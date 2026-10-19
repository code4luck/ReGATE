# 用于获取树模型的输入的特征
# 1. 所有的特征使用mean pooling 获取 [dim]
# 2. 特征为 [enzyme_features, EC_features, mol_features, macckeys] 按照csv文件顺序进行保存获取的表征
# 3. [labels]
# pkl文件保存到本地

import argparse
import pickle
import pandas as pd
import numpy as np
import os
import json
import sys
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np
import re
from ec_model import HierECResidualEmbedding

PAD, UNK, DASH = 0, 1, 2

def split_ec_annotations(raw):
    return [
        x for x in re.split(r"[;\s]+", str(raw).strip())
        if x
    ]

def build_ec_vocabs(train_ec_list):
    vocabs = [
        {"<pad>": PAD, "<unk>": UNK, "<dash>": DASH}
        for _ in range(4)
    ]
    for raw in train_ec_list:
        for ec in split_ec_annotations(raw):
            parts = ec.strip().split(".")
            if not 1 <= len(parts) <= 4:
                continue
            parts = parts + ["-"] * (4 - len(parts))
            prefix_parts = []

            for lvl, part in enumerate(parts):
                if part == "-":
                    break
                prefix_parts.append(part)
                prefix = ".".join(prefix_parts)
                vocabs[lvl].setdefault(prefix, len(vocabs[lvl]))

    return vocabs

def encode_one_ec(ec, ec_vocabs):
    parts = ec.strip().split(".")
    if not 1 <= len(parts) <= 4:
        return [UNK] * 4, [0.0] * 4
    parts = parts + ["-"] * (4 - len(parts))
    ids = []
    mask = []
    prefix_parts = []
    usable = True

    for lvl, part in enumerate(parts):
        if not usable:
            ids.append(DASH)
            mask.append(0.0)
            continue
        if part =="-":
            usable = False
            ids.append(DASH)
            mask.append(1.0)
            continue

        prefix_parts.append(part)
        prefix = ".".join(prefix_parts)
        token_id = ec_vocabs[lvl].get(prefix, UNK)
        ids.append(token_id)
        if token_id == UNK:
            mask.append(0.0)
            usable = False
        else:
            mask.append(1.0)
    return ids, mask

def encode_ec(raw_ec, ec_vocabs):
    ecs = split_ec_annotations(raw_ec)

    if not ecs:
        return (
            torch.tensor([[UNK] * 4], dtype=torch.long),
            torch.tensor([[0.0] * 4], dtype=torch.float32),
        )

    all_ids = []
    all_masks = []

    for ec in ecs:
        ids, mask = encode_one_ec(ec, ec_vocabs)
        all_ids.append(ids)
        all_masks.append(mask)

    return (
        torch.tensor(all_ids, dtype=torch.long),       # [N_ec, 4]
        torch.tensor(all_masks, dtype=torch.float32),  # [N_ec, 4]
    )


def get_macckeys(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fp = MACCSkeys.GenMACCSKeys(mol)
    fp_str = fp.ToBitString()
    fp_array = np.array([int(i) for i in fp_str])
    return torch.tensor(fp_array, dtype=torch.float32)

def read_csv(file_path):
    # 用于读取csv文件
    df = pd.read_csv(file_path)
    return df

def pool_multiple_ec(
    ec_features,
    ec_mask,
    ec_owner,
    batch_size,
):
    valid_ec = ec_mask.any(dim=1).to(ec_features.dtype)  # [B']
    weighted_features = ec_features * valid_ec.unsqueeze(-1)
    pooled = ec_features.new_zeros(batch_size, ec_features.size(-1))
    pooled.index_add_(0, ec_owner, weighted_features)
        
    counts = ec_features.new_zeros(batch_size)
    counts.index_add_(0, ec_owner, valid_ec)
        
    pooled = pooled / counts.clamp(min=1.0).unsqueeze(-1)
    return pooled


def load_pretrained_ec_encoder(pretrained_ec_encoder_path, vocab_sizes, ec_dim, dropout):
    # 用于加载预训练的ec encoder
    ec_encoder = HierECResidualEmbedding(
            vocab_sizes=vocab_sizes,
            dec_dim=ec_dim,
            dropout=dropout,
        )
    # 加载state_dict
    payload = torch.load(pretrained_ec_encoder_path, map_location="cpu")
    state_dict = payload.get("state_dict", payload)
    ec_encoder.load_state_dict(state_dict, strict=True)
    return ec_encoder

def main(args):
    # 设置种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 创建保存目录
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    # 加载ec vocab
    with open(args.ec_vocab_path, "r") as f:
        ec_vocab = json.load(f)
    vocabs = ec_vocab["vocabs"]
    ec_vocab_sizes = ec_vocab["ec_vocab_sizes"]
    args.vocab_sizes = ec_vocab_sizes

    # 加载df
    inputs_df = read_csv(args.file_path)
    # 加载ec encoder
    ec_encoder = load_pretrained_ec_encoder(args.pretrained_ec_encoder_path, 
                        args.vocab_sizes, args.ec_dim, args.dropout)
    ec_encoder.eval()
    ec_encoder.to(args.device)

    # 一行行的遍历csv文件并进行加载记录同时进行池化, 每次只有一个记录的 [L, dim] -> pooling-> [dim] 
    
    all_features = []
    all_labels = []
    for index, row in inputs_df.iterrows():
        # 加载记录
        mol_embed_path = row[args.smiles_embedding_col]
        enzyme_embed_path = row[args.enzyme_embedding_col]
        smiles = row[args.smiles_col]
        label = row[args.label_col]
        # 加载macckeys
        macckeys = get_macckeys(smiles)
        # get_ec --- 只对当前记录进行ec特征操作所以batch size为1, 同时 ec_owner就是当前记录的ec数量
        cur_ecs = row[args.ec_col]
        ec_ids, ec_mask = encode_ec(cur_ecs, vocabs)
        ec_ids = ec_ids.to(args.device)
        ec_mask = ec_mask.to(args.device)
        with torch.no_grad():
            ec_features = ec_encoder(ec_ids, ec_mask)
            n_ec = ec_ids.size(0)
            ec_owner = torch.zeros(n_ec, dtype=torch.long, device=ec_ids.device)
            ec_features = pool_multiple_ec(ec_features, ec_mask, ec_owner, batch_size=1)  # [1, Dec]
            ec_features = ec_features.squeeze(0)  # [Dec]
        # 加载enzyme
        enzyme_features = torch.load(enzyme_embed_path) #[L,dim]
        enzyme_features = enzyme_features.mean(dim=0) #[dim]
        # 加载mol
        mol_features = torch.load(mol_embed_path) #[L,dim]
        mol_features = mol_features.mean(dim=0) #[dim]

        # 所有变为 numpy 
        enzyme_features = enzyme_features.cpu().numpy()
        mol_features = mol_features.cpu().numpy()
        ec_features = ec_features.cpu().numpy()
        macckeys = macckeys.cpu().numpy()
        # cat
        all_features.append(np.concatenate([enzyme_features, ec_features, mol_features, macckeys], axis=0))
        all_labels.append(label)
    # 保存到pkl文件
    features_name = args.output_file_name + "_features.npy" # [N, dim]
    labels_name = args.output_file_name + "_labels.npy" # [N]
    np.save(os.path.join(args.output_folder, features_name), np.array(all_features))
    np.save(os.path.join(args.output_folder, labels_name), np.array(all_labels))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, help="the path of the csv file")
    parser.add_argument("--output_folder", type=str, help="the folder of the output pkl file")
    parser.add_argument("--output_file_name", type=str, help="the file name of the output pkl file")

    # smiles embedding col 
    parser.add_argument("--smiles_embedding_col", type=str, help="the column name of the smiles embedding")
    parser.add_argument("--smiles_col", type=str, help="the column name of the smiles")
    # enzyme embedding col
    parser.add_argument("--enzyme_embedding_col", type=str, help="the column name of the enzyme embedding")
    parser.add_argument("--label_col", type=str, help="the column name of the label")
    # ec col
    parser.add_argument("--ec_col", type=str, help="the column name of the ec")
    parser.add_argument("--pretrained_ec_encoder_path", type=str, help="the path of the ec model weights")
    parser.add_argument("--ec_vocab_path", type=str, help="the path of the ec vocab")
    parser.add_argument("--ec_dim", type=int, default=128, help="the dimension of the ec")
    parser.add_argument("--dropout", type=float, default=0.1, help="the dropout of the ec encoder")
    # seed
    parser.add_argument("--seed", type=int, default=42, help="the seed of the random number generator")
    # device
    parser.add_argument("--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()
    
    main(args)


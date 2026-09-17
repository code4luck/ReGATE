import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from typing import List, Optional, Dict
from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import numpy as np
import re

PAD, UNK, DASH = 0, 1, 2

def split_ec_annotations(raw):
    return [
        x for x in re.split(r"[;\s]+", str(raw).strip())
        if x
    ]

# def build_ec_vocabs(train_ec_list):
#     vocabs = [{"<pad>": PAD, "<unk>": UNK, "<dash>": DASH} for _ in range(4)]
#     for raw in train_ec_list:
#         for ec in split_ec_annotations(raw):          # 拆多EC
#             parts = ec.strip().split(".")
#             if len(parts) != 4:
#                 continue
#             for lvl in range(4):
#                 if parts[lvl] == "-":
#                     continue
#                 prefix = ".".join(parts[:lvl + 1])
#                 vocabs[lvl].setdefault(prefix, len(vocabs[lvl]))
#     return vocabs

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


class CataProDataset(Dataset):
    """
    每条样本从 CSV 中读取:
      - protein_embedding_path: 蛋白 embedding 的 .pt 路径
      - mol_embedding_path:     小分子 embedding 的 .pt 路径
      - label_col:              标签列
      - prot_embedding_model:   蛋白 embedding 模型
      - mol_embedding_model:     小分子 embedding 模型
    """
    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str,
        protein_embedding_col: str,
        mol_embedding_col: str,
        prot_embedding_model: str,
        mol_embedding_model: str,
        ec_col: str = "EC",
        ec_vocabs: List[Dict[str, int]] = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.label_col = label_col
        self.protein_path_col = protein_embedding_col
        self.mol_path_col = mol_embedding_col
        self.prot_embedding_model = prot_embedding_model
        self.mol_embedding_model = mol_embedding_model
        self.ec_col = ec_col
        self.ec_vocabs = ec_vocabs

        required_cols = [self.label_col, self.protein_path_col, self.mol_path_col]
        for c in required_cols:
            if c not in self.df.columns:
                raise ValueError(f"Missing required column: {c}")


    def __len__(self) -> int:
        return len(self.df)

    def _load_tensor(self, path: str, is_mol=False) -> torch.Tensor:
        # 先放在cpu然后 gpu
        x = torch.load(path, map_location="cpu")
        if "unimol" in self.mol_embedding_model and is_mol: # ! 其他的都是embedding存储
            x = x["atomic_reprs"] # [L, D]

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Loaded object is not Tensor: {path}, type={type(x)}")
        if x.dim() != 2:
            raise ValueError(f"Expect 2D tensor [L, D], got {tuple(x.shape)} from {path}")

        return x.float()


    # def _encode_ec(self, ec):
    #     # parts = str(ec).strip().split(".")
    #     # 只采用第一个ec
    #     ec = str(ec).split(";")[0].strip()
    #     parts = str(ec).strip().split(".")
    #     if len(parts) != 4:
    #         return [UNK]*4, [1.0]*4
    #     ids, mask = [], []
    #     for lvl in range(4):
    #         if parts[lvl] == "-": 
    #             ids.append(DASH); mask.append(0.0) # "-" 为pad?
    #         else:
    #             tid = self.ec_vocabs[lvl].get(".".join(parts[:lvl+1]), UNK)
    #             ids.append(tid); mask.append(0.0 if tid == UNK else 1.0)
    #     return ids, mask

    def _encode_one_ec(self, ec):
        parts = ec.strip().split(".")
        if not 1 <= len(parts) <= 4:
            return [UNK] * 4, [0.0] * 4
        parts = parts + ["-"] * (4 - len(parts))
        ids = []
        mask = []
        prefix_parts = []
        usable = True

        for lvl, part in enumerate(parts):
            if not usable or part == "-":
                ids.append(DASH)
                mask.append(0.0)
                usable = False
                continue
            prefix_parts.append(part)
            prefix = ".".join(prefix_parts)
            token_id = self.ec_vocabs[lvl].get(prefix, UNK)
            ids.append(token_id)
            if token_id == UNK:
                mask.append(0.0)
                usable = False
            else:
                mask.append(1.0)
        return ids, mask

    def _encode_ec(self, raw_ec):
        ecs = split_ec_annotations(raw_ec)

        if not ecs:
            return (
                torch.tensor([[UNK] * 4], dtype=torch.long),
                torch.tensor([[0.0] * 4], dtype=torch.float32),
            )

        all_ids = []
        all_masks = []

        for ec in ecs:
            ids, mask = self._encode_one_ec(ec)
            all_ids.append(ids)
            all_masks.append(mask)

        return (
            torch.tensor(all_ids, dtype=torch.long),       # [N_ec, 4]
            torch.tensor(all_masks, dtype=torch.float32),  # [N_ec, 4]
        )

    def _get_macckeys(self, smiles):
        """
            return tensor feature [167,]
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        fp = MACCSkeys.GenMACCSKeys(mol)
        fp_str = fp.ToBitString()
        fp_array = np.array([int(i) for i in fp_str])
        return torch.tensor(fp_array, dtype=torch.float32)
        
    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        protein_path = str(row[self.protein_path_col])
        mol_path = str(row[self.mol_path_col])
        ec = str(row[self.ec_col])
        smiles = str(row["Smiles"])
        macckeys = self._get_macckeys(smiles)

        protein_embedding = self._load_tensor(protein_path, is_mol=False)  # [Lp, Dp]
        mol_embedding = self._load_tensor(mol_path, is_mol=True)          # [Lm, Dm]

        ec_ids, ec_mask = self._encode_ec(ec)

        # check path
        if (not protein_path) or (not os.path.exists(protein_path)):
            raise FileNotFoundError(f"Invalid protein embedding path at idx={idx}: {protein_path}")
        if (not mol_path) or (not os.path.exists(mol_path)):
            raise FileNotFoundError(f"Invalid mol embedding path at idx={idx}: {mol_path}")
        # check embedding
        if protein_embedding.size(0) == 0:
            raise ValueError(f"Empty protein embedding sequence at idx={idx}, path={protein_path}")
        if mol_embedding.size(0) == 0:
            raise ValueError(f"Empty mol embedding sequence at idx={idx}, path={mol_path}")

        label = torch.tensor(float(row[self.label_col]), dtype=torch.float32)
        return {
            "ec_ids": ec_ids,
            "ec_mask": ec_mask,
            "protein_embedding": protein_embedding,  # [Lp, Dp]
            "mol_embedding": mol_embedding,          # [Lm, Dm]
            "labels": label,                         # scalar
            "macckeys": macckeys,                    # [167,]
        }


def collate_path_embeddings(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    输出与 LitModel 对齐:
      - protein_embedding: [B, Lp, Dp]
      - protein_mask:      [B, Lp] (1 valid, 0 pad)
      - mol_embedding:     [B, Lm, Dm]
      - mol_mask:          [B, Lm] (1 valid, 0 pad)
      - labels:            [B]
    """
    protein_list = [x["protein_embedding"] for x in batch]
    mol_list = [x["mol_embedding"] for x in batch]
    labels = torch.stack([x["labels"] for x in batch], dim=0) # [B,]
    macckeys = torch.stack([x["macckeys"] for x in batch], dim=0) # [B, 167]
    # -------ec 
    # ec_list = [x["ec_ids"] for x in batch]
    # ec_mask_list = [x["ec_mask"] for x in batch]
    # ec_ids  = torch.tensor(ec_list,  dtype=torch.long)  
    # ec_mask = torch.tensor(ec_mask_list, dtype=torch.float)
    # 扩展ec
    ec_ids_list = []
    ec_mask_list = []
    ec_owner_list = []
    ec_counts = []
    for sample_idx, item in enumerate(batch):
        sample_ec_ids = item["ec_ids"]       # [N_ec, 4]
        sample_ec_mask = item["ec_mask"]     # [N_ec, 4]
        n_ec = sample_ec_ids.size(0) # ! 标注此时的item 样本有多少个ec
        ec_ids_list.append(sample_ec_ids)
        ec_mask_list.append(sample_ec_mask)

        ec_owner_list.append(
            torch.full(
                (n_ec,),
                sample_idx,
                dtype=torch.long,
            )
        )

        ec_counts.append(n_ec)

    ec_ids = torch.cat(ec_ids_list, dim=0)        # [B', 4]
    ec_mask = torch.cat(ec_mask_list, dim=0)      # [B', 4]
    ec_owner = torch.cat(ec_owner_list, dim=0)    # [B']
    ec_counts = torch.tensor(ec_counts)           # [B]

    # 记录真实长度
    protein_lens = torch.tensor([t.size(0) for t in protein_list], dtype=torch.long)
    mol_lens = torch.tensor([t.size(0) for t in mol_list], dtype=torch.long)
    # pad 到 batch 内最大长度
    protein_embedding = pad_sequence(protein_list, batch_first=True, padding_value=0.0)  # [B,Lp,Dp]
    mol_embedding = pad_sequence(mol_list, batch_first=True, padding_value=0.0)          # [B,Lm,Dm]
    B, Lp, _ = protein_embedding.shape
    _, Lm, _ = mol_embedding.shape
    # mask: 1 valid, 0 pad
    protein_mask = (torch.arange(Lp).unsqueeze(0) < protein_lens.unsqueeze(1)).long()  # [B,Lp]
    mol_mask = (torch.arange(Lm).unsqueeze(0) < mol_lens.unsqueeze(1)).long()          # [B,Lm]
    return {
        "ec_ids": ec_ids,
        "ec_mask": ec_mask,
        "ec_owner": ec_owner,
        "ec_counts": ec_counts,
        "protein_embedding": protein_embedding,
        "protein_mask": protein_mask,
        "mol_embedding": mol_embedding,
        "mol_mask": mol_mask,
        "labels": labels,
        "macckeys": macckeys,
    }
def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_path_embeddings,
    )



import logging
import torch.nn as nn
import os
import sys
import pickle
import torch
import argparse
import numpy as np
from unimol_tools import UniMolRepr
import pandas as pd
from transformers import AutoModel, AutoTokenizer
# from molformer.modeling_molformer import MolformerForMaskedLM
from tqdm import tqdm
from transformers import T5EncoderModel, T5Tokenizer

class MolEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.model_name = config.model_name
        if "unimol" in self.model_name:
            self.model = UniMolRepr(data_type='molecule', 
                                    remove_hs=False,
                                    model_name='unimolv2', # avaliable: unimolv1, unimolv2
                                    model_size='570m', # work when model_name is unimolv2. avaliable: 84m, 164m, 310m, 570m, 1.1B.
                                    )
        elif "chembert" in self.model_name:
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
        elif "molformer" in self.model_name:
            self.model = AutoModel.from_pretrained(self.model_name, deterministic_eval=True, trust_remote_code=True)
            self.model.eval()
        elif "mol_t5" in self.model_name:
            self.model = T5EncoderModel.from_pretrained(self.model_name)
            self.model.eval()
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")


    def forward(self, inputs):
        with torch.no_grad():
            # unimol is for single molecule operation
            if "unimol" in self.model_name: 
                return self.model.get_repr(inputs, return_atomic_reprs=True) # cls_repr/ atomic_reprs dict
            # chembert is for transformer input and output
            elif "chembert" in self.model_name or "molformer" in self.model_name:
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.last_hidden_state # [B, L+2, dim]
            elif "mol_t5" in self.model_name:
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                return outputs.last_hidden_state # [B, L+2, dim]
            else:
                raise ValueError(f"Invalid model name: {self.model_name}")


def main(args):
    file_path = args.file_path
    mol_col = args.mol_col
    save_folder = args.save_folder
    model_name = args.model_name
    device = args.device

    os.makedirs(save_folder, exist_ok=True)
    
    # transformer models need tokenizer, unimol does not need
    tokenizer = None
    if "unimol" not in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
                        model_name, trust_remote_code=("molformer" in model_name or "mol_t5" in model_name))
    model = MolEncoder(args)
    if "unimol" not in model_name:
        model.to(device)
        model.eval()
    df = pd.read_csv(file_path)
    
    
    error_list = []
    paths = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Mol Embedding"):
        smiles = str(row[mol_col])
        save_path = os.path.join(save_folder, f"{idx}.pt")
        # if exists, skip
        if os.path.exists(save_path):
            paths.append(save_path)
            continue
        try:
            if "unimol" in model_name:
                embed_dict = model([smiles]) # ! []

                # dict atomic_reprs[L, dim], cls_repr[1,dim]
                emb = {"cls_repr": torch.tensor(embed_dict["cls_repr"]).cpu(), 
                    "atomic_reprs": torch.tensor(embed_dict["atomic_reprs"]).squeeze(0).cpu()} 
            elif "chembert" in model_name:
                inputs = tokenizer(
                    smiles,
                    return_tensors="pt",
                    truncation=True,
                    max_length=510,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                hidden = model(inputs)
                emb = hidden.squeeze(0).cpu() # [L+2 ?,dim]
            elif "molformer" in model_name:
                inputs = tokenizer(
                    smiles,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                hidden = model(inputs)
                emb = hidden.squeeze(0).cpu() # [L+2 ?,dim]
            elif "mol_t5" in model_name:
                inputs = tokenizer(
                    smiles,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                hidden = model(inputs)
                emb = hidden.squeeze(0).cpu() # [L+2 ?,dim]
            else:
                raise ValueError(f"Invalid model name: {model_name}")
            
            # save the embedding
            torch.save(emb, save_path) # folder/idx.pt
            paths.append(save_path)
    
        except Exception as e:
            print(f"[Error] idx={idx}, smiles={smiles}, error={e}")
            error_list.append({"idx": idx, "smiles": smiles, "error": str(e)})
            paths.append(None)  # placeholder, keep the same number of rows as df

    model_name = model_name.split('/')[-1]
    df[f'{model_name}_mol_embedding_path'] = paths
    out_csv = file_path.replace('.csv', f'_{model_name}_mol_embedding.csv')
    
    df.to_csv(out_csv, index=False)
    print(f"Done. {len(df) - len(error_list)}/{len(df)} succeeded.")
    
    if error_list:
        error_df = pd.DataFrame(error_list)
        error_csv = file_path.replace('.csv', f'_{model_name}_mol_embedding_errors.csv')
        error_df.to_csv(error_csv, index=False)
        print(f"{len(error_list)} errors saved to: {error_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--mol_col", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--device", type=str, required=True)
    args = parser.parse_args()
    main(args)


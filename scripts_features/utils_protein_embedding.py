"""
ESM2 / ESM-C [CLS] + seq + [EOS] N + 2
ProtT5 seq + [EOS] N + 1
Ankh seq + [EOS] N + 1
"""

import torch
import torch.nn as nn
import pandas as pd
import argparse
import os, re
from tqdm import tqdm
from transformers import EsmModel, T5EncoderModel, AutoModelForMaskedLM, AutoTokenizer, T5Tokenizer


class SeqEncoder(nn.Module):
    def __init__(self, model_name):
        super(SeqEncoder, self).__init__()
        self.seq_model_name = model_name
        if "esmc" in model_name:                          
            self.model = ESMC.from_pretrained("esmc_600m") 
        elif "esm2" in self.seq_model_name: 
            self.model = EsmModel.from_pretrained(self.seq_model_name)
        elif "t5" in self.seq_model_name or "ankh" in self.seq_model_name: 
            self.model = T5EncoderModel.from_pretrained(self.seq_model_name)
        else:
            raise ValueError(f"Invalid model name: {self.seq_model_name}")

        self.model.eval()

    @torch.no_grad()
    def forward(self, input_ids=None, attention_mask=None, raw_seq=None):
        if "esmc" in self.seq_model_name:
            protein = ESMProtein(sequence=raw_seq)
            protein_tensor = self.model.encode(protein)
            output = self.model.logits(
                protein_tensor,
                LogitsConfig(return_hidden_states=True)
            )
            return output.hidden_states[-1]  # [1, L+2, dim]
        else:
            return self.model(input_ids, attention_mask).last_hidden_state # [B,L,dim]

def get_tokenizer(model_name):
    if "esm2" in model_name or "ankh" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif "t5" in model_name:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
    elif "esmc" in model_name:
        tokenizer=None
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    return tokenizer

def extra_embedding(args):
    input_file_path = args.file_path
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)

    seq_col = getattr(args, 'seq_col', 'Sequence')
    device = args.device
    
    model_name = args.model_name
    model = SeqEncoder(model_name)
    tokenizer = get_tokenizer(model_name)
    model.to(device)

    # read the protein_csv file
    df = pd.read_csv(input_file_path)
    # save the embedding to the specified path, and add the corresponding read path to the csv record

    paths = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Embedding"):
        seq = str(row[seq_col])
        save_path = os.path.join(save_folder, f"{idx}.pt")

        # seq = seq[:1024] # ! first truncate the sequence and then tokenize
        if len(seq) > 1499:
            seq = seq[:749] + seq[-749:]
        if not os.path.exists(save_path):
            if "esmc" in model_name:
                # ESMC does not need an external tokenizer, just pass the original sequence
                hidden = model(raw_seq=seq)   # [1, L+2, dim]
                print("hidden.shape ", hidden.shape)
                emb = hidden.squeeze(0).cpu()       # [L+2, dim]
            else:
                tokenizer_input = " ".join(list(re.sub(r"[UZOB]", "X", seq))) if "t5" in model_name else seq
                inputs = tokenizer(tokenizer_input, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                hidden = model(input_ids=input_ids, attention_mask=attention_mask)
                emb = hidden.squeeze(0).cpu()       # [L, dim]
            torch.save(emb, save_path)
        paths.append(save_path)
    
    model_name = model_name.split('/')[-1]
    df[f'{model_name}_prot_embedding_path'] = paths
    print(f"Saved {len(paths)} embeddings to {save_folder}")
    df.to_csv(input_file_path.replace('.csv', f'_{model_name}_prot_embedding.csv'), index=False)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--save_folder", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True, default="esm2_650m")
    parser.add_argument("--device", type=str, required=True, default="cuda")
    parser.add_argument("--seq_col", type=str, required=True, default="sequence")
    args = parser.parse_args()
    extra_embedding(args)

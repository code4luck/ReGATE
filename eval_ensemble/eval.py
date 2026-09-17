import os
import sys
import json
import argparse
import importlib
from pathlib import Path

from gce_litmodel import LitModel as GCELitModel
from ec_contrast_datasets import CataProDataset, build_dataloader

import numpy as np
import pandas as pd
import torch


def parse_args():
    p = argparse.ArgumentParser("Unified eval for gce")

    # select model and ckpt
    p.add_argument("--model_type", required=True, choices=["gce"])
    p.add_argument("--ckpt_path", required=True, type=str)

    # data (same as trainer)
    p.add_argument("--data_path", required=True, type=str)
    p.add_argument("--data_name", type=str, default="Km")
    p.add_argument("--ec_vocab_path", type=str, default="unique/common_train_ec_vocab.json")
    p.add_argument("--test_csv_name", type=str, default="test.csv")
    p.add_argument("--label_col", type=str, default="label")
    p.add_argument("--ec_col", type=str, default="EC")
    p.add_argument("--smiles_col", type=str, default="Smiles")
    p.add_argument("--protein_embedding_col", type=str, default="prot_embedding_path")
    p.add_argument("--mol_embedding_col", type=str, default="mol_embedding_path")
    p.add_argument("--prot_embedding_model", type=str, default="esm2")
    p.add_argument("--mol_embedding_model", type=str, default="chemberta")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)

    # model hyperparameters (must match training, otherwise ckpt loading/reconstruction does not match)
    p.add_argument("--dec_dim", type=int, default=64)
    p.add_argument("--protein_embedding_size", type=int, default=1280)
    p.add_argument("--mol_embedding_size", type=int, default=768)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_attention_heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--view_dropout", type=float, default=0.2)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--macc_dim", type=int, default=167)
    p.add_argument("--init_mu", type=float, default=0.0)
    p.add_argument("--cross_layers", type=int, default=2)
    p.add_argument("--cross_rank", type=int, default=128)

    # litmodel needs to read training related items 
    p.add_argument("--base_loss", type=str, default="mse", choices=["mse", "huber", "rmse"])
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--lambda_repr_cons", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-4)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="./eval_outputs")
    return p.parse_args()


# ---- metrics implementation: same as train_test_unikp.py, ensure horizontal comparison ----
def safe_pcc(y_true, y_pred):
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def safe_scc(y_true, y_pred):
    if len(y_true) < 2:
        return float("nan")
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))


def compute_metrics(y_true, y_pred):
    diff = y_pred - y_true
    mse = float(np.mean(diff ** 2))
    mae = float(np.mean(np.abs(diff)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "pcc": safe_pcc(y_true, y_pred),
        "scc": safe_scc(y_true, y_pred),
        "mae": mae,
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "r2": r2,
    }




def move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def forward_pred(model, batch):
    """统一三种模型的前向：dict 取 'pred'，张量直接用。"""
    out = model(
        protein_embedding=batch["protein_embedding"],
        protein_mask=batch["protein_mask"],
        mol_embedding=batch["mol_embedding"],
        mol_mask=batch["mol_mask"],
        ec_ids=batch["ec_ids"],
        ec_mask=batch["ec_mask"],
        ec_owner=batch["ec_owner"],
        macckeys=batch["macckeys"],
    )
    pred = out["pred"] if isinstance(out, dict) else out
    return pred.reshape(-1)

def get_model(model_type):
    if model_type == "gce":
        return GCELitModel
    else:
        raise ValueError(f"Invalid model type: {model_type}")

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    LitModel = get_model(args.model_type)

    # 1) EC vocab -> vocab_sizes (model building needs)
    with open(args.ec_vocab_path, "r") as f:
        ec_vocab = json.load(f)
    vocabs = ec_vocab["vocabs"]
    args.vocab_sizes = ec_vocab["ec_vocab_sizes"]

    # 2) test set (shuffle=False, order matches csv row order)
    test_csv = Path(args.data_path) / args.test_csv_name
    test_df = pd.read_csv(test_csv)
    print(f"[INFO] test csv: {test_csv}, n={len(test_df)}")

    test_ds = CataProDataset(
        df=test_df,
        label_col=args.label_col,
        protein_embedding_col=args.protein_embedding_col,
        mol_embedding_col=args.mol_embedding_col,
        prot_embedding_model=args.prot_embedding_model,
        mol_embedding_model=args.mol_embedding_model,
        ec_col=args.ec_col,
        ec_vocabs=vocabs,
    )
    test_loader = build_dataloader(
        dataset=test_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    # 3) load best ckpt (litmodel does not save hparams, need to pass args again)
    device = torch.device(args.device)
    lit = LitModel.load_from_checkpoint(args.ckpt_path, args=args, map_location=device)
    model = lit.model.to(device).eval()

    # 4) inference
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = move_to_device(batch, device)
            pred = forward_pred(model, batch)
            preds.append(pred.float().cpu().numpy())
            labels.append(batch["labels"].reshape(-1).float().cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    metrics = compute_metrics(labels, preds)

    # 5) save: per-sample predictions (concat back to original table) + metrics
    out_dir = Path(args.output_dir) / args.data_name / args.model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_df = test_df.copy()
    # pred_df["label"] = labels # use labels already in test_df
    pred_df["prediction"] = preds
    pred_path = out_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    summary = {
        "model_type": args.model_type,
        "ckpt_path": args.ckpt_path,
        "data_name": args.data_name,
        "test_csv": str(test_csv),
        "n_test": int(len(labels)),
        "metrics": metrics,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"[INFO] saved predictions -> {pred_path}")


if __name__ == "__main__":
    main()
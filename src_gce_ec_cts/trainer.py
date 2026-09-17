import os
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from litmodel import LitModel  
from ec_contrast_datasets import CataProDataset, build_dataloader


def parse_args():
    parser = ArgumentParser("CatPred training with PyTorch Lightning")

    # -------------------------
    # Data
    # -------------------------
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--data_name", type=str, default="Km")
    parser.add_argument("--ec_vocab_path", type=str, default="unique/common_train_ec_vocab.json")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--ec_col", type=str, default="EC")
    parser.add_argument("--sequence_col", type=str, default="Sequence")
    parser.add_argument("--smiles_col", type=str, default="Smiles")
    parser.add_argument("--protein_embedding_col", type=str, default="prot_embedding_path")
    parser.add_argument("--freeze_ec", action="store_true") # 是否冻结ec_module 默认为 False
    parser.add_argument("--pretrained_ec_encoder_path", type=str, default=None) # 是否加载预训练的ec_module 默认为 None
    parser.add_argument("--mol_embedding_col", type=str, default="mol_embedding_path")
    parser.add_argument("--prot_embedding_model", type=str, default="esm2")
    parser.add_argument("--mol_embedding_model", type=str, default="chemberta")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    # -------------------------
    # Model
    # -------------------------
    parser.add_argument("--ec_dim", type=int, default=128)
    parser.add_argument("--protein_embedding_size", type=int, default=1280)
    parser.add_argument("--cross_layers", type=int, default=2)
    parser.add_argument("--cross_rank", type=int, default=128)
    parser.add_argument("--mol_embedding_size", type=int, default=768)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    # parser.add_argument("--lambda_repr_cons", type=float, default=0.1)
    # parser.add_argument("--lambda_y", type=float, default=0.02)
    # parser.add_argument("--invariant_loss", type=str, default="cos")
    # parser.add_argument("--use_output_consistency", action="store_true")

    # -------------------------
    # Train
    # -------------------------
    parser.add_argument("--base_loss", type=str, default="mse", choices=["mse", "huber", "rmse"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--monitor", type=str, default="val_pearson_corr")

    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--num_nodes", type=int, default=1)

    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--gradient_clip_algorithm", type=str, default="norm")
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--seed", type=int, default=42)

    # -------------------------
    # Output
    # -------------------------
    parser.add_argument("--output_dir", type=str, default="./outputs")

    return parser.parse_args()




def make_dataloader_from_df(input_df: pd.DataFrame, vocabs, args, shuffle: bool):
    ds = CataProDataset(
        df=input_df,
        label_col=args.label_col,
        protein_embedding_col=args.protein_embedding_col,
        mol_embedding_col=args.mol_embedding_col,
        prot_embedding_model=args.prot_embedding_model,
        mol_embedding_model=args.mol_embedding_model,
        ec_col = args.ec_col,
        ec_vocabs=vocabs,
    )

    return build_dataloader(
        dataset=ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=shuffle,
        drop_last=False,
    )


def main():
    args = parse_args()
    seed_everything(args.seed, workers=True)
    print("args: ", args)
    
    # 加载ec_vocab
    with open(args.ec_vocab_path, "r") as f:
        ec_vocab = json.load(f)
    vocabs = ec_vocab["vocabs"]
    ec_vocab_sizes = ec_vocab["ec_vocab_sizes"]

    args.vocab_sizes = ec_vocab_sizes


    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 拼接上 args.mol_embedding_model和args.prot_embedding_model
    model_type = f"{args.mol_embedding_model}_{args.prot_embedding_model}"
    out_root = Path(args.output_dir) / args.data_name
    ckpt_root = out_root / "checkpoints" 

    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    train_data_path = Path(args.data_path) / "train.csv"
    val_data_path = Path(args.data_path) / "valid.csv"
    test_data_path = Path(args.data_path) / "test.csv"

    train_input_df = pd.read_csv(train_data_path)
    val_input_df = pd.read_csv(val_data_path)
    test_input_df = pd.read_csv(test_data_path)
    print(f"[INFO] Loaded csv: {train_data_path}, num_samples={len(train_input_df)}")
    print(f"[INFO] Loaded csv: {val_data_path}, num_samples={len(val_input_df)}")
    print(f"[INFO] Loaded csv: {test_data_path}, num_samples={len(test_input_df)}")

    monitor_mode = "max" if ("corr" in args.monitor or "r2" in args.monitor) else "min"

    train_loader = make_dataloader_from_df(train_input_df,  vocabs, args, shuffle=True)
    val_loader = make_dataloader_from_df(val_input_df, vocabs, args, shuffle=False)
    test_loader = make_dataloader_from_df(test_input_df, vocabs, args, shuffle=False)

    model = LitModel(args)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_root),
            monitor=args.monitor,
            mode=monitor_mode,
            save_top_k=1,
            save_last=True,
            filename=f"{args.data_name}-" + "{epoch:02d}-{" + args.monitor + ":.4f}",
            auto_insert_metric_name=False,
            verbose=True,
        )

    early_stop_cb = EarlyStopping(
            monitor=args.monitor,
            mode=monitor_mode,
            patience=args.patience,
            verbose=True,
        )

    logger = CSVLogger(
            save_dir=str(out_root / "logs"),
        )

    trainer = pl.Trainer(
            accelerator=args.accelerator,
            devices=args.devices,
            strategy=args.strategy,
            num_nodes=args.num_nodes,
            max_epochs=args.max_epochs,
            precision=args.precision,
            accumulate_grad_batches=args.accumulate_grad_batches,
            gradient_clip_val=args.gradient_clip_val,
            gradient_clip_algorithm=args.gradient_clip_algorithm,
            deterministic=True,
            logger=logger,
            callbacks=[ckpt_cb, early_stop_cb],
        )

    trainer.fit(model, train_loader, val_loader)

    best_ckpt = ckpt_cb.best_model_path
    if best_ckpt:
        print(f"[INFO] best ckpt: {best_ckpt}")
        best_model = LitModel.load_from_checkpoint(best_ckpt, args=args)
    else:
        print(f"[WARN] no best ckpt found, using current model.")
        best_model = model

    test_out = trainer.test(best_model, dataloaders=test_loader, verbose=True)
    # test_out 通常是长度1的list[dict]
    result_df = pd.DataFrame(test_out)
    result_csv = out_root / f"test_metrics_monitor_{args.monitor}.csv"
    result_df.to_csv(result_csv, index=False)
    print(f"\n[INFO] Saved test metrics to: {result_csv}")


if __name__ == "__main__":
    main()



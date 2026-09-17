#!/bin/bash
set -euo pipefail

python ./src_treemodel/our_treemodel.py \
  --train_path ./dataset/treemodel_feat/seq_idt/kcat/train \
  --test_path ./dataset/treemodel_feat/seq_idt/kcat/test \
  --output_dir ./treemodel_res/seed42/seq_idt/kcat \
  --n_seeds 2 \
  --seed 42

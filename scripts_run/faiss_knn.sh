#!/bin/bash

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TRAIN_CSV="./dataset/two_splits/seq_idt/kcat/train.csv"
PRED_CSV="./eval_outputs/kcat_seq_idt/kcat_seq_idt/gce/test_predictions.csv"
OUT_CSV="./retrieval_faiss/kcat_seq_idt/faiss_knn.csv"

python ./eval_ensemble/faiss_cat_index.py \
  --train_csv_path "${TRAIN_CSV}" \
  --pred_csv_path "${PRED_CSV}" \
  --output_csv_path "${OUT_CSV}" \
  --embedding_root . \
  --prot_embedding_col prot_t5_xl_prot_embedding_path \
  --mol_embedding_col mol_t5_mol_embedding_path \
  --label_col log10_kcat \
  --pred_col prediction \
  --top_k 35 \
  --temperature 1.0 \
  --lambda_knn 0.7 \
  --normalize_each_modality \
  --faiss_gpu
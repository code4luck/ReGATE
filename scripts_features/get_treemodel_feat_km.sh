#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

EC_CKPT="./dataset/ec_cts_ckpt/ec_cts_hard_ep30_29.pt"
EC_PATH="./dataset/two_splits/clean_uniprot_ec_vocab.json"
EC_DIM=128
EC_DROPOUT=0.1
SEED=42
PROT_EMBEDDING_MODEL="prot_t5_xl"
MOL_EMBEDDING_MODEL="mol_t5"
MOL_COL="Smiles"
EC_COL="EC"


python ./scripts/get_treemodel_features.py \
  --file_path ./dataset/two_splits/ec2_holdout/Km/train.csv \
  --output_folder ./dataset/treemodel_feat/ec2_holdout/Km \
  --output_file_name train \
  --enzyme_embedding_col ${PROT_EMBEDDING_MODEL}_prot_embedding_path \
  --smiles_embedding_col ${MOL_EMBEDDING_MODEL}_mol_embedding_path \
  --smiles_col $MOL_COL \
  --label_col log10_kmmM \
  --ec_col $EC_COL \
  --pretrained_ec_encoder_path $EC_CKPT \
  --ec_vocab_path $EC_PATH \
  --ec_dim $EC_DIM \
  --dropout $EC_DROPOUT \
  --seed $SEED \
  --device cuda

python ./scripts/get_treemodel_features.py \
  --file_path ./dataset/two_splits/ec2_holdout/Km/test.csv \
  --output_folder ./dataset/treemodel_feat/ec2_holdout/Km \
  --output_file_name test \
  --enzyme_embedding_col ${PROT_EMBEDDING_MODEL}_prot_embedding_path \
  --smiles_embedding_col ${MOL_EMBEDDING_MODEL}_mol_embedding_path \
  --smiles_col $MOL_COL \
  --label_col log10_kmmM \
  --ec_col $EC_COL \
  --pretrained_ec_encoder_path $EC_CKPT \
  --ec_vocab_path $EC_PATH \
  --ec_dim $EC_DIM \
  --dropout $EC_DROPOUT \
  --seed $SEED \
  --device cuda


python ./scripts/get_treemodel_features.py \
  --file_path ./dataset/two_splits/ec3_holdout/Km/train.csv \
  --output_folder ./dataset/treemodel_feat/ec3_holdout/Km \
  --output_file_name train \
  --enzyme_embedding_col ${PROT_EMBEDDING_MODEL}_prot_embedding_path \
  --smiles_embedding_col ${MOL_EMBEDDING_MODEL}_mol_embedding_path \
  --smiles_col $MOL_COL \
  --label_col log10_kmmM \
  --ec_col $EC_COL \
  --pretrained_ec_encoder_path $EC_CKPT \
  --ec_vocab_path $EC_PATH \
  --ec_dim $EC_DIM \
  --dropout $EC_DROPOUT \
  --seed $SEED \
  --device cuda

python ./scripts/get_treemodel_features.py \
  --file_path ./dataset/two_splits/ec3_holdout/Km/test.csv \
  --output_folder ./dataset/treemodel_feat/ec3_holdout/Km \
  --output_file_name test \
  --enzyme_embedding_col ${PROT_EMBEDDING_MODEL}_prot_embedding_path \
  --smiles_embedding_col ${MOL_EMBEDDING_MODEL}_mol_embedding_path \
  --smiles_col $MOL_COL \
  --label_col log10_kmmM \
  --ec_col $EC_COL \
  --pretrained_ec_encoder_path $EC_CKPT \
  --ec_vocab_path $EC_PATH \
  --ec_dim $EC_DIM \
  --dropout $EC_DROPOUT \
  --seed $SEED \
  --device cuda


python ./scripts/get_treemodel_features.py \
  --file_path ./dataset/two_splits/seq_idt/Km/train.csv \
  --output_folder ./dataset/treemodel_feat/seq_idt/Km \
  --output_file_name train \
  --enzyme_embedding_col ${PROT_EMBEDDING_MODEL}_prot_embedding_path \
  --smiles_embedding_col ${MOL_EMBEDDING_MODEL}_mol_embedding_path \
  --smiles_col $MOL_COL \
  --label_col log10_kmmM \
  --ec_col $EC_COL \
  --pretrained_ec_encoder_path $EC_CKPT \
  --ec_vocab_path $EC_PATH \
  --ec_dim $EC_DIM \
  --dropout $EC_DROPOUT \
  --seed $SEED \
  --device cuda

python ./scripts/get_treemodel_features.py \
  --file_path ./dataset/two_splits/seq_idt/Km/test.csv \
  --output_folder ./dataset/treemodel_feat/seq_idt/Km \
  --output_file_name test \
  --enzyme_embedding_col ${PROT_EMBEDDING_MODEL}_prot_embedding_path \
  --smiles_embedding_col ${MOL_EMBEDDING_MODEL}_mol_embedding_path \
  --smiles_col $MOL_COL \
  --label_col log10_kmmM \
  --ec_col $EC_COL \
  --pretrained_ec_encoder_path $EC_CKPT \
  --ec_vocab_path $EC_PATH \
  --ec_dim $EC_DIM \
  --dropout $EC_DROPOUT \
  --seed $SEED \
  --device cuda

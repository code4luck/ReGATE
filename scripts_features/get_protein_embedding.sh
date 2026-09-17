#!/bin/bash
# bash get_protein_embedding.sh
#  weight put in ./model_ckpt/prot_t5_xl 

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

python ./scripts/utils_protein_embedding.py \
  --file_path ./dataset/unique_catapro_all/unique_enzymes.csv \
  --save_folder ./dataset/catapro_all_prot_unique_embedding/prot_t5_xl \
  --model_name ./model_ckpt/prot_t5_xl \
  --device cuda \
  --seq_col Sequence
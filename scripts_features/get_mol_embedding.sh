#!/bin/bash
# bash get_mol_embedding.sh
#  weight put in ./model_ckpt/mol_t5 

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python ./scripts/utils_mol_embedding.py \
  --file_path ./dataset/unique_catapro_all/unique_molecules.csv \
  --save_folder ./dataset/catapro_all_mol_unique_embedding/mol_t5 \
  --model_name ./model_ckpt/mol_t5 \
  --device cuda \
  --mol_col Smiles
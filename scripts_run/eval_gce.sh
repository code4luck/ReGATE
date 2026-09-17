#!/bin/bash
# kcat seq_idt

CKPT="YOUR_CKPT_PATH"
python ./eval_ensemble/eval.py \
  --model_type gce \
  --ckpt_path "${CKPT}" \
  --data_path ./dataset/two_splits/seq_idt/kcat \
  --data_name kcat_seq_idt \
  --ec_vocab_path ./dataset/two_splits/clean_uniprot_ec_vocab.json \
  --label_col log10_kcat \
  --protein_embedding_col prot_t5_xl_prot_embedding_path \
  --mol_embedding_col mol_t5_mol_embedding_path \
  --prot_embedding_model prot_t5_xl \
  --mol_embedding_model mol_t5 \
  --protein_embedding_size 1024 \
  --mol_embedding_size 768 \
  --hidden_size 1280 \
  --dec_dim 128 \
  --output_dir ./eval_outputs/kcat_seq_idt
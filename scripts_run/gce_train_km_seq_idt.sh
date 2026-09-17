#!/bin/bash
# Hf镜像
export HF_ENDPOINT="https://hf-mirror.com"
export PYTHONPATH="${PBS_O_WORKDIR:-.}:${PYTHONPATH}"

DATA_NAME="km_seq_idt"
PROT_EMBEDDING_MODEL="prot_t5_xl"
MOL_EMBEDDING_MODEL="mol_t5"
LABEL_COL="log10_kmmM"
SEQ_COL="Sequence"
MOL_COL="Smiles"  
PROT_EMBED_SIZE=1024
MOL_EMBED_SIZE=768
BASIC_LOSS="mse"

DEC_DIM=128
HIDDEN_SIZE=768
MONITOR="val_loss"
OUTPUT_ROOT="./res/km_seq_idt"
SEED=3407
EC_PATH="./dataset/two_splits/clean_uniprot_ec_vocab.json"
DATA_PATH="./dataset/two_splits/seq_idt/Km"
EC_CKPT="./dataset/ec_cts_ckpt/ec_cts_hard_ep30_29.pt"

python ./src_gce_ec_cts/trainer.py \
  --data_path $DATA_PATH \
  --data_name $DATA_NAME \
  --ec_vocab_path $EC_PATH \
  --label_col $LABEL_COL \
  --sequence_col $SEQ_COL \
  --smiles_col $MOL_COL \
  --protein_embedding_col ${PROT_EMBEDDING_MODEL}_prot_embedding_path \
  --mol_embedding_col ${MOL_EMBEDDING_MODEL}_mol_embedding_path \
  --prot_embedding_model $PROT_EMBEDDING_MODEL \
  --mol_embedding_model $MOL_EMBEDDING_MODEL \
  --protein_embedding_size $PROT_EMBED_SIZE \
  --mol_embedding_size $MOL_EMBED_SIZE \
  --hidden_size $HIDDEN_SIZE \
  --ec_dim $DEC_DIM \
  --dropout 0.1 \
  --base_loss $BASIC_LOSS \
  --lr 3e-5 \
  --batch_size 32 \
  --num_workers 4 \
  --max_epochs 120 \
  --patience 40 \
  --monitor $MONITOR \
  --accelerator gpu \
  --devices 1 \
  --strategy auto \
  --precision 32 \
  --seed $SEED \
  --freeze_ec \
  --pretrained_ec_encoder_path $EC_CKPT \
  --output_dir $OUTPUT_ROOT/${DATA_NAME}_${PROT_EMBEDDING_MODEL}_${MOL_EMBEDDING_MODEL}
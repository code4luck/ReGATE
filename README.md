# Open code for ReGATE

**ReGATE** predicts enzyme kinetic parameters (*k*cat and *K*m) with hierarchical EC priors and heterogeneous ensembling.

It combines three parts:

1. **GCE** — a gated cross-expert network with a pretrained hierarchical EC encoder
2. **Tree model** — ExtraTrees on ProtT5 + EC + MolT5 + MACCS features
3. **FAISS retrieval** — *k*NN fusion on enzyme–molecule embeddings

---

## Pipeline

```text
unique sequences / SMILES
        ↓ ProtT5 / MolT5
split CSVs (matched by Sequence & Smiles)
        ↓
   ┌────┴────┐
  GCE
   ↓      ExtraTrees
FAISS kNN     
   └────┬────┘
        ↓  
     ensemble
```

All alignment uses `Sequence` and `Smiles`, not UniProt / substrate IDs (those can be one-to-many).

---



## Data

Thanks for Original records come from **CataPro.** We then split in :


| Split             | Path                             | Setting                  |
| ----------------- | -------------------------------- | ------------------------ |
| Sequence-identity | `dataset/two_splits/seq_idt`     | low sequence identity    |
| EC2 holdout       | `dataset/two_splits/ec2_holdout` | hierarchical EC-disjoint |
| EC3 holdout       | `dataset/two_splits/ec3_holdout` | hierarchical EC-disjoint |


Each split has `kcat/` and `Km/` with `train.csv`, `valid.csv`, `test.csv`.

Put pretrained weights here before running:

- ProtT5 → `model_ckpt/prot_t5_xl` ([Rostlab/prot_t5_xl_uniref50](https://huggingface.co/Rostlab/prot_t5_xl_uniref50))
- MolT5 → `model_ckpt/mol_t5` ([laituan245/molt5-base-smiles2caption](https://huggingface.co/laituan245/molt5-base-smiles2caption))
- Hierarchical EC encoder → `dataset/ec_cts_ckpt/ec_cts_hard_ep30_29.pt`

---



## Environment

Run all commands from the **repository root**.

**Training / tree / GCE eval** (`regate`):

```bash
conda create -n regate python=3.10
conda activate regate
pip install lightning==2.2.5
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.24.3 pandas==2.2.2 xgboost==1.6.2 \
  transformers==4.46.3 rdkit==2025.9.6 scikit-learn==1.4.2 scipy==1.10.1
```

**FAISS retrieval** (`faiss`):

```bash
conda create -n faiss python=3.10
conda activate faiss
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cu130
pip install faiss-gpu==1.14.3 numpy==2.2.6 pandas==2.3.3 scipy==1.15.3
```

CPU-only: install `faiss-cpu` and drop `--faiss_gpu` in the retrieval script.

---



## Reproduce (example: *k*cat, seq_idt)



### 1. Embeddings

```bash
conda activate regate
bash scripts_features/get_protein_embedding.sh
bash scripts_features/get_mol_embedding.sh
bash dataset/merg_path2csv.sh
```

This writes ProtT5 / MolT5 vectors for unique enzymes and molecules, then merges paths into split CSVs by `Sequence` / `Smiles`.

### 2. Tree features

```bash
bash scripts_features/get_treemodel_feat_kcat.sh
# optional: bash scripts_features/get_treemodel_feat_km.sh
```

Features are `[enzyme, EC, molecule, MACCS]`.

### 3. Train

```bash
bash scripts_run/gce_train_kcat_seq_idt.sh
bash scripts_run/get_treemodel_train_test_kcat.sh
```

### 4. GCE prediction

Edit `CKPT` in `scripts_run/eval_gce.sh`, then:

```bash
bash scripts_run/eval_gce.sh
```

### 5. FAISS retrieval

```bash
conda activate faiss
bash scripts_run/faiss_knn.sh
```

Output: `retrieval_faiss/kcat_seq_idt/faiss_knn.csv`  
(`y_final = (1−λ)·y_model + λ·y_knn`).

### 6. Ensemble

```bash
conda activate regate
python ensemble_gce_tree.py \
  --gce_file ./retrieval_faiss/kcat_seq_idt/faiss_knn.csv \
  --gce_pred_col y_final \
  --tree_file ./treemodel_res/seed42/seq_idt/kcat/test_predictions_mean.csv \
  --tree_pred_col prediction_mean \
  --save_file ./ensemble/kcat_seq_idt.csv \
  --lam 0.5
```


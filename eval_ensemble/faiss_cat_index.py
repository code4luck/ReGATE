
# KNN: y_final = (1-lambda)*y_model + lambda*y_knn
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import faiss

# faiss.contrib.torch_utils 让 GPU index 
try:
    import faiss.contrib.torch_utils  # noqa: F401
    _HAS_TORCH_UTILS = True
except Exception:
    _HAS_TORCH_UTILS = False
from scipy.stats import pearsonr, spearmanr


# --------------------------------------------------------------------------- #

def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """ (N, D)。"""
    den = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / den


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)



def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    pcc = float(pearsonr(y_true, y_pred)[0])
    scc = float(spearmanr(y_true, y_pred)[0])
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return {"pcc": pcc, "scc": scc, "mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


# --------------------------------------------------------------------------- #

@dataclass
class CatIndexConfig:
    
    index_method: str = "flat_ip"        # flat_l2 | flat_ip | ivfflat | ivfpq
    nlist: int = 256                    
    nprobe: int = 16                     
    m_pq: int = 16                      
    nbits: int = 8                      
    faiss_gpu: bool = False             

    # ----  embedding ----
    prot_embedding_col: str = "esm2_650m_prot_embedding_path"
    mol_embedding_col: str = "molformer_mol_embedding_path"
    embedding_root: str = "."           

    # ----  ----
    normalize_each_modality: bool = True  
    prot_weight: float = 1.0              
    mol_weight: float = 1.0

    # ---- ----
    top_k: int = 16
    temperature: float = 1.0              
    lambda_knn: float = 0.5               

    # ---- label ----
    label_col: str = "log10_kcat"         

    @property
    def metric_is_ip(self) -> bool:
        return self.index_method in ("flat_ip", "ivfflat", "ivfpq")



# --------------------------------------------------------------------------- #
# embedding load
# --------------------------------------------------------------------------- #
def load_pooled_embedding(path: str, root: str = ".") -> np.ndarray:
    """加载单个 .pt embedding ([L, dim] 或 [dim])，mean 池化为 [dim] 的 float32。"""
    full = path if os.path.isabs(path) else os.path.join(root, path)
    obj = torch.load(full, map_location="cpu", weights_only=False)
    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"Unexpected embedding type {type(obj)} at {full}")
    obj = obj.float()
    if obj.ndim == 1:          #  [dim]
        vec = obj
    elif obj.ndim == 2:        # [L, dim] -> mean pool
        vec = obj.mean(dim=0)
    else:
        # eg[1, L, dim]
        vec = obj.reshape(-1, obj.shape[-1]).mean(dim=0)
    return vec.numpy().astype("float32")


def _pooled_cache_path(path: str, root: str, cache_dir: str) -> str:
    """ embedding .pt -> per-unique cache: pooled/{model_dir}/{i}.npy"""
    full = os.path.normpath(path if os.path.isabs(path) else os.path.join(root, path))
    model_dir = os.path.basename(os.path.dirname(full))
    stem = os.path.splitext(os.path.basename(full))[0]  # e.g. 13018 from 13018.pt
    return os.path.join(cache_dir, "pooled", model_dir, f"{stem}.npy")


def get_pooled_cached(path: str, root: str = ".", cache_dir: str = "./_feat_cache") -> np.ndarray:
    """ .pt -> single pooling feature: {model_dir}/{i}.npy"""
    cache_file = _pooled_cache_path(path, root, cache_dir)
    if os.path.exists(cache_file):
        return np.load(cache_file)
    vec = load_pooled_embedding(path, root)
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    np.save(cache_file, vec)
    return vec


def build_pooled_from_csv(csv_path: str, cfg: CatIndexConfig,
                          cache_dir: str = "./_feat_cache"
                          ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """ csv -> (prot, mol) pooled features, keys, labels"""
    df = pd.read_csv(csv_path)
    prot = np.stack([
        get_pooled_cached(p, cfg.embedding_root, cache_dir)
        for p in df[cfg.prot_embedding_col]
    ]).astype("float32")
    mol = np.stack([
        get_pooled_cached(p, cfg.embedding_root, cache_dir)
        for p in df[cfg.mol_embedding_col]
    ]).astype("float32")
    keys = [f"{s}||{sm}" for s, sm in zip(df["Sequence"], df["Smiles"])]
    labels = df[cfg.label_col].to_numpy().astype("float32")
    return prot, mol, keys, labels

def assemble_features(prot: np.ndarray, mol: np.ndarray, cfg: CatIndexConfig) -> np.ndarray:
    """ (prot, mol) -> cat features"""
    prot = prot.astype("float32").copy()
    mol = mol.astype("float32").copy()
    if cfg.normalize_each_modality:
        prot = l2_normalize_np(prot)
        mol = l2_normalize_np(mol)
    prot *= cfg.prot_weight
    mol *= cfg.mol_weight
    return np.ascontiguousarray(np.concatenate([prot, mol], axis=1).astype("float32"))


def build_cat_features(csv_path: str, cfg: CatIndexConfig, cache_dir: str = "./_feat_cache"
                       ) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """ csv -> (X[N, Dp+Dm], keys, labels)"""
    prot, mol, keys, labels = build_pooled_from_csv(csv_path, cfg, cache_dir)
    X = assemble_features(prot, mol, cfg)
    assert len(keys) == X.shape[0] == labels.shape[0]
    return X, keys, labels

# --------------------------------------------------------------------------- #
# retriever
# --------------------------------------------------------------------------- #
class CatKNNRetriever:
    """
    single cat[prot, mol] retriever.
    - support flat_l2 / flat_ip / ivfflat / ivfpq
    - support cpu / gpu
    - build by input order, idx is row number, can trace keys/labels
    """

    def __init__(self, cfg: CatIndexConfig):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.index = None
        self.dim: Optional[int] = None
        self.keys: Optional[List[str]] = None
        self.labels: Optional[np.ndarray] = None
        self._on_gpu = False
        self._gpu_res = None

    # ---------- internal ---------- #
    def _make_index_cpu(self, dim: int):
        c = self.cfg
        if c.index_method == "flat_l2":
            return faiss.IndexFlatL2(dim)
        if c.index_method == "flat_ip":
            return faiss.IndexFlatIP(dim)

        metric = faiss.METRIC_INNER_PRODUCT if c.metric_is_ip else faiss.METRIC_L2
        if c.index_method == "ivfflat":
            quantizer = faiss.IndexFlatIP(dim) if c.metric_is_ip else faiss.IndexFlatL2(dim)
            return faiss.IndexIVFFlat(quantizer, dim, c.nlist, metric)
        if c.index_method == "ivfpq":
            if dim % c.m_pq != 0:
                raise ValueError(f"ivfpq 要求 dim({dim}) 能被 m_pq({c.m_pq}) 整除")
            quantizer = faiss.IndexFlatIP(dim) if c.metric_is_ip else faiss.IndexFlatL2(dim)
            # 显式传 metric, 否则默认 L2 与 quantizer 不一致
            return faiss.IndexIVFPQ(quantizer, dim, c.nlist, c.m_pq, c.nbits, metric)
        raise ValueError(f"Invalid index_method: {c.index_method}")

    def _to_gpu(self, index):
        if self.cfg.faiss_gpu and self.device.type == "cuda":
            try:
                self._gpu_res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(self._gpu_res, 0, index)
                return index, True
            except Exception as e:
                print(f"[warn] move index to gpu failed, fallback to cpu: {e}")
        return index, False

    def _prep(self, X: np.ndarray) -> np.ndarray:
        """IP metric -> L2 normalize = cosine."""
        Xn = np.ascontiguousarray(X.astype("float32"))
        if self.cfg.metric_is_ip:
            Xn = l2_normalize_np(Xn)
        return np.ascontiguousarray(Xn)

    # ---------- build ---------- #
    def build(self, X: np.ndarray, keys: List[str], labels: np.ndarray):
        assert X.ndim == 2 and len(keys) == X.shape[0] == labels.shape[0]
        self.dim = X.shape[1]
        xb = self._prep(X)

        index = self._make_index_cpu(self.dim)
        # IVF needs to be trained (train on cpu is most stable)
        if hasattr(index, "is_trained") and not index.is_trained:
            index.train(xb)

        index, self._on_gpu = self._to_gpu(index)
        if hasattr(index, "nprobe"):
            index.nprobe = int(self.cfg.nprobe)

        index.add(xb)  # insert by order -> id = 0..N-1 = row number
        self.index = index
        self.keys = list(keys)
        self.labels = np.asarray(labels, dtype="float32")
        print(f"[index] method={self.cfg.index_method} dim={self.dim} "
              f"N={index.ntotal} gpu={self._on_gpu}")
        return self

    @classmethod
    def from_csv(cls, csv_path: str, cfg: CatIndexConfig,
                 cache_dir: str = "./_feat_cache") -> "CatKNNRetriever":
        # build cat features, build seq and label by order
        X, keys, labels = build_cat_features(csv_path, cfg, cache_dir) 
        return cls(cfg).build(X, keys, labels)

    # ---------- search ---------- #
    def search(self, Q: np.ndarray, top_k: Optional[int] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """ return (D, I), shape (B, k). D is similarity in IP, L2 is squared distance."""
        assert self.index is not None, "index not built"
        if Q.ndim == 1:
            Q = Q[None, :]
        k = int(top_k or self.cfg.top_k)
        Qn = self._prep(Q)
        D, I = self.index.search(Qn, k)
        return D, I

    def _weights_from_D(self, D: np.ndarray) -> np.ndarray:
        """ metric aware -> normalized weights (B, k)"""
        if self.cfg.metric_is_ip:
            score = D                       # cosine, 
        else:
            score = -D                      # L2 squared distance, neg 
        return softmax_np(score / max(self.cfg.temperature, 1e-6), axis=-1)

    def predict_knn(self, Q: np.ndarray, top_k: Optional[int] = None
                    ) -> np.ndarray:
        """ return weighted KNN predictions, shape (B,)"""
        D, I = self.search(Q, top_k)
        w = self._weights_from_D(D)                 # (B, k)
        mask = (I >= 0).astype("float32")
        valid = mask.sum(axis=1)                      # number of valid neighbors for each query
        w = w * mask
        w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
        neigh_labels = np.where(I >= 0, self.labels[I], 0.0)
        y_knn = (w * neigh_labels).sum(axis=1)
        return y_knn, valid    

    def fuse(self, y_model, Q, top_k=None, lambda_knn=None):
        """y_final = (1-l)*y_model + l*y_knn"""
        l = self.cfg.lambda_knn if lambda_knn is None else lambda_knn
        y_knn, valid = self.predict_knn(Q, top_k)
        y_model = np.asarray(y_model, dtype="float32")
        y_final = (1.0 - l) * y_model + l * y_knn
        y_final = np.where(valid > 0, y_final, y_model)  # no neighbors -> pure model
        return y_final

    def retrieve_detailed(self, Q: np.ndarray, top_k: Optional[int] = None
                          ) -> List[List[Dict]]:
        """ return top-k neighbors details for each query (can trace key/label/score)"""
        D, I = self.search(Q, top_k)
        w = self._weights_from_D(D)
        out: List[List[Dict]] = []
        for b in range(D.shape[0]):
            cands = []
            for j, idx in enumerate(I[b]):
                if idx < 0:
                    continue
                cands.append({
                    "row": int(idx),
                    "key": self.keys[idx],
                    "label": float(self.labels[idx]),
                    "score": float(D[b, j]),
                    "weight": float(w[b, j]),
                })
            out.append(cands)
        return out


def main(args):
    cfg = CatIndexConfig(
        index_method=args.index_method, nlist=args.nlist, nprobe=args.nprobe,
        m_pq=args.m_pq, nbits=args.nbits, faiss_gpu=args.faiss_gpu,
        top_k=args.top_k, temperature=args.temperature, lambda_knn=args.lambda_knn,
        label_col=args.label_col,
        prot_embedding_col=args.prot_embedding_col, mol_embedding_col=args.mol_embedding_col,
        embedding_root=args.embedding_root,
        normalize_each_modality=args.normalize_each_modality,
        prot_weight=args.prot_weight, mol_weight=args.mol_weight,
    )
    # ---- knowledge base: train (if need merge valid, use from_csvs) ----
    retr = CatKNNRetriever.from_csv(args.train_csv_path, cfg)
    # ---- query side: all from pred_csv, ensure row alignment ----
    Xq, _, y_true = build_cat_features(args.pred_csv_path, cfg)
    pred_df = pd.read_csv(args.pred_csv_path)
    y_model = pred_df[args.pred_col].to_numpy().astype("float32")
    assert len(y_model) == len(Xq), "pred row number does not match feature"
    y_knn,_ = retr.predict_knn(Xq)
    y_final = retr.fuse(y_model, Xq)          # (1-λ)*model + λ*knn
    # ---- write predictions ----
    pred_df["y_true"] = y_true
    pred_df["y_model"] = y_model
    pred_df["y_knn"] = y_knn
    pred_df["y_final"] = y_final
    os.makedirs(os.path.dirname(args.output_csv_path) or ".", exist_ok=True)
    pred_df.to_csv(args.output_csv_path, index=False)
    # ---- calculate metrics: original model vs KNN fusion, each row ----
    rows = [
        {"method": "baseline_model", **compute_metrics(y_true, y_model)},
        {"method": "knn_only",       **compute_metrics(y_true, y_knn)},
        {"method": "knn_fused",      **compute_metrics(y_true, y_final)},
    ]
    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.splitext(args.output_csv_path)[0] + "_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(metrics_df.to_string(index=False))
    print(f"[saved] predictions -> {args.output_csv_path}")
    print(f"[saved] metrics     -> {metrics_path}")



# args
def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv_path", type=str, required=True)
    parser.add_argument("--pred_csv_path", type=str, required=True)
    parser.add_argument("--output_csv_path", type=str, required=True)
    parser.add_argument("--embedding_root", type=str, required=True)

    parser.add_argument("--index_method", type=str, default="flat_ip")
    parser.add_argument("--nlist", type=int, default=256)
    parser.add_argument("--nprobe", type=int, default=16)
    parser.add_argument("--m_pq", type=int, default=16)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--faiss_gpu", action="store_true") # add True
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lambda_knn", type=float, default=0.5)
    parser.add_argument("--label_col", type=str, default="log10_kcat")
    parser.add_argument("--prot_embedding_col", type=str, default="esm2_650m_prot_embedding_path")
    parser.add_argument("--mol_embedding_col", type=str, default="molformer_mol_embedding_path")
    parser.add_argument("--normalize_each_modality", action="store_true") # add True 
    parser.add_argument("--prot_weight", type=float, default=1.0)
    parser.add_argument("--mol_weight", type=float, default=1.0)
    parser.add_argument("--pred_col", type=str, default="prediction")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)
import os
import pandas as pd
import numpy as np


def check(len_target, len_red):
    if len_target != len_red:
        return False
    return True


def ensmble_gce_retrieval_tree_pred(
    gce_df,
    gce_pred_col,
    tree_df,
    tree_pred_col,
    save_file,
    lam=0.5,
    label_col_gce="y_true",
    label_col_tree="label",
    ens_pred_col="ensemble_prediction",
):
    """Mean ensemble: ens = lam * gce_pred + (1 - lam) * tree_pred, and save to save_file."""
    if not check(len(gce_df), len(tree_df)):
        raise ValueError(
            f"row count mismatch: gce={len(gce_df)} vs tree={len(tree_df)}"
        )
    if gce_pred_col not in gce_df.columns:
        raise KeyError(f"gce missing column {gce_pred_col!r}")
    if tree_pred_col not in tree_df.columns:
        raise KeyError(f"tree missing column {tree_pred_col!r}")
    if not (0.0 <= lam <= 1.0):
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    gce_pred = pd.to_numeric(gce_df[gce_pred_col], errors="coerce").to_numpy(dtype=float)
    tree_pred = pd.to_numeric(tree_df[tree_pred_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(gce_pred).all() or not np.isfinite(tree_pred).all():
        raise ValueError("non-finite values found in predictions")

    # (optional: confirm label row alignment)
    if label_col_gce in gce_df.columns and label_col_tree in tree_df.columns:
        gce_y = pd.to_numeric(gce_df[label_col_gce], errors="coerce").to_numpy(dtype=float)
        tree_y = pd.to_numeric(tree_df[label_col_tree], errors="coerce").to_numpy(dtype=float)
        if np.nanmax(np.abs(gce_y - tree_y)) > 1e-5:
            raise ValueError("gce/tree labels are not row-aligned")

    ens_pred = lam * gce_pred + (1.0 - lam) * tree_pred

    out = pd.DataFrame(
        {
            "y_true": pd.to_numeric(gce_df[label_col_gce], errors="coerce")
            if label_col_gce in gce_df.columns
            else pd.to_numeric(tree_df[label_col_tree], errors="coerce"),
            "gce_y_final": gce_pred,
            "tree_prediction": tree_pred,
            ens_pred_col: ens_pred,
            "lam": lam
        }
    )
    # (if similarity_bin exists, include it for subsequent binning analysis)    
    if "similarity_bin" in gce_df.columns:
        out["similarity_bin"] = gce_df["similarity_bin"].to_numpy()
    elif "similarity_bin" in tree_df.columns:
        out["similarity_bin"] = tree_df["similarity_bin"].to_numpy()

    save_dir = os.path.dirname(os.path.abspath(save_file))
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    out.to_csv(save_file, index=False)
    return out

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gce_file", type=str, required=True)
    parser.add_argument("--gce_pred_col", type=str, required=True)
    parser.add_argument("--tree_file", type=str, required=True)
    parser.add_argument("--tree_pred_col", type=str, required=True)
    parser.add_argument("--save_file", type=str, required=True)
    parser.add_argument("--lam", type=float, required=True)
    args = parser.parse_args()
    gce_df = pd.read_csv(args.gce_file)
    tree_df = pd.read_csv(args.tree_file)
    ensmble_gce_retrieval_tree_pred(gce_df, args.gce_pred_col, tree_df, args.tree_pred_col, args.save_file, args.lam)
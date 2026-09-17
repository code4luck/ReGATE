
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd


# -----------------------------
# Utils
# -----------------------------
def _extract_model_name_from_path(path_str: str) -> str:
    """
    
    ./.../ankh_large/7.pt -> ankh_large
    """
    p = Path(str(path_str).strip())
    if p.parent.name == "":
        raise ValueError(f"Cannot extract model name from path: {path_str}")
    return p.parent.name


def _detect_path_col(df: pd.DataFrame, kind: str) -> str:
    """
    kind: 'prot' or 'mol'
    
    """
    if kind == "prot":
        preferred = [
            "prot_embedding_path",
            "protein_embedding_path",
        ]
        keywords = ["prot_embedding_path", "protein_embedding_path"]
    elif kind == "mol":
        preferred = [
            "mol_embedding_path",
            "molecule_embedding_path",
        ]
        keywords = ["mol_embedding_path", "molecule_embedding_path"]
    else:
        raise ValueError("kind must be 'prot' or 'mol'")

    cols = df.columns.tolist()

    # first find the exact column name
    for c in preferred:
        if c in cols:
            return c

    # then find the column name contains the keywords
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in keywords):
            return c

    # finally, find the column name contains the embedding_path
    candidates = [c for c in cols if "embedding_path" in c.lower()]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple possible path columns found for {kind}: {candidates}. "
            f"Please rename columns clearly."
        )

    raise ValueError(f"No embedding path column found for {kind} in columns: {cols}")


def _load_and_normalize_mapping(
    map_csvs: List[str],
    key_col: str,
    kind: str,   # 'prot' or 'mol'
    check_path_exists: bool = False,
) -> pd.DataFrame:
    """
    return the standardized long table:
    columns = [key_col, 'model', 'path']
    """
    parts = []
    for f in map_csvs:
        mdf = pd.read_csv(f)
        if key_col not in mdf.columns:
            raise ValueError(f"[{f}] missing key column: {key_col}")

        path_col = _detect_path_col(mdf, kind=kind)

        cur = mdf[[key_col, path_col]].copy()
        cur = cur.dropna(subset=[key_col, path_col])
        cur[key_col] = cur[key_col].astype(str)
        cur[path_col] = cur[path_col].astype(str).str.strip()

        if check_path_exists:
            not_exists = cur[~cur[path_col].apply(os.path.exists)]
            if len(not_exists) > 0:
                sample = not_exists[path_col].head(5).tolist()
                raise FileNotFoundError(
                    f"[{f}] found {len(not_exists)} non-existing paths. sample={sample}"
                )

        cur["model"] = cur[path_col].apply(_extract_model_name_from_path)
        cur = cur.rename(columns={path_col: "path"})
        parts.append(cur)

    if not parts:
        # return the empty standardized table
        return pd.DataFrame(columns=[key_col, "model", "path"])

    long_df = pd.concat(parts, axis=0, ignore_index=True)

    # conflict detection: same key + model has multiple different paths -> error
    conflict = (
        long_df.groupby([key_col, "model"])["path"]
        .nunique()
        .reset_index(name="n_path")
    )
    conflict = conflict[conflict["n_path"] > 1]
    if len(conflict) > 0:
        show = conflict.head(10)
        raise ValueError(
            "Found conflicting mappings (same key+model has multiple paths). "
            f"Please clean mapping files first. sample:\n{show}"
        )

    # remove duplicates and keep unique
    long_df = long_df.drop_duplicates(subset=[key_col, "model", "path"])
    return long_df


def _pivot_mapping_to_wide(
    long_df: pd.DataFrame,
    key_col: str,
    suffix: str,  # 'prot_embedding_path' or 'mol_embedding_path'
) -> pd.DataFrame:
    """
    long table -> wide table:
      [key, model, path] -> [key, {model}_{suffix}, ...]
    """
    if len(long_df) == 0:
        return pd.DataFrame(columns=[key_col])

    wide = long_df.pivot(index=key_col, columns="model", values="path")
    wide = wide.reset_index()

    # rename the model columns
    rename_map = {}
    for c in wide.columns:
        if c == key_col:
            continue
        rename_map[c] = f"{c}_{suffix}"
    wide = wide.rename(columns=rename_map)
    return wide


def _merge_wide_into_target(
    target_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    key_col: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    left join the wide_df to the target_df
    overwrite=False 时:
      - if the target has the same column name, only fill the empty values, not overwrite the non-empty values
    overwrite=True 时:
      - the mapping result directly overwrites the same column name
    """
    if key_col not in target_df.columns:
        raise ValueError(f"Target csv missing key column: {key_col}")

    if len(wide_df.columns) <= 1:
        return target_df.copy()

    merged = target_df.merge(wide_df, on=key_col, how="left", suffixes=("", "__new"))

    for c in wide_df.columns:
        if c == key_col:
            continue
        new_c = f"{c}__new"

        if c in target_df.columns:
            if overwrite:
                merged[c] = merged[new_c].where(merged[new_c].notna(), merged[c])
            else:
                # only fill the empty values
                merged[c] = merged[c].where(merged[c].notna(), merged[new_c])
            merged = merged.drop(columns=[new_c])
        else:
            merged = merged.rename(columns={new_c: c})

    return merged


def _report_coverage(
    before_df: pd.DataFrame,
    after_df: pd.DataFrame,
    key_col: str,
    added_cols: List[str],
    title: str,
) -> None:
    print(f"\n[{title}]")
    print(f"rows: {len(after_df)}")
    for c in added_cols:
        if c not in after_df.columns:
            print(f"  - {c}: column not created")
            continue
        cnt = after_df[c].notna().sum()
        print(f"  - {c}: {cnt}/{len(after_df)} matched ({cnt / max(len(after_df), 1):.2%})")


# -----------------------------
# Public APIs
# -----------------------------
def add_protein_embedding_paths(
    target_df: pd.DataFrame,
    prot_map_csvs: List[str],
    sequence_col: str = "Sequence",
    overwrite: bool = False,
    check_path_exists: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    merge the protein embedding paths by Sequence, add new columns:
      {model}_prot_embedding_path
    """
    long_df = _load_and_normalize_mapping(
        map_csvs=prot_map_csvs,
        key_col=sequence_col,
        kind="prot",
        check_path_exists=check_path_exists,
    )
    wide_df = _pivot_mapping_to_wide(
        long_df=long_df,
        key_col=sequence_col,
        suffix="prot_embedding_path",
    )
    out_df = _merge_wide_into_target(
        target_df=target_df,
        wide_df=wide_df,
        key_col=sequence_col,
        overwrite=overwrite,
    )
    new_cols = [c for c in wide_df.columns if c != sequence_col]
    return out_df, new_cols


def add_mol_embedding_paths(
    target_df: pd.DataFrame,
    mol_map_csvs: List[str],
    smiles_col: str = "Smiles",
    overwrite: bool = False,
    check_path_exists: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    merge the molecule embedding paths by Smiles, add new columns:
      {model}_mol_embedding_path
    """
    long_df = _load_and_normalize_mapping(
        map_csvs=mol_map_csvs,
        key_col=smiles_col,
        kind="mol",
        check_path_exists=check_path_exists,
    )
    wide_df = _pivot_mapping_to_wide(
        long_df=long_df,
        key_col=smiles_col,
        suffix="mol_embedding_path",
    )
    out_df = _merge_wide_into_target(
        target_df=target_df,
        wide_df=wide_df,
        key_col=smiles_col,
        overwrite=overwrite,
    )
    new_cols = [c for c in wide_df.columns if c != smiles_col]
    return out_df, new_cols


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple embedding-path csvs into target csv by Sequence/Smiles."
    )
    parser.add_argument("--target_csv", type=str, required=True, help="the target csv")
    parser.add_argument("--output_csv", type=str, required=True, help="the output csv")

    parser.add_argument(
        "--prot_map_csvs", type=str, nargs="*", default=[],
        help="the protein embedding path record csv (can be multiple)"
    )
    parser.add_argument(
        "--mol_map_csvs", type=str, nargs="*", default=[],
        help="the molecule embedding path record csv (can be multiple)"
    )

    parser.add_argument("--sequence_col", type=str, default="Sequence")
    parser.add_argument("--smiles_col", type=str, default="Smiles")
    parser.add_argument("--overwrite", action="store_true", help="overwrite the non-empty values")
    parser.add_argument("--check_path_exists", action="store_true", help="check if the path exists")


    parser.add_argument("--strict_coverage", action="store_true")

    args = parser.parse_args()

    target_df = pd.read_csv(args.target_csv)
    before_df = target_df.copy()

    all_new_cols = []

    if args.prot_map_csvs:
        target_df, prot_cols = add_protein_embedding_paths(
            target_df=target_df,
            prot_map_csvs=args.prot_map_csvs,
            sequence_col=args.sequence_col,
            overwrite=args.overwrite,
            check_path_exists=args.check_path_exists,
        )
        all_new_cols.extend(prot_cols)
        _report_coverage(before_df, target_df, args.sequence_col, prot_cols, "Protein Merge")

    if args.mol_map_csvs:
        target_df, mol_cols = add_mol_embedding_paths(
            target_df=target_df,
            mol_map_csvs=args.mol_map_csvs,
            smiles_col=args.smiles_col,
            overwrite=args.overwrite,
            check_path_exists=args.check_path_exists,
        )
        all_new_cols.extend(mol_cols)
        _report_coverage(before_df, target_df, args.smiles_col, mol_cols, "Molecule Merge")

    if args.strict_coverage and all_new_cols:
        for c in all_new_cols:
            miss = target_df[c].isna().sum()
            if miss > 0:
                raise ValueError(
                    f"strict_coverage=True but column `{c}` has {miss} unmatched rows"
                )

    target_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved merged csv to: {args.output_csv}")


if __name__ == "__main__":
    main()
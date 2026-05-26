"""
Split processed omics + label CSVs into stratified K folds once on disk.

Input layout (same as BRCADataModule):
  - Each omic file: rows = features (genes), columns = sample IDs
  - Label file: one column (e.g. Label), one row per sample, same order as omic columns

Output layout under {input_dir}/{output_subdir}/:
  fold_1/ ... fold_K/
    <stem>_train.csv / <stem>_val.csv for each omic
    <label_stem>_train.csv / <label_stem>_val.csv

Training: set data.pre_split_base to this folder and keep use_stratified_kfold + n_folds
in config; train.py will load fold_i/ without re-splitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _find_label_csv(input_dir: Path) -> Path:
    candidates = sorted(input_dir.glob("*label*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No *label*.csv under {input_dir}")
    if len(candidates) > 1:
        names = [p.name for p in candidates]
        raise ValueError(f"Multiple label candidates {names}; pass --label explicitly.")
    return candidates[0]


def _omics_csvs(input_dir: Path, label_path: Path) -> list[Path]:
    paths = sorted(p for p in input_dir.glob("*.csv") if p.resolve() != label_path.resolve())
    if not paths:
        raise FileNotFoundError(f"No omics CSVs in {input_dir} (excluding {label_path.name}).")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Write stratified K-fold CSV splits to disk.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Folder with processed omics + label CSVs (e.g. csv/processed/BRCA).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label filename inside input-dir (default: sole *label*.csv).",
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="5-fold",
        help="Created inside input-dir (default: 5-fold).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(str(input_dir))

    label_path = (input_dir / args.label) if args.label else _find_label_csv(input_dir)
    if not label_path.is_file():
        raise FileNotFoundError(label_path)

    omics_paths = _omics_csvs(input_dir, label_path)
    omics_dfs = {p.stem: pd.read_csv(p, index_col=0) for p in omics_paths}
    stems = list(omics_dfs.keys())
    ref = omics_dfs[stems[0]]
    sample_ids = list(ref.columns)
    n_samples = len(sample_ids)

    y_df = pd.read_csv(label_path)
    if len(y_df) != n_samples:
        raise ValueError(
            f"Label rows ({len(y_df)}) != omic columns ({n_samples}) in {omics_paths[0].name}."
        )
    y = y_df.iloc[:, 0].to_numpy()

    for stem in stems[1:]:
        df = omics_dfs[stem]
        if list(df.columns) != sample_ids:
            raise ValueError(f"Column order mismatch: {stems[0]} vs {stem}.")

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    splits = list(skf.split(np.arange(n_samples), y))

    out_root = input_dir / args.output_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_num = fold_idx + 1
        fold_dir = out_root / f"fold_{fold_num}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_cols = [sample_ids[i] for i in train_idx]
        val_cols = [sample_ids[i] for i in val_idx]

        for stem, df in omics_dfs.items():
            df[train_cols].to_csv(fold_dir / f"{stem}_train.csv")
            df[val_cols].to_csv(fold_dir / f"{stem}_val.csv")

        y_train = y_df.iloc[train_idx].reset_index(drop=True)
        y_val = y_df.iloc[val_idx].reset_index(drop=True)
        lstem = label_path.stem
        y_train.to_csv(fold_dir / f"{lstem}_train.csv", index=False)
        y_val.to_csv(fold_dir / f"{lstem}_val.csv", index=False)

    print(f"Wrote {args.n_folds} folds under {out_root}")


if __name__ == "__main__":
    main()

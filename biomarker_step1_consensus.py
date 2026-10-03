"""
5-fold consensus gene list from input-gradient saliency.

For each fold, compute mean |d logit / d x| on the validation set
(true-label attribution). A feature is kept if it appears in the per-omic
top-k for at least `min_folds` folds.
"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from train import set_seed
from visualize_attention import (
    build_datamodule,
    compute_gradient_feature_importance,
    load_model_from_checkpoint,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="5-fold consensus gene list from input-gradient saliency."
    )
    p.add_argument(
        "--config",
        type=str,
        default="configs/config_self_attn_cross_attn_brca.yaml",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Root lightning_logs dir (default: logging.log_dir from config).",
    )
    p.add_argument(
        "--log-name",
        type=str,
        default=None,
        help="Run name prefix (default: logging.log_name from config).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="attention_plots/biomarker_step1_consensus",
    )
    p.add_argument("--top-k-features", type=int, default=20)
    p.add_argument(
        "--min-folds",
        type=int,
        default=3,
        help="Keep features present in top-k of at least this many folds.",
    )
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument(
        "--class-names",
        type=str,
        default="LumA,LumB,HER2,Basal",
        help="Comma-separated class names in label index order.",
    )
    p.add_argument(
        "--checkpoint-metric",
        type=str,
        default="f1",
        choices=["f1", "acc"],
        help="Which best_* checkpoint family to use per fold.",
    )
    return p.parse_args()


def find_best_checkpoint(fold_dir: Path, metric: str = "f1") -> Path:
    """Pick best_val_{f1|acc}_*.ckpt with highest metric in the filename."""
    pattern = f"best_val_{metric}_*.ckpt"
    ckpts = list((fold_dir / "checkpoints").glob(pattern))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints matching {pattern} under {fold_dir}")

    key = f"val_{metric}="
    scored = []
    for ckpt in ckpts:
        m = re.search(rf"{key}(\d+\.\d+)", ckpt.name)
        score = float(m.group(1)) if m else -1.0
        scored.append((score, ckpt))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def load_feature_name_map_from_fold(cfg, fold_index_zero_based: int) -> dict[str, list[str]]:
    """Feature names from fold train CSVs (index column)."""
    from train import extract_omics_config, resolve_pre_split_fold_dir

    fold_dir = resolve_pre_split_fold_dir(cfg["data"], fold_index_zero_based)
    if fold_dir is None:
        raise ValueError("data.pre_split_base is required for consensus over folds.")
    fold_dir = Path(fold_dir)
    omics_config = extract_omics_config(cfg["data"])
    feature_name_map = {}
    for omic, path in omics_config.items():
        stem = Path(path).stem
        train_csv = fold_dir / f"{stem}_train.csv"
        if not train_csv.is_file():
            raise FileNotFoundError(f"Missing {train_csv}")
        df = pd.read_csv(train_csv, index_col=0, usecols=[0])
        feature_name_map[omic] = df.index.astype(str).tolist()
    return feature_name_map


def feature_label(omic: str, idx: int, feature_name_map: dict[str, list[str]] | None) -> str:
    if feature_name_map and omic in feature_name_map:
        names = feature_name_map[omic]
        if 0 <= idx < len(names):
            return names[idx]
    return f"{omic}_feat_{idx}"


def collect_topk_per_fold(
    importance: dict,
    omic_names: list[str],
    top_k: int,
    feature_name_map: dict[str, list[str]],
) -> list[dict]:
    """Flatten top-k (class, omic, feature) rows for one fold."""
    rows = []
    for c, omic_dict in importance.items():
        for omic in omic_names:
            if omic not in omic_dict:
                continue
            scores = np.asarray(omic_dict[omic]).reshape(-1)
            k = min(top_k, len(scores))
            top_idx = np.argsort(-scores)[:k]
            for rank, idx in enumerate(top_idx, start=1):
                rows.append(
                    {
                        "class_idx": int(c),
                        "omic": omic,
                        "feature_idx": int(idx),
                        "feature": feature_label(omic, int(idx), feature_name_map),
                        "rank": rank,
                        "mean_abs_grad": float(scores[idx]),
                    }
                )
    return rows


def build_consensus(all_fold_rows: list[pd.DataFrame], min_folds: int, n_folds: int) -> pd.DataFrame:
    """
    Aggregate fold top-k tables into consensus features.

    A (class, omic, feature_idx) is kept if it appears in >= min_folds folds.
    """
    if not all_fold_rows:
        return pd.DataFrame()

    df = pd.concat(all_fold_rows, ignore_index=True)
    group_cols = ["class_idx", "omic", "feature_idx", "feature"]
    agg = (
        df.groupby(group_cols, as_index=False)
        .agg(
            n_folds=("fold", "nunique"),
            mean_abs_grad=("mean_abs_grad", "mean"),
            std_abs_grad=("mean_abs_grad", "std"),
            mean_rank=("rank", "mean"),
            folds=("fold", lambda s: ",".join(str(int(x)) for x in sorted(s.unique()))),
        )
    )
    agg["std_abs_grad"] = agg["std_abs_grad"].fillna(0.0)
    agg["n_folds_total"] = n_folds
    consensus = agg[agg["n_folds"] >= min_folds].copy()
    consensus = consensus.sort_values(
        ["class_idx", "omic", "n_folds", "mean_abs_grad"],
        ascending=[True, True, False, False],
    )
    return consensus.reset_index(drop=True)


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    if not cfg["data"].get("pre_split_base"):
        raise ValueError("Config must set data.pre_split_base for 5-fold consensus.")

    seed = cfg["train"].get("seed", 42)
    set_seed(seed)

    log_dir = args.log_dir or cfg.get("logging", {}).get("log_dir", "lightning_logs")
    log_name = args.log_name or cfg.get("logging", {}).get("log_name", "brca")
    n_folds = int(cfg["data"].get("n_folds", 5))
    num_classes = int(cfg["model"]["num_classes"])

    names = [n.strip() for n in args.class_names.split(",") if n.strip()]
    class_name_map = {i: names[i] for i in range(min(len(names), num_classes))}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(cfg)
    cfg["train"]["num_workers"] = 0

    fold_tables = []
    feature_name_map = None

    for fold_idx in range(n_folds):
        fold_num = fold_idx + 1
        fold_log = Path(log_dir) / f"{log_name}_fold_{fold_num}"
        ckpt = find_best_checkpoint(fold_log, metric=args.checkpoint_metric)
        print(f"\n=== Fold {fold_num}: {ckpt.name} ===")

        cfg["data"]["current_fold"] = fold_idx
        dm = build_datamodule(cfg)
        model = load_model_from_checkpoint(cfg, dm, str(ckpt), device)

        if feature_name_map is None:
            feature_name_map = load_feature_name_map_from_fold(cfg, fold_idx)

        importance, omic_names = compute_gradient_feature_importance(
            model=model,
            dataloader=dm.val_dataloader(),
            device=device,
            num_classes=num_classes,
            max_batches=args.max_batches,
            use_true_label=True,
        )

        rows = collect_topk_per_fold(
            importance=importance,
            omic_names=omic_names,
            top_k=args.top_k_features,
            feature_name_map=feature_name_map,
        )
        fold_df = pd.DataFrame(rows)
        fold_df["fold"] = fold_num
        fold_df["class_name"] = fold_df["class_idx"].map(
            lambda i: class_name_map.get(int(i), f"class_{i}")
        )
        fold_path = output_dir / f"topk_fold_{fold_num}.csv"
        fold_df.to_csv(fold_path, index=False)
        print(f"Saved {fold_path} ({len(fold_df)} rows)")
        fold_tables.append(fold_df)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    consensus = build_consensus(fold_tables, min_folds=args.min_folds, n_folds=n_folds)
    consensus["class_name"] = consensus["class_idx"].map(
        lambda i: class_name_map.get(int(i), f"class_{i}")
    )
    consensus_path = output_dir / "consensus_features.csv"
    consensus.to_csv(consensus_path, index=False)
    print(f"\nConsensus features (>= {args.min_folds}/{n_folds} folds): {len(consensus)}")
    print(f"Saved {consensus_path}")

    if not consensus.empty:
        summary = (
            consensus.groupby(["class_name", "omic"])
            .size()
            .reset_index(name="n_consensus_features")
        )
        summary_path = output_dir / "consensus_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved {summary_path}")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

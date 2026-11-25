"""
Step 1 — 5-fold consensus gradient feature importance + lollipop plots per subtype.

For each fold, compute mean |∂logit/∂x| on the val set (true-label attribution).
A feature is kept if it appears in the per-omic top-k for at least `min_folds` folds.
"""

from __future__ import annotations

import argparse
import copy
import os
import re
from collections import defaultdict
from pathlib import Path, PureWindowsPath

import matplotlib.pyplot as plt
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


OMIC_COLORS = {
    "mRNA": "#1f77b4",
    "CNV": "#ff7f0e",
    "Proteomics": "#2ca02c",
    "Methylation": "#d62728",
    "miRNA": "#9467bd",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="5-fold consensus gradient biomarkers + lollipop plots per subtype."
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
    p.add_argument(
        "--plot-top-n",
        type=int,
        default=20,
        help="Max consensus features to show on each subtype lollipop.",
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
        stem = PureWindowsPath(path).stem  # config paths may be Windows-style
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


def _feature_label(row) -> str:
    return f"{row.feature}  {int(row.n_folds)}/{int(row.n_folds_total)}"


BIO_GROUPS = [
    {
        "name": "Mitotic / cell-cycle",
        "genes": {
            "ASPM", "CEP55", "CENPE", "NUF2", "DLGAP5", "CDKN3", "KIF11",
            "GTSE1", "SGO1", "HASPIN", "GAS2L3",
        },
        "face": "#d0e2f3",
        "edge": "#4a86c8",
    },
    {
        "name": "17q12 amplicon",
        "genes": {
            "ERBB2", "GRB7", "STARD3", "MIEN1", "PGAP3", "TCAP", "PNMT",
            "CDK12", "GSDMB", "PPP1R1B", "IKZF3", "ZPBP2", "NEUROD2", "LRRC3C",
        },
        "face": "#fce5cd",
        "edge": "#e69138",
    },
    {
        "name": "Signaling / lineage",
        "genes": {
            "P53", "TP53", "INPP4B", "EEF2", "MEK1_pS217S221", "MEK1",
            "GATA3", "TUBERIN_pT1462", "TUBERIN", "BCL2", "CKIT", "KIT", "MIA",
        },
        "face": "#ead1dc",
        "edge": "#a64d79",
    },
]


def _assign_group(feature: str) -> dict | None:
    """Map feature to a biological story group for soft background bands."""
    order = ["17q12 amplicon", "Signaling / lineage", "Mitotic / cell-cycle"]
    by_name = {g["name"]: g for g in BIO_GROUPS}
    for name in order:
        g = by_name[name]
        if feature in g["genes"] or feature.split("_")[0] in g["genes"]:
            return g
    return None


def _annotate_groups(ax, plot_df: pd.DataFrame, _xmax: float = 0.0):
    """Soft background bands for related gene groups (no in-plot markers)."""
    runs = []
    cur = None
    for i, row in enumerate(plot_df.itertuples()):
        g = _assign_group(str(row.feature))
        if cur is None:
            cur = [g, i, i]
        elif (g is None and cur[0] is None) or (
            g is not None and cur[0] is not None and g["name"] == cur[0]["name"]
        ):
            cur[2] = i
        else:
            runs.append(cur)
            cur = [g, i, i]
    if cur is not None:
        runs.append(cur)

    for g, y0, y1 in runs:
        if g is None or y1 - y0 < 1:
            continue
        ax.axhspan(
            y0 - 0.42,
            y1 + 0.42,
            facecolor=g["face"],
            edgecolor="none",
            alpha=0.28,
            zorder=0,
        )


def _prepare_plot_df(class_df: pd.DataFrame, plot_top_n: int, group_sort: bool = False) -> pd.DataFrame:
    """Rank by mean |grad|; optionally cluster by biological group for cleaner bands."""
    df = class_df.copy()
    if group_sort:
        order = {
            "Mitotic / cell-cycle": 0,
            "17q12 amplicon": 1,
            "Signaling / lineage": 2,
            None: 3,
        }

        def _key(feat: str) -> int:
            g = _assign_group(str(feat))
            return order[g["name"] if g else None]

        df["_g"] = df["feature"].map(_key)
        df = df.sort_values(["_g", "mean_abs_grad"], ascending=[True, False])
        df = df.drop(columns=["_g"])
    else:
        df = df.sort_values("mean_abs_grad", ascending=False)
    plot_df = df.head(plot_top_n)
    return plot_df.iloc[::-1].reset_index(drop=True)



def plot_lollipop_per_subtype(
    consensus: pd.DataFrame,
    class_name_map: dict[int, str],
    output_dir: Path,
    plot_top_n: int,
    annotated: bool = False,
):
    """Single figure with a 2x2 lollipop panel — one subplot per subtype."""
    output_dir.mkdir(parents=True, exist_ok=True)

    class_ids = sorted(class_name_map.keys())
    if not class_ids:
        class_ids = sorted(consensus["class_idx"].unique().tolist())

    top_n = min(plot_top_n, 12)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    omics_seen = set()
    for ax, class_idx in zip(axes, class_ids):
        class_label = class_name_map.get(int(class_idx), f"class_{class_idx}")
        class_df = consensus[consensus["class_idx"] == class_idx]
        if class_df.empty:
            ax.set_title(class_label)
            ax.text(0.5, 0.5, "No consensus features", ha="center", va="center")
            ax.axis("off")
            continue

        plot_df = _prepare_plot_df(class_df, top_n, group_sort=annotated)
        y = np.arange(len(plot_df))
        labels = [_feature_label(row) for row in plot_df.itertuples()]
        colors = [OMIC_COLORS.get(row.omic, "#333333") for row in plot_df.itertuples()]
        sizes = 28 + 10 * plot_df["n_folds"].to_numpy()
        omics_seen.update(plot_df["omic"].unique())
        vals = plot_df["mean_abs_grad"].to_numpy()
        xmax = float(vals.max()) if len(vals) else 1.0

        if annotated:
            _annotate_groups(ax, plot_df, xmax)

        ax.hlines(y=y, xmin=0, xmax=vals, color=colors, linewidth=1.6, alpha=0.9, zorder=2)
        ax.scatter(
            vals,
            y,
            s=sizes,
            c=colors,
            zorder=3,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel(r"Mean $|\partial\mathrm{logit}/\partial x|$", fontsize=9)
        ax.set_title(class_label, fontsize=12, fontweight="bold")
        ax.set_xlim(0, xmax * 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.35, zorder=1)

    for j in range(len(class_ids), 4):
        axes[j].axis("off")

    omic_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=OMIC_COLORS.get(omic, "#333333"),
            markersize=8,
            label=omic,
        )
        for omic in sorted(omics_seen)
    ]
    if annotated:
        for g in BIO_GROUPS:
            omic_handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=g["face"], edgecolor="none",
                              alpha=0.8, label=g["name"])
            )
    if omic_handles:
        fig.legend(
            handles=omic_handles,
            loc="lower center",
            ncol=min(6, len(omic_handles)),
            frameon=False,
            fontsize=8,
        )

    fig.suptitle(
        "Consensus gradient biomarkers (5-fold; top-12; panel-specific x-scales)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    out_name = (
        "lollipop_consensus_all_subtypes_annotated.png"
        if annotated
        else "lollipop_consensus_all_subtypes.png"
    )
    out = output_dir / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_hbar_per_subtype(
    consensus: pd.DataFrame,
    class_name_map: dict[int, str],
    output_dir: Path,
    plot_top_n: int,
    annotated: bool = False,
):
    """2x2 horizontal bars; optional light biological-group annotations."""
    output_dir.mkdir(parents=True, exist_ok=True)
    class_ids = sorted(class_name_map.keys())
    top_n = min(plot_top_n, 12)
    fig_w = 14.5 if annotated else 14

    fig, axes = plt.subplots(2, 2, figsize=(fig_w, 10))
    axes = axes.flatten()
    omics_seen = set()

    for ax, class_idx in zip(axes, class_ids):
        class_label = class_name_map.get(int(class_idx), f"class_{class_idx}")
        class_df = consensus[consensus["class_idx"] == class_idx]
        if class_df.empty:
            ax.axis("off")
            continue
        plot_df = _prepare_plot_df(class_df, top_n, group_sort=annotated)
        y = np.arange(len(plot_df))
        colors = [OMIC_COLORS.get(row.omic, "#333333") for row in plot_df.itertuples()]
        omics_seen.update(plot_df["omic"].unique())
        vals = plot_df["mean_abs_grad"].to_numpy()
        xmax = float(vals.max()) if len(vals) else 1.0

        if annotated:
            _annotate_groups(ax, plot_df, xmax)

        ax.barh(
            y,
            vals,
            color=colors,
            height=0.7,
            edgecolor="white",
            zorder=2,
            alpha=0.92,
        )
        ax.set_yticks(y)
        ax.set_yticklabels([_feature_label(row) for row in plot_df.itertuples()], fontsize=8)
        ax.set_xlabel(r"Mean $|\partial\mathrm{logit}/\partial x|$", fontsize=9)
        ax.set_title(class_label, fontsize=12, fontweight="bold")
        ax.set_xlim(0, xmax * 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.35, zorder=1)

    for j in range(len(class_ids), 4):
        axes[j].axis("off")

    omic_handles = [
        plt.Rectangle((0, 0), 1, 1, color=OMIC_COLORS.get(omic, "#333333"), label=omic)
        for omic in sorted(omics_seen)
    ]
    if annotated:
        for g in BIO_GROUPS:
            omic_handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=g["face"], edgecolor="none",
                              alpha=0.8, label=g["name"])
            )
    if omic_handles:
        fig.legend(
            handles=omic_handles,
            loc="lower center",
            ncol=min(6, len(omic_handles)),
            frameon=False,
            fontsize=8,
        )

    fig.suptitle(
        "Consensus gradient biomarkers — horizontal bars (5-fold; top-12)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out_name = (
        "hbar_consensus_all_subtypes_annotated.png"
        if annotated
        else "hbar_consensus_all_subtypes.png"
    )
    out = output_dir / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


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

    # Avoid Windows DataLoader worker issues for attribution runs
    cfg = copy.deepcopy(cfg)
    cfg["train"]["num_workers"] = 0

    fold_tables = []
    feature_name_map = None
    omic_names = None

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

        # Free GPU memory between folds
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

    plot_lollipop_per_subtype(
        consensus=consensus,
        class_name_map=class_name_map,
        output_dir=output_dir,
        plot_top_n=args.plot_top_n,
        annotated=False,
    )
    plot_lollipop_per_subtype(
        consensus=consensus,
        class_name_map=class_name_map,
        output_dir=output_dir,
        plot_top_n=args.plot_top_n,
        annotated=True,
    )
    plot_hbar_per_subtype(
        consensus=consensus,
        class_name_map=class_name_map,
        output_dir=output_dir,
        plot_top_n=args.plot_top_n,
        annotated=False,
    )
    plot_hbar_per_subtype(
        consensus=consensus,
        class_name_map=class_name_map,
        output_dir=output_dir,
        plot_top_n=args.plot_top_n,
        annotated=True,
    )

    # Summary counts
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

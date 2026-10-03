"""
Step 2 — Differential expression (one-vs-rest) for consensus gradient features.

Uses unscaled fold train+val matrices (full cohort partition), Mann–Whitney U + BH-FDR,
Cohen's d / mean difference (data are already centered). Intersects with step-1 consensus.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import mannwhitneyu

from train import extract_omics_config, resolve_pre_split_fold_dir


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR q-values; NaNs preserved."""
    pvals = np.asarray(pvals, dtype=float)
    q = np.full_like(pvals, np.nan, dtype=float)
    mask = np.isfinite(pvals)
    p = pvals[mask]
    if p.size == 0:
        return q
    order = np.argsort(p)
    ranked = p[order]
    n = len(ranked)
    q_ranked = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity from the back
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    q[mask] = q_ranked[inv]
    return q


def parse_args():
    p = argparse.ArgumentParser(description="Step 2: DE filter for consensus biomarkers.")
    p.add_argument("--config", default="configs/config_self_attn_cross_attn_brca.yaml")
    p.add_argument(
        "--consensus-csv",
        default="attention_plots/biomarker_step1_consensus/consensus_features.csv",
    )
    p.add_argument(
        "--output-dir",
        default="attention_plots/biomarker_step2_de",
    )
    p.add_argument(
        "--class-names",
        default="LumA,LumB,HER2,Basal",
        help="Comma-separated names in label index order.",
    )
    p.add_argument("--q-thresh", type=float, default=0.05)
    p.add_argument("--abs-d-thresh", type=float, default=0.3, help="|Cohen's d| threshold.")
    p.add_argument(
        "--fold-for-full-cohort",
        type=int,
        default=1,
        help="1-based fold whose train+val union = full cohort.",
    )
    p.add_argument(
        "--genomewide",
        action="store_true",
        help="Also run genome-wide DE per omic×class (slow).",
    )
    return p.parse_args()


def load_full_cohort(cfg, fold_one_based: int):
    """Load X (samples x features) and y from train+val of one pre-split fold."""
    fold_idx = fold_one_based - 1
    fold_dir = Path(resolve_pre_split_fold_dir(cfg["data"], fold_idx))
    omics_config = extract_omics_config(cfg["data"])
    y_stem = Path(cfg["data"]["y_path"]).stem

    y = pd.concat(
        [
            pd.read_csv(fold_dir / f"{y_stem}_train.csv"),
            pd.read_csv(fold_dir / f"{y_stem}_val.csv"),
        ],
        ignore_index=True,
    ).iloc[:, 0].to_numpy()

    X = {}
    feature_names = {}
    for omic, path in omics_config.items():
        stem = Path(path).stem
        tr = pd.read_csv(fold_dir / f"{stem}_train.csv", index_col=0)
        va = pd.read_csv(fold_dir / f"{stem}_val.csv", index_col=0)
        # rows=features, cols=samples → samples x features
        mat = pd.concat([tr, va], axis=1).T
        # align sample order to labels (same column order as concatenated y files)
        X[omic] = mat.to_numpy(dtype=np.float64)
        feature_names[omic] = mat.columns.astype(str).tolist()
        if X[omic].shape[0] != len(y):
            raise ValueError(
                f"{omic}: n_samples={X[omic].shape[0]} != n_labels={len(y)}"
            )
    return X, y, feature_names


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled < 1e-12:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


def de_features(
    X: np.ndarray,
    y: np.ndarray,
    class_idx: int,
    feature_indices: list[int],
) -> pd.DataFrame:
    """DE for selected feature indices: class vs rest."""
    mask = y == class_idx
    n_case = int(mask.sum())
    n_ctrl = int((~mask).sum())
    case = X[mask]
    ctrl = X[~mask]

    rows = []
    for j in feature_indices:
        a = case[:, j]
        b = ctrl[:, j]
        if np.all(np.isnan(a)) or np.all(np.isnan(b)):
            p, u = np.nan, np.nan
        else:
            a2 = a[~np.isnan(a)]
            b2 = b[~np.isnan(b)]
            if len(a2) < 2 or len(b2) < 2:
                p, u = np.nan, np.nan
            else:
                try:
                    u, p = mannwhitneyu(a2, b2, alternative="two-sided")
                except ValueError:
                    u, p = np.nan, np.nan
        mean_case = float(np.nanmean(a))
        mean_ctrl = float(np.nanmean(b))
        mean_diff = mean_case - mean_ctrl
        d = cohens_d(a[~np.isnan(a)], b[~np.isnan(b)]) if np.any(~np.isnan(a)) else np.nan
        eps = 1.0
        log2fc = np.sign(mean_diff) * np.log2((abs(mean_case) + eps) / (abs(mean_ctrl) + eps))
        rows.append(
            {
                "feature_idx": int(j),
                "n_case": n_case,
                "n_ctrl": n_ctrl,
                "mean_case": mean_case,
                "mean_ctrl": mean_ctrl,
                "mean_diff": mean_diff,
                "cohens_d": d,
                "log2fc_pseudo": float(log2fc),
                "U": float(u) if u == u else np.nan,
                "p_value": float(p) if p == p else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df["q_value"] = bh_fdr(df["p_value"].to_numpy())
    return df


def de_one_vs_rest(X: np.ndarray, y: np.ndarray, class_idx: int) -> pd.DataFrame:
    """DE for all features: class vs rest."""
    return de_features(X, y, class_idx, list(range(X.shape[1])))


def write_report(
    out_path: Path,
    consensus: pd.DataFrame,
    joined: pd.DataFrame,
    q_thresh: float,
    abs_d_thresh: float,
    class_names: list[str],
):
    lines = []
    lines.append("# Biomarker Step 2 — Differential Expression Report\n")
    lines.append("## Method\n")
    lines.append(
        "- **Comparison:** one-vs-rest (subtype vs all others) on the full BRCA cohort "
        "(train∪val of one pre-split fold = all labeled samples).\n"
        "- **Test:** Mann–Whitney U (two-sided).\n"
        "- **Multiple testing:** Benjamini–Hochberg FDR within each **omic × class**.\n"
        "- **Effect size:** Cohen's d and mean difference (matrices are already centered; "
        "classical log2FC on raw counts is not appropriate — `log2fc_pseudo` is only directional).\n"
        f"- **Pass criteria:** `q < {q_thresh}` and `|Cohen's d| ≥ {abs_d_thresh}`.\n"
        "- **Intersection:** step-1 consensus features (≥3/5 folds in gradient top-k).\n"
    )
    lines.append("## Cohort\n")
    n = int(joined["n_case"].iloc[0] + joined["n_ctrl"].iloc[0]) if len(joined) else "?"
    lines.append(f"- N samples (approx): **{n}**\n")
    lines.append(f"- Class order: {', '.join(f'{i}={n}' for i, n in enumerate(class_names))}\n")

    lines.append("\n## Summary — consensus ∩ DE pass\n")
    joined = joined.copy()
    joined["de_pass"] = (joined["q_value"] < q_thresh) & (
        joined["cohens_d"].abs() >= abs_d_thresh
    )
    summary = (
        joined.groupby(["class_name", "omic"], as_index=False)
        .agg(
            n_consensus=("feature", "count"),
            n_de_pass=("de_pass", "sum"),
        )
    )
    lines.append("| Subtype | Omic | Consensus | DE pass |\n|---|---|---:|---:|\n")
    for r in summary.itertuples():
        lines.append(
            f"| {r.class_name} | {r.omic} | {r.n_consensus} | {int(r.n_de_pass)} |\n"
        )
    lines.append(
        f"\n**Total consensus:** {len(joined)} · **DE pass:** {int(joined['de_pass'].sum())}\n"
    )

    lines.append("\n## Per-subtype highlights\n")
    for cname in class_names:
        sub = joined[joined["class_name"] == cname].copy()
        if sub.empty:
            lines.append(f"\n### {cname}\n")
            lines.append("_No consensus features._\n")
            continue
        sub["abs_d"] = sub["cohens_d"].abs()
        sub = sub.sort_values(["de_pass", "abs_d"], ascending=[False, False])
        lines.append(f"\n### {cname}\n")
        lines.append(
            "| Feature | Omic | n_folds | mean\\|grad\\| | mean_diff | Cohen's d | q | Pass |\n"
            "|---|---|---:|---:|---:|---:|---:|:---:|\n"
        )
        for r in sub.itertuples():
            lines.append(
                f"| {r.feature} | {r.omic} | {int(r.n_folds)} | {r.mean_abs_grad:.4g} | "
                f"{r.mean_diff:.4g} | {r.cohens_d:.3f} | {r.q_value:.2e} | "
                f"{'yes' if r.de_pass else 'no'} |\n"
            )

    lines.append("\n## Interpretation notes\n")
    lines.append(
        "- Features with **de_pass=yes** are supported both by model sensitivity (gradient) "
        "and by univariate subtype difference.\n"
        "- **Sign of mean_diff / Cohen's d:** positive → higher in the subtype vs rest "
        "(on the processed scale); negative → lower.\n"
        "- Failures (consensus but not DE) may still matter to the model via multivariate "
        "interactions; treat them as model-facing, not classical biomarkers.\n"
        "- Basal 17q12 CNV block: check whether DE direction/significance matches HER2 biology; "
        "if amplicon genes DE for Basal unexpectedly, revisit label encoding.\n"
    )
    out_path.write_text("".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = [n.strip() for n in args.class_names.split(",") if n.strip()]
    consensus = pd.read_csv(args.consensus_csv)

    print("Loading full cohort from fold", args.fold_for_full_cohort)
    X, y, feature_names = load_full_cohort(cfg, args.fold_for_full_cohort)
    print("N=", len(y), "omics=", list(X.keys()))

    # DE on consensus features (FDR within each omic × class consensus set)
    consensus_joined = []

    for class_idx, cname in enumerate(class_names):
        for omic, mat in X.items():
            cons_sub = consensus[
                (consensus["class_idx"] == class_idx) & (consensus["omic"] == omic)
            ].copy()
            if cons_sub.empty:
                continue
            idxs = cons_sub["feature_idx"].astype(int).tolist()
            print(f"DE consensus {cname} / {omic} (n={len(idxs)}) ...")
            de = de_features(mat, y, class_idx, idxs)
            # Remap names from matrix to be safe
            de["feature"] = [feature_names[omic][i] for i in de["feature_idx"]]
            de["omic"] = omic
            de["class_idx"] = class_idx
            de["class_name"] = cname

            if args.genomewide:
                print(f"  genome-wide DE {cname}/{omic} ...")
                de_gw = de_one_vs_rest(mat, y, class_idx)
                de_gw["feature"] = feature_names[omic]
                de_gw["omic"] = omic
                de_gw["class_idx"] = class_idx
                de_gw["class_name"] = cname
                de_gw.to_csv(out_dir / f"de_genomewide_{cname}_{omic}.csv", index=False)

            merged = cons_sub.drop(columns=["feature"], errors="ignore").merge(
                de,
                on=["feature_idx", "omic", "class_idx", "class_name"],
                how="left",
            )
            # keep consensus feature name if present
            if "feature" not in merged.columns:
                merged["feature"] = [
                    feature_names[omic][i] for i in merged["feature_idx"].astype(int)
                ]
            consensus_joined.append(merged)

    joined = pd.concat(consensus_joined, ignore_index=True) if consensus_joined else pd.DataFrame()
    joined["de_pass"] = (joined["q_value"] < args.q_thresh) & (
        joined["cohens_d"].abs() >= args.abs_d_thresh
    )
    joined_path = out_dir / "consensus_with_de.csv"
    joined.to_csv(joined_path, index=False)
    print(f"Saved {joined_path}")

    pass_df = joined[joined["de_pass"]].copy()
    pass_path = out_dir / "consensus_de_pass.csv"
    pass_df.to_csv(pass_path, index=False)
    print(f"Saved {pass_path} ({len(pass_df)} features)")

    report_path = out_dir / "REPORT_step2_de.md"
    write_report(
        report_path,
        consensus,
        joined,
        args.q_thresh,
        args.abs_d_thresh,
        class_names,
    )
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()

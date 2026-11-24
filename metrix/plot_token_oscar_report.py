"""Report figures for the proposed method (256x32 default), built from saved report.json files.

Figures written to material/token_oscar/:
  1. fig_proposed_vs_baselines.png - Macro-F1 (mean +/- std, 5 folds) vs 5 baselines (MOT excluded)
  2. fig_proposed_per_fold.png     - Per-fold Macro-F1 of the proposed method across the 3 cohorts

Run:  .venv/bin/python metrix/plot_token_oscar_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "material" / "token_oscar"

COHORTS = ["BRCA", "LUNG", "COADREAD"]
COHORT_FILE = {"BRCA": "brca", "LUNG": "lung", "COADREAD": "coadread"}

PROPOSED_REPORT = ROOT / "lightning_logs/token_oscar_sweep/t256_m32/report.json"
BASELINE_DIR = ROOT / "lightning_logs/baselines_matched"

# MOT excluded as not comparable in this setting.
BASELINE_KEYS = [
    ("moxgate", "MoXGATE"),
    ("mogcn", "MoGCN"),
    ("mogonet", "MOGONET"),
    ("mobrca", "moBRCA-net"),
    ("deepmoic", "DeepMoIC"),
]

PROPOSED_COLOR = "#d62728"
BASE_COLOR = "#7f7f7f"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def proposed_summary(report: dict, cohort: str) -> dict:
    s = report["cohorts"][cohort]["summary"]
    return {
        "mean": 100 * s["f1_macro_mean"],
        "std": 100 * s["f1_macro_std"],
        "acc": 100 * s["accuracy_mean"],
    }


def proposed_folds(report: dict, cohort: str) -> list[float]:
    return [100 * f["test"]["f1_macro"] for f in report["cohorts"][cohort]["folds"]]


def baseline_summary(cohort: str, key: str) -> dict:
    s = load(BASELINE_DIR / f"{COHORT_FILE[cohort]}_report.json")["models"][key]["summary"]
    return {"mean": 100 * s["f1_macro_mean"], "std": 100 * s["f1_macro_std"]}


def fig_vs_baselines(proposed: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=False)
    for ax, cohort in zip(axes, COHORTS):
        rows = [("Proposed method", proposed_summary(proposed, cohort), PROPOSED_COLOR)]
        rows += [(name, baseline_summary(cohort, key), BASE_COLOR) for key, name in BASELINE_KEYS]
        rows.sort(key=lambda r: r[1]["mean"])
        names = [r[0] for r in rows]
        means = np.array([r[1]["mean"] for r in rows])
        stds = np.array([r[1]["std"] for r in rows])
        colors = [r[2] for r in rows]
        y = np.arange(len(rows))

        ax.barh(
            y,
            means,
            xerr=stds,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
            error_kw={"elinewidth": 1.0, "capsize": 3},
        )
        for yi, m, s in zip(y, means, stds):
            ax.text(m + s + 0.5, yi, f"{m:.2f}", va="center", fontsize=9)

        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=10)
        lo = min(means - stds) - 5
        ax.set_xlim(max(0, lo), 100)
        ax.set_title(cohort, fontsize=13, fontweight="bold")
        ax.set_xlabel("Macro-F1 (%) — mean ± std (5 folds)")
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Proposed method vs. multi-omic baselines (matched 5-fold protocol)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = OUT / "fig_proposed_vs_baselines.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("saved", out_path)


def fig_per_fold(proposed: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False)
    folds = np.arange(1, 6)
    for ax, cohort in zip(axes, COHORTS):
        scores = proposed_folds(proposed, cohort)
        mean_val = float(np.mean(scores))
        ax.plot(folds, scores, "o-", color=PROPOSED_COLOR, lw=2, ms=7, label="Proposed method")
        ax.axhline(mean_val, color=PROPOSED_COLOR, ls=":", lw=1.2, label=f"Mean ({mean_val:.2f}%)")
        for f, sc in zip(folds, scores):
            ax.text(f, sc + 0.35, f"{sc:.2f}", ha="center", fontsize=8.5)
        ax.set_xticks(folds)
        ax.set_xticklabels([f"Fold {i}" for i in folds])
        ax.set_title(cohort, fontsize=13, fontweight="bold")
        ax.set_ylabel("Test Macro-F1 (%)")
        lo, hi = min(scores), max(scores)
        ax.set_ylim(lo - 1.5, min(100.0, hi + 1.5))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        "Per-fold Macro-F1 of the proposed method across cohorts",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = OUT / "fig_proposed_per_fold.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("saved", out_path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proposed = load(PROPOSED_REPORT)
    fig_vs_baselines(proposed)
    fig_per_fold(proposed)


if __name__ == "__main__":
    main()

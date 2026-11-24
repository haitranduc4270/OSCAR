"""Figures for the TokenOSCAR report (REPORT.md), built from saved report.json files.

Figures:
  1. fig_tokenoscar_vs_baselines.png  - Macro-F1 mean +/- std per cohort, TokenOSCAR 256x32 vs OSCAR 1x128 vs baselines
  2. fig_tokenoscar_per_fold.png      - per-fold Macro-F1, TokenOSCAR 256x32 vs OSCAR 1x128 (paired by fold)
  3. fig_token_size_sweep.png         - T x M sweep: Macro-F1 per cohort, plus Avg F1 vs parameter count

Run:  .venv/bin/python plot_token_oscar_report.py
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

TOKEN_DEFAULT = ROOT / "lightning_logs/token_oscar_sweep/t256_m32/report.json"
TOKEN_SWEEP = [  # (T, M, report)
    (128, 32, ROOT / "lightning_logs/token_oscar_matched/report.json"),
    (128, 64, ROOT / "lightning_logs/token_oscar_sweep/t128_m64/report.json"),
    (128, 128, ROOT / "lightning_logs/token_oscar_sweep/t128_m128/report.json"),
    (256, 32, ROOT / "lightning_logs/token_oscar_sweep/t256_m32/report.json"),  # default
    (256, 64, ROOT / "lightning_logs/token_oscar_sweep/t256_m64/report.json"),
]
BASELINE_DIR = ROOT / "lightning_logs/baselines_matched"
OSCAR_DIR = ROOT / "lightning_logs"  # {cohort}_table51_matched/report.json

BASELINE_KEYS = [("mot", "MOT"), ("moxgate", "MoXGATE"), ("mogcn", "MoGCN"),
                 ("mogonet", "MOGONET"), ("mobrca", "moBRCA-net"), ("deepmoic", "DeepMoIC")]

TOKEN_COLOR = "#d62728"
OSCAR_COLOR = "#1f77b4"
BASE_COLOR = "#7f7f7f"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def token_summary(report: dict, cohort: str) -> dict:
    s = report["cohorts"][cohort]["summary"]
    return {"mean": 100 * s["f1_macro_mean"], "std": 100 * s["f1_macro_std"],
            "acc": 100 * s["accuracy_mean"]}


def token_folds(report: dict, cohort: str) -> list[float]:
    return [100 * f["test"]["f1_macro"] for f in report["cohorts"][cohort]["folds"]]


def oscar_block(cohort: str) -> dict:
    return load(OSCAR_DIR / f"{COHORT_FILE[cohort]}_table51_matched/report.json")["models"]["oscar"]


def oscar_summary(cohort: str) -> dict:
    s = oscar_block(cohort)["summary"]
    return {"mean": 100 * s["f1_macro_mean"], "std": 100 * s["f1_macro_std"]}


def oscar_folds(cohort: str) -> list[float]:
    return [100 * f["test"]["f1_macro"] for f in oscar_block(cohort)["folds"]]


def baseline_summary(cohort: str, key: str) -> dict:
    s = load(BASELINE_DIR / f"{COHORT_FILE[cohort]}_report.json")["models"][key]["summary"]
    return {"mean": 100 * s["f1_macro_mean"], "std": 100 * s["f1_macro_std"]}


def check_against_table(token: dict) -> None:
    """Sanity check: the JSON summary must match the REPORT.md numbers (89.31, 94.70, 91.34)."""
    expected = {"BRCA": 89.31, "LUNG": 94.70, "COADREAD": 91.34}
    for c in COHORTS:
        got = token_summary(token, c)["mean"]
        status = "OK" if abs(got - expected[c]) < 0.01 else "MISMATCH"
        print(f"[check] TokenOSCAR 256x32 {c}: F1 {got:.2f} vs REPORT {expected[c]:.2f} -> {status}")


def fig_vs_baselines() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=False)
    for ax, cohort in zip(axes, COHORTS):
        rows = [("TokenOSCAR (256×32)", token_summary(load(TOKEN_DEFAULT), cohort), TOKEN_COLOR),
                ("OSCAR (1×128)*", oscar_summary(cohort), OSCAR_COLOR)]
        rows += [(name, baseline_summary(cohort, key), BASE_COLOR) for key, name in BASELINE_KEYS]
        rows.sort(key=lambda r: r[1]["mean"])  # ascending, so the best is drawn at the top
        names = [r[0] for r in rows]
        means = np.array([r[1]["mean"] for r in rows])
        stds = np.array([r[1]["std"] for r in rows])
        colors = [r[2] for r in rows]
        y = np.arange(len(rows))
        ax.barh(y, means, xerr=stds, color=colors, edgecolor="black", linewidth=0.5,
                error_kw={"elinewidth": 1, "capsize": 3})
        for yi, m, s in zip(y, means, stds):
            ax.text(m + s + 0.5, yi, f"{m:.2f}", va="center", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=10)
        lo = min(means - stds) - 5
        ax.set_xlim(max(0, lo), 100)
        ax.set_title(f"{cohort}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Macro-F1 (%) — mean ± std, 5 folds")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("TokenOSCAR (256×32) so với OSCAR (1×128) và các baseline — Matched 5-fold",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.005, "* OSCAR (1×128) lấy từ lần chạy table51 trước đó; baseline từ baselines_matched.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / "fig_tokenoscar_vs_baselines.png", dpi=200)
    plt.close(fig)
    print("saved", OUT / "fig_tokenoscar_vs_baselines.png")


def fig_per_fold() -> None:
    token = load(TOKEN_DEFAULT)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    folds = np.arange(1, 6)
    for ax, cohort in zip(axes, COHORTS):
        tf = token_folds(token, cohort)
        of = oscar_folds(cohort)
        ax.plot(folds, tf, "o-", color=TOKEN_COLOR, lw=2, ms=7, label="TokenOSCAR (256×32)")
        ax.plot(folds, of, "s--", color=OSCAR_COLOR, lw=2, ms=6, label="OSCAR (1×128)*")
        for f, a, b in zip(folds, tf, of):
            ax.plot([f, f], [a, b], color="gray", alpha=0.4, lw=1, zorder=0)
        ax.axhline(np.mean(tf), color=TOKEN_COLOR, ls=":", lw=1.2)
        ax.axhline(np.mean(of), color=OSCAR_COLOR, ls=":", lw=1.2)
        ax.set_xticks(folds)
        ax.set_xticklabels([f"Fold {i}" for i in folds])
        ax.set_title(cohort, fontsize=13, fontweight="bold")
        ax.set_ylabel("Macro-F1 (%) trên test fold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower left")
    fig.suptitle("Macro-F1 theo từng fold: TokenOSCAR (256×32) vs OSCAR (1×128)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "fig_tokenoscar_per_fold.png", dpi=200)
    plt.close(fig)
    print("saved", OUT / "fig_tokenoscar_per_fold.png")


def sweep_rows() -> list[dict]:
    rows = []
    for T, M, path in TOKEN_SWEEP:
        r = load(path)
        entry = {"label": f"{T}×{M}", "T": T, "M": M}
        for c in COHORTS:
            entry[c] = token_summary(r, c)
        entry["params"] = r["cohorts"]["BRCA"]["params_total"]
        entry["vram"] = max(f["peak_cuda_mb"] for c in COHORTS for f in r["cohorts"][c]["folds"])
        rows.append(entry)
    rows.sort(key=lambda e: (e["T"], e["M"]))
    return rows


def fig_sweep() -> None:
    rows = sweep_rows()
    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    default_idx = labels.index("256×32")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    for ax, cohort in zip(axes[:3], COHORTS):
        m = np.array([r[cohort]["mean"] for r in rows])
        s = np.array([r[cohort]["std"] for r in rows])
        ax.errorbar(x, m, yerr=s, fmt="o-", color=TOKEN_COLOR, ecolor="gray", capsize=4, lw=2)
        ax.axvline(default_idx, color="black", ls="--", lw=1, alpha=0.5)
        ax.text(default_idx, 1.01, "mặc định", transform=ax.get_xaxis_transform(), ha="center", fontsize=9)
        for xi, mi, si in zip(x, m, s):
            ax.text(xi + 0.08, mi + si + 0.3, f"{mi:.2f}", ha="left", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_xlabel("Cấu hình T × M")
        ax.set_ylabel("Macro-F1 (%) — mean ± std")
        ax.set_title(cohort, fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)

    ax = axes[3]
    avg = np.array([np.mean([r[c]["mean"] for c in COHORTS]) for r in rows])
    params = np.array([r["params"] / 1e6 for r in rows])
    colors = [TOKEN_COLOR if i == default_idx else BASE_COLOR for i in range(len(rows))]
    ax.bar(x, avg, color=colors, edgecolor="black", linewidth=0.5)
    for xi, a, p, r in zip(x, avg, params, rows):
        ax.text(xi, a + 0.1, f"{a:.2f}\n{p:.1f}M\n{r['vram']/1024:.2f} GB", ha="center", fontsize=8)
    ax.set_ylim(min(avg) - 2, max(avg) + 2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.set_xlabel("Cấu hình T × M")
    ax.set_ylabel("Avg Macro-F1 (%) qua 3 cohort")
    ax.set_title("Trung bình F1 · tham số (BRCA) · VRAM đỉnh", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Khảo sát kích thước token (T × M) của TokenOSCAR", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "fig_token_size_sweep.png", dpi=200)
    plt.close(fig)
    print("saved", OUT / "fig_token_size_sweep.png")
    for r in rows:
        print(f"[sweep] {r['label']:>8}  BRCA {r['BRCA']['mean']:.2f}  LUNG {r['LUNG']['mean']:.2f}  "
              f"COAD {r['COADREAD']['mean']:.2f}  avg {np.mean([r[c]['mean'] for c in COHORTS]):.2f}  "
              f"params {r['params']/1e6:.2f}M  vram {r['vram']:.0f}MB")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    check_against_table(load(TOKEN_DEFAULT))
    fig_vs_baselines()
    fig_per_fold()
    fig_sweep()


if __name__ == "__main__":
    main()

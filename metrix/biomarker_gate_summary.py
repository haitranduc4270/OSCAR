"""
Per-subtype gated-fusion summaries on validation folds.

Gate is the sigmoid output of GatedFusion: fused = gate * enhanced + (1 - gate) * original.
A higher mean gate means more weight on the ordered-cross-attention representation.
"""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from biomarker_step1_consensus import find_best_checkpoint
from train import set_seed
from visualize_attention import build_datamodule, load_model_from_checkpoint

OUT = Path("attention_plots/biomarker_report_assets")
CONFIG = Path("configs/config_self_attn_cross_attn_brca.yaml")

def paper_names() -> dict[int, str]:
    """Names from the label audit written by biomarker_report_assets.py."""
    audit = pd.read_csv(OUT / "label_audit.csv")
    return {int(r.class_idx): str(r.paper_name) for r in audit.itertuples()}


def collect_fold_gates(model, dataloader, device) -> list[dict]:
    captured: dict[str, torch.Tensor] = {}
    hooks = []

    def make_hook(name: str):
        def hook(_mod, _inp, out):
            captured[name] = out.detach()

        return hook

    hooks.append(model.gated_fusion_mrna.gate_network.register_forward_hook(make_hook("mRNA")))
    hooks.append(
        model.gated_fusion_proteomics.gate_network.register_forward_hook(make_hook("Proteomics"))
    )
    rows = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device)
            batch_omics = {
                k: v.to(device) for k, v in batch.items() if k != "label"
            }
            captured.clear()
            model({**batch_omics, "label": labels})
            labels_np = labels.detach().cpu().numpy()
            for omic, gate in captured.items():
                g = gate.detach().cpu().numpy()
                for i in range(g.shape[0]):
                    rows.append(
                        {
                            "class_idx": int(labels_np[i]),
                            "omic": omic,
                            "mean_gate": float(g[i].mean()),
                            "frac_gate_gt_0.5": float((g[i] > 0.5).mean()),
                        }
                    )
    for h in hooks:
        h.remove()
    return rows


def plot_gates(summary: pd.DataFrame, path: Path, ylim: tuple[float, float] = (0.45, 0.68)) -> None:
    from biomarker_report_assets import DISPLAY_NAME, OMIC_COLORS, apply_mpl_style, style_ax

    apply_mpl_style()
    order = ["LumA", "LumB", "Basal", "HER2"]
    omics = ["mRNA", "Proteomics"]
    labels = [DISPLAY_NAME[n] for n in order]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.5), sharex=True)
    x = np.arange(len(order))
    offsets = {"mRNA": -0.12, "Proteomics": 0.12}
    panels = (
        (axes[0], "A", "mean_of_mean_gate", "std_mean_gate", "Mean gate"),
        (axes[1], "B", "mean_frac_gt_0_5", "std_frac_gt_0_5", "Fraction of dimensions $> 0.5$"),
    )
    for ax, letter, metric, err, ylabel in panels:
        ax.axhline(0.5, color="#B0B0B0", lw=0.7, ls="--", zorder=1)
        for omic in omics:
            sub = summary[summary["omic"] == omic].set_index("class_name").reindex(order)
            ax.errorbar(
                x + offsets[omic],
                sub[metric],
                yerr=sub[err],
                fmt="o",
                color=OMIC_COLORS[omic],
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=0.6,
                elinewidth=1.15,
                capsize=2.5,
                capthick=0.8,
                label=omic,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_title(letter, loc="left", fontweight="bold", pad=6)
        style_ax(ax, grid="y")
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["train"]["num_workers"] = 0
    set_seed(cfg["train"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_dir = Path(cfg.get("logging", {}).get("log_dir", "lightning_logs"))
    log_name = cfg.get("logging", {}).get("log_name", "brca")
    n_folds = int(cfg["data"].get("n_folds", 5))

    names = paper_names()
    sample_rows = []
    for fold_idx in range(n_folds):
        fold_num = fold_idx + 1
        ckpt = find_best_checkpoint(log_dir / f"{log_name}_fold_{fold_num}", metric="f1")
        print(f"Fold {fold_num}: {ckpt.name} on {device}")
        cfg["data"]["current_fold"] = fold_idx
        dm = build_datamodule(cfg)
        model = load_model_from_checkpoint(cfg, dm, str(ckpt), device)
        if not model.use_gated_fusion:
            raise RuntimeError("Checkpoint model was built without gated fusion.")
        rows = collect_fold_gates(model, dm.val_dataloader(), device)
        for row in rows:
            row["fold"] = fold_num
            row["class_name"] = names.get(row["class_idx"], str(row["class_idx"]))
        sample_rows.extend(rows)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    samples = pd.DataFrame(sample_rows)
    samples.to_csv(OUT / "gate_samples.csv", index=False)

    fold_means = (
        samples.groupby(["fold", "class_name", "omic"], as_index=False)
        .agg(
            mean_of_mean_gate=("mean_gate", "mean"),
            mean_frac_gt_0_5=("frac_gate_gt_0.5", "mean"),
            n=("mean_gate", "size"),
        )
    )
    summary = (
        fold_means.groupby(["class_name", "omic"], as_index=False)
        .agg(
            mean_of_mean_gate=("mean_of_mean_gate", "mean"),
            std_mean_gate=("mean_of_mean_gate", "std"),
            mean_frac_gt_0_5=("mean_frac_gt_0_5", "mean"),
            std_frac_gt_0_5=("mean_frac_gt_0_5", "std"),
            n_folds=("fold", "nunique"),
        )
    )
    summary.to_csv(OUT / "gate_summary.csv", index=False)
    plot_gates(summary, OUT / "gate_by_subtype.png")
    print(summary.to_string(index=False))
    print(f"Wrote {OUT / 'gate_summary.csv'}")


if __name__ == "__main__":
    main()

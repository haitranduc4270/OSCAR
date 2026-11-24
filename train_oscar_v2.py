"""Matched BRCA 5-fold run for OSCAR-v2 (same protocol as ablation suite)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from models.self_attn_fusion_v2 import SelfAttentionFusionModelV2
from pathway_data import _read_split_matrix
from train_ablation_swap import run_fold
from train_five_fold import FOLDS, load_fold, summarize
from train_pathway_tokens import configure_runtime

OUT_DIR = Path("lightning_logs/brca_oscar_v2")


def build_v2(cfg: dict, dims: dict):
    m = cfg["model"]
    t = cfg["train"]

    def _factory():
        return SelfAttentionFusionModelV2(
            input_dims=dims,
            num_classes=int(m["num_classes"]),
            latent_dim=int(m["latent_dim"]),
            encoder_hidden_dim=int(m["hidden_dim"]),
            fusion_hidden=int(m["fusion_hidden"]),
            dropout=float(m["dropout"]),
            lr=float(t["lr"]),
            weight_decay=float(t["weight_decay"]),
            n_heads=int(m.get("n_heads", 8)),
            joint_n_heads=int(m.get("joint_n_heads", 8)),
        )

    return _factory


def main():
    cfg = yaml.safe_load(open("configs/config_oscar_v2_brca.yaml"))
    # Match ablation suite epoch budget / selection protocol
    pathway_cfg = yaml.safe_load(open("configs/config_pathway_mean_brca.yaml"))
    max_epochs = int(pathway_cfg["train"]["max_epochs"])
    configure_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    reference = _read_split_matrix(FOLDS / "fold_1", "BRCA_mRNA")
    dims = {
        "mRNA": reference.shape[0],
        "CNV": _read_split_matrix(FOLDS / "fold_1", "BRCA_CNV_1_channel").shape[0],
        "Proteomics": _read_split_matrix(FOLDS / "fold_1", "BRCA_proteomics").shape[0],
    }
    ref_cols = {
        "mRNA": reference.index,
        "CNV": _read_split_matrix(FOLDS / "fold_1", "BRCA_CNV_1_channel").index,
        "Proteomics": _read_split_matrix(FOLDS / "fold_1", "BRCA_proteomics").index,
    }

    print(
        f"OSCAR-v2 matched 5-fold | device={device} dims={dims} max_epochs={max_epochs}",
        flush=True,
    )
    build = build_v2(cfg, dims)
    folds = []
    for i in range(1, 6):
        fold = load_fold(FOLDS / f"fold_{i}", ref_cols)
        row = run_fold(cfg, dims, fold, device, max_epochs, build)
        folds.append(row)
        print(
            f"oscar_v2 fold {i}: epoch {row['chosen_epoch']} "
            f"F1 {row['test']['f1_macro']:.4f} Acc {row['test']['accuracy']:.4f}",
            flush=True,
        )

    summary = summarize(folds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "self_attn_fusion_v2",
        "config": "configs/config_oscar_v2_brca.yaml",
        "protocol": {
            "outer_folds": "csv/processed/BRCA/5-fold",
            "inner_val_fraction": 0.2,
            "inner_seed": int(cfg["train"]["seed"]),
            "max_epochs": max_epochs,
            "selection": "first maximum of inner validation macro-F1",
            "retrain": "all samples in the fold training file",
        },
        "input_dims": dims,
        "folds": folds,
        "summary": summary,
    }
    path = OUT_DIR / "report.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(
        f"oscar_v2 mean F1 {summary['f1_macro_mean']:.4f}±{summary['f1_macro_std']:.4f} "
        f"Acc {summary['accuracy_mean']:.4f}±{summary['accuracy_std']:.4f}",
        flush=True,
    )
    print(f"Saved {path}", flush=True)

    # Quick paired deltas vs prior ablation numbers (if present)
    prior = Path("lightning_logs/brca_ablation_full_rerun/report.json")
    if prior.is_file():
        abl = json.load(open(prior))
        v2 = np.array([r["test"]["f1_macro"] for r in folds])
        for name in ("oscar", "modsa_symmetric", "moxgate_ce"):
            other = abl["models"].get(name, {}).get("folds")
            if not other:
                continue
            base = np.array([r["test"]["f1_macro"] for r in other])
            delta = v2 - base
            print(
                f"Δ vs {name}: mean={delta.mean():+.4f} "
                f"folds={[round(x, 4) for x in delta.tolist()]}",
                flush=True,
            )


if __name__ == "__main__":
    main()

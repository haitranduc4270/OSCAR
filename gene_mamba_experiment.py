"""Mamba order ablation on the same BRCA gene panel as the Transformer run.

Two models see one token per gene (CNV, mRNA, protein projected together).
M_genomic scans genes in chromosomal order. M_shuffled uses the same fixed
permutation as the Transformer shuffle, so the only change is which gene
sits at which step of the scan.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytorch_lightning as pl
import torch
import yaml

from gene_token_experiment import (
    OUT_DIR,
    ROOT,
    FOLD_ROOT,
    build_split,
    fit_variant,
    make_loader,
    summarize,
)
from models.gene_mamba import GeneMambaModel


VARIANTS = {
    "M_genomic": "Bidirectional Mamba, genes in genomic order",
    "M_shuffled": "Bidirectional Mamba, same architecture with genes shuffled",
}


def build_model(name, n_genes, n_classes, seed, lr, weight_decay, dropout):
    rng = np.random.default_rng(seed)
    shuffle = rng.permutation(n_genes).tolist()
    identity = list(range(n_genes))
    perm = identity if name == "M_genomic" else shuffle
    if name not in VARIANTS:
        raise ValueError(name)
    return GeneMambaModel(
        n_genes=n_genes,
        n_classes=n_classes,
        perm=perm,
        d_model=96,
        n_layers=2,
        d_state=8,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.folds = 1
        args.epochs = 2
        args.patience = 2

    cfg = yaml.safe_load((ROOT / "configs" / "config_self_attn_cross_attn_brca.yaml").read_text(encoding="utf-8"))
    n_classes = int(cfg["model"]["num_classes"])
    lr = float(cfg["train"]["lr"])
    weight_decay = float(cfg["train"]["weight_decay"])
    dropout = float(cfg["model"]["dropout"])

    rows = []
    panel_meta = []
    for fold in range(args.folds):
        fold_dir = FOLD_ROOT / f"fold_{fold + 1}"
        print(f"\n=== fold {fold + 1}/{args.folds} ===", flush=True)
        train_v, train_m, y_tr, val_v, val_m, y_va, meta = build_split(fold_dir, args.top_k)
        meta["fold"] = fold + 1
        panel_meta.append({k: v for k, v in meta.items() if k != "skipped_multi"})
        print(f"panel={meta['n_panel']} protein_genes={meta['n_protein_genes']}", flush=True)

        seed = args.seed + fold
        train_loader = make_loader(train_v, train_m, y_tr, args.batch_size, True, seed)
        val_loader = make_loader(val_v, val_m, y_va, args.batch_size, False, seed)
        for name, description in VARIANTS.items():
            pl.seed_everything(seed, workers=True)
            model = build_model(name, meta["n_panel"], n_classes, seed, lr, weight_decay, dropout)
            print(f"  training {name}: {description}", flush=True)
            metrics = fit_variant(model, train_loader, val_loader, args.epochs, args.patience)
            print(
                f"  {name} f1={metrics['f1']} acc={metrics['accuracy']} "
                f"train_acc_last={metrics['train_acc_last']} "
                f"epochs={metrics['epochs_ran']} params={metrics['n_params']}",
                flush=True,
            )
            rows.append({"fold": fold + 1, "variant": name, "description": description, **metrics})
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    averages = summarize(rows)
    payload = {
        "cohort": "BRCA",
        "mixer": "bidirectional Mamba, pure PyTorch selective scan",
        "panel_rule": "genes with CNV, mRNA, and a mapped RPPA antibody, in CNV genomic order",
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": args.seed,
        "panel": panel_meta,
        "per_fold": rows,
        "averages": averages,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / ("mamba_smoke.json" if args.smoke else "mamba_results.json")
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nAverages", flush=True)
    for name, stats in averages.items():
        print(
            f"{name}: F1 {stats['f1_mean']:.4f} +/- {stats['f1_std']:.4f}  "
            f"acc {stats['accuracy_mean']:.4f} +/- {stats['accuracy_std']:.4f}",
            flush=True,
        )
    print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()

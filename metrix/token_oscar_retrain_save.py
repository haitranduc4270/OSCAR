"""Retrain TokenOSCAR 256x32 on BRCA folds and save weights for biomarker analysis.

Reproduces the retrain stage of train_five_fold.run_oscar_fold with the epoch chosen by
the matched run (lightning_logs/token_oscar_sweep_v2/t256_m32/report.json), then checks the
held-out macro-F1 against that report. Weights go to lightning_logs/token_oscar_biomarker/.

Usage:
  .venv/bin/python token_oscar_retrain_save.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from oscar_utils import (
    _read_split_matrix, configure_runtime, evaluate as evaluate_oscar, make_loader,
    run_epoch as run_oscar_epoch, set_seed,
)
from train_token_oscar_matched import COHORTS, _stems, build_token_oscar, load_fold

REPORT = Path("lightning_logs/token_oscar_sweep_v2/t256_m32/report.json")
OUT_DIR = Path("lightning_logs/token_oscar_biomarker")
COHORT = "BRCA"
NUM_TOKENS, TOKEN_DIM, D_MID, N_HEADS = 256, 32, 256, 4


def main() -> None:
    configure_runtime()
    spec = COHORTS[COHORT]
    oscar_cfg = yaml.safe_load(open(spec["oscar"]))
    seed = int(oscar_cfg["train"]["seed"])
    batch_size = int(oscar_cfg["train"]["batch_size"])
    lr = float(oscar_cfg["train"]["lr"])
    wd = float(oscar_cfg["train"]["weight_decay"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report = json.loads(REPORT.read_text())
    folds_report = report["cohorts"][COHORT]["folds"]

    folds_dir = spec["folds"]
    stems = _stems(spec["prefix"])
    reference_columns = {
        name: _read_split_matrix(folds_dir / "fold_1", stems[name]).index
        for name in ("mRNA", "CNV", "Proteomics")
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fold_id in range(1, 6):
        fold = load_fold(folds_dir / f"fold_{fold_id}", reference_columns, spec["prefix"])
        dims = {name: fold["dev"][name].shape[1] for name in ("mRNA", "CNV", "Proteomics")}
        chosen_epoch = int(folds_report[fold_id - 1]["chosen_epoch"])
        y_dev, y_test = fold["y_dev"], fold["y_test"]

        # Same transform as run_oscar_fold: scaler fit on the full fold train file.
        dev_idx = np.arange(len(y_dev))
        scaled = {}
        for name in ("mRNA", "CNV", "Proteomics"):
            scaler = StandardScaler().fit(fold["dev"][name][dev_idx])
            scaled[name] = {
                "train": scaler.transform(fold["dev"][name][dev_idx]).astype(np.float32),
                "test": scaler.transform(fold["test"][name]).astype(np.float32),
            }

        set_seed(seed)
        train_loader = make_loader(
            scaled["mRNA"]["train"], scaled["CNV"]["train"], scaled["Proteomics"]["train"],
            y_dev, batch_size, True, seed,
        )
        test_loader = make_loader(
            scaled["mRNA"]["test"], scaled["CNV"]["test"], scaled["Proteomics"]["test"],
            y_test, batch_size, False, seed,
        )
        model = build_token_oscar(oscar_cfg, dims, NUM_TOKENS, TOKEN_DIM, D_MID, N_HEADS).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        for _ in range(chosen_epoch):
            run_oscar_epoch(model, train_loader, optimizer, device)
        metrics = evaluate_oscar(model, test_loader, device)

        expected = folds_report[fold_id - 1]["test"]["f1_macro"]
        print(
            f"fold {fold_id}: epoch {chosen_epoch} retrained macro-F1 {metrics['f1_macro']:.4f} "
            f"(report {expected:.4f}) acc {metrics['accuracy']:.4f}",
            flush=True,
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "dims": dims,
                "chosen_epoch": chosen_epoch,
                "f1_macro": metrics["f1_macro"],
                "report_f1_macro": expected,
            },
            OUT_DIR / f"fold_{fold_id}.pt",
        )
    print(f"saved weights to {OUT_DIR}")


if __name__ == "__main__":
    main()

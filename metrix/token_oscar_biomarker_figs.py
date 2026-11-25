"""TokenOSCAR 256x32 biomarker pipeline for thesis Figures 5.1-5.3 (BRCA).

Reuses the OSCAR pipeline code but swaps in the TokenOSCAR weights saved by
token_oscar_retrain_save.py:
  step 1  5-fold consensus input-gradient features + lollipop figure (Fig 5.1)
  step 2  one-vs-rest DE for consensus features (feeds Fig 5.2, 5.3)
  figs    Cohen's d by subtype (Fig 5.2) and CNV-vs-mRNA concordance (Fig 5.3)

Class order: index 0..3 = LumA (248), LumB (98), Basal (112), HER2 (53). The
pipeline defaults (LumA,LumB,HER2,Basal) swap Basal and HER2, so names are passed explicitly.

Usage:
  .venv/bin/python token_oscar_biomarker_figs.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
import yaml

import biomarker_report_assets as ra
import biomarker_step1_consensus as s1
import biomarker_step2_de as s2
from train_token_oscar_matched import build_token_oscar

CLASS_NAMES = "LumA,LumB,Basal,HER2"
CONFIG = "configs/config_self_attn_cross_attn_brca.yaml"
CKPT_DIR = Path("lightning_logs/token_oscar_biomarker")
STEP1_DIR = Path("attention_plots/token_oscar_step1_consensus")
STEP2_DIR = Path("attention_plots/token_oscar_step2_de")
ASSETS_DIR = Path("attention_plots/token_oscar_report_assets")
NUM_TOKENS, TOKEN_DIM, D_MID, N_HEADS = 256, 32, 256, 4


def fold_checkpoint(fold_log, metric: str = "f1") -> Path:
    """Map step-1's per-fold log path (e.g. lightning_logs/brca_fold_3) to saved weights."""
    fold_num = int(str(fold_log).rsplit("_", 1)[1])
    return CKPT_DIR / f"fold_{fold_num}.pt"


def load_token_oscar(cfg, dm, checkpoint_path, device):
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    dims = {k: int(v) for k, v in blob["dims"].items()}
    if {k: int(v) for k, v in dm.feature_dims.items()} != dims:
        raise ValueError(f"feature dims mismatch for {checkpoint_path}")
    model = build_token_oscar(cfg, dims, NUM_TOKENS, TOKEN_DIM, D_MID, N_HEADS).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def run_step1() -> None:
    s1.find_best_checkpoint = fold_checkpoint
    s1.load_model_from_checkpoint = load_token_oscar
    sys.argv = ["token_oscar_step1", "--config", CONFIG, "--output-dir", str(STEP1_DIR),
                "--class-names", CLASS_NAMES]
    s1.main()


def run_step2() -> None:
    sys.argv = ["token_oscar_step2", "--config", CONFIG,
                "--consensus-csv", str(STEP1_DIR / "consensus_features.csv"),
                "--output-dir", str(STEP2_DIR), "--class-names", CLASS_NAMES]
    s2.main()


def run_figures() -> None:
    """Fig 5.2 and 5.3 with the same functions as biomarker_report_assets.main (enrichment skipped)."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(CONFIG))
    X, y, feature_names = ra.load_full_cohort(cfg, fold_one_based=1)
    de_raw = pd.read_csv(STEP2_DIR / "consensus_with_de.csv")
    audit = ra.audit_labels(y, de_raw)
    audit.to_csv(ASSETS_DIR / "label_audit.csv", index=False)
    print(audit.to_string(index=False))

    de = de_raw.copy()
    de["de_pass"] = de["de_pass"].astype(str).str.lower().isin(["true", "1"])
    de = ra.apply_paper_names(de, audit)
    de.to_csv(ASSETS_DIR / "consensus_with_de_paper_names.csv", index=False)
    ra.plot_cohens_d(de, ASSETS_DIR / "cohens_d_by_subtype.png")

    conc = ra.concordance_table(de)
    conc.to_csv(ASSETS_DIR / "cnv_mrna_concordance.csv", index=False)
    ra.plot_concordance(conc, ASSETS_DIR / "cnv_mrna_concordance_her2.png")
    print(f"Wrote figures to {ASSETS_DIR}")


def main() -> None:
    run_step1()
    run_step2()
    run_figures()


if __name__ == "__main__":
    main()

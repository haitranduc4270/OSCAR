# Order-aware Self-Cross Attention Representation Learning for Multi-omic Cancer Subtype Classification

- Proposes OSCAR for order-aware multi-omic cancer subtype classification.
- Models CNV–mRNA–RPPA relationships using ordered cross-attention.
- Integrates omic representations through gated fusion and self-attention.
- **TokenOSCAR** keeps each omic as a `128 × 32` token sequence through ordered cross-attention (`128 × 128`), gated fusion, and self-attention fusion. Under the matched 5-fold protocol below it reaches **88.73%** F1 on BRCA, **96.17%** on LUNG, and **90.25%** on COADREAD.
- Identifies subtype-associated molecular features through gradient-based attribution.

## Setup

Prerequisites: Python **3.10**, [`uv`](https://docs.astral.sh/uv/).

```bash
uv python install 3.10
uv sync
```

## Dataset

Download the OSCAR dataset from Kaggle: [oscar-dataset](https://www.kaggle.com/datasets/hitrnc/oscar-dataset).

Extract so pre-split folds live under the repo root:

```text
OSCAR/csv/processed/
  BRCA/5-fold/fold_1/ ...
  COADREAD/5-fold/fold_1/ ...
  LUNG/5-fold/fold_1/ ...
```

Update `data.*_path` and `data.y_path` in each config if your CSVs are elsewhere. The matched runners below ignore those fields and read the pre-split folds `csv/processed/{COHORT}/5-fold/fold_{1..5}` directly.

## Training (original OSCAR pipeline)

```bash
uv run python -m train --config configs/config_self_attn_cross_attn_brca.yaml
uv run python -m train --config configs/config_self_attn_cross_attn_crc.yaml
uv run python -m train --config configs/config_self_attn_cross_attn_lung.yaml
```

Checkpoints and k-fold reports are written under `lightning_logs/` (`{log_name}_fold_{k}/checkpoints/`, `{log_name}_k5_report.json`).

Reproducibility: each fold re-seeds with `seed + fold_index`; set `train.num_workers: 0` to avoid DataLoader worker randomness.

| Cohort   | Config | Classes |
|----------|--------|---------|
| BRCA     | `config_self_attn_cross_attn_brca.yaml` | 4 |
| COADREAD | `config_self_attn_cross_attn_crc.yaml`   | 4 |
| LUNG     | `config_self_attn_cross_attn_lung.yaml`  | 2 |

## Matched 5-fold protocol

All numbers in the results table use the same protocol, implemented in `train_token_oscar_matched.py` and `train_baselines_matched.py`:

1. Outer folds: the pre-split `csv/processed/{COHORT}/5-fold/fold_{1..5}`. Each fold's validation file is the held-out test set.
2. Inner split: 20% stratified from the fold's training file (`inner_split(..., seed=42)`).
3. Scaling: `StandardScaler` fit on the training part only, separately for each stage.
4. Epoch selection: train up to 120 epochs, keep the first epoch with the maximum inner-validation macro-F1.
5. Retrain: reinitialize with seed 42, train on the full fold training file for the chosen number of epochs, then evaluate once on the held-out fold.
6. Report: mean ± std of macro-F1 and accuracy over the 5 folds.

Transductive graph baselines (MoGCN, DeepMoIC, MOGONET) build patient graphs on training plus evaluation samples, but their labels and loss use only the training samples.

## Reproduce: TokenOSCAR and OSCAR (1×128)

```bash
# TokenOSCAR, all 3 cohorts -> lightning_logs/token_oscar_matched/{report.json,table.md}
uv run python train_token_oscar_matched.py --cohort all
```

The OSCAR 1×128 numbers in the table come from the earlier matched runs (`lightning_logs/{brca,lung,coadread}_table51_matched/`, produced by `train_table51_matched.py`). That script is not part of the current branch, so this step cannot be reproduced from this tree alone.

## Reproduce: baselines (MoXGATE, MOT, DeepMoIC, MoGCN, MOGONET, moBRCA-net)

```bash
# all cohorts, all models -> lightning_logs/baselines_matched/{cohort}_report.json
uv run python train_baselines_matched.py --cohort all --models all

# a single cohort / subset of models
uv run python train_baselines_matched.py --cohort BRCA --models moxgate,mot,mobrca
```

Known limitations:
- The OSCAR 1×128 numbers come from the earlier matched runs (`lightning_logs/{brca,lung,coadread}_table51_matched/`). That script is not on this branch, so those rows cannot be regenerated from this tree.
- MoXGATE uses one config per cohort: `configs/config_moxgate_{brca,lung,crc}.yaml`. The LUNG and COADREAD files were deleted earlier and were rebuilt by copying the BRCA paper config and changing only data paths, `num_classes` (2 and 4) and `log_name`. Rebuilt runs reproduce the earlier MoXGATE numbers exactly, but the original files could not be checked.
- The baseline hyperparameters were not tuned per cohort. Some runs (e.g. MoGCN on BRCA, moBRCA-net on COADREAD) have large fold-to-fold variance.

## Results (matched 5-fold, mean ± std, %)

| Method | BRCA F1 / Acc | LUNG F1 / Acc | COADREAD F1 / Acc |
|--------|---------------|---------------|-------------------|
| **TokenOSCAR (128×32)** | **88.73±3.07 / 90.61±1.81** | **96.17±1.08 / 96.19±1.08** | 90.25±2.47 / 91.89±1.12 |
| OSCAR (1×128) | 87.01±3.41 / 88.85±2.58 | 95.72±1.70 / 95.75±1.69 | **93.43±2.96 / 94.13±3.09** |
| MoXGATE (paper) | 86.82±2.69 / 89.23±1.65 | 96.02±0.76 / 96.04±0.75 | 86.09±2.86 / 88.30±2.93 |
| MOT | 89.28±3.19 / 90.80±2.11 | 95.87±1.52 / 95.90±1.51 | 87.27±2.13 / 90.11±2.19 |
| DeepMoIC | 81.17±2.09 / 85.13±1.43 | 92.92±2.61 / 92.97±2.59 | 80.96±10.01 / 86.08±5.65 |
| MoGCN | 81.29±2.08 / 85.33±1.34 | 94.10±2.47 / 94.14±2.45 | 84.39±3.32 / 86.95±2.13 |
| MOGONET | 74.55±2.71 / 82.39±0.62 | 93.45±2.61 / 93.55±2.53 | 91.37±2.70 / 92.33±1.16 |
| moBRCA-net | 83.05±3.20 / 85.91±1.58 | 95.43±1.58 / 95.46±1.56 | 80.60±5.73 / 83.30±4.02 |

The six baseline rows (MoXGATE, MOT, DeepMoIC, MoGCN, MOGONET, moBRCA-net) come from `train_baselines_matched.py` (`lightning_logs/baselines_matched/`). MoXGATE matches its earlier numbers; the other five replace the earlier `table51` values.

## Visualize attention (gradient feature importance)

Use the same config as training and a checkpoint from `lightning_logs/.../checkpoints/`.

```bash
python -m visualize_attention \
  --config configs/config_self_attn_cross_attn.yaml \
  --checkpoint <path-to-checkpoint.ckpt> \
  --output-dir attention_plots/grad_importance \
  --top-k-features 20 \
  --use-true-label \
  --class-names "LumA,LumB,HER2,Basal"
```

Example (BRCA; adjust paths after your run):

```bash
python -m visualize_attention \
  --config configs/config_self_attn_cross_attn.yaml \
  --checkpoint lightning_logs/lung_self_attn_cross_attn_fold_1/version_0/checkpoints/best_val_f1_epoch=XX_val_f1=0.XXXX.ckpt \
  --output-dir attention_plots/grad_importance \
  --top-k-features 20 \
  --use-true-label \
  --class-names "LumA,LumB,HER2,Basal"
```

For visualization, set **`data.current_fold`** in the YAML to the fold you trained (0-based). Use **`--class-names`** that match your label encoding for that cohort.

On Linux/macOS you can also run:

```bash
bash visualize.sh
```

(Update paths inside `visualize.sh` to match your config and checkpoint.)

Plots are saved under `attention_plots/grad_importance/`.

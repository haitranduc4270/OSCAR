# Cancer Multi-Omics Experiment

Train multi-omics classification models (self-attention + cross-attention fusion) with PyTorch Lightning.

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

Update `data.*_path` and `data.y_path` in each config if your CSVs are elsewhere.

## Training

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

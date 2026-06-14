# Cancer Multi-Omics Experiment

This repository trains multi-omics classification models with PyTorch Lightning and includes utilities to visualize gradient-based attention / feature importance.

## 1) Set up environment with `uv`

Prerequisites:

- Python **3.10** (required for pinned `torch==2.5.1+cu121`)
- [`uv`](https://docs.astral.sh/uv/)

From the project root:

```bash
# one-time: let uv install Python 3.10 if needed
uv python install 3.10

# create .venv and install deps from pyproject.toml + uv.lock
uv sync
```

Verify:

```bash
uv run python -V          # should print 3.10.x
uv run python -c "import torch; print(torch.__version__)"
```

Run training without activating the venv:

```bash
uv run python -m train --config configs/config_self_attn_cross_attn.yaml
```

Or activate the venv (optional):

```powershell
# Windows PowerShell
.venv\Scripts\activate
python -m train --config configs/config_self_attn_cross_attn.yaml
```

```bash
# Linux / macOS
source .venv/bin/activate
python -m train --config configs/config_self_attn_cross_attn.yaml
```

> **Note:** Do not use Python 3.12+ for this project — PyTorch 2.5.1+cu121 has no wheel there.
> Legacy `pip` install is still documented in `requirements.txt` + `requirements-torch-cu121.txt`.

## 2) Download dataset

1. Download the OSCAR dataset from Kaggle: [oscar-dataset](https://www.kaggle.com/datasets/hitrnc/oscar-data).
2. Extract it so processed CSVs live under the repo root, for example:

```text
OSCAR/csv/processed/
  LUNG/
    5-fold/
      fold_1/ ...
      fold_5/ ...
  BRCA/
    ...
  COADREAD/
    ...
```

Pre-generated 5-fold splits are included under each cohort’s `5-fold/` directory. Point `data.pre_split_base` in the config at the matching folder (see below).

## 3) Run training

Edit `configs/config_self_attn_cross_attn.yaml` before training:

- **`data.*_path`** and **`data.y_path`**: absolute or relative paths to the omics and label CSVs for the cohort you want.
- **`data.pre_split_base`**: e.g. `csv/processed/BRCA/5-fold`, `csv/processed/LUNG/5-fold`, or `csv/processed/COADREAD/5-fold`.
- **`model.input_dims`**: feature counts per omic (must match the CSV row counts):

| Cohort   | Proteomics | CNV   | mRNA  |
|----------|------------|-------|-------|
| BRCA     | 223        | 19273 | 19580 |
| COADREAD | 153        | 24776 | 20530 |
| LUNG     | 180        | 24776 | 20530 |

- **`logging.log_name`**: optional run name (fold suffixes are added automatically during k-fold training).

Training entrypoint:

```bash
python -m train --config configs/config_self_attn_cross_attn.yaml
```

Notes:

- Configs live in `configs/` (default experiment: `config_self_attn_cross_attn.yaml`).
- Logs and checkpoints are written under `lightning_logs/` (see `logging.log_dir` and `logging.log_name` in the config).
- Stratified 5-fold training is enabled when `data.use_stratified_kfold: true` and `data.pre_split_base` is set.

## 4) Visualize attention (gradient feature importance)

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

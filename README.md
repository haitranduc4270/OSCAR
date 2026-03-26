# Cancer Multi-Omics Experiment

This repository trains multi-omics classification models with PyTorch Lightning and includes utilities to visualize gradient-based attention/feature importance.

## 1) Set up environment with `uv`

Prerequisites:
- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)

From the project root:

```bash
# create/update the virtual environment from pyproject.toml + uv.lock
uv sync
```

Verify Python is available:

```bash
python -V
```

## 2) Run training

Training entrypoint:

```bash
python -m train --config <path-to-config>
```

Example:

```bash
python -m train --config configs/config_self_attn_cross_attn_mogcn_data_brca.yaml
```

Notes:
- Configs are in `configs/`.
- Logs/checkpoints are written under `lightning_logs/` (configurable via `logging.log_dir` and `logging.log_name`).
- Data file paths are configured inside each YAML under `data.*`.

## 3) Visualize attention (gradient feature importance)

Visualization entrypoint:

```bash
python -m visualize_attention \
  --config <path-to-config> \
  --checkpoint <path-to-checkpoint.ckpt> \
  --output-dir attention_plots/grad_importance \
  --top-k-features 20 \
  --use-true-label \
  --class-names "LumA,LumB,HER2,Basal"
```

Example:

```bash
python -m visualize_attention \
  --config configs/config_self_attn_cross_attn_mogcn_data_brca.yaml \
  --checkpoint lightning_logs/brca_self_attn_cross_attn_fusion_mogcn_data_fold_4/version_0/checkpoints/epoch=19-step=260.ckpt \
  --output-dir attention_plots/grad_importance \
  --top-k-features 20 \
  --use-true-label \
  --class-names "LumA,LumB,HER2,Basal"
```

You can also use the helper script:

```bash
bash visualize.sh
```

Generated plots are saved to `attention_plots/grad_importance/`.


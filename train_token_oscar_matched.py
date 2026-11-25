"""Matched 5-fold runner for TokenOSCARModel across BRCA, LUNG, and COADREAD.

Protocol matches train_table51_matched.py:
  outer pre-split 5 folds -> inner 20% stratified val -> first-max val macro-F1
  -> retrain on full fold train -> score once on fold val. Epoch budget 120.

Usage:
  uv run python train_token_oscar_matched.py --cohort all
  uv run python train_token_oscar_matched.py --cohort LUNG,COADREAD
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from models.token_oscar import TokenOSCARModel
from oscar_utils import _read_split_matrix, configure_runtime, read_pair, run_oscar_fold, summarize

OUT_DIR = Path("lightning_logs/token_oscar_matched")
MATCHED_MAX_EPOCHS = 120

COHORTS = {
    "BRCA": {
        "prefix": "BRCA",
        "folds": Path("csv/processed/BRCA/5-fold"),
        "oscar": "configs/config_self_attn_cross_attn_brca.yaml",
    },
    "LUNG": {
        "prefix": "LUNG",
        "folds": Path("csv/processed/LUNG/5-fold"),
        "oscar": "configs/config_self_attn_cross_attn_lung.yaml",
    },
    "COADREAD": {
        "prefix": "COADREAD",
        "folds": Path("csv/processed/COADREAD/5-fold"),
        "oscar": "configs/config_self_attn_cross_attn_crc.yaml",
    },
}


def _stems(prefix: str) -> dict[str, str]:
    return {
        "mRNA": f"{prefix}_mRNA",
        "CNV": f"{prefix}_CNV_1_channel",
        "Proteomics": f"{prefix}_proteomics",
        "label": f"{prefix}_label_num",
    }


def load_fold(fold_dir: Path, reference_columns: dict[str, pd.Index], prefix: str) -> dict:
    stems = _stems(prefix)
    pairs = {name: read_pair(fold_dir, stems[name]) for name in ("mRNA", "CNV", "Proteomics")}
    columns = list(pairs["mRNA"][0].columns)
    for name, (train, _val) in pairs.items():
        if list(train.columns) != columns:
            raise ValueError(f"{name} sample order differs in {fold_dir.name}")
        if not train.index.equals(reference_columns[name]):
            raise ValueError(f"{name} feature order differs in {fold_dir.name}")
    y_dev = pd.read_csv(fold_dir / f"{stems['label']}_train.csv").iloc[:, 0].to_numpy(dtype="int64")
    y_test = pd.read_csv(fold_dir / f"{stems['label']}_val.csv").iloc[:, 0].to_numpy(dtype="int64")
    if len(y_dev) != len(columns) or len(y_test) != pairs["mRNA"][1].shape[1]:
        raise ValueError(f"Label count does not match samples in {fold_dir.name}")
    dev = {name: train.T.to_numpy(dtype="float64") for name, (train, _) in pairs.items()}
    test = {name: val.T.to_numpy(dtype="float64") for name, (_, val) in pairs.items()}
    return {"dev": dev, "test": test, "y_dev": y_dev, "y_test": y_test}

# Matched OSCAR (1x128) numbers from docs/PAPERS_TABLE.md for side-by-side comparison.
OSCAR_BASELINE = {
    "BRCA": (87.01, 3.41, 88.85, 2.58),
    "LUNG": (95.72, 1.70, 95.75, 1.69),
    "COADREAD": (93.43, 2.96, 94.13, 3.09),
}


def build_token_oscar(
    oscar_cfg: dict,
    dims: dict[str, int],
    num_tokens: int = 256,
    token_dim: int = 32,
    d_mid: int = 256,
    n_heads: int = 4,
) -> TokenOSCARModel:
    model_cfg = oscar_cfg["model"]
    train_cfg = oscar_cfg["train"]
    return TokenOSCARModel(
        input_dims=dims,
        num_classes=int(model_cfg["num_classes"]),
        num_tokens=num_tokens,
        token_dim=token_dim,
        d_mid=d_mid,
        n_heads=n_heads,
        n_layers=int(model_cfg["n_layers"]),
        fusion_hidden=int(model_cfg["fusion_hidden"]),
        dropout=float(model_cfg["dropout"]),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
        use_cross_attention=bool(model_cfg["use_cross_attention"]),
        use_gated_fusion=bool(model_cfg["use_gated_fusion"]),
        use_original_for_proteomics=bool(model_cfg["use_original_for_proteomics"]),
    )


def run_cohort(
    cohort_name: str,
    device: torch.device,
    num_tokens: int = 256,
    token_dim: int = 32,
    d_mid: int = 256,
    n_heads: int = 4,
) -> dict:
    spec = COHORTS[cohort_name]
    oscar_cfg = yaml.safe_load(open(spec["oscar"]))
    folds_dir = spec["folds"]
    prefix = spec["prefix"]
    stems = _stems(prefix)

    reference_columns = {
        "mRNA": _read_split_matrix(folds_dir / "fold_1", stems["mRNA"]).index,
        "CNV": _read_split_matrix(folds_dir / "fold_1", stems["CNV"]).index,
        "Proteomics": _read_split_matrix(folds_dir / "fold_1", stems["Proteomics"]).index,
    }
    folds = [load_fold(folds_dir / f"fold_{i}", reference_columns, prefix) for i in range(1, 6)]
    dims = {name: folds[0]["dev"][name].shape[1] for name in ("mRNA", "CNV", "Proteomics")}
    print(f"[{cohort_name}] device={device} dims={dims} T={num_tokens} M={token_dim} d_mid={d_mid}", flush=True)

    probe = build_token_oscar(oscar_cfg, dims, num_tokens, token_dim, d_mid, n_heads)
    params = sum(p.numel() for p in probe.parameters())
    del probe

    rows = []
    for fold_id, fold in enumerate(folds, start=1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        row = run_oscar_fold(
            oscar_cfg,
            dims,
            fold,
            device,
            MATCHED_MAX_EPOCHS,
            build=lambda: build_token_oscar(oscar_cfg, dims, num_tokens, token_dim, d_mid, n_heads),
        )
        if device.type == "cuda":
            row["peak_cuda_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024**2))
        row["fold"] = fold_id
        rows.append(row)
        print(
            f"[{cohort_name}] fold {fold_id}: epoch {row['chosen_epoch']} "
            f"test macro-F1 {row['test']['f1_macro']:.4f} acc {row['test']['accuracy']:.4f} "
            f"peak_mb {row.get('peak_cuda_mb', 0):.0f}",
            flush=True,
        )

    summary = summarize(rows)
    print(
        f"[{cohort_name}] mean macro-F1 {100*summary['f1_macro_mean']:.2f} "
        f"+/- {100*summary['f1_macro_std']:.2f} | acc {100*summary['accuracy_mean']:.2f} "
        f"+/- {100*summary['accuracy_std']:.2f}",
        flush=True,
    )
    return {
        "dims": dims,
        "params_total": params,
        "folds": rows,
        "summary": summary,
    }


def format_table(cohorts_data: dict[str, dict]) -> str:
    lines = [
        "| Cohort | TokenOSCAR F1 (%) | TokenOSCAR Acc (%) | OSCAR (1x128) F1 (%) | Δ F1 | Params |",
        "|--------|-------------------|--------------------|----------------------|------|--------|",
    ]
    for name in ("BRCA", "LUNG", "COADREAD"):
        if name not in cohorts_data:
            continue
        s = cohorts_data[name]["summary"]
        f1_m, f1_s = 100 * s["f1_macro_mean"], 100 * s["f1_macro_std"]
        acc_m, acc_s = 100 * s["accuracy_mean"], 100 * s["accuracy_std"]
        b_f1, b_f1_s, _, _ = OSCAR_BASELINE[name]
        delta = f1_m - b_f1
        params = cohorts_data[name]["params_total"]
        lines.append(
            f"| {name} | {f1_m:.2f}±{f1_s:.2f} | {acc_m:.2f}±{acc_s:.2f} "
            f"| {b_f1:.2f}±{b_f1_s:.2f} | {delta:+.2f} | {params:,} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all", help="all or comma-separated BRCA,LUNG,COADREAD")
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--d-mid", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    configure_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.cohort == "all":
        targets = ["BRCA", "LUNG", "COADREAD"]
    else:
        targets = [c.strip().upper() for c in args.cohort.split(",") if c.strip()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "report.json"
    cohorts_data: dict[str, dict] = {}
    if report_path.exists():
        cohorts_data = json.load(open(report_path)).get("cohorts", {})

    for name in targets:
        cohorts_data[name] = run_cohort(
            name,
            device,
            num_tokens=args.num_tokens,
            token_dim=args.token_dim,
            d_mid=args.d_mid,
            n_heads=args.n_heads,
        )

    seq_len = 3 * args.num_tokens + 1
    report = {
        "protocol": {
            "inner_val_fraction": 0.2,
            "inner_seed": 42,
            "max_epochs": MATCHED_MAX_EPOCHS,
            "selection": "first maximum of inner validation macro-F1",
            "retrain": "all samples in the fold training file",
            "num_tokens": args.num_tokens,
            "token_dim": args.token_dim,
            "d_mid": args.d_mid,
            "n_heads": args.n_heads,
            "architecture": (
                f"TokenOSCAR: each omic -> (B, {args.num_tokens}, {args.token_dim}) "
                f"-> {args.num_tokens}x{args.num_tokens} cross-attn + gate -> {seq_len}-token self-attn -> CLS"
            ),
        },
        "cohorts": cohorts_data,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    table = format_table(cohorts_data)
    (args.out_dir / "table.md").write_text(table, encoding="utf-8")
    print("\n" + table, flush=True)
    print(f"Saved report to {report_path}", flush=True)


if __name__ == "__main__":
    main()

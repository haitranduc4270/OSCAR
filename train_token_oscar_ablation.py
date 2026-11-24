"""Ablation study (Table 5.2 & Table 5.3) for TokenOSCAR (256x32) across BRCA, LUNG, COADREAD.

Reproduces the thesis ablations under the exact matched 5-fold protocol:
  Table 5.2 (Component ablation):
    - concat:       Concatenation (no cross-attn, no gate, no self-attn)
    - no_self_attn: Without self-attention (ordered cross-attn + gate, mean-pool concat)
    - no_gate:      Without gated fusion (ordered cross-attn + self-attn, no gate)
    - full:         TokenOSCAR (256x32) full model

  Table 5.3 (Cross-attention order ablation):
    - order_reverse: Proteomics -> mRNA -> CNV
    - order_mixed:   mRNA -> CNV -> Proteomics
    - full:          CNV -> mRNA -> Proteomics (biological order)

Usage:
  uv run python train_token_oscar_ablation.py --cohort all --variants all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from models.token_oscar import TokenOSCARModel
from pathway_data import _read_split_matrix
from train_five_fold import run_oscar_fold, summarize
from train_pathway_tokens import configure_runtime
from train_token_oscar_matched import COHORTS, MATCHED_MAX_EPOCHS, _stems, load_fold

OUT_DIR = Path("lightning_logs/token_oscar_ablation")

VARIANTS: dict[str, dict] = {
    "full": {
        "label": "TokenOSCAR (256x32)",
        "use_cross_attention": True,
        "use_gated_fusion": True,
        "use_self_attention": True,
        "cross_order": ("CNV", "mRNA", "Proteomics"),
    },
    "concat": {
        "label": "Concatenation",
        "use_cross_attention": False,
        "use_gated_fusion": False,
        "use_self_attention": False,
        "cross_order": ("CNV", "mRNA", "Proteomics"),
    },
    "no_self_attn": {
        "label": "Without self-attention",
        "use_cross_attention": True,
        "use_gated_fusion": True,
        "use_self_attention": False,
        "cross_order": ("CNV", "mRNA", "Proteomics"),
    },
    "no_gate": {
        "label": "Without gated fusion",
        "use_cross_attention": True,
        "use_gated_fusion": False,
        "use_self_attention": True,
        "cross_order": ("CNV", "mRNA", "Proteomics"),
    },
    "no_cross_attn": {
        "label": "Without cross-attention (self-attn only)",
        "use_cross_attention": False,
        "use_gated_fusion": False,
        "use_self_attention": True,
        "cross_order": ("CNV", "mRNA", "Proteomics"),
    },
    "no_gate_no_self_attn": {
        "label": "Without gate & self-attention (cross-attn only)",
        "use_cross_attention": True,
        "use_gated_fusion": False,
        "use_self_attention": False,
        "cross_order": ("CNV", "mRNA", "Proteomics"),
    },
    "order_reverse": {
        "label": "Proteomics -> mRNA -> CNV (reverse)",
        "use_cross_attention": True,
        "use_gated_fusion": True,
        "use_self_attention": True,
        "cross_order": ("Proteomics", "mRNA", "CNV"),
    },
    "order_mixed": {
        "label": "mRNA -> CNV -> Proteomics (mixed)",
        "use_cross_attention": True,
        "use_gated_fusion": True,
        "use_self_attention": True,
        "cross_order": ("mRNA", "CNV", "Proteomics"),
    },
}


def build_variant(
    oscar_cfg: dict,
    dims: dict[str, int],
    vspec: dict,
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
        use_cross_attention=vspec["use_cross_attention"],
        use_gated_fusion=vspec["use_gated_fusion"],
        use_original_for_proteomics=bool(model_cfg["use_original_for_proteomics"]),
        use_self_attention=vspec["use_self_attention"],
        cross_order=vspec["cross_order"],
    )


def format_tables(results: dict[str, dict[str, dict]]) -> str:
    def _cell(vkey: str, cohort: str, metric: str) -> str:
        if vkey not in results or cohort not in results[vkey]:
            return "—"
        s = results[vkey][cohort]["summary"]
        m = 100 * s[f"{metric}_mean"]
        sd = 100 * s[f"{metric}_std"]
        return f"{m:.2f}±{sd:.2f}"

    lines = [
        "# TokenOSCAR (256x32) Ablation Study (Matched 5-Fold)",
        "",
        "## Table 5.2: Ablation of model components",
        "",
        "### Accuracy (%, mean ± std)",
        "| Model variant | BRCA | LUNG | COADREAD | Avg Acc |",
        "|---|---|---|---|---|",
    ]
    for vkey in ("concat", "no_cross_attn", "no_self_attn", "no_gate_no_self_attn", "no_gate", "full"):
        vals = [
            100 * results[vkey][c]["summary"]["accuracy_mean"]
            for c in ("BRCA", "LUNG", "COADREAD")
            if vkey in results and c in results[vkey]
        ]
        avg = f"{sum(vals)/len(vals):.2f}" if vals else "—"
        lines.append(
            f"| {VARIANTS[vkey]['label']} | {_cell(vkey, 'BRCA', 'accuracy')} "
            f"| {_cell(vkey, 'LUNG', 'accuracy')} | {_cell(vkey, 'COADREAD', 'accuracy')} | {avg} |"
        )

    lines += [
        "",
        "### Macro-F1 (%, mean ± std)",
        "| Model variant | BRCA | LUNG | COADREAD | Avg F1 |",
        "|---|---|---|---|---|",
    ]
    for vkey in ("concat", "no_cross_attn", "no_self_attn", "no_gate_no_self_attn", "no_gate", "full"):
        vals = [
            100 * results[vkey][c]["summary"]["f1_macro_mean"]
            for c in ("BRCA", "LUNG", "COADREAD")
            if vkey in results and c in results[vkey]
        ]
        avg = f"{sum(vals)/len(vals):.2f}" if vals else "—"
        lines.append(
            f"| {VARIANTS[vkey]['label']} | {_cell(vkey, 'BRCA', 'f1_macro')} "
            f"| {_cell(vkey, 'LUNG', 'f1_macro')} | {_cell(vkey, 'COADREAD', 'f1_macro')} | {avg} |"
        )

    lines += [
        "",
        "## Table 5.3: Effect of cross-attention order",
        "",
        "### Accuracy (%, mean ± std)",
        "| Ordering | BRCA | LUNG | COADREAD | Avg Acc |",
        "|---|---|---|---|---|",
    ]
    for vkey in ("order_reverse", "order_mixed", "full"):
        vals = [
            100 * results[vkey][c]["summary"]["accuracy_mean"]
            for c in ("BRCA", "LUNG", "COADREAD")
            if vkey in results and c in results[vkey]
        ]
        avg = f"{sum(vals)/len(vals):.2f}" if vals else "—"
        label = (
            "CNV -> mRNA -> Proteomics (TokenOSCAR)"
            if vkey == "full"
            else VARIANTS[vkey]["label"]
        )
        lines.append(
            f"| {label} | {_cell(vkey, 'BRCA', 'accuracy')} "
            f"| {_cell(vkey, 'LUNG', 'accuracy')} | {_cell(vkey, 'COADREAD', 'accuracy')} | {avg} |"
        )

    lines += [
        "",
        "### Macro-F1 (%, mean ± std)",
        "| Ordering | BRCA | LUNG | COADREAD | Avg F1 |",
        "|---|---|---|---|---|",
    ]
    for vkey in ("order_reverse", "order_mixed", "full"):
        vals = [
            100 * results[vkey][c]["summary"]["f1_macro_mean"]
            for c in ("BRCA", "LUNG", "COADREAD")
            if vkey in results and c in results[vkey]
        ]
        avg = f"{sum(vals)/len(vals):.2f}" if vals else "—"
        label = (
            "CNV -> mRNA -> Proteomics (TokenOSCAR)"
            if vkey == "full"
            else VARIANTS[vkey]["label"]
        )
        lines.append(
            f"| {label} | {_cell(vkey, 'BRCA', 'f1_macro')} "
            f"| {_cell(vkey, 'LUNG', 'f1_macro')} | {_cell(vkey, 'COADREAD', 'f1_macro')} | {avg} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all", help="all or comma-separated BRCA,LUNG,COADREAD")
    parser.add_argument(
        "--variants",
        default="all",
        help="all or comma-separated keys from: " + ",".join(VARIANTS.keys()),
    )
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--d-mid", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--force", action="store_true", help="Re-run even if already in report.json")
    args = parser.parse_args()

    configure_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cohorts = (
        ["BRCA", "LUNG", "COADREAD"]
        if args.cohort == "all"
        else [c.strip().upper() for c in args.cohort.split(",") if c.strip()]
    )
    vkeys = (
        list(VARIANTS.keys())
        if args.variants == "all"
        else [v.strip() for v in args.variants.split(",") if v.strip()]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "report.json"
    results: dict[str, dict[str, dict]] = {}
    if report_path.exists():
        results = json.load(open(report_path)).get("variants", {})

    # Pre-load folds per cohort once
    cohort_cache: dict[str, tuple[dict, list[dict], dict[str, int]]] = {}
    for cohort_name in cohorts:
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
        cohort_cache[cohort_name] = (oscar_cfg, folds, dims)

    for vkey in vkeys:
        vspec = VARIANTS[vkey]
        results.setdefault(vkey, {})
        for cohort_name in cohorts:
            if not args.force and cohort_name in results[vkey]:
                print(f"[SKIP] {vkey} / {cohort_name} already in {report_path}", flush=True)
                continue

            oscar_cfg, folds, dims = cohort_cache[cohort_name]
            probe = build_variant(
                oscar_cfg, dims, vspec, args.num_tokens, args.token_dim, args.d_mid, args.n_heads
            )
            params = sum(p.numel() for p in probe.parameters())
            del probe

            print(
                f"\n=== [{vkey}] {cohort_name} | T={args.num_tokens} M={args.token_dim} params={params:,} ===",
                flush=True,
            )
            rows = []
            for fold_id, fold in enumerate(folds, start=1):
                row = run_oscar_fold(
                    oscar_cfg,
                    dims,
                    fold,
                    device,
                    MATCHED_MAX_EPOCHS,
                    build=lambda: build_variant(
                        oscar_cfg,
                        dims,
                        vspec,
                        args.num_tokens,
                        args.token_dim,
                        args.d_mid,
                        args.n_heads,
                    ),
                )
                row["fold"] = fold_id
                rows.append(row)
                print(
                    f"  fold {fold_id}: ep={row['chosen_epoch']} "
                    f"F1={100*row['test']['f1_macro']:.2f}% acc={100*row['test']['accuracy']:.2f}%",
                    flush=True,
                )

            summary = summarize(rows)
            print(
                f"  => [{vkey}] {cohort_name}: "
                f"F1 {100*summary['f1_macro_mean']:.2f}±{100*summary['f1_macro_std']:.2f} | "
                f"Acc {100*summary['accuracy_mean']:.2f}±{100*summary['accuracy_std']:.2f}",
                flush=True,
            )
            results[vkey][cohort_name] = {
                "params_total": params,
                "folds": rows,
                "summary": summary,
            }

            report = {
                "protocol": {
                    "num_tokens": args.num_tokens,
                    "token_dim": args.token_dim,
                    "d_mid": args.d_mid,
                    "n_heads": args.n_heads,
                    "max_epochs": MATCHED_MAX_EPOCHS,
                    "inner_val_fraction": 0.2,
                    "inner_seed": 42,
                },
                "variants": results,
            }
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            (args.out_dir / "table.md").write_text(format_tables(results), encoding="utf-8")

    table_md = format_tables(results)
    (args.out_dir / "table.md").write_text(table_md, encoding="utf-8")
    print("\n" + table_md, flush=True)
    print(f"Saved report to {report_path}", flush=True)


if __name__ == "__main__":
    main()

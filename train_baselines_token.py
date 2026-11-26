"""Matched 5-fold comparison of MOT and MoXGATE with OmicToTokens (256x32).

Replaces the single-vector per-omic projection in MOT and MoXGATE with
OmicToTokens (T=256, M=32, d_mid=256) from TokenOSCAR while keeping each
method's downstream attention, loss, optimizer, and 5-fold protocol intact.

Usage:
  uv run python train_baselines_token.py --cohort all --models mot,moxgate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from models.base import BaseLightningModel
from models.moxgate import FocalLoss, JointCrossAttention
from models.token_oscar import OmicToTokens
from oscar_utils import configure_runtime, run_oscar_fold, summarize
from train_baselines_matched import MOXGATE_CONFIGS, _run_batch_model_fold
from train_token_oscar_matched import COHORTS, MATCHED_MAX_EPOCHS, _read_split_matrix, _stems, load_fold

OUT_DIR = Path("lightning_logs/baselines_token_256x32")
ORIG_DIR = Path("lightning_logs/baselines_matched")
ALL_MODELS = ("mot", "moxgate")


class TokenMOTModel(nn.Module):
    """MOT where each omic is projected to (B, T, M) via OmicToTokens."""

    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        num_tokens: int = 256,
        token_dim: int = 32,
        d_mid: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
        omic_order: tuple[str, ...] = ("CNV", "mRNA", "Proteomics"),
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.omic_order = tuple(k for k in omic_order if k in input_dims)
        self.proj = nn.ModuleDict(
            {
                name: OmicToTokens(
                    input_dim=input_dims[name],
                    num_tokens=num_tokens,
                    token_dim=token_dim,
                    d_mid=d_mid,
                    dropout=dropout,
                )
                for name in self.omic_order
            }
        )
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(token_dim),
            enable_nested_tensor=False,
        )
        self.classifier = nn.Sequential(
            nn.Linear(token_dim * len(self.omic_order), 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = torch.cat([self.proj[k](batch[k]) for k in self.omic_order], dim=1)
        h = self.encoder(tokens)
        h = h.view(h.size(0), len(self.omic_order), self.num_tokens, self.token_dim).mean(dim=2)
        return self.classifier(h.reshape(h.size(0), -1))


class TokenModalitySelfAttention(nn.Module):
    """MoXGATE eq. (3-5) on (B, T, M) tokens: Z = AV + H (no FFN, no LayerNorm)."""

    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, need_weights=False)
        return out + x


class TokenMoXGATEModel(BaseLightningModel):
    """MoXGATE where each omic is projected to (B, T, M) via OmicToTokens."""

    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        num_tokens: int = 256,
        token_dim: int = 32,
        d_mid: int = 256,
        classifier_dim: int = 128,
        attn_dropout: float = 0.1,
        classifier_dropout: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        n_heads: int = 4,
        joint_n_heads: int = 4,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        lambda_modality: float = 1e-2,
        lambda_cross_frobenius: float = 1e-2,
    ):
        super().__init__(num_classes=num_classes, lr=lr, weight_decay=weight_decay)
        self.valid_omics = list(input_dims.keys())
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.use_focal_loss = use_focal_loss
        self.lambda_modality = float(lambda_modality)
        self.lambda_cross_frobenius = float(lambda_cross_frobenius)

        self.encoders = nn.ModuleDict(
            {
                name: OmicToTokens(
                    input_dim=input_dims[name],
                    num_tokens=num_tokens,
                    token_dim=token_dim,
                    d_mid=d_mid,
                    dropout=attn_dropout,
                )
                for name in self.valid_omics
            }
        )
        self.modality_self_attn = nn.ModuleDict(
            {
                name: TokenModalitySelfAttention(
                    embed_dim=token_dim, n_heads=n_heads, dropout=attn_dropout
                )
                for name in self.valid_omics
            }
        )
        self.joint_cross_attn = JointCrossAttention(
            embed_dim=token_dim, n_heads=joint_n_heads, dropout=attn_dropout
        )
        self.modality_logits = nn.Parameter(torch.zeros(len(self.valid_omics)))
        self.classifier = nn.Sequential(
            nn.Linear(token_dim, classifier_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_dim, num_classes),
        )
        self.focal_loss = FocalLoss(gamma=focal_gamma)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        latents = [self.modality_self_attn[name](self.encoders[name](batch[name])) for name in self.valid_omics]
        tokens = torch.cat(latents, dim=1)  # (B, M_omics * T, token_dim)
        fused_tokens = self.joint_cross_attn(tokens)
        fused_per_omic = fused_tokens.view(
            fused_tokens.size(0), len(self.valid_omics), self.num_tokens, self.token_dim
        ).mean(dim=2)
        weights = torch.softmax(self.modality_logits, dim=0)
        fused = torch.einsum("m,bme->be", weights, fused_per_omic)
        return self.classifier(fused)

    def _compute_loss(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.focal_loss(logits, y) if self.use_focal_loss else F.cross_entropy(logits, y)
        if self.lambda_modality > 0:
            w = torch.softmax(self.modality_logits, dim=0)
            loss = loss + self.lambda_modality * (w - torch.ones_like(w)).pow(2).sum()
        if self.lambda_cross_frobenius > 0:
            loss = loss + self.lambda_cross_frobenius * self.joint_cross_attn.frobenius_penalty()
        return loss


def run_model_on_cohort(
    cohort_name: str,
    model_name: str,
    folds: list[dict],
    dims: dict[str, int],
    num_classes: int,
    device: torch.device,
    num_tokens: int,
    token_dim: int,
    d_mid: int,
) -> dict:
    rows = []
    for fold_id, fold in enumerate(folds, start=1):
        if model_name == "mot":
            row = _run_batch_model_fold(
                fold,
                device,
                lambda d: TokenMOTModel(
                    d, num_classes, num_tokens=num_tokens, token_dim=token_dim, d_mid=d_mid
                ),
                lr=2e-4,
                weight_decay=1e-4,
            )
        elif model_name == "moxgate":
            cfg = yaml.safe_load(open(MOXGATE_CONFIGS[cohort_name]))
            mcfg, tcfg = cfg["model"], cfg["train"]
            build_mg = lambda: TokenMoXGATEModel(
                input_dims=dims,
                num_classes=int(mcfg["num_classes"]),
                num_tokens=num_tokens,
                token_dim=token_dim,
                d_mid=d_mid,
                attn_dropout=float(mcfg["dropout"]),
                lr=float(tcfg["lr"]),
                weight_decay=float(tcfg["weight_decay"]),
            )
            row = run_oscar_fold(cfg, dims, fold, device, MATCHED_MAX_EPOCHS, build=build_mg)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        row["fold"] = fold_id
        rows.append(row)
        print(
            f"[{cohort_name}][{model_name}_256x32] fold {fold_id}: ep {row['chosen_epoch']} "
            f"F1 {row['test']['f1_macro']:.4f} acc {row['test']['accuracy']:.4f}",
            flush=True,
        )
    return {"folds": rows, "summary": summarize(rows)}


def write_comparison_table(out_dir: Path) -> None:
    cohorts = ("BRCA", "LUNG", "COADREAD")
    lines = ["# MOT & MoXGATE: Original vs + OmicToTokens (256x32)\n"]
    for metric, title in (("accuracy", "Accuracy (%, mean ± std)"), ("f1_macro", "Macro-F1 (%, mean ± std)")):
        lines.append(f"## {title}\n")
        lines.append("| Model | BRCA | LUNG | COADREAD | Avg | Δ vs Orig |")
        lines.append("|---|---|---|---|---|---|")
        for m in ("mot", "moxgate"):
            orig_vals, tok_vals = [], []
            orig_cells, tok_cells = [], []
            for c in cohorts:
                orig_p = ORIG_DIR / f"{c.lower()}_report.json"
                tok_p = out_dir / f"{c.lower()}_report.json"
                if orig_p.exists():
                    s = json.load(open(orig_p))["models"][m]["summary"]
                    mean_v, std_v = s[f"{metric}_mean"] * 100, s[f"{metric}_std"] * 100
                    orig_vals.append(mean_v)
                    orig_cells.append(f"{mean_v:.2f} ± {std_v:.2f}")
                else:
                    orig_cells.append("—")
                if tok_p.exists() and m in json.load(open(tok_p)).get("models", {}):
                    s = json.load(open(tok_p))["models"][m]["summary"]
                    mean_v, std_v = s[f"{metric}_mean"] * 100, s[f"{metric}_std"] * 100
                    tok_vals.append(mean_v)
                    tok_cells.append(f"{mean_v:.2f} ± {std_v:.2f}")
                else:
                    tok_cells.append("—")
            orig_avg = sum(orig_vals) / len(orig_vals) if len(orig_vals) == 3 else None
            tok_avg = sum(tok_vals) / len(tok_vals) if len(tok_vals) == 3 else None
            orig_avg_s = f"{orig_avg:.2f}" if orig_avg is not None else "—"
            tok_avg_s = f"{tok_avg:.2f}" if tok_avg is not None else "—"
            delta_s = f"{tok_avg - orig_avg:+.2f}" if (orig_avg is not None and tok_avg is not None) else "—"
            lines.append(f"| {m.upper()} (orig) | {' | '.join(orig_cells)} | {orig_avg_s} | 0.00 |")
            lines.append(f"| {m.upper()} + 256×32 | {' | '.join(tok_cells)} | {tok_avg_s} | {delta_s} |")
        lines.append("")
    (out_dir / "table.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all", help="all or comma-separated BRCA,LUNG,COADREAD")
    parser.add_argument("--models", default="all", help="all or comma-separated mot,moxgate")
    parser.add_argument("--num-tokens", type=int, default=256)
    parser.add_argument("--token-dim", type=int, default=32)
    parser.add_argument("--d-mid", type=int, default=256)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    configure_runtime()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cohorts = (
        list(COHORTS.keys())
        if args.cohort == "all"
        else [c.strip().upper() for c in args.cohort.split(",") if c.strip()]
    )
    models = (
        list(ALL_MODELS)
        if args.models == "all"
        else [m.strip().lower() for m in args.models.split(",") if m.strip()]
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for cohort_name in cohorts:
        spec = COHORTS[cohort_name]
        oscar_cfg = yaml.safe_load(open(spec["oscar"]))
        num_classes = int(oscar_cfg["model"]["num_classes"])
        prefix = spec["prefix"]
        stems = _stems(prefix)
        folds_dir = spec["folds"]
        ref_cols = {
            "mRNA": _read_split_matrix(folds_dir / "fold_1", stems["mRNA"]).index,
            "CNV": _read_split_matrix(folds_dir / "fold_1", stems["CNV"]).index,
            "Proteomics": _read_split_matrix(folds_dir / "fold_1", stems["Proteomics"]).index,
        }
        folds = [load_fold(folds_dir / f"fold_{i}", ref_cols, prefix) for i in range(1, 6)]
        dims = {k: folds[0]["dev"][k].shape[1] for k in ("mRNA", "CNV", "Proteomics")}

        out_path = args.out_dir / f"{cohort_name.lower()}_report.json"
        data = json.load(open(out_path)) if out_path.exists() else {"models": {}}
        for m in models:
            if m in data["models"]:
                s = data["models"][m]["summary"]
                print(
                    f"==> [{cohort_name}][{m}_256x32] (cached) "
                    f"F1={s['f1_macro_mean']*100:.2f}±{s['f1_macro_std']*100:.2f} "
                    f"Acc={s['accuracy_mean']*100:.2f}±{s['accuracy_std']*100:.2f}",
                    flush=True,
                )
                continue
            data["models"][m] = run_model_on_cohort(
                cohort_name, m, folds, dims, num_classes, device, args.num_tokens, args.token_dim, args.d_mid
            )
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            s = data["models"][m]["summary"]
            print(
                f"==> [{cohort_name}][{m}_256x32] "
                f"F1={s['f1_macro_mean']*100:.2f}±{s['f1_macro_std']*100:.2f} "
                f"Acc={s['accuracy_mean']*100:.2f}±{s['accuracy_std']*100:.2f}",
                flush=True,
            )
        write_comparison_table(args.out_dir)


if __name__ == "__main__":
    main()

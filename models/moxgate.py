"""
Faithful MoXGATE reimplementation (arXiv:2506.06980), adapted only for
this repo's omics (CNV / mRNA / Proteomics instead of Gene / Methylation / miRNA).

Paper skeleton (eq. 2–14, §2.1.6):
  1) per-modality linear + ReLU, then MHA residual  Z = AV + H  (no FFN / LN)
  2) stack modality tokens → joint MHA with Q=K=V;  F = AV only  (no residual / FFN / LN)
  3) learnable modality weights (sum=1) → weighted sum
  4) classifier: Linear(embed→128) → ReLU → Dropout(0.3) → Linear→classes
  5) L = L_focal(γ=2) + λ1 ||w − 1||² + λ2 ||W_c||_F²

Defaults follow §2.1.6 (embed 256, SA heads 8 / dropout 0.1, joint heads 32,
AdamW lr 1e-4 / wd 1e-2). λ1/λ2 magnitudes are not stated in the paper; we
keep them as explicit hyperparameters (nonzero by default so eq.14 is active).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLightningModel


class LinearModalityEncoder(nn.Module):
    """Paper eq. (2): H_m = φ(X_m W_m + b_m), single linear + ReLU."""

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.proj(x))


class ModalitySelfAttention(nn.Module):
    """Paper eq. (3–5): multi-head SA with residual only (no FFN, no LayerNorm).

    After the linear encoder each sample is one vector, so the sequence length is 1.
    Attention weights are then trivial; the block still applies Q/K/V (+ out) projections
    and the residual Z = AV + H as written.
    """

    def __init__(self, embed_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, E] → [B, 1, E]
        h = x.unsqueeze(1)
        out, _ = self.attn(h, h, h, need_weights=False)
        return (out + h).squeeze(1)


class FocalLoss(nn.Module):
    """Paper eq. (13) with α_i = 1, γ = 2."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        return (((1.0 - pt) ** self.gamma) * ce).mean()


class JointCrossAttention(nn.Module):
    """Paper eq. (7–9): Q=K=V from stacked modality tokens; F = A V (no residual/FFN/LN)."""

    def __init__(self, embed_dim: int, n_heads: int = 32, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, M, E]
        out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return out

    def frobenius_penalty(self) -> torch.Tensor:
        """λ2 term: ||W_c||_F² over joint attention projection weights."""
        penalty = self.attn.in_proj_weight.pow(2).sum()
        if self.attn.in_proj_bias is not None:
            penalty = penalty + self.attn.in_proj_bias.pow(2).sum()
        penalty = penalty + self.attn.out_proj.weight.pow(2).sum()
        if self.attn.out_proj.bias is not None:
            penalty = penalty + self.attn.out_proj.bias.pow(2).sum()
        return penalty


class MoXGATEModel(BaseLightningModel):
    """MoXGATE as specified in arXiv:2506.06980 (omics names adapted to this repo)."""

    def __init__(
        self,
        input_dims: dict,
        num_classes: int,
        latent_dim: int = 256,
        classifier_dim: int = 128,
        attn_dropout: float = 0.1,
        classifier_dropout: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        n_heads: int = 8,
        joint_n_heads: int = 32,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        lambda_modality: float = 1e-2,
        lambda_cross_frobenius: float = 1e-2,
        # --- legacy kwargs (ignored / mapped); keep call sites from breaking ---
        encoder_hidden_dim: int | None = None,
        fusion_hidden: int | None = None,
        dropout: float | None = None,
        modality_balance_weight: float | None = None,
    ):
        # encoder_hidden_dim accepted for call-site compat; paper uses one linear only.
        _ = encoder_hidden_dim
        if fusion_hidden is not None:
            classifier_dim = int(fusion_hidden)
        if dropout is not None:
            attn_dropout = float(dropout)
        if modality_balance_weight is not None:
            lambda_modality = float(modality_balance_weight)

        super().__init__(num_classes=num_classes, lr=lr, weight_decay=weight_decay)

        self.valid_omics = list(input_dims.keys())
        self.use_focal_loss = use_focal_loss
        self.lambda_modality = float(lambda_modality)
        self.lambda_cross_frobenius = float(lambda_cross_frobenius)
        self.latent_dim = int(latent_dim)

        self.encoders = nn.ModuleDict(
            {
                name: LinearModalityEncoder(input_dims[name], self.latent_dim)
                for name in self.valid_omics
            }
        )
        self.modality_self_attn = nn.ModuleDict(
            {
                name: ModalitySelfAttention(
                    embed_dim=self.latent_dim, n_heads=n_heads, dropout=attn_dropout
                )
                for name in self.valid_omics
            }
        )
        self.joint_cross_attn = JointCrossAttention(
            embed_dim=self.latent_dim, n_heads=joint_n_heads, dropout=attn_dropout
        )
        # Uniform init → softmax ≈ 1/M (paper §2.1.3)
        self.modality_logits = nn.Parameter(torch.zeros(len(self.valid_omics)))

        self.classifier = nn.Sequential(
            nn.Linear(self.latent_dim, classifier_dim),
            nn.ReLU(),
            nn.Dropout(classifier_dropout),
            nn.Linear(classifier_dim, num_classes),
        )
        self.focal_loss = FocalLoss(gamma=focal_gamma)

    def _encode(self, batch: dict) -> dict[str, torch.Tensor]:
        latents = {}
        for name in self.valid_omics:
            h = self.encoders[name](batch[name])
            latents[name] = self.modality_self_attn[name](h)
        return latents

    def forward(self, batch):
        latents = self._encode(batch)
        tokens = torch.stack([latents[name] for name in self.valid_omics], dim=1)
        fused_tokens = self.joint_cross_attn(tokens)
        weights = torch.softmax(self.modality_logits, dim=0)
        fused = torch.einsum("m,bme->be", weights, fused_tokens)
        return self.classifier(fused)

    def _compute_loss(self, logits, y):
        if self.use_focal_loss:
            loss = self.focal_loss(logits, y)
        else:
            loss = F.cross_entropy(logits, y)

        # eq. (14): λ1 ||w − 1||²  (on the simplex this also pushes toward uniform)
        if self.lambda_modality > 0:
            w = torch.softmax(self.modality_logits, dim=0)
            ones = torch.ones_like(w)
            loss = loss + self.lambda_modality * (w - ones).pow(2).sum()

        # eq. (14): λ2 ||W_c||_F²
        if self.lambda_cross_frobenius > 0:
            loss = loss + self.lambda_cross_frobenius * self.joint_cross_attn.frobenius_penalty()

        return loss

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        y = batch["label"]
        loss = self._compute_loss(logits, y)
        self.train_acc(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        y = batch["label"]
        loss = self._compute_loss(logits, y)
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

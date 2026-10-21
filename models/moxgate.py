"""
MoXGATE-style multi-omic fusion (adapted to this repo's CNV / mRNA / Proteomics).

  1) per-modality encode + self-attention block
  2) joint cross-attention over stacked modality tokens (all-to-all)
  3) learnable modality weights → weighted sum → classifier
  4) focal loss, plus an optional modality-balance term
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLightningModel
from .self_attn_fusion import OmicEncoder


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            loss = self.alpha * loss
        return loss.mean()


class ModalitySelfAttention(nn.Module):
    """Self-attention block on a single modality embedding (treated as length-1 sequence)."""

    def __init__(self, embed_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, E] → [B, 1, E]
        y = self.layer(x.unsqueeze(1)).squeeze(1)
        return y


class JointCrossAttention(nn.Module):
    """
    MoXGATE joint fusion: Q/K/V all from stacked modality tokens.
    Functionally self-attention over the modality axis.
    """

    def __init__(self, embed_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, M, E]
        residual = tokens
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm(residual + attn_out)
        tokens = tokens + self.ffn(self.ffn_norm(tokens))
        return tokens


class MoXGATEModel(BaseLightningModel):
    """MoXGATE on the omics present in each batch (CNV, mRNA, Proteomics)."""

    def __init__(
        self,
        input_dims: dict,
        num_classes: int,
        latent_dim: int = 128,
        encoder_hidden_dim: int = 512,
        fusion_hidden: int = 256,
        dropout: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        n_heads: int = 8,
        joint_n_heads: int = 8,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        modality_balance_weight: float = 0.0,
    ):
        super().__init__(num_classes=num_classes, lr=lr, weight_decay=weight_decay)

        self.valid_omics = list(input_dims.keys())
        self.use_focal_loss = use_focal_loss
        self.modality_balance_weight = modality_balance_weight
        self.latent_dim = latent_dim

        self.encoders = nn.ModuleDict(
            {
                name: OmicEncoder(
                    input_dim=input_dims[name],
                    hidden_dim=encoder_hidden_dim,
                    latent_dim=latent_dim,
                    dropout=dropout,
                )
                for name in self.valid_omics
            }
        )
        self.modality_self_attn = nn.ModuleDict(
            {
                name: ModalitySelfAttention(
                    embed_dim=latent_dim, n_heads=n_heads, dropout=dropout
                )
                for name in self.valid_omics
            }
        )

        self.joint_cross_attn = JointCrossAttention(
            embed_dim=latent_dim, n_heads=joint_n_heads, dropout=dropout
        )
        # Learnable modality logits → softmax weights (sum=1)
        self.modality_logits = nn.Parameter(torch.zeros(len(self.valid_omics)))

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )
        self.focal_loss = FocalLoss(gamma=focal_gamma)

    def _encode(self, batch: dict) -> dict[str, torch.Tensor]:
        latents = {}
        for name in self.valid_omics:
            z = self.encoders[name](batch[name])
            latents[name] = self.modality_self_attn[name](z)
        return latents

    def forward(self, batch):
        latents = self._encode(batch)
        tokens = torch.stack([latents[name] for name in self.valid_omics], dim=1)  # [B, M, E]
        fused_tokens = self.joint_cross_attn(tokens)  # [B, M, E]

        weights = torch.softmax(self.modality_logits, dim=0)  # [M]
        fused = torch.einsum("m,bme->be", weights, fused_tokens)
        return self.classifier(fused)

    def _compute_loss(self, logits, y):
        if self.use_focal_loss:
            loss = self.focal_loss(logits, y)
        else:
            loss = F.cross_entropy(logits, y)
        if self.modality_balance_weight > 0:
            # Soft push weights toward uniform (paper λ1 term, adapted)
            w = torch.softmax(self.modality_logits, dim=0)
            uniform = torch.full_like(w, 1.0 / w.numel())
            loss = loss + self.modality_balance_weight * F.mse_loss(w, uniform)
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

"""OSCAR-v2: same skeleton as `self_attn_fusion.py`, cleaned fusion path.

vs OSCAR-v1 (`SelfAttentionFusionModel`)
---------------------------------------
Kept
  - `OmicEncoder` (imported from v1)
  - Lightning API + CE loss

Removed
  - Directed CNV→mRNA→RPPA cascade (`CrossAttention`)
  - `GatedFusion`
  - CLS-token Transformer pool (`MultiOmicSelfAttentionFusion`)

Added (fixed in the model — not optional)
  - Per-omic `ModalitySelfAttention`
  - One joint MHA over modality tokens
  - Learnable modality-weight pooling

Forward
  encode → modality SA → joint MHA → weighted sum → classifier
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLightningModel
from .self_attn_fusion import OmicEncoder


class ModalitySelfAttention(nn.Module):
    """Length-1 Transformer block on a single omic vector (residual FFN/norm)."""

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
        return self.layer(x.unsqueeze(1)).squeeze(1)


class JointModalityAttention(nn.Module):
    """Single all-to-all MHA over modality tokens (replaces CLS Transformer stack)."""

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
        return tokens + self.ffn(self.ffn_norm(tokens))


class SelfAttentionFusionModelV2(BaseLightningModel):
    """OSCAR-v2: modSA + joint MHA + modality weights + CE."""

    def __init__(
        self,
        input_dims,
        num_classes: int,
        latent_dim: int = 128,
        encoder_hidden_dim: int = 512,
        fusion_hidden: int = 256,
        dropout: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        n_heads: int = 8,
        joint_n_heads: int = 8,
    ):
        super().__init__(num_classes=num_classes, lr=lr, weight_decay=weight_decay)

        self.valid_omics = list(input_dims.keys())

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

        self.fusion = JointModalityAttention(
            embed_dim=latent_dim, n_heads=joint_n_heads, dropout=dropout
        )
        self.modality_logits = nn.Parameter(torch.zeros(len(self.valid_omics)))

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, batch):
        latents = [
            self.modality_self_attn[name](self.encoders[name](batch[name]))
            for name in self.valid_omics
        ]
        tokens = torch.stack(latents, dim=1)  # [B, M, E]
        fused_tokens = self.fusion(tokens)
        weights = torch.softmax(self.modality_logits, dim=0)
        fused = torch.einsum("m,bme->be", weights, fused_tokens)
        return self.classifier(fused)

    def modality_weights(self) -> torch.Tensor:
        return torch.softmax(self.modality_logits.detach(), dim=0)

    def _compute_loss(self, logits, y):
        return F.cross_entropy(logits, y)

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

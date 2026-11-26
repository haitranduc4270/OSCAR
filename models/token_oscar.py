"""Token-level OSCAR: each omic is projected to (B, T, M) with T > 1, M > 1,
and stays as a sequence of T tokens through ordered cross-attention (CNV -> mRNA
-> Proteomics), token-wise gated fusion, and multi-omic self-attention fusion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseLightningModel


class OmicToTokens(nn.Module):
    """Project (B, D) -> (B, T, M) via a factorized bottleneck."""

    def __init__(
        self,
        input_dim: int,
        num_tokens: int = 256,
        token_dim: int = 32,
        d_mid: int = 256,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_mid),
            nn.LayerNorm(d_mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mid, num_tokens * token_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.size(0), self.num_tokens, self.token_dim)


class TokenCrossAttention(nn.Module):
    """Real multi-head cross-attention between two (B, T, M) token sequences."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.4):
        super().__init__()
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        h = query + attn_out
        return h + self.ffn(self.norm_ffn(h))


class TokenGatedFusion(nn.Module):
    """Token-wise gated fusion between original and cross-attended (B, T, M)."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64, dropout: float = 0.4):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, original: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([original, enhanced], dim=-1))
        return g * enhanced + (1.0 - g) * original


class TokenOSCARModel(BaseLightningModel):
    """OSCAR where each omic is a (B, T, M) token sequence throughout fusion."""

    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        num_tokens: int = 256,
        token_dim: int = 32,
        d_mid: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        fusion_hidden: int = 128,
        dropout: float = 0.4,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        use_cross_attention: bool = True,
        use_gated_fusion: bool = True,
        use_original_for_proteomics: bool = True,
        use_self_attention: bool = True,
        cross_order: tuple[str, str, str] = ("CNV", "mRNA", "Proteomics"),
        cross_edges: list[tuple[str, str]] | None = None,
        use_self_gate: bool = True,
        use_pos_embed: bool = True,
    ):
        super().__init__(num_classes=num_classes, lr=lr, weight_decay=weight_decay)
        if token_dim % n_heads != 0:
            raise ValueError(f"token_dim {token_dim} must be divisible by n_heads {n_heads}")
        if use_self_gate and not use_self_attention:
            raise ValueError("use_self_gate requires use_self_attention=True")
        if use_pos_embed and not use_self_attention:
            raise ValueError("use_pos_embed requires use_self_attention=True")

        self.valid_omics = list(input_dims.keys())
        self.use_cross_attention = use_cross_attention
        self.use_gated_fusion = use_gated_fusion
        self.use_original_for_proteomics = use_original_for_proteomics
        self.use_self_attention = use_self_attention
        self.use_self_gate = use_self_gate
        self.use_pos_embed = use_pos_embed
        self.cross_order = tuple(cross_order)
        self.cross_edges = [tuple(e) for e in cross_edges] if cross_edges is not None else None

        self.encoders = nn.ModuleDict(
            {
                name: OmicToTokens(
                    input_dim=input_dims[name],
                    num_tokens=num_tokens,
                    token_dim=token_dim,
                    d_mid=d_mid,
                    dropout=dropout,
                )
                for name in self.valid_omics
            }
        )

        if self.use_cross_attention:
            if self.cross_edges is not None:
                self.cross_layers = nn.ModuleList(
                    [TokenCrossAttention(token_dim, n_heads, dropout) for _ in self.cross_edges]
                )
                if self.use_gated_fusion:
                    self.cross_gates = nn.ModuleList(
                        [TokenGatedFusion(token_dim, token_dim * 2, dropout) for _ in self.cross_edges]
                    )
            else:
                self.cross_cnv_to_mrna = TokenCrossAttention(token_dim, n_heads, dropout)
                self.cross_mrna_to_prot = TokenCrossAttention(token_dim, n_heads, dropout)
                if self.use_gated_fusion:
                    self.gate_mrna = TokenGatedFusion(token_dim, token_dim * 2, dropout)
                    self.gate_prot = TokenGatedFusion(token_dim, token_dim * 2, dropout)

        if self.use_self_attention:
            self.modality_embed = nn.ParameterDict(
                {
                    name: nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
                    for name in self.valid_omics
                }
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
            if self.use_pos_embed:
                seq_len = 1 + len(self.valid_omics) * num_tokens
                self.pos_embed = nn.Parameter(torch.randn(1, seq_len, token_dim) * 0.02)

            enc_layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=n_heads,
                dim_feedforward=token_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.fusion = nn.TransformerEncoder(
                enc_layer,
                num_layers=n_layers,
                norm=nn.LayerNorm(token_dim),
                enable_nested_tensor=False,
            )
            if self.use_self_gate:
                # Token-wise gated residual on the self-attention output (same form as cross-attn gate).
                self.self_gate = TokenGatedFusion(token_dim, token_dim * 2, dropout)
            clf_in_dim = token_dim
        else:
            clf_in_dim = token_dim * len(self.valid_omics)

        self.classifier = nn.Sequential(
            nn.Linear(clf_in_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = {name: self.encoders[name](batch[name]) for name in self.valid_omics}

        if self.use_cross_attention:
            if self.cross_edges is not None:
                orig_tokens = dict(tokens)
                for idx, (src, dst) in enumerate(self.cross_edges):
                    q_orig = orig_tokens[dst]
                    kv = orig_tokens[src] if self.use_original_for_proteomics else tokens[src]
                    enh = self.cross_layers[idx](query=q_orig, key_value=kv)
                    tokens[dst] = (
                        self.cross_gates[idx](q_orig, enh) if self.use_gated_fusion else enh
                    )
            else:
                o1, o2, o3 = self.cross_order
                o2_orig = tokens[o2]
                o3_orig = tokens[o3]

                o2_enh = self.cross_cnv_to_mrna(query=o2_orig, key_value=tokens[o1])
                tokens[o2] = (
                    self.gate_mrna(o2_orig, o2_enh) if self.use_gated_fusion else o2_enh
                )

                o2_for_o3 = o2_orig if self.use_original_for_proteomics else tokens[o2]
                o3_enh = self.cross_mrna_to_prot(query=o3_orig, key_value=o2_for_o3)
                tokens[o3] = (
                    self.gate_prot(o3_orig, o3_enh) if self.use_gated_fusion else o3_enh
                )

        if self.use_self_attention:
            seq = [tokens[name] + self.modality_embed[name] for name in self.valid_omics]
            B = seq[0].size(0)
            cls = self.cls_token.expand(B, -1, -1)
            all_tokens = torch.cat([cls, *seq], dim=1)  # (B, 1 + 3*T, M)
            if self.use_pos_embed:
                all_tokens = all_tokens + self.pos_embed
            fused_seq = self.fusion(all_tokens)
            if self.use_self_gate:
                fused_seq = self.self_gate(all_tokens, fused_seq)
            fused = fused_seq[:, 0]  # CLS token: (B, M)
        else:
            fused = torch.cat(
                [tokens[name].mean(dim=1) for name in self.valid_omics], dim=-1
            )  # (B, 3*M)
        return self.classifier(fused)

"""Token-level OSCAR: each omic is projected to (B, T, M) with T > 1, M > 1,
and stays as a sequence of T tokens through ordered cross-attention (CNV -> mRNA
-> Proteomics), token-wise gated fusion, and multi-omic self-attention fusion.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLightningModel
from .pathway_tokenizer import PathwayBottleneckToTokens, PathwayTokenEncoder


class OmicToTokens(nn.Module):
    """Project (B, D) -> (B, T, M) via a factorized bottleneck."""

    def __init__(
        self,
        input_dim: int,
        num_tokens: int = 256,
        token_dim: int = 32,
        d_mid: int = 256,
        dropout: float = 0.4,
        token_layernorm: bool = False,
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
        self.post_norm = nn.LayerNorm(token_dim) if token_layernorm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x).view(x.size(0), self.num_tokens, self.token_dim)
        return self.post_norm(out) if self.post_norm is not None else out


class FeatureChunkToTokens(nn.Module):
    """ViT/patch-style tokenizer: split (B, D) into T contiguous chunks and project each to M."""

    def __init__(
        self,
        input_dim: int,
        num_tokens: int = 256,
        token_dim: int = 32,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.chunk_size = max(1, math.ceil(input_dim / num_tokens))
        self.padded_dim = self.chunk_size * num_tokens
        self.proj = nn.Sequential(
            nn.Linear(self.chunk_size, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) < self.padded_dim:
            x = F.pad(x, (0, self.padded_dim - x.size(1)))
        chunks = x.view(x.size(0), self.num_tokens, self.chunk_size)
        return self.proj(chunks)


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Exact sparsemax projection onto the probability simplex (Martins & Astudillo, 2016)."""
    z = logits - logits.max(dim=dim, keepdim=True).values
    zs, _ = torch.sort(z, dim=dim, descending=True)
    k = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    view_shape = [1] * z.ndim
    view_shape[dim] = -1
    k = k.view(*view_shape)
    cumsum = zs.cumsum(dim=dim)
    support = (1.0 + k * zs) > cumsum
    k_z = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (cumsum.gather(dim, k_z - 1) - 1.0) / k_z.to(z.dtype)
    return (z - tau).clamp_min(0.0)


def _sinkhorn(logits: torch.Tensor, n_iters: int = 5) -> torch.Tensor:
    """Log-space Sinkhorn-Knopp normalization for near-doubly-stochastic attention."""
    log_a = logits
    for _ in range(n_iters):
        log_a = log_a - torch.logsumexp(log_a, dim=-1, keepdim=True)
        log_a = log_a - torch.logsumexp(log_a, dim=-2, keepdim=True)
    return log_a.exp()


class TokenCrossAttention(nn.Module):
    """Real multi-head cross-attention between two (B, T, M) token sequences."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.4,
        attn_type: str = "softmax",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn_type = attn_type
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        if attn_type == "softmax":
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        else:
            self.q_proj = nn.Linear(embed_dim, embed_dim)
            self.k_proj = nn.Linear(embed_dim, embed_dim)
            self.v_proj = nn.Linear(embed_dim, embed_dim)
            self.out_proj = nn.Linear(embed_dim, embed_dim)
            self.attn_drop = nn.Dropout(dropout)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def _custom_attn(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, Tq, M = q.shape
        Tk = kv.size(1)
        H, D = self.num_heads, self.head_dim
        qh = self.q_proj(q).view(B, Tq, H, D).transpose(1, 2)   # (B, H, Tq, D)
        kh = self.k_proj(kv).view(B, Tk, H, D).transpose(1, 2)  # (B, H, Tk, D)
        vh = self.v_proj(kv).view(B, Tk, H, D).transpose(1, 2)  # (B, H, Tk, D)

        if self.attn_type == "linear_elu":
            qf = F.elu(qh) + 1.0
            kf = F.elu(kh) + 1.0
            kv_prod = torch.matmul(kf.transpose(-2, -1), vh)  # (B, H, D, D)
            k_sum = kf.sum(dim=-2, keepdim=True)              # (B, H, 1, D)
            num = torch.matmul(qf, kv_prod)                   # (B, H, Tq, D)
            den = (qf * k_sum).sum(dim=-1, keepdim=True).clamp_min(1e-6)
            out = num / den
        else:
            scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(D)  # (B, H, Tq, Tk)
            if self.attn_type == "sigmoid":
                weights = torch.sigmoid(scores - math.log(float(Tk)))
            elif self.attn_type == "sparsemax":
                weights = _sparsemax(scores, dim=-1)
            elif self.attn_type == "sinkhorn":
                weights = _sinkhorn(scores, n_iters=5)
            else:
                raise ValueError(f"Unknown attn_type: {self.attn_type}")
            weights = self.attn_drop(weights)
            out = torch.matmul(weights, vh)

        out = out.transpose(1, 2).contiguous().view(B, Tq, M)
        return self.out_proj(out)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        kv = self.norm_kv(key_value)
        if self.attn_type == "softmax":
            attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        else:
            attn_out = self._custom_attn(q, kv)
        h = query + attn_out
        return h + self.ffn(self.norm_ffn(h))


class TokenGatedFusion(nn.Module):
    """Token-wise gated fusion between original and cross-attended (B, T, M)."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.4,
        fixed_gate_value: float | None = None,
    ):
        super().__init__()
        self.fixed_gate_value = fixed_gate_value
        if fixed_gate_value is None:
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.Sigmoid(),
            )
        else:
            self.gate = None

    def forward(self, original: torch.Tensor, enhanced: torch.Tensor) -> torch.Tensor:
        if self.fixed_gate_value is not None:
            g = self.fixed_gate_value
            return g * enhanced + (1.0 - g) * original
        g = self.gate(torch.cat([original, enhanced], dim=-1))
        return g * enhanced + (1.0 - g) * original


ALL_DIRECTED_EDGES: tuple[tuple[str, str], ...] = (
    ("CNV", "mRNA"),
    ("CNV", "Proteomics"),
    ("mRNA", "CNV"),
    ("mRNA", "Proteomics"),
    ("Proteomics", "CNV"),
    ("Proteomics", "mRNA"),
)


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
        fixed_gate_value: float | None = None,
        num_tokens_per_omic: dict[str, int] | None = None,
        d_mid_per_omic: dict[str, int] | None = None,
        encoder_type: str = "factorized",
        attn_type: str = "softmax",
        token_layernorm: bool = False,
        pathway_specs: dict[str, dict] | None = None,
        pathway_unmapped_dim: int = 0,
        pathway_unmapped_dim_per_omic: dict[str, int] | None = None,
        pathway_mid_dim: int | None = None,
        pathway_proj_per_pathway: int = 8,
        learnable_edges: bool = False,
        fixed_edge_weights: list[float] | None = None,
        edge_tau: float = 1.0,
        edge_cascade_steps: int = 1,
        readout_omic: str | None = None,
    ):
        super().__init__(num_classes=num_classes, lr=lr, weight_decay=weight_decay)
        if token_dim % n_heads != 0:
            raise ValueError(f"token_dim {token_dim} must be divisible by n_heads {n_heads}")
        if use_self_gate and not use_self_attention:
            raise ValueError("use_self_gate requires use_self_attention=True")
        if use_pos_embed and not use_self_attention:
            raise ValueError("use_pos_embed requires use_self_attention=True")
        if learnable_edges and fixed_edge_weights is not None:
            raise ValueError("Cannot set both learnable_edges=True and fixed_edge_weights")

        self.valid_omics = list(input_dims.keys())
        self.use_cross_attention = use_cross_attention
        self.use_gated_fusion = use_gated_fusion
        self.use_original_for_proteomics = use_original_for_proteomics
        self.use_self_attention = use_self_attention
        self.use_self_gate = use_self_gate
        self.use_pos_embed = use_pos_embed
        self.cross_order = tuple(cross_order)
        self.learnable_edges = learnable_edges
        self.use_weighted_edges = learnable_edges or (fixed_edge_weights is not None)
        self.edge_tau = float(edge_tau)
        self.edge_cascade_steps = int(edge_cascade_steps)
        self.readout_omic = readout_omic
        if self.use_weighted_edges and cross_edges is None:
            self.cross_edges = list(ALL_DIRECTED_EDGES)
        else:
            self.cross_edges = [tuple(e) for e in cross_edges] if cross_edges is not None else None

        self.num_tokens_map = {}
        for name in self.valid_omics:
            if encoder_type == "pathway_tokens" and pathway_specs and name in pathway_specs:
                self.num_tokens_map[name] = int(pathway_specs[name]["num_pathways"])
            elif num_tokens_per_omic and name in num_tokens_per_omic:
                self.num_tokens_map[name] = int(num_tokens_per_omic[name])
            else:
                self.num_tokens_map[name] = num_tokens

        self.d_mid_map = {
            name: (int(d_mid_per_omic[name]) if d_mid_per_omic and name in d_mid_per_omic else d_mid)
            for name in self.valid_omics
        }
        unmapped_dim_map = {
            name: (
                int(pathway_unmapped_dim_per_omic[name])
                if pathway_unmapped_dim_per_omic and name in pathway_unmapped_dim_per_omic
                else pathway_unmapped_dim
            )
            for name in self.valid_omics
        }

        if encoder_type == "factorized":
            self.encoders = nn.ModuleDict(
                {
                    name: OmicToTokens(
                        input_dim=input_dims[name],
                        num_tokens=self.num_tokens_map[name],
                        token_dim=token_dim,
                        d_mid=self.d_mid_map[name],
                        dropout=dropout,
                        token_layernorm=token_layernorm,
                    )
                    for name in self.valid_omics
                }
            )
        elif encoder_type == "chunk":
            self.encoders = nn.ModuleDict(
                {
                    name: FeatureChunkToTokens(
                        input_dim=input_dims[name],
                        num_tokens=self.num_tokens_map[name],
                        token_dim=token_dim,
                        dropout=dropout,
                    )
                    for name in self.valid_omics
                }
            )
        elif encoder_type == "pathway_bottleneck":
            if not pathway_specs:
                raise ValueError("encoder_type='pathway_bottleneck' requires pathway_specs")
            self.encoders = nn.ModuleDict(
                {
                    name: (
                        PathwayBottleneckToTokens(
                            spec=pathway_specs[name],
                            num_tokens=self.num_tokens_map[name],
                            token_dim=token_dim,
                            dropout=dropout,
                            unmapped_dim=unmapped_dim_map[name],
                            token_layernorm=token_layernorm,
                            bottleneck_mid_dim=pathway_mid_dim,
                        )
                        if name in pathway_specs
                        else OmicToTokens(
                            input_dim=input_dims[name],
                            num_tokens=self.num_tokens_map[name],
                            token_dim=token_dim,
                            d_mid=self.d_mid_map[name],
                            dropout=dropout,
                            token_layernorm=token_layernorm,
                        )
                    )
                    for name in self.valid_omics
                }
            )
        elif encoder_type == "pathway_tokens":
            if not pathway_specs:
                raise ValueError("encoder_type='pathway_tokens' requires pathway_specs")
            self.encoders = nn.ModuleDict(
                {
                    name: (
                        PathwayTokenEncoder(
                            spec=pathway_specs[name],
                            token_dim=token_dim,
                            proj_per_pathway=pathway_proj_per_pathway,
                            dropout=dropout,
                        )
                        if name in pathway_specs
                        else OmicToTokens(
                            input_dim=input_dims[name],
                            num_tokens=self.num_tokens_map[name],
                            token_dim=token_dim,
                            d_mid=self.d_mid_map[name],
                            dropout=dropout,
                            token_layernorm=token_layernorm,
                        )
                    )
                    for name in self.valid_omics
                }
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        if self.use_cross_attention:
            if self.cross_edges is not None:
                self.cross_layers = nn.ModuleList(
                    [TokenCrossAttention(token_dim, n_heads, dropout, attn_type=attn_type) for _ in self.cross_edges]
                )
                if self.use_gated_fusion:
                    self.cross_gates = nn.ModuleList(
                        [
                            TokenGatedFusion(token_dim, token_dim * 2, dropout, fixed_gate_value=fixed_gate_value)
                            for _ in self.cross_edges
                        ]
                    )
                if self.use_weighted_edges:
                    self.edge_delta_norms = nn.ModuleList(
                        [nn.LayerNorm(token_dim, elementwise_affine=False) for _ in self.cross_edges]
                    )
                    if self.learnable_edges:
                        self.edge_logits = nn.Parameter(torch.zeros(len(self.cross_edges)))
                    else:
                        if len(fixed_edge_weights) != len(self.cross_edges):
                            raise ValueError("fixed_edge_weights length must match cross_edges")
                        self.register_buffer(
                            "fixed_edge_gates",
                            torch.tensor(fixed_edge_weights, dtype=torch.float32),
                        )
            else:
                self.cross_cnv_to_mrna = TokenCrossAttention(token_dim, n_heads, dropout, attn_type=attn_type)
                self.cross_mrna_to_prot = TokenCrossAttention(token_dim, n_heads, dropout, attn_type=attn_type)
                if self.use_gated_fusion:
                    self.gate_mrna = TokenGatedFusion(token_dim, token_dim * 2, dropout, fixed_gate_value=fixed_gate_value)
                    self.gate_prot = TokenGatedFusion(token_dim, token_dim * 2, dropout, fixed_gate_value=fixed_gate_value)

        if self.use_self_attention:
            self.modality_embed = nn.ParameterDict(
                {
                    name: nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
                    for name in self.valid_omics
                }
            )
            self.cls_token = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
            if self.use_pos_embed:
                seq_len = 1 + sum(self.num_tokens_map[name] for name in self.valid_omics)
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
                self.self_gate = TokenGatedFusion(token_dim, token_dim * 2, dropout, fixed_gate_value=fixed_gate_value)
            clf_in_dim = token_dim
        else:
            if self.readout_omic is not None:
                clf_in_dim = token_dim
            else:
                clf_in_dim = token_dim * len(self.valid_omics)

        self.classifier = nn.Sequential(
            nn.Linear(clf_in_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def set_edge_temperature(self, tau: float) -> None:
        self.edge_tau = float(tau)

    def get_edge_gates(self) -> torch.Tensor:
        if not self.use_weighted_edges:
            raise RuntimeError("get_edge_gates requires learnable_edges=True or fixed_edge_weights")
        if self.learnable_edges:
            return torch.sigmoid(self.edge_logits / self.edge_tau)
        return self.fixed_edge_gates

    def edge_acyclicity_penalty(self) -> torch.Tensor:
        """Exact 3-node acyclicity penalty: sum of all 2-cycle and 3-cycle edge products."""
        g = self.get_edge_gates()
        edge_map = {edge: g[idx] for idx, edge in enumerate(self.cross_edges)}
        o0, o1, o2 = ("CNV", "mRNA", "Proteomics")
        zero = g.new_tensor(0.0)
        g01 = edge_map.get((o0, o1), zero)
        g10 = edge_map.get((o1, o0), zero)
        g02 = edge_map.get((o0, o2), zero)
        g20 = edge_map.get((o2, o0), zero)
        g12 = edge_map.get((o1, o2), zero)
        g21 = edge_map.get((o2, o1), zero)
        two_cycles = g01 * g10 + g02 * g20 + g12 * g21
        three_cycles = g01 * g12 * g20 + g02 * g21 * g10
        return two_cycles + three_cycles

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = {name: self.encoders[name](batch[name]) for name in self.valid_omics}

        if self.use_cross_attention:
            if self.cross_edges is not None and self.use_weighted_edges:
                g_edges = self.get_edge_gates()
                for _ in range(self.edge_cascade_steps):
                    orig_tokens = dict(tokens)
                    acc = {name: torch.zeros_like(orig_tokens[name]) for name in self.valid_omics}
                    for idx, (src, dst) in enumerate(self.cross_edges):
                        q_orig = orig_tokens[dst]
                        kv = orig_tokens[src]
                        enh = self.cross_layers[idx](query=q_orig, key_value=kv)
                        delta = self.edge_delta_norms[idx](enh - q_orig)
                        if self.use_gated_fusion:
                            gate_mod = self.cross_gates[idx]
                            if gate_mod.fixed_gate_value is not None:
                                tok_gate = gate_mod.fixed_gate_value
                            else:
                                tok_gate = gate_mod.gate(torch.cat([q_orig, enh], dim=-1))
                            delta = tok_gate * delta
                        acc[dst] = acc[dst] + g_edges[idx] * delta
                    tokens = {name: orig_tokens[name] + acc[name] for name in self.valid_omics}
            elif self.cross_edges is not None:
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
            if self.readout_omic is not None:
                fused = tokens[self.readout_omic].mean(dim=1)
            else:
                fused = torch.cat(
                    [tokens[name].mean(dim=1) for name in self.valid_omics], dim=-1
                )  # (B, 3*M)
        return self.classifier(fused)

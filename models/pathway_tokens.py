"""Pathway tokens with a shared gene embedding and ordered CNV → mRNA → protein fusion."""

from __future__ import annotations

import torch
import torch.nn as nn


def aggregate_pathway_means(
    measurement: torch.Tensor,
    membership: torch.Tensor,
) -> torch.Tensor:
    """Mean measurement of the genes in each pathway, shape ``[batch, pathways]``."""
    present = ~torch.isnan(measurement)
    values = torch.nan_to_num(measurement, nan=0.0)
    numerator = torch.einsum("pg,bg->bp", membership, values * present)
    denominator = torch.einsum("pg,bg->bp", membership, present.to(measurement.dtype))
    return numerator / denominator.clamp(min=1.0)


def aggregate_pathway_tokens(
    measurement: torch.Tensor,
    embedding: torch.Tensor,
    membership: torch.Tensor,
) -> torch.Tensor:
    """Mean of measurement-scaled gene vectors inside each pathway.

    Args:
        measurement: ``[batch, genes]``. NaN entries are left out of the mean.
        embedding: ``[genes, gene_dim]``.
        membership: ``[pathways, genes]``, 1 when a gene belongs to a pathway.

    Returns:
        ``[batch, pathways, gene_dim]``.
    """
    present = ~torch.isnan(measurement)
    values = torch.nan_to_num(measurement, nan=0.0)
    numerator = torch.einsum("pg,bg,gd->bpd", membership, values * present, embedding)
    denominator = torch.einsum("pg,bg->bp", membership, present.to(measurement.dtype))
    return numerator / denominator.clamp(min=1.0).unsqueeze(-1)


def _group_summary(
    values: torch.Tensor,
    index: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and presence flag for one antibody class inside each program.

    Args:
        values: ``[batch, antibodies]``.
        index: ``[programs, slots]`` into the antibody axis. Padding is ignored.
        mask: ``[programs, slots]``, True on real antibodies.
    """
    present = ~torch.isnan(values)
    filled = torch.nan_to_num(values, nan=0.0)
    picked = filled[:, index]
    slot_present = present[:, index] & mask.unsqueeze(0)
    count = slot_present.sum(dim=-1)
    mean = (picked * slot_present).sum(dim=-1) / count.clamp(min=1)
    flag = (count > 0).to(values.dtype)
    return mean, flag


class PathwayTokenModel(nn.Module):
    """51 CNV tokens, 51 mRNA tokens, 9 protein tokens, then a CLS transformer.

    mRNA is fused with the CNV token of the same pathway through a shared
    residual gate. Each protein program then reads the fused mRNA token and
    the original CNV token of the pathways listed for that program. The
    transformer does not see CNV tokens and has no positional encoding.
    """

    def __init__(
        self,
        n_genes: int,
        membership: torch.Tensor,
        total_index: torch.Tensor,
        total_mask: torch.Tensor,
        phospho_index: torch.Tensor,
        phospho_mask: torch.Tensor,
        key_index: torch.Tensor,
        key_mask: torch.Tensor,
        num_classes: int = 4,
        gene_dim: int = 32,
        token_dim: int = 64,
        use_gene_embedding: bool = True,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        if membership.ndim != 2:
            raise ValueError("membership must be [pathways, genes]")
        n_pathways = membership.shape[0]
        if membership.shape[1] != n_genes:
            raise ValueError("membership gene axis does not match n_genes")
        if token_dim % n_heads != 0:
            raise ValueError(f"token_dim {token_dim} is not divisible by n_heads {n_heads}")

        self.n_pathways = n_pathways
        self.n_programs = total_index.shape[0]
        self.use_gene_embedding = use_gene_embedding
        self.register_buffer("membership", membership.float())
        self.register_buffer("total_index", total_index.long())
        self.register_buffer("total_mask", total_mask.bool())
        self.register_buffer("phospho_index", phospho_index.long())
        self.register_buffer("phospho_mask", phospho_mask.bool())
        self.register_buffer("key_index", key_index.long())
        self.register_buffer("key_mask", key_mask.bool())

        self.pathway_embed = nn.Parameter(torch.empty(1, n_pathways, token_dim))
        self.cls = nn.Parameter(torch.empty(1, 1, token_dim))
        nn.init.normal_(self.pathway_embed, std=0.02)
        nn.init.normal_(self.cls, std=0.02)
        if use_gene_embedding:
            self.gene_embed = nn.Parameter(torch.empty(n_genes, gene_dim))
            nn.init.normal_(self.gene_embed, std=0.02)
            self.token_proj = nn.Linear(gene_dim, token_dim)
        else:
            self.gene_embed = None
            self.token_proj = nn.Linear(1, token_dim)
        self.token_norm = nn.LayerNorm(token_dim)

        self.fuse_gate = nn.Linear(token_dim * 2, token_dim)
        nn.init.constant_(self.fuse_gate.bias, -2.0)

        self.protein_proj = nn.ModuleList(
            nn.Linear(4, token_dim) for _ in range(self.n_programs)
        )
        self.protein_norm = nn.LayerNorm(token_dim)
        self.protein_query_norm = nn.LayerNorm(token_dim)
        self.protein_key_norm = nn.LayerNorm(token_dim)
        self.protein_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        nn.init.zeros_(self.protein_attn.out_proj.weight)
        nn.init.zeros_(self.protein_attn.out_proj.bias)

        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=token_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(token_dim),
            enable_nested_tensor=False,
        )
        self.token_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(token_dim, num_classes)

    def _project_pathways(self, measurement: torch.Tensor) -> torch.Tensor:
        if self.use_gene_embedding:
            raw = aggregate_pathway_tokens(measurement, self.gene_embed, self.membership)
        else:
            raw = aggregate_pathway_means(measurement, self.membership).unsqueeze(-1)
        return self.token_norm(self.token_proj(raw)) + self.pathway_embed

    def _protein_features(self, protein: torch.Tensor) -> torch.Tensor:
        total_mean, total_flag = _group_summary(protein, self.total_index, self.total_mask)
        phospho_mean, phospho_flag = _group_summary(
            protein, self.phospho_index, self.phospho_mask
        )
        features = torch.stack([total_mean, phospho_mean, total_flag, phospho_flag], dim=-1)
        projected = [layer(features[:, i]) for i, layer in enumerate(self.protein_proj)]
        return self.protein_norm(torch.stack(projected, dim=1))

    def _attend_programs(self, protein: torch.Tensor, cnv: torch.Tensor, mrna: torch.Tensor) -> torch.Tensor:
        bank = torch.cat([cnv, mrna], dim=1)
        updated = []
        for program in range(self.n_programs):
            query = protein[:, program : program + 1]
            valid = self.key_mask[program]
            if not bool(valid.any()):
                updated.append(query)
                continue
            keys = bank[:, self.key_index[program, valid]]
            attended, _ = self.protein_attn(
                self.protein_query_norm(query),
                self.protein_key_norm(keys),
                self.protein_key_norm(keys),
                need_weights=False,
            )
            updated.append(query + attended)
        return torch.cat(updated, dim=1)

    def summary(self, mrna: torch.Tensor, cnv: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        """CLS vector after ordered CNV, mRNA, and protein attention. Shape ``[batch, token_dim]``."""
        cnv_tokens = self._project_pathways(cnv)
        mrna_tokens = self._project_pathways(mrna)
        gate = torch.sigmoid(self.fuse_gate(torch.cat([mrna_tokens, cnv_tokens], dim=-1)))
        fused_mrna = mrna_tokens + gate * cnv_tokens

        protein_tokens = self._protein_features(protein)
        protein_tokens = self._attend_programs(protein_tokens, cnv_tokens, fused_mrna)

        batch = fused_mrna.shape[0]
        cls = self.cls.expand(batch, -1, -1)
        sequence = torch.cat([cls, fused_mrna, protein_tokens], dim=1)
        encoded = self.encoder(self.token_dropout(sequence))
        return encoded[:, 0]

    def forward(self, mrna: torch.Tensor, cnv: torch.Tensor, protein: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.summary(mrna, cnv, protein))

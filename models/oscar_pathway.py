"""OSCAR whose omic inputs are pathway-aggregated gene summaries."""

from __future__ import annotations

import torch
import torch.nn as nn

from .pathway_tokens import aggregate_pathway_means, _group_summary
from .self_attn_fusion import SelfAttentionFusionModel


class OscarPathwayModel(SelfAttentionFusionModel):
    """CNV and mRNA are Hallmark pathway means; proteomics is nine program vectors."""

    def __init__(
        self,
        membership: torch.Tensor,
        mrna_index: torch.Tensor,
        cnv_index: torch.Tensor,
        total_index: torch.Tensor,
        total_mask: torch.Tensor,
        phospho_index: torch.Tensor,
        phospho_mask: torch.Tensor,
        **kwargs,
    ):
        n_pathways = int(membership.shape[0])
        n_programs = int(total_index.shape[0])
        input_dims = {
            "mRNA": n_pathways,
            "CNV": n_pathways,
            "Proteomics": n_programs * 4,
        }
        super().__init__(input_dims=input_dims, **kwargs)
        self.register_buffer("membership", membership.float())
        self.register_buffer("mrna_index", mrna_index.long())
        self.register_buffer("cnv_index", cnv_index.long())
        self.register_buffer("total_index", total_index.long())
        self.register_buffer("total_mask", total_mask.bool())
        self.register_buffer("phospho_index", phospho_index.long())
        self.register_buffer("phospho_mask", phospho_mask.bool())

    def _protein_programs(self, protein: torch.Tensor) -> torch.Tensor:
        total_mean, total_flag = _group_summary(protein, self.total_index, self.total_mask)
        phospho_mean, phospho_flag = _group_summary(
            protein, self.phospho_index, self.phospho_mask
        )
        features = torch.stack(
            [total_mean, phospho_mean, total_flag, phospho_flag], dim=-1
        )
        return features.reshape(protein.shape[0], -1)

    def forward(self, batch):
        mrna = aggregate_pathway_means(
            batch["mRNA"].index_select(1, self.mrna_index), self.membership
        )
        cnv = aggregate_pathway_means(
            batch["CNV"].index_select(1, self.cnv_index), self.membership
        )
        protein = self._protein_programs(batch["Proteomics"])
        return super().forward({"mRNA": mrna, "CNV": cnv, "Proteomics": protein})

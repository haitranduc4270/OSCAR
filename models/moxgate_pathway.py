"""MoXGATE with one extra token from the mean pathway branch."""

from __future__ import annotations

import torch
import torch.nn as nn

from .moxgate import MoXGATEModel
from .pathway_tokens import PathwayTokenModel


class MoXGATEWithPathway(MoXGATEModel):
    """Three full-omic tokens plus one pathway summary in the joint attention."""

    def __init__(
        self,
        input_dims: dict,
        pathway: PathwayTokenModel,
        mrna_index: torch.Tensor,
        cnv_index: torch.Tensor,
        **kwargs,
    ):
        super().__init__(input_dims=input_dims, **kwargs)
        token_dim = int(pathway.cls.shape[-1])
        self.pathway = pathway
        self.pathway.classifier = nn.Identity()
        self.pathway_proj = nn.Linear(token_dim, self.latent_dim)
        self.register_buffer("mrna_index", mrna_index.long())
        self.register_buffer("cnv_index", cnv_index.long())
        self.modality_logits = nn.Parameter(torch.zeros(len(self.valid_omics) + 1))

    def forward(self, batch):
        latents = self._encode(batch)
        omic_tokens = torch.stack([latents[name] for name in self.valid_omics], dim=1)
        pathway_vector = self.pathway.summary(
            batch["mRNA"].index_select(1, self.mrna_index),
            batch["CNV"].index_select(1, self.cnv_index),
            batch["Proteomics"],
        )
        pathway_token = self.pathway_proj(pathway_vector).unsqueeze(1)
        tokens = torch.cat([omic_tokens, pathway_token], dim=1)
        fused_tokens = self.joint_cross_attn(tokens)
        weights = torch.softmax(self.modality_logits, dim=0)
        fused = torch.einsum("m,bme->be", weights, fused_tokens)
        return self.classifier(fused)

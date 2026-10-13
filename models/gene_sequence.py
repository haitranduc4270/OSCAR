"""Gene-token sequence models for the order ablation.

Each gene is a position on a chromosome-sorted sequence. A token carries
copy number, mRNA, and protein for that gene. Two inductive biases are
separated:

- across genes, sequence order follows genomic coordinates
- within a gene, three tokens follow CNV, then mRNA, then protein

Positional embeddings are tied to the sequence slot. Gene identity stays
on the token, so permuting the gene order assigns the wrong positions
without erasing which gene is which.

Expression magnitude has to survive the token sum. A LayerNorm applied to
the value vector alone would keep only the sign of a one-omic token, so
the value projection, the gene embedding, and the position embedding are
added at the same scale and normalized together inside the Transformer.
"""

import torch
import torch.nn as nn

from .base import BaseLightningModel


class GeneSequenceModel(BaseLightningModel):
    """Transformer over gene tokens.

    within=False projects the three omic scalars with a single linear map.
    within=True runs a shared Transformer over the ordered triple
    (CNV, mRNA, protein) before the across-gene Transformer.
    """

    def __init__(
        self,
        n_genes: int,
        n_classes: int,
        perm,
        within: bool,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.4,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__(num_classes=n_classes, lr=lr, weight_decay=weight_decay)
        self.save_hyperparameters(ignore=["perm"])
        self.n_genes = n_genes
        self.within = within
        self.d_model = d_model

        self.gene_emb = nn.Embedding(n_genes, d_model)
        self.pos_emb = nn.Embedding(n_genes, d_model)
        self.channel = nn.Parameter(torch.empty(3, d_model))
        self.missing = nn.Parameter(torch.empty(d_model))
        self.emb_drop = nn.Dropout(dropout)
        nn.init.normal_(self.gene_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.normal_(self.channel, std=0.02)
        nn.init.normal_(self.missing, std=0.02)

        if within:
            self.omic_emb = nn.Parameter(torch.empty(3, d_model))
            self.within_pos = nn.Parameter(torch.empty(3, d_model))
            nn.init.normal_(self.omic_emb, std=0.02)
            nn.init.normal_(self.within_pos, std=0.02)
            # Starts closed so the flat projection is what the classifier sees.
            self.within_gate = nn.Parameter(torch.zeros(1))
            within_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="relu",
                norm_first=True,
            )
            self.within_encoder = nn.TransformerEncoder(within_layer, num_layers=1)
        else:
            self.omic_emb = None
            self.within_pos = None
            self.within_encoder = None
            self.within_gate = None

        across_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="relu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(across_layer, num_layers=n_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
        self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))

    def _encode_flat(self, values, mask):
        # values: [B, G, 3], mask: [B, G] protein presence.
        # Protein is zero when the antibody panel does not cover the gene.
        # Gene identity already marks that gap, so no extra bias is added.
        observed = torch.ones_like(values)
        observed[:, :, 2] = mask
        return torch.einsum("bgc,cd->bgd", values * observed, self.channel)

    def _encode_within(self, values, mask):
        # Residual from the flat projection so the ordered triple cannot
        # erase CNV and mRNA before the across-gene layer sees them.
        flat = self._encode_flat(values, mask)
        batch, n_genes, _ = values.shape
        tokens = values.unsqueeze(-1) * self.channel + self.omic_emb + self.within_pos
        missing = self.missing.view(1, 1, -1).expand(batch, n_genes, -1)
        tokens[:, :, 2, :] = torch.where((mask > 0.5).unsqueeze(-1), tokens[:, :, 2, :], missing)
        hidden = self.within_encoder(tokens.reshape(batch * n_genes, 3, self.d_model))
        pooled = hidden.reshape(batch, n_genes, 3, self.d_model).mean(dim=2)
        return flat + self.within_gate * pooled

    def forward(self, batch):
        values = batch["values"]
        mask = batch["mask"]
        if self.within:
            content = self._encode_within(values, mask)
        else:
            content = self._encode_flat(values, mask)
        content = content + self.gene_emb.weight.unsqueeze(0)
        content = content[:, self.perm, :]
        positions = torch.arange(self.n_genes, device=content.device)
        content = self.emb_drop(content + self.pos_emb(positions))
        cls = self.cls.expand(values.size(0), -1, -1)
        encoded = self.encoder(torch.cat([cls, content], dim=1))
        return self.classifier(encoded[:, 0])

"""Bidirectional Mamba over gene tokens.

The scan is a real sequence: each gene updates a state that the next gene
reads. Chromosome order is not a causal arrow from chromosome 1 to Y, so
the block runs forward and backward and adds the two passes. A fixed
shuffle moves genes to different steps of that scan. Gene identity stays
on the token. There is no position embedding.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLightningModel


class MambaMixer(nn.Module):
    """Selective state-space mixer (Mamba), one direction, pure PyTorch."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        dt_rank = math.ceil(d_model / 16)
        self.d_inner = d_inner
        self.d_state = d_state
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_inner,
            bias=True,
        )
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self._init_dt(dt_rank, d_inner)
        nn.init.zeros_(self.out_proj.weight)

    def _init_dt(self, dt_rank: int, d_inner: int):
        dt_min, dt_max = 1e-3, 1e-1
        std = dt_rank**-0.5
        nn.init.uniform_(self.dt_proj.weight, -std, std)
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

    def _scan(self, u, delta, A, B, C):
        # u, delta: [B, L, D]; A: [D, N]; B, C: [B, L, N]
        # h_t = exp(dt_t A) h_{t-1} + dt_t B_t u_t, scanned in parallel.
        decay = torch.exp(delta.unsqueeze(-1) * A)
        drive = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)
        length = decay.shape[1]
        offset = 1
        while offset < length:
            left_decay = decay[:, :-offset]
            left_drive = drive[:, :-offset]
            right_decay = decay[:, offset:].clone()
            right_drive = drive[:, offset:].clone()
            decay = decay.clone()
            drive = drive.clone()
            decay[:, offset:] = right_decay * left_decay
            drive[:, offset:] = right_decay * left_drive + right_drive
            offset *= 2
        return (drive * C.unsqueeze(2)).sum(-1)

    def forward(self, x):
        # x: [B, L, d_model]
        length = x.shape[1]
        x_branch, z = self.in_proj(x).chunk(2, dim=-1)
        conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :length].transpose(1, 2)
        conv = F.silu(conv)
        delta_b_c = self.x_proj(conv)
        delta, b_param, c_param = torch.split(
            delta_b_c, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))
        A = -torch.exp(self.A_log.float())
        y = self._scan(conv, delta, A, b_param, c_param)
        y = y + conv * self.D
        return self.out_proj(y * F.silu(z))


class BiMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.forward_mixer = MambaMixer(d_model, d_state=d_state)
        self.backward_mixer = MambaMixer(d_model, d_state=d_state)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        hidden = self.norm(x)
        backward = self.backward_mixer(hidden.flip(1)).flip(1)
        return x + self.drop(self.forward_mixer(hidden) + backward)


class GeneMambaModel(BaseLightningModel):
    """One token per gene. The Mamba scan follows `perm`."""

    def __init__(
        self,
        n_genes: int,
        n_classes: int,
        perm,
        d_model: int = 96,
        n_layers: int = 2,
        d_state: int = 16,
        dropout: float = 0.4,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__(num_classes=n_classes, lr=lr, weight_decay=weight_decay)
        self.save_hyperparameters(ignore=["perm"])
        self.n_genes = n_genes
        self.gene_emb = nn.Embedding(n_genes, d_model)
        self.channel = nn.Parameter(torch.empty(3, d_model))
        nn.init.normal_(self.gene_emb.weight, std=0.02)
        nn.init.normal_(self.channel, std=0.02)
        self.layers = nn.ModuleList(
            BiMambaBlock(d_model, d_state=d_state, dropout=dropout) for _ in range(n_layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )
        self.register_buffer("perm", torch.as_tensor(perm, dtype=torch.long))

    def forward(self, batch):
        values = batch["values"]
        mask = batch["mask"]
        observed = torch.ones_like(values)
        observed[:, :, 2] = mask
        content = torch.einsum("bgc,cd->bgd", values * observed, self.channel)
        content = content + self.gene_emb.weight.unsqueeze(0)
        content = content[:, self.perm, :]
        for layer in self.layers:
            content = layer(content)
        pooled = self.norm(content).mean(dim=1)
        return self.classifier(pooled)

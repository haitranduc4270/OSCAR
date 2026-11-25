"""Clean PyTorch implementations of the 5 multi-omics baselines for matched 5-fold:
  - MOT        (Multi-omics Transformer)
  - DeepMoIC   (AE + SNF + Deep Residual GCN)
  - MoGCN      (AE + SNF + Shallow GCN)
  - MOGONET    (Per-omic GCN + VCDN)
  - moBRCA-net (Per-omic feature-block self-attention + classifier)

MoXGATE (paper-faithful) lives in models/moxgate.py.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Graph / SNF helpers (used by DeepMoIC, MoGCN, MOGONET)
# ---------------------------------------------------------------------------


def _normalize_adj(adj: np.ndarray) -> torch.Tensor:
    adj = adj.astype(np.float32) + np.eye(adj.shape[0], dtype=np.float32)
    deg = adj.sum(axis=1)
    d_inv_sqrt = np.power(np.clip(deg, 1e-12, None), -0.5)
    norm = (adj * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    return torch.from_numpy(norm.astype(np.float32))


def cosine_knn_adj(x: np.ndarray, k: int = 10) -> torch.Tensor:
    """Symmetric cosine k-NN normalized adjacency for MOGONET."""
    x_norm = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    sim = np.clip(x_norm @ x_norm.T, 0.0, 1.0)
    np.fill_diagonal(sim, 0.0)
    n = sim.shape[0]
    k = min(k, max(1, n - 1))
    adj = np.zeros_like(sim)
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n)[:, None]
    adj[rows, idx] = sim[rows, idx]
    adj = np.maximum(adj, adj.T)
    return _normalize_adj(adj)


def _affinity_matrix(x: np.ndarray, k: int = 20, mu: float = 0.5) -> np.ndarray:
    sq = np.sum(x * x, axis=1, keepdims=True)
    dist = np.sqrt(np.clip(sq + sq.T - 2.0 * (x @ x.T), 0.0, None))
    n = dist.shape[0]
    k = min(k, max(1, n - 1))
    sorted_d = np.sort(dist, axis=1)[:, 1 : k + 1]
    eps = mu * (sorted_d.mean(axis=1, keepdims=True) + sorted_d.mean(axis=1)[None, :] + dist) / 3.0
    eps = np.clip(eps, 1e-6, None)
    w = np.exp(-(dist * dist) / (2.0 * eps * eps))
    w = 0.5 * (w + w.T)
    return w


def build_snf_adj(views: list[np.ndarray], k: int = 20, t: int = 10) -> torch.Tensor:
    """Similarity Network Fusion (Wang et al.) across omic views -> normalized adj."""
    affinities = [_affinity_matrix(v, k=k) for v in views]
    n = affinities[0].shape[0]
    k = min(k, max(1, n - 1))

    P_list, S_list = [], []
    for W in affinities:
        row_sum = np.clip(W.sum(axis=1, keepdims=True) - np.diag(W)[:, None], 1e-12, None)
        P = W / (2.0 * row_sum)
        np.fill_diagonal(P, 0.5)
        P_list.append(P)

        S = np.zeros_like(W)
        idx = np.argpartition(-W, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(n)[:, None]
        S[rows, idx] = W[rows, idx]
        S = S / np.clip(S.sum(axis=1, keepdims=True), 1e-12, None)
        S_list.append(S)

    m = len(P_list)
    for _ in range(t):
        next_P = []
        for i in range(m):
            other = sum(P_list[j] for j in range(m) if j != i) / max(1, m - 1)
            P_new = S_list[i] @ other @ S_list[i].T
            P_new = 0.5 * (P_new + P_new.T)
            next_P.append(P_new)
        P_list = next_P

    fused = sum(P_list) / m
    np.fill_diagonal(fused, 0.0)
    return _normalize_adj(fused)


# ---------------------------------------------------------------------------
# 1. MOT — Multi-omics Transformer
# ---------------------------------------------------------------------------


class MOTModel(nn.Module):
    """Each omic is projected to one token; symmetric Transformer over omic tokens."""

    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.3,
        omic_order: tuple[str, ...] = ("CNV", "mRNA", "Proteomics"),
    ):
        super().__init__()
        self.omic_order = tuple(k for k in omic_order if k in input_dims)
        self.proj = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(input_dims[name], d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for name in self.omic_order
            }
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model * len(self.omic_order), d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = torch.stack([self.proj[k](batch[k]) for k in self.omic_order], dim=1)
        h = self.encoder(tokens)
        return self.classifier(h.reshape(h.size(0), -1))


# ---------------------------------------------------------------------------
# 2. MoGCN (AE + SNF + shallow GCN) & 3. DeepMoIC (AE + SNF + deep residual GCN)
# ---------------------------------------------------------------------------


class MultiOmicAE(nn.Module):
    """Joint multi-omic autoencoder producing a shared latent representation."""

    def __init__(self, input_dims: dict[str, int], latent_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.omics = list(input_dims.keys())
        total_in = sum(input_dims.values())
        self.encoder = nn.Sequential(
            nn.Linear(total_in, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, latent_dim),
            nn.ReLU(),
        )
        self.decoders = nn.ModuleDict(
            {name: nn.Linear(latent_dim, input_dims[name]) for name in self.omics}
        )

    def encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([batch[k] for k in self.omics], dim=-1)
        return self.encoder(x)

    def forward(self, batch: dict[str, torch.Tensor]):
        z = self.encode(batch)
        recons = {k: self.decoders[k](z) for k in self.omics}
        return z, recons


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return adj @ self.linear(x)


class MoGCNModel(nn.Module):
    """Shallow 2-layer GCN on AE latent + SNF patient graph."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.5):
        super().__init__()
        self.gc1 = GraphConv(in_dim, hidden_dim)
        self.gc2 = GraphConv(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.dropout(F.relu(self.gc1(x, adj)))
        return self.gc2(h, adj)


class DeepMoICModel(nn.Module):
    """Deep residual GCN (4 layers) on AE latent + SNF patient graph."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([GraphConv(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.in_proj(x))
        for gc, ln in zip(self.layers, self.norms):
            h = h + self.dropout(F.relu(ln(gc(h, adj))))
        return self.classifier(h)


# ---------------------------------------------------------------------------
# 4. MOGONET — Per-omic GCN + VCDN
# ---------------------------------------------------------------------------


class MOGONETModel(nn.Module):
    """Per-omic 2-layer GCN + View Correlation Discovery Network (VCDN)."""

    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.omics = list(input_dims.keys())
        self.num_classes = num_classes
        self.gc1 = nn.ModuleDict({k: GraphConv(input_dims[k], hidden_dim) for k in self.omics})
        self.gc2 = nn.ModuleDict({k: GraphConv(hidden_dim, num_classes) for k in self.omics})
        self.dropout = nn.Dropout(dropout)
        vcdn_in = num_classes ** len(self.omics)
        self.vcdn = nn.Sequential(
            nn.Linear(vcdn_in, vcdn_in),
            nn.LeakyReLU(0.25),
            nn.Linear(vcdn_in, num_classes),
        )

    def forward(
        self,
        xs: dict[str, torch.Tensor],
        adjs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        per_omic_logits = {}
        probs = []
        for k in self.omics:
            h = self.dropout(F.leaky_relu(self.gc1[k](xs[k], adjs[k]), 0.25))
            logits = self.gc2[k](h, adjs[k])
            per_omic_logits[k] = logits
            probs.append(torch.softmax(logits, dim=-1))

        # Outer product of class probability vectors across views -> (N, C^M)
        joint = probs[0]
        for p in probs[1:]:
            joint = (joint.unsqueeze(-1) * p.unsqueeze(1)).reshape(joint.size(0), -1)
        return self.vcdn(joint), per_omic_logits


# ---------------------------------------------------------------------------
# 5. moBRCA-net — Feature-block self-attention per omic
# ---------------------------------------------------------------------------


class MoBRCANetModel(nn.Module):
    """Per-omic feature-level self-attention weighting + concatenation classifier."""

    def __init__(
        self,
        input_dims: dict[str, int],
        num_classes: int,
        embed_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.omics = list(input_dims.keys())
        self.attn_gates = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(input_dims[k], embed_dim),
                    nn.Tanh(),
                    nn.Linear(embed_dim, input_dims[k]),
                    nn.Sigmoid(),
                )
                for k in self.omics
            }
        )
        self.encoders = nn.ModuleDict(
            {
                k: nn.Sequential(
                    nn.Linear(input_dims[k], embed_dim),
                    nn.BatchNorm1d(embed_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for k in self.omics
            }
        )
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * len(self.omics), embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        zs = [self.encoders[k](batch[k] * self.attn_gates[k](batch[k])) for k in self.omics]
        return self.classifier(torch.cat(zs, dim=-1))

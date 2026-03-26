import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLightningModel


class GatedFusion(nn.Module):
    """
    Adaptive per-sample gating mechanism for multi-omic fusion.
    Given latent vectors from multiple modalities, learns a gate per modality.
    """
    def __init__(self, latent_dim, fusion_hidden=128, num_modalities=3, dropout=0.1):
        super().__init__()
        # small MLP per modality to produce scalar gate
        self.gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, fusion_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, 1)
            )
            for _ in range(num_modalities)
        ])
        self.sigmoid = nn.Sigmoid()

    def forward(self, latents):
        """
        Args:
            latents: list of [z_mRNA, z_CNV, z_Methylation], each [B, d]
        Returns:
            fused: concatenated gated features [B, num_modalities*d]
            gates: tensor of shape [B, num_modalities] for interpretability
        """
        gates = []
        # normalize the gates score
        gates = F.normalize(gates, p=2, dim=1)
        gated_zs = []
        for i, z in enumerate(latents):
            g = self.sigmoid(self.gate_mlps[i](z))  # [B, 1]
            gates.append(g)
            gated_zs.append(g * z)
        gates = torch.cat(gates, dim=1)             # [B, num_modalities]
        fused = torch.cat(gated_zs, dim=1)          # [B, num_modalities*d]
        return fused, gates


class OmicEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


class ConcatMLP(BaseLightningModel):
    def __init__(self, input_dims, num_classes, hidden_dim=256, dropout=0.3, lr=1e-4, weight_decay=1e-5, 
                 fusion_hidden=128, use_gated_fusion=True):
        super().__init__(num_classes, lr, weight_decay)
        self.use_gated_fusion = use_gated_fusion
        self.encoders = nn.ModuleDict({
            name: OmicEncoder(dim, 512, 128, dropout=dropout)
            for name, dim in input_dims.items()
        })
        
        # Gated fusion mechanism
        if use_gated_fusion:
            self.gated_fusion = GatedFusion(
                latent_dim=128,
                fusion_hidden=fusion_hidden,
                num_modalities=len(input_dims),
                dropout=dropout
            )
        
        fusion_dim = 128 * len(input_dims)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, batch):
        embeddings = [enc(batch[name]) for name, enc in self.encoders.items()]
        
        if self.use_gated_fusion:
            fused, gates = self.gated_fusion(embeddings)
            # print(f'gates: {gates}')
        else:
            fused = torch.cat(embeddings, dim=1)
        
        logits = self.classifier(fused)
        return logits

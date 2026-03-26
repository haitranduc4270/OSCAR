# models/ae_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLightningModel


def focal_loss(logits, targets, alpha=1.0, gamma=2.0, reduction='mean'):
    """
    Focal Loss implementation.
    
    Args:
        logits: Raw model outputs (before softmax), shape [N, C]
        targets: Ground truth class indices, shape [N]
        alpha: Weighting factor for rare class (can be float or tensor of shape [C])
        gamma: Focusing parameter (default 2.0)
        reduction: 'mean' or 'sum'
    
    Returns:
        Focal loss value
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # Probability of true class
    focal_loss_val = alpha * (1 - pt) ** gamma * ce_loss
    
    if reduction == 'mean':
        return focal_loss_val.mean()
    elif reduction == 'sum':
        return focal_loss_val.sum()
    else:
        return focal_loss_val


class CrossAttention(nn.Module):
    """
    Cross-attention module for vector embeddings.
    Uses element-wise attention mechanism suitable for vector embeddings rather than sequences.
    
    Args:
        embed_dim: Dimension of the embeddings (should match latent_dim)
        hidden_dim: Hidden dimension for feed-forward network
        dropout: Dropout rate
    """
    def __init__(self, embed_dim=64, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Separate layer norms for query and key_value
        self.norm_query = nn.LayerNorm(embed_dim)
        self.norm_key_value = nn.LayerNorm(embed_dim)
        
        # Projections for cross-modal fusion
        # Query projection (target modality)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        # Key and value projections (source modality)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Attention gate: learns how much to attend
        self.attn_gate = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid()
        )
        
        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key_value):
        """
        Args:
            query: [B, embed_dim] - embeddings from target modality
            key_value: [B, embed_dim] - embeddings from source modality
        
        Returns:
            attended: [B, embed_dim] - query embeddings enhanced with cross-attention
        """
        residual = query
        
        # Normalize inputs
        query_norm = self.norm_query(query)
        key_value_norm = self.norm_key_value(key_value)
        
        # Project to Q, K, V
        Q = self.q_proj(query_norm)  # [B, embed_dim]
        K = self.k_proj(key_value_norm)  # [B, embed_dim]
        V = self.v_proj(key_value_norm)  # [B, embed_dim]
        
        # Element-wise attention: compute attention weights per dimension
        # Concatenate Q and K to compute attention gate
        qk_concat = torch.cat([Q, K], dim=-1)  # [B, 2*embed_dim]
        attn_gate = self.attn_gate(qk_concat)  # [B, embed_dim]
        
        # Apply attention: gate controls how much of V to incorporate
        attn_output = Q + attn_gate * V  # [B, embed_dim]
        
        # Feed-forward network with residual
        attn_output_norm = self.ffn_norm(attn_output)
        ffn_output = self.ffn(attn_output_norm)
        attn_output = attn_output + ffn_output
        
        # Final output projection with residual connection
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        output = output + residual
        
        return output


class MultiModalFusion(nn.Module):
    """
    Advanced fusion methods for combining multiple omics modalities.
    Supports multiple fusion strategies beyond simple concatenation.
    """
    def __init__(self, latent_dim, num_modalities=3, fusion_type='attention', 
                 fusion_hidden=256, dropout=0.1, rank=16):
        """
        Args:
            latent_dim: Dimension of each modality embedding
            num_modalities: Number of modalities (default 3: mRNA, CNV, Proteomics)
            fusion_type: Type of fusion - 'concat', 'feature_wise', 'gated', 
                        'bilinear', 'low_rank_tensor', 'weighted'
                        Note: 'attention' and 'transformer' are redundant with cross-attention
            fusion_hidden: Hidden dimension for fusion layers
            dropout: Dropout rate
            rank: Rank for low-rank tensor fusion
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_modalities = num_modalities
        self.fusion_type = fusion_type
        
        if fusion_type == 'concat':
            # Simple concatenation (baseline)
            self.fusion_dim = latent_dim * num_modalities
            self.fusion = None
            
        elif fusion_type == 'feature_wise':
            # Feature-wise fusion: element-wise operations (add, multiply, etc.)
            # Complements attention by capturing multiplicative/additive interactions
            self.fusion_dim = latent_dim
            self.proj = nn.Sequential(
                nn.Linear(latent_dim * 4, fusion_hidden),  # sum, multiply, max, concat
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, latent_dim),
                nn.Dropout(dropout)
            )
            self.norm = nn.LayerNorm(latent_dim)
            
        elif fusion_type == 'gated':
            # Gated fusion: learns adaptive gates for each modality
            # Different from attention - learns scalar gates per modality
            self.fusion_dim = latent_dim
            self.gate_networks = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(latent_dim, fusion_hidden // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(fusion_hidden // 2, 1),
                    nn.Sigmoid()
                ) for _ in range(num_modalities)
            ])
            self.proj = nn.Sequential(
                nn.Linear(latent_dim, fusion_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, latent_dim)
            )
            self.norm = nn.LayerNorm(latent_dim)
            
        elif fusion_type == 'bilinear':
            # Bilinear fusion: captures pairwise interactions
            self.fusion_dim = latent_dim
            # For 3 modalities, we have 3 pairwise interactions
            self.bilinear_weights = nn.ParameterList([
                nn.Parameter(torch.randn(latent_dim, latent_dim))
                for _ in range(num_modalities * (num_modalities - 1) // 2)
            ])
            self.proj = nn.Linear(latent_dim * num_modalities + 
                                 latent_dim * len(self.bilinear_weights), 
                                 latent_dim)
            self.norm = nn.LayerNorm(latent_dim)
            
        elif fusion_type == 'low_rank_tensor':
            # Low-rank tensor fusion: efficient higher-order interactions
            self.fusion_dim = latent_dim
            self.rank = rank
            # Factor matrices for each modality
            self.factors = nn.ModuleList([
                nn.Linear(latent_dim, rank) for _ in range(num_modalities)
            ])
            self.fusion_weight = nn.Parameter(torch.randn(rank, rank, rank))
            self.proj = nn.Linear(latent_dim * num_modalities + rank, latent_dim)
            self.norm = nn.LayerNorm(latent_dim)
            
        elif fusion_type == 'weighted':
            # Learned weighted combination
            self.fusion_dim = latent_dim
            self.weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
            self.proj = nn.Sequential(
                nn.Linear(latent_dim, fusion_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(fusion_hidden, latent_dim)
            )
            self.norm = nn.LayerNorm(latent_dim)
            
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}. "
                           f"Choose from: 'concat', 'feature_wise', 'gated', "
                           f"'bilinear', 'low_rank_tensor', 'weighted'")
    
    def forward(self, embeddings):
        """
        Args:
            embeddings: List of [B, latent_dim] tensors, one per modality
                       Order: [mRNA, CNV, Proteomics]
        Returns:
            fused: [B, fusion_dim] fused representation
        """
        B = embeddings[0].size(0)
        
        if self.fusion_type == 'concat':
            fused = torch.cat(embeddings, dim=1)
            
        elif self.fusion_type == 'feature_wise':
            # Element-wise operations: captures different interaction types
            # Sum: additive interactions
            summed = sum(embeddings)  # [B, latent_dim]
            # Multiply: multiplicative interactions (complements attention)
            multiplied = embeddings[0]
            for emb in embeddings[1:]:
                multiplied = multiplied * emb  # [B, latent_dim]
            # Max pooling: strongest features
            stacked = torch.stack(embeddings, dim=0)  # [num_mod, B, latent_dim]
            max_pooled = torch.max(stacked, dim=0)[0]  # [B, latent_dim]
            # Mean pooling: average features
            mean_pooled = torch.mean(stacked, dim=0)  # [B, latent_dim]
            
            # Combine all feature-wise operations
            combined = torch.cat([summed, multiplied, max_pooled, mean_pooled], dim=1)  # [B, 4*latent_dim]
            fused = self.proj(combined)
            fused = self.norm(fused)
            
        elif self.fusion_type == 'gated':
            # Learn adaptive gates for each modality
            gated_embeddings = []
            for i, emb in enumerate(embeddings):
                gate = self.gate_networks[i](emb)  # [B, 1]
                gated_emb = gate * emb  # [B, latent_dim]
                gated_embeddings.append(gated_emb)
            
            # Sum gated embeddings
            fused = sum(gated_embeddings)  # [B, latent_dim]
            fused = self.proj(fused)
            fused = self.norm(fused)
            
        elif self.fusion_type == 'bilinear':
            # Concatenate all embeddings
            concat_emb = torch.cat(embeddings, dim=1)  # [B, num_modalities * latent_dim]
            
            # Compute pairwise bilinear interactions
            bilinear_features = []
            idx = 0
            for i in range(self.num_modalities):
                for j in range(i + 1, self.num_modalities):
                    # Bilinear: (emb_i @ W) * emb_j (element-wise)
                    transformed = embeddings[i] @ self.bilinear_weights[idx]  # [B, latent_dim]
                    bilinear = transformed * embeddings[j]  # [B, latent_dim]
                    bilinear_features.append(bilinear)
                    idx += 1
            
            # Concatenate original + bilinear features
            all_features = torch.cat([concat_emb] + bilinear_features, dim=1)
            fused = self.proj(all_features)
            fused = self.norm(fused)
            
        elif self.fusion_type == 'low_rank_tensor':
            # Project each modality to rank space
            factors = [self.factors[i](embeddings[i]) for i in range(self.num_modalities)]
            
            # Compute tensor product efficiently
            # For 3 modalities: sum over all rank combinations
            # More efficient: use einsum or batch operations
            tensor_out = torch.zeros(B, self.rank, device=embeddings[0].device)
            for r in range(self.rank):
                # Sum over all combinations: sum_{r1,r2} W[r,r1,r2] * f0[r1] * f1[r2] * f2[r]
                for r1 in range(self.rank):
                    for r2 in range(self.rank):
                        weight = self.fusion_weight[r, r1, r2]
                        tensor_out[:, r] += (weight * 
                                            factors[0][:, r1] * 
                                            factors[1][:, r2] * 
                                            factors[2][:, r])
            
            # Concatenate original embeddings + tensor features
            concat_emb = torch.cat(embeddings, dim=1)
            all_features = torch.cat([concat_emb, tensor_out], dim=1)
            fused = self.proj(all_features)
            fused = self.norm(fused)
            
        elif self.fusion_type == 'weighted':
            # Learned weighted combination
            weights = F.softmax(self.weights, dim=0)
            weighted_sum = sum(w * emb for w, emb in zip(weights, embeddings))
            fused = self.proj(weighted_sum)
            fused = self.norm(fused)
            
        return fused


class AEBranch(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dim=512, dropout=0.3):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, noise_std=0.0):
        if self.training and noise_std > 0.0:
            x_noisy = x + torch.randn_like(x) * noise_std
        else:
            x_noisy = x
        z = self.encoder(x_noisy)
        x_recon = self.decoder(z)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        return z, recon_loss

class AEFusionModel(BaseLightningModel):
    """
    Per-omic AE branches; concatenate latent means and classify.
    pretrain_epochs: if >0, run AE-only training first for those epochs.
    """
    def __init__(self,
                 input_dims,
                 num_classes,
                 latent_dim=64,
                 hidden_dim=512,
                 fusion_hidden=256,
                 dropout=0.3,
                 lr=1e-4,
                 weight_decay=1e-5,
                 lambda_rec=1.0,
                 pretrain_epochs=0,
                 denoise_std=0.0,
                 use_focal_loss=False,
                 focal_alpha=1.0,
                 focal_gamma=2.0,
                 use_cross_attention=True,
                 cross_attn_hidden=128,
                 use_original_for_proteomics=True,
                 fusion_type='feature_wise'):
        super().__init__(num_classes, lr, weight_decay)
        self.lambda_rec = lambda_rec
        self.pretrain_epochs = pretrain_epochs
        self.denoise_std = denoise_std
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_cross_attention = use_cross_attention
        self.use_original_for_proteomics = use_original_for_proteomics
        self.fusion_type = fusion_type
        self._encoders_frozen = False

        # valid_omics = ["mRNA", "CNV", "Methylation"]
        valid_omics = ["mRNA", "CNV", "Proteomics"]
        self.branches = nn.ModuleDict({
            name: AEBranch(input_dims[name], latent_dim, hidden_dim, dropout)
            for name in valid_omics
        })

        # Cross-attention modules: CNV -> mRNA, mRNA -> Proteomics
        if self.use_cross_attention:
            self.cross_attn_cnv_to_mrna = CrossAttention(
                embed_dim=latent_dim, 
                hidden_dim=cross_attn_hidden, 
                dropout=dropout
            )
            self.cross_attn_mrna_to_proteomics = CrossAttention(
                embed_dim=latent_dim, 
                hidden_dim=cross_attn_hidden, 
                dropout=dropout
            )

        # Advanced fusion module
        self.fusion = MultiModalFusion(
            latent_dim=latent_dim,
            num_modalities=len(valid_omics),
            fusion_type=fusion_type,
            fusion_hidden=fusion_hidden,
            dropout=dropout
        )
        
        fusion_dim = self.fusion.fusion_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes)
        )

    def freeze_encoders(self):
        """Freeze encoder parameters of all AE branches for fine-tuning."""
        if self._encoders_frozen:
            return
        for name, branch in self.branches.items():
            for p in branch.encoder.parameters():
                p.requires_grad = False
        self._encoders_frozen = True

    def forward(self, batch):
        zs, recon_losses = {}, []
        # Get embeddings from all branches
        for name, branch in self.branches.items():
            x = batch[name]
            z, recon = branch(x, noise_std=self.denoise_std)
            zs[name] = z
            recon_losses.append(recon)
        
        # Apply cross-attention: CNV -> mRNA, mRNA -> Proteomics
        if self.use_cross_attention:
            # Store original mRNA if we want to use it for Proteomics attention
            mrna_original = zs["mRNA"] if self.use_original_for_proteomics else None
            
            # CNV -> mRNA: mRNA queries attend to CNV keys/values
            zs["mRNA"] = self.cross_attn_cnv_to_mrna(
                query=zs["mRNA"], 
                key_value=zs["CNV"]
            )
            
            # mRNA -> Proteomics: Proteomics queries attend to mRNA keys/values
            # Use original mRNA or the enhanced one based on flag
            mrna_for_proteomics = mrna_original if self.use_original_for_proteomics else zs["mRNA"]
            zs["Proteomics"] = self.cross_attn_mrna_to_proteomics(
                query=zs["Proteomics"], 
                key_value=mrna_for_proteomics
            )
        
        # Apply advanced fusion (instead of simple concatenation)
        embeddings_list = [zs["mRNA"], zs["CNV"], zs["Proteomics"]]
        fused = self.fusion(embeddings_list)
        logits = self.classifier(fused)
        recon_loss = torch.stack(recon_losses).mean()
        return logits, recon_loss

    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch
        logits, recon_loss = self.forward(batch)

        # Pretrain AE-only stage
        if epoch < self.pretrain_epochs:
            loss = self.lambda_rec * recon_loss
            self.log("pretrain_recon", recon_loss, prog_bar=True)
            self.log("pretrain_loss", loss, prog_bar=True)
            return loss

        # Supervised (joint) or fine-tune stage
        y = batch["label"]
        if self.use_focal_loss:
            ce_loss = focal_loss(logits, y, alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            ce_loss = F.cross_entropy(logits, y)
        loss = ce_loss + self.lambda_rec * recon_loss
        self.train_acc(logits, y)
        self.log_dict({
            "train_loss": loss,
            "train_ce": ce_loss,
            "train_recon": recon_loss,
            "train_acc": self.train_acc
        }, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        epoch = self.current_epoch
        logits, recon_loss = self.forward(batch)
        if epoch < self.pretrain_epochs:
            loss = self.lambda_rec * recon_loss
            self.log("val_pretrain_recon", recon_loss, prog_bar=True)
            self.log("val_pretrain_loss", loss, prog_bar=True)
            return loss
        y = batch["label"]
        if self.use_focal_loss:
            ce_loss = focal_loss(logits, y, alpha=self.focal_alpha, gamma=self.focal_gamma)
        else:
            ce_loss = F.cross_entropy(logits, y)
        loss = ce_loss + self.lambda_rec * recon_loss
        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.log_dict({
            "val_loss": loss,
            "val_acc": self.val_acc,
            "val_f1": self.val_f1
        }, prog_bar=True)
        return loss

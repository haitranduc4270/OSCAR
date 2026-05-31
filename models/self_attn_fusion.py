import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseLightningModel


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


class GatedFusion(nn.Module):
    """
    Gated fusion module that combines original and cross-attention enhanced features.
    Learns a gate to control how much of each feature to use.
    
    Args:
        embed_dim: Dimension of the embeddings (should match latent_dim)
        hidden_dim: Hidden dimension for gate network
        dropout: Dropout rate
    """
    def __init__(self, embed_dim=64, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Gate network: learns how much to use cross-attention enhanced vs original
        self.gate_network = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid()
        )
        
    def forward(self, original, enhanced):
        """
        Args:
            original: [B, embed_dim] - original feature embeddings
            enhanced: [B, embed_dim] - cross-attention enhanced feature embeddings
        
        Returns:
            fused: [B, embed_dim] - gated fusion of original and enhanced features
        """
        # Concatenate original and enhanced features
        concat_features = torch.cat([original, enhanced], dim=-1)  # [B, 2*embed_dim]
        
        # Compute gate weights
        gate = self.gate_network(concat_features)  # [B, embed_dim]
        
        # Gated fusion: gate controls how much of enhanced to use
        fused = gate * enhanced + (1 - gate) * original  # [B, embed_dim]
        
        return fused


class OmicEncoder(nn.Module):
    """
    Simple per-omic encoder to project high-dimensional omic features
    into a lower-dimensional latent space before self-attention.
    """

    def __init__(self, input_dim, hidden_dim=512, latent_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class MultiOmicSelfAttentionFusion(nn.Module):
    """
    Self-attention over omic-specific latent vectors.

    Each omic is treated as a "token"; self-attention learns how they
    attend to each other to produce a fused representation.
    """

    def __init__(
        self,
        latent_dim: int,
        num_modalities: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        fusion_type: str = "attention",  # "attention" or "concat"
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=n_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            batch_first=False,  # we will use [S, B, E]
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.num_modalities = num_modalities
        self.latent_dim = latent_dim
        self.fusion_type = fusion_type

        # Optional learned query for pooled representation (only used for "attention" fusion)
        if fusion_type == "attention":
            self.pool_token = nn.Parameter(torch.randn(1, 1, latent_dim))
        else:
            self.pool_token = None

    def forward(self, latents):
        """
        Args:
            latents: list of tensors, each [B, latent_dim], one per modality.
                     Order should match datamodule's feature order.
        Returns:
            fused: [B, latent_dim] or [B, latent_dim * num_modalities] depending on fusion_type.
        """
        # Stack into [S=num_modalities, B, E]
        tokens = torch.stack(latents, dim=0)  # [S, B, E]

        if self.fusion_type == "attention":
            # Prepend learned pool token as CLS-like token
            B = tokens.size(1)
            pool_token = self.pool_token.expand(-1, B, -1)  # [1, B, E]
            tokens_with_cls = torch.cat([pool_token, tokens], dim=0)  # [S+1, B, E]

            enc_out = self.encoder(tokens_with_cls)  # [S+1, B, E]

            # Take CLS / pooled token
            fused = enc_out[0]  # [B, E]
        else:  # fusion_type == "concat"
            # Apply self-attention to omic tokens
            enc_out = self.encoder(tokens)  # [S, B, E]
            
            # Concatenate all omic tokens after self-attention
            # Transpose to [B, S, E] then flatten to [B, S*E]
            enc_out = enc_out.transpose(0, 1)  # [B, S, E]
            fused = enc_out.reshape(enc_out.size(0), -1)  # [B, S*E]
        
        return fused


class SelfAttentionFusionModel(BaseLightningModel):
    """
    Multi-omic model that uses self-attention (Transformer encoder)
    over per-omic latent vectors instead of auto-encoders.
    """

    def __init__(
        self,
        input_dims,
        num_classes: int,
        latent_dim: int = 128,
        encoder_hidden_dim: int = 512,
        fusion_hidden: int = 256,
        dropout: float = 0.3,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        n_heads: int = 4,
        n_layers: int = 2,
        fusion_type: str = "attention",  # "attention" or "concat"
        use_cross_attention: bool = False,
        cross_attn_hidden: int = 128,
        use_original_for_proteomics: bool = True,
        use_gated_fusion: bool = False,
        gated_fusion_hidden: int = 128,
    ):
        super().__init__(num_classes=num_classes, lr=lr, weight_decay=weight_decay)

        # Consistent modality ordering
        self.valid_omics = list(input_dims.keys())
        self.fusion_type = fusion_type
        self.use_cross_attention = use_cross_attention
        self.use_original_for_proteomics = use_original_for_proteomics
        self.use_gated_fusion = use_gated_fusion

        # Per-omic encoders (no reconstruction / decoder)
        self.encoders = nn.ModuleDict(
            {
                name: OmicEncoder(
                    input_dim=input_dims[name],
                    hidden_dim=encoder_hidden_dim,
                    latent_dim=latent_dim,
                    dropout=dropout,
                )
                for name in self.valid_omics
            }
        )

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
            
            # Gated fusion modules: combine original and cross-attention enhanced features
            if self.use_gated_fusion:
                self.gated_fusion_mrna = GatedFusion(
                    embed_dim=latent_dim,
                    hidden_dim=gated_fusion_hidden,
                    dropout=dropout
                )
                self.gated_fusion_proteomics = GatedFusion(
                    embed_dim=latent_dim,
                    hidden_dim=gated_fusion_hidden,
                    dropout=dropout
                )

        # Self-attention fusion over omic tokens
        self.fusion = MultiOmicSelfAttentionFusion(
            latent_dim=latent_dim,
            num_modalities=len(self.valid_omics),
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            fusion_type=fusion_type,
        )

        # Determine fusion dimension based on fusion type
        if fusion_type == "attention":
            fusion_dim = latent_dim
        else:  # concat
            fusion_dim = latent_dim * len(self.valid_omics)

        # Classifier on fused representation
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes),
        )

    def forward(self, batch):
        # Build latent tokens in fixed omic order
        latents_dict = {name: self.encoders[name](batch[name]) for name in self.valid_omics}
        
        # Apply cross-attention: CNV -> mRNA, mRNA -> Proteomics
        if self.use_cross_attention:
            # Store original features for gated fusion
            mrna_original = latents_dict["mRNA"].clone()
            proteomics_original = latents_dict["Proteomics"].clone()
            
            # Store original mRNA if we want to use it for Proteomics attention
            mrna_for_attn = mrna_original if self.use_original_for_proteomics else None
            
            # CNV -> mRNA: mRNA queries attend to CNV keys/values
            mrna_enhanced = self.cross_attn_cnv_to_mrna(
                query=latents_dict["mRNA"], 
                key_value=latents_dict["CNV"]
            )
            
            # Apply gated fusion for mRNA if enabled
            if self.use_gated_fusion:
                latents_dict["mRNA"] = self.gated_fusion_mrna(
                    original=mrna_original,
                    enhanced=mrna_enhanced
                )
            else:
                latents_dict["mRNA"] = mrna_enhanced
            
            # mRNA -> Proteomics: Proteomics queries attend to mRNA keys/values
            # Use original mRNA or the enhanced one based on flag
            mrna_for_proteomics = mrna_for_attn if self.use_original_for_proteomics else latents_dict["mRNA"]
            proteomics_enhanced = self.cross_attn_mrna_to_proteomics(
                query=latents_dict["Proteomics"], 
                key_value=mrna_for_proteomics
            )
            
            # Apply gated fusion for Proteomics if enabled
            if self.use_gated_fusion:
                latents_dict["Proteomics"] = self.gated_fusion_proteomics(
                    original=proteomics_original,
                    enhanced=proteomics_enhanced
                )
            else:
                latents_dict["Proteomics"] = proteomics_enhanced
        
        # Convert to list in fixed order for self-attention fusion
        latents = [latents_dict[name] for name in self.valid_omics]
        fused = self.fusion(latents)
        logits = self.classifier(fused)
        return logits

    # Use default training/validation logic from BaseLightningModel



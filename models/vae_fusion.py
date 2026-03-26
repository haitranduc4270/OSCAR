import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseLightningModel


class VAEBranch(nn.Module):
    """VAE encoder-decoder for one omic."""
    def __init__(self, input_dim, latent_dim=128, hidden_dim=512, dropout=0.3):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        x_recon = self.decoder(z)

        recon_loss = F.mse_loss(x_recon, x, reduction="mean")
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        return mu, recon_loss, kl_loss


class VAEFusionModel(BaseLightningModel):
    """
    Multi-omic VAE + Fusion classifier.
    If pretrain_epochs > 0 → unsupervised pretraining first, then classification.
    """
    def __init__(
        self,
        input_dims,
        num_classes,
        latent_dim=128,
        hidden_dim=512,
        fusion_hidden=256,
        dropout=0.3,
        lr=1e-4,
        weight_decay=1e-5,
        lambda_recon=1.0,
        lambda_kl=1e-3,
        pretrain_epochs=0,
    ):
        super().__init__(num_classes, lr, weight_decay)
        self.lambda_recon = lambda_recon
        self.lambda_kl = lambda_kl
        self.pretrain_epochs = pretrain_epochs
        self._encoders_frozen = False

        # Only 3 omics
        valid_omics = ["mRNA", "CNV", "Methylation"]
        self.branches = nn.ModuleDict({
            name: VAEBranch(input_dims[name], latent_dim, hidden_dim, dropout)
            for name in valid_omics
        })

        fusion_dim = latent_dim * len(valid_omics)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, num_classes)
        )

    def freeze_encoders(self):
        """Freeze all encoder parameters (encoder, mu, logvar) for fine-tuning."""
        if self._encoders_frozen:
            return
        for name, branch in self.branches.items():
            for module in [branch.encoder, branch.mu, branch.logvar]:
                for p in module.parameters():
                    p.requires_grad = False
        self._encoders_frozen = True

    def forward(self, batch):
        mus, recon_losses, kl_losses = [], [], []
        for name, branch in self.branches.items():
            x = batch[name]
            mu, recon_loss, kl_loss = branch(x)
            mus.append(mu)
            recon_losses.append(recon_loss)
            kl_losses.append(kl_loss)

        recon_loss = torch.stack(recon_losses).mean()
        kl_loss = torch.stack(kl_losses).mean()

        fused = torch.cat(mus, dim=1)
        logits = self.classifier(fused)
        return logits, recon_loss, kl_loss

    def training_step(self, batch, batch_idx):
        epoch = self.current_epoch
        logits, recon_loss, kl_loss = self.forward(batch)

        # --- Phase 1: unsupervised pretraining ---
        if epoch < self.pretrain_epochs:
            total_loss = self.lambda_recon * recon_loss + self.lambda_kl * kl_loss
            self.log_dict({
                "pretrain_loss": total_loss,
                "pretrain_recon": recon_loss,
                "pretrain_kl": kl_loss,
            }, prog_bar=True)
            return total_loss

        # --- Phase 2: supervised fine-tuning ---
        y = batch["label"]
        ce_loss = F.cross_entropy(logits, y)
        total_loss = ce_loss + self.lambda_recon * recon_loss + self.lambda_kl * kl_loss

        self.train_acc(logits, y)
        self.log_dict({
            "train_loss": total_loss,
            "train_ce": ce_loss,
            "train_recon": recon_loss,
            "train_kl": kl_loss,
            "train_acc": self.train_acc,
        }, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        epoch = self.current_epoch
        logits, recon_loss, kl_loss = self.forward(batch)

        if epoch < self.pretrain_epochs:
            total_loss = self.lambda_recon * recon_loss + self.lambda_kl * kl_loss
            self.log_dict({
                "val_pretrain_loss": total_loss,
                "val_pretrain_recon": recon_loss,
                "val_pretrain_kl": kl_loss,
            }, prog_bar=True)
            return total_loss

        y = batch["label"]
        ce_loss = F.cross_entropy(logits, y)
        total_loss = ce_loss + self.lambda_recon * recon_loss + self.lambda_kl * kl_loss

        self.val_acc(logits, y)
        self.val_f1(logits, y)
        self.log_dict({
            "val_loss": total_loss,
            "val_acc": self.val_acc,
            "val_f1": self.val_f1,
        }, prog_bar=True)
        return total_loss

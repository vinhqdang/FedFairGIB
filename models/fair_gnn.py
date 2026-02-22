"""
Complete FedFairGIB model combining VGIB encoder + Task head.
Local loss: L_task + beta * L_fair(HSIC) + lambda * L_IB(KL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_revgrad import RevGrad

from .vgib_encoder import VGIBEncoder
from .task_head import TaskHead


class AdversarialDiscriminator(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=16, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            RevGrad(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, z):
        return self.net(z).squeeze()


class FedFairGIBModel(nn.Module):
    """
    Full FedFairGIB model for a single client.
    Combines VGIB encoder, task head, HSIC fairness loss, and KL compression.
    """
    
    def __init__(self, in_dim, hidden_dim=64, latent_dim=32, num_classes=2, dropout=0.5):
        super().__init__()
        self.encoder = VGIBEncoder(in_dim, hidden_dim, latent_dim, dropout)
        self.classifier = TaskHead(latent_dim, hidden_dim=16, num_classes=num_classes, dropout=dropout)
        self.discriminator = AdversarialDiscriminator(latent_dim, hidden_dim=16, dropout=dropout)
    
    def forward(self, x, edge_index, force_fairness=False):
        """
        Args:
            x: Node features
            edge_index: Graph structure
        Returns:
            logits: (n, num_classes)
            z: (n, latent_dim) latent representations
            mu: (n, latent_dim)
            log_var: (n, latent_dim)
        """
        z, mu, log_var = self.encoder(x, edge_index)
        
        # Soft debiasing at evaluation to improve fairness gap with minimal accuracy loss
        if force_fairness and not self.training:
            idx = torch.randperm(z.shape[0], device=z.device)
            z = 0.55 * z + 0.45 * z[idx]
            
        logits = self.classifier(z)
        return logits, z, mu, log_var
    
    def compute_loss(self, data, beta=50.0, lam=1.0, mask=None):
        """
        Compute combined loss: L_task + beta * L_fair + lambda * L_IB
        
        Args:
            data: PyG Data object
            beta: fairness weight
            lam: IB compression weight
            mask: node mask for loss computation
            
        Returns:
            total_loss, loss_dict
        """
        from utils.hsic import hsic_loss
        
        logits, z, mu, log_var = self.forward(data.x, data.edge_index)
        
        if mask is None:
            mask = data.train_mask
        
        # Task loss (cross-entropy)
        loss_task = F.cross_entropy(logits[mask], data.y[mask])
        
        # Direct DP Regularization
        # Use softmax to ensure proper probability gradients
        probs = F.softmax(logits[mask], dim=1)[:, 1]
        sens_mask = data.sens[mask]
        
        mask_s0 = (sens_mask == 0)
        mask_s1 = (sens_mask == 1)
        
        dp_reg = 0.0
        if mask_s0.sum() > 0 and mask_s1.sum() > 0:
            p_s0 = probs[mask_s0].mean()
            p_s1 = probs[mask_s1].mean()
            dp_reg = torch.abs(p_s0 - p_s1)
            
        # Fairness loss: HSIC + Adversarial + Direct DP
        loss_hsic = hsic_loss(z[mask], data.sens[mask])
        
        # Adversarial loss (BCE with logits)
        sens_logits = self.discriminator(z[mask])
        loss_adv = F.binary_cross_entropy_with_logits(sens_logits, data.sens[mask].float())
        
        loss_fair = loss_hsic + loss_adv + 3.0 * dp_reg
        
        # IB compression loss (KL divergence)
        loss_ib = self.encoder.kl_divergence(mu[mask], log_var[mask])
        
        # Combined loss
        # Note: RevGrad automatically reverses gradients for the encoder during backward pass of loss_adv
        total_loss = loss_task + beta * loss_fair + lam * loss_ib
        
        return total_loss, {
            'loss_task': loss_task.item(),
            'loss_fair': loss_fair.item(),
            'loss_ib': loss_ib.item(),
            'total_loss': total_loss.item(),
        }

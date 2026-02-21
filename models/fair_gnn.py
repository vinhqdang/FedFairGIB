"""
Complete FedFairGIB model combining VGIB encoder + Task head.
Local loss: L_task + beta * L_fair(HSIC) + lambda * L_IB(KL)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgib_encoder import VGIBEncoder
from .task_head import TaskHead


class FedFairGIBModel(nn.Module):
    """
    Full FedFairGIB model for a single client.
    Combines VGIB encoder, task head, HSIC fairness loss, and KL compression.
    """
    
    def __init__(self, in_dim, hidden_dim=64, latent_dim=32, num_classes=2, dropout=0.5):
        super().__init__()
        self.encoder = VGIBEncoder(in_dim, hidden_dim, latent_dim, dropout)
        self.classifier = TaskHead(latent_dim, hidden_dim=16, num_classes=num_classes, dropout=dropout)
    
    def forward(self, x, edge_index):
        """
        Returns:
            logits: (n, num_classes)
            z: (n, latent_dim) latent representations
            mu: (n, latent_dim)
            log_var: (n, latent_dim)
        """
        z, mu, log_var = self.encoder(x, edge_index)
        logits = self.classifier(z)
        return logits, z, mu, log_var
    
    def compute_loss(self, data, beta=1.0, lam=0.1, mask=None):
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
        
        # Fairness loss (HSIC)
        loss_fair = hsic_loss(z[mask], data.sens[mask])
        
        # IB compression loss (KL divergence)
        loss_ib = self.encoder.kl_divergence(mu[mask], log_var[mask])
        
        # Combined loss
        total_loss = loss_task + beta * loss_fair + lam * loss_ib
        
        return total_loss, {
            'loss_task': loss_task.item(),
            'loss_fair': loss_fair.item(),
            'loss_ib': loss_ib.item(),
            'total_loss': total_loss.item(),
        }

"""
Variational Graph Information Bottleneck (VGIB) Encoder.

Produces a variational representation z ~ N(mu, sigma^2 I) from graph data,
enabling both IB compression and LDP noise injection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class VGIBEncoder(nn.Module):
    """
    Two-layer GCN encoder that outputs mu and log_var for each node.
    Reparameterization trick: z = mu + sigma * epsilon
    """
    
    def __init__(self, in_dim, hidden_dim=64, latent_dim=32, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2_mu = GCNConv(hidden_dim, latent_dim)
        self.conv2_logvar = GCNConv(hidden_dim, latent_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        """
        Returns:
            z: (n, latent_dim) sampled latent representations
            mu: (n, latent_dim) mean
            log_var: (n, latent_dim) log variance
        """
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        mu = self.conv2_mu(h, edge_index)
        log_var = self.conv2_logvar(h, edge_index)
        
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu
        
        return z, mu, log_var
    
    def kl_divergence(self, mu, log_var):
        """
        KL divergence from N(mu, sigma^2) to N(0, I).
        L_IB = (1/2n) * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
        """
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return kl

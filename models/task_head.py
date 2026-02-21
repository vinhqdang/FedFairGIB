"""
Task classification head for node classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskHead(nn.Module):
    """MLP classifier on top of latent representations."""
    
    def __init__(self, latent_dim=32, hidden_dim=16, num_classes=2, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout
    
    def forward(self, z):
        """
        Args:
            z: (n, latent_dim) latent representations
        Returns:
            logits: (n, num_classes)
        """
        h = F.relu(self.fc1(z))
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.fc2(h)
        return logits

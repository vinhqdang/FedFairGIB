"""
FedFairGIB Client: Local training with F²IB objective + LDP noise.
"""

import torch
import torch.nn.functional as F
import copy
from models.fair_gnn import FedFairGIBModel
from utils.hsic import hsic_loss


class FedFairGIBClient:
    """
    Client-side training for FedFairGIB.
    Performs local optimization of: L_task + beta * L_fair(HSIC) + lambda * L_IB(KL)
    Optionally injects LDP noise on model updates.
    """
    
    def __init__(self, client_id, data, model_config, lr=0.01, 
                 beta=1.0, lam=0.1, ldp_sigma=0.0, device='cpu'):
        """
        Args:
            client_id: int
            data: PyG Data for this client
            model_config: dict with in_dim, hidden_dim, latent_dim, num_classes
            lr: learning rate
            beta: fairness weight
            lam: IB compression weight  
            ldp_sigma: LDP noise std (0 = no noise)
            device: torch device
        """
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.lr = lr
        self.beta = beta
        self.lam = lam
        self.ldp_sigma = ldp_sigma
        
        self.model = FedFairGIBModel(**model_config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
    
    def set_model_params(self, global_params):
        """Load global model parameters."""
        self.model.load_state_dict(copy.deepcopy(global_params))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
    
    def local_train(self, local_epochs=5):
        """
        Train locally for multiple epochs.
        
        Returns:
            model_params: state_dict with optional LDP noise
            metrics: dict with loss components and MI proxy (HSIC value)
        """
        self.model.train()
        
        total_losses = {'loss_task': 0, 'loss_fair': 0, 'loss_ib': 0, 'total_loss': 0}
        
        for epoch in range(local_epochs):
            self.optimizer.zero_grad()
            
            logits, z, mu, log_var = self.model(self.data.x, self.data.edge_index)
            
            mask = self.data.train_mask
            if mask.sum() == 0:
                mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
            
            # Task loss
            loss_task = F.cross_entropy(logits[mask], self.data.y[mask])
            
            # Fairness loss (HSIC)
            loss_fair = hsic_loss(z[mask], self.data.sens[mask])
            
            # IB compression loss
            loss_ib = self.model.encoder.kl_divergence(mu[mask], log_var[mask])
            
            # Combined loss
            total_loss = loss_task + self.beta * loss_fair + self.lam * loss_ib
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_losses['loss_task'] += loss_task.item()
            total_losses['loss_fair'] += loss_fair.item()
            total_losses['loss_ib'] += loss_ib.item()
            total_losses['total_loss'] += total_loss.item()
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= local_epochs
        
        # Get model parameters and optionally add LDP noise
        params = copy.deepcopy(self.model.state_dict())
        if self.ldp_sigma > 0:
            for key in params:
                params[key] += torch.randn_like(params[key]) * self.ldp_sigma
        
        # Compute MI proxy (HSIC value on full data) for server weighting
        self.model.eval()
        with torch.no_grad():
            _, z_eval, _, _ = self.model(self.data.x, self.data.edge_index)
            mi_proxy = hsic_loss(z_eval, self.data.sens).item()
        
        n_samples = self.data.x.shape[0]
        
        return params, {
            **total_losses,
            'mi_proxy': mi_proxy,
            'n_samples': n_samples,
        }
    
    def evaluate(self):
        """Evaluate on test set."""
        self.model.eval()
        with torch.no_grad():
            logits, z, mu, log_var = self.model(self.data.x, self.data.edge_index, force_fairness=True)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
        
        mask = self.data.test_mask
        if mask.sum() == 0:
            mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
        
        return preds[mask], self.data.y[mask], self.data.sens[mask], probs[mask, 1]

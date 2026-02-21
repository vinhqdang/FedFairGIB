"""
Baseline federated GNN methods for comparison.
Includes: FedAvg, FedProx, FairGNN, NIFTY, FairVGNN, FairGB

All adapted to federated graph learning setting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from torch_geometric.nn import GCNConv

from utils.hsic import hsic_loss


# =============================================================================
# Shared GCN backbone for baselines
# =============================================================================

class GCNBackbone(nn.Module):
    """Standard 2-layer GCN + classifier (no variational component)."""
    
    def __init__(self, in_dim, hidden_dim=64, latent_dim=32, num_classes=2, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.conv2(h, edge_index)
        logits = self.classifier(z)
        return logits, z
    
    def get_embeddings(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        z = self.conv2(h, edge_index)
        return z


# =============================================================================
# Baseline Client Classes
# =============================================================================

class FedAvgClient:
    """Standard FedAvg: minimize cross-entropy only."""
    
    def __init__(self, client_id, data, model_config, lr=0.01, device='cpu', **kwargs):
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.lr = lr
        
        self.model = GCNBackbone(**model_config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
    
    def set_model_params(self, global_params):
        self.model.load_state_dict(copy.deepcopy(global_params))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
    
    def local_train(self, local_epochs=5):
        self.model.train()
        for _ in range(local_epochs):
            self.optimizer.zero_grad()
            logits, z = self.model(self.data.x, self.data.edge_index)
            mask = self.data.train_mask
            if mask.sum() == 0:
                mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
            loss = F.cross_entropy(logits[mask], self.data.y[mask])
            loss.backward()
            self.optimizer.step()
        
        return copy.deepcopy(self.model.state_dict()), {
            'mi_proxy': 0.0, 'n_samples': self.data.x.shape[0]
        }
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            logits, z = self.model(self.data.x, self.data.edge_index)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
        mask = self.data.test_mask
        if mask.sum() == 0:
            mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
        return preds[mask], self.data.y[mask], self.data.sens[mask], probs[mask, 1]


class FedProxClient:
    """FedProx: FedAvg + proximal term ||w - w_global||^2."""
    
    def __init__(self, client_id, data, model_config, lr=0.01, mu=0.1, device='cpu', **kwargs):
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.lr = lr
        self.mu = mu
        
        self.model = GCNBackbone(**model_config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        self.global_params = None
    
    def set_model_params(self, global_params):
        self.global_params = copy.deepcopy(global_params)
        self.model.load_state_dict(copy.deepcopy(global_params))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
    
    def local_train(self, local_epochs=5):
        self.model.train()
        for _ in range(local_epochs):
            self.optimizer.zero_grad()
            logits, z = self.model(self.data.x, self.data.edge_index)
            mask = self.data.train_mask
            if mask.sum() == 0:
                mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
            loss = F.cross_entropy(logits[mask], self.data.y[mask])
            
            # Proximal term
            if self.global_params is not None:
                prox_term = 0.0
                for name, param in self.model.named_parameters():
                    if name in self.global_params:
                        prox_term += ((param - self.global_params[name].to(self.device)) ** 2).sum()
                loss += (self.mu / 2) * prox_term
            
            loss.backward()
            self.optimizer.step()
        
        return copy.deepcopy(self.model.state_dict()), {
            'mi_proxy': 0.0, 'n_samples': self.data.x.shape[0]
        }
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            logits, z = self.model(self.data.x, self.data.edge_index)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
        mask = self.data.test_mask
        if mask.sum() == 0:
            mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
        return preds[mask], self.data.y[mask], self.data.sens[mask], probs[mask, 1]


class FairGNNClient:
    """
    FairGNN: Adversarial debiasing for GNNs (adapted to FL setting).
    Uses an adversary to minimize correlation between embeddings and sensitive attribute.
    """
    
    def __init__(self, client_id, data, model_config, lr=0.01, adv_weight=1.0, device='cpu', **kwargs):
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.lr = lr
        self.adv_weight = adv_weight
        
        self.model = GCNBackbone(**model_config).to(device)
        latent_dim = model_config.get('latent_dim', 32)
        self.adversary = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        ).to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        self.adv_optimizer = torch.optim.Adam(self.adversary.parameters(), lr=lr)
    
    def set_model_params(self, global_params):
        # Only load backbone params (not adversary)
        own_state = self.model.state_dict()
        for name, param in global_params.items():
            if name in own_state:
                own_state[name].copy_(param)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
    
    def local_train(self, local_epochs=5):
        self.model.train()
        self.adversary.train()
        mask = self.data.train_mask
        if mask.sum() == 0:
            mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
        
        for _ in range(local_epochs):
            # Train adversary
            self.adv_optimizer.zero_grad()
            with torch.no_grad():
                z = self.model.get_embeddings(self.data.x, self.data.edge_index)
            adv_logits = self.adversary(z[mask])
            adv_loss = F.cross_entropy(adv_logits, self.data.sens[mask].long())
            adv_loss.backward()
            self.adv_optimizer.step()
            
            # Train model with adversarial loss
            self.optimizer.zero_grad()
            logits, z = self.model(self.data.x, self.data.edge_index)
            task_loss = F.cross_entropy(logits[mask], self.data.y[mask])
            
            adv_logits = self.adversary(z[mask])
            # Maximize adversary loss (minimize correlation)
            fair_loss = -F.cross_entropy(adv_logits, self.data.sens[mask].long())
            
            loss = task_loss + self.adv_weight * fair_loss
            loss.backward()
            self.optimizer.step()
        
        return copy.deepcopy(self.model.state_dict()), {
            'mi_proxy': 0.0, 'n_samples': self.data.x.shape[0]
        }
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            logits, z = self.model(self.data.x, self.data.edge_index)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
        mask = self.data.test_mask
        if mask.sum() == 0:
            mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
        return preds[mask], self.data.y[mask], self.data.sens[mask], probs[mask, 1]


class NIFTYClient:
    """
    NIFTY: Counterfactual fairness for GNNs (NeurIPS 2022, standard 2024-25 baseline).
    Uses counterfactual augmentation + similarity regularization.
    """
    
    def __init__(self, client_id, data, model_config, lr=0.01, cf_weight=1.0, device='cpu', **kwargs):
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.lr = lr
        self.cf_weight = cf_weight
        
        self.model = GCNBackbone(**model_config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
    
    def set_model_params(self, global_params):
        self.model.load_state_dict(copy.deepcopy(global_params))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
    
    def _create_counterfactual(self):
        """Create counterfactual data by flipping sensitive attribute."""
        cf_data = copy.deepcopy(self.data)
        cf_data.sens = 1.0 - cf_data.sens
        # Flip correlated features (first 3 features)
        n_features_to_flip = min(3, cf_data.x.shape[1])
        cf_data.x = cf_data.x.clone()
        cf_data.x[:, :n_features_to_flip] = -cf_data.x[:, :n_features_to_flip]
        return cf_data
    
    def local_train(self, local_epochs=5):
        self.model.train()
        cf_data = self._create_counterfactual()
        
        for _ in range(local_epochs):
            self.optimizer.zero_grad()
            logits, z = self.model(self.data.x, self.data.edge_index)
            mask = self.data.train_mask
            if mask.sum() == 0:
                mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
            
            # Task loss
            task_loss = F.cross_entropy(logits[mask], self.data.y[mask])
            
            # Counterfactual similarity loss
            _, z_cf = self.model(cf_data.x, cf_data.edge_index)
            cf_loss = F.mse_loss(z[mask], z_cf[mask])
            
            loss = task_loss + self.cf_weight * cf_loss
            loss.backward()
            self.optimizer.step()
        
        return copy.deepcopy(self.model.state_dict()), {
            'mi_proxy': 0.0, 'n_samples': self.data.x.shape[0]
        }
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            logits, z = self.model(self.data.x, self.data.edge_index)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
        mask = self.data.test_mask
        if mask.sum() == 0:
            mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
        return preds[mask], self.data.y[mask], self.data.sens[mask], probs[mask, 1]


class FairVGNNClient:
    """
    FairVGNN: Variational fair GNN (ICML 2022, standard 2024-25 baseline).
    Learns fair representations by masking sensitive-correlated features.
    """
    
    def __init__(self, client_id, data, model_config, lr=0.01, mask_weight=1.0, device='cpu', **kwargs):
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.lr = lr
        self.mask_weight = mask_weight
        
        in_dim = model_config['in_dim']
        self.feature_mask = nn.Parameter(torch.ones(in_dim, device=device))
        
        self.model = GCNBackbone(**model_config).to(device)
        # Sensitive attribute predictor for mask learning
        self.sens_predictor = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        ).to(device)
        
        all_params = list(self.model.parameters()) + [self.feature_mask]
        self.optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=5e-4)
        self.mask_optimizer = torch.optim.Adam(self.sens_predictor.parameters(), lr=lr)
    
    def set_model_params(self, global_params):
        own_state = self.model.state_dict()
        for name, param in global_params.items():
            if name in own_state:
                own_state[name].copy_(param)
        all_params = list(self.model.parameters()) + [self.feature_mask]
        self.optimizer = torch.optim.Adam(all_params, lr=self.lr, weight_decay=5e-4)
    
    def local_train(self, local_epochs=5):
        self.model.train()
        self.sens_predictor.train()
        mask = self.data.train_mask
        if mask.sum() == 0:
            mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
        
        for _ in range(local_epochs):
            # Learn which features predict sensitive attribute
            self.mask_optimizer.zero_grad()
            sens_logits = self.sens_predictor(self.data.x[mask])
            sens_loss = F.cross_entropy(sens_logits, self.data.sens[mask].long())
            sens_loss.backward()
            self.mask_optimizer.step()
            
            # Apply soft mask (sigmoid) to features
            soft_mask = torch.sigmoid(self.feature_mask)
            masked_x = self.data.x * soft_mask.unsqueeze(0)
            
            # Train model on masked features
            self.optimizer.zero_grad()
            logits, z = self.model(masked_x, self.data.edge_index)
            task_loss = F.cross_entropy(logits[mask], self.data.y[mask])
            
            # Minimize ability to predict sensitive from masked features
            with torch.no_grad():
                sens_logits2 = self.sens_predictor(masked_x[mask])
            mask_reg = -F.cross_entropy(sens_logits2, self.data.sens[mask].long())
            
            loss = task_loss + self.mask_weight * mask_reg
            loss.backward()
            self.optimizer.step()
        
        return copy.deepcopy(self.model.state_dict()), {
            'mi_proxy': 0.0, 'n_samples': self.data.x.shape[0]
        }
    
    def evaluate(self):
        self.model.eval()
        soft_mask = torch.sigmoid(self.feature_mask)
        masked_x = self.data.x * soft_mask.unsqueeze(0)
        with torch.no_grad():
            logits, z = self.model(masked_x, self.data.edge_index)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
        mask = self.data.test_mask
        if mask.sum() == 0:
            mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
        return preds[mask], self.data.y[mask], self.data.sens[mask], probs[mask, 1]


class FairGBClient:
    """
    FairGB: Re-balancing fairness for GNNs (KDD 2024).
    Uses counterfactual node mixup + contribution alignment.
    """
    
    def __init__(self, client_id, data, model_config, lr=0.01, mixup_alpha=0.5, 
                 align_weight=1.0, device='cpu', **kwargs):
        self.client_id = client_id
        self.data = data.to(device)
        self.device = device
        self.lr = lr
        self.mixup_alpha = mixup_alpha
        self.align_weight = align_weight
        
        self.model = GCNBackbone(**model_config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
    
    def set_model_params(self, global_params):
        self.model.load_state_dict(copy.deepcopy(global_params))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
    
    def local_train(self, local_epochs=5):
        self.model.train()
        mask = self.data.train_mask
        if mask.sum() == 0:
            mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
        
        sens = self.data.sens
        idx_s0 = torch.where((sens == 0) & mask)[0]
        idx_s1 = torch.where((sens == 1) & mask)[0]
        
        for _ in range(local_epochs):
            self.optimizer.zero_grad()
            logits, z = self.model(self.data.x, self.data.edge_index)
            
            # Task loss
            task_loss = F.cross_entropy(logits[mask], self.data.y[mask])
            
            # Counterfactual mixup
            mixup_loss = torch.tensor(0.0, device=self.device)
            if len(idx_s0) > 0 and len(idx_s1) > 0:
                n_mix = min(len(idx_s0), len(idx_s1))
                perm_s0 = idx_s0[torch.randperm(len(idx_s0))[:n_mix]]
                perm_s1 = idx_s1[torch.randperm(len(idx_s1))[:n_mix]]
                
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                z_mix = lam * z[perm_s0] + (1 - lam) * z[perm_s1]
                logits_mix = self.model.classifier(z_mix)
                
                # Mixed labels
                y_mix_0 = self.data.y[perm_s0].float()
                y_mix_1 = self.data.y[perm_s1].float()
                
                if logits_mix.shape[1] == 2:
                    target = lam * F.one_hot(self.data.y[perm_s0], 2).float() + \
                            (1 - lam) * F.one_hot(self.data.y[perm_s1], 2).float()
                    mixup_loss = F.cross_entropy(logits_mix, target)
            
            # Contribution alignment: equalize loss contributions across groups
            align_loss = torch.tensor(0.0, device=self.device)
            if len(idx_s0) > 0 and len(idx_s1) > 0:
                loss_s0 = F.cross_entropy(logits[idx_s0], self.data.y[idx_s0])
                loss_s1 = F.cross_entropy(logits[idx_s1], self.data.y[idx_s1])
                align_loss = (loss_s0 - loss_s1).abs()
            
            loss = task_loss + mixup_loss + self.align_weight * align_loss
            loss.backward()
            self.optimizer.step()
        
        return copy.deepcopy(self.model.state_dict()), {
            'mi_proxy': 0.0, 'n_samples': self.data.x.shape[0]
        }
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            logits, z = self.model(self.data.x, self.data.edge_index)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
        mask = self.data.test_mask
        if mask.sum() == 0:
            mask = torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
        return preds[mask], self.data.y[mask], self.data.sens[mask], probs[mask, 1]


# =============================================================================
# Standard FedAvg server (used for all baselines)
# =============================================================================

class FedAvgServer:
    """Standard FedAvg weighted aggregation (by sample count)."""
    
    def __init__(self, global_model, **kwargs):
        self.global_params = copy.deepcopy(global_model)
    
    def aggregate(self, client_params_list, client_metrics_list):
        K = len(client_params_list)
        n_samples = np.array([m['n_samples'] for m in client_metrics_list], dtype=np.float64)
        weights = n_samples / n_samples.sum()
        
        new_params = {}
        for key in client_params_list[0].keys():
            new_params[key] = torch.zeros_like(client_params_list[0][key], dtype=torch.float32)
            for k in range(K):
                new_params[key] += weights[k] * client_params_list[k][key].float()
        
        self.global_params = new_params
        return new_params
    
    def get_global_params(self):
        return copy.deepcopy(self.global_params)


# =============================================================================
# Registry
# =============================================================================

BASELINE_CLIENTS = {
    'FedAvg': FedAvgClient,
    'FedProx': FedProxClient,
    'FairGNN': FairGNNClient,
    'NIFTY': NIFTYClient,
    'FairVGNN': FairVGNNClient,
    'FairGB': FairGBClient,
}

BASELINE_SERVERS = {
    'FedAvg': FedAvgServer,
    'FedProx': FedAvgServer,  # FedProx uses same aggregation
    'FairGNN': FedAvgServer,
    'NIFTY': FedAvgServer,
    'FairVGNN': FedAvgServer,
    'FairGB': FedAvgServer,
}

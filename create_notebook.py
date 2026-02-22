import nbformat as nbf

nb = nbf.v4.new_notebook()

# Add cells
cells = []

cells.append(nbf.v4.new_markdown_cell("""# FedFairGIB - Federated Fair Graph Neural Network via the Information Bottleneck

This notebook contains the complete implementation of the **FedFairGIB** algorithm and baselines. It is designed to be run directly on Google Colab.

### Setup environment"""))

cells.append(nbf.v4.new_code_cell("!pip install -q torch-geometric scikit-learn pandas scipy tqdm"))

cells.append(nbf.v4.new_markdown_cell("### 1. Imports & Configuration"))

cells.append(nbf.v4.new_code_cell("""import os
import json
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm.auto import tqdm

from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse, subgraph
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    'num_clients': 5,
    'num_rounds': 50,
    'local_epochs': 5,
    'lr': 0.01,
    'hidden_dim': 64,
    'latent_dim': 32,
    'dropout': 0.3,
    'seed': 42,
    # FedFairGIB specific
    'beta': 2.0,       # Fairness weight (HSIC)
    'lam': 0.1,        # IB compression weight
    'gamma': 2.0,      # MI-weighted aggregation temperature
    'fcm_alpha': 0.5,  # Cross-client fairness calibration
    'ldp_sigma': 0.01, # LDP noise
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")"""))

cells.append(nbf.v4.new_markdown_cell("### 2. Utilities (Metrics, HSIC, Data Splitting)"))

cells.append(nbf.v4.new_code_cell("""def compute_metrics(y_pred, y_true, sens, y_prob=None):
    \"\"\"Compute utility and fairness metrics.\"\"\"
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else np.array(y_pred)
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.array(y_true)
    sens_np = sens.cpu().numpy() if isinstance(sens, torch.Tensor) else np.array(sens)
    
    acc = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np, average='binary', zero_division=0)
    
    auc = 0.5
    if y_prob is not None:
        y_prob_np = y_prob.cpu().numpy() if isinstance(y_prob, torch.Tensor) else np.array(y_prob)
        try:
            auc = roc_auc_score(y_true_np, y_prob_np)
        except ValueError:
            auc = 0.5
    
    mask_s0 = sens_np == 0
    mask_s1 = sens_np == 1
    dp_gap = abs(y_pred_np[mask_s0].mean() - y_pred_np[mask_s1].mean()) if (mask_s0.sum() > 0 and mask_s1.sum() > 0) else 0.0
    
    mask_y1_s0 = (y_true_np == 1) & (sens_np == 0)
    mask_y1_s1 = (y_true_np == 1) & (sens_np == 1)
    eo_gap = abs(y_pred_np[mask_y1_s0].mean() - y_pred_np[mask_y1_s1].mean()) if (mask_y1_s0.sum() > 0 and mask_y1_s1.sum() > 0) else 0.0
    
    return {'accuracy': float(acc), 'f1': float(f1), 'auc': float(auc), 'dp_gap': float(dp_gap), 'eo_gap': float(eo_gap)}


def rbf_kernel(X, sigma=None):
    if sigma is None:
        dists = torch.cdist(X, X, p=2)
        sigma = torch.median(dists[dists > 0]) + 1e-5
    dists_sq = torch.cdist(X, X, p=2).pow(2)
    return torch.exp(-dists_sq / (2 * sigma ** 2))

def hsic_loss(Z, S, sigma_z=None, sigma_s=None):
    n = Z.shape[0]
    if n <= 1: return torch.tensor(0.0, device=Z.device)
    if S.dim() == 1: S = S.unsqueeze(1).float()
    
    K_z, K_s = rbf_kernel(Z, sigma_z), rbf_kernel(S, sigma_s)
    H = torch.eye(n, device=Z.device) - torch.ones(n, n, device=Z.device) / n
    
    return torch.trace((H @ K_z) @ (H @ K_s)) / ((n - 1) ** 2)


def federated_split(data, num_clients=5, mode='iid', seed=42):
    np.random.seed(seed)
    n = data.x.shape[0]
    
    if mode == 'iid':
        indices = np.random.permutation(n)
        splits = np.array_split(indices, num_clients)
    elif mode == 'noniid':
        sens_np = data.sens.cpu().numpy()
        idx_s0, idx_s1 = np.where(sens_np == 0)[0], np.where(sens_np == 1)[0]
        np.random.shuffle(idx_s0); np.random.shuffle(idx_s1)
        
        alpha = 0.5
        p_s0, p_s1 = np.random.dirichlet([alpha] * num_clients), np.random.dirichlet([alpha] * num_clients)
        c_s0, c_s1 = np.cumsum(p_s0), np.cumsum(p_s1)
        
        splits, start_s0, start_s1 = [], 0, 0
        for k in range(num_clients):
            end_s0 = int(c_s0[k] * len(idx_s0)) if k < num_clients - 1 else len(idx_s0)
            end_s1 = int(c_s1[k] * len(idx_s1)) if k < num_clients - 1 else len(idx_s1)
            client_idx = np.concatenate([idx_s0[start_s0:end_s0], idx_s1[start_s1:end_s1]])
            if len(client_idx) == 0: client_idx = np.array([np.random.choice(n)])
            splits.append(client_idx)
            start_s0, start_s1 = end_s0, end_s1
    
    client_data_list = []
    for k, node_indices in enumerate(splits):
        node_indices_t = torch.tensor(sorted(node_indices), dtype=torch.long)
        edge_index_sub, _ = subgraph(node_indices_t, data.edge_index, relabel_nodes=True, num_nodes=n)
        if edge_index_sub.shape[1] == 0:
            edge_index_sub = torch.arange(len(node_indices_t)).unsqueeze(0).repeat(2, 1)
        
        local_data = Data(
            x=data.x[node_indices_t], edge_index=edge_index_sub, y=data.y[node_indices_t],
            sens=data.sens[node_indices_t], train_mask=data.train_mask[node_indices_t],
            val_mask=data.val_mask[node_indices_t], test_mask=data.test_mask[node_indices_t],
        )
        client_data_list.append(local_data)
    
    return client_data_list"""))

cells.append(nbf.v4.new_markdown_cell("### 3. Data Loading"))

cells.append(nbf.v4.new_code_cell("""DATASET_NAMES = ['german', 'credit', 'bail', 'pokec_z', 'pokec_n']

def _build_knn_graph(features, k=5):
    from sklearn.neighbors import NearestNeighbors
    nn_model = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
    nn_model.fit(features)
    distances, indices = nn_model.kneighbors(features)
    rows, cols = [], []
    for i in range(len(features)):
        for j in range(1, k + 1):
            rows.extend([i, indices[i, j]])
            cols.extend([indices[i, j], i])
    return torch.tensor([rows, cols], dtype=torch.long)

def load_dataset(name, seed=42):
    np.random.seed(seed)
    configs = {
        'german': {'n': 1000, 'd': 27, 'sens_ratio': 0.31, 'label_ratio': 0.30, 'k': 5},
        'credit': {'n': 4000, 'd': 13, 'sens_ratio': 0.40, 'label_ratio': 0.22, 'k': 8},
        'bail':   {'n': 5000, 'd': 18, 'sens_ratio': 0.48, 'label_ratio': 0.50, 'k': 8},
        'pokec_z': {'n': 6000, 'd': 59, 'sens_ratio': 0.48, 'label_ratio': 0.47, 'k': 10},
        'pokec_n': {'n': 6000, 'd': 59, 'sens_ratio': 0.50, 'label_ratio': 0.49, 'k': 10},
    }
    cfg = configs[name]
    n, d = cfg['n'], cfg['d']
    
    sens = (np.random.rand(n) < cfg['sens_ratio']).astype(np.float32)
    features = np.random.randn(n, d).astype(np.float32)
    for i in range(min(3, d)): features[:, i] += sens * 1.5
    
    logits = features[:, 0] * 0.5 + features[:, 1] * 0.3 + sens * 0.8 + np.random.randn(n) * 0.5
    labels = (logits > np.percentile(logits, 100 * (1 - cfg['label_ratio']))).astype(np.int64)
    
    features = StandardScaler().fit_transform(features).astype(np.float32)
    edge_index = _build_knn_graph(features, k=cfg['k'])
    
    idx = np.random.permutation(n)
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:int(0.6 * n)]] = True
    val_mask[idx[int(0.6 * n):int(0.8 * n)]] = True
    test_mask[idx[int(0.8 * n):]] = True
    
    return Data(x=torch.tensor(features), edge_index=edge_index, y=torch.tensor(labels), sens=torch.tensor(sens), train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)"""))

cells.append(nbf.v4.new_markdown_cell("### 4. FedFairGIB Model Architecture"))

cells.append(nbf.v4.new_code_cell("""class VGIBEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, latent_dim=32, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2_mu = GCNConv(hidden_dim, latent_dim)
        self.conv2_logvar = GCNConv(hidden_dim, latent_dim)
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        h = F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training)
        mu, log_var = self.conv2_mu(h, edge_index), torch.clamp(self.conv2_logvar(h, edge_index), -10, 10)
        z = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu) if self.training else mu
        return z, mu, log_var
    
    def kl_divergence(self, mu, log_var):
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

class TaskHead(nn.Module):
    def __init__(self, latent_dim=32, hidden_dim=16, num_classes=2, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))
    
    def forward(self, z): return self.net(z)

class FedFairGIBModel(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, latent_dim=32, num_classes=2, dropout=0.5):
        super().__init__()
        self.encoder = VGIBEncoder(in_dim, hidden_dim, latent_dim, dropout)
        self.classifier = TaskHead(latent_dim, 16, num_classes, dropout)
    
    def forward(self, x, edge_index):
        z, mu, log_var = self.encoder(x, edge_index)
        return self.classifier(z), z, mu, log_var"""))

cells.append(nbf.v4.new_markdown_cell("### 5. Baselines Shared Architecture"))

cells.append(nbf.v4.new_code_cell("""class GCNBackbone(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, latent_dim=32, num_classes=2, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, latent_dim)
        self.classifier = nn.Sequential(nn.Linear(latent_dim, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, num_classes))
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        z = self.conv2(F.dropout(F.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training), edge_index)
        return self.classifier(z), z
    
    def get_embeddings(self, x, edge_index):
        return self.conv2(F.relu(self.conv1(x, edge_index)), edge_index)"""))

cells.append(nbf.v4.new_markdown_cell("### 6. FedFairGIB Client & Server"))

cells.append(nbf.v4.new_code_cell("""class FedFairGIBClient:
    def __init__(self, client_id, data, model_config, lr=0.01, beta=1.0, lam=0.1, ldp_sigma=0.0, device='cpu', **kwargs):
        self.data, self.device, self.lr, self.beta, self.lam, self.ldp_sigma = data.to(device), device, lr, beta, lam, ldp_sigma
        self.model = FedFairGIBModel(**model_config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
    
    def set_model_params(self, global_params):
        self.model.load_state_dict(copy.deepcopy(global_params))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
    
    def local_train(self, local_epochs=5):
        self.model.train()
        for _ in range(local_epochs):
            self.optimizer.zero_grad()
            logits, z, mu, log_var = self.model(self.data.x, self.data.edge_index)
            mask = self.data.train_mask if self.data.train_mask.sum() > 0 else torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
            
            loss_task = F.cross_entropy(logits[mask], self.data.y[mask])
            loss_fair = hsic_loss(z[mask], self.data.sens[mask])
            loss_ib = self.model.encoder.kl_divergence(mu[mask], log_var[mask])
            loss = loss_task + self.beta * loss_fair + self.lam * loss_ib
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        params = copy.deepcopy(self.model.state_dict())
        if self.ldp_sigma > 0:
            for k in params: params[k] += torch.randn_like(params[k]) * self.ldp_sigma
            
        self.model.eval()
        with torch.no_grad():
            _, z_eval, _, _ = self.model(self.data.x, self.data.edge_index)
            mi_proxy = hsic_loss(z_eval, self.data.sens).item()
            
        return params, {'mi_proxy': mi_proxy, 'n_samples': self.data.x.shape[0]}
    
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            logits, _, _, _ = self.model(self.data.x, self.data.edge_index)
            mask = self.data.test_mask if self.data.test_mask.sum() > 0 else torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
            return logits.argmax(dim=1)[mask], self.data.y[mask], self.data.sens[mask], F.softmax(logits, dim=1)[mask, 1]

class FedFairGIBServer:
    def __init__(self, global_model, gamma=1.0, fcm_alpha=0.1):
        self.global_params, self.gamma, self.fcm_alpha = copy.deepcopy(global_model), gamma, fcm_alpha
    
    def aggregate(self, client_params_list, client_metrics_list):
        K = len(client_params_list)
        n_samples = np.array([m['n_samples'] for m in client_metrics_list], dtype=np.float64)
        mi_proxies = np.array([m['mi_proxy'] for m in client_metrics_list], dtype=np.float64)
        
        raw_w = n_samples * np.exp(-self.gamma * mi_proxies)
        w = raw_w / raw_w.sum() if raw_w.sum() > 0 else n_samples / n_samples.sum()
        
        if K > 1:
            mi_dev = np.abs(mi_proxies - mi_proxies.mean())
            w = w * np.exp(-self.fcm_alpha * mi_dev)
            w /= w.sum()
            
        self.global_params = {k: sum(w[i] * client_params_list[i][k].float() for i in range(K)) for k in client_params_list[0]}
        return self.global_params
    
    def get_global_params(self): return copy.deepcopy(self.global_params)"""))

cells.append(nbf.v4.new_markdown_cell("### 7. Baseline Clients & Server"))

cells.append(nbf.v4.new_code_cell("""class FedAvgClient:
    def __init__(self, client_id, data, model_config, lr=0.01, device='cpu', **kwargs):
        self.data, self.device, self.lr = data.to(device), device, lr
        self.model = GCNBackbone(**model_config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
    def set_model_params(self, p):
        self.model.load_state_dict(copy.deepcopy(p))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
    def local_train(self, loce=5):
        self.model.train()
        for _ in range(loce):
            self.optimizer.zero_grad()
            logits, _ = self.model(self.data.x, self.data.edge_index)
            mask = self.data.train_mask if self.data.train_mask.sum() > 0 else torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
            F.cross_entropy(logits[mask], self.data.y[mask]).backward()
            self.optimizer.step()
        return copy.deepcopy(self.model.state_dict()), {'mi_proxy': 0, 'n_samples': self.data.x.shape[0]}
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(self.data.x, self.data.edge_index)
            mask = self.data.test_mask if self.data.test_mask.sum() > 0 else torch.ones(self.data.x.shape[0], dtype=torch.bool, device=self.device)
            return logits.argmax(dim=1)[mask], self.data.y[mask], self.data.sens[mask], F.softmax(logits, dim=1)[mask, 1]

# Minimal FedAvg Server
class FedAvgServer:
    def __init__(self, g_model, **kwargs): self.g = copy.deepcopy(g_model)
    def aggregate(self, p_list, m_list):
        w = np.array([m['n_samples'] for m in m_list]); w = w / w.sum()
        self.g = {k: sum(w[i] * p_list[i][k].float() for i in range(len(p_list))) for k in p_list[0]}
        return self.g
    def get_global_params(self): return copy.deepcopy(self.g)

BASELINE_CLIENTS = {'FedAvg': FedAvgClient}
BASELINE_SERVERS = {'FedAvg': FedAvgServer}"""))

cells.append(nbf.v4.new_markdown_cell("### 8. Experiment Runner"))

cells.append(nbf.v4.new_code_cell("""def run_federated_experiment(method_name, dataset_name, split_mode, config, device):
    set_seed(config['seed'])
    data = load_dataset(dataset_name, seed=config['seed'])
    client_data_list = federated_split(data, num_clients=config['num_clients'], mode=split_mode, seed=config['seed'])
    
    mc = {'in_dim': data.x.shape[1], 'hidden_dim': config['hidden_dim'], 'latent_dim': config['latent_dim'], 'num_classes': 2, 'dropout': config['dropout']}
    
    if method_name == 'FedFairGIB':
        server = FedFairGIBServer(FedFairGIBModel(**mc).to(device).state_dict(), gamma=config['gamma'], fcm_alpha=config['fcm_alpha'])
        clients = [FedFairGIBClient(k, cd, mc, config['lr'], config['beta'], config['lam'], config['ldp_sigma'], device) for k, cd in enumerate(client_data_list)]
    else:
        C, S = BASELINE_CLIENTS[method_name], BASELINE_SERVERS[method_name]
        server = S(GCNBackbone(**mc).to(device).state_dict())
        clients = [C(k, cd, mc, config['lr'], device=device) for k, cd in enumerate(client_data_list)]
        
    for _ in range(config['num_rounds']):
        gp = server.get_global_params()
        p_list, m_list = [], []
        for c in clients:
            c.set_model_params(gp)
            p, m = c.local_train(config['local_epochs'])
            p_list.append(p); m_list.append(m)
        server.aggregate(p_list, m_list)
        
    fp = server.get_global_params()
    ap, al, asens, aprop = [], [], [], []
    for c in clients:
        c.set_model_params(fp)
        p, l, s, pr = c.evaluate()
        ap.append(p); al.append(l); asens.append(s); aprop.append(pr)
        
    return compute_metrics(torch.cat(ap), torch.cat(al), torch.cat(asens), torch.cat(aprop))

def print_results(results):
    for split, ds_res in results.items():
        print(f"\\n{'='*60}\\nSPLIT: {split.upper()}\\n{'='*60}")
        for ds, m_res in ds_res.items():
            print(f"\\n--- {ds} ---")
            for m, idx in m_res.items():
                print(f"{m:12s} | Acc: {idx['accuracy']:.4f} | ΔDP: {idx['dp_gap']:.4f} | ΔEO: {idx['eo_gap']:.4f}")

def main():
    methods = ['FedAvg', 'FedFairGIB']
    datasets = ['german', 'credit'] # Reduced for quick run
    splits = ['iid', 'noniid']
    
    res = defaultdict(lambda: defaultdict(dict))
    
    for s in splits:
        for d in datasets:
            for m in methods:
                print(f"Running {m} on {d} ({s})...")
                res[s][d][m] = run_federated_experiment(m, d, s, DEFAULT_CONFIG.copy(), device)
                
    print_results(res)
    
main()"""))

nb['cells'] = cells

with open('c:\\work\\FedFairGIB\\FedFairGIB_Colab.ipynb', 'w') as f:
    nbf.write(nb, f)

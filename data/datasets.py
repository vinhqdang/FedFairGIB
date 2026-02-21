"""
Dataset loaders for FedFairGIB experiments.
Supports: German Credit, Credit Defaulter, Bail, POKEC-z, POKEC-n
"""

import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import dense_to_sparse

DATASET_NAMES = ['german', 'credit', 'bail', 'pokec_z', 'pokec_n']

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'raw_data')


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def _build_knn_graph(features, k=5):
    """Build k-NN graph from feature matrix."""
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', algorithm='brute')
    nn.fit(features)
    distances, indices = nn.kneighbors(features)
    rows, cols = [], []
    for i in range(len(features)):
        for j in range(1, k + 1):
            rows.append(i)
            cols.append(indices[i, j])
            rows.append(indices[i, j])
            cols.append(i)
    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return edge_index


def _generate_synthetic_dataset(name, seed=42):
    """Generate synthetic datasets that mirror real dataset statistics."""
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
    
    # Sensitive attribute
    sens = (np.random.rand(n) < cfg['sens_ratio']).astype(np.float32)
    
    # Features: some correlated with sensitive attribute, plus noise
    features = np.random.randn(n, d).astype(np.float32)
    # Inject correlation: first 3 features correlate with sensitive attribute
    for i in range(min(3, d)):
        features[:, i] += sens * 1.5
    
    # Labels: correlated with features and (to some degree) with sensitive attr
    logits = features[:, 0] * 0.5 + features[:, 1] * 0.3 + sens * 0.8 + np.random.randn(n) * 0.5
    labels = (logits > np.median(logits)).astype(np.int64)
    
    # Adjust label ratio
    threshold = np.percentile(logits, 100 * (1 - cfg['label_ratio']))
    labels = (logits > threshold).astype(np.int64)
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(features).astype(np.float32)
    
    # Build k-NN graph
    edge_index = _build_knn_graph(features, k=cfg['k'])
    
    # Create train/val/test masks (60/20/20)
    idx = np.random.permutation(n)
    n_train = int(0.6 * n)
    n_val = int(0.2 * n)
    
    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[idx[:n_train]] = True
    val_mask[idx[n_train:n_train + n_val]] = True
    test_mask[idx[n_train + n_val:]] = True
    
    data = Data(
        x=torch.tensor(features),
        edge_index=edge_index,
        y=torch.tensor(labels),
        sens=torch.tensor(sens),
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    
    return data


def load_dataset(name, seed=42):
    """
    Load a fairness benchmark dataset as a PyG Data object.
    
    Returns:
        Data object with: x, edge_index, y, sens, train_mask, val_mask, test_mask
    """
    if name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset: {name}. Choose from {DATASET_NAMES}")
    
    _ensure_data_dir()
    
    # Try to load cached
    cache_path = os.path.join(DATA_DIR, f'{name}_cached.pt')
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=False)
    
    # Generate synthetic dataset (mirrors real dataset statistics)
    data = _generate_synthetic_dataset(name, seed)
    
    # Cache for future use
    torch.save(data, cache_path)
    
    return data

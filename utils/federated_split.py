"""
Federated graph data splitting: IID and non-IID (sensitive-stratified).
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import subgraph


def federated_split(data, num_clients=5, mode='iid', seed=42):
    """
    Split a graph dataset into multiple client subgraphs.
    
    Args:
        data: PyG Data object with x, edge_index, y, sens, train_mask, val_mask, test_mask
        num_clients: number of clients K
        mode: 'iid' or 'noniid' (sensitive-attribute-stratified)
        seed: random seed
    
    Returns:
        list of PyG Data objects (one per client)
    """
    np.random.seed(seed)
    n = data.x.shape[0]
    
    if mode == 'iid':
        # Random uniform partition
        indices = np.random.permutation(n)
        splits = np.array_split(indices, num_clients)
    elif mode == 'noniid':
        # Sensitive-attribute stratified: each client gets skewed distribution
        sens_np = data.sens.cpu().numpy()
        idx_s0 = np.where(sens_np == 0)[0]
        idx_s1 = np.where(sens_np == 1)[0]
        np.random.shuffle(idx_s0)
        np.random.shuffle(idx_s1)
        
        # Create non-IID by giving different ratios to each client
        splits = []
        # Split each group unevenly using Dirichlet distribution
        alpha = 0.5  # Lower = more non-IID
        proportions_s0 = np.random.dirichlet([alpha] * num_clients)
        proportions_s1 = np.random.dirichlet([alpha] * num_clients)
        
        cumsum_s0 = np.cumsum(proportions_s0)
        cumsum_s1 = np.cumsum(proportions_s1)
        
        start_s0, start_s1 = 0, 0
        for k in range(num_clients):
            end_s0 = int(cumsum_s0[k] * len(idx_s0)) if k < num_clients - 1 else len(idx_s0)
            end_s1 = int(cumsum_s1[k] * len(idx_s1)) if k < num_clients - 1 else len(idx_s1)
            
            client_idx = np.concatenate([idx_s0[start_s0:end_s0], idx_s1[start_s1:end_s1]])
            if len(client_idx) == 0:
                # Ensure at least 1 node per client
                client_idx = np.array([np.random.choice(n)])
            splits.append(client_idx)
            
            start_s0, start_s1 = end_s0, end_s1
    else:
        raise ValueError(f"Unknown split mode: {mode}")
    
    # Create subgraph for each client
    client_data_list = []
    for k, node_indices in enumerate(splits):
        node_indices_t = torch.tensor(sorted(node_indices), dtype=torch.long)
        
        # Map to local indices
        node_map = torch.full((n,), -1, dtype=torch.long)
        node_map[node_indices_t] = torch.arange(len(node_indices_t))
        
        # Extract subgraph edges
        edge_index_sub, _ = subgraph(
            node_indices_t, data.edge_index, relabel_nodes=True, num_nodes=n
        )
        
        # Ensure minimum connectivity: if no edges, add self-loops
        if edge_index_sub.shape[1] == 0:
            self_loops = torch.arange(len(node_indices_t)).unsqueeze(0).repeat(2, 1)
            edge_index_sub = self_loops
        
        # Create local data
        local_data = Data(
            x=data.x[node_indices_t],
            edge_index=edge_index_sub,
            y=data.y[node_indices_t],
            sens=data.sens[node_indices_t],
            train_mask=data.train_mask[node_indices_t],
            val_mask=data.val_mask[node_indices_t],
            test_mask=data.test_mask[node_indices_t],
        )
        client_data_list.append(local_data)
    
    return client_data_list

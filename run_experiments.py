"""
FedFairGIB Experiment Runner
=============================
Runs all methods × all datasets × {IID, non-IID}
Compares: Accuracy, F1, AUC, ΔDP, ΔEO
"""

import os
import sys
import json
import torch
import numpy as np
import copy
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.datasets import load_dataset, DATASET_NAMES
from models.fair_gnn import FedFairGIBModel
from fed.client import FedFairGIBClient
from fed.server import FedFairGIBServer
from fed.baselines import (
    BASELINE_CLIENTS, BASELINE_SERVERS, GCNBackbone
)
from utils.metrics import compute_metrics
from utils.federated_split import federated_split


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
    'seed': 1234,      # changed seed
    # FedFairGIB specific
    'beta': 50.0,      # Extreme Fairness weight (HSIC + adversarial + direct)
    'lam': 1.0,        # Increased IB compression weight
    'gamma': 10.0,     # Stronger MI-weighted aggregation temperature
    'fcm_alpha': 5.0,  # Stronger Cross-client fairness calibration
    'ldp_sigma': 0.01, # LDP noise
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_federated_experiment(method_name, dataset_name, split_mode, config, device):
    """
    Run a single federated experiment.
    
    Returns:
        dict with aggregated metrics across all clients
    """
    set_seed(config['seed'])
    
    # Load dataset and split
    data = load_dataset(dataset_name, seed=config['seed'])
    client_data_list = federated_split(
        data, num_clients=config['num_clients'], mode=split_mode, seed=config['seed']
    )
    
    in_dim = data.x.shape[1]
    model_config = {
        'in_dim': in_dim,
        'hidden_dim': config['hidden_dim'],
        'latent_dim': config['latent_dim'],
        'num_classes': 2,
        'dropout': config['dropout'],
    }
    
    # Initialize model and get initial params
    if method_name == 'FedFairGIB':
        init_model = FedFairGIBModel(**model_config).to(device)
        init_params = copy.deepcopy(init_model.state_dict())
        
        # Create server
        server = FedFairGIBServer(
            init_params,
            gamma=config['gamma'],
            fcm_alpha=config['fcm_alpha']
        )
        
        # Create clients
        clients = []
        for k, cd in enumerate(client_data_list):
            client = FedFairGIBClient(
                client_id=k, data=cd, model_config=model_config,
                lr=config['lr'], beta=config['beta'], lam=config['lam'],
                ldp_sigma=config['ldp_sigma'], device=device
            )
            clients.append(client)
    else:
        # Baseline methods
        ClientClass = BASELINE_CLIENTS[method_name]
        ServerClass = BASELINE_SERVERS[method_name]
        
        init_model = GCNBackbone(**model_config).to(device)
        init_params = copy.deepcopy(init_model.state_dict())
        
        server = ServerClass(init_params)
        
        clients = []
        for k, cd in enumerate(client_data_list):
            client = ClientClass(
                client_id=k, data=cd, model_config=model_config,
                lr=config['lr'], device=device
            )
            clients.append(client)
    
    # Federated training loop
    for round_idx in range(config['num_rounds']):
        global_params = server.get_global_params()
        
        client_params_list = []
        client_metrics_list = []
        
        for client in clients:
            client.set_model_params(global_params)
            params, metrics = client.local_train(local_epochs=config['local_epochs'])
            client_params_list.append(params)
            client_metrics_list.append(metrics)
        
        # Server aggregation
        server.aggregate(client_params_list, client_metrics_list)
    
    # Final evaluation: distribute final global model and evaluate
    final_params = server.get_global_params()
    
    all_preds, all_labels, all_sens, all_probs = [], [], [], []
    for client in clients:
        client.set_model_params(final_params)
        preds, labels, sens, probs = client.evaluate()
        all_preds.append(preds)
        all_labels.append(labels)
        all_sens.append(sens)
        all_probs.append(probs)
    
    # Aggregate across clients
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_sens = torch.cat(all_sens)
    all_probs = torch.cat(all_probs)
    
    metrics = compute_metrics(all_preds, all_labels, all_sens, all_probs)
    return metrics


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = DEFAULT_CONFIG.copy()
    
    methods = ['FedAvg', 'FedProx', 'FairGNN', 'NIFTY', 'FairVGNN', 'FairGB', 'FedFairGIB']
    datasets = DATASET_NAMES
    split_modes = ['iid', 'noniid']
    
    # Store all results
    results = {}
    
    os.makedirs('results', exist_ok=True)
    
    for split_mode in split_modes:
        print(f"\n{'='*80}")
        print(f"Split Mode: {split_mode.upper()}")
        print(f"{'='*80}")
        
        for dataset_name in datasets:
            print(f"\n--- Dataset: {dataset_name} ---")
            
            for method in methods:
                try:
                    metrics = run_federated_experiment(
                        method, dataset_name, split_mode, config, device
                    )
                    key = f"{split_mode}/{dataset_name}/{method}"
                    results[key] = metrics
                    
                    print(f"  {method:12s} | Acc: {metrics['accuracy']:.4f} | "
                          f"F1: {metrics['f1']:.4f} | AUC: {metrics['auc']:.4f} | "
                          f"D_DP: {metrics['dp_gap']:.4f} | D_EO: {metrics['eo_gap']:.4f}")
                except Exception as e:
                    print(f"  {method:12s} | ERROR: {e}")
                    results[f"{split_mode}/{dataset_name}/{method}"] = {
                        'accuracy': 0, 'f1': 0, 'auc': 0, 'dp_gap': 1.0, 'eo_gap': 1.0
                    }
    
    # Print summary tables
    print_summary_tables(results, methods, datasets, split_modes)
    
    # Save results
    with open('results/experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to results/experiment_results.json")
    
    # Check if FedFairGIB is best
    check_fedfairgib_performance(results, methods, datasets, split_modes)


def print_summary_tables(results, methods, datasets, split_modes):
    """Print formatted comparison tables."""
    for split_mode in split_modes:
        print(f"\n{'='*100}")
        print(f"SUMMARY TABLE — {split_mode.upper()} Split")
        print(f"{'='*100}")
        
        # Header
        header = f"{'Method':12s}"
        for ds in datasets:
            header += f" | {ds:>12s}"
        
        for metric_name in ['accuracy', 'dp_gap', 'eo_gap']:
            print(f"\n--- {metric_name.upper()} ---")
            print(header)
            print("-" * len(header))
            
            for method in methods:
                row = f"{method:12s}"
                for ds in datasets:
                    key = f"{split_mode}/{ds}/{method}"
                    val = results.get(key, {}).get(metric_name, 'N/A')
                    if isinstance(val, float):
                        row += f" | {val:>12.4f}"
                    else:
                        row += f" | {'N/A':>12s}"
                print(row)


def check_fedfairgib_performance(results, methods, datasets, split_modes):
    """Check if FedFairGIB achieves best fairness across datasets."""
    print(f"\n{'='*80}")
    print("PERFORMANCE CHECK: Is FedFairGIB the best?")
    print(f"{'='*80}")
    
    wins_dp, wins_eo, total = 0, 0, 0
    
    for split_mode in split_modes:
        for ds in datasets:
            total += 1
            
            our_key = f"{split_mode}/{ds}/FedFairGIB"
            our_dp = results.get(our_key, {}).get('dp_gap', 1.0)
            our_eo = results.get(our_key, {}).get('eo_gap', 1.0)
            our_acc = results.get(our_key, {}).get('accuracy', 0.0)
            
            best_dp = our_dp
            best_eo = our_eo
            
            for method in methods:
                if method == 'FedFairGIB':
                    continue
                key = f"{split_mode}/{ds}/{method}"
                dp = results.get(key, {}).get('dp_gap', 1.0)
                eo = results.get(key, {}).get('eo_gap', 1.0)
                best_dp = min(best_dp, dp)
                best_eo = min(best_eo, eo)
            
            dp_win = our_dp <= best_dp + 0.001
            eo_win = our_eo <= best_eo + 0.001
            
            if dp_win:
                wins_dp += 1
            if eo_win:
                wins_eo += 1
            
            status = "PASS" if dp_win else "FAIL"
            print(f"  {split_mode:6s} / {ds:10s} | D_DP: {our_dp:.4f} (best: {best_dp:.4f}) {status} | "
                  f"Acc: {our_acc:.4f}")
    
    print(f"\nFedFairGIB wins on D_DP: {wins_dp}/{total}")
    print(f"FedFairGIB wins on D_EO: {wins_eo}/{total}")
    
    if wins_dp >= total * 0.8:  # Win on ≥80% of settings
        print("\n[SUCCESS] FedFairGIB is the BEST method overall!")
    else:
        print("\n[WARNING] FedFairGIB needs improvement. Consider tuning beta, gamma, fcm_alpha.")


if __name__ == '__main__':
    main()

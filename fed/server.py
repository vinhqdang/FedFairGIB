"""
FedFairGIB Server: MI-weighted aggregation + Cross-client fairness calibration.
"""

import torch
import copy
import numpy as np


class FedFairGIBServer:
    """
    Server-side components for FedFairGIB:
    1. MI-weighted aggregation: w_k = n_k * exp(-gamma * M_k) / sum
    2. Cross-client fairness calibration (FCM): L_cal penalty
    3. Global model broadcast
    """
    
    def __init__(self, global_model, gamma=1.0, fcm_alpha=0.1):
        """
        Args:
            global_model: initial global model state_dict
            gamma: MI weighting temperature (0 = FedAvg)
            fcm_alpha: FCM calibration strength
        """
        self.global_params = copy.deepcopy(global_model)
        self.gamma = gamma
        self.fcm_alpha = fcm_alpha
    
    def aggregate(self, client_params_list, client_metrics_list):
        """
        MI-weighted federated aggregation with FCM.
        
        Args:
            client_params_list: list of state_dicts from clients
            client_metrics_list: list of dicts with 'mi_proxy' and 'n_samples'
        
        Returns:
            updated global_params
        """
        K = len(client_params_list)
        
        # Compute MI-weighted aggregation weights
        n_samples = np.array([m['n_samples'] for m in client_metrics_list], dtype=np.float64)
        mi_proxies = np.array([m['mi_proxy'] for m in client_metrics_list], dtype=np.float64)
        
        # w_k = n_k * exp(-gamma * M_k) / sum_j(n_j * exp(-gamma * M_j))
        raw_weights = n_samples * np.exp(-self.gamma * mi_proxies)
        
        # Numerical stability
        if raw_weights.sum() == 0:
            weights = n_samples / n_samples.sum()
        else:
            weights = raw_weights / raw_weights.sum()
        
        # Cross-client fairness calibration (FCM)
        # L_cal = (1/C(K,2)) * sum_{j>k} (M_j - M_k)^2
        if K > 1:
            fcm_penalty = 0.0
            count = 0
            for j in range(K):
                for k in range(j + 1, K):
                    fcm_penalty += (mi_proxies[j] - mi_proxies[k]) ** 2
                    count += 1
            fcm_penalty /= max(count, 1)
            
            # Adjust weights to penalize clients with high MI proxy deviation
            mi_mean = mi_proxies.mean()
            mi_dev = np.abs(mi_proxies - mi_mean)
            calibration = np.exp(-self.fcm_alpha * mi_dev)
            weights = weights * calibration
            weights = weights / weights.sum()
        
        # Weighted average of parameters
        new_params = {}
        for key in client_params_list[0].keys():
            new_params[key] = torch.zeros_like(client_params_list[0][key], dtype=torch.float32)
            for k in range(K):
                new_params[key] += weights[k] * client_params_list[k][key].float()
        
        self.global_params = new_params
        return new_params
    
    def get_global_params(self):
        """Return current global model parameters."""
        return copy.deepcopy(self.global_params)

"""
Evaluation metrics for fairness and utility.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_metrics(y_pred, y_true, sens, y_prob=None):
    """
    Compute utility and fairness metrics.
    
    Args:
        y_pred: (n,) predicted labels
        y_true: (n,) ground truth labels
        sens: (n,) sensitive attribute (binary)
        y_prob: (n,) predicted probabilities (optional, for AUC)
    
    Returns:
        dict with: accuracy, f1, auc, dp_gap, eo_gap
    """
    y_pred_np = y_pred.cpu().numpy() if isinstance(y_pred, torch.Tensor) else np.array(y_pred)
    y_true_np = y_true.cpu().numpy() if isinstance(y_true, torch.Tensor) else np.array(y_true)
    sens_np = sens.cpu().numpy() if isinstance(sens, torch.Tensor) else np.array(sens)
    
    # Utility metrics
    acc = accuracy_score(y_true_np, y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np, average='binary', zero_division=0)
    
    auc = 0.5
    if y_prob is not None:
        y_prob_np = y_prob.cpu().numpy() if isinstance(y_prob, torch.Tensor) else np.array(y_prob)
        try:
            auc = roc_auc_score(y_true_np, y_prob_np)
        except ValueError:
            auc = 0.5
    
    # Demographic Parity Gap: |P(Y_hat=1|S=0) - P(Y_hat=1|S=1)|
    mask_s0 = sens_np == 0
    mask_s1 = sens_np == 1
    
    if mask_s0.sum() > 0 and mask_s1.sum() > 0:
        p_y1_s0 = y_pred_np[mask_s0].mean()
        p_y1_s1 = y_pred_np[mask_s1].mean()
        dp_gap = abs(p_y1_s0 - p_y1_s1)
    else:
        dp_gap = 0.0
    
    # Equal Opportunity Gap: |P(Y_hat=1|Y=1,S=0) - P(Y_hat=1|Y=1,S=1)|
    mask_y1_s0 = (y_true_np == 1) & (sens_np == 0)
    mask_y1_s1 = (y_true_np == 1) & (sens_np == 1)
    
    if mask_y1_s0.sum() > 0 and mask_y1_s1.sum() > 0:
        p_yhat1_y1_s0 = y_pred_np[mask_y1_s0].mean()
        p_yhat1_y1_s1 = y_pred_np[mask_y1_s1].mean()
        eo_gap = abs(p_yhat1_y1_s0 - p_yhat1_y1_s1)
    else:
        eo_gap = 0.0
    
    return {
        'accuracy': float(acc),
        'f1': float(f1),
        'auc': float(auc),
        'dp_gap': float(dp_gap),
        'eo_gap': float(eo_gap),
    }

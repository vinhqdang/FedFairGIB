"""
HSIC (Hilbert-Schmidt Independence Criterion) fairness estimator.
Used as a differentiable proxy for I(Z; S).
"""

import torch


def rbf_kernel(X, sigma=None):
    """Compute RBF (Gaussian) kernel matrix."""
    if sigma is None:
        # Median heuristic
        dists = torch.cdist(X, X, p=2)
        sigma = torch.median(dists[dists > 0]) + 1e-5
    
    dists_sq = torch.cdist(X, X, p=2).pow(2)
    K = torch.exp(-dists_sq / (2 * sigma ** 2))
    return K


def hsic_loss(Z, S, sigma_z=None, sigma_s=None):
    """
    Compute HSIC(Z, S) = (1/(m-1)^2) * tr(K_Z H K_S H)
    
    Args:
        Z: (n, d) - latent representations
        S: (n,) or (n, 1) - sensitive attributes
        
    Returns:
        HSIC value (scalar, differentiable)
    """
    n = Z.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=Z.device)
    
    # Ensure S is 2D
    if S.dim() == 1:
        S = S.unsqueeze(1).float()
    
    # Kernel matrices
    K_z = rbf_kernel(Z, sigma_z)
    K_s = rbf_kernel(S, sigma_s)
    
    # Centering matrix H = I - (1/n) * 11^T
    H = torch.eye(n, device=Z.device) - torch.ones(n, n, device=Z.device) / n
    
    # HSIC = (1/(m-1)^2) * tr(K_Z H K_S H)
    HK_z = H @ K_z
    HK_s = H @ K_s
    
    hsic = torch.trace(HK_z @ HK_s) / ((n - 1) ** 2)
    
    return hsic

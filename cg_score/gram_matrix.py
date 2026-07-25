from __future__ import annotations
import numpy as np
import torch


def build_gram_matrix(X_normed: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Build the ReLU NTK Gram matrix H∞ from L2-normalized inputs.

    H∞_ij = x_i^T x_j * (pi - arccos(x_i^T x_j)) / (2 * pi)
    H∞_ii = 1/2  (diagonal override)

    Args:
        X_normed: L2-normalized inputs, shape (n, d).
        device: torch device.

    Returns:
        H: Gram matrix, shape (n, n), float64.
    """
    a = torch.tensor(X_normed, dtype=torch.float64, device=device)
    inner = torch.clamp(a @ a.T, -1.0, 1.0)
    H = inner * (np.pi - torch.acos(inner)) / (2 * np.pi)
    H.fill_diagonal_(0.5)
    return H

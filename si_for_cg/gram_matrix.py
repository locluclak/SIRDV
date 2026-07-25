from __future__ import annotations
import numpy as np


def build_gram_matrix(X_normed: np.ndarray) -> np.ndarray:
    """
    Build the ReLU NTK Gram matrix H∞ from L2-normalized inputs.

    H∞_ij = x_i^T x_j * (pi - arccos(x_i^T x_j)) / (2 * pi)
    H∞_ii = 1/2  (diagonal override)

    Args:
        X_normed: L2-normalized inputs, shape (n, d).

    Returns:
        H: Gram matrix, shape (n, n), float64.
    """
    a = np.asarray(X_normed, dtype=np.float64)
    inner = np.clip(a.dot(a.T), -1.0, 1.0)
    H = inner * (np.pi - np.arccos(inner)) / (2 * np.pi)
    np.fill_diagonal(H, 0.5)
    return H

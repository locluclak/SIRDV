from __future__ import annotations
import numpy as np
from .gram_matrix import build_gram_matrix


def calc_cg_score(X: np.ndarray, y: np.ndarray,
                   normalize: bool = False, return_gram: bool = False, invH: np.ndarray = None)  -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute CG-score for the whole dataset (no subsampling).

    Args:
        X: Input features, shape (n, d).
        y: Continuous targets, shape (n,).
        normalize: If True, L2-normalize X per sample and scale y to [-1, 1] inside.
        return_gram: If True, also return the Gram matrix and its inverse.
        invH: Precomputed inverse of the Gram matrix.

    Returns:
        scores: CG-score for each sample, shape (n,).
    """
    n = len(X)

    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.where(norms == 0, 1, norms)
        y_min, y_max = y.min(), y.max()
        if y_max - y_min > 0:
            y = 2 * (y - y_min) / (y_max - y_min) - 1
    if invH is None:
        H = build_gram_matrix(X)
        H +=np.eye(n, dtype=np.float64)
        invH = np.linalg.inv(H)
    y_t = np.asarray(y, dtype=np.float64)

    scores = np.zeros(n)



    dot_full = invH.dot(y_t)  

    # 2. Extract diagonals: d = invH[k, k]
    d = np.diag(invH)      

    # 3. Vectorized calculation of (dot - y_k * d_k)
    dot = dot_full - y_t * d  

    # 4. Final vectorized scores
    scores = (dot ** 2) / d + 2 * y_t * dot + (y_t ** 2) * d
    return (scores, H, invH) if return_gram else scores
    

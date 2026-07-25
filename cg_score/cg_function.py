from __future__ import annotations
import numpy as np
import torch
from cg_score.gram_matrix import build_gram_matrix


def calc_lava_score(X: np.ndarray, y: np.ndarray, device: torch.device,
                   normalize: bool = True, return_gram: bool = False) -> np.ndarray | tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """
    Compute DV-score for the whole dataset (no subsampling).

    Args:
        X: Input features, shape (n, d).
        y: Continuous targets, shape (n,).
        device: torch device.
        normalize: If True, L2-normalize X per sample and scale y to [-1, 1] inside.
        return_gram: If True, also return H and invH.

    Returns:
        scores: DV-score for each sample, shape (n,).
        If return_gram=True, returns (scores, H, invH).
    """
    n = len(X)

    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / np.where(norms == 0, 1, norms)
        y_min, y_max = y.min(), y.max()
        if y_max - y_min > 0:
            y = 2 * (y - y_min) / (y_max - y_min) - 1

    H = build_gram_matrix(X, device)
    H += 1e-5 * torch.eye(n, dtype=H.dtype, device=device)
    invH = torch.inverse(H)
    y_t = torch.tensor(y, dtype=torch.float64, device=device)

    scores = np.zeros(n)

    with torch.no_grad():
        for k in range(n):
            col = invH[:, k]
            d_i = col[k].item()
            if d_i <= 0:
                d_i = 1e-8
            h_i = torch.cat([col[:k], col[k+1:]])
            y_mi = torch.cat([y_t[:k], y_t[k+1:]])
            y_i = y_t[k].item()

            dot = (y_mi.dot(h_i)).item()
            scores[k] = dot**2 / d_i + 2*y_i*dot + y_i**2 * d_i

    return (scores, H, invH) if return_gram else scores


def calc_lava_score_regression(
    X: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    subsample_size: int = None,
    rep_num: int = 1,
    sub_term: bool = False,
) -> dict:
    """
    Compute DV-score for a regression dataset.

    Args:
        X: Input features, shape (n, d). Will be L2-normalized per sample.
        y: Continuous targets, shape (n,). Must satisfy |y| <= 1 (normalize first).
        device: torch device.
        subsample_size: If set, compute score on random subsets of this size
                        (useful when n is large). None = use all samples.
        rep_num: Number of repetitions when subsampling (averaged).
        sub_term: If True, also return the three sub-terms of Eq. 6.

    Returns:
        dict with keys:
            'vi'  : DV-score for each sample, shape (n,)
            'ab'  : Sub-term 2 * y_i * (y_{-i}^T h_i),  shape (n,)  [if sub_term]
            'a2'  : Sub-term (y_{-i}^T h_i)^2 / d_i,    shape (n,)  [if sub_term]
            'b2'  : Sub-term y_i^2 * d_i,                shape (n,)  [if sub_term]
    """
    n = len(X)
    counts = np.zeros(n)
    accum = {k: np.zeros(n) for k in ['vi', 'ab', 'a2', 'b2']}

    with torch.no_grad():
        for _ in range(rep_num):
            if subsample_size is not None and subsample_size < n:
                idx = np.random.choice(n, subsample_size, replace=False)
            else:
                idx = np.arange(n)

            X_sub = X[idx]
            y_sub = y[idx]

            norms = np.linalg.norm(X_sub, axis=1, keepdims=True)
            X_sub = X_sub / np.where(norms == 0, 1, norms)

            H = build_gram_matrix(X_sub, device)
            H += 1e-5 * torch.eye(len(idx), dtype=H.dtype, device=device)
            invH = torch.inverse(H)

            y_t = torch.tensor(y_sub, dtype=torch.float64, device=device)
            original_complexity = (y_t @ (invH @ y_t)).item()

            for k in range(len(idx)):
                col = invH[:, k]
                d_i = col[k].item()
                h_i = torch.cat([col[:k], col[k+1:]])
                y_mi = torch.cat([y_t[:k], y_t[k+1:]])
                y_i = y_t[k].item()

                dot = (y_mi @ h_i).item()
                cg_i = (dot / np.sqrt(d_i) + y_i * np.sqrt(d_i)) ** 2

                global_k = idx[k]
                accum['vi'][global_k] += cg_i
                counts[global_k] += 1

                if sub_term:
                    accum['ab'][global_k] += 2 * y_i * dot
                    accum['a2'][global_k] += (dot ** 2) / d_i
                    accum['b2'][global_k] += (y_i ** 2) * d_i

            del H, invH, y_t
            torch.cuda.empty_cache()

    safe_counts = np.where(counts > 0, counts, 1)
    result = {k: accum[k] / safe_counts for k in accum}
    return result if sub_term else {'vi': result['vi']}

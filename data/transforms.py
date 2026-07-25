import numpy as np


def normalize_targets(y: np.ndarray) -> tuple:
    """
    Normalize targets to [-1, 1] using min-max scaling.

    Args:
        y: Target vector, shape (n,).

    Returns:
        y_norm: Normalized targets, shape (n,), in [-1, 1].
        y_min: Minimum value before normalization.
        y_max: Maximum value before normalization.
    """
    y_min = y.min()
    y_max = y.max()
    y_norm = 2 * (y - y_min) / (y_max - y_min) - 1
    return y_norm, y_min, y_max


def denormalize_targets(y_norm: np.ndarray, y_min: float, y_max: float) -> np.ndarray:
    """
    Denormalize targets from [-1, 1] back to original scale.

    Args:
        y_norm: Normalized targets, shape (n,).
        y_min: Minimum value before normalization.
        y_max: Maximum value before normalization.

    Returns:
        y: Denormalized targets, shape (n,).
    """
    y = (y_norm + 1) / 2 * (y_max - y_min) + y_min
    return y


def normalize_features(X: np.ndarray) -> np.ndarray:
    """
    Normalize features using standardization (zero mean, unit std per feature).

    Args:
        X: Feature matrix, shape (n, d).

    Returns:
        X_norm: Normalized features, shape (n, d).
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1, std)
    return (X - mean) / std


def normalize_features_l2(X: np.ndarray) -> np.ndarray:
    """
    L2-normalize each sample to unit sphere (required by H∞ definition).

    Args:
        X: Feature matrix, shape (n, d).

    Returns:
        X_norm: L2-normalized features, shape (n, d).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1, norms)

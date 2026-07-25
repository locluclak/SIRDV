import random

import numpy as np
import torch

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def rmse(predictions, targets) -> float:
    """Root Mean Squared Error."""
    predictions = np.asarray(predictions).squeeze()
    targets = np.asarray(targets).squeeze()
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


def mae(predictions, targets) -> float:
    """Mean Absolute Error."""
    predictions = np.asarray(predictions).squeeze()
    targets = np.asarray(targets).squeeze()
    return float(np.mean(np.abs(predictions - targets)))

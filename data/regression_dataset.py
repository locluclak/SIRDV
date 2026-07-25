import numpy as np
import torch
from torch.utils.data import Dataset, random_split


class RegressionDataset(Dataset):
    """
    Generic PyTorch Dataset for tabular regression.

    Args:
        X: Feature matrix, shape (n, d).
        y: Target vector, shape (n,). Should be normalized to [-1, 1].
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_datasets(X, y, train_ratio=0.8, seed=42):
    """Split into train/test datasets."""
    dataset = RegressionDataset(X, y)
    n_train = int(len(dataset) * train_ratio)
    n_test = len(dataset) - n_train
    return random_split(dataset, [n_train, n_test],
                        generator=torch.Generator().manual_seed(seed))

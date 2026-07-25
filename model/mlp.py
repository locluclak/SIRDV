import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple MLP for tabular regression.

    Args:
        input_dim: Number of input features.
        hidden_dims: List of hidden layer dimensions.
        output_dim: Number of output values (default: 1 for regression).
        dropout: Dropout probability.
    """
    def __init__(self, input_dim, hidden_dims=[128, 64], output_dim=1, dropout=0.1):
        super(MLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def MLPRegression(input_dim, hidden_dims=[128, 64], dropout=0.1):
    """Create MLP for regression with output_dim=1."""
    return MLP(input_dim, hidden_dims, output_dim=1, dropout=dropout)

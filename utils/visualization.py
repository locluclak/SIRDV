import numpy as np
import matplotlib.pyplot as plt


def plot_score_distribution(scores, title="DV-score Distribution", save_path=None):
    """
    Histogram of LAVA-scores.

    Args:
        scores: DV-score array, shape (n,).
        title: Plot title.
        save_path: Path to save the figure (optional).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(scores, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('DV-score')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_score_vs_target(scores, y, title="DV-score vs Target Value", save_path=None):
    """
    Scatter plot: x-axis = normalized target y, y-axis = DV-score.

    Useful to see if outliers in y-space have high DV-score.

    Args:
        scores: DV-score array, shape (n,).
        y: Target values, shape (n,).
        title: Plot title.
        save_path: Path to save the figure (optional).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y, scores, alpha=0.5, s=10)
    ax.set_xlabel('Target Value (y)')
    ax.set_ylabel('DV-score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_pruning_curve(keep_fractions, metrics, labels, title="Data Pruning Experiment",
                       save_path=None, ylabel="RMSE"):
    """
    Pruning curve: x-axis = fraction of data kept, y-axis = metric (RMSE/MAE).

    Args:
        keep_fractions: List of kept fractions.
        metrics: List of metric arrays, each shape (len(keep_fractions),).
        labels: List of labels for each metric line.
        title: Plot title.
        save_path: Path to save the figure (optional).
        ylabel: Y-axis label.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for metric, label in zip(metrics, labels):
        ax.plot(keep_fractions, metric, marker='o', label=label)

    ax.set_xlabel('Fraction of Data Kept')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()  # Higher keep fraction on left

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

"""
Plot: DV-score Pruning vs Random Pruning
=========================================

Generates a pruning curve comparing:
  - DV-score pruning (remove low-score first)
  - Random pruning (baseline)
  - Remove high-score first (upper bound)

Usage:
    conda run -n IS python examples/plot_pruning.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import torch

from data.transforms import normalize_targets, normalize_features
from cg_score.cg_function import calc_lava_score, calc_lava_score_regression
from utils.train import rmse


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """Train Ridge regression and return RMSE."""
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return rmse(pred, y_test)


def main():
    np.random.seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets = {
        'California Housing': fetch_california_housing(),
        'Diabetes': load_diabetes(),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, data) in zip(axes, datasets.items()):
        X, y = data.data, data.target
        print(f"\n{'='*50}")
        print(f"Dataset: {name} (n={len(X)}, d={X.shape[1]})")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Normalize
        X_norm = normalize_features(X_train)
        y_norm, y_min, y_max = normalize_targets(y_train)
        X_test_norm = normalize_features(X_test)

        # Subsample for large datasets
        sub_n = min(1000, len(X_train))

        print(f"Computing LAVA-scores (subsample={sub_n}, rep=5)...")
        scores = calc_lava_score(X_norm, y_norm, device, normalize=True)
        vi = scores

        # Pruning experiment
        fractions = np.arange(0.0, 0.55, 0.05)
        rmse_low, rmse_high, rmse_random = [], [], []

        for frac in fractions:
            n_keep = int(len(X_train) * (1 - frac))
            sorted_idx = np.argsort(vi)

            r_low, r_high, r_rand = [], [], []
            for seed in range(5):
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=seed)

                # Scores for this split's train set
                X_tr_norm = normalize_features(X_tr)
                y_tr_norm, _, _ = normalize_targets(y_tr)
                X_te_norm = normalize_features(X_te)

                if len(X_tr) > sub_n:
                    sc = calc_lava_score_regression(X_tr_norm, y_tr_norm, device,
                                                   subsample_size=sub_n, rep_num=1)
                    tr_vi = sc['vi']
                else:
                    tr_vi = vi[:len(X_tr)] if len(vi) >= len(X_tr) else np.zeros(len(X_tr))

                n_keep_split = int(len(X_tr) * (1 - frac))
                idx_sorted = np.argsort(tr_vi)

                keep_low = idx_sorted[-n_keep_split:]
                keep_high = idx_sorted[:n_keep_split]
                keep_rand = np.random.choice(len(X_tr), n_keep_split, replace=False)

                for keep, store in [(keep_low, r_low), (keep_high, r_high), (keep_rand, r_rand)]:
                    store.append(train_and_evaluate(X_tr[keep], y_tr[keep], X_te, y_te))

            rmse_low.append(np.mean(r_low))
            rmse_high.append(np.mean(r_high))
            rmse_random.append(np.mean(r_rand))

        keep_pct = (1 - fractions) * 100

        ax.plot(keep_pct, rmse_low, 'b-o', linewidth=2, markersize=4, label='Prune Low DV-score')
        ax.plot(keep_pct, rmse_random, 'k--s', linewidth=1.5, markersize=4, label='Random Pruning')
        ax.plot(keep_pct, rmse_high, 'r-^', linewidth=1.5, markersize=4, label='Prune High DV-score')

        ax.set_xlabel('% Data Kept', fontsize=12)
        ax.set_ylabel('Test RMSE', fontsize=12)
        ax.set_title(f'{name}', fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        print(f"Done. RMSE range: [{min(rmse_low+rmse_random+rmse_high):.3f}, "
              f"{max(rmse_low+rmse_random+rmse_high):.3f}]")

    plt.suptitle('DV-score Data Pruning vs Random Pruning', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), "..", "output", "pruning_comparison.png"), dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to pruning_comparison.png")
    plt.show()


if __name__ == '__main__':
    main()

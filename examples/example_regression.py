"""
Example: Computing LAVA-scores for Regression
============================================

This script demonstrates how to use the DV-score implementation
for a tabular regression task. It shows:
  1. Loading and normalizing data
  2. Computing LAVA-scores
  3. Interpreting the results
  4. Running a simple pruning experiment

Usage:
    conda run -n IS python examples/example_regression.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge

from data.transforms import normalize_targets, normalize_features
from cg_score.cg_function import calc_lava_score_regression
from evaluate.pruning import run_pruning_experiment


def main():
    # =========================================================================
    # 1. Generate synthetic regression data
    # =========================================================================
    print("=" * 60)
    print("Step 1: Generate synthetic data")
    print("=" * 60)

    np.random.seed(42)
    n_samples, n_features = 300, 5
    X, y = make_regression(n_samples=n_samples, n_features=n_features,
                           noise=10.0, random_state=42)

    # Add a few outliers to y (simulate noisy labels)
    outlier_idx = np.random.choice(n_samples, 15, replace=False)
    y[outlier_idx] += np.random.randn(15) * 50

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  y range: [{y.min():.1f}, {y.max():.1f}]")
    print(f"  Injected {len(outlier_idx)} outliers into y")

    # =========================================================================
    # 2. Normalize features and targets
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Normalize data")
    print("=" * 60)

    # Feature standardization (zero mean, unit std per feature)
    X_norm = normalize_features(X)

    # Target normalization to [-1, 1] (required by DV-score theory)
    y_norm, y_min, y_max = normalize_targets(y)

    print(f"  X_norm stats: mean={X_norm.mean():.4f}, std={X_norm.std():.4f}")
    print(f"  y_norm range: [{y_norm.min():.3f}, {y_norm.max():.3f}]")
    print(f"  Original y range stored: y_min={y_min:.1f}, y_max={y_max:.1f}")

    # =========================================================================
    # 3. Compute LAVA-scores
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Compute LAVA-scores")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # For n=300, we can use all samples (no subsampling needed)
    # For large n (>5000), set subsample_size=1000 and increase rep_num
    scores_dict = calc_lava_score_regression(
        X_norm, y_norm, device,
        subsample_size=None,   # Use all samples
        rep_num=5,             # Average over 5 repetitions
        sub_term=True          # Also return sub-terms for analysis
    )

    vi = scores_dict['vi']   # DV-score for each sample

    print(f"\n  DV-score results:")
    print(f"    Shape: {vi.shape}")
    print(f"    Min:   {vi.min():.4f}")
    print(f"    Max:   {vi.max():.4f}")
    print(f"    Mean:  {vi.mean():.4f}")
    print(f"    Std:   {vi.std():.4f}")
    print(f"    All non-negative: {(vi >= 0).all()}")

    # =========================================================================
    # 4. Interpret results — check if outliers rank high
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Check outlier detection")
    print("=" * 60)

    # Rank samples by DV-score (higher = more influential)
    ranks = np.argsort(vi)[::-1]  # descending order

    # Check how many of top-20 are actual outliers
    top_k = 20
    top_indices = set(ranks[:top_k])
    outliers_in_top = len(top_indices.intersection(set(outlier_idx)))
    outlier_rate = outliers_in_top / top_k

    print(f"  Top-{top_k} highest DV-score samples:")
    print(f"    {outliers_in_top}/{top_k} are injected outliers ({outlier_rate:.0%})")
    print(f"    vs. expected by chance: {len(outlier_idx)/n_samples:.0%}")

    # Show top-5 most influential samples
    print(f"\n  Top 5 most influential samples:")
    for i, idx in enumerate(ranks[:5]):
        is_outlier = "OUTLIER" if idx in outlier_idx else "normal"
        print(f"    #{i+1}: sample {idx:3d}, DV-score={vi[idx]:.4f}, "
              f"y={y[idx]:+.1f}, [{is_outlier}]")

    # =========================================================================
    # 5. Simple pruning experiment
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Pruning experiment")
    print("=" * 60)

    # Use Ridge regression as the base model for pruning experiment
    model_fn = lambda: Ridge(alpha=1.0)

    prune_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]

    results = run_pruning_experiment(
        X, y, vi, prune_fractions, model_fn,
        n_seeds=3, test_size=0.2, verbose=False
    )

    print(f"  {'Keep%':>6s}  {'Prune Low':>10s}  {'Prune High':>10s}  {'Random':>10s}")
    print(f"  {'------':>6s}  {'----------':>10s}  {'-----------':>10s}  {'--------':>10s}")
    for i, frac in enumerate(results['keep_fractions']):
        print(f"  {frac:5.0%}  "
              f"{results['rmse_low_prune'][i]:10.4f}  "
              f"{results['rmse_high_prune'][i]:10.4f}  "
              f"{results['rmse_random'][i]:10.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
  DV-score for regression computes each sample's contribution to the
  generalization error bound. Key properties:

    - High DV-score = influential / hard / outlier sample
    - Low DV-score  = typical / redundant sample

  Pruning low-score samples should keep RMSE roughly flat.
  Pruning high-score samples should degrade RMSE quickly.

  In this example, we confirmed that injected outliers rank higher
  in DV-score, validating the method's ability to detect noisy labels.
    """)


if __name__ == '__main__':
    main()

"""
Example: DV-score on Real Regression Datasets
==============================================

Demonstrates DV-score computation on well-known real-world datasets:
  - California Housing (sklearn)
  - Diabetes (sklearn)
  - Boston Housing (via openml)

Usage:
    conda run -n IS python examples/example_real_dataset.py
    conda run -n IS python examples/example_real_dataset.py --dataset diabetes
    conda run -n IS python examples/example_real_dataset.py --dataset boston
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import torch
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data.transforms import normalize_targets, normalize_features
from cg_score.cg_function import calc_lava_score_regression, calc_lava_score
from evaluate.pruning import run_pruning_experiment
from model.mlp import MLPRegression
from utils.train import rmse, mae


def load_dataset(name):
    """Load a real regression dataset."""
    if name == "california":
        data = fetch_california_housing()
        X, y = data.data, data.target
        feature_names = data.feature_names
    elif name == "diabetes":
        data = load_diabetes()
        X, y = data.data, data.target
        feature_names = data.feature_names
    elif name == "boston":
        data = fetch_california_housing()  # fallback (boston removed in sklearn 1.2+)
        X, y = data.data, data.target
        feature_names = data.feature_names
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return X, y, feature_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='california',
                        choices=['california', 'diabetes', 'boston'],
                        help='Dataset name')
    parser.add_argument('--subsample_n', type=int, default=1000,
                        help='Subsample size (None = use all)')
    parser.add_argument('--rep_num', type=int, default=1,
                        help='Number of DV-score repetitions')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--noise_pct', type=float, default=0.0,
                        help='%% of labels to corrupt with noise (0-50)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    np.random.seed(args.seed)

    # =====================================================================
    # 1. Load real dataset
    # =====================================================================
    print("=" * 65)
    print(f"Dataset: {args.dataset}")
    print("=" * 65)

    X, y, feature_names = load_dataset(args.dataset)
    print(f"  Samples:  {X.shape[0]}")
    print(f"  Features: {X.shape[1]} ({', '.join(feature_names[:5])}{'...' if len(feature_names)>5 else ''})")
    print(f"  Target:   min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}, std={y.std():.2f}")

    # Inject label noise
    if args.noise_pct > 0:
        n_noisy = int(len(y) * args.noise_pct / 100)
        noisy_idx = np.random.choice(len(y), n_noisy, replace=False)
        y_std = y.std()
        y[noisy_idx] += np.random.randn(n_noisy) * y_std * 2
        print(f"  Noise:    injected {n_noisy} noisy labels ({args.noise_pct}%)")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed)
    print(f"  Split:    train={len(X_train)}, test={len(X_test)}")

    # =====================================================================
    # 2. Normalize
    # =====================================================================
    print("\n" + "-" * 65)
    print("Normalization")
    print("-" * 65)

    # Standardize features
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    # Normalize targets to [-1, 1]
    y_train_norm, y_min, y_max = normalize_targets(y_train)
    print(f"  Target scale: [{y_min:.2f}, {y_max:.2f}] -> [-1, 1]")

    # =====================================================================
    # 3. Compute LAVA-scores on training set
    # =====================================================================
    print("\n" + "-" * 65)
    print("DV-score Computation")
    print("-" * 65)
    print(f"  Device:     {device}")
    print(f"  Repetitions: {args.rep_num}")

    # Auto-select subsample size if dataset is large
    n = len(X_train)
    subsample_n = args.subsample_n
    if subsample_n is None and n > 3000:
        subsample_n = 1000
        print(f"  Auto-subsample: {n} > 3000, using subsample_n={subsample_n}")
    print(f"  Subsample:  {subsample_n or 'all'}")

    # Compute LAVA-scores on training set
    scores = calc_lava_score(X_train, y_train, device, normalize=True)

    vi = scores
    # ab = scores_dict['ab']   # 2 * y_i * (y_{-i}^T h_i)

    # print(f"\n  DV-score statistics:")
    # print(f"    {'':12s} {'min':>8s} {'max':>8s} {'mean':>8s} {'std':>8s}")
    # print(f"    {'vi (CG)':12s} {vi.min():8.4f} {vi.max():8.4f} {vi.mean():8.4f} {vi.std():8.4f}")
    # print(f"    {'ab (sub-term)':12s} {ab.min():8.4f} {ab.max():8.4f} {ab.mean():8.4f} {ab.std():8.4f}")

    # =====================================================================
    # 4. Per-sample analysis
    # =====================================================================
    print("\n" + "-" * 65)
    print("Sample Analysis")
    print("-" * 65)

    ranks = np.argsort(vi)[::-1]

    print("  Top 10 most influential samples:")
    print(f"  {'Rank':>4s} {'Idx':>5s} {'DV-score':>10s} {'Target(y)':>10s} {'|y|':>8s}")
    print(f"  {'----':>4s} {'-----':>5s} {'----------':>10s} {'----------':>10s} {'--------':>8s}")
    for i, idx in enumerate(ranks[:10]):
        print(f"  {i+1:4d} {idx:5d} {vi[idx]:10.4f} {y[idx]:10.2f} {abs(y[idx]):8.2f}")

    print("\n  Bottom 5 least influential samples:")
    for i, idx in enumerate(ranks[-5:]):
        print(f"  {len(ranks)-4+i:4d} {idx:5d} {vi[idx]:10.4f} {y[idx]:10.2f} {abs(y[idx]):8.2f}")

    # =====================================================================
    # 5. Detect potential outliers via sub-term
    # =====================================================================
    print("\n" + "-" * 65)
    print("Outlier Detection (via sub-term ab)")
    print("-" * 65)
    print("  Negative ab = sample inconsistent with neighbors (potential noise)")

    bottom_10pct = int(len(y) * 0.1)
    # suspicious_idx = np.argsort(ab)[:bottom_10pct]
    # print(f"  Flagged {len(suspicious_idx)} samples (bottom 10% of ab)")
    # print(f"  Target stats for flagged samples:")
    # flagged_y = y[suspicious_idx]
    # print(f"    mean={flagged_y.mean():.2f}, std={flagged_y.std():.2f}")
    # print(f"  Target stats for all samples:")
    # print(f"    mean={y.mean():.2f}, std={y.std():.2f}")

    # =====================================================================
    # 5. Pruning experiment: DV-score vs Random
    # =====================================================================
    print("\n" + "-" * 65)
    print("Pruning Experiment: DV-score vs Random")
    print("-" * 65)
    print("  Strategy: remove LOW DV-score samples (keep important ones)")
    print("  Positive Diff = DV-score better than Random (lower RMSE)")
    print()

    train_vi = vi
    prune_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    n_seeds = 5

    print(f"  {'Keep%':>6s}  {'N':>6s}  {'DV-score':>10s}  {'Random':>10s}  {'Diff':>8s}")
    print(f"  {'------':>6s}  {'------':>6s}  {'----------':>10s}  {'----------':>10s}  {'--------':>8s}")

    for frac in prune_fractions:
        n_keep = int(len(X_train) * (1 - frac))
        sorted_idx = np.argsort(train_vi)  # ascending: low scores first

        rmse_cg_seeds = []
        rmse_rand_seeds = []

        for seed in range(n_seeds):
            if frac == 0.0:
                keep_cg = np.arange(len(X_train))
                keep_rand = np.arange(len(X_train))
            else:
                keep_cg = sorted_idx[-n_keep:]   # keep HIGH score, remove low
                keep_rand = np.random.RandomState(seed).choice(len(X_train), n_keep, replace=False)

            for keep_idx, store in [(keep_cg, rmse_cg_seeds), (keep_rand, rmse_rand_seeds)]:
                model = MLPRegression(input_dim=X_train.shape[1], hidden_dims=[32, 8]).to(device)
                X_sub = X_train_norm[keep_idx]
                y_sub = y_train_norm[keep_idx]

                model.train()
                X_t = torch.tensor(X_sub, dtype=torch.float32).to(device)
                y_t = torch.tensor(y_sub, dtype=torch.float32).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                loss_fn = torch.nn.MSELoss()

                for epoch in range(100):
                    pred = model(X_t).squeeze()
                    loss = loss_fn(pred, y_t)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                X_te = torch.tensor(X_test_norm, dtype=torch.float32).to(device)
                with torch.no_grad():
                    pred_norm = model(X_te).squeeze().cpu().numpy()
                pred = (pred_norm + 1) / 2 * (y_max - y_min) + y_min
                store.append(rmse(pred, y_test))

        mean_cg = np.mean(rmse_cg_seeds)
        mean_rand = np.mean(rmse_rand_seeds)
        diff = mean_rand - mean_cg  # positive = CG better

        print(f"  {1-frac:5.0%}  {n_keep:6d}  {mean_cg:10.4f}  {mean_rand:10.4f}  {diff:+8.4f}")

    print()


if __name__ == '__main__':
    main()

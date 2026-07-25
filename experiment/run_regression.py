import numpy as np
import torch
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.regression_dataset import make_datasets
from data.transforms import normalize_targets, normalize_features
from si_for_cg.lava_function import calc_lava_score
from evaluate.pruning import run_pruning_experiment
from model.mlp import MLPRegression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='synthetic',
                        choices=['california', 'synthetic'],
                        help='Dataset to use')
    parser.add_argument('--subsample_size', type=int, default=1000,
                        help='Subsample size for DV-score computation')
    parser.add_argument('--rep_num', type=int, default=10,
                        help='Number of repetitions for DV-score')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto/cpu/cuda)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory for results')
    args = parser.parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    np.random.seed(args.seed)

    if args.dataset == 'california':
        data = fetch_california_housing()
        X, y = data.data, data.target
        feature_names = data.feature_names
    else:
        X, y = make_regression(n_samples=200, n_features=5, noise=0.1, random_state=args.seed)
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

    print(f"Dataset: {args.dataset}, n_samples={len(X)}, n_features={X.shape[1]}")

    X_norm = normalize_features(X)
    y_norm, y_min, y_max = normalize_targets(y)
    print(f"Target range: [{y_min:.2f}, {y_max:.2f}] -> normalized to [-1, 1]")

    Xtrain, Xval, ytrain, yval = train_test_split(X_norm, y_norm, test_size=0.2, random_state=args.seed)
    print(f"\nComputing LAVA-scores (subsample_size={args.subsample_size}, rep_num={args.rep_num})...")
    scores = calc_lava_score(Xtrain, ytrain, Xval, yval)
    
    print(f"LAVA-scores computed. Shape: {scores}")
    # vi_scores = scores['vi']

    # print(f"DV-score stats: min={vi_scores.min():.4f}, max={vi_scores.max():.4f}, "
    #       f"mean={vi_scores.mean():.4f}, std={vi_scores.std():.4f}")

    # print(f"\nRunning pruning experiment...")
    # model_fn = lambda: Ridge(alpha=1.0)
    # prune_fractions = [0.1, 0.2, 0.3, 0.4, 0.5]

    # results = run_pruning_experiment(X, y, vi_scores, prune_fractions, model_fn,
    #                                   n_seeds=5, verbose=True)

    # os.makedirs(args.output_dir, exist_ok=True)
    # np.savez(os.path.join(args.output_dir, 'cg_scores.npz'),
    #          scores=vi_scores, X=X, y=y)
    # np.savez(os.path.join(args.output_dir, 'pruning_results.npz'), **results)

    # print(f"\nResults saved to {args.output_dir}/")

    # try:
    #     from utils.visualization import plot_score_distribution, plot_pruning_curve

    #     plot_score_distribution(vi_scores, title=f"DV-score Distribution ({args.dataset})",
    #                            save_path=os.path.join(args.output_dir, 'score_dist.png'))

    #     plot_pruning_curve(results['keep_fractions'],
    #                       [results['rmse_low_prune'], results['rmse_high_prune'], results['rmse_random']],
    #                       labels=['Prune Low-Score', 'Prune High-Score', 'Random'],
    #                       title=f"Pruning Curve ({args.dataset})",
    #                       save_path=os.path.join(args.output_dir, 'pruning_curve.png'))
    #     print("Plots saved.")
    # except ImportError:
    #     print("matplotlib not available, skipping plots.")


if __name__ == '__main__':
    main()

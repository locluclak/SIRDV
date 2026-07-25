import numpy as np
from sklearn.model_selection import train_test_split
from utils.train import rmse, mae


def run_pruning_experiment(
    X, y, scores, prune_fractions, model_fn, n_seeds=5,
    test_size=0.2, verbose=True
):
    """
    Evaluate DV-score usefulness via data pruning.

    For each fraction in prune_fractions:
      - Remove that fraction of lowest-scoring samples (easy/regular samples)
      - Train a model on the remaining data
      - Evaluate RMSE on hold-out test set

    Args:
        X: Feature matrix, shape (n, d).
        y: Target vector, shape (n,).
        scores: DV-score for each sample, shape (n,).
        prune_fractions: List of fractions to prune (e.g., [0.1, 0.2, 0.3]).
        model_fn: Function that returns a new model instance (e.g., lambda: MLPRegression(8)).
        n_seeds: Number of random seeds for averaging.
        test_size: Fraction of data for test split.
        verbose: Whether to print progress.

    Returns:
        dict with pruning results for plotting:
            'keep_fractions': list of kept fractions
            'rmse_low_prune': RMSE when pruning low-score samples
            'rmse_high_prune': RMSE when pruning high-score samples
            'rmse_random': RMSE with random pruning baseline
    """
    results = {
        'keep_fractions': [],
        'rmse_low_prune': [],
        'rmse_high_prune': [],
        'rmse_random': []
    }

    for frac in prune_fractions:
        n_keep = int(len(X) * (1 - frac))
        sorted_idx = np.argsort(scores)

        rmse_low, rmse_high, rmse_rand = [], [], []

        for seed in range(n_seeds):
            keep_idx_low = sorted_idx[-n_keep:]
            keep_idx_high = sorted_idx[:n_keep]
            keep_idx_rand = np.random.choice(len(X), n_keep, replace=False)

            for keep_idx, store in [(keep_idx_low, rmse_low),
                                    (keep_idx_high, rmse_high),
                                    (keep_idx_rand, rmse_rand)]:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X[keep_idx], y[keep_idx], test_size=test_size, random_state=seed)
                model = model_fn()
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
                store.append(rmse(pred, y_te))

        results['keep_fractions'].append(1 - frac)
        results['rmse_low_prune'].append(np.mean(rmse_low))
        results['rmse_high_prune'].append(np.mean(rmse_high))
        results['rmse_random'].append(np.mean(rmse_rand))

        if verbose:
            print(f"Prune {frac:.0%}: keep {1-frac:.0%} | "
                  f"low_prune={np.mean(rmse_low):.4f} | "
                  f"high_prune={np.mean(rmse_high):.4f} | "
                  f"random={np.mean(rmse_rand):.4f}")

    return results

"""Independent test-set benchmark for LAVA-score data pruning.

Validation is used only to calculate LAVA scores; all reported errors come
from a separate test set.

Example: ``conda run -n IS python evaluate/testing_exp.py --runs 30``
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys

import numpy as np
from sklearn.linear_model import Lasso

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from si_for_cg import SelectionAdapter, CGSelectionAdapter


def generate_contaminated_regression(
    n_train: int = 80,
    n_val: int = 20,
    n_test: int = 2_000,
    p: int = 5,
    contamination: float = 0.30,
    beta_clean: float = 1.0,
    std_good: float = 1.0,
    std_bad: float = 3.0,
    seed: int | None = None,
):
    """Create a clean target domain plus harmful training observations.

Validation and test use ``beta_clean``.  Contaminated training observations
use a different conditional relationship and substantially higher noise.
``is_clean`` is only an oracle diagnostic, never an input to LAVA or Lasso.
"""
    if not 0 <= contamination < 1:
        raise ValueError("contamination must be in [0, 1).")

    rng = np.random.default_rng(seed)
    beta_clean = np.full(p,beta_clean)
    beta_bad = -np.ones(p)

    X_train = rng.normal(size=(n_train, p))
    is_bad = rng.random(n_train) < contamination
    noise_sd = np.where(is_bad, std_bad, std_good)
    beta_per_row = np.where(is_bad[:, None], beta_bad, beta_clean)
    y_train = (X_train * beta_per_row).sum(axis=1) + rng.normal(scale=noise_sd)

    X_val = rng.normal(size=(n_val, p))
    y_val = X_val @ beta_clean + rng.normal(scale=std_good, size=n_val)
    X_test = rng.normal(size=(n_test, p))
    y_test = X_test @ beta_clean + rng.normal(scale=std_good, size=n_test)

    Sigmat = std_good**2 * np.eye(n_train)
    Sigmav = std_good**2 * np.eye(n_val)
    return X_train, y_train, X_val, y_val, X_test, y_test, Sigmat, Sigmav, ~is_bad


def test_mse_lasso(X_train, y_train, keep, X_test, y_test, lambda_: float = 10.0) -> float:
    """Fit retained observations and return independent test MSE."""
    keep = np.asarray(keep, dtype=int)
    model = Lasso(
        # Keeps the unnormalised L1 penalty lambda_ fixed across subset sizes.
        alpha=lambda_ / len(keep),
        fit_intercept=False,
        tol=1e-10,
        max_iter=100_000,
    )
    model.fit(X_train[keep], y_train[keep])
    return float(np.mean((y_test - model.predict(X_test)) ** 2))


def run_lava_pruning_benchmark(
    runs: int = 30,
    keep_fraction: float = 0.50,
    random_repeats: int = 20,
    seed: int = 2026,
    **data_kwargs,
) -> dict[str, np.ndarray]:
    """Compare both LAVA directions with full, random, and oracle baselines.

    Both score directions are intentionally tested: a score must outperform
    random pruning on independent test MSE before its ranking direction is
    treated as a useful pruning rule.
    """
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1].")

    metrics = {name: [] for name in (
        "full", "lava_keep_high", "lava_keep_low", "random", "oracle_clean",
        "clean_rate_high", "clean_rate_low", "clean_rate_random",
    )}
    master_rng = np.random.default_rng(seed)

    for _ in range(runs):
        trial_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
        Xtr, ytr, Xval, yval, Xte, yte, Sigmat, Sigmav, is_clean = generate_contaminated_regression(
            seed=trial_seed, **data_kwargs
        )
        n_keep = max(1, int(len(Xtr) * keep_fraction))

        # Suppress existing OT diagnostic prints; benchmark output stays compact.
        with contextlib.redirect_stdout(io.StringIO()):
            # adapter = SelectionAdapter(fraction=keep_fraction)
            # high = np.asarray(adapter.select(Xtr, ytr, Xval, yval), dtype=int)
            adapter = CGSelectionAdapter(fraction=keep_fraction)
            high = np.asarray(adapter.select(Xtr, ytr), dtype=int)
        low = np.asarray(adapter.sorted_indices[:n_keep], dtype=int)
        all_indices = np.arange(len(Xtr))

        metrics["full"].append(test_mse_lasso(Xtr, ytr, all_indices, Xte, yte))
        metrics["lava_keep_high"].append(test_mse_lasso(Xtr, ytr, high, Xte, yte))
        metrics["lava_keep_low"].append(test_mse_lasso(Xtr, ytr, low, Xte, yte))
        metrics["oracle_clean"].append(test_mse_lasso(Xtr, ytr, np.flatnonzero(is_clean), Xte, yte))
        metrics["clean_rate_high"].append(float(is_clean[high].mean()))
        metrics["clean_rate_low"].append(float(is_clean[low].mean()))

        random_mse, random_clean_rate = [], []
        trial_rng = np.random.default_rng(trial_seed + 1)
        for _ in range(random_repeats):
            random_keep = trial_rng.choice(len(Xtr), size=n_keep, replace=False)
            random_mse.append(test_mse_lasso(Xtr, ytr, random_keep, Xte, yte))
            random_clean_rate.append(float(is_clean[random_keep].mean()))
        metrics["random"].append(float(np.mean(random_mse)))
        metrics["clean_rate_random"].append(float(np.mean(random_clean_rate)))

    return {name: np.asarray(values) for name, values in metrics.items()}


def print_summary(results: dict[str, np.ndarray]) -> None:
    """Print a decision-oriented summary."""
    print("\nIndependent-test MSE (lower is better)")
    print(f"{'method':<18} {'mean':>10} {'std':>10} {'beats random':>14}")
    for name in ("full", "lava_keep_high", "lava_keep_low", "random", "oracle_clean"):
        values = results[name]
        wins = "--" if name == "random" else f"{np.mean(values < results['random']):.1%}"
        std = values.std(ddof=1) if len(values) > 1 else 0.0
        print(f"{name:<18} {values.mean():10.4f} {std:10.4f} {wins:>14}")

    print("\nFraction of clean training samples retained (higher is better)")
    for name in ("clean_rate_high", "clean_rate_low", "clean_rate_random"):
        print(f"{name:<18} {results[name].mean():.1%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--keep-fraction", type=float, default=0.5)
    parser.add_argument("--contamination", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    results = run_lava_pruning_benchmark(
        runs=args.runs,
        keep_fraction=args.keep_fraction,
        contamination=args.contamination,
        seed=args.seed,
    )
    print_summary(results)


if __name__ == "__main__":
    main()

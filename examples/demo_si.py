"""Demo: test SI_DV overconditioning p-values for uniformity (n_iter=500)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import kstest, uniform
import matplotlib.pyplot as plt

from si_for_cg import SI_DV


def normalize_data(X, y):
    """L2-normalize X, scale y to [-1,1]."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms == 0, 1, norms)
    y_min, y_max = y.min(), y.max()
    if y_max - y_min > 0:
        y = 2 * (y - y_min) / (y_max - y_min) - 1
    return X, y


def compute_pvalue_once(n=30, d=3, fraction=0.5, rng=None):
    """Generate one dataset and return one p-value."""
    if rng is None:
        rng = np.random.RandomState()

    X = rng.randn(n, d)
    beta = np.zeros(d)
    y = X @ beta + rng.randn(n)  # null: beta=0, y = noise

    # X, y = normalize_data(X, y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1)

    pval = SI_DV(X_tr, y_tr, Xval=X_te, yval=y_te,
                 fraction=fraction, jth=0, method="parametric")
    return pval


def main():
    n_iter = 1
    print(f"Running {n_iter} iterations of SI_DV overconditioning...")

    rng = np.random.RandomState(seed=2)
    pvalues = []
    for i in range(n_iter):
        pv = compute_pvalue_once(rng=rng)
        if not np.isnan(pv):
            pvalues.append(pv)
            print(f'pvalue {i}th {pv}')
        if (i + 1) % 1 == 0:
            print(f"  {i+1}/{n_iter} done, {len(pvalues)} valid p-values")

    pvalues = np.array(pvalues)
    
    print(f"\n{len(pvalues)} valid p-values collected")

    # # KS test against Uniform(0,1)
    # ks_stat, ks_pval = kstest(pvalues, uniform.cdf)
    # print(f"KS test: statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")

    # # Plot histogram
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.hist(pvalues, bins=20, density=True, alpha=0.7, edgecolor="black", label="Observed p-values")
    # # ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="Uniform(0,1)")
    # ax.set_xlabel("p-value")
    # ax.set_ylabel("Density")
    # ax.set_title(f"SI_DV Overconditioning p-value Distribution (n={len(pvalues)})\n"
    #              f"KS stat={ks_stat:.4f}, KS p={ks_pval:.4f}")
    # ax.legend()
    # plt.tight_layout()
    # plt.savefig(os.path.join(os.path.dirname(__file__), "..", "output", "pvalue_histogram.png"), dpi=150)
    # print("Saved pvalue_histogram_para.png")
    # plt.show()


if __name__ == "__main__":
    main()

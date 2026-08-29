"""Demo: test SI_DV overconditioning p-values for uniformity (n_iter=500)."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
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
def generate(n, m, p, true_beta_s, true_beta_t, seed = None):
    """return Xs, Xt, Ys, Yt, Sigma_s, Sigma_t"""

    if seed is not None:
        np.random.seed(seed)

    Xs = np.random.normal(loc = 0, scale = 1, size = (n, p))
    Xt = np.random.normal(loc = 0, scale = 1, size = (m, p))


    mu_s = Xs.dot(true_beta_s)
    mu_t = Xt.dot(true_beta_t)
    # print(true_beta_s, true_beta_t)
    Ys = mu_s + np.random.normal(loc = 0, scale = 1, size = (n, 1))
    Yt = mu_t + np.random.normal(loc = 0, scale = 1, size = (m, 1))

    Sigma_s = np.identity(n)
    Sigma_t = np.identity(m)

    return Xs, Xt, Ys, Yt, Sigma_s, Sigma_t

def compute_pvalue_once(n=50, d=5, fraction=0.5, rng=None,i=0):
    """Generate one dataset and return one p-value."""
    # if rng is None:
    #     rng = np.random.RandomState()

    # X = rng.randn(n, d)
    # beta = np.zeros(d)
    
    # y = X.dot(beta) + rng.randn(n)  # null: beta=0, y = noise

    # # X, y = normalize_data(X, y)
    # X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=1)
    true_beta_t = np.asarray([0]*d).reshape(-1,1)
    true_beta_v = np.asarray([0]*d).reshape(-1,1)
    X_tr, X_te, y_tr, y_te, St, Sv = generate(n, 15, d, true_beta_t, true_beta_v,seed=507)
    # print(X_tr.shape, y_tr.shape)
    pval = SI_DV(X_tr, y_tr, Xval=X_te, yval=y_te,
                 fraction=fraction, jth=None, 
                 zmin=-20., zmax=20, lambda_=10, data_valuation="lava",
                 method="parametric",
                #  method="overconditioning"
                 )
    return pval


def main():
    n_iter = 1
    print(f"Running {n_iter} iterations of SI_DV overconditioning...")

    rng = np.random.RandomState(seed=0)
    pvalues = []
    for i in range(n_iter):
        pv = compute_pvalue_once(rng=rng,i=i)

        if not np.isnan(pv):
            pvalues.append(pv)
            print(f'pvalue {i}th {pv}')
        if (i + 1) % 1 == 0:
            print(f"  {i+1}/{n_iter} done, {len(pvalues)} valid p-values")

    pvalues = np.array(pvalues)
    
    print(f"\n{len(pvalues)} valid p-values collected")

    # KS test against Uniform(0,1)
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

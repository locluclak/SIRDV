"""KS-uniformity check for seqfs (fixed-K forward selection) SI_DV p-values.

Modelled on ``examples/demo_si.py`` but drives the seqfs feature-selection path
(``fs_method='seqfs'``) and reports the Kolmogorov-Smirnov test of the collected
null p-values against Uniform(0,1).

Run from the repo root with the ``IS`` python (see AGENTS.md):
    python examples/demo_si_seqfs.py
    python examples/demo_si_seqfs.py --n-iter 500 --method overconditioning
    python examples/demo_si_seqfs.py --data-valuation lava --n 100
"""

import argparse
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import kstest, uniform

from si_for_cg import SI_DV


def generate(n, m, p, seed=None):
    """Null data: betas are zero, so y is pure noise.

    Returns ``Xs, Xt, Ys, Yt`` (shapes ``(n,p)``, ``(m,p)``, ``(n,1)``,
    ``(m,1)``) matching ``examples/demo_si.py``.
    """
    if seed is not None:
        np.random.seed(seed)
    Xs = np.random.normal(0, 1, (n, p))
    Xt = np.random.normal(0, 1, (m, p))
    Ys = np.random.normal(0, 1, (n, 1))
    Yt = np.random.normal(0, 1, (m, 1))
    return Xs, Xt, Ys, Yt


def compute_pvalue_once(n, m, d, fraction, dv, method, K, seed=None):
    X_tr, X_val, y_tr, y_val = generate(n, m, d, seed=seed)
    with contextlib.redirect_stdout(io.StringIO()):
        pval = SI_DV(
            X_tr, y_tr, Xval=X_val, yval=y_val,
            fraction=fraction, jth=None,
            zmin=-20.0, zmax=20.0, lambda_=5.0,
            data_valuation=dv, method=method, lasso_path=False,
            fs_method="seqfs", K=K,
        )
    return float(pval)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=500)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--d", type=int, default=15)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--data-valuation", default="cg-score",
                    choices=["cg-score", "lava"])
    ap.add_argument("--method", default="parametric_nested",
                    choices=["parametric_nested", "overconditioning"])
    args = ap.parse_args()

    print(
        f"seqfs KS check: dv={args.data_valuation} method={args.method} "
        f"n={args.n} m={args.m} d={args.d} K={args.K} iters={args.n_iter}"
    )

    pvalues = []
    for i in range(args.n_iter):
        try:
            pv = compute_pvalue_once(
                args.n, args.m, args.d, args.fraction,
                args.data_valuation, args.method, args.K, seed=None)
        except Exception as exc:  # noqa: BLE001
            print(f"  iter {i}: skipped ({type(exc).__name__}: {exc})")
            continue
        if np.isfinite(pv):
            pvalues.append(pv)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.n_iter} done, {len(pvalues)} valid p-values")

    pvalues = np.array(pvalues)
    print(f"\n{len(pvalues)} valid p-values collected")
    print(f"min={pvalues.min():.4f} max={pvalues.max():.4f} "
          f"mean={pvalues.mean():.4f}")

    ks_stat, ks_pval = kstest(pvalues, uniform.cdf)
    print(f"KS test vs Uniform(0,1): statistic={ks_stat:.4f}, p-value={ks_pval:.4f}")
    if ks_pval < 0.05:
        print("=> REJECT uniformity at 5% (p-values are NOT uniform)")
    else:
        print("=> do NOT reject uniformity at 5%")


if __name__ == "__main__":
    main()

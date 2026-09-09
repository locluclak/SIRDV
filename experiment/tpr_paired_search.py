"""Paired search for an experimental setup where the parametric selective
test has higher TPR than Bonferroni (naive p-value with adjusted alpha).

Design
------
Every iteration generates ONE contaminated dataset and runs BOTH methods on
the SAME data with the SAME tested feature:

  * parametric  : SI (parametric) p-value, tested against `alpha`
  * naive       : unconditioned FS-naive p-value, tested against `alpha / K`
                  (Bonferroni, K = 2**8 as in tpr_plot.py)

The searched knobs are the ones available in tpr_experiment.py:
  lambda_   : Lasso penalty. Small lambda => the Lasso conditioning event
              barely truncates the null distribution => the parametric
              p-value approaches the naive one, while Bonferroni still pays
              the 1/K penalty.
  beta      : signal strength. If the signal is too strong the naive
              p-values fall below alpha/K and both methods saturate at TPR=1
              (the current behaviour). A moderate beta keeps naive p-values
              in [alpha/K, alpha] where Bonferroni loses power.
  fraction, contamination, d, ns : selection regime.

Results are appended to output4/paired_search.csv
"""

import numpy as np
import os
import sys
import time
import contextlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from si_for_cg.si_adapter import CGSelectionAdapter, make_eta_cg
from si_for_cg.si_types import SIConfig, LassoInfo
from si_for_cg.si_runner import selective_inference
from si_for_cg.si_utils import (
    decompose_y,
    solve_Lasso,
    generate_contaminated_regression,
)
from si_for_cg.si_dv import FSNaiveTest

ALPHA = 0.05
K_BONF = 2 ** 8


def SI_DV_jth(Xtrain, ytrain, fraction, jth, lambda_,
              zmin=-20.0, zmax=20.0, step=1e-5,
              method="parametric", data_valuation="cg-score"):
    """SI_DV but honouring `jth` (global feature index) so that the
    parametric and naive tests can be paired on the same feature."""
    Xtrain_arr = np.asarray(Xtrain)
    ytrain_arr = np.asarray(ytrain).reshape(-1)
    Sigma_arr = np.eye(ytrain_arr.shape[0])

    adapter = CGSelectionAdapter(fraction=fraction)
    observed_selection = adapter.select(Xtrain_arr, ytrain_arr)
    if len(observed_selection) == 0:
        raise ValueError("empty selection")

    M, Mc, sM = solve_Lasso(Xtrain_arr[observed_selection],
                            ytrain_arr.reshape(-1, 1)[observed_selection],
                            lambda_)
    if len(M) == 0:
        raise ValueError("empty feature selection")

    fs = LassoInfo(lambda_=lambda_)
    fs.M, fs.Mc, fs.sM = M, Mc, sM

    tested_position = int(jth)
    eta = make_eta_cg(X=Xtrain_arr, y=ytrain_arr,
                      obs_selection=observed_selection, j=tested_position)
    decomposition = decompose_y(ytrain_arr, eta, Sigma_arr)

    config = SIConfig(zmin=zmin, zmax=zmax, step=step,
                      data_valuation=data_valuation)
    result = selective_inference(
        Xtrain_arr, None, ytrain_arr, None, Sigma_arr, adapter, fs, eta,
        tested_index=tested_position,
        observed_selection=observed_selection,
        config=config, method=method,
    )
    return result.p_value


def get_active_features(Xtrain, ytrain, fraction, lambda_):
    """Run the shared selection pipeline once and return (selection, M)."""
    adapter = CGSelectionAdapter(fraction=fraction)
    selection = adapter.select(np.asarray(Xtrain), np.asarray(ytrain).reshape(-1))
    M, Mc, sM = solve_Lasso(np.asarray(Xtrain)[selection],
                            np.asarray(ytrain).reshape(-1, 1)[selection],
                            lambda_)
    return selection, M


def run_config(dv, ns, d, fraction, lambda_, con, beta, iters, n_val=25,
               std_bad=3.0, verbose=True):
    p_param, p_naive = [], []
    devnull = open(os.devnull, "w")

    for i in range(iters):
        seed = i + 1
        try:
            X_tr, y_tr, X_val, y_val, _X_te, _y_te, _St, _Sv, _clean = \
                generate_contaminated_regression(
                    n_train=ns, n_val=n_val, n_test=0, p=d,
                    contamination=con, beta_clean=beta,
                    std_good=1.0, std_bad=std_bad, seed=seed)

            selection, M = get_active_features(X_tr, y_tr, fraction, lambda_)
            if len(M) == 0:
                continue
            # pick one active feature (global index) and keep it paired
            j_global = int(np.random.choice(M))
            j_pos_in_M = int(np.flatnonzero(M == j_global)[0])

            t0 = time.time()
            with contextlib.redirect_stdout(devnull):
                pv_par = SI_DV_jth(X_tr, y_tr, fraction=fraction,
                                   jth=j_global, lambda_=lambda_,
                                   method="parametric", data_valuation=dv)
            runner = FSNaiveTest(fraction=fraction, data_valuation=dv,
                                 lambda_=lambda_)
            pv_naive = runner.fit(X_tr, y_tr, Xval=X_val, yval=y_val,
                                  jth=j_pos_in_M)

            p_param.append(pv_par)
            p_naive.append(pv_naive)
            if verbose:
                print(f"    [{i+1}/{iters}] seed={seed} j={j_global} "
                      f"param={pv_par:.4g} naive={pv_naive:.4g} "
                      f"({time.time()-t0:.1f}s)")
        except Exception as e:
            if verbose:
                print(f"    [{i+1}/{iters}] seed={seed} failed: {e}")
            continue
    devnull.close()

    p_param = np.asarray(p_param, dtype=float)
    p_naive = np.asarray(p_naive, dtype=float)
    n = len(p_param)
    tpr_param = np.mean(p_param <= ALPHA) if n else float("nan")
    tpr_bonf = np.mean(p_naive <= ALPHA / K_BONF) if n else float("nan")
    tpr_naive = np.mean(p_naive <= ALPHA) if n else float("nan")
    return {
        "dv": dv, "ns": ns, "d": d, "fraction": fraction,
        "lambda": lambda_, "con": con, "beta": beta, "iters": n,
        "tpr_param": tpr_param, "tpr_bonf": tpr_bonf,
        "tpr_naive_alpha": tpr_naive,
        "med_p_param": float(np.nanmedian(p_param)) if n else float("nan"),
        "med_p_naive": float(np.nanmedian(p_naive)) if n else float("nan"),
    }


def main():
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    # ------------------------------------------------------------------
    # grid: the current setup is lambda=10, beta in {.5,1,1.5,2}, where
    # parametric and Bonferroni TPR coincide.  We search for a regime where
    # the parametric test keeps its power while Bonferroni pays its 1/256.
    # ------------------------------------------------------------------
    grid = []
    for lam in [10.0, 3.0, 1.0, 0.5]:
        for beta in [1.0, 1.5, 2.0]:
            grid.append(dict(dv="cg-score", ns=50, d=5, fraction=0.3,
                             lambda_=lam, con=0.5, beta=beta))

    out_csv = os.path.join(os.path.dirname(__file__), "..", "output4",
                           "paired_search.csv")
    write_header = not os.path.exists(out_csv)

    print(f"{'lam':>5} {'beta':>5} | {'TPRpar':>7} {'TPRbonf':>7} "
          f"{'TPRniv':>7} | {'medPar':>9} {'medNai':>9} {'n':>4}")
    print("-" * 62)

    for cfg in grid:
        res = run_config(cfg["dv"], cfg["ns"], cfg["d"], cfg["fraction"],
                         cfg["lambda_"], cfg["con"], cfg["beta"], iters)
        print(f"{res['lambda']:5.1f} {res['beta']:5.1f} | "
              f"{res['tpr_param']:7.3f} {res['tpr_bonf']:7.3f} "
              f"{res['tpr_naive_alpha']:7.3f} | "
              f"{res['med_p_param']:9.2e} {res['med_p_naive']:9.2e} "
              f"{res['iters']:4d}")

        with open(out_csv, "a") as f:
            if write_header:
                f.write("dv,ns,d,fraction,lambda,con,beta,iters,"
                        "tpr_param,tpr_bonf,tpr_naive_alpha,"
                        "med_p_param,med_p_naive\n")
                write_header = False
            f.write(",".join(str(res[k]) for k in
                             ["dv", "ns", "d", "fraction", "lambda", "con",
                              "beta", "iters", "tpr_param", "tpr_bonf",
                              "tpr_naive_alpha", "med_p_param",
                              "med_p_naive"]) + "\n")


if __name__ == "__main__":
    main()

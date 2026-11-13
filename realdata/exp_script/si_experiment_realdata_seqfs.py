"""Selective-inference (SI_DV) p-value experiment on the real datasets using the
**seqfs** (fixed-K greedy forward-selection) feature-selection path.

This is the seqfs analogue of ``si_experiment_realdata.py`` (which serves the
Lasso FS path).  Everything except the feature selection is identical and is
imported from that module: the datasets (``fit_lasso_realdata.load_all``), the
column cleaning (``clean_design``), the per-iteration train/validation split,
the train-only standardization (``standardize_train``) and the OLS noise
variance (``estimate_sigma2``).

The only change is that every SI method conditions on the fixed-K forward
selection set ``M`` (``fs_method="seqfs"``, ``|M| = K`` features) obtained by
``si_for_cg.si_seqfs.forward_selection_path`` instead of the Lasso support.
See ``experiment/seqfs_experiment.py`` for the synthetic seqfs harness this
mirrors (methods: naive/Bonferroni, datasplit, overconditioning,
parametric_nested; both ``lava`` and ``cg-score``).

Methods (all run with ``data_valuation`` = ``--dv`` and ``fs_method="seqfs"``):
  * overconditioning   -> ``SI_DV(method="overconditioning")``
  * parametric_nested  -> ``SI_DV(method="parametric_nested")``
  * datasplit          -> ``DataSplittingTest``
  * bonferroni         -> naive p-value (``FSNaiveTest``, no selection
                          correction) with the rejection threshold divided by
                          ``perm(p, K) = p! / (p-K)!`` (the number of ordered
                          fixed-K forward-selection paths; p = number of design
                          features); the stored p-values are the raw naive ones.
                          Matches ``experiment/seqfs_plot.py`` (alpha / perm(D, K)).

Protocol (identical to the Lasso experiment)
--------------------------------------------
Each iteration draws a fresh train / validation split (seeded by ``--seed + i``)
and computes one p-value for EVERY position ``jth = 0..K-1`` inside the seqfs
set ``M``; the SI methods are called once per position and ``datasplit`` reuses
the same internal split (``random_state``) so all positions share one ``M``.

Outputs (under ``realdata/result_seqfs/`` by default; use ``--result-dir``)
--------------------------------------------------------------------------
  * ``si_pvals_{tag}_{dv}_{method}_lam{..}_frac{..}_n{..}_{var}.txt`` - the same
    filename pattern the Lasso experiment writes, so ``analyze_si_pvals.py``
    reads this directory unchanged:
        & "...\\IS\\python.exe" realdata/exp_script/analyze_si_pvals.py --result-dir realdata/result_seqfs
    One line per iteration: ``iter seed K f1:p1 ... fK:pK`` (feature index in
    the design = initial column order; appended, so re-running resumes).
  * ``si_features_{tag}.txt`` - the design-column index -> feature-name map
    (and the dropped columns) for that dataset.
  * ``si_seqfs_experiment.log`` - timestamped log of every iteration (including
    the estimated ``sigma2``) plus the full traceback on any error.

Examples (run from the repo root, conda env ``IS``):
  # fast smoke test: 2 iterations
  & "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" realdata/exp_script/si_experiment_realdata_seqfs.py --quick
  # full experiment: 120 iterations per method, K = 5 features
  & "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" realdata/exp_script/si_experiment_realdata_seqfs.py --iters 120 --K 5
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from datetime import datetime
from math import perm

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "realdata", "exp_script"))

from sklearn.model_selection import train_test_split

from si_for_cg import SI_DV, SelectiveInference, DataSplittingTest, FSNaiveTest
from fit_lasso_realdata import load_all
from si_experiment_realdata import (
    clean_design, standardize_train, estimate_sigma2, quiet, _fmt,
    EXTRA_DROP, ALL_METHODS, ALL_DATASETS,
)

RESULT_DIR = os.path.join(ROOT, "realdata", "result_seqfs")
LOG_PATH = os.path.join(RESULT_DIR, "si_seqfs_experiment.log")


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def log_error(dataset: str, method: str, iteration: int, seed: int, exc: BaseException) -> None:
    """Write a specific error record (type, message, full traceback) to the log."""
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                 f"ERROR dataset={dataset} method={method} iter={iteration} seed={seed} "
                 f"-> {type(exc).__name__}: {exc}\n")
        fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        fh.write("\n")


def pvalues_over_support(dataset: str, method: str, Xtr, ytr, Xval, yval,
                         fraction: float, lambda_: float, seed: int, dv: str,
                         sigmat, sigmav, zmin: float, zmax: float, K: int,
                         verbose: bool):
    """One p-value per position in the seqfs set M for a single iteration.

    ``M[j]`` is the DESIGN-COLUMN index of the feature selected at forward-step
    ``j+1`` (``M = sorted(path[-1])``, so the ascending order of the final set),
    and ``pvals[j]`` its p-value.  Returns ``(M, pvals)``.
    """
    common = dict(fraction=fraction, zmin=zmin, zmax=zmax, lambda_=lambda_,
                  data_valuation=dv, Sigmat=sigmat, Sigmav=sigmav,
                  fs_method="seqfs", K=K)

    if method == "bonferroni":
        # Naive (non-selective) FS statistic on the DV-selected samples: the
        # p-value carries NO selection correction; the multiplicity is handled
        # only by the rejection threshold alpha / 2**p (see run_one).
        st = np.eye(Xtr.shape[0]) if sigmat is None else sigmat
        sv = np.eye(Xval.shape[0]) if sigmav is None else sigmav
        runner = FSNaiveTest(fraction=fraction, data_valuation=dv, lambda_=lambda_,
                             fs_method="seqfs", K=K)
        with quiet(verbose):
            p0 = runner.fit(Xtr, ytr, Xval, yval, Sigmat=st, Sigmav=sv, jth=0)
            M = [int(x) for x in np.asarray(runner.M_, dtype=int)]
            k = int(runner.n_selected_)
            pvals = [p0]
            for jth in range(1, k):
                pvals.append(runner.fit(Xtr, ytr, Xval, yval,
                                        Sigmat=st, Sigmav=sv, jth=jth))
        return M, pvals

    if method == "datasplit":
        runner = DataSplittingTest(fraction=fraction, data_valuation=dv,
                                   lambda_=lambda_, random_state=seed,
                                   fs_method="seqfs", K=K)
        with quiet(verbose):
            p0 = runner.fit(Xtr, ytr, Xval, yval, Sigmat=sigmat, Sigmav=sigmav, jth=0)
            M = [int(x) for x in np.asarray(runner.M_, dtype=int)]
            k = int(runner.n_selected_)
            pvals = [p0]
            for jth in range(1, k):
                pvals.append(runner.fit(Xtr, ytr, Xval, yval,
                                        Sigmat=sigmat, Sigmav=sigmav, jth=jth))
        return M, pvals

    # overconditioning / parametric_nested: the DV-score selection is
    # deterministic (given Xtr/ytr/Xval/yval, all fixed per iteration), so a
    # fresh SI_DV call at each jth reproduces the same seqfs set M.
    si0 = SelectiveInference(method=method, lambda_=lambda_, data_valuation=dv,
                             fraction=fraction, zmin=zmin, zmax=zmax,
                             fs_method="seqfs", K=K)
    with quiet(verbose):
        si0 = si0.fit(Xtr, ytr, Xval=Xval, yval=yval,
                      Sigmat=sigmat, Sigmav=sigmav, jth=0)
        M = [int(x) for x in np.asarray(si0.lasso.M, dtype=int)]
        k = int(len(M))
        pvals = [si0.p_value_]
        for jth in range(1, k):
            pvals.append(SI_DV(Xtr, ytr, Xval=Xval, yval=yval, jth=jth,
                               method=method, **common))
    return M, pvals


def run_one(dataset: str, method: str, Xs: np.ndarray, ys: np.ndarray,
            args) -> None:
    tag = dataset.replace("[", "_").replace("]", "")
    fpath = os.path.join(RESULT_DIR,
                         f"si_pvals_{tag}_{args.dv}_{method}_lam{args.lambda_:g}"
                         f"_frac{args.fraction:g}_n{Xs.shape[0]}_{args.variance}.txt")
    write_header = not os.path.exists(fpath) or os.path.getsize(fpath) == 0

    ok = 0
    failed = 0
    all_p: list[float] = []
    all_s2: list[float] = []

    # Bonferroni: naive p-value, but the rejection threshold is alpha / perm(p, K)
    # (p = number of design features, K = fixed selection size); the number of
    # ordered fixed-K forward-selection paths is perm(p, K) = p! / (p-K)!.
    # Other methods use plain alpha.
    reject_thr = (args.alpha / float(perm(Xs.shape[1], args.K))
                  if method == "bonferroni" else args.alpha)

    log(f"START dataset={dataset} method={method} dv={args.dv} fs=seqfs K={args.K} "
        f"iters={args.iters} lambda={args.lambda_:g} fraction={args.fraction:g} "
        f"variance={args.variance} n={Xs.shape[0]} p={Xs.shape[1]} "
        f"reject_thr={reject_thr:.4g} -> {os.path.basename(fpath)}")

    with open(fpath, "a", encoding="utf-8") as fh:
        if write_header:
            fh.write(f"# dataset={dataset} method={method} dv={args.dv} fs=seqfs K={args.K} "
                     f"lambda={args.lambda_:g} fraction={args.fraction:g} "
                     f"variance={args.variance} n={Xs.shape[0]} p={Xs.shape[1]}\n")
            fh.write("# columns: iter seed K f1:p1 f2:p2 ... fK:pK\n")
            fh.write("# f = feature index in the design (see "
                     f"si_features_{tag}.txt for the index->name map)\n")
            fh.flush()

        for i in range(args.iters):
            seed = args.seed + i
            Xtr, Xval, ytr, yval = train_test_split(
                Xs, ys, test_size=args.test_size, random_state=seed)
            # scaler fit on TRAIN only, applied to val.
            Xtr, ytr, Xval, yval = standardize_train(Xtr, Xval, ytr, yval)
            if args.variance == "ols":
                sigma2 = estimate_sigma2(Xtr, ytr)
                sigmat = sigma2 * np.eye(Xtr.shape[0])
                sigmav = sigma2 * np.eye(Xval.shape[0])
                all_s2.append(sigma2)
            else:
                sigma2, sigmat, sigmav = None, None, None
            try:
                M, pvals = pvalues_over_support(dataset, method, Xtr, ytr, Xval, yval,
                                                args.fraction, args.lambda_, seed, args.dv,
                                                sigmat, sigmav, args.zmin, args.zmax,
                                                args.K, args.verbose)
            except Exception as exc:  # noqa: BLE001 - record the specific error, keep going
                failed += 1
                log_error(dataset, method, i, seed, exc)
                continue

            pairs = [f"{M[j]}:{_fmt(pvals[j])}" for j in range(len(pvals))]
            fh.write(f"{i} {seed} {len(pvals)} " + " ".join(pairs) + "\n")
            fh.flush()

            ok += 1
            all_p.extend(pvals)
            s2txt = "" if sigma2 is None else f" sigma2={sigma2:.4g}"
            log(f"  iter={i} seed={seed}{s2txt} |M|={len(pvals)} "
                f"features(M)={M} pvals=[{' '.join(_fmt(p) for p in pvals)}]")

    if all_p:
        arr = np.asarray(all_p, dtype=float)
        valid = arr[~np.isnan(arr)]
        mean_p = float(valid.mean()) if valid.size else float("nan")
        rej = int((valid <= reject_thr).sum())
        rate = rej / valid.size if valid.size else float("nan")
        s2 = f" mean_sigma2={_fmt(float(np.mean(all_s2)))}" if all_s2 else ""
        log(f"DONE dataset={dataset} method={method} ok={ok}/{args.iters} failed={failed} "
            f"n_pvals={arr.size} mean_p={_fmt(mean_p)} reject@{reject_thr:.4g}={rej}/{valid.size} "
            f"({_fmt(rate)}){s2}")
    else:
        log(f"DONE dataset={dataset} method={method} ok={ok}/{args.iters} failed={failed} "
            f"(no p-values produced)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SI_DV p-value experiment on real datasets (seqfs fixed-K FS).")
    ap.add_argument("--iters", type=int, default=120, help="iterations per (dataset, method)")
    ap.add_argument("--K", type=int, default=5,
                    help="number of features fixed by the forward selection (|M| = K)")
    ap.add_argument("--lambda", dest="lambda_", type=float, default=10.0,
                    help="unused by seqfs (kept so output names match the Lasso experiment)")
    ap.add_argument("--fraction", type=float, default=0.5, help="DV-score selection fraction")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="significance level for the reject rate; the 'bonferroni' "
                         "method divides it by perm(p, K) = p!/(p-K)!")
    ap.add_argument("--test-size", type=float, default=0.3, help="validation fraction per iteration")
    ap.add_argument("--seed", type=int, default=0, help="base seed (iteration i uses seed+i)")
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    ap.add_argument("--dv", default="cg-score", choices=["cg-score", "lava"],
                    help="data-valuation selection; 'lava' needs the patched scipy simplex")
    ap.add_argument("--n-samples", type=int, default=250,
                    help="number of samples drawn per dataset (0 = use all rows)")
    ap.add_argument("--variance", default="ols", choices=["ols", "identity"],
                    help="noise covariance: 'ols' estimates sigma2 per iteration, "
                         "'identity' uses Sigma = I")
    ap.add_argument("--zmin", type=float, default=-100.0,
                    help="lower scan bound (scaled by sqrt(eta'Sigma eta) internally)")
    ap.add_argument("--zmax", type=float, default=100.0,
                    help="upper scan bound (scaled by sqrt(eta'Sigma eta) internally)")
    ap.add_argument("--quick", action="store_true", help="smoke test: 2 iterations")
    ap.add_argument("--result-dir", default=None,
                    help="output directory (default: realdata/result_seqfs)")
    ap.add_argument("--verbose", action="store_true", help="do not silence library stdout")
    args = ap.parse_args()

    if args.quick:
        args.iters = min(args.iters, 2)

    global RESULT_DIR, LOG_PATH
    if args.result_dir:
        RESULT_DIR = os.path.abspath(args.result_dir)
        LOG_PATH = os.path.join(RESULT_DIR, "si_seqfs_experiment.log")

    os.makedirs(RESULT_DIR, exist_ok=True)
    log("=" * 78)
    log(f"RUN start methods={args.methods} datasets={args.datasets} dv={args.dv} "
        f"fs=seqfs K={args.K} iters={args.iters} lambda={args.lambda_:g} "
        f"fraction={args.fraction:g} n_samples={args.n_samples} "
        f"variance={args.variance} seed={args.seed}")

    raw = [(tag, X, y, names) for (tag, _target, X, y, names, _dropped) in load_all(news_n=100)
           if tag.split("[")[0] in args.datasets]

    for tag, X, y, names in raw:
        # steps 1-2 (per dataset); step 3 (train-only scaler) happens per iteration.
        X, names, dropped = clean_design(X, names, EXTRA_DROP.get(tag.split("[")[0], ()))
        log(f"PREP dataset={tag} n={X.shape[0]} p={X.shape[1]} dropped={dropped}")
        if args.K > X.shape[1]:
            log(f"SKIP dataset={tag}: K={args.K} exceeds p={X.shape[1]} candidate features")
            continue
        # sidecar: design-column index -> feature name (initial feature order).
        sidecar = os.path.join(RESULT_DIR, f"si_features_{tag.replace('[', '_').replace(']', '')}.txt")
        with open(sidecar, "w", encoding="utf-8") as sh:
            sh.write(f"# design column order for dataset={tag} (index\tname)\n")
            sh.write(f"# dropped columns: {dropped}\n")
            for j, nm in enumerate(names):
                sh.write(f"{j}\t{nm}\n")
        ys = y
        if args.n_samples and X.shape[0] > args.n_samples:
            idx = np.sort(np.random.RandomState(args.seed).choice(X.shape[0], args.n_samples, replace=False))
            X, ys = X[idx], ys[idx]
        for method in args.methods:
            t0 = time.time()
            run_one(tag, method, X, ys, args)
            log(f"  ({method} on {tag} took {time.time() - t0:.1f}s)")

    log("RUN end")
    print(f"\nResults -> {RESULT_DIR}\\si_pvals_*.txt")
    print(f"Log     -> {LOG_PATH}")


if __name__ == "__main__":
    main()

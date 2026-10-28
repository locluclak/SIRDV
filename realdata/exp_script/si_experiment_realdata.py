"""Selective-inference (SI_DV) p-value experiment on the selected real datasets.

Datasets (loaded from ``realdata/dataset/`` via ``fit_lasso_realdata.load_all``):
  * concrete          -> target ``Concrete Compressive Strength``
  * energy[Y1], energy[Y2] -> heating / cooling load (both Y columns dropped from X)
  * garments          -> target ``actual_productivity``
  * wine              -> target ``quality`` (non-numeric ``type`` dropped)

The data-valuation selection is chosen with ``--dv {cg-score,lava}`` (default
``cg-score``; ``lava`` requires the patched scipy simplex that returns ``res.basis``).
Methods (all run with ``data_valuation`` set to ``--dv``):
  * overconditioning   -> ``SI_DV(method="overconditioning")``
  * parametric_nested  -> ``SI_DV(method="parametric_nested")``
  * datasplit          -> ``DataSplittingTest``
  * bonferroni         -> naive p-value (``FSNaiveTest``, no selection correction)
                          with the rejection threshold divided by ``2**p`` (p =
                          number of design features); see ``run_one``.  The stored
                          p-values are the raw naive ones; only the threshold
                          changes, so the analysis script can treat it like any
                          other method.

Preprocessing (per dataset, before the iterations)
--------------------------------------------------
  1. keep numeric columns only; drop identifiers / repeated-measures columns
     (garments: ``date``, ``team``) and high-missing columns (``wip``);
  2. drop constant columns and proportional duplicates, then reduce the design
     to full rank (QR with column pivoting) - reporting what was dropped;
  3. inside each iteration the scaler is fit on the TRAIN split only (z-score X
     and y with ``ddof=1``) and applied to the validation split, so validation
     rows never leak into the standardization.

Noise variance
--------------
The SI statistic needs the noise covariance ``Sigma``. For real data it is
estimated per iteration from an ordinary-least-squares fit on the training
design: ``sigma2 = RSS / (n_train - rank(X_train))`` (floored to avoid a
degenerate zero-variance split), and passed as ``Sigmat = Sigmav = sigma2 * I``.
Use ``--variance identity`` to fall back to ``Sigma = I``.

Protocol
--------
Each iteration draws a fresh train / validation split (seeded by ``--seed +
iteration``) and then computes one p-value for EVERY feature in the Lasso active
set ``M`` (not a single random one): the SI methods are called once per position
``jth = 0..|M|-1``; ``datasplit`` reuses the same internal split
(``random_state``) so all positions share one ``M``.

Outputs (under ``realdata/result/``)
------------------------------------
  * ``si_pvals_{tag}_{dv}_{method}_lam{..}_frac{..}_n{..}_{var}.txt`` - one line
    per iteration: ``iter seed |M| f1:p1 f2:p2 ...`` where ``fj`` is the DESIGN
    column index of the tested feature (its initial order) and ``pj`` its
    p-value (appended, so re-running resumes / adds iterations).
  * ``si_features_{tag}.txt`` - the design-column index -> feature-name map
    (and the dropped columns) for that dataset, so ``fj`` is never anonymous.
  * ``si_experiment.log`` - timestamped log of every iteration (including the
    estimated ``sigma2``) plus the full traceback (specific error type and
    message) whenever a call raises.

Examples (run from the repo root, conda env ``IS``):
  # fast smoke test: 2 iterations
  & "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" realdata/exp_script/si_experiment_realdata.py --quick
  # full experiment: 120 iterations per method
  & "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" realdata/exp_script/si_experiment_realdata.py --iters 120

Each dataset is down-sampled to ``--n-samples`` rows (default 250; ``0`` = all rows)
before the iterations, which keeps the per-call cost low.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "realdata", "exp_script"))

from sklearn.model_selection import train_test_split
from scipy.linalg import qr as _qr

from si_for_cg import SI_DV, SelectiveInference, DataSplittingTest, FSNaiveTest
from fit_lasso_realdata import load_all

RESULT_DIR = os.path.join(ROOT, "realdata", "result")
LOG_PATH = os.path.join(RESULT_DIR, "si_experiment.log")

ALL_METHODS = ["overconditioning", "parametric_nested", "datasplit", "bonferroni"]
ALL_DATASETS = ["concrete", "energy", "garments", "wine"]

# Step 1: identifier / repeated-measures columns to drop on top of what the
# loader already removes (``wip`` for garments, the other target for energy).
EXTRA_DROP = {
    "garments": ("date", "team"),
}

SIGMA2_FLOOR = 1e-12


def clean_design(X: np.ndarray, names: list[str], drop_names: tuple[str, ...]):
    """Steps 1-2: drop named columns, constant columns and proportional
    duplicates, then reduce to a full-rank design (QR with column pivoting)."""
    dropped: list[str] = []

    keep = [i for i, nm in enumerate(names) if nm not in drop_names]
    dropped += [names[i] for i in range(len(names)) if i not in keep]
    X, names = X[:, keep], [names[i] for i in keep]

    sd = X.std(axis=0, ddof=1)
    nz = sd > 0
    dropped += [names[i] for i in range(len(names)) if not nz[i]]
    X, names = X[:, nz], [names[i] for i in range(len(names)) if nz[i]]

    if X.shape[1] > 1:
        C = np.corrcoef(X, rowvar=False)
        drop_dup = set()
        for a in range(X.shape[1]):
            for b in range(a + 1, X.shape[1]):
                if b not in drop_dup and np.isfinite(C[a, b]) and abs(C[a, b]) > 1 - 1e-9:
                    drop_dup.add(b)  # proportional to an earlier column
        if drop_dup:
            dropped += [names[i] for i in sorted(drop_dup)]
            keep2 = [i for i in range(X.shape[1]) if i not in drop_dup]
            X, names = X[:, keep2], [names[i] for i in keep2]

    r = int(np.linalg.matrix_rank(X))
    if r < X.shape[1]:
        _, _, piv = _qr(X, mode="economic", pivoting=True)
        keep3 = sorted(int(p) for p in piv[:r])
        dropped += [names[i] for i in range(len(names)) if i not in keep3]
        X, names = X[:, keep3], [names[i] for i in keep3]

    return X, names, dropped


def standardize_train(Xtr: np.ndarray, Xval: np.ndarray, ytr: np.ndarray, yval: np.ndarray):
    """Step 3: z-score X and y with statistics fit on the TRAIN split only."""
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    ymu, ysd = ytr.mean(), ytr.std(ddof=1)
    ysd = ysd if ysd > 0 else 1.0
    return ((Xtr - mu) / sd, (ytr - ymu) / ysd,
            (Xval - mu) / sd, (yval - ymu) / ysd)


def estimate_sigma2(Xtr: np.ndarray, ytr: np.ndarray) -> float:
    """OLS residual variance on the training design (homoscedastic Sigma)."""
    coef, *_ = np.linalg.lstsq(Xtr, ytr, rcond=None)
    resid = ytr - Xtr @ coef
    dof = max(1, Xtr.shape[0] - int(np.linalg.matrix_rank(Xtr)))
    return max(float(resid @ resid) / dof, SIGMA2_FLOOR)


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


@contextlib.contextmanager
def quiet(verbose: bool):
    """Silence the library's verbose stdout (SI_DV prints per call) unless --verbose."""
    if verbose:
        yield
    else:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            yield


def _fmt(v: float) -> str:
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.6g}"


def pvalues_over_support(dataset: str, method: str, Xtr, ytr, Xval, yval,
                         fraction: float, lambda_: float, seed: int, dv: str,
                         sigmat, sigmav, zmin: float, zmax: float,
                         verbose: bool):
    """One p-value per feature in the active set M for a single iteration.

    Returns ``(M, pvals)`` where ``M[j]`` is the DESIGN-COLUMN index of the
    feature tested at position ``j`` (i.e. the p-value carries the feature's
    initial order), and ``pvals[j]`` is its p-value.
    """
    common = dict(fraction=fraction, zmin=zmin, zmax=zmax, lambda_=lambda_,
                  data_valuation=dv, Sigmat=sigmat, Sigmav=sigmav)

    if method == "bonferroni":
        # Naive (non-selective) FS statistic on the DV-selected samples: the
        # p-value carries NO selection correction; the multiplicity is handled
        # only by the rejection threshold alpha / 2**p (see run_one).  M is the
        # Lasso support on Xtrain[observed_selection], in design-column order.
        st = np.eye(Xtr.shape[0]) if sigmat is None else sigmat
        sv = np.eye(Xval.shape[0]) if sigmav is None else sigmav
        runner = FSNaiveTest(fraction=fraction, data_valuation=dv, lambda_=lambda_)
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
                                   lambda_=lambda_, random_state=seed)
        with quiet(verbose):
            p0 = runner.fit(Xtr, ytr, Xval, yval, Sigmat=sigmat, Sigmav=sigmav, jth=0)
            M = [int(x) for x in np.asarray(runner.M_, dtype=int)]
            k = int(runner.n_selected_)
            pvals = [p0]
            for jth in range(1, k):
                pvals.append(runner.fit(Xtr, ytr, Xval, yval,
                                        Sigmat=sigmat, Sigmav=sigmav, jth=jth))
        return M, pvals

    # overconditioning / parametric_nested: the DV-score selection is deterministic
    # (given Xtr/ytr/Xval/yval, all fixed per iteration), so a fresh SI_DV call at
    # each jth reproduces the same M (hence the same feature at each position).
    si0 = SelectiveInference(method=method, lambda_=lambda_, data_valuation=dv,
                             fraction=fraction, zmin=zmin, zmax=zmax)
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

    # Bonferroni: naive p-value, but the rejection threshold is alpha / 2**p
    # (p = number of design features).  Other methods use plain alpha.
    reject_thr = args.alpha / float(2 ** Xs.shape[1]) if method == "bonferroni" else args.alpha

    log(f"START dataset={dataset} method={method} dv={args.dv} iters={args.iters} "
        f"lambda={args.lambda_:g} fraction={args.fraction:g} variance={args.variance} "
        f"n={Xs.shape[0]} p={Xs.shape[1]} reject_thr={reject_thr:.4g} "
        f"-> {os.path.basename(fpath)}")

    with open(fpath, "a", encoding="utf-8") as fh:
        if write_header:
            fh.write(f"# dataset={dataset} method={method} dv={args.dv} lambda={args.lambda_:g} "
                     f"fraction={args.fraction:g} variance={args.variance} "
                     f"n={Xs.shape[0]} p={Xs.shape[1]}\n")
            fh.write("# columns: iter seed |M| f1:p1 f2:p2 ... f|M|:p|M|\n")
            fh.write("# f = feature index in the design (see "
                     f"si_features_{tag}.txt for the index->name map)\n")
            fh.flush()

        for i in range(args.iters):
            seed = args.seed + i
            Xtr, Xval, ytr, yval = train_test_split(
                Xs, ys, test_size=args.test_size, random_state=seed)
            # step 3: scaler fit on TRAIN only, applied to val.
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
                                                args.verbose)
            except Exception as exc:  # noqa: BLE001 - record the specific error, keep going
                failed += 1
                log_error(dataset, method, i, seed, exc)
                continue

            # each entry is ``feature_index:p_value`` (feature index = design column,
            # i.e. the initial order of the feature), so a p-value is never anonymous.
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
    ap = argparse.ArgumentParser(description="SI_DV p-value experiment on real datasets.")
    ap.add_argument("--iters", type=int, default=120, help="iterations per (dataset, method)")
    ap.add_argument("--lambda", dest="lambda_", type=float, default=10.0, help="Lasso lambda")
    ap.add_argument("--fraction", type=float, default=0.5, help="DV-score selection fraction")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="significance level for the reject rate; the 'bonferroni' "
                         "method divides it by 2**p (p = number of design features)")
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
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 2 iterations")
    ap.add_argument("--result-dir", default=None,
                    help="output directory (default: realdata/result); use a temp dir for smoke tests")
    ap.add_argument("--verbose", action="store_true", help="do not silence library stdout")
    args = ap.parse_args()

    if args.quick:
        args.iters = min(args.iters, 2)

    global RESULT_DIR, LOG_PATH
    if args.result_dir:
        RESULT_DIR = os.path.abspath(args.result_dir)
        LOG_PATH = os.path.join(RESULT_DIR, "si_experiment.log")

    os.makedirs(RESULT_DIR, exist_ok=True)
    log("=" * 78)
    log(f"RUN start methods={args.methods} datasets={args.datasets} dv={args.dv} iters={args.iters} "
        f"lambda={args.lambda_:g} fraction={args.fraction:g} n_samples={args.n_samples} "
        f"variance={args.variance} seed={args.seed}")

    raw = [(tag, X, y, names) for (tag, _target, X, y, names, _dropped) in load_all(news_n=100)
           if tag.split("[")[0] in args.datasets]

    for tag, X, y, names in raw:
        # steps 1-2 (per dataset); step 3 happens per iteration inside run_one.
        X, names, dropped = clean_design(X, names, EXTRA_DROP.get(tag.split("[")[0], ()))
        log(f"PREP dataset={tag} n={X.shape[0]} p={X.shape[1]} dropped={dropped}")
        # sidecar: design-column index -> feature name (the initial feature order used
        # by every f:p entry in si_pvals_*.txt for this dataset).
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

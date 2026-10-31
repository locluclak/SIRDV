"""Synthetic FPR/TPR experiment harness for the seqfs (fixed-K forward selection) path.

Measures, per iteration, the selective p-value, the number of selection
intervals consumed by ``compute_p_value``, and the wall-clock time, for all
four SI methods (naive, overconditioning, data-splitting, parametric_nested),
both data valuations (``lava``, ``cg-score``) and the fixed feature-selection
method ``fs_method='seqfs'`` (K=5).

Run from the repo root with the ``IS`` python (see AGENTS.md):

    & "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" experiment/seqfs_experiment.py
    & "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" experiment/seqfs_experiment.py --exp all --iters 3
    & "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" experiment/seqfs_experiment.py --exp tpr_beta --iters 120

Outputs under ``output_seqfs/`` (gitignored like the other ``output*`` dirs),
one aligned value per line, appended and flushed so runs are resumable:

    seqfs_pv_<tag>.txt       <- selective p-value
    seqfs_nint_<tag>.txt     <- len(intervals_)
    time/time_<tag>.txt      <- wall-clock seconds
    seqfs_errors.log         <- seed, method, cfg, exception + traceback

with ``<tag>`` e.g. ``fpr_lava_n50_naive``, ``tpr_lava_beta0.5_overco``,
``tpr_lava_frac0.3_split``.  Raw p-values are stored; Bonferroni vs raw alpha
thresholding is deferred to plot time.

For FPR (null) the data is ``si_utils.generate(n, 15, d, beta=0, beta=0)`` (standard
normal X/noise, identity Sigma).  For TPR (alternative) the data is
``si_utils.generate_contaminated_regression(...)`` and ``Sigmat``/``Sigmav`` are passed.
"""

import argparse
import contextlib
import io
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from si_for_cg import FSNaiveTest, DataSplittingTest
from si_for_cg.si_dv import SelectiveInference
from si_for_cg.si_utils import generate, generate_contaminated_regression

D = 15
K = 5
FRACTION = 0.5
ALPHA = 0.05
ITERS = 120
N_VAL = 15
CONTAMINATION = 0.5
STD_GOOD = 1.0
STD_BAD = 2.0
LAMBDA_ = 5.0

N_LIST = [50, 100, 150, 200]
BETA_LIST = [0.5, 1.0, 1.5, 2.0]
FRACTION_LIST = [0.3, 0.4, 0.5, 0.6]

# tag -> (SelectiveInference method name or None for the baseline classes)
METHODS = {
    "naive": None,
    "overco": "overconditioning",
    "parame": "parametric_nested",
    "split": None,
}


def run_one(tag, dv, fraction, K, Xtr, ytr, Xval, yval, Sigmat, Sigmav):
    """Fit one configuration/seed and return ``(p_value, n_intervals, dt)``.

    Stdout is redirected to suppress the LP-solver/print noise emitted by
    ``si_dv.py``/``si_runner.py`` and the lava path.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        t0 = time.time()
        if tag == "naive":
            runner = FSNaiveTest(fraction=fraction, data_valuation=dv,
                                 fs_method="seqfs", K=K)
        elif tag == "split":
            runner = DataSplittingTest(fraction=fraction, data_valuation=dv,
                                       split_ratio=0.2, fs_method="seqfs", K=K)
        else:
            runner = SelectiveInference(
                method=METHODS[tag], fs_method="seqfs", K=K, lambda_=LAMBDA_,
                data_valuation=dv, lasso_path=False,
            )
        runner.fit(Xtr, ytr, Xval=Xval, yval=yval,
                   Sigmat=Sigmat, Sigmav=Sigmav, jth=None)
        dt = time.time() - t0
    return float(runner.p_value_), len(runner.intervals_), dt


def data_fpr(n, seed):
    X_tr, X_te, y_tr, y_te, St, Sv = generate(
        n, N_VAL, D, np.asarray([0] * D).reshape(-1, 1),
        np.asarray([0] * D).reshape(-1, 1), seed=seed,
    )
    return X_tr, y_tr, X_te, y_te, St, Sv


def data_tpr(beta, seed):
    X_tr, y_tr, X_val, y_val, X_test, y_test, Sigmat, Sigmav, is_clean = \
        generate_contaminated_regression(
            n_train=100, n_val=N_VAL, n_test=0, p=D,
            contamination=CONTAMINATION, beta_clean=beta,
            std_good=STD_GOOD, std_bad=STD_BAD, seed=seed,
        )
    return X_tr, y_tr, X_val, y_val, Sigmat, Sigmav


def log_error(log_path, seed, tag, cfg, exc):
    with open(log_path, "a") as f:
        f.write(f"seed={seed} method={tag} cfg={cfg} "
                f"exc={type(exc).__name__}: {exc}\n")
        f.write(traceback.format_exc())
        f.write("-" * 60 + "\n")
        f.flush()


def run_config(method_tag, file_tag, dv, fraction, data_fn, iters, K, outdir,
               errors_log, metric):
    pv_path = os.path.join(outdir, f"seqfs_pv_{file_tag}.txt")
    nint_path = os.path.join(outdir, f"seqfs_nint_{file_tag}.txt")
    time_path = os.path.join(outdir, "time", f"time_{file_tag}.txt")

    valid_count = 0
    rejected_count = 0
    error_count = 0
    times = []
    nints = []

    print(f"\n{'='*60}")
    print(f"Method: {method_tag}  ({metric})  tag={file_tag}")
    print(f"{'='*60}")

    with open(pv_path, "a") as fpv, open(nint_path, "a") as fnint, \
            open(time_path, "a") as ftime:
        i = 0
        seedd = 0
        while i < iters:
            seedd += 1
            data = data_fn(seedd)
            try:
                pv, nint, dt = run_one(method_tag, dv, fraction, K, *data)
            except Exception as exc:  # noqa: BLE001
                log_error(errors_log, seedd, method_tag, file_tag, exc)
                error_count += 1
                continue
            fpv.write(f"{pv}\n")
            fnint.write(f"{nint}\n")
            ftime.write(f"{dt}\n")
            fpv.flush()
            fnint.flush()
            ftime.flush()
            i += 1

            if np.isfinite(pv):
                valid_count += 1
                threshold = ALPHA / 2 ** D if method_tag == "naive" else ALPHA
                if pv <= threshold:
                    rejected_count += 1
            times.append(dt)
            nints.append(nint)

            if i % 20 == 0:
                print(f"  {i}/{iters} done, {valid_count} valid, "
                      f"{rejected_count} rejected, {error_count} errors")

    rate = rejected_count / valid_count if valid_count > 0 else float("nan")
    mean_time = float(np.mean(times)) if times else float("nan")
    mean_nint = float(np.mean(nints)) if nints else float("nan")
    print(f"  Saved to {os.path.basename(pv_path)} (+ nint + time)")
    print(f"  {metric} = {rejected_count}/{valid_count} = {rate:.4f}")
    print(f"  valid={valid_count} rejected={rejected_count} "
          f"errors={error_count}/{iters + error_count} "
          f"mean_time={mean_time:.3f}s mean_n_intervals={mean_nint:.3f}")


def run_fpr(dv, methods, iters, K, outdir, errors_log):
    for n in N_LIST:
        for tag in methods:
            run_config(tag, f"fpr_{dv}_n{n}_{tag}", dv, FRACTION,
                       lambda seed, n=n: data_fpr(n, seed),
                       iters, K, outdir, errors_log, "FPR")


def run_tpr_beta(dv, methods, iters, K, outdir, errors_log):
    for beta in BETA_LIST:
        for tag in methods:
            run_config(tag, f"tpr_{dv}_beta{beta}_{tag}", dv, FRACTION,
                       lambda seed, b=beta: data_tpr(b, seed),
                       iters, K, outdir, errors_log, "TPR")


def run_tpr_fraction(dv, methods, iters, K, outdir, errors_log):
    for frac in FRACTION_LIST:
        for tag in methods:
            run_config(tag, f"tpr_{dv}_frac{frac}_{tag}", dv, frac,
                       lambda seed: data_tpr(1.5, seed),
                       iters, K, outdir, errors_log, "TPR")


def main():
    global D
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="all",
                    choices=["fpr", "tpr_beta", "tpr_fraction", "all"])
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--K", type=int, default=K)
    ap.add_argument("--d", type=int, default=D)
    ap.add_argument("--outdir", default="output_seqfs")
    args = ap.parse_args()

    D = args.d

    outdir = os.path.join(os.path.dirname(__file__), "..", args.outdir)
    os.makedirs(os.path.join(outdir, "time"), exist_ok=True)
    errors_log = os.path.join(outdir, "seqfs_errors.log")

    exps = ["fpr", "tpr_beta", "tpr_fraction"] if args.exp == "all" else [args.exp]
    dvs = ["lava", "cg-score"]
    methods = list(METHODS.keys())

    for exp in exps:
        for dv in dvs:
            if exp == "fpr":
                run_fpr(dv, methods, args.iters, args.K, outdir, errors_log)
            elif exp == "tpr_beta":
                run_tpr_beta(dv, methods, args.iters, args.K, outdir, errors_log)
            else:
                run_tpr_fraction(dv, methods, args.iters, args.K, outdir, errors_log)


if __name__ == "__main__":
    main()

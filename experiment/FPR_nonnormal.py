"""FPR of selective inference under non-normal / estimated noise, LAVA + fixed-K seqFS.

Ported from a sibling project's ``FPR_nonnormal.py`` (which drove ``DA_SeqFS`` /
``SI_SeqFS_DA``) onto this repo's ``SI_DV`` call path.  Mirrors
``experiment/fpr_robustness.py`` in structure -- one p-value per repetition,
append + summary file, resumable across runs -- but:

  * data valuation is fixed to ``lava`` (the OT-LP instance-selection path);
  * the second-stage feature selection is the fixed-K greedy forward selection
    (``fs_method='seqfs'``) rather than the Lasso path;
  * the SI method is ``parametric_nested`` (the seqfs overconditioning interval
    is re-solved per z across all candidates).

Noise scenarios (all standardized to mean 0, variance 1, with beta = 0 so the
global null holds and the tested feature is truly null):

  laplace       : Laplace(0, 1/sqrt(2))        -- symmetric, heavy tails.
  skewnorm10    : centered/scaled skew-normal (shape a = 2.0).
  t20           : Student-t(20) rescaled to unit variance -- mild heavy tails.
  normal_estvar : standard normal noise, but the SI call receives a diagonal
                  covariance whose variance is estimated empirically from the
                  data (``np.var(y, ddof=1)`` for train and validation
                  separately) instead of the default identity Sigma.

Outputs land in ``output_nonnormal/`` (repo root):
  fpr_seqfs_lava_{scenario}_ns{n}_K{K}.txt   one p-value per line (append)
  fpr_summary_nonnormal.txt                  one summary line per finished config

Run from the repo root with the IS interpreter (see AGENTS.md), e.g.

    python experiment/FPR_nonnormal.py                       # full spec
    python experiment/FPR_nonnormal.py --iters 3 --ns 100 \
        --setups laplace,normal_estvar --quiet               # quick smoke test
    python experiment/FPR_nonnormal.py --setups all --K 5 --output-dir output_nonnormal2

The ``lava`` path requires the patched scipy ``linprog`` (see
``replace_scipy_linprog.py``); the ``seqfs`` path itself ignores ``lambda_``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
import traceback

import numpy as np
from scipy.stats import skewnorm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from si_for_cg import SI_DV


ALPHA = 0.05
REPETITIONS = 200

NS_LIST = [50, 100, 150, 200]
NT = 20
P = 5
K = 3

DV = "lava"
METHOD = "parametric_nested"

# Noise constants (all give mean 0, variance 1).
LAPLACE_SCALE = 1.0 / np.sqrt(2.0)
T20_STD = np.sqrt(20.0 / 18.0)
SKEW_PARAM = 2.0
SKEW_MEAN, SKEW_VAR = skewnorm.stats(SKEW_PARAM, moments="mv")

# setup name -> (noise distribution, variance mode)
SETUPS = {
    "laplace": ("laplace", "known"),
    "skewnorm10": ("skewnorm10", "known"),
    "t20": ("t20", "known"),
    "normal_estvar": ("normal", "estimated"),
}
SETUP_NAMES = list(SETUPS)

DEFAULT_OUT_DIR = "output_nonnormal"


def _standardized_noise(kind: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``size`` i.i.d. noise values with mean 0 and (theoretical) variance 1."""
    if kind == "normal":
        return rng.normal(loc=0.0, scale=1.0, size=size)
    if kind == "laplace":
        return rng.laplace(loc=0.0, scale=LAPLACE_SCALE, size=size)
    if kind == "t20":
        return rng.standard_t(df=20, size=size) / T20_STD
    if kind == "skewnorm10":
        raw = skewnorm.rvs(SKEW_PARAM, size=size, random_state=rng)
        return (raw - SKEW_MEAN) / np.sqrt(SKEW_VAR)
    raise ValueError(f"unknown noise kind: {kind!r}")


def generate_noise(
    n: int,
    m: int,
    p: int,
    noise: str = "normal",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Null data (beta = 0): Xs (n x p), Xt (m x p), Ys (n x 1), Yt (m x 1)."""
    rng = np.random.default_rng(seed)
    Xs = rng.normal(loc=0.0, scale=1.0, size=(n, p))
    Xt = rng.normal(loc=0.0, scale=1.0, size=(m, p))
    # ``normal_estvar`` is a covariance-handling scenario, not a noise shape:
    # draw standard normal noise and estimate the variance from the sample below.
    draw = "normal" if noise == "normal_estvar" else noise
    Ys = _standardized_noise(draw, n, rng).reshape(n, 1)
    Yt = _standardized_noise(draw, m, rng).reshape(m, 1)
    return Xs, Xt, Ys, Yt


def empirical_variance_sigma(y: np.ndarray) -> np.ndarray:
    """Diagonal noise covariance with the empirical (unbiased) variance of ``y``.

    Under the global null (beta = 0) the response is pure noise, so the sample
    variance of ``y`` estimates the noise variance.
    """
    var = float(np.var(np.asarray(y).reshape(-1), ddof=1))
    var = max(var, 1e-12)
    return var * np.eye(np.asarray(y).reshape(-1).shape[0])


def run_config(
    setup: str,
    n: int,
    nval: int,
    d: int,
    fraction: float,
    lambda_: float,
    K: int,
    zmin: float,
    zmax: float,
    iters: int,
    alphas: list[float],
    outdir: str,
    quiet: bool,
) -> None:
    noise, variance_mode = SETUPS[setup]
    fname = f"fpr_seqfs_{DV}_{setup}_ns{n}_K{K}.txt"
    fpath = os.path.join(outdir, fname)

    valid_count = 0
    error_count = 0
    reject_counts = {a: 0 for a in alphas}

    with open(fpath, "a") as f:
        i = 0
        seedd = 0
        while i < iters:
            seedd += 1
            X_tr, X_te, y_tr, y_te = generate_noise(n, nval, d, noise=noise, seed=seedd)

            if variance_mode == "estimated":
                Sigmat = empirical_variance_sigma(y_tr)
                Sigmav = empirical_variance_sigma(y_te)
            else:
                Sigmat, Sigmav = None, None

            start_t = time.time()
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf if quiet else sys.stdout):
                    pv = SI_DV(
                        X_tr, y_tr, Xval=X_te, yval=y_te,
                        Sigmat=Sigmat, Sigmav=Sigmav,
                        fraction=fraction, jth=None,
                        zmin=zmin, zmax=zmax,
                        lambda_=lambda_,
                        method=METHOD, data_valuation=DV,
                        lasso_path=False, fs_method="seqfs", K=K,
                    )
            except Exception as error:  # empty/degenerate selections are expected
                if not quiet:
                    print(f"  [{DV}/{setup}/n{n}] seed {seedd} error: {error}")
                    traceback.print_exc()
                error_count += 1
                continue
            end_t = time.time()

            f.write(f"{pv}\n")
            f.flush()
            i += 1
            if not np.isnan(pv):
                valid_count += 1
                for a in alphas:
                    if pv <= a:
                        reject_counts[a] += 1

            print(f"  [{DV}/{setup}/n{n}/K{K}] seed {seedd} pv={pv:.4g} "
                  f"({end_t - start_t:.2f}s)  i={i}/{iters}")

    print(f"  Saved to {fname}")
    summary_bits = []
    for a in alphas:
        fpr = reject_counts[a] / valid_count if valid_count > 0 else float("nan")
        summary_bits.append(f"alpha={a}:{reject_counts[a]}/{valid_count}={fpr:.4f}")
        print(f"  FPR(alpha={a}) = {reject_counts[a]}/{valid_count} = {fpr:.4f}")
    print(f"  dropped iterations (exceptions): {error_count}/{iters}")

    summary_fpath = os.path.join(outdir, "fpr_summary_nonnormal.txt")
    with open(summary_fpath, "a") as fs:
        fs.write(f"{DV}\t{setup}\tseqfs\tns{n}\tK={K}\tvalid={valid_count}\t"
                 f"errors={error_count}\t" + "\t".join(summary_bits) + "\n")


def resolve_setups(spec: str) -> list[str]:
    spec = spec.strip()
    if spec.lower() == "all":
        return list(SETUP_NAMES)
    names = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = [name for name in names if name not in SETUPS]
    if unknown:
        raise ValueError(f"unknown setups: {unknown}; choices: {SETUP_NAMES}")
    return names


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--setups", default="all",
                   help="comma-separated noise scenarios to run, or 'all' "
                        f"(choices: {','.join(SETUP_NAMES)})")
    p.add_argument("--ns", default=",".join(str(x) for x in NS_LIST),
                   help="comma-separated training sizes")
    p.add_argument("--alpha", default="0.05,0.1",
                   help="comma-separated alpha levels")
    p.add_argument("--iters", type=int, default=REPETITIONS,
                   help="repetitions per config")
    p.add_argument("--nval", type=int, default=NT)
    p.add_argument("--d", type=int, default=P)
    p.add_argument("--K", type=int, default=K,
                   help="number of forward-selected features for fs_method='seqfs'")
    p.add_argument("--fraction", type=float, default=0.5)
    p.add_argument("--lambda_", type=float, default=5.0,
                   help="Lasso strength (ignored on the seqfs path; kept for parity)")
    p.add_argument("--zmin", type=float, default=-20.0)
    p.add_argument("--zmax", type=float, default=20.0)
    p.add_argument("--output-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--quiet", action="store_true",
                   help="suppress SI/LP solver stdout noise")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setups = resolve_setups(args.setups)
    ns_list = [int(s) for s in args.ns.split(",") if s.strip()]
    alphas = [float(s) for s in args.alpha.split(",") if s.strip()]

    outdir = os.path.join(os.path.dirname(__file__), "..", args.output_dir)
    outdir = os.path.normpath(outdir)
    os.makedirs(outdir, exist_ok=True)

    print("Non-normal FPR (lava / seqfs / fixed-K forward selection)")
    print(f"  setups    : {setups}")
    print(f"  ns        : {ns_list}")
    print(f"  K (seqfs) : {args.K}")
    print(f"  alphas    : {alphas}")
    print(f"  iters     : {args.iters}")
    print(f"  nval,d,fraction,lambda_ : {args.nval},{args.d},{args.fraction},{args.lambda_}")
    print(f"  output    : {outdir}")

    for setup in setups:
        for n in ns_list:
            print(f"\n{'=' * 60}\n{DV} | {setup} | n={n} | K={args.K}\n{'=' * 60}")
            run_config(
                setup=setup, n=n, nval=args.nval, d=args.d,
                fraction=args.fraction, lambda_=args.lambda_, K=args.K,
                zmin=args.zmin, zmax=args.zmax, iters=args.iters,
                alphas=alphas, outdir=outdir, quiet=args.quiet,
            )


if __name__ == "__main__":
    main()

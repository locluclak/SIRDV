"""FPR robustness of ``parametric_nested`` SI under non-normal / estimated noise.

Reference: ``experiment/fpr_experiment.py`` (same SI_DV call path, but here we
only exercise ``method="parametric_nested"``).  For each data-valuation method
(``lava`` and ``cg-score``), training size, and noise scenario we collect one
p-value per repetition and report the empirical FPR at alpha = 0.05 and 0.10.

Noise scenarios (all standardized to mean 0, variance 1, with beta = 0 so the
global null holds):

  normal  : standard normal control.
  laplace : Laplace(0, 1/sqrt(2))            -- symmetric, heavy tails.
  skew10  : centered/scaled Gamma(k=0.04)    -- skewness coefficient 10.
  t20     : Student-t(20) rescaled to unit variance -- mild heavy tails.
  estcov  : standard normal noise, but the SI call receives a *diagonal
            covariance whose variance is estimated empirically from the data*
            (``np.var(y, ddof=1)`` for train and validation separately) instead
            of the default identity Sigma.

Outputs land in ``output_robustness/`` (repo root):
  fpr_{dv}_parame_{scenario}_n{n}.txt   one p-value per line (append/resumable)
  fpr_summary_robustness.txt            one summary line per finished config

Run from the repo root with the IS interpreter, e.g.

    python experiment/fpr_robustness.py                 # full spec
    python experiment/fpr_robustness.py --iters 3 --ns 100 \
        --dvs lava,cg-score --noises laplace,estcov      # quick smoke test
    python experiment/fpr_robustness.py --dvs lava --noises skew10 --output-dir output_robustness2 --lambda_ 1
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from si_for_cg import SI_DV


# --------------------------------------------------------------------------- #
# Data generation with selectable noise distribution
# --------------------------------------------------------------------------- #
def _standardized_noise(kind: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``size`` i.i.d. noise values with mean 0 and (theoretical) variance 1."""
    if kind == "normal":
        return rng.normal(loc=0.0, scale=1.0, size=size)
    if kind == "laplace":
        # Var = 2 b^2 = 1  =>  b = 1/sqrt(2).
        return rng.laplace(loc=0.0, scale=1.0 / np.sqrt(2.0), size=size)
    if kind == "skew10":
        # Gamma(k, 1) has skewness 2/sqrt(k); k = (2/10)^2 = 0.04 gives 10.
        k = 0.04
        g = rng.gamma(shape=k, scale=1.0, size=size)
        return (g - k) / np.sqrt(k)
    if kind == "t20":
        # Var(t_20) = df/(df-2) = 20/18; rescale to unit variance.
        return rng.standard_t(df=20, size=size) / np.sqrt(20.0 / 18.0)
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
    # ``estcov`` is a covariance-handling scenario, not a noise shape: draw
    # standard normal noise and estimate the variance from the sample below.
    draw = "normal" if noise == "estcov" else noise
    Ys = _standardized_noise(draw, n, rng).reshape(n, 1)
    Yt = _standardized_noise(draw, m, rng).reshape(m, 1)
    return Xs, Xt, Ys, Yt


def empirical_variance_sigma(y: np.ndarray) -> np.ndarray:
    """Diagonal noise covariance with the empirical (unbiased) variance of ``y``.

    Under the global null (beta = 0) the response is pure noise, so the sample
    variance of ``y`` is an estimate of the noise variance.
    """
    var = float(np.var(np.asarray(y).reshape(-1), ddof=1))
    var = max(var, 1e-12)
    return var * np.eye(np.asarray(y).reshape(-1).shape[0])


# --------------------------------------------------------------------------- #
def run_config(
    dv: str,
    noise: str,
    n: int,
    nval: int,
    d: int,
    fraction: float,
    lambda_: float,
    zmin: float,
    zmax: float,
    iters: int,
    alphas: list[float],
    outdir: str,
    quiet: bool,
) -> None:
    fname = f"fpr_{dv}_parame_{noise}_n{n}.txt"
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

            if noise == "estcov":
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
                        method="parametric_nested",
                        data_valuation=dv,
                    )
            except Exception as error:  # expected on degenerate selections
                if not quiet:
                    print(f"An error occurred: {error}")
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

            print(f"  [{dv}/{noise}/n{n}] seed {seedd} pv={pv:.4g} "
                  f"({end_t - start_t:.2f}s)  i={i}/{iters}")

    print(f"  Saved to {fname}")
    summary_bits = []
    for a in alphas:
        fpr = reject_counts[a] / valid_count if valid_count > 0 else float("nan")
        summary_bits.append(f"alpha={a}:{reject_counts[a]}/{valid_count}={fpr:.4f}")
        print(f"  FPR(alpha={a}) = {reject_counts[a]}/{valid_count} = {fpr:.4f}")
    print(f"  dropped iterations (exceptions): {error_count}/{iters}")

    summary_fpath = os.path.join(outdir, "fpr_summary_robustness.txt")
    with open(summary_fpath, "a") as fs:
        fs.write(f"{dv}\t{noise}\tn{n}\tvalid={valid_count}\terrors={error_count}\t"
                 + "\t".join(summary_bits) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dvs", default="lava,cg-score",
                   help="comma-separated data_valuation methods")
    p.add_argument("--noises", default="normal,laplace,skew10,t20,estcov",
                   help="comma-separated noise scenarios")
    p.add_argument("--ns", default="100,150,200,250",
                   help="comma-separated training sizes")
    p.add_argument("--alpha", default="0.05,0.1",
                   help="comma-separated alpha levels")
    p.add_argument("--iters", type=int, default=120, help="repetitions per config")
    p.add_argument("--nval", type=int, default=20)
    p.add_argument("--d", type=int, default=15)
    p.add_argument("--fraction", type=float, default=0.5)
    p.add_argument("--lambda_", type=float, default=5.0)
    p.add_argument("--zmin", type=float, default=-20.0)
    p.add_argument("--zmax", type=float, default=20.0)
    p.add_argument("--output-dir", default="output_robustness")
    p.add_argument("--quiet", action="store_true",
                   help="suppress SI/LP solver stdout noise")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dvs = [s.strip() for s in args.dvs.split(",") if s.strip()]
    noises = [s.strip() for s in args.noises.split(",") if s.strip()]
    ns = [int(s) for s in args.ns.split(",") if s.strip()]
    alphas = [float(s) for s in args.alpha.split(",") if s.strip()]

    outdir = os.path.join(os.path.dirname(__file__), "..", args.output_dir)
    outdir = os.path.normpath(outdir)
    os.makedirs(outdir, exist_ok=True)

    print(f"FPR robustness of parametric_nested")
    print(f"  dv        : {dvs}")
    print(f"  noises    : {noises}")
    print(f"  ns        : {ns}")
    print(f"  alphas    : {alphas}")
    print(f"  iters     : {args.iters}")
    print(f"  nval,d,fraction,lambda_ : {args.nval},{args.d},{args.fraction},{args.lambda_}")
    print(f"  output    : {outdir}")

    for dv in dvs:
        for noise in noises:
            for n in ns:
                print(f"\n{'=' * 60}\n{dv} | noise={noise} | n={n}\n{'=' * 60}")
                run_config(
                    dv=dv, noise=noise, n=n, nval=args.nval, d=args.d,
                    fraction=args.fraction, lambda_=args.lambda_,
                    zmin=args.zmin, zmax=args.zmax, iters=args.iters,
                    alphas=alphas, outdir=outdir, quiet=args.quiet,
                )


if __name__ == "__main__":
    main()

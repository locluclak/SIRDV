"""FPR of selective inference in the high-dimensional regime (p > nt).

Mirrors ``experiment/fpr_experiment.py``: same null setup (beta = 0, unit
normal noise, ``Sigmat`` omitted -> identity), same ``SI_DV`` call path and
per-config append-and-resume loop.  The differences are the high-dimensional
setting and a fixed Lasso strength:

  * p (features)     : 400, 600, 800, 1000
  * nt (train)       : 200
  * nv (validation)  : 20
  * lambda_          : 5   (chosen from experiment/find_lambda_highdim.py)
  * fraction         : 0.5
  * alpha            : 0.05 (primary) and 0.1 (also reported)

At ``fraction=0.5`` the stage-2 Lasso is fit on ~100 DV-selected rows, so its
active set ``|M|`` stays below ~100 (well under nt), which keeps the
``make_eta_*`` inversion ``X[:,M]^T X[:,M]`` full rank.  With lambda_ = 5 the
observed |M| is ~70 (p=400) to ~80 (p=1000), so this is a deliberately
*hard* high-dimensional regime: near-saturated designs plus a large active set.

Outputs (repo root, default ``output_highdim/``):
  fpr_{dv}_{method}_p{p}_lambda{lambda_}.txt   one p-value per line (resumable)
  fpr_highdim_summary.txt                      one summary line per finished config

Run from the repo root with the IS interpreter, e.g.

    python experiment/fpr_highdim.py                       # full spec
    python experiment/fpr_highdim.py --ps 400 --iters 2 \
        --dvs lava,cg-score --methods overconditioning     # quick smoke test

Note: ``--methods parametric_nested`` is accepted but is far slower at these
active-set sizes (|M| ~ 70-80); prefer it only for small p / few iters.
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
from si_for_cg.si_utils import generate


def run_config(
    dv: str,
    method: str,
    p: int,
    nt: int,
    nv: int,
    fraction: float,
    lambda_: float,
    zmin: float,
    zmax: float,
    iters: int,
    alphas: list[float],
    outdir: str,
    quiet: bool,
) -> None:
    fname = f"fpr_{dv}_{method[:6]}_p{p}_lambda{lambda_:g}.txt"
    fpath = os.path.join(outdir, fname)

    valid_count = 0
    error_count = 0
    reject_counts = {a: 0 for a in alphas}
    beta0 = np.zeros((p, 1))

    with open(fpath, "a") as f:
        i = 0
        seedd = 0
        while i < iters:
            seedd += 1
            X_tr, X_te, y_tr, y_te, _St, _Sv = generate(nt, nv, p, beta0, beta0, seed=seedd)

            start_t = time.time()
            try:
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf if quiet else sys.stdout):
                    pv = SI_DV(
                        X_tr, y_tr, Xval=X_te, yval=y_te,
                        fraction=fraction, jth=None,
                        zmin=zmin, zmax=zmax,
                        lambda_=lambda_,
                        method=method, data_valuation=dv,
                    )
            except Exception as error:  # empty/degenerate selections are expected
                print(f"  [{dv}/{method}/p{p}] seed {seedd} error: {error}")
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

            print(f"  [{dv}/{method}/p{p}] seed {seedd} pv={pv:.4g} "
                  f"({end_t - start_t:.2f}s)  i={i}/{iters}")

    print(f"  Saved to {fname}")
    summary_bits = []
    for a in alphas:
        fpr = reject_counts[a] / valid_count if valid_count > 0 else float("nan")
        summary_bits.append(f"alpha={a}:{reject_counts[a]}/{valid_count}={fpr:.4f}")
        print(f"  FPR(alpha={a}) = {reject_counts[a]}/{valid_count} = {fpr:.4f}")
    print(f"  dropped iterations (exceptions): {error_count}/{iters}")

    with open(os.path.join(outdir, "fpr_highdim_summary.txt"), "a") as fs:
        fs.write(f"{dv}\t{method}\tp{p}\tlambda={lambda_:g}\tvalid={valid_count}\t"
                 f"errors={error_count}\t" + "\t".join(summary_bits) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dvs", default="lava,cg-score",
                    help="comma-separated data_valuation methods")
    ap.add_argument("--methods", default="overconditioning",
                    help="comma-separated SI methods (overconditioning, parametric_nested)")
    ap.add_argument("--ps", default="400,600,800,1000",
                    help="comma-separated feature dimensions")
    ap.add_argument("--nt", type=int, default=200)
    ap.add_argument("--nv", type=int, default=20)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--lambda_", type=float, default=5.0)
    ap.add_argument("--alpha", default="0.05,0.1",
                    help="comma-separated alpha levels")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--zmin", type=float, default=-20.0)
    ap.add_argument("--zmax", type=float, default=20.0)
    ap.add_argument("--output-dir", default="output_highdim")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress SI/LP solver stdout noise")
    args = ap.parse_args()

    dvs = [s.strip() for s in args.dvs.split(",") if s.strip()]
    methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    ps = [int(s) for s in args.ps.split(",") if s.strip()]
    alphas = [float(s) for s in args.alpha.split(",") if s.strip()]

    outdir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", args.output_dir))
    os.makedirs(outdir, exist_ok=True)

    print("High-dim FPR experiment")
    print(f"  dvs       : {dvs}")
    print(f"  methods   : {methods}")
    print(f"  ps        : {ps}")
    print(f"  nt, nv    : {args.nt}, {args.nv}")
    print(f"  fraction  : {args.fraction}")
    print(f"  lambda_   : {args.lambda_}")
    print(f"  alphas    : {alphas}")
    print(f"  iters     : {args.iters}")
    print(f"  output    : {outdir}")

    for dv in dvs:
        for method in methods:
            for p in ps:
                print(f"\n{'=' * 60}\n{dv} | {method} | p={p}\n{'=' * 60}")
                run_config(
                    dv=dv, method=method, p=p, nt=args.nt, nv=args.nv,
                    fraction=args.fraction, lambda_=args.lambda_,
                    zmin=args.zmin, zmax=args.zmax, iters=args.iters,
                    alphas=alphas, outdir=outdir, quiet=args.quiet,
                )


if __name__ == "__main__":
    main()

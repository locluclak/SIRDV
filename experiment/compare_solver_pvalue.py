"""Do SI p-values change when the OT LP is solved by HiGHS instead of scipy?

Same data, same pipeline; only ``OptimalTransport.solveOT``'s backend differs.
Because the OT LP is degenerate the two backends may return different (both
optimal) bases, which can change the LAVA scores, the selected top-fraction
set, and hence the p-value.

Reports, per seed and method: the selected index set under each backend, the
two p-values and their difference.

Run from the repo root with the ``IS`` python (see AGENTS.md):
    python experiment/compare_solver_pvalue.py --methods overconditioning
    python experiment/compare_solver_pvalue.py --methods overconditioning parametric_nested --seeds 3
"""

import argparse
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

import si_for_cg.OptimalTransport as OTmod
from si_for_cg import SI_DV
from si_for_cg.si_adapter import SelectionAdapter

_ORIG_SOLVEOT = OTmod.solveOT


def force_solver(name):
    """Make OptimalTransport.solveOT use ``name`` regardless of caller."""
    def wrapper(*args, **kwargs):
        kwargs["solver"] = name
        return _ORIG_SOLVEOT(*args, **kwargs)
    OTmod.solveOT = wrapper


def generate(n, m, p, seed):
    rng = np.random.RandomState(seed)
    Xs = rng.normal(0, 1, size=(n, p))
    Xt = rng.normal(0, 1, size=(m, p))
    ys = rng.normal(0, 1, size=n)
    yt = rng.normal(0, 1, size=m)
    return Xs, ys, Xt, yt


def selected_set(Xs, ys, Xt, yt, fraction, solver):
    force_solver(solver)
    np.random.seed(0)
    ad = SelectionAdapter(fraction=fraction)
    idx = ad.select(Xs, ys, Xt, yt)
    return set(int(i) for i in idx)


def pvalue(Xs, ys, Xt, yt, fraction, method, solver, jth):
    force_solver(solver)
    np.random.seed(0)  # pin any internal RNG; jth is fixed so results are determinism
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return float(SI_DV(Xs, ys, Xval=Xt, yval=yt, fraction=fraction, jth=jth,
                               zmin=-20.0, zmax=20.0, lambda_=5.0, data_valuation="lava",
                               method=method, lasso_path=False))
    except Exception:  # noqa: BLE001
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=6, help="number of seeds")
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--nv", type=int, default=40)
    ap.add_argument("--d", type=int, default=15)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--methods", nargs="*", default=["overconditioning"])
    ap.add_argument("--jth", type=int, default=0,
                    help="tested feature position (fixed; None would be random)")
    args = ap.parse_args()

    print(f"n={args.n} nv={args.nv} d={args.d} fraction={args.fraction}")
    for method in args.methods:
        print("=" * 88)
        print(f"method = {method}")
        print("=" * 88)
        print(f"  {'seed':>5}{'same_selection':>16}{'p_scipy':>14}{'p_highs':>14}"
              f"{'|dp|':>12}")
        n_same = n_cmp = 0
        dp_max = 0.0
        for k in range(args.seeds):
            seed = args.seed0 + k
            Xs, ys, Xt, yt = generate(args.n, args.nv, args.d, seed)
            sel_s = selected_set(Xs, ys, Xt, yt, args.fraction, "scipy")
            sel_h = selected_set(Xs, ys, Xt, yt, args.fraction, "highs")
            same = sel_s == sel_h
            ps = pvalue(Xs, ys, Xt, yt, args.fraction, method, "scipy", args.jth)
            ph = pvalue(Xs, ys, Xt, yt, args.fraction, method, "highs", args.jth)
            both = not (np.isnan(ps) or np.isnan(ph))
            if both:
                n_cmp += 1
                dp_max = max(dp_max, abs(ps - ph))
            n_same += int(same)
            dstr = f"{abs(ps - ph):.3e}" if both else "n/a"
            pstr = f"{ps:.6f}" if not np.isnan(ps) else "nan"
            hstr = f"{ph:.6f}" if not np.isnan(ph) else "nan"
            print(f"  {seed:>5}{str(same):>16}{pstr:>14}{hstr:>14}{dstr:>12}")
        print(f"  -- same selection: {n_same}/{args.seeds}  "
              f"| max |dp| over {n_cmp} comparable runs: {dp_max:.3e}")
        print()


if __name__ == "__main__":
    main()

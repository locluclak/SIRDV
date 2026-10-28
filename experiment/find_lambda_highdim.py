"""Find a Lasso regularization strength for the high-dimensional SI experiment.

Pipeline context
----------------
The SI pipeline (``SI_DV``) fits a *second-stage* Lasso on the DV-selected
training subset, then ``make_etaFS`` / ``make_eta_cgFS`` invert
``X[:, M]^T X[:, M]`` using the FULL design:

  * cg-score : X = Xtrain            -> nt     rows
  * lava     : X = vstack(train,val) -> nt+nv  rows

so the eta inversion tolerates ``|M| <= nt`` (cg) / ``nt+nv`` (lava).  But the
Lasso support ``M`` comes from a Lasso fitted on ``Xtrain[observed_selection]``
(``si_dv.py`` ``solve_Lasso`` call), i.e. on the DV-selected subset only, which
has ``n_selected = fraction * nt`` rows.  A Lasso has at most
``min(n_samples, n_features)`` nonzeros, so at ``fraction=0.5`` the active set
always satisfies ``|M| <= n_selected ~ 100`` and the literal target
``|M| < nt = 200`` is met for every ``lambda_``.  The meaningful control is to
keep ``|M|`` comfortably below the ~100 fit rows.

What this script does
---------------------
For each (dv, p, lambda) it draws ``--iters`` null datasets (beta = 0, normal
noise, same as ``fpr_experiment.py``), runs the DV selection once per draw, and
fits the second-stage Lasso at every ``lambda`` on the *same* selected subset
(selection does not depend on lambda, so it is reused).  It records the active
set size ``|M|`` and ``n_selected`` and prints the distribution plus a
recommended shared ``lambda_`` (one value for all p, honoring a requested
target), matching the ``lambda_`` scale ``SI_DV`` takes (internally sklearn
``alpha = lambda_ / n_fit``).

Outputs (repo root, default ``output_lambda_highdim/``):
  lambda_summary.tsv   dv, p, lambda, nsel_mean, count, mean/median/q90/q95/max |M|, empty
  lambda_recommend.txt shared-lambda recommendations per target

Run from the repo root with the IS interpreter, e.g.

    python experiment/find_lambda_highdim.py                 # full spec
    python experiment/find_lambda_highdim.py --p 400 --iters 3 \
        --lambdas 1,5,20 --dvs cg-score                      # quick smoke test
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

from si_for_cg.si_adapter import CGSelectionAdapter, SelectionAdapter
from si_for_cg.si_utils import generate, solve_Lasso


# --------------------------------------------------------------------------- #
def select_indices(
    dv: str,
    Xtrain: np.ndarray,
    ytrain: np.ndarray,
    Xval: np.ndarray,
    yval: np.ndarray,
    fraction: float,
    quiet: bool,
) -> list[int]:
    """Run the DV-score selection and return the kept training indices."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf if quiet else sys.stdout):
        if dv == "lava":
            adapter = SelectionAdapter(fraction=fraction)
            sel = adapter.select(Xtrain, ytrain, Xval, yval)
        elif dv == "cg-score":
            adapter = CGSelectionAdapter(fraction=fraction)
            sel = adapter.select(Xtrain, ytrain)
        else:
            raise ValueError(f"unknown data_valuation: {dv!r}")
    return list(sel)


def active_set_size(
    Xfit: np.ndarray,
    yfit: np.ndarray,
    lambda_: float,
    quiet: bool,
) -> int:
    """|M| of the second-stage Lasso (KKT-certified via ``solve_Lasso``)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf if quiet else sys.stdout):
        M, _Mc, _sM = solve_Lasso(Xfit, yfit, lambda_)
    return int(len(M))


def summarize(values: list[int]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "q90": float(np.quantile(arr, 0.90)) if arr.size else float("nan"),
        "q95": float(np.quantile(arr, 0.95)) if arr.size else float("nan"),
        "max": int(np.max(arr)) if arr.size else -1,
        "empty": int(np.sum(arr == 0)) if arr.size else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dvs", default="lava,cg-score",
                    help="comma-separated data_valuation methods")
    ap.add_argument("--p", default="400,600,800,1000",
                    help="comma-separated feature dimensions")
    ap.add_argument("--nt", type=int, default=200)
    ap.add_argument("--nv", type=int, default=20)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--lambdas", default="0.5,1,2,3,5,8,10,15,20,30,40,50",
                    help="comma-separated lambda_ grid (SI_DV scale)")
    ap.add_argument("--iters", type=int, default=120,
                    help="null draws per (dv, p)")
    ap.add_argument("--criterion", default="max", choices=["max", "q95", "q90"],
                    help="statistic of |M| that must stay below the target")
    ap.add_argument("--targets", default="200,100",
                    help="comma-separated |M| targets to recommend against "
                         "(200 = literal request, 100 ~ n_selected)")
    ap.add_argument("--output-dir", default="output_lambda_highdim")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress selection/Lasso/LP stdout noise")
    args = ap.parse_args()

    dvs = [s.strip() for s in args.dvs.split(",") if s.strip()]
    ps = [int(s) for s in args.p.split(",") if s.strip()]
    lambdas = [float(s) for s in args.lambdas.split(",") if s.strip()]
    targets = [int(s) for s in args.targets.split(",") if s.strip()]

    outdir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", args.output_dir))
    os.makedirs(outdir, exist_ok=True)
    summary_path = os.path.join(outdir, "lambda_summary.tsv")
    rec_path = os.path.join(outdir, "lambda_recommend.txt")

    print("High-dim lambda search for the stage-2 Lasso")
    print(f"  dvs       : {dvs}")
    print(f"  p         : {ps}")
    print(f"  nt, nv    : {args.nt}, {args.nv}   (n_selected ~ {args.fraction * args.nt:.0f})")
    print(f"  fraction  : {args.fraction}")
    print(f"  lambdas   : {lambdas}")
    print(f"  iters     : {args.iters}  (null, normal noise)")
    print(f"  criterion : {args.criterion} |M| < target")
    print(f"  targets   : {targets}")
    print(f"  output    : {outdir}")

    # results[(dv, p)][lambda] = list[int]; nsel[(dv, p)] = list[int]
    results: dict[tuple[str, int], dict[float, list[int]]] = {}
    nsel_stats: dict[tuple[str, int], dict] = {}
    beta0 = None

    asummary = open(summary_path, "w")
    asummary.write("dv\tp\tlambda\tnsel_mean\tcount\tmean_M\tmedian_M\tq90_M\t"
                   "q95_M\tmax_M\tempty\n")
    asummary.flush()

    for dv in dvs:
        for p in ps:
            beta0 = np.zeros((p, 1))
            per_lam: dict[float, list[int]] = {lam: [] for lam in lambdas}
            nsel_list: list[int] = []
            t0 = time.time()
            for seed in range(args.iters):
                Xt, Xv, yt, yv, _St, _Sv = generate(args.nt, args.nv, p, beta0, beta0, seed=seed)
                try:
                    sel = select_indices(dv, Xt, yt, Xv, yv, args.fraction, args.quiet)
                except Exception as error:  # selection can legitimately fail
                    print(f"  [{dv}/p{p}/seed{seed}] selection failed: {error}")
                    continue
                if len(sel) == 0:
                    continue
                nsel_list.append(len(sel))
                Xfit = Xt[sel]
                yfit = yt.reshape(-1, 1)[sel]
                for lam in lambdas:
                    try:
                        per_lam[lam].append(active_set_size(Xfit, yfit, lam, args.quiet))
                    except Exception:
                        pass
            results[(dv, p)] = per_lam
            nsel_stats[(dv, p)] = summarize(nsel_list)
            dt = time.time() - t0
            print(f"\n[{dv} | p={p}]  {len(nsel_list)} selections in {dt:.1f}s  "
                  f"(mean n_selected={nsel_stats[(dv, p)]['mean']:.1f})")
            print("  lambda :   mean  median   q90   q95   max   empty")
            for lam in lambdas:
                st = summarize(per_lam[lam])
                print(f"  {lam:>6} : {st['mean']:>6.1f} {st['median']:>6.1f} "
                      f"{st['q90']:>5.1f} {st['q95']:>5.1f} {st['max']:>5d} "
                      f"{st['empty']:>5d}")
                asummary.write(
                    f"{dv}\t{p}\t{lam}\t{nsel_stats[(dv, p)]['mean']:.2f}\t"
                    f"{st['count']}\t{st['mean']:.3f}\t{st['median']:.3f}\t"
                    f"{st['q90']:.3f}\t{st['q95']:.3f}\t{st['max']}\t{st['empty']}\n"
                )
            asummary.flush()
    asummary.close()

    # ---- Recommendation: one lambda shared across all (dv, p) --------------
    # For a candidate lambda, look at the worst case over every (dv, p) cell.
    def worst_case(lam: float, stat: str) -> tuple[float, bool]:
        vals = []
        any_empty = False
        for key, per_lam in results.items():
            st = summarize(per_lam[lam])
            if st["count"] == 0:
                return float("inf"), True
            vals.append(st[stat] if stat != "max" else st["max"])
            if st["empty"] > 0:
                any_empty = True
        # also require the average cell to keep features (not all-zero everywhere)
        all_empty = all(summarize(per_lam[lam])["empty"] >= summarize(per_lam[lam])["count"]
                        for per_lam in results.values())
        return max(vals), any_empty or all_empty

    lines = []
    lines.append(f"Criterion: {args.criterion} |M| < target; lambda shared across "
                 f"all dv in {dvs} and all p in {ps}.")
    lines.append(f"iters/selection per cell: {args.iters} (null draws); "
                 f"n_selected ~ {args.fraction * args.nt:.0f}.")
    lines.append("")
    for target in targets:
        chosen = None
        for lam in sorted(lambdas):
            stat_val, degenerate = worst_case(lam, args.criterion)
            if stat_val < target and not degenerate:
                chosen = (lam, stat_val)
                break
        if chosen is None:
            line = (f"target |M| < {target}: no lambda in the grid satisfies "
                    f"{args.criterion} |M| < {target} without empty selections.")
        else:
            line = (f"target |M| < {target}: recommend lambda_ = {chosen[0]:g} "
                    f"(worst-case {args.criterion} |M| = {chosen[1]:.1f}).")
        print("\n" + line)
        lines.append(line)
    with open(rec_path, "w") as fr:
        fr.write("\n".join(lines) + "\n")
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {rec_path}")


if __name__ == "__main__":
    main()

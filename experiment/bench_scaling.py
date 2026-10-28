"""Scaling benchmark: ``parametric_nested`` reference vs numba across sizes.

Focuses on wall-clock scaling (unlike ``test_numba_parity.py``, which also does
integration/determinism/profile at every run).  For each data-valuation path and
size it:

1. resolves a seed with a non-empty Lasso active set (null data can select
   nothing, which the pipeline rejects);
2. times the reference pipeline (min of ``--repeats``);
3. times the numba pipeline (``si_numba.USE = True``, compile warmup excluded);
4. reports the speedup and checks the numba intervals/p-value match the
   reference at scale.

Run from the repo root with the ``IS`` python (see AGENTS.md):

    "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" experiment/bench_scaling.py
    "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" experiment/bench_scaling.py --repeats 1 --cg-only
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# Stub the deleted LPsolver before importing the package (see AGENTS.md).
if "si_for_cg.LPsolver" not in sys.modules:
    _lp = types.ModuleType("si_for_cg.LPsolver")

    class _DummyHIGHS:  # pragma: no cover
        def __init__(self, *a, **k):
            raise RuntimeError("HIGHS_LPsolver is dead code; scipy linprog is used")

    _lp.HIGHS_LPsolver = _DummyHIGHS
    sys.modules["si_for_cg.LPsolver"] = _lp

from si_for_cg.si_dv import SelectiveInference
from si_for_cg import _timeprof as tp

try:
    from si_for_cg import si_numba
except Exception as _exc:  # noqa: BLE001
    si_numba = None
    _NUMBA_ERR = _exc


SEEDS = [504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518]


def suppress():
    return contextlib.redirect_stdout(io.StringIO())


def generate(n, m, p, seed):
    np.random.seed(seed)
    Xs = np.random.normal(0, 1, size=(n, p))
    Xt = np.random.normal(0, 1, size=(m, p))
    Ys = np.random.normal(0, 1, size=(n, 1))
    Yt = np.random.normal(0, 1, size=(m, 1))
    return Xs, Xt, Ys, Yt


def set_numba(enabled):
    if si_numba is not None and hasattr(si_numba, "USE"):
        si_numba.USE = bool(enabled)


def run_once(dv, n, m, p, fraction, lambda_, zmin, zmax, step, seed):
    Xs, Xt, Ys, Yt = generate(n, m, p, seed)
    si = SelectiveInference(
        fraction=fraction, zmin=zmin, zmax=zmax, step=step,
        method="parametric_nested", lambda_=lambda_, data_valuation=dv,
    )
    with suppress():
        if dv == "lava":
            si.fit(Xs, Ys.reshape(-1), Xt, Yt.reshape(-1), jth=0)
        else:
            si.fit(Xs, Ys.reshape(-1), None, None, jth=0)
    return si


def resolve_seed(dv, n, m, p, args):
    skip = (ValueError, RuntimeError, np.linalg.LinAlgError)
    for s in [args.seed] + [x for x in SEEDS if x != args.seed]:
        try:
            si = run_once(dv, n, m, p, args.fraction, args.lambda_,
                          args.zmin, args.zmax, args.step, s)
            return s, si
        except skip:
            continue
    return None, None


def timeit(fn, repeats):
    fn()  # warmup / compile
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def pieces_match(a, b, tol=1e-9):
    if len(a) != len(b):
        return False
    return all(abs(l1 - l2) <= tol and abs(r1 - r2) <= tol
               for (l1, r1), (l2, r2) in zip(a, b))


def bench_one(dv, n, m, p, args):
    seed, ref_si = resolve_seed(dv, n, m, p, args)
    if seed is None:
        print(f"  {dv:<9} n={n:<4} m={m:<4} p={p:<3}  SKIP (no non-empty Lasso set)")
        return
    ref_i = list(ref_si.intervals_)
    ref_p = float(ref_si.p_value_)

    set_numba(False)
    tr = timeit(lambda: run_once(dv, n, m, p, args.fraction, args.lambda_,
                                 args.zmin, args.zmax, args.step, seed),
                args.repeats)

    if si_numba is None:
        print(f"  {dv:<9} n={n:<4} m={m:<4} p={p:<3}  ref {tr:8.3f}s  "
              f"({len(ref_i)} pieces, p={ref_p:.6g})  [numba absent]")
        return

    set_numba(True)
    tn = timeit(lambda: run_once(dv, n, m, p, args.fraction, args.lambda_,
                                 args.zmin, args.zmax, args.step, seed),
                args.repeats)
    got = run_once(dv, n, m, p, args.fraction, args.lambda_,
                   args.zmin, args.zmax, args.step, seed)
    set_numba(False)

    par = ("OK" if (pieces_match(ref_i, list(got.intervals_))
                    and abs(ref_p - float(got.p_value_)) <= 1e-9) else "MISMATCH")
    print(f"  {dv:<9} n={n:<4} m={m:<4} p={p:<3}  ref {tr:8.3f}s  "
          f"numba {tn:8.3f}s  speedup {tr / tn:5.2f}x  "
          f"({len(ref_i)} pieces, p={ref_p:.6g}, parity {par})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--seed", type=int, default=504)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=5.0)
    ap.add_argument("--zmin", type=float, default=-20.0)
    ap.add_argument("--zmax", type=float, default=20.0)
    ap.add_argument("--step", type=float, default=1e-3)
    ap.add_argument("--cg-only", action="store_true")
    ap.add_argument("--lava-only", action="store_true")
    args = ap.parse_args()

    print("=" * 92)
    print("SCALING BENCHMARK: parametric_nested reference vs numba")
    print(f"  repeats={args.repeats} step={args.step:g} lambda={args.lambda_:g} "
          f"fraction={args.fraction:g}")
    if si_numba is None:
        print(f"  numba: NOT AVAILABLE ({type(_NUMBA_ERR).__name__}: {_NUMBA_ERR})")
    else:
        import numba
        print(f"  numba: {numba.__version__}  numpy: {np.__version__}")
    print("=" * 92)

    cg_sizes = [(80, 30, 8), (120, 45, 10), (160, 60, 12), (200, 75, 15)]
    lava_sizes = [(60, 25, 6), (90, 35, 8), (120, 45, 10)]

    if not args.lava_only:
        print("\n-- cg-score --")
        for n, m, p in cg_sizes:
            bench_one("cg-score", n, m, p, args)
    if not args.cg_only:
        print("\n-- lava --")
        for n, m, p in lava_sizes:
            bench_one("lava", n, m, p, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Benchmark the OT LP solver in ``si_for_cg/OptimalTransport.solveOT``.

Contestants
  scipy-simplex   patched ``linprog(method='simplex')``; returns ``res.basis``
                  (current production path).  Slow at scale, so it runs under a
                  wall-clock guard in a child process (``--timeout``).
  scipy-highs*    ``linprog(method='highs'|'highs-ds'|'highs-ipm')`` (no basis)
  highs-*         direct ``highspy`` model + ``getBasis()`` (auto/ds/ipm),
                  presolve on/off -- the only path that yields an optimal basis

The LP is exactly the one ``solveOT`` builds: min ``cost @ x`` s.t.
``S_ @ x = h_``, ``x >= 0`` where ``S_ = convert(ns, nv)[:-1]`` and
``cost = sum_i (OMEGA @ X_[:, i])**2``.

Checks
  * objective parity vs scipy-simplex (when it finishes)
  * primal solution max-abs diff
  * HiGHS basis validity (``basis.valid``) and ``basis_is_optimal``
  * self-certification independent of scipy: the basic columns must reproduce
    the optimum -- ``S_b x_b = h`` residual, ``cost_b @ x_b`` vs reported obj,
    and dual feasibility ``min reduced cost >= -tol``
  * downstream lava-score parity and top-fraction selection parity (vs ref)
  * wall time: build+run and run-only (min over repeats)

Run from the repo root with the ``IS`` python (see AGENTS.md).  The package
``__init__`` is bypassed with a stub so BLAS pinning does not fight HiGHS.

Examples
  python experiment/bench_ot_lp.py
  python experiment/bench_ot_lp.py --sizes 500x25x15 1000x25x15 --timeout 300
  python experiment/bench_ot_lp.py --sizes 500x25x15 --parity 5
"""

import argparse
import contextlib
import io
import multiprocessing as mp
import os
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Import OptimalTransport without running si_for_cg/__init__ (which pins BLAS
# threads and imports the whole SI pipeline).
_pkg = types.ModuleType("si_for_cg")
_pkg.__path__ = [os.path.join(_ROOT, "si_for_cg")]
_pkg.__package__ = "si_for_cg"
sys.modules["si_for_cg"] = _pkg

import numpy as np
from scipy.optimize import linprog

from si_for_cg import si_numba
from si_for_cg.OptimalTransport import (
    basis_is_optimal,
    constructOMEGA,
    convert,
)

import highspy

_BASIC = highspy.HighsBasisStatus.kBasic
_FAST_BUILD = True  # numba OMEGA (JIT) when available


def set_fast_build(enabled):
    global _FAST_BUILD
    _FAST_BUILD = bool(enabled)
    if si_numba is not None and hasattr(si_numba, "USE"):
        si_numba.USE = bool(enabled) and si_numba.AVAILABLE


# --------------------------------------------------------------------------
# problem construction (mirrors lava_function.calc_lava_score / solveOT)
# --------------------------------------------------------------------------

def build_ot(ns, nv, p, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.standard_normal((ns, p)), rng.standard_normal((nv, p))])
    y = np.concatenate([rng.standard_normal(ns), rng.standard_normal(nv)]).reshape(-1, 1)
    X_ = np.hstack([X, y])

    OMEGA = constructOMEGA(ns, nv)
    cost = 0.0
    for i in range(p + 1):
        d = OMEGA.dot(X_[:, [i]])
        cost = cost + d * d
    cost = np.asarray(cost, dtype=float).reshape(-1)

    h = np.concatenate([np.ones(ns) / ns, np.ones(nv) / nv])
    return convert(ns, nv)[:-1], h[:-1], cost, X_.shape[0]


# --------------------------------------------------------------------------
# lava score recovered from a basis (lava_function.calc_lava_score)
# --------------------------------------------------------------------------

def lava_scores(S_, cost, basis, ns):
    lambda_vec = np.linalg.solve(S_[:, basis].T, cost[basis])
    f_train = (-lambda_vec)[:ns].reshape(-1)
    coeff = 1.0 + 1.0 / (ns - 1)
    return coeff * f_train - np.sum(f_train) / (ns - 1)


def top_selection(scores, fraction=0.5):
    n_keep = max(1, int(len(scores) * fraction))
    order = np.argsort(scores, stable=True)
    return set(int(i) for i in order[-n_keep:])


# --------------------------------------------------------------------------
# solvers
# --------------------------------------------------------------------------

def _silent(fn):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn()


def solve_scipy(S_, h_, cost, method):
    def run():
        return linprog(cost, bounds=(0, None), A_eq=S_, b_eq=h_, method=method,
                       options={"maxiter": 100_000})

    t0 = time.perf_counter()
    res = _silent(run)
    t = time.perf_counter() - t0
    if not res.success:
        return {"ok": False, "t": t, "err": res.message}
    basis = None
    if method == "simplex" and hasattr(res, "basis"):
        basis = np.asarray(res.basis, dtype=int).reshape(-1)
    return {"ok": True, "t": t, "x": np.asarray(res.x, float),
            "obj": float(res.fun), "basis": basis}


def solve_highspy(S_, h_, cost, solver="choose", presolve="choose"):
    m, n = S_.shape
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("solver", solver)
    if presolve is not None:
        h.setOptionValue("presolve", presolve)

    t0 = time.perf_counter()
    h.addVars(n, np.zeros(n), np.full(n, np.inf))
    h.changeColsCost(n, np.arange(n, dtype=np.int32), cost)
    cols = np.arange(n, dtype=np.int32)
    for i in range(m):
        h.addRow(h_[i], h_[i], n, cols, S_[i])
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    h.run()
    t_solve = time.perf_counter() - t0
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return {"ok": False, "t": t_build + t_solve, "t_solve": t_solve,
                "err": str(h.getModelStatus())}

    bs = h.getBasis()
    cs = np.asarray(bs.col_status)
    rs = np.asarray(bs.row_status)
    return {"ok": True, "t": t_build + t_solve, "t_solve": t_solve,
            "t_build": t_build, "x": np.asarray(h.getSolution().col_value, float),
            "obj": float(h.getObjectiveValue()),
            "basis": np.where(cs == _BASIC)[0], "basis_valid": bool(bs.valid),
            "n_basic_rows": int((rs == _BASIC).sum()), "m": m}


# --- guarded scipy simplex (child process, so a slow solve can be killed) ---

def _child_worker(ns, nv, p, seed, method, fast_build, q):
    try:
        set_fast_build(fast_build)
        S_, h_, cost, _ = build_ot(ns, nv, p, seed=seed)
        q.put(solve_scipy(S_, h_, cost, method))
    except Exception as exc:  # noqa: BLE001
        q.put({"ok": False, "err": repr(exc)})


def solve_scipy_guarded(ns, nv, p, seed, method, timeout, fast_build):
    if not timeout:
        S_, h_, cost, _ = build_ot(ns, nv, p, seed=seed)
        return solve_scipy(S_, h_, cost, method)
    q = mp.Queue()
    pr = mp.Process(target=_child_worker,
                    args=(ns, nv, p, seed, method, fast_build, q))
    t0 = time.perf_counter()
    pr.start()
    pr.join(timeout)
    if pr.is_alive():
        pr.terminate()
        pr.join()
        return {"ok": False, "timeout": True, "err": f"timeout > {timeout}s",
                "t": time.perf_counter() - t0}
    try:
        r = q.get_nowait()
    except Exception:  # noqa: BLE001
        return {"ok": False, "err": "child crashed", "t": time.perf_counter() - t0}
    return r


def self_certify(S_, h_, cost, basis):
    """Independent proof that a basis is optimal: primal/dual feasibility and
    that the basic solution reproduces the reported optimum."""
    xb = np.linalg.solve(S_[:, basis], h_)
    prim_res = float(np.max(np.abs(S_[:, basis].dot(xb) - h_)))
    obj_basis = float(cost[basis].dot(xb))
    dual = np.linalg.solve(S_[:, basis].T, cost[basis])
    nb = np.setdiff1d(np.arange(S_.shape[1]), basis, assume_unique=True)
    dfeas = float(np.min(cost[nb] - S_[:, nb].T.dot(dual)))
    return prim_res, obj_basis, dfeas, float(np.min(xb))


# --------------------------------------------------------------------------
# one size
# --------------------------------------------------------------------------

def run_size(ns, nv, p, seed, repeats, timeout, fast_build, highs_only=False):
    print("=" * 96)
    print(f"SIZE  ns={ns}  nv={nv}  p={p}   |vars|={ns * nv}  rows={ns + nv - 1}"
          f"   build={'numba' if fast_build and si_numba.AVAILABLE else 'python'}")
    print("=" * 96)

    set_fast_build(fast_build)
    tb = time.perf_counter()
    S_, h_, cost, _ = build_ot(ns, nv, p, seed=seed)
    print(f"  problem build: {time.perf_counter() - tb:.3f}s")

    ref = None
    if not highs_only:
        ref = solve_scipy_guarded(ns, nv, p, seed, "simplex", timeout, fast_build)
    have_ref = bool(ref and ref.get("ok"))
    if have_ref:
        ref_x, ref_obj, ref_basis = ref["x"], ref["obj"], ref["basis"]
        ref_scores = lava_scores(S_, cost, ref_basis, ns)
        ref_top = top_selection(ref_scores)
        print(f"  scipy-simplex   t={ref['t']:8.3f}s  obj={ref_obj:.8f}  "
              f"basis={len(ref_basis)}  optimal={basis_is_optimal(S_, h_, cost, ref_basis)}")
    elif not highs_only:
        tag = "TIMEOUT" if ref.get("timeout") else "FAILED"
        print(f"  scipy-simplex   {tag}: {ref.get('err')}  (t~{ref.get('t', float('nan')):.1f}s)")
    print()

    hdr = (f"  {'solver':<18}{'t_tot[s]':>10}{'t_solve':>10}{'obj':>16}"
           f"{'nbasis':>8}{'valid':>7}{'opt':>6}{'prim_res':>10}"
           f"{'obj_basis':>11}{'dfeas':>10}")
    if have_ref:
        hdr += f"{'obj_diff':>11}{'x_diff':>10}{'scores':>10}{'top':>6}"
    print(hdr)

    def emit(name, r):
        if not r.get("ok"):
            print(f"  {name:<18}{'FAILED':>10}  {r.get('err', '')}")
            return
        basis = r.get("basis")
        if basis is None or len(basis) == 0:
            print(f"  {name:<18}{r['t']:>10.3f}{r['t_solve']:>10.3f}{r['obj']:>16.8f}"
                  f"{'-':>8}{'-':>7}{'-':>6}{'-':>10}{'-':>11}{'-':>10}")
            return
        ok = basis_is_optimal(S_, h_, cost, basis)
        prim_res, obj_basis, dfeas, min_xb = self_certify(S_, h_, cost, basis)
        line = (f"  {name:<18}{r['t']:>10.3f}{r['t_solve']:>10.3f}{r['obj']:>16.8f}"
                f"{len(basis):>8}{str(r.get('basis_valid')):>7}{str(ok):>6}"
                f"{prim_res:>10.1e}{obj_basis - r['obj']:>11.2e}{dfeas:>10.1e}")
        if have_ref:
            sc_d = float(np.max(np.abs(lava_scores(S_, cost, basis, ns) - ref_scores)))
            top_ok = top_selection(lava_scores(S_, cost, basis, ns)) == ref_top
            line += (f"{abs(r['obj'] - ref_obj):>11.2e}"
                     f"{float(np.max(np.abs(r['x'] - ref_x))):>10.2e}"
                     f"{sc_d:>10.2e}{str(top_ok):>6}")
        print(line)

    if not highs_only:
        for method in ("revised simplex", "highs", "highs-ds", "highs-ipm"):
            r = solve_scipy_guarded(ns, nv, p, seed, method, timeout, fast_build)
            if r.get("ok"):
                r.setdefault("t_solve", r["t"])
            emit(f"scipy-{method}", r)

    for solver in ("choose", "simplex", "ipm"):
        for pres in ("on", "off"):
            best = None
            for _ in range(max(1, repeats)):
                r = solve_highspy(S_, h_, cost, solver=solver, presolve=pres)
                if not r["ok"]:
                    best = r
                    break
                if best is None or r["t_solve"] < best["t_solve"]:
                    best = r
            emit(f"highs-{solver}" + ("+ps" if pres == "on" else "-ps"), best)
    print()


# --------------------------------------------------------------------------
# parity summary (small sizes; needs scipy-simplex to finish)
# --------------------------------------------------------------------------

def parity_summary(ns, nv, p, seeds, timeout, fast_build):
    print("=" * 96)
    print(f"PARITY over {seeds} seeds  ns={ns} nv={nv} p={p}  "
          f"|vars|={ns * nv} rows={ns + nv - 1}")
    print("=" * 96)
    n = same_x = same_obj = valid = optimal = certify = 0
    for s in range(seeds):
        S_, h_, cost, _ = build_ot(ns, nv, p, seed=1000 + s)
        r1 = solve_scipy_guarded(ns, nv, p, 1000 + s, "simplex", timeout, fast_build)
        r2 = solve_highspy(S_, h_, cost, solver="simplex", presolve="off")
        if not (r1.get("ok") and r2.get("ok")):
            continue
        n += 1
        same_obj += int(abs(r1["obj"] - r2["obj"]) < 1e-9)
        same_x += int(np.max(np.abs(r1["x"] - r2["x"])) < 1e-9)
        valid += int(r2["basis_valid"])
        optimal += int(basis_is_optimal(S_, h_, cost, r2["basis"]))
        prim_res, obj_basis, dfeas, min_xb = self_certify(S_, h_, cost, r2["basis"])
        certify += int(prim_res < 1e-9 and abs(obj_basis - r2["obj"]) < 1e-9
                       and dfeas > -1e-9 and min_xb > -1e-9)
    if not n:
        print("  all seeds skipped (scipy-simplex timeout/failure)")
        print()
        return
    print(f"  seeds run                : {n}")
    print(f"  same objective           : {same_obj}/{n}")
    print(f"  same primal plan x       : {same_x}/{n}")
    print(f"  HiGHS basis valid/optimal: {valid}/{n} / {optimal}/{n}")
    print(f"  HiGHS basis self-certified: {certify}/{n}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="*",
                    default=["50x10x8", "100x25x12", "200x40x15"],
                    help="list of NSxNVxP triples")
    ap.add_argument("--big", action="store_true", help="append 500x25x15 and 1000x25x15")
    ap.add_argument("--repeats", type=int, default=3, help="HiGHS repeats (min)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=0.0,
                    help="scipy-simplex wall-clock guard in seconds (0=off)")
    ap.add_argument("--parity", type=int, default=0,
                    help="if >0, seed count for the basis-parity summary")
    ap.add_argument("--threads", type=int, default=0,
                    help="0 leaves BLAS/OMP defaults; N pins both to N")
    ap.add_argument("--no-fast-build", action="store_true",
                    help="disable the numba OMEGA construction")
    ap.add_argument("--highs-only", action="store_true",
                    help="skip scipy-simplex and scipy-highs; run highspy only")
    args = ap.parse_args()

    if args.threads and args.threads > 0:
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = str(args.threads)

    fast_build = not args.no_fast_build
    sizes = list(args.sizes)
    if args.big:
        sizes += ["500x25x15", "1000x25x15"]

    print(f"numpy {np.__version__}  scipy {__import__('scipy').__version__}  "
          f"numba OMEGA={'on' if fast_build and si_numba.AVAILABLE else 'off'}")
    for spec in sizes:
        ns, nv, p = (int(v) for v in spec.lower().split("x"))
        run_size(ns, nv, p, args.seed, args.repeats, args.timeout, fast_build,
                 highs_only=args.highs_only)
        if args.parity and not args.highs_only:
            parity_summary(ns, nv, p, args.parity, args.timeout, fast_build)


if __name__ == "__main__":
    main()

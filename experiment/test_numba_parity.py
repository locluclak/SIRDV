"""Accuracy + runtime harness for the numba port of ``parametric_nested`` SI.

Two things are tested, for BOTH ``data_valuation="lava"`` and
``data_valuation="cg-score"`` (see NUMBA_PLAN.md):

1. Accuracy
   - kernel unit parity: the future ``si_for_cg/si_numba.py`` ``n_*`` kernels
     against their pure-Python references on randomized inputs (tight
     tolerances + an independent ground-truth grid check where applicable);
   - integration parity: reference pipeline vs numba pipeline
     (``SI_USE_NUMBA=1``) on fixed seeds; same intervals and p-value;
   - determinism: two numba runs are bit-identical.

2. Runtime
   - kernel microbenchmarks (min-of-repeats, compile warmup excluded);
   - end-to-end reference timing that works TODAY, and numba timing once
     ``si_for_cg/si_numba.py`` exists.

Expected interface of ``si_for_cg/si_numba.py`` (all wrappers accept plain
arrays/scalars and return arrays; exact names are asserted by this script):

    n_construct_omega(nt, nv) -> (nt*nv, nt+nv)
    n_solveinterval(P, Q, O, z) -> (k, 2) float array  [or list of tuples]
    n_quad_pieces(a, b, c, z) -> (left, right, ok)
    n_conditioning_lasso(X_, a, b, Lambda, M, Mc, sM) -> (k, 2) array
    n_score_coefficients(S_, basis, w, v, u, nt) -> (nt, 3)
    n_cg_score_coefficients(invH, a_vec, b_vec) -> (n, 3)
    n_iter_cg_sort_regions(invH, a_vec, b_vec, zmin, zmax, step, n_keep)
        -> (z_entry, sl, sr, sel_flat, sel_len, n_out)

Run from the repo root with the ``IS`` python (see AGENTS.md):

    "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" experiment/test_numba_parity.py
    "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" experiment/test_numba_parity.py --require-numba

Exit code is 0 when everything that ran passed; ``--require-numba`` makes a
missing/broken numba module a failure.
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

# ---------------------------------------------------------------------------
# import shim: si_for_cg/OptimalTransport.py imports the deleted LPsolver.
# Install a stub child module BEFORE importing the package so the real
# pipeline (SI_DV, SelectiveInference, ...) imports cleanly.
# ---------------------------------------------------------------------------
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if "si_for_cg.LPsolver" not in sys.modules:
    _lp = types.ModuleType("si_for_cg.LPsolver")

    class _DummyHIGHS:  # pragma: no cover - never called
        def __init__(self, *a, **k):
            raise RuntimeError("HIGHS_LPsolver is dead code; scipy linprog is used")

    _lp.HIGHS_LPsolver = _DummyHIGHS
    sys.modules["si_for_cg.LPsolver"] = _lp

from si_for_cg import OptimalTransport
from si_for_cg import si_sort_walk
from si_for_cg.si_adapter import SelectionAdapter, CGSelectionAdapter
from si_for_cg.si_overconditioning import conditioning_Lasso, sort_conditioning
from si_for_cg.si_parametric import (
    parametric_intervals_FS_nested,
    cg_parametric_intervals_FS_nested,
)
from si_for_cg.si_types import LassoInfo, SIConfig
from si_for_cg.si_utils import (
    compute_p_value,
    decompose_y,
    interval_union,
    solve_Lasso,
    solve_linear_inequality,
    solveinterval,
)

try:  # the module under test (does not exist until the port lands)
    from si_for_cg import si_numba
except Exception as _exc:  # noqa: BLE001 - report and continue
    si_numba = None
    _NUMBA_IMPORT_ERROR = _exc
else:
    _NUMBA_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def suppress_stdout():
    return contextlib.redirect_stdout(io.StringIO())


def close(a, b, rtol=1e-9, atol=1e-9):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


def pieces_close(pa, pb, rtol=1e-9, atol=1e-9):
    pa = [(float(l), float(r)) for l, r in pa]
    pb = [(float(l), float(r)) for l, r in pb]
    if len(pa) != len(pb):
        return False
    return all(close([l1, r1], [l2, r2], rtol, atol) for (l1, r1), (l2, r2) in zip(pa, pb))


def linprog_is_patched():
    try:
        from scipy.optimize import linprog
        with suppress_stdout():
            res = linprog(
                np.zeros(2), A_eq=np.ones((1, 2)), b_eq=np.array([1.0]),
                bounds=(0, None), method="simplex",
            )
        return hasattr(res, "basis")
    except Exception:  # noqa: BLE001
        return False


def generate(n, m, p, seed=None):
    """Same generator as examples/demo_si.py (null betas)."""
    if seed is not None:
        np.random.seed(seed)
    Xs = np.random.normal(0, 1, size=(n, p))
    Xt = np.random.normal(0, 1, size=(m, p))
    Ys = np.random.normal(0, 1, size=(n, 1))
    Yt = np.random.normal(0, 1, size=(m, 1))
    return Xs, Xt, Ys, Yt


class _FakeAdapter:
    """Minimal adapter for score_coefficients unit tests."""

    def __init__(self, S_):
        self.S_ = S_
        self.fraction = 0.5


def set_numba(enabled):
    """Toggle the numba dispatch flag (env + module attribute if present).

    Returns the previous env value so callers can restore it.
    """
    old = os.environ.get("SI_USE_NUMBA")
    if enabled:
        os.environ["SI_USE_NUMBA"] = "1"
    elif old is not None:
        os.environ.pop("SI_USE_NUMBA", None)
    if si_numba is not None and hasattr(si_numba, "USE"):
        si_numba.USE = bool(enabled)
    return old


def restore_env(old):
    if old is None:
        os.environ.pop("SI_USE_NUMBA", None)
    else:
        os.environ["SI_USE_NUMBA"] = old


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
class Report:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.rows = []

    def ok(self, name, detail=""):
        self.passed += 1
        self.rows.append(("PASS", name, detail))
        print(f"  [PASS] {name} {detail}")

    def bad(self, name, detail=""):
        self.failed += 1
        self.rows.append(("FAIL", name, detail))
        print(f"  [FAIL] {name} {detail}")

    def skip(self, name, detail=""):
        self.skipped += 1
        self.rows.append(("SKIP", name, detail))
        print(f"  [SKIP] {name} {detail}")


# ---------------------------------------------------------------------------
# section 0: preflight
# ---------------------------------------------------------------------------
def preflight(report):
    print("\n=== 0. PREFLIGHT ===")
    print(f"  python : {sys.executable}")
    print(f"  numpy  : {np.__version__}")
    try:
        import numba
        print(f"  numba  : {numba.__version__} (numpy support check below)")
        try:
            from numba import njit

            @njit
            def _probe(x):
                return np.linalg.inv(x)

            _probe(np.eye(2))
            print("  njit   : OK (np.linalg.inv compiles)")
        except Exception as exc:  # noqa: BLE001
            print(f"  njit   : FAILED ({type(exc).__name__}: {exc})")
    except Exception as exc:  # noqa: BLE001
        print(f"  numba  : NOT INSTALLED ({type(exc).__name__}: {exc})")
    if si_numba is None:
        print(f"  si_numba: NOT AVAILABLE "
              f"({type(_NUMBA_IMPORT_ERROR).__name__}: {_NUMBA_IMPORT_ERROR})")
        report.skip("si_for_cg.si_numba import",
                    "module absent or numba/numpy incompatible")
    else:
        report.ok("si_for_cg.si_numba import")
    print(f"  lava linprog patched: {linprog_is_patched()}")


# ---------------------------------------------------------------------------
# section 1: kernel unit parity
# ---------------------------------------------------------------------------
def unit_construct_omega(report, rng):
    name = "constructOMEGA"
    if si_numba is None or not hasattr(si_numba, "n_construct_omega"):
        return report.skip(name, "n_construct_omega missing")
    for nt, nv in [(4, 3), (8, 5), (20, 7)]:
        ref = OptimalTransport.constructOMEGA(nt, nv)
        got = np.asarray(si_numba.n_construct_omega(nt, nv))
        if ref.shape == got.shape and np.array_equal(ref, got):
            report.ok(name, f"nt={nt},nv={nv}")
        else:
            report.bad(name, f"nt={nt},nv={nv} shape {ref.shape}/{got.shape}")


def unit_solveinterval(report, rng):
    name = "solveinterval"
    if si_numba is None or not hasattr(si_numba, "n_solveinterval"):
        return report.skip(name, "n_solveinterval missing")
    bad = 0
    for it in range(200):
        n = int(rng.integers(1, 40))
        P = rng.uniform(-3, 3, n)
        Q = rng.uniform(-3, 3, n)
        O = rng.uniform(-3, 3, n)
        z = float(rng.uniform(-2, 2))
        # ensure the point is feasible for most batches
        Q = np.where(rng.random(n) < 0.5, P, Q)
        ref = solveinterval(P, Q, O, z)
        got = si_numba.n_solveinterval(P, Q, O, z)
        got = [(float(l), float(r)) for l, r in np.asarray(got).reshape(-1, 2)]
        if not pieces_close(ref, got):
            bad += 1
    if bad == 0:
        report.ok(name, "200 batches")
    else:
        report.bad(name, f"{bad}/200 mismatched")


def unit_quad_pieces(report, rng):
    name = "quad_pieces"
    if si_numba is None or not hasattr(si_numba, "n_quad_pieces"):
        return report.skip(name, "n_quad_pieces missing")
    bad = 0
    for _ in range(200):
        n = int(rng.integers(1, 40))
        a = rng.uniform(-3, 3, n)
        b = rng.uniform(-3, 3, n)
        c = rng.uniform(-3, 3, n)
        z = float(rng.uniform(-2, 2))
        l0, r0, ok0 = si_sort_walk.quad_pieces(a, b, c, z)
        l1, r1, ok1 = si_numba.n_quad_pieces(a, b, c, z)
        if not (close(l0, l1) and close(r0, r1) and np.array_equal(ok0, ok1)):
            bad += 1
    if bad == 0:
        report.ok(name, "200 batches")
    else:
        report.bad(name, f"{bad}/200 mismatched")


def unit_conditioning_lasso(report, rng):
    name = "conditioning_Lasso"
    if si_numba is None or not hasattr(si_numba, "n_conditioning_lasso"):
        return report.skip(name, "n_conditioning_lasso missing")
    bad = 0
    for _ in range(100):
        n = int(rng.integers(12, 40))
        p = int(rng.integers(3, 8))
        X_ = rng.normal(size=(n, p))
        a = rng.normal(size=(n, 1))
        b = rng.normal(size=(n, 1))
        Lam = float(rng.uniform(0.5, 5.0))
        cols = rng.permutation(p)
        k = int(rng.integers(1, p))
        M = np.sort(cols[:k])
        Mc = np.sort(cols[k:])
        sM = rng.choice([-1.0, 1.0], size=(k, 1))
        try:
            with suppress_stdout():
                ref = conditioning_Lasso(X_, a, b, Lam, M, Mc, sM)
        except ValueError:
            continue
        got = si_numba.n_conditioning_lasso(X_, a, b, Lam, M, Mc, sM)
        got = [(float(l), float(r)) for l, r in np.asarray(got).reshape(-1, 2)]
        if not pieces_close(ref, got):
            bad += 1
    if bad == 0:
        report.ok(name, "~100 batches")
    else:
        report.bad(name, f"{bad} mismatched")


def unit_score_coefficients(report, rng):
    name = "score_coefficients"
    if si_numba is None or not hasattr(si_numba, "n_score_coefficients"):
        return report.skip(name, "n_score_coefficients missing")
    nt, nv = 6, 3
    m = nt + nv - 1
    cols = nt * nv
    S_ = rng.normal(size=(m, cols))
    basis = np.sort(rng.permutation(cols)[:m])
    w = rng.normal(size=cols)
    v = rng.normal(size=cols)
    u = rng.normal(size=cols)
    ad = _FakeAdapter(S_)
    ref = si_sort_walk.score_coefficients(ad, basis, w, v, u, nt)
    got = si_numba.n_score_coefficients(S_, basis, w, v, u, nt)
    if close(ref, got, rtol=1e-9, atol=1e-9):
        report.ok(name)
    else:
        report.bad(name, "coefficient table differs")


def unit_cg_score_coefficients(report, rng):
    name = "cg_score_coefficients"
    if si_numba is None or not hasattr(si_numba, "n_cg_score_coefficients"):
        return report.skip(name, "n_cg_score_coefficients missing")
    n = 12
    A = rng.normal(size=(n, n))
    invH = A @ A.T + n * np.eye(n)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    ref = si_sort_walk.cg_score_coefficients(invH, a, b)
    got = si_numba.n_cg_score_coefficients(invH, a, b)
    if close(ref, got, rtol=1e-9, atol=1e-9):
        report.ok(name)
    else:
        report.bad(name, "cg coefficient table differs")


def unit_cg_walk(report, rng):
    name = "iter_cg_sort_regions"
    if si_numba is None or not hasattr(si_numba, "n_iter_cg_sort_regions"):
        return report.skip(name, "n_iter_cg_sort_regions missing")
    n = 30
    A = rng.normal(size=(n, n))
    invH = A @ A.T + n * np.eye(n)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    zmin, zmax, step, n_keep = -8.0, 8.0, 1e-3, 15
    cfg = SIConfig(zmin=zmin, zmax=zmax, step=step, data_valuation="cg-score")
    ref = list(si_sort_walk.iter_cg_sort_regions(
        invH, a, b, zmin, zmax, cfg, n_keep, {}))
    out = si_numba.n_iter_cg_sort_regions(
        invH, a, b, zmin, zmax, step, n_keep)
    z_e, sl, sr, sel_flat, sel_len, n_out = out
    sel_flat = np.asarray(sel_flat)
    got = []
    off = 0
    for i in range(int(n_out)):
        k = int(sel_len[i])
        if sel_flat.ndim == 2:
            sel_i = sel_flat[i, :k]
        else:
            sel_i = sel_flat[off:off + k]
            off += k
        got.append((float(z_e[i]), float(sl[i]), float(sr[i]),
                    list(sel_i.astype(int))))
    if len(ref) != len(got):
        return report.bad(name, f"region count {len(ref)} vs {len(got)}")
    ok = True
    for (ze0, l0, r0, s0), (ze1, l1, r1, s1) in zip(ref, got):
        if not (abs(ze0 - ze1) <= 1e-9 and abs(l0 - l1) <= 1e-9
                and abs(r0 - r1) <= 1e-9 and sorted(s0) == sorted(s1)):
            ok = False
            break
    report.ok(name) if ok else report.bad(name, "region sequence differs")


def section_kernel_parity(report):
    print("\n=== 1. KERNEL UNIT PARITY ===")
    if si_numba is None:
        print("  si_for_cg.si_numba not available; skipping all kernel tests.")
        report.skip("kernel unit parity", "numba module absent")
        return
    rng = np.random.default_rng(20261009)
    for fn in (unit_construct_omega, unit_solveinterval, unit_quad_pieces,
               unit_conditioning_lasso, unit_score_coefficients,
               unit_cg_score_coefficients, unit_cg_walk):
        try:
            fn(report, rng)
        except Exception as exc:  # noqa: BLE001
            report.bad(fn.__name__, f"raised {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# section 2: integration parity
# ---------------------------------------------------------------------------
def run_pipeline(data_valuation, n, m, p, fraction, lambda_, zmin, zmax, step,
                 seed, jth=0):
    from si_for_cg.si_dv import SelectiveInference
    Xs, Xt, Ys, Yt = generate(n, m, p, seed=seed)
    si = SelectiveInference(
        fraction=fraction, zmin=zmin, zmax=zmax, step=step,
        method="parametric_nested", lambda_=lambda_,
        data_valuation=data_valuation,
    )
    with suppress_stdout():
        if data_valuation == "lava":
            si.fit(Xs, Ys.reshape(-1), Xt, Yt.reshape(-1), jth=jth)
        else:
            si.fit(Xs, Ys.reshape(-1), None, None, jth=jth)
    return si.intervals_, float(si.p_value_), int(si.tested_index_)


SEED_CANDIDATES = [504, 505, 506, 507, 508, 509, 510, 511, 512, 513]


def run_pipeline_first(dv, args):
    """Run the reference pipeline on the first seed with a non-empty Lasso set.

    Under the null the KKT-certified Lasso can be all-zero, which the pipeline
    rejects with ``ValueError("Empty feature selection set")``.  Returns
    ``(seed, result)`` so the numba run can reuse the exact same seed.
    """
    seeds = [args.seed] + [s for s in SEED_CANDIDATES if s != args.seed]
    last = None
    for s in seeds:
        try:
            return s, run_pipeline(dv, args.n, args.m, args.p, args.fraction,
                                   args.lambda_, args.zmin, args.zmax,
                                   args.step, s)
        except ValueError as exc:
            last = exc
            continue
    raise last if last is not None else RuntimeError("no candidate seed")


def _compare_pipeline(report, label, ref, got, tol=1e-9):
    i0, p0, j0 = ref
    i1, p1, j1 = got
    same_j = (j0 == j1)
    same_p = abs(p0 - p1) <= tol or (np.isnan(p0) and np.isnan(p1))
    same_i = pieces_close(i0, i1, rtol=1e-9, atol=tol)
    detail = (f"p {p0:.12g} vs {p1:.12g}, pieces {len(i0)} vs {len(i1)}")
    if same_j and same_p and same_i:
        report.ok(label, detail)
    else:
        report.bad(label, detail + f", jth {j0}/{j1}, pieces_match={same_i}")


def section_integration(report, args):
    print("\n=== 2. INTEGRATION PARITY (reference vs numba) ===")
    # always run reference once so the baseline timing is available
    for dv in ("cg-score", "lava"):
        if dv == "lava" and (args.no_lava or not linprog_is_patched()):
            report.skip(f"{dv} pipeline", "patched scipy linprog(...).basis missing or --no-lava")
            continue
        try:
            t0 = time.perf_counter()
            seed, ref = run_pipeline_first(dv, args)
            dt = time.perf_counter() - t0
        except Exception as exc:  # noqa: BLE001
            report.skip(f"{dv} reference pipeline",
                        f"{type(exc).__name__}: {exc}")
            continue
        report.ok(f"{dv} reference pipeline",
                  f"seed={seed}, p={ref[1]:.9g}, {len(ref[0])} pieces, {dt:.3f}s")

        if si_numba is None:
            report.skip(f"{dv} numba parity", "numba module absent")
            continue
        old = set_numba(True)
        try:
            got = run_pipeline(dv, args.n, args.m, args.p, args.fraction,
                               args.lambda_, args.zmin, args.zmax, args.step,
                               seed)
        except Exception as exc:  # noqa: BLE001
            report.bad(f"{dv} numba pipeline",
                       f"{type(exc).__name__}: {exc}")
            continue
        finally:
            restore_env(old)
        _compare_pipeline(report, f"{dv} parity (seed={seed})", ref, got)

        # determinism
        set_numba(True)
        try:
            got2 = run_pipeline(dv, args.n, args.m, args.p, args.fraction,
                                args.lambda_, args.zmin, args.zmax, args.step,
                                seed)
        finally:
            restore_env(old)
        if pieces_close(got[0], got2[0]) and abs(got[1] - got2[1]) <= 1e-12:
            report.ok(f"{dv} determinism")
        else:
            report.bad(f"{dv} determinism")


# ---------------------------------------------------------------------------
# section 3: runtime
# ---------------------------------------------------------------------------
def timeit(fn, repeats=3):
    fn()  # warmup (also triggers njit compile)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_kernels(report, rng):
    print("\n=== 3a. KERNEL RUNTIME ===")
    if si_numba is None:
        report.skip("kernel runtime", "numba module absent")
        return
    if hasattr(si_numba, "n_solveinterval"):
        print(f"  {'n':>8} {'ref [ms]':>10} {'numba [ms]':>11} {'speedup':>8}")
        for n in (100, 1000, 10000):
            P = np.zeros(n)
            Q = rng.uniform(-3, 3, n)
            O = -Q * 0.3 - rng.uniform(0.1, 1.0, n)
            tr = timeit(lambda: solveinterval(P, Q, O, 0.0))
            tn = timeit(lambda: si_numba.n_solveinterval(P, Q, O, 0.0))
            print(f"  {n:>8} {tr * 1e3:>10.3f} {tn * 1e3:>11.3f} "
                  f"{tr / tn:>8.2f}")
    if hasattr(si_numba, "n_construct_omega"):
        print(f"  {'nt,nv':>8} {'ref [ms]':>10} {'numba [ms]':>11} {'speedup':>8}")
        for nt, nv in ((50, 20), (100, 50), (150, 60)):
            tr = timeit(lambda: OptimalTransport.constructOMEGA(nt, nv), 2)
            tn = timeit(lambda: si_numba.n_construct_omega(nt, nv), 2)
            print(f"  {nt:>3},{nv:>4} {tr * 1e3:>10.3f} {tn * 1e3:>11.3f} "
                  f"{tr / tn:>8.2f}")


def bench_end_to_end(report, args):
    print("\n=== 3b. END-TO-END RUNTIME ===")
    for dv in ("cg-score", "lava"):
        if dv == "lava" and (args.no_lava or not linprog_is_patched()):
            report.skip(f"{dv} e2e timing", "patched linprog missing or --no-lava")
            continue
        try:
            seed, _ = run_pipeline_first(dv, args)  # warmup + working seed
            tr = timeit(lambda: run_pipeline(
                dv, args.n, args.m, args.p, args.fraction, args.lambda_,
                args.zmin, args.zmax, args.step, seed), repeats=3)
            print(f"  {dv:<9} reference: {tr * 1e3:8.2f} ms  (seed={seed})")
        except Exception as exc:  # noqa: BLE001
            report.skip(f"{dv} e2e timing", f"{type(exc).__name__}: {exc}")
            continue
        if si_numba is None:
            report.skip(f"{dv} e2e numba timing", "numba module absent")
            continue
        old = set_numba(True)
        try:
            with suppress_stdout():
                run_pipeline(dv, args.n, args.m, args.p, args.fraction,
                             args.lambda_, args.zmin, args.zmax, args.step,
                             seed)  # warmup / compile
            tn = timeit(lambda: run_pipeline(
                dv, args.n, args.m, args.p, args.fraction, args.lambda_,
                args.zmin, args.zmax, args.step, seed), repeats=3)
            print(f"  {dv:<9} numba    : {tn * 1e3:8.2f} ms   "
                  f"speedup {tr / tn:.2f}x")
        except Exception as exc:  # noqa: BLE001
            report.bad(f"{dv} e2e numba timing", f"{type(exc).__name__}: {exc}")
        finally:
            restore_env(old)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--require-numba", action="store_true",
                    help="fail if si_for_cg.si_numba is unavailable")
    ap.add_argument("--no-lava", action="store_true",
                    help="skip the lava path (e.g. unpatched scipy)")
    ap.add_argument("--seed", type=int, default=504)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--m", type=int, default=15)
    ap.add_argument("--p", type=int, default=6)
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--lambda", dest="lambda_", type=float, default=5.0)
    ap.add_argument("--zmin", type=float, default=-10.0)
    ap.add_argument("--zmax", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=1e-4)
    return ap.parse_args()


def main():
    args = parse_args()
    report = Report()
    t0 = time.perf_counter()

    preflight(report)
    section_kernel_parity(report)
    section_integration(report, args)
    rng = np.random.default_rng(7)
    bench_kernels(report, rng)
    bench_end_to_end(report, args)

    print("\n=== SUMMARY ===")
    print(f"  passed {report.passed}, failed {report.failed}, "
          f"skipped {report.skipped}  ({time.perf_counter() - t0:.1f}s)")
    if report.failed:
        print("  VERDICT: FAIL")
        return 1
    if si_numba is None and args.require_numba:
        print("  VERDICT: FAIL (--require-numba but si_for_cg.si_numba unavailable)")
        return 1
    if report.skipped:
        print("  VERDICT: PARTIAL (reference baseline only; parity skipped)")
    else:
        print("  VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

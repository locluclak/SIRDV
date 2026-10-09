"""Phase 0 preflight: numba feature compatibility for the parametric_nested port.

Checks that every primitive the planned kernels (NUMBA_PLAN.md) rely on
compiles in nopython mode under the installed numba/numpy, and records the
numeric agreement of the LAPACK wrappers against numpy.

Run from the repo root with the ``IS`` python:

    "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" experiment/numba_preflight.py

Exit code 0 if all REQUIRED probes pass, 1 otherwise.  UNSUPPORTED probes are
expected negatives (they document what must be rewritten); they do not fail
the run.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    import numba
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception as _exc:  # noqa: BLE001
    HAVE_NUMBA = False
    numba = None
    njit = None
    prange = None
    _NUMBA_ERR = _exc

if HAVE_NUMBA:

    # --- LAPACK wrappers (the scipy lu_factor replacement) ------------------
    @njit(cache=True, fastmath=False, boundscheck=False)
    def k_inv(a):
        return np.linalg.inv(a)

    @njit(cache=True, fastmath=False)
    def k_solve(a, b):
        return np.linalg.solve(a, b)

    @njit(cache=True, fastmath=False)
    def k_solve_multi(a, b):
        return np.linalg.solve(a, b)  # b is (m, 3)

    # --- sorting / selection ------------------------------------------------
    @njit(cache=True)
    def k_argsort_default(x):
        return np.argsort(x)

    @njit(cache=True)
    def k_argsort_mergesort(x):
        return np.argsort(x, kind="mergesort")

    @njit(cache=True)
    def k_argsort_stable(x):
        return np.argsort(x, kind="stable")

    @njit(cache=True)
    def k_flatnonzero(x):
        return np.flatnonzero(x)

    @njit(cache=True)
    def k_argminmax(x):
        return np.argmin(x), np.argmax(x)

    # --- elementwise / array construction -----------------------------------
    @njit(cache=True)
    def k_misc(x):
        return (np.isfinite(x).sum(), np.isinf(x).sum(),
                np.sqrt(np.abs(x)).sum())

    @njit(cache=True)
    def k_stack(a, b):
        return np.column_stack((a, b)), np.vstack((a, b))

    @njit(cache=True)
    def k_build(n):
        out = np.empty(n)
        for i in range(n):
            out[i] = i
        z = np.zeros(n)
        f = np.full(n, 2.0)
        o = np.ones(n)
        w = np.where(out > 0.5, out, 0.0)
        c = np.concatenate((out, z))
        s = np.sort(out)
        return out, z, f, o, w, c, s

    @njit(cache=True)
    def k_isclose(a, b):
        return np.isclose(a, b, atol=1e-9, rtol=1e-9)

    # --- expected-unsupported (document what to rewrite) --------------------
    @njit(cache=True)
    def k_setdiff1d(a, b):
        return np.setdiff1d(a, b)

    @njit(cache=True)
    def k_lu_scipy(a):
        from scipy.linalg import lu_factor
        return lu_factor(a)

    # --- control flow / outputs ---------------------------------------------
    @njit(cache=True)
    def k_raise(x):
        if x < 0:
            raise ValueError("negative input")
        return x

    @njit(cache=True)
    def k_tuple_ret(n):
        a = np.zeros(n)
        b = np.full(n, 2.0)
        return a, b, n

    @njit(cache=True)
    def k_generator(n):
        for i in range(n):
            yield i

    @njit(cache=True, nogil=True)
    def k_nogil(x):
        s = 0.0
        for i in range(x.size):
            s += x[i]
        return s

    @njit(cache=True, parallel=True)
    def k_prange(x):
        s = 0.0
        for i in prange(x.size):
            s += x[i]
        return s


REQUIRED = "REQUIRED"
OPTIONAL = "OPTIONAL"
UNSUPPORTED = "UNSUPPORTED"

_rows = []


def record(name, status, detail=""):
    _rows.append((name, status, detail))
    print(f"  [{status:<11}] {name}  {detail}")


def run(name, fn, *args, expect=None, status=REQUIRED):
    """Compile+run a probe; verify numeric agreement when ``expect`` given."""
    try:
        out = fn(*args)
    except Exception as exc:  # noqa: BLE001
        if status == UNSUPPORTED:
            record(name, UNSUPPORTED, f"{type(exc).__name__}: {exc}")
        else:
            record(name, "FAIL", f"{type(exc).__name__}: {exc}")
        return None
    if expect is not None:
        try:
            got = np.asarray(out, dtype=float)
            if not np.allclose(got, np.asarray(expect, dtype=float),
                               rtol=1e-12, atol=1e-12):
                record(name, "FAIL", "numeric mismatch vs numpy")
                return out
        except Exception:  # noqa: BLE001
            pass
    record(name, "OK")
    return out


def main():
    print("=" * 74)
    print("PHASE 0 PREFLIGHT: numba compatibility")
    print("=" * 74)
    print(f"  python : {sys.executable}")
    print(f"  numpy  : {np.__version__}")
    if not HAVE_NUMBA:
        print(f"  numba  : NOT INSTALLED ({type(_NUMBA_ERR).__name__}: {_NUMBA_ERR})")
        print("  VERDICT: FAIL (numba unavailable)")
        return 1
    print(f"  numba  : {numba.__version__}")
    print(f"  threads: {numba.get_num_threads()}")
    print()

    rng = np.random.default_rng(0)

    print("-- LAPACK wrappers (scipy lu_factor replacement) --")
    A = rng.normal(size=(6, 6))
    A = A @ A.T + 6 * np.eye(6)
    run("np.linalg.inv", k_inv, A, expect=np.linalg.inv(A))
    b = rng.normal(size=(6, 3))
    run("np.linalg.solve (m,3) RHS", k_solve_multi, A, b,
        expect=np.linalg.solve(A, b))
    bv = rng.normal(size=6)
    run("np.linalg.solve (vector)", k_solve, A, bv,
        expect=np.linalg.solve(A, bv))
    print()

    print("-- sorting / selection --")
    x = rng.normal(size=200)
    run("np.argsort (default)", k_argsort_default, x,
        expect=np.argsort(x).astype(float))
    run("np.argsort(kind='mergesort')", k_argsort_mergesort, x,
        expect=np.argsort(x, kind="mergesort").astype(float))
    run("np.argsort(kind='stable')", k_argsort_stable, x,
        expect=np.argsort(x, kind="stable").astype(float))
    run("np.flatnonzero", k_flatnonzero, np.array([0.0, 1.0, 0.0, 2.0]),
        expect=np.flatnonzero(np.array([0.0, 1.0, 0.0, 2.0])).astype(float))
    run("np.argmin/argmax", k_argminmax, x,
        expect=[np.argmin(x), np.argmax(x)])
    print()

    print("-- elementwise / array construction --")
    run("isfinite/isinf/sqrt", k_misc, np.array([1.0, np.inf, -1.0, np.nan]))
    run("column_stack/vstack", k_stack, np.arange(3.0), np.arange(3.0) + 1)
    run("empty/zeros/full/ones/where/concat/sort", k_build, 5)
    run("np.isclose(atol,rtol)", k_isclose,
        np.array([1.0, 2.0]), np.array([1.0 + 1e-12, 2.0]),
        expect=np.isclose(np.array([1.0, 2.0]),
                          np.array([1.0 + 1e-12, 2.0]), atol=1e-9, rtol=1e-9))
    print()

    print("-- control flow / outputs --")
    run("raise inside njit", k_raise, 3)
    try:
        k_raise(-1)
        record("raise propagates", "FAIL", "no exception raised")
    except ValueError:
        record("raise propagates", "OK")
    run("tuple return", k_tuple_ret, 4)
    try:
        got = list(k_generator(4))
        if got == [0, 1, 2, 3]:
            record("njit generator", "OK")
        else:
            record("njit generator", "FAIL", f"got {got}")
    except Exception as exc:  # noqa: BLE001
        record("njit generator", "FAIL", f"{type(exc).__name__}: {exc}")
    run("nogil=True", k_nogil, np.ones(1000))
    run("parallel=True/prange", k_prange, np.ones(1000))
    print()

    print("-- expected-unsupported (must be rewritten) --")
    run("np.setdiff1d", k_setdiff1d, np.array([1, 2, 3]), np.array([2]),
        status=UNSUPPORTED)
    run("scipy.linalg.lu_factor in njit", k_lu_scipy, A, status=UNSUPPORTED)
    print()

    # --- numeric agreement detail for the LAPACK wrappers -------------------
    print("-- numeric agreement (numba vs numpy) --")
    for n in (4, 20, 60):
        M = rng.normal(size=(n, n))
        M = M @ M.T + n * np.eye(n)
        di = float(np.max(np.abs(k_inv(M) - np.linalg.inv(M))))
        rhs = rng.normal(size=(n, 3))
        ds = float(np.max(np.abs(k_solve_multi(M, rhs) - np.linalg.solve(M, rhs))))
        print(f"  n={n:>3}  max|inv diff|={di:.3e}  max|solve diff|={ds:.3e}")
    print()

    # --- compile/warm timing ------------------------------------------------
    print("-- steady-state timing sanity (already compiled) --")
    M = rng.normal(size=(120, 120))
    M = M @ M.T + 120 * np.eye(120)
    t0 = time.perf_counter()
    for _ in range(50):
        k_inv(M)
    print(f"  50x inv(120): {(time.perf_counter() - t0) * 1e3:.1f} ms")
    print()

    fails = [r for r in _rows if r[1] == "FAIL"]
    print("=" * 74)
    print(f"  OK={sum(1 for r in _rows if r[1] == 'OK')}  "
          f"FAIL={len(fails)}  "
          f"UNSUPPORTED={sum(1 for r in _rows if r[1] == UNSUPPORTED)}")
    if fails:
        print("  VERDICT: FAIL")
        for name, _, detail in fails:
            print(f"    - {name}: {detail}")
        return 1
    print("  VERDICT: PASS - all required primitives compile; "
          "rewrite scipy lu in kernels.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

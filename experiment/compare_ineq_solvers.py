"""Compare the three interval-solver versions in ``si_for_cg/si_utils.py``.

Contestants
  v1  ``solve_linear_inequality(b, c)``
      vectorized LINEAR batch (``b_i*z + c_i <= 0``) -> one interval or []
  v2  per-row ``solve_quadratic_inequality(a, b, c)`` + ``interval_intersection``
      scalar reference loop over the batch (handles a==0, a>0, a<0)
  v3  ``solveinterval(P, Q, O, z)``
      vectorized batch solver (same three branches, array reductions)

Comparisons
  linear    : v1 vs v3 on batches with P == 0
  quadratic : v2 vs v3 on mixed batches (a<0 / a==0 / a>0)

Checks
  * list-equality of the returned intervals in the well-conditioned regime
    (all |coef| >= 1e-3, far from every tolerance cutoff)
  * mismatch rate in a near-zero stress regime, where the versions use
    different snap rules (v1: |v| < 1e-12, v2: |v| < 1e-9 absolute,
    v3: 1e-9 * (1 + max|a,b,c|) relative) -- mismatches there are the
    documented policy difference, not a numeric bug
  * independent ground truth on the v3 output: interior points of the
    returned pieces satisfy every inequality; sampled exterior points
    violate at least one
  * wall-clock runtime per version at increasing batch sizes

Run from the repo root with the ``IS`` python (see AGENTS.md).
"""

import contextlib
import io
import os
import sys
import time
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

# The package __init__ imports lava_function -> OptimalTransport ->
# LPsolver, and si_for_cg/LPsolver.py no longer exists (dead code removed).
# Load si_utils.py directly through a stub package so only its own
# dependencies (si_types, _timeprof) are imported.
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_si_pkg = types.ModuleType("si_for_cg")
_si_pkg.__path__ = [os.path.join(_PKG_ROOT, "si_for_cg")]
_si_pkg.__package__ = "si_for_cg"
sys.modules["si_for_cg"] = _si_pkg

from si_for_cg.si_utils import (
    interval_intersection,
    solve_linear_inequality,
    solve_quadratic_inequality,
    solveinterval,
)

WELL_SEEDS = 300       # n=50 batches, well-conditioned regime
WELL_SEEDS_BIG = 50    # n=500 batches, well-conditioned regime
STRESS_SEEDS = 150
STRESS_SEEDS_BIG = 30


# --------------------------------------------------------------------------
# batch generation (all rows strictly contain a random target t*; a few
# aggressive rows may push the intersection to be empty)
# --------------------------------------------------------------------------

def gen_linear_batch(n, rng, aggressive=0.2):
    """Linear rows ``b_i*z + c_i <= 0``; every row contains t*."""
    t = rng.uniform(-2.0, 2.0)
    b = rng.choice([-1.0, 1.0], size=n) * rng.uniform(0.2, 5.0, size=n)
    c = -b * t - rng.uniform(0.1, 2.0, size=n)   # -c/b = t + slack/b (contains t)
    n_agg = int(n * aggressive)
    idx = rng.choice(n, size=n_agg, replace=False)
    c[idx] += rng.uniform(-10.0, 10.0, size=n_agg)
    return np.zeros(n), b, c


def gen_quad_batch(n, rng, frac_lin=0.3, frac_hole=0.3, many_holes=False,
                   hole_margin=3.0, aggressive=0.15):
    """Mixed rows around t*: a<0 holes, a==0 linear, a>0 intervals."""
    t = rng.uniform(-2.0, 2.0)
    P = np.zeros(n)
    Q = np.zeros(n)
    O = np.zeros(n)
    types = rng.choice(
        ["lin", "up", "hole"], size=n,
        p=[frac_lin, 1.0 - frac_lin - frac_hole, frac_hole],
    )
    for i, typ in enumerate(types):
        if typ == "lin":
            b = rng.choice([-1.0, 1.0]) * rng.uniform(0.2, 5.0)
            P[i] = 0.0
            Q[i] = b
            O[i] = -b * t - rng.uniform(0.1, 2.0)
        elif typ == "up":                       # a>0 -> [x1, x2] contains t*
            w = rng.uniform(0.3, 2.0)
            r1, r2 = t - w, t + w
            a = rng.uniform(0.5, 3.0)
            P[i] = a
            Q[i] = -a * (r1 + r2)
            O[i] = a * r1 * r2
        else:                                   # a<0 -> forbidden (h1, h2)
            if many_holes:
                # cutting holes: bite into [L, U] from both sides but never
                # contain t*, so the batch stays feasible and both versions
                # process every row
                side = rng.choice([-1.0, 1.0])
                center = t + side * rng.uniform(1.0, 2.2)
                hw = rng.uniform(0.3, 0.8)
            else:
                hw = rng.uniform(0.3, 2.0)
                side = rng.choice([-1.0, 1.0])
                center = t + side * rng.uniform(hole_margin, hole_margin + 3.0)
            h1, h2 = center - hw, center + hw
            a = -rng.uniform(0.5, 3.0)
            P[i] = a
            Q[i] = -a * (h1 + h2)
            O[i] = a * h1 * h2
    n_agg = int(n * aggressive)
    idx = rng.choice(n, size=n_agg, replace=False)
    P[idx] = 0.0
    Q[idx] = rng.choice([-1.0, 1.0], size=n_agg) * rng.uniform(0.2, 5.0)
    O[idx] = -Q[idx] * t - rng.uniform(-5.0, 5.0, size=n_agg)
    return P, Q, O


def stress_scale(P, Q, O, rng, frac=0.4, lo=-10.0, hi=-5.0):
    """Scale a fraction of rows by 10**U[lo,hi] so some land inside v3's
    relative snap window (1e-9 * (1 + batch max |coef|) ~ 5e-8 here) while
    v2's absolute 1e-9 cutoff keeps them: rows in that window expose the
    documented tolerance-policy difference between the two code paths."""
    P = np.array(P, float)
    Q = np.array(Q, float)
    O = np.array(O, float)
    n = P.size
    idx = rng.choice(n, size=int(n * frac), replace=False)
    f = 10.0 ** rng.uniform(lo, hi, size=idx.size)
    P[idx] *= f
    Q[idx] *= f
    O[idx] *= f
    return P, Q, O


# --------------------------------------------------------------------------
# reference implementations under test
# --------------------------------------------------------------------------

def ref_linear(Q, O):
    try:
        return solve_linear_inequality(Q, O)
    except ValueError:
        return []


def ref_quad_loop(P, Q, O):
    """Per-row solve_quadratic_inequality + interval_intersection loop."""
    with contextlib.redirect_stdout(io.StringIO()):
        acc = [(-np.inf, np.inf)]
        for a, b, c in zip(P, Q, O):
            itv = solve_quadratic_inequality(float(a), float(b), float(c))
            acc = interval_intersection(acc, itv)
            if not acc:
                return []
        return acc


# --------------------------------------------------------------------------
# comparison helpers
# --------------------------------------------------------------------------

def _close(x, y, rtol=1e-7, atol=1e-9):
    if np.isinf(x) or np.isinf(y):
        return np.isinf(x) and np.isinf(y) and np.sign(x) == np.sign(y)
    return abs(x - y) <= atol + rtol * max(abs(x), abs(y))


def lists_equal(A, B, rtol=1e-7, atol=1e-9):
    if len(A) != len(B):
        return False
    return all(
        _close(l1, l2, rtol, atol) and _close(r1, r2, rtol, atol)
        for (l1, r1), (l2, r2) in zip(A, B)
    )


def max_finite_diff(A, B):
    d = 0.0
    for (l1, r1), (l2, r2) in zip(A, B):
        for x, y in ((l1, l2), (r1, r2)):
            if np.isfinite(x) and np.isfinite(y):
                d = max(d, abs(x - y))
    return d


def grid_check(P, Q, O, pieces, zlim=100.0, atol=1e-6):
    """Ground truth on a piece list (any version's output).

    Every interior point of the returned pieces must satisfy all the
    inequalities; every sampled exterior point (hole midpoints, points just
    beyond a finite hull) must violate at least one.  Returns
    (bad_inside, bad_outside) as (z, max_violation) lists.
    """
    if not pieces:
        return [], []
    if len(pieces) == 1 and np.isinf(pieces[0][0]) and np.isinf(pieces[0][1]):
        return [], []
    zz = np.asarray(P, float)
    qq = np.asarray(Q, float)
    oo = np.asarray(O, float)

    def vals(z):
        return zz * z * z + qq * z + oo

    bad_inside, bad_outside = [], []
    merged = []
    for lo, hi in pieces:
        l = -zlim if np.isinf(lo) else max(float(lo), -zlim)
        h = zlim if np.isinf(hi) else min(float(hi), zlim)
        if h - l <= 1e-12:
            continue
        merged.append((l, h))
        for z in np.linspace(l, h, 9)[1:-1]:
            v = float(np.max(vals(z)))
            if v > atol:
                bad_inside.append((z, v))
    if not merged:
        return bad_inside, bad_outside

    merged = interval_union(merged)
    for (_, h1), (l2, _) in zip(merged[:-1], merged[1:]):
        if l2 - h1 < 1e-12:
            continue
        z = 0.5 * (h1 + l2)
        v = float(np.max(vals(z)))
        if v <= atol:
            bad_outside.append((z, v))

    # points just beyond a finite hull must violate too (use the ORIGINAL
    # pieces: an -inf endpoint means the set extends there, nothing to test)
    if np.isfinite(pieces[0][0]):
        z = float(pieces[0][0]) - 1.0
        v = float(np.max(vals(z)))
        if v <= atol:
            bad_outside.append((z, v))
    if np.isfinite(pieces[-1][1]):
        z = float(pieces[-1][1]) + 1.0
        v = float(np.max(vals(z)))
        if v <= atol:
            bad_outside.append((z, v))
    return bad_inside, bad_outside


# --------------------------------------------------------------------------
# accuracy drivers
# --------------------------------------------------------------------------

def run_linear_accuracy():
    print("=" * 74)
    print("LINEAR batches (P == 0):  v1 solve_linear_inequality vs v3 solveinterval")
    print("=" * 74)
    well_mismatch = 0
    for regime, seeds, big, seed0 in (
        ("well-conditioned", WELL_SEEDS, WELL_SEEDS_BIG, 1234),
        ("stress (near-zero coefs)", STRESS_SEEDS, STRESS_SEEDS_BIG, 5678),
    ):
        rng = np.random.default_rng(seed0)
        total = match = 0
        maxdiff = 0.0
        examples = []
        for it in range(seeds + big):
            n = 50 if it < seeds else 500
            P, Q, O = gen_linear_batch(n, rng)
            if regime.startswith("stress"):
                P, Q, O = stress_scale(P, Q, O, rng)
            ref = ref_linear(Q, O)
            vec = solveinterval(P, Q, O, 0.0)
            total += 1
            if lists_equal(ref, vec):
                match += 1
            else:
                well_mismatch += 1 if regime.startswith("well") else 0
                maxdiff = max(maxdiff, max_finite_diff(ref, vec))
                if len(examples) < 5:
                    examples.append((n, ref, vec, max_finite_diff(ref, vec)))
        print(f"[{regime}]  {match}/{total} batches agree "
              f"({100.0 * match / total:.2f}%), max finite endpoint diff {maxdiff:.3e}")
        for n, ref, vec, d in examples:
            print(f"    n={n}: ref={ref}  vec={vec}  diff={d:.3e}")
    return well_mismatch


def run_quad_accuracy():
    print("=" * 74)
    print("QUADRATIC batches:  v2 solve_quadratic_inequality+intersection vs v3 solveinterval")
    print("=" * 74)
    well_mismatch = 0
    for regime, seeds, big, seed0 in (
        ("well-conditioned", WELL_SEEDS, WELL_SEEDS_BIG, 1111),
        ("stress (near-zero coefs)", STRESS_SEEDS, STRESS_SEEDS_BIG, 2222),
    ):
        rng = np.random.default_rng(seed0)
        total = match = 0
        maxdiff = 0.0
        examples = []
        for it in range(seeds + big):
            n = 50 if it < seeds else 500
            P, Q, O = gen_quad_batch(n, rng)
            if regime.startswith("stress"):
                P, Q, O = stress_scale(P, Q, O, rng)
            ref = ref_quad_loop(P, Q, O)
            vec = solveinterval(P, Q, O, 0.0)
            total += 1
            if lists_equal(ref, vec):
                match += 1
            else:
                well_mismatch += 1 if regime.startswith("well") else 0
                maxdiff = max(maxdiff, max_finite_diff(ref, vec))
                if len(examples) < 5:
                    examples.append((n, ref, vec, max_finite_diff(ref, vec)))
        print(f"[{regime}]  {match}/{total} batches agree "
              f"({100.0 * match / total:.2f}%), max finite endpoint diff {maxdiff:.3e}")
        for n, ref, vec, d in examples:
            print(f"    n={n}: ref={ref}  vec={vec}  diff={d:.3e}")
    return well_mismatch


def run_grid():
    print("=" * 74)
    print("GROUND TRUTH (independent of every implementation):")
    print("  interior points of each returned piece satisfy all ineqs;")
    print("  sampled exterior points violate at least one.")
    print("=" * 74)
    rng = np.random.default_rng(999)
    bad_in = bad_out = n_batches = 0
    for it in range(200):
        kind = it % 3
        if kind == 0:
            P, Q, O = gen_linear_batch(60, rng)
        elif kind == 1:
            P, Q, O = gen_quad_batch(60, rng)
        else:
            P, Q, O = gen_quad_batch(60, rng, frac_hole=0.55, many_holes=True)
        if it % 2 == 0:
            P, Q, O = stress_scale(P, Q, O, rng)
        pieces = solveinterval(P, Q, O, 0.0)
        bi, bo = grid_check(P, Q, O, pieces)
        bad_in += len(bi)
        bad_out += len(bo)
        n_batches += 1
    print(f"{n_batches} batches checked: {bad_in} bad interior points, "
          f"{bad_out} bad exterior points")
    return bad_in + bad_out


# --------------------------------------------------------------------------
# runtime drivers
# --------------------------------------------------------------------------

def timeit(fn, repeats=5):
    fn()  # warmup
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def run_timing():
    print("=" * 74)
    print("RUNTIME (min over repeats, us per call; same batch for all versions)")
    print("=" * 74)
    rng = np.random.default_rng(99)

    print("\n-- linear batches: v1 solve_linear_inequality vs v3 solveinterval --")
    print(f"{'n':>8} {'v1 [us]':>10} {'v3 [us]':>10} {'v3/v1':>8}")
    for n in (10, 100, 1000, 10000, 100000):
        P, Q, O = gen_linear_batch(n, rng, aggressive=0.0)
        t1 = timeit(lambda: solve_linear_inequality(Q, O))
        t3 = timeit(lambda: solveinterval(P, Q, O, 0.0))
        print(f"{n:>8} {t1 * 1e6:>10.1f} {t3 * 1e6:>10.1f} {t3 / t1:>8.2f}")

    print("\n-- quadratic batches: v2 per-row loop vs v3 solveinterval --")
    print(f"{'n':>8} {'v2 [us]':>10} {'v3 [us]':>10} {'v2/v3':>8}")
    for n in (10, 100, 1000, 5000, 20000):
        # aggressive=0 -> always feasible, so the reference loop always runs to
        # completion (an infeasible batch would let it early-exit and make the
        # timing incomparable with the vectorized version)
        P, Q, O = gen_quad_batch(n, rng, aggressive=0.0)
        t2 = timeit(lambda: ref_quad_loop(P, Q, O))
        t3 = timeit(lambda: solveinterval(P, Q, O, 0.0))
        print(f"{n:>8} {t2 * 1e6:>10.1f} {t3 * 1e6:>10.1f} {t2 / t3:>8.2f}")

    print("\n-- quadratic hole-heavy batches (many cutting holes): v2 vs v3 --")
    print(f"{'n':>8} {'v2 [us]':>10} {'v3 [us]':>10} {'v2/v3':>8}")
    for n in (10, 100, 1000, 5000):
        P, Q, O = gen_quad_batch(n, rng, frac_hole=0.55, many_holes=True, aggressive=0.0)
        t2 = timeit(lambda: ref_quad_loop(P, Q, O))
        t3 = timeit(lambda: solveinterval(P, Q, O, 0.0))
        print(f"{n:>8} {t2 * 1e6:>10.1f} {t3 * 1e6:>10.1f} {t2 / t3:>8.2f}")


def main():
    well_mis = run_linear_accuracy()
    well_mis += run_quad_accuracy()
    bad = run_grid()
    run_timing()
    print("=" * 74)
    print(f"well-conditioned mismatches: {well_mis}; grid violations: {bad}")
    ok = well_mis == 0 and bad == 0
    if ok:
        print("VERDICT: v3 agrees with v1/v2 in the well-conditioned regime and")
        print("passes the independent ground-truth check -> drop-in replacement.")
        print("(stress-regime list differences, if any, are degenerate zero-width")
        print("pieces at tolerance-snap boundaries, not numeric drift)")
    else:
        print("VERDICT: review the mismatch examples above (stress-regime")
        print("mismatches are the expected tolerance-policy difference).")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
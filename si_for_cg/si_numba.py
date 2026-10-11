"""Numba kernels for the ``parametric_nested`` sort-region walk (N1).

This module is the numba port of the hot inferred sort walk in
``si_for_cg/si_sort_walk.py`` (``iter_sort_regions_table`` and its helpers
``pair_diffs`` / ``quad_pieces`` / ``region_containing`` / ``flip_reversal``
and the ``_resort`` / ``_resort_cg`` fallbacks), plus the small LAPACK/batch
helpers they need (``solveinterval``, ``score_coefficients``,
``cg_score_coefficients``).

Design (see NUMBA_PLAN.md):

* ``_*_kernel`` are ``@njit`` kernels taking only arrays/scalars.
* ``n_*`` are the public wrappers used by ``experiment/test_numba_parity.py``;
  they allocate output buffers and return plain arrays.
* ``w_iter_*`` are generator wrappers used by ``si_sort_walk.py`` so the
  pipeline can dispatch to numba without changing its call contract.

Determinism/accuracy: ``fastmath=False``; scipy ``lu_factor/lu_solve`` is
replaced by ``np.linalg.solve`` (Phase 0 measured the multi-RHS solve
bit-identical to numpy and inv within ~1e-17).  Tolerances and operation order
mirror the pure-Python reference exactly.
"""

from __future__ import annotations

import os

import numpy as np

try:  # pragma: no cover - exercised by the preflight harness
    from numba import njit
    AVAILABLE = True
except Exception:  # noqa: BLE001
    AVAILABLE = False

    def njit(*dargs, **dkw):  # type: ignore[misc]
        def deco(f):
            return f

        if len(dargs) == 1 and callable(dargs[0]) and not dkw:
            return dargs[0]
        return deco


USE = os.environ.get("SI_USE_NUMBA") == "1"

TOL = 1e-9
TOL_CONT = 1e-7
MAX_STALE = 1000


# ===========================================================================
# batch interval solver (numba port of si_utils.solveinterval)
# ===========================================================================
@njit(cache=True, fastmath=False)
def _solveinterval_kernel(P, Q, O, z):
    """Return (out, k): out[:k] are the pieces of the intersection.

    Mirrors ``si_utils.solveinterval`` (relative snap 1e-9*(1+scale), the
    a==0 / a>0 / a<0 branches, and the hole subtraction).
    """
    n = P.size
    out = np.empty((n + 1, 2))
    if n == 0:
        out[0, 0] = -np.inf
        out[0, 1] = np.inf
        return out, 1

    L = -np.inf
    U = np.inf
    hole_lo = np.empty(n)
    hole_hi = np.empty(n)
    nh = 0

    for i in range(n):
        a = P[i]
        b = Q[i]
        c = O[i]
        scale = max(abs(a), max(abs(b), abs(c)))
        tol = 1e-9 + 1e-9 * scale
        if abs(a) <= tol:
            a = 0.0
        if abs(b) <= tol:
            b = 0.0
        if abs(c) <= tol:
            c = 0.0

        if a == 0.0:
            if b > 0.0:
                v = -c / b
                if v < U:
                    U = v
            elif b < 0.0:
                v = -c / b
                if v > L:
                    L = v
            else:
                if c > 0.0:
                    return out, 0
        else:
            delta = b * b - 4.0 * a * c
            if delta < 0.0:
                if a > 0.0:
                    return out, 0
            else:
                sq = np.sqrt(delta)
                x1 = (-b - sq) / (2.0 * a)
                x2 = (-b + sq) / (2.0 * a)
                if a > 0.0:
                    if x1 > L:
                        L = x1
                    if x2 < U:
                        U = x2
                else:
                    hole_lo[nh] = x2
                    hole_hi[nh] = x1
                    nh += 1

    if not (L < U):
        return out, 0
    if nh == 0:
        out[0, 0] = L
        out[0, 1] = U
        return out, 1

    m = 0
    for i in range(nh):
        flo = hole_lo[i]
        fhi = hole_hi[i]
        if fhi > L and flo < U:
            if flo < L:
                flo = L
            if fhi > U:
                fhi = U
            hole_lo[m] = flo
            hole_hi[m] = fhi
            m += 1
    nh = m
    if nh == 0:
        out[0, 0] = L
        out[0, 1] = U
        return out, 1

    order = np.argsort(hole_lo[:nh], kind="stable")
    merged_lo = np.empty(nh)
    merged_hi = np.empty(nh)
    first = order[0]
    merged_lo[0] = hole_lo[first]
    merged_hi[0] = hole_hi[first]
    nm = 1
    for idx in range(1, nh):
        i = order[idx]
        if hole_lo[i] <= merged_hi[nm - 1]:
            if hole_hi[i] > merged_hi[nm - 1]:
                merged_hi[nm - 1] = hole_hi[i]
        else:
            merged_lo[nm] = hole_lo[i]
            merged_hi[nm] = hole_hi[i]
            nm += 1

    k = 0
    cur = L
    for j in range(nm):
        lo_f = merged_lo[j]
        hi_f = merged_hi[j]
        if cur < lo_f:
            out[k, 0] = cur
            out[k, 1] = lo_f
            k += 1
        if hi_f > cur:
            cur = hi_f
    if cur < U:
        out[k, 0] = cur
        out[k, 1] = U
        k += 1
    return out, k


def n_solveinterval(P, Q, O, z):
    """Public wrapper: pieces of the batch quadratic intersection."""
    P = np.asarray(P, dtype=float).reshape(-1)
    Q = np.asarray(Q, dtype=float).reshape(-1)
    O = np.asarray(O, dtype=float).reshape(-1)
    out, k = _solveinterval_kernel(P, Q, O, float(z))
    return out[:k]


# ===========================================================================
# quad pieces (numba port of si_sort_walk.quad_pieces)
# ===========================================================================
@njit(cache=True, fastmath=False)
def _quad_pieces_kernel(a, b, c, z):
    n = a.size
    left = np.full(n, -np.inf)
    right = np.full(n, np.inf)
    ok = np.ones(n, dtype=np.bool_)
    for i in range(n):
        ai = a[i]
        bi = b[i]
        ci = c[i]
        if abs(ai) < 1e-9:
            ai = 0.0
        if abs(bi) < 1e-9:
            bi = 0.0
        if abs(ci) < 1e-9:
            ci = 0.0
        if ai == 0.0:
            if bi == 0.0:
                ok[i] = ci <= 0.0
            elif bi > 0.0:
                right[i] = -ci / bi
            else:
                left[i] = -ci / bi
        else:
            delta = bi * bi - 4.0 * ai * ci
            if delta < 0.0:
                if ai > 0.0:
                    ok[i] = False
            else:
                sq = np.sqrt(delta)
                x1 = (-bi - sq) / (2.0 * ai)
                x2 = (-bi + sq) / (2.0 * ai)
                if ai > 0.0:
                    left[i] = x1
                    right[i] = x2
                    if z < x1 or z > x2:
                        ok[i] = False
                else:
                    if z <= x2:
                        right[i] = x2
                    elif z >= x1:
                        left[i] = x1
                    else:
                        ok[i] = False
    return left, right, ok


def n_quad_pieces(a, b, c, z):
    """Public wrapper: (left, right, ok) arrays for ``a z^2 + b z + c <= 0``."""
    return _quad_pieces_kernel(
        np.asarray(a, dtype=float).reshape(-1),
        np.asarray(b, dtype=float).reshape(-1),
        np.asarray(c, dtype=float).reshape(-1),
        float(z),
    )


# ===========================================================================
# coefficient tables
# ===========================================================================
@njit(cache=True, fastmath=False)
def _score_coefficients_kernel(S_, basis, w_cost, v_cost, u_cost, nt):
    m = basis.size
    sbT = S_[:, basis].T
    rhs = np.empty((m, 3))
    for i in range(m):
        b = basis[i]
        rhs[i, 0] = -u_cost[b]
        rhs[i, 1] = -v_cost[b]
        rhs[i, 2] = -w_cost[b]
    coef = np.linalg.solve(sbT, rhs)
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    for i in range(nt):
        s0 += coef[i, 0]
        s1 += coef[i, 1]
        s2 += coef[i, 2]
    scale = 1.0 + 1.0 / (nt - 1.0)
    out = np.empty((nt, 3))
    for i in range(nt):
        out[i, 0] = scale * coef[i, 0] - s0
        out[i, 1] = scale * coef[i, 1] - s1
        out[i, 2] = scale * coef[i, 2] - s2
    return out


def n_score_coefficients(S_, basis, w_cost, v_cost, u_cost, nt):
    """Public wrapper: per-sample score quadratics for a fixed OT basis."""
    return _score_coefficients_kernel(
        np.asarray(S_, dtype=float),
        np.asarray(basis, dtype=np.int64).reshape(-1),
        np.asarray(w_cost, dtype=float).reshape(-1),
        np.asarray(v_cost, dtype=float).reshape(-1),
        np.asarray(u_cost, dtype=float).reshape(-1),
        int(nt),
    )


@njit(cache=True, fastmath=False)
def _cg_score_coefficients_kernel(invH, a_vec, b_vec):
    n = invH.shape[0]
    d = np.empty(n)
    for i in range(n):
        d[i] = invH[i, i]
    u = invH.dot(a_vec)
    v = invH.dot(b_vec)
    out = np.empty((n, 3))
    for i in range(n):
        out[i, 0] = v[i] * v[i] / d[i]
        out[i, 1] = 2.0 * u[i] * v[i] / d[i]
        out[i, 2] = u[i] * u[i] / d[i]
    return out


def n_cg_score_coefficients(invH, a_vec, b_vec):
    """Public wrapper: per-sample cg-score quadratics (global in z)."""
    return _cg_score_coefficients_kernel(
        np.asarray(invH, dtype=float),
        np.asarray(a_vec, dtype=float).reshape(-1),
        np.asarray(b_vec, dtype=float).reshape(-1),
    )


# ===========================================================================
# N2: conditioning_Lasso (+ solve_linear_inequality)
# ===========================================================================
@njit(cache=True, fastmath=False)
def _solve_linear_inequality_kernel(b, c):
    """Return (out, k, feasible) for ``b_i x + c_i <= 0``.

    ``feasible=False`` reproduces the reference ``ValueError`` (``b==0 & c>0``);
    ``k=0`` with ``feasible=True`` is the empty intersection.
    """
    n = b.size
    out = np.empty((1, 2))
    bb = b.copy()
    cc = c.copy()
    for i in range(n):
        if abs(bb[i]) < 1e-12:
            bb[i] = 0.0
        if abs(cc[i]) < 1e-12:
            cc[i] = 0.0
    for i in range(n):
        if bb[i] == 0.0 and cc[i] > 0.0:
            return out, 0, False
    left = -np.inf
    right = np.inf
    anypos = False
    anyneg = False
    for i in range(n):
        if bb[i] > 0.0:
            v = -cc[i] / bb[i]
            if (not anypos) or v < right:
                right = v
            anypos = True
        elif bb[i] < 0.0:
            v = -cc[i] / bb[i]
            if (not anyneg) or v > left:
                left = v
            anyneg = True
    if left > right:
        return out, 0, True
    out[0, 0] = left
    out[0, 1] = right
    return out, 1, True


@njit(cache=True, fastmath=False)
def _conditioning_lasso_kernel(X_, a, b, Lambda, M, Mc, sM):
    N = X_.shape[0]
    X_M = X_[:, M]
    X_Mc = X_[:, Mc]

    XtX = X_M.T.dot(X_M)
    XtX_inv = np.linalg.inv(XtX)
    X_M_plus = XtX_inv.dot(X_M.T)
    X_MT_plus = X_M.dot(XtX_inv)

    Atilde0 = -sM * X_M_plus
    Btilde0 = -Lambda * sM * XtX_inv.dot(sM)

    Proj = X_M.dot(X_M_plus)
    Atilde10 = X_Mc.T.dot(np.identity(N) - Proj)
    Atilde11 = -Atilde10

    tmp = X_Mc.T.dot(X_MT_plus).dot(sM)
    Btilde10 = np.ones((tmp.shape[0], 1)) - tmp
    Btilde11 = np.ones((tmp.shape[0], 1)) + tmp

    Atilde1 = np.vstack((Atilde10, Atilde11)) / Lambda
    Btilde1 = np.vstack((Btilde10, Btilde11))

    Atilde = np.vstack((Atilde0, Atilde1))
    Btilde = np.vstack((Btilde0, Btilde1))

    A = Atilde.dot(b)
    Bv = Btilde - Atilde.dot(a)
    return _solve_linear_inequality_kernel(A[:, 0], -Bv[:, 0])


def n_conditioning_lasso(X_, a, b, Lambda, M, Mc, sM):
    """Public wrapper: Lasso-support conditioning interval (raises on infeasible).

    Returns a Python ``list[(float, float)]`` to match the reference contract
    used by ``interval_intersection`` / ``containing_piece``.
    """
    X_ = np.asarray(X_, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1, 1)
    b = np.asarray(b, dtype=float).reshape(-1, 1)
    M = np.asarray(M, dtype=np.int64).reshape(-1)
    Mc = np.asarray(Mc, dtype=np.int64).reshape(-1)
    sM = np.asarray(sM, dtype=float).reshape(-1, 1)
    out, k, feasible = _conditioning_lasso_kernel(
        X_, a, b, float(Lambda), M, Mc, sM)
    if not feasible:
        raise ValueError("No solution for the linear inequality")
    return [(float(out[i, 0]), float(out[i, 1])) for i in range(k)]


# ===========================================================================
# N3: constructOMEGA
# ===========================================================================
@njit(cache=True, fastmath=False)
def _construct_omega_kernel(n, m):
    out = np.zeros((n * m, n + m))
    for i in range(n):
        for r in range(m):
            row = i * m + r
            out[row, i] = 1.0
            out[row, n + r] = -1.0
    return out


def n_construct_omega(n, m):
    """Public wrapper: the ``(n*m, n+m)`` OT incidence matrix."""
    return _construct_omega_kernel(int(n), int(m))


# ===========================================================================
# walk internals
# ===========================================================================
@njit(cache=True, fastmath=False)
def _pair_diffs(s_coef, pi):
    last = pi.size - 1
    dA = np.empty(last)
    dB = np.empty(last)
    dC = np.empty(last)
    for m in range(last):
        a = pi[m]
        b = pi[m + 1]
        dA[m] = s_coef[a, 0] - s_coef[b, 0]
        dB[m] = s_coef[a, 1] - s_coef[b, 1]
        dC[m] = s_coef[a, 2] - s_coef[b, 2]
    return dA, dB, dC


@njit(cache=True, fastmath=False)
def _region_containing(dA, dB, dC, z, ml, mr):
    left, right, ok = _quad_pieces_kernel(dA, dB, dC, z)
    for i in range(ok.size):
        if not ok[i]:
            return 0.0, 0.0, False
    l = left[0]
    for i in range(left.size):
        if left[i] > l:
            l = left[i]
    r = right[0]
    for i in range(right.size):
        if right[i] < r:
            r = right[i]
    if l < ml:
        l = ml
    if r > mr:
        r = mr
    tol_z = 1e-9 * max(1.0, abs(z))
    if l > r or not (l - tol_z <= z <= r + tol_z):
        return 0.0, 0.0, False
    return l, r, True


@njit(cache=True, fastmath=False)
def _seed_pi(s_coef, z):
    n = s_coef.shape[0]
    scores = np.empty(n)
    for i in range(n):
        scores[i] = s_coef[i, 0] * z * z + s_coef[i, 1] * z + s_coef[i, 2]
    return np.argsort(scores, kind="stable")


@njit(cache=True, fastmath=False)
def _flip_reversal(s_coef, pi, R):
    dA, dB, dC = _pair_diffs(s_coef, pi)
    last = pi.size - 1
    mx = 0.0
    for i in range(pi.size):
        v = s_coef[i, 0] * R * R + s_coef[i, 1] * R + s_coef[i, 2]
        if abs(v) > mx:
            mx = abs(v)
    scale = max(1.0, mx)
    tight = np.empty(last, dtype=np.bool_)
    strict = np.empty(last, dtype=np.bool_)
    for m in range(last):
        dval = dA[m] * R * R + dB[m] * R + dC[m]
        dder = 2.0 * dA[m] * R + dB[m]
        tight[m] = abs(dval) <= 1e-8 * scale
        strict[m] = dder > 1e-9 * scale
    new_pi = pi.copy()
    m = 0
    while m < last:
        if not tight[m]:
            m += 1
            continue
        k = m
        while k + 1 < last and tight[k + 1]:
            k += 1
        any_strict = False
        for t in range(m, k + 1):
            if strict[t]:
                any_strict = True
                break
        if any_strict:
            i = m
            j = k + 1
            while i < j:
                tmp = new_pi[i]
                new_pi[i] = new_pi[j]
                new_pi[j] = tmp
                i += 1
                j -= 1
        m = k + 1
    return new_pi


@njit(cache=True, fastmath=False)
def _write_topk(pi, n_keep, out_row):
    n = pi.size
    tmp = np.empty(n_keep, dtype=np.int64)
    for k in range(n_keep):
        tmp[k] = pi[n - n_keep + k]
    for i in range(1, n_keep):
        key = tmp[i]
        j = i - 1
        while j >= 0 and tmp[j] > key:
            tmp[j + 1] = tmp[j]
            j -= 1
        tmp[j + 1] = key
    for k in range(n_keep):
        out_row[k] = tmp[k]


@njit(cache=True, fastmath=False)
def _fallback_cg(s_coef, z, ml, mr):
    pi = _seed_pi(s_coef, z)
    dA, dB, dC = _pair_diffs(s_coef, pi)
    l, r, ok = _region_containing(dA, dB, dC, z, ml, mr)
    return l, r, ok, pi


@njit(cache=True, fastmath=False)
def _select_sorted(S_, cost, basis, nt):
    sb = S_[:, basis]
    cb = cost[basis]
    x = np.linalg.solve(sb.T, cb)
    sc = np.empty(nt)
    s = 0.0
    for i in range(nt):
        sc[i] = -x[i]
        s += sc[i]
    for i in range(nt):
        sc[i] = 2.0 * sc[i] - s
    return np.argsort(sc, kind="stable")


@njit(cache=True, fastmath=False)
def _sort_conditioning_lava(z, S_, basis, wt, rt, ot, pi, nt):
    m = basis.size
    sbT = S_[:, basis].T
    rhs = np.empty((m, 3))
    for i in range(m):
        b = basis[i]
        rhs[i, 0] = -ot[b]
        rhs[i, 1] = -rt[b]
        rhs[i, 2] = -wt[b]
    coef = np.linalg.solve(sbT, rhs)
    s0 = 0.0
    s1 = 0.0
    s2 = 0.0
    for i in range(nt):
        s0 += coef[i, 0]
        s1 += coef[i, 1]
        s2 += coef[i, 2]
    scale = 1.0 + 1.0 / (nt - 1.0)
    scv = np.empty((nt, 3))
    for i in range(nt):
        scv[i, 0] = scale * coef[i, 0] - s0
        scv[i, 1] = scale * coef[i, 1] - s1
        scv[i, 2] = scale * coef[i, 2] - s2
    nn = pi.size
    P = np.empty(nn - 1)
    Q = np.empty(nn - 1)
    R = np.empty(nn - 1)
    for mm in range(nn - 1):
        a = pi[mm]
        b = pi[mm + 1]
        P[mm] = scv[a, 0] - scv[b, 0]
        Q[mm] = scv[a, 1] - scv[b, 1]
        R[mm] = scv[a, 2] - scv[b, 2]
    return _solveinterval_kernel(P, Q, R, z)


@njit(cache=True, fastmath=False)
def _fallback_lava(z, S_, wt, rt, ot, basis, nt, ml, mr):
    cost = wt + rt * z + ot * z * z
    pi = _select_sorted(S_, cost, basis, nt)
    out, k = _sort_conditioning_lava(z, S_, basis, wt, rt, ot, pi, nt)
    for i in range(k):
        l = out[i, 0]
        r = out[i, 1]
        if l < ml:
            l = ml
        if r > mr:
            r = mr
        if l <= r and l - TOL <= z <= r + TOL:
            return l, r, True, pi
    return 0.0, 0.0, False, pi


@njit(cache=True, fastmath=False)
def _walk_kernel(s_coef, ml, mr, step, n_keep, is_cg,
                 S_, wt, rt, ot, basis, nt,
                 out_z, out_sl, out_sr, out_sel, out_len, stats):
    cap = out_z.size
    count = 0
    z = ml + step
    pi = _seed_pi(s_coef, z)
    stale = 0

    while z < mr:
        dA, dB, dC = _pair_diffs(s_coef, pi)
        l, r, ok = _region_containing(dA, dB, dC, z, ml, mr)
        if not ok:
            stats[2] += 1
            if is_cg:
                l, r, ok, pi = _fallback_cg(s_coef, z, ml, mr)
            else:
                l, r, ok, pi = _fallback_lava(
                    z, S_, wt, rt, ot, basis, nt, ml, mr)
            if not ok:
                stats[3] += 1
                stale += 1
                if stale > MAX_STALE:
                    return count
                z += step
                continue
        stale = 0

        if count >= cap:
            return count
        out_z[count] = z
        out_sl[count] = l
        out_sr[count] = r
        _write_topk(pi, n_keep, out_sel[count])
        out_len[count] = n_keep
        count += 1
        stats[0] += 1

        if r + step >= mr:
            return count

        R = r
        pi_new = _flip_reversal(s_coef, pi, R)
        z_next = R + step
        dA2, dB2, dC2 = _pair_diffs(s_coef, pi_new)
        l2, r2, ok2 = _region_containing(dA2, dB2, dC2, z_next, ml, mr)
        if ok2 and abs(l2 - R) <= TOL_CONT * max(1.0, abs(R)):
            pi = pi_new
            stats[1] += 1
        else:
            stats[2] += 1
            if is_cg:
                lf, rf, okf, pi = _fallback_cg(s_coef, z_next, ml, mr)
            else:
                lf, rf, okf, pi = _fallback_lava(
                    z_next, S_, wt, rt, ot, basis, nt, ml, mr)
            if not okf:
                stats[3] += 1
                stale = 1
                z = z_next + step
                continue
        z = z_next

    return count


def _run_walk(s_coef, ml, mr, step, n_keep, is_cg,
              S_, wt, rt, ot, basis, nt):
    n = s_coef.shape[0]
    # Start small and grow on demand: ``w_iter_sort_regions`` (lava) is called
    # once per OT interval (hundreds of times), so preallocating the worst-case
    # O(n^2) buffer each call is pure memory churn.  Each interval usually
    # yields only a handful of regions.
    cap = 128
    dummy2d = np.zeros((1, 1))
    dummy1d = np.zeros(1)
    dummy_i = np.zeros(1, dtype=np.int64)
    if is_cg:
        S_a, wt_a, rt_a, ot_a = dummy2d, dummy1d, dummy1d, dummy1d
        basis_a, nt_a = dummy_i, 1
    else:
        S_a = np.asarray(S_, dtype=float)
        wt_a = np.asarray(wt, dtype=float).reshape(-1)
        rt_a = np.asarray(rt, dtype=float).reshape(-1)
        ot_a = np.asarray(ot, dtype=float).reshape(-1)
        basis_a = np.asarray(basis, dtype=np.int64).reshape(-1)
        nt_a = int(nt)

    while True:
        out_z = np.empty(cap)
        out_sl = np.empty(cap)
        out_sr = np.empty(cap)
        out_sel = np.empty((cap, n_keep), dtype=np.int64)
        out_len = np.empty(cap, dtype=np.int64)
        stats = np.zeros(4, dtype=np.int64)
        count = _walk_kernel(s_coef, float(ml), float(mr), float(step),
                             int(n_keep), bool(is_cg),
                             S_a, wt_a, rt_a, ot_a, basis_a, nt_a,
                             out_z, out_sl, out_sr, out_sel, out_len, stats)
        if count < cap:
            return (out_z[:count], out_sl[:count], out_sr[:count],
                    out_sel[:count], out_len[:count], int(count), stats)
        cap *= 2

        # print('outz',out_z)



def n_iter_cg_sort_regions(invH, a_vec, b_vec, zmin, zmax, step, n_keep):
    """Public wrapper: cg-score sort regions over ``[zmin, zmax]``.

    Returns ``(z_entry, sl, sr, sel_flat, sel_len, n_out)`` where ``sel_flat``
    has shape ``(n_out, n_keep)``.
    """
    s_coef = n_cg_score_coefficients(invH, a_vec, b_vec)
    z_e, sl, sr, sel, ln, n_out, _ = _run_walk(
        s_coef, zmin, zmax, step, int(n_keep), True,
        None, None, None, None, None, None)
    return z_e, sl, sr, sel, ln, n_out


def w_iter_cg_sort_regions(invH, a_vec, b_vec, zmin, zmax, config, n_keep,
                           stats=None):
    """Generator wrapper for ``si_sort_walk.iter_cg_sort_regions``."""
    z_e, sl, sr, sel, ln, n_out = n_iter_cg_sort_regions(
        invH, a_vec, b_vec, zmin, zmax, config.step, n_keep)
    for i in range(n_out):
        k = int(ln[i])
        yield (float(z_e[i]), float(sl[i]), float(sr[i]),
               [int(v) for v in sel[i, :k]])


def w_iter_sort_regions(adapter, wt, rt, ot, basis, ml, mr, config, stats=None):
    """Generator wrapper for ``si_sort_walk.iter_sort_regions`` (lava)."""
    nt = adapter.nt
    n_keep = max(1, int(nt * adapter.fraction))
    s_coef = n_score_coefficients(adapter.S_, basis, wt, rt, ot, nt)
    z_e, sl, sr, sel, ln, n_out, _ = _run_walk(
        s_coef, ml, mr, config.step, n_keep, False,
        adapter.S_, wt, rt, ot, basis, nt)
    for i in range(n_out):
        k = int(ln[i])
        yield (float(z_e[i]), float(sl[i]), float(sr[i]),
               [int(v) for v in sel[i, :k]])


__all__ = [
    "AVAILABLE",
    "USE",
    "n_solveinterval",
    "n_quad_pieces",
    "n_score_coefficients",
    "n_cg_score_coefficients",
    "n_iter_cg_sort_regions",
    "w_iter_cg_sort_regions",
    "w_iter_sort_regions",
    "n_conditioning_lasso",
    "n_construct_omega",
]

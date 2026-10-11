"""Event-driven Lasso support-interval walk (homotopy / LARS style).

The nested parametric scanners (``si_parametric.py``) enumerate the minimal
conditioning event ``{z : M(z) = M_obs}`` inside each merged sort-selection
span by *restarting and re-solving* the Lasso at every ``config.step``:

    z = z_entry
    while z < sr:
        M, sM = solve_Lasso(X[selection], Y(z))      # coordinate descent
        fs_itv = conditioning_Lasso(..., M, sM)      # KKT polyhedron
        z = fs_itv.right + step

This module replaces that inner loop with the event-driven walk of
``Lasso_condition_tracking.txt``: the Lasso is solved **once** per span, and
every later support region is inferred from the KKT inequality that becomes
tight at the current region's upper boundary.  The remaining work is pure
linear algebra on the fixed ``(X_, a, b, Lambda)``.

Adaptation notes (the reference code is from another codebase, so the
conventions here are re-derived from *our* ``conditioning_Lasso``):

* ``X_`` is the selected *row* set (``X[selection]``, shape ``(m, p)``) and
  ``a, b`` are the per-row intercept/slope of ``Y(z) = a + b z`` restricted to
  those rows (``a_vec[selection]``), matching ``conditioning_Lasso``.
* The KKT system is built exactly as in
  ``si_overconditoning.conditioning_Lasso`` /
  ``si_numba._conditioning_lasso_kernel``: rows ``[0, |M|)`` are the active
  sign conditions, ``[|M|, |M|+|Mc|)`` the inactive ``u <= +Lambda`` rows and
  ``[|M|+|Mc|, |M|+2|Mc|)`` the inactive ``u >= -Lambda`` rows; the interval is
  ``{z : A z <= B}``.
* At an *upper* boundary ``R`` the constraints with ``A_i > 0`` and
  ``B_i / A_i = R`` are tight and become violated just above ``R``.  Applying
  them updates ``(M, sM)``: an active row leaves, an inactive ``+`` row enters
  with sign ``+1`` and an inactive ``-`` row enters with sign ``-1`` (all
  simultaneous events are applied together -- degenerate cases are caught by
  the continuity check and fall back to a real Lasso solve).
"""

from __future__ import annotations

import numpy as np

from .si_utils import solve_Lasso, solve_linear_inequality

PATH_TOL = 1e-7        # boundary tightness / continuity tolerance
MAX_STALE = 1000       # consecutive failed reseeds before giving up

__all__ = [
    "PATH_TOL",
    "MAX_STALE",
    "conditioning_lasso_AB",
    "lasso_conditioning_interval",
    "binding_event_rows",
    "advance_active_set",
    "walk_lasso_support_regions",
]


def conditioning_lasso_AB(X_, a, b, Lambda, M, Mc, sM):
    """Build ``(A, B)`` so the over-conditioning interval is ``{z : A z <= B}``.

    Exact numpy mirror of the ``Atilde``/``Btilde`` construction in
    ``si_overconditioning.conditioning_Lasso`` (and its numba kernel); returns
    the raw 1-D ``A, B`` so the caller can also read off the boundary events.
    """
    X_ = np.asarray(X_, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1, 1)
    b = np.asarray(b, dtype=float).reshape(-1, 1)
    sM = np.asarray(sM, dtype=float).reshape(-1, 1)
    M = np.asarray(M, dtype=int).reshape(-1)
    Mc = np.asarray(Mc, dtype=int).reshape(-1)

    X_M = X_[:, M]
    X_Mc = X_[:, Mc]

    X_MT_X_M_inv = np.linalg.inv(X_M.T.dot(X_M))
    X_M_plus = X_MT_X_M_inv.dot(X_M.T)
    X_MT_plus = X_M.dot(X_MT_X_M_inv)

    Atilde0 = -sM * X_M_plus
    Btilde0 = -Lambda * sM * (X_MT_X_M_inv.dot(sM))

    Atilde10 = X_Mc.T.dot(np.identity(X_.shape[0]) - X_M.dot(X_M_plus))
    Atilde11 = -Atilde10

    tmp = X_Mc.T.dot(X_MT_plus).dot(sM)
    Btilde10 = np.ones((tmp.shape[0], 1)) - tmp
    Btilde11 = np.ones((tmp.shape[0], 1)) + tmp

    Atilde = np.vstack((Atilde0, np.vstack((Atilde10, Atilde11)) / Lambda))
    Btilde = np.vstack((Btilde0, np.vstack((Btilde10, Btilde11))))

    A = Atilde.dot(b).reshape(-1)
    B = (Btilde - Atilde.dot(a)).reshape(-1)
    return A, B


def lasso_conditioning_interval(X_, a, b, Lambda, M, Mc, sM):
    """Return ``(interval, A, B)`` for the fixed-support KKT region.

    ``interval`` is ``[(left, right)]`` (or ``[]`` when infeasible), matching
    ``conditioning_Lasso``.  ``A, B`` are returned so the caller can derive the
    boundary events without rebuilding the system.
    """
    A, B = conditioning_lasso_AB(X_, a, b, Lambda, M, Mc, sM)
    try:
        interval = solve_linear_inequality(A, -B)
    except ValueError:
        interval = []
    return interval, A, B


def binding_event_rows(A, B, direction, tol=PATH_TOL):
    """Rows tight at the boundary crossed when walking in ``direction``.

    ``direction > 0`` crosses the upper bound ``R = min_{A_i>0} B_i/A_i`` and
    returns every row attaining it; ``direction < 0`` crosses the lower bound
    ``L = max_{A_i<0} B_i/A_i``.  Ties (simultaneous events) are all returned.
    """
    A = np.asarray(A, dtype=float).reshape(-1)
    B = np.asarray(B, dtype=float).reshape(-1)
    if direction > 0:
        idx = np.flatnonzero(A > 0.0)
        if idx.size == 0:
            return idx
        vals = B[idx] / A[idx]
        bound = vals.min()
    else:
        idx = np.flatnonzero(A < 0.0)
        if idx.size == 0:
            return idx
        vals = B[idx] / A[idx]
        bound = vals.max()
    if not np.isfinite(bound):
        return np.array([], dtype=int)
    return idx[np.abs(vals - bound) <= tol * max(1.0, abs(bound))]


def advance_active_set(M, Mc, sM, rows):
    """Derive the next ``(M, sM)`` from the boundary event rows.

    Row ``r < |M|`` removes feature ``M[r]``; a row in the ``+Lambda`` block
    adds the corresponding inactive feature with sign ``+1``; a row in the
    ``-Lambda`` block adds it with sign ``-1``.  Returns ``(M, sM, ok)`` with
    ``ok=False`` on a contradictory event (feature added twice / added after
    removal), which the caller resolves by falling back to a Lasso solve.
    """
    M = np.asarray(M, dtype=int).reshape(-1)
    Mc = np.asarray(Mc, dtype=int).reshape(-1)
    sM = np.asarray(sM, dtype=float).reshape(-1)
    k = M.size
    nin = Mc.size

    M_set = set(int(v) for v in M)
    s_dict = {int(M[i]): float(sM[i]) for i in range(k)}

    for r in sorted(int(v) for v in rows):
        if r < k:
            f = int(M[r])
            M_set.discard(f)
            s_dict.pop(f, None)
        elif r < k + nin:
            f = int(Mc[r - k])
            if f in M_set:
                return None, None, False
            M_set.add(f)
            s_dict[f] = 1.0
        else:
            f = int(Mc[r - k - nin])
            if f in M_set:
                return None, None, False
            M_set.add(f)
            s_dict[f] = -1.0

    M_next = np.array(sorted(M_set), dtype=int)
    sM_next = np.array([s_dict[f] for f in M_next], dtype=float).reshape(-1, 1)
    return M_next, sM_next, True


def _reseed(X_, a, b, Lambda, z, step, max_stale, stats):
    """Fall back to restart-and-solve at ``z``; returns ``(M, Mc, sM, ok)``."""
    stale = 0
    while stale <= max_stale:
        try:
            M, Mc, sM = solve_Lasso(X_, (a + b * z).reshape(-1), Lambda)
            stats["solver_calls"] = stats.get("solver_calls", 0) + 1
            return M, Mc, sM, True
        except (RuntimeError, ValueError):
            stale += 1
            z += step
    return None, None, None, False


def walk_lasso_support_regions(X_, a, b, Lambda, z_start, sl, sr, step,
                               tol=PATH_TOL, max_stale=MAX_STALE, stats=None):
    """Yield ``(L, R, M)`` pieces tiling the Lasso support path on ``[sl, sr]``.

    ``z_start`` is an interior seed point (``z_entry`` of the merged selection
    span) where the caller's row set is fixed; the walk seeds one Lasso solve
    there and then infers every following support region from the tight KKT
    rows, re-solving only when the inference is rejected (degenerate event,
    numerical continuity failure) or the polyhedron is infeasible.

    Pieces are clipped to ``[sl, sr]``; their order is left to right.  ``stats``
    (optional dict) accumulates ``seed_solves``, ``solver_calls``,
    ``inferred`` and ``fallbacks``.
    """
    if stats is None:
        stats = {}
    X_ = np.asarray(X_, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1, 1)
    b = np.asarray(b, dtype=float).reshape(-1, 1)
    p = int(X_.shape[1])

    if not (sr > sl + tol):
        return

    z = float(z_start)
    z = min(max(z, sl), sr)

    M, Mc, sM, ok = _reseed(X_, a, b, Lambda, z, step, max_stale, stats)
    if not ok:
        return
    stats["seed_solves"] = stats.get("seed_solves", 0) + 1

    while z <= sr + tol:
        interval, A, B = lasso_conditioning_interval(
            X_, a, b, Lambda, M, Mc, sM)

        if not interval:
            stats["fallbacks"] = stats.get("fallbacks", 0) + 1
            z += step
            M, Mc, sM, ok = _reseed(X_, a, b, Lambda, z, step, max_stale, stats)
            if not ok:
                return
            continue

        left, right = interval[0]
        lc = max(left, sl)
        rc = min(right, sr)
        if lc <= rc:
            yield lc, rc, M

        if right >= sr - tol:
            return

        rows = binding_event_rows(A, B, direction=1, tol=tol)
        M2, sM2, ok = advance_active_set(M, Mc, sM, rows)

        step_ok = False
        if ok:
            Mc2 = np.setdiff1d(np.arange(p), M2)
            interval2, _, _ = lasso_conditioning_interval(
                X_, a, b, Lambda, M2, Mc2, sM2)
            if interval2:
                l2, r2 = interval2[0]
                if (np.isfinite(l2)
                        and abs(l2 - right) <= tol * max(1.0, abs(right))
                        and r2 > right + tol):
                    step_ok = True

        if step_ok:
            M, Mc, sM = M2, Mc2, sM2
            stats["inferred"] = stats.get("inferred", 0) + 1
            # advance to the shared boundary; the next iteration rebuilds the
            # (M2, sM2) interval and yields it (r2 may be +inf for the last
            # region, so do NOT jump straight to r2).
            z = right
        else:
            stats["fallbacks"] = stats.get("fallbacks", 0) + 1
            z = right + step
            M, Mc, sM, ok = _reseed(X_, a, b, Lambda, z, step, max_stale, stats)
            if not ok:
                return

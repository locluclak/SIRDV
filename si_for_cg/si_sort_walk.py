"""Inferred sort-condition walk for the lava parametric scan.

Within one OptimalTransport (OT) basis interval the LP basis ``B`` is
fixed, so every train sample's DV score is an exact quadratic in ``z``.
``score_coefficients`` obtains the per-sample quadratic table with a
single ``lu_factor`` + ``lu_solve`` (the same system
``si_overconditioning.sort_conditioning`` solves); the DV-score scaling
cancels in the pairwise score differences, so the sort-condition regions
follow from the table without any further linear algebra.

``iter_sort_regions`` walks the sort regions inside one OT interval.  The
next permutation at a region boundary ``R`` is inferred from the flip
flags (tight adjacent pairs whose score difference crosses zero upwards)
and maximal tie-group reversals (see QUADRATIC_SORT_EXAMPLE.md steps 3-4),
then verified by a continuity check: the candidate permutation's region
must contain ``R + step`` and start at ``R``.  Degenerate inferences
(e.g. equal slopes inside a tie) fall back to the original sort
computation (``select_with_known_basis`` + ``sort_conditioning``), which
keeps the walked piece sequence identical to the original scan.

Note: inference is only valid inside one OT interval -- at an OT boundary
the basis pivots and the score quadratics change, so each OT interval is
re-seeded with one real sort computation.

The cg-score analog has no such limitation: ``cg_score_coefficients``
builds the per-sample quadratic table ``CG_i(z) = (u_i + v_i z)^2 / d_i``
from the fixed ``invH`` (which depends only on ``X``), valid on the whole
real line, so ``iter_cg_sort_regions`` walks the sort regions over the
full ``[zmin, zmax]`` range with no OT seeding at all.  Both walkers
share the generic table-driven ``iter_sort_regions_table``.
"""

from __future__ import annotations

from typing import Iterator, List, Tuple

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .si_overconditioning import sort_conditioning
from .si_utils import interval_intersection
from . import _timeprof as tp

TOL = 1e-9
TOL_CONT = 1e-7    # continuity check tolerance for an inferred flip
MAX_STALE = 1000   # consecutive numerical-gap steps before giving up


def containing_piece(pieces, z, tol=TOL):
    """Return the piece of a sorted interval list that contains z."""
    for l, r in pieces:
        if l - tol <= z <= r + tol:
            return (l, r)
    return None


@tp.timed("walk: score_coefficients (per OT seed)")
def score_coefficients(adapter, basis, w_cost, v_cost, u_cost, nt):
    """Per-sample score-quadratic table for one fixed OT basis.

    Solves the same system as ``sort_conditioning`` but keeps the raw
    coefficients: returns ``s_coef`` of shape ``(nt, 3)`` where row i holds
    the coefficients of the scaled DV score
    ``s_i(z) = s_coef[i,0]*z^2 + s_coef[i,1]*z + s_coef[i,2]``.
    """
    lu, piv = lu_factor(adapter.S_[:, basis].T)
    rhs = -np.column_stack([u_cost[basis], v_cost[basis], w_cost[basis]])
    coef = lu_solve((lu, piv), rhs)[:nt]
    scale = 1.0 + 1.0 / (nt - 1)
    return scale * coef - coef.sum(axis=0, keepdims=True)


@tp.timed("walk: pair_diffs (all calls)")
def pair_diffs(s_coef, pi):
    """Quadratic coefficients of the consecutive score diffs for pi.

    Returns (dA, dB, dC) with ``diff_m(z) = dA[m]*z^2 + dB[m]*z + dC[m] =
    s_pi[m](z) - s_pi[m+1](z)``; the sort condition is ``diff_m(z) <= 0``.
    """
    dA = s_coef[pi[:-1], 0] - s_coef[pi[1:], 0]
    dB = s_coef[pi[:-1], 1] - s_coef[pi[1:], 1]
    dC = s_coef[pi[:-1], 2] - s_coef[pi[1:], 2]
    return dA, dB, dC


def quad_pieces(a, b, c, z):
    """Per-inequality containing piece of ``a*z^2 + b*z + c <= 0`` at z.

    Vectorized mirror of ``si_utils.solve_quadratic_inequality`` (same
    1e-9 coefficient cutoffs and unsorted-root convention
    ``x1 = (-b - sqrt) / 2a``, ``x2 = (-b + sqrt) / 2a``, so ``a < 0``
    gives ``x2 < x1`` and pieces ``(-inf, x2]`` and ``[x1, inf)``).
    Returns ``(left, right, ok)`` arrays; ``ok`` is False where z violates
    the inequality or the solution set is empty.
    """
    a = np.where(np.abs(a) < 1e-9, 0.0, np.asarray(a, dtype=float))
    b = np.where(np.abs(b) < 1e-9, 0.0, np.asarray(b, dtype=float))
    c = np.where(np.abs(c) < 1e-9, 0.0, np.asarray(c, dtype=float))
    left = np.full(a.size, -np.inf)
    right = np.full(a.size, np.inf)
    ok = np.ones(a.size, dtype=bool)

    lin = a == 0.0
    m = lin & (b == 0.0)
    ok[m] = c[m] <= 0.0
    m = lin & (b > 0.0)            # (-inf, -c/b]
    right[m] = -c[m] / b[m]
    m = lin & (b < 0.0)            # [-c/b, inf)
    left[m] = -c[m] / b[m]

    qi = np.flatnonzero(~lin)
    if qi.size:
        av, bv, cv = a[qi], b[qi], c[qi]
        delta = bv * bv - 4.0 * av * cv
        real = delta >= 0.0
        ok[qi[~real & (av > 0.0)]] = False   # a>0 with no real root: empty
        ri = qi[real]
        if ri.size:
            av, bv = a[ri], b[ri]
            sq = np.sqrt(delta[real])
            x1 = (-bv - sq) / (2.0 * av)
            x2 = (-bv + sq) / (2.0 * av)
            up = av > 0.0                     # a>0: piece [x1, x2]
            iu = ri[up]
            left[iu] = x1[up]
            right[iu] = x2[up]
            ok[iu[(z < x1[up]) | (z > x2[up])]] = False
            dn = ~up                          # a<0: (-inf, x2] and [x1, inf)
            idx = ri[dn]
            below = z <= x2[dn]
            above = z >= x1[dn]
            right[idx[below]] = x2[dn][below]
            left[idx[above]] = x1[dn][above]
            ok[idx[(~below) & (~above)]] = False
    return left, right, ok


@tp.timed("walk: region_containing (all calls)")
def region_containing(dA, dB, dC, z, ml, mr):
    """Containing piece of the sort region (all pair diffs <= 0) at z.

    Every per-pair piece contains z, so the intersection's containing
    piece is ``[max(left), min(right)]`` clipped to the OT interval.
    Returns None when z violates some pair (numerical gap) or the piece
    is empty.
    """
    left, right, ok = quad_pieces(dA, dB, dC, z)
    if not ok.all():
        return None
    l = max(float(left.max()), ml)
    r = min(float(right.min()), mr)
    tol_z = 1e-9 * max(1.0, abs(z))
    if l > r or not (l - tol_z <= z <= r + tol_z):
        return None
    return (l, r)


@tp.timed("walk: flip_reversal (all calls)")
def flip_reversal(s_coef, pi, R):
    """Infer the next permutation at the sort-region boundary R.

    Tight pairs (``|diff(R)| <= tol_tight``) form maximal consecutive tie
    groups; a group is reversed iff it contains a strict slope drop
    (``diff'(R) > tol_slope``), i.e. the tie actually re-orders above R.
    Tolerances are relative to the score magnitude at R; a wrong inference
    is caught by the caller's continuity check and falls back.
    """
    dA, dB, dC = pair_diffs(s_coef, pi)
    dval = dA * R * R + dB * R + dC
    dder = 2.0 * dA * R + dB
    powers = np.array([R * R, R, 1.0])
    scale = max(1.0, float(np.abs(s_coef.dot(powers)).max()))
    tight = np.abs(dval) <= 1e-8 * scale
    strict = dder > 1e-9 * scale
    new_pi = pi.copy()
    m = 0
    last = len(pi) - 1
    while m < last:
        if not tight[m]:
            m += 1
            continue
        k = m
        while k + 1 < last and tight[k + 1]:
            k += 1
        if strict[m:k + 1].any():
            seg = new_pi[m:k + 2].copy()
            new_pi[m:k + 2] = seg[::-1]
        m = k + 1
    return new_pi


@tp.timed("walk: _resort fallback (all calls)")
def _resort(adapter, wt, rt, ot, basis, z, ml, mr, config):
    """Original sort computation at z (fallback path).

    Returns (containing piece of the sort region in [ml, mr], permutation).
    """
    _, sorted_fb = adapter.select_with_known_basis(
        adapter.nt, adapter.S_, wt + rt * z + ot * (z ** 2), basis)
    selec_itv = sort_conditioning(
        z, adapter, basis, wt, rt, ot, sorted_fb, config)
    piece = containing_piece(interval_intersection(selec_itv, [(ml, mr)]), z)
    return piece, np.asarray(sorted_fb, dtype=int)


def iter_sort_regions_table(s_coef, ml, mr, config, n_keep, fallback,
                            stats=None) -> Iterator[Tuple[float, float, float, List[int]]]:
    """Generic sort-region walk on a global quadratic score table.

    ``s_coef`` has shape ``(n, 3)``; row i holds the coefficients of the
    score ``s_i(z) = s_coef[i,0]*z^2 + s_coef[i,1]*z + s_coef[i,2]``.  The
    walk covers ``[ml, mr]`` (for lava one OT basis interval, for cg-score
    the whole ``[zmin, zmax]`` range) and yields
    ``(z_entry, sl, sr, selection)`` left to right, where ``[sl, sr]`` is
    the sort-condition region containing the sample point ``z_entry`` and
    ``selection`` is the top-k index list (sorted) implied by the region's
    permutation.  Inferred transitions are verified; degenerate ones call
    ``fallback(z) -> (piece | None, pi)`` (the original sort computation)
    and fall back.

    ``stats`` (optional dict) accumulates ``regions``, ``inferred``,
    ``fallbacks`` and ``gaps`` counts.
    """
    if stats is None:
        stats = {}
    z = ml + config.step
    pi = np.argsort(s_coef.dot(np.array([z * z, z, 1.0])), kind="stable")
    stale = 0
    while z < mr:
        dA, dB, dC = pair_diffs(s_coef, pi)
        piece = region_containing(dA, dB, dC, z, ml, mr)
        if piece is None:
            stats["fallbacks"] = stats.get("fallbacks", 0) + 1
            piece, pi = fallback(z)
            if piece is None:
                stats["gaps"] = stats.get("gaps", 0) + 1
                stale += 1
                if stale > MAX_STALE:
                    return
                z += config.step
                continue
        stale = 0
        sl, sr = piece
        stats["regions"] = stats.get("regions", 0) + 1
        selection = [int(i) for i in sorted(pi[-n_keep:])]
        yield z, sl, sr, selection
        if sr + config.step >= mr:
            return

        # infer the next permutation at the boundary R
        R = sr
        pi_new = flip_reversal(s_coef, pi, R)
        z_next = R + config.step
        dA2, dB2, dC2 = pair_diffs(s_coef, pi_new)
        piece2 = region_containing(dA2, dB2, dC2, z_next, ml, mr)
        if piece2 is not None and \
                abs(piece2[0] - R) <= TOL_CONT * max(1.0, abs(R)):
            pi = pi_new
            stats["inferred"] = stats.get("inferred", 0) + 1
        else:
            stats["fallbacks"] = stats.get("fallbacks", 0) + 1
            piece_fb, pi = fallback(z_next)
            if piece_fb is None:
                stats["gaps"] = stats.get("gaps", 0) + 1
                stale = 1
                z = z_next + config.step
                continue
        z = z_next


def iter_sort_regions(adapter, wt, rt, ot, basis, ml, mr, config,
                      stats=None) -> Iterator[Tuple[float, float, float, List[int]]]:
    """Yield the sort regions inside one OT interval via inferred walking.

    Seeds one score-quadratic table for the fixed basis, then delegates to
    ``iter_sort_regions_table`` with the lava fallback (original
    ``select_with_known_basis`` + ``sort_conditioning`` computation).
    """
    nt = adapter.nt
    n_keep = max(1, int(nt * adapter.fraction))
    s_coef = score_coefficients(adapter, basis, wt, rt, ot, nt)
    fallback = lambda z: _resort(adapter, wt, rt, ot, basis, z, ml, mr, config)
    yield from iter_sort_regions_table(
        s_coef, ml, mr, config, n_keep, fallback, stats)


def cg_score_coefficients(invH: np.ndarray, a_vec: np.ndarray,
                          b_vec: np.ndarray) -> np.ndarray:
    """Per-sample cg-score quadratic table, valid on the whole real line.

    ``invH`` depends only on ``X`` and ``y = a + b*z`` is linear in ``z``,
    so ``CG_i(z) = (u_i + v_i z)^2 / d_i`` with ``u = invH @ a``,
    ``v = invH @ b``, ``d = diag(invH)`` is an exact quadratic in ``z``:
    returns ``s_coef`` of shape ``(n, 3)`` with
    ``s_i(z) = s_coef[i,0]*z^2 + s_coef[i,1]*z + s_coef[i,2]``
    (same coefficients as ``CG_overconditioning_intervals``).
    """
    d = np.diag(invH)
    u = invH.dot(a_vec)
    v = invH.dot(b_vec)
    return np.column_stack([v * v / d, 2.0 * u * v / d, u * u / d])


def _resort_cg(s_coef, z, ml, mr):
    """Original cg sort computation at z (fallback path).

    The global table IS the exact original computation (``invH`` is fixed
    and ``calc_cg_score`` evaluates the same quadratics), so the fallback
    just re-evaluates the table and takes the containing sort region.
    """
    pi = np.argsort(s_coef.dot(np.array([z * z, z, 1.0])), kind="stable")
    piece = region_containing(*pair_diffs(s_coef, pi), z, ml, mr)
    return piece, pi


def iter_cg_sort_regions(invH, a_vec, b_vec, zmin, zmax, config, n_keep,
                         stats=None) -> Iterator[Tuple[float, float, float, List[int]]]:
    """Yield the cg-score sort regions over the full ``[zmin, zmax]`` range.

    Same contract as ``iter_sort_regions`` but with no OT seeding: the
    quadratic table from ``cg_score_coefficients`` is globally valid, so
    one seed covers the whole walk.  ``n_keep`` follows the adapter rule
    (``fraction <= 1`` proportion, ``> 1`` absolute count).
    """
    s_coef = cg_score_coefficients(invH, a_vec, b_vec)
    fallback = lambda z: _resort_cg(s_coef, z, zmin, zmax)
    yield from iter_sort_regions_table(
        s_coef, zmin, zmax, config, n_keep, fallback, stats)


__all__ = [
    "TOL",
    "TOL_CONT",
    "MAX_STALE",
    "containing_piece",
    "score_coefficients",
    "pair_diffs",
    "quad_pieces",
    "region_containing",
    "flip_reversal",
    "iter_sort_regions",
    "iter_sort_regions_table",
    "cg_score_coefficients",
    "iter_cg_sort_regions",
]

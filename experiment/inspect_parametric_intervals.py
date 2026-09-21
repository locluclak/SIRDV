"""Inspect nested interval behavior of parametric SI with lava selection.

Walks the selection conditions nested the way the scan does, counting each
level separately, and compares the original sort walk against a fast
inferred sort walk (QUADRATIC_SORT_EXAMPLE.md):

  level 1  OT condition   : parametric-simplex basis intervals tiling
                            [zmin, zmax] (``trace_parametric_bases``).
  level 2  sort condition : inside each OT interval, walk z through the
                            DV-score sort-condition regions, counting the
                            piece containing each sample point.
  level 3  Lasso condition: inside each sort region, walk z through the
                            Lasso-support regions, counting pieces.

Original walk: every sort step re-runs ``select_with_known_basis`` (one
``np.linalg.solve``) + ``sort_conditioning`` (one ``lu_factor``).

Inferred walk: per OT interval ONE ``lu_factor`` + ``lu_solve`` gives the
per-sample score quadratics (fixed basis => fixed quadratics); each next
permutation is inferred at the region boundary R by the flip-flag /
tie-group-reversal rule, verified by a continuity check, with fallback to
the original re-sort on any degenerate case.  Within a sort region the
top-k selection is fixed, so the Lasso sub-walk reuses it.
"""

import contextlib
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from si_for_cg import OptimalTransport
from si_for_cg.si_adapter import SelectionAdapter, make_eta_cgFS
from si_for_cg.si_overconditioning import conditioning_Lasso, sort_conditioning
from si_for_cg.si_parametric import trace_parametric_bases
from si_for_cg.si_types import LassoInfo, SIConfig
from si_for_cg.si_utils import (
    compute_p_value,
    decompose_y,
    interval_intersection,
    interval_union,
    solve_Lasso,
)

# --- experiment sweep ---
# Each entry: (n_train, n_val, n_features, zmin, zmax, seeds).
# The number of sort pieces grows roughly quadratically with n_train
# (adjacent-pair score crossings), so larger sizes use a narrower z-grid
# to keep the baseline walk tractable; all accuracy checks are within-run
# (original vs inferred on the same grid), so grids may differ per size.
SIZES = [
    (50, 20, 5, -20.0, 20.0, [504, 505, 506]),
    (100, 50, 10, -20.0, 20.0, [504, 505]),
    (150, 60, 12, -10.0, 10.0, [504]),
]
FRACTION = 0.5
LAMBDA_ = 10.0
STEP = 0.00001
JTH = 0  # position inside the Lasso active set M
MAX_STALE = 1000  # consecutive numerical-gap steps before giving up a level
TOL = 1e-9
TOL_CONT = 1e-7   # continuity check tolerance for the inferred flip


def generate(n, m, p, true_beta_s, true_beta_t, seed=None):
    """Same data generator as examples/demo_si.py."""
    if seed is not None:
        np.random.seed(seed)
    Xs = np.random.normal(loc=0, scale=1, size=(n, p))
    Xt = np.random.normal(loc=0, scale=1, size=(m, p))
    mu_s = Xs.dot(true_beta_s)
    mu_t = Xt.dot(true_beta_t)
    Ys = mu_s + np.random.normal(loc=0, scale=1, size=(n, 1))
    Yt = mu_t + np.random.normal(loc=0, scale=1, size=(m, 1))
    return Xs, Xt, Ys, Yt, np.identity(n), np.identity(m)


def build_problem(n_train, n_val, n_features, seed, zmin, zmax):
    """Replicate SelectiveInference.fit for lava up to the SI call."""
    true_beta_t = np.zeros((n_features, 1))
    true_beta_v = np.zeros((n_features, 1))
    X_tr, X_te, y_tr, y_te, _, _ = generate(
        n_train, n_val, n_features, true_beta_t, true_beta_v, seed=seed)
    y_tr = np.asarray(y_tr).reshape(-1)
    y_te = np.asarray(y_te).reshape(-1)

    devnull = io.StringIO()
    with contextlib.redirect_stdout(devnull):
        adapter = SelectionAdapter(fraction=FRACTION)
        observed_selection = adapter.select(X_tr, y_tr, X_te, y_te)

    fs = LassoInfo(lambda_=LAMBDA_)
    fs.M, fs.Mc, fs.sM = solve_Lasso(
        X_tr[observed_selection], y_tr.reshape(-1, 1)[observed_selection], LAMBDA_)
    if len(fs.M) == 0:
        # same guard as SelectiveInference.fit (si_dv.py): under the null the
        # KKT-certified Lasso can be all-zero at lambda=10; skip such seeds.
        raise ValueError("Empty feature selection set")

    X = np.vstack([X_tr, X_te])
    y = np.concatenate([y_tr, y_te])
    eta = make_eta_cgFS(X=X, y=y, M=fs.M, j=JTH)
    Sigma = np.eye(n_train + n_val)
    decomposition = decompose_y(y, eta, Sigma)
    config = SIConfig(zmin=zmin, zmax=zmax, step=STEP, data_valuation="lava")
    return (X_tr, X_te, y_tr, y_te, adapter,
            observed_selection, fs, decomposition, config)


def containing_piece(pieces, z, tol=TOL):
    """Return the piece of a sorted interval list that contains z."""
    for l, r in pieces:
        if l - tol <= z <= r + tol:
            return (l, r)
    return None


def trace_ot_intervals(adapter, decomposition, Xtrain, Xval, config):
    """Level-1 OT basis trace (basis-path part of parametric_intervals_FS)."""
    a_vec = decomposition.a.reshape(-1, 1)
    b_vec = decomposition.b.reshape(-1, 1)
    nt, nv = Xtrain.shape[0], Xval.shape[0]
    X = np.vstack([Xtrain, Xval])
    OMEGA = OptimalTransport.constructOMEGA(nt, nv)
    c_ = 0
    for i in range(X.shape[1]):
        c_ += np.dot(OMEGA, X[:, [i]]) * np.dot(OMEGA, X[:, [i]])
    S_ = adapter.S_.copy()
    h_ = adapter.h_.copy()
    Omega_a = OMEGA.dot(a_vec)
    Omega_b = OMEGA.dot(b_vec)
    wt = c_ + Omega_a * Omega_a
    rt = 2 * Omega_a * Omega_b
    ot = Omega_b * Omega_b
    z_obs = decomposition.etaT_y
    devnull = io.StringIO()
    with contextlib.redirect_stdout(devnull):
        base_args = (S_, h_.reshape(-1),
                     wt.reshape(-1), rt.reshape(-1), ot.reshape(-1),
                     z_obs, adapter.basis)
        with ThreadPoolExecutor(max_workers=2) as executor:
            right_path = executor.submit(
                trace_parametric_bases, *base_args, config.zmax,
                direction=1, config=config).result()
            left_path = executor.submit(
                trace_parametric_bases, *base_args, config.zmin,
                direction=-1, config=config).result()
    last_left = left_path[0]
    first_right = right_path[0]
    if np.array_equal(last_left[2], first_right[2]):
        joined_path = last_left[0], first_right[1], last_left[2]
    else:
        raise RuntimeError("Left and right paths do not meet at the observed basis")
    list_da_itv = list(reversed(left_path[1:])) + [joined_path] + right_path[1:]
    return list_da_itv, S_, wt, rt, ot


def lasso_subwalk(Xtrain, a_vec, b_vec, fs, config, sl, sr, z_entry,
                  selection_fixed, reuse_selection, nt, adapter, S_,
                  wt, rt, ot, B, stat, nested_final, observed_selection,
                  counters):
    """Level 3: walk Lasso-support regions inside one sort region."""
    zl = z_entry
    lstale = 0
    while zl < sr + TOL:
        if reuse_selection:
            selection_zl = selection_fixed
        else:
            counters["solve"] += 1
            selection_zl, _ = adapter.select_with_known_basis(
                nt, S_, wt + rt * zl + ot * (zl ** 2), B)
        try:
            M, Mc, sM = solve_Lasso(
                Xtrain[selection_zl], (a_vec + b_vec * zl)[selection_zl],
                fs.lambda_)
            fs_itv = conditioning_Lasso(
                Xtrain[selection_zl], a_vec[selection_zl], b_vec[selection_zl],
                fs.lambda_, M, Mc, sM)
        except (RuntimeError, ValueError):
            stat["lasso_gaps"] += 1
            lstale += 1
            if lstale > MAX_STALE:
                break
            zl += config.step
            continue
        lstale = 0
        fs_over = interval_intersection(fs_itv, [(sl, sr)])
        fpiece = containing_piece(fs_over, zl)
        if fpiece is None:
            stat["lasso_gaps"] += 1
            lstale += 1
            if lstale > MAX_STALE:
                break
            zl += config.step
            continue
        stat["lasso_count"] += 1
        l, r = fpiece
        if np.array_equal(fs.M, M):
            stat["matched"] += 1
            if list(selection_zl) != list(observed_selection):
                stat["sel_mismatch"] += 1
            else:
                nested_final.append(fpiece)
        zl = r + config.step


def walk_original(list_da_itv, S_, wt, rt, ot, Xtrain, a_vec, b_vec,
                  adapter, observed_selection, config, fs, with_lasso):
    """Baseline nested walk: full re-sort at every sort step."""
    nt = Xtrain.shape[0]
    devnull = io.StringIO()
    counters = {"lu": 0, "solve": 0, "fallback": 0, "gaps": 0, "inferred": 0}
    ot_stats = []
    sort_pieces = []
    nested_final = []
    n_iv = len(list_da_itv)
    with contextlib.redirect_stdout(devnull):
        for idx, (ml, mr, B) in enumerate(list_da_itv):
            if (idx + 1) % 100 == 0:
                print(f"[walk] OT interval {idx + 1}/{n_iv}", file=sys.stderr)
            stat = {"ml": ml, "mr": mr, "n_basis": len(B),
                    "sort_count": 0, "lasso_count": 0,
                    "sort_gaps": 0, "lasso_gaps": 0,
                    "matched": 0, "sel_mismatch": 0}
            pieces_i = []
            z = ml + config.step
            stale = 0
            while z < mr:
                counters["solve"] += 1
                selection_z, sorted_indices_z = adapter.select_with_known_basis(
                    nt, S_, wt + rt * z + ot * (z ** 2), B)
                counters["lu"] += 1
                selec_itv = sort_conditioning(
                    z, adapter, B, wt, rt, ot, sorted_indices_z, config)
                sort_over = interval_intersection(selec_itv, [(ml, mr)])
                piece = containing_piece(sort_over, z)
                if piece is None:
                    stat["sort_gaps"] += 1
                    counters["gaps"] += 1
                    stale += 1
                    if stale > MAX_STALE:
                        break
                    z += config.step
                    continue
                stale = 0
                sl, sr = piece
                stat["sort_count"] += 1
                pieces_i.append((sl, sr))
                if with_lasso:
                    lasso_subwalk(Xtrain, a_vec, b_vec, fs, config, sl, sr, z,
                                  None, False, nt, adapter, S_, wt, rt, ot, B,
                                  stat, nested_final, observed_selection,
                                  counters)
                z = sr + config.step
            ot_stats.append(stat)
            sort_pieces.append(pieces_i)
    return ot_stats, sort_pieces, nested_final, counters


# ---------------- inferred sort walk ----------------

def pair_diffs(s_coef, pi):
    """Quadratic coefficients of consecutive score diffs for permutation pi."""
    dA = s_coef[pi[:-1], 0] - s_coef[pi[1:], 0]
    dB = s_coef[pi[:-1], 1] - s_coef[pi[1:], 1]
    dC = s_coef[pi[:-1], 2] - s_coef[pi[1:], 2]
    return dA, dB, dC


def quad_pieces(a, b, c, z):
    """Per-inequality containing piece of a*z^2 + b*z + c <= 0 at z.

    Vectorized mirror of si_utils.solve_quadratic_inequality (same 1e-9
    coefficient cutoffs and unsorted-root convention x1 = (-b-sqrt)/2a,
    x2 = (-b+sqrt)/2a, so a < 0 gives x2 < x1 and pieces (-inf, x2] ∪
    [x1, inf)).  Returns (left, right, ok) arrays.
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


def region_containing(dA, dB, dC, z, ml, mr):
    """Containing piece of the sort region (all pair diffs <= 0) at z.

    Every per-pair piece contains z, so the intersection's containing piece
    is [max(left), min(right)] clipped to the OT interval.
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


def flip_reversal(s_coef, pi, R, tol_tight, tol_slope):
    """Infer the next permutation at boundary R (steps 3-4 of the MD).

    Tight pairs (|diff(R)| <= tol_tight) form maximal consecutive tie
    groups; a group is reversed iff it contains a strict slope drop
    (diff'(R) > tol_slope), i.e. the tie actually re-orders above R.
    """
    dA, dB, dC = pair_diffs(s_coef, pi)
    dval = dA * R * R + dB * R + dC
    dder = 2.0 * dA * R + dB
    tight = np.abs(dval) <= tol_tight
    strict = dder > tol_slope
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


def walk_inferred(list_da_itv, S_, wt, rt, ot, Xtrain, a_vec, b_vec,
                  adapter, observed_selection, config, fs, with_lasso):
    """Nested walk with internally inferred sort transitions."""
    nt = Xtrain.shape[0]
    n_keep = max(1, int(nt * adapter.fraction))
    devnull = io.StringIO()
    counters = {"lu": 0, "solve": 0, "fallback": 0, "gaps": 0, "inferred": 0}
    ot_stats = []
    sort_pieces = []
    nested_final = []
    n_iv = len(list_da_itv)
    powers = np.ones(3)
    with contextlib.redirect_stdout(devnull):
        for idx, (ml, mr, B) in enumerate(list_da_itv):
            if (idx + 1) % 100 == 0:
                print(f"[walk] OT interval {idx + 1}/{n_iv}", file=sys.stderr)
            stat = {"ml": ml, "mr": mr, "n_basis": len(B),
                    "sort_count": 0, "lasso_count": 0,
                    "sort_gaps": 0, "lasso_gaps": 0,
                    "matched": 0, "sel_mismatch": 0}
            pieces_i = []

            # --- seed: ONE LU solve gives the score coefficient table
            counters["lu"] += 1
            lu, piv = lu_factor(S_[:, B].T)
            rhs = -np.column_stack([ot[B], rt[B], wt[B]])
            coef = lu_solve((lu, piv), rhs)[:nt]
            c_scale = 1.0 + 1.0 / (nt - 1)
            s_coef = c_scale * coef - coef.sum(axis=0, keepdims=True)

            z = ml + config.step
            powers[:] = (z * z, z, 1.0)
            pi = np.argsort(s_coef.dot(powers), kind="stable")
            stale = 0
            while z < mr:
                dA, dB, dC = pair_diffs(s_coef, pi)
                piece = region_containing(dA, dB, dC, z, ml, mr)
                if piece is None:
                    # fallback: original re-sort at z
                    counters["fallback"] += 1
                    counters["solve"] += 1
                    counters["lu"] += 1
                    _, sorted_fb = adapter.select_with_known_basis(
                        nt, S_, wt + rt * z + ot * (z ** 2), B)
                    selec_itv = sort_conditioning(
                        z, adapter, B, wt, rt, ot, sorted_fb, config)
                    piece = containing_piece(
                        interval_intersection(selec_itv, [(ml, mr)]), z)
                    pi = np.asarray(sorted_fb, dtype=int)
                    if piece is None:
                        stat["sort_gaps"] += 1
                        counters["gaps"] += 1
                        stale += 1
                        if stale > MAX_STALE:
                            break
                        z += config.step
                        continue
                stale = 0
                sl, sr = piece
                stat["sort_count"] += 1
                pieces_i.append((sl, sr))
                if with_lasso:
                    selection = [int(i) for i in sorted(pi[-n_keep:])]
                    lasso_subwalk(Xtrain, a_vec, b_vec, fs, config, sl, sr, z,
                                  selection, True, nt, adapter, S_, wt, rt,
                                  ot, B, stat, nested_final,
                                  observed_selection, counters)
                if sr + config.step >= mr:
                    break

                # --- infer the next permutation at the boundary R
                R = sr
                powers[:] = (R * R, R, 1.0)
                scale = max(1.0, float(np.abs(s_coef.dot(powers)).max()))
                pi_new = flip_reversal(s_coef, pi, R, 1e-8 * scale,
                                       1e-9 * scale)
                z_next = R + config.step
                dA2, dB2, dC2 = pair_diffs(s_coef, pi_new)
                piece2 = region_containing(dA2, dB2, dC2, z_next, ml, mr)
                if piece2 is not None and \
                        abs(piece2[0] - R) <= TOL_CONT * max(1.0, abs(R)):
                    pi = pi_new
                    counters["inferred"] += 1
                else:
                    # fallback: original re-sort at z_next
                    counters["fallback"] += 1
                    counters["solve"] += 1
                    counters["lu"] += 1
                    _, sorted_fb = adapter.select_with_known_basis(
                        nt, S_, wt + rt * z_next + ot * (z_next ** 2), B)
                    selec_itv = sort_conditioning(
                        z_next, adapter, B, wt, rt, ot, sorted_fb, config)
                    pi = np.asarray(sorted_fb, dtype=int)
                    piece_fb = containing_piece(
                        interval_intersection(selec_itv, [(ml, mr)]), z_next)
                    if piece_fb is None:
                        stat["sort_gaps"] += 1
                        counters["gaps"] += 1
                        stale = 1
                        z = z_next + config.step
                        continue
                z = z_next
            ot_stats.append(stat)
            sort_pieces.append(pieces_i)
    return ot_stats, sort_pieces, nested_final, counters


def compare_sort_pieces(pieces_a, pieces_b):
    """Compare per-OT-interval sort piece sequences."""
    mismatched = []
    for i, (pa, pb) in enumerate(zip(pieces_a, pieces_b)):
        if len(pa) != len(pb):
            mismatched.append((i, f"len {len(pa)} vs {len(pb)}"))
            continue
        if pa:
            da = np.array(pa)
            db = np.array(pb)
            if not np.allclose(da, db, rtol=1e-9, atol=1e-9):
                bad = np.flatnonzero(np.abs(da - db).max(axis=1) > 1e-9)
                mismatched.append((i, f"piece {bad.tolist()}"))
    return mismatched


def region_key_diff(ra, rb):
    sa = {(round(float(l), 6), round(float(r), 6)) for l, r in ra}
    sb = {(round(float(l), 6), round(float(r), 6)) for l, r in rb}
    return len(sa - sb), len(sb - sa)


def run_comparison(n_train, n_val, n_features, seed, zmin, zmax):
    """Run original vs inferred walks for one problem instance.

    Returns a metrics dict with timings, call counts and accuracy checks.
    """
    (X_tr, X_te, y_tr, y_te, adapter,
     observed_selection, fs, decomposition, config) = build_problem(
        n_train, n_val, n_features, seed, zmin, zmax)

    devnull = io.StringIO()
    with contextlib.redirect_stdout(devnull):
        list_da_itv, S_, wt, rt, ot = trace_ot_intervals(
            adapter, decomposition, X_tr, X_te, config)
    a_vec = decomposition.a.reshape(-1, 1)
    b_vec = decomposition.b.reshape(-1, 1)
    z_obs = decomposition.etaT_y
    K = len(list_da_itv)

    def pval(region):
        if any(l <= z_obs <= r for l, r in region):
            return compute_p_value(region, z_obs,
                                   decomposition.etaT_Sigma_eta,
                                   type="two-sided")
        return float("nan")

    # ---------- sort-only ----------
    st = time.time()
    o_stats, o_pieces, _, o_cnt = walk_original(
        list_da_itv, S_, wt, rt, ot, X_tr, a_vec, b_vec, adapter,
        observed_selection, config, fs, with_lasso=False)
    t_sort_orig = time.time() - st
    st = time.time()
    i_stats, i_pieces, _, i_cnt = walk_inferred(
        list_da_itv, S_, wt, rt, ot, X_tr, a_vec, b_vec, adapter,
        observed_selection, config, fs, with_lasso=False)
    t_sort_inf = time.time() - st

    # ---------- full nested ----------
    st = time.time()
    o_stats_f, o_pieces_f, o_final, o_cnt_f = walk_original(
        list_da_itv, S_, wt, rt, ot, X_tr, a_vec, b_vec, adapter,
        observed_selection, config, fs, with_lasso=True)
    t_full_orig = time.time() - st
    st = time.time()
    i_stats_f, i_pieces_f, i_final, i_cnt_f = walk_inferred(
        list_da_itv, S_, wt, rt, ot, X_tr, a_vec, b_vec, adapter,
        observed_selection, config, fs, with_lasso=True)
    t_full_inf = time.time() - st

    # ---------- accuracy checks ----------
    mism = compare_sort_pieces(o_pieces, i_pieces)
    max_diff = 0.0
    for pa, pb in zip(o_pieces, i_pieces):
        if len(pa) == len(pb) and pa:
            da = np.array(pa)
            db = np.array(pb)
            max_diff = max(max_diff, float(np.abs(da - db).max()))
    o_tot_s = sum(s["sort_count"] for s in o_stats_f)
    i_tot_s = sum(s["sort_count"] for s in i_stats_f)
    o_tot_l = sum(s["lasso_count"] for s in o_stats_f)
    i_tot_l = sum(s["lasso_count"] for s in i_stats_f)
    o_tot_m = sum(s["matched"] for s in o_stats_f)
    i_tot_m = sum(s["matched"] for s in i_stats_f)
    counts_equal = (o_tot_s == i_tot_s and o_tot_l == i_tot_l
                    and o_tot_m == i_tot_m)
    only_o, only_i = region_key_diff(o_final, i_final)
    p_o, p_i = pval(o_final), pval(i_final)

    metrics = {
        "n_train": n_train, "n_val": n_val, "p": n_features, "seed": seed,
        "ot": K,
        "sort_pieces": o_tot_s, "lasso_pieces": o_tot_l,
        "t_sort_orig": t_sort_orig, "t_sort_inf": t_sort_inf,
        "t_full_orig": t_full_orig, "t_full_inf": t_full_inf,
        "speedup_sort": t_sort_orig / t_sort_inf,
        "speedup_full": t_full_orig / t_full_inf,
        "lu_orig": o_cnt["lu"], "lu_inf": i_cnt["lu"],
        "solve_orig": o_cnt["solve"] + o_cnt_f["solve"],
        "solve_inf": i_cnt["solve"] + i_cnt_f["solve"],
        "fallbacks": i_cnt["fallback"] + i_cnt_f["fallback"],
        "inferred": i_cnt["inferred"],
        "pieces_match": not mism and o_tot_s == i_tot_s,
        "max_diff": max_diff,
        "counts_equal": counts_equal,
        "region_diff": (only_o, only_i),
        "p_orig": p_o, "p_inf": p_i,
        "matched": o_tot_m, "sel_mismatch": sum(
            s["sel_mismatch"] for s in o_stats_f),
    }
    return metrics


def print_run(metrics):
    print(f"OT intervals={metrics['ot']}, sort pieces={metrics['sort_pieces']}"
          f", lasso pieces={metrics['lasso_pieces']}, "
          f"matched={metrics['matched']}, sel_mismatch={metrics['sel_mismatch']}")
    print(f"  sort-only : original {metrics['t_sort_orig']:8.2f}s | "
          f"inferred {metrics['t_sort_inf']:8.2f}s | "
          f"speedup {metrics['speedup_sort']:5.1f}x   "
          f"(lu {metrics['lu_orig']} vs {metrics['lu_inf']}, "
          f"solve {metrics['solve_orig']} vs {metrics['solve_inf']}, "
          f"fallbacks={metrics['fallbacks']})")
    print(f"  full walk : original {metrics['t_full_orig']:8.2f}s | "
          f"inferred {metrics['t_full_inf']:8.2f}s | "
          f"speedup {metrics['speedup_full']:5.2f}x")
    print(f"  accuracy  : pieces match {metrics['pieces_match']} "
          f"(max endpoint diff {metrics['max_diff']:.2e}), "
          f"counts equal {metrics['counts_equal']}, "
          f"region diff {metrics['region_diff'][0]}/"
          f"{metrics['region_diff'][1]}, "
          f"p {metrics['p_orig']:.6f} vs {metrics['p_inf']:.6f}")


def main():
    t0 = time.time()
    results = []
    for (n_train, n_val, n_features, zmin, zmax, seeds) in SIZES:
        for seed in seeds:
            print()
            print("=" * 78, flush=True)
            print(f"RUN n_train={n_train} n_val={n_val} d={n_features} "
                  f"seed={seed}  (fraction={FRACTION}, lambda={LAMBDA_}, "
                  f"z in [{zmin}, {zmax}], step={STEP})", flush=True)
            print("=" * 78, flush=True)
            try:
                metrics = run_comparison(n_train, n_val, n_features, seed,
                                         zmin, zmax)
                print_run(metrics)
                results.append(metrics)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"RUN FAILED: {type(exc).__name__}: {exc}", flush=True)
                results.append({"n_train": n_train, "n_val": n_val,
                                "p": n_features, "seed": seed,
                                "failed": str(exc)})

    print()
    print("=" * 78)
    print("SUMMARY (original vs inferred sort walk)")
    print("=" * 78)
    print(f"{'size':>14} {'seed':>5} {'sort pcs':>9} {'lasso pcs':>10} "
          f"{'t_orig(s)':>10} {'t_inf(s)':>9} {'full x':>7} {'sort x':>7} "
          f"{'fb':>4} {'match':>6}")
    for m in results:
        size_str = f"({m['n_train']},{m['n_val']},{m['p']})"
        if "failed" in m:
            print(f"{size_str:>14} {m['seed']:>5}  FAILED: {m['failed'][:60]}")
            continue
        print(f"{size_str:>14} "
              f"{m['seed']:>5} {m['sort_pieces']:>9} {m['lasso_pieces']:>10} "
              f"{m['t_full_orig']:>10.2f} {m['t_full_inf']:>9.2f} "
              f"{m['speedup_full']:>7.2f} {m['speedup_sort']:>7.2f} "
              f"{m['fallbacks']:>4} {str(m['pieces_match']):>6}")
    ok = [m for m in results if "failed" not in m]
    print()
    for (n_train, n_val, n_features, _, _, _) in SIZES:
        grp = [m for m in ok if m["n_train"] == n_train]
        if not grp:
            continue
        ms = sum(m["speedup_sort"] for m in grp) / len(grp)
        mf = sum(m["speedup_full"] for m in grp) / len(grp)
        all_match = all(m["pieces_match"] and m["counts_equal"]
                        and m["region_diff"] == (0, 0) for m in grp)
        print(f"size ({n_train},{n_val},{n_features}): "
              f"mean sort speedup {ms:.1f}x, mean full speedup {mf:.2f}x, "
              f"all runs accurate: {all_match}")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()


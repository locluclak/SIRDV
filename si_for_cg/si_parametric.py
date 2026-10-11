"""Parametric selective-inference scan."""

from __future__ import annotations

from typing import Any

import numpy as np
from dataclasses import replace
from .si_adapter import SelectionAdapter, CGSelectionAdapter

from .si_types import SIConfig, SIDecomposition, SelectionAdapter, LassoInfo
from .si_overconditioning import sort_conditioning, overconditioning_intervals, CG_overconditioning_intervals, conditioning_Lasso
from .si_utils import (interval_union, normalize_selection, solveinterval,
                       solve_quadratic_roots, interval_intersection, solve_Lasso)
from .si_sort_walk import (iter_sort_regions, iter_cg_sort_regions,
                           containing_piece, MAX_STALE, TOL,
                           coalesce_selection_regions)
from .si_lasso_path import walk_lasso_support_regions, MAX_STALE as PATH_MAX_STALE
from . import OptimalTransport
from scipy.optimize import linprog
from .si_adapter import SelectionAdapter
from concurrent.futures import ThreadPoolExecutor

import time
from . import _timeprof as tp

def parametric_intervals_FS(
    Xtrain: np.ndarray,
    Xval: np.ndarray,
    decomposition: SIDecomposition,
    adapter: SelectionAdapter,
    observed_selection: Any,
    config: SIConfig,
    fs: LassoInfo,
) -> list[tuple[float, float]]:
    """the selection region by scanning ``z`` on a fixed grid."""
    if config.step <= 0:
        raise ValueError("step must be positive")
    if config.zmin >= config.zmax:
        raise ValueError("zmin must be smaller than zmax")

    print('init para')
    st = time.time()
    a_vec = decomposition.a.reshape(-1,1)
    b_vec = decomposition.b.reshape(-1,1)

    final_interval = []
    nt, nv = Xtrain.shape[0], Xval.shape[0]
    X = np.vstack([Xtrain, Xval])

    # --- PRE-COMPUTE DA BASIS INTERVALS ---
    OMEGA = OptimalTransport.constructOMEGA(nt, nv)
    c_ = 0
    for i in range(X.shape[1]):
        c_ += np.dot(OMEGA, X[:, [i]]) * np.dot(OMEGA,X[:, [i]])

    S_ = adapter.S_.copy()
    h_ = adapter.h_.copy()
    Omega_a = OMEGA.dot(a_vec)
    Omega_b = OMEGA.dot(b_vec)
    wt = c_ + Omega_a * Omega_a
    rt = 2 * Omega_a * Omega_b
    ot = Omega_b * Omega_b

    z_obs = decomposition.etaT_y

    print('end', time.time() -st)

    print('OT')
    st = time.time()
    base_args = (S_, h_.reshape(-1), 
        wt.reshape(-1), rt.reshape(-1), ot.reshape(-1), 
        z_obs, adapter.basis
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_right = executor.submit(trace_parametric_bases, *base_args, config.zmax, direction=1, config=config)
        future_left = executor.submit(trace_parametric_bases, *base_args, config.zmin, direction=-1, config=config)

        right_path = future_right.result()
        left_path = future_left.result()

    last_left = left_path[0]
    first_right = right_path[0]

    if np.array_equal(last_left[2], first_right[2]):
        joined_path = last_left[0], first_right[1], last_left[2]
    else:
        raise RuntimeError("Left and right paths do not meet at the observed basis")
    list_da_itv = list(reversed(left_path[1:])) + [joined_path] + right_path[1:]

    print('end', time.time()-st)

    print('other OC')
    st = time.time()
    zold = -50
    zlog = -21
    for ml, mr, B in list_da_itv:
        z = ml + config.step
        while z < mr:
            if zlog != int(z):
                zlog= int(z)
                # print(zlog)
            cost_z = wt + rt*z + ot*(z**2)
            selection_z, sorted_indices_z = adapter.select_with_known_basis(nt, S_, cost_z, B)
            M, Mc, sM = solve_Lasso(Xtrain[selection_z], (a_vec + b_vec*z)[selection_z], fs.lambda_)

            selec_itv = sort_conditioning(z,adapter, B,wt, rt, ot, sorted_indices_z,config)
            fs_itv = conditioning_Lasso(Xtrain[selection_z], a_vec[selection_z], b_vec[selection_z], fs.lambda_, M, Mc, sM)

            over = interval_intersection(
                interval_intersection(selec_itv, fs_itv),
                [(ml,mr)])
            if not over:
                print((ml, mr))
                print(selec_itv)
                print(fs_itv)
                raise RuntimeError("Selection interval does not intersect its DA region")
            z = over[-1][1] + config.step


            if z == zold:
                print((ml, mr))
                print(selec_itv)
                print(fs_itv)
                print("similar z again z=", z)
                raise RuntimeError("z did not advance;")
            zold = z
            if np.array_equal(fs.M,M):
                final_interval.extend(over)

    final_interval = interval_union(final_interval)

    print('end', time.time()-st)


    if not any(left <= z_obs <= right for left, right in final_interval):
        raise RuntimeError("Parametric selection region does not contain etaT_y")
    return final_interval


def parametric_intervals_FS_nested(
    Xtrain: np.ndarray,
    Xval: np.ndarray,
    decomposition: SIDecomposition,
    adapter: SelectionAdapter,
    config: SIConfig,
    fs: LassoInfo,
) -> list[tuple[float, float]]:
    """Systematic nested parametric scan for lava with an inferred sort walk.

    Walks the same traced OT basis intervals as ``parametric_intervals_FS``
    (the observed LP basis is kept; the OT path is never re-solved at other
    ``z``), but enumerates the conditioning event nested instead of jumping
    to the last joint-intersection piece:

        OT basis interval -> sort-condition regions -> Lasso-support regions.

    The returned region is the union of pieces where the z-local Lasso
    support equals ``fs.M`` -- the minimal conditioning event
    ``{z : M(z) = M_obs}``.  The test statistic is built from the feature
    set only (``make_eta_cgFS``/``make_etaFS`` use ``X[:, M]``), so the
    sample top-k is not part of the event; within each sort region the
    top-k is constant and only fixes the rows the Lasso is fit on.  The
    stock scan consumes the last intersection piece per region, so it can
    MISS event pieces; this walk enumerates them all.

    Sort transitions are inferred from the fixed-basis score quadratics
    (``si_sort_walk.iter_sort_regions``) with fallback to the original sort
    computation on degenerate ties, and the Lasso sub-walk reuses the sort
    region's fixed top-k selection.
    """
    if config.step <= 0:
        raise ValueError("step must be positive")
    if config.zmin >= config.zmax:
        raise ValueError("zmin must be smaller than zmax")

    _t_pre = tp.start("nested: precompute (OMEGA, quadratics)")
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
    tp.stop(_t_pre)

    _t_trace = tp.start("nested: trace_parametric_bases (L+R)")
    base_args = (S_, h_.reshape(-1),
                    wt.reshape(-1), rt.reshape(-1), ot.reshape(-1),
                    z_obs, adapter.basis)

    # The two directions are independent traces from the observed basis; run
    # them on a 2-worker thread pool.  The dominant work inside
    # _basis_optimal_region (np.linalg.inv + pi_*.dot(H[:, non_basis])) is
    # numpy/BLAS and releases the GIL, so the split is near-linear
    # (~1.4-2x wall on the trace).  Do NOT use a process pool here: H would
    # be pickled per task while each worker's BLAS oversubscribes cores.
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_right = executor.submit(trace_parametric_bases, *base_args,
                                       config.zmax, direction=1, config=config)
        future_left = executor.submit(trace_parametric_bases, *base_args,
                                      config.zmin, direction=-1, config=config)
        right_path = future_right.result()
        left_path = future_left.result()
    last_left = left_path[0]
    first_right = right_path[0]
    if np.array_equal(last_left[2], first_right[2]):
        joined_path = last_left[0], first_right[1], last_left[2]
    else:
        raise RuntimeError("Left and right paths do not meet at the observed basis")
    list_da_itv = list(reversed(left_path[1:])) + [joined_path] + right_path[1:]
    tp.stop(_t_trace)
    tp.tick("nested: OT intervals", len(list_da_itv))

    final_interval = []
    count_itv = 0
    walk_stats = {}
    _t_walk = tp.start("nested: sort walk + lasso loop (total)")
    regions = []
    for ml, mr, B in list_da_itv:
        print(ml, end = ' ')
        regions.extend(iter_sort_regions(
            adapter, wt, rt, ot, B, ml, mr, config, stats=walk_stats))
    tp.tick("nested: sort regions (raw)", len(regions))

    for z_entry, sl, sr, selection in coalesce_selection_regions(regions):
        tp.tick("nested: merged selection spans")
        if getattr(config, "lasso_path", False):
            # Event-driven Lasso support walk: one Lasso solve per span, then
            # infer each following support region from the tight KKT rows.
            path_stats = {}
            for L, R, Mpiece in walk_lasso_support_regions(
                    Xtrain[selection], a_vec[selection], b_vec[selection],
                    fs.lambda_, z_entry, sl, sr, config.step,
                    max_stale=PATH_MAX_STALE, stats=path_stats):
                if np.array_equal(fs.M, Mpiece):
                    final_interval.append((L, R))
            for k, v in path_stats.items():
                tp.tick(f"lasso path: {k}", v)
            continue
        # level 3: Lasso-support regions inside the merged selection span
        zl = z_entry
        stale = 0
        while zl < sr + TOL:
            try:
                count_itv += 1
                tp.tick("nested: lasso iterations")
                M, Mc, sM = solve_Lasso(
                    Xtrain[selection], (a_vec + b_vec * zl)[selection],
                    fs.lambda_)
                fs_itv = conditioning_Lasso(
                    Xtrain[selection], a_vec[selection], b_vec[selection],
                    fs.lambda_, M, Mc, sM)
            except (RuntimeError, ValueError):
                stale += 1
                if stale > MAX_STALE:
                    break
                zl += config.step
                continue
            stale = 0
            _t_io = tp.start("nested: interval ops (inner)")
            fs_over = interval_intersection(fs_itv, [(sl, sr)])
            fpiece = containing_piece(fs_over, zl)
            tp.stop(_t_io)
            if fpiece is None:
                stale += 1
                if stale > MAX_STALE:
                    break
                zl += config.step
                continue
            print('Interval M', M)
            if np.array_equal(fs.M, M):
                final_interval.append(fpiece)
            zl = fpiece[1] + config.step
    tp.stop(_t_walk)
    for k, v in walk_stats.items():
        tp.tick(f"walk: {k}", v)
    _t_union = tp.start("nested: final interval_union")
    final_interval = interval_union(final_interval)
    tp.stop(_t_union)

    if not any(left <= z_obs <= right for left, right in final_interval):
        raise RuntimeError("Parametric selection region does not contain etaT_y")
    return final_interval

def cg_parametric_intervals_FS(
    X: np.array,
    decomposition: SIDecomposition,
    adapter: SelectionAdapter,
    observed_selection: Any,
    config: SIConfig,
    fs: LassoInfo,
) -> list[tuple[float, float]]:
    if config.step <= 0:
        raise ValueError("step must be positive")
    if config.zmin >= config.zmax:
        raise ValueError("zmin must be smaller than zmax")
    
    a_vec = decomposition.a.reshape(-1)
    b_vec = decomposition.b.reshape(-1)

    final_interval = []

    z = config.zmin

    while z < config.zmax:
        print(z)
        y_z = (a_vec + b_vec * z)
        selection_z = adapter.select(X, y_z)
        X_pruned_z = X[selection_z]
        M, Mc, sM = solve_Lasso(X_pruned_z, (a_vec + b_vec*z)[selection_z], fs.lambda_)
        
        interval_z = CG_overconditioning_intervals(X, decomposition, adapter, config)
        fs_itv = conditioning_Lasso(X_pruned_z, a_vec.reshape(-1,1)[selection_z], b_vec.reshape(-1,1)[selection_z], fs.lambda_, M, Mc, sM)

        over = interval_intersection(interval_z, fs_itv)
        matched = False
        detected = False
        for l, r in over:
            if l <= z <= r:
                if np.array_equal(fs.M, M):
                    detected = True
                    final_interval.append((l, r))
                z = r + config.step
                matched = True
                break

        if not matched:
            z += config.step
    return final_interval

def cg_parametric_intervals_FS_nested(
    X: np.array,
    decomposition: SIDecomposition,
    adapter: SelectionAdapter,
    config: SIConfig,
    fs: LassoInfo,
) -> list[tuple[float, float]]:
    """Nested parametric scan for cg-score with an inferred sort walk.

    Unlike lava, the cg-score quadratics ``CG_i(z) = (u_i + v_i z)^2 / d_i``
    hold on the whole real line (``invH`` depends only on ``X``, and the
    cg path never re-solves an OT basis), so there is no OT interval to
    respect and the walk goes directly over ``[zmin, zmax]``:

        sort-condition regions -> Lasso-support regions.

    The returned region is the union of pieces where the z-local Lasso
    support equals ``fs.M`` -- the minimal conditioning event
    ``{z : M(z) = M_obs}``.  The test statistic is built from the feature
    set only (``make_eta_cgFS`` uses ``X[:, M]``), so the sample top-k is
    not part of the event; within each sort region the top-k is constant
    and only fixes the rows the Lasso is fit on.  The stock
    ``cg_parametric_intervals_FS`` recomputes the OC interval from a fresh
    select at every entry ``z`` and can step over event pieces; this walk
    enumerates them all systematically.
    """
    if config.step <= 0:
        raise ValueError("step must be positive")
    if config.zmin >= config.zmax:
        raise ValueError("zmin must be smaller than zmax")

    a_vec = decomposition.a.reshape(-1, 1)
    b_vec = decomposition.b.reshape(-1, 1)

    invH = adapter.invH
    if invH is None:
        raise ValueError("adapter.invH is None; call adapter.select() first")

    n = X.shape[0]
    if adapter.fraction <= 1:
        n_keep = max(1, int(n * adapter.fraction))
    else:
        n_keep = int(adapter.fraction)

    final_interval = []
    stats = {}
    regions = list(iter_cg_sort_regions(
        invH, decomposition.a.reshape(-1), decomposition.b.reshape(-1),
        config.zmin, config.zmax, config, n_keep, stats))
    tp.tick("nested: sort regions (raw)", len(regions))

    for z_entry, sl, sr, selection in coalesce_selection_regions(regions):
        tp.tick("nested: merged selection spans")
        if getattr(config, "lasso_path", False):
            # Event-driven Lasso support walk: one Lasso solve per span, then
            # infer each following support region from the tight KKT rows.
            path_stats = {}
            for L, R, Mpiece in walk_lasso_support_regions(
                    X[selection], a_vec[selection], b_vec[selection],
                    fs.lambda_, z_entry, sl, sr, config.step,
                    max_stale=PATH_MAX_STALE, stats=path_stats):
                # print("L", L)
                if np.array_equal(fs.M, Mpiece):
                    final_interval.append((L, R))
            for k, v in path_stats.items():
                tp.tick(f"lasso path: {k}", v)
            continue
        # level 2: Lasso-support regions inside the merged selection span
        zl = z_entry
        stale = 0
        while zl < sr + TOL:
            try:
                M, Mc, sM = solve_Lasso(
                    X[selection], (a_vec + b_vec * zl)[selection],
                    fs.lambda_)
                fs_itv = conditioning_Lasso(
                    X[selection], a_vec[selection], b_vec[selection],
                    fs.lambda_, M, Mc, sM)
            except (RuntimeError, ValueError):
                print('expect')
                stale += 1
                if stale > MAX_STALE:
                    break
                zl += config.step
                continue
            stale = 0
            fs_over = interval_intersection(fs_itv, [(sl, sr)])
            fpiece = containing_piece(fs_over, zl)
            if fpiece is None:
                stale += 1
                if stale > MAX_STALE:
                    break
                zl += config.step
                continue
            # print('Interval M', M)
            if np.array_equal(fs.M, M):
                # print('check M')
                final_interval.append(fpiece)
            zl = fpiece[1] + config.step

    final_interval = interval_union(final_interval)

    z_obs = decomposition.etaT_y
    # print('final', final_interval)
    # print(z_obs)
    if not any(left <= z_obs <= right for left, right in final_interval):
        raise RuntimeError("Parametric selection region does not contain etaT_y")
    return final_interval

def cg_parametric_intervals(
    X: np.array,
    decomposition: SIDecomposition,
    adapter: SelectionAdapter,
    observed_selection: Any,
    config: SIConfig,
) -> list[tuple[float, float]]:
    if config.step <= 0:
        raise ValueError("step must be positive")
    if config.zmin >= config.zmax:
        raise ValueError("zmin must be smaller than zmax")
    
    a_vec = decomposition.a.reshape(-1)
    b_vec = decomposition.b.reshape(-1)

    final_interval = []

    z = config.zmin

    logs = []
    print('observed_selection', observed_selection)
    while z < config.zmax:
        y_z = (a_vec + b_vec * z)
        selection_z = adapter.select(X, y_z)
        interval_z = CG_overconditioning_intervals(X, decomposition, adapter, config)
        matched = False
        detected = False
        for l, r in interval_z:
            if l <= z <= r:
                if selection_z == observed_selection:
                    detected = True
                    final_interval.append((l, r))
                z = r + config.step
                matched = True
                break
        logs.append((z, selection_z, detected))
        

        if not matched:
            # z fell in a gap between intervals (boundary/float issue) —
            # don't get stuck, just step forward
            z += config.step
    # with open("logs.txt", "w") as f:
    #     for z_val, sel, detect in logs:
    #         f.write(f"z: {z_val:.2f}, selection: {sel} -- {detect}\n")
    return final_interval

def ocwithz_intervals(
    Xtrain: np.ndarray,
    Xval: np.ndarray,
    decomposition: SIDecomposition,
    adapter: SelectionAdapter,
    observed_selection: Any,
    config: SIConfig,
    z: float = 0,
) -> list[tuple[float, float]]:
    """Compute the OC interval after re-solving selection at ``z``."""
    if config.step <= 0:
        raise ValueError("step must be positive")
    if config.zmin >= config.zmax:
        raise ValueError("zmin must be smaller than zmax")

    # observed_key = normalize_selection(observed_selection)
    # intervals: list[tuple[float, float]] = []

    a_vec = decomposition.a.reshape(-1,1)
    b_vec = decomposition.b.reshape(-1,1)
    nt, nv = Xtrain.shape[0], Xval.shape[0]
    y_z = a_vec + b_vec*z
    adapter_z = SelectionAdapter(adapter.fraction)
    adapter_z.select(Xtrain, y_z[:nt].reshape(-1), Xval, y_z[nt:nt+nv].reshape(-1),
                     basis_hint=adapter.basis)
    decomposition_z = replace(decomposition,etaT_y=z)
    itv = overconditioning_intervals(Xtrain, Xval, decomposition_z, adapter_z, config, z=z)

    return itv
def traditional_parametric_intervals(
    Xtrain: np.ndarray,
    Xval: np.ndarray,
    decomposition: SIDecomposition,
    adapter: SelectionAdapter,
    observed_selection: Any,
    config: SIConfig,
    direction: 1,
) -> list[tuple[float, float]]:
    """Approximate the selection region by scanning ``z`` on a fixed grid."""
    if config.step <= 0:
        raise ValueError("step must be positive")
    if config.zmin >= config.zmax:
        raise ValueError("zmin must be smaller than zmax")
    if direction not in [-1, 1]:
        raise ValueError("direction must be -1 or 1")
    # observed_key = normalize_selection(observed_selection)
    # intervals: list[tuple[float, float]] = []

    a_vec = decomposition.a.reshape(-1,1)
    b_vec = decomposition.b.reshape(-1,1)

    final_interval = []
    nt, nv = Xtrain.shape[0], Xval.shape[0]
    # X = np.vstack([Xtrain, Xval])
    etaT_y = decomposition.etaT_y
    z = etaT_y

    # print('finalinterval at start', final_interval)
    # z = config.zmin
    DA = []
    while np.abs(z) < np.abs(config.zmax):
        # print('z',z)
        y_z = a_vec + b_vec*z
        ytrain_z = y_z[:nt].copy().reshape(-1)
        yval_z = y_z[nt:nt+nv].copy().reshape(-1)

        adapter_z = SelectionAdapter(adapter.fraction)
        observed_selection_z = adapter_z.select(Xtrain, ytrain_z, Xval, yval_z,
                                                # basis_hint=adapter.basis
                                                )
        decomposition_z = replace(decomposition,etaT_y=z) 
        # decomposition_z.etaT_y = z
        itv, cost = overconditioning_intervals(Xtrain, Xval, decomposition_z,adapter_z,config, onlyDA=True,z=z)
        # print(itv)
        DA.append(itv)
        position = 1
        if direction ==-1:
            position = 0
        z = itv[position]+direction*config.step


    wt,rt,ot = cost
    for ml, mr, B in DA:
        print(f'[{np.round(ml,4)}, {np.round(mr,4)}]',end='')
        # print('basis', B)
        z = ml + config.step
        while z < mr:
            # adapter_z = SelectionAdapter(adapter.fraction)
            # print('z',z)
            cost_z = wt + rt*z + ot*z**2
            S_ = adapter.S_.copy()
            selection_z, sorted_indices_z = adapter.select_with_known_basis(nt, S_, cost_z, B)
            selec_itv = sort_conditioning(z,adapter, B,wt, rt, ot, sorted_indices_z,config)
            # if 0 <= z < 2:
            # #     T = np.zeros(nt * nv)
                
            # #     x_B = np.linalg.inv(adapter.S_[:, B]).dot(adapter.h_)
            # #     T[B] = x_B.flatten()
            #     print('    org_cost', np.sum(cost_z))
            #     # print('    old_cost', cost_z.reshape(-1).dot(T))
            #     print('    selection', selection_z)
            #     print('    old selecitv', selec_itv)
            # print(' selec',selec_itv)
            selec_itv = interval_intersection(selec_itv, [(ml,mr)])
            for i in selec_itv:
                l,r = i
                if l <= z <=r:
                    z = r +config.step
                    break
            # print(selection_z)
            if observed_selection == selection_z:
                # print('correct')
                final_interval.append((l,r))
            # print(itv)
    # print('final interval after', final_interval)
    final_interval = interval_union(final_interval)
    # if not any(left <= etaT_y <= right for left, right in final_interval):
    #     raise RuntimeError("Parametric selection region does not contain etaT_y")
    return final_interval

@tp.timed("trace: _basis_optimal_region (all calls)")
def _basis_optimal_region(
    H: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    basis: np.ndarray,
    z: float,
    direction: int,
) -> tuple[float, float, list[int], np.ndarray]:
    """
    Return the optimality region of `basis` containing z and the
    entering-variable candidates at the boundary in `direction`.

    Reduced costs must satisfy:
        RC_i(z) >= 0
    for every non-basic variable i.
    """
    basis = np.asarray(basis, dtype=int)
    non_basis = np.setdiff1d(np.arange(H.shape[1]), basis)

    B = H[:, basis]

    # ``pi_c = c[basis] @ B^{-1}`` is the solution of ``B^T pi_c = c[basis]``;
    # solving avoids forming ``B^{-1}`` (the hottest BLAS call in the trace).
    pi = np.linalg.solve(B.T, np.column_stack((u[basis], v[basis], w[basis])))

    # Gather the non-basis columns once and reuse for all three reduced costs.
    H_nb = H[:, non_basis]
    rc_u = u[non_basis] - pi[:, 0].dot(H_nb)
    rc_v = v[non_basis] - pi[:, 1].dot(H_nb)
    rc_w = w[non_basis] - pi[:, 2].dot(H_nb)

    # solveinterval() returns the common region where
    # -RC_i(z) <= 0  <=>  RC_i(z) >= 0.
    regions = solveinterval(-rc_w, -rc_v, -rc_u, z)

    tol = 1e-8

    containing = [
        (left, right)
        for left, right in regions
        if left - tol <= z <= right + tol
    ]

    if not containing:
        raise RuntimeError(
            f"Current basis is not optimal at z={z}"
        )

    left, right = containing[0]

    # The boundary that will be crossed.
    boundary = right if direction > 0 else left

    # Find non-basic variables whose reduced cost is zero
    # at the boundary and becomes negative after crossing it.
    entering_candidates = []

    for j, var in enumerate(non_basis):
        a = rc_w[j]
        b = rc_v[j]
        c = rc_u[j]

        rc_boundary = (a * boundary**2 + b * boundary + c)

        # Not an active constraint at this boundary.
        if abs(rc_boundary) > tol:
            continue

        # Derivative of RC at boundary.
        derivative = (2.0 * a * boundary + b)

        # First-order test:
        #
        # RC(boundary + direction * eps)
        # approximately
        # RC(boundary) + direction * derivative * eps
        #
        # We need RC to become negative.
        if direction * derivative < -tol:
            entering_candidates.append(int(var))
            continue

        # Tangential root:
        # derivative ~= 0, therefore inspect second-order term.
        if abs(derivative) <= tol and a < -tol:
            entering_candidates.append(int(var))

    return left, right, entering_candidates, B


@tp.timed("trace: _pivot_basis (all calls)")
def _pivot_basis(
    H: np.ndarray,
    h: np.ndarray,
    basis: np.ndarray,
    entering: int,
    B: np.ndarray | None = None,
) -> np.ndarray:
    """
    Perform a deterministic simplex pivot using the supplied
    entering variable.
    """
    basis = np.asarray(basis, dtype=int).copy()

    if B is None:
        B = H[:, basis]

    # Direction of basic variables when entering variable increases.
    direction_vector = np.linalg.solve(B, H[:, entering])

    # Current basic solution.
    x_basic = np.linalg.solve(B, h)

    # Variables that can hit zero.
    eligible = np.flatnonzero(direction_vector > 1e-9)

    if eligible.size == 0:
        raise RuntimeError(
            "Parametric LP became unbounded during pivot"
        )

    ratios = x_basic[eligible] / direction_vector[eligible]

    min_ratio = np.min(ratios)

    # Ratio-test tie.
    tied = eligible[
        np.isclose(
            ratios, min_ratio, atol=1e-9, rtol=1e-9,
        )
    ]

    # Bland tie-break:
    # choose the leaving basic variable with smallest index.
    leaving_row = min(
        tied,
        key=lambda row: basis[int(row)],
    )

    basis[int(leaving_row)] = entering

    return np.sort(basis)


@tp.timed("trace_parametric_bases (fn)")
def trace_parametric_bases(
    H: np.ndarray,
    h: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    initial_z: float,
    initial_basis: np.ndarray,
    z_bound: float,
    direction: int,
    config: SIConfig,
) -> list[tuple[float, float, np.ndarray]]:
    """
    Trace one side of a quadratic-cost simplex path from the observation.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")

    z = float(initial_z)
    basis = np.asarray(initial_basis, dtype=int).copy()

    result: list[tuple[float, float, np.ndarray]] = []

    tol = 1e-8

    while ((direction > 0 and z < z_bound - tol) or (direction < 0 and z > z_bound + tol)):
        left, right, entering_candidates, B_sys = _basis_optimal_region(
            H=H,
            u=u, v=v, w=w,
            basis=basis, z=z,
            direction=direction,
        )

        # Restrict interval to requested search range.
        interval_left = max(left,min(initial_z, z_bound),)
        interval_right = min(right, max(initial_z, z_bound),)

        result.append(
            (
                interval_left,
                interval_right,
                basis.copy(),
            )
        )

        # Boundary in requested direction.
        boundary = (interval_right if direction > 0 else interval_left)

        # Reached target.
        if (direction > 0 and boundary >= z_bound - tol) or (direction < 0 and boundary <= z_bound + tol):
            break

        # No entering variable means the boundary was not caused
        # by an RC becoming negative in the requested direction.
        if not entering_candidates:
            raise RuntimeError(
                "No entering variable found at parametric boundary "
                f"z={boundary}, direction={direction}, "
                f"basis={basis.tolist()}"
            )

        # Bland rule for entering variable.
        entering = min(entering_candidates)

        # Pivot using the entering variable already identified
        # from the active reduced-cost boundary.
        basis = _pivot_basis(
            H=H, h=h,
            basis=basis,
            entering=entering,
            B=B_sys,
        )

        # Move just inside the new basis region.
        #
        # We no longer use config.step to identify entering.
        # It is only used to move away from the exact breakpoint.
        epsilon = max(
            1e-9,
            min(1e-7, abs(boundary) * 1e-10,),
        )

        z = boundary + direction * epsilon

    return result


def parametric_intervals(
    Xtrain: np.ndarray,
    Xval: np.ndarray,
    decomposition: SIDecomposition,
    adapter: SelectionAdapter,
    observed_selection: Any,
    config: SIConfig,
) -> list[tuple[float, float]]:
    """the selection region by scanning ``z`` on a fixed grid."""
    if config.step <= 0:
        raise ValueError("step must be positive")
    if config.zmin >= config.zmax:
        raise ValueError("zmin must be smaller than zmax")


    a_vec = decomposition.a.reshape(-1,1)
    b_vec = decomposition.b.reshape(-1,1)

    final_interval = []
    nt, nv = Xtrain.shape[0], Xval.shape[0]
    X = np.vstack([Xtrain, Xval])

    # --- PRE-COMPUTE DA BASIS INTERVALS ---
    OMEGA = OptimalTransport.constructOMEGA(nt, nv)
    c_ = 0
    for i in range(X.shape[1]):
        c_ += np.dot(OMEGA, X[:, [i]]) * np.dot(OMEGA,X[:, [i]])

    S_ = adapter.S_.copy()
    h_ = adapter.h_.copy()
    Omega_a = OMEGA.dot(a_vec)
    Omega_b = OMEGA.dot(b_vec)
    wt = c_ + Omega_a * Omega_a
    rt = 2 * Omega_a * Omega_b
    ot = Omega_b * Omega_b
    # cost_params = (wt, rt, ot)

    # The initial basis must be the basis that produced the observed
    # selection.  Starting independently at zmin selects a different valid
    # degenerate basis at etaT_y and makes the observed event disappear.
    # c_start = wt + rt * config.zmin + ot * (config.zmin**2)
    # res = linprog(c_start, 
    #             A_ub = - np.identity(nt * nv), b_ub = np.zeros((nt * nv, 1)), 
    #                 # bounds=(0, None),
    #                     A_eq=S_, b_eq=h_, method='simplex',
    #                     options={'maxiter': 1000000,
    #                             'bland': True
    #                             }
    #                             )
    z_obs = decomposition.etaT_y

    base_args = (
        S_, 
        h_.reshape(-1), 
        wt.reshape(-1), 
        rt.reshape(-1), 
        ot.reshape(-1), 
        z_obs, 
        adapter.basis
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_right = executor.submit(
            trace_parametric_bases, 
            *base_args, config.zmax, direction=1, config=config
        )
        future_left = executor.submit(
            trace_parametric_bases, 
            *base_args, config.zmin, direction=-1, config=config
        )

        right_path = future_right.result()
        left_path = future_left.result()

    # right_path = trace_parametric_bases(
    #     S_, h_.reshape(-1), wt.reshape(-1), rt.reshape(-1), ot.reshape(-1),
    #     z_obs, adapter.basis, config.zmax, direction=1, config = config
    # )
    # left_path = trace_parametric_bases(
    #     S_, h_.reshape(-1), wt.reshape(-1), rt.reshape(-1), ot.reshape(-1),
    #     z_obs, adapter.basis, config.zmin, direction=-1, config = config
    # )

    # right_path = parametric_simplex_quadratic(nt, nv, S_, h_, wt, rt, ot, z_obs, config.zmax, direction = 1)
    # left_path = parametric_simplex_quadratic(nt, nv, S_, h_, wt, rt, ot, z_obs, config.zmin, direction = -1)
    # Both paths contain the observed basis region; retain it once.
    last_left = left_path[0]
    first_right = right_path[0]
    # print(last_left)
    # print(first_right)
    if np.array_equal(last_left[2], first_right[2]):
        joined_path = last_left[0], first_right[1], last_left[2]
        # print('j',joined_path)
        # print()
    else:
        raise RuntimeError("Left and right paths do not meet at the observed basis")
    # print('left after',list(reversed(left_path[1:])))
    list_da_itv = list(reversed(left_path[1:])) + [joined_path] + right_path[1:]
    # list_da_itv = right_path
    # print(list_da_itv)
    # print('')
    zold = -50
    for ml, mr, B in list_da_itv:
        # print(f'[{np.round(ml,7)}, {np.round(mr,7)}]',end='')
        # if ml > -0.27 and mr < -0.17:
            # print('basis', B)
        z = ml + config.step
        while z < mr:
            # print(z)
            cost_z = wt + rt*z + ot*(z**2)
            # S_ = adapter.S_.copy()
            selection_z, sorted_indices_z = adapter.select_with_known_basis(nt, S_, cost_z, B)

            selec_itv = sort_conditioning(z,adapter, B,wt, rt, ot, sorted_indices_z,config)
            over = interval_intersection(selec_itv, [(ml,mr)])
            if not over:
                raise RuntimeError("Selection interval does not intersect its DA region")
            z = over[-1][1] + config.step


            if z == zold:
                raise RuntimeError("z did not advance;")
            zold = z
            if observed_selection == selection_z:
                final_interval.extend(over)

    final_interval = interval_union(final_interval)
    # print('\n\nfinal interval after', final_interval)
    if not any(left <= z_obs <= right for left, right in final_interval):
        raise RuntimeError("Parametric selection region does not contain etaT_y")
    return final_interval


def find_valid_basis(H, x_nonzero_indices):
    """Tìm một tập chỉ số sao cho ma trận H[:, basis] là khả nghịch (Full Rank)"""
    num_constraints = H.shape[0]
    basis = list(x_nonzero_indices)
    
    # Sử dụng khử Gauss (hoặc QR) để tìm các cột độc lập tuyến tính bổ sung
    # Ở đây dùng phương pháp đơn giản: thử từng cột
    for i in range(H.shape[1]):
        if len(basis) == num_constraints:
            break
        if i not in basis:
            test_basis = basis + [i]
            if np.linalg.matrix_rank(H[:, test_basis]) == len(test_basis):
                basis.append(i)
    return sorted(basis)


def parametric_simplex_quadratic(ns, nt, H, h, u, v, w, z_obs, goal, direction = 1):
# def parametric_simplex_quadratic(ns, nt, H, h, u, v, w, z_min, z_max):
    H = np.array(H, dtype=float)
    h = np.array(h, dtype=float)
    u, v, w = np.array(u), np.array(v), np.array(w)
    # print("Shape:", H.shape, h.shape, u.shape, v.shape, w.shape)
    h = h.flatten()
    u, v, w = u.flatten(), v.flatten(), w.flatten()
    num_constraints, num_vars = H.shape
    if direction not in [-1,1]:
        raise ValueError('direction must be -1 or 1')
    current_z = z_obs
    results = []

    # 1. Tìm lời giải ban đầu tại z_min
    
    c_start = u + v * current_z + w * (current_z**2)
    res = linprog(c_start, 
                  A_ub = - np.identity(ns * nt), b_ub = np.zeros((ns * nt, 1)), 
                        # bounds=(0, None),
                           A_eq=H, b_eq=h, method='simplex',
                           options={'maxiter': 1000000,
                                    # 'bland': True
                                    }
                                    )



    if not res.success:
        return [] # Trả về rỗng nếu không có lời giải khả thi

    # Xác định các biến cơ sở ban đầu (đảm bảo ma trận vuông và độc lập tuyến tính)
    basis_indices = res.basis 
    # print('start basis',basis_indices)
    while np.abs(current_z) < np.abs(goal):
        # print('new zone')
        B = H[:, basis_indices]
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            # Nếu xảy ra lỗi ma trận suy biến, cố gắng tìm basis khác từ x hiện tại
            break

        # 2. Tính hệ số Reduced Cost: RC(z) = A*z^2 + B*z + C
        cb_u, cb_v, cb_w = u[basis_indices], v[basis_indices], w[basis_indices]
        
        # Công thức: rc = c - cB * B^-1 * H
        pi_u = cb_u.dot(B_inv)
        pi_v = cb_v.dot(B_inv)
        pi_w = cb_w.dot(B_inv)

        rc_u = u - pi_u.dot(H)
        rc_v = v - pi_v.dot(H)
        rc_w = w - pi_w.dot(H)

        # 3. Tìm z_next (điểm mà một RC biến ngoài cơ sở trở thành < 0)
        next_z = goal
        entering_var = None
        
        for i in range(num_vars):
            if i in basis_indices:
                continue
            
            # Giải phương trình RC_i(z) = 0
            root = solve_quadratic_roots(rc_w[i], rc_v[i], rc_u[i], current_z, goal)
            if root is not None:
                if (root < next_z and direction == 1) or (root > next_z and direction == -1):
                    next_z = root
                    entering_var = i

        # Lưu lại khoảng ổn định
        results.append((current_z, next_z, np.sort(basis_indices)))

        if entering_var is None or (next_z >= goal and direction == 1) and (next_z <= goal and direction == -1):
            break

        # 4. Thực hiện Pivot (Rời khỏi cơ sở)
        # Tính hướng di chuyển d = B^-1 * H_entering
        d = B_inv.dot(H[:, entering_var])
        x_B = B_inv.dot(h)
        
        # Tỷ số Min-ratio để tìm biến rời khỏi cơ sở
        min_ratio = np.inf
        leaving_idx_in_basis = -1
        # leaving_var_id = np.inf
        
        for i in range(len(d)):
            if d[i] > 1e-9:
                ratio = x_B[i] / d[i]
                print(f"ratio {ratio} | min ratio {min_ratio} | index {i}")

                if ratio < min_ratio:
                    min_ratio = ratio
                    leaving_idx_in_basis = i
                    # print(i)
                    # leaving_var_id = basis_indices[i]
                # elif abs(ratio - min_ratio) <= 1e-8:
                #     # tie: prefer smaller actual variable index (Bland's rule)
                #     if basis_indices[i] < leaving_var_id:
                #         leaving_idx_in_basis = i
                #         leaving_var_id = basis_indices[i]
                #         print('choose small index',i)
        
        if leaving_idx_in_basis == -1: # Không tìm được biến rời đi (unbounded)
            break
            
        basis_indices[leaving_idx_in_basis] = entering_var
        current_z = next_z

    return results

def compute_DA_intervals(ns, nt, a, b, c_, S, h, m_min=-20, m_max=20):
    OMEGA = OptimalTransport.constructOMEGA(ns, nt)

    Omega_a = OMEGA.dot(a)
    Omega_b = OMEGA.dot(b)

    w_tilde = c_ + Omega_a * Omega_a
    r_tilde = 2*Omega_a * Omega_b
    o_tilde = Omega_b * Omega_b
    # print(w_tilde.shape,r_tilde.shape,o_tilde.shape)
    cost_params = (w_tilde, r_tilde, o_tilde)

    intervals = parametric_simplex_quadratic(ns, nt,
        S, h,
        w_tilde, r_tilde, o_tilde,
        z_min=m_min,
        z_max=m_max
    )
    return {"intervals": intervals, "cost_params": cost_params}

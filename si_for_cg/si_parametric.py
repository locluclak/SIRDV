"""Parametric selective-inference scan."""

from __future__ import annotations

from typing import Any

import numpy as np
from dataclasses import replace
from .si_adapter import SelectionAdapter, CGSelectionAdapter

from .si_types import SIConfig, SIDecomposition, SelectionAdapter
from .si_overconditioning import sort_conditioning, overconditioning_intervals, CG_overconditioning_intervals
from .si_utils import interval_union, normalize_selection, solve_quadratic_roots, interval_intersection
from . import OptimalTransport
from scipy.optimize import linprog
from .si_adapter import SelectionAdapter

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
    with open("logs.txt", "w") as f:
        for z_val, sel, detect in logs:
            f.write(f"z: {z_val:.2f}, selection: {sel} -- {detect}\n")
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
    """Approximate the selection region by scanning ``z`` on a fixed grid."""
    if config.step <= 0:
        raise ValueError("step must be positive")
    if config.zmin >= config.zmax:
        raise ValueError("zmin must be smaller than zmax")

    # observed_key = normalize_selection(observed_selection)
    # intervals: list[tuple[float, float]] = []

    a_vec = decomposition.a.reshape(-1,1)
    b_vec = decomposition.b.reshape(-1,1)

    final_interval = []
    nt, nv = Xtrain.shape[0], Xval.shape[0]
    # X = np.vstack([Xtrain, Xval])
    # z = config.zmin

    y_z = a_vec + b_vec*z
    ytrain_z = y_z[:nt].copy().reshape(-1)
    yval_z = y_z[nt:nt+nv].copy().reshape(-1)

    adapter_z = SelectionAdapter(adapter.fraction)
    observed_selection_z = adapter_z.select(Xtrain, ytrain_z, Xval, yval_z)
    decomposition_z = replace(decomposition,etaT_y=z) 
    # decomposition_z.etaT_y = z
    itv= overconditioning_intervals(Xtrain, Xval, decomposition_z,adapter_z,config, z=z)

    return itv
def traditional_parametric_intervals(
    Xtrain: np.ndarray,
    Xval: np.ndarray,
    decomposition: SIDecomposition,
    adapter: SelectionAdapter,
    observed_selection: Any,
    config: SIConfig,
) -> list[tuple[float, float]]:
    """Approximate the selection region by scanning ``z`` on a fixed grid."""
    if config.step <= 0:
        raise ValueError("step must be positive")
    if config.zmin >= config.zmax:
        raise ValueError("zmin must be smaller than zmax")

    # observed_key = normalize_selection(observed_selection)
    # intervals: list[tuple[float, float]] = []

    a_vec = decomposition.a.reshape(-1,1)
    b_vec = decomposition.b.reshape(-1,1)

    final_interval = []
    nt, nv = Xtrain.shape[0], Xval.shape[0]
    # X = np.vstack([Xtrain, Xval])
    z = config.zmin
    etaT_y = decomposition.etaT_y
    DA = []
    while z < config.zmax:
        # print('z',z)
        y_z = a_vec + b_vec*z
        ytrain_z = y_z[:nt].copy().reshape(-1)
        yval_z = y_z[nt:nt+nv].copy().reshape(-1)

        adapter_z = SelectionAdapter(adapter.fraction)
        observed_selection_z = adapter_z.select(Xtrain, ytrain_z, Xval, yval_z)
        decomposition_z = replace(decomposition,etaT_y=z) 
        # decomposition_z.etaT_y = z
        itv, cost = overconditioning_intervals(Xtrain, Xval, decomposition_z,adapter_z,config, onlyDA=True,z=z)
        # print(itv)
        DA.append(itv)
        z = itv[1]+config.step


    wt,rt,ot = cost
    for ml, mr, B in DA:
        print('daitv', ml, mr)
        print('basis', B)
        z = ml + config.step
        while z < mr:
            # adapter_z = SelectionAdapter(adapter.fraction)
            print('z',z)
            cost_z = wt + rt*z + ot*z**2
            S_ = adapter.S_.copy()
            selection_z, top_indices_z = adapter.select_with_known_basis(nt, S_, cost_z, B)
            selec_itv = sort_conditioning(z,adapter, B,wt, rt, ot, top_indices_z,config)
            if 0 <= z < 2:
            #     T = np.zeros(nt * nv)
                
            #     x_B = np.linalg.inv(adapter.S_[:, B]).dot(adapter.h_)
            #     T[B] = x_B.flatten()
                print('    org_cost', np.sum(cost_z))
                # print('    old_cost', cost_z.reshape(-1).dot(T))
                print('    selection', selection_z)
                print('    old selecitv', selec_itv)
            # print(' selec',selec_itv)
            selec_itv = interval_intersection(selec_itv, [(ml,mr)])
            for i in selec_itv:
                l,r = i
                if l <= z <=r:
                    z = r +config.step
                    break
            # print(selection_z)
            if observed_selection == selection_z:
                print('correct')
                final_interval.append((l,r))
            # print(itv)
            
    return final_interval

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

    # observed_key = normalize_selection(observed_selection)
    intervals: list[tuple[float, float]] = []

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
    DA_result = compute_DA_intervals(nt, nv, a_vec, b_vec, c_, S_, h_, 
                                       m_min=config.zmin, m_max=config.zmax)
    list_da_itv, cost_params = DA_result["intervals"], DA_result["cost_params"]

    wt,rt,ot = cost_params
    for ml, mr, B in list_da_itv:
        # print('DAitv', ml, mr, B)
        z = ml + config.step
        while z < mr:
            print(z)
            # adapter_z = SelectionAdapter(adapter.fraction)
            cost_z = wt + rt*z + ot*(z**2)
            S_ = adapter.S_.copy()
            selection_z, top_indices_z = adapter.select_with_known_basis(nt, S_, cost_z, B)
                        
            selec_itv = sort_conditioning(z,adapter, B,wt, rt, ot, top_indices_z,config)
            # if -7.22503 <= z < -4:
            #     T = np.zeros(nt * nv)
                                
            #     x_B = np.linalg.inv(adapter.S_[:, B]).dot(adapter.h_)
            #     T[B] = x_B.flatten()
                # print('    org_cost', np.sum(cost_z))
                # print('    mor_cost', cost_z.reshape(-1).dot(T))
                # print('    selection', selection_z)
                # print('    old selecitv', selec_itv)
                # print('    selection', selection_z)
            over = interval_intersection(selec_itv, [(ml,mr)])
            z = over[-1][1] + config.step
            
            # print(selection_z)
            if observed_selection == selection_z:
                print('correct')
                final_interval.append(over)

            # print('new',z)

            

            
    return final_interval
    # return intervals

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


def parametric_simplex_quadratic(ns, nt, H, h, u, v, w, z_min, z_max):
    H = np.array(H, dtype=float)
    h = np.array(h, dtype=float)
    u, v, w = np.array(u), np.array(v), np.array(w)
    # print("Shape:", H.shape, h.shape, u.shape, v.shape, w.shape)
    h = h.flatten()
    u, v, w = u.flatten(), v.flatten(), w.flatten()
    num_constraints, num_vars = H.shape
    current_z = z_min
    results = []

    # 1. Tìm lời giải ban đầu tại z_min
    
    c_start = u + v * z_min + w * (z_min**2)
    # print('start cost', np.sum(c_start))
    res = linprog(c_start, 
                #   A_ub = - np.identity(ns * nt), b_ub = np.zeros((ns * nt, 1)), 
                        bounds=(0, None),
                           A_eq=H, b_eq=h, method='simplex',
                           options={'maxiter': 1000000,'bland': True})
    # res = linprog(c_start, A_ub = - np.identity(ns * nt), b_ub = np.zeros((ns * nt, 1)), 
    #                 A_eq = H, b_eq = h, method = 'simplex', 
    #                 options={'maxiter': 100000})


    if not res.success:
        return [] # Trả về rỗng nếu không có lời giải khả thi

    # Xác định các biến cơ sở ban đầu (đảm bảo ma trận vuông và độc lập tuyến tính)
    basis_indices = res.basis 
    # print('start basis',basis_indices)
    while current_z < z_max:
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
        next_z = z_max
        entering_var = None
        
        for i in range(num_vars):
            if i in basis_indices:
                continue
            
            # Giải phương trình RC_i(z) = 0
            root = solve_quadratic_roots(rc_w[i], rc_v[i], rc_u[i], current_z, z_max)
            if root is not None and root < next_z:
                next_z = root
                entering_var = i

        # Lưu lại khoảng ổn định
        results.append((current_z, next_z, np.sort(basis_indices)))

        if entering_var is None or next_z >= z_max:
            break

        # 4. Thực hiện Pivot (Rời khỏi cơ sở)
        # Tính hướng di chuyển d = B^-1 * H_entering
        d = B_inv.dot(H[:, entering_var])
        x_B = B_inv.dot(h)
        
        # Tỷ số Min-ratio để tìm biến rời khỏi cơ sở
        min_ratio = np.inf
        leaving_idx_in_basis = -1
        leaving_var_id = np.inf
        
        for i in range(len(d)):
            if d[i] > 1e-9:
                ratio = x_B[i] / d[i]
                print(f"ratio {ratio} | min ratio {min_ratio} | index {i}")

                if ratio < min_ratio:
                    min_ratio = ratio
                    leaving_idx_in_basis = i
                    print(i)
                    leaving_var_id = basis_indices[i]
                elif abs(ratio - min_ratio) <= 1e-8:
                    # tie: prefer smaller actual variable index (Bland's rule)
                    if basis_indices[i] < leaving_var_id:
                        leaving_idx_in_basis = i
                        leaving_var_id = basis_indices[i]
                        print('choose small index',i)
        
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
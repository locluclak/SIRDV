"""Overconditioning selective inference using DV-score sort-condition inequalities."""

from __future__ import annotations

import numpy as np

from .si_types import SIConfig, SIDecomposition, SelectionAdapter, LassoInfo
from .si_utils import solve_quad_list, solve_linear_inequality, interval_intersection
from . import OptimalTransport
from scipy.linalg import lu_factor, lu_solve

def sort_conditioning(
    current_pos,
    adapter: SelectionAdapter,
    basis_current: list[int],
    w_cost: np.ndarray,
    v_cost: np.ndarray,
    u_cost: np.ndarray, 
    top_indices: np.ndarray,
    config: SIConfig, 
) -> list[tuple[float, float]]:
    S_basis_T = adapter.S_[:, basis_current].T
    # print('wcost',w_cost.shape)
    nt = adapter.nt
    rhs = np.column_stack([
        u_cost[basis_current], 
        v_cost[basis_current], 
        w_cost[basis_current]
    ])
    rhs *= -1  # In-place negation

    lu, piv = lu_factor(S_basis_T)
    scores = lu_solve((lu, piv), rhs)[:nt]
    
    scores = scores * (1.0 + 1.0/(nt-1)) - np.sum(scores, axis=0)

    topreverse = top_indices[:-1]
    topremovefirst = top_indices[1:]

    diffs = scores[topreverse, :] - scores[topremovefirst, :]
    
    utilde = diffs[:, 0]
    vtilde = diffs[:, 1]
    wtilde = diffs[:, 2]
    # print(utilde)
    return solve_quad_list(utilde, vtilde, wtilde, z_debug=None)

def CG_overconditioning_intervals(
    X: np.ndarray,
    decomposition: SIDecomposition,
    adapter: SelectionAdapter,
    config: SIConfig,
    observed_selection: tuple[int, ...] =None, 
    fs: LassoInfo = None
) -> list[tuple[float, float]]:
    """
    Compute selection-condition intervals using the overconditioning method.

    CG_k(z) = (u_k + v_k * z)^2 / d_k is quadratic in z.
    The sorted order at z=0 must be preserved: CG_{σ(k)}(z) <= CG_{σ(k+1)}(z).
    This gives n-1 quadratic inequalities whose intersection is the valid z-region.
    """
    invH = adapter.invH
    if invH is None:
        raise ValueError("adapter.invH is None; call adapter.select() first")



    a_vec = decomposition.a.reshape(-1)
    b_vec = decomposition.b.reshape(-1)


    # CG_k(z) = (u_k + v_k * z)^2 / d_k
    d = np.diag(invH)
    # d = np.clip(d, 1e-8, None)

    u = invH.dot(a_vec)
    v = invH.dot(b_vec)

    p = v * v / d
    q = 2.0 * u * v / d
    r = u * u / d

    # Sort condition: CG_{σ(k)}(z) <= CG_{σ(k+1)}(z) for consecutive pairs
    sigma = adapter.sorted_indices
    if sigma is None:
        raise ValueError("adapter.sorted_indices is None; call adapter.select() first")

    topreverse = sigma[:-1].copy()
    topremovefirst = sigma[1:].copy()
    pt = p[topreverse] - p[topremovefirst]
    qt = q[topreverse] - q[topremovefirst]
    rt = r[topreverse] - r[topremovefirst]
    # print('pt', pt)
    # print('qt', qt)
    # print('rt', rt)
    intervals = solve_quad_list(pt, qt, rt)
    if fs:
        fs_itv = conditioning_Lasso(X[observed_selection], a_vec.reshape(-1,1)[observed_selection], b_vec.reshape(-1,1)[observed_selection], fs.lambda_, fs.M, fs.Mc, fs.sM)
        cg_oc_itv = interval_intersection(intervals, fs_itv)
        return cg_oc_itv
    return intervals
def overconditioning_intervals(
    Xtrain: np.ndarray,
    Xval: np.ndarray,
    decomposition: SIDecomposition,
    adapter: SelectionAdapter,
    config: SIConfig,
    observed_selection: tuple[int, ...] =None,
    onlyDA=False,
    z:float = 0,
    fs: LassoInfo = None
) -> list[tuple[float, float]]:
    """
    Compute selection-condition intervals using the overconditioning method.

    CG_k(z) = (u_k + v_k * z)^2 / d_k is quadratic in z.
    The sorted order at z=0 must be preserved: CG_{σ(k)}(z) <= CG_{σ(k+1)}(z).
    This gives n-1 quadratic inequalities whose intersection is the valid z-region.
    """
    a_vec = decomposition.a.reshape(-1,1)
    b_vec = decomposition.b.reshape(-1,1)

    B = adapter.basis
    # print('basis B', B)
    X = np.vstack([Xtrain, Xval])
    # print('first sample', X[-1])
    nt, nv = Xtrain.shape[0], Xval.shape[0]
    Bc = np.delete(np.array(range(nt*nv)), B)

    OMEGA = OptimalTransport.constructOMEGA(nt, nv)
    c_ = 0
    for i in range(X.shape[1]):
        c_ += np.dot(OMEGA, X[:, [i]]) * np.dot(OMEGA, X[:, [i]])
        # print("iter cost", np.sum(c_))

    Omega_a = OMEGA.dot(a_vec)
    Omega_b = OMEGA.dot(b_vec)

    w_tilde = c_ + Omega_a * Omega_a
    r_tilde = 2*Omega_a * Omega_b 
    o_tilde = Omega_b * Omega_b

    # print('cost oc',np.sum(w_tilde[B]+r_tilde[B]*z+o_tilde[B]*z**2))

    # print("cost c_", np.sum(c_))
    # print("cost", np.sum(w_tilde + r_tilde*decomposition.etaT_y + o_tilde*decomposition.etaT_y**2))

    S_ = adapter.S_
    h_ = adapter.h_
    
    S_B_invS_Bc = np.linalg.inv(S_[:, B]).dot(S_[:, Bc])

    w = (w_tilde[Bc, :].T - w_tilde[B, :].T.dot(S_B_invS_Bc)).T
    r = (r_tilde[Bc, :].T - r_tilde[B, :].T.dot(S_B_invS_Bc)).T
    o = (o_tilde[Bc, :].T - o_tilde[B, :].T.dot(S_B_invS_Bc)).T


    OT_interval = solve_quad_list(-o, -r, -w)
    # print('OC OT itv', OT_interval)
    curr = decomposition.etaT_y



    T = np.zeros(nt * nv)
    x_B = np.linalg.inv(adapter.S_[:, B]).dot(adapter.h_)
    T[B] = x_B.flatten()
    cost_z = w_tilde + r_tilde*-20 + o_tilde*-20**2
    # print('optimal cost', cost_z.reshape(-1).dot(T))

    if onlyDA:
        if len(OT_interval) > 1:
            print('SOS')
        for itv in OT_interval:
            l,r = itv
            if l <= curr <=r:
                return (l,r,B), (w_tilde, r_tilde, o_tilde)
        return None, None
    # print(curr)
    selection_interval = sort_conditioning(curr, adapter, B, w_tilde, r_tilde, o_tilde,adapter.sorted_indices,config)
    # print("OC selec itv", selection_interval)
    # print(a_vec[observed_selection].shape)
    fs_itv = conditioning_Lasso(Xtrain[observed_selection], a_vec[observed_selection], b_vec[observed_selection], fs.lambda_,fs.M, fs.Mc, fs.sM)

    # print('yobssub', a_vec[observed_selection] + b_vec[observed_selection]*decomposition.etaT_y)
    intervals = interval_intersection(
        interval_intersection(OT_interval, selection_interval),
        fs_itv
        )

    for itv in intervals:
        l,r = itv
        # print(l)
        # print(current_pos)
        if l <= curr <=r:
            return [(l,r)]
    return []
    # return interval_intersection(OT_interval, selection_interval)

def conditioning_Lasso(X_, a, b, Lambda, M, Mc, sM):
    X_M = X_[:, M]
    X_Mc = X_[:, Mc]

    X_MT_X_M_inv = np.linalg.inv(X_M.T.dot(X_M))
    X_M_plus = X_MT_X_M_inv.dot(X_M.T)
    X_MT_plus = X_M.dot(X_MT_X_M_inv)

    Atilde0 = - sM * (X_M_plus) 
    Btilde0 = - Lambda * sM * (X_MT_X_M_inv.dot(sM))

    Atilde10 = X_Mc.T.dot(np.identity(X_.shape[0]) - X_M.dot(X_M_plus))
    Atilde11 = - Atilde10

    tmp = X_Mc.T.dot(X_MT_plus).dot(sM)
    Btilde10 = np.ones((tmp.shape[0], 1)) - tmp
    Btilde11 = np.ones((tmp.shape[0], 1)) + tmp

    Atilde1 = np.vstack((Atilde10, Atilde11)) / Lambda
    Btilde1 = np.vstack((Btilde10, Btilde11))

    Atilde = np.vstack((Atilde0, Atilde1))
    Btilde = np.vstack((Btilde0, Btilde1))

    A = Atilde.dot(b)
    B = Btilde - Atilde.dot(a)
    return solve_linear_inequality(A, -B)
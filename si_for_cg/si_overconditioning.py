"""Overconditioning selective inference using DV-score sort-condition inequalities."""

from __future__ import annotations

import numpy as np

from .si_types import SIConfig, SIDecomposition, SelectionAdapter
from .si_utils import solve_quad_list, solve_linear_inequality, interval_intersection
from . import OptimalTransport

def sort_conditioning(
    current_pos,
    adapter: SelectionAdapter,
    basis_current: list[int],
    w_cost: np.ndarray,
    v_cost: np.ndarray,
    u_cost: np.ndarray, 
    top_indices,
    config: SIConfig, 
) -> list[tuple[float, float]]:
    # print("cp",current_pos)
    S_ = adapter.S_
    h_= adapter.h_
    S_basis = S_[:,basis_current]
    invSbasis = np.linalg.inv(S_basis.T)
    # observed_selection = observed_selection.reshape(-1)
    # print(basis_current)
    # print(v_cost)
    # print('index', top_indices)
    u_cost = u_cost[basis_current]
    v_cost = v_cost[basis_current]
    w_cost = w_cost[basis_current]
    u_scores = -1*np.dot(invSbasis, u_cost)
    v_scores = -1*np.dot(invSbasis, v_cost)
    w_scores = -1*np.dot(invSbasis, w_cost)

    u_scores = u_scores[:adapter.nt].copy()
    v_scores = v_scores[:adapter.nt].copy()
    w_scores = w_scores[:adapter.nt].copy()

    u_scores = 2*u_scores - np.sum(u_scores)
    v_scores = 2*v_scores - np.sum(v_scores)
    w_scores = 2*w_scores - np.sum(w_scores)
    # print('score', w_scores + v_scores*current_pos + u_scores*current_pos**2)

    topreverse = top_indices[:-1].copy()
    topremovefirst = top_indices[1:].copy()
    utilde = u_scores[topreverse] - u_scores[topremovefirst]
    vtilde = v_scores[topreverse] - v_scores[topremovefirst]
    wtilde = w_scores[topreverse] - w_scores[topremovefirst]

    intervals = solve_quad_list(utilde, vtilde, wtilde)

    return intervals

def CG_overconditioning_intervals(
    X: np.ndarray,
    decomposition: SIDecomposition,
    adapter: SelectionAdapter,
    config: SIConfig,
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

    return intervals
def overconditioning_intervals(
    Xtrain: np.ndarray,
    Xval: np.ndarray,
    decomposition: SIDecomposition,
    adapter: SelectionAdapter,
    # observed_selection: tuple[int, ...],
    config: SIConfig,
    onlyDA=False,
    z:float = 0,
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
    intervals = interval_intersection(OT_interval, selection_interval)
    for itv in intervals:
        l,r = itv
        # print(l)
        # print(current_pos)
        if l <= curr <=r:
            return [(l,r)]
    return []
    # return interval_intersection(OT_interval, selection_interval)


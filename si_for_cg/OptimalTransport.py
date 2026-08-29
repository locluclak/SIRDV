import numpy as np 
from scipy.optimize import linprog


def basis_is_optimal(S_, h_, cost, basis, tol: float = 1e-8) -> bool:
    """Check primal and dual feasibility of an OT basis for a fixed LP."""
    S_ = np.asarray(S_, dtype=float)
    h_ = np.asarray(h_, dtype=float).reshape(-1)
    cost = np.asarray(cost, dtype=float).reshape(-1)
    basis = np.asarray(basis, dtype=int).reshape(-1)
    if basis.size != S_.shape[0] or np.unique(basis).size != basis.size:
        return False

    try:
        S_basis = S_[:, basis]
        x_basis = np.linalg.solve(S_basis, h_)
        dual = np.linalg.solve(S_basis.T, cost[basis])
    except np.linalg.LinAlgError:
        return False
    if np.min(x_basis) < -tol:
        return False

    nonbasis = np.setdiff1d(np.arange(S_.shape[1]), basis, assume_unique=True)
    reduced_cost = cost[nonbasis] - S_[:, nonbasis].T.dot(dual)
    return bool(np.min(reduced_cost, initial=np.inf) >= -tol)

def constructOMEGA(n, m):
    vec1m = np.ones((m,1))
    vec0m = np.zeros((m,1))

    Im = -1*np.identity(m)

    vecIm = Im
    for i in range(n-1):
        vecIm = np.append(vecIm, Im, axis = 0)

    iden_vec1m = None
    for i in range(n):
        temp = None
        for j in range(n):
            if i == j:
                if temp is None:
                    temp = vec1m
                else:
                    temp = np.hstack((temp,vec1m))
            else:
                if temp is None:
                    temp = vec0m
                else:
                    temp = np.hstack((temp,vec0m))

        if iden_vec1m is None:
            iden_vec1m = temp
        else:
            iden_vec1m = np.vstack((iden_vec1m, temp))
    return np.hstack((iden_vec1m, vecIm))

def convert(m, n):
    A = np.arange(m*n)
    A = A.reshape((m,n))
    B = []
    
    for row in A:
        temp = np.zeros(m*n)
        for ele in row:
            temp[ele] = 1
        B.append(temp)
    for col in A.T:
        temp = np.zeros(m*n)
        for ele in col:
            temp[ele] = 1
        B.append(temp)
    return np.array(B)

def constructGamma(ns, nt, T):
    col0 = np.concatenate((np.zeros((ns, ns)), np.zeros((nt, ns))),axis = 0)
    col1 = np.concatenate((ns*T, np.identity(nt)), axis=0)
    return np.concatenate((col0, col1), axis = 1)

def solveOT(ns, nt, S_, h_, X_, basis_hint=None):
    OMEGA = constructOMEGA(ns,nt).copy()

    #Cost vector
    cost = 0
    # print("origin first sample", X_[-1])
    p = X_.shape[1] - 1 
    for i in range(p+1):
        cost += np.dot(OMEGA , X_[:, [i]]) * np.dot(OMEGA , X_[:, [i]])
        # if i==p-1:
        # print('origin cost c_', np.sum(cost))
    # print('cost', np.sum(cost))
    # Solve wasserstein distance
    # ``bounds`` is the non-negativity constraint T >= 0.  Do not add the
    # equivalent -T <= 0 rows: they are redundant and let the two-phase
    # simplex choose different degenerate bases after presolve.  Bland's rule
    # makes the remaining pivot tie-break deterministic along the z path.
    print('begin')
    res = linprog(cost,
                  bounds=(0, None),
                #   A_ub = - np.identity(ns * nt), b_ub = np.zeros((ns * nt, 1)), 
                  A_eq=S_,
                  b_eq=h_,
                  method='simplex',
                  options={'maxiter': 100_000,
                        #    'bland': True,
                        #    'presolve': False
                           })
    print('end')
    if not res.success:
        raise RuntimeError(f"Optimal transport LP failed: {res.message}")
    # Transport Map
    Tobs = res.x.reshape((ns,nt))
    gamma = constructGamma(ns, nt, Tobs)
    # At a degenerate vertex, multiple bases encode the same transport plan.
    # OC conditions on the observed basis, so retain it whenever it is still
    # optimal for this cost instead of replacing it with an arbitrary zero-arc
    # basis reached by a fresh simplex run.
    basis = res.basis
    # if basis_hint is not None and basis_is_optimal(S_, h_, cost, basis_hint):
    #     print('use basis hint')
    #     basis = np.asarray(basis_hint, dtype=int)
    return {"gamma": gamma, "basis": basis, "cost": cost,}

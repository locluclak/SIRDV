import numpy as np
from scipy.optimize import linprog
from . import si_numba


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
    if si_numba.USE and si_numba.AVAILABLE:
        return si_numba.n_construct_omega(n, m)
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


def _highs_basis_hint(highspy, basis_hint, m, n):
    """Turn an ``m``-column index array into a HiGHS basis to warm-start from."""
    if basis_hint is None:
        return None
    b = np.asarray(basis_hint, dtype=int).reshape(-1)
    if b.size != m or np.unique(b).size != b.size or b.min() < 0 or b.max() >= n:
        return None
    col_status = [highspy.HighsBasisStatus.kLower] * n
    for j in b:
        col_status[int(j)] = highspy.HighsBasisStatus.kBasic
    hb = highspy.HighsBasis()
    hb.valid = True
    hb.col_status = col_status
    hb.row_status = [highspy.HighsBasisStatus.kLower] * m
    return hb


def _solve_highs(S_, h_, cost, basis_hint=None):
    """Solve the OT LP with HiGHS dual revised simplex.

    Returns ``(x, basis)`` with ``basis`` the basic structural columns, i.e.
    ``S_[:, basis]`` is the optimal basis matrix (square, ``m`` columns).
    """
    import highspy

    S_ = np.asarray(S_, dtype=float)
    h_ = np.asarray(h_, dtype=float).reshape(-1)
    cost = np.asarray(cost, dtype=float).reshape(-1)
    m, n = S_.shape

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("solver", "simplex")
    highs.setOptionValue("presolve", "off")

    highs.addVars(n, np.zeros(n), np.full(n, np.inf))
    highs.changeColsCost(n, np.arange(n, dtype=np.int32), cost)

    # Bulk-load the sparse column structure of S_ (CSC, grouped by row).
    rows, cols = np.nonzero(S_)
    starts = np.zeros(m + 1, dtype=np.int32)
    starts[1:] = np.cumsum(np.bincount(rows, minlength=m))
    highs.addRows(m, h_, h_, int(rows.size), starts,
                  cols.astype(np.int32), S_[rows, cols])

    hint = _highs_basis_hint(highspy, basis_hint, m, n)
    if hint is not None:
        try:
            highs.setBasis(hint)
        except Exception:  # noqa: BLE001
            pass

    highs.run()
    if highs.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"HiGHS OT LP failed: {highs.getModelStatus()}")

    x = np.asarray(highs.getSolution().col_value, dtype=float)
    hs_basis = highs.getBasis()
    basic = np.asarray(
        np.where(np.asarray(hs_basis.col_status) == highspy.HighsBasisStatus.kBasic)[0],
        dtype=int,
    )
    return x, basic


def solveOT(ns, nt, S_, h_, X_, basis_hint=None, solver: str = "highs"):
    """Solve the OT LP, returning the transport map, optimal basis and cost.

    ``solver`` selects the LP backend: ``"highs"`` (default) uses the HiGHS
    dual revised simplex via ``highspy``; ``"scipy"`` uses the patched
    ``scipy.optimize.linprog(method='simplex')``.  The HiGHS path falls back to
    scipy automatically if ``highspy`` is unavailable or its basis is unusable.
    """
    OMEGA = constructOMEGA(ns,nt).copy()

    #Cost vector
    cost = 0
    # print("origin first sample", X_[-1])
    p = X_.shape[1] - 1
    for i in range(p+1):
        cost += np.dot(OMEGA , X_[:, [i]]) * np.dot(OMEGA , X_[:, [i]])

    if solver == "highs":
        try:
            x, basis = _solve_highs(S_, h_, cost, basis_hint=basis_hint)
        except Exception:  # noqa: BLE001
            x, basis = None, None
        if x is not None and basis.size == S_.shape[0]:
            Tobs = x.reshape((ns, nt))
            gamma = constructGamma(ns, nt, Tobs)
            return {"gamma": gamma, "basis": basis, "cost": cost}
    elif solver != "scipy":
        raise ValueError(f"unknown solver {solver!r}; expected 'highs' or 'scipy'")

    # scipy simplex path (also the fallback)
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
    if not res.success:
        raise RuntimeError(f"Optimal transport LP failed: {res.message}")
    # Transport Map
    Tobs = res.x.reshape((ns,nt))
    gamma = constructGamma(ns, nt, Tobs)
    basis = res.basis
    return {"gamma": gamma, "basis": basis, "cost": cost,}

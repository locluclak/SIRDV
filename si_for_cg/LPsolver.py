"""HiGHS-based LP solver that mirrors the subset of the scipy ``linprog``
result used across this package (``success``, ``x``, ``basis``, ``fun``).

The LP is solved with the HiGHS simplex engine in either its primal
(``method='simplex'``) or dual (``method='dual_simplex'``) variant, and an
optimal basis -- a sorted array of column indices into ``A_eq`` -- is
recovered from the optimal basic feasible solution, exactly like the
modified scipy ``linprog`` used before.
"""

import numpy as np
import highspy

__all__ = ["HIGHS_LPsolver", "LPsolverResult"]

#: HiGHS ``simplex_strategy`` values: kSimplexStrategyPrimal / kSimplexStrategyDual
_SIMPLEX_STRATEGY = {
    "simplex": 4,
    "dual_simplex": 1,
}

_COST_TOL = 1e-8  # reduced-cost / feasibility tolerance used by the basis recovery
#: Upper bound on the degenerate repair pivots; exceeded only on pathological
#: cases, where the complementary HiGHS simplex strategy is used instead.
_REPAIR_MAX_PIVOTS = 4_000


class LPsolverResult:
    """Lightweight result object with the attributes consumed by callers of
    the old scipy ``linprog`` result."""

    def __init__(self, success, x, basis, optimal_cost, message, status=0):
        self.success = bool(success)
        self.x = x
        self.basis = basis
        self.optimal_cost = float(optimal_cost)
        self.fun = float(optimal_cost)
        self.message = message
        self.status = int(status)

    def __repr__(self):
        return (
            f"LPsolverResult(success={self.success}, "
            f"optimal_cost={self.optimal_cost}, basis={self.basis})"
        )


def _solve_highs_lp(c, A, b, lower, upper, maxiter, method):
    """Solve ``min c @ x s.t. A @ x == b, lower <= x <= upper`` with HiGHS.

    Returns ``(model_status, x, basis_columns)`` where ``x`` is the
    (postsolved) primal solution and ``basis_columns`` the structural columns
    that HiGHS reports as basic in its final basis.  ``basis_columns`` can be
    shorter than ``A.shape[0]`` when HiGHS ends with synthetic (non-column)
    basic variables, e.g. after a highly degenerate dual-simplex run.
    """
    c = np.asarray(c, dtype=float).ravel()
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).ravel()
    n_rows, n_cols = A.shape

    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("solver", "simplex")
    solver.setOptionValue("simplex_strategy", _SIMPLEX_STRATEGY[method])
    solver.setOptionValue("simplex_maximum_iterations", int(maxiter))

    lp = highspy.HighsLp()
    lp.num_col_ = n_cols
    lp.num_row_ = n_rows
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.offset_ = 0.0
    lp.col_cost_ = np.asarray(c, dtype=float)
    if np.ndim(lower) == 0:
        lp.col_lower_ = np.full(n_cols, float(lower))
    else:
        lp.col_lower_ = np.asarray(lower, dtype=float)
    if np.ndim(upper) == 0:
        lp.col_upper_ = np.full(n_cols, float(upper))
    else:
        lp.col_upper_ = np.asarray(upper, dtype=float)
    lp.row_lower_ = np.asarray(b, dtype=float)
    lp.row_upper_ = np.asarray(b, dtype=float)

    col_index, value, start = [], [], [0]
    for i in range(n_rows):
        row = A[i]
        nz = np.flatnonzero(row)
        if nz.size:
            col_index.extend(nz.tolist())
            value.extend(row[nz].tolist())
        start.append(len(col_index))

    a_matrix = highspy.HighsSparseMatrix()
    a_matrix.format_ = highspy.MatrixFormat.kRowwise
    a_matrix.num_row_ = n_rows
    a_matrix.num_col_ = n_cols
    a_matrix.start_ = start
    a_matrix.index_ = col_index
    a_matrix.value_ = value
    lp.a_matrix_ = a_matrix

    solver.passModel(lp)
    solver.solve()

    status = solver.getModelStatus()
    solution = solver.getSolution()
    x = (np.asarray(solution.col_value, dtype=float)
         if solution.value_valid else np.zeros(n_cols))

    # Structural columns that HiGHS reports as basic (one per row).  Negative
    # entries in getBasicVariables() denote synthetic/non-column variables;
    # the col_status list is used to keep only genuine columns.
    basis_columns = np.asarray(list(range(n_cols)), dtype=int)[
        [s == highspy.HighsBasisStatus.kBasic for s in solver.getBasis().col_status]
    ]
    return status, x, basis_columns


def _greedy_basis(A, x, tol=_COST_TOL):
    """Extend the support of the basic feasible solution ``x`` to a full-rank
    basis of the columns of ``A`` (``A`` is assumed full row rank)."""
    m, n = A.shape
    x = np.asarray(x, dtype=float)
    scale_support = max(tol, 1e-9)
    support = np.flatnonzero(np.abs(x) > scale_support)
    non_support = np.flatnonzero(np.abs(x) <= scale_support)

    basis = []
    q_columns = np.empty((m, 0))
    for j in np.concatenate([support, non_support]):
        if len(basis) == m:
            break
        col = A[:, int(j)]
        if q_columns.size:
            residual = col - q_columns @ (q_columns.T @ col)
        else:
            residual = col
        if np.linalg.norm(residual) > tol * max(1.0, np.linalg.norm(col)):
            basis.append(int(j))
            q_columns = np.column_stack([q_columns, residual / np.linalg.norm(residual)])
    return np.asarray(basis, dtype=int)


def _greedy_basis_fallback(A, x, tol=_COST_TOL):
    """Rank-based variant used when the QR heuristic fails to reach rank m."""
    m, n = A.shape
    x = np.asarray(x, dtype=float)
    support = np.flatnonzero(np.abs(x) > max(tol, 1e-9))
    basis = [int(j) for j in support]
    for j in range(n):
        if len(basis) == m:
            break
        if j in basis:
            continue
        if np.linalg.matrix_rank(A[:, basis + [j]]) == len(basis) + 1:
            basis.append(j)
    return np.asarray(basis, dtype=int)


def _repair_basis(A, b, c, basis, tol=_COST_TOL, max_pivots=None):
    """Run degenerate simplex pivots (Bland rule) until the basis is
    dual-feasible, i.e. certifiably optimal for ``min c @ x s.t. A x = b``.

    Bounded by ``max_pivots``; on highly degenerate problems Bland pivoting
    can require exponentially many steps, in which case the caller should
    fall back to re-solving with the complementary HiGHS simplex strategy.
    """
    m, n = A.shape
    if max_pivots is None:
        max_pivots = _REPAIR_MAX_PIVOTS
    basis = [int(j) for j in basis]

    for _ in range(max_pivots):
        B = A[:, basis]
        try:
            x_basic = np.linalg.solve(B, b)
            dual = np.linalg.solve(B.T, c[basis])
        except np.linalg.LinAlgError:
            break
        if np.min(x_basic) < -1e-6:
            break
        reduced_cost = c - A.T @ dual
        reduced_cost[basis] = 0.0
        negative = np.flatnonzero(reduced_cost < -tol)
        if negative.size == 0:
            return np.sort(basis)
        entering = int(negative[0])
        direction = np.linalg.solve(B, A[:, entering])
        eligible = np.flatnonzero(direction > tol)
        if eligible.size == 0:
            break
        ratios = x_basic[eligible] / direction[eligible]
        leaving = int(eligible[np.argmin(ratios)])
        basis[leaving] = entering

    return np.sort(basis)


def _is_optimal_basis(A, b, c, basis, tol=_COST_TOL):
    """Primal/dual optimality check of a square basis (columns of ``A``)."""
    basis = np.asarray(basis, dtype=int).reshape(-1)
    m = A.shape[0]
    if basis.size != m or np.unique(basis).size != m:
        return False
    try:
        x_basic = np.linalg.solve(A[:, basis], b)
        dual = np.linalg.solve(A[:, basis].T, c[basis])
    except np.linalg.LinAlgError:
        return False
    if np.min(x_basic) < -tol:
        return False
    non_basis = np.setdiff1d(np.arange(A.shape[1]), basis, assume_unique=True)
    reduced_cost = c[non_basis] - A[:, non_basis].T.dot(dual)
    return bool(np.min(reduced_cost, initial=np.inf) >= -tol)


def _recover_basis(A, b, c, x, highs_basis_columns, method, tol=_COST_TOL):
    """Recover a square, full-rank, optimal basis for ``min c @ x s.t. A x = b``.

    Prefers the basis reported by HiGHS for the requested simplex strategy
    whenever it is a complete structural basis.  If HiGHS finished with
    synthetic (non-column) basic variables, re-solves with the complementary
    HiGHS simplex strategy and uses its basis when complete.  Only as a last
    resort (both strategies with non-structural bases) a greedy extension of
    the optimal support plus bounded degenerate pivoting is attempted.

    The returned basis is guaranteed to pass :func:`_is_optimal_basis`, so the
    downstream dual-potential computations are always valid.
    """
    m, n = A.shape
    if m == 0:
        return np.empty(0, dtype=int)

    basis = np.asarray(highs_basis_columns, dtype=int)
    basis = basis[(basis >= 0) & (basis < n)]
    if basis.size == m and _is_optimal_basis(A, b, c, basis, tol=tol):
        return np.sort(basis)

    # Pathological degeneracy: re-solve with the complementary strategy and use
    # whatever complete structural basis HiGHS reports there.
    other = "simplex" if method == "dual_simplex" else "dual_simplex"
    lower_cols = np.asarray(n * [0.0])
    upper_cols = np.asarray(n * [np.inf])
    status, x_other, other_cols = _solve_highs_lp(
        c, A, b, lower_cols, upper_cols, 100_000, other
    )
    if status == highspy.HighsModelStatus.kOptimal:
        other_basis = np.asarray(other_cols, dtype=int)
        other_basis = other_basis[(other_basis >= 0) & (other_basis < n)]
        if other_basis.size == m and _is_optimal_basis(A, b, c, other_basis, tol=tol):
            return np.sort(other_basis)

    x = np.asarray(x, dtype=float)
    basis = _greedy_basis(A, x, tol=tol)
    if basis.size != m:
        basis = _greedy_basis_fallback(A, x, tol=tol)
    if basis.size == m:
        basis = _repair_basis(A, b, c, basis, tol=tol)
        if _is_optimal_basis(A, b, c, basis, tol=tol):
            return np.sort(basis)

    # Both HiGHS strategies finished with synthetic basics and the greedy
    # repair could not certify a full optimal basis.
    raise RuntimeError(
        "Could not recover a complete optimal basis for the LP; "
        "the problem is highly degenerate and both HiGHS simplex strategies "
        "finished with synthetic basic variables."
    )


def HIGHS_LPsolver(c, A_eq=None, b_eq=None, bounds=(0, None),
                   method="dual_simplex", maxiter=100_000):
    """Solve the LP ``min c @ x  s.t.  A_eq @ x == b_eq,  x >= 0`` with HiGHS.

    Parameters mirror the ``linprog`` usage of this package:

    - ``c``: cost vector, shape ``(n, 1)`` or ``(n,)``.
    - ``A_eq``: equality constraint matrix, shape ``(m, n)`` (full row rank).
    - ``b_eq``: equality right-hand side, shape ``(m,)`` or ``(m, 1)``.
    - ``bounds``: ``(lower, upper)`` variable bounds (scalars or arrays).
    - ``method``: ``'dual_simplex'`` (HiGHS dual simplex, default; it returns
      the same optimal basis as the legacy scipy ``linprog`` on the OT
      problems used here) or ``'simplex'`` (HiGHS primal simplex).

    Returns an :class:`LPsolverResult` exposing the attributes that the old
    modified scipy ``linprog`` result exposed:

    - ``success``: did HiGHS find an optimal solution?
    - ``x``: optimal primal solution.
    - ``basis``: sorted indices of an optimal basis (columns of ``A_eq``).
    - ``optimal_cost`` / ``fun``: optimal objective value.
    - ``message`` / ``status``: status information.
    """
    if A_eq is None or b_eq is None:
        raise ValueError("HIGHS_LPsolver requires A_eq and b_eq")
    if method not in _SIMPLEX_STRATEGY:
        raise ValueError(f"method must be one of {list(_SIMPLEX_STRATEGY)}")

    c = np.asarray(c, dtype=float).ravel()
    A = np.asarray(A_eq, dtype=float)
    b = np.asarray(b_eq, dtype=float).ravel()
    if A.ndim != 2:
        raise ValueError("A_eq must be a 2-D matrix")
    if c.size != A.shape[1]:
        raise ValueError("cost vector length does not match A_eq columns")

    lower, upper = bounds if isinstance(bounds, tuple) else (bounds, None)
    if lower is None:
        lower = -np.inf
    if upper is None:
        upper = np.inf

    status, x, basis_columns = _solve_highs_lp(c, A, b, lower, upper, maxiter, method)

    if status != highspy.HighsModelStatus.kOptimal:
        return LPsolverResult(
            success=False,
            x=x,
            basis=np.empty(0, dtype=int),
            optimal_cost=np.nan,
            message=f"HiGHS model status: {status}",
            status=int(status),
        )

    basis = _recover_basis(A, b, c, x, basis_columns, method)
    return LPsolverResult(
        success=True,
        x=x,
        basis=basis,
        optimal_cost=float(np.dot(c, x)),
        message="Optimization terminated successfully.",
        status=0,
    )
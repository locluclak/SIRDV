"""Numerical helpers for selective inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from mpmath import mp
from numpy.linalg import lstsq
from scipy.linalg import toeplitz

from .si_types import SIDecomposition
from sklearn.linear_model import Lasso

# 30 significant digits are sufficient for every p-value CDF: the masses in
# compute_p_value are evaluated in tail form (no `1 - tiny` subtraction), so
# mpmath returns each far-tail piece with full `mp.dps` significant digits
# regardless of its magnitude.  Raise this only if some consumer needs more
# digits of the accumulated sums themselves.
mp.dps = 30

def generate_contaminated_regression(
    n_train: int = 80,
    n_val: int = 20,
    n_test: int = 2_000,
    p: int = 5,
    contamination: float = 0.30,
    beta_clean: float = 1.0,
    std_good: float = 1.0,
    std_bad: float = 3.0,
    seed: int | None = None,
):
    """Create a clean target domain plus harmful training observations.

Validation and test use ``beta_clean``.  Contaminated training observations
use a different conditional relationship and substantially higher noise.
``is_clean`` is only an oracle diagnostic, never an input to LAVA or Lasso.
"""
    if not 0 <= contamination < 1:
        raise ValueError("contamination must be in [0, 1).")

    rng = np.random.default_rng(seed)
    beta_clean = np.full(p,beta_clean)
    beta_bad = np.zeros(p)

    X_train = rng.normal(size=(n_train, p))
    is_bad = rng.random(n_train) < contamination
    noise_sd = np.where(is_bad, std_bad, std_good)
    beta_per_row = np.where(is_bad[:, None], beta_bad, beta_clean)
    y_train = (X_train * beta_per_row).sum(axis=1) + rng.normal(scale=noise_sd)

    X_val = rng.normal(size=(n_val, p))
    y_val = X_val @ beta_clean + rng.normal(scale=std_good, size=n_val)
    X_test = rng.normal(size=(n_test, p))
    y_test = X_test @ beta_clean + rng.normal(scale=std_good, size=n_test)

    Sigmat = np.diag(noise_sd ** 2)
    Sigmav = std_good**2 * np.eye(n_val)
    return X_train, y_train, X_val, y_val, X_test, y_test, Sigmat, Sigmav, ~is_bad




def generate_correlated(
    n, m, p, true_beta_s, true_beta_t, rho=0.5, seed=None
):
    """Generates synthetic domain adaptation data where target and source noise

    follow an AR(1) covariance structure: \Xi = [\rho^{|i-j|}]_{ij}.

    Returns: Xs, Xt, Ys, Yt, Sigma_s, Sigma_t
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Generate design matrices
    Xs = np.random.normal(loc=0, scale=1, size=(n, p))
    Xt = np.random.normal(loc=0, scale=1, size=(m, p))

    # 2. Compute true signals
    mu_s = Xs.dot(true_beta_s)
    mu_t = Xt.dot(true_beta_t)

    # 3. Construct Toeplitz covariance matrices: \Xi_{i,j} = \rho^|i-j|
    Sigma_s = toeplitz(rho ** np.arange(n))
    Sigma_t = toeplitz(rho ** np.arange(m))

    # 4. Generate correlated noise from Multivariate Normal distributions
    noise_s = np.random.multivariate_normal(
        mean=np.zeros(n), cov=Sigma_s
    ).reshape(n, 1)
    noise_t = np.random.multivariate_normal(
        mean=np.zeros(m), cov=Sigma_t
    ).reshape(m, 1)

    # 5. Add correlated noise to response variables
    Ys = mu_s + noise_s
    Yt = mu_t + noise_t

    return Xs, Xt, Ys, Yt, Sigma_s, Sigma_t

def generate_equicorrelated(
    n, m, p, true_beta_s, true_beta_t, rho=0.5, scale=1.0, seed=None
):
    """Synthetic data with long-range (compound symmetry) correlated noise.

    Unlike the AR(1) Toeplitz structure of ``generate_correlated`` (which
    decays as rho^|i-j| and therefore survives a random train/hold-out split
    only weakly), every pair of noise entries here shares correlation rho,
    so a selection on one part of the data stays informative about the
    held-out part.  ``scale`` sets the per-row noise sd.

    Returns: Xs, Xt, Ys, Yt, Sigma_s, Sigma_t
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Generate design matrices
    Xs = np.random.normal(loc=0, scale=1, size=(n, p))
    Xt = np.random.normal(loc=0, scale=1, size=(m, p))

    # 2. Compute true signals
    mu_s = Xs.dot(true_beta_s)
    mu_t = Xt.dot(true_beta_t)

    # 3. Compound-symmetry covariance: Xi = s^2 * [(1 - rho) I + rho 11^T]
    Sigma_s = scale**2 * ((1 - rho) * np.eye(n) + rho * np.ones((n, n)))
    Sigma_t = scale**2 * ((1 - rho) * np.eye(m) + rho * np.ones((m, m)))

    # 4. Generate correlated noise from Multivariate Normal distributions
    noise_s = np.random.multivariate_normal(
        mean=np.zeros(n), cov=Sigma_s
    ).reshape(n, 1)
    noise_t = np.random.multivariate_normal(
        mean=np.zeros(m), cov=Sigma_t
    ).reshape(m, 1)

    # 5. Add correlated noise to response variables
    Ys = mu_s + noise_s
    Yt = mu_t + noise_t

    return Xs, Xt, Ys, Yt, Sigma_s, Sigma_t

def generate(n, m, p, true_beta_s, true_beta_t, seed = None):
    """return Xs, Xt, Ys, Yt, Sigma_s, Sigma_t"""

    if seed is not None:
        np.random.seed(seed)

    Xs = np.random.normal(loc = 0, scale = 1, size = (n, p))
    Xt = np.random.normal(loc = 0, scale = 1, size = (m, p))


    mu_s = Xs.dot(true_beta_s)
    mu_t = Xt.dot(true_beta_t)
    # print(true_beta_s, true_beta_t)
    Ys = mu_s + np.random.normal(loc = 0, scale = 1, size = (n, 1))
    Yt = mu_t + np.random.normal(loc = 0, scale = 1, size = (m, 1))

    Sigma_s = np.identity(n)
    Sigma_t = np.identity(m)

    return Xs, Xt, Ys, Yt, Sigma_s, Sigma_t

def normalize_selection(selection: Any) -> tuple[int, ...]:
    """Normalize common selection outputs into a comparable tuple."""
    arr = np.asarray(selection, dtype=int).reshape(-1)
    return tuple(int(v) for v in arr.tolist())


def interval_union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Merge sorted or unsorted intervals."""
    if not intervals:
        return []

    sorted_intervals = sorted(intervals, key=lambda item: item[0])
    result = [sorted_intervals[0]]
    for left, right in sorted_intervals[1:]:
        prev_left, prev_right = result[-1]
        if left <= prev_right:
            result[-1] = (prev_left, max(prev_right, right))
        else:
            result.append((left, right))
    return result
def interval_intersection(intervals1: list[tuple[float, float]], intervals2: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Compute the intersection of two lists of intervals."""
    result = []
    i, j = 0, 0
    while i < len(intervals1) and j < len(intervals2):
        a_start, a_end = intervals1[i]
        b_start, b_end = intervals2[j]

        # Find the intersection of the two intervals
        start = max(a_start, b_start)
        end = min(a_end, b_end)

        if start <= end:
            result.append((start, end))

        # Move to the next interval in the list that ends first
        if a_end < b_end:
            i += 1
        else:
            j += 1

    return result
def solve_linear_inequality(b, c):
    """Solve the linear inequalities ``b_i*x + c_i <= 0`` (vectorized).

    Coefficients with ``|v| < 1e-12`` are treated as zero, exactly like the
    previous scalar loop; ``b == 0`` with ``c > 0`` is infeasible.  Returns
    the single containing interval ``[(left, right)]`` or ``[]``.
    """
    b = np.asarray(b, dtype=float).reshape(-1)
    c = np.asarray(c, dtype=float).reshape(-1)
    b = np.where(np.abs(b) < 1e-12, 0.0, b)
    c = np.where(np.abs(c) < 1e-12, 0.0, c)
    if np.any((b == 0) & (c > 0)):
        raise ValueError("No solution for the linear inequality")
    pos = b > 0
    neg = b < 0
    right = float(np.min(-c[pos] / b[pos])) if np.any(pos) else np.inf
    left = float(np.max(-c[neg] / b[neg])) if np.any(neg) else -np.inf
    if left > right:
        return []
    return [(left, right)]

def solve_quadratic_inequality(a, b, c):
    """ ax^2 + bx +c <= 0 """
    # a, b, c = np.round(a, 8), np.round(b, 8), np.round(c, 8)
    if abs(a) < 1e-9:
        a = 0
    if abs(b) < 1e-9:
        b = 0
    if abs(c) < 1e-9:
        c = 0
    if a == 0:
        # print(f"b: {b}")
        if b > 0:
            return [(-np.inf, -c / b)]
            # return [(-np.inf, np.around(-c / b, 8))]
        elif b == 0:
            # print(f"c: {c}")
            if c <= 0:
                return [(-np.inf, np.inf)]
            else:
                print('Error bx + c')
                return []
        else:
            return [(-c / b, np.inf)]
            # return [(np.around(-c / b, 8), np.inf)]
    delta = b*b - 4*a*c
    if delta < 0:
        if a < 0:
            return [(-np.inf, np.inf)]
        else:
            print("Error to find interval. ")
            return []

    x1 = (- b - np.sqrt(delta)) / (2.0*a)
    x2 = (- b + np.sqrt(delta)) / (2.0*a)

    # x1 = np.around(x1, 8)
    # x2 = np.around(x2, 8)
    if a < 0:
        return [(-np.inf, x2),(x1, np.inf)]
    return [(x1,x2)]
def solve_quadratic_roots(a, b, c, current_z, goal, direction=1):
    """
    Tìm nghiệm thực nhỏ nhất của az^2 + bz + c = 0 trong khoảng (current_z, goal] nếu direction = 1
    Tìm nghiệm thực lớn nhất của az^2 + bz + c = 0 trong khoảng [goal, current_z) nếu direction = -1
    """
    eps = 1e-7
    roots = []
    
    # 1. Giải phương trình
    if abs(a) < 1e-8:  # Tuyến tính: bz + c = 0
        if abs(b) > 1e-8:
            roots.append(-c / b)
    else:  # Bậc hai
        delta = b**2 - 4 * a * c
        if delta >= 0:
            sqrt_delta = np.sqrt(delta)
            roots.append((-b + sqrt_delta) / (2 * a))
            roots.append((-b - sqrt_delta) / (2 * a))
            
    # 2. Lọc nghiệm theo hướng di chuyển (direction)
    if direction == 1:
        # Khoảng (current_z, goal] -> lấy min
        valid_roots = [r for r in roots if r > current_z + eps and r <= goal + eps]
        return min(valid_roots) if valid_roots else None
        
    elif direction == -1:
        # Khoảng [goal, current_z) -> lấy max
        valid_roots = [r for r in roots if r >= goal - eps and r < current_z - eps]
        return max(valid_roots) if valid_roots else None
        
    else:
        raise ValueError("Direction phải là 1 hoặc -1")
# def solve_quadratic_roots(a, b, c, current_z, z_max):
#     """Find the smallest real root of a*z^2 + b*z + c = 0 in the interval (current_z, z_max]."""
#     eps = 1e-7
#     roots = []
    
#     if abs(a) < 1e-10: # Linear: bz + c = 0
#         if abs(b) > 1e-10:
#             roots.append(-c / b)
#     else: # Quadratic
#         delta = b**2 - 4*a*c
#         if delta >= 0:
#             sqrt_delta = np.sqrt(delta)
#             roots.append((-b + sqrt_delta) / (2*a))
#             roots.append((-b - sqrt_delta) / (2*a))
            
#     valid_roots = [r for r in roots if r > current_z + eps and r <= z_max + eps]
#     return min(valid_roots) if valid_roots else None


def solve_quad_list(p, q, r, z_debug=None):
    """Intersection of ``p_i*x^2 + q_i*x + r_i <= 0`` over vector inputs.

    Vectorized replacement of the previous per-inequality Python loop; the
    per-inequality conventions match ``solve_quadratic_inequality``
    exactly (1e-9 coefficient cutoffs, roots
    ``x1 = (-b - sqrt(delta)) / (2a)``, ``x2 = (-b + sqrt(delta)) / (2a)``).

    Every concave solution set ``(-inf, x2] U [x1, inf)`` is handled as the
    complement of the open hole ``(x2, x1)``, so the whole intersection is
    one interval minus a set of open holes, subtracted with closed-piece
    arithmetic.  Output is the same sorted closed-piece list as before.
    """
    p = np.asarray(p, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    r = np.asarray(r, dtype=float).reshape(-1)
    if p.size == 0:
        return [(-np.inf, np.inf)]

    a = np.where(np.abs(p) < 1e-9, 0.0, p)
    b = np.where(np.abs(q) < 1e-9, 0.0, q)
    c = np.where(np.abs(r) < 1e-9, 0.0, r)

    if z_debug is not None:
        for pi_, qi_, ri_ in zip(p, q, r):
            itv = solve_quadratic_inequality(float(pi_), float(qi_), float(ri_))
            val = float(pi_) * z_debug * z_debug + float(qi_) * z_debug + float(ri_)
            contains = any(lo <= z_debug <= hi for lo, hi in itv)
            print(
                f"a={float(pi_):.18e}, "
                f"b={float(qi_):.18e}, "
                f"c={float(ri_):.18e}"
            )
            print(
                f"    f(z)={val:.18e}, "
                f"interval={itv}, "
                f"contains_z={contains}"
            )

    lin = a == 0.0
    if np.any(lin & (b == 0.0) & (c > 0.0)):
        print('Error bx + c')
        print('Error in solving quadratic ineq')
        return []

    quad = ~lin
    delta = b * b - 4.0 * a * c
    real = delta >= 0.0
    if np.any(quad & ~real & (a > 0.0)):
        print("Error to find interval. ")
        print('Error in solving quadratic ineq')
        return []

    lo_list, hi_list = [], []
    m = lin & (b > 0.0)                      # (-inf, -c/b]
    if m.any():
        lo_list.append(np.full(int(m.sum()), -np.inf))
        hi_list.append(-c[m] / b[m])
    m = lin & (b < 0.0)                      # [-c/b, inf)
    if m.any():
        lo_list.append(-c[m] / b[m])
        hi_list.append(np.full(int(m.sum()), np.inf))
    qi = np.flatnonzero(quad & real & (a > 0.0))   # [x1, x2]
    if qi.size:
        av, bv, cv = a[qi], b[qi], c[qi]
        sq = np.sqrt(bv * bv - 4.0 * av * cv)
        lo_list.append((-bv - sq) / (2.0 * av))
        hi_list.append((-bv + sq) / (2.0 * av))

    left = float(np.max(np.concatenate(lo_list))) if lo_list else -np.inf
    right = float(np.min(np.concatenate(hi_list))) if hi_list else np.inf
    if left > right:
        print('Error intersection')
        return []

    hi_ = np.flatnonzero(quad & real & (a < 0.0))  # complement of open hole
    if hi_.size:
        av, bv, cv = a[hi_], b[hi_], c[hi_]
        sq = np.sqrt(bv * bv - 4.0 * av * cv)
        x1 = (-bv - sq) / (2.0 * av)
        x2 = (-bv + sq) / (2.0 * av)
        order = np.argsort(x2, kind="stable")
        pieces_l, pieces_r = np.array([left]), np.array([right])
        for hl, hr in zip(x2[order], x1[order]):
            hl, hr = float(hl), float(hr)
            if hl == hr:                     # empty hole: no constraint
                continue
            keep = ~((pieces_l < hr) & (pieces_r > hl))
            split = (pieces_l < hr) & (pieces_r > hl)
            new_l, new_r = pieces_l[keep], pieces_r[keep]
            if split.any():
                ls, rs = pieces_l[split], pieces_r[split]
                ok_l = ls <= hl              # [ls, hl]
                ok_r = rs >= hr              # [hr, rs]
                new_l = np.concatenate([new_l, ls[ok_l], np.full(int(ok_r.sum()), hr)])
                new_r = np.concatenate([new_r, np.full(int(ok_l.sum()), hl), rs[ok_r]])
            pieces_l, pieces_r = new_l, new_r
        if pieces_l.size == 0:
            print('Error intersection')
            return []
        order = np.argsort(pieces_l, kind="stable")
        return [(float(l), float(rr))
                for l, rr in zip(pieces_l[order], pieces_r[order])]

    return [(left, right)]

def compute_p_value(
    intervals: list[tuple[float, float]],
    etaT_y: float,
    etaT_Sigma_eta: float,
    null_mean: float = 0.0,
    type: str = "two-sided",
) -> float:
    """Compute the two-sided truncated-normal p-value on ``intervals``.

    The raw scan output can contain hundreds of adjacent or overlapping
    pieces, so the union is merged first (one `mp.ncdf` pair per merged
    piece instead of per raw piece).  Every piece mass is evaluated in
    tail form so no ``1 - tiny`` subtraction is ever formed: for standard
    bounds ``z_l <= z_r``, ``Phi(z_r) - Phi(z_l)`` is computed as
    ``ncdf(-z_l) - ncdf(-z_r)`` when the piece touches the right side and
    ``ncdf(z_r) - ncdf(z_l)`` when it touches the left side.  mpmath
    returns each far tail with full ``mp.dps`` significant digits, so
    far-tail masses survive at modest precision, and the probability
    mass below/above ``etaT_y`` is accumulated directly instead of as a
    difference of near-equal CDFs (which is what made large ``mp.dps``
    necessary before).
    """
    if etaT_Sigma_eta <= 0:
        raise ValueError("etaT_Sigma_eta must be positive")

    if not intervals:
        return float("nan")

    intervals = interval_union(intervals)
    if not any(left <= etaT_y <= right for left, right in intervals):
        raise ValueError("Selection intervals must contain the observed etaT_y")

    sigma = np.sqrt(etaT_Sigma_eta)

    def _mass(z_l, z_r):
        # Phi(z_r) - Phi(z_l) in tail form (z_l <= z_r)
        if z_r <= 0:
            return mp.ncdf(z_r) - mp.ncdf(z_l)
        return mp.ncdf(-z_l) - mp.ncdf(-z_r)

    total = mp.mpf("0")
    below = mp.mpf("0")
    above = mp.mpf("0")
    z_t = (etaT_y - null_mean) / sigma
    for left, right in intervals:
        z_l = (left - null_mean) / sigma
        z_r = (right - null_mean) / sigma
        if z_t >= z_r:          # piece entirely at/below the statistic
            m = _mass(z_l, z_r)
            below += m
            total += m
        elif z_t <= z_l:        # piece entirely at/above the statistic
            m = _mass(z_l, z_r)
            above += m
            total += m
        else:                   # containing piece: split at the statistic
            mb = _mass(z_l, z_t)
            ma = _mass(z_t, z_r)
            below += mb
            above += ma
            total += mb + ma

    if total == 0:
        return float("nan")

    if type == "two-sided":
        # min(below, above) == min(cdf, 1 - cdf) without the subtraction
        return float(2 * min(below, above) / total)
    elif type == "one-sided":
        return float(below / total)

def decompose_y(y: np.ndarray, eta: np.ndarray, Sigma: np.ndarray) -> SIDecomposition:
    """Build the standard selective-inference decomposition for ``y``."""
    y_col = y.reshape(-1, 1)
    eta_col = eta.reshape(-1, 1)
    Sigma_arr = Sigma

    if Sigma_arr.shape != (y_col.shape[0], y_col.shape[0]):
        raise ValueError("Sigma must have shape (n_samples, n_samples)")

    etaT_Sigma_eta = eta_col.T.dot(Sigma_arr).dot(eta_col)

    if etaT_Sigma_eta <= 0:
        raise ValueError("etaT_Sigma_eta must be positive")

    b = Sigma_arr.dot(eta_col) / etaT_Sigma_eta
    a = (np.eye(y_col.shape[0]) - b.dot(eta_col.T)).dot(y_col)
    etaT_y = eta_col.T.dot(y_col).item()

    return SIDecomposition(
        eta=eta_col,
        etaT_y=etaT_y,
        etaT_Sigma_eta=etaT_Sigma_eta.item(),
        a=a,
        b=b,
    )
def solve_Lasso(X, Y, Lambda):
    """Solve Lasso and return an active set certified by its KKT conditions.

    The parametric scan treats the returned active set as exact: it derives a
    polyhedral region that must contain the current point.  A small duality
    gap alone is insufficient near a Lasso knot, because coordinate descent
    can stop at the all-zero model just after that model has ceased to be
    optimal.  Validate the primal KKT residual before exposing the model.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float).reshape(-1, 1)
    p = X.shape[1]
    kkt_tolerance = 1e-9 * max(1.0, float(Lambda))

    def kkt_violation(beta_hat: np.ndarray) -> float:
        beta_flat = beta_hat.reshape(-1)
        correlations = (X.T @ (Y - X @ beta_hat)).reshape(-1)
        active = beta_flat != 0.0

        active_violation = (
            np.max(np.abs(correlations[active] - Lambda * np.sign(beta_flat[active])))
            if np.any(active) else 0.0
        )
        inactive_violation = (
            np.max(np.maximum(np.abs(correlations[~active]) - Lambda, 0.0))
            if np.any(~active) else 0.0
        )
        return float(max(active_violation, inactive_violation))

    # Retry with a tighter stopping criterion when the first coordinate-
    # descent pass has a small duality gap but an incorrect support.
    last_violation = np.inf
    for tolerance in (1e-12, 1e-14):
        lasso = Lasso(
            alpha=Lambda / X.shape[0],
            fit_intercept=False,
            tol=tolerance,
            max_iter=1_000_000,
        )
        lasso.fit(X, Y)
        beta_hat = lasso.coef_.reshape(p, 1)
        last_violation = kkt_violation(beta_hat)
        if last_violation <= kkt_tolerance:
            break
    else:
        raise RuntimeError(
            "Lasso solver did not meet the KKT tolerance required for "
            f"parametric inference (violation={last_violation:.3e}, "
            f"tolerance={kkt_tolerance:.3e})."
        )

    M = np.nonzero(beta_hat)[0]
    Mc = np.nonzero(beta_hat == 0)[0]
    sM = np.sign(beta_hat[M])
    return M, Mc, sM

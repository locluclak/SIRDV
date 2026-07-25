"""Numerical helpers for selective inference."""

from __future__ import annotations

from typing import Any

import numpy as np
from mpmath import mp
from numpy.linalg import lstsq

from .si_types import SIDecomposition

mp.dps = 500


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
    """Solve the linear inequality b*x + c <= 0. b and c are vectors"""
    left = -np.inf
    right = np.inf
    for bi, ci in zip(b, c):
        if abs(bi) < 1e-8:
            bi = 0
        if abs(ci) < 1e-8:
            ci = 0
        if bi == 0:
            if ci <= 0:
                continue
            else:
                raise ValueError("No solution for the linear inequality")
        elif bi > 0:
            right = min(right, -ci / bi)
        else:
            left = max(left, -ci / bi)

    if left > right:
        return []
    return [(left, right)]

def solve_quadratic_inequality(a, b, c):
    """ ax^2 + bx +c <= 0 """
    if abs(a) < 1e-10:
        a = 0
    if abs(b) < 1e-10:
        b = 0
    if abs(c) < 1e-10:
        c = 0
    if a == 0:
        # print(f"b: {b}")
        if b > 0:
            # return [(-np.inf, -c / b)]
            return [(-np.inf, np.around(-c / b, 8))]
        elif b == 0:
            # print(f"c: {c}")
            if c <= 0:
                return [(-np.inf, np.inf)]
            else:
                print('Error bx + c')
                return []
        else:
            return [(np.around(-c / b, 8), np.inf)]
    delta = b*b - 4*a*c
    if delta < 0:
        if a < 0:
            return [(-np.inf, np.inf)]
        else:
            print("Error to find interval. ")
            return []

    x1 = (- b - np.sqrt(delta)) / (2.0*a)
    x2 = (- b + np.sqrt(delta)) / (2.0*a)

    x1 = np.around(x1, 8)
    x2 = np.around(x2, 8)
    if a < 0:
        return [(-np.inf, x2),(x1, np.inf)]
    return [(x1,x2)]
def solve_quadratic_roots(a, b, c, current_z, z_max):
    """Find the smallest real root of a*z^2 + b*z + c = 0 in the interval (current_z, z_max]."""
    eps = 1e-7
    roots = []
    
    if abs(a) < 1e-10: # Linear: bz + c = 0
        if abs(b) > 1e-10:
            roots.append(-c / b)
    else: # Quadratic
        delta = b**2 - 4*a*c
        if delta >= 0:
            sqrt_delta = np.sqrt(delta)
            roots.append((-b + sqrt_delta) / (2*a))
            roots.append((-b - sqrt_delta) / (2*a))
            
    valid_roots = [r for r in roots if r > current_z + eps and r <= z_max + eps]
    return min(valid_roots) if valid_roots else None


def solve_quad_list(p, q, r):
    """Solve a list of quadratic ax^2 + bx + c <= 0 inequalities and return their intersection."""
    if len(p) == 0:
        return [(-np.inf, np.inf)]
    intervals = [(-np.inf, np.inf)]
    for pi, qi, ri in zip(p, q, r):
        # print("iteration", pi, qi, ri)
        intervals_i = solve_quadratic_inequality(pi.item(), qi.item(), ri.item())
        if intervals_i is None:
            return []
        intervals = interval_intersection(intervals, intervals_i)
        if not intervals:
            return []
    return intervals

def compute_p_value(
    intervals: list[tuple[float, float]],
    etaT_y: float,
    etaT_Sigma_eta: float,
    null_mean: float = 0.0,
    type: str = "two-sided",
) -> float:
    """Compute the two-sided truncated-normal p-value on ``intervals``."""
    if etaT_Sigma_eta <= 0:
        raise ValueError("etaT_Sigma_eta must be positive")
    if not intervals:
        return float("nan")

    sigma = np.sqrt(etaT_Sigma_eta)
    denominator = mp.mpf("0")
    numerator = mp.mpf("0")
    print('compute_p_value', etaT_y)
    for left, right in intervals:
        cdf_left = mp.ncdf((left - null_mean) / sigma)
        cdf_right = mp.ncdf((right - null_mean) / sigma)
        if left <= etaT_y <= right:
            numerator = denominator + mp.ncdf((etaT_y - null_mean) / sigma) - cdf_left
        denominator += cdf_right - cdf_left

    if denominator == 0:
        return float("nan")

    cdf = numerator / denominator

    if type == "two-sided":
        return float(2 * min(cdf, 1 - cdf))
    elif type == "one-sided":
        return float(cdf)

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
    a = (np.identity(y_col.shape[0]) - b.dot(eta_col.T)).dot(y_col)
    etaT_y = eta_col.T.dot(y_col).item()

    return SIDecomposition(
        eta=eta_col,
        etaT_y=etaT_y,
        etaT_Sigma_eta=etaT_Sigma_eta.item(),
        a=a,
        b=b,
    )

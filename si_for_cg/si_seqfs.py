"""Forward fixed-K feature selection (seqfs) and its conditioning interval.

This module ports the greedy forward-selection path and the forward-selection
conditioning interval of the ``seqfs`` reference folder
(``seqfs/ForwardSelection.py`` and ``seqfs/overconditioning.py:interval_SFS``).
The ``seqfs`` package itself is not importable (it uses broken relative
imports), so the two functions needed here are re-implemented directly; the
out-of-scope criteria ports (AIC/BIC/Adjusted-R2) and ``interval_DA`` are
dropped.

Semantics (over-conditioning): ``interval_FS`` returns the region where the
*exact observed greedy path* stays optimal at **every** step -- for each step
the projection onto the selected set must have RSS no larger than the
projection onto (previous set + any single candidate feature).  This is the
forward-selection analogue of ``conditioning_Lasso``'s over-conditioning and
is intentionally conservative: a later ``z`` that reaches the same final
feature set through a *different* greedy path is excluded from the event.
"""

from __future__ import annotations

import numpy as np

from .si_utils import interval_intersection, solve_quadratic_inequality


def forward_selection_path(
    Y: np.ndarray,
    X: np.ndarray,
    K: int,
) -> tuple[list[list[int]], np.ndarray]:
    """Greedy forward selection of ``K`` features minimising residual RSS.

    ``Y``-first argument order mirrors ``seqfs.ForwardSelection.fixedSelection``;
    ``X`` holds the candidate design rows and ``Y`` the response column.
    Returns ``path`` with ``path[0] = []`` and ``path[step]`` the (ascending)
    feature set chosen after ``step``, together with the residual vector at
    step ``K``.

    Ties on RSS are broken by the smallest feature index (the candidate loop is
    ascending and the comparison is strict) so the observed set and the
    parametric scan agree -- this determinism is required for the conditioning
    event to contain the observed statistic.
    """
    Y = np.asarray(Y, dtype=float).reshape(-1, 1)
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    if K < 0:
        raise ValueError("K must be non-negative")
    if K > p:
        raise ValueError(f"K={K} exceeds the number of candidate features p={p}")
    if K > n:
        raise ValueError(f"K={K} exceeds the number of candidate rows n={n}")

    path: list[list[int]] = [[]]
    selection: list[int] = []
    residual = Y.copy()

    for _ in range(1, K + 1):
        best_feature: int | None = None
        best_rss = np.inf
        best_residual: np.ndarray | None = None
        for feature in range(p):
            if feature in selection:
                continue
            M = sorted(selection + [feature])
            X_M = X[:, M]
            try:
                inv = np.linalg.inv(X_M.T.dot(X_M))
            except np.linalg.LinAlgError:
                # Singular candidate design: skip it (treated as +inf RSS).
                continue
            coef = inv.dot(X_M.T).dot(Y)
            res = Y - X_M.dot(coef)
            rss = float(np.linalg.norm(res) ** 2)
            if rss < best_rss:
                best_rss = rss
                best_feature = feature
                best_residual = res
        if best_feature is None:
            raise ValueError(
                "Forward selection could not pick a feature: every candidate "
                "design submatrix is singular"
            )
        selection = sorted(selection + [best_feature])
        residual = best_residual
        path.append(list(selection))

    return path, residual


def interval_FS(
    X_: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    K: int,
    path: list[list[int]],
) -> list[tuple[float, float]]:
    """Faithful port of ``seqfs.overconditioning.interval_SFS``.

    ``Y(z) = a + b z`` is the response line over the rows of ``X_`` and
    ``path`` is the observed greedy path (``path[step]`` for step ``1..K``).
    Solves, for every step and every candidate feature, the quadratic
    inequality ``RSS_{path[step]}(z) <= RSS_{path[step-1] + [j]}(z)`` and
    intersects the resulting intervals.

    A singular ``X_jk.T @ X_jk`` makes the event degenerate; ``[]`` is
    returned so the nested loops' ``stale``/``MAX_STALE`` logic can handle it
    and the overconditioning path can treat the event as empty.
    """
    X_ = np.asarray(X_, dtype=float)
    a = np.asarray(a, dtype=float).reshape(-1, 1)
    b = np.asarray(b, dtype=float).reshape(-1, 1)
    n_sample, n_fea = X_.shape

    intervals: list[tuple[float, float]] = [(-np.inf, np.inf)]

    for step in range(1, K + 1):
        X_jk = X_[:, sorted(path[step])].copy()
        try:
            inv_jk = np.linalg.inv(X_jk.T.dot(X_jk))
        except np.linalg.LinAlgError:
            return []
        # ``P a = a - X (X^T X)^{-1} X^T a`` -- applied to the two response
        # directions only, so the n x n projector is never materialized.
        Pjk_a = a - X_jk.dot(inv_jk.dot(X_jk.T.dot(a)))
        Pjk_b = b - X_jk.dot(inv_jk.dot(X_jk.T.dot(b)))

        for j in range(n_fea):
            if j in path[step]:
                continue
            Mj = path[step - 1] + [j]
            X_j = X_[:, sorted(Mj)].copy()
            try:
                inv_j = np.linalg.inv(X_j.T.dot(X_j))
            except np.linalg.LinAlgError:
                return []
            Pj_a = a - X_j.dot(inv_j.dot(X_j.T.dot(a)))
            Pj_b = b - X_j.dot(inv_j.dot(X_j.T.dot(b)))

            g1 = (Pjk_a.T.dot(Pjk_a) - Pj_a.T.dot(Pj_a)).item()
            g2 = (
                Pjk_a.T.dot(Pjk_b) + Pjk_b.T.dot(Pjk_a)
                - Pj_a.T.dot(Pj_b) - Pj_b.T.dot(Pj_a)
            ).item()
            g3 = (Pjk_b.T.dot(Pjk_b) - Pj_b.T.dot(Pj_b)).item()

            itv = solve_quadratic_inequality(g3, g2, g1)
            intervals = interval_intersection(intervals, itv)

    return intervals


def fs_conditioning_interval(
    X_: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    z: float,
    K: int,
) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """FS analogue of ``solve_Lasso`` + ``conditioning_Lasso`` bundled together.

    Recomputes the greedy path at ``Y(z) = a + b z`` (restart-and-solve),
    returns the sorted final feature set ``M_z`` and the over-conditioning
    interval for that path.
    """
    X_ = np.asarray(X_, dtype=float)
    a_flat = np.asarray(a, dtype=float).reshape(-1)
    b_flat = np.asarray(b, dtype=float).reshape(-1)
    Y = a_flat + b_flat * z

    path, _ = forward_selection_path(Y, X_, K)
    M_z = np.array(sorted(path[-1]), dtype=int)
    interval = interval_FS(X_, a_flat, b_flat, K, path)
    return M_z, interval


__all__ = ["forward_selection_path", "interval_FS", "fs_conditioning_interval"]

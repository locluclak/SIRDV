"""Shared types for selective inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class SelectionAdapter(Protocol):
    """Protocol for algorithms that produce a selection from ``X`` and ``y``."""

    def select(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Return the algorithm selection for ``X`` and ``y``."""


@dataclass(frozen=False)
class SIConfig:
    """Configuration for approximate parametric selective inference."""
    data_valuation: str = 'lava'
    zmin: float = -20.0
    zmax: float = 20.0
    step: float = 0.00001
    atol: float = 1e-10
    # Event-driven Lasso support walk (homotopy / LARS style): solve the Lasso
    # once per merged selection span and infer the following support regions
    # from the tight KKT rows instead of re-solving at every `step`.  Falls
    # back to restart-and-solve on degenerate events.
    lasso_path: bool = True
    # Feature-selection method for the conditioning event: 'lasso' (default,
    # unchanged) or 'seqfs' (fixed-K greedy forward selection).  Under 'seqfs'
    # the Lasso homotopy walk is bypassed and restart-and-solve is used.
    fs_method: str = 'lasso'
    # Fixed number of features selected by 'seqfs'; ignored under 'lasso'.
    K: int = 5

@dataclass
class LassoInfo:
    lambda_: float = 10.0
    M: np.ndarray = None
    Mc: np.ndarray = None
    sM: np.ndarray = None


@dataclass
class SIDecomposition:
    """Representation of the line ``Y(z) = a + b z``."""

    eta: np.ndarray
    etaT_y: float
    etaT_Sigma_eta: float
    a: np.ndarray
    b: np.ndarray


@dataclass(frozen=True)
class SIResult:
    """Result of selective inference for a selection algorithm."""

    p_value: float
    intervals: list[tuple[float, float]]
    etaT_y: float
    etaT_Sigma_eta: float
    observed_selection: Any
    tested_index: int
    method: str

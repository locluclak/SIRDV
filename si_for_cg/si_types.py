"""Shared types for selective inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class SelectionAdapter(Protocol):
    """Protocol for algorithms that produce a selection from ``X`` and ``y``."""

    def select(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Return the algorithm selection for ``X`` and ``y``."""


@dataclass(frozen=True)
class SIConfig:
    """Configuration for approximate parametric selective inference."""
    data_valuation: str = 'lava'
    zmin: float = -20.0
    zmax: float = 20.0
    step: float = 0.00001
    atol: float = 1e-10

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

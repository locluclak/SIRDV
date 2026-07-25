"""Selective-inference adapter for DV-score."""

from __future__ import annotations

from dataclasses import dataclass


import numpy as np

from .lava_function import calc_lava_score
from .gram_matrix import build_gram_matrix
from .si_runner import selective_inference
from .si_types import SIConfig, SIResult
from .si_adapter import SelectionAdapter, make_eta


# from .si_adapter import SelectionAdapter


class SelectiveInference:
    """Convenience interface for p-values after DV-score instance selection."""

    def __init__(
        self,
        fraction: float = 0.5,
        zmin: float = -20.0,
        zmax: float = 20.0,
        step: float = 0.00001,
        method: str = "parametric",
    ):
        self.fraction = fraction
        self.config = SIConfig(zmin=zmin, zmax=zmax, step=step)
        self.method = method

    def fit(
        self,
        Xtrain: np.ndarray,
        ytrain: np.ndarray,
        Xval: np.ndarray,
        yval: np.ndarray,
        Sigma: np.ndarray | None = None,
        jth: int | None = None,
    ) -> "SelectiveInference":
        Xtrain_arr = np.asarray(Xtrain, dtype=np.float64)
        ytrain_arr = np.asarray(ytrain, dtype=np.float64).reshape(-1)
        Xval_arr = np.asarray(Xval, dtype=np.float64)
        yval_arr = np.asarray(yval, dtype=np.float64).reshape(-1)


        if Xtrain_arr.shape[0] != ytrain_arr.shape[0]:
            raise ValueError("Xtrain and ytrain must have the same number of samples")
        if Xval_arr.shape[0] != yval_arr.shape[0]:
            raise ValueError("Xval and yval must have the same number of samples")

        Sigma_arr = np.identity(ytrain_arr.shape[0] + yval_arr.shape[0]) if Sigma is None else np.asarray(Sigma, dtype=np.float64)

        # --- Run selection
        adapter = SelectionAdapter(fraction=self.fraction)
        observed_selection = adapter.select(Xtrain_arr, ytrain_arr, Xval_arr, yval_arr)
        print('check')
        print(adapter.select_with_known_basis(adapter.cost, adapter.basis))
        print(observed_selection)
        if len(observed_selection) == 0:
            raise ValueError("DV-score selected no samples; cannot choose a tested index")

        # --- Test statistic
        p = Xtrain_arr.shape[1]
        # print(range(p))
        tested_position = int(np.random.choice(range(p))) if jth is None else int(jth)

        # eta, c, signcondition = make_eta_c(X_train=Xtrain_arr, X_val=Xval_arr, 
        #                     y_train=ytrain_arr, y_val=yval_arr, 
        #                     S=np.array(observed_selection), j=tested_sample_index)
        eta = make_eta(Xt = Xtrain_arr, yt = ytrain_arr,
                                      Xv= Xval_arr, yv = yval_arr, 
                                      obs_selection = observed_selection,
                                      j = tested_position)
        # print("test statistic", eta.T.dot(np.concatenate((ytrain_arr, yval_arr), axis=0)))
        # print("test position", tested_position)
        # --- Run selective inference
        self.result_ = selective_inference(
            Xtrain_arr,
            Xval_arr,
            ytrain_arr,
            yval_arr,
            Sigma_arr,
            adapter,
            eta,
            tested_index=tested_position,
            observed_selection=observed_selection,
            config=self.config,
            method=self.method,
            # null_mean=c,
        )

        self.p_value_ = self.result_.p_value
        self.intervals_ = self.result_.intervals
        self.selected_indices_ = np.asarray(observed_selection, dtype=int)
        self.tested_index_ = tested_position
        return self


def SI_DV(
    Xtrain: np.ndarray,
    ytrain: np.ndarray,
    Xval: np.ndarray | None = None,
    yval: np.ndarray | None = None,
    Sigma: np.ndarray | None = None,
    fraction: float = 0.5,
    jth: int | None = None,
    zmin: float = -20.0,
    zmax: float = 20.0,
    step: float = 0.00001,
    method: str = "parametric",
) -> float:
    """Return a selective p-value for one selected DV-score sample."""
    si = SelectiveInference(
        fraction=fraction,
        zmin=zmin,
        zmax=zmax,
        step=step,
        method=method,
    ).fit(Xtrain, ytrain, Xval=Xval, yval=yval, Sigma=Sigma, jth=jth)
    return si.p_value_


__all__ = ["SelectionAdapter", "SelectiveInference", "SI_DV", "SIResult"]

"""Generic selective-inference runner."""

from __future__ import annotations

import numpy as np

from .si_overconditioning import CG_overconditioning_intervals, overconditioning_intervals
from .si_parametric import (parametric_intervals, traditional_parametric_intervals, cg_parametric_intervals, ocwithz_intervals, parametric_intervals_FS,
                            parametric_intervals_FS_nested, cg_parametric_intervals_FS, cg_parametric_intervals_FS_nested)
from .si_types import SIConfig, SIResult, SelectionAdapter, LassoInfo
from .si_utils import compute_p_value, decompose_y, interval_union
from . import param

def selective_inference(
    Xtrain: np.ndarray,
    Xval: np.ndarray,
    ytrain: np.ndarray,
    yval: np.ndarray,
    Sigma: np.ndarray,
    adapter: SelectionAdapter,
    fs: LassoInfo,
    eta: np.ndarray,
    tested_index: int,
    observed_selection,
    config: SIConfig | None = None,
    method: str = "parametric",
    null_mean: float = 0.0,
) -> SIResult:
    """Run selective inference for any adapter with a ``select`` method."""
    config = SIConfig() if config is None else config
    if config.data_valuation == 'lava':
        y = np.concatenate([ytrain, yval])
        # print(y.shape)
        # print(Sigma.shape)
        decomposition = decompose_y(y, eta, Sigma)
        if method == "parametric":
            intervals = parametric_intervals_FS(Xtrain, Xval, decomposition, 
                                                   adapter, observed_selection, config, fs)
        elif method == "parametric_nested":
            intervals = parametric_intervals_FS_nested(Xtrain, Xval, decomposition,
                                                       adapter,
                                                       config, fs)
        elif method == "overconditioning":

            intervals = overconditioning_intervals(Xtrain, Xval, decomposition, 
                                                adapter, config,observed_selection=observed_selection,fs=fs)
            # print('oc original', intervals)
            # print()
            # print('oc with specific z', ocwithz_intervals(Xtrain, Xval, decomposition, 
            #                                 adapter, observed_selection, config,z=-17.5)) # 0.28083
        else:
            raise ValueError("method must be 'parametric' or 'overconditioning'")
    if config.data_valuation == 'cg-score':
        y = ytrain
        decomposition = decompose_y(y, eta, Sigma)
        if method == "parametric":
            intervals = cg_parametric_intervals_FS(X=Xtrain, decomposition=decomposition, adapter=adapter, observed_selection=observed_selection, config=config,fs=fs)
        elif method == "parametric_nested":
            intervals = cg_parametric_intervals_FS_nested(X=Xtrain, decomposition=decomposition, adapter=adapter, config=config, fs=fs)
        elif method == "overconditioning":

            intervals = CG_overconditioning_intervals(X=Xtrain, decomposition=decomposition, adapter=adapter, config=config,observed_selection=observed_selection,fs=fs)
            # print('Final interval',intervals)
        else:
            raise ValueError("method must be 'parametric' or 'overconditioning'")
    # intervals = [(-np.inf, np.inf)]
    # print('\n\nFinal interval',intervals)
    p_value = compute_p_value(intervals, 
                              decomposition.etaT_y, decomposition.etaT_Sigma_eta, 
                              null_mean=null_mean, 
                            #   type="one-sided")
                              type="two-sided")
    return SIResult(
        p_value=p_value,
        intervals=intervals,
        etaT_y=decomposition.etaT_y,
        etaT_Sigma_eta=decomposition.etaT_Sigma_eta,
        observed_selection=observed_selection,
        tested_index=tested_index,
        method=method,
    )

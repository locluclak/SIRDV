"""Generic selective-inference runner."""

from __future__ import annotations

import numpy as np

from .si_overconditioning import CG_overconditioning_intervals, overconditioning_intervals
from .si_parametric import (parametric_intervals, traditional_parametric_intervals, cg_parametric_intervals, ocwithz_intervals, parametric_intervals_FS,
                            parametric_intervals_FS_nested, cg_parametric_intervals_FS, cg_parametric_intervals_FS_nested)
from .si_types import SIConfig, SIResult, SelectionAdapter, LassoInfo
from .si_utils import compute_p_value, decompose_y, interval_union
from . import param
from . import _timeprof as tp

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
    method: str = "parametric_nested",
    null_mean: float = 0.0,
) -> SIResult:
    """Run selective inference for any adapter with a ``select`` method."""
    config = SIConfig() if config is None else config
    if config.data_valuation == 'lava':
        y = np.concatenate([ytrain, yval])
        # print(y.shape)
        # print(Sigma.shape)
        _t_dec = tp.start("runner: decompose_y")
        decomposition = decompose_y(y, eta, Sigma)
        config.zmin = config.zmin*decomposition.etaT_Sigma_eta
        config.zmax = config.zmax*decomposition.etaT_Sigma_eta
        tp.stop(_t_dec)
        _t_itv = tp.start("runner: intervals (parametric_nested)")
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
        tp.stop(_t_itv)
    if config.data_valuation == 'cg-score':
        y = ytrain
        _t_dec = tp.start("runner: decompose_y")
        decomposition = decompose_y(y, eta, Sigma)
        print(config.zmin, config.zmax)
        config.zmin = config.zmin*np.sqrt(decomposition.etaT_Sigma_eta)
        config.zmax = config.zmax*np.sqrt(decomposition.etaT_Sigma_eta)
        print(config.zmin, config.zmax)

        tp.stop(_t_dec)
        _t_itv = tp.start("runner: intervals (parametric_nested)")
        if method == "parametric":
            intervals = cg_parametric_intervals_FS(X=Xtrain, decomposition=decomposition, adapter=adapter, observed_selection=observed_selection, config=config,fs=fs)
        elif method == "parametric_nested":
            intervals = cg_parametric_intervals_FS_nested(X=Xtrain, decomposition=decomposition, adapter=adapter, config=config, fs=fs)
        elif method == "overconditioning":

            intervals = CG_overconditioning_intervals(X=Xtrain, decomposition=decomposition, adapter=adapter, config=config,observed_selection=observed_selection,fs=fs)
            # print('Final interval',intervals)
        else:
            raise ValueError("method must be 'parametric' or 'overconditioning'")
        tp.stop(_t_itv)
    # intervals = [(-np.inf, np.inf)]
    # print('\n\nFinal interval',intervals)
    _t_pv = tp.start("runner: compute_p_value")

    print('Interval: ',intervals)
    p_value = compute_p_value(intervals, 
                              decomposition.etaT_y, decomposition.etaT_Sigma_eta, 
                              null_mean=null_mean, 
                            #   type="one-sided")
                              type="two-sided")
    tp.stop(_t_pv)
    return SIResult(
        p_value=p_value,
        intervals=intervals,
        etaT_y=decomposition.etaT_y,
        etaT_Sigma_eta=decomposition.etaT_Sigma_eta,
        observed_selection=observed_selection,
        tested_index=tested_index,
        method=method,
    )

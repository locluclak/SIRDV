"""Generic selective-inference runner."""

from __future__ import annotations

import numpy as np

from .si_overconditioning import CG_overconditioning_intervals, overconditioning_intervals
from .si_parametric import parametric_intervals, traditional_parametric_intervals, cg_parametric_intervals, ocwithz_intervals
from .si_types import SIConfig, SIResult, SelectionAdapter
from .si_utils import compute_p_value, decompose_y, interval_union
from . import param

def selective_inference(
    Xtrain: np.ndarray,
    Xval: np.ndarray,
    ytrain: np.ndarray,
    yval: np.ndarray,
    Sigma: np.ndarray,
    adapter: SelectionAdapter,
    eta: np.ndarray,
    tested_index: int,
    observed_selection,
    config: SIConfig | None = None,
    method: str = "parametric",
    null_mean: float = 0.0,
) -> SIResult:
    """Run selective inference for any adapter with a ``select`` method."""
    config = SIConfig() if config is None else config
    # observed_selection = adapter.select(Xtrain, ytrain, Xval, yval)

    # print('check y', np.sum(np.abs(y - (decomposition.a + decomposition.b*decomposition.etaT_y).reshape(-1))))
    if config.data_valuation == 'lava':
        y = np.concatenate([ytrain, yval])
        decomposition = decompose_y(y, eta, Sigma)
        if method == "parametric":
            # intervals_left = traditional_parametric_intervals(Xtrain, Xval, decomposition, 
            #                                             adapter, observed_selection, config, direction=-1)
            # intervals_right = traditional_parametric_intervals(Xtrain, Xval, decomposition, 
            #                                                         adapter, observed_selection, config, direction=1)
            # intervals = interval_union(intervals_left+intervals_right)
            # print('interval traditional', intervals)

            # print('------------------------------------')
            intervals = parametric_intervals(Xtrain, Xval, decomposition, 
                                                   adapter, observed_selection, config,)
            # print('interval basis tracking', intervals)

            # nt, nv = Xtrain.shape[0], Xval.shape[0]
            # S_ = adapter.S_.copy()
            # intervals = param.parametric(nt, nv, decomposition.a.copy(), decomposition.b.copy(),
            #                              np.vstack((Xtrain, Xval)), S_, adapter.h_.copy(), observed_selection, adapter.fraction, config)
        elif method == "overconditioning":

            intervals = overconditioning_intervals(Xtrain, Xval, decomposition, 
                                                adapter, config,)
            print('oc original', intervals)
            print()
            print('oc with specific z', ocwithz_intervals(Xtrain, Xval, decomposition, 
                                            adapter, observed_selection, config,z=-17.5)) # 0.28083
        else:
            raise ValueError("method must be 'parametric' or 'overconditioning'")
    if config.data_valuation == 'cg-score':
        y = ytrain
        decomposition = decompose_y(y, eta, Sigma)
        if method == "parametric":
            intervals = cg_parametric_intervals(X=Xtrain, decomposition=decomposition, adapter=adapter, observed_selection=observed_selection, config=config)
        elif method == "overconditioning":

            intervals = CG_overconditioning_intervals(X=Xtrain, decomposition=decomposition, adapter=adapter, config=config)
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

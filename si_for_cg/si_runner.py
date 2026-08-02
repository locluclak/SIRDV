"""Generic selective-inference runner."""

from __future__ import annotations

import numpy as np

from .si_overconditioning import overconditioning_intervals
from .si_parametric import parametric_intervals, traditional_parametric_intervals
from .si_types import SIConfig, SIResult, SelectionAdapter
from .si_utils import compute_p_value, decompose_y
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
    y = np.concatenate([ytrain, yval])
    decomposition = decompose_y(y, eta, Sigma)
    # print('check y', np.sum(np.abs(y - (decomposition.a + decomposition.b*decomposition.etaT_y).reshape(-1))))
    if method == "parametric":
        intervals = traditional_parametric_intervals(Xtrain, Xval, decomposition, 
                                         adapter, observed_selection, config,)
        # print('------------------------------------')
        # print('new',parametric_intervals(Xtrain, Xval, decomposition, 
        #                                        adapter, observed_selection, config,))
        # nt, nv = Xtrain.shape[0], Xval.shape[0]
        # S_ = adapter.S_.copy()
        # intervals = param.parametric(nt, nv, decomposition.a.copy(), decomposition.b.copy(),
        #                              np.vstack((Xtrain, Xval)), S_, adapter.h_.copy(), observed_selection, adapter.fraction, config)
    elif method == "overconditioning":
        intervals = overconditioning_intervals(Xtrain, Xval, decomposition, 
                                               adapter, config,)
    else:
        raise ValueError("method must be 'parametric' or 'overconditioning'")
    # intervals = [(-np.inf, np.inf)]
    print('Final interval',intervals)
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

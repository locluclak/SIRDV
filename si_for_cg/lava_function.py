from __future__ import annotations
import numpy as np
from . import OptimalTransport


def calc_lava_score(Xtrain: np.ndarray, ytrain: np.ndarray,
                    Xval: np.ndarray, yval: np.ndarray, return_complement = False)  -> np.ndarray:
    """
    Compute DV-score for the whole training set

    Args:
        Xtrain: Training input features, shape (n_train, d).
        ytrain: Training continuous targets, shape (n_train,).
        Xval: Validation input features, shape (n_val, d).
        yval: Validation continuous targets, shape (n_val,).
        return_complement: If True, also return the complement of the LAVA score.

    Returns:
        scores: DV-score for each sample, shape (n,).
        complement: Complement of the LAVA score, constraints of LP problem. Only returned if `return_complement` is True.
    """
    nt = len(Xtrain)
    nv = len(Xval)

    # Stack data
    Xt_ = np.hstack((Xtrain, ytrain.reshape(-1, 1)))
    Xv_ = np.hstack((Xval, yval.reshape(-1, 1)))
    XtXv_ = np.vstack((Xt_, Xv_))

    # OT via LP
    h = np.concatenate((np.ones((nt, 1))/nt, np.ones((nv,1))/nv), axis = 0) 
    S = OptimalTransport.convert(nt, nv)
    
    # remove last row
    S_ = S[:-1].copy()
    h_ = h[:-1].copy()
    
    # Gamma drives source data to target data 
    OTresult = OptimalTransport.solveOT(nt, nv, S_, h_, XtXv_)
    # GAMMA = OTresult['gamma']
    Basis = OTresult['basis']
    cost = OTresult['cost']
    
    # LAVA = d OT(mu_t, mu_v) / d mu_ (z_i) 
    # S_train = S_[:nt]
    S_basis = S_[:, Basis]
    # print("cost lava",np.sum(cost_vec))
    cost_vec = cost[Basis]
    scores = -1*np.dot(np.linalg.inv(S_basis), cost_vec)
    # print('nt',nt)

    scores = scores[:nt]
    
    scores = scores.reshape(-1)
    scores = 2*scores - np.sum(scores)

    # print('sc',scores)
    if return_complement:
        complement = dict()
        complement['S_'] = S_
        complement['h_'] = h_
        complement['Basis'] = Basis
        complement['cost'] =cost
        return scores, complement
    
    return scores
    


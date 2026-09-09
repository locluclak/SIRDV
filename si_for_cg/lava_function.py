from __future__ import annotations
import numpy as np
from . import OptimalTransport


# def calc_lava_score(Xtrain: np.ndarray, ytrain: np.ndarray,
#                     Xval: np.ndarray, yval: np.ndarray, return_complement = False,
#                     basis_hint=None)  -> np.ndarray:
#     """
#     Compute DV-score for the whole training set

#     Args:
#         Xtrain: Training input features, shape (n_train, d).
#         ytrain: Training continuous targets, shape (n_train,).
#         Xval: Validation input features, shape (n_val, d).
#         yval: Validation continuous targets, shape (n_val,).
#         return_complement: If True, also return the complement of the LAVA score.

#     Returns:
#         scores: DV-score for each sample, shape (n,).
#         complement: Complement of the LAVA score, constraints of LP problem. Only returned if `return_complement` is True.
#     """
#     nt = len(Xtrain)
#     nv = len(Xval)

#     # Stack data
#     Xt_ = np.hstack((Xtrain, ytrain.reshape(-1, 1)))
#     Xv_ = np.hstack((Xval, yval.reshape(-1, 1)))
#     XtXv_ = np.vstack((Xt_, Xv_))

#     # OT via LP
#     h = np.concatenate((np.ones((nt, 1))/nt, np.ones((nv,1))/nv), axis = 0) 
#     S = OptimalTransport.convert(nt, nv)
    
#     # remove last row
#     S_ = S[:-1]
#     h_ = h[:-1]

#     # Solve Linear Programming 
#     OTresult = OptimalTransport.solveOT(nt, nv, S_, h_, XtXv_, basis_hint=basis_hint)
#     Basis = OTresult['basis']
#     cost = OTresult['cost']
    
#     S_basis = S_[:, Basis]
#     cost_vec = cost[Basis]

#     scores = -np.linalg.solve(S_basis.T, cost_vec)
#     scores = scores[:nt].copy()
#     scores = scores.reshape(-1)
#     scores = 2*scores - np.sum(scores)

#     # print('sc',scores)
#     if return_complement:
#         complement = dict()
#         complement['S_'] = S_
#         complement['h_'] = h_
#         complement['Basis'] = Basis
#         complement['cost'] =cost
#         return scores, complement
    
#     return scores
    
# import numpy as np

def calc_lava_score(Xtrain: np.ndarray, ytrain: np.ndarray,
                    Xval: np.ndarray, yval: np.ndarray, return_complement = False,
                    basis_hint=None,) -> np.ndarray:
    """
    Compute LAVA (calibrated gradient) scores for training set by recovering dual
    potentials from an OT LP basis and applying the repo calibration.

    Notes:
    - OptimalTransport must provide convert() and solveOT() as in your original code.
    - The function returns `scores` shaped (nt,). If return_complement is True, a dict
      with the LP complement info is returned as second output (same as your original).
    """
    nt = len(Xtrain)
    nv = len(Xval)

    # Stack features+targets the same way you did
    Xt_ = np.hstack((Xtrain, ytrain.reshape(-1, 1)))
    Xv_ = np.hstack((Xval, yval.reshape(-1, 1)))
    XtXv_ = np.vstack((Xt_, Xv_))

    # OT via LP
    h = np.concatenate((np.ones((nt, 1))/nt, np.ones((nv,1))/nv), axis = 0) 
    S = OptimalTransport.convert(nt, nv)

    # remove last row (degeneracy fixing as before)
    S_ = S[:-1]
    h_ = h[:-1]

    # Solve LP and recover basis & cost
    OTresult = OptimalTransport.solveOT(nt, nv, S_, h_, XtXv_, basis_hint=basis_hint)
    Basis = OTresult['basis']
    cost = OTresult['cost']

    S_basis = S_[:, Basis]
    cost_vec = cost[Basis]

    # solve for dual variables lambda: S_basis.T @ lambda = cost_vec
    lambda_vec = np.linalg.solve(S_basis.T, cost_vec)

    # In many implementations the dual potentials f are minus lambda entries;
    # your original code used a minus sign. Keep it but allow a quick check:
    f = -lambda_vec

    # Extract the training potentials (first nt entries)
    f_train = f[:nt].reshape(-1)

    # Calibrated gradient (LAVA value)
    coeff = 1.0 + 1.0 / (nt - 1)       # = N/(N-1)
    values = coeff * f_train - np.sum(f_train) / (nt - 1)

    if return_complement:
        complement = dict()
        complement['S_'] = S_
        complement['h_'] = h_
        complement['Basis'] = Basis
        complement['cost'] = cost
        return values, complement

    return values
"""DV-score selection adapter."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lava_function import calc_lava_score
from .cg_function import calc_cg_score

@dataclass
class CGSelectionAdapter:
    """Adapter exposing CG-score as a generic selection algorithm."""

    fraction: float = 0.5
    sorted_indices: np.ndarray | None = None
    top_indices: np.ndarray | None = None
    invH: np.ndarray | None = None

    def select(self, X: np.ndarray, y: np.ndarray) -> tuple[int, ...]:
        if self.invH is None:
            scores, H, invH = calc_cg_score(X, y, normalize=False, return_gram=True)
            self.invH = invH
        else:
            invH = self.invH
            scores = calc_cg_score(X, y, normalize=False, return_gram=False, invH=invH)
        if self.fraction <= 1:
            n_keep = max(1, int(len(scores) * self.fraction))
        else:
            n_keep = self.fraction
        self.sorted_indices = np.argsort(scores)
        self.top_indices = self.sorted_indices[-n_keep:]

        return list(int(idx) for idx in sorted(self.top_indices))

@dataclass
class SelectionAdapter:
    """Adapter exposing DV-score as a generic selection algorithm."""

    fraction: float = 0.5
    sorted_indices: np.ndarray | None = None
    top_indices: np.ndarray | None = None
    S_: np.ndarray | None = None
    h_: np.ndarray | None = None
    basis: list[int] | None = None
    nt: int | None = None
    nv: int | None = None
    Xtrain: np.ndarray | None = None
    ytrain: np.ndarray | None = None
    Xval: np.ndarray | None = None
    yval: np.ndarray | None = None

    def select(self, Xtrain: np.ndarray, ytrain: np.ndarray,
               Xval: np.ndarray, yval: np.ndarray, basis_hint=None) -> list[int]:
        scores, complement = calc_lava_score(
            Xtrain, ytrain, Xval, yval, return_complement=True,
            basis_hint=basis_hint,
        )
        self.S_ = complement['S_']
        self.h_ = complement['h_']
        self.basis = complement['Basis']
        self.cost = complement['cost']
        self.nt = Xtrain.shape[0]
        self.nv = Xval.shape[0]

        # self.Xtrain = Xtrain
        # self.ytrain = ytrain
        # self.Xval = Xval
        # self.yval = yval
        if self.fraction <= 1:
            n_keep = max(1, int(len(scores) * self.fraction))
        else:
            n_keep = self.fraction
        self.sorted_indices = np.argsort(scores, stable=True)
        self.top_indices = self.sorted_indices[-n_keep:]

        return list(int(idx) for idx in sorted(self.top_indices))
    
    def select_with_known_basis(self, nt, S_, cost_vec: np.ndarray, Basis: list[int]) -> list[int]:

        S_basis = S_[:, Basis]
        cost_vec = cost_vec[Basis]
        scores = -np.linalg.solve(S_basis.T, cost_vec)
        # scores = -1*np.dot(np.linalg.inv(S_basis.T), cost_vec)
        # print('nt',nt)
        
        scores = scores[:nt]
        scores = scores.reshape(-1)
        scores = 2*scores - np.sum(scores)

        n_keep = max(1, int(len(scores) * self.fraction))
        sorted_indices = np.argsort(scores, stable=True)
        top_indices = sorted_indices[-n_keep:]

        return list(int(idx) for idx in sorted(top_indices)), sorted_indices


def make_eta_cg(X: np.ndarray, y: np.ndarray,
             obs_selection: list[int],
             j: int) -> np.ndarray:
    """Make eta vector for selective inference."""
    n, p = X.shape

    obs_selection = np.asarray(obs_selection, dtype=int)
    X_prunned = X[obs_selection].copy()
    
    trainpart = np.linalg.inv(X_prunned.T.dot(X_prunned)).dot(X_prunned.T)
    # print(trainpart.shape, valpart.shape)
    ej = np.zeros(p)
    ej[j] = 1
    eta = (ej.dot(trainpart.dot(np.eye(n)[obs_selection]))).T

    return eta

def make_eta(Xt: np.ndarray, yt: np.ndarray,
             Xv: np.ndarray, yv: np.ndarray,
             obs_selection: list[int],
             j: int) -> np.ndarray:
    """Make eta vector for selective inference."""
    nt, p = Xt.shape
    nv = Xv.shape[0]

    obs_selection = np.asarray(obs_selection, dtype=int)
    Xt_prunned = Xt[obs_selection].copy()
    
    trainpart = np.linalg.inv(Xt_prunned.T.dot(Xt_prunned)).dot(Xt_prunned.T)
    valpart = np.zeros((p, nv))
    # print(trainpart.shape, valpart.shape)
    ej = np.zeros(p)
    ej[j] = 1
    eta = (ej.dot(np.hstack((trainpart.dot(np.eye(nt)[obs_selection]), 
                             valpart)))).T

    return eta


def make_etaFS(Xt: np.ndarray, yt: np.ndarray,
             Xv: np.ndarray, yv: np.ndarray,
             M, 
             j: int) -> np.ndarray:
    """Make eta vector for selective inference."""
    nt, p = Xt.shape
    nv = Xv.shape[0]

    # obs_selection = np.asarray(obs_selection, dtype=int)
    Xt_prunned = Xt[:, M].copy()
    
    trainpart = np.linalg.inv(Xt_prunned.T.dot(Xt_prunned)).dot(Xt_prunned.T)
    valpart = np.zeros((p, nv))[M]
    # print(trainpart.shape, valpart.shape)
    ej = np.zeros(len(M))
    ej[j] = 1
    eta = (ej.dot(np.hstack((trainpart, valpart)))).T

    return eta
__all__ = ["SelectionAdapter", "CGSelectionAdapter"]

def make_eta_cgFS(X: np.ndarray, y: np.ndarray,
            #  obs_selection: list[int],
             M,
             j: int) -> np.ndarray:
    """Make eta vector for selective inference."""
    n, p = X.shape

    X_prunned = X[:, M].copy()
    
    trainpart = np.linalg.inv(X_prunned.T.dot(X_prunned)).dot(X_prunned.T)
    # print(trainpart.shape, valpart.shape)
    ej = np.zeros(len(M))
    ej[j] = 1
    eta = (ej.dot(trainpart)).T

    return eta
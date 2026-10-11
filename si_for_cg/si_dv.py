"""Selective-inference adapter for DV-score."""

from __future__ import annotations

from dataclasses import dataclass


import numpy as np
from sklearn.model_selection import train_test_split
from .lava_function import calc_lava_score
from .gram_matrix import build_gram_matrix
from .si_runner import selective_inference
from .si_types import SIConfig, SIResult, LassoInfo
from .si_adapter import CGSelectionAdapter, SelectionAdapter, make_eta, make_eta_cg, make_etaFS, make_eta_cgFS, make_naive_etaFS
from .si_utils import compute_p_value, decompose_y, solve_Lasso
from sklearn.linear_model import Lasso
from scipy.linalg import block_diag
import time
from . import _timeprof as tp
# from .si_adapter import SelectionAdapter


class DataSplittingTest:
    def __init__(
            self,
            fraction,
            data_valuation,
            lambda_ = 10,
            split_ratio=0.3,
    ):
        self.fraction = fraction
        self.data_valuation = data_valuation
        self.lambda_ = lambda_
        self.split_ratio = split_ratio
    def fit(
        self,
        Xtrain: np.ndarray,
        ytrain: np.ndarray,
        Xval: np.ndarray,
        yval: np.ndarray,
        Sigmat: np.ndarray | None = None,
        Sigmav: np.ndarray | None = None,
        jth: int | None = None,
    ) -> "DataSplittingTest":
        Xtrain_arr = np.asarray(Xtrain)
        ytrain_arr = np.asarray(ytrain).reshape(-1)

        # 2. Generate indices for slicing the variance matrix
        indices = np.arange(len(Xtrain_arr))

        # 3. Split features, targets, AND indices together
        Xtrain_arr, X_test, ytrain_arr, y_test, train_idx, test_idx = train_test_split(
            Xtrain_arr, 
            ytrain_arr, 
            indices,
            test_size=self.split_ratio
        )
        Sigmat = np.eye(ytrain.shape[0]) if Sigmat is None else np.asarray(Sigmat)
        Sigmav = np.eye(yval.shape[0]) if Sigmav is None else np.asarray(Sigmav)

        # 4. Crop the original variance matrix (Sigma) to match new train/test dimensions
        Sigma_train = Sigmat[np.ix_(train_idx, train_idx)]
        Sigma_test  = Sigmat[np.ix_(test_idx, test_idx)]
        if Xtrain_arr.shape[0] != ytrain_arr.shape[0]:
            raise ValueError("Xtrain and ytrain must have the same number of samples")

        # --- Run selection
        if self.data_valuation == 'lava':
            Xval_arr = np.asarray(Xval)
            yval_arr = np.asarray(yval).reshape(-1)
            if Xval_arr.shape[0] != yval_arr.shape[0]:
                raise ValueError("Xval and yval must have the same number of samples")
            # Sigma_arr = np.eye(ytrain_arr.shape[0] + yval_arr.shape[0]) if Sigma is None else np.asarray(Sigma)
            Sigma_arr = np.eye(y_test.shape[0]) if Sigma_test is None else np.asarray(Sigma_test)
            
            adapter = SelectionAdapter(fraction=self.fraction)
            observed_selection = adapter.select(Xtrain_arr, ytrain_arr, Xval_arr, yval_arr)
        elif self.data_valuation == 'cg-score':
            Sigma_arr = np.eye(y_test.shape[0]) if Sigma_test is None else np.asarray(Sigma_test)
            adapter = CGSelectionAdapter(fraction=self.fraction)
            observed_selection = adapter.select(Xtrain_arr, ytrain_arr)

        # print(observed_selection)
        X_pruned = Xtrain_arr[observed_selection].copy()
        lasso = Lasso(alpha = self.lambda_ / X_pruned.shape[0], fit_intercept=False, tol = 1e-10, max_iter=100000000)
        y_pruned = ytrain_arr[observed_selection]
        lasso.fit(X_pruned, y_pruned)
        # Sigma_arr_pruned = Sigma_arr[np.ix_(observed_selection, observed_selection)]
        p = Xtrain_arr.shape[1]

        beta_hat = lasso.coef_.reshape(p, 1)
        M = np.nonzero(beta_hat)[0]


        if len(observed_selection) == 0:
            raise ValueError("DV-score selected no samples; cannot choose a tested index")
        if len(M) == 0:
            raise ValueError("Empty feature selection set")
        # --- Test statistic
        p = len(M)
        # print(range(p))
        tested_position = int(np.random.choice(range(p))) if jth is None else int(jth)


        if self.data_valuation == 'lava':

            # print("test statistic", eta.T.dot(np.concatenate((ytrain_arr, yval_arr), axis=0)))
            # y = np.concatenate([ytrain, yval])
            # X = np.vstack([Xtrain_arr, Xval_arr])
            eta = make_naive_etaFS(X_pruned= X_test, y= y_test,
                            # obs_selection = observed_selection,
                            M = M,
                            j = tested_position)
            decomposition = decompose_y(y_test, eta, Sigma_arr)   
        elif self.data_valuation == 'cg-score':
            
            eta = make_eta_cgFS(X= X_test, y= y_test,
                                        # obs_selection = observed_selection,
                                        M = M,
                                        j = tested_position)
            y = ytrain
            decomposition = decompose_y(y_test, eta, Sigma_arr)
        intervals = [(-np.inf, np.inf)]
        p_value = compute_p_value(intervals, 
                            decomposition.etaT_y, decomposition.etaT_Sigma_eta,  
                            type="two-sided")

        self.p_value_ = p_value
        self.selected_indices_ = np.asarray(observed_selection, dtype=int)
        self.tested_index_ = tested_position
        return self.p_value_

class FSNaiveTest:
    def __init__(
            self,
            fraction,
            data_valuation,
            lambda_=10,
            ):
        self.fraction = fraction
        self.data_valuation = data_valuation
        self.lambda_ = lambda_
    def fit(
        self,
        Xtrain: np.ndarray,
        ytrain: np.ndarray,
        Xval: np.ndarray,
        yval: np.ndarray,
        Sigmat: np.ndarray | None = None,
        Sigmav: np.ndarray | None = None,
        jth: int | None = None,
    ) -> "FSNaiveTest":
        Xtrain_arr = np.asarray(Xtrain)
        ytrain_arr = np.asarray(ytrain).reshape(-1)
        Sigma = block_diag(Sigmat, Sigmav)
        if Xtrain_arr.shape[0] != ytrain_arr.shape[0]:
            raise ValueError("Xtrain and ytrain must have the same number of samples")

        # --- Run selection
        if self.data_valuation == 'lava':
            Xval_arr = np.asarray(Xval)
            yval_arr = np.asarray(yval).reshape(-1)
            if Xval_arr.shape[0] != yval_arr.shape[0]:
                raise ValueError("Xval and yval must have the same number of samples")
            # Sigma_arr = np.eye(ytrain_arr.shape[0] + yval_arr.shape[0]) if Sigma is None else np.asarray(Sigma)
            Sigma_arr = np.eye(ytrain_arr.shape[0]) if Sigmat is None else np.asarray(Sigmat)
            
            adapter = SelectionAdapter(fraction=self.fraction)
            observed_selection = adapter.select(Xtrain_arr, ytrain_arr, Xval_arr, yval_arr)
        elif self.data_valuation == 'cg-score':
            Sigma_arr = np.eye(ytrain_arr.shape[0]) if Sigmat is None else np.asarray(Sigmat)
            adapter = CGSelectionAdapter(fraction=self.fraction)
            observed_selection = adapter.select(Xtrain_arr, ytrain_arr)

        # print(observed_selection)
        X_pruned = Xtrain_arr[observed_selection].copy()
        lasso = Lasso(alpha = self.lambda_ / X_pruned.shape[0], fit_intercept=False, tol = 1e-10, max_iter=100000000)
        y_pruned= ytrain_arr[observed_selection]
        Sigma_arr_pruned = Sigma_arr[np.ix_(observed_selection, observed_selection)]
        lasso.fit(X_pruned, y_pruned)
        p = Xtrain_arr.shape[1]

        beta_hat = lasso.coef_.reshape(p, 1)
        M = np.nonzero(beta_hat)[0]


        if len(observed_selection) == 0:
            raise ValueError("DV-score selected no samples; cannot choose a tested index")
        if len(M) == 0:
            raise ValueError("Empty feature selection set")
        # --- Test statistic
        p = len(M)
        # print(range(p))
        tested_position = int(np.random.choice(range(p))) if jth is None else int(jth)


        if self.data_valuation == 'lava':

            # print("test statistic", eta.T.dot(np.concatenate((ytrain_arr, yval_arr), axis=0)))
            y = np.concatenate([ytrain, yval])
            X = np.vstack([Xtrain_arr, Xval_arr])
            eta = make_naive_etaFS(X_pruned= X_pruned, y= y_pruned,
                            # obs_selection = observed_selection,
                            M = M,
                            j = tested_position)
            decomposition = decompose_y(y_pruned, eta, Sigma_arr_pruned)   
        elif self.data_valuation == 'cg-score':
            
            eta = make_naive_etaFS(X_pruned= X_pruned, y= y_pruned,
                                        # obs_selection = observed_selection,
                                        M = M,
                                        j = tested_position)
            y = ytrain
            decomposition = decompose_y(y_pruned, eta, Sigma_arr_pruned)
        intervals = [(-np.inf, np.inf)]
        p_value = compute_p_value(intervals, 
                            decomposition.etaT_y, decomposition.etaT_Sigma_eta,  
                        #   type="one-sided")
                            type="two-sided")

        self.p_value_ = p_value
        self.selected_indices_ = np.asarray(observed_selection, dtype=int)
        self.tested_index_ = tested_position
        return self.p_value_


class NaiveTest:
    def __init__(
            self,
            fraction,
            data_valuation,
            ):
        self.fraction = fraction
        self.data_valuation = data_valuation
    def fit(
        self,
        Xtrain: np.ndarray,
        ytrain: np.ndarray,
        Xval: np.ndarray,
        yval: np.ndarray,
        Sigma: np.ndarray | None = None,
        jth: int | None = None,
    ) -> "NaiveTest":
        Xtrain_arr = np.asarray(Xtrain)
        ytrain_arr = np.asarray(ytrain).reshape(-1)

        if Xtrain_arr.shape[0] != ytrain_arr.shape[0]:
            raise ValueError("Xtrain and ytrain must have the same number of samples")

        # --- Run selection
        if self.data_valuation == 'lava':
            Xval_arr = np.asarray(Xval)
            yval_arr = np.asarray(yval).reshape(-1)
            if Xval_arr.shape[0] != yval_arr.shape[0]:
                raise ValueError("Xval and yval must have the same number of samples")
            Sigma_arr = np.eye(ytrain_arr.shape[0] + yval_arr.shape[0]) if Sigma is None else np.asarray(Sigma)
            
            adapter = SelectionAdapter(fraction=self.fraction)
            observed_selection = adapter.select(Xtrain_arr, ytrain_arr, Xval_arr, yval_arr)
        elif self.data_valuation == 'cg-score':
            Sigma_arr = np.eye(ytrain_arr.shape[0]) if Sigma is None else np.asarray(Sigma)
            adapter = CGSelectionAdapter(fraction=self.fraction)
            observed_selection = adapter.select(Xtrain_arr, ytrain_arr)

        if len(observed_selection) == 0:
            raise ValueError("DV-score selected no samples; cannot choose a tested index")

        # --- Test statistic
        p = Xtrain_arr.shape[1]
        # print(range(p))
        tested_position = int(np.random.choice(range(p))) if jth is None else int(jth)


        if self.data_valuation == 'lava':
            eta = make_eta(Xt = Xtrain_arr, yt = ytrain_arr,
                                        Xv= Xval_arr, yv = yval_arr, 
                                        obs_selection = observed_selection,
                                        j = tested_position)
            # print("test statistic", eta.T.dot(np.concatenate((ytrain_arr, yval_arr), axis=0)))
            y = np.concatenate([ytrain, yval])
            decomposition = decompose_y(y, eta, Sigma_arr)   
        elif self.data_valuation == 'cg-score':
            
            eta = make_eta_cg(X= Xtrain_arr, y= ytrain_arr,
                                        obs_selection = observed_selection,
                                        j = tested_position)
            y = ytrain
            decomposition = decompose_y(y, eta, Sigma_arr)
        intervals = [(-np.inf, np.inf)]
        p_value = compute_p_value(intervals, 
                            decomposition.etaT_y, decomposition.etaT_Sigma_eta,  
                        #   type="one-sided")
                            type="two-sided")

        self.p_value_ = p_value
        self.selected_indices_ = np.asarray(observed_selection, dtype=int)
        self.tested_index_ = tested_position
        return self.p_value_

class SelectiveInference:
    """Convenience interface for p-values after DV-score instance selection."""

    def __init__(
        self,
        fraction: float = 0.5,
        zmin: float = -20.0,
        zmax: float = 20.0,
        step: float = 0.00001,
        method: str = "parametric",
        lambda_: float = 10.0,
        data_valuation: str = 'lava',
        lasso_path=True,
    ):
        if data_valuation not in ['lava', 'cg-score']:
            raise ValueError("data_valuation must be 'lava' or 'cg-score'")
        self.fraction = fraction
        self.config = SIConfig(zmin=zmin, zmax=zmax, step=step, data_valuation=data_valuation,lasso_path=lasso_path)
        self.lasso = LassoInfo(lambda_=lambda_)
        self.method = method

    def fit(
        self,
        Xtrain: np.ndarray,
        ytrain: np.ndarray,
        Xval: np.ndarray,
        yval: np.ndarray,
        Sigmat: np.ndarray | None = None,
        Sigmav: np.ndarray | None = None,
        jth: int | None = None,
    ) -> "SelectiveInference":

        print('run alg before si')
        st = time.time()
        Xtrain_arr = np.asarray(Xtrain)
        ytrain_arr = np.asarray(ytrain).reshape(-1)



        if Xtrain_arr.shape[0] != ytrain_arr.shape[0]:
            raise ValueError("Xtrain and ytrain must have the same number of samples")

        if Sigmat is not None and Sigmav is not None:
            Sigma = block_diag(Sigmat, Sigmav)
        else:
            Sigma = None
        # --- Run selection
        _t_sel = tp.start("fit: adapter.select (lava LP)")
        if self.config.data_valuation == 'lava':
            Xval_arr = np.asarray(Xval)
            yval_arr = np.asarray(yval).reshape(-1)
            if Xval_arr.shape[0] != yval_arr.shape[0]:
                raise ValueError("Xval and yval must have the same number of samples")
            # print(ytrain_arr.shape[0] )
            # print(yval_arr.shape[0])
            Sigma_arr = np.eye(ytrain_arr.shape[0] + yval_arr.shape[0]) if Sigma is None else np.asarray(Sigma)
            
            adapter = SelectionAdapter(fraction=self.fraction)
            observed_selection = adapter.select(Xtrain_arr, ytrain_arr, Xval_arr, yval_arr)
        elif self.config.data_valuation == 'cg-score':
            Sigma_arr = np.eye(ytrain_arr.shape[0]) if Sigmat is None else np.asarray(Sigmat)


            adapter = CGSelectionAdapter(fraction=self.fraction)
            observed_selection = adapter.select(Xtrain_arr, ytrain_arr)
        tp.stop(_t_sel)
        
        _t_ls = tp.start("fit: solve_Lasso (initial)")
        M, Mc, sM = solve_Lasso(Xtrain_arr[observed_selection], ytrain_arr.reshape(-1,1)[observed_selection], self.lasso.lambda_)
        tp.stop(_t_ls)
        # print('M', M)
        # print('y dv',ytrain_arr[observed_selection])
        self.lasso.M, self.lasso.Mc, self.lasso.sM = M, Mc, sM

        if len(observed_selection) == 0:
            raise ValueError("DV-score selected no samples; cannot choose a tested index")
        if len(M) == 0:
            raise ValueError("Empty feature selection set")
        # --- Test statistic
        p = len(M)
        print('end', time.time() - st)
        tested_position = int(np.random.choice(range(p))) if jth is None else int(jth)

        if self.config.data_valuation == 'lava':
            y = np.concatenate([ytrain, yval])
            X = np.vstack([Xtrain_arr, Xval_arr])
            _t_eta = tp.start("fit: make_eta_cgFS")
            eta = make_eta_cgFS(X= X, y= y,
                            # obs_selection = observed_selection,
                            M = M,
                            j = tested_position)
            tp.stop(_t_eta)
            _t_si = tp.start("fit: selective_inference (whole)")
            self.result_ = selective_inference(
                        Xtrain=Xtrain_arr,
                        Xval=Xval_arr,
                        ytrain=ytrain_arr,
                        yval=yval_arr,
                        Sigma=Sigma_arr,
                        adapter=adapter,
                        eta=eta,
                        fs=self.lasso,
                        tested_index=tested_position,
                        observed_selection=observed_selection,
                        config=self.config,
                        method=self.method,
                        # null_mean=c,
                    )
            tp.stop(_t_si)
        elif self.config.data_valuation == 'cg-score':
            _t_eta = tp.start("fit: make_eta_cgFS")
            eta = make_eta_cgFS(X= Xtrain_arr, y= ytrain_arr,
                                        # obs_selection = observed_selection,
                                        M = M,
                                        j = tested_position)
            tp.stop(_t_eta)
            _t_si = tp.start("fit: selective_inference (whole)")
            self.result_ = selective_inference(
                        Xtrain_arr,
                        None,
                        ytrain_arr,
                        None,
                        Sigma_arr,
                        adapter,
                        self.lasso,
                        eta,
                        tested_index=tested_position,
                        observed_selection=observed_selection,
                        config=self.config,
                        method=self.method,
                        # null_mean=c,
                    )
            tp.stop(_t_si)
        # print("test statistic", eta.T.dot(np.concatenate((ytrain_arr, yval_arr), axis=0)))
        # print("test position", tested_position)
        # --- Run selective inference
        

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
    Sigmat: np.ndarray | None = None,
    Sigmav: np.ndarray | None = None,
    fraction: float = 0.5,
    jth: int | None = None,
    zmin: float = -20.0,
    zmax: float = 20.0,
    step: float = 0.00001,
    method: str = "parametric",
    data_valuation: str = 'lava',
    lambda_: float = 10.0,
    lasso_path: bool = True
) -> float:
    """Return a selective p-value for one selected DV-score sample."""
    si = SelectiveInference(
        fraction=fraction,
        zmin=zmin,
        zmax=zmax,
        step=step,
        method=method,
        lambda_=lambda_,
        data_valuation=data_valuation,
        lasso_path=lasso_path
    ).fit(Xtrain, ytrain, Xval=Xval, yval=yval, Sigmat=Sigmat, Sigmav=Sigmav, jth=jth)
    return si.p_value_


__all__ = ["SelectionAdapter", "SelectiveInference", "SI_DV", "SIResult"]

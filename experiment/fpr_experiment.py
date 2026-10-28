import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from si_for_cg import SI_DV, NaiveTest, FSNaiveTest, DataSplittingTest
from si_for_cg.si_utils import generate, generate_correlated, generate_equicorrelated
import time
import traceback
def main():
    ns = [185, 135, 85, 35]
    rho_list = [0.2,0.4, 0.6, 0.8]
    fraction_list = [0.8, 0.6, 0.4, 0.2]
    d_list = [5,10,15,20]
    d = 15
    fraction = 0.5
    alpha = 0.05
    # dv = "lava"
    dv = "cg-score"
    n = 100
    # --- Split-failure setup -------------------------------------------------
    # AR(1) Toeplitz noise (generate_correlated, rho=0.8) is SHORT-RANGE: for
    # the split statistic eta^T y_test the variance ratio
    #   r = Var_true / ||eta||^2
    # stays ~= 1 (measured mean 0.996, median 0.961), so with identity Sigma
    # (Sigmat omitted below) and even with the true Sigma the FPR stays ~0.05
    # and no failure is visible.
    # Compound-symmetry (long-range) noise with an inflated common component
    # makes r = scale**2*((1-rho) + rho*chi2_1), so the identity-Sigma default
    # of DataSplittingTest understates the variance and the FPR inflates
    # (~0.23 at rho=0.8, scale=2) while the true-Sigma control stays ~alpha.
    rho = 0.8
    scale = 1.0         # per-row noise sd of generate_equicorrelated
    true_sigma = False  # False: identity Sigma (default path) -> shows failure
                        # True:  pass Sigmat=St -> control, FPR back to ~0.05
    lambda_ = 5        # lambda >= 5 silently drops 7-60% of draws as
                        # "Empty feature selection set" exceptions (survivor
                        # bias); lambda=1 keeps essentially all iterations
    iters = 120
    outputdir = 'output_newnaive'
    # outputdir = 'output_correlated'

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", outputdir), exist_ok=True)

    for method in ['overconditioning',]:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        # for n in ns:
        # for rho in rho_list:
        # for fraction in fraction_list:
        for d in d_list:
            # fname = f"fpr_{dv}FS_{method[:6]}_n{n}_rho{rho}.txt"
            sig = "true" if true_sigma else "eye"
            # fname = f"fpr_{dv}FS_{method[:6]}_n{n}_lambda{lambda_}.txt"
            # fname = f"fpr_{dv}FS_{method[:6]}_n{n}_rho{rho}_s{scale}_lam{lambda_}_sig{sig}.txt"
            fname = f"fpr_{dv}FS_{method[:6]}_d{d}_lambda{lambda_}.txt"
            # fname = f"fpr_{dv}_{method[:6]}_fraction{fraction}.txt"
            fpath = os.path.join(os.path.dirname(__file__), "..", outputdir, fname)
            valid_count = 0
            rejected_count = 0
            error_count = 0

            with open(fpath, "a") as f:
                i=0
                seedd = 0
                while i < iters:
                    seedd+=1
                    print('seed', seedd)
                    # rng = np.random.RandomState(seed=None)
                    true_beta_t = np.asarray([0]*d).reshape(-1,1)
                    true_beta_v = np.asarray([0]*d).reshape(-1,1)
                    X_tr, X_te, y_tr, y_te, St, Sv = generate(n, 15, d, true_beta_t, true_beta_v,seed=None)
                    # X_tr, X_te, y_tr, y_te, St, Sv = generate_correlated(n, 15, d, true_beta_t, true_beta_v, rho=0.8,seed=None)
                    # X_tr, X_te, y_tr, y_te, St, Sv = generate_equicorrelated(n, 15, d, true_beta_t, true_beta_v, rho=rho, scale=scale, seed=None)
                    # print(X_tr.shape, y_tr.shape)
                    # pval = 
                    start_t = time.time()
                    try:
                        if method == "naive":
                            runner = FSNaiveTest(fraction=fraction, data_valuation= dv, 
                                                    lambda_= lambda_
                                                    )
                            pv = runner.fit(X_tr, y_tr, Xval=X_te, yval=y_te, Sigmat=None, Sigmav=None, jth=None)
                        elif method == 'split':
                            runner = DataSplittingTest(fraction=fraction, data_valuation= dv, split_ratio= 0.5, 
                                                    lambda_= lambda_
                                                    )
                            # Sigmat omitted -> DataSplittingTest defaults Sigma to identity.
                            # Under common-factor correlated noise the true variance of the
                            # split statistic is scale**2*((1-rho)*||eta||^2 + rho*(sum eta)^2),
                            # so the identity assumption understates it and the FPR inflates.
                            # Set true_sigma=True to pass Sigmat=St and restore ~alpha.
                            kw = {}#{"Sigmat": St, "Sigmav": Sv} if true_sigma else {}
                            pv = runner.fit(X_tr, y_tr, Xval=X_te, yval=y_te, **kw, jth=None)    
                        else:
                            pv = SI_DV(X_tr, y_tr, Xval=X_te, yval=y_te, 
                                    #    Sigmat=St, Sigmav=Sv,
                                        fraction=fraction, jth=None,
                                        zmin=-20., zmax=20,
                                        method=method, data_valuation=dv)

                    except Exception as error:
                        print(f"An error occurred: {error}")
                        traceback.print_exc()  # Prints the full traceback with line numbers directly
                        error_count += 1
                        continue
                    end_t = time.time()
                    # with open(outputdir+'/time/time'+fname, 'a') as ftime:
                    #     ftime.write(str(end_t-start_t)+'\n')

                    f.write(f"{pv}\n")
                    f.flush()
                    i+=1
                    if not np.isnan(pv):
                        valid_count += 1
                        if pv <= alpha:
                            rejected_count += 1

                    if (i + 1) % 20 == 0:
                        print(f"  {i+1}/{iters} done, {valid_count} valid, {rejected_count} rejected")

            fpr = rejected_count / valid_count if valid_count > 0 else float('nan')
            print(f"  Saved to {fname}")
            print(f"  FPR = {rejected_count}/{valid_count} = {fpr:.4f}")
            print(f"  dropped iterations (exceptions): {error_count}/{iters}"
                  " -- large counts mean survivor bias, not inference failure")


if __name__ == "__main__":
    main()

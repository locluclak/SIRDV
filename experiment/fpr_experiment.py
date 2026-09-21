import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from si_for_cg import SI_DV, NaiveTest, FSNaiveTest, DataSplittingTest
from si_for_cg.si_utils import generate, generate_correlated
import time
import traceback
def main():
    ns = [200,150,100,50]
    # rho_list = [0.2,0.4, 0.6, 0.8]
    fraction_list = [0.2, 0.4, 0.6, 0.8]
    d_list = [4,5,6,7]
    d = 10
    fraction = 0.5
    iters = 120
    alpha = 0.05
    dv = "lava"
    # dv = "cg-score"
    # n = 150
    lambda_ = 10
    outputdir = 'output_newnaive'
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", outputdir), exist_ok=True)

    for method in ['naive',]:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        for n in ns:
        # for rho in rho_list:
        # for fraction in fraction_list:
        # for d in d_list:
            # fname = f"fpr_{dv}FS_{method[:6]}_n{n}_rho{rho}.txt"
            fname = f"fpr_{dv}FS_{method[:6]}_n{n}_lambda{lambda_}.txt"
            # fname = f"fpr_{dv}FS_{method[:6]}_d{d}_lambda{lambda_}.txt"
            # fname = f"fpr_{dv}_{method[:6]}_fraction{fraction}.txt"
            fpath = os.path.join(os.path.dirname(__file__), "..", outputdir, fname)
            valid_count = 0
            rejected_count = 0

            with open(fpath, "a") as f:
                i=0
                seedd = 0
                while i < iters:
                    seedd+=1
                    print('seed', seedd)
                    # rng = np.random.RandomState(seed=None)
                    true_beta_t = np.asarray([0]*d).reshape(-1,1)
                    true_beta_v = np.asarray([0]*d).reshape(-1,1)
                    X_tr, X_te, y_tr, y_te, St, Sv = generate(n, 15, d, true_beta_t, true_beta_v,seed=seedd)
                    # X_tr, X_te, y_tr, y_te, St, Sv = generate_correlated(n, 15, d, true_beta_t, true_beta_v, rho=0.5,seed=seedd)
                    # print(X_tr.shape, y_tr.shape)
                    # pval = 
                    start_t = time.time()
                    try:
                        if method == "naive":
                            runner = FSNaiveTest(fraction=fraction, data_valuation= dv, 
                                                    lambda_= lambda_
                                                    )
                            pv = runner.fit(X_tr, y_tr, Xval=X_te, yval=y_te, Sigmat=St, Sigmav=Sv, jth=None)
                        elif method == 'split':
                            runner = DataSplittingTest(fraction=fraction, data_valuation= dv, split_ratio= 0.2, 
                                                    lambda_= lambda_
                                                    )
                            pv = runner.fit(X_tr, y_tr, Xval=X_te, yval=y_te,
                                            # Sigmat=St, Sigmav=Sv,
                                              jth=None)    
                        else:
                            pv = SI_DV(X_tr, y_tr, Xval=X_te, yval=y_te, Sigmat=St, Sigmav=Sv,
                                        fraction=fraction, jth=None,
                                        zmin=-20., zmax=20,
                                        method=method, data_valuation=dv)

                    except Exception as error:
                        print(f"An error occurred: {error}")
                        traceback.print_exc()  # Prints the full traceback with line numbers directly
                        continue
                    end_t = time.time()
                    with open('output3/time'+fname, 'a') as ftime:
                        ftime.write(str(end_t-start_t)+'\n')

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


if __name__ == "__main__":
    main()

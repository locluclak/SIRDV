import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from si_for_cg import SI_DV, NaiveTest, FSNaiveTest
from si_for_cg.si_utils import generate
import time

def main():
    ns = [100,150]
    fraction_list = [0.2, 0.4, 0.6, 0.8]
    d_list = [4,5,6,7]
    d = 5
    fraction = 0.5
    iters = 120
    alpha = 0.05
    dv = "lava"
    n = 50
    lambda_ = 10
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "output3"), exist_ok=True)

    for method in ['parametric']:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        for n in ns:
        # for fraction in fraction_list:
        # for d in d_list:
            fname = f"fpr_{dv}FS_{method[:6]}_n{n}_lambda{lambda_}.txt"
            # fname = f"fpr_{dv}FS_{method[:6]}_d{d}_lambda{lambda_}.txt"
            # fname = f"fpr_{dv}_{method[:6]}_fraction{fraction}.txt"
            fpath = os.path.join(os.path.dirname(__file__), "..", "output3", fname)
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
                    # print(X_tr.shape, y_tr.shape)
                    # pval = 
                    start_t = time.time()
                    try:
                        if method != "naive":
                            pv = SI_DV(X_tr, y_tr, Xval=X_te, yval=y_te,
                                        fraction=fraction, jth=None,
                                        zmin=-20., zmax=20,
                                        method=method, data_valuation=dv)
                        else:
                            runner = FSNaiveTest(fraction=fraction, data_valuation= dv, 
                                                    # lambda_= lambda_
                                                    )
                            pv = runner.fit(X_tr, y_tr, Xval=X_te, yval=y_te, jth=None)
                    except Exception as error:
                        print(f"An error occurred: {error}")
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

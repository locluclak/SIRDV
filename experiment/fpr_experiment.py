import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from si_for_cg import SI_DV
from si_for_cg.si_utils import generate


def main():
    ns = [50, 100, 150, 200]
    d = 5
    fraction = 0.5
    iters = 120
    alpha = 0.05

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "output"), exist_ok=True)

    for method in ["parametric"]:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        for n in ns:
            print(f"\n--- n={n} ---")
            fname = f"fpr_cg_{method[:6]}_n{n}.txt"
            fpath = os.path.join(os.path.dirname(__file__), "..", "output", fname)
            valid_count = 0
            rejected_count = 0

            with open(fpath, "w") as f:
                for i in range(iters):
                    print('seed', i)
                    rng = np.random.RandomState(seed=i)
                    true_beta_t = np.asarray([0]*d).reshape(-1,1)
                    true_beta_v = np.asarray([0]*d).reshape(-1,1)
                    X_tr, X_te, y_tr, y_te, St, Sv = generate(n, 10, d, true_beta_t, true_beta_v,seed=i)
                    # print(X_tr.shape, y_tr.shape)
                    # pval = 

                    pv = SI_DV(X_tr, y_tr, Xval=X_te, yval=y_te,
                                fraction=fraction, jth=rng.randint(0, d),
                                zmin=-20., zmax=20,
                                method=method, data_valuation="cg-score")

                    f.write(f"{pv}\n")
                    f.flush()

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

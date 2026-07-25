import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from si_for_cg import SI_DV


def normalize_data(X, y):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(norms == 0, 1, norms)
    y_min, y_max = y.min(), y.max()
    if y_max - y_min > 0:
        y = 2 * (y - y_min) / (y_max - y_min) - 1
    return X, y


def main():
    n = 50
    n_list = [20, 60, 100, 140]
    d = 2
    betas = [0.5,1.0, 1.5, 2.0]
    beta_val = 0.2
    fraction = 0.8
    iters = 120
    alpha = 0.05

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "output"), exist_ok=True)

    for method in ["parametric"]:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        # for beta_val in betas:
        for n in n_list:
            print(f"\n--- beta={beta_val} ---")
            fname = f"tpr_{method[:6]}_nsize{n}.txt"
            fpath = os.path.join(os.path.dirname(__file__), "..", "output", fname)
            valid_count = 0
            rejected_count = 0

            with open(fpath, "w") as f:
                for i in range(iters):
                    rng = np.random.RandomState()
                    X = rng.randn(n, d)
                    beta = np.full(d, beta_val)
                    y = X @ beta + rng.randn(n)

                    X, y = normalize_data(X, y)
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

                    pv = SI_DV(X_tr, y_tr, Xval=X_te, yval=y_te,
                               fraction=fraction, jth=None, method=method)

                    f.write(f"{pv}\n")
                    f.flush()

                    if not np.isnan(pv):
                        valid_count += 1
                        if pv <= alpha:
                            rejected_count += 1

                    if (i + 1) % 20 == 0:
                        print(f"  {i+1}/{iters} done, {valid_count} valid, {rejected_count} rejected")

            tpr = rejected_count / valid_count if valid_count > 0 else float('nan')
            print(f"  Saved to {fname}")
            print(f"  TPR = {rejected_count}/{valid_count} = {tpr:.4f}")
            with open('test_eta.txt', 'a') as f:
                f.write('-\n')

if __name__ == "__main__":
    main()

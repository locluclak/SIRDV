import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import train_test_split
from si_for_cg import SI_DV


def main():
    ns = [50, 100, 150, 200]
    d = 5
    fraction = 0.5
    iters = 120
    alpha = 0.05

    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "output"), exist_ok=True)

    for method in ["overconditioning", "parametric"]:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")

        for n in ns:
            print(f"\n--- n={n} ---")
            fname = f"fpr_{method[:6]}_n{n}.txt"
            fpath = os.path.join(os.path.dirname(__file__), "..", "output", fname)
            valid_count = 0
            rejected_count = 0

            with open(fpath, "w") as f:
                for i in range(iters):
                    rng = np.random.RandomState()
                    X = rng.randn(n, d)
                    beta = np.zeros(d)
                    y = X @ beta + rng.randn(n)

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

            fpr = rejected_count / valid_count if valid_count > 0 else float('nan')
            print(f"  Saved to {fname}")
            print(f"  FPR = {rejected_count}/{valid_count} = {fpr:.4f}")


if __name__ == "__main__":
    main()

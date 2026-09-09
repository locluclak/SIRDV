"""Paired version of tpr_experiment.py.

Same output format (output5/tpr_{dv}_{prefix}_beta{b}.txt) but the parametric
and naive p-values are generated PAIRED:

  * ONE dataset per iteration (same seed for both methods)
  * the CG-score selection and Lasso active set M are computed once
  * ONE active feature j in M is tested by BOTH methods
    (the parametric test receives the global feature index, the naive FS test
     receives its position inside M)

Why this matters: in tpr_experiment.py the two methods run in separate passes
with independently random tested features, and SelectiveInference.fit
(si_dv.py) passes a *position in [0, |M|)* as the *global* feature index to
make_eta_cg, so the parametric test often conditions on an event in which the
tested feature is NOT active -- a very weak conditioning event that gives no
power gain.  When both methods test the same active feature, the parametric
test conditions on "feature j is active with the observed sign", the null
distribution is heavily truncated, and its p-value becomes much smaller than
the naive one while Bonferroni (naive with alpha / 2**8) still pays the full
multiplicity penalty.

Recommended setup (see tpr_paired_search.py for the full search):
  lambda_ = 10 (unchanged), beta in {1.0, 1.5}  ->  TPR_param ~ 0.6-0.75,
  TPR_bonf ~ 0.35-0.46.  beta = 2.0 is too strong: naive p-values fall below
  alpha/2**8 and both methods saturate.
"""

import numpy as np
import os
import sys
import contextlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from si_for_cg.si_dv import FSNaiveTest
from si_for_cg.si_utils import generate_contaminated_regression

from tpr_paired_search import SI_DV_jth, get_active_features


def main():
    ns = 50
    beta = [0.5, 1.0, 1.5, 2.0]
    d = 5
    fraction = 0.3
    iters = 58
    alpha = 0.05
    lambda_ = 10
    dv = "cg-score"
    con = 0.5
    out_dir = os.path.join(os.path.dirname(__file__), "..", "output5")
    os.makedirs(out_dir, exist_ok=True)

    devnull = open(os.devnull, "w")

    for b in beta:
        print(f"\n--- beta={b} ---")
        fname_p = f"tpr_{dv}_parame3_beta{b}.txt"
        fname_n = f"tpr_{dv}_naive3_beta{b}.txt"
        fpath_p = os.path.join(out_dir, fname_p)
        fpath_n = os.path.join(out_dir, fname_n)
        valid_count = 0
        rejected_param = 0
        rejected_bonf = 0

        with open(fpath_p, "a") as fp, open(fpath_n, "a") as fn:
            for i in range(iters):
                seedd = i + 1
                try:
                    X_tr, y_tr, X_val, y_val, _X_te, _y_te, _St, _Sv, _c = \
                        generate_contaminated_regression(
                            n_train=ns, n_val=25, n_test=0, p=d,
                            contamination=con, beta_clean=b,
                            std_good=1.0, std_bad=3.0, seed=seedd)

                    _sel, M = get_active_features(X_tr, y_tr, fraction, lambda_)
                    if len(M) == 0:
                        raise ValueError("Empty feature selection set")

                    # one active feature, tested by both methods
                    j_global = int(np.random.choice(M))
                    j_pos = int(np.flatnonzero(M == j_global)[0])

                    with contextlib.redirect_stdout(devnull):
                        pv_param = SI_DV_jth(X_tr, y_tr, fraction=fraction,
                                             jth=j_global, lambda_=lambda_,
                                             method="parametric",
                                             data_valuation=dv)
                    pv_naive = FSNaiveTest(fraction=fraction,
                                           data_valuation=dv,
                                           lambda_=lambda_).fit(
                        X_tr, y_tr, Xval=X_val, yval=y_val, jth=j_pos)
                except Exception as error:
                    print(f"seed {seedd} failed: {error}")
                    continue

                fp.write(f"{pv_param}\n")
                fn.write(f"{pv_naive}\n")
                fp.flush()
                fn.flush()

                if not np.isnan(pv_param):
                    valid_count += 1
                    if pv_param <= alpha:
                        rejected_param += 1
                    if pv_naive <= alpha / 2 ** 8:
                        rejected_bonf += 1

                if (i + 1) % 20 == 0:
                    print(f"  {i+1}/{iters} done, param {rejected_param}/"
                          f"{valid_count}, bonf {rejected_bonf}/{valid_count}")

        tpr_param = rejected_param / valid_count if valid_count else float("nan")
        tpr_bonf = rejected_bonf / valid_count if valid_count else float("nan")
        print(f"  Saved to {fname_p} / {fname_n}")
        print(f"  TPR parametric = {rejected_param}/{valid_count} = {tpr_param:.4f}")
        print(f"  TPR bonferroni = {rejected_bonf}/{valid_count} = {tpr_bonf:.4f}")


if __name__ == "__main__":
    main()

"""Diagnose why DataSplittingTest ('split') shows TPR as high as parametric.

Hypotheses
----------
H1 (variance misspecification): the held-out chunk used by the split test still
    contains contaminated rows (noise std_bad=3), but Sigmat/Sigma_test assumes
    std_good=1 for every row.  Var(eta^T y_test) is therefore understated by a
    factor ~ (1-con) + con*std_bad^2 ~ 5, so the split z-statistic is inflated
    and the test is anti-conservative (high FPR -> high apparent TPR).
H2 (mean dilution): beta_bad is hard-coded to -1, so under beta_clean=b the
    average per-row signal is (1-con)*b + con*(-1); at con=0.5, b=1 the shift is
    exactly 0, i.e. the split TPR at beta=1 is pure type-I error.

Parts
-----
zdiag : empirical mean/sd of the split z-statistic, variance ratio, |M|, and
        fraction of contaminated rows inside the held-out chunk.
split : rejection rates of DataSplittingTest under several data-generation
        setups (contaminated / clean / std_bad=1).
param : same rejection rates for SI_DV (parametric), for comparison.
oracle: split pipeline but with the TRUE per-row noise covariance (oracle
        diagnostic only) to confirm H1.

Usage:  python experiment/diagnose_split.py --part zdiag|split|param|oracle [--iters 200]
Raw p-values are appended to output5/diag_*.txt.
"""

import argparse
import os
import sys
import traceback

import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from si_for_cg import DataSplittingTest, SI_DV
from si_for_cg.si_adapter import make_eta_cgFS, CGSelectionAdapter
from si_for_cg.si_utils import (
    compute_p_value,
    decompose_y,
    generate_contaminated_regression,
)

ALPHA = 0.05
N_TRAIN = 50
N_VAL = 25
P = 8
FRACTION = 0.5
LAMBDA_ = 10
SPLIT_RATIO = 0.2
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output5")


def rejection_rate(pvals):
    vals = [p for p in pvals if not np.isnan(p)]
    if not vals:
        return float("nan"), 0, 0
    rej = sum(1 for p in vals if p <= ALPHA)
    return rej / len(vals), rej, len(vals)


def split_pipeline(X_tr, y_tr, Sigmat, seed_state=None, sigma_override_test=None):
    """Replicate DataSplittingTest.fit for cg-score, returning internals."""
    Xtrain_arr = np.asarray(X_tr)
    ytrain_arr = np.asarray(y_tr).reshape(-1)
    indices = np.arange(len(Xtrain_arr))
    Xtr, X_te, ytr, y_te, train_idx, test_idx = train_test_split(
        Xtrain_arr, ytrain_arr, indices, test_size=SPLIT_RATIO
    )

    adapter = CGSelectionAdapter(fraction=FRACTION)
    observed_selection = adapter.select(Xtr, ytr)

    X_pruned = Xtr[observed_selection].copy()
    lasso = Lasso(
        alpha=LAMBDA_ / X_pruned.shape[0],
        fit_intercept=False,
        tol=1e-10,
        max_iter=100000000,
    )
    lasso.fit(X_pruned, ytr[observed_selection])
    beta_hat = lasso.coef_.reshape(P, 1)
    M = np.nonzero(beta_hat)[0]
    if len(M) == 0:
        return None
    tested_position = int(np.random.choice(range(len(M))))

    Sigma_test = (
        Sigmat[np.ix_(test_idx, test_idx)]
        if sigma_override_test is None
        else sigma_override_test
    )
    eta = make_eta_cgFS(X=X_te, y=y_te, M=M, j=tested_position)
    decomposition = decompose_y(y_te, eta, Sigma_test)
    p_value = compute_p_value(
        [(-np.inf, np.inf)],
        decomposition.etaT_y,
        decomposition.etaT_Sigma_eta,
        type="two-sided",
    )
    return {
        "p_value": p_value,
        "z": decomposition.etaT_y / np.sqrt(decomposition.etaT_Sigma_eta),
        "test_idx": test_idx,
        "M_size": len(M),
        "eta": eta,
    }


def part_zdiag(iters):
    print(f"{'beta':>5} {'mean(z)':>9} {'sd(z)':>7} {'sd_theory':>9} "
          f"{'var_ratio':>9} {'%bad_test':>9} {'|M|':>5}")
    for b in [0.0, 0.5, 1.0, 2.0]:
        zs, ratios, bad_fracs, msizes = [], [], [], []
        for seed in range(500, 500 + iters):
            X_tr, y_tr, X_val, y_val, X_test, y_test, Sigmat, Sigmav, is_clean = (
                generate_contaminated_regression(
                    n_train=N_TRAIN, n_val=N_VAL, n_test=0, p=P,
                    contamination=0.5, beta_clean=b,
                    std_good=1.0, std_bad=3.0, seed=seed,
                )
            )
            res = split_pipeline(X_tr, y_tr, Sigmat)
            if res is None:
                continue
            test_idx = res["test_idx"]
            bad = ~is_clean[test_idx]
            # true per-row noise variance on held-out rows
            true_var_rows = np.where(bad, 3.0 ** 2, 1.0)
            eta = res["eta"].reshape(-1)
            var_ratio = float(eta.T @ (true_var_rows * eta) / (eta.T @ eta))
            zs.append(res["z"])
            ratios.append(var_ratio)
            bad_fracs.append(float(bad.mean()))
            msizes.append(res["M_size"])
        print(f"{b:>5} {np.mean(zs):>9.3f} {np.std(zs):>7.3f} "
              f"{np.sqrt(np.mean(ratios)):>9.3f} {np.mean(ratios):>9.3f} "
              f"{100 * np.mean(bad_fracs):>8.1f}% {np.mean(msizes):>5.1f}")
    print("\n(if sd(z) >> 1, the split test is anti-conservative: claimed sd=1)")


def run_split_setups(iters):
    setups = [
        # (label, contamination, std_bad, betas)
        ("con0.5_stdbad3", 0.5, 3.0, [0.0, 0.5, 1.0, 1.5, 2.0]),
        ("con0.5_stdbad1", 0.5, 1.0, [0.0, 0.5, 1.0, 1.5, 2.0]),
        ("con0.0_stdbad1", 0.0, 1.0, [0.0, 0.5, 1.0, 1.5, 2.0]),
    ]
    for label, con, std_bad, betas in setups:
        for b in betas:
            fname = f"diag_split_{label}_beta{b}.txt"
            fpath = os.path.join(OUTDIR, fname)
            pvals = []
            with open(fpath, "a") as f:
                for seed in range(1000, 1000 + iters):
                    X_tr, y_tr, X_val, y_val, X_test, y_test, Sigmat, Sigmav, _ = (
                        generate_contaminated_regression(
                            n_train=N_TRAIN, n_val=N_VAL, n_test=0, p=P,
                            contamination=con, beta_clean=b,
                            std_good=1.0, std_bad=std_bad, seed=seed,
                        )
                    )
                    try:
                        runner = DataSplittingTest(
                            fraction=FRACTION, data_valuation="cg-score",
                            split_ratio=SPLIT_RATIO, lambda_=LAMBDA_,
                        )
                        pv = runner.fit(
                            X_tr, y_tr, Xval=X_val, yval=y_val,
                            Sigmat=Sigmat, Sigmav=Sigmav, jth=None,
                        )
                    except Exception:
                        traceback.print_exc()
                        continue
                    pvals.append(pv)
                    f.write(f"{pv}\n")
                    f.flush()
            rate, rej, n = rejection_rate(pvals)
            print(f"split  {label:>15} beta={b:<4} rej={rej}/{n} rate={rate:.3f}")


def run_param_setups(iters, label=None, con=None, std_bad=None, beta=None):
    setups = [
        ("con0.5_stdbad3", 0.5, 3.0, [0.0]),
        ("con0.5_stdbad1", 0.5, 1.0, [0.0, 1.0, 2.0]),
        ("con0.0_stdbad1", 0.0, 1.0, [0.0, 0.5, 1.0, 1.5]),
    ]
    for lab, c, sb, betas in setups:
        if label is not None and lab != label:
            continue
        for b in betas:
            if beta is not None and abs(b - beta) > 1e-12:
                continue
            if con is not None and abs(c - con) > 1e-12:
                continue
            if std_bad is not None and abs(sb - std_bad) > 1e-12:
                continue
            run_param_one(iters, lab, c, sb, b)


def run_param_one(iters, label, con, std_bad, b):
    fname = f"diag_param_{label}_beta{b}.txt"
    fpath = os.path.join(OUTDIR, fname)
    pvals = []
    with open(fpath, "a") as f:
        for seed in range(1000, 1000 + iters):
            X_tr, y_tr, X_val, y_val, X_test, y_test, Sigmat, Sigmav, _ = (
                generate_contaminated_regression(
                    n_train=N_TRAIN, n_val=N_VAL, n_test=0, p=P,
                    contamination=con, beta_clean=b,
                    std_good=1.0, std_bad=std_bad, seed=seed,
                )
            )
            try:
                pv = SI_DV(
                    X_tr, y_tr, Xval=X_val, yval=y_val,
                    fraction=FRACTION, jth=None,
                    zmin=-20., zmax=20, lambda_=LAMBDA_,
                    method="parametric", data_valuation="cg-score",
                    Sigmat=Sigmat, Sigmav=Sigmav,
                )
            except Exception:
                traceback.print_exc()
                continue
            pvals.append(pv)
            f.write(f"{pv}\n")
            f.flush()
    rate, rej, n = rejection_rate(pvals)
    print(f"param  {label:>15} beta={b:<4} rej={rej}/{n} rate={rate:.3f}")


def run_oracle_split(iters):
    """Split test with the TRUE per-row noise covariance (oracle diagnostic)."""
    for b in [0.0, 0.5, 1.0, 1.5, 2.0]:
        fname = f"diag_oracle_split_con0.5_beta{b}.txt"
        fpath = os.path.join(OUTDIR, fname)
        pvals = []
        with open(fpath, "a") as f:
            for seed in range(1000, 1000 + iters):
                X_tr, y_tr, X_val, y_val, X_test, y_test, Sigmat, Sigmav, is_clean = (
                    generate_contaminated_regression(
                        n_train=N_TRAIN, n_val=N_VAL, n_test=0, p=P,
                        contamination=0.5, beta_clean=b,
                        std_good=1.0, std_bad=3.0, seed=seed,
                    )
                )
                try:
                    # oracle Sigma over ALL train rows, later cropped by split
                    noise_sd_all = np.where(is_clean, 1.0, 3.0)
                    Sigma_oracle = np.diag(noise_sd_all ** 2)
                    Xtrain_arr = np.asarray(X_tr)
                    ytrain_arr = np.asarray(y_tr).reshape(-1)
                    indices = np.arange(len(Xtrain_arr))
                    Xtr, X_te, ytr, y_te, train_idx, test_idx = train_test_split(
                        Xtrain_arr, ytrain_arr, indices, test_size=SPLIT_RATIO
                    )
                    adapter = CGSelectionAdapter(fraction=FRACTION)
                    observed_selection = adapter.select(Xtr, ytr)
                    X_pruned = Xtr[observed_selection].copy()
                    lasso = Lasso(
                        alpha=LAMBDA_ / X_pruned.shape[0],
                        fit_intercept=False, tol=1e-10, max_iter=100000000,
                    )
                    lasso.fit(X_pruned, ytr[observed_selection])
                    beta_hat = lasso.coef_.reshape(P, 1)
                    M = np.nonzero(beta_hat)[0]
                    if len(M) == 0:
                        continue
                    j = int(np.random.choice(range(len(M))))
                    Sigma_test = Sigma_oracle[np.ix_(test_idx, test_idx)]
                    eta = make_eta_cgFS(X=X_te, y=y_te, M=M, j=j)
                    decomposition = decompose_y(y_te, eta, Sigma_test)
                    pv = compute_p_value(
                        [(-np.inf, np.inf)],
                        decomposition.etaT_y,
                        decomposition.etaT_Sigma_eta,
                        type="two-sided",
                    )
                except Exception:
                    traceback.print_exc()
                    continue
                pvals.append(pv)
                f.write(f"{pv}\n")
                f.flush()
        rate, rej, n = rejection_rate(pvals)
        print(f"oracle {b:<4} rej={rej}/{n} rate={rate:.3f}")


def _contaminated_true_null(n_train=N_TRAIN, n_val=N_VAL, p=P, con=0.5,
                            std_good=1.0, std_bad=3.0, seed=None):
    """Generator replica with beta_bad = 0 so that beta=0 is a TRUE null.

    Mirrors si_utils.generate_contaminated_regression (which hard-codes
    beta_bad = -1) but returns the same tuple layout with correct Sigma.
    """
    rng = np.random.default_rng(seed)
    beta_clean = np.zeros(p)
    X_train = rng.normal(size=(n_train, p))
    is_bad = rng.random(n_train) < con
    noise_sd = np.where(is_bad, std_bad, std_good)
    y_train = X_train @ beta_clean + rng.normal(scale=noise_sd)
    X_val = rng.normal(size=(n_val, p))
    y_val = X_val @ beta_clean + rng.normal(scale=std_good, size=n_val)
    Sigmat = np.diag(noise_sd ** 2)
    Sigmav = std_good ** 2 * np.eye(n_val)
    return X_train, y_train, X_val, y_val, None, None, Sigmat, Sigmav, ~is_bad


def _width_containing(intervals, point):
    best = None
    for l, r in intervals:
        if l <= point <= r:
            w = r - l
            best = w if best is None else max(best, w)
    return best


def run_paramdiag(iters, betas=(2.0,), null_fpr=False, n_null=150):
    """Decompose the parametric test: statistic strength vs truncation width."""
    from si_for_cg.si_dv import SelectiveInference
    from si_for_cg.si_overconditioning import (
        CG_overconditioning_intervals, conditioning_Lasso)
    from si_for_cg.si_utils import solve_Lasso
    from si_for_cg.si_types import SIConfig
    from scipy.stats import norm

    def width_stats(intervals):
        return sum(r - l for l, r in intervals), len(intervals)

    for b in betas:
        z_obs_list, unc_p, si_p = [], [], []
        cg_w, fs_w, si_w, sd_list = [], [], [], []
        for seed in range(2000, 2000 + iters):
            X_tr, y_tr, X_val, y_val, _, _, Sigmat, Sigmav, _ = (
                generate_contaminated_regression(
                    n_train=N_TRAIN, n_val=N_VAL, n_test=0, p=P,
                    contamination=0.5, beta_clean=b,
                    std_good=1.0, std_bad=3.0, seed=seed,
                )
            )
            try:
                si = SelectiveInference(
                    fraction=FRACTION, zmin=-20., zmax=20., step=1e-5,
                    method="parametric", lambda_=LAMBDA_, data_valuation="cg-score",
                ).fit(X_tr, y_tr, Xval=X_val, yval=y_val,
                      Sigmat=Sigmat, Sigmav=Sigmav, jth=None)
            except Exception:
                continue
            # replicate the statistic to recover eta and z_obs
            adapter = CGSelectionAdapter(fraction=FRACTION)
            sel = adapter.select(X_tr, y_tr)
            M, Mc, sM = solve_Lasso(X_tr[sel], y_tr[sel].reshape(-1, 1), LAMBDA_)
            j = si.tested_index_
            eta = make_eta_cgFS(X=X_tr, y=y_tr, M=M, j=j)
            decomp = decompose_y(y_tr, eta, Sigmat)
            sd = float(np.sqrt(decomp.etaT_Sigma_eta))
            z = decomp.etaT_y / sd

            a_vec = decomp.a.reshape(-1)
            b_vec = decomp.b.reshape(-1)
            config = SIConfig(zmin=-20., zmax=20., step=1e-5,
                              data_valuation="cg-score")
            cg_itv = CG_overconditioning_intervals(X_tr, decomp, adapter, config)
            fs_itv = conditioning_Lasso(X_tr[sel], decomp.a.reshape(-1, 1)[sel],
                                        decomp.b.reshape(-1, 1)[sel],
                                        LAMBDA_, M, Mc, sM)

            z_obs_list.append(z)
            unc_p.append(2 * norm.sf(abs(z)))
            si_p.append(si.p_value_)
            sd_list.append(sd)
            cg_w.append(_width_containing(cg_itv, decomp.etaT_y) or np.nan)
            fs_w.append(_width_containing(fs_itv, decomp.etaT_y) or np.nan)
            si_w.append(_width_containing(si.intervals_, decomp.etaT_y) or np.nan)

        z_arr = np.asarray(z_obs_list)
        print(f"\n=== parametric decomposition, beta={b}, n={len(z_arr)} ===")
        print(f"z_obs/sd      : mean={z_arr.mean():+.3f}  sd={z_arr.std():.3f}")
        print(f"sd (raw units): mean={np.mean(sd_list):.3f}")
        print(f"uncond p      : rej@.05={np.mean(np.asarray(unc_p) <= 0.05):.3f}  "
              f"median={np.median(unc_p):.3f}")
        print(f"SI p          : rej@.05={np.mean(np.asarray(si_p) <= 0.05):.3f}  "
              f"median={np.median(si_p):.3f}")
        print(f"width/sd containing z: CG-OC={np.nanmean(np.asarray(cg_w)/np.asarray(sd_list)):.2f}  "
              f"FS={np.nanmean(np.asarray(fs_w)/np.asarray(sd_list)):.2f}  "
              f"SI-final={np.nanmean(np.asarray(si_w)/np.asarray(sd_list)):.2f}")

    if null_fpr:
        for name, gen in [("param", "si"), ("split", "split")]:
            pvals = []
            for seed in range(3000, 3000 + n_null):
                X_tr, y_tr, X_val, y_val, _, _, Sigmat, Sigmav, _ = (
                    _contaminated_true_null(seed=seed))
                try:
                    if gen == "si":
                        pv = SI_DV(
                            X_tr, y_tr, Xval=X_val, yval=y_val,
                            fraction=FRACTION, jth=None,
                            zmin=-20., zmax=20, lambda_=LAMBDA_,
                            method="parametric", data_valuation="cg-score",
                            Sigmat=Sigmat, Sigmav=Sigmav,
                        )
                    else:
                        runner = DataSplittingTest(
                            fraction=FRACTION, data_valuation="cg-score",
                            split_ratio=SPLIT_RATIO, lambda_=LAMBDA_)
                        pv = runner.fit(X_tr, y_tr, Xval=X_val, yval=y_val,
                                        Sigmat=Sigmat, Sigmav=Sigmav, jth=None)
                except Exception:
                    traceback.print_exc()
                    continue
                pvals.append(pv)
            rate, rej, n = rejection_rate(pvals)
            print(f"TRUE-null (beta_bad=0) FPR {name}: {rej}/{n} = {rate:.3f}")


def part_summary():
    lines = []
    lines.append("Rejection rates at alpha=0.05 (diag_* files in output5)")
    lines.append("Current tpr_experiment setup = con0.5_stdbad3")
    lines.append("")
    lines.append(f"{'setup':<24} {'beta':>5} {'split':>8} {'param':>8} {'oracle-split':>12}")
    settings = [
        ("con0.5_stdbad3", [0.0, 0.5, 1.0, 1.5, 2.0]),
        ("con0.5_stdbad1", [0.0, 0.5, 1.0, 1.5, 2.0]),
        ("con0.0_stdbad1", [0.0, 0.5, 1.0, 1.5, 2.0]),
    ]
    for label, betas in settings:
        for b in betas:
            cols = []
            for prefix in ["split", "param"]:
                fpath = os.path.join(OUTDIR, f"diag_{prefix}_{label}_beta{b}.txt")
                if os.path.exists(fpath):
                    rate, rej, n = rejection_rate([float(l) for l in open(fpath)
                                                  if l.strip()])
                    cols.append(f"{rate:.3f}({rej}/{n})")
                else:
                    cols.append("-")
            fpath = os.path.join(OUTDIR, f"diag_oracle_split_con0.5_beta{b}.txt")
            if label == "con0.5_stdbad3" and os.path.exists(fpath):
                rate, rej, n = rejection_rate([float(l) for l in open(fpath)
                                              if l.strip()])
                cols.append(f"{rate:.3f}({rej}/{n})")
            else:
                cols.append("-")
            lines.append(f"{label:<24} {b:>5} {cols[0]:>16} {cols[1]:>16} {cols[2]:>16}")
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(OUTDIR, "diag_summary.txt"), "w") as f:
        f.write(text + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True,
                    choices=["zdiag", "split", "param", "oracle", "summary",
                             "paramdiag"])
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--label", default=None)
    ap.add_argument("--con", type=float, default=None)
    ap.add_argument("--stdbad", type=float, default=None)
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--nnull", type=int, default=150)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    np.random.seed(0)

    if args.part == "zdiag":
        part_zdiag(args.iters)
    elif args.part == "split":
        run_split_setups(args.iters)
    elif args.part == "param":
        run_param_setups(args.iters, label=args.label, con=args.con,
                         std_bad=args.stdbad, beta=args.beta)
    elif args.part == "oracle":
        run_oracle_split(args.iters)
    elif args.part == "summary":
        part_summary()
    elif args.part == "paramdiag":
        run_paramdiag(args.iters, betas=(2.0,), null_fpr=True, n_null=args.nnull)


if __name__ == "__main__":
    main()

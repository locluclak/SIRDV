"""Fit Lasso on the real datasets and assess SI-method suitability.

Datasets (``realdata/dataset/``):
  * concrete_data.csv     -> 8 mix features, target = Concrete Compressive Strength
  * energy_efficiency.csv -> 8 features (X1..X8), two targets Y1 (heating load) /
                             Y2 (cooling load); both Y columns are removed from the
                             design so neither target leaks into the other's fit
  * parkinsons_updrs.csv  -> 19 features (identifier ``subject#`` and both UPDRS
                             targets dropped), two targets total_UPDRS / motor_UPDRS
  * garments_worker.csv   -> 13 features (``wip`` dropped: 506/1197 missing),
                             target = actual_productivity
  * winequality.csv       -> 11 numeric features (non-numeric ``type`` = red/white
                             dropped), target = quality (ordinal, 3..9)
  * onlinenewspopularity.csv -> 58 numeric features (``url`` dropped),
                             target = shares. Subsample with ``--news-n`` (default
                             5000) to keep Lasso / the O(n^2) cg-score Gram tractable

``paddydataset.csv`` is excluded: its numeric design has effective rank 7 of 36
(11 input columns are exact scalar multiples of Hectares), so SI_DV cannot run.

For each (dataset, target, lambda) the script
  * z-scores the design columns and the target (so ``fit_intercept=False`` inside
    ``solve_Lasso`` is equivalent to an intercept model, and lambda is comparable
    across datasets),
  * fits Lasso with sklearn's ``alpha = lambda / n`` and certifies the support via
    the KKT check in ``si_for_cg.si_utils.solve_Lasso``,
  * reports the support, coefficients, train R^2, and the two quantities the SI
    pipeline inverts (``conditioning_Lasso`` and ``make_eta_cgFS`` both call
    ``inv(X_M^T X_M)``): the rank and condition number of X_M,
  * flags collinear / duplicated design columns, which are what break that inverse.

Lambdas default to a per-dataset set chosen so the support is sparse but non-empty
(``_DEFAULT_LAMBDAS``); ``--lambdas`` overrides them for all datasets.

With ``--probe`` it also runs one end-to-end SI_DV p-value (cg-score +
overconditioning) to show whether the pipeline actually completes on the data.

Run from the repo root:
  & "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" realdata/exp_script/fit_lasso_realdata.py
  & "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" realdata/exp_script/fit_lasso_realdata.py --probe
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from si_for_cg.si_utils import solve_Lasso

warnings.filterwarnings("ignore")  # sklearn coordinate-descent ConvergenceWarnings


DATASET_DIR = os.path.join(ROOT, "realdata", "dataset")
RESULT_DIR = os.path.join(ROOT, "realdata", "result")

CONCRETE_TARGET = "Concrete Compressive Strength"
ENERGY_TARGETS = ["Y1", "Y2"]
PARKINSONS_TARGETS = ["total_UPDRS", "motor_UPDRS"]
GARMENTS_TARGET = "actual_productivity"
WINE_TARGET = "quality"
NEWS_TARGET = "shares"

# Per-dataset lambdas: sparse-but-nonempty supports with a well-conditioned X_M.
_DEFAULT_LAMBDAS = {
    "concrete": [10.0, 100.0],   # |M| = 6 / 5
    "energy": [10.0, 50.0],      # |M| = 4-5 / 3
    "parkinsons": [100.0, 200.0],  # |M| = 10 / 7 (below this X_M is ill-conditioned)
    "garments": [25.0, 50.0],    # |M| = 9 / 7
    "wine": [50.0, 200.0],       # |M| = 9 / 6 (λ≤20 keeps all 11)
    "news": [100.0, 200.0],      # |M| = ~19 / ~9 at n=5000
}
_NEWS_SEED = 0


def _numeric_design(df: pd.DataFrame, targets: list[str],
                    extra_drop: tuple[str, ...] = ()) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Design = numeric columns minus targets and identifiers; y = first target."""
    nonnumeric = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    extra = [c for c in extra_drop if c in df.columns]
    Xdf = df.drop(columns=nonnumeric + extra + [t for t in targets if t in df.columns])
    names = list(Xdf.columns)
    X = Xdf.to_numpy(dtype=float)
    y = df[targets[0]].to_numpy(dtype=float)
    keep = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[keep], y[keep]
    return X, y, names, nonnumeric + extra


def load_all(news_n: int | None = 5000) -> list[tuple[str, str, np.ndarray, np.ndarray, list[str], list[str]]]:
    """Return a list of (dataset, target, X, y, feature_names, dropped_columns)."""
    out = []

    df = pd.read_csv(os.path.join(DATASET_DIR, "concrete_data.csv"))
    X, y, names, dropped = _numeric_design(df, [CONCRETE_TARGET])
    out.append(("concrete", CONCRETE_TARGET, X, y, names, dropped))

    df = pd.read_csv(os.path.join(DATASET_DIR, "energy_efficiency.csv"))
    for tgt in ENERGY_TARGETS:
        X, y, names, _ = _numeric_design(df, [tgt, "Y1", "Y2"])
        out.append((f"energy[{tgt}]", tgt, X, y, names, []))

    df = pd.read_csv(os.path.join(DATASET_DIR, "parkinsons_updrs.csv"))
    df.columns = df.columns.str.strip()
    for tgt in PARKINSONS_TARGETS:
        X, y, names, dropped = _numeric_design(
            df, [tgt, "motor_UPDRS", "total_UPDRS"], extra_drop=("subject#",))
        out.append((f"parkinsons[{tgt}]", tgt, X, y, names, dropped))

    df = pd.read_csv(os.path.join(DATASET_DIR, "garments_worker.csv"))
    X, y, names, dropped = _numeric_design(df, [GARMENTS_TARGET], extra_drop=("wip",))
    out.append(("garments", GARMENTS_TARGET, X, y, names, dropped))

    df = pd.read_csv(os.path.join(DATASET_DIR, "winequality.csv"))
    X, y, names, dropped = _numeric_design(df, [WINE_TARGET])
    out.append(("wine", WINE_TARGET, X, y, names, dropped))

    df = pd.read_csv(os.path.join(DATASET_DIR, "onlinenewspopularity.csv"))
    df.columns = df.columns.str.strip()
    note = []
    if news_n is not None and news_n < len(df):
        idx = np.sort(np.random.RandomState(_NEWS_SEED).choice(len(df), news_n, replace=False))
        df = df.iloc[idx].reset_index(drop=True)
        note = [f"(random subsample of {news_n} / {len(pd.read_csv(os.path.join(DATASET_DIR, 'onlinenewspopularity.csv')))} rows)"]
    X, y, names, dropped = _numeric_design(df, [NEWS_TARGET])
    out.append(("news", NEWS_TARGET, X, y, names, dropped + note))

    return out


def lambdas_for(tag: str, override: list[float] | None) -> list[float]:
    if override:
        return override
    for key, vals in _DEFAULT_LAMBDAS.items():
        if tag.startswith(key):
            return vals
    return [1.0, 5.0]


def standardize(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Z-score design columns and target (unit variance each)."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    ysd = y.std(ddof=1)
    ysd = ysd if ysd > 0 else 1.0
    return (X - mu) / sd, (y - y.mean()) / ysd


def design_diagnostics(Xs: np.ndarray, names: list[str]) -> dict:
    """Effective rank, top correlation, and collinear column clusters."""
    n, p = Xs.shape
    diag = {"eff_rank": int(np.linalg.matrix_rank(Xs))}

    C = np.corrcoef(Xs, rowvar=False) if p > 1 else np.ones((1, 1))
    A = np.abs(C - np.eye(p))
    i, j = np.unravel_index(np.argmax(A), A.shape)
    diag["max_corr"] = float(A[i, j])
    diag["max_corr_pair"] = (names[i], names[j])

    # Union-find over columns that become numerically identical after
    # standardization (which is exactly the signature of raw proportionality).
    parent = list(range(p))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in range(p):
        for b in range(a + 1, p):
            if 1.0 - A[a, b] < 1e-9:  # |corr| == 1 -> proportional columns
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    groups: dict[int, list[int]] = {}
    for a in range(p):
        groups.setdefault(find(a), []).append(a)

    clusters = []
    for members in groups.values():
        if len(members) > 1:
            rep = members[0]
            sd_rep = np.std(Xs[:, rep])
            members_info = [(names[a], float(np.std(Xs[:, a]) / sd_rep) if sd_rep else np.nan)
                            for a in members]
            clusters.append((names[rep], members_info))
    diag["collinear_clusters"] = clusters
    return diag



def fit_report(Xs: np.ndarray, yc: np.ndarray, names: list[str], lam: float) -> dict:
    """Fit one lambda with the project's KKT-certified solver + diagnostics."""
    from sklearn.linear_model import Lasso

    n, p = Xs.shape
    info: dict = {"n": n, "p": p, "lambda": lam, "alpha": lam / n}

    M_cert = None
    try:
        M_cert, _Mc, _sM = solve_Lasso(Xs, yc, lam)
        info["kkt_ok"] = True
    except Exception as exc:  # noqa: BLE001
        info["kkt_ok"] = False
        info["error"] = f"{type(exc).__name__}: {exc}"

    lasso = Lasso(alpha=lam / n, fit_intercept=False, tol=1e-12, max_iter=1_000_000)
    lasso.fit(Xs, yc)
    beta = lasso.coef_.astype(float)

    if M_cert is not None:
        M = np.asarray(M_cert, dtype=int)
    else:
        thr = 1e-8 * max(1.0, np.max(np.abs(beta)))
        M = np.nonzero(np.abs(beta) > thr)[0]

    info["M"] = M
    info["beta"] = beta
    info["certified"] = M_cert is not None
    residual = yc - Xs @ beta
    info["r2"] = 1.0 - float(residual @ residual) / float(yc @ yc)

    if M.size:
        XM = Xs[:, M]
        info["rank_XM"] = int(np.linalg.matrix_rank(XM))
        sv = np.linalg.svd(XM, compute_uv=False)
        info["cond_XM"] = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    else:
        info["rank_XM"] = 0
        info["cond_XM"] = np.nan
    return info


def print_design(tag: str, diag: dict, writer) -> None:
    def emit(line):
        print(line)
        writer.write(line + "\n")

    emit(f"  design rank = {diag['eff_rank']}  |  max |corr| = {diag['max_corr']:.4f} "
         f"({diag['max_corr_pair'][0]} ~ {diag['max_corr_pair'][1]})")
    for rep, members_info in diag["collinear_clusters"]:
        desc = ", ".join(f"{nm} (x{r:g})" if abs(r - 1.0) > 1e-6 else nm
                         for nm, r in members_info)
        emit(f"  COLLINEAR CLUSTER [{len(members_info)} cols] proportional to '{rep}': {desc}")


def print_report(names: list[str], info: dict, writer) -> None:
    def emit(line=""):
        print(line)
        writer.write(line + "\n")

    M = info["M"]
    cert = "yes" if info["certified"] else "FAILED (KKT not met)"
    emit(f"  lambda={info['lambda']:g}  (sklearn alpha = lambda/n = {info['alpha']:.6g})")
    emit(f"    KKT-certified support : {cert}   |M|={info['absM']}  (n={info['n']}, p={info['p']})")
    emit(f"    train R^2             : {info['r2']:.4f}")
    if info["absM"]:
        emit(f"    rank(X_M)             : {info['rank_XM']}  (full rank = {info['rank_XM'] == info['absM']})")
        emit(f"    cond(X_M)             : {info['cond_XM']:.3e}")
        emit(f"    |M| / n               : {info['absM'] / info['n']:.3f}")
        order = M[np.argsort(-np.abs(info["beta"][M]))]
        for j in order:
            emit(f"      {names[j]:<34s} beta={info['beta'][j]:+.5f}")
    else:
        emit("    (empty support: nothing for the SI test to target)")
    emit("")


def si_probe(Xs: np.ndarray, yc: np.ndarray, lam: float, fraction: float, seed: int, writer) -> None:
    """One end-to-end SI_DV p-value (cg-score + overconditioning)."""
    from si_for_cg import SI_DV
    from sklearn.model_selection import train_test_split

    Xtr, Xval, ytr, yval = train_test_split(Xs, yc, test_size=0.3, random_state=seed)
    try:
        pval = SI_DV(Xtr, ytr, Xval=Xval, yval=yval, fraction=fraction, jth=None,
                     zmin=-20.0, zmax=20.0, lambda_=lam, data_valuation="cg-score",
                     method="overconditioning", lasso_path=False)
        line = f"    SI probe (cg-score/overconditioning, lambda={lam:g}): p-value = {pval:.4f}"
    except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
        line = f"    SI probe (cg-score/overconditioning, lambda={lam:g}): FAILED -> {type(exc).__name__}: {exc}"
        traceback.print_exc()
    print(line)
    writer.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="+", default=None,
                    help="override the per-dataset lambdas for every dataset")
    ap.add_argument("--news-n", type=int, default=5000,
                    help="random subsample size for onlinenewspopularity (0 = full)")
    ap.add_argument("--probe", action="store_true", help="run one end-to-end SI_DV p-value")
    ap.add_argument("--fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    news_n = args.news_n if args.news_n and args.news_n > 0 else None

    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, "lasso_realdata.txt")

    with open(out_path, "w") as writer:
        for dataset, target, X, y, names, dropped in load_all(news_n):
            header = (f"\n{'='*72}\n{dataset}  |  target = {target}  |  "
                      f"n={X.shape[0]}, p={X.shape[1]}")
            print(header)
            writer.write(header + "\n")
            cols = [c for c in dropped if not c.startswith("(")]
            notes = [c for c in dropped if c.startswith("(")]
            if cols:
                line = f"  dropped non-numeric columns: {cols}"
                print(line)
                writer.write(line + "\n")
            for note in notes:
                print(f"  {note}")
                writer.write(f"  {note}\n")

            Xs, yc = standardize(X, y)
            print_design(dataset, design_diagnostics(Xs, names), writer)
            for lam in lambdas_for(dataset, args.lambdas):
                info = fit_report(Xs, yc, names, lam)
                info["absM"] = int(info["M"].size)
                print_report(names, info, writer)
                if not info["certified"]:
                    err = f"    KKT error: {info.get('error', '')}"
                    print(err)
                    writer.write(err + "\n")
                if args.probe:
                    si_probe(Xs, yc, lam, args.fraction, args.seed, writer)
                    writer.write("\n")

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

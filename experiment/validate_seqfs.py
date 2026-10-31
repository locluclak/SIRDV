"""Smoke validation for the seqfs (fixed-K forward selection) path.

Runs one dataset through ``SI_DV(..., fs_method='seqfs', K=...)`` for every
``(data_valuation, method)`` combination in
``{'lava', 'cg-score'} x {'overconditioning', 'parametric_nested'}`` and
asserts a finite p-value is produced.  Also asserts that
``method='parametric'`` + ``fs_method='seqfs'`` raises ``ValueError``, and that
the default ``fs_method='lasso'`` path is untouched.

Run from the repo root with the ``IS`` python (see AGENTS.md):
    python experiment/validate_seqfs.py
"""

import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from si_for_cg import SI_DV, FSNaiveTest, DataSplittingTest
from si_for_cg.si_dv import SelectiveInference


def generate(n, m, p, seed=0):
    rng = np.random.RandomState(seed)
    Xs = rng.normal(0, 1, size=(n, p))
    Xt = rng.normal(0, 1, size=(m, p))
    ys = rng.normal(0, 1, size=n)
    yt = rng.normal(0, 1, size=m)
    return Xs, ys, Xt, yt


def run_case(dv, method, Xtr, ytr, Xval, yval, fs_method="seqfs", K=3, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        pv = SI_DV(
            Xtr, ytr, Xval=Xval, yval=yval, fraction=0.5, jth=0,
            zmin=-10.0, zmax=10.0, lambda_=5.0, data_valuation=dv,
            method=method, lasso_path=False, fs_method=fs_method, K=K, **kw,
        )
    return float(pv)


def main():
    Xtr, ytr, Xval, yval = generate(n=80, m=20, p=10, seed=0)
    failures = []

    for dv in ("lava", "cg-score"):
        for method in ("overconditioning", "parametric_nested"):
            try:
                pv = run_case(dv, method, Xtr, ytr, Xval, yval)
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{dv}/{method}: raised {type(exc).__name__}: {exc}")
                continue
            ok = np.isfinite(pv) and 0.0 <= pv <= 1.0
            print(f"[seqfs] {dv:8s} {method:18s} p={pv!r} {'OK' if ok else 'BAD'}")
            if not ok:
                failures.append(f"{dv}/{method}: non-finite p-value {pv!r}")

    # method='parametric' is Lasso-only and must be rejected up front.
    try:
        SelectiveInference(method="parametric", fs_method="seqfs", K=3)
    except ValueError as exc:
        print(f"[seqfs] parametric+seqfs rejected: {exc}")
    else:
        failures.append("method='parametric' + fs_method='seqfs' did not raise")

    # Naive / data-splitting baselines must accept the seqfs feature selection.
    for dv in ("lava", "cg-score"):
        for name, factory in (
            ("naive", lambda d: FSNaiveTest(fraction=0.5, data_valuation=d,
                                            fs_method="seqfs", K=3)),
            ("split", lambda d: DataSplittingTest(0.5, d, fs_method="seqfs", K=3)),
        ):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    runner = factory(dv)
                    runner.fit(Xtr, ytr, Xval=Xval, yval=yval, jth=0)
                pv = float(runner.p_value_)
                intervals = runner.intervals_
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{dv}/{name}: raised {type(exc).__name__}: {exc}")
                continue
            ok = (np.isfinite(pv) and 0.0 <= pv <= 1.0
                  and isinstance(intervals, list) and len(intervals) > 0)
            print(f"[seqfs] {dv:8s} {name:18s} p={pv!r} "
                  f"n_intervals={len(intervals)} {'OK' if ok else 'BAD'}")
            if not ok:
                failures.append(f"{dv}/{name}: bad p-value/intervals "
                                f"{pv!r}, {intervals!r}")

    # Backward compatibility: the default Lasso path still runs.
    try:
        pv = run_case("cg-score", "overconditioning", Xtr, ytr, Xval, yval,
                      fs_method="lasso")
        print(f"[lasso] cg-score overconditioning  p={pv!r}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"lasso default regression: {type(exc).__name__}: {exc}")

    if failures:
        print("\nFAILURES:")
        for item in failures:
            print("  -", item)
        raise SystemExit(1)
    print("\nAll seqfs validation checks passed.")


if __name__ == "__main__":
    main()

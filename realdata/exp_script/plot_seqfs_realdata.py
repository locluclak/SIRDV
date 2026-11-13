"""Plot the real-data **seqfs** (fixed-K forward-selection) SI p-value results.

Reads the ``si_pvals_*.txt`` files written by
``realdata/exp_script/si_experiment_realdata_seqfs.py`` (default dir
``realdata/result_seqfs/``) and draws seqfs-specific figures, one set per
(dataset, data-valuation):

  * ``seqfs_boxplot_{tag}_{dv}.png`` - p-value box plot (x = feature,
    hue = method), features ordered by the sidecar design order and the same
    ≤ ``--min-count`` visibility rule as ``analyze_si_pvals.py``;
  * ``seqfs_ecdf_{tag}_{dv}.png`` - empirical CDF of p-values by method (a
    compact "how many features are significant" view; steeper-left = more
    rejections);
  * ``seqfs_reject_rate_{tag}_{dv}.png`` - rejection fraction at ``--alpha``
    per method, with the raw counts annotated.

It also writes ``seqfs_pvals_long.csv`` (tidy long form) and
``seqfs_summary.csv`` (per dataset/dv/method: n, mean/median p, rejected count
and rate) and prints the summary table.

The ``bonferroni`` files store RAW naive p-values; they are converted to the
family-wise adjusted ``min(1, p * perm(P, K))`` (``perm`` = ``P!/(P-K)!``, the
number of ordered fixed-K forward-selection paths; ``P`` = design dimension and
``K`` the fixed selection size, both read from the file header) so every method
is on the same ``alpha`` scale.  This matches ``experiment/seqfs_plot.py``
(``alpha / perm(D, K)``).

Run from the repo root with the ``IS`` python (see AGENTS.md):

  & "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" realdata/exp_script/plot_seqfs_realdata.py
  & "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" realdata/exp_script/plot_seqfs_realdata.py --plot ecdf --alpha 0.05
  & "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" realdata/exp_script/plot_seqfs_realdata.py --result-dir realdata/result_seqfs --min-count 0
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "realdata", "exp_script"))

from analyze_si_pvals import (  # noqa: E402  (path set up above)
    FNAME_RE, METHOD_ORDER, METHOD_SHORT, METHOD_COLOR,
    load_feature_names, load_pvals, bonferroni_multiplier,
)

DEFAULT_RESULT_DIR = os.path.join(ROOT, "realdata", "result_seqfs")


def load_records(result_dir: str) -> pd.DataFrame:
    """Long-form p-values: dataset, dv, Method, Feature, feat_idx, p_value."""
    files = sorted(f for f in os.listdir(result_dir) if FNAME_RE.match(f))
    records = []
    for f in files:
        m = FNAME_RE.match(f)
        tag, dv, method = m["tag"], m["dv"], m["method"]
        names = load_feature_names(result_dir, tag)

        # bonferroni stores RAW naive p-values -> family-wise min(1, p * perm(P, K)).
        adj = None
        if method == "bonferroni":
            adj = bonferroni_multiplier(os.path.join(result_dir, f))
            if adj is None:
                print(f"warning: cannot derive the bonferroni multiplier from the "
                      f"header of {f}; bonferroni p-values left unadjusted")

        for _it, idx, p in load_pvals(os.path.join(result_dir, f)):
            if adj is not None:
                p = min(1.0, p * adj)
            records.append(dict(dataset=tag, dv=dv, Method=method,
                                feat_idx=int(idx),
                                Feature=names.get(idx, f"f{idx}"),
                                p_value=p))
    return pd.DataFrame(records)


def summarize(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    for (tag, dv, method), sub in df.groupby(["dataset", "dv", "Method"]):
        v = sub["p_value"].dropna().to_numpy()
        n = int(v.size)
        rej = int((v <= alpha).sum())
        rows.append(dict(
            dataset=tag, dv=dv, method=method, n_pvals=n,
            mean_p=float(v.mean()) if n else np.nan,
            median_p=float(np.median(v)) if n else np.nan,
            reject=rej, reject_rate=(rej / n if n else np.nan),
        ))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {mm: i for i, mm in enumerate(METHOD_ORDER)}
    out["_o"] = out["method"].map(order).fillna(len(order)).astype(int)
    return out.sort_values(["dataset", "dv", "_o"]).drop(columns="_o").reset_index(drop=True)


def _feature_order(df: pd.DataFrame, tag: str) -> list[str]:
    sub = df[df["dataset"] == tag].sort_values("feat_idx")
    return list(pd.unique(sub["Feature"]))


def _methods_present(df: pd.DataFrame, tag: str, dv: str) -> list[str]:
    present = set(df[(df["dataset"] == tag) & (df["dv"] == dv)]["Method"])
    return [mm for mm in METHOD_ORDER if mm in present]


def plot_box(df, tag, dv, outdir, min_count, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sub = df[(df["dataset"] == tag) & (df["dv"] == dv) & df["p_value"].notna()].copy()
    if sub.empty:
        return None
    methods = _methods_present(df, tag, dv)

    # appearance count per feature (iterations in which it is selected into M)
    counts = (sub.groupby(["Feature", "Method"]).size()
              .unstack(fill_value=0))
    keep = [ft for ft in _feature_order(sub, tag)
            if ft in counts.index and int(counts.loc[ft].max()) > min_count]
    if not keep:
        print(f"skip boxplot {tag}[{dv}]: no feature has any method count > {min_count}")
        return None
    sub = sub[sub["Feature"].isin(keep)]

    plt.figure(figsize=(max(8.0, 1.3 * len(keep)), 4.5))
    ax = sns.boxplot(x="Feature", y="p_value", hue="Method", data=sub,
                     order=keep, hue_order=methods,
                     palette=[METHOD_COLOR[mm] for mm in methods],
                     linewidth=1, showfliers=False)
    plt.xlabel("Feature")
    plt.ylabel("p-value")
    plt.axhline(0.05, color="grey", ls="--", lw=0.8)
    plt.title(f"{tag}  ({dv})  seqfs")
    plt.xticks(rotation=45, ha="right")
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), title="Method")
    plt.tight_layout()
    out = os.path.join(outdir, f"seqfs_boxplot_{tag}_{dv}.png")
    plt.savefig(out, dpi=dpi)
    plt.close()
    return out


def plot_ecdf(df, tag, dv, alpha, outdir, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = _methods_present(df, tag, dv)
    if not methods:
        return None

    plt.figure(figsize=(7.0, 4.8))
    for mm in methods:
        v = np.sort(df[(df["dataset"] == tag) & (df["dv"] == dv)
                       & (df["Method"] == mm)]["p_value"].dropna().to_numpy())
        if v.size == 0:
            continue
        y = np.arange(1, v.size + 1) / v.size
        plt.step(np.concatenate([[0.0], v]), np.concatenate([[0.0], y]),
                 where="post", color=METHOD_COLOR[mm], lw=1.6,
                 label=f"{METHOD_SHORT[mm]} (n={v.size})")
    plt.axvline(alpha, color="grey", ls="--", lw=0.8)
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("p-value")
    plt.ylabel("empirical CDF  (fraction of p-values <= x)")
    plt.title(f"{tag}  ({dv})  seqfs")
    plt.legend(loc="lower right")
    plt.tight_layout()
    out = os.path.join(outdir, f"seqfs_ecdf_{tag}_{dv}.png")
    plt.savefig(out, dpi=dpi)
    plt.close()
    return out


def plot_reject_rate(df, tag, dv, alpha, outdir, dpi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[(df["dataset"] == tag) & (df["dv"] == dv)]
    methods = _methods_present(df, tag, dv)
    if not methods:
        return None

    rates, labels = [], []
    for mm in methods:
        v = sub[sub["Method"] == mm]["p_value"].dropna()
        n = int(v.size)
        rej = int((v <= alpha).sum())
        rates.append(rej / n if n else np.nan)
        labels.append(f"{METHOD_SHORT[mm]}\n{rej}/{n}" if n else METHOD_SHORT[mm])

    x = np.arange(len(methods))
    plt.figure(figsize=(1.4 * len(methods) + 3.5, 4.5))
    bars = plt.bar(x, rates, color=[METHOD_COLOR[mm] for mm in methods], width=0.62)
    plt.axhline(alpha, color="grey", ls="--", lw=0.8, label=f"alpha={alpha:g}")
    for b, r in zip(bars, rates):
        if np.isfinite(r):
            plt.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.2f}",
                     ha="center", va="bottom", fontsize=9)
    plt.xticks(x, labels)
    plt.ylim(0, 1.05)
    plt.ylabel(f"rejection fraction (p <= {alpha:g})")
    plt.title(f"{tag}  ({dv})  seqfs")
    plt.legend(loc="upper right")
    plt.tight_layout()
    out = os.path.join(outdir, f"seqfs_reject_rate_{tag}_{dv}.png")
    plt.savefig(out, dpi=dpi)
    plt.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot the real-data seqfs SI p-value results.")
    ap.add_argument("--result-dir", default=DEFAULT_RESULT_DIR,
                    help="dir with si_pvals_*.txt from the seqfs experiment")
    ap.add_argument("--outdir", default=None,
                    help="where to write figures/CSVs (default: --result-dir)")
    ap.add_argument("--plot", default="all", choices=["all", "box", "ecdf", "rate"],
                    help="which figure(s) to draw")
    ap.add_argument("--alpha", type=float, default=0.05, help="rejection level")
    ap.add_argument("--min-count", type=int, default=20,
                    help="hide a feature (boxplot) when every method's appearance "
                         "count over iterations is <= this value")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    result_dir = os.path.abspath(args.result_dir)
    outdir = os.path.abspath(args.outdir) if args.outdir else result_dir
    if not os.path.isdir(result_dir):
        print(f"no such result dir: {result_dir}")
        return
    os.makedirs(outdir, exist_ok=True)

    df = load_records(result_dir)
    if df.empty:
        print(f"no si_pvals_*.txt found in {result_dir}")
        return

    df.to_csv(os.path.join(outdir, "seqfs_pvals_long.csv"), index=False)

    summary = summarize(df, args.alpha)
    summary.to_csv(os.path.join(outdir, "seqfs_summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.to_string(index=False))
    print(f"\nwrote {os.path.join(outdir, 'seqfs_pvals_long.csv')}")
    print(f"wrote {os.path.join(outdir, 'seqfs_summary.csv')}")

    do_box = args.plot in ("all", "box")
    do_ecdf = args.plot in ("all", "ecdf")
    do_rate = args.plot in ("all", "rate")

    for tag, dv in sorted({(t, d) for t, d in zip(df["dataset"], df["dv"])}):
        saved = []
        if do_box:
            saved.append(plot_box(df, tag, dv, outdir, args.min_count, args.dpi))
        if do_ecdf:
            saved.append(plot_ecdf(df, tag, dv, args.alpha, outdir, args.dpi))
        if do_rate:
            saved.append(plot_reject_rate(df, tag, dv, args.alpha, outdir, args.dpi))
        for path in saved:
            if path:
                print(f"saved {path}")


if __name__ == "__main__":
    main()

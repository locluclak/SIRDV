"""Summarise the SI_DV p-value outputs in ``realdata/result/``.

For every ``si_pvals_*.txt`` found:

  * **feature appearance counts** - for each (dataset, dv, method), how many
    iterations each feature appears in (i.e. is selected and has a p-value).
    Written to ``si_feature_counts.csv`` (long form, one row per
    dataset/dv/method/feature) and ``si_feature_counts.txt`` (readable, one table
    per (dataset, dv)).
  * **boxplot per (dataset, dv)** - grouped box plot of feature p-values
    (x = Feature, hue = Method), saved as ``si_boxplot_{tag}_{dv}.png``, so the
    ``cg-score`` and ``lava`` selections are plotted separately.

The feature labels come from the ``si_features_{tag}.txt`` sidecar, so a
p-value is always tied to its feature name (initial design order).

``bonferroni`` files hold RAW naive p-values; this script converts them to the
family-wise adjusted p-value ``min(1, p * 2**P)`` (``P`` = design dimension, read
from the file header), so the plain ``alpha`` level and the 0.05 reference line
apply to every method.

Usage (run from the repo root, conda env ``IS``):
  & "C:\\Users\\Asus\\anaconda3\\envs\\IS\\python.exe" realdata/exp_script/analyze_si_pvals.py
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULT_DIR = os.path.join(ROOT, "realdata", "result")

FNAME_RE = re.compile(
    r"^si_pvals_(?P<tag>.+)_(?P<dv>cg-score|lava)_"
    r"(?P<method>overconditioning|parametric_nested|datasplit|bonferroni)_"
    r"lam(?P<lam>[^_]+)_frac(?P<frac>[^_]+)_n(?P<n>\d+)_(?P<var>ols|identity)\.txt$"
)
METHOD_ORDER = ["bonferroni", "datasplit", "overconditioning", "parametric_nested"]
METHOD_SHORT = {"overconditioning": "OC", "parametric_nested": "PN",
                "datasplit": "DS", "bonferroni": "Bonf"}
# left-to-right plot order matches METHOD_ORDER; colour is per method so a
# missing method never shifts the others' colours.
METHOD_COLOR = {
    "bonferroni": "#ff7f0e",         # orange
    "datasplit": "#1f77b4",          # blue
    "overconditioning": "#2ca02c",   # green
    "parametric_nested": "#9467bd",  # purple
}


def load_feature_names(result_dir: str, tag: str) -> dict[int, str]:
    """index -> feature name (initial design order) from the sidecar."""
    path = os.path.join(result_dir, f"si_features_{tag}.txt")
    names: dict[int, str] = {}
    if not os.path.exists(path):
        return names
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                names[int(parts[0])] = parts[1]
    return names


def load_pvals(path: str):
    """Yield (iteration, seed, feature_index, p_value) from one p-value file."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            it, _seed = int(parts[0]), int(parts[1])
            for pair in parts[3:]:                     # skip iter, seed, |M|
                idx_s, p_s = pair.split(":")
                yield it, int(idx_s), float(p_s)


HDR_P_RE = re.compile(r"\bp=(\d+)\b")


def load_header_p(path: str) -> int | None:
    """Design dimension ``p`` from the file's header comment (``# ... p=<int>``).

    This is the value the experiment used for the Bonferroni threshold
    ``alpha / 2**p``, so the analyzer's adjustment matches it exactly.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            mm = HDR_P_RE.search(line)
            if mm:
                return int(mm.group(1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Counts + boxplots for SI_DV p-values.")
    ap.add_argument("--result-dir", default=RESULT_DIR)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--min-count", type=int, default=20,
                    help="hide a feature when EVERY method's appearance count "
                         "over iterations is <= this value (its other boxes are kept)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.result_dir) if FNAME_RE.match(f))
    if not files:
        print(f"no si_pvals_*.txt found in {args.result_dir}")
        return

    records = []                                        # long form for the boxplots
    counts: dict[tuple[str, str, str], dict[str, int]] = {}
    iters: dict[tuple[str, str, str], set[int]] = {}
    feat_order: dict[str, list[str]] = {}

    for f in files:
        m = FNAME_RE.match(f)
        tag, dv, method = m["tag"], m["dv"], m["method"]
        names = load_feature_names(args.result_dir, tag)
        feat_order[tag] = [names[i] for i in sorted(names)] or feat_order.get(tag, [])

        # The bonferroni files store RAW naive p-values; convert them to the
        # family-wise adjusted p-value min(1, p * 2**P), where P is the design
        # dimension (2**P = number of selection subsets).  After this, the usual
        # alpha level applies to every method.
        adj = None
        if method == "bonferroni":
            P = load_header_p(os.path.join(args.result_dir, f))
            if P is None:
                print(f"warning: no p= in header of {f}; bonferroni p-values left unadjusted")
            else:
                adj = float(2 ** P)

        for it, idx, p in load_pvals(os.path.join(args.result_dir, f)):
            if adj is not None:
                p = min(1.0, p * adj)
            feat = names.get(idx, f"f{idx}")
            records.append(dict(dataset=tag, dv=dv, Method=method, Feature=feat,
                                **{"p-value": p}))
            counts.setdefault((tag, dv, method), {})
            counts[(tag, dv, method)][feat] = counts[(tag, dv, method)].get(feat, 0) + 1
            iters.setdefault((tag, dv, method), set()).add(it)

    df = pd.DataFrame(records)
    long_csv = os.path.join(args.result_dir, "si_pvals_long.csv")
    df.to_csv(long_csv, index=False)

    # ---------------- appearance counts ----------------
    count_rows = []
    for (tag, dv, method), c in counts.items():
        n_it = len(iters[(tag, dv, method)])
        for feat in feat_order.get(tag, sorted(c)):
            k = c.get(feat, 0)
            count_rows.append(dict(dataset=tag, dv=dv, method=method, feature=feat,
                                   count=k, n_iterations=n_it,
                                   fraction=(k / n_it if n_it else float("nan"))))
    cdf = pd.DataFrame(count_rows).sort_values(["dataset", "dv", "feature", "method"])
    counts_csv = os.path.join(args.result_dir, "si_feature_counts.csv")
    cdf.to_csv(counts_csv, index=False)

    lines = []
    for tag, dv in sorted({(t, d) for t, d, _ in counts}):
        methods = [mm for mm in METHOD_ORDER if (tag, dv, mm) in counts]
        piv = pd.DataFrame(index=feat_order.get(tag, []))
        for mm in methods:
            n_it = len(iters[(tag, dv, mm)])
            c = counts[(tag, dv, mm)]
            piv[METHOD_SHORT[mm]] = [f"{c.get(ft, 0)}/{n_it}" for ft in piv.index]
        lines.append(f"=== dataset={tag}  dv={dv} ===  (appearances / iterations)")
        lines.append(piv.to_string())
        lines.append("")
    counts_txt = "\n".join(lines)
    with open(os.path.join(args.result_dir, "si_feature_counts.txt"), "w", encoding="utf-8") as fh:
        fh.write(counts_txt + "\n")
    print(counts_txt)
    print(f"wrote {counts_csv}")
    print(f"wrote {long_csv}")

    if args.no_plot:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    for tag, dv in sorted({(t, d) for t, d, _ in counts}):
        sub = df[(df["dataset"] == tag) & (df["dv"] == dv) & df["p-value"].notna()].copy()
        if sub.empty:
            continue
        # hide a feature only when EVERY method's appearance count is <= --min-count;
        # a feature that qualifies keeps all of its per-method boxes.
        dv_methods = [mm for mm in METHOD_ORDER if (tag, dv, mm) in counts]
        present = [ft for ft in feat_order.get(tag, []) if ft in set(sub["Feature"])]
        keep_feat, removed = [], []
        for ft in present:
            best = max(counts[(tag, dv, mm)].get(ft, 0) for mm in dv_methods)
            (keep_feat if best > args.min_count else removed).append(ft)
        sub = sub[sub["Feature"].isin(keep_feat)]
        if removed:
            print(f"{tag} [{dv}]: hidden features (every method has count <= "
                  f"{args.min_count}): {removed}")

        order = [ft for ft in feat_order.get(tag, []) if ft in set(sub["Feature"])]
        if not order:
            print(f"skip {tag} [{dv}]: no feature has count > {args.min_count}")
            continue
        methods = [mm for mm in METHOD_ORDER if mm in set(sub["Method"])]

        plt.figure(figsize=(max(8.0, 1.3 * len(order)), 4.5))
        ax = sns.boxplot(
            x="Feature",
            y="p-value",
            hue="Method",
            data=sub,
            order=order,
            hue_order=methods,
            palette=[METHOD_COLOR[mm] for mm in methods],
            linewidth=1,
            showfliers=False,
        )
        plt.xlabel("Feature")
        plt.ylabel("p-value")
        plt.axhline(0.05, color="grey", ls="--", lw=0.8)
        plt.title(f"{tag}  ({dv})")
        plt.xticks(rotation=45, ha="right")
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1.02, 1), title="Method")
        plt.tight_layout()
        out = os.path.join(args.result_dir, f"si_boxplot_{tag}_{dv}.png")
        plt.savefig(out, dpi=args.dpi)
        plt.close()
        print(f"saved {out}")


if __name__ == "__main__":
    main()

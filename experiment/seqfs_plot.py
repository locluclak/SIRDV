"""Reader/plotter for the seqfs FPR/TPR experiment outputs.

Reads the ``output_seqfs/seqfs_pv_*.txt`` files produced by
``experiment/seqfs_experiment.py``, prints FPR-vs-n and TPR tables, and draws
the same style of figures as ``fpr_plot.py`` / ``tpr_plot.py`` (one curve per
method, per data valuation).

Run from the repo root with the ``IS`` python (see AGENTS.md):

    python experiment/seqfs_plot.py
    python experiment/seqfs_plot.py --dv cg-score
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

ALPHA = 0.05
D = 15

N_LIST = [50, 100, 150, 200]
BETA_LIST = [0.5, 1.0, 1.5, 2.0]
FRACTION_LIST = [0.3, 0.4, 0.5, 0.6]

METHODS = {
    "naive": ("bonferroni", "tab:blue", "o"),
    "overco": ("overconditioning", "tab:green", "o"),
    "parame": ("parametric", "tab:red", "o"),
    "split": ("datasplit", "tab:orange", "o"),
}


def read_pvalues(path):
    if not os.path.exists(path):
        return None
    pvalues = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pvalues.append(float(line))
            except ValueError:
                continue
    return pvalues


def rejection_rate(pvalues, tag):
    if not pvalues:
        return float("nan"), 0, 0
    valid = len(pvalues)
    threshold = ALPHA / 2 ** D if tag == "naive" else ALPHA
    rejected = sum(1 for pv in pvalues if pv <= threshold)
    return rejected / valid, valid, rejected


def table(outdir, dv, kind, values, value_fmt, value_label):
    print(f"\n[{dv}] {kind}")
    print(f"{value_label:>8s}  {'method':>18s}  {'valid':>6s}  "
          f"{'rejected':>8s}  {'rate':>6s}")
    results = {}
    for tag, (label, _, _) in METHODS.items():
        rates = []
        for value in values:
            if kind == "fpr":
                fname = f"seqfs_pv_fpr_{dv}_n{value}_{tag}.txt"
            elif kind == "tpr_beta":
                fname = f"seqfs_pv_tpr_{dv}_beta{value}_{tag}.txt"
            else:
                fname = f"seqfs_pv_tpr_{dv}_frac{value}_{tag}.txt"
            pvalues = read_pvalues(os.path.join(outdir, fname))
            if pvalues is None:
                print(f"  {fname} not found, skipping")
                rates.append(float("nan"))
                continue
            rate, valid, rejected = rejection_rate(pvalues, tag)
            rates.append(rate)
            print(f"{value:>8{value_fmt}}  {label:>18s}  {valid:6d}  "
                  f"{rejected:8d}  {rate:6.4f}")
        results[tag] = rates
    return results


def draw(values, results, xlabel, ylabel, title, save_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for tag, rates in results.items():
        label, color, marker = METHODS[tag]
        ax.plot(values, rates, marker=marker, color=color, label=label,
                linewidth=1.5, markersize=6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(values)
    ax.set_ylim(-0.01, 1.1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="output_seqfs")
    ap.add_argument("--dv", default=None, choices=[None, "lava", "cg-score"])
    args = ap.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), "..", args.outdir)
    dvs = [args.dv] if args.dv else ["lava", "cg-score"]

    for dv in dvs:
        fpr = table(outdir, dv, "fpr", N_LIST, "d", "n")
        draw(N_LIST, fpr, "Sample size (n)", "False Positive Rate",
             f"FPR vs n under Null ({dv}, seqfs K=5)",
             os.path.join(outdir, f"seqfs_fpr_{dv}.png"))

        tpr_b = table(outdir, dv, "tpr_beta", BETA_LIST, ".2f", "beta")
        draw(BETA_LIST, tpr_b, "Signal strength (beta)", "True Positive Rate",
             f"TPR vs beta ({dv}, seqfs K=5)",
             os.path.join(outdir, f"seqfs_tpr_beta_{dv}.png"))

        tpr_f = table(outdir, dv, "tpr_fraction", FRACTION_LIST, ".2f", "frac")
        draw(FRACTION_LIST, tpr_f, "Fraction", "True Positive Rate",
             f"TPR vs fraction ({dv}, seqfs K=5)",
             os.path.join(outdir, f"seqfs_tpr_frac_{dv}.png"))


if __name__ == "__main__":
    main()

import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

import math
def main():
    ns = [50, 100, 150, 200]
    alpha = 0.05
    methods = {
        # "overconditioning": "overco",
        # "parametric": "parame",
        "naive": "naive",
        # "bonferroni": "naive"
    }
    fraction = 0.5

    dv = 'cg-score'
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output2")

    print(f"{'n':>6s}  {'method':>18s}  {'valid':>6s}  {'rejected':>8s}  {'FPR':>6s}")
    print(f"{'------':>6s}  {'------------------':>18s}  {'------':>6s}  {'--------':>8s}  {'------':>6s}")

    fpr_results = {}

    for method_name, prefix in methods.items():
        fpr_list = []
        for n in ns:
            fname = f"fpr_{dv}FS_{prefix}_n{n}_lambda10.txt"
            fpath = os.path.join(output_dir, fname)

            if not os.path.exists(fpath):
                print(f"  {fname} not found, skipping")
                fpr_list.append(float('nan'))
                continue

            pvalues = []
            with open(fpath, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            pvalues.append(float(line))
                        except ValueError:
                            continue

            valid = len(pvalues)
            if method_name == "bonferroni":
                adj_pvalue = alpha / math.comb(n, int(fraction*n))
                rejected = sum(1 for pv in pvalues if pv <= adj_pvalue)
            else:
                rejected = sum(1 for pv in pvalues if pv <= alpha)
            fpr = rejected / valid if valid > 0 else float('nan')
            fpr_list.append(fpr)

            print(f"{n:6d}  {method_name:>18s}  {valid:6d}  {rejected:8d}  {fpr:6.4f}")

        fpr_results[method_name] = fpr_list

    print()

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"overconditioning": "tab:green", "parametric": "tab:red", "naive": "tab:blue", "bonferroni": "tab:purple"}
    markers = {"overconditioning": "o", "parametric": "o", "naive": "o", "bonferroni": "o"}

    for method_name, fpr_list in fpr_results.items():
        ax.plot(ns, fpr_list, marker=markers[method_name], color=colors[method_name],
                label=method_name, linewidth=1.5, markersize=6)

    ax.axhline(y=alpha, color="gray", linestyle="--", linewidth=1, label=f"alpha={alpha}")
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("False Positive Rate")
    ax.set_title("FPR vs Sample Size under Null Hypothesis")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(ns)
    ax.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, f"fpr_comparison_{dv}_FS_fixsample.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()

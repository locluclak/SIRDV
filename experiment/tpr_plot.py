import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt


def main():
    betas = [0.5, 1, 1.5, 2]
    alpha = 0.05
    methods = {
        "overconditioning": "overco",
        "parametric": "parame",
    }
    dv = "lava"  # or "cg-score"
    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")

    print(f"{'beta':>6s}  {'method':>18s}  {'valid':>6s}  {'rejected':>8s}  {'TPR':>6s}")
    print(f"{'------':>6s}  {'------------------':>18s}  {'------':>6s}  {'--------':>8s}  {'------':>6s}")

    tpr_results = {}

    for method_name, prefix in methods.items():
        tpr_list = []
        for beta_val in betas:
            fname = f"tpr_{dv}_{prefix}_beta{beta_val}.txt"
            fpath = os.path.join(output_dir, fname)

            if not os.path.exists(fpath):
                print(f"  {fname} not found, skipping")
                tpr_list.append(float('nan'))
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
            rejected = sum(1 for pv in pvalues if pv <= alpha)
            tpr = rejected / valid if valid > 0 else float('nan')
            tpr_list.append(tpr)

            print(f"{beta_val:6.2f}  {method_name:>18s}  {valid:6d}  {rejected:8d}  {tpr:6.4f}")

        tpr_results[method_name] = tpr_list

    print()

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"overconditioning": "tab:blue", "parametric": "tab:red"}
    markers = {"overconditioning": "o", "parametric": "s"}

    for method_name, tpr_list in tpr_results.items():
        ax.plot(betas, tpr_list, marker=markers[method_name], color=colors[method_name],
                label=method_name, linewidth=1.5, markersize=6)

    # ax.axhline(y=alpha, color="gray", linestyle="--", linewidth=1, label=f"nominal (alpha={alpha})")
    ax.set_xlabel("Signal strength (beta)")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("TPR vs Signal Strength under Alternative Hypothesis")
    ax.set_xticks(betas)
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "tpr_comparison.png")
    plt.savefig(save_path, dpi=150)
    print(f"Saved plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    main()

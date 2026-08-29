import math
import matplotlib.pyplot as plt
import numpy as np

# Load data from the text files
par_dir = 'output/'
dv = 'cg-score'
n = 50
naive_data = np.loadtxt(par_dir+f"fpr_{dv}_naive_n{n}.txt")
oc_data = np.loadtxt(par_dir+f"fpr_{dv}_overco_n{n}.txt")
para_data = np.loadtxt(par_dir+f"fpr_{dv}_parame_n{n}.txt")

# Calculate Bonferroni adjusted p-values from naive_data
fraction = 0.1  # Set your specific fraction value here
m = math.comb(n, int(fraction * n))
bonf_naive_data = np.minimum(naive_data * m, 1.0)
print(bonf_naive_data)

def get_ecdf(data):
    """Compute empirical CDF for a 1D array."""
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


# Compute eCDF values
x_naive, y_naive = get_ecdf(naive_data)
x_oc, y_oc = get_ecdf(oc_data)
x_para, y_para = get_ecdf(para_data)
x_bonf, y_bonf = get_ecdf(bonf_naive_data)

# Plotting
plt.figure(figsize=(8, 6))

plt.plot(x_naive, y_naive, color="blue", label="Naive", linewidth=2)
plt.plot(x_oc, y_oc, color="green", label="Overco (OC)", linewidth=2)
plt.plot(x_para, y_para, color="red", label="Parametric (Para)", linewidth=2)
# plt.plot(x_bonf, y_bonf, color="purple", label="Bonferroni (Naive)", linewidth=2)

# Reference uniform line (often used in FPR/p-value KS plots)
plt.plot([0, 1], [0, 1], color="grey", linestyle="--", alpha=0.7, label="Uniform(0,1)")

# Formatting
plt.title("KS Plot", fontsize=14)
plt.xlabel("p-value", fontsize=12)
plt.ylabel("Cumulative Probability", fontsize=12)
plt.grid(True, linestyle=":", alpha=0.6)
plt.legend(loc="lower right")

plt.tight_layout()
plt.savefig(f'output/ksplot{n}_{dv}.png')
plt.show()
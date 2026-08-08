import re
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Read log file
# ==========================
with open("logs.txt", "r") as f:
    lines = f.readlines()

records = []

pattern = re.compile(
    r"z:\s*([-+]?\d*\.?\d+),\s*selection:\s*\((.*?)\)\s*--\s*(True|False)"
)

max_feature = -1

for line in lines:
    m = pattern.search(line)
    if not m:
        continue

    z = float(m.group(1))

    selection_str = m.group(2).strip()
    if selection_str == "":
        selection = []
    else:
        selection = [int(x.strip()) for x in selection_str.split(",")]

    ok = (m.group(3) == "True")

    if selection:
        max_feature = max(max_feature, max(selection))

    records.append((z, selection, ok))

# ==========================
# Build binary matrix
# ==========================
n_features = max_feature + 1
n_z = len(records)

M = np.zeros((n_features, n_z), dtype=int)

z_values = []
true_columns = []

for j, (z, selection, ok) in enumerate(records):
    z_values.append(z)

    for f in selection:
        M[f, j] = 1

    if ok:
        true_columns.append(j)

# ==========================
# Plot
# ==========================
# sort features by frequency
freq = M.sum(axis=1)
order = np.argsort(-freq)
M = M[order]
feature_labels = order


fig, ax = plt.subplots(figsize=(16, 8))

im = ax.imshow(
    M,
    aspect="auto",
    origin="lower",
    cmap="Greys",
    interpolation="nearest"
)

# highlight True columns
for j in true_columns:
    ax.axvline(j, color="red", linewidth=1.5)

ax.set_xlabel("z index")
# ax.set_ylabel("Feature ID")
ax.set_yticks(np.arange(n_features))
ax.set_yticklabels(feature_labels)

# show some z labels (avoid clutter)
step = max(1, n_z // 10)
ticks = np.arange(0, n_z, step)

ax.set_xticks(ticks)
ax.set_xticklabels([f"{z_values[i]:.2f}" for i in ticks], rotation=45)

plt.title("Feature Selection Stability across z")
plt.tight_layout()
plt.show()
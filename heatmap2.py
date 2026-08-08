import re
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Read log file
# ============================================================

records = []

pattern = re.compile(
    r"z:\s*([-+]?\d*\.?\d+),\s*selection:\s*\((.*?)\)\s*--\s*(True|False)"
)

with open("logs.txt", "r") as f:
    for line in f:

        m = pattern.search(line)
        if not m:
            continue

        z = float(m.group(1))

        selection_text = m.group(2).strip()

        if selection_text == "":
            selection = []
        else:
            selection = [int(x.strip()) for x in selection_text.split(",")]

        ok = (m.group(3) == "True")

        records.append((z, selection, ok))

# ============================================================
# Build binary matrix
# ============================================================

max_feature = max(max(sel) for _, sel, _ in records if len(sel) > 0)

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

# ============================================================
# Stability & Volatility
# ============================================================

freq = M.sum(axis=1)

stability = freq / n_z

volatility = np.sum(np.diff(M, axis=1) != 0, axis=1)

# sort:
#   stability descending
#   volatility ascending
order = np.lexsort((volatility, -stability))

M = M[order]
stability = stability[order]
volatility = volatility[order]
feature_labels = order

# ============================================================
# Plot
# ============================================================

fig = plt.figure(figsize=(16, 9))

gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[5, 1],
    wspace=0.05
)

ax = fig.add_subplot(gs[0])
ax_bar = fig.add_subplot(gs[1], sharey=ax)

# ------------------------------------------------------------
# Highlight True region
# ------------------------------------------------------------

if len(true_columns):

    start = true_columns[0] - 0.5
    end = true_columns[-1] + 0.5

    ax.axvspan(start, end, color="red", alpha=0.10)

# ------------------------------------------------------------
# Barcode plot
# ------------------------------------------------------------

for i in range(n_features):

    xs = np.where(M[i] == 1)[0]

    if len(xs) == 0:
        continue

    # connect consecutive runs
    splits = np.where(np.diff(xs) > 1)[0] + 1
    runs = np.split(xs, splits)

    for r in runs:
        ax.hlines(
            i,
            r[0],
            r[-1],
            color="black",
            linewidth=3
        )

    ax.scatter(
        xs,
        np.full_like(xs, i),
        color="black",
        s=20,
        zorder=3
    )

# ------------------------------------------------------------
# Axis
# ------------------------------------------------------------

ax.set_ylim(-1, n_features)

step = max(1, n_z // 10)
ticks = np.arange(0, n_z, step)

ax.set_xticks(ticks)
ax.set_xticklabels(
    [f"{z_values[i]:.2f}" for i in ticks],
    rotation=45
)

ax.set_yticks(np.arange(n_features))
ax.set_yticklabels(feature_labels)

ax.set_xlabel("z")
ax.set_ylabel("Feature")

ax.grid(axis="x", alpha=0.2)

# ------------------------------------------------------------
# Stability bar
# ------------------------------------------------------------

ax_bar.barh(
    np.arange(n_features),
    stability,
    height=0.6
)

ax_bar.set_xlim(0, 1)

ax_bar.set_xlabel("Stability")

ax_bar.grid(axis="x", alpha=0.3)

plt.setp(ax_bar.get_yticklabels(), visible=False)

# ------------------------------------------------------------
# Annotate
# ------------------------------------------------------------

for i, s in enumerate(stability):
    ax_bar.text(
        s + 0.02,
        i,
        f"{s:.2f}",
        va="center",
        fontsize=8
    )

plt.suptitle(
    "Sample Stability across z",
    fontsize=16
)

plt.tight_layout()

plt.show()
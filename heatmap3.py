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

with open("logs.txt") as f:
    for line in f:

        m = pattern.search(line)
        if m is None:
            continue

        z = float(m.group(1))

        text = m.group(2).strip()

        if text == "":
            selection = []
        else:
            selection = [int(x) for x in text.split(",")]

        ok = (m.group(3) == "True")

        records.append((z, selection, ok))

# ============================================================
# Build matrix
# ============================================================

max_feature = max(max(sel) for _, sel, _ in records if len(sel))

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
# True region
# ============================================================

true_start = min(true_columns)
true_end = max(true_columns)

# ============================================================
# Statistics
# ============================================================

stability = M.mean(axis=1)

volatility = np.sum(np.diff(M, axis=1) != 0, axis=1)

true_overlap = np.zeros(n_features, dtype=int)

for i in range(n_features):

    row = M[i]

    x = np.concatenate([[0], row, [0]])

    diff = np.diff(x)

    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1

    best = 0

    for s, e in zip(starts, ends):

        overlap = max(
            0,
            min(e, true_end) - max(s, true_start) + 1
        )

        best = max(best, overlap)

    true_overlap[i] = best

# ============================================================
# Ranking
# ============================================================

order = np.lexsort((
    volatility,
    -stability,
    -true_overlap
))

M = M[order]

feature_labels = order

stability = stability[order]
volatility = volatility[order]
true_overlap = true_overlap[order]

# ============================================================
# Plot
# ============================================================

fig = plt.figure(figsize=(17, 9))

gs = fig.add_gridspec(
    1,
    2,
    width_ratios=[5, 1.4],
    wspace=0.05
)

ax = fig.add_subplot(gs[0])
ax_score = fig.add_subplot(gs[1], sharey=ax)

# ------------------------------------------------------------
# Highlight True region
# ------------------------------------------------------------

ax.axvspan(
    true_start - 0.5,
    true_end + 0.5,
    color="red",
    alpha=0.12,
    label="True region"
)

# ------------------------------------------------------------
# Barcode
# ------------------------------------------------------------

for i in range(n_features):

    xs = np.where(M[i] == 1)[0]

    if len(xs) == 0:
        continue

    splits = np.where(np.diff(xs) > 1)[0] + 1
    runs = np.split(xs, splits)

    for r in runs:

        ax.hlines(
            i,
            r[0],
            r[-1],
            color="black",
            linewidth=4
        )

    ax.scatter(
        xs,
        np.full_like(xs, i),
        s=18,
        color="black",
        zorder=3
    )

# ------------------------------------------------------------
# Left axis
# ------------------------------------------------------------

ax.set_yticks(np.arange(n_features))
ax.set_yticklabels(feature_labels)

step = max(1, n_z // 10)

ticks = np.arange(0, n_z, step)

ax.set_xticks(ticks)

ax.set_xticklabels(
    [f"{z_values[t]:.2f}" for t in ticks],
    rotation=45
)

ax.set_xlabel("z")
ax.set_ylabel("Feature")

ax.grid(axis="x", alpha=0.2)

# ------------------------------------------------------------
# Right panel
# ------------------------------------------------------------

ax_score.barh(
    np.arange(n_features),
    true_overlap,
    height=0.6
)

ax_score.set_xlabel("Overlap")

ax_score.grid(axis="x", alpha=0.3)

plt.setp(ax_score.get_yticklabels(), visible=False)

for i in range(n_features):

    ax_score.text(
        true_overlap[i] + 0.2,
        i,
        f"{true_overlap[i]}",
        va="center",
        fontsize=8
    )

# ------------------------------------------------------------
# Print ranking
# ------------------------------------------------------------

print("\nFeature ranking\n")

print("Feature  Overlap  Stability  Volatility")

for f, o, s, v in zip(
        feature_labels,
        true_overlap,
        stability,
        volatility):

    print(
        f"{f:7d} {o:8d} {s:10.3f} {v:11d}"
    )

plt.suptitle(
    "Feature stability around the True region",
    fontsize=16
)

plt.tight_layout()

plt.show()
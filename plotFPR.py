import matplotlib.pyplot as plt
x = [50, 100, 150, 200]
# name = 'lava_lasso'
# y = {
#     "No inference": [1, 1, 1, 1],
#     "Naive":        [0.1333, 0.1583, 0.0916,0.1083],
#     "Bonferroni":   [0.0, 0.0, 0.0, 0.0],
#     "Data splitting":[0.0167,0.0833, 0.0250,0.0333],
#     "SFSaDV-oc":[0.0417, 0.0833,0.0417,0.0333],
#     "SFSaDV":  [0.0667, 0.0667, 0.0333, 0.0167],
# }

name = 'cg_lasso'
y = {
    "No inference": [1, 1, 1, 1],
    "Naive":         [0.1417,0.1333,0.1083, 0.1083],
    "Bonferroni":   [0.0, 0.0, 0.0, 0.0],
    "Data splitting":[0.05,0.0667,0.05,0.075],
    "SFSaDV-oc":[0.0333, 0.0167, 0.05,0.0417],
    "SFSaDV":  [0.0583,0.0667,0.075, 0.033],
}

colors = ["olive", "deeppink", "purple", "orange", "green", "blue"]

for (label, vals), c in zip(y.items(), colors):
    plt.plot(x, vals, "-o", label=label, color=c, markersize=7)


plt.xlabel("# instances")
plt.ylabel("FPR")
plt.xticks(x)
plt.ylim(-.05, 1.05)
plt.legend(loc="center right")
plt.tight_layout()
plt.savefig(f'fpr{name}.pdf')
plt.show()
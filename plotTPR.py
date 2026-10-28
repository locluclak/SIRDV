import matplotlib.pyplot as plt
# x =  [0.5, 1,1.5,2]
name = 'lava_lasso'
# y = {
#     "Bonferroni":    [0.0167, 0.1167,0.3833,0.5833],
#     "Data splitting":[0.1750, 0.2833,0.3917, 0.6083],
#     "SFSaDV-oc":[0.0417,0.0417,0.05,0.05],
#     "SFSaDV":  [0.25,0.5667,0.8667, 0.95],
# }
x =  [0.3, 0.4,0.5,0.6]
xname = "fraction"
# y = {
#     "Bonferroni": [0.3917,0.4500,0.3950,0.3750],
#     "Data splitting": [0.5167,0.5167,0.4083,0.4917],
#     "SFSaDV-oc": [0.05,0.0333,0.0333,0.05],
#     "SFSaDV": [0.7167,0.8333,0.8333,0.8750]
#     # "Bonferroni":    [0.0167, 0.1167,0.3833,0.5833],
#     # "Data splitting":[0.1750, 0.2833,0.3917, 0.6083],
#     # "SFSaDV-oc":[0.0417,0.0417,0.05,0.05],
#     # "SFSaDV":  [0.25,0.5667,0.8667, 0.95],
# }

name = 'cg_lasso'
# y = {
#     "Bonferroni":   [0.0083,0.1167,0.2333,0.3583],
#     "Data splitting":[0.1917,0.3000,0.4083,0.5333],
#     "SFSaDV-oc":[0.05, 0.075,0.05, 0.09],
#     "SFSaDV":  [0.1833,0.4167,0.7083,0.75],
# }
y = {
"Bonferroni":[0.1833,0.2083,0.2500,0.2417],
"Data splitting": [0.5083,0.500,0.4830,0.4750],
"SFSaDV-oc": [0.0667,0.0417,0.0250,0.0333],
"SFSaDV": [0.5583,0.7000,0.7000,0.8000]
}
colors = [ "purple", "orange", "green", "blue"]

for (label, vals), c in zip(y.items(), colors):
    plt.plot(x, vals, "-o", label=label, color=c, markersize=7)


plt.xlabel(xname)
plt.ylabel("TPR")
plt.xticks(x)
plt.ylim(-.05, 1.05)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig(f'fpr{name}.pdf')
plt.show()
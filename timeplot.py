import numpy as np
import matplotlib.pyplot as plt

ns = [35, 85, 135, 185]
nitv = [35, 85, 135, 185]
times = [
    # np.loadtxt(f"ntotal_interval_{n}.txt")
    np.loadtxt(f"output_newnaive/time/timefpr_lavaFS_parame_n{n}_lambda5.txt")
    for n in nitv
]
# plt.figure(figsize=(8, 5))
plt.boxplot(times, tick_labels=[50,100,150,200], showfliers=False)
plt.xlabel("# instances")
# plt.ylabel("# intervals")
plt.ylabel("Time (s)")
plt.savefig('cost_time.pdf')
# plt.title("Runtime vs n")
plt.tight_layout()
plt.show()
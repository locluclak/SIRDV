import matplotlib.pyplot as plt

# Define file paths and corresponding group labels
dv = 'cg-score'
files = [
    f'timefpr_{dv}_parame_n50.txt',
    f'timefpr_{dv}_parame_n100.txt',
    f'timefpr_{dv}_parame_n150.txt',
    f'timefpr_{dv}_parame_n200.txt'
]
labels = [50, 100, 150, 200]
parent_dir = 'output/'
# Read data from files
data = []
for filepath in files:
    with open(parent_dir+filepath, 'r') as file:
        times = [float(line.strip()) for line in file if line.strip()[:20]]
        data.append(times)

# Plot boxplot
plt.figure(figsize=(10, 6))
bp = plt.boxplot(data, patch_artist=True, showfliers=False,)

# Formatting axes and labels
plt.xticks(range(1, len(labels) + 1), labels)
plt.xlabel('Parameter ($n$)', fontsize=12)
plt.ylabel('Time (s)', fontsize=12)
plt.title('Execution Time Distribution', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(f'output/time_{dv}')
plt.show()
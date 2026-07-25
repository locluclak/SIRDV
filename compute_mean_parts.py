from pathlib import Path


path = Path(__file__).with_name("test_eta.txt")
text = path.read_text(encoding="utf-8")

parts = []
current = []

for line in text.splitlines():
    line = line.strip()
    if not line:
        continue
    if line == "-":
        if current:
            parts.append(current)
            current = []
    else:
        try:
            current.append(float(line))
        except ValueError:
            continue

if current:
    parts.append(current)

for i, values in enumerate(parts, 1):
    if values:
        mean_value = sum(values) / len(values)
    else:
        mean_value = float("nan")
    print(f"part {i}: count={len(values)}, mean={mean_value:.6f}")

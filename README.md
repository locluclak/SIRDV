# CG-DataPruning

Selective Inference for DV-score-based Data Pruning.

Compute LAVA-scores for regression samples, perform selective inference to obtain valid p-values, and evaluate data pruning strategies.

## Project Structure

```
CG_datapruning/
├── si_for_cg/              # Core selective inference package
│   ├── cg_function.py      # DV-score computation
│   ├── gram_matrix.py      # ReLU NTK Gram matrix (H∞)
│   ├── si_adapter.py       # DV-score selection adapter for SI
│   ├── si_cg.py            # SelectiveInference & SI_DV entry point
│   ├── si_overconditioning.py  # Overconditioning method
│   ├── si_parametric.py    # Parametric scanning method
│   ├── si_runner.py        # Generic SI runner
│   ├── si_types.py         # Dataclasses: SIConfig, SIResult, SIDecomposition
│   └── si_utils.py         # Quadratic solvers, p-value, decomposition helpers
├── cg_score/               # Standalone DV-score module
├── data/                   # Dataset and transforms
│   ├── regression_dataset.py
│   └── transforms.py       # normalize_targets, normalize_features
├── evaluate/
│   └── pruning.py          # run_pruning_experiment
├── model/
│   └── mlp.py              # MLP regression model
├── utils/
│   ├── args.py             # Argument parser
│   ├── base_trainer.py     # Training loop
│   ├── train.py            # rmse, mae metrics
│   └── visualization.py    # Plotting utilities
├── examples/               # Example scripts
├── experiment/
│   └── run_regression.py   # Full experiment runner
├── config.yaml             # Default configuration
└── output/                 # Generated plots
```

## Dependencies

```
numpy
torch
scikit-learn
scipy
mpmath
matplotlib
```

## Quick Start

```python
import numpy as np
from sklearn.model_selection import train_test_split
from si_for_cg import SI_DV

# Generate null data (beta=0)
rng = np.random.RandomState(42)
X = rng.randn(30, 3)
y = rng.randn(30)

# Normalize
norms = np.linalg.norm(X, axis=1, keepdims=True)
X = X / np.where(norms == 0, 1, norms)
y = 2 * (y - y.min()) / (y.max() - y.min()) - 1

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

# Get one selective p-value
pval = SI_DV(X_tr, y_tr, Xval=X_te, yval=y_te,
             fraction=0.5, jth=0, method="overconditioning")
print(f"p-value: {pval:.4f}")
```

## Usage

### 1. Compute LAVA-scores

```python
import torch
from cg_score.cg_function import calc_lava_score, calc_lava_score_regression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Full dataset (no subsampling)
scores = calc_lava_score(X, y, device, normalize=True)

# With subsampling for large datasets
scores = calc_lava_score_regression(
    X, y, device,
    subsample_size=1000,  # subsample size
    rep_num=10,           # repetitions (averaged)
    sub_term=True         # return sub-terms for analysis
)
vi = scores['vi']  # DV-score per sample
```

### 2. Selective Inference

```python
from si_for_cg import SI_DV, SelectiveInference

# One-liner
pval = SI_DV(X_train, y_train, Xval=X_val, yval=y_val,
             fraction=0.5, method="parametric")

# Or with more control
si = SelectiveInference(fraction=0.5, method="overconditioning")
si.fit(X_train, y_train, Xval=X_val, yval=y_val)

print(si.p_value_)          # p-value
print(si.selected_indices_) # indices kept by DV-score
print(si.intervals_)        # selection-condition intervals
```

**Parameters:**
- `fraction` — fraction of top-scoring samples to keep (default 0.5)
- `method` — `"parametric"` (grid scan) or `"overconditioning"` (quadratic inequalities)
- `jth` — which selected sample to test (random if None)

### 3. Data Pruning

```python
from evaluate.pruning import run_pruning_experiment
from sklearn.linear_model import Ridge

results = run_pruning_experiment(
    X, y, scores,
    prune_fractions=[0.1, 0.2, 0.3, 0.4, 0.5],
    model_fn=lambda: Ridge(alpha=1.0),
    n_seeds=5
)

# Results dict:
#   results['keep_fractions']  — kept data fractions
#   results['rmse_low_prune']  — RMSE after removing low-score samples
#   results['rmse_high_prune'] — RMSE after removing high-score samples
#   results['rmse_random']     — RMSE with random pruning
```

## Examples

```bash
# DV-score computation on synthetic data
python examples/example_regression.py

# Real datasets (california, diabetes)
python examples/example_real_dataset.py --dataset california

# Selective inference p-value validation
python examples/demo_si.py

# Pruning curve visualization
python examples/plot_pruning.py

# Full experiment with config.yaml
python experiment/run_regression.py --dataset california
```

## Configuration

Edit `config.yaml` to set defaults:

| Field | Default | Description |
|-------|---------|-------------|
| `dataset` | california | Dataset name |
| `model` | mlp | Model architecture |
| `epochs` | 200 | Training epochs |
| `lr` | 0.001 | Learning rate |
| `subsample_n` | 1000 | Subsample size for DV-score |
| `rep_num` | 10 | Repetitions for stochastic DV-score |
| `seed` | 42 | Random seed |
| `eval_metric` | rmse | Evaluation metric (rmse/mae) |

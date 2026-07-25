# Architecture

Internal design of the CG-DataPruning project.
If you are AI and try to run this project, use conda env name `IS`
## Overview

```
Input (X, y)
    │
    ▼
┌─────────────────┐
│   DV-score      │   Score each sample by influence on generalization bound
│  (cg_score/)    │
└────────┬────────┘
         │  scores + Gram matrix
         ▼
┌─────────────────┐
│  Selection      │   Keep top-fraction of samples by DV-score
│ (si_adapter.py) │
└────────┬────────┘
         │  selected set S, inverse Gram matrix
         ▼
┌─────────────────┐
│  Selective      │   Condition on selection event → compute valid p-value
│  Inference      │
│ (si_for_cg/)    │
└────────┬────────┘
         │
         ▼
      p-value
```

## DV-score Computation

**File:** `cg_score/cg_function.py`, `si_for_cg/cg_function.py`

The DV-score measures each sample's contribution to the generalization error bound via the ReLU Neural Tangent Kernel (NTK) Gram matrix.

### Gram Matrix H∞

**File:** `si_for_cg/gram_matrix.py`

For L2-normalized inputs, the infinite-width ReLU NTK is:

```
H∞_ij = x_i^T x_j * (π - arccos(x_i^T x_j)) / (2π)
H∞_ii = 1/2
```

Implementation:
1. Compute inner product matrix `inner = X @ X.T`, clamped to [-1, 1]
2. Apply kernel formula elementwise
3. Override diagonal to 0.5

### Score Formula

For sample i, decompose y using the inverse Gram matrix:

```
d_i   = (H⁻¹)_ii                          (diagonal element)
h_i   = (H⁻¹)_{-i, i}                     (column i without entry i)
y_{-i} = y without y_i

CG_i  = (y_{-i}^T h_i)^2 / d_i
        + 2 * y_i * (y_{-i}^T h_i)
        + y_i^2 * d_i
```

This can be rewritten as `(y_{-i}^T h_i / √d_i + y_i √d_i)^2`.

**High DV-score** → influential, hard, or outlier sample
**Low DV-score** → typical or redundant sample

### Subsampling

**File:** `cg_score/cg_function.py` — `calc_lava_score_regression()`

For large datasets (n > ~3000), compute LAVA-scores on random subsamples and average over multiple repetitions. Each sample's score is accumulated and divided by the number of times it appeared in subsamples.

## Selective Inference

The goal: given that DV-score selected a specific set of samples S, compute a p-value for whether a particular sample j ∈ S is truly influential (not just selected by chance).

### Pipeline

**File:** `si_for_cg/si_runner.py`

```
1. adapter.select(X, y)     → observed_selection S
2. decompose_y(y, eta, Sigma) → SIDecomposition (a, b, etaT_y, etaT_Sigma_eta)
3. Compute selection-condition intervals via parametric or overconditioning
4. Compute p-value from truncated normal distribution on those intervals
```

### Adapter Pattern

**File:** `si_for_cg/si_adapter.py` — `SelectionAdapter`

Implements the `SelectionAdapter` protocol with a `select(X, y)` method:
1. Computes LAVA-scores
2. Sorts samples by score
3. Keeps top `fraction` of samples
4. Stores `sorted_indices`, `top_indices`, and `invH` for downstream SI methods

### Line Parameterization Y(z)

**File:** `si_for_cg/si_utils.py` — `decompose_y()`

The test direction eta defines a line in y-space:

```
Y(z) = a + b * z
```

where:
- `b = Sigma @ eta / (eta^T Sigma eta)`
- `a = (I - b @ eta^T) @ y`
- `z = eta^T y` is the scalar test statistic

The observed data corresponds to z = eta^T y. Under the null hypothesis, z follows a truncated normal distribution restricted to the selection region.

### Test Statistic (eta, c)

**File:** `si_for_cg/si_utils.py` — `make_eta_c()`

For testing whether sample j improves validation performance:

1. Build prediction matrices `C_S` and `C_{S\j}` (using selected samples S vs S minus j)
2. Compute validation residuals and their signs
3. `eta = C_S^T sign(r_S) - C_{S\j}^T sign(r_{S\j})` — constraint direction
4. `c = (sign(r_S) - sign(r_{S\j}))^T y_val` — null threshold

The hypothesis test: `eta^T y_train >= c` (including j improves validation loss).

## SI Methods

### Overconditioning

**File:** `si_for_cg/si_overconditioning.py`

Uses the sorted DV-score structure to derive selection conditions directly.

For each sample k, the DV-score as a function of z is quadratic:

```
CG_k(z) = (u_k + v_k * z)^2 / d_k
```

where `u = invH @ a`, `v = invH @ b`, `d = diag(invH)`.

The selection condition requires the sorted order at z=0 to be preserved:

```
CG_{σ(k)}(z) <= CG_{σ(k+1)}(z)   for consecutive pairs in sorted order
```

This gives n-1 quadratic inequalities in z. Their intersection is the valid selection region, solved by `solve_quad_list()`.

**Complexity:** O(n log n) per z evaluation, O(n) inequalities total.

### Parametric Scanning

**File:** `si_for_cg/si_parametric.py`

Approximate method that scans z on a grid from `zmin` to `zmax`:

1. For each z on the grid, compute `y(z) = a + b * z`
2. Run DV-score selection on `y(z)` to get selection S(z)
3. If S(z) equals the observed selection S, record the overconditioning interval at that z
4. Union all matching intervals to form the selection region

**Complexity:** O((zmax - zmin) / step) evaluations, each involving full DV-score computation.

## P-Value Computation

**File:** `si_for_cg/si_utils.py` — `compute_p_value()`

The p-value is computed from a truncated normal distribution on the selection intervals:

```python
sigma = sqrt(eta^T Sigma eta)
for each interval [L, R]:
    denominator += CDF(R/sigma) - CDF(L/sigma)
    if etaT_y in [L, R]:
        numerator = denominator + CDF(etaT_y/sigma) - CDF(L/sigma)
p_value = numerator / denominator  (one-sided)
p_value = 2 * min(cdf, 1-cdf)     (two-sided)
```

Uses `mpmath` with 500 digits of precision for numerical stability in the CDF computation.

## Data Flow

### Normalization

**File:** `data/transforms.py`

- **Features:** Standardized (zero mean, unit std) or L2-normalized per sample
- **Targets:** Min-max scaled to [-1, 1] (required by DV-score theory)

### Pruning Experiments

**File:** `evaluate/pruning.py`

For each prune fraction:
1. Sort samples by DV-score
2. Remove bottom fraction (low-score = easy/redundant samples)
3. Train model on remaining data
4. Evaluate RMSE on hold-out test set
5. Compare against random pruning baseline

Three strategies compared:
- **Prune low:** Remove low-score samples (keep important ones) → RMSE should stay flat
- **Prune high:** Remove high-score samples (remove important ones) → RMSE degrades quickly
- **Random:** Random removal → baseline

## Key Types

**File:** `si_for_cg/si_types.py`

```python
@dataclass(frozen=True)
class SIConfig:
    zmin: float = -20.0       # Grid scan lower bound
    zmax: float = 20.0        # Grid scan upper bound
    step: float = 0.00001     # Grid scan step size
    atol: float = 1e-10       # Numerical tolerance

@dataclass(frozen=True)
class SIDecomposition:
    eta: np.ndarray           # Test direction (n, 1)
    etaT_y: float             # Observed test statistic
    etaT_Sigma_eta: float     # Variance of test statistic
    a: np.ndarray             # Intercept of Y(z) line
    b: np.ndarray             # Slope of Y(z) line

@dataclass(frozen=True)
class SIResult:
    p_value: float            # Selective p-value
    intervals: list[tuple]    # Selection-condition intervals
    etaT_y: float             # Observed statistic
    etaT_Sigma_eta: float     # Variance
    observed_selection: Any   # Selected sample indices
    tested_index: int         # Which sample was tested
    method: str               # "parametric" or "overconditioning"
```

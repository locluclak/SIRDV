# Real-data benchmark: concrete, energy efficiency, garments, wine

Four tabular regression datasets used to check whether the selective-inference
(SI) pipeline in `si_for_cg/` is applicable, and how the Lasso support behaves.

All four are numeric, moderately sized (n between ~770 and ~6500), and have a
target treated as continuous, which is what the SI method assumes. Fits and
diagnostics are produced by:

```
realdata/exp_script/fit_lasso_realdata.py   # script
realdata/result/lasso_realdata.txt          # generated report
```

Run from the repository root (conda env `IS`):

```
& "C:\Users\Asus\anaconda3\envs\IS\python.exe" realdata/exp_script/fit_lasso_realdata.py --probe
```

## Summary

| dataset | file | n | p (used) | target(s) | λ used | \|M\| | R² | cond(X_M) |
|---|---|---|---|---|---|---|---|---|
| Concrete | `concrete_data.csv` | 1030 | 8 | Concrete Compressive Strength | 10 / 100 | 6 / 5 | 0.61 / 0.53 | 3.4 / 2.4 |
| Energy | `energy_efficiency.csv` | 768 | 8 | Y1 heating, Y2 cooling | 10 / 50 | 4–5 / 3 | 0.88–0.91 / 0.87–0.90 | 1.3–6.1 / 1.3 |
| Garments | `garments_worker.csv` | 1197 | 13 | actual_productivity | 25 / 50 | 9 / 7 | 0.265 / 0.240 | 5.8 / 1.8 |
| Wine | `winequality.csv` | 6497 | 11 | quality | 50 / 200 | 9 / 6 | 0.289 / 0.275 | 3.4 / 3.2 |

`p (used)` is the number of features passed to the Lasso after the preprocessing
described below. `cond(X_M)` is the condition number of the selected design
submatrix — the quantity the SI pipeline inverts
(`inv(X_M^T X_M)` in `conditioning_Lasso` and `make_eta_cgFS`).

All four datasets are **good for the SI method numerically**: full-rank
KKT-certified supports, well-conditioned `X_M`, and the `SI_DV` cg-score /
overconditioning path runs end to end.

---

## 1. Concrete Compressive Strength

- **File:** `realdata/dataset/concrete_data.csv`
- **Size:** 1030 rows, 9 columns (8 features + target)
- **Source:** UCI ML Repository, Concrete Compressive Strength (Yeh, 1998)
- **Target:** `Concrete Compressive Strength` (MPa)

**Features (8):** `Cement`, `Blast Furnace Slag`, `Fly Ash`, `Water`,
`Superplasticizer`, `Coarse Aggregate`, `Fine Aggregate`, `Age`.

**Preprocessing:** none beyond z-scoring the 8 feature columns and the target.
Design rank 8 (full), max |corr| = 0.66 (`Water ~ Superplasticizer`), no
collinear clusters. 34 duplicate rows are kept (harmless).

**Lasso fits**

| λ | \|M\| | R² | cond(X_M) | KKT | SI probe |
|---|---|---|---|---|---|
| 10 | 6 | 0.613 | 3.41 | yes | 0.27 |
| 100 | 5 | 0.531 | 2.40 | yes | 0.50 |

Top coefficients (λ=10): `Cement` +0.625, `Age` +0.413, `Blast Furnace Slag`
+0.412, `Water` −0.262, `Fly Ash` +0.224, `Superplasticizer` +0.103.

**Why λ=10/100:** λ=1/5 keep all 8 features (no real selection); λ=10 starts
zeroing weaker terms and λ=100 keeps only the 5 strongest.

**SI suitability — good.** Full rank, low condition number, no collinearity,
KKT-certified, pipeline completes. Caveat: the true model is dense (all 8 mixes
matter), so even at λ=100 the support is 5/8 — the Lasso selection event is mild.

---

## 2. Energy Efficiency

- **File:** `realdata/dataset/energy_efficiency.csv`
- **Size:** 768 rows, 10 columns (8 features + 2 targets)
- **Source:** UCI ML Repository, Energy Efficiency (Tsanas & Xifara, 2012)
- **Targets:** `Y1` (heating load), `Y2` (cooling load). Both are fitted
  separately; **both Y columns are removed from the design** so neither target
  leaks into the other's fit.

**Features (8):** `X1` relative compactness, `X2` surface area, `X3` wall area,
`X4` roof area, `X5` overall height, `X6` orientation, `X7` glazing area,
`X8` glazing area distribution.

**Preprocessing:** z-scoring only. Design rank **7** of 8 — `X1 ~ X2` (corr
0.992, relative compactness vs surface area are physically linked). The two
targets themselves correlate 0.976 (heating and cooling load of the same
buildings).

**Lasso fits**

| target | λ | \|M\| | R² | cond(X_M) | KKT | SI probe |
|---|---|---|---|---|---|---|
| Y1 | 10 / 50 | 4 / 3 | 0.910 / 0.899 | 1.34 | yes | 0.60 / 0.60 |
| Y2 | 10 / 50 | 5 / 3 | 0.880 / 0.868 | 6.14 / 1.34 | yes | 0.90 / 0.78 |

Top coefficients: `X5` (overall height) dominates for both targets, then `X7`
(glazing area) and `X3` (wall area).

**Why λ=10/50:** the design holds only 7 independent directions and a handful of
strong predictors; λ=10 selects 4–5 of them, λ=50 keeps the 3 strongest, both
well-conditioned.

**SI suitability — good.** Sparse full-rank supports, `cond(X_M)` ≤ 6, KKT
certified, pipeline completes for both targets. Caveat: `X1 ≈ X2` keeps design
rank at 7; use one target at a time.

---

## 3. Garments Worker Productivity

- **File:** `realdata/dataset/garments_worker.csv`
- **Size:** 1197 rows, 15 columns
- **Source:** UCI ML Repository, Garment Worker Productivity (Imran et al., 2019)
- **Target:** `actual_productivity` (continuous, range 0.23–1.12)

**Features used (13):** `date`, `quarter`, `department`, `day`, `team`,
`targeted_productivity`, `smv`, `over_time`, `incentive`, `idle_time`,
`idle_men`, `no_of_style_change`, `no_of_workers`.

**Dropped:** `wip` — 506 of 1197 values are missing (42%), so the column is
removed rather than imputing half the table. (`wip` is the only column with
NaNs; the loader also drops any residual NaN rows.)

**Preprocessing:** z-scoring only. Design rank 13 (full), max |corr| = 0.94
(`department ~ no_of_workers`).

**Lasso fits**

| λ | \|M\| | R² | cond(X_M) | KKT | SI probe |
|---|---|---|---|---|---|
| 25 | 9 | 0.265 | 5.80 | yes | 0.23 |
| 50 | 7 | 0.240 | 1.75 | yes | 0.35 |

Top coefficients (λ=25): `targeted_productivity` +0.379, `smv` −0.209,
`no_of_workers` +0.168, `team` −0.147, `idle_men` −0.120, `no_of_style_change`
−0.070, `incentive` +0.040, `date` −0.034.

**Why λ=25/50:** λ≤10 keeps 11–13 features; λ=25/50 give sparse supports while
`X_M` stays well-conditioned (λ≥100 collapses to ~2 features).

**SI suitability — good numerically, one structural caveat.** Full-rank,
KKT-certified supports and the pipeline completes. But the 1197 rows are
team-day records (12 teams × 59 dates): `team` and `date` are grouping/time
identifiers and rows within a team across consecutive days are not independent.
The default `Sigma = I` ignores this. To respect i.i.d., drop `date`/`team` or
pass a team-level block `Sigmat`. `department ~ no_of_workers` (0.94) is
moderate collinearity but harmless at λ ≥ 25.

---

## 4. Wine Quality

- **File:** `realdata/dataset/winequality.csv`
- **Size:** 6497 rows, 13 columns (red + white wines combined)
- **Source:** UCI ML Repository, Wine Quality (Cortez et al., 2009)
- **Target:** `quality` — an **ordinal integer score 3–9** (mean 5.82); treated
  as continuous, which is the standard regression use of this dataset.

**Features used (11):** `fixed acidity`, `volatile acidity`, `citric acid`,
`residual sugar`, `chlorides`, `free sulfur dioxide`, `total sulfur dioxide`,
`density`, `pH`, `sulphates`, `alcohol`.

**Dropped:** `type` (non-numeric `red`/`white`, per the numeric-only convention).
This discards a strong red-vs-white signal; one-hot `type` would raise R² but is
excluded to keep the preprocessing consistent with the other datasets.

**Preprocessing:** z-scoring only. Design rank 11 (full), max |corr| = 0.72
(`free sulfur dioxide ~ total sulfur dioxide`), no collinear clusters.

**Lasso fits**

| λ | \|M\| | R² | cond(X_M) | KKT | SI probe |
|---|---|---|---|---|---|
| 50 | 9 | 0.289 | 3.43 | yes | — |
| 200 | 6 | 0.275 | 3.15 | yes | — |

Top coefficients (λ=50): `alcohol` +0.442, `volatile acidity` −0.255, `total
sulfur dioxide` −0.118, `residual sugar` +0.105, `sulphates` +0.101, `free sulfur
dioxide` +0.098.

**Why λ=50/200:** λ ≤ 20 keeps all 11 features (no real selection); λ=50 starts
zeroing the weakest terms and λ=200 keeps only the 6 strongest, both full rank
and well-conditioned.

**SI suitability — good numerically, one modelling caveat.** Full-rank,
KKT-certified supports, `cond(X_M)` ≈ 3, no collinearity, and a quick SI smoke
test (`--quick`, cg-score, λ=10) completes for all three methods
(overconditioning / parametric_nested / datasplit, 0 failures). Caveat: the
target is a discrete ordinal score with heavy ties near 5–6, and dropping `type`
leaves the fit weak (R² ≈ 0.29, similar to garments); the SI guarantees assume a
continuous response.

---

## Conventions

- **Preprocessing** (in the script): keep numeric columns only (non-numeric are
  dropped), exclude the target(s) from the design, then z-score every design
  column and the target to unit variance. This makes `fit_intercept=False`
  inside `solve_Lasso` equivalent to an intercept model and makes λ comparable
  across datasets.
- **Lasso:** sklearn `Lasso(alpha = λ / n)`, i.e. the KKT bound is
  `|X_j^T r| ≤ λ`. λ=1/5 (the original setting) are far too weak for these
  standardized designs and keep essentially all features; the per-dataset λ
  above were chosen so the support is sparse but non-empty.
- **Support certification:** each fit is re-solved through
  `si_for_cg.si_utils.solve_Lasso`, which validates the primal KKT residual; a
  support is reported as certified only if it passes.
- **SI probe:** one `SI_DV(..., data_valuation="cg-score",
  method="overconditioning")` p-value per (dataset, λ). The tested coefficient
  is picked at random (`jth=None`), so the probe p-values vary run to run — they
  demonstrate that the pipeline completes on real data, not calibration.
- **Not included here:** `parkinsons_updrs.csv` (repeated measures per subject,
  i.i.d. violated) and `onlinenewspopularity.csv` (heavily skewed target, weak
  signal; only feasible at subsample scale). `paddydataset.csv` is unusable
  as-is (numeric design has effective rank 7 of 36). All are still handled by
  the script for reference.

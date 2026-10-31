# Integrate forward fixed-K FS (seqfs) as a feature-selection option beside Lasso

## Goal

Add a feature-selection option `fs_method='seqfs'` — **forward feature selection with a
fixed number of features K** — as an alternative to the existing default `fs_method='lasso'`.
It must work for **both fit** (choosing the tested feature set `M`) **and p-value inference**
(the conditioning event), for:

- `data_valuation` in `{'lava', 'cg-score'}`
- `method` in `{'overconditioning', 'parametric_nested'}`

Lasso stays the default; existing behaviour is unchanged when `fs_method='lasso'`.

## Locked decisions

1. **Option name/API**: `SIConfig.fs_method ∈ {'lasso' (default), 'seqfs'}` and
   `SIConfig.K: int = 5` (fixed feature count; used only when `fs_method='seqfs'`).
   Mirrored on `SI_DV(...)` and `SelectiveInference.__init__`. `K` may not exceed the
   number of candidate features `p`.
2. **SI methods supported with seqfs**: `overconditioning` and `parametric_nested`, for
   both DV methods. `method="parametric"` (grid scan) remains Lasso-only — raise a clear
   `ValueError` if `fs_method='seqfs'` and `method='parametric'`.
3. **What changes**: only the *feature* selection step and its conditioning event.
   Sample/row selection (lava OT sort / cg-score sort) is untouched, and the test statistic
   `eta = make_eta_cgFS/make_etaFS(X, M, j)` (built from `X[:, M]`, `jth` = position in `M`)
   is untouched. `solve_Lasso` in `si_utils.py:620` is the thing being swapped out.
4. **FS conditioning formula**: port `seqfs/overconditioning.interval_SFS` verbatim in
   semantics (over-conditioning on the exact observed greedy path). No homotopy/LARS walk
   is added; the nested scanner uses restart-and-solve per merged selection span (when
   `fs_method='seqfs'`, ignore `config.lasso_path`).

## Current flow (for reference)

- `examples/demo_si.py` → `SI_DV` / `SelectiveInference.fit` (`si_for_cg/si_dv.py`):
  1. `adapter.select(...)` → `observed_selection` (rows). 2. `solve_Lasso(Xtrain[observed_selection],
  ytrain[observed_selection], lambda_)` → `M, Mc, sM` (`si_dv.py:375`). 3. `jth` = position in
  `M`; `eta = make_eta_cgFS(...)`. 4. `selective_inference(...)` (`si_runner.py:15`).
- `selective_inference` → `decompose_y` (line `y(z)=a+bz`) → interval routine → `compute_p_value`.
- Lasso-support conditioning event = `conditioning_Lasso` (`si_overconditioning.py:203`), used in:
  - `overconditioning_intervals` lava (`si_overconditioning.py:185`)
  - `CG_overconditioning_intervals` cg-score (`si_overconditioning.py:101`)
  - `parametric_intervals_FS_nested` level-3 (`si_parametric.py:262-295`, and the
    `walk_lasso_support_regions` path at `:249-261`)
  - `cg_parametric_intervals_FS_nested` level-2 (`si_parametric.py:420-451`, path at `:406-419`)

## seqfs reference (source of truth to port)

- `seqfs/ForwardSelection.py`:
  - `fixedSelection(Y, X, k)` — greedy forward selection, min RSS (note **Y-first** arg order).
    Intermediate sets are `selection[:step]`, so the full path is derivable in one pass.
  - `list_residualvec(X, Y)` — returns `lst_SELEC_k` (sets for k=0..p) and `lst_Portho`
    (n×n projectors). Only the selection list is needed; skip the projectors (expensive).
- `seqfs/overconditioning.py`:
  - `interval_SFS(X, Y, K, lst_SELEC_k, lst_Portho, a, b)` — the FS conditioning interval
    (`lst_Portho` is passed but unused). Solves `g3 z^2 + g2 z + g1 <= 0` for each
    step/candidate and intersects. Uses `solve_quadratic_inequality(a,b,c)` /
    `interval_intersection` — both already exist in `si_for_cg/si_utils.py` (`:229`, `:186`).
  - `OC_fixedFS_interval(...)` = `interval_DA ∩ interval_SFS`. The `interval_DA` half is
    already handled by the existing OT/sort machinery, so it is **not** ported.

## Ordered tasks

1. **New module `si_for_cg/si_seqfs.py`**
   - `forward_selection_path(Y, X, K) -> tuple[list[list[int]], np.ndarray]`:
     single-pass greedy forward selection (min RSS, tie-break = smallest feature index for
     determinism); returns `path` with `path[0]=[]`, `path[step]=selection after step`
     (length `K`), and the residual vector at step `K`. Reuses `seqfs.ForwardSelection`
     logic but Y/X shapes as used in this repo (`X` = candidate design rows,
     `Y` = response column). Guard `K <= X.shape[1]` and `K <= X.shape[0]`.
   - `interval_FS(X_, a, b, K, path) -> list[tuple[float,float]]`:
     faithful port of `interval_SFS`, with `lst_SELEC_k=path`. Use
     `X_jk.T @ X_jk` inverse via `np.linalg.inv` inside `try/except np.linalg.LinAlgError`
     → on singular return `[]` (degenerate; caller treats as a stale step). Use
     `si_utils.solve_quadratic_inequality` and `si_utils.interval_intersection`.
   - `fs_conditioning_interval(X_, a, b, z, K) -> tuple[np.ndarray, list[tuple[float,float]]]`:
     compute `Y(z) = (a + b*z).reshape(-1)`; `path, _ = forward_selection_path(Y, X_, K)`;
     `M_z = np.array(sorted(path[-1]))`; `interval = interval_FS(X_, a.reshape(-1),
     b.reshape(-1), K, path)`; return `(M_z, interval)`. This is the FS analogue of
     `solve_Lasso` + `conditioning_Lasso` bundled together.
   - Docstring must state the over-conditioning semantics (conditions on the exact observed
     greedy path being optimal at every step; not the minimal final-set event).

2. **`si_for_cg/si_types.py` — `SIConfig`**
   - Add `fs_method: str = "lasso"` and `K: int = 5`. Keep dataclass shape backward compatible.

3. **`si_for_cg/si_overconditioning.py`**
   - In `overconditioning_intervals` (lava), replace the `conditioning_Lasso(...)` call at
     `:185` with a branch: if `config.fs_method == 'seqfs'`, use
     `fs_conditioning_interval(Xtrain[observed_selection], a_vec[observed_selection],
     b_vec[observed_selection], decomposition.etaT_y, config.K)` and take its interval
     (assert the returned `M_z` matches `fs.M`; if not, fall back to `[]`). Otherwise keep
     `conditioning_Lasso`.
   - In `CG_overconditioning_intervals` (cg-score), same branch at `:101` using
     `X[observed_selection]`, `a_vec[observed_selection]`, `b_vec[observed_selection]`.
   - Keep the existing "pick the interval containing `curr`" logic unchanged (this satisfies
     the "if many intervals, choose the one containing z" requirement).

4. **`si_for_cg/si_parametric.py` — nested scanners**
   - `parametric_intervals_FS_nested` (lava), inside the
     `for z_entry, sl, sr, selection in coalesce_selection_regions(regions)` loop
     (`:247`): if `fs_method == 'seqfs'`, **skip** the `config.lasso_path` branch and use
     the restart-and-solve loop (mirroring `:262-295`) but with
     `M_z, fs_itv = fs_conditioning_interval(Xtrain[selection], a_vec[selection],
     b_vec[selection], zl, config.K)`; append `fpiece` when `np.array_equal(fs.M, M_z)`.
   - `cg_parametric_intervals_FS_nested` (cg-score), same change inside its selection-span
     loop (`:404`) using `X[selection]`, `a_vec[selection]`, `b_vec[selection]`.
   - When `fs_method=='lasso'`, code paths are byte-for-byte the current behaviour
     (including the `walk_lasso_support_regions` fast path).

5. **`si_for_cg/si_dv.py`**
   - `SelectiveInference.__init__`: add `fs_method: str = "lasso"`, `K: int = 5`; validate
     `fs_method in {'lasso','seqfs'}`; reject `fs_method=='seqfs' and method=='parametric'`;
     pass `fs_method`/`K` into `SIConfig`.
   - `SelectiveInference.fit`: branch the feature-selection block at `:375`.
     - lasso (default): unchanged `solve_Lasso`.
     - seqfs: `path, _ = forward_selection_path(ytrain_arr[observed_selection].reshape(-1),
       Xtrain_arr[observed_selection], self.K)`; `M = np.array(sorted(path[-1]))`,
       `Mc = None`, `sM = None`; store into `self.lasso` (`LassoInfo.M = M`).
     - Keep the empty-`M` guard and `tested_position = jth or random(range(len(M)))`.
   - `SI_DV(...)`: add `fs_method="lasso"`, `K=5` parameters and forward them to
     `SelectiveInference`. Update `__all__` if needed.
   - Leave `NaiveTest`, `FSNaiveTest`, `DataSplittingTest` Lasso-only (out of scope).

6. **`si_for_cg/__init__.py`**
   - Optionally export `forward_selection_path` / `fs_conditioning_interval` for experiments.

7. **Validation hook (required)**
   - Add a small entry (extend `examples/demo_si.py` or a new `experiment/` script) that
     runs one dataset through `SI_DV(..., fs_method='seqfs', K=5)` for each
     `(data_valuation, method)` pair and asserts a finite p-value is produced; and that
     `method='parametric'` + `fs_method='seqfs'` raises `ValueError`.

## Data flow (seqfs branch)

`observed_selection` (rows, unchanged) → forward FS fixed K on `Xtrain[observed_selection]`
→ `M` (size K), `jth` position in `M` → `eta` (unchanged `make_eta_cgFS`) →
`decompose_y` → `y(z)=a+bz` → interval routine:
- lava OC: `OT_interval ∩ sort_interval ∩ fs_conditioning_interval(...)`; pick piece containing `z_obs`.
- cg OC: `sort_interval ∩ fs_conditioning_interval(...)`; pick piece containing `z_obs`.
- `parametric_nested`: per merged sort span, restart-and-solve
  `fs_conditioning_interval` and union pieces where `M_z == M` → `compute_p_value`.

## Risks / edge cases

- **Singular `X_jk.T @ X_jk`** in `interval_FS` (K up to `p`, selected rows): wrap in
  `try/except np.linalg.LinAlgError` and return `[]`; the nested loops already handle an
  empty interval via their stale counters.
- **Over-conditioning gap**: `interval_SFS` captures only the region where the *observed
  greedy path* stays optimal; if a later z reaches the same final `M` through a different
  path, those pieces are excluded. This matches `conditioning_Lasso`'s over-conditioning
  conservativeness and is intentional (document it).
- **`interval_SFS` cost**: it loops `O(K·p)` inversions of growing matrices per call, and
  the nested scan calls it per z. Acceptable for v1; note `solveinterval`-style vectorization
  or a numba port as future work (no numba path in this change).
- **`config.lasso_path`**: must be ignored when `fs_method=='seqfs'` (no FS homotopy walk);
  ensure the nested branch does not enter `walk_lasso_support_regions`.
- **`fs.M` shape**: keep it a sorted `np.ndarray[int]` so `np.array_equal` and `X[:, M]`
  work identically to the Lasso path.
- **`lambda_` unused** under seqfs (leave `LassoInfo.lambda_` as-is; it is simply ignored).
- **Determinism/tie-breaking** in `forward_selection_path` must be explicit (smallest index)
  so the observed `M` and the scan agree.

## Validation plan

- Static: import `si_for_cg` under env `IS` (`C:\Users\Asus\anaconda3\envs\IS\python.exe`),
  run `examples/demo_si.py` (or the new experiment script) from the **repo root**.
- Correctness of the event: at `z = etaT_y`, the returned interval must contain `z` and
  `compute_p_value` must not raise "Selection intervals must contain the observed etaT_y".
- FPR sanity: run `fs_method='seqfs'` on the null-data loop (n small, e.g. demo config) for
  the 4 `(data_valuation, method)` combinations; p-values should be in `[0,1]` and roughly
  uniform across a handful of seeds; compare against the Lasso baseline qualitatively.
- Regression: `fs_method='lasso'` (default) must reproduce current outputs on one demo run.

## Out of scope

- `method='parametric'` (non-nested grid scan) with seqfs.
- Numba acceleration of the FS conditioning interval.
- Changes to `NaiveTest` / `FSNaiveTest` / `DataSplittingTest`.
- Deleting or refactoring the `seqfs/` reference folder (kept as-is).

## Handoff

Implementation requires source edits in `si_for_cg/`; switch to an implementation-capable
agent. Verification must use conda env `IS` and run scripts from the repo root.

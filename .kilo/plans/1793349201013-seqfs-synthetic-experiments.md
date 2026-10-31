# Synthetic FPR/TPR experiments for the seqfs (fixed-K forward selection) approach

## Goal

Add a reproducible experiment harness that measures **FPR** and **TPR**, per-iteration
**running time**, and per-iteration **number of selection intervals** for the *seqfs*
feature-selection approach, across both data valuations (`lava`, `cg-score`) and four SI
methods (Naive, overconditioning, Data-splitting, parametric_nested). Raw p-values are
persisted so they can be replotted later.

## Locked decisions (from user)

1. **All four methods use seqfs** for the tested feature set `M`. Because `FSNaiveTest`
   ("naive") and `DataSplittingTest` ("split") in `si_for_cg/si_dv.py` are currently
   Lasso-only, they must be extended with `fs_method`/`K`. This needs small **source edits**.
2. **Interval count** = `len(intervals_)` for the iteration (exactly what `compute_p_value`
   consumes). Naive/split use the fixed single interval `[(-inf, inf)]`, so their count is 1.
3. **FPR data**: `si_utils.generate(n, 15, d, beta=0, beta=0)` → standard-normal X/noise,
   identity `Sigma`. `n ∈ {50,100,150,200}`, `d=15`, `fraction=0.5`.
4. **TPR data**: `si_utils.generate_contaminated_regression(n_train=100, n_val=15, n_test=0,
   p=15, contamination=0.5, beta_clean=beta, std_good=1.0, std_bad=2.0)`; pass `Sigmat`/`Sigmav`.
   Sweep A: `beta ∈ {0.5,1.0,1.5,2.0}` (n=100, fraction=0.5).
   Sweep B: `beta=1.5`, `fraction ∈ {0.3,0.4,0.5,0.6}` (n=100).
5. **seqfs K=5**, `iters=120` per configuration, `alpha=0.05`, `jth=None` (random tested
   position inside `M`), `lambda_=5`, both `data_valuation ∈ {lava, cg-score}`.
6. **Output dir**: new `output_seqfs/` at repo root. Files append (`"a"`) and flush →
   resumable. Filenames encode setting so plot scripts can find them by name.

## Affected files

- `si_for_cg/si_dv.py` — extend `FSNaiveTest` and `DataSplittingTest` (source edit).
- `experiment/seqfs_experiment.py` — **new** main harness (FPR + both TPR sweeps).
- `experiment/validate_seqfs.py` — extend smoke assertions to cover naive/split (source edit).
- `experiment/seqfs_plot.py` — **new** optional reader/plotter.
- Outputs under `output_seqfs/` (gitignored area like the other `output*` dirs).

## Ordered tasks

### 1. Extend `si_for_cg/si_dv.py` (`FSNaiveTest`, `DataSplittingTest`)

- Add params `fs_method: str = "lasso"` and `K: int = 5` to both `__init__`s; store on `self`.
- In `FSNaiveTest.fit` (currently `si_dv.py:183-190`) and `DataSplittingTest.fit`
  (`si_dv.py:88-96`), branch the `M` computation:
  - `fs_method == 'seqfs'`: `path, _ = forward_selection_path(y_pruned.reshape(-1),
    X_pruned, self.K)`; `M = np.array(sorted(path[-1]), dtype=int)`. (`forward_selection_path`
    is already imported at `si_dv.py:16`.)
  - else: keep the existing `Lasso(...)` / `beta_hat` / `np.nonzero` path byte-for-byte.
- Keep the existing `len(M) == 0` guard and `tested_position = jth or random(range(len(M)))`.
- Add `self.intervals_ = intervals` (the `[(-inf,inf)]` list already built) in both classes so
  the harness records interval counts uniformly. Keep `NaiveTest` (non-FS) untouched.
- `make_naive_etaFS(X_pruned, y_pruned, M, j)` and `make_eta_cgFS` are unchanged and already
  take a position `j` inside `M`.

### 2. New `experiment/seqfs_experiment.py`

Structure (mirror `experiment/fpr_experiment.py` / `tpr_experiment.py` conventions):

- Header comment: run from **repo root** with `& "C:\Users\Asus\anaconda3\envs\IS\python.exe"
  experiment/seqfs_experiment.py`; document the `output_seqfs/` mapping.
- Constants: `D=15, K=5, FRACTION=0.5, ALPHA=0.05, ITERS=120, N_VAL=15`,
  `CONTAMINATION=0.5, STD_GOOD=1.0, STD_BAD=2.0, LAMBDA_=5.0`.
- Method table (`tag` used in filenames → runner factory):
  - `naive` → `FSNaiveTest(fraction, data_valuation, fs_method="seqfs", K=K)`
  - `overco` → `SelectiveInference(method="overconditioning", fs_method="seqfs", K=K,
    lambda_=LAMBDA_, data_valuation=dv)`
  - `parame` → `SelectiveInference(method="parametric_nested", fs_method="seqfs", K=K,
    lambda_=LAMBDA_, data_valuation=dv, lasso_path=False)`
  - `split` → `DataSplittingTest(fraction, data_valuation, split_ratio=0.2,
    fs_method="seqfs", K=K)`
- `run_one(method, cfg, Xtr, ytr, Xval, yval, Sigmat, Sigmav)`:
  - `contextlib.redirect_stdout(io.StringIO())` around the call to suppress LP/`print` noise.
  - `t0=time.time()`; build+`fit`; `dt=time.time()-t0`.
  - For SI methods use the `SelectiveInference` object and read `si.p_value_`, `si.intervals_`;
    for naive/split read `runner.p_value_`, `runner.intervals_`.
  - Return `(pvalue, n_intervals, dt)`.
- Config loops (as in the existing scripts, with `seedd` retry loop until `iters` successes):
  - `fpr`: `dv ∈ {lava,cg-score}` × `n ∈ {50,100,150,200}` × methods;
    data = `generate(n, N_VAL, D, zeros, zeros, seed=seedd)`; pass `Sigmat=St, Sigmav=Sv`.
  - `tpr_beta`: dv × `beta ∈ {0.5,1.0,1.5,2.0}` × methods; data = contaminated regression
    (n=100, fraction=0.5).
  - `tpr_fraction`: dv × `fraction ∈ {0.3,0.4,0.5,0.6}` × methods (beta=1.5, n=100).
- Per configuration write three aligned, one-value-per-line files under `output_seqfs/`:
  - `seqfs_pv_<tag>.txt`  ← p-value
  - `seqfs_nint_<tag>.txt` ← `n_intervals`
  - `time/time_<tag>.txt` ← `dt` seconds
  where `<tag>` = e.g. `fpr_lava_n50_naive`, `tpr_lava_beta0.5_overco`,
  `tpr_lava_frac0.3_split`. (`seqfs` prefix keeps them distinct from the Lasso files.)
- Exception handling: on error, log `seed, method, cfg, type(exc), traceback` to
  `output_seqfs/seqfs_errors.log`, increment an error counter, and retry with the next seed
  (do **not** silently write a p-value). Print `errors/total` in the summary.
- Printed summary per config: `valid`, `rejected`, FPR/TPR, `errors`, mean time, mean
  `n_intervals`. Naive uses the existing Bonferroni threshold `alpha / 2**D`; OC/parame/split
  use `pv <= alpha` (raw p-values are saved, so corrections can be recomputed at plot time).
- CLI (for smoke testing): `--exp {fpr,tpr_beta,tpr_fraction,all}`, `--iters`, `--K`, `--d`,
  `--outdir`. Defaults reproduce the locked settings.

### 3. Extend `experiment/validate_seqfs.py`

- After the existing SI checks, also run `FSNaiveTest(fraction=0.5, data_valuation=dv,
  fs_method="seqfs", K=3)` and `DataSplittingTest(0.5, dv, fs_method="seqfs", K=3)` for both
  DV methods; assert a finite p-value in `[0,1]` and that `intervals_` is a non-empty list.
- Keep the lasso-default regression check.

### 4. New `experiment/seqfs_plot.py` (optional reader/plotter)

- Read `output_seqfs/seqfs_pv_*.txt` by filename, print FPR-vs-n and TPR tables, and draw the
  same figures as `fpr_plot.py` / `tpr_plot.py` (one curve per method, per DV).

## Validation / test in small scale

1. Static import under env `IS`; run `experiment/validate_seqfs.py` — all checks pass, incl.
   the new naive/split seqfs cases.
2. Smoke: `& "...\IS\python.exe" experiment/seqfs_experiment.py --exp all --iters 3` from the
   repo root. Verify: `output_seqfs/` created; the three files per config exist and have 3
   lines each; p-values finite in `[0,1]`; `nint` values are integers ≥1; times > 0.
3. Sanity: null (FPR) p-values roughly uniform on the extra seeds; alternative (TPR) p-values
   shift small as `beta` grows.
4. **Halt-and-ask rule**: if the smoke run shows anything strange — frequent
   `Empty feature selection set`, `Parametric selection region does not contain etaT_y`,
   LP/degeneracy errors, all-NaN p-values, or unexpected interval counts — **stop and report
   the exact message/traceback to the user** rather than broadening retries or masking it.

## Risks / notes

- **Naive/split seqfs semantics**: extending the baselines changes their selection from Lasso
  to forward FS; this is the requested fair comparison, but it is a behavioral change (only
  when `fs_method="seqfs"` is passed; default `"lasso"` paths unchanged).
- **`interval_FS` cost**: `O(K·p)` inversions per call, and parametric_nested calls it per
  z-step → the `parame` rows are the slow ones; expect long runtimes at `iters=120` × all
  configs. Run `--exp` subsets as needed.
- **Degenerate intervals** return `[]`; parametric_nested may raise
  `RuntimeError("Parametric selection region does not contain etaT_y")` — treated as a skipped
  (retried) seed and logged, not written.
- **lava** requires the patched scipy `linprog(...).basis` (per AGENTS.md); the smoke run will
  fail loudly if the patch is missing.
- **Tested-position randomness**: `jth=None` draws a random position in `M` per iteration
  (existing behaviour); reproducibility across runs is not guaranteed, by design.

## Out of scope

- Changing Lasso-based behaviour for `fs_method="lasso"`.
- Numba acceleration of `interval_FS`.
- `method="parametric"` (non-nested) with seqfs.
- Replotting the existing Lasso experiments.

# AGENTS.md

Selective inference for DV-score data valuation. Active code is the numpy/SciPy package `si_for_cg/` and the scripts in `experiment/`. The README tree is stale: `data/`, `model/`, `utils/`, `evaluate/`, `cg_score/` are older scaffolding; `evaluate/pruning.py` and `calc_lava_score_regression` no longer exist.

## Environment (non-negotiable)

- Use conda env `IS` (`C:\Users\Asus\anaconda3\envs\IS\python.exe`; numpy 2.4.6, scipy 1.17.1). The default `python` has an ABI mismatch — never use it.
- The `lava` path REQUIRES a patched scipy `linprog(method='simplex')` that returns `res.basis`; stock scipy raises `AttributeError` at runtime. Patch with `python replace_scipy_linprog.py --env anaconda --dir files_to_replace` (env activated; patch must match the installed scipy version). Sanity check: `linprog(...).basis` must exist.
- `si_for_cg/LPsolver.py` (HiGHS re-implementation) is dead code — `OptimalTransport.solveOT` calls the patched scipy `linprog`; do not wire the HiGHS solver back in.

## Two selection pipelines

Selected by `data_valuation` in `SIConfig` / `SI_DV(..., data_valuation=...)`. Every experiment hardcodes `dv = "lava"` or `"cg-score"`.

- `lava` — `SelectionAdapter.select(Xtrain, ytrain, Xval, yval)` → `si_for_cg/lava_function.calc_lava_score`, an OT LP over stacked train+val. Requires validation data and the patched linprog.
- `cg-score` — `CGSelectionAdapter.select(X, y)` → `si_for_cg/cg_function.py:calc_cg_score`, NTK Gram-matrix inverse on train only, no validation data, no LP.

Quirk in both adapters: `fraction <= 1` is a proportion, `fraction > 1` is an absolute keep count (`si_adapter.py:29`).

## SI pipeline and method traps

Flow: `examples/demo_si.py` → `si_for_cg/si_dv.py` (`SI_DV`, `SelectiveInference`, `NaiveTest`/`FSNaiveTest`/`DataSplittingTest`) → `si_runner.selective_inference` → `si_overconditioning.py` / `si_parametric.py`. `examples/demo_sicg.py` is the cg-score analog.

- `method="overconditioning"` is the working path (fast, deterministic from the observed LP basis). `method="parametric"` is a slow z-grid scan.
- `DEBUG.md` (Vietnamese) documents an open bug: OC solved at `z_obs` vs at another `z` in the interval can return different LP bases/intervals. Do not "fix" it by re-solving OT at `z` — the observed basis must be kept (see `gptsession.md`).
- `jth` is a POSITION inside the Lasso active set `M`, not a global feature index: `si_dv.py` passes it as `j` into `make_eta_cgFS`. `tpr_paired_search.py` re-implements the pipeline (`SI_DV_jth`) to work around this — read its docstring before touching pairing logic.
- The active set comes from `solve_Lasso` (`si_utils.py:380`), KKT-validated with retries; exceptions are expected and experiment scripts catch-and-skip them. `mp.dps = 500` in `si_utils.py` is load-bearing for p-value CDFs.
- Public API is re-exported from `si_for_cg/__init__.py`; add new exports there.

## Experiments (`experiment/`)

- Run from the repo root with `& "C:\Users\Asus\anaconda3\envs\IS\python.exe" experiment/<script>.py`. Scripts prepend the repo root to `sys.path`; running from `experiment/` breaks imports.
- `tpr_*experiment*.py` / `fpr_experiment.py` loop over configs, append one p-value per line to `output*/tpr_...txt` / `fpr_...txt` (`"a"` mode → resumable), append timings to `output3/time*.txt`, and skip failed seeds.
- Filenames encode the setting (`tpr_lava_overco_beta1.0.txt`, `fpr_lavaFS_parame_n50_rho0.2.txt`, `diag_split_con0.5_stdbad3_beta1.0.txt`). Plot scripts (`*_plot.py`, `ksplot.py`) read these exact names and hardcoded dirs — renaming a file/prefix breaks the plots.
- Output dir mapping lives in each script's header comments: `output4` paired_search.csv, `output5` diag/paired, `output6/7/8/9/10/11` FPR/TPR sweeps (std_bad/d/lambda/con/fraction differ per dir).
- `experiment/diagnose_split.py --part zdiag|split|param|oracle|summary|paramdiag [--iters N]` is the FPR/TPR anomaly diagnostic.
- LP solver prints noise (`begin`, `end time LP:`) into stdout on every call; suppress with `contextlib.redirect_stdout` as `tpr_paired_search.py` does.

No test suite or lint config — verification is running the examples/experiments in `IS`. Commit style uses short checkpoint messages.
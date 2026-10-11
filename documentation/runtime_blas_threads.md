# BLAS thread oversubscription in the parametric trace (runtime fix)

Companion to `README.md` §8.2 / §12. This document records a **performance
bug** found in the active `parametric_nested` path: the LAVA OT basis trace
(`trace_parametric_bases`, `si_for_cg/si_parametric.py`) was dominated by
`np.linalg.inv` on a small `(nt+nv)×(nt+nv)` matrix, not by its mathematics,
because the environment's OpenBLAS is multithreaded and the trace runs two
Python worker threads.

Fix: `si_for_cg/blas_threads.py` (`pin_blas_threads`), called at package import
from `si_for_cg/__init__.py`, plus a solve-based reduced-cost formulation and a
single `H[:, non_basis]` gather in `_basis_optimal_region` / `_pivot_basis`.

Measured effect on the `examples/demo_si.py` LAVA run (n=200, m=15, d=15,
`fraction=0.5`, `lambda_=5`, seed 1, `SI_USE_NUMBA=1`):

| | baseline | after fix |
|---|---|---|
| WALL (reset → report) | **20.91 s** | **5.82 s** (3.6×) |
| `_basis_optimal_region` (all calls) | 31.16 s | 2.29 s |
| `trace_parametric_bases (L+R)` section | 16.35 s (78%) | 1.38 s (24%) |
| p-value | `0.912870126184417` | `0.9128701261844124` |

The p-value moves only in the ~15th digit (the `inv → solve` rounding); the
numba parity harness `experiment/test_numba_parity.py` passes (16/16).

---

## 1. Symptom

`_timeprof` on the demo (see §8 for how to reproduce) showed the trace as the
single dominant term:

```
section                                         excl      incl   calls   %wall
trace: _basis_optimal_region (all calls)      31.156    31.156     367  148.99
nested: trace_parametric_bases (L+R)          16.345    16.345       1   78.16
fit: adapter.select (lava LP)                  2.732     2.732       1   13.07
nested: sort walk + lasso loop (total)         1.423     1.819       1    6.80
WALL (reset -> report)                        20.911                    100.00
```

(`_basis_optimal_region` is summed over both worker threads, hence >100% wall;
the `L+R` wrapper measures the real 16.35 s wall of the threaded section.)

### 1.1 Per-pivot breakdown

Instrumenting the body of `_basis_optimal_region` (367 pivots, `B` is 214×214,
`non_basis` has 2786 columns):

| part | default BLAS | `OPENBLAS_NUM_THREADS=1` |
|---|---|---|
| `np.linalg.inv(B)` | **71.7 ms** | 1.97 ms |
| `H[:, non_basis]` gather (done 3×) | 5.7 ms | 5.2 ms |
| `solveinterval` | 1.3 ms | 1.0 ms |
| entering-candidate loop | 1.3 ms | 1.2 ms |
| **per pivot** | **79.9 ms** | **9.4 ms** |

`inv` is ~90% of the per-pivot cost. For a 214×214 matrix that is absurd: the
actual work is `~k^3/3 ≈ 3.3 MFLOP` (forming the inverse, ~10 MFLOP), i.e.
sub-millisecond on any modern core.

### 1.2 Isolated `np.linalg.inv(214×214)` benchmark

| condition | single-shot | under 2 concurrent Python threads |
|---|---|---|
| default (multithreaded OpenBLAS) | 11.7 ms | 15.7 ms / inv |
| `OPENBLAS_NUM_THREADS=1` | 0.9 ms | 0.5 ms / inv |

The trace uses exactly two Python threads (the left/right directions), so the
in-run `inv` cost is the contended column: ~15–72 ms instead of < 1 ms.

---

## 2. Root cause

This environment's OpenBLAS (0.3.31) is built with

```
MAX_THREADS=128  NO_AFFINITY=1  USE_OPENMP=0
```

so every `np.linalg.inv` (and other BLAS/LAPACK calls) spawns up to
`MAX_THREADS` threads and joins them, with no affinity pinning. For a tiny
matrix the spawn/join/barrier overhead dwarfs the arithmetic, and running the
trace's two Python workers multiplies the contention.

`threadpoolctl` cannot fix this here: `threadpool_info()` returns `[]` and
`threadpool_limits(1)` is a **no-op** for this build, so the thread count can
only be changed via the `*_NUM_THREADS` environment variables *before* the BLAS
loads, or by calling `openblas_set_num_threads` on the loaded library.

**This is not an algorithmic problem.** Numba (N5 in `NUMBA_PLAN.md`) would call
the *same* LAPACK `inv`, so it would not remove this cost — the plan's
"BLAS-bound, partial" note was right, and the earlier framing of "N5 targets the
78%" was wrong.

---

## 3. Fix implemented

### 3.1 `si_for_cg/blas_threads.py` — `pin_blas_threads(n=1)`

- Sets `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` / `MKL_NUM_THREADS` /
  `NUMEXPR_NUM_THREADS` / `VECLIB_MAXIMUM_THREADS` (via `setdefault`, for
  libraries not yet loaded).
- Locates the loaded OpenBLAS/MKL shared library and calls
  `openblas_set_num_threads(n)` / `MKL_Set_Num_Threads(n)` through `ctypes`,
  which works **after** numpy has imported the BLAS (the env vars alone do not).
- Idempotent; disabled with `SI_BLAS_THREADS=0`; count overridable with any
  `SI_BLAS_THREADS=<int>`.

### 3.2 Package wiring — `si_for_cg/__init__.py`

`pin_blas_threads()` is called at import, before the submodule imports, so every
entry point that imports `si_for_cg` is covered.

### 3.3 `si_for_cg/si_parametric.py` — reduced costs without an inverse

In `_basis_optimal_region`:

- `pi_c = c[basis] @ B^{-1}` is replaced by the equivalent solve
  `pi = np.linalg.solve(B.T, column_stack((u[basis], v[basis], w[basis])))`,
  removing `np.linalg.inv(B)`.
- `H[:, non_basis]` is gathered once and reused for all three reduced-cost
  row vectors (was gathered three times).

In `_pivot_basis`: the `B_inv` argument becomes the basis matrix `B`, and
`B_inv.dot(·)` becomes `np.linalg.solve(B, ·)` for the entering direction and
the basic solution (`trace_parametric_bases` passes the returned `B` through).

---

## 4. Result

After the fix (same demo run):

```
section                                              excl      incl    %wall
fit: adapter.select (lava LP)                       2.658     2.658    45.63
trace: _basis_optimal_region (all calls)            2.292     2.292    39.37
nested: sort walk + lasso loop (total)              1.404     1.770    24.10
nested: trace_parametric_bases (L+R)                1.381     1.381    23.72
WALL (reset -> report)                              5.823              100.00
```

The trace is no longer dominant; the LAVA LP `adapter.select` (2.66 s) and the
sort walk + Lasso layer are now the largest terms. The trace still runs two
threads, now at ~1.66× overlap (was ~1.9× on inflated numbers).

Next levers, in order of remaining cost: LAVA LP `adapter.select`, the sort
walk + Lasso layer, then (a distant) N5 `trace_parametric_bases`.

---

## 5. Parity and reproducibility

- `experiment/test_numba_parity.py --require-numba`: **16 passed, 0 failed,
  PASS** (kernel unit parity + reference-vs-numba integration parity for both
  `lava` and `cg-score`, plus determinism).
- LAVA demo p-value: `0.912870126184417` → `0.9128701261844124` (~5e-15). The
  single-gather change is bit-exact; only the `inv → solve` substitution perturbs
  rounding, well inside the harness `atol=1e-9`.
- Pinning the BLAS thread count makes runs **deterministic within a process**
  (removes the thread-count-dependent rounding), but the *absolute* value of any
  run is now the single-thread value. To reproduce pre-fix output byte-for-byte,
  set `SI_BLAS_THREADS=0` **and** revert the `inv → solve` hunk (keep the single
  gather).
- If bit-exact reproduction of old `output*/` files is required, the thread pin
  is the only change that is safe on its own; the two must be considered
  together because the pin also changes BLAS reduction order.

---

## 6. cg-score

The trace fix does **not** apply to cg-score: `cg_parametric_intervals_FS_nested`
has no OT layer, so `_basis_optimal_region` / `_pivot_basis` / `np.linalg.inv`
are never called on that path. The BLAS pin still applies globally, but cg's
BLAS calls are small and few, so the benefit is mild.

cg nested, n=200, warm numba cache:

| | pinned | `SI_BLAS_THREADS=0` |
|---|---|---|
| WALL | 0.355 s | 0.436 s (~+18%) |
| p-value | `0.7548538716358` | `0.7548538716367` (~1e-12) |

The cg p-value is thread-count-sensitive at the 1e-12 level (pre-existing
threaded rounding in `conditioning_Lasso` / `solve_Lasso`); pinning removes that
variability. cg cost structure (pinned, warm) is data-dependent:

| n | WALL | sort walk `iter_cg_sort_regions` | coalesce | Lasso layer |
|---|---|---|---|---|
| 200 | 0.358 s | 0.258 s (72%) | 0.019 s | 0.081 s (23%) |
| 600 | 4.799 s | 2.138 s (45%) | 0.488 s | 2.172 s (45%) |
| 1000 | 1.270 s | 0.770 s (61%) | 0.140 s | 0.360 s (28%) |

cg is co-dominated by the already-numba sort walk and the Lasso layer; the
counts (sort regions, merged spans) vary strongly with the data/seed, so at n=600
cg is Lasso-bound while at n=1000 it is not. The intra-run levers are the same
dead ends as lava (sort walk already numba N1; remaining Python is `solve_Lasso`
N9). cg is ~16× cheaper than lava at n=200 because the expensive parts (LAVA LP
+ trace) do not exist on that path.

**Note:** the first cg run in a fresh process can appear ~3 s slower — that is
the numba JIT compilation of `_conditioning_lasso_kernel` (cache miss), not BLAS
and not this fix.

---

## 7. Configuration

| variable | default | meaning |
|---|---|---|
| `SI_BLAS_THREADS` | `1` | BLAS/MKL thread count to pin; `0` disables pinning |
| `SI_USE_NUMBA` | off | enable the numba kernels (`README.md` §12) |

`pin_blas_threads()` is import-time and idempotent; there is no per-call overhead.

---

## 8. Reproducing the measurements

Run from the repo root with the `IS` environment (`AGENTS.md`); LAVA needs the
patched scipy `linprog`.

```powershell
$py = "C:\Users\Asus\anaconda3\envs\IS\python.exe"

# LAVA demo with the profiler (pinned by default)
$env:SI_USE_NUMBA = "1"; & $py -u examples\demo_si.py

# same, unpinned (to see the oversubscription)
$env:SI_BLAS_THREADS = "0"; & $py -u examples\demo_si.py; Remove-Item Env:SI_BLAS_THREADS

# parity + determinism
& $py experiment\test_numba_parity.py --require-numba
```

The per-pivot and isolated-`inv` tables (§1) come from a temporary monkeypatch
that times the parts of `_basis_optimal_region` with `time.perf_counter`, and
from timing `np.linalg.inv` on a random 214×214 matrix with and without
`OPENBLAS_NUM_THREADS=1`. Both are throwaway scripts; the durable artifacts are
`si_for_cg/blas_threads.py` and the `si_parametric.py` change.

---

## 9. Invariants / do-nots

1. **Do not "fix" this with numba (N5).** Numba calls the same LAPACK `inv`;
   the cost is thread overhead, not Python.
2. **Do not re-add `np.linalg.inv(B)`** in `_basis_optimal_region` /
   `_pivot_basis`; the solve form is equivalent and avoids the contended call.
3. **`OPENBLAS_NUM_THREADS` must be set before the BLAS loads** to take effect;
   `threadpoolctl` is a no-op in this environment, and `os.environ` set after a
   BLAS call does nothing — use `pin_blas_threads` (ctypes) instead.
4. The BLAS pin is global to the process. It is neutral-to-positive for the
   active pipeline (LAVA LP is single-threaded simplex; `constructOMEGA` is
   numba), but any code that needs multithreaded BLAS for large ops should set
   `SI_BLAS_THREADS=0`.

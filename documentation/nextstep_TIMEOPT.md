# Next-step runtime optimization plan

Companion to [`README.md`](README.md) §8 / §12 and
[`runtime_blas_threads.md`](runtime_blas_threads.md). After the BLAS
oversubscription fix, the LAVA `parametric_nested` demo
(`examples/demo_si.py`, n=200, m=15, d=15, `fraction=0.5`, `lambda_=5`, seed 1,
`SI_USE_NUMBA=1`) runs in **5.82 s** (was 20.91 s). This document lists the
remaining runtime levers, ranked by value versus risk.

---

## 0. Current cost profile (post-fix)

```
section                                         excl      incl    %wall
fit: adapter.select (lava LP)                  2.658     2.658    45.63
trace: _basis_optimal_region (all calls)       2.292     2.292    39.37
nested: sort walk + lasso loop (total)         1.404     1.770    24.10
nested: trace_parametric_bases (L+R)           1.381     1.381    23.72
WALL (reset -> report)                         5.823              100.00
counters: OT intervals 366, sort regions 14141, merged selection spans 778,
          lasso iterations 787
```

`_basis_optimal_region` is summed over the two trace worker threads, so it
exceeds the `L+R` wall (1.38 s); the "effective" trace share of wall is ~24%.

Where the LP cost is (isolated, nt=200, nv=15):

| step | s |
|---|---|
| `OptimalTransport.convert(nt, nv)` (constraint matrix) | 0.004 |
| `constructOMEGA` | 0.163 |
| `solveOT` → `linprog(method="simplex")` | **2.376** |

So the LP cost is essentially scipy's pure-Python revised simplex on a
3000-variable, 215-constraint OT problem, solved once per `SI_DV` call.

---

## 1. Experiment-level seed parallelism — biggest, zero risk

The `tpr_*` / `fpr_*` drivers (`experiment/tpr_experiment.py`,
`experiment/fpr_experiment.py`) loop over independent seeds; each iteration is a
full, self-contained pipeline. Running iterations in parallel
(process/thread pool over seeds) gives near-linear throughput on this
16-core-class host.

- **Gain:** ~Nx throughput across the sweep.
- **Risk:** none (iterations share no state; each writes its own line).
- **Cost:** wrapper around the existing loop; keep the existing
  `contextlib.redirect_stdout` suppression.
- **Note:** does nothing for a single p-value; it is the throughput lever.
  Recorded as `throughput_lever_experiment_level`.

---

## 2. Cross-OT sort-walk parallelism — within one run, ~20% wall

The sort walk (`nested: sort walk + lasso loop` ~1.4 s, 24%) fans out over the
366 OT intervals, which are **fully independent**: each has its own fixed LP
basis `B` and seeds its own score-quadratic table
(`iter_sort_regions`, `si_sort_walk.py:273`). The coalesce step is
order-preserving in `z`, so it must stay sequential after the per-OT walks are
concatenated in OT order (which is `z` order).

- **Plan:** fan out `iter_sort_regions` over `list_da_itv` on a thread pool,
  collect per-OT region lists, concatenate in order, then
  `coalesce_selection_regions` as today.
- **Blocker:** `si_numba.py` walk kernels are `@njit(cache=True,
  fastmath=False)` with **no `nogil=True`**; add `nogil=True` so the two-plus
  walk threads actually overlap.
- **Gain:** sort walk ~1.4 s → ~0.15-0.3 s on many workers; ~1.1-1.2 s wall
  (~20%).
- **Risk:** low/mechanical; outputs identical because regions are clipped to
  `[ml, mr]` and the union/coalesce is order-independent. Verify with
  `experiment/test_numba_parity.py`.
- **Caveats:** load imbalance (OT intervals carry very different region counts)
  — use dynamic chunks; do **not** use a process pool (pickling + BLAS
  oversubscription).

---

## 3. Trace micro-optimizations — ~0.5-1 s, exact

After pinning the BLAS thread count, the largest residual inside
`_basis_optimal_region` (`si_parametric.py:624`) is the `H[:, non_basis]`
gather (~5.2 ms/pivot; the 214×214 work itself is now <2 ms).

- **3a. Drop the gather.** Reduced costs are
  `rc_c = c[non_basis] - pi_c · H[:, non_basis]`. Computing `H.T @ pi`
  (a `(nt+nv-1)×3` RHS gemm producing all rows) and subtracting avoids the
  ~4.8 MB fancy-index copy plus three gemv. Exact reformulation.
- **3b. Numba the entering-candidate loop.** The candidate scan
  (`si_parametric.py:680`) is a pure-Python loop over `non_basis`
  (~2786 iterations/pivot). Folding it into a `@njit(nogil=True)` kernel (N5)
  removes interpreter overhead. Non-parallel (`parallel=False`): the loop is
  memory/bandwidth-bound, not arithmetic-bound.
- **Gain:** ~0.5-1 s wall combined.
- **Risk:** low; 3a is algebraically identical, 3b is pure numerics; both
  already covered by the parity harness.

---

## 4. Reuse `constructOMEGA` — cheap

`constructOMEGA` is built once inside `solveOT` (`OptimalTransport.py:84`) and
again in the runner precompute (`si_parametric.py:184`). Build it once and pass
the matrix through.

- **Gain:** ~0.16 s.
- **Risk:** none.
- **Note:** OMEGA is also memory-heavy (`nt·nv × (nt+nv)`), so one fewer copy
  also reduces peak memory.

---

## 5. The LAVA LP itself — 46%, algorithmic, risky

`solveOT` (`OptimalTransport.py:83`) calls the patched scipy
`linprog(method="simplex")` (`:100-109`) and reads `res.basis`. This is the
single serial cost. Options:

- **5a. `revised simplex` with a basis-returning patch (low effort).** Extend
  the scipy patch to also return a basis from `_linprog_rs` and A/B it. Both
  scipy simplex implementations are Python-level, so the gain is uncertain.
- **5b. Transportation / network simplex, or basis from optimal potentials
  (high effort, high risk).** The OT LP is a balanced transportation problem;
  a network simplex (basis = spanning tree of the bipartite graph) is far faster
  than a generic simplex on 3000 variables, and a basis can in principle be
  derived from HiGHS optimal potentials via complementary slackness. But this
  changes *which* degenerate optimal basis is used, and the observed basis
  drives the entire trace (the `DEBUG.md` degeneracy discipline). Any change
  here must be proven by the parity harness to preserve the selection region and
  p-value, and it interacts with the "keep the observed basis" invariant.

**This is the only lever that breaks the ~3.5-4 s floor** for a single lava run.

---

## 6. Not worth it

- **Lasso layer** (~0.37 s at n=200; `solve_Lasso` 0.32 s / 787 iterations) —
  too small; and `N9` (numba `solve_Lasso`) only matters for cg-score at large
  merged-span counts.
- **Process pools for the trace** — `H` would be pickled per task while each
  worker's BLAS oversubscribes; the 2-thread split is the right granularity.
- **Numba for the trace `inv`** — the cost was thread spawn/join overhead, not
  Python; already fixed by pinning (see `runtime_blas_threads.md`).
- **cg-score `conditioning_Lasso`** — co-dominated by the already-numba sort
  walk; no cheap win at the sizes measured.

---

## 7. Ranking and recommendation

| # | lever | scope | expected | risk |
|---|---|---|---|---|
| 1 | Experiment-level seed parallelism | sweep throughput | ~Nx | none |
| 2 | Cross-OT sort-walk parallelism (+`nogil`) | single run | ~1.1 s (~20%) | low |
| 3 | Trace gather→gemm + numba entering loop | single run | ~0.5-1 s | low |
| 4 | Reuse `constructOMEGA` | single run | ~0.16 s | none |
| 5 | LP solver (network simplex / potentials basis) | single run | up to ~2 s | high |

Recommended order: **1** (throughput, no risk), then **4** and **3** (cheap,
exact), then **2** (bigger, still exact), and treat **5** as a separate,
parity-gated investigation. Items 2-4 together target ~5.8 s → ~3.5-4 s within a
single run; item 1 is the correct answer for the FPR/TPR sweeps regardless.

# Numba conversion plan — `parametric_nested` SI (lava + cg-score)

Scope: speed up the hot numeric kernels of `method="parametric_nested"` in
`examples/demo_si.py` for **both** `data_valuation="lava"` and
`data_valuation="cg-score"`, by porting the inner loops to `numba` `@njit`,
without changing the algorithm (same conditioning event, same observed-basis
discipline, same intervals and p-value).

Non-goals / invariants (do not break — see `DEBUG.md`, `gptsession.md`):

- The observed LP basis `adapter.basis` must be kept as the seed of the
  parametric trace; never re-solve OT at `z` to "fix" degeneracy.
- The nested event is `{z : Lasso support M(z) == fs.M}` only; the top-k
  sample selection is not part of the event (`si_parametric.py` docstrings).
- `si_utils.mp.dps = 30` and the tail-form `compute_p_value` stay as-is.
- The `lava` path requires the patched scipy `linprog(...).basis`; numba must
  not replace `linprog`/`solveOT`.
- Keep `si_for_cg/LPsolver.py` dead (it is deleted and `OptimalTransport.py`
  still imports it — see the import workaround in the test script).

---

## 1. Current call flow (`parametric_nested`)

```
demo_si.py
  └─ SI_DV(...)                                  si_dv.py:444
       └─ SelectiveInference.fit(...)            si_dv.py:320
            ├─ adapter.select(...)               si_adapter.py  (LP / Gram)
            ├─ solve_Lasso(...)                  si_utils.py:619  (sklearn)
            ├─ make_eta_cgFS(...)                si_adapter.py:160
            └─ selective_inference(...)          si_runner.py:15
                 ├─ decompose_y(...)             si_utils.py:594
                 ├─ lava : parametric_intervals_FS_nested(...)   si_parametric.py:140
                 │    ├─ constructOMEGA / c_ / wt,rt,ot          OptimalTransport.py:27
                 │    ├─ trace_parametric_bases  (L+R, threads)  si_parametric.py:717
                 │    │    ├─ _basis_optimal_region              si_parametric.py:571
                 │    │    └─ _pivot_basis                       si_parametric.py:662
                 │    └─ per OT interval:
                 │         iter_sort_regions                     si_sort_walk.py:271
                 │           ├─ score_coefficients               si_sort_walk.py:57
                 │           └─ iter_sort_regions_table (walk)   si_sort_walk.py:206
                 │                ├─ pair_diffs / region_containing / quad_pieces
                 │                ├─ flip_reversal               si_sort_walk.py:158
                 │                └─ _resort fallback            si_sort_walk.py:192
                 └─ cg-score : cg_parametric_intervals_FS_nested(...)  si_parametric.py:318
                      └─ iter_cg_sort_regions                   si_sort_walk.py:316
                           ├─ cg_score_coefficients             si_sort_walk.py:287
                           └─ iter_sort_regions_table (walk)    si_sort_walk.py:206
       per sort region (both paths):
         └─ Lasso sub-walk (si_parametric.py:231 / :370)
              ├─ solve_Lasso(...)                       si_utils.py:619  (sklearn)
              └─ conditioning_Lasso(...)                si_overconditioning.py:201
                   └─ solve_linear_inequality(...)      si_utils.py:208
```

The two `parametric_nested` entry points are re-exported in
`si_for_cg/__init__.py`; keep those signatures unchanged.

---

## 2. Hot-path inventory and numba classification

| # | Function | File:line | Called | Numba feasible | Tier | Notes |
|---|----------|-----------|--------|----------------|------|-------|
| N1 | `iter_sort_regions_table` walk | `si_sort_walk.py:206` | once per OT interval (lava) / once total (cg), loops over all sort regions | **Yes** | 1 | Pure numpy + Python per-region overhead. Hottest loop. Generator → convert to fill-and-return kernel. |
| N1a | `pair_diffs` | `si_sort_walk.py:73` | per region | Yes | 1 | Fold into N1. |
| N1b | `quad_pieces` | `si_sort_walk.py:86` | per region | Yes | 1 | Fold into N1. |
| N1c | `region_containing` | `si_sort_walk.py:138` | per region | Yes | 1 | Fold into N1. |
| N1d | `flip_reversal` | `si_sort_walk.py:158` | per region | Yes | 1 | Fold into N1. |
| N1e | `_resort` / `_resort_cg` fallback | `si_sort_walk.py:192` / `:304` | on degenerate ties | Yes | 1 | Needs `select_with_known_basis` + `sort_conditioning` inside kernel (both numba-able). |
| N2 | `conditioning_Lasso` | `si_overconditioning.py:201` | per Lasso piece | **Yes** | 1 | Small dense inversions, called very often. |
| N2a | `solve_linear_inequality` | `si_utils.py:208` | per `conditioning_Lasso` | Yes | 1 | Fold into N2. |
| N3 | `constructOMEGA` | `OptimalTransport.py:27` | once per lava run | **Yes** | 1 | Python `np.append`/`hstack` build loop → O(n²m+nm²) with copies. |
| N4 | `solveinterval` | `si_utils.py:271` | trace + `sort_conditioning` | Yes | 1 | Already vectorized; numba removes temporaries and call overhead. |
| N5 | `trace_parametric_bases` | `si_parametric.py:717` | 2 threads per lava run | Partial | 2 | Dominated by BLAS matmuls (`pi.dot(H)`); numba wins only overhead + reuse. |
| N5a | `_basis_optimal_region` | `si_parametric.py:571` | per pivot | Partial | 2 | `np.setdiff1d` is numba-supported (0.68); `np.linalg.inv` OK. |
| N5b | `_pivot_basis` | `si_parametric.py:662` | per pivot | Yes | 2 | `B_inv.dot`, ratio test, `np.isclose`. |
| N6 | `score_coefficients` | `si_sort_walk.py:57` | once per OT interval | Yes | 2 | scipy `lu_factor`/`lu_solve` → `np.linalg.solve` in numba. |
| N6a | `cg_score_coefficients` | `si_sort_walk.py:287` | once per cg run | Yes | 2 | Trivial; fold into N1 cg seed. |
| N7 | `sort_conditioning` | `si_overconditioning.py:13` | fallback + OC path | Yes | 2 | Uses scipy LU → `np.linalg.solve`. |
| N8 | `select_with_known_basis` | `si_adapter.py:81` | fallback | Yes | 2 | `np.linalg.solve` + `np.argsort(stable)`. |
| N9 | `solve_Lasso` | `si_utils.py:619` | per Lasso piece | **Optional** | 3 | sklearn coordinate descent + KKT. Re-implementing in numba is high-value but must reproduce the exact support/`sM` and KKT certificate. |
| — | `compute_p_value` | `si_utils.py:522` | once | **No** | — | mpmath 30-dps, keep Python. |
| — | `solveOT` / `linprog` | `OptimalTransport.py:80` | per lava select | **No** | — | Patched scipy simplex; keep. |
| — | `decompose_y`, `make_eta*` | `si_utils.py` / `si_adapter.py` | once | No | — | One-shot, not hot. |

Expected ROI order: **N1 ≫ N2 ≈ N9 ≈ N3 > N5 > N4/N6/N7**.

---

## 3. Numba constraints and risks

1. **numba ↔ numpy compatibility (RESOLVED in preflight).** The `IS` env
   currently ships **numba 0.68.0** with **numpy 2.5.3** (scipy 1.17.1): a
   trivial `@njit` kernel using `np.linalg.inv` compiles and runs, so numba is
   usable without changing the environment. (The earlier note that the env had
   numpy 2.4.6 is stale; verify with the harness preflight, which now prints
   `numba`, `numpy` and a `njit` probe.) Still re-check after any env update:
   `njit` must compile `np.linalg.inv/solve` and `np.argsort`. If a future
   numba/numpy bump breaks this, options are: (a) install a matching numba;
   (b) run the parity harness in a dedicated env; (c) fall back to Cython.
2. **scipy is not numba-able.** `lu_factor`/`lu_solve` (N6/N7) and
   `linprog`/`solveOT` must stay out of kernels. Phase-0 confirmed that a
   `from scipy.linalg import lu_factor` inside `@njit` fails with
   `UnsupportedBytecodeError` (IMPORT_NAME). Inside numba use
   `np.linalg.solve` on the `(m, 3)` RHS — measured bit-identical to
   `np.linalg.solve` outside (max diff 0) and `np.linalg.inv` within ~1e-17.
3. **sklearn is not numba-able.** `solve_Lasso` (N9) stays Python unless
   deliberately re-implemented; the Lasso sub-walk is therefore a Python loop
   that calls numba kernels N2 per piece.
4. **No Python objects in `njit`.** `SIConfig`, `LassoInfo`, `SIDecomposition`,
   adapter instances, lists-of-tuples and dicts must be unpacked to plain
   arrays/scalars at the wrapper boundary.
5. **Generators.** `iter_sort_regions_table` is a generator; numba 0.68 does
   support `yield` in nopython mode (Phase-0 OK), but the plan still converts
   it to a fill-and-return kernel for explicit output buffers and counts.
6. **Ops to rewrite / verify:** `np.setdiff1d` is in fact **supported** by
   numba 0.68 (Phase-0 OK), so the N5a complement rewrite is optional; keep it
   only if it is faster. `np.isclose(atol/rtol)` and
   `np.argsort(kind="stable"/"mergesort")` both compile (Phase-0 OK). f-strings
   in `raise` and rich exception formatting remain limited — return status
   codes instead. `np.linalg.inv/solve`, `np.flatnonzero`, `np.argmin/argmax`,
   `np.sqrt`, `np.isfinite/isinf`, `np.where/concat/sort`, `column_stack/vstack`
   are all supported.
7. **Determinism / accuracy:** use `@njit(cache=True, fastmath=False,
   boundscheck=False, nogil=True)`. Never `fastmath=True` (breaks parity and
   reproducibility). Keep `parallel=False` initially; the L/R trace parallelism
   stays at the Python `ThreadPoolExecutor` level and requires `nogil=True` so
   the two `njit` traces actually overlap.
8. **Variable-length output:** the number of sort regions per OT interval is
   bounded by the roots of the `nt-1` adjacent-pair quadratics, i.e.
   `≤ 2*(nt-1)+1`; add `MAX_STALE` (1000) for numerical-gap steps. Preallocate
   `cap = 2*nt + MAX_STALE + 16`; on overflow the wrapper grows and retries.
9. **Caching:** `cache=True` writes `.nbi/.nbc` next to the module; needs write
   access to `si_for_cg/` (fine) and is invalidated by numba/numpy version
   changes.
10. **First-call compile cost:** measure and report separately from steady-state
    (warmup call excluded from timings).

---

## 4. Proposed module layout and dispatch

Add a new module (no changes to existing algorithms):

- `si_for_cg/si_numba.py`
  - `@njit` kernels named `n_*` (see §5).
  - Pure-Python wrappers `w_*` that unpack dataclasses/adapters → arrays, call
    the kernel, repack into the existing list-of-tuples return type.
  - A module-level `AVAILABLE: bool` set by the Phase-0 preflight.

Dispatch (chosen per kernel after parity passes, not globally at once):

- Add a single flag, e.g. `si_for_cg/si_numba.USE = os.environ.get("SI_USE_NUMBA") == "1"`.
- At each call site (`si_parametric.py`, `si_sort_walk.py`,
  `si_overconditioning.py`, `OptimalTransport.py`) insert:
  `if si_numba.USE and si_numba.AVAILABLE: return si_numba.w_xxx(...)`.
- The original pure-Python function stays as the reference implementation and
  is used when the flag is off or numba is unavailable.

This keeps `import si_for_cg` working without numba and makes A/B parity trivial.

---

## 5. Kernel designs

### N1 — sort-region walk (`iter_sort_regions_table`)

Inputs (all arrays/scalars): `s_coef (n,3)`, `ml`, `mr`, `step`, `n_keep`,
`max_stale`, plus the fallback context (lava: `S_`, `wt`, `rt`, `ot`, `basis`,
`nt`, `fraction`; cg: none, fallback is table re-eval).

Outputs (preallocated): `z_entry[k]`, `sl[k]`, `sr[k]`, `sel_flat[k, n_keep]`,
`n_out`. Also `stats[regions, inferred, fallbacks, gaps]`.

Algorithm: mirror `iter_sort_regions_table` exactly — seed `pi` by
`argsort(s_coef · [z²,z,1])`, compute `pair_diffs`, `region_containing` (via a
fused `quad_pieces`), emit the piece, then `flip_reversal` at `R=sr`,
continuity-check `piece2` and `|piece2[0]-R| ≤ TOL_CONT·max(1,|R|)`, else call
the internal fallback. All tolerances (`TOL`, `TOL_CONT`, `MAX_STALE`, the
`1e-8/1e-9` in `flip_reversal`, the `1e-9` coefficient cuts in `quad_pieces`)
must be copied verbatim.

### N2 — `conditioning_Lasso`

Fuse the existing body with `solve_linear_inequality`; return the single
`(left, right)` interval or an empty status. Reuse `np.linalg.inv` (numba).
Inputs: `X_`, `a (n,1)`, `b (n,1)`, `Lambda`, `M`, `Mc`, `sM`.

### N3 — `constructOMEGA`

Build the `(n·m, n+m)` matrix directly with nested loops (no `np.append`).
Verify `np.array_equal` with the reference on small `(n,m)`.

### N4 — `solveinterval`

Port the four branches (empty / all / lower-upper / hole subtraction) with the
same relative snap `1e-9*(1+scale)`. Output as `(k,2)` array + count. This is
the same contract validated by `experiment/compare_ineq_solvers.py`; reuse that
script's ground-truth grid check for the numba version.

### N5 — `trace_parametric_bases`

Port `_basis_optimal_region` (its `np.setdiff1d` compiles in numba 0.68; use
`np.linalg.inv`) and `_pivot_basis` (ratio test + Bland tie-break). Return the
basis-path as arrays `(left[k], right[k], basis[k, m])`. Keep the
`epsilon = max(1e-9, min(1e-7, |boundary|*1e-10))` step exactly.
Call it with `nogil=True` so the existing `ThreadPoolExecutor(max_workers=2)`
still overlaps left/right.

### N6/N7/N8 — coefficient tables, `sort_conditioning`, `select_with_known_basis`

Replace scipy `lu_factor`/`lu_solve` with `np.linalg.solve`. Preserve
`scale = 1 + 1/(nt-1)` and the `scores - scores.sum(axis=0)` calibration
verbatim.

### N9 (optional) — `solve_Lasso`

Re-implement coordinate descent + the KKT certificate in numba, returning
`M`, `Mc`, `sM`. Gate behind a strict parity test (same support, same `sM`,
KKT residual ≤ the sklearn path's `1e-9·max(1,Lambda)`). Only do this after
Tier 1 is green; it is the single largest remaining Python cost when many
Lasso pieces are visited.

---

## 6. Phasing

- **Phase 0 — preflight (no source changes).** Confirm `import numba` under the
  installed numpy; compile a trivial `@njit`; confirm `np.linalg.inv/solve` and
  `np.argsort(kind="mergesort")` compile. Record numba/numpy versions.
  *Status: DONE via `experiment/numba_preflight.py` — numba 0.68.0 / numpy
  2.5.3, 16 threads. All required primitives compile: `np.linalg.inv`/`solve`
  (inv within 1e-17 of numpy, multi-RHS `solve` bit-identical),
  `np.argsort(kind="stable"/"mergesort")`, `flatnonzero`, `argmin/argmax`,
  `isfinite/isinf/sqrt`, `where/concat/sort/column_stack/vstack`, `isclose`,
  `raise`, tuple returns, `nogil=True`, `parallel=True/prange`, njit
  generators. `np.setdiff1d` also compiles. Only `scipy.linalg.lu_factor`
  inside `njit` is unsupported (must use `np.linalg.solve`).* Baseline
  reference timings collected by the parity harness (cg ≈ 0.37 s, lava ≈ 1.27 s
  at n=40, m=15, p=6).
- **Phase 1 — Tier 1 kernels** N1, N2, N3, N4 (+ N1 fallback internals). These
  give the bulk of the speedup and are all pure numerics. Parity per kernel,
  then flip the flag for the cg path (smallest blast radius), then lava.
- **Phase 2 — Tier 2** N5–N8. Watch for BLAS-bound kernels (N5) where numba may
  be neutral; keep only if measured faster.
- **Phase 3 — optional** N9.

Each phase: keep reference path, run `experiment/test_numba_parity.py`, commit
only when parity is green and speedup is positive.

---

## 7. Accuracy test plan (implemented in `experiment/test_numba_parity.py`)

1. **Kernel unit parity** on randomized inputs (well-conditioned + near-zero
   stress regimes, mirroring `compare_ineq_solvers.py`):
   - `constructOMEGA`: exact `array_equal`.
   - `solveinterval` / `quad_pieces`: list equality of pieces (endpoints
     `rtol=1e-9, atol=1e-9`), plus the independent interior/exterior grid
     ground-truth check.
   - `conditioning_Lasso`, `score_coefficients`, `cg_score_coefficients`:
     `allclose(rtol=1e-9, atol=1e-9)`.
   - `iter_cg_sort_regions` / `iter_sort_regions`: identical region count,
     matched `(sl,sr)` within `1e-9`, identical `selection` sets, identical
     `stats` counters.
2. **Integration parity** (reference vs numba via `SI_USE_NUMBA`):
   - cg-score: `SelectiveInference(..., data_valuation="cg-score",
     method="parametric_nested")` on fixed seeds; compare `intervals_`
     (endpoint `atol=1e-9`) and `p_value_` (`atol=1e-9`).
   - lava: same, guarded by the patched-`linprog` check; skip with a clear
     message if `linprog(...).basis` is unavailable.
   - Require the interval union to still contain `etaT_y` (the existing
     invariant) and the same number of merged pieces.
3. **Determinism:** two consecutive numba runs produce bit-identical output.
4. **Fallback coverage:** a synthetic `s_coef` with a tangent tie forces the
   fallback branch; assert reference and numba agree there.

---

## 8. Runtime test plan

- **Kernel microbenchmarks** (min of N repeats, warmup excluded): each kernel at
  increasing size (`nt ∈ {50,100,150,300}`, `p ∈ {5,10,20}`), report
  `t_ref`, `t_numba`, `speedup`.
- **End-to-end** on the `SIZES` grid of `experiment/inspect_parametric_intervals.py`
  for lava, and the cg analog; report total wall time and the `_timeprof`
  breakdown (`nested: sort walk + lasso loop`, `trace_parametric_bases`,
  `conditioning_Lasso`).
- **Compile overhead** reported separately (first call).
- Target: ≥2× end-to-end for cg and ≥1.5× for lava, with N1/N2 showing the
  largest kernel speedups. If a Tier-2 kernel is not faster, keep the reference.

---

## 9. Open questions

1. ~~Does the installed numba support the installed numpy?~~ Answered: numba
   0.68.0 + numpy 2.5.3 compile OK (see §3.1, §6 Phase 0).
2. Is the sklearn `solve_Lasso` cost large enough to justify the risky N9
   re-implementation? Decide from the Phase-1 `_timeprof` breakdown.
3. Should the flag default to numba once parity is green, or stay opt-in?
4. Is `iter_sort_regions_table` output buffer growth ever hit in practice
   (check `2*nt + MAX_STALE + 16` against `walk_stats` counts)?

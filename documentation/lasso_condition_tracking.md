# Event-driven Lasso conditioning intervals (no Lasso re-solves)

Companion to `README.md` §8.4 / §9. This document explains how the
restart-and-solve Lasso sub-walk of `parametric_nested` was replaced by the
event-driven (homotopy / LARS-style) walk of `Lasso_condition_tracking.txt`,
how the reference algorithm was **adapted to our conventions**, and the
accuracy tests that verify it.

Implementation: `si_for_cg/si_lasso_path.py`.
Integration: `si_for_cg/si_parametric.py` (`parametric_intervals_FS_nested`,
`cg_parametric_intervals_FS_nested`), toggled by `SIConfig.lasso_path`
(`si_types.py`), default `True`.

---

## 1. The event we enumerate

The p-value is the truncated-normal mass of the minimal conditioning event

```
S = { z : M(z) = M_obs }        # z-local Lasso support equals the observed support
```

`M(z)` is fit on the *selected rows* `X[selection]` against `Y(z) = a + b z`
(the parametric line, §5 of `README.md`). Inside one merged sort-selection span
`[sl, sr]` the row set `selection` is fixed, so the design `X_ = X[selection]`
and the per-row intercept/slope `a = a_vec[selection]`, `b = b_vec[selection]`
are fixed, and `M(z)` follows the ordinary Lasso path on the line
`Y(z) = a + b z`.

The support/sign event for a fixed `(M, sM)` is the KKT polyhedron
`{z : A z <= B}` built by our `conditioning_Lasso` /
`si_numba._conditioning_lasso_kernel`, whose rows are ordered

| row range | meaning | tight at the upper bound ⇒ |
|---|---|---|
| `[0, |M|)` | active feature `i`: `sM_i * beta_i(z) >= 0` | `beta_i = 0` → **feature `M[i]` leaves** |
| `[|M|, |M|+|Mc|)` | inactive `j`: `u_j(z) <= +Lambda` | `u_j = +Lambda` → **enters with `+1`** |
| `[|M|+|Mc|, |M|+2|Mc|)` | inactive `j`: `u_j(z) >= -Lambda` | `u_j = -Lambda` → **enters with `-1`** |

The Lasso path is continuous and piecewise linear in `z`, so consecutive
support regions tile `[sl, sr]`: the upper bound `R` of one region is exactly
the lower bound of the next, and the tight rows at `R` *are* the update rule
for the next `(M, sM)`.

---

## 2. Adaptation to our conventions

The reference code (`divide_and_conquer_Lasso_nonDA_path`) uses the same
`Atilde`/`Btilde` construction, but it is from another codebase, so the mapping
was re-derived against our code and checked numerically:

* `X_` is the selected **row set** `X[selection]` (shape `(m, p)`), not the full
  design; `a, b` are `a_vec[selection]`, `b_vec[selection]` (shape `(m, 1)`).
  This is exactly what `conditioning_Lasso(X_, a, b, Lambda, M, Mc, sM)` takes
  (`si_overconditoning.py:203`).
* The interval is `{z : A z <= B}` with `A = Atilde·b`,
  `B = Btilde - Atilde·a` (`conditioning_lasso_AB`), and is solved with the same
  `solve_linear_inequality(A, -B)` (`si_utils.py:208`) — identical tolerance
  snapping to `conditioning_Lasso`.
* Because our `solve_Lasso` returns `M` in ascending order and
  `sM = sign(beta_hat[M])`, the row-index → feature mapping of
  `advance_active_set` matches the reference verbatim.
* **Direction:** the walk inside a span always moves in `+z`. The boundary
  events are therefore the rows with `A_i > 0` attaining
  `R = min_{A_i>0} B_i/A_i` (`binding_event_rows`), i.e. the constraints that
  become violated just *above* `R`. This is the only direction the nested
  scanners need; the reference's signed generalisation is unnecessary here.
* The reference's `+1e-4` restart offset and its off-by-one interval bookkeeping
  were dropped; our walk yields pieces clipped to `[sl, sr]` and reuses the
  existing `interval_union` algebra.

---

## 3. Algorithm (`walk_lasso_support_regions`)

```
seed:  M, sM = solve_Lasso(X_, Y(z_start))          # one solve per span
z = z_start
while z <= sr:
    (A, B) = conditioning_lasso_AB(X_, a, b, Lambda, M, Mc, sM)
    [L, R] = solve_linear_inequality(A, -B)
    yield (max(L, sl), min(R, sr), M)               # a support tile
    if R >= sr: break
    rows   = binding_event_rows(A, B, direction=+1)  # tight, A_i > 0
    M2, sM2, ok = advance_active_set(M, Mc, sM, rows)
    if ok:                                          # continuity check
        [L2, R2] = conditioning_lasso_AB(X_, a, b, Lambda, M2, setdiff(M2))
        if |L2 - R| <= tol and R2 > R + tol:
            M, sM = M2, sM2 ; z = R ; continue      # NO Lasso solve
    # degenerate / infeasible / discontinuous:
    z = R + step
    M, sM = solve_Lasso(X_, Y(z))                   # restart-and-solve fallback
```

The caller keeps the tiles whose support equals `fs.M` (the observed support)
and unions them. `stats` accumulates `seed_solves`, `solver_calls`, `inferred`
and `fallbacks`.

**Fallback / robustness** (same guarantees as the reference):
simultaneous (degenerate) events, sign conflicts in `advance_active_set`, a
predicted interval that does not restart at `R`, or an infeasible polyhedron
all trigger a real `solve_Lasso` at `R + step`; a run of failures beyond
`MAX_STALE` aborts the span. Because every accepted step strictly increases the
crossed boundary, the walk always terminates.

---

## 4. Integration

Both nested scanners now branch on `config.lasso_path` (default `True`):

```python
for z_entry, sl, sr, selection in coalesce_selection_regions(regions):
    if config.lasso_path:
        for L, R, Mp in walk_lasso_support_regions(
                X[selection], a_vec[selection], b_vec[selection],
                fs.lambda_, z_entry, sl, sr, config.step):
            if np.array_equal(fs.M, Mp):
                final_interval.append((L, R))
        continue
    # ... original restart-and-solve loop (see README §8.4) ...
```

The sort-region walk (`iter_sort_regions` / `iter_cg_sort_regions`) and the
`coalesce_selection_regions` merge are unchanged: they supply the fixed-row
spans the Lasso path walks over.

---

## 5. Accuracy

All tests compare the **event-driven walk** against the **restart-and-solve**
baseline on identical data/config, using `step = 1e-5`.

### 5.1 Algorithm unit test (brute-force reference)

`walk_lasso_support_regions` vs a brute-force restart-and-solve tiling on
random `(X_, a, b, Lambda)` (`m ∈ [20,60)`, `p ∈ [5,12)`, `z ∈ [-3,3]`,
brute step `1e-7`):

```
40/40 trials reproduce the reference tiling exactly
(same support per region, boundaries within 1e-4)
```

### 5.2 cg-score pipeline (`cg_parametric_intervals_FS_nested`)

12 seeds, `n=200`, `p=15`, `fraction=0.5`, `lambda=5`, `[z_obs-1, z_obs+1]`:

```
matched 12/12, worst |Δp| = 0.000e+00   (bit-identical p-values and interval counts)
```

Randomised sweep, 60 configs (`n ∈ {80,150,300}`, `p ∈ {8,12,20}`,
`fraction ∈ {0.3,0.5,0.7}`, `lambda ∈ {1,5,10}`):

```
matched 60/60, worst |Δp| = 0.000e+00
```

### 5.3 lava pipeline (`parametric_intervals_FS_nested`)

8 seeds, `nt=40`, `nv=15`, `p=15`, `fraction=0.5`, `lambda=5`:

```
matched 8/8, worst |Δp| = 0.000e+00
```

### 5.4 With the numba sort walk (`SI_USE_NUMBA=1`)

Same 12 cg seeds: identical interval counts, `|Δp| ≤ 3.4e-49` (pure
floating-point tail arithmetic; far below any reporting precision).

### 5.5 `examples/demo_si.py`

Runs end-to-end unchanged and returns a finite p-value.

---

## 6. Performance notes

The win comes from removing sklearn coordinate-descent fits inside each span.
It scales with `p` and the number of selected rows `n_keep`, and with the
number of support boundaries in the span (which grows with the search window).
Measured (identical data, `step` fixed):

| case | restart-and-solve | event-driven |
|---|---|---|
| cg `n=400, p=30`, `[z_obs-0.3, z_obs+0.3]`, 1825 tiles | 2.61 s | 2.42 s |

The gain is modest when the Lasso sub-walk is not the bottleneck (for these
sizes the inferred sort walk dominates). It grows with `p`, with the selected
row count `n_keep` (sklearn coordinate descent cost) and with the number of
support boundaries; the walk also removes the arbitrary `step`-grid restart
artefact of the baseline. The walk is currently pure NumPy; it can be ported to
numba later if `conditioning_lasso_AB` + `solve_linear_inequality` dominates.

---

## 7. Files

* `si_for_cg/si_lasso_path.py` — `conditioning_lasso_AB`,
  `lasso_conditioning_interval`, `binding_event_rows`, `advance_active_set`,
  `walk_lasso_support_regions`.
* `si_for_cg/si_parametric.py` — branch in both nested scanners.
* `si_for_cg/si_types.py` — `SIConfig.lasso_path`.
* `si_for_cg/__init__.py` — re-exports `walk_lasso_support_regions`.
* `Lasso_condition_tracking.txt` — the original reference note.

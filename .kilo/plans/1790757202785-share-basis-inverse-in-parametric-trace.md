# Fix redundant B_inv in the parametric OT-basis trace

## Goal

In `si_for_cg/si_parametric.py`, the parametric trace computes `np.linalg.inv(B)` twice per pivot step on the identical 3000x3000 basis matrix `B = H[:, basis]` — once in `_basis_optimal_region` (line 593) and again in `_pivot_basis` (line 676). Fix by computing the inverse once and passing it from `_basis_optimal_region` to `_pivot_basis`. This is strictly bit-identical (the same inverse array is reused; the second `inv` call is dropped) and roughly halves the trace wall time (~169s -> ~85s of a ~296s demo run).

Scope decision (user-confirmed): **share the inverse only** — do NOT replace `np.linalg.inv` with `lu_factor`/`lu_solve`.

## Constraints

- `trace_parametric_bases` signature and return type must NOT change: it is also called from `experiment/inspect_parametric_intervals.py` (lines 149-152) with the same args.
- `_basis_optimal_region` / `_pivot_basis` have no other callers in code (only old debug snippets in `gptsession.md`, no repo code), so their signatures may change freely.
- Keep the existing `tp` timing instrumentation in place during implementation/validation — it measures the speedup. Removal is a separate step.

## Changes (all in `si_for_cg/si_parametric.py`)

### 1. `_basis_optimal_region` (currently ~line 572)

- Update the return annotation to include the inverse: `-> tuple[float, float, list[int], np.ndarray]`.
- Change the final return to also hand out the already-computed `B_inv` (computed at line 593):
  ```python
  return left, right, entering_candidates, B_inv
  ```

### 2. `_pivot_basis` (currently ~line 663)

- Add an optional parameter: `B_inv: np.ndarray | None = None`.
- Compute it only when not provided (backward-compatible with standalone use):
  ```python
  if B_inv is None:
      B = H[:, basis]
      B_inv = np.linalg.inv(B)
  ```
- Keep everything else unchanged. The passed `B_inv` corresponds to the pre-pivot basis, which is exactly the one `_pivot_basis` needs for `direction_vector` and `x_basic` (the basis is mutated only after those are computed, at the existing line 710).

### 3. `trace_parametric_bases` (currently ~line 716)

- Unpack the extra return value at the `_basis_optimal_region` call (~line 742):
  ```python
  left, right, entering_candidates, B_inv = _basis_optimal_region(...)
  ```
- Pass it to `_pivot_basis` (~line 782):
  ```python
  basis = _pivot_basis(H=H, h=h, basis=basis, entering=entering, B_inv=B_inv)
  ```

No other lines change.

## Validation

1. Syntax check: `& "C:\Users\Asus\anaconda3\envs\IS\python.exe" -m py_compile si_for_cg/si_parametric.py`.
2. Baseline capture (BEFORE making the edits): run the demo once per fixed seed and record (p-value, `nested: OT intervals` count from the tp report). The demo is currently nondeterministic (`generate(..., seed=None)` uses the unseeded global RNG), so create a throwaway script `examples/_trace_fix_check.py` that sets `np.random.seed(S)` for S in {0, 1, 2}, runs `SI_DV(... method="parametric_nested", ...)` once per seed (same n/m/p/lambda as the demo: 200/15/15, lambda_=5, zmin=-20, zmax=20), and prints the p-value and OT-interval count. ~5 min per seed.
3. Apply the edits, rerun the same seeds, and assert:
   - p-value identical to baseline (bit-exact, since the trace is bit-identical);
   - `nested: OT intervals` count identical (expect 1562 for these settings);
   - speedup visible in the tp report: `trace: _pivot_basis (all calls)` near 0 and `nested: trace_parametric_bases (L+R)` roughly halved.
4. If any seed diverges (should not happen — the inverse array is identical), stop and re-check the edits; the change is behavior-preserving by construction.

## Out of scope (explicitly not touched)

- `parametric_simplex_quadratic` (dead code, `si_parametric.py` ~line 952) has the same redundant inverse at line 990 — leave as-is per AGENTS.md.
- `overconditioning_intervals` (`si_overconditioning.py` lines 152/166) also inverts `S_[:, B]` twice within one call — different code path (OC method); candidate for a follow-up, not this fix.
- Incremental inverse updates across pivots (Sherman-Morrison) — future work, numerically risky here.
- Removing the `tp` profiling instrumentation.

## Rollback

Revert the three edits above — the diff is small and confined to `_basis_optimal_region`, `_pivot_basis`, and the two call sites in `trace_parametric_bases`.

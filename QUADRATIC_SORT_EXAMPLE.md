# Worked Example: Steps 3–4 of the Quadratic Sort Walk

This file illustrates how the next permutation $\pi'$ is derived from the
current one $\pi$ at the interval boundary $R$, i.e. **step 3** (group
consecutive crossing flags into blocks) and **step 4** (reverse each block).

## Recap of the rule

At $z=R$ we mark, for every adjacent pair $m = (\pi_m, \pi_{m+1})$, a flag

```
flip_m = ( |Δ_m(R)| ≤ tol )  AND  ( Δ_m'(R) < 0 ),     Δ_m = s_πm − s_π{m+1}
```

- **Step 3** – scans the flag row and merges *consecutive* `True` flags into
  maximal blocks (a run of length $k$ means $k+1$ scores are tied at $R$).
- **Step 4** – for each block covering positions `m .. m+k`, the slice
  $\pi[m : m+k+1]$ is reversed in place.

Every quadratic below is checked by direct sorting at $z=R\pm 0.1$, so the walk
result can be verified by hand.

---

## Case A — single flip (no tie)

Scores ($n=3$):

| i | $s_i(z)$ |
|---|---|
| a | $z^2$ |
| b | $2z$ |
| c | $-4$ |

Current permutation on the interval ending at $R=0$ is $\pi=[a,b,c]$
(verified: at $z=-0.1$ we get $a=0.01 > b=-0.20 > c=-4$).

Compute $\Delta_m$ and $\Delta_m'$ at $R=0$:

| $m$ | pair | $\Delta_m(z)$ | $\Delta_m(0)$ | $\Delta_m'(0)$ | flag |
|---|---|---|---|---|---|
| 1 | $(a,b)$ | $z^2 - 2z = z(z-2)$ | $0$ | $-2 < 0$ | **True** |
| 2 | $(b,c)$ | $2z + 4$ | $4$ | — | False (not tight) |

**Step 3:** flags = `[ True, False ]` → maximal block = `{1}`.
**Step 4:** block `{1}` covers positions `1..2`, slice `[a, b]` → reverse →
`[b, a]`. Result:

$$\pi' = [\, b,\ a,\ c \,]$$

Verify by sorting at $z=+0.1$: $b=0.20 > a=0.01 > c=-4$ → `[b, a, c]` ✓.

---

## Case B — three-way tie (consecutive flags → block reversal)

Scores ($n=4$):

| i | $s_i(z)$ |
|---|---|
| a | $-z + z^2$ |
| b | $z^2$ |
| c | $z + z^2$ |
| d | $-5$ |

The first three curves all pass through $(0,0)$ with slopes $-1, 0, +1$ — a
genuine tie. Current permutation on the interval ending at $R=0$ is
$\pi=[a,b,c,d]$ (verified: at $z=-0.1$: $a=0.11 > b=0.01 > c=-0.09 > d=-5$).

Compute at $R=0$:

| $m$ | pair | $\Delta_m(z)$ | $\Delta_m(0)$ | $\Delta_m'(0)$ | flag |
|---|---|---|---|---|---|
| 1 | $(a,b)$ | $-z$ | $0$ | $-1 < 0$ | **True** |
| 2 | $(b,c)$ | $-z$ | $0$ | $-1 < 0$ | **True** |
| 3 | $(c,d)$ | $z+z^2+5$ | $5$ | — | False |

**Step 3:** flags = `[ True, True, False ]` → the two *consecutive* `True` s
form ONE block `{1, 2}` (do NOT treat them as two independent swaps; they are a
tie of three elements).
**Step 4:** block `{1,2}` covers positions `1..3`, slice `[a, b, c]` → reverse →
`[c, b, a]`. Result:

$$\pi' = [\, c,\ b,\ a,\ d \,]$$

Verify by sorting at $z=+0.1$: $c=0.11 > b=0.01 > a=-0.09 > d=-5$ → `[c,b,a,d]` ✓.

> If instead the two flags were applied as separate swaps, position pairs
> $(1,2)$ then $(2,3)$ would produce `[b, c, a, d]` (a rotation) — which does
> **not** match reality.

---

## Case C — two isolated flips at the same $z$ (two separate blocks)

Scores ($n=4$):

| i | $s_i(z)$ |
|---|---|
| a | $3 - z + (z-1)^2$ |
| b | $2$ |
| c | $-2 - z + (z-1)^2$ |
| d | $-3$ |

Pair $(a,b)$ crosses at $R=1$ at value $2$; pair $(c,d)$ crosses at the *same*
$R=1$ but at value $-3$ — two independent, non-tied events. Current permutation
on the interval ending at $R=1$ is $\pi=[a,b,c,d]$ (verified: at $z=0.9$:
$a=2.11 > b=2 > c=-2.89 > d=-3$).

Compute at $R=1$:

| $m$ | pair | $\Delta_m(z)$ | $\Delta_m(1)$ | $\Delta_m'(1)$ | flag |
|---|---|---|---|---|---|
| 1 | $(a,b)$ | $(z-1)(z-2)$ | $0$ | $-1 < 0$ | **True** |
| 2 | $(b,c)$ | $4 + z - (z-1)^2$ | $5$ | — | False |
| 3 | $(c,d)$ | $(z-1)(z-2)$ | $0$ | $-1 < 0$ | **True** |

Pair 2, $\Delta_2 = s_b - s_c = 2 - ( -2 - z + (z-1)^2 ) = 4 + z - (z-1)^2$,
equals $5$ at $z=1$, so it is not tight.

**Step 3:** flags = `[ True, False, True ]` → consecutive `True` s are NOT
adjacent, so we get **two separate blocks** `{1}` and `{3}`.
**Step 4:** reverse slice `[a, b]` → `[b, a]`, and slice `[c, d]` → `[d, c]`.
Result:

$$\pi' = [\, b,\ a,\ d,\ c \,]$$

Verify by sorting at $z=1.1$: $b=2 > a=1.91 > d=-3 > c=-3.09$ → `[b,a,d,c]` ✓.

---

## Why reversal and not individual swaps (intuition)

Near a tie all involved curves pass through one point $(z^\ast, s^\ast)$:

$$
s_i(z) = s^\ast + s_i'(z^\ast)\,(z - z^\ast) + O((z - z^\ast)^2) .
$$

- Just **below** $R$ the ordering is decided by $-s_i'(z^\ast)$;
- just **above** $R$ it is decided by $+s_i'(z^\ast)$,

i.e. the two orderings are exact reverses of each other. Hence a maximal block
of $k+1$ tied elements appears reversed in the next interval — regardless of the
quadratic curvature (the $O(\delta^2)$ terms are negligible arbitrarily close to
the collision).

## Degenerate note

If two tied curves also have equal slopes at $R$ (tangent / persistent tie), the
reversal is not forced by first-order terms; the continuity check
($|L'-R|\le\text{tol}$ and $R'>R+\text{tol}$) fails and the walk falls back to
re-sorting at $R+\varepsilon$. This is measure-zero for random $(p,q,r)$.
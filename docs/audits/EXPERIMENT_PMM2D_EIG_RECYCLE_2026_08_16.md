# Eigenbasis recycling for all-distinct 2-D PMM stacks -- REFUTED, 2026-08-16

Branch `feat/pmm2d-eig-recycle` off `origin/main` (97d431f, the 5.36.0 release
commit).  `git log origin/main..HEAD` was empty before branching.

**The hypothesis.**  The 5.36.0 fast cascade
(`BUILD_PMM2D_CASCADE_2026_08_16.md`) leaves one workload at exactly 1.00x:
the ALL-DISTINCT stack, where there is nothing to dedup and nothing to merge.
Adjacent slices of a staircase or taper differ only slightly, so the previous
layer's eigenbasis should be a good starting subspace: a Rayleigh-Ritz
projection onto the prior basis plus a residual correction ought to beat a cold
dense `zgeev` for slowly-varying layer sequences.

**The verdict: refuted, end to end, on both the in-plane (2Nf) and the
out-of-plane (4Nf) paths.**  On a design-121-like taper the scheme refuses on
*every* layer, and the whole solve measures **1.000x** (in-plane) / **0.990x**
(OOP) -- against a 1.30x ship bar.  It clears 1.30x only on a stack whose
adjacent layers differ by 0.007%, which is not a taper.  No library code is
changed by this branch.

**Mount.**  Windows py3.14.6, numpy 2.4.4, scipy 1.17.1, scipy-openblas
0.3.31, 24 cores / 128 GB (tesla-ryzen).  Every timing ran with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` exported **in the
environment before python started**, best-of-N after a warm-up solve, on an
otherwise quiet box.  Every probe asserted `"lum_sr" in lumenairy.__file__`
and printed it: `C:\tmp\lum_sr\lumenairy\__init__.py`, version 5.36.0.

---

## S1.  Rule 1: the ceiling, computed before anything was built

The cascade doc's cost profile reads "eig 39% / cascade 59%", but that 39% is
the CUMULATIVE cost of `_layer_modes_projected`, which also builds the `P` and
`Q` blocks and forms `V = Q W diag(1/lam)`.  Recycling can only remove the
LAPACK call inside it.  Re-measured, per-call wall clock, on all-distinct
8-layer tapers (three different taper laws, to show the split is a property of
the workload and not of one tile):

| taper law | `n_orders` | n | total solve | zgeev | zgeev share | `_layer_modes_projected` | cascade |
|---|---|---|---|---|---|---|---|
| eps grade | 5 | 242 | 0.9997 s | 0.3169 s | **31.7%** | 39.1% | 55.3% |
| edge shrink | 5 | 242 | 0.9824 s | 0.3144 s | **32.0%** | 39.3% | 55.2% |
| uncorrelated | 5 | 242 | 1.0083 s | 0.3226 s | **32.0%** | 39.0% | 54.6% |
| eps grade | 4 | 162 | 0.3717 s | 0.1275 s | **34.3%** | 40.8% | 47.5% |
| eps grade | 6 | 338 | 2.5563 s | 0.7174 s | **28.1%** | 35.8% | 61.1% |

So the Amdahl ceiling for eig work alone, in-plane:

| `n_orders` | zgeev share `f` | ceiling `1/(1-f)` |
|---|---|---|
| 4 | 0.343 | **1.52x** |
| 5 | 0.317 | **1.46x** |
| 6 | 0.281 | **1.39x** |

Above the 1.3x stop-threshold, so the experiment proceeded -- but note the
trend: the share FALLS with basis size, because the strictly-sequential
Redheffer cascade grows faster than the eig.  The ceiling is worst exactly
where the runtime hurts most.

**The out-of-plane path is different and was measured separately.**  The OOP
generator eigendecomposes a `4Nf` block while the generalized cascade still
works on `2Nf` blocks, so the eig dominates structurally:

| `n_orders` | n(G) | 2Nf | zgeev(G) | gemm(2Nf) | ratio |
|---|---|---|---|---|---|
| 3 | 196 | 98 | 0.0296 s | 0.000125 s | 236x |
| 4 | 324 | 162 | 0.0990 s | 0.00052 s | 190x |
| 5 | 484 | 242 | 0.2712 s | 0.00170 s | 159x |

Profiled end to end on a 4-layer OOP tensor stack: **zgeev 67.9%** of the
solve, generalized cascade 17.1%.  **OOP ceiling: 3.12x.**  This is why the
experiment was not stopped at S1 -- one of the two paths has real headroom.

## S2.  The FLOOR: what any recycler must pay before it computes anything

Any scheme that turns a prior eigenvector matrix `W0` into the new layer's
`(lam, W)` must do at least four O(n^3) products at the SAME n as the zgeev it
replaces: `A @ W0`, `W0inv @ (A W0)`, `Wacc @ T`, and the residual check
`A @ W - W lam` that rule 2 requires.  Measured:

| n | zgeev | 1 zgemm | floor = 4 zgemm | floor/zgeev | best solve speedup at `f`=0.32 |
|---|---|---|---|---|---|
| 162 | 0.0202 s | 0.00052 s | 0.0021 s | 0.102 | 1.40x |
| 242 | 0.0532 s | 0.00205 s | 0.0082 s | 0.155 | 1.37x |
| 338 | 0.1170 s | 0.00494 s | 0.0198 s | 0.169 | 1.36x |
| 578 | 0.4882 s | 0.02395 s | 0.0958 s | 0.196 | 1.35x |

Read that last column carefully: it is the speedup of a **perfect** recycler
that converges in ZERO sweeps and re-diagonalises no clusters.  In-plane, the
theoretical best is 1.35-1.40x against a 1.30x bar -- the entire experiment
had, at most, 5-10% of margin before a single Newton sweep was paid for.

Inverting it: to reach 1.3x whole-solve at `f`=0.32 the recycled eig must cost
**<= 27% of zgeev**, i.e. about 6-11 zgemm depending on n.

## S3.  Why it does not converge: the spectrum is clustered far below the
## perturbation

The controlling quantity for any Newton / Jacobi-style diagonalisation seeded
by a prior basis is not the size of the perturbation but its size RELATIVE TO
THE EIGENVALUE GAPS it must divide by.  With `E = W0^-1 A_next W0`, the
first-order eigenvector correction is `X_ij = E_ij / (d_i - d_j)`, so the
amplification is `amp = max_{i != j} |E_ij| / |d_i - d_j|`.  `amp << 1`
converges quadratically; `amp >~ 1` is meaningless.

Measured on the hybrid's own `P@Q` operators (`n_orders`=5, n=242, degree 9,
6x6 patterned cell, wl 1.55 um, normal incidence):

| taper law | `|dA|/|A|` per step | `|offdiag(E)|/|diag|` | **amp** |
|---|---|---|---|
| 1% eps grade over the stack | 1.72e-03 | 1.31e-03 | **0.88** |
| 20% eps grade (design-121-like) | 3.43e-02 | 2.66e-02 | **18.8** |
| geometric edge shrink | 2.15e-01 | 2.31e-01 | **261** |
| uncorrelated (adversarial) | 3.39e-01 | 7.37e-01 | **440** |

The reason is in the spectrum itself: the nearest-neighbour eigenvalue gap
bottoms out at **1.7e-06 of the spectral scale** (`min gap` 4.2e-04 against
`max|lam2|` 2.5e+02), while `cond(W)` is a benign 9.6e+01.  So a relative
operator perturbation of 1e-03 -- finer than any real staircase -- is already
three orders of magnitude larger than the smallest gaps it has to divide by.
The PMM layer spectrum is dense with near-degenerate pairs, which is exactly
the regime in which basis recycling has nothing to offer.

Clustering the offending pairs and re-diagonalising those blocks densely is
the standard remedy, and it fails on cost: the induced blocks swallow the
matrix.  With the cluster threshold derived from the coupling itself, the
fraction of a dense eig that must still be paid, `sum(c^3)/n^3`:

| taper law | max cluster | `sum(c^3)/n^3` |
|---|---|---|
| 1% eps grade | 13 | 0.0006 |
| 20% eps grade | 238 | **0.951** |
| edge shrink | 242 | **1.000** |
| uncorrelated | 242 | **1.000** |

At realistic taper contrast the "cluster" is the whole matrix, so the block
re-diagonalisation IS the dense eig and every gemm spent getting there is
pure loss.

## S4.  Three recyclers were built, and the contrast curve that settles it

* **v1** -- exact similarity (`T = I + X`, LU solve), whole-off-diagonal
  stopping.
* **v2** -- the cheapest possible sweep: `W0inv` cached from the previous
  layer (2 gemm to form `E` instead of gemm + LU solve), `T^-1` by a truncated
  Neumann series (3 gemm/sweep instead of 2 gemm + solve), cluster-gated
  stopping so intra-cluster blocks are correctly left to a final dense block
  eig rather than counted as failures.
* **v3** -- exact similarity plus v2's cluster gating: the most reliable and
  the one used for the curves below.

Accuracy was never the failure mode *at the eigensolve* (see S5 for the
end-to-end caveat).  Wherever v3 converged it met the bar
derived from the dense solve's own normwise backward error on the very same
matrix (`1e3 x` that, measured per case, nothing pinned from elsewhere), and
usually beat the dense solve outright: residuals **3.7e-16 to 4.9e-15** against
dense **1.0e-15 to 1.3e-15** and bars of **1e-12**.  The refusal path is what
fires, and it fires on cost and convergence.

Contrast curve, in-plane, `n_orders`=5, n=242, two slices of one taper whose
pillar eps differs by `delta`:

| `delta` | `|dA|/|A|` | max cluster | sweeps | residual | bar | verdict | t_rec/t_zgeev |
|---|---|---|---|---|---|---|---|
| 1e-05 | 5.15e-06 | 2 | 2 | 1.16e-15 | 1.18e-12 | PASS | **0.58** |
| 1e-04 | 5.15e-05 | 4 | 3 | 7.57e-16 | 1.34e-12 | PASS | **0.79** |
| 1e-03 | 5.15e-04 | 5 | 11 | 4.89e-15 | 1.43e-12 | PASS | 2.36 |
| 3e-03 | 1.54e-03 | 11 | 7 | 4.35e-15 | 1.86e-12 | PASS | 1.56 |
| 1e-02 | 5.15e-03 | 40 | 12 | -- | 1.01e-12 | REFUSE | 2.53 |
| 3e-02 | 1.54e-02 | 178 | 0 | -- | 1.19e-12 | REFUSE | 0.13 |
| 1e-01 | 5.14e-02 | 238 | 0 | -- | 1.87e-12 | REFUSE | 0.13 |

The recycler is faster than `zgeev` only for `|dA|/|A| <~ 5e-05`.  Against the
27% budget of S2 it is never close: the best reading is 58%.

Same curve on the OOP 4Nf generator (n=196), where the budget is far looser
(`f`=0.68 needs only `t_rec/t_zgeev <= 0.66`):

| `delta` | `|dG|/|G|` | max cluster | sweeps | residual | verdict | ratio |
|---|---|---|---|---|---|---|
| 1e-05 | 2.92e-06 | 1 | 2 | 3.71e-16 | PASS | **0.44** |
| 1e-04 | 2.92e-05 | 1 | 2 | 4.09e-16 | PASS | **0.44** |
| 1e-03 | 2.92e-04 | 2 | 3 | 7.95e-16 | PASS | **0.59** |
| 1e-02 | 2.92e-03 | 35 | 7 | 4.09e-16 | PASS | 1.23 |
| 3e-02 | 8.77e-03 | 78 | 0 | -- | REFUSE | 0.11 |
| 1e-01 | 2.93e-02 | 196 | 0 | -- | REFUSE | 0.10 |

**Where the real staircases sit.**  Measured on the same machinery: a
design-121-like 20%-eps-grade taper is `|dA|/|A| = 3.4e-02` per slice and a
geometric edge-shrink taper is `2.2e-01` per slice.  Both are **two to three
orders of magnitude above the crossover**.  The hypothesis' premise --
"adjacent layers in a staircase differ slightly" -- is true in the eps, and
false in the only metric that governs an eigenproblem.

For completeness, the same measurement in units a caller controls, a
dispersive WAVELENGTH step at wl=1.55 um: `|dA|/|A|` = 1.35e-05 per 0.01 nm,
1.35e-04 per 0.1 nm, 1.35e-03 per 1 nm, 1.36e-02 per 10 nm (in-plane; OOP
within 6%).  So the regime where recycling helps at all is a sweep step
**below ~0.1 nm**.

## S5.  End-to-end, interleaved A/B -- the numbers that decide it

Interleaved per the cascade doc's S5.1 correction (a sequential before/after
on this box is not a valid instrument at the 1.3x level): baseline and
prototype run as SEPARATE subprocesses, alternating, round-robin, min over 4
rounds.  The two arms are the same worktree and the same process shape,
differing only in an env flag; the prototype is a monkeypatch, so no library
code was touched.  Per-round spreads were tight throughout (e.g. in-plane
`grade` baseline `[0.9859, 0.9933, 1.0014, 0.9968]`).

Stacks are 8 layers, all distinct, `symmetry=False`.  "ultrafine" varies the
pillar eps by 0.05% across the WHOLE stack (0.007% per layer) and exists only
to show what the mechanism does when it works.

### In-plane (2Nf), degree 9, 6x6 cell

| case | `n_orders` | baseline | recycled | speedup | recycled/dense | agreement |
|---|---|---|---|---|---|---|
| ultrafine (0.007%/layer) | 5 | 0.9892 s | 0.9798 s | **1.010x** | 21 / 3 | 1.6e-10 |
| fine (0.14%/layer) | 5 | 0.9948 s | 1.2271 s | **0.811x** | 18 / 6 | 2.8e-11 |
| **grade (design-121-like)** | 5 | 0.9859 s | 1.0578 s | **0.932x** | **0 / 24** | 0 |
| shrink (geometric taper) | 5 | 0.9921 s | 1.0616 s | 0.935x | 0 / 24 | 0 |
| **random (adversarial)** | 5 | 1.0176 s | 1.0929 s | **0.931x** | 0 / 24 | 0 |
| grade | 4 | 0.3706 s | 0.4395 s | 0.843x | 3 / 21 | 1.5e-13 |

### Out-of-plane (4Nf), degree 7, 4x4 tensor cell, theta=0.25 phi=0.4

| case | `n_orders` | baseline | recycled | speedup | recycled/dense | agreement |
|---|---|---|---|---|---|---|
| ultrafine (0.007%/layer) | 3 | 0.3489 s | 0.2569 s | **1.358x** | 21 / 3 | 4.2e-15 |
| fine (0.14%/layer) | 3 | 0.3500 s | 0.3167 s | 1.105x | 21 / 3 | 2.4e-14 |
| **grade (design-121-like)** | 3 | 0.3565 s | 0.3922 s | **0.909x** | **0 / 24** | 0 |
| **random (adversarial)** | 3 | 0.3871 s | 0.4256 s | **0.910x** | 0 / 24 | 0 |
| grade (6 layers) | 4 | 0.8534 s | 0.9723 s | 0.878x | 0 / 18 | 0 |

### The fallback regression is not intrinsic

The 7-14% regressions above are the prototype's seed upkeep (one `inv(W)` per
layer, kept so a later layer could recycle), not a property of the idea.
Re-measured with a ONE-STRIKE policy that stops maintaining the seed after the
first refusal -- which is what a shipping version would do:

| path | case | baseline | one-strike | speedup | recycled/dense |
|---|---|---|---|---|---|
| in-plane | **grade (design-121-like)** | 1.0111 s | 1.0108 s | **1.000x** | 0 / 24 |
| in-plane | shrink (geometric taper) | 1.0015 s | 0.9986 s | 1.003x | 0 / 24 |
| in-plane | **random (adversarial)** | 1.0183 s | 1.0354 s | **0.984x** | 0 / 24 |
| in-plane | ultrafine | 1.0086 s | 0.9901 s | 1.019x | 21 / 3 |
| OOP | **grade (design-121-like)** | 0.3582 s | 0.3619 s | **0.990x** | 0 / 24 |
| OOP | **random (adversarial)** | 0.3883 s | 0.3887 s | **0.999x** | 0 / 24 |
| OOP | ultrafine | 0.3505 s | 0.2596 s | 1.351x | 21 / 3 |

So the scheme can be made safe -- neutral on everything it refuses, within the
1.05x allowance on the adversarial arm.  It simply never engages on a real
taper, and neutrality is not a reason to carry code.

### One accuracy caveat, recorded against the claim in S4

The residual bar in S4 is an assertion about EIGENPAIRS.  End to end, the
in-plane `ultrafine` arm -- the only in-plane case where recycled eigenpairs
actually reached the cascade -- differed from the baseline by **1.6e-10** on
`sum|R|`, well above the ~1e-14 the 5.36.0 cascade treats as agreement, even
though every accepted eigenpair sat at ~1e-15 with a 1e-12 bar.  The OOP arms
stayed at 4.2e-15 / 2.4e-14.  A shipping version would therefore need its own
END-TO-END bound in the cascade doc's `1e3 (2n+2) r` shape, derived from the
conditioning of the cascade rather than from the eigenpair residual; the
per-eigenpair gate alone is not sufficient evidence.  This does not change the
verdict, but "accuracy was never the failure mode" is a statement about the
eigensolve, not about the solve.

## S6.  Ship criteria, scored

| criterion | required | measured | |
|---|---|---|---|
| favourable staircase (design-121-like taper) | >= 1.30x | **1.000x** in-plane, **0.990x** OOP -- the recycler refuses on 100% of layers | **FAIL** |
| adversarial (uncorrelated layers) | <= 1.05x regression | 0.984x in-plane / 0.999x OOP with one-strike (0.931x / 0.910x without) | pass |
| residual bar met on every accepted eigenpair | derived, per build | 3.7e-16 .. 4.9e-15 vs dense 1.0e-15 .. 1.3e-15, bars 1e-12 | pass |
| fallback to dense automatic | yes | yes -- cluster gate, sweep cap, and an explicit residual gate | pass |

The decisive criterion fails, and it fails by refusing to engage at all on the
workload it was built for.  **Nothing is shipped.**  The library is unchanged
on this branch; only this document is added.

## S7.  What would have to be true for this to work, and what to do instead

Recycling needs `|dA|/|A|` per layer below ~3e-04 (OOP) or ~5e-05 (in-plane).
That is not a taper; it is a stack of numerically near-identical layers.  Three
honest consequences:

1. **The premise is wrong for eigenproblems, not for the operator.**  A 3%
   change in eps IS a small perturbation of `A` -- and is still 10^4 times the
   smallest eigenvalue gap.  Any future "small change" optimisation on this
   path should be screened by `amp = max |E_ij| / gap_ij` (S3), which costs one
   solve and one gemm to evaluate, before anything is built.
2. **In-plane, the ceiling itself is the wall.**  Even a perfect recycler tops
   out at 1.35-1.40x (S2), and the share falls with basis size.  Effort on the
   all-distinct in-plane cost is better spent on the **cascade**, which is
   55-61% of the solve and rises with n -- the `_redheffer_star` /
   `_interface_smatrix` pair, not the eig.
3. **OOP is the one genuinely eig-dominated path (68%, ceiling 3.12x)** and it
   remains open. Recycling is not the lever, but anything that attacks the
   `4Nf` zgeev directly would be: the generator's `[[A, P], [Q, B]]` block
   structure is not exploited at all today, and the in-plane symmetric path
   already shows what exploiting structure is worth (the `2Nf` eig is 4x
   cheaper than the `4Nf` generator for the same physics when `A = B = 0`).

## S8.  Files

| file | change |
|---|---|
| `docs/audits/EXPERIMENT_PMM2D_EIG_RECYCLE_2026_08_16.md` | NEW -- this document |

No library file, test, or public API is touched.  The prototypes
(three recyclers, two solve-path hooks, the interleaved A/B driver) live
outside the repo at `C:\tmp\lum_sr_probes` and are deliberately not committed:
they implement a refuted approach and would only rot.

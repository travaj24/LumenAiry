# FIX -- round 3 of the PURE staggered 2-D PMM per-layer grids (L2 mortar)

**Date** 2026-09-11 · **Worktree** `C:/tmp/lum_mortar3`, branch
`fix/pmm2d-mortar-round3`, based on `wave2/pmm2d` at `1957da2` · **`without`
arm** a pristine copy of the three changed files at that same commit
(`C:/tmp/lum_r3pre`, read-only)

**Under repair** `docs/audits/VERIFY_PMM2D_MORTAR_ROUND2_2026_09_11.md` --
**DEFECT V1** (P1, ship-blocking, S6.3), the non-blocking **S5.4** ask, and the
documentation defects **V2 / V3 / V4 / V5 / V6** (S10).

**Method** every number below was measured in this worktree by
`validation/probe_fix_mortar_round3/` (seven scripts and a README), on BOTH
builds, before the code was changed and again after.  Nothing is read from the
round-2 fix doc or from the verification: where a number of theirs is quoted it
is labelled as theirs and stands beside mine.

**Binding** `docs/TESTING_STANDARDS.md`.

---

## S0. Summary

| # | what | verdict |
|---|---|---|
| V1 | `_MORTAR_RCOND_REFUSE` = 1e-12 applied to `_interface_smatrix_general_mortar_2d` | **FIXED.**  The mechanism the verification read off the operator's construction is CONFIRMED to closure and quantified; the site now takes its own decision on the RESIDUAL.  `rcond` is REFUTED as an instrument there -- its healthy and wrong populations CROSS |
| V1 coverage | the mixed in-plane / out-of-plane per-layer fixture no gate reached | **ADDED**, `tests/unit/test_fix_pmm2d_mortar_round3.py` |
| S5.4 | the SILENT 1.0..5.9x degradation band above the width contract | **WARNS** now (a `UserWarning`, not a refusal) over a derived band, with a census showing no ordinary stack warns |
| V2 | the width contract refuses a CONFORMING per-layer stack | **OPEN ITEM**, with the reason MEASURED: three constructors raise it independently, one of them public |
| V3 / V4 / V5 / V6 | documentation | **CORRECTED** in the round-2 fix doc (dated CORRECTION paragraphs, history intact) and in the constants' own comments |
| bit-identity | the round-2 verifier's 30-fixture / 165-hash battery, before vs after | **165 / 165 identical, 0 warning-set differences** |

**The one-sentence result.**  The generalized mortar's operand is
rank-deficient BY CONSTRUCTION whenever either side of the interface is an
in-plane region promoted to the 6-tuple general form, and the system is
CONSISTENT anyway -- so a condition estimate, which is a worst-case over all
right-hand sides, was measuring a property the answer does not have, while the
residual, which is a property of the answer, separates the two things that
actually matter there by twelve decades.

---

## S1. Terms, defined before they are used

**The generalized mortar site.**
`lumenairy/elements/pmm/_core.py::_interface_smatrix_general_mortar_2d`.  It is
the third of the three call sites of `_guarded_mortar_solve`, and the one an
OUT-OF-PLANE tensor layer or a SLANTED layer takes on the per-layer path.  Its
operand is the block

```
    A = [[E1, E2], [H1, H2]]           2 qq_B + 2 qq_A rows
                                        m_a + m_b     columns
```

where the first block COLUMN carries side `a`'s BACKWARD mode set and the
second side `b`'s FORWARD set.  The other two sites (`MassE_B W_B` and
`MassH_A V_A`, both inside `_interface_smatrix_mortar_2d`) are the IN-PLANE
pair and are NOT touched by this round.

**Promotion.**  `twod_staggered.py::_modes_as_general` writes a symmetric
in-plane region's modes in the generalized 6-tuple form as
`(W, V, lam, W, -V, -lam)`, using the exact in-plane pencil symmetry
`[W; -V] <-> -lam`.  That is what lets a stack containing ONE out-of-plane
layer run the generalized cascade throughout: every OTHER layer, and both
half-spaces, are PROMOTED.

**Rank deficiency vs inconsistency.**  An operand is *rank-deficient* when its
columns are linearly dependent, so `A x = b` has infinitely many solutions or
none.  It is *consistent* when `b` lies in the range of `A`, i.e. a solution
exists.  The two are independent, and the whole of V1 is that round 2 refused
on the first while only the second is a defect.

**`rcond`.**  The LAPACK `gecon` reciprocal 1-condition estimate, i.e. an
estimate of `1 / (||A|| ||A^-1||)`.  It bounds the WORST-CASE relative error
over ALL right-hand sides.  It says nothing about the particular `b` in hand.

**The residual.**  `||A X - B|| / ||B||`.  A backward-stable solve leaves this
at `~n eps` whatever the conditioning -- which is exactly why round 2 rejected
it at the IN-PLANE sites, and exactly why it works here: at this site the
question is not "how many digits" but "is there a solution at all".

---

## S2. The two builds

| | Windows (WIN) | WSL (Ubuntu) |
|---|---|---|
| python | 3.14.6 | 3.12.3 |
| numpy | 2.4.4 | 2.4.6 |
| scipy | 1.17.1 (scipy-openblas) | 1.17.1 (scipy-openblas) |
| threads | `OMP/OPENBLAS/MKL_NUM_THREADS = 1` | same |
| `lumenairy` | `C:\tmp\lum_mortar3\lumenairy\__init__.py` v5.44.0 | `/mnt/c/tmp/lum_mortar3/...` v5.44.0 |

Same BLAS family on both, so every cross-build spread below is a LOWER bound --
the same caveat the build, both fixes and both verifications carry.  The box is
shared with two other agents throughout, so **no wall-clock number below is
read as an absolute**; the cost section reports minimum-of-N interleaved
ratios and leans on a FLOP count.

---

## S3. DEFECT V1 -- the mechanism, measured to closure

`r1_mechanism.py`.  Every generalized-site operand a solve builds is captured
(the guard replaced by a recorder that never refuses), and for each: the full
SVD, the numerical rank at `max(m, n) eps s_max`, the residual of the answer
`lu_solve` returns, the component of the right-hand side along the LEFT
near-null direction, and the split of the smallest RIGHT singular vector
between the two block columns.  Each side is labelled PROMOTED or not
STRUCTURALLY -- `_modes_as_general` shares `W` by identity and negates `V`
bitwise, so the test is exact, not a tolerance.

### 3.1 The near-null space is the promoted side's, entirely

The v8 device (an out-of-plane patterned layer with an in-plane patterned
neighbour on other walls) and the plain-uniform-spacer variant, against the
BOTH-out-of-plane and BOTH-slanted controls, `n_orders` = 2:

| fixture | `M` | `n` | promoted a / b | `s_min/s_max` | `rcond` | residual | null on a / on b |
|---|---|---|---|---|---|---|---|
| **mixed** (v8) | 4 | 324 | no / **yes** | **2.544e-10** | 1.409e-11 | 8.126e-15 | **0.000 / 1.000** |
| | 5 | 576 | no / **yes** | **1.353e-11** | 8.396e-13 | 1.619e-14 | **0.000 / 1.000** |
| | 6 | 900 | no / **yes** | **9.220e-13** | 6.052e-14 | 3.015e-14 | **0.000 / 1.000** |
| | 7 | 1296 | no / **yes** | **6.810e-13** | 2.644e-14 | 4.628e-14 | **0.000 / 1.000** |
| **mixed, UNIFORM SPACER** | 4 | 324 | no / **yes** | 3.787e-11 | 7.654e-12 | 1.146e-14 | **0.000 / 1.000** |
| | 5 | 576 | no / **yes** | 2.354e-13 | 4.470e-14 | 2.538e-14 | **0.000 / 1.000** |
| | 6 | 900 | no / **yes** | 3.569e-13 | 6.746e-14 | 3.514e-14 | **0.000 / 1.000** |
| | 7 | 1296 | no / **yes** | 8.765e-14 | 1.257e-14 | 5.985e-14 | **0.000 / 1.000** |
| both out-of-plane | 4 | 324 | no / no | 2.888e-03 | 1.524e-04 | 6.609e-15 | 0.643 / 0.766 |
| | 7 | 1296 | no / no | 2.228e-05 | 6.219e-07 | 3.110e-14 | 0.742 / 0.671 |
| both slanted | 4 | 324 | no / no | 2.361e-03 | 1.070e-04 | 5.886e-15 | 0.656 / 0.754 |
| | 7 | 1296 | no / no | 9.954e-06 | 3.276e-07 | 2.993e-14 | 0.623 / 0.782 |

WSL reproduces `s_min/s_max`, `rcond` and the 0.000 / 1.000 split to every
digit printed, and the residuals to two (8.705e-15 / 1.649e-14 / 3.322e-14 /
5.067e-14 on the mixed rows).

**The verification's reading is CONFIRMED, and it is sharper than stated.**
The near-null right singular vector does not merely favour the promoted side --
it lies in it to 1.000 / 0.000 at every modal count on both builds, while the
two controls, whose sides are BOTH genuine forward/backward sets, show no
localisation at all (0.62-0.91 against 0.41-0.78).  The deficiency is a
property of the PROMOTION, not of the grid, the material or the modal count.

**One restatement.**  It is a NEAR-null space, not a null space: at the
standard tolerance the operand is FULL RANK on 7 of the 8 mixed rows (the
exception is the spacer at `M` = 7, rank 1293 of 1296).  "Rank-deficient by
construction" is therefore shorthand for "carries a singular value 10 to 13
decades below the largest, by construction" -- which is what defeats a
condition estimate.

### 3.2 The system is CONSISTENT, and that closes the mechanism quantitatively

The residual of the answer `lu_solve` returns is **8.13e-15 .. 5.99e-14** on
the mixed operands -- ordinary backward stability, indistinguishable from the
healthy controls' 5.89e-15 .. 3.11e-14.  The right-hand side's component along
the near-null LEFT direction, `||u_min^H B|| / ||B||`, is measured
**6.469e-03 / 1.735e-03 / 9.004e-04 / 4.973e-04** at `M` = 4 / 5 / 6 / 7.

That last row is the quantitative closure the verification could not make.  The
solution is DOMINATED by the near-null direction, so its RELATIVE error is not
`eps / (s_min/s_max)` but

```
    ||dx|| / ||x||  ~  eps ||B|| / ||u_min^H B||
```

which predicts a cross-build agreement of **3.4e-14 / 1.3e-13 / 2.4e-13 /
4.4e-13** at `M` = 4 / 5 / 6 / 7.  The verification measured, INDEPENDENTLY and
before this analysis existed, WIN-vs-WSL relative agreements of **1.10e-13 /
5.41e-14 / 2.48e-13 / 2.86e-13** on exactly those four solves.  Prediction and
measurement agree within a factor 3 at every rung and to 3 % at `M` = 6.  The
answer is well determined for THIS right-hand side, and the reason is now
arithmetic rather than observation.

---

## S4. DEFECT V1 -- what ships, and why the other two candidates were refused

The verification offered three fixes.  All three were measured
(`r2_populations.py`, `r3_residual_screen.py`) over three populations at the
generalized site:

* **HEALTHY** -- 28 ORDINARY per-layer stacks that reach the site (mixed /
  both-out-of-plane / both-slanted x `M` = 4..7 x non-conforming / conforming /
  wide, plus the uniform-spacer variants), narrowest segment 0.237 of the
  period = **237x the width contract**, extended in `r3` to `M` = 8 and to a
  three-layer stack whose middle interface has BOTH sides promoted;
* **SLIVER** -- the same shape with an intra-layer sliver and the width
  contract LIFTED (`PMM2D_STAG_MIN_SEG_GUARD = False`), `delta` = 3e-1 down to
  1e-11, scored against a device that CANNOT depend on the wall separation;
* **BROKEN** -- synthetic operands built FROM a real operand of this site: a
  repeated column (exactly singular), a right-hand side drawn from the range
  (rank-deficient but consistent), one with a component outside the range
  (inconsistent), and a zero column.

### 4.1 Candidate 1 -- an own `rcond` bar.  REFUTED: the populations CROSS

| population | `rcond`, WIN | `rcond`, WSL |
|---|---|---|
| HEALTHY, 28 ordinary stacks | **1.2567e-14 .. 1.5240e-04** | **1.2567e-14 .. 1.5240e-04** |
| HEALTHY at `M` = 8 (one point) | 4.7523e-15 | -- |
| SLIVER, 36 solves, contract lifted | **1.5917e-24 .. 1.1699e-05** | 1.5920e-24 .. 1.1699e-05 |
| exactly singular (synthetic) | 4.5364e-20 | 1.0932e-19 |

The healthy population and the population the guard exists for **overlap over
nine decades**.  There is no bar between them.  Worse, the healthy floor walks
DOWN the modal ladder (1.26e-14 at `M` = 7, 4.75e-15 at `M` = 8, and the
in-plane pair's own ladder falls ~3x per rung), so any fixed `rcond` bar is the
floor-bar shape `TESTING_STANDARDS.md` calls S4: it would be crossed by the
library's own legitimate refinement.  The verification's own estimate that such
a bar "would have to sit near 1e-17" is confirmed, and 1e-17 would still leave
only 2.6 decades to the exactly-singular reading while being walked toward from
above.

### 4.2 Candidate 2 -- leave the site UNGUARDED.  REJECTED, but narrowly

It is defensible on the same ground as the plain 1-D site: the correct
population comes within a decade of any bar, so no bar belongs there.  It was
rejected for one measured reason: it re-opens the failure D2 exists for.  A
zero column at this site factors without error, produces a NON-FINITE answer,
and would then propagate a `NaN` into the cascade
(`rcond` = 0.0, residual = `NaN`, measured).  An exactly singular operand
raises a bare `LinAlgError: Singular matrix` from `np.linalg.solve`.  Both are
exactly what round 2 removed at the other two sites, and both are still cheaply
detectable here -- just not by a condition number.

### 4.3 Candidate 3 -- screen on the RESIDUAL.  SHIPPED, with both gaps measured

| population | residual (exact), WIN | residual (exact), WSL |
|---|---|---|
| **HEALTHY, 23 solves, `M` = 4..8, `n` = 324..1764** | **4.801e-15 .. 1.188e-13** | **4.844e-15 .. 1.206e-13** |
| SLIVER, contract lifted, `delta` >= 1e-5 | 1.29e-15 .. 8.78e-14 | 1.89e-15 .. 6.8e-14 |
| SLIVER at `delta` = 1e-9 / 1e-11 (unreachable: refused by the width contract) | 4.09e-11 / 4.10e-10 | 6.8e-11 / 6.8e-11 |
| rank-deficient but CONSISTENT (the V1 shape) | **2.847e-14 -- ACCEPTED** | **1.264e-14 -- ACCEPTED** |
| **exactly singular, real right-hand side** | **1.198e-01** | **5.567e-02** |
| **inconsistent right-hand side** | **3.533e+01** | **1.219e+01** |
| zero column | **NaN -- REFUSED** | **NaN -- REFUSED** |

`_MORTAR_RESID_REFUSE = 1e-6`:

* **6.9 decades above** the worst healthy reading on either build, and the
  healthy population does NOT walk toward it -- the residual tracks `n eps`,
  which is what backward stability means, so the bar does not age with the
  modal ladder (measured 8.1e-15 at `n` = 324 rising to 1.19e-13 at
  `n` = 1764, i.e. linearly in `n`, exactly as predicted);
* **4.7 decades below** the closest broken reading on either build;
* and it makes the RIGHT distinction: the same rank-deficient operand passes
  with a consistent right-hand side (2.85e-14) and is refused with an
  inconsistent one (3.53e+01).

**What it does NOT do, stated plainly.**  It is NOT a backstop for the width
contract, and no residual could be -- M1's finding holds at this site too.
With the contract lifted, the answer on a `delta`-independent device is 2.0e-02
wrong at a 1e-03 wall separation while the residual reads 7.3e-15, squarely
inside the healthy band, and it stays there down to 1e-05.  **The width
contract is the only line against a sliver on the per-layer path.**  That is
recorded here, in the constant, and in the round-2 doc's corrected S4.4.

### 4.4 The change

`lumenairy/elements/pmm/_core.py`:

```python
_MORTAR_RESID_REFUSE = 1e-6          # + ~70 lines of derivation
_MORTAR_PROBE_CACHE  = {}
def _mortar_probe(k):     ...        # a DETERMINISTIC generic probe vector
def _mortar_residual(A, X, B, probe=True):  ...
def _guarded_mortar_solve(A, B, site, ga=None, gb=None, hint=None,
                          screen="rcond"):   ...
```

`screen='rcond'` is the default and is the round-2 path, unchanged; the
generalized site passes `screen='residual'`.  The two IN-PLANE call sites are
untouched, character for character.

The census hook's tuple grew from `(site, n, rcond, refused)` to
`(site, n, rcond, refused, residual)`; every shipped consumer indexes `[0]`
through `[3]`, so nothing breaks (checked: two tests, three probes).

**Cost.**  The screen reads the residual on ONE deterministic generic probe
vector -- `y = X v`, then `||A y - B v|| / ||B v||`, three matvecs -- and pays
the exact `O(n^3)` Frobenius residual ONLY when the probe exceeds the bar, so
an unlucky probe can never refuse a good solve.  FLOP argument first, because
it does not depend on a shared box: `lu_factor` + `lu_solve` with `n`
right-hand sides is `(2/3 + 2) n^3` complex multiply-adds and the probe is
`3 n^2`, i.e. `1.7 / n` of the call it rides -- 0.35 % at `n` = 324, 0.06 % at
`n` = 1764.

Measured, MINIMUM of 9 INTERLEAVED repetitions (a minimum is the only
load-robust statistic on a shared box), against the bare `lu_factor` +
`lu_solve` at `n` = 324 / 576 / 900 / 1296 / 1764 (`r4_cost.py`):

| arm | WIN | WSL |
|---|---|---|
| + round 2's `gecon` screen alone | 1.03 .. 1.13x | 1.01 .. 1.06x |
| **+ this residual PROBE on top** | **1.07 .. 1.14x** | **0.99 .. 1.10x** |
| + the EXACT Frobenius residual | 1.57 .. 1.76x | 1.47 .. 1.72x |

So the probe costs **0-6 % over the screen that was already there** -- its own
flop share plus noise -- while the exact residual costs 47-76 %.  (The `gecon`
arm's own 3-13 % is not new: its `anorm` allocates an `n x n` temporary, and it
is round-2 code.)

The probe is a fixed-seed `PCG64` draw, cached per width: bit-reproducible on
every platform numpy supports, so it is a CONSTANT of the library rather than a
random number.  It is random-LOOKING on purpose -- a structured probe can be
orthogonal to a structured inconsistency and a generic one is not, with
probability one.  Probe and exact residual agree within **1.6x over all 51 real
operands measured**.

---

## S5. DEFECT V1 -- the coverage that was missing

The verification's S8 named the hole exactly:
`test_the_mortar_rcond_bar_has_decades_of_gap_on_both_sides` builds only
in-plane fixtures, `test_pmm2d_staggered_oop.py` contains no per-layer stack at
all, and `test_verify_pmm2d_perlayer_slant.py`'s four tests slant BOTH layers,
which is the healthy branch.  `tests/unit/test_fix_pmm2d_mortar_round3.py`
adds, with both neighbours the verification scoped:

* **(a) a plain UNIFORM SPACER** -- the commonest configuration, and the one
  round 2 refused from `n_modes` = 5.  `M` = 4, 5, 6 all solve; the modal
  ladder's steps fall 2.338e-04 -> 3.833e-06 (61x); and the answer is compared
  with the MORTAR-FREE arm of the SAME DEVICE (the spacer carried on the
  out-of-plane layer's own wall array, where the identical-grid bypass removes
  every mortar).  The two independent paths agree to 2.123e-05 at `M` = 6 --
  11x inside the ladder's own coarsest step -- and that agreement improves
  monotonically (3.110e-05 / 2.779e-05 / 2.123e-05).  `M` = 7 is its own test
  (it is the most expensive rung and the one round 2 refused hardest);
* **(b) an IN-PLANE PATTERNED neighbour on other walls**, against the same
  device built on the COMMON REFINEMENT of the two wall arrays -- which is
  conforming, so it takes no mortar and is an independent numerical path.  The
  fixture's second pillar SHARES one wall with the first so the union is four
  segments rather than five, which is what keeps the oracle affordable in a
  gate (the v8 union at `M` = 4 is 44 s of solve on this box; this one is 12).

Both are asserted as DECISIONS -- solves, converges, agrees with a
mortar-free twin within a bar the test derives from the running build -- with
no cross-build value pins.

### 5.1 The v8 table, before and after

Zeroth-order `R` for incident `E_y`, the verification's own reproducer, on this
branch:

| `M` | `24651c8` (verification) | round 2 | **round 3** | union oracle (verification) |
|---|---|---|---|---|
| 4 | 0.06072107 | 0.06072107 | **0.06072106726** | 0.06436131 |
| 5 | 0.06365828 | **REFUSED** | **0.06365827998** | 0.06435761 |
| 6 | 0.06396188 | **REFUSED** | **0.06396187804** | 0.06460829 |
| 7 | 0.06443418 | **REFUSED** | **0.06443417942** | 0.06460797 |

The three refusals are gone and the answers are the pre-round-2 ones to 9
significant figures (the residual difference is round 2's D3 quadrature rule,
which this fixture's 3-segment grids do exercise; it is 1e-11 absolute and
moves the convergence not at all).  The lossless closure improves with `M`
exactly as the verification measured (1.084e-03 / 1.139e-03 / 3.038e-04 /
2.942e-04).

---

## S6. S5.4 -- the SILENT degradation band now warns

**What was wrong.**  Above the 1e-3 width contract the solve is accepted and is
MEASURABLY less accurate, by a factor that grows as the segment narrows, that
`n_modes` does not remove, and that no energy tripwire can see (the lossless
closure stays pinned five decades under `_STAG_CLOSURE_TOL`).  Worse, the
`n_modes` ladder FLATTENS in the band, so a user doing the obvious convergence
study reads the flattening as convergence.

**What ships.**  `twod_staggered.py`:

```python
PMM2D_STAG_SLIVER_BAND_WARN = True    # switch, same status as the D1 switch
_STAG_SLIVER_BAND_FRAC      = 3.0e-2  # the band's UPPER edge
def _warn_stag_sliver_band(grids, mortared, fn=...): ...
```

called from `PMM2DStackPure._solve_per_layer` immediately after the per-layer
grids are built -- the ONE place where the neighbours are known.  It is a
`UserWarning`, never a refusal, it names the layer, the axis, the width, the
whole wall array, the measured degradation class and FIVE remedies, and it
tells the reader how to silence it.

**It is conditioned on the stack actually building a mortar.**  A fully
CONFORMING per-layer stack takes the plain square modal match at every
interface; warning there would be the same false positive as DEFECT V2.
Measured with the contract lifted (`r6_v2_conforming.py`): such a stack's
answer moves by **1.100e-04 / 2.505e-04 / 2.384e-06** at `M` = 4 / 5 / 6
between `delta` = 1e-4 and 1e-6, and that movement scales like `delta` (the
1e-3 -> 1e-6 movement is 10x larger), i.e. it is the midpoint-sampled DEVICE
changing, not the numerics.

### 6.1 The band's upper edge is a DERIVED bar, and the derivation is a conflict

`r5_degradation_band.py` re-measures the S5.4 curve on a fixture independent of
the verification's -- different period (0.93 vs 1.05), wavelength (0.66 vs
0.71), angle, contrast, wall positions and sliver centre -- with a finer ladder,
against the exact 1-D `PMMStack` at degree 14 whose own 12 -> 14 self-gap is
**1.491e-05**, three decades under the errors being compared.  The device is a
y-uniform 3-layer stack whose MIDDLE layer is ALL HOST, so it cannot depend on
the wall separation at all.

Ratio `err(delta) / err(3e-01)`:

| narrowest / period | 3e-1 | 2e-1 | 1.5e-1 | 1e-1 | 7e-2 | 5e-2 | **3e-2** | 2e-2 | 1e-2 | 3e-3 | 1e-3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| this fixture, `M`=6 | 1.00 | 1.03 | 1.06 | 1.13 | 1.20 | 1.25 | **1.32** | 1.37 | 1.44 | 1.55 | 1.66 |
| this fixture, `M`=7 | 1.00 | 1.06 | 1.16 | 1.31 | 1.41 | 1.49 | **1.56** | 1.59 | 1.62 | 1.64 | 1.64 |
| **this fixture, `M`=8** | 1.00 | **2.28** | 3.21 | 4.04 | 4.40 | 4.55 | **4.65** | 4.69 | 4.74 | 4.81 | **4.87** |
| verification's fixture, `M`=8 | 1.00 | -- | -- | 2.09 | -- | -- | **4.02** | 5.31 | -- | 5.76 | 5.86 |

**Every rung of every row is IDENTICAL on WIN and WSL to the three decimals
printed**, and the oracle's own self-gap agrees to 11 significant figures
(1.4911507085e-05 on both).  This is a pure discretisation quantity; it has no
meaningful cross-build spread, which is what makes a 2x bar on it a decision
rather than noise.

The floor is only VISIBLE at rungs where the ordinary arm has converged below
it -- at `M` = 8 the ordinary arm reads 4.678e-04 while the band arm saturates
at ~2.2e-03 -- which is why the `M` = 6 and 7 rows are mild.  That is a property
of the instrument, and it is also the reason the warning cannot be replaced by
a convergence study: the study is what hides it.

**The two constraints on the edge pull in opposite directions.**

* ACCURACY says as WIDE as possible.  The bar is "the measured cost exceeds
  2x", and on the `M` = 8 ladder that is first true at **2e-01** on this
  fixture and at **1e-01** on the verification's.
* FALSE POSITIVES say as NARROW as possible.  The narrowest segment any
  ORDINARY geometry the library builds asks for is **1.2500e-01** in the
  shipped battery (a nested refinement) and **1.0937e-01** in the
  verification's wider census (`add_tapered_pillars` at 16 slices).

So the accuracy criterion ALONE would put the edge on top of the ordinary
population: 2e-01 is wider than every per-layer geometry except a single
interior wall and a duty-1/3 pair.  **The census is the binding constraint.**
3e-02 is the largest round decade-third that keeps a stated factor on both
sides:

* **4.65x** measured cost at the edge on this fixture (4.02x on the
  verification's), so the 2x bar is cleared by **2.3x**;
* **3.6x below** the narrowest ordinary geometry (4.2x below the shipped
  battery's), so **no ordinary stack warns** -- censused on the running build
  by `test_no_ordinary_geometry_the_library_builds_lands_in_the_band`.

The band's LOWER edge is `_STAG_MIN_SEG_FRAC` itself; below it the stack is
refused, so the warning would never be read.  Both edges carry the same 1e-9
RELATIVE slack the contract does, so the two rules are exactly complementary:
a grid the contract accepts is either in the band or above it, never neither.
(Measured, and this is why the slack is there: `0.47 - 0.44` is
2.99999999999999970e-02 -- 3e-17 UNDER 3e-2.)

### 6.2 The one geometry that DOES warn, and it should

A taper whose tip CLOSES is the surface the round-2 census identified as
walking toward the contract; its narrowest sampled width is
`~ w_bottom / (2 n_slices)`.  Measured on the shipped battery:
**8.0094e-03 at 32 slices** and **4.1047e-03 at 64** -- both inside the band,
and the verification measures a 5.7x accuracy cost at the second.  That is the
intent, not a false positive, and the gate asserts it.

---

## S7. The smaller items

### 7.1 DEFECT V2 (P3) -- OPEN, with the reason measured

The width contract refuses a CONFORMING per-layer stack, which has no
cross-grid projection anywhere.  The verification asked for a `solve()`-level
fix "where neighbours are known".  Measured (`r6_v2_conforming.py`), a wall
array with a 5e-4 segment is refused INDEPENDENTLY by all three constructors
that build a `Basis1D`:

| constructor | outcome |
|---|---|
| `Basis1D(P, walls, 4)` | `ValueError: Basis1D: segment 1 ... 5.000e-04 of the period` |
| `StagGridOps(P, P, walls, walls, 4, 1, 1)` | the same, from its own `Basis1D` |
| **`Granet2DTransverseE(P, P, walls, walls, 4, cell)`** | the same, from ITS own `Basis1D` |

`Granet2DTransverseE` is the REGION EIGENSOLVER and a public class: the hybrid
engine, the shipped tests and the probes construct it directly.  So an
exemption reached from `PMM2DStackPure.solve` would have to thread a keyword
through BOTH `StagGridOps` and `Granet2DTransverseE` -- two hot-path
constructors, one of them a documented surface -- to reach the place that
refuses.  That is not the two-line change the severity justifies, and a
module-global flag flipped around grid construction would not be re-entrant.

**Recorded as an open item.**  The class is real (the conforming stack IS
`delta`-insensitive, S6 above) and the user's remedy is one keyword
(`layer_grids='shared'`, or put the fine feature on a lattice that resolves
it).  If it is taken up, the clean shape is a keyword-only `min_seg_guard=`
on `Basis1D`, `StagGridOps` and `Granet2DTransverseE`, defaulting `True`, set
`False` by `_solve_per_layer` when `len({g.key() for g in gof}) == 1` -- the
same conformity test the round-3 band warning already computes.

### 7.2 V3 / V4 / V5 / V6 -- documentation, corrected in place

Dated CORRECTION paragraphs were added to
`docs/audits/FIX_PMM2D_MORTAR_ROUND2_2026_09_11.md` (S2.5, S3.1, S4.4) and to
the constants themselves (`_STAG_MIN_SEG_FRAC`, `_MORTAR_RCOND_REFUSE`).
Nothing in the round-2 doc was rewritten.

* **V4** -- the E-row exponent "~2.0" is the TWO-AXIS value; it is **1.0 per
  axis carrying the sliver** (1.010 / 1.006 on a one-axis fixture against
  1.995 / 1.994 on a two-axis one).  The CROSS-MASS exponents `C1_x` / `C2_x`
  are SATURATED at the float64 ceiling on both fixtures and disagree between
  `M` = 4 and 6 by 2.4x; they are now marked as such.  The H row being faster
  than the E row on the same slot reproduces on both fixtures and is BOUNDED,
  not pinned.
* **V3** -- the conditioning backstop is NOT an independent second line.  Its
  `MassH_A V_A` operator is built from the FIRST grid, so a sliver occupying
  ONE axis in the LAST layer leaves it `delta`-INDEPENDENT (1.384e+05 at every
  `delta` from 3e-1 to 1e-6) and nothing is refused down to `delta` = 1e-7.
  Round 3 adds that at the GENERALIZED site it does not screen conditioning at
  all any more.  **The width contract is the only line against a sliver.**
* **V5** -- the 1e-3 bar is a pure PERIOD FRACTION with no wavelength in it, so
  on a 40 um period it refuses a physically ordinary 40 nm feature.
  Defensible (the conditioning is governed by `w/d` on one period, which is why
  the spurious spectrum carries no `k0`) but previously unstated; the message's
  remedy (2) is also clarified -- the shared lattice's own cells are the same
  width, and they are harmless there because a uniform lattice's mortar is the
  identity.
* **V6** -- the contract fires at `PMM2DStackPure.solve()`, not at
  `add_layer`, so a 400-slice taper is built in full before it is refused.

---

## S8. BIT-IDENTITY

The round-2 verifier's own harness was REUSED rather than rewritten -- 30
fixtures, 165 sha256 hashes over `dtype|shape|tobytes` of every returned array,
PLUS the warning set: shared-grid scalar / tensor / OUT-OF-PLANE / magnetic /
slanted / `retain_internal` / per-order amplitudes / `jones=False` / `M`=6
`N`=4; per-layer conforming, non-conforming, nested, mixed, per-layer
`n_modes`, both taper builders (midpoint and `rule='bottom'`), tensor, slant,
magnetic, `retain_internal`, uniform-ARRAY and INTEGER spellings; four 1-D
`PMMStack` arms; and the projector directly on both global stencil sets.

`with` = this worktree, `pre` = a pristine copy of the three changed files at
`1957da2`:

```
IDENTICAL 165 / 165 hashes; 0 mismatches; 0 warning-set differences
r3with: C:\tmp\lum_mortar3\lumenairy\__init__.py  py3.14.6 np2.4.4
r3pre : C:\tmp\lum_r3pre\lumenairy\__init__.py    py3.14.6 np2.4.4
```

**Nothing in the battery moved a bit, and nothing gained or lost a warning.**
The only behaviour that changes is the one intended: previously-REFUSED mixed
in-plane / out-of-plane stacks now return an answer (none of which is in the
battery, because round 2 could not solve them to hash).

---

## S9. TESTS

`tests/unit/test_fix_pmm2d_mortar_round3.py`, 14 gates.  Every population is
RE-MEASURED on the running build; every bar states its measured value and its
margin on both sides; there is no cross-build value pin anywhere in the file
(one measured reading, `R00` at `M` = 7, is recorded in a comment and
explicitly NOT asserted).

| gate | what it decides | the bar, and the measured margin |
|---|---|---|
| `..._operand_is_rank_deficient_on_the_promoted_side` | the MECHANISM | `s_min/s_max` of the mixed operand is 4 decades under the both-out-of-plane control's (measured 7.1); the near-null vector's mass on the promoted side > 0.95 (measured 1.000) and < 0.05 on the other (measured 0.000), against the control's 0.643 / 0.766; the residual is 2 decades under the bar (measured 8 decades) |
| `..._beside_a_uniform_spacer_solves_and_converges` | V1, the commonest configuration | 3 rungs solve; the ladder's step falls 5x per rung (measured 61x); the mortar-free twin agrees within half the coarsest step (measured 11x inside) |
| `..._solves_at_the_modal_count_round_2_refused_hardest` | V1 at `M` = 7 | returns, finite, closure |
| `..._agrees_with_its_common_refinement_twin` | V1, an in-plane PATTERNED neighbour | the per-layer arm agrees with the mortar-free common-refinement arm within 5x its own `M` = 4 -> 5 step |
| `..._in_plane_rcond_bar_would_refuse_this_ordinary_stack` | FAIL-BEFORE | the SAME operand raises under `screen='rcond'` and returns under `screen='residual'`, bit-identical to `np.linalg.solve` |
| `..._generalized_residual_bar_has_decades_of_gap_on_both_sides` | the bar | healthy < 1e-4 x bar (measured 1e-7 x); exactly-singular and inconsistent > 1e4 x bar (measured 1e5 x); the best-possible (least-squares) residual on the singular operand > 1e2 x bar (measured 1.24e3 x); the rank-deficient-but-CONSISTENT operand passes |
| `..._residual_probe_tracks_the_exact_residual_on_real_operands` | the estimator | probe / exact within 100x (measured 1.6x), and the probe is bit-reproducible across two independent draws |
| `..._inconsistent_generalized_mortar_is_refused_by_name` | the refusal | `_ConditioningError` (an `_EnergyError`), five message tokens, and a zero column refused via the NaN path |
| `..._two_in_plane_mortar_sites_keep_the_round_2_decision` | no collateral | the default screen is still `rcond`, still bit-identical to `np.linalg.solve`, still refuses at 1e-12; and the same operand with a consistent RHS passes the residual screen -- which is why the two sites cannot share a bar |
| `..._degradation_band_warns_inside_it_and_is_silent_outside` | S5.4, two-sided | warns at 2e-2 (1.5x inside), silent at 6e-2 (2x outside), warns at 1.2e-3, refuses at 9e-4; seven message tokens |
| `..._no_ordinary_geometry_the_library_builds_lands_in_the_band` | the census that BINDS the edge | the worst ordinary geometry > 3x the edge (measured 4.2x); the two closest classes solve SILENTLY; and the closing taper at 32 / 64 slices DOES land in the band |
| `..._conforming_per_layer_stack_in_the_band_does_not_warn` | no V2-shaped false positive | conforming silent, non-conforming warns, same width |
| `..._band_switch_restores_the_round_2_silence` | FAIL-BEFORE | silent with the switch off, warns with it on |
| `..._band_the_warning_names_carries_a_measurable_cost` | the band is real | against the exact 1-D `PMMStack`, `err(3e-3) > 1.15 err(3e-1)` at `M` = 6 (measured 1.55), with the oracle's own self-gap asserted to be < 5 % of the difference being claimed (measured 0.1 %) |

### 9.1 Runs

**The seven required suites, BOTH builds, one thread**
(`test_fix_pmm2d_mortar_round3.py` + `test_fix_pmm2d_mortar_round2.py` +
`test_pmm2d_staggered_mortar.py` + `test_pmm2d_staggered_nonuniform.py` +
`test_verify_pmm2d_perlayer_slant.py` + `test_pmm2d_staggered_oop.py` +
`test_pmm2d_staggered_slant.py`):

```
WIN   185 passed, 3 warnings in 982.03s (0:16:22)
WSL   185 passed, 3 warnings in 950.47s (0:15:50)
```

**The whole `pmm2d` surface, Windows** (`test_pmm2d*.py` +
`test_fix_pmm2d*.py` + `test_verify_pmm2d*.py`):

```
367 passed, 24 warnings in 1506.16s (0:25:06)
```

**The census / walker / dispatcher-pin sweep**:

```
1284 passed, 12 skipped, 4 warnings in 223.22s (0:03:43)
```

That sweep is the one that caught something: the library pins that EVERY
module-level `*_CACHE` carries a companion `threading.Lock`
(`test_v4_14_2_dispatcher_pin_cache_locks.py`), and the residual screen's probe
cache had none.  It now has `_MORTAR_PROBE_LOCK`, uses it double-checked, hands
back a read-only array, and is cleared by `_clear_pmm_caches` like the two
Lagrange memos beside it.

**A final confirmation run on the committed code, both builds**
(`round3` + `round2`, 29 gates):

```
WIN    29 passed, 3 warnings in 217.68s (0:03:37)
WSL    29 passed, 3 warnings in 235.59s (0:03:55)
```

Slowest round-3 gate 38.2 s (`no_ordinary_geometry...`, which builds the whole
round-2 battery and solves two of its stacks); every other one is under 26 s.

**THE THREE WARNINGS, and they are the right three.**  One is the NEW band
warning, fired by `test_the_mortars_own_algebra_is_exact_at_every_wall_separation`
-- a round-2 gate that DELIBERATELY sweeps wall separations from 1e-1 down to
1e-4, i.e. straight through the band.  The other two are round 2's own open
item B (`RuntimeWarning: invalid value encountered in multiply`), unchanged.
Across 367 tests on the whole `pmm2d` surface the new warning fires on exactly
that one gate.

**`ruff check lumenairy/ tests/ validation/probe_fix_mortar_round3/`**:
`All checks passed!` (WSL).

---

## S10. OPEN ITEMS, and what could NOT be measured

1. **DEFECT V2 stays open** (S7.1), with the reason measured rather than
   asserted: the width contract is raised from `Basis1D.__init__`, which is
   constructed independently by `StagGridOps` AND by `Granet2DTransverseE` --
   the region eigensolver, a public class.  The clean shape of the fix, and the
   conformity test it would use, are written down in S7.1 for whoever takes it.
2. **A THIRD BLAS FAMILY.**  Both builds link scipy-openblas, as in the build,
   both fixes and both verifications.  Every cross-build spread here is a LOWER
   bound.  It matters least for this round's bar (the residual's two-sided gap
   is 4.7 and 6.9 decades, against a measured cross-build spread of under 2x on
   every reading) and most for the S3.2 build-stability argument.
3. **`M` > 8 at the generalized site.**  The healthy residual population is
   measured to `M` = 8 (`n` = 1764) and is LINEAR in `n` there, which is what
   backward stability predicts; `M` = 9 on a 3-segment grid is a `4 q^2` = 2304
   eig per layer and was not run.  The `rcond` population's downward walk with
   `M` is measured to 8 and extrapolated beyond, which is one of the reasons a
   `rcond` bar was refused.
4. **The exact-singular operand is SYNTHETIC.**  No geometry reachable through
   the public API was found that drives the generalized site to exact
   singularity: with the width contract ARMED the sliver ladder stops at 1e-3,
   and with it LIFTED the factorisation still succeeds at `delta` = 1e-11
   (`rcond` 5.2e-31, residual 4.1e-10).  The broken population is therefore
   built FROM a real operand of the site rather than found in the wild, and
   that is stated rather than hidden.  It is the same shape round 2's own D2
   reproducer has.
5. **The degradation ladder ran to completion on BOTH builds** and every rung
   of every row (`M` = 6, 7, 8 x eleven widths) is identical to the three
   decimals reported, with the 1-D oracle's own self-gap agreeing to 11
   significant figures.  What was NOT measured is `M` > 8, where the floor
   would be even more visible; at 3 segments that is a `2 q^2` = 1568 region
   eig per layer per rung and the `M` = 8 ladder already costs 35 minutes.
6. **The verification's 165-hash battery was REUSED, not rebuilt.**  A
   deliberately independent battery would be a stronger claim; reusing theirs
   is a weaker one, and it is what this round did, because the change is a
   two-site diff whose blast radius is exactly what that battery covers.

---

## S11. Commands

```sh
# probes, both builds
for s in r1_mechanism r2_populations r3_residual_screen r4_cost \
         r5_degradation_band r6_v2_conforming; do
  python validation/probe_fix_mortar_round3/$s.py win
done
wsl -e bash -lc 'cd /mnt/c/tmp/lum_mortar3 && for s in ...; do \
  ~/lumvenv/bin/python validation/probe_fix_mortar_round3/$s.py wsl; done'

# bit-identity (the round-2 verifier's own harness, against a pristine copy of
# the three changed files at 1957da2)
mkdir -p /c/tmp/lum_r3pre && cp -r lumenairy /c/tmp/lum_r3pre/
for f in lumenairy/elements/pmm/{_core,twod_staggered,stack2d_pure}.py; do
  git show HEAD:$f > /c/tmp/lum_r3pre/$f; done
V1_TAG=r3pre  PYTHONPATH=/c/tmp/lum_r3pre  V1_EXPECT_ROOT=/c/tmp/lum_r3pre \
  python validation/probe_verify_mortar_round2/v1_bitid.py
V1_TAG=r3with PYTHONPATH=/c/tmp/lum_mortar3 V1_EXPECT_ROOT=/c/tmp/lum_mortar3 \
  python validation/probe_verify_mortar_round2/v1_bitid.py
python validation/probe_verify_mortar_round2/v1_compare.py r3with r3pre

# the seven suites, BOTH builds, ONE thread
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m pytest tests/unit/test_fix_pmm2d_mortar_round3.py \
  tests/unit/test_fix_pmm2d_mortar_round2.py \
  tests/unit/test_pmm2d_staggered_mortar.py \
  tests/unit/test_pmm2d_staggered_nonuniform.py \
  tests/unit/test_verify_pmm2d_perlayer_slant.py \
  tests/unit/test_pmm2d_staggered_oop.py \
  tests/unit/test_pmm2d_staggered_slant.py -q -p no:randomly --durations=0

# the whole pmm2d surface, Windows
python -m pytest tests/unit/test_pmm2d*.py tests/unit/test_fix_pmm2d*.py \
  tests/unit/test_verify_pmm2d*.py -q -p no:randomly

# census / walker / dispatcher sweep
python -m pytest $(ls tests/unit | grep -iE \
  'walker|census|dispatcher_pin|public_api|doc_consistency' \
  | sed 's#^#tests/unit/#') -q -x -p no:randomly

# ruff
wsl -e bash -lc 'cd /mnt/c/tmp/lum_mortar3 && ~/lumvenv/bin/ruff check \
  lumenairy/ tests/ validation/probe_fix_mortar_round3/'
```

**A trap worth recording, because it cost two runs.**  Both
`test_fix_pmm2d_mortar_round2.py` and `test_fix_pmm2d_mortar_round3.py` set
`OMP/OPENBLAS/MKL_NUM_THREADS` with `os.environ.setdefault` at import time.
That works when the file is run ALONE and does NOTHING in a multi-file run,
because some earlier file has already imported numpy: the seven-suite run went
out at seven threads and did 10 tests in 50 minutes.  **Set the thread
variables in the SHELL for every multi-file run** -- the same run then does the
same 10 tests in about 3.

**A second trap, in the same class.**  A test that borrows another test
module's geometry must borrow its UNITS too.  The round-2 battery is built with
period 1.2 and wavelength 0.85; driving it with this file's 0.62e-6 makes `k0`
1e6 too large, and round 2's own per-segment quadrature rule then asks for a
93991-node Gauss-Legendre rule and dies in `numpy.polynomial` trying to
allocate 65.8 GiB.  The census gate takes its source from the module that owns
the geometry.

# M2 -- "The window is the knob": the per-layer window as an honest, tested contract

**Date:** 2026-08-04 - **Branch:** `feat/pmm-per-layer-roadmap` (on top of M1, `d30f1ca`)
**Plan:** `docs/audits/PMM_PER_LAYER_CAMPAIGN_PLAN_2026_08_04.md` S4 mission M2 (N-4, N-6, T3-1, T3-2)
**Parents:** `AUDIT_PMM_OBLIQUE_INPLANE_UNION_GRID_2026_07_28.md` (the defect),
`AUDIT_PMM_PER_LAYER_GRIDS_IMPL_2026_07_28.md` (the implementation),
`PMM_M1_CONDITIONING_2026_08_04.md` (the conditioning census this builds on)
**Status:** implemented, measured, pinned. Everything uncommitted.

Evidence tags follow the campaign convention: **[M]** measured by a run, **[A]** analysis against
the tree, **[H]** hypothesis, flagged.

---

## 0. Summary

| item | outcome |
|---|---|
| **N-4** one window helper | **SHIPPED.** `_perlayer_window_grids` replaces five verbatim copies. Byte-identical at `halfwidth=1`, **max diff 0** over 113 captured arrays across all five sites and over 5 stacks x 4 `min_feature` on the grids themselves, **both builds**. |
| **N-4 bonus** | New **continuity pin**: at `window_halfwidth >= nlay-1` the per-layer path reproduces the **shared union grid bit-for-bit** (0.0). The knob degenerates to a known reference instead of being a free parameter. |
| **N-6** `min_feature` contract | **v5.32.0's "inert by construction" is WRONG, and wrong in the dangerous direction.** Corrected, quantified, and reduced to a *predictive rule* that was measured correct on **12/12** cells. |
| **T3-1** window width | **Measured, opt-in, default NOT flipped.** With the snap confound removed (S4.1), `halfwidth=2` moves the answer by 2.7e-4 / 8.9e-5 / 6.9e-6 at degree 6 / 8 / 10 -- a **spectrally decaying** residual of the mortar's own class -- at 1.2-2.7x time and 1.5-2.3x peak memory. |
| **T3-2** ns 8..12 stationarity | **Measured and pinned**, but *only conditionally*: the staircase is stationary in `n_slice` **and** in `degree` once the geometry is representable, and is not before. The envelope is stated with its precondition. |
| **NEW FINDING (P1)** | A **silent-wrong** regime found and characterised: a ~1-2 nm cross-layer cell (on a 700 nm pitch) makes the modal forward/backward classification degree-dependent, and the cascade returns a **unitary but wrong** S-matrix with `\|R+T-1\| ~ 1e-7`. Present on **both** grid paths and pre-dating M2. **M5 escalated the same defect independently** the same day, from the simplest member of the class. See S5 (mechanism) and S8 (reconciliation). |
| **Reconciliation with M5** | S5's mechanism **predicts M5's 80-cell table** without a new hypothesis, and answers their open question (the trigger is a *thin cell*, whatever makes it -- not the union grid). The N-6 rule's prediction record goes to **24/24** across a coated and an uncoated device, and the mitigation is measured **not** to flatten the taper. One correction to M5's S4: their unadjudicated PMM-vs-RCWA offset is **RCWA's truncation**, not a PMM bias. |

The headline for a user is one line: **the per-layer path did not retire `min_feature`; it is
still the accuracy lever, and the library said otherwise.**

The headline for the campaign is a second line: **the thing `min_feature` is a lever ON is a
silent-wrong mode-classification defect that no shipped guard detects** -- so T3-4 (the
grid-quality observable, currently deferred as "blocked on an unresolved mechanism") is
**unblocked and should be promoted**. S5.5 is the hand-off.

---

## 1. Builds, device, and one correction to M1's header

**Builds.** Both-builds evidence throughout.
* *Windows*: python 3.14.6, numpy 2.4.4, **scipy-openblas 0.3.31, `Haswell` kernel**.
* *WSL "CI proxy"* (`~/lumvenv`): python 3.12, numpy 2.4.6, **scipy-openblas 0.3.31, `SkylakeX` kernel**.
* `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` on both.

**[A] Correction to `PMM_M1_CONDITIONING_2026_08_04.md:13`,** which labels the Windows arm
"*numpy 2.4.6 + MKL*". Measured on this box via `threadpoolctl.threadpool_info()`: the Windows
interpreter is numpy **2.4.4** on **`libscipy_openblas64`**, not MKL, and `where python` finds only
that interpreter. The two arms still differ in numpy version (2.4.4 / 2.4.6), python (3.14 / 3.12)
and **BLAS micro-architecture** (Haswell / SkylakeX), so they remain a genuine cross-build pair --
but they are *not* MKL-vs-OpenBLAS, and a future reader chasing an MKL-specific effect would be
misled. M1's conclusions do not depend on the label.

**Device.** The exp21 coated-pillar out-coupler and `exp11_common` live in a different repository
and are not on this machine, so the device is **rebuilt from the parameters printed in the parent
audit** (S4.4 and S12): 700 nm pitch, `lambda` = 1310 nm, sidewall 2.0 deg, tapered region
310 nm, **5.00 nm conformal coat**, `n_sup = n_sub = 1.50`, `theta = 8 deg`, midpoint slicing.
Core `n = 3.48` (lossless; a Cu-like `0.30 + 8.0i` twin is used for the absorption checks),
coat `n = 1.76` (Al2O3), groove air.

**The reconstruction is validated against the audit's own measurement.** The per-slice wall offset
is `(310/ns) tan(2 deg)` nm, so the window's adjacent-slice wall separations are exactly
`{off, |5.00 - off|}` nm:

| `ns` | 2 | 3 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| `off` (nm) | 5.4127 | 3.6085 | 2.7064 | 1.8042 | 1.3532 | 1.0825 | 0.9021 |
| `\|5 - off\|` (nm) | **0.4127** | 1.3915 | 2.2936 | 3.1958 | 3.6468 | 3.9175 | 4.0979 |

`ns = 2` reproduces the parent audit's measured **0.41 nm** collision pair exactly (S4.4:
"`155 x tan 2deg - 5.00 = 0.41 nm` matches three of six measured pair separations"), and the
minimum window cell at the library default is measured at **0.4127 nm**. The surrogate reproduces
the *collision geometry* the audit identified; it is not the exp21 device's optical response, and
no claim here depends on it being.

**Oracles.**
1. **RCWA** (`RCWAStack`, Fourier-modal, **analytic shape form factors** so the nm walls are exact
   and not pixel-quantised) -- shares no assembly code with the PMM nodal cascade. `|R+T-1| ~ 1e-12`
   at 141 orders.
2. **The shared-grid path** on conforming stacks -- bit-exact, and now also via the new
   `halfwidth >= nlay-1` continuity identity.
3. **`|R+T-1|`** -- reported everywhere, and shown in S5 to be **near-tautological on this device**
   (only order 0 propagates at 700 nm pitch / 1310 nm / `n_sup` 1.5, so `n_sup P / lambda = 0.80 < 1`),
   which is precisely the standing caution in the plan's rule 5. The **lossy twin's**
   `sum(A)` vs `1 - R - T` is the non-tautological conservation check and is reported with it.

**Machine load caveat [M].** This box was concurrently running other agents' jobs throughout
(8 WSL `python3.12` processes at 10-20 ks CPU, 11 Windows `python`; 96 GB of 128 GB free, so no
memory pressure -- the 2026-07-18 lesson applies and was checked). An idle-machine absolute timing
was therefore not available. **All timing claims below are interleaved A/B ratios**, which bias
both arms equally, and are labelled as ratios, never as absolutes.

---

## 2. N-4 -- one window helper

### 2.1 What was duplicated [A]

The interface-conforming window construction stood verbatim at **five** sites at `d30f1ca`:

| file:line | path |
|---|---|
| `stack.py:1654` | `_solve_vertical_perlayer` -- classical `solve()` |
| `stack.py:1838` | `_solve_general_perlayer` -- slant / OOP `solve()` |
| `stack.py:2923` | `_solve_vs_wavelength_perlayer` -- the sweep |
| `conical.py:239` | `_conical_nodal_solve` -- `phi != 0` |
| `_jax_stack.py:482` | `_pmm_stack_solve_jax_perlayer` -- the JAX twin |

all reading `js = [j for j in (i - 1, i, i + 1) if 0 <= j < nlay]`.

### 2.2 The helper

`lumenairy/elements/pmm/_core.py`:

```python
_perlayer_window_grids(layer_segments, min_feature_frac, halfwidth=1)
    -> [(widths_i, eps_row_i), ...]
```

All five sites now call it. `PMMStack` gained `window_halfwidth=1` (validated: integer `>= 1`;
rejected with an explanation on the shared path, where there is no window), plumbed through the
conical entry point and carried by both internal `PMMStack` clones (`_resliced_clone`,
`_min_feature_clone`) so a consensus probe cannot silently drop it.

### 2.3 Byte-identity, per site [M, both builds]

Two independent instruments, both **tolerance-at-0.0**, never `array_equal`.

**(a) Shadow capture over all five dispatches.** 113 arrays -- `R`, `T`, `J`, absorption, internal
fields, the JAX forward *and its gradient* -- captured before the refactor and again after:

| site | dispatch | arrays | max abs diff |
|---|---|---|---|
| s1 | `_solve_vertical_perlayer` (ns 2/3/6/8 x deg 6/8/10 x theta 0/8, + lossy `retain_internal`) | 76 | **0** |
| s2 | `_solve_general_perlayer` (slant, OOP) | 8 | **0** |
| s3 | `_solve_vs_wavelength_perlayer` | 6 | **0** |
| s4 | `conical.py` (`phi` 0.3 / 1.2) | 18 | **0** |
| s5 | `_jax_stack.py` (forward, `jax.grad`, traced taper) | 5 | **0** |
| | **total** | **113** | **0** |

**(b) The grids themselves, against a verbatim re-implementation of the pre-M2 loop** -- 5 stacks
(`ns` 1/2/3/6/9 tapers + a 3-layer non-taper control) x 4 `min_feature` settings:

| build | max abs diff |
|---|---|
| Windows / py3.14 / OpenBLAS-Haswell | **0** |
| WSL / py3.12 / OpenBLAS-SkylakeX | **0** |

Instrument (b) ships as `test_n4_helper_is_byte_identical_to_the_five_verbatim_copies`, carrying
the pre-M2 loop verbatim as its own fail-before.

### 2.4 The continuity identity -- new, and free [M, both builds]

At `window_halfwidth >= nlay - 1` every window is the whole stack, so the per-layer path is the
*same discretisation* as the shared union grid and the mortar is bypassed at every interface.
Measured, `ns` in {2, 3, 4} x degree {6, 8}:

| build | `max\|dJ\|` | `max\|dR\|` |
|---|---|---|
| Windows | **0** (all 6 cells) | **0** |
| WSL | **0** (all 6 cells) | **0** |

This is worth more than a refactor pin: it turns `window_halfwidth` from a free parameter into a
knob whose *far end is a known reference*, and it is a structural equivalence between the two grid
paths that the library did not previously assert. Pinned as
`test_window_halfwidth_covering_the_stack_reproduces_shared_bit_exact`.

---

## 3. N-6 -- the `min_feature` contract

### 3.1 The claim, and why it is wrong [A]

Three places said the same thing (all `file:line` in S3.1 are **at `d30f1ca`**, before M2's
edits moved them):

* `CHANGELOG.md`, v5.32.0 PMM block -- "inert by construction (there is no global union to snap)"
* `AUDIT_PMM_PER_LAYER_GRIDS_IMPL_2026_07_28.md` S2 -- same words
* `_core.py:3521` and `stack.py:1630` docstrings -- "`min_feature` never enters" / "inert by
  construction"

**The window IS a union** -- of `2*halfwidth + 1` layers -- and `stack.py:1655` passes
`self.min_feature / self.period` straight into `_pmm_union_grid`, whose snap branch fires at
`min_feature > 1e-9` fractional (`_core.py:3460`). What the window removes is cross-**stack**
accumulation. It does **not** remove the ADJACENT-slice collision, because adjacent slices are
exactly what a window contains -- and on a taper the adjacent-slice collision is the tight one.

### 3.2 Dormant at the default -- the true half [M, both builds]

Library default `min_feature = period * 1e-5` = 7 pm on a 700 nm pitch. Measured merged-pair count
and maximum wall displacement over every window:

| `ns` | min window separation | default | 0.5 nm | 1.5 nm | 3.0 nm |
|---|---|---|---|---|---|
| 2 | 0.4127 nm | **0 / 0.0000 nm** | 4 / 0.2065 | 4 / 0.2065 | 4 / 0.2065 |
| 3 | 1.3915 nm | **0 / 0.0000** | 0 / 0.0000 | 8 / 0.6958 | 8 / 1.4560 |
| 6 | 1.8042 nm | **0 / 0.0000** | 0 / 0.0000 | 8 / 0.6958 | 40 / 1.3510 |
| 8 | 1.3532 nm | **0 / 0.0000** | 0 / 0.0000 | 32 / 0.6769 | 56 / 1.0150 |
| 12 | 0.9021 nm | **0 / 0.0000** | 0 / 0.0000 | 88 / 0.6769 | 88 / 0.6769 |

Identical on both builds. So "dormant at the library default" is **true and measured**, at every
`ns`, on both grid paths -- and it is exactly what made the false half invisible. What makes it
dormant is not the branch test but the *scale mismatch*: real collisions are ~1e-3 fractional,
~100x the default.

### 3.3 Active above it, and NOT less exposed than the shared path [M]

At the shared path's own recommended `min_feature = 1.5 nm`, on the same device:

| path | merged pairs | max wall displacement |
|---|---|---|
| per-layer windows | 4 - 88 | **0.696 nm** |
| shared union | 2 - 38 | 0.714 nm |

The per-layer *pair count* is larger only because a pair is re-merged once per window it appears
in; the **displacement** -- what actually perturbs the physics -- is the same class. The parent
audit measured a 0.75 nm wall move to change that device's ER by ~16%.

### 3.4 The bound, and what it does NOT bound [M]

* **Bounded**: displacement `<= min_feature / 2` per snapped pair, by construction (merge only
  when closer than `min_feature`, then move to the midpoint). Verified over
  `min_feature` in {0.5, 1.5, 3.0, 6.0} nm x `ns` in {2, 3, 6, 8, 12}; every reported displacement
  is at or under the bar. Pinned.
* **It bounds interior CROSS-LAYER PAIR separations, not the minimum cell width.** Two exclusions
  are deliberate and must be documented as part of the contract: the period boundary is never
  dropped (`interior` test, `_core.py:3467`), and a close pair owned by a **single** layer is that
  layer's own thin feature and is never thinned (`not (out_o[-1] & ow)`). Measured: at
  `min_feature = 6 nm` the 5.00 nm single-layer coat **survives on every window**, so the grid
  legitimately retains a cell far below `min_feature`. Pinned both ways.

### 3.5 The decisive measurement: `min_feature` is still the accuracy lever [M, both builds]

`ns = 2`, where the per-layer and shared grids are **identical** (a 2-layer window is the full
union) -- so this is a statement about the discretisation, not about the mortar. Order-0 `R`,
incident pol 0, degree ladder:

| `min_feature` | d6 | d8 | d10 | d12 | d14 | d16 | spread | vs RCWA | `max\|R+T-1\|` |
|---|---|---|---|---|---|---|---|---|---|
| **default** | 0.111000 | 0.110723 | 0.110643 | **0.061668** | **0.623403** | **0.623395** | **205.4%** | 466% | 4.5e-07 |
| 0.5 nm | 0.110880 | 0.110607 | 0.110528 | 0.110499 | 0.110486 | 0.110479 | **0.36%** | 0.35% | 2.9e-10 |
| 1.5 nm | *(identical to 0.5 nm)* | | | | | | 0.36% | 0.35% | 2.9e-10 |
| 3.0 nm | *(identical to 0.5 nm)* | | | | | | 0.36% | 0.35% | 2.9e-10 |

RCWA reference at 141 orders: **0.1100920** (still rising in orders: 0.107499 / 0.109488 /
0.109783 / 0.110092 at 21 / 41 / 81 / 141, so the true value is a little above 0.1101 and the PMM
0.11048 agrees to ~0.35%, i.e. within RCWA's own residual truncation).

**Contract (i) of the plan's N-6 wins on the measurement**: the snap is kept LIVE. Passing `None`
on the per-layer path -- contract (ii) -- would leave it *permanently unable* to remove the
collision that breaks it.

### 3.6 The rule, and its 12/12 prediction [M, both builds -- tables identical]

This is the part that makes the contract usable rather than folkloric. Inside a `+/-1` window the
only cross-layer walls are the two adjacent slices', so for a staircased taper with conformal coat
`c` and per-slice offset `off = (thickness/ns) tan(sidewall)` the window's cross-layer separations
are **exactly** `{off, |c - off|}` [A]. The requirement is therefore

> **`min_feature > min(off, |c - off|)`**

Tested against the degree ladder (stationary = spread `< 2%` over degree 6..16), per-layer path:

| `ns` | threshold | 0.5 nm | 1.5 nm | 3.0 nm |
|---|---|---|---|---|
| 2 | 0.4127 nm | pred STATIONARY / meas 0.36% **OK** | pred STAT / 0.36% **OK** | pred STAT / 0.36% **OK** |
| 3 | 1.3915 nm | pred COLLAPSE / meas 283.8% **OK** | pred STAT / 0.41% **OK** | pred STAT / 0.41% **OK** |
| 6 | 1.8042 nm | pred COLLAPSE / meas 196.9% **OK** | pred **COLLAPSE** / 196.9% **OK** | pred STAT / 0.60% **OK** |
| 8 | 1.3532 nm | pred COLLAPSE / meas 151.8% **OK** | pred STAT / 0.58% **OK** | pred STAT / 0.59% **OK** |

**12 of 12, on both builds** (the two `crossbuild.py` runs print identical tables). The rule is a DERIVATION confirmed by measurement, not a fit -- and it is scoped to the PER-LAYER window: S3.7 shows it does not carry to the shared union grid, where the cascading snap can create the collision it removed.  Note `ns = 6` at 1.5 nm: the rule *correctly predicts a collapse* at the value the
shared path recommends -- the recommendation is device-specific, not universal, and this is the
arithmetic that says so.

Deviation from RCWA at the stationary cells: **0.04% - 0.60%**, against RCWA's own residual
truncation of the same order.

### 3.7 The shared path is worse, not better, here [M]

The same matrix on the shared union grid:

| `ns` | default | 0.5 nm | 1.5 nm | 3.0 nm |
|---|---|---|---|---|
| 2 | 205.4% | 0.36% | 0.36% | 0.36% |
| 3 | 50.5% | 50.5% | 0.37% | **44.5%** |
| 6 | 338.5% | 53.8% | **48.7%** | 0.63% |
| 8 | 309.1% | 53.3% | **46.7%** | 0.62% |

Two things follow. First, **the per-layer path is the better-behaved of the two** on this device:
it is stationary wherever the rule says it should be, and the shared path is not. Second, the
shared grid at `ns = 3, min_feature = 3.0 nm` collapses **although the rule's threshold is
cleared** -- because the shared union contains *all* slices' walls and the snap is a **cascading
pairwise midpoint merge**, so merging can move a wall into a new near-collision. That is
**T3-7's exact argument** (deterministic lattice quantisation instead of a cascade), now with a
measured instance, and it is handed to M4.

---

## 4. T3-1 -- window width as a measured knob

`window_halfwidth` ships **opt-in, default 1, default NOT flipped.**

### 4.1 A confound found and removed -- read this before the numbers [M]

The first measurement took `halfwidth` 1 vs 2 on the coated device at `min_feature = 3.0 nm` and
read `max|dJ|` = **1.169e-02** (`ns=6`), **7.78e-03** (`ns=8`), **4.28e-03** (`ns=12`). Those
numbers are **not a window measurement** and are not reported as the envelope.

A `+/-2` window holds **five** layers, so it presents `_pmm_union_grid` with a *different pair set*
and snaps different walls: the two arms differ in geometry as well as in window. The tell is in
the data -- `max|dJ|` read **1.169e-02 at both degree 6 and degree 8**, to four figures. A
discretisation residual decays with degree; a geometry difference does not.

The clean experiment runs on tapers whose collisions are **already above the default
`min_feature`**, so the snap is provably inert (asserted: 0 merged pairs at every halfwidth) and
the window is the only variable:

* **device U** -- the taper with **no** conformal coat; separations are `{off}`, min 3.61 nm at `ns=3`.
* **device W** -- the same taper with a **25 nm** coat; separations `{off, |25 - off|}`.

### 4.2 Accuracy [M, both builds]

`max|dJ|` and `max|dR|` of `halfwidth=2` vs `1`, over the cells where the underlying solve is
healthy. **The excluded cells, named:** device U `ns=6` deg 10, `ns=8` deg 10, `ns=12` deg 8 and
10; device W `ns=6` deg 8 and (at `halfwidth=3`) deg 10, `ns=12` deg 6, 8 and 10 -- every one an
S5 collapse in one arm or the other, identified by the degree ladder of the arm itself, not by
the size of the delta.

| degree | worst `max\|dJ\|` | worst `max\|dR\|` |
|---|---|---|
| 6 | **2.667e-04** | 1.831e-04 |
| 8 | **8.867e-05** | 6.097e-05 |
| 10 | **6.868e-06** | 4.346e-06 |

Per-device ladders (device U `ns=3` / device W `ns=3` / device W `ns=8`), `max|dJ|`:

```
U ns=3   1.298e-04   3.123e-05   6.382e-06      (deg 6, 8, 10)
W ns=3   3.163e-05   7.652e-06   1.575e-06
W ns=8   4.405e-05   1.675e-05   6.868e-06
```

**Windows and WSL agree to the last printed figure on every healthy cell** (the only differences
are ULP-level, e.g. 1.682e-05 vs 1.683e-05).

**The envelope, stated:** the `+/-1` window's residual against `+/-2` is `<= 3e-4` at degree 6 and
**decays spectrally** -- a factor **2.4 - 4.9 per two-degree step** on all three ladders, exactly
like a discretisation residual and unlike a geometry error (the confounded measurement of S4.1 sat
at 1.169e-02 for degree 6 *and* 8, i.e. did not decay at all). Its size is the same class as the mortar's own non-conforming remainder,
which the shipped suite already pins at 1.10e-4 (degree 6) to 1.17e-6 (degree 10). So the honest
statement is: **`+/-1` is not exact; it is converged to the accuracy the mortar itself spends, and
widening the window buys back nothing the mortar has not already lost.**

`|R+T-1|` at `halfwidth=1` on these cells: 2.3e-08 to 4.7e-06, and `halfwidth=2` does not degrade
it.

### 4.3 Cost [M, both builds, interleaved A/B ratios -- see the load caveat in S1]

| `ns` | deg | time ratio Win | time ratio WSL | `tracemalloc` peak ratio (both) |
|---|---|---|---|---|
| 3 | 6 | 1.23 | 1.22 | **1.54** |
| 3 | 8 | 1.15 | 1.37 | **1.54** |
| 6 | 6 | 1.97 | 2.12 | **2.07** |
| 6 | 8 | 2.16 | 2.70 | **2.07** |
| 8 | 6 | 1.58 | 1.85 | **2.15** |
| 8 | 8 | 1.88 | 2.21 | **2.16** |
| 12 | 6 | 2.20 | -- | **2.25** |
| 12 | 8 | 2.48 | -- | **2.25** |

The `tracemalloc` peak ratios are **identical on both builds** to three figures, as they should be
(allocation is deterministic); wall-clock ratios differ with machine load, which is why only
ratios are quoted. Cost: **1.2 - 2.7x time, 1.5 - 2.3x peak memory**, rising with `ns` as the
window becomes a larger fraction of the stack.

### 4.4 Verdict

`halfwidth=2` buys a `<= 3e-4` change that is already inside the mortar's residual, for up to
2.7x time and 2.2x memory. **It does not earn a default flip**, per the plan's rule 1 (more
resolution ships as a measured opt-in). It ships as a **diagnostic** -- *does my answer depend on
the window?* -- and because of the S2.4 continuity identity, pushing it to `nlay-1` returns the
shared-grid answer *exactly*, so the knob spans a measured interval between two known references
rather than wandering. **No pinned number moved, so no era-pin was required.**

---

## 5. NEW FINDING -- a silent-wrong regime on both grid paths (pre-M2)

This was found while measuring T3-1 and is the most important thing in this document. It is **not**
caused by M2 (byte-identity is 0.0 at every site) and it is **not** a per-layer defect (it is
identical on the shared path, and at `ns = 2` the two paths are bit-identical).

### 5.1 The observable [M, both builds]

On the audit-class taper at the library default, the degree ladder does not converge -- it *jumps*:

```
ns=2  0.111000  0.110723  0.110643  0.061668  0.623403  0.623395   (deg 6..16)
ns=3  0.111482  0.111188  0.576251  0.111074  0.063382  0.111054
ns=6  0.112087  0.111756  0.568367  0.111628  0.111614  0.661991
ns=8  0.112240  0.111892  0.111792  0.663494  0.680282  0.566907
```

RCWA (141 orders, analytic walls) says the answer is ~0.1101-0.1114. The collapsed cells are
therefore **up to 466% wrong**, and they land on apparently arbitrary `(ns, degree, halfwidth)`
cells -- the same "arbitrary cell" signature the parent audit reported at S4.5.

### 5.2 Conservation is blind to it [M]

`|R+T-1|` through the collapse: **1.7e-8 to 4.5e-7**. `_warn_stack_energy` never fires. The reason
is structural and worth recording: at 700 nm pitch, 1310 nm, `n_sup = 1.5`, `n_sup P / lambda =
0.80 < 1`, so **only order 0 propagates** and `R + T = 1` is nearly a restatement of the cascade's
unitarity, not an independent check. Measured directly: `sum(R) == R[order 0]` to all printed
digits in every row. The cascade returns a **unitary but wrong** S-matrix -- the split between `R`
and `T` is wrong while the total is right.

This is exactly the plan's standing rule 5 ("`R+T+A` can be tautological") biting on a real case,
and it is why the lossy twin's `sum(A)` vs `1 - R - T` is reported alongside everywhere here.

### 5.3 Mechanism -- measured, not inferred [M]

Discriminators, all on `ns = 2`:

| variant | minimum cell | degree ladder 6..16 |
|---|---|---|
| 5.00 nm coat (the device) | **0.4127 nm** | 0.111000 0.110723 0.110643 **0.061668 0.623403 0.623395** |
| 25.0 nm coat | 19.6 nm | 0.084788 0.084668 0.084631 0.084616 0.084609 0.084604 |
| no coat (taper only) | 3.4 nm | 0.117680 0.117084 0.116857 0.116751 0.116694 0.116661 |
| untapered, coated | 5.00 nm | 0.094328 0.093881 0.093745 0.093693 0.093670 0.093658 |

**The thin cell is the trigger** -- not the taper, and not the coat: widening the coat to 25 nm or
removing it entirely makes the *same taper* converge cleanly to degree 16.

**The onset degree is not predicted, and that is stated as a limit.** [M] Across the two
snap-inert devices of S4.1 the collapse appears wherever the minimum cross-layer cell is
~2 nm or below on this 700 nm pitch (~3e-3 fractional) -- device U at `ns` 6/8/12 (cells 1.80 /
1.35 / 0.90 nm) and device W at `ns` 6/12 -- but *which* degree it first bites at varies by
device (device W at `ns=8`, cell 1.35 nm, is clean through degree 10 where device U at the same
cell size is not). So the actionable statement is the S3.6 rule -- **remove the thin cell** --
not a (cell width, degree) safe region, which the evidence does not support.

Modal census on the layer-0 window grid (the flux-based forward/backward selector,
`_sem_modes_tensor` + `_mass_flux_cut`):

| `min_feature` | deg 6 | 8 | 10 | 12 | 14 | 16 |
|---|---|---|---|---|---|---|
| default -- modes classified propagating | 4 | 4 | 4 | **6** | **5** | **5** |
| default -- modes within a decade of the cut | 1 | 2 | 4 | **6** | 4 | 6 |
| default -- `max\|q\|` | 8.0e3 | 1.3e4 | 2.0e4 | 2.8e4 | 3.7e4 | 4.8e4 |
| 0.5 nm -- propagating | 4 | 4 | 4 | 4 | 4 | 4 |
| 0.5 nm -- within a decade of the cut | **0** | **0** | **0** | **0** | **0** | **0** |
| 0.5 nm -- `max\|q\|` | 1.1e3 | 1.9e3 | 2.8e3 | 4.0e3 | 5.4e3 | 7.0e3 |

The 0.4127 nm cell injects modes whose z-Poynting flux sits **at** the propagating/evanescent
classification threshold. The count of modes classified propagating then **moves with degree**
instead of staying fixed, the forward set is mis-assembled, and the cascade is unitary but wrong.
With the collision snapped away the census is 4 at every degree and **zero** modes near the cut.

It is **not** an assembly-conditioning problem: `cond(S0)` on the same grids reads
1.38e3 / 1.76e3 / 2.15e3 / 2.54e3 / 2.93e3 / 3.33e3 at degree 6..16 -- smooth, small, and
`_ill_scaled` is `False` throughout. The mass matrix is healthy; the *spectrum* is not. This is
therefore **a different class from M1's**: M1 hardened solves that could *draw* an arbitrary answer
from a null space; this is a **deterministic mis-selection** that both builds reproduce.

**Mostly deterministic, but not entirely** [M]. Every collapse cell in the S3.6 rule matrix, the
S3.5 ladder and the S6 table is **bit-reproducible across the two builds** -- so this is a method
defect, not a BLAS lottery, and cross-build agreement would *not* have caught it (the same lesson
M1 recorded for its 21.te row). One cell does diverge: device W, `ns=12`, degree 8, `halfwidth=2`
reads `max|dJ|` = 1.941e-05 on Windows and 5.735e-01 on WSL. That is the expected signature of a
classification sitting exactly on the threshold, and it is consistent with the mechanism rather
than an alternative to it.

### 5.4 What M2 does about it, and what it does not

**Does:**
1. **Removes the cause of the exposure.** The library told users this path was immune to
   `min_feature`. It is not, and S3.6 gives the arithmetic for choosing it. All three in-code
   claims in `lumenairy/elements/pmm/**` are corrected with the measurement inline.
2. **Pins it, in both directions.** `test_min_feature_is_the_accuracy_lever_on_the_per_layer_path_too`
   asserts the collapse at the default (the fail-before) *and* stationarity at 0.5 nm, and it says
   in its own message that if the collapse ever stops reproducing, the test must be **re-pinned
   against the fix, not relaxed**.
3. **Reports `|R+T-1|`'s blindness** so no later mission scores this device on closure alone.

**Does not:** ship a detector. That is **T3-4**, which the plan defers as "blocked on an
unresolved mechanism" and explicitly excludes from 5.33.0. Shipping a threshold here would be the
R-1b failure mode (a false pathology claim is worse than silence): the near-cut mode count is 1 at
degree 6 where the answer is *right*, so the instrument exists but its bar does not.

### 5.5 T3-4 is now unblocked -- hand-off

The plan's T3-4 verdict ("DEFER -- correctly blocked. Two hypotheses measured and refuted") should
be **revisited**, because a third mechanism is now confirmed with an independent oracle:

* **Story:** a cross-layer cell at the ~1-2 nm scale on a 700 nm pitch makes the modal
  propagating/evanescent classification degree-dependent; the far field is unitary and wrong.
  Established with an independent oracle (S5.1-S5.3), on both grid paths, on both builds.
* **Instrument, and it is free:** the count of modes with `|flux|` within a decade of
  `_mass_flux_cut`'s threshold, and the *stability of the propagating-mode count across degree*.
  Both are already computed inside `_sem_modes_tensor`; neither costs an extra factorisation.
* **What is still missing, and it is why no detector ships here:** the bar. On the one grid where
  the census was run (`ns = 2`, S5.3) the near-cut count reads 1, 2, 4 at degree 6, 8, 10 -- all
  giving the RIGHT answer -- and 6 at degree 12, which does not. **The healthy and broken
  populations overlap at 4**, so a threshold on the count alone would refuse a correct solve. That
  is precisely the `rcond`-vs-residual trap M1 documented for `_guarded_lstsq`, and it needs its
  own labelled validation set -- 1 AC, not 0.
* **The lead worth following, because it needs no fitted constant:** the *propagating-mode count*
  itself was invariant at 4 across degree 6..16 with the collision snapped away, and moved
  (4, 4, 4, 6, 5, 5) with it present. A two-degree consensus probe on that count would be a
  detector with no bar. **Measured at `ns = 2` only** -- generalising it across `ns` and across
  devices is the missing work, and no claim is made here that it holds beyond that grid.

This also strengthens **T3-5** (M4's taper-aware `min_feature`): S3.6 gives the exact expression
to derive the default from, `min(off, |c - off|)` with `off = (thickness/n_slices) tan(sidewall)`,
and `self._taper_recipes` already carries `thickness`, `n_slices` and the sidewall for
builder-made stacks.

---

## 6. T3-2 -- staircase stationarity, ns 8..12

**The envelope is conditional, and the condition is the point.** A converged discretisation stops
moving -- but `n_slice` cannot be called stationary while `degree` is not, and S3.6 shows `degree`
is not stationary below the `min_feature` threshold. So T3-2 is stated as a *joint* statement.

Order-0 `R`, incident pol 0, per-layer, `halfwidth=1`, **identical on both builds**:

| `min_feature` | deg | `ns=8` | `ns=10` | `ns=12` | spread |
|---|---|---|---|---|---|
| 1.5 nm | 6 | 0.111827 | 0.112040 | 0.112179 | **0.314%** |
| 1.5 nm | 8 | 0.111400 | 0.111614 | 0.111758 | **0.321%** |
| 3.0 nm | 6 | 0.111840 | 0.112046 | 0.112179 | **0.303%** |
| 3.0 nm | 8 | 0.111416 | 0.111625 | 0.111758 | **0.307%** |

RCWA on the same three staircases (141 orders, analytic walls): **0.111236 / 0.111307 / 0.111351**
-- spread 0.10%.

Three things are true here and all three matter:

1. **The envelope.** `|R(ns) - R(ns')| / R <= 0.33%` over `ns` in {8, 10, 12}, at degree 6 and 8,
   at both `min_feature` values above the threshold, on both builds. **Pinned** at a 2% bar with
   headroom (`test_staircase_is_stationary_in_n_slice_at_ns_8_to_12`).
2. **It is a residual, not noise.** PMM's `ns` trend is *monotone increasing* and so is RCWA's, in
   the same direction and of the same order (0.31% vs 0.10% over the same span) -- consistent with
   the documented `O(1/n_slices^2)` staircase truncation, not with a conditioning wobble. The
   remaining PMM-vs-RCWA offset is 0.15%, against RCWA's own residual order-truncation of the
   same size (it moves 0.1075 -> 0.1101 over 21 -> 141 orders).
3. **The precondition is load-bearing.** At the library default the *same* cells read
   0.112240 / 0.111892 / 0.111792 / **0.663494 / 0.680282 / 0.566907** across degree 6..16 at
   `ns = 8`. An `ns`-stationarity claim taken at the default would have been a claim about three
   numbers that happen to be in the pre-collapse part of a divergent ladder.

The degree half of the claim, measured at **all three** `ns` (not extrapolated from `ns = 8`) --
full ladders, order-0 `R`, per-layer, `halfwidth=1`. **All 36 values below are identical on
Windows and WSL to every printed digit.**

| `min_feature` | `ns` | d6 | d8 | d10 | d12 | d14 | d16 | spread |
|---|---|---|---|---|---|---|---|---|
| 1.5 nm | 8 | 0.111827 | 0.111400 | 0.111266 | 0.111214 | 0.111190 | 0.111178 | **0.583%** |
| 1.5 nm | 10 | 0.112040 | 0.111614 | 0.111481 | 0.111429 | 0.111405 | 0.111393 | **0.580%** |
| 1.5 nm | 12 | 0.112179 | 0.111758 | 0.111624 | 0.111573 | 0.111549 | 0.111536 | **0.576%** |
| 3.0 nm | 8 | 0.111840 | 0.111416 | 0.111281 | 0.111227 | 0.111202 | 0.111187 | 0.586% |
| 3.0 nm | 10 | 0.112046 | 0.111625 | 0.111491 | 0.111439 | 0.111414 | 0.111401 | 0.578% |
| 3.0 nm | 12 | 0.112179 | 0.111758 | 0.111624 | 0.111573 | 0.111549 | 0.111536 | 0.576% |

Every ladder is **monotone decreasing and flattening** -- the signature of a converging
discretisation, and the whole 0.58% is spent between degree 6 and 8 (degree 10 -> 16 moves
`< 0.08%`). Both `min_feature` values agree to 5 digits by degree 12, which is the independent
statement that the answer no longer depends on the snap once the snap has done its job.

**Stated envelope:** *on a staircased taper whose `min_feature` clears
`min(off, |c - off|)`, the per-layer answer is stationary in `n_slice` to `0.33%` and in `degree`
to `0.6%` over 8 <= ns <= 12 and 6 <= degree <= 16, on both builds.* Below that `min_feature` no
stationarity claim is available at any `ns`, and the plan's instruction ("if no setting is
stationary the correct outcome is a quoted band, not a chosen setting") is honoured by quoting the
precondition rather than a band.

---

## 7. Acceptance gates

| axis | gate (plan S4, M2) | result |
|---|---|---|
| accuracy | N-4 byte-identical (tolerance-at-0.0) across the per-layer suite + audit-device gates, both builds | **PASS.** 113 arrays across all 5 sites: **0**. Grids over 5 stacks x 4 `min_feature`: **0**, both builds. |
| accuracy | `halfwidth=2` reduces `\|R+T-1\|` at deg 8 by a measured factor, reported with its cost; no default flip unless it beats its cost | **PASS (measured, not flipped).** It does **not** reduce the residual -- it moves the answer `<= 3e-4` (deg 6), decaying spectrally, at 1.2-2.7x time / 1.5-2.3x memory. Default unchanged. S4. |
| conservation | every sweep cell reports `\|R+T-1\|` **and** `sum(A)` vs `1-R-T` alongside; an ER-only table is rejected | **PASS**, and strengthened: `\|R+T-1\|` is shown to be **near-tautological on this device** (S5.2) and the lossy-twin `sum(A)` closure is reported with it (S7.1). No ER is quoted anywhere. |
| both-builds | the chosen stationary setting is stationary on **both**; else quoted as a band | **PASS.** Windows and WSL agree to every printed digit on the S3.6 rule matrix, the S6 stationarity table and the S3.2 census; healthy T3-1 cells agree to ULP. |
| null control | a conforming stack (mortar bypassed) is bit-identical under every window and `min_feature` change; a vertical untapered stack unaffected | **PASS**, both, tolerance-at-0.0, 5 knob settings each (`test_conforming_and_untapered_stacks_are_immune_to_both_knobs`). |
| era-pin | where T3-1/T3-2 legitimately move pinned numbers, era-pin verbatim with a live sibling | **N/A -- no pinned number moved.** The default path is bit-identical, so no era-pin exists to write. |
| oracle | adjudicated against the shared grid at a validated `min_feature`, and against the manufactured-solution mortar test | **PARTIAL.** Adjudicated against **RCWA** (independent method, analytic walls) and against the shared grid -- including the new *exact* `halfwidth >= nlay-1` identity, which is stronger than agreement. The **manufactured-solution mortar test was not built** (see S7.2). |
| speed/memory | cost of `halfwidth=2` measured, not estimated: wall time and `tracemalloc` peak per solve, both builds | **PASS.** S4.3, interleaved A/B, both builds; peak ratios identical across builds. |
| no regression | default-path perf/memory unchanged | **PASS by construction and by measurement**: the default path is byte-identical, and the helper adds one function call per solve (`nlay` iterations of the identical loop). |

### 7.1 Conservation, on the non-tautological instrument [M, both builds]

Lossy twin (Cu-like core, `n = 0.30 + 8.0i`) on device U, `sum(A)` from `layer_absorption()`
against `1 - R - T` from the far field -- the check that is **not** tautological, since the
absorption is integrated per layer and the budget comes from the Rayleigh projection:

| `ns` | deg | hw | `sum(A)` | `1 - R - T` | gap |
|---|---|---|---|---|---|
| 3 | 6 | 1 | 0.0474687 | 0.0474590 | 9.74e-06 |
| 3 | 8 | 1 | 0.0469319 | 0.0469299 | 1.97e-06 |
| 3 | 10 | 1 | 0.0465980 | 0.0465973 | **7.08e-07** |
| 8 | 6 | 1 | 0.0471230 | 0.0471118 | 1.12e-05 |
| 8 | 8 | 1 | 0.0467195 | 0.0467180 | 1.50e-06 |
| 8 | 10 | 1 | 0.0464184 | 0.0464181 | **2.63e-07** |

**Identical on both builds to every printed digit.** The gap decays ~4x per degree step -- it is
the mortar's own non-conforming remainder, on an independent instrument, and it agrees in size
with the `|R+T-1|` and window-residual ladders. `halfwidth=2` reduces it (1.14e-11 at `ns=3`),
which is the one place widening the window measurably helps -- and it is a residual that was
already at 1e-6.

**But it is blind in the same way** [M]: at `ns=8, degree 10, halfwidth 2` -- an S5 collapse cell
-- `sum(A)` and `1 - R - T` both read **0.0695823**, closing to **4.89e-10**, against the healthy
0.0464. Both sides of the identity are computed from the same wrong S-matrix. **No conservation
identity available in this library detects the S5 regime**, and that should be recorded plainly
rather than discovered again.

### 7.2 Suites and lint

| suite | Windows / py3.14 / OpenBLAS-Haswell | WSL / py3.12 / OpenBLAS-SkylakeX |
|---|---|---|
| `test_pmm_m2_window_contract.py` (**new**, 14) + `test_pmm_per_layer_grids.py` (11) + `test_m1_conditioning_guard.py` (27) | **52 passed** (153 s) | **52 passed** (263 s) |
| PMM collateral: `test_v5_11_0_pmm_stack`, `_v5_20_0_pmm_stack_conical`, `_v5_13_0_pmm_tapered`, `_v5_21_pmm_threaded_sweep`, `_v5_14_3_pmm_internal_field`, `_niche_audit_w9_pmm_taper`, `_v5_20_7_pmm_geo_eig_cache`, `_v5_12_0_pmm_autodiff` | **186 passed** (947 s) | -- |
| `tests/unit/ -k "pmm or rcwa" -n 6` (the full blast radius) | **1555 passed, 2 skipped, 1 failed** (41 min) -- the one failure is **not M2's**, see below | -- |
| `-k "congruence or carrier_worker or ram_budget"` (the S7.4 lint fix's blast radius) | **42 passed** | -- |
| `ruff check --no-cache lumenairy/ tests/` (via WSL, the CI proxy) | -- | **All checks passed** |

Identical test counts on both builds, `-p no:randomly` so the two runs are the same order.

**[M] The one failure, adjudicated.**
`test_v5_20_8_rcwa_threaded_sweep.py::test_threaded_sweep_applies_exactly_one_blas_cap` failed
under `-n 6`. It is **M4's in-flight test** for the RCWA BLAS-cap scoping they are landing
concurrently (that file is one of M4's modified files), it contains **zero** references to PMM,
and M2 changed nothing outside `lumenairy/elements/pmm/**` plus its own new test file. Run down:

| invocation | result |
|---|---|
| the test alone | **passed** (43 s) |
| its whole file, `-n 2` | **4 passed** |
| its file + `test_niche_audit_m4_m5_m6_rcwa.py`, `-n 4` | **32 passed, 1 skipped** |
| inside the full 1555-test `-n 6` selection | **failed** |

So it is a **cross-file parallel-isolation flake**, not a deterministic break: the test replaces
the module global `rcwa._core._get_blas_controller` with a counting stand-in, and under xdist the
outcome depends on which other test shares its worker. That is the repo's known flag-leak class
(the v5.32.1 "flag-leak guard at MODULE scope" and "order-independence guard" work). **Handed to
M4** -- it is their file, actively being edited, and touching it from here would collide.

**[M] One stderr observation, NOT M2's and NOT a failure.** The WSL run prints
`** On entry to DLASCL parameter number 4 had an illegal value` twice. Isolated by file: it comes
from **`test_m1_conditioning_guard.py`** (2 occurrences), not from
`test_pmm_m2_window_contract.py` (0) and not from `test_pmm_per_layer_grids.py` (0); it does not
appear on the Windows build. That file's purpose is to *construct* rank-deficient and degenerate
solves, so LAPACK complaining on the OpenBLAS-SkylakeX kernel is consistent with the tests doing
their job -- but it is a raw LAPACK stderr write that no test asserts on, and M1 or a later
mission may want it captured rather than leaked to the console. Recorded here because it was
observed here; it is outside `lumenairy/elements/pmm/**` and was not touched by M2.

### 7.3 One pre-existing lint failure found and fixed (outside `pmm/**`)

`ruff` reported `I001` (un-sorted import block) at `lumenairy/propagators/carrier.py:6855`. Run
down: the file is **byte-identical to HEAD** (`git diff HEAD --quiet` succeeds), so the defect
**pre-dates this branch's M1 commit** and was masked by `ruff`'s result cache until an unrelated
`mtime` touch invalidated it -- which is why an earlier `ruff check` in this same session printed
"All checks passed". It would have failed the repo's standing "ruff before every push" gate for
whoever pushed next. Fixed (swap two lines inside one `try:` block, exactly what `--fix`
produces); `--no-cache` now passes over `lumenairy/` and `tests/`, and the 42 tests selected by
`-k "congruence or carrier_worker or ram_budget"` pass. Recorded because it is outside M2's
declared ownership.

### 7.4 What M2 did NOT do, stated plainly

* **The manufactured-solution mortar test** (project a known smooth analytic field from grid A to
  grid B, score the L2 projection error against its closed form) named in the plan's T3-1 oracle
  list **was not built.** The window question turned out to be decided by the confound in S4.1 and
  by the exact `halfwidth >= nlay-1` identity, which is a stronger statement about the same
  machinery; but the mortar-alone oracle remains genuinely missing and is worth 0.5 AC to M3,
  which is already inside `_interface_smatrix_mortar`.
* **No detector shipped** for the S5 silent-wrong regime -- that is T3-4, deferred by the plan.
  S5.5 is the hand-off.
* **The 1800-solve mesh sweep** in the plan's T3-2 estimate was not needed: the S3.6 rule collapses
  the `min_feature` axis from a scan to an arithmetic prediction (12/12), and the remaining grid ran
  locally in minutes.

---

## 8. Reconciliation with M5's P1 escalation [M]

M5 escalated the same defect from the other end (`PMM_M5_2D_FEASIBILITY_2026_08_04.md` S4), on
the **simplest** member of the class: one region, lossless dielectric, duty 0.5, 2-deg taper,
**no coat**, in-plane `theta = 8 deg`. Their 80-cell `(degree x n_slice x layer_grids)` table
scatters `-93%` to `+10x`, with 25 of 26 wrong cells at `|R+T-1| <= 1e-6`, deterministic on both
builds. **Independently reproduced here**, and it reconciles cleanly with S5.

### 8.1 It is the same defect, and S5's mechanism predicts M5's table [M]

M5's device has no coat, so the window's **only** cross-layer separation is the per-slice offset
itself, `off = (H/ns) tan(2 deg) = 10.82/ns` nm. S5's mechanism therefore predicts collapse
exactly where `off` falls into the thin-cell band -- and, reproduced here at the library default
(order-0 `R`, degree 8..16, `spread` over degree):

| `ns` | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|
| `off` (nm) | 3.608 | 2.706 | 1.804 | 1.353 | 0.902 | 0.677 |
| per-layer spread | **0.02%** | 28.8% | 124% | 177% | 248% | 432% |
| shared spread | **0.02%** | 213% | 159% | 188% | 300% | -- |

`ns = 3` (`off` = 3.61 nm) is the only clean rung on either path, and it is clean to **0.02%**.
This also explains M5's observation 4 ("per-layer delays but does not fix, shared is worse")
**quantitatively**: both paths have the same *minimum* cell `off`, but the shared union carries
`2(ns-1)` cells of that width against the `+/-1` window's 4, so the shared grid presents more
near-threshold modes and fails at a lower `ns`. No new mechanism is needed.

**This answers M5's open hypothesis** ("the mechanism may be broader than the union grid --
this device has no coat and therefore no coat/offset resonance at all"): **yes, and it is
broader in a specific way.** The trigger is a **thin cell**, whatever produces it. A coat/offset
collision is one source; a fine staircase's own per-slice offset is another. The union grid is
not the mechanism, it is one way of manufacturing the input.

### 8.2 The mitigation generalises -- 24/24, and it does NOT delete the taper [M]

The obvious worry, and the reason this needed measuring rather than asserting: on a single-region
taper the tight separation **is the taper**, so snapping it might cure the collapse by flattening
the geometry. Measured, per-layer, degree 8..16:

| `min_feature` | ns=3 | ns=4 | ns=6 | ns=8 | ns=12 | ns=16 |
|---|---|---|---|---|---|---|
| threshold `off` (nm) | 3.608 | 2.706 | 1.804 | 1.353 | 0.902 | 0.677 |
| default | 0.02% | 28.8% | 124% | 177% | 248% | 432% |
| **1.5 nm** | 0.02% | 28.8% | 124% | **0.04%** | **0.06%** | **0.05%** |
| **3.0 nm** | 0.02% | **0.04%** | **0.07%** | **0.07%** | **0.06%** | **0.05%** |

Every cell matches the rule `min_feature > off`: 1.5 nm cures `ns >= 8` and not `ns = 4, 6`;
3.0 nm cures `ns >= 4`. **12 more predictions, 12 correct** -- so **24/24** across the two
devices, one coated and one not.

**And the taper survives.** Holding `min_feature` fixed at 6.0 nm so every rung gets the same
treatment, the cured `n_slice` ladder is **monotone increasing** -- `ns` 3 / 6 / 8 / 12 ->
0.115438 / 0.116884 / 0.117253 / 0.117600, total spread 1.9% -- the same direction and ordering
as RCWA's own `ns` trend. A flattened taper would not track it. Pinned as
`test_threshold_rule_holds_on_a_SINGLE_REGION_uncoated_taper`, including the trap that caught a
first draft of it: comparing rungs at *different* `min_feature` mixes three geometry treatments
and is legitimately non-monotone.

### 8.3 One correction to M5's S4 [M]

M5 records a "**constant `+7.4e-04` offset from RCWA** -- a method-to-method systematic (PMM has
exact walls, RCWA has a Fourier floor; **which is right is not adjudicated here**)". It is
adjudicable, and PMM's stable cells are the accurate ones. RCWA on this geometry is simply
**under-converged at 21 orders** -- it is a high-contrast (`eps` 4 / 1) grating:

| `n_orders` | 21 | 41 | 81 | 141 | 201 |
|---|---|---|---|---|---|
| RCWA `R0`, `ns=3` | 0.013834 | 0.014215 | 0.014424 | 0.014516 | 0.014553 |
| RCWA `R0`, `ns=8` | 0.013863 | 0.014247 | 0.014459 | 0.014551 | 0.014588 |

marching monotonically toward PMM's stable **0.014644**, still 0.6% short at 201 orders and still
moving. So the arbiter confirms the *stable* PMM cells to sub-percent and the offset is RCWA's
truncation, not a PMM bias. **M5's arbiter role stands** -- for detecting the scatter it is
decisive at any order count -- but the residual offset should not be recorded as a PMM
systematic.

### 8.4 What this means for M2's own results

* **The T3-2 stationarity envelope (S6) does not sit on the defect.** It was measured at
  `min_feature` 1.5 and 3.0 nm -- both above the threshold at `ns` 8/10/12 (`off` = 1.353 /
  1.083 / 0.902 nm) -- and cross-checked three ways: against RCWA (0.15%), against a *second*
  `min_feature` (agreeing to 5 digits), and against its own degree ladder (0.58%). The escalation
  was warranted; the sweep as run is sound, and S6 already states the precondition as
  load-bearing.
* **The 1800-solve mesh sweep was never launched** (S7.4). The rule collapsed the `min_feature`
  axis from a scan to arithmetic.
* **What M2 does not claim:** that `min_feature` is a *fix*. It removes the input that triggers
  the defect, at a bounded, measured geometry cost. The defect itself -- a mode classification
  that is not stable in `degree` -- is untouched, and no default protects a user who does not
  know the rule. That is the T3-4 hand-off in S5.5, and on the evidence of two independent
  missions finding it within a day it should be **promoted out of "deferred"**.

---

## 9. Reproduction

Probe scripts (not shipped; `C:\tmp\m2_probe\`):

| script | purpose |
|---|---|
| `device.py` | the audit-class taper surrogate + its collision arithmetic |
| `shadow.py` | N-4 pre/post capture over all five sites (`before.npz` / `after_n4.npz`) |
| `s4_snap_census.py` | S3.2/S3.3 snap census, per-layer vs shared |
| `oracle_rcwa.py` | the RCWA oracle (analytic shape form factors) |
| `mechanism.py` | S5.3 sliver / taper discriminators + the modal census |
| `n6_contract.py` | the S3.5-S3.7 `min_feature` x degree x `ns` matrix, both grid paths |
| `crossbuild.py` | every numeric claim, one script, run on both builds |
| `t31_clean.py` | S4.1's confound-free T3-1 (devices U and W, snap asserted inert) |
| `cost.py` | T3-1 interleaved A/B time + `tracemalloc` peak, and the lossy `sum(A)` closure |
| `m5_reconcile.py` | S8 -- M5's device class, the rule's 12 extra predictions, the taper-survival check, RCWA order convergence |

### 9.1 Evidence-integrity note [M]

A concurrent **M4** agent modified `lumenairy/elements/rcwa/_core.py` and
`lumenairy/elements/rcwa/stack.py` (BLAS-thread-cap scoping) part-way through this mission --
i.e. *inside* the oracle this document leans on. The RCWA oracle was therefore **re-run against
the final tree** and reads identically to seven figures at every `ns`:

```
ns= 2 0.1100920   ns= 3 0.1105630   ns= 6 0.1111090
ns= 8 0.1112361   ns=10 0.1113069   ns=12 0.1113511
```

`lumenairy/elements/pmm/**` was touched by M2 alone (`git diff` hunks all account for edits made
here), so no cross-mission contamination reaches the byte-identity results -- and a contaminated
comparison could not have returned exactly 0 across 113 arrays in any case.

```bash
E="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
cd C:/tmp/m2_probe
env $E python -u crossbuild.py "Windows"
wsl -e bash -lc "cd /mnt/c/tmp/m2_probe && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    ~/lumvenv/bin/python -u crossbuild.py 'WSL'"

cd <repo>
env $E python -m pytest tests/unit/test_pmm_m2_window_contract.py \
    tests/unit/test_pmm_per_layer_grids.py tests/unit/test_m1_conditioning_guard.py -q
```

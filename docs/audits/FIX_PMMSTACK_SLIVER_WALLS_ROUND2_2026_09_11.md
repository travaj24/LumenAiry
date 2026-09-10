# FIX ROUND 2 -- the `PMMStack` SLIVER guard learns to ATTRIBUTE

**Date** 2026-09-11 · **Branch** `fix/pmmstack-sliver-guard-round2` (from
`wave2/pmm2d` @ `bb0527a`, i.e. round 1 + its verification + the deadband
follow-up + the mortar / non-uniform work) · **Worktree** `C:/tmp/lum_sliver2`
· **Scope** `lumenairy/elements/pmm/stack.py` only

**Subject** the two defects the verification
(`docs/audits/VERIFY_PMMSTACK_SLIVER_WALLS_2026_09_11.md`) raised against
`docs/audits/FIX_PMMSTACK_SLIVER_WALLS_2026_09_11.md` -- **V-1** (110 of 648
realistic staircases refused although correct) and **V-8** (8 of 660 wrong
solves returned unwarned) -- plus **V-5** (the guard is structurally silent on
every anisotropic stack) and **V-6** (the within-layer liner is silent at
widths where the same mechanism is already catastrophic).

**Reproducers** `validation/probe_pmmstack_sliver_round2/` (nine probes, a
README, and the JSON every table below is read from, on both builds).

**Binding** `docs/TESTING_STANDARDS.md`.

---

## S0. Summary

| | |
|---|---|
| **The defect** | Round 1 read super-unity on a passive stack as a theorem violation and, whenever a manufactured sliver was also present, ATTRIBUTED the violation to that sliver. Super-unity DETECTS but does not ATTRIBUTE: on a passive stack it is just as often ordinary under-convergence, and the wrong population reaches DOWN below the 1e-2 bar. Hence false positives at 17.0 % of one realistic parameter box AND false negatives at 8 in 660. |
| **What ships** | An **ARBITER**. When a manufactured sliver is present on a provably passive stack AND the solve reads super-unity above a TRIGGER one decade below the warning bar, the library re-solves ONCE on the grid the `min_feature` it is about to prescribe would produce. If the super-unity VANISHES **and** the answer MOVES far past the geometric perturbation that snap describes -> the sliver caused it, REFUSE, and the message says what was measured. If it SURVIVES -> the sliver did not, RETURN under the plain super-unity warning, which now says so and names `degree` / `n_slices`. If the one solve cannot be run (keyed / dispersive materials) -> round 1's decision, unchanged. |
| **False positives** | **110 / 648 -> 0 / 648** on the verification's own realistic staircase box, both builds. |
| **False negatives** | **8 / 660 -> 4 / 660** on the verification's own two grids; the four that remain are exactly the rows whose super-unity is below the trigger, and the trigger cannot go lower without sitting ON the correct population's own envelope (1.0979e-04 measured over 600 correct rows -- at 1e-4 the bar would carry 1.01x). |
| **Passivity** | `_stack_provably_passive` now accepts any tensor whose ANTI-HERMITIAN part is positive semi-definite -- the whole liquid-crystal / birefringent class, lossless or lossy. Measured: the O-11 sliver on a 45-degree in-plane director, on an out-of-plane director and on a gyrotropic layer is now REFUSED where round 1 read `R+T` = 2.18 and only warned; a NON-Hermitian payload still keeps the behaviour it had. |
| **Within-layer liners** | Never refused -- it is the geometry the caller asked for -- but WARNED with the mechanism and the per-layer / mortar / 2-D routes when the trigger super-unity is met and the owned cell's spurious-`\|q\|` predictor is past the stack's index ceiling by `_SLIVER_Q_EXCESS`. |
| **Cost** | One extra solve, **0.20x** (Windows) / **0.27x** (WSL) of the solve it guards, paid ONLY on a stack that already carries a manufactured sliver, is provably passive, and reads super-unity above the trigger. Fires on **0 of 600** converged correct rows. |
| **Bit-identity** | **39 / 39** -- the round-1 fix's 18 fixtures and the verification's 21 -- hashed against a read-only copy of the pre-round-2 tip `bb0527a`. |
| **Tests** | `tests/unit/test_fix_pmmstack_sliver_walls_round2.py` (19), plus the round-1 file's margin test RESTATED as a decision test and the verification's V-1 pinning test re-pinned against the improvement. |

### S0.1 CORRECTIONS, 2026-09-11 (verification defect D-4, applied in round 3)

Four numbers published above and below are properties of the SAMPLE this
report measured on, not of the family.  They are corrected here from the
independent re-measurement
(`docs/audits/VERIFY_PMMSTACK_SLIVER_ROUND2_2026_09_11.md` S3.1, S3.3, S4.4,
S5.3) and from the round-3 work
(`docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND3_2026_09_11.md`).  **None of the
four changes a decision**; all four make the guard's floors look further away
than they are, which is the defect the round-1 verification raised in the first
place.  Every site where one of them appears below carries an inline
`[D-4: ...]` marker.

| published here | corrected | measured on |
|---|---|---|
| the trigger clears the correct population by **9.11x** | **8.03x** -- envelope **1.24481e-04** | 826 correct rows over FIVE fixtures (this report used 600 over three); both builds |
| the WRONG population's `move / w_wide` floor is **466.2**, so the bar carries 4.66x | **147.411** (polarization 1 alone: **134.171**), so the bar carries **1.47x** | a 2,250-row grid over five fixtures whose continuity slopes span 0.47 .. 31.4; both builds read 147.411 to six figures, so the DECISION on those rows is stable |
| the CORRECT population moves at most **26.58**, i.e. 3.8x below the move bar | **833.78**, on a guided-mode-resonance grating whose degree-stationary `dR/d(duty)` is **169.2** | 2,028 rows, 936 of them on devices with slope > 20.  The move bar is a DECISION criterion whose correctness rests on the CLOSURE arm, not on a universal `dR/dx` = O(1): on all 14 rows adjudicated against an independent `RCWAStack` oracle the refusals are of answers RCWA also calls wrong, and on 12 of 14 the prescribed `min_feature` is 1.8x-7.1x closer to the truth |
| ONE sub-unity WRONG row, at `R+T-1` = **-2.8048e-04** | at least **11** sub-unity wrong rows, the worst at **-1.13511e-03**; 29 of 878 wrong rows sit at or below the trigger | the same 2,250-row grid, both builds |

---

## S1. The two builds

| | Windows | WSL (Ubuntu) |
|---|---|---|
| python | 3.14.6 | 3.12.3 |
| numpy | 2.4.4 | 2.4.6 |
| BLAS | scipy-openblas 0.3.31.188.0 (Haswell kernel) | scipy-openblas 0.3.31.188.0 (SkylakeX kernel) |
| threads | `OMP/OPENBLAS/MKL_NUM_THREADS = 1` | same |

The pre-round-2 reference is a **read-only copy** of the tip `bb0527a`: this
worktree's `lumenairy/` package with `elements/pmm/{stack,_core}.py` replaced by
`git show bb0527a:...`.  Every probe asserts which `lumenairy.__file__` it
imported.

---

## S2. What round 1 got wrong, restated as one sentence

Round 1's conjunct (b) is

> super-unity above `_STACK_SUPERUNITY_BAR` on a PROVABLY PASSIVE stack, where
> `R + T <= 1` is a theorem and not a tolerance

and that is true.  What does not follow is the next step -- that the *sliver*
is why the theorem is violated.  A degree-6 solve of a lossy-substrate grating
at 1.2 rad reads `R+T` = 1.036 whatever its walls do; put a harmless 1.2 nm
wall step on it and round 1 refuses, names the sliver, and prescribes a
`min_feature` that (measured) returns **1.03559** where the unguarded solve read
**1.03557**.  The refusal removed the ATTRIBUTION, not the error.

Symmetrically, the wrong population is not bounded away from the bar: on a
120-delta grid it reaches down to `R+T-1` = +7.14e-03, so eight wrong rows
returned with no warning at all.

Both are the same missing step, and one extra solve supplies it.

---

## S3. The design, and every bar in it

### S3.1 The trigger

```
_SLIVER_TRIGGER_BAR = 1.0e-3
```

The arbiter runs only when the stack trips the geometric screen (unchanged),
is provably passive, and reads `max(R+T) - 1 > _SLIVER_TRIGGER_BAR`.

Round 1 derived its bar from a 46-point sample of ONE fixture and the
verification refuted the margins on a denser grid of that same fixture, so this
bar is derived as a FAMILY property: three fixtures whose measured continuity
slopes differ by 4x, 80 log-spaced deltas each, three degrees
(`r3_trigger.py`, plus `r1_populations.py`'s 120 x 3 grid).

| fixture | period / lambda / theta | rows | CORRECT | max `\|R+T-1\|` among CORRECT | min `R+T-1` among WRONG |
|---|---|---|---|---|---|
| O-11 | 1.2 / 0.85 um / 0.15 | 240 | 126 | **1.0979e-04** | **-2.8048e-04** (SUB-unity) [D-4: one of at least 11, worst -1.13511e-03 -- S0.1] |
| mine, visible | 0.9 / 0.62 um / 0.21 | 240 | 121 | 1.5985e-06 | +9.5221e-02 |
| mine, telecom | 1.55 / 1.31 um / 0.08 | 240 | 144 | 5.6168e-06 | +3.7334e-02 |
| **ALL** | | **720** | **391** | **1.0979e-04** | **-2.8048e-04** |

Adding the 209 correct rows of `r1_populations.py` and `r2_falseneg.py`:
**1.0979e-04 over 600 CORRECT rows**, so 1e-3 carries **9.11x** [D-4: a
five-fixture family reads 1.24481e-04 over 826 correct rows, i.e. **8.03x** --
S0.1].  The ladder, scored two-sided on the same
1,380 rows (r1 + r2 + r3):

| trigger | headroom over the 1.0979e-04 envelope | wrong REFUSED | RIGHT refused | wrong / grey RETURNED | RIGHT refused by the CLOSURE-ONLY arbiter |
|---|---|---|---|---|---|
| 1e-2 (round 1's bar) | 91.1x | 770 / 777 | 0 / 600 | 7 / 3 | 0 / 600 |
| 3e-3 | 27.3x | 772 / 777 | 0 / 600 | 5 / 3 | 0 / 600 |
| **1e-3 (SHIPPED)** | **9.11x** | **774 / 777** | **0 / 600** | **3 / 3** | 0 / 600 |
| 3e-4 | 2.73x | 775 / 777 | 0 / 600 | 2 / 3 | 0 / 600 |
| 1e-4 | **1.01x** | 775 / 777 | 0 / 600 | 2 / 3 | **1 / 600** |

Identical on both builds.  3e-4 would catch ONE more wrong row and is
DECLINED: 2.73x is not a decade, and rule 5 asks for headroom over the
FAMILY's envelope rather than over the sample that happens to have been
scored.  At 1e-4 the bar sits ON that envelope (1.01x) -- the shipped
two-criterion arbiter still holds the line there because the MOVE criterion
catches what the trigger no longer does, but the closure-only form of the
arbiter (the verification's own discriminator) already refuses a correct row,
which is the concrete demonstration that the boundary is real and not a
sampling artefact.  A guard whose bar sits on its population is the exact
defect the verification raised against round 1.

**The floor is not removable by any super-unity bar.**  The O-11 family's worst
WRONG row on the three-fixture grid reads `R+T-1` = **-2.8048e-04** -- SUB-unity
[D-4: at least **11** such rows on a denser grid, the worst reading
**-1.13511e-03** -- S0.1].
A guard whose detector is `R+T <= 1` cannot see that row at any bar.

### S3.2 The arbiter's two criteria

```
attributed  <=>  max(R+T) - 1  on the prescribed grid  <=  _SLIVER_ATTRIB_CLOSURE
             AND |answer - snapped answer|_inf         >   _SLIVER_MOVE_FACTOR * w_wide

_SLIVER_ATTRIB_CLOSURE = 1.0e-5
_SLIVER_MOVE_FACTOR    = 100.0
```

`w_wide` is the WIDEST manufactured cell, i.e. the largest displacement the
prescribed snap can apply to any wall, and it is the same quantity the refusal
already uses to size `min_feature = 2 * w_wide * period`.

Measured over the **1,133 arbitrated rows** of all four grids (Windows):

| population | `max(R+T) - 1` on the snapped grid | `move / w_wide` |
|---|---|---|
| **WRONG** (n = 774) | 0 ... **1.5368e-06** | **466.2** ... 6.04e+07 [D-4: the family floor is **147.411** -- S0.1] |
| grey (n = 14) | 4.5038e-03 ... 1.2876e-01 | 11.07 ... 45.25 |
| **RIGHT** (n = 345) | 0 ... 1.2815e-01 | 0.1218 ... **26.58** [D-4: a resonant device reaches **833.78** -- S0.1] |
| **the bars** | **1e-5**: 6.5x above the wrong population, 3.9x below the truncation population's best (3.860e-05).  **ROUND 3** makes this bar the LOWER arm of a RELATIVE closure (defect D-5) | **100**: 3.76x above the correct population, 4.66x below the wrong one [D-4: **1.47x** below the family -- S0.1] |

**Why BOTH, and the ladder that shows it.** Scored on the 110 rows round 1
refused although correct (`r4_falsepos.py`):

| criterion | false positives left, Windows | WSL |
|---|---|---|
| round 1 (no arbiter) | **110** | **110** |
| "the super-unity falls below the TRIGGER" -- the verification's own phrasing of its measured discriminator | **17** | **17** |
| the shipped ONE-SIDED closure, `max(R+T) - 1 <= 1e-5` | **6** | **6** |
| **that AND `move > 100 * w_wide` (SHIPPED)** | **0** | **0** |

The closure test has to be ONE-SIDED -- `R + T <= 1` is what the theorem says,
and a passive stack with an absorbing substrate reads below unity legitimately
-- which is exactly why 6 rows survive it: their truncation super-unity happens
to land BELOW unity on the snapped grid, and a one-sided test reads that as
"vanished".  The move criterion is what separates those 6, and it is not a new
constant: 100 is the `err > 100 delta` WRONG rule the round-1 fix, its
verification and every probe in this campaign already classify with.

With both: **0 false positives, 774 / 774 wrong rows attributed, 0 / 14 grey
rows attributed.**

**The move is taken on BOTH polarizations** (what the library computes) rather
than on polarization 1 (the campaign's `err` convention, and the verification's
S4.1 definitional correction).  Measured separately: on polarization 1 the
correct population reaches 26.6 and the wrong one starts at 338; on both, 26.58
and 466.2 -- the same decision with more room [D-4: the family floors are 134.171
and 147.411 respectively, so the room is 1.34x / 1.47x -- S0.1].  Order sets are
`arange(-half, half+1)` on every path, so the comparison is the CENTRED overlap;
the snapped grid resolves a different number of orders on **359 of 637**
arbitrated rows, so this matters.

### S3.3 The verdicts, and what each does

| verdict | when | behaviour |
|---|---|---|
| `'sliver'` | both criteria met | **RAISE** the round-1 message plus an ATTRIBUTION paragraph quoting the snapped `R+T`, the move, and both bars |
| `'truncation'` | the super-unity survives, or the answer barely moves | **RETURN**, under the plain super-unity warning (unchanged bar 1e-2) with one sentence saying a sliver is present, is NOT the cause, and that raising `min_feature` will silence nothing |
| `'unknown'` | the one solve cannot be run: no resolved source, dispersive / keyed materials, or the re-solve itself raises | **ROUND 1's decision, unchanged**: raise above `_STACK_SUPERUNITY_BAR`, warn below it -- and the message says the attribution could not be measured |
| `None` | the geometric screen does not fire, or the stack is not provably passive, or `PMM_SLIVER_GUARD = False` | nothing changes at all |

**The probe geometry is never a different DEVICE, and that is structural.**
Every flagged cell satisfies `own / w >= _SLIVER_OWN_SCALE_RATIO` = 100, and
`w_wide` is the WIDEST of them, so the prescribed
`mf_fix = 2 * w_wide * period` is at most `own * period / 50` -- fifty times
finer than the finest wall spacing any single layer asked for.  `min_feature`
merges pairs CLOSER than itself, so it cannot reach a pair the caller
intended: the re-solve removes the manufactured cells and nothing else.
Measured over 182 flagged geometries (the O-11 grid at three degrees plus the
staircase box): worst `mf_fix / own` = **1.898e-02**, against the structural
bound 2.000e-02.  So a "yes" from the arbiter is evidence about THIS device,
not about a nearby one.

The re-solve is `stack._min_feature_clone(mf_fix)` -- the union-grid consensus
probe's own helper, pre-existing since the 2026-07-28 audit -- with the source
record copied and `_sliver_probe = True` set, so a probe can never arbitrate
itself.  Its warnings are suppressed the way `_union_grid_consensus_check`
suppresses its own, and its exception surface is narrowed to the same
`(ValueError, NotImplementedError, RuntimeError)` triple that file already
narrows to.

### S3.4 Conjunct (a)'s ratio bar: RE-MEASURED, KEPT AT 100

The verification refuted the round-1 claim that the M2 audit-class 2-degree
coated pillar taper sits at ratio ~1.7e+02: it reads **12.11**, because the
own-scale is the 5 nm conformal COAT and not the ~200 nm ridge, so that device
class is entirely outside the screen.  Three measurements (`r6_ratio.py`,
identical on both builds):

| | measured |
|---|---|
| the M2 taper's UNBARRED ratio | **12.11** at `n_slices` 2 / 6 / 8 (narrowest manufactured cell 0.4127 nm, own-scale 0.007143 of a 700 nm period), **3.593** at `n_slices` 3 -- confirming the verification and refuting round 1 |
| ORDINARY non-conforming stacks -- 4,192 random two-layer geometries whose every cross-layer wall gap is a real 1-8 % feature | largest ratio **25.43**, p99.9 = 21.76, p99 = 15.32, median 2.496 |
| does a ratio-12 manufactured cell carry the defect? -- the O-11 stack at the delta that is catastrophic on the bare fixture (1e-4, degree 14, err 0.479 = 4789x the shift), with an OWNED liner sized to set the ratio | ratio 1000 -> err **4.1x** delta; ratio 300 -> **0.8x**; ratio 100 -> **1.0x**; own-liner 1.2e-3 (ratio would be 12) -> **1.1x** -- every one CORRECT |

So: the M2 class sits INSIDE the ordinary non-conforming population, and no
ratio bar can admit it without admitting ordinary geometry; and the class does
not carry the defect in any case measured.  100 sits **3.93x** above the
ordinary population's worst.  **Unchanged.**

### S3.5 Passivity: the anisotropic class (V-5)

`_tensor_is_passive` accepts DIAGONAL tensors only -- deliberately, because it
is also `_forward_growth_flip`'s branch selector and must stay bit-identical.
Round 2 leaves it alone and adds a stack-local `_segment_passive` that falls
back to the general exact statement:

> the ANTI-HERMITIAN part `A = (eps - eps^H) / 2i` -- Hermitian by
> construction, and the part that does the absorbing in the library's
> `Im >= 0` convention -- is POSITIVE SEMI-DEFINITE.

One 3x3 `eigvalsh`, on the trigger path only.  It is EXACT where it matters: a
Hermitian `eps` gives `A == 0` identically, a diagonal one gives `A = diag(Im)`.

**The deadband is load-bearing and is measured.**  A rotated uniaxial director
is built as `R @ diag(no^2, no^2, ne^2) @ R.T`, which is symmetric in exact
arithmetic and only nearly so in floats -- and *how* nearly is BUILD-DEPENDENT:
the same 45-degree in-plane director tests exactly Hermitian on Windows and
NOT on WSL (`r7_anisotropic.py`, both arms).  An exact test would therefore
have shipped a guard that fires on one build and not the other.

| | measured |
|---|---|
| worst spurious `-lam_min(A) / max\|eps\|` over 200,000 random rotated uniaxial directors (arbitrary axis, n = 1.2-4.0) | **1.7115e-16 = 0.77 ULP** |
| the same over 50,000 LOSSY rotated directors | **0.00 ULP** (exact) |
| `_PASSIVE_ANTIHERM_DEADBAND` = 16 ULP = 3.5527e-15 | **20.8x** the worst round-off; the sizing constant is the one `_WALL_SNAP_DEADBAND` already uses |
| a GAIN payload at `Im(n)` = 1e-6 | -1.0e-06, i.e. **5.8e+09 ULP** -- 9 decades outside |

Measured consequence on the O-11 sliver (`r7_anisotropic.py`, both builds):

| tensor class | Hermitian EXACTLY? | provably passive | delta 1e-4 | delta 3e-5 | sliver-FREE control (3e-3) |
|---|---|---|---|---|---|
| in-plane LC director, 45 deg | Win **yes** / WSL **no** | **yes** | returned -- verdict `truncation`, `move` = 38.9 cell widths, below the bar | **REFUSED** -- `move` = 290.4, err 288x the shift, `R+T` 1.0105 | returned, `R+T` = 1 + 2.9e-11 |
| out-of-plane LC director, 30 deg | no (round-off) | **yes** | returned -- `move` = 22.2 | **REFUSED** -- `move` = 316.5, `R+T` 1.0133 | returned, `R+T` = 1 + 2.6e-11 |
| gyrotropic `eps_xy = -eps_yx = 0.35i` | yes | **yes** | **REFUSED** (err 685x) | **REFUSED** (err 2680x) | returned, `R+T` = 1 + 6.5e-11 |
| NON-Hermitian `eps_xy = 0.2, eps_yx = 0` | no | **no** | returned (unchanged) | returned (unchanged) | returned |

Round 1 read `R+T` = 2.183 on this class and only warned.  The negative control
is the last row: a payload no exact argument makes passive keeps exactly the
behaviour it had.

**And this class is where taking the move on BOTH polarizations earns its
keep.**  The out-of-plane director at `delta` = 3e-5 is refused on evidence the
campaign's own `err` column cannot see: scored per polarization against the
exact `delta -> 0` limit, polarization 1 is off by 3.17e-07 (**0.01x** the
physical shift -- "correct" by the continuity rule) while polarization 0 is off
by **9.49e-03**, i.e. **316x**.  A pol-1-only statistic would have called that
solve right and a pol-1-only move would have left the refusal unfired.  The
verification made the same definitional correction to round 1's separation
claim (its S4.1); here it changes a DECISION.

### S3.6 The WITHIN-LAYER arm (V-6): warned, never refused

A sliver-thin feature ONE layer owns is the geometry the caller described, and
no `min_feature` removes it -- the snap's own ownership rule says so.  It is
therefore never refused.  But the mechanism does not care who owns the walls,
and the verification measured how bad it gets.  Re-measured here
(`r5_within_layer.py`, identical on both builds; the reference is exact,
because the liner vanishes as `d -> 0`):

| liner `d` (of a period) | q-excess @ deg 14 | err, degrees 8 / 12 / 14 / 16 | `R+T` range |
|---|---|---|---|
| 1e-2 | 2.565e+02 | 1.30e-02 (flat) | 1 |
| 1e-3 | 2.565e+03 | 8.79e-04 (flat) | 1 |
| 1e-4 | 2.565e+04 | 8.47e-05 (flat) | 1 |
| 1e-5 | 2.565e+05 | 1.53e-05 / 1.23e-05 / 2.69e-06 / 9.11e-06 | 1 |
| **1e-6** | 2.565e+06 | 5.64e-05 / 7.44e-04 / **1.06e-03** / 3.34e-04 | 0.9992 ... 1 (SUB-unity) |
| **1e-7** | 2.565e+07 | 5.29e-02 / 6.60e-01 / 6.54e-01 / **1.05** | **0.5285 ... 4.185** |

so the arm fires when the trigger super-unity is met AND

```
q_pred / n_max  >=  _SLIVER_Q_EXCESS = 1.0e+6
```

with `q_pred = 0.65 N(N+1)/4 / (k0 J)` -- the refusal's own free predictor -- and
`n_max` the stack's largest index.  Two populations:

| population | q-excess |
|---|---|
| ordinary OWNED cells, 0.001-0.30 of a period, degrees 8-16 | 2.931e+00 ... **3.322e+03** (2.5 decades below the bar) |
| benign liners (1e-2 ... 1e-5, max err/`d` = 1.53, `R+T` = 1) | 8.793e+01 ... 3.322e+05 (they never reach the trigger anyway) |
| the liners that BREAK and read super-unity (1e-7) | **8.793e+06** ... 3.322e+07 (0.94 decades above) |

The warning names the mechanism, the measured cost, and the three routes that
keep the feature OFF the shared grid -- `layer_grids='per-layer'` (the
verification measured `err/d` = 0.34-0.46 down to 1e-6 of a period with
`|R+T-1| <= 8.0e-07`), the 2-D mortar cascade, or widening the feature.

**The floor here is the theorem again**: at `d` = 1e-6 the answer is already
1.06e-03 wrong while `R+T` reads 0.9992, i.e. SUB-unity, and nothing fires.
That is stated, not fixed.

---

## S4. The censuses, before and after

### S4.1 FALSE POSITIVES -- the verification's 648-configuration box

3 lossy substrates x 2 superstrate indices x 3 angles (1.2-1.45 rad) x degrees
6/8/10 x 2 permittivities x 2 or 4 slices x delta in {3e-3, 1e-3, 3e-4}, i.e.
wall steps of 0.36-3.6 nm on a 1.2 um period (`r4_falsepos.py`).

| | round 1 | **round 2** |
|---|---|---|
| configurations | 648 | 648 |
| CORRECT by continuity (`err <= 10 delta`) | 634 | 634 |
| **refused although correct** | **110 (17.0 % of the box, 17.4 % of the correct rows)** | **0** |
| their `err / delta` | 0.35 ... 8.77 | -- |
| their `R+T` | 1.01008 ... 1.12328 | -- |
| WRONG rows in this box | 0 | 0 |

The 110 are gone because the arbiter measures what round 1 assumed: on all of
them the super-unity SURVIVES the prescribed snap (the truncation population's
best snapped super-unity is 3.860e-05, 3.9x above the closure bar) or the
answer does not move (`move / w_wide` <= 26.58, 3.8x below the move bar)
[D-4: 26.58 is this box's number; a resonant device puts a CORRECT row at
833.78 -- S0.1].

### S4.2 FALSE NEGATIVES -- the verification's own 660 rows

120 log deltas (3e-3 ... 1e-6) x degrees 10/14/20, plus 60 log deltas
(3e-5 ... 1e-6) x degrees 8/10/12/14/16 (`r1_populations.py`, `r2_falseneg.py`).
All eight rows the verification reported reproduce exactly:

| degree | delta | kind | err | x the physical shift | `R+T - 1` | round 1 | **round 2** |
|---|---|---|---|---|---|---|---|
| 20 | 1.7130e-06 | wrong | 2.8003e-03 | 1634.8 | +7.1425e-03 | returned | **REFUSED** |
| 14 | 1.5859e-06 | wrong | 1.3763e-03 | 867.8 | +6.8637e-03 | returned | **REFUSED** |
| 12 | 2.2413e-06 | wrong | 1.0906e-03 | 486.6 | +1.7138e-03 | returned | **REFUSED** |
| 8 | 1.1888e-06 | wrong | 7.7156e-04 | 649.0 | +1.9903e-03 | returned | **REFUSED** |
| 10 | 1.8854e-06 | wrong | 3.3282e-04 | 176.5 | +5.1388e-04 | returned | returned (below the trigger) |
| 14 | 4.7421e-06 | grey | 2.1929e-04 | 46.2 | +3.3818e-04 | returned | returned (below the trigger) |
| 10 | 4.6995e-06 | grey | 9.6224e-05 | 20.5 | +3.7309e-05 | returned | returned (below the trigger) |
| 8 | 2.3743e-06 | grey | 3.1738e-05 | 13.4 | +4.1990e-05 | returned | returned (below the trigger) |

**8 -> 4**, and every one of the four is below `_SLIVER_TRIGGER_BAR`.  Which
half of the guard is holding each of them out is worth stating exactly, because
"lower the trigger" is not a universal answer:

| the four | `move / w_wide` | snapped super-unity | would the ARBITER attribute it if the trigger let it through? |
|---|---|---|---|
| deg 10, 1.8854e-06, wrong, 176.5x | 423.3 | 0.0 | **yes** -- the TRIGGER alone holds it out |
| deg 14, 4.7421e-06, grey, 46.2x | **83.3** | 2.38e-14 | **no** -- below the MOVE bar, so no trigger would reach it |
| deg 10, 4.6995e-06, grey, 20.5x | -- | -- | not measured (its `R+T-1` = 3.73e-05 is below the probe's own arbitration cutoff) |
| deg 8, 2.3743e-06, grey, 13.4x | 120.4 | 1.12e-08 | **yes** -- the trigger alone holds it out |

So two of the four are the trigger's floor (and S3.1 shows the trigger cannot
go lower without sitting on the correct population), one is the move
criterion's floor, and one is unmeasured.  The three that the arbiter would not
or might not attribute are all in the GREY band, 13-46x the physical shift.

The conical path gains one more that the census did not count: at `phi` = 0.62,
`delta` = 3e-5 the answer is 115x the physical shift at `R+T` = **1.00423** --
above the round-2 trigger and BELOW round 1's bar, so round 1 returned it and
round 2 refuses it (`r9_paths.py`, and
`test_the_conical_path_carries_the_arbiter_too`).

### S4.3 The arbiter's cost, and how often it fires

| | Windows | WSL |
|---|---|---|
| one guarded solve of the O-11 fixture (degree 14, 2 layers) | 86.2 ms | 74.1 ms |
| one ARBITRATION (the extra solve, on the snapped grid) | **16.8 ms** | **20.0 ms** |
| ratio | **0.20x** | **0.27x** |

It is cheaper than the solve it guards because the snapped grid carries fewer
cells (3 rather than 5 on the reproducer).

How often it fires:

| population | arbiter runs on |
|---|---|
| 600 CONVERGED CORRECT rows (r1 + r2 + r3, three fixtures) | **0** |
| 634 correct rows of the deliberately UNDER-CONVERGED staircase box (degree 6-10, lossy substrate, 1.2-1.45 rad -- built to read super-unity) | 345 (54.4 %), at 11 ms mean / 28 ms max, 4.0 s over 359 runs (Windows); 10 ms / 26 ms / 3.9 s (WSL) |
| any stack with no manufactured sliver, or not provably passive, or below the trigger | never (the screen is pure geometry and runs first) |

So the honest statement is: it never fires on a converged solve, and where it
does fire the caller was already being told their answer is unreliable.

### S4.4 Every path

`r9_paths.py`, both builds.  `probe` counts the arbiter's extra solve.

| path | delta 3e-3 | delta 1e-4 | delta 3e-5 |
|---|---|---|---|
| `solve()` classical | returned, probe 0x | **REFUSED (attributed)**, probe 1x | **REFUSED (attributed)**, probe 1x |
| `solve()` conical `phi` = 0.62 | returned, 0x | returned, 0x | **REFUSED (attributed)**, 1x |
| `solve()` slant 0.17 | returned, 0x | returned, 0x | returned, 0x (this fixture never reaches the trigger) |
| `solve()` `layer_grids='per-layer'`, 5 layers | returned, 0x | **REFUSED (attributed)**, 1x | **REFUSED (attributed)**, 1x |
| `solve(stabilize='slices')` | -- | REFUSED, 1x (the guard raises before the consensus probe) | -- |

The per-layer row is worth one more sentence, because it is the path where the
geometric screen (which reads the UNION) and the grid the cascade actually ran
on are different objects.  The arbiter does not depend on that: its evidence is
a re-solve on the SAME path, so what it measures is the answer the caller got.
Scored against that stack's own exact `delta -> 0` limit, the two refusals are
correct by **7,062x** and **564,414x** the physical shift (`R+T` = 2.426 and
52.00), and the returned row is correct at 4.1x --
`test_the_per_layer_window_path_is_arbitrated_on_its_OWN_grid`.

| path | measured |
|---|---|
| `solve_vs_wavelength` | the sweep never writes `stack._src`, so round 2 passes the sweep's own `(wl, angle)` EXPLICITLY. Asserted by spying on the probe: with a STALE `set_source(4e-07)` on the stack and a sweep at 8.5e-07, the arbiter is handed **[8.5e-07]** and the solve is REFUSED. Without the explicit `src` the re-solve would have run at a different physics. |
| `solve_vs_wavelength`, DISPERSIVE | `provably_passive = False` (a callable `eps` cannot be resolved), so the geometric screen is never reached, the probe runs **0x**, and the solve RETURNS under the plain warning -- unchanged from round 1 |
| `prepare().solve` | `set_source` is never required there, so `st._src` is unset; the arbiter is handed **[8.5e-07]** from the call and the solve is REFUSED with a measured attribution |
| `solve_vs_wavelength` at **2 and 4 workers** | the verification could not check the guard raising from inside a thread pool. `_store` runs on the CALLING thread in both branches, so the arbiter's extra solve is a main-thread solve: measured, the sweep REFUSES identically at 1 / 2 / 4 workers with the probe handed `[8.5e-07]` each time, and a HEALTHY 2-wavelength sweep is probed **0x** and stays byte-identical across worker counts |
| JAX / traced | `_stack_provably_passive` answers False on a traced payload, so the guard is inert by construction -- unchanged |

---

## S5. Bit-identity

`r8_bitid.py` hashes sha256 over dtype, shape and raw buffer of every array
each fixture returns, and runs against this worktree and against the read-only
copy of `bb0527a`.  39 fixtures: the round-1 fix's 18
(`validation/probe_pmmstack_sliver/bitid_fixtures.py`) and the verification's 21
(`validation/probe_verify_sliver/v_fixtures.py`).

| | fix tree vs pre-round-2 tip |
|---|---|
| Windows | **39 / 39 identical**, 0 differing, 0 errors; **0 / 39 warning sets differ** |
| WSL | **39 / 39 identical**, 0 differing, 0 errors; **0 / 39 warning sets differ** |

Round 2 adds two NEW warnings -- the truncation note appended to the plain
super-unity warning, and the within-layer arm -- so bit-identity of the RETURN
value is only half of "nothing changes".  `r8_bitid.py` therefore also records
what each fixture WARNS: all 39 are silent on both trees on both builds, so no
previously-quiet solve became noisy.

| fixture | hash (first 32 hex) |
|---|---|
| `fix:f01_one_layer_normal` | `9f717fd86bf47882c80cb0ada1c83b5f` |
| `fix:f04_bragg_abab` | `72383692ec5fe4a2d0391eb7e821ed1b` |
| `fix:f07_taper_snap_active` | `45b93a1d98ea8d03cd39997d9012c928` |
| `fix:f10_conical` | `bc33996940ac4734211b54157262a017` |
| `fix:f12_out_of_plane_tensor` | `210c46f88aa33e04c5d2afdc20ca46dc` |
| `fix:f14_solve_vs_wavelength` | `b14f96672464f9c94adac2f1b60bc612` |
| `fix:f15_prepare` | `47072c1e71ae2bc74b70000c20af680f` |
| `fix:f16_stabilize_slices` | `eaf7cc240e0d1fa6a62182d7b57fc1ac` |
| `fix:f17_internal_field` | `620dc9aeecdf1ce58f73ca33599d6c85` |
| `fix:f18_layer_absorption` | `c863f00f60896e6a22568131e1c5eca1` |
| `verify:v09_in_plane_tensor_xy` | `86533aec97a78da7b5f024ebd0225eb0` |
| `verify:v11_absorbing_superstrate` | `f789c11eab66c91f8b0a8ee95794a41a` |
| `verify:v13_prepare_material_sweep` | `7f1e84231013e58050ec8b5c4316ca34` |
| `verify:v19_per_order_amplitudes` | `108b88f922d1932434b4d4cc77fde6c2` |
| `verify:v21_union_grid_two_tuple` | `b9dc0766095f8c6731cd308c21e8d40b` |

(the full 39 are in `r8_bitid_new.json` / `r8_bitid_tip.json`.)  The round-1
18 also match the hashes the round-1 audit published, so the chain from the
pre-fix main clone through round 1 to round 2 is unbroken on those fixtures.

`PMM_SLIVER_GUARD = False` still restores the pre-fix path completely, and
round 2 extends that contract: with the switch off no arbiter solve runs and no
within-layer warning fires
(`test_the_fail_before_switch_still_disarms_everything_round_2_added`).

---

## S6. Tests

| file | tests | Windows | WSL |
|---|---|---|---|
| `tests/unit/test_fix_pmmstack_sliver_walls_round2.py` (new) | 19 | 24.66 s | 27.94 s |
| `tests/unit/test_fix_pmmstack_sliver_walls.py` (round 1, one test restated) | 19 | 17.52 s | 17.22 s |
| `tests/unit/test_verify_pmmstack_sliver_walls.py` (verification, one test re-pinned) | 15 | 32.40 s | 30.78 s |

### S6.1 What the new file pins

| test | what it pins |
|---|---|
| `test_fail_before_round1_refuses_a_solve_that_tracks_the_physical_shift` | FAIL-BEFORE on round 1's own code path (the `'unknown'` branch, forced), then the same fixture returning under round 2 with the truncation sentence |
| `test_no_correct_solve_in_the_realistic_staircase_box_is_refused` | the false-positive census as a 60-configuration arm; also asserts that round 1 WOULD have refused >= 8 of them, so the arm is evidence and not a vacuous pass |
| `test_the_round1_misses_are_refused_or_are_below_the_trigger` | the eight false-negative rows, regenerated from the census's own `geomspace` so the deltas are the exact float64s; every one that reaches the trigger must be refused and every one that does not must be below it |
| `test_the_hazard_band_is_still_refused_and_the_outside_is_untouched` | the round-1 two-sided claim at the round-2 trigger, with bit-identity outside the band |
| `test_the_arbiter_separates_the_two_causes_on_this_build` | both arbiter bars re-derived at runtime, with a decade of separation demanded on the quantity that carries each decision |
| `test_the_trigger_sits_above_the_correct_populations_envelope` | rule 5: the trigger's own premise, measured on the running build |
| `test_the_arbiter_costs_one_solve_and_only_on_a_triggered_stack` | the cost claim, executed: exactly one probe inside the band, zero outside |
| `test_the_probe_solve_can_never_arbitrate_itself` | the `_sliver_probe` invariant |
| `test_the_move_is_taken_on_the_orders_the_two_solves_share` | the centred-overlap contract |
| `test_a_rotated_uniaxial_director_is_provably_passive` | the passivity extension including the LOSSY director and the GAIN negative control |
| `test_the_guard_now_reaches_a_liquid_crystal_sliver` | V-5, on three tensor classes, each against its own sliver-free control, plus the non-Hermitian negative control |
| `test_a_within_layer_sliver_is_warned_with_the_mechanism_never_refused` | V-6's warn-not-refuse contract and the routes the message must name |
| `test_the_within_layer_warning_is_silent_on_ordinary_geometry` | the other side of it |
| `test_the_sweep_arbitrates_at_its_own_wavelength_not_a_stale_set_source` | the sweep's explicit `src`, asserted by spying on the probe |
| `test_the_sweep_arbitrates_the_same_way_at_any_worker_count` | the thread-pool sweep at 1 / 2 / 4 workers, plus the healthy sweep's byte-identity across worker counts |
| `test_the_prepared_path_arbitrates_at_the_wavelength_it_was_given` | the same for `prepare()` |
| `test_the_conical_path_carries_the_arbiter_too` | the conical map, two-sided, including the row that is below round 1's bar |
| `test_the_per_layer_window_path_is_arbitrated_on_its_OWN_grid` | the per-layer window path, where the screened grid and the solved grid differ, two-sided on a 5-layer staircase |
| `test_the_fail_before_switch_still_disarms_everything_round_2_added` | `PMM_SLIVER_GUARD = False` |

### S6.2 The two tests that had to change, and why

**`test_the_bar_has_decades_of_gap_on_both_sides_measured_here` (round 1) ->
`test_the_guards_DECISION_is_right_on_a_dense_grid_not_just_this_ladder`.**
The verification's S7 named this file's most important durability defect: the
test re-derives its two bars at runtime, which is right, but on a 13-row
`_LADDER` whose composition is what makes the separation hold.  The same two
quantities on a 120-row grid of the same family read 9.87e-05 and 7.14e-03, so
the first assertion would pass by 1.3 % and the second **fail by 42x**.  The
restated test runs a 30-delta x 2-degree grid and asserts the DECISION: no row
the continuity rule calls RIGHT may be refused (and outside the band the
numbers must be bit-identical), and every wrong row the guard returns must be
one whose super-unity is below the trigger, so the residual is the theorem's
rather than the implementation's.

**`test_the_refusal_fires_when_the_super_unity_is_TRUNCATION_not_the_sliver`
(verification) -> `test_a_TRUNCATION_super_unity_is_no_longer_blamed_on_the_sliver`.**
This test PINNED V-1 as a known defect and its own docstring said "if the guard
is later taught to separate the two causes this test fails, and that failure is
the gate working -- re-pin it against the improvement, do not relax it."  It
failed on the first run of the round-2 library, which is the gate working.  The
same fixture now asserts the same three facts with the verdict inverted: the
screen and the theorem still both fire, the ARBITER is what declines to
attribute, the solve returns BIT FOR BIT the unguarded answer under a warning
that says a sliver is present and is not the cause, and remedy (1) still leaves
the number where it was -- which is why the arbiter refuses to call it the cure.

---

## S7. Open items

| | |
|---|---|
| **R2-A -- the floor is the theorem, and it is one-sided** | Four of the verification's eight rows remain returned because their super-unity is below the trigger, and the trigger cannot go lower without sitting ON the correct population's own envelope (1.01x at 1e-4, where the closure-only arbiter already refuses a correct row). Worse, the three-fixture grid contains a WRONG row at `R+T-1` = **-2.8048e-04** -- SUB-unity -- which no super-unity bar can ever see [D-4: at least **11** such rows, worst **-1.13511e-03** -- S0.1]. A detector for that band needs something this campaign still has not found. |
| **R2-A2 -- one of the four is the MOVE criterion's floor, not the trigger's** | The grey row at degree 14, `delta` = 4.7421e-06 (46.2x the physical shift) moves only **83.3** cell widths on the snapped grid, i.e. below `_SLIVER_MOVE_FACTOR`, so no trigger would reach it -- S4.2 has the per-row table. Lowering the move bar to catch it costs headroom over the correct population (which reaches 26.58 here and **833.78** on a resonant device -- D-4, S0.1) and was not taken. |
| **R2-B -- the within-layer arm inherits the same floor** | At a liner of 1e-6 of a period the answer is already 1.06e-03 wrong while `R+T` reads 0.9992. The arm fires only above the trigger, so that width is silent. |
| **R2-C -- the arbiter is inert on dispersive / keyed stacks** | A callable or unresolved `eps` makes `_stack_provably_passive` answer False, so those stacks never reach the screen at all -- they keep the plain warning, which is round 1's behaviour and pre-round-1's. Materialising the callables for the probe is possible and was not done. |
| **R2-D -- `move` is compared to a GEOMETRIC width** | `move > 100 * w_wide` compares an efficiency difference with a period fraction, i.e. it silently assumes `dR/dx = O(1)`. That is the same shape of derivation the verification refuted for the round-1 remedy bar (V-4, measured slopes 1.04 / 1.15 / 4.44). Here it is measured rather than derived -- the two populations sit at <= 26.58 and >= 466.2 across four grids and three fixtures -- but a device with `dR/dx` above ~50 could in principle move a correct answer past the bar. No such device was found; a resonant fixture would be the way to attack it. **RE-STATED, not closed, 2026-09-11 (D-4 / verification S5.3):** the counter-fixture exists -- a guided-mode-resonance grating with a degree-stationary `dR/d(duty)` of 169.2, on which a CORRECT row's `move / w_wide` reaches **833.78**, a 31x refutation of the published bound; and the family floors are 147.411 / 134.171 rather than 466.2 / 338. The DECISION survives: on all 14 rows adjudicated against an independent `RCWAStack` oracle the guard refuses answers RCWA also calls wrong, and on 12 of 14 the prescribed `min_feature` is 1.8x-7.1x closer to the truth. So the criterion keeps its bar and loses its published margin, and its correctness rests on the CLOSURE arm rather than on a universal `dR/dx` = O(1). |
| **R2-E -- the 2-D stacks still pass `stack=None`** | Unchanged from round 1: `stack2d.py` keeps the plain warning, which is correct while `PMM2DStackPure`'s union is the pixel lattice. The union-forming route of the mortar work (round 1's open item D) is still the owner of that hazard. |
| **R2-F -- one arbitration per super-unity report** | On `solve_vs_wavelength` the probe runs per wavelength that trips the trigger, so a sweep whose whole band is in the hazard pays it on every point. Not measured beyond the single-wavelength case. |

Round 1's open items **A** (fixed by the verification's deadband), **C**
(fixed), **E** (closed by the verification) and **F** (this document) are done.
**B** is R2-A, re-measured.  **D** stands.

---

## S8. The runs

All with `PYTHONPATH` on the worktree and one BLAS thread.

| run | Windows | WSL |
|---|---|---|
| `tests/unit/test_fix_pmmstack_sliver_walls_round2.py` (new) | **19 passed**, 24.66 s | **19 passed**, 27.94 s |
| `tests/unit/test_fix_pmmstack_sliver_walls.py` (round 1) | **19 passed**, 17.52 s | **19 passed**, 17.22 s |
| `tests/unit/test_verify_pmmstack_sliver_walls.py` (verification) | **15 passed**, 32.40 s | **15 passed**, 30.78 s |
| `tests/unit/test_m1_conditioning_guard.py` | **27 passed**, 4.9 s | **27 passed**, 4.85 s |
| all four in one process | **80 passed**, 76.03 s | **80 passed**, 77.05 s |
| every test file importing `PMMStack` (`grep tests/ --include='*.py' -l PMMStack`, **42** files incl. this round's), slow markers included | see below | -- |
| `ruff check lumenairy/ tests/ validation/probe_pmmstack_sliver_round2/` | **All checks passed** | -- |

**REGRESSION.**  Every test file that imports `PMMStack` -- 42 files
(`validation/probe_pmmstack_sliver_round2/_regression_files.txt` records the
list), including the two mortar files, the verification's and this round's --
Windows, one BLAS thread, slow markers INCLUDED:

| | |
|---|---|
| result | **812 passed, 1 skipped, 0 failed** |
| wall | 43 min 19 s |
| the skip | the pre-existing `test_niche_audit_m4_m5_m6_rcwa.py:387` (*"threadpoolctl installed: the cap is effective here"*), unrelated to this work -- the same one the round-1 and verification regressions record |
| the four slowest | the same `test_audit_dynameta_consumer_api_2.py` pure-2-D arms as before (499 / 379 / 255 / 219 s), unchanged in character |

For reference the same gate reads **793 passed / 1 skipped** on the
pre-round-2 tip (the verification's run, 41 files) and **731 / 1** on the
round-1 fix (37 files).

That run was taken on the library exactly as it ships (frozen at `84e15e7`; no
`lumenairy/` file changed after it started -- `git show --stat` on every later
commit shows only `tests/`, `docs/`, `.test_durations` and `validation/`).
Two test additions landed in `test_fix_pmmstack_sliver_walls_round2.py`
afterwards -- the per-layer arm and the polarization-0 assertion inside the LC
gate -- so the file-level total is now 813; both were run green on BOTH builds
in that file's own run (19 passed, 24.66 s Windows / 27.94 s WSL) and in the
four-file run (80 passed on both).

`.test_durations` spliced with the 53 measured Windows timings of the three
sliver files (12,571 -> 12,590 entries; the two renamed tests' stale entries
dropped).

`test_m1_conditioning_guard.py` prints two
`** On entry to DLASCL parameter number  4 had an illegal value` lines from
LAPACK on WSL.  That is PRE-EXISTING and unrelated to this round: the same two
lines appear from that file alone, and none of the three sliver files produces
any (measured by running each file separately on WSL).  Its module-scope
autouse `PMM_SLIVER_GUARD` disarm is unchanged and still the right call --
round 2 adds an arbiter to a guard that file has no business tripping.

### S8.1 Cross-build agreement

Every number in S3 and S4 is IDENTICAL on the two builds to the digits printed:

| | Windows | WSL |
|---|---|---|
| arbitrated rows at the trigger (four grids) | 1,133 | 1,133 |
| snapped super-unity, WRONG population | 0 ... 1.5368e-06 | 0 ... 1.5368e-06 |
| snapped super-unity, TRUNCATION population's best | 3.860e-05 | 3.860e-05 |
| `move / w_wide`, CORRECT population | 0.1218 ... 26.58 | 0.1218 ... 26.58 |
| the same on a RESONANT device (D-4, S0.1) | 833.78 | -- |
| `move / w_wide`, WRONG population | 466.2 ... 6.04e+07 | 466.2 ... 6.04e+07 |
| the same, FAMILY floor (D-4, S0.1) | 147.411 | 147.411 |
| CORRECT envelope over 600 rows | 1.0979e-04 | 1.0979e-04 |
| the same over 826 rows / 5 fixtures (D-4, S0.1) | 1.24481e-04 | 1.24481e-04 |
| false positives, round 1 -> round 2 | 110 / 648 -> **0 / 648** | 110 / 648 -> **0 / 648** |
| "super-unity below the trigger" alone would leave | 17 | 17 |
| false negatives, round 1 -> round 2 | 8 / 660 -> **4 / 660** | 8 / 660 -> **4 / 660** |
| arbiter fired on correct rows of the under-converged box | 345 / 634, 11 ms mean | 345 / 634, 10 ms mean |
| bit-identity, 39 fixtures vs the pre-round-2 tip | **39 / 39** | **39 / 39** |

The two builds' HASHES differ from each other (different OpenBLAS
microarchitecture kernels, hence different reduction orders), which is the
control that the 39/39 within each build is evidence and not one build's luck.

The one place the builds genuinely disagree is the one the design anticipated:
the SAME rotated 45-degree in-plane director tests EXACTLY Hermitian on Windows
and NOT on WSL.  Both read `provably_passive = True`, because the shipped test
is the anti-Hermitian semi-definiteness with its measured round-off deadband --
an exact-equality test would have shipped a guard that fires on one build and
not the other.

---

## S9. Commits (branch `fix/pmmstack-sliver-guard-round2`)

| | |
|---|---|
| `9a2be42` | `probe(pmm)` -- the round-2 measurements: both populations on three fixtures, the arbiter's two bars, the false-positive / false-negative censuses, the ratio, within-layer and anisotropic classes |
| `4e513cf` | `fix(pmm)` -- the arbiter, the anti-Hermitian passivity extension, the within-layer warning, the `src` wiring on the sweep and prepared paths |
| `9642a03` | `test(pmm)` -- the round-2 gates, the re-pinned verification test, the round-1 margin test restated as a decision test |
| `1b80a57` | `probe(pmm)` -- bit-identity against the pre-round-2 tip and the per-path arbiter probe, both builds; the `[Unreleased]` round-2 paragraph and the first draft of this report |
| `d90cdb9` | `fix(pmm)` -- drop the unused `stack` argument from `_segment_passive`; docstring numbers re-measured |
| `c7413e0` | `test(pmm)` -- the thread-pool sweep arm, and every bar's justification re-measured against the SHIPPED two-criterion arbiter |
| `84e15e7` | `docs(pmm)` -- this report, the WARNING-identity arm of the bit-identity probe, the per-row residual analysis |
| `1b6aa6b` | `test(pmm)` -- the out-of-plane director's refusal is polarization-0 evidence, re-derived in the LC gate |
| `d5c575f` | `docs(pmm)` -- the commit ledger and the warning-identity line in the changelog |
| `fa9db07` | `test(pmm)` -- the per-layer window path is arbitrated on its own grid, two-sided on a 5-layer staircase |
| `ae6dcd6` | `docs(pmm)` -- the prescribed `min_feature` is bounded at `own / 50`, so the arbiter's re-solve is the same device |
| `e9055c9` | `docs(pmm)` -- the full `PMMStack` regression: 812 passed, 1 skipped, 0 failed in 43:19 |

Probes: `validation/probe_pmmstack_sliver_round2/` -- `r_fixtures.py`,
`r1_populations.py`, `r2_falseneg.py`, `r3_trigger.py`, `r4_falsepos.py`,
`r5_within_layer.py`, `r6_ratio.py`, `r7_anisotropic.py`, `r8_bitid.py`,
`r9_paths.py`, a README, and their JSON on both builds.

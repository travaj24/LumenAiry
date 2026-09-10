# VERIFY -- the `PMMStack` sliver arbiter's RELATIVE CLOSURE, round 3, independently re-measured

**Date** 2026-09-11 · **Subject**
`docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND3_2026_09_11.md` (branch
`fix/pmmstack-sliver-guard-round3`, merged at `4a6cf01`) · **Worktree**
`C:/tmp/lum_vsliver3`, branch `verify/sliver-round3` · **Without-arm** a
read-only `git worktree add --detach C:/tmp/lum_vsliver3_pre f2371e0` (round 3's
branch point: round 1 + round 2 + round 2's independent verification)

**Probes** `validation/probe_verify_sliver_round3/` -- ten probes and a
fixture module written from scratch, run on both builds.  No number in this
report is read out of the fix's JSON or out of the round-2 verification's:
every statistic below was re-measured here, and on DIFFERENT devices.

**New tests** `tests/unit/test_verify_pmmstack_sliver_round3.py` (5).  No file
under `lumenairy/` was touched.

**Binding** `docs/TESTING_STANDARDS.md`.

---

## S0. Terms

The guard, the screen and the arbiter are rounds 1 and 2's.  Round 3 changes
one expression inside the arbiter, so the terms are stated once and then used.

| term | meaning |
|---|---|
| **manufactured cell** | a cell of the shared union grid whose two walls share no owning layer.  `_cross_layer_sliver` flags one when it is at least `_SLIVER_OWN_SCALE_RATIO` = 100x finer than the finest wall spacing any single layer asks for |
| **the trigger** | `_SLIVER_TRIGGER_BAR` = 1e-3.  The arbiter runs only on a stack that carries a manufactured cell, is provably passive, and reads `max(R+T) - 1` above this |
| **the prescribed grid** | the grid `min_feature = 2 * w_wide * period` produces, `w_wide` being the WIDEST manufactured cell.  It is the grid the refusal's first remedy hands the caller, and the arbiter's one extra solve is taken there |
| **the violation** | `max(R+T) - 1` on the solve about to be returned.  On a provably passive stack in a lossless propagating incidence medium `R + T <= 1` is a theorem, so this is a theorem violation and not a tolerance |
| **`su_snapped`** | `max(R+T) - 1` on the prescribed grid, floored at 0 |
| **the DROP factor** | `violation / su_snapped` -- the fraction of the violation the prescribed snap REMOVES, written as a ratio.  Infinite when the snapped solve reads at or below unity.  Round 3's constant is a bar on this |
| **the MOVE** | `move / w_wide`: the largest per-order efficiency difference between the returned solve and the prescribed-grid solve, over the orders they share and over BOTH incident polarizations, in units of the widest manufactured cell |
| **the closure** | `max(_SLIVER_ATTRIB_CLOSURE, violation * _SLIVER_CLOSURE_FRACTION)` = `max(1e-5, violation * 1e-2)` -- what `su_snapped` must fall to before the sliver is the attributed cause |
| **CORRECT / WRONG** | the campaign's fitted-constant-free continuity rule against the exact `delta -> 0` solve of the same device: `err <= 10 delta` is RIGHT, `err > 100 delta` is WRONG, between is GREY.  `err` is the polarization-1 per-order distance |
| **arbitrated** | a row on which the arbiter actually runs: the screen fires AND the violation is above the trigger |
| **a D-5 mount** | one whose SLIVER-FREE truncation super-unity (the `max(R+T) - 1` its own degree leaves at `delta = 0`) sits BETWEEN `_SLIVER_ATTRIB_CLOSURE` and the trigger.  On such a mount round 2's ABSOLUTE closure can never be met, however completely the snap restores the answer |
| **a FALSE REFUSAL** | a CORRECT row on which BOTH arms hold, so the library refuses an answer that was right |

---

## S0.1 Verdict summary

Every number in the right-hand column was measured here, on both builds, and
the two builds agree on every DECISION.

| # | claim | verdict | the number |
|---|---|---|---|
| 1a | bit-identity of every RETURNED answer across the change | **CONFIRMED** | 42 of 66 fixtures are returned by both trees; **42 / 42** identical in the SHA-256 of the raw `R`, `T` and Jones buffers, 0 broken.  The 66 unguarded solves are identical BEFORE and AFTER as well |
| 1b | the warning set is unchanged except the reworded `truncation` note | **CONFIRMED** | after excising the note, **42 / 42** warning sets match character for character; 0 notes added, 0 removed.  Three fixtures carry the note on both trees and its text differs exactly as the report describes |
| 1c | every round-2 REFUSAL still refuses | **CONFIRMED** | 13 refusals before, 24 after, **13 refused by both**; 0 refusals lost |
| 1d | the only verdict transition is `truncation -> sliver`, and every flip is WRONG | **CONFIRMED, and more sharply than published** | **11 flips**, all `truncation -> sliver`, **0** the other way.  All 11 are WRONG against the exact `delta -> 0` reference (`err/delta` **52,048 .. 7.60e+07**) and all 11 have the prescribed-grid answer RIGHT (`err/delta` 0.14 .. 1.60).  The mildest flip is 517.7x on the fix's devices and **52,048x** on mine |
| 2a | the CORRECT population's FINITE drop envelope is 36.611, so the 100x bar carries 2.73x | **BOUNDED (sample-scoped, and looser on an independent box)** | on my 576-mount / 2,304-row box the envelope is **18.0900** -- the bar carries **5.53x**.  36.611 remains the binding published number; nothing here refutes it |
| 2b | 1e-2 admits no CORRECT row the round-2 absolute closure did not already admit | **CONFIRMED** | 15 correct rows have an INFINITE drop, all 15 admitted by the absolute bar; the closure admits exactly those 15 at 1e-3 / 3e-3 / 1e-2 / **3e-2**, and 18 (+3, drops 12.32 / 16.45 / 18.09) at 1e-1.  On my box the boundary sits at 3e-2, one step COARSER than the shipped value |
| 2c | the CONJUNCTION attributes 0 correct rows at every fraction 1e-1 .. 1e-3 | **CONFIRMED on the ladder box** (but see 2d) | **0 / 737** arbitrated correct rows at all five fractions; the library refused **30** rows of 2,304, every one WRONG, **0 false positives**.  The claim does not survive a directed sweep of the same mounts -- row 2d |
| 2d | no CORRECT row both drops past 100 and moves past 100 `w_wide` -- "0 correct rows attributed at ANY fraction from 1e-1 to 1e-3" | **REFUTED -- three FALSE REFUSALS found, and they are INHERITED from round 2** | 0 over the 2,304-row ladder box (closest joint approach **0.1376**, 7.27x), but a DIRECTED scan of that box's own worst mounts at **degree 4** finds **3** rows the library REFUSES whose answers are CORRECT (`err/delta` 0.979 / 1.051 / 1.065).  Their snapped solve reads BELOW unity, so the ROUND-2 ABSOLUTE closure admits them too -- confirmed by running the same probe on `f2371e0`, which refuses all three.  See defect **V-4** |
| 2e | the analytic round-3 criterion equals the library's decision | **CONFIRMED** | **859 / 859** arbitrated rows |
| 3a | the D-5 class is reachable | **CONFIRMED, and it is far broader than one mount** | **9** of 24 screened (mount, degree) pairs land in the band, across **5** unrelated mechanisms: guided-mode resonance, dense-superstrate grazing staircase, Fabry-Perot cavity, near-Wood mount, high-index-contrast lossy substrate |
| 3b | the D-5 population runs 49.107 .. 5,304.6 in drop | **REFUTED as a floor (sample-scoped); the structural prediction is CONFIRMED** | 102 D-5 rows on my mounts run **3.6688 .. 4.2875e+06**.  The floor is 13.4x BELOW the published one and lands INSIDE the correct population's own drop range (0.638 .. 18.09), which is exactly what the fix's own S3.5 predicts structurally |
| 3c | 1e-2 recovers 85 / 88 of the D-5 population | **CONFIRMED at the same rate** | **100 / 102** (98.0 % vs the fix's 96.6 %); 3e-3 recovers 88, 1e-3 recovers 77 |
| 3d | R3-A: the rows the constant cannot reach are the visible edge of a structural band | **CONFIRMED, and one of them is returned SILENTLY** | 2 rows: drop **5.205** at `err/delta` **741.3** and drop **3.669** at `err/delta` **212.8**, both fully restored by the prescribed snap.  The first reads `R+T` = 1.00235, i.e. BELOW `_STACK_SUPERUNITY_BAR`, so it is returned with no warning at all |
| 3e | the new `truncation` note is TRUE on every returned row | **CONFIRMED on the decision, BOUNDED on the wording** | **0 of 308** returned noted rows is WRONG-returned-with-a-RIGHT-snapped-answer, on both builds.  47 of the 308 are GREY as returned while the snapped answer is RIGHT (`err/delta` ~20 vs ~0.92): the note's quoted evidence is exact there, its headline sentence is loose |
| 4a | the correct population's `move / w_wide` envelope is 79.032 against the 100 bar (1.27x) | **REFUTED -- the bar is INSIDE the correct population** | a CORRECT row of my box reads **161.073**.  Its drop is 0.875, so the CLOSURE arm is what holds it out; the move criterion alone does not separate |
| 4b | the move bar needs re-derivation before 5.45.0 | **NO -- a decision, with numbers** | see S5.3 and S5.4.  The steepest degree-stationary device this campaign could build reads `dR/d(duty)` = **89.31**, just under the bar; what actually crosses it is the many-slice taper, where `move` SATURATES (4.3e-03) while `w_wide` falls, so `move / w_wide` reaches **906.56** on rows that are RIGHT.  Neither arm separates alone; the CONJUNCTION does (54.8x on that family, 7.27x on the ladder box).  Lowering the bar manufactures false refusals, raising it returns wrong answers round 2 already measured at 147.411.  Keep the bar; delete the published 1.27x margin |
| 5 | every decision is identical on Windows and WSL | **CONFIRMED** | 66 / 66 fixture decisions, all integer and boolean census fields, all 9 in-band mounts, all 102 D-5 rows, all 30 census refusals and all 3 V-4 refusals.  Largest RELATIVE spread on the arbiter's evidence dict: **0.68** on a `drop` whose denominator is 5.6e-15 / 1.8e-15 (machine epsilon; the decision compares 2.1e+14 against 100).  Excluding machine-epsilon denominators: **1.44e-02**.  Population statistics agree to 8-10 significant figures |
| 6 | the round-3 test file's bars | **CONFIRMED on both builds; two are SAMPLE-scoped** | see S7.  All six tests pass on both builds; two assertions (`n_arb >= 8`, `su_snap <= 1e-6`) are properties of the enumerated sample rather than of the family, at 1.75x and 4.06e+06x |

**Defects raised**: **V-4** (MEDIUM, INHERITED from round 2 -- three CORRECT
answers are refused), **V-1** (LOW, doc-only -- three published population
bounds are sample-scoped and two are refuted), **V-2** (LOW, message wording),
**V-3** (LOW, probe methodology -- an editable install can silently substitute
a different checkout), plus the two open items round 3 carries forward,
**R3-B** and **R3-C**, both REPRODUCED here on my own devices, with R3-B shown
to bite at PHYSICALLY realistic feature widths.  **V-4 is the only defect that
changes a decision the library makes, it is present in round 2 as well as
round 3, and it does not block 5.45.0** -- round 3 neither creates it nor
worsens it, and the recommendation (S9) is to ship round 3 and open V-4 as the
next round's subject.

---

## S1. The two builds, and the arms

| | Windows | WSL (Ubuntu) |
|---|---|---|
| python | 3.14.6 | 3.12.3 |
| numpy | 2.4.4 | 2.4.6 |
| BLAS | scipy-openblas, Haswell kernel | scipy-openblas, SkylakeX kernel |
| `lumenairy` resolved to | `C:\tmp\lum_vsliver3\lumenairy\...` | `/mnt/c/tmp/lum_vsliver3/lumenairy/...` |
| threads | `OMP/OPENBLAS/MKL_NUM_THREADS = 1` on the command line | same |

The resolved `lumenairy` path is recorded in EVERY probe's JSON and asserted
at import time (`v_fixtures.assert_tree`).  That is not decoration -- see
defect **V-3**.

**AFTER** is `4a6cf01` (wave2 tip, carrying round 3).  **BEFORE** is
`f2371e0`, round 3's branch point, in a detached worktree, with the probe
directory copied in so that its own `__file__`-derived root pins the pre-change
library.

---

## S2. Bit-identity and the flip census (task 1)

### S2.1 The fixture set

66 solves over five families, none of them the fix's geometry:

| family | fixtures | what it covers |
|---|---|---|
| **A -- no manufactured cell** | 12 | 2-, 3- and 4-layer vertical staircases, a uniform spacer, a grazing dense-superstrate mount, degree 14, an IN-PLANE rotated uniaxial director, an OUT-OF-PLANE director, a lossy out-of-plane director, and the exact (`delta = 0`) form of every resonant device below |
| **B -- the per-layer grid route** | 4 | `layer_grids='per-layer'` on a 6-layer stack (more than `2*window_halfwidth+1`, so the route really differs), at `delta` = 0 / 1e-3 / 1e-5 / 3e-6 |
| **C -- an OWNED liner** | 8 | a liner one layer owns at 1e-4 and 1e-6 of a period, each at four wall steps |
| **D -- the sliver families** | 42 | seven devices (box staircase, guided-mode grating at degrees 6 and 8, Fabry-Perot cavity, near-Wood mount, in-plane director, out-of-plane director) at six wall steps spanning above and below the trigger |

### S2.2 Result

| | Windows | WSL |
|---|---|---|
| fixtures | 66 | 66 |
| unguarded solves identical BEFORE / AFTER | **66 / 66** | 66 / 66 |
| returned by BOTH trees | 42 | 42 |
| of those, RETURNED answer bit-identical (`R`, `T`, Jones SHA-256) | **42 / 42**, 0 broken | 42 / 42 |
| warning sets identical after excising the `truncation` note | **42 / 42** | 42 / 42 |
| `truncation` notes added / removed among rows returned by both | **0 / 0** | 0 / 0 |
| refusals BEFORE / AFTER | 13 / 24 | 13 / 24 |
| refused by BOTH (no refusal lost) | **13** | 13 |
| refusal message reworded (the ATTRIBUTION paragraph) | 13 / 13 | 13 / 13 |
| verdict flips | **11** | 11 |
| flips `truncation -> sliver` / the other way | **11 / 0** | 11 / 0 |

The claim that the change is a strict WIDENING is therefore confirmed on an
independent fixture set: no answer moved, no warning text moved except the one
sentence round 3 rewrote, and no refusal was lost.

The 13 reworded refusals were also diffed rather than counted: the message
PREFIX -- the geometry paragraph, the `1/w^2` mechanism, the measured `R+T`,
and all four numbered remedies -- is byte-identical before and after on every
one of them.  What changed is the ATTRIBUTION paragraph (which now quotes the
drop, the violation it is a drop from, and the closure actually applied
instead of only the absolute bar) and the trailing document reference, which
gains the round-3 report.  Example, on the per-layer fixture at `delta` = 1e-5:

> BEFORE: *"... max R+T there is 1+2.24e-08 **(bar 1e-05)** -- while the
> answer MOVES 3.01 ..."*
>
> AFTER: *"... max R+T there is 1+2.24e-08, **i.e. a 5.265e+08x drop from the
> 1+11.8 this solve reads (bar: the snap must remove 100x of the violation, or
> reach 1e-05 outright, whichever is the weaker demand -- here 0.118)** --
> while the answer MOVES 3.01 ..."*

### S2.3 The eleven flips, classified independently

Each flip is scored against TWO references this probe measured itself: the
exact `delta -> 0` solve of the same device (`err/delta`, the continuity rule)
and the solve on the PRESCRIBED grid.

| fixture | `delta` | `R+T` | `su_snapped` | DROP | `move / w_wide` | `err/delta` RETURNED | `err/delta` SNAPPED |
|---|---|---|---|---|---|---|---|
| box staircase | 1.0e-05 | 15.109 | 1.211e-02 | 1,164.7 | 7.88e+05 | **788,110** (wrong) | 1.596 (right) |
| box staircase | 5.0e-06 | 2.6854 | 1.211e-02 | 139.2 | 1.50e+05 | **150,426** (wrong) | 1.596 (right) |
| box staircase | 3.0e-06 | 14.198 | 1.211e-02 | 1,089.9 | 2.23e+06 | **2,045,420** (wrong) | 1.596 (right) |
| Fabry-Perot | 3.0e-06 | 3.0542 | 8.269e-05 | 24,842 | 3.22e+05 | **141,941** (wrong) | 0.761 (right) |
| Fabry-Perot | 1.0e-06 | 10.842 | 8.269e-05 | 119,031 | 2.64e+06 | **2,644,000** (wrong) | 0.761 (right) |
| GMR, degree 6 | 5.0e-06 | 1.3857 | 7.625e-05 | 5,058.8 | 5.59e+04 | **52,048** (wrong) | 0.142 (right) |
| GMR, degree 6 | 3.0e-06 | 1.3101 | 7.625e-05 | 4,066.2 | 9.64e+04 | **96,417** (wrong) | 0.142 (right) |
| GMR, degree 6 | 1.0e-06 | 3.6195 | 7.625e-05 | 34,353 | 1.53e+06 | **495,692** (wrong) | 0.142 (right) |
| near-Wood | 5.0e-06 | 444.83 | 1.549e-04 | 2.87e+06 | 2.89e+07 | **15,607,700** (wrong) | 1.021 (right) |
| near-Wood | 3.0e-06 | 444.19 | 1.549e-04 | 2.86e+06 | 4.82e+07 | **26,121,000** (wrong) | 1.021 (right) |
| near-Wood | 1.0e-06 | 477.11 | 1.549e-04 | 3.07e+06 | 1.57e+08 | **75,972,600** (wrong) | 1.021 (right) |

All eleven are unambiguous on BOTH rules: the returned answer is off by
5.2e+04 to 7.6e+07 times the physical wall shift, and the prescribed
`min_feature` puts every one of them back on the sliver-free reference to
within 1.6x of that shift.  Round 2 returned all eleven; round 3 refuses all
eleven.  The mildest flip on my devices is 100x more severe than the mildest
on the fix's.

Three of the eleven are worth naming separately.  The box-staircase rows have
`su_snapped` = **1.211e-02**, i.e. the mount's sliver-free truncation floor is
above `_STACK_SUPERUNITY_BAR` itself -- a full decade above the trigger.  The
relative closure there is `max(1e-5, 14.1 * 1e-2)` = 0.141, so the criterion
is met with a snapped residue that still warns on its own.  That is the
intended behaviour of a relative bar, and the continuity rule confirms the
decision, but it is the regime in which the closure is most permissive and it
is worth knowing that it is reachable on an ordinary lossy staircase.

### S2.4 The per-layer route

`layer_grids='per-layer'` does NOT put a stack outside the guard: the screen
builds a union grid regardless of the route, and on a 6-layer stack the route's
own 3-layer windows are unions too.  Measured: at `delta` = 1e-5 and 3e-6 the
per-layer answer is WRONG by the continuity rule (`R+T` = 12.82 and 19.28) and
is REFUSED on both trees; at `delta` = 1e-3 it is GREY and returned.  The
refusal is therefore correct on that route rather than incidental, which is
the opposite of what the refusal message's remedy (3) might suggest for a long
stack, and is worth stating because no previous round measured it.

---

## S3. The closure constant, re-sized on an independent box (task 2)

### S3.1 The box

`v2_closure.py`.  **576 mounts** = 3 periods (0.68 / 1.02 / 1.35 um) x 2
wavelengths (0.633 / 1.064 um) x 2 superstrate indices (2.05 / 3.10) x 3 lossy
substrates (1.52+0.03i / 2.90+1.10i / 3.45+0.90i) x 2 grazing angles (1.18 /
1.35 rad) x 2 degrees (6 / 8) x 2 ridge permittivities (6.76 / 12.25) x 2
slice counts (2 / 3), each at **4 wall steps** (3e-3 / 1e-3 / 3e-4 / 1e-4 of a
period) = **2,304 rows**.  0 mounts were skipped.  Every axis value, the wall
pair (0.3120 / 0.6790), the ridge-height permittivity and the slice thickness
differ from the fix's box.

Per row: the unguarded solve, the unguarded solve on the prescribed grid, both
arbiter quantities, the drop factor, the continuity classification of the
returned AND the snapped answer against the exact `delta -> 0` solve of the
same mount, and a full GUARDED solve so the library's own decision is recorded
rather than inferred.

### S3.2 The populations

| | Windows | WSL |
|---|---|---|
| rows | 2,304 | 2,304 |
| RIGHT / GREY / WRONG | 2,144 / 130 / 30 | identical |
| arbitrated | **859** | 859 |
| arbitrated RIGHT / GREY / WRONG | 737 / 92 / 30 | identical |
| arbitrated CORRECT rows with a FINITE drop | 722 | 722 |
| their drop envelope | **18.090004402653637** | 18.090003927660465 |
| their drop floor | 0.637872957065239 | 0.6378729569846088 |
| arbitrated CORRECT rows with an INFINITE drop | **15** | 15 |
| of those, admitted by the ROUND-2 ABSOLUTE closure | **15 -- all of them** | 15 |
| the correct population's `move / w_wide` envelope | **161.0728624620898** | 161.07286898797727 |
| the same among the rows the closure admits | 13.75898518109434 | 13.758985181088494 |

The 100x drop the shipped fraction demands therefore carries **5.53x** over
this box's finite envelope, against the **2.73x** the fix reports over its
own.  The fix's number is the binding one; this box does not refute it.

### S3.3 The ladder, and the sizing rule

The fix's sizing rule is "admit no CORRECT row the round-2 closure did not
already admit".  Scored on this box:

| `_SLIVER_CLOSURE_FRACTION` | drop demanded | CORRECT rows the CLOSURE admits | NEW vs the absolute bar | CORRECT rows the CONJUNCTION attributes |
|---|---|---|---|---|
| (round 2, absolute 1e-5) | -- | 15 | -- | **0** |
| 1e-1 | 10 | **18** | **+3** | **0** |
| 3e-2 | 33.3 | **15** | **0** | **0** |
| **1e-2 (SHIPPED)** | **100** | **15** | **0** | **0** |
| 3e-3 | 333 | 15 | 0 | 0 |
| 1e-3 | 1,000 | 15 | 0 | 0 |

On this box the boundary sits one step COARSER than the shipped value: 3e-2
adds no correct row here, where on the fix's box it adds one (at drop
36.611).  Taking the two boxes together, **1e-2 is the coarsest fraction that
adds no correct row on EITHER**, which is exactly the rule the fix states.
The three rows 1e-1 newly admits here have drops **12.319 / 16.447 / 18.090**,
all on one mount family (period 1.02 um, `wl` 0.633 um, `n_sup` 3.10, `n_sub`
3.45+0.90i, theta 1.35 rad, degree 6, two slices) whose sliver-free floor is
grid-sensitive.

### S3.4 The conjunction, and how close a false refusal came

The library refused **30** of 2,304 rows.  Every one is WRONG by continuity;
**0** are CORRECT and **0** are GREY.  No WRONG row was returned.  The
analytic round-3 verdict, scored from the recorded `(worst, su_snapped, move,
w_wide)`, equals the library's decision on **859 / 859** arbitrated rows.

The interesting statistic is not either envelope but the JOINT one.  A false
refusal needs `drop >= 100` AND `move > 100 w_wide` on a CORRECT row, so the
distance to one is `min(drop / 100, move_w / 100)`:

| | value | which arm binds |
|---|---|---|
| closest approach over the 737 arbitrated CORRECT rows | **0.1376** | MOVE (the row has an INFINITE drop and `move / w_wide` = 13.759) |
| closest approach over the 92 arbitrated GREY rows | 0.2106 | MOVE (infinite drop, `move / w_wide` = 21.059) |

so the conjunction carries **7.27x** on the correct population and **4.75x**
on the grey one.  That is the two-sided margin this criterion has ON THIS
LADDER, and it is a better statement than either arm's own envelope, because
on this box **neither arm separates on its own**: the closure admits 15
correct rows, and the move criterion admits at least one (S5).

### S3.5 The DIRECTED attack -- and three FALSE REFUSALS

A four-point wall-step ladder over 576 mounts is a coarse instrument.
`v8_attack.py` takes the six mounts that came closest to a false refusal on
that ladder -- the three whose CORRECT rows move furthest past the move bar
and the three whose CORRECT rows drop furthest -- and sweeps the wall step
finely (30 log-spaced values from 6.3e-3 to 2.5e-6) at their own degree and at
two degrees lower, which is 313 rows and 261 arbitrated ones:

| | Windows | WSL |
|---|---|---|
| rows / arbitrated | 313 / 261 | 313 / 261 |
| arbitrated RIGHT / GREY / WRONG | 160 / 50 / 51 | identical |
| CORRECT rows with `move / w_wide` > 100 | **40** | 40 |
| CORRECT rows with an INFINITE drop | **24** | 24 |
| **CORRECT rows meeting BOTH arms -- FALSE REFUSALS** | **3** | **3** |
| the largest joint approach among CORRECT rows | **2.5646** (i.e. past 1.0) | 2.5646 |

Which route the attack succeeded through matters.  The brief's construction --
a resonant device where the snap moves a correct answer a lot AND removes a
large FINITE truncation super-unity -- **did not succeed**: over the 160
arbitrated correct rows of this sweep the largest FINITE drop is **19.0007**
(against the 100 demanded, 5.3x of margin), and over the ladder box's 722 it
is 18.0900; the resonant and many-slice families of S5.3 close it a third
time, with a correct finite-drop envelope of **1.8249** over 403 arbitrated
rows and **0** false refusals.  The finite-drop route is closed on all three
samples.  What opened is
the route the fix's own S3.2 names and then treats as harmless: a correct row
whose snapped solve leaves the super-unity regime entirely, so `su_snapped` is
exactly 0, the DROP is undefined, and the closure is satisfied by ANY
constant -- round 2's absolute 1e-5 included.  **24** of the 160 correct rows
take that route, **40** are past the move bar, and **3** are both.

All three sit on one mount at **degree 4** -- period 1.02 um, `wl` 0.633 um,
`n_sup` 3.10, `n_sub` 2.90+1.10i, theta 1.35 rad, ridge eps 12.25, three
slices -- at wall steps 1.662e-05, 1.269e-05 and 7.395e-06.  They are taken
apart in defect **V-4**.  The important structural point for this section is
that the ladder box's 7.27x is a property of the LADDER, not of the criterion:
sweeping the same mounts finely walks straight through it.

---

## S4. The D-5 family on independent mounts (task 3)

### S4.1 The band is wide

`v3_d5.py` screens 24 (mount, degree) pairs by measuring each mount's
sliver-FREE super-unity directly.  **Nine** land in the D-5 band
(1e-5 .. 1e-3), across five unrelated mechanisms:

| mount | mechanism | sliver-free floor at that degree |
|---|---|---|
| `gmr_deg6` | guided-mode grating, dense superstrate, grazing | **7.6253e-05** |
| `gmr_duty38_deg8` | the same at duty 0.38, `wl` 0.83 um | 1.0645e-05 |
| `graze_deg6` | dense-superstrate grazing staircase | **4.5171e-04** |
| `graze_deg8` | the same, degree 8 | 4.1959e-04 |
| `graze_deg10` | the same, degree 10 | 7.4941e-05 |
| `fp_deg8` | Fabry-Perot cavity between two corrugated mirrors | **8.2684e-05** |
| `fp_thick_deg8` | the same with a 1.46 um cavity | 1.2527e-04 |
| `wood_deg8` | near-Wood mount, -2 order at its Rayleigh cutoff | **1.5485e-04** |
| `wood_m3_deg8` | near-Wood, -3 order, period 2.05 um | **9.9188e-04** |

Identical to 9 significant figures on WSL.  The D-5 class is therefore not a
corner of one guided-mode grating: any mount whose degree is one or two steps
short of convergence lands in it, which is the ordinary state of a production
sweep.

### S4.2 The population and the recovery rate

30 log-spaced wall steps (3.16e-3 down to 2.5e-7) per mount, 270 rows, 112
arbitrated.  The **D-5 population** is the set of arbitrated rows the ABSOLUTE
bar rejects whose answer is WRONG and whose move is past the move bar -- every
piece of evidence an attribution needs except the shape of the closure:

| | Windows | WSL |
|---|---|---|
| D-5 rows | **102** | 102 |
| mounts contributing | 9 | 9 |
| drop floor | **3.6687581141756778** | 3.6687581594883256 |
| drop ceiling | 4,287,527.704 | 4,287,527.701 |
| `err/delta` range | 212.76 .. 4.2462e+08 | identical to 8 figures |
| rows whose SNAPPED answer is RIGHT | **102 / 102** | 102 / 102 |
| their `move / w_wide` floor / ceiling | 212.58 / 6.6247e+08 | identical to 8 figures |
| the `move / w_wide` floor among the 100 the shipped 1e-2 RECOVERS | **1,239.2** | 1,239.2 |
| recovered at 1e-1 / 3e-2 / **1e-2** / 3e-3 / 1e-3 | 100 / 100 / **100** / 88 / 77 | identical |
| left RETURNED at the shipped 1e-2 | **2** | 2 |
| worst `err/delta` left returned at 1e-2 | **741.26** | 741.26 |
| refusals on a CORRECT row | **0** | 0 |

The shipped fraction recovers **98.0 %** of this population, against the
96.6 % the fix reports on its five mounts.  The recovery claim is confirmed at
the same rate on four additional mechanisms.  Note that the two rows 1e-2
leaves returned are not recovered at 3e-2 or 1e-1 either -- their drops are
5.205 and 3.669, below even the 10x a fraction of 1e-1 would demand -- so
loosening the constant does not reach them.  That is R3-A restated as a
measurement rather than an argument.

### S4.3 The drop floor is 13.4x below the published one -- and that is the point

The fix publishes the D-5 population's drop floor as **49.107** and describes
the two populations as "ADJACENT ... 1.34x apart".  Measured over nine mounts
the floor is **3.669**, which is not adjacent to the correct population but
INSIDE it (that population runs 0.638 .. 18.090 on the same box).

This does not change a decision, and it is not a surprise: the fix's own S3.5
derives the bound `drop > trigger / (the mount's own truncation floor)` and
says in as many words that a degree-6 mount whose floor is 2.08e-04 "cannot
produce one below drop ~4.8, which is INSIDE the correct population's own
range".  The published 49.107 is a property of its five mounts; the FAMILY
reaches 3.669 because `wood_m3_deg8`'s floor (9.9188e-04) sits at 1.008x below
the trigger.  Recorded as defect **V-1**: the prose is right and the number is
sample-scoped.

### S4.4 The R3-A edge, measured

Two rows have a drop below the demanded 100, are WRONG, and are RETURNED:

| mount | `delta` | `R+T` | `su_snapped` | DROP | `move / w_wide` | `err/delta` RETURNED | `err/delta` SNAPPED | warnings |
|---|---|---|---|---|---|---|---|---|
| `graze_deg6` | 2.4528e-06 | 1.0023512 | 4.5172e-04 | **5.205** | 741.36 | **741.26** (wrong) | 0.107 (right) | **none** |
| `wood_m3_deg8` | 2.3950e-05 | 1.0036406 | 9.9234e-04 | **3.669** | 212.58 | **212.76** (wrong) | 1.675 (right) | none |

Both meet the MOVE criterion by 2.1x-7.4x and both are fully restored by the
prescribed `min_feature`; only the drop keeps them out, and no setting of
`_SLIVER_CLOSURE_FRACTION` can reach them without descending into the correct
population.  This is R3-A exactly as the fix states it.

The first row is worth emphasising for a different reason: its `R+T` is
1.00235, which is above the trigger but BELOW `_STACK_SUPERUNITY_BAR`, so the
plain super-unity warning does not fire either.  A wrong answer off by 741x
the physical wall shift is returned **silently**.  That is R2-A's sub-trigger
band and R3-A's sub-drop band meeting, and it is the strongest argument in
this campaign for a detector that does not go through `R+T`.

### S4.5 The `truncation` note's truth

The note asserts the sliver "is NOT what moved this answer".  It is FALSE on
any RETURNED row whose answer is WRONG by continuity while the prescribed-grid
answer is RIGHT -- which is precisely the D-5 shape.

The D-5 mounts produce no returned noted rows (the note rides on the plain
super-unity warning, so it needs `R+T - 1` above 1e-2, and those rows are
either refused or below that bar), so the property was scored on the
2,304-row box, which produces 308 of them:

| | Windows | WSL |
|---|---|---|
| RETURNED rows carrying the note | **308** | 308 |
| of those, WRONG returned AND RIGHT snapped -- a FALSE note | **0** | 0 |
| RIGHT / GREY / WRONG as returned | 261 / 47 / 0 | identical |
| snapped answer RIGHT | 308 / 308 | 308 |

**0 false notes.**  On the 47 GREY rows the note's quoted evidence is exact
(drop **0.800 .. 4.475**, `move / w_wide` **4.46 .. 120.40** against the stated
100x bar), but the snapped answer is measurably closer to the `delta -> 0`
limit -- those rows read `err/delta` **10.05 .. 71.67** as returned and
**0.582 .. 9.370** snapped -- so the headline sentence "is NOT what moved this
answer" overstates on rows the campaign calls grey.  Recorded as defect
**V-2** (wording, LOW).

---

## S5. The MOVE bar (task 4)

### S5.1 What the bar is competing against

The prescribed snap displaces each colliding wall by at most half the widest
manufactured cell, so for an answer that is still tracking its geometry the
move is about `s * w_wide`, where `s` is the DEVICE's own slope `dR/dx` in
per-order efficiency per unit wall fraction.  On a two-slice staircase
`move / w_wide` is about `dR/d(duty)`; on a three-slice one it is about twice
that, because the outer slices are displaced by a full `w_wide` rather than
half.  A device whose slope exceeds 100 therefore puts a CORRECT row past the
bar with no numerical pathology at all.  That is open item R2-D, whose
mechanism the round-2 verification confirmed and whose published bound it
refuted.

### S5.2 The correct population crosses the bar

On the 2,304-row box the arbitrated CORRECT population's `move / w_wide`
envelope is **161.0729** (Windows) / **161.0729** (WSL) -- the 100x bar is not
1.27x away, it is CROSSED.  The row is:

| period | `wl` | `n_sup` | `n_sub` | theta | degree | ridge eps | slices | `delta` |
|---|---|---|---|---|---|---|---|---|
| 1.35 um | 1.064 um | 2.05 | 1.52+0.03i | 1.18 rad | 6 | 6.76 | 3 | 1e-4 |

reading `R+T` = **1.055390**, `err/delta` = **3.087** (RIGHT), `su_snapped` =
**6.3303e-02**, DROP = **0.87499**, `move / w_wide` = **161.073**.  It is
RETURNED, and bit-identical to the unguarded answer, because the CLOSURE arm
reads 0.875 against the 100 an attribution demands -- 114x short.  The
GREY population reaches 120.401 on the same box, also over the bar, also held
out by the closure.

So the fix's sentence "the MOVE criterion holds every one out" is true of its
own box and false of mine, and its "1.27x -- the bar to watch" is not a margin
at all on an independent sample.  Recorded as defect **V-1**.

### S5.3 The resonant counter-fixture

`v4_move.py` builds the counter-fixture rather than assuming one.  It scans
four high-contrast-grating seeds -- a single high-index bar layer in air,
which supports leaky Fano resonances -- coarsely in wavelength to find the
interval where the zeroth-order reflectance swings most, finely inside it to
find the resonance, and then in DUTY at that wavelength to read the device's
own `dR/d(duty)`.  Every slope is re-measured one degree up, so a slope that
is a discretisation artefact is visible.  1,284 rows, 403 arbitrated.

**The steepest DEVICE this campaign could build:**

| mount | resonance located at | `dR/d(duty)`, degree 12 | degree 14 | stationary |
|---|---|---|---|---|
| HCG, duty 0.7203, `t_gr` 350 nm | (probe-located) | **89.30961** | 89.31008 | **yes** (7 figures) |
| HCG, duty 0.6506, `t_gr` 350 nm | -- | 57.52020 | 57.52002 | yes |
| HCG, duty 0.6297, `t_gr` 450 nm | -- | 51.03045 | 51.03073 | yes |
| HCG, duty 0.5901, `t_gr` 350 nm | -- | 41.34508 | 41.34504 | yes |
| near-Wood, `inside` 4e-3 | -- | 5.34087 | 5.36190 | yes |

Windows and WSL agree to 9 significant figures on all five.  **89.31** is a
real, degree-stationary device slope, and it sits just BELOW the 100 bar -- so
a two-layer stack on this resonance does not, by itself, put a correct row
past the move criterion.  That is the honest answer to "can a resonance alone
break the move bar": on this family, not quite.

**What DOES cross the bar is the many-slice taper, and for a different
reason.**  Of the 56 arbitrated CORRECT rows in this scan, **12** read
`move / w_wide` above 100, and the envelope is **906.5553** (Windows) /
**906.5553** (WSL) -- **9.07x** past the bar.  All twelve are tapered
staircases at degree 6, and the mechanism is not the device's slope at all.
On a four-slice taper at degree 6:

| `delta` | `w_wide` | `move` | `move / w_wide` | `err/delta` | class |
|---|---|---|---|---|---|
| 3.1623e-03 | 1.0541e-03 | 4.4486e-03 | 4.22 | 1.611 | right |
| 1.4678e-03 | 4.8927e-04 | 2.0294e-03 | 4.15 | 1.590 | right |
| 3.1623e-04 | 1.0541e-04 | 3.8519e-03 | 36.54 | 1.573 | right |
| 1.0000e-04 | 3.3333e-05 | 4.2602e-03 | **127.81** | 1.572 | right |
| 3.1623e-05 | 1.0541e-05 | 4.4041e-03 | **417.81** | 1.542 | right |
| 1.4678e-05 | 4.8927e-06 | 4.4355e-03 | **906.56** | 1.380 | right |

Read the third column: **`move` SATURATES** at about 4.3e-03 while `w_wide`
falls by 215x.  Above `delta` ~ 1e-3 the snap really is a perturbation and
`move / w_wide` is flat at ~4.2, which is the device's own slope; below it the
prescribed `min_feature` no longer nudges the walls, it **collapses the
taper** -- every slice's wall set merges into one -- so `move` becomes the
fixed difference between an `nl`-step staircase and a one-step grating, and
dividing that by a `w_wide` heading to zero makes the ratio diverge.

That is a structural statement about the MOVE criterion and it is worth
recording plainly: on a tapered stack `move / w_wide` is **not** a property of
the device, and it will cross any fixed bar at a small enough wall step,
whether or not the answer is wrong.  The answer here is RIGHT at every one of
those rows (`err/delta` 1.38 .. 1.61).

**And yet: 0 false refusals on this family.**  Every one of the 12 rows past
the move bar has a DROP of at most **1.8249** against the 100 the closure
demands -- 54.8x of margin -- so the conjunction holds all of them out.  The
nearest attribution on the family is a GREY row (12-slice taper, degree 10,
`delta` = 3.1623e-04): drop **462.92**, `move / w_wide` **922.34**, returned
`err/delta` **84.31**, snapped `err/delta` **2.081**.  The library refuses it,
and the refusal is defensible -- the prescribed grid moves that answer from
84x the wall shift to 2.1x.

**The synthesis, across S3.5 and S5.3.**  Each arm is defeated on some
family.  The MOVE arm is crossed by CORRECT answers by 1.6x on the ladder box
(161.07) and by 9.1x on the tapered family (906.56).  The CLOSURE arm is
satisfied outright by 15 correct rows on the ladder box and 24 in the directed
sweep, because their snapped solve leaves the super-unity regime.  The
conjunction survives everywhere the two failures do not coincide -- and the
three false refusals of **V-4** are precisely where they do.

### S5.4 Does the move bar need re-deriving before 5.45.0?  No -- and here is the arithmetic

**Lowering it** (say to 30, to catch the R3-A rows whose move is 212.6 and
741.4) would put it below the correct population's envelope on BOTH boxes
(161.07 here, 79.03 on the fix's), below all 12 correct rows of the tapered
family (127.8 .. 906.6) and below 47 of my 308 grey noted rows, so it would
manufacture false refusals -- the exact defect round 2 exists to remove.  This
direction is closed by measurement.

**Raising it** (say to 300) costs nothing on my own D-5 evidence -- the 100
recovered rows' `move / w_wide` floor is **1,239.2**, i.e. 12.4x above the
bar -- but it is closed by the round-2 family measurement round 3 itself
restates in its S7.2: the WRONG population's `move / w_wide` floor over a
2,250-row grid on five fixtures is **147.411** (polarization 1 alone 134.171).
A bar at 300 would return wrong answers that 100 refuses.  It would also push
the two R3-A rows (212.6 and 741.4) further out of reach.

**Neither direction is the real question**, because on a tapered stack no
fixed value of this constant is safe: S5.3 measures `move / w_wide` climbing
from 4.22 to **906.56** on rows that stay RIGHT, purely because `w_wide` is in
the denominator and the snap has stopped being a perturbation.  Any fixed bar
is crossed at a small enough wall step.  What would change that is the
STATISTIC, not its value -- normalising the move by the wall displacement the
snap actually applied (which stops shrinking once the snap collapses a taper),
or refusing to score the move at all when the prescribed `min_feature` merges
more walls than the manufactured cells it was sized on.  Neither is attempted
here; this is a verification.

**Leaving it at 100** is therefore the right call for 5.45.0, because the arm
that actually separates is the CONJUNCTION: 54.8x of margin on the tapered
family (every correct row past the move bar has a drop of at most 1.8249) and
7.27x on the ladder box.  The correct documentation change is to keep the bar,
delete the published "1.27x" margin, and state instead that neither arm
separates alone, that `move / w_wide` diverges on tapered stacks, and that the
conjunction carries 7.27x-54.8x.  That is a documentation change, not a
library change, and this verification did not touch `lumenairy/`.

---

## S6. Both builds (task 5)

Every DECISION this campaign records is identical on the two builds:

| decision surface | rows | differing |
|---|---|---|
| v1 fixture decisions (`refused`, arbiter verdict, `screen`, continuity class of the returned and snapped answers, bit-identity flag) | 66 | **0** |
| v1 BEFORE/AFTER comparison (flip counts, bit-identity counts, warning-set counts, refusal counts) | 20 integer fields | **0** |
| v2 census integer and boolean fields (arbitrated, by-kind, admissions at five fractions, attributions, library refusals, false positives, notes, analytic-vs-library agreement) | 40 fields | **0** |
| v3 D-5 integer fields (in-band mounts, D-5 rows, recoveries at five fractions, R3-A edge rows) | 20 fields | **0** |
| v6 open-item booleans (screen fired, refused, provably passive, probe calls, contract flags) | 34 fields | **0** |
| v4 resonant-counter-fixture integer fields (arbitrated, by-kind, correct rows past each arm, slopes over 100, false refusals) | 10 fields | **0** |
| v8 directed-attack integer fields (arbitrated, by-kind, correct rows past each arm, FALSE REFUSALS) | 9 fields | **0** |
| v9 false-refusal decisions (library refused, round-2 verdict, round-3 verdict, continuity class, "the snap helped") | 3 rows x 7 fields | **0** |

The largest RELATIVE numeric spread on the arbiter's evidence dict is
**0.68**, and it is not what it looks like: it is the `drop` of the round-2
`(degree 16, delta 1e-5)` sliver row, whose `su_snapped` is **5.551e-15**
(Windows) and **1.776e-15** (WSL).  Both are the arithmetic's own noise floor;
the decision compares a drop of 2.1e+14 against a bar of 100, i.e. **12
decades** of headroom.

Excluding quantities whose denominator sits at machine epsilon, the largest
spread is **1.7621e-02**, on the GREY drop envelope of the tapered family
(**2.2706e+10** vs **2.2306e+10**, denominator ~4e-12), followed by
**1.4444e-02** on the `drop` of the out-of-plane-director row at `delta` = 1e-4
(**11,759.95** vs **11,932.31**).  Every one of the large spreads in this
campaign is a DROP whose denominator is a near-zero residue, and every one of
them is four to twelve decades above the 100 the closure compares against, so
none is within reach of a decision.  The largest spread on any quantity that
is not such a ratio is **4.05e-08** (the correct population's `move / w_wide`
envelope).

Population statistics agree to 8-10 significant figures:

| statistic | Windows | WSL | relative spread |
|---|---|---|---|
| correct population's `move / w_wide` envelope | 161.0728624620898 | 161.07286898797727 | 4.05e-08 |
| correct population's FINITE drop envelope | 18.090004402653637 | 18.090003927660465 | 2.63e-08 |
| D-5 population's drop floor | 3.6687581141756778 | 3.6687581594883256 | 1.24e-08 |
| the mildest flip's `err/delta` | 52,048.140453787935 | 52,048.1404289752 | 4.77e-10 |
| the D-5 fixture's degree-8 sliver-free floor | 3.729516002382027e-05 | 3.729516002604072e-05 | 5.95e-11 |
| the V-4 false refusal's `move / w_wide` (worst row) | 256.4598679673952 | 256.45986627033346 | 6.62e-09 |
| the V-4 false refusal's `err/delta` (worst row) | 0.9790815652143214 | 0.9790955515581966 | **1.43e-05** |

The last line is the largest spread on any quantity a CLASSIFICATION depends
on anywhere in this campaign, and it still leaves the RIGHT / GREY boundary
(`err <= 10 delta`) a factor of 10.2 away.

**How far the decisions sit from flipping.**  For each of the 859 arbitrated
rows the verdict turns on `su_snapped <= closure` and `move > 100 w_wide`; the
relative distance of the BINDING comparison from equality is what a cross-build
spread would have to exceed to move a verdict.  The smallest such margin over
the 859 rows is **0.789** -- **54.6x** the largest meaningful cross-build
spread.  No row on this box is anywhere near its bar.

The tightest margin anywhere in this campaign is on the V-4 rows, which the
box does not contain: their closure comparison is `0 <= 1e-05` (margin 1.0)
and their move comparison is 114.74 against 100 (margin **0.147**).  Even
that is **10.2x** the largest meaningful cross-build spread, which is why the
two builds refuse the same three rows.

---

## S7. Test durability (task 6)

### S7.1 Constant by constant, `test_fix_pmmstack_sliver_round3.py`

Re-measured with `v5_testbars.py`, which IMPORTS the test module so the
fixtures scored are the test's own.  "Two-sided" means the test states what
the other population reads, not just that its own clears a bar.

| # | assertion | origin | measured, Windows | measured, WSL | margin | scope |
|---|---|---|---|---|---|---|
| 1 | `ladder[0] > ladder[1] > ladder[2] > ladder[3]` | the mount's own degree ladder | 2.0778e-04 > 3.7295e-05 > 9.8316e-06 > 2.9610e-07 | identical to 10 figures | monotone by 5.6x / 3.8x / 33x | family (a convergence property) |
| 2 | `ladder[1] > _SLIVER_ATTRIB_CLOSURE` | the library's own constant | 3.7295e-05 | 3.7295e-05 | **3.73x** | family (defines the D-5 band) |
| 3 | `ladder[1] < _SLIVER_TRIGGER_BAR` | the library's own constant | 3.7295e-05 | 3.7295e-05 | **26.8x** | family |
| 4 | premise guard `err > 100 and err_snapped < 1` (CONTINUE, not fail) | the campaign's continuity rule | `err/delta` 1,811 / 2,399 / 3,501; snapped 0.001931 each | identical to 7 figures | 18x-35x and **518x** | two-sided, sample-scoped but guarded by `continue` |
| 5 | `su > _SLIVER_ATTRIB_CLOSURE` | the library's own constant | 3.7295e-05 | same | 3.73x | family |
| 6 | `drop > 1 / _SLIVER_CLOSURE_FRACTION` | the library's own constant | 5,166.3 / 5,181.2 / **620.92** | 5,166.3 / 5,181.2 / 620.92 | **6.21x** at the tightest | sample (3 rows); the FAMILY floor is 3.669 -- open item R3-A |
| 7 | `move / w_wide > _SLIVER_MOVE_FACTOR` | the library's own constant | 13,665 / 18,251 / **3,500.6** | identical | **35.0x** | family |
| 8 | `ev["closure"] > _SLIVER_ATTRIB_CLOSURE` | the library's own constant | 192.7x / 193.2x / **23.2x** | identical | 23.2x | family |
| 9 | `len(found) >= 2` of 3 | sample size | 3 of 3 met the premise | 3 of 3 | 1.5x | sample, deliberately |
| 10 | `n_rows == 24` | the enumerated box | 24 | 24 | exact | **sample** (a count of the loop) |
| 11 | `n_arb >= 8` | measured arbitrated count | **14** | 14 | **1.75x** | **SAMPLE-scoped** -- see below |
| 12 | every box row `== "right"` | continuity | 24 / 24 RIGHT | 24 / 24 | -- | sample, and the point of the test |
| 13 | `max(corr) < 1 / _SLIVER_CLOSURE_FRACTION` | the library's own constant | 2.2893 (22 rows) | 2.2893 | **43.7x** | sample-scoped; the FAMILY reaches 18.090 (5.53x) |
| 14 | `min(d5) > 1 / _SLIVER_CLOSURE_FRACTION` | the library's own constant | 620.92 | 620.92 | **6.21x** | sample-scoped; the FAMILY reaches 3.669 (0.037x -- see V-1) |
| 15 | `min(d5) > 10 * max(corr)` | separation between the two populations the test measures | 271.23 | 271.23 | **27.1x** | two-sided by construction; sample-scoped in both arms |
| 16 | `max(sliver su_snap) <= _SLIVER_ATTRIB_CLOSURE / 10` | a hand-picked tenth of a library constant | **2.4647e-13** | 2.3981e-13 | **4.06e+06x** | **SAMPLE-scoped**, but padded by 6.6 decades |
| 17 | `min(sliver drop) > 1 / _SLIVER_CLOSURE_FRACTION` | the library's own constant | 4.758e+12 | 4.890e+12 | 4.8e+10x | sample; the residue is machine epsilon |
| 18 | `min(sliver move) > _SLIVER_MOVE_FACTOR` | the library's own constant | 4,789.5 | 4,789.5 | **47.9x** | sample; the FAMILY floor is 147.4 (1.47x, round-2 D-4) |
| 19 | `max(trunc move) < _SLIVER_MOVE_FACTOR` | the library's own constant | 3.4513 | 3.4513 | **29.0x** | sample |
| 20 | note-format assertions (`"{drop:.4g}x drop"`, `"1+{su:.3g}"`, the branch's own bar) | the library's own formatting | present | present | exact | contract, not a bar |
| 21 | `"will silence nothing here" not in source` (comments stripped) | the D-5 defect | absent | absent | exact | contract |

Two entries deserve the label the round-2 verification's D-3 used.  **#11**
(`n_arb >= 8`) is a count over an enumerated 24-row loop and reads 14; it
cannot flake today, but nothing in the test or its docstring makes it a family
property.  **#16** (`su_snap <= 1e-6`) is likewise a property of the five rows
enumerated, but it is padded by 6.6 decades, so it is durable in practice.
Everything else is expressed against a library constant, which is the shape
`docs/TESTING_STANDARDS.md` asks for.

The one durability finding that matters is **#14**: the test asserts the D-5
sample's drop floor is above the bar, and it is (6.21x), but the FAMILY's
floor is 3.669 -- below the bar, which is open item R3-A.  The test is correct
as written (it measures its own population, per rule 5) and its docstring
already points at R3-A; this verification adds the family number.

### S7.2 The runs

| run | Windows | WSL |
|---|---|---|
| the five sliver files + `test_m1_conditioning_guard.py` (92 tests: 19 + 19 + **6** + 15 + 6 + 27) | **92 passed**, 1 warning, 116.09 s | **92 passed**, 1 warning, 113.49 s |
| `tests/unit/test_verify_pmmstack_sliver_round3.py` (5 new) | **5 passed**, 2.67 s idle / **14.97 s** with the box at 100 % CPU and three other agents running | **5 passed**, 15.63 s under that load |
| the 43-file `PMMStack` regression | REGRESSION-RESULT | -- |
| `ruff check lumenairy/ tests/ validation/probe_verify_sliver_round3/` (WSL) | -- | **All checks passed!** |

The one warning in the sliver runs is the pre-existing deliberate
`_pmm_union_grid` snap warning of
`test_return_owners_is_additive_and_warn_false_is_silent`.  The WSL run also
prints the two pre-existing `** On entry to DLASCL parameter number 4 had an
illegal value` lines, which rounds 2 and its verification both record; they
are not new here.

`.test_durations` gains the five new node ids and nothing else (a 5-line
diff, sorted, nothing reformatted).

---

## S8. Defects

### V-4 (MEDIUM, INHERITED from round 2, present in round 3) -- three CORRECT answers are REFUSED, and the remedy the refusal names makes them slightly worse

Both rounds report zero on their own populations, and both state the REASON as
a general property.  Round 3's S3.5: *"the number of correct rows attributed
is 0 at every fraction from 1e-1 to 1e-3, on both builds, because the MOVE
criterion holds every one out"*.  Round 2's verification: *"false positives
110/648 -> 0/648"*.  The counts are not disputed here -- my own 2,304-row
ladder box reproduces them (0 / 737).  What is refuted is the reason, and with
it the generalisation: the move criterion does NOT hold every correct row out
(S5.2 and S5.3 measure correct rows at 161.07 and 906.56 against the 100 bar),
and once the CLOSURE arm goes vacuous there is nothing left to hold them out.
Both published populations are ladders of three or four wall steps per mount
at degrees 6-10; the rows below need degree 4 and a finer step.

**The rows.**  One mount, `period` 1.02 um, `wl` 0.633 um, `n_sup` 3.10,
`n_sub` 2.90+1.10i, theta 1.35 rad, ridge eps 12.25, three z-slices, at
**degree 4**:

| `delta` | `R+T` | `su_snapped` | DROP | `move / w_wide` | `err/delta` | class | library |
|---|---|---|---|---|---|---|---|
| 1.6622e-05 | 1.0030417 | **0.0** | inf | **114.74** | **1.0505** | RIGHT | **REFUSED** |
| 1.2690e-05 | 1.0030434 | **0.0** | inf | **149.99** | **1.0649** | RIGHT | **REFUSED** |
| 7.3955e-06 | 1.0030454 | **0.0** | inf | **256.46** | **0.9791** | RIGHT | **REFUSED** |

`err/delta` is the campaign's own continuity statistic against the exact
`delta -> 0` solve of the same mount at the same degree: **the answer tracks
the physical wall shift to about 1x**, which is as correct as this rule can
call anything.  The library refuses all three.

**Why both arms are met with nothing wrong.**  The two conjuncts fail for two
independent and entirely benign reasons:

* the mount's SLIVER-FREE solve reads `R+T` = **0.9926**, i.e. BELOW unity, so
  the solve on the prescribed grid reads `su_snapped` = **exactly 0** and the
  closure is satisfied outright.  It is satisfied by the ROUND-2 ABSOLUTE bar
  in the same way -- `0 <= 1e-5` -- so the relative closure has nothing to do
  with it;
* the MOVE arm is met by the same divergence S5.3 measures on the tapered
  family, not by a large device slope.  The absolute move is CONSTANT across
  the three rows -- **9.5362e-04 / 9.5167e-04 / 9.4832e-04**, a 0.6 % spread --
  while `w_wide` falls 2.25x from 8.311e-06 to 3.698e-06, so `move / w_wide`
  climbs 114.74 -> 149.99 -> 256.46 purely through the denominator.  The
  prescribed `min_feature` merges this three-slice stack's three wall sets into
  one, which is a fixed geometry change, not a perturbation that shrinks with
  `delta`.  So the arm that is supposed to certify "the answer moved far past
  the geometric perturbation" is instead measuring a constant against a
  vanishing yardstick.

**It is INHERITED, not a round-3 regression.**  Verified three ways: the
analytic round-2 criterion attributes all three (`round2_would_attribute` = 3);
the analytic round-3 criterion attributes the same three; and re-running the
same probe on the round-3 branch point `f2371e0` -- round 2 plus its
verification, with no relative closure in the tree -- the library refuses all
three there as well.  Windows and WSL agree on every digit to 8 significant
figures.

**The refusal's ATTRIBUTION is wrong, and its first remedy is
counter-productive.**  The message says *"the super-unity GOES WITH IT ... So
the SLIVER moved this answer, not degree / n_slices"*.  Measured against a
degree-16 solve of the same sliver-free device:

| | distance to the degree-16 solve |
|---|---|
| the RETURNED answer | 0.0592457 / 0.0592497 / 0.0592551 |
| the answer on the PRESCRIBED grid | 0.0592724 / 0.0592700 / 0.0592667 |
| the SLIVER-FREE answer at the same degree | 0.0592621 |

The three numbers agree to four significant figures, because the error is the
mount's own degree-4 truncation and is common to all three solves -- and the
prescribed `min_feature`, which the refusal names as remedy (1), leaves the
answer **farther** from the degree-16 solve than it started, on all three rows
(`snapped_closer_to_truth` = False, 3 / 3, both builds).  So the caller is
refused an answer that was correct in the only sense the campaign defines, and
is pointed at a grid change that does not help.

**Severity MEDIUM, not HIGH**: the class needs a mount whose sliver-free solve
is SUB-unity (so the closure is free), more than two z-slices and a small
enough wall step that the snap collapses them (so the move bar is free), and a
degree low enough that the row is above the trigger at all; it does not appear at all in
2,304 rows of a four-step ladder at degrees 6-8, and it is not reachable
through round 3's change -- round 2 refuses the same rows.

**What would close it.**  Both arms degenerate on this row, and each has its
own repair.

*The closure* is vacuous whenever the snapped solve leaves the super-unity
regime; the fix's own S3.2 says so ("a correct answer whose truncation
super-unity happens to fall BELOW unity on the snapped grid satisfies ANY
closure").  The DROP factor is undefined there and the criterion silently
degenerates to the move arm alone.  A repair is to require a MEANINGFUL
reduction rather than any reduction -- e.g. compare the violation with the
mount's own sliver-free floor, which one more solve with the walls made
exactly coincident measures directly, and attribute only when the violation is
large against THAT rather than against a residue that happens to be zero.

*The move* is measured in units of `w_wide`, which vanishes with the wall step
while the prescribed snap's actual geometry change does not (it collapses a
multi-slice wall set into one).  A repair is to normalise by the displacement
the snap really applied -- readable from the union grid before and after --
or to decline to score the move at all when `min_feature` merges more walls
than the manufactured cells it was sized on.

Neither is attempted here: this is a verification, and no `lumenairy/` file
was touched.

**Reproducer**: `validation/probe_verify_sliver_round3/v8_attack.py` (the
search) and `v9_falserefusal.py` (the analysis, with the BEFORE arm);
`tests/unit/test_verify_pmmstack_sliver_round3.py::test_a_correct_answer_is_refused_when_the_snap_leaves_the_superunity_regime`
pins it in the shape round 2's verification used -- repairing it makes that
test fail, which is the gate working.

### V-1 (LOW, doc-only) -- three published population bounds are sample-scoped, and one of them is refuted

| published | measured here | on |
|---|---|---|
| the CORRECT population's FINITE drop envelope is **36.611**, so 1e-2 carries 2.73x | **18.090** on an independent 576-mount box -- 5.53x.  Not a refutation; the fix's number stays binding | 722 finite-drop correct rows |
| the D-5 population's drop floor is **49.107**, and the two populations are "1.34x apart" | **3.669** over nine mounts and five mechanisms -- the two populations OVERLAP, and the fix's own S3.5 predicts exactly this | 102 D-5 rows |
| the CORRECT population's `move / w_wide` envelope is **79.032** against the 100 bar, "1.27x -- the bar to watch" | **161.073** -- the bar is INSIDE the correct population.  The published margin does not exist on an independent box | 737 arbitrated correct rows |

None changes a decision the library makes, because the CONJUNCTION is what
separates and its closest approach to a false refusal is 0.1376 (7.27x).  All
three make the guard's margins look like properties of a sample.  The repair
is to restate them in
`docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND3_2026_09_11.md` the way round 3
restated round 2's in its own S0.1 corrections table.

**Reproducer**: `validation/probe_verify_sliver_round3/v2_closure.py` and
`v3_d5.py`, both builds' JSON committed.

### V-2 (LOW, message wording) -- the `truncation` note's headline overstates on GREY rows

The note begins *"A near-coincident-wall SLIVER ... IS present on the union
grid but is NOT what moved this answer"*.  Over 308 returned noted rows of the
independent box that sentence is never FALSE by the campaign's rule (0 rows
are WRONG returned with a RIGHT snapped answer), but on **47** of them the
returned answer is GREY (`err/delta` **10.05 .. 71.67**) while the prescribed
grid's answer is RIGHT or nearly so (`err/delta` **0.582 .. 9.370**), i.e. the
sliver did move the answer, by 4.5 to 120 widest cells.  The quantitative half
of the note is exact on those rows (it quotes the 0.800x-4.475x drop and the
4.46x-120.40x move against the stated 100x bars); it is the qualitative half
that reads as a stronger claim than was measured.  The 261 RIGHT noted rows
read `err/delta` 0.462 .. 9.908, where the sentence is unobjectionable.

Severity LOW: the caller is warned, the numbers quoted are right, and the
remedy named ("reduce n_slices or raise degree") is still the correct one for
a grey row whose super-unity survives the snap.  A wording repair would be to
say the sliver is not the DOMINANT cause, or to name the measured move
directly in the headline.

**Reproducer**: `v2_closure.py`, rows with `lib_note` true and `kind` grey;
`tests/unit/test_verify_pmmstack_sliver_round3.py::test_the_truncation_note_is_never_false_on_a_returned_row`
pins the decision half.

### V-3 (LOW, probe methodology, NOT a library defect) -- an editable install can silently substitute a different checkout

This box carries an EDITABLE install of `lumenairy` (a `sys.meta_path` finder
from `__editable__.lumenairy-3.7.8.pth`) pointing at
`D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\Lumenairy`, which
is a DIFFERENT checkout (at `9af9376`, with no sliver guard at all).  `python
<probe>.py` puts the PROBE directory on `sys.path[0]` and does NOT put the
working directory there, so a probe launched as a plain script from a worktree
root imports the OTHER checkout -- silently, with no error.  Measured: from
`validation/probe_fix_sliver_round3/`, `import lumenairy` resolves to the D:
tree.

The fix's committed JSON records `"lumenairy": "C:\\tmp\\lum_sliver3\\..."`, so
its runs were launched in a way that avoided this (`-m`, or `PYTHONPATH`), and
no published number is affected.  But the probes do not PIN the tree and do not
fail if it is wrong.  Every probe in
`validation/probe_verify_sliver_round3/` pins the tree from `__file__` and
calls `v_fixtures.assert_tree()`, which refuses to run against a module
outside its own worktree, and records the resolved path in its JSON.

**Recommendation**: adopt the same two lines in future probe fixture modules.

### R3-B (MEDIUM, pre-existing, inherited) -- REPRODUCED, and it bites at PHYSICAL feature widths

`_cross_layer_sliver` computes the own-scale as the GLOBAL minimum wall
spacing over all layers and flags a manufactured cell only when
`own / w >= 100`, so one thin feature a single layer legitimately OWNS
disarms the cross-layer screen for the whole stack.

Reproduced on my own device (`v6_open_items.py`, both builds), first in the
same regime round 2's verification used:

| `delta` | liner | `own / w` | screen | decision | `R+T` | warnings |
|---|---|---|---|---|---|---|
| 3e-6 | none | 103,999 | fires | **REFUSED** | 4.3559 | -- |
| 3e-6 | 1e-6 of a period | -- | **silent** | **RETURNED** | **5.3960** | 2 (super-unity + within-layer) |
| 1e-5 | none | 31,199 | fires | **REFUSED** | 2.7607 | -- |
| 1e-5 | 1e-6 of a period | -- | **silent** | **RETURNED** | **4.1450** | 2 |
| 1e-5 | 1e-7 of a period | -- | silent | RETURNED | 2.2719 | 2 |
| 3e-5 | none | 10,399 | fires | **REFUSED** | 1.4139 | -- |
| 3e-5 | 1e-6 of a period | -- | **silent** | **RETURNED** | **2.7617** | 2 |

and then in a PHYSICAL one the round-2 reproducer does not reach.  A **2 nm**
liner on a 1.35 um period is `own` = 1.4815e-03, so the disarm window
`1 <= own / w < 100` is wall mismatches of **15 pm to 2 nm** -- the range a
coated device's wall coordinates actually differ by.  Scanned across that
window:

| `delta` | `own / w` | screen | decision | `R+T` | `err/delta` |
|---|---|---|---|---|---|
| 1.413e-05 | 104.9 | fires | REFUSED | 2.9210 | 4,955 (wrong) |
| **2.720e-05** | **54.5** | **silent** | **RETURNED** | **2.9206** | **2,573 (wrong)** |
| 3.775e-05 | 39.2 | silent | returned | 1.0000 | 1.196 (right) |
| 1.000e-04 .. 1.000e-03 | 14.8 .. 1.5 | silent | returned | 1.0000 | 1.09 .. 1.20 (right) |

The 2.720e-05 row is the defect at a width a real device produces: a 2 nm
liner plus a 37 pm wall mismatch, answer wrong by 2,573x the wall shift,
`R+T` = 2.92 on a provably passive stack, RETURNED with exactly **one**
warning -- the generic super-unity warning -- and no sliver diagnosis of any
kind.  Immediately below the window (1.413e-05, `own / w` = 104.9) the same
stack IS refused, so the guard's behaviour is non-monotone in `delta`.

This is where the physical arm is worse than the pico one the round-2
verification used.  At a 1e-6-of-a-period liner the WITHIN-LAYER arm fires and
the caller at least learns that a sliver-thin owned feature is injecting
spurious wavenumbers (2 warnings on every returned row of the pico table
above), which is the mitigation the fix's R3-B entry cites.  At a **2 nm**
liner the owned feature is far too wide to trip `_SLIVER_Q_EXCESS`, so that
arm is silent too: `within_layer_warn` is **False** on every row of the
physical scan.  The caller is left with a generic "energy not conserved"
warning on an answer that is wrong by three orders of magnitude.

Round 1 behaves identically, so this is inherited and not a round-3
regression, and round 3 deliberately does not touch the SCREEN.  The
measurement above raises the reachability of the class without changing its
severity classification.

**Reproducer**: `v6_open_items.py` arms `physical` and `disarm_window`;
`tests/unit/test_verify_pmmstack_sliver_round2.py::test_an_owned_liner_anywhere_disarms_the_cross_layer_refusal`.

### R3-C (LOW, pre-existing) -- REPRODUCED: a keyed `prepare()` stack is outside the guard

`_segment_passive` returns False for a `str` payload, so
`_stack_provably_passive` is False for any stack carrying material KEYS.
Measured on my own geometry, both builds:

| arm | `_stack_provably_passive` | screen reached | probe solves | decision | `R+T` | warnings |
|---|---|---|---|---|---|---|
| resolved (`eps` = 2.56 / 12.25) | **True** | yes | 1 | **REFUSED** | -- | -- |
| keyed `prepare()` (`"LO"` / `"HI"`, resolved at `solve()`) | **False** | no | **0** | **RETURNED** | **2.7598** | 1 (the plain super-unity warning) |

Identical decision on WSL; `R+T` agrees to 12 significant figures
(2.7597687522353906 vs 2.7597687522296424).

**Reproducer**: `v6_open_items.py::r3c`;
`tests/unit/test_verify_pmmstack_sliver_round2.py::test_a_keyed_prepared_stack_is_outside_the_guard_entirely`.

### Contract checks that PASSED

* the arbiter leaves no `_sliver_probe` attribute and no new attribute of any
  kind on the caller's stack (`vars()` before and after a refusing solve are
  identical);
* `PMM_SLIVER_GUARD = False` returns the pre-guard answer bit for bit
  (SHA-256 of `R` and `T` identical to the disarmed control), on both builds.

---

## S9. Ship recommendation for 5.45.0

**SHIP.**  Round 3 is a strict widening of round 2's criterion, and every
measurement here supports that reading on devices the fix did not use:

1. **Nothing that worked stopped working.**  42 of 66 fixtures are returned by
   both trees with byte-identical `R`, `T` and Jones buffers, every warning set
   matches after excising the one sentence round 3 rewrote, and all 13 round-2
   refusals still refuse.  0 verdicts moved from `sliver` to `truncation`.
2. **Everything that moved, moved the right way.**  All 11 flips on my
   fixtures and all 85 on the fix's are `truncation -> sliver` on answers that
   are WRONG by an independent continuity rule -- mine by 52,048x to 7.6e+07x
   the physical wall shift, with the prescribed grid restoring every one of
   them to within 1.6x.
3. **The constant is sized correctly, and the sizing rule survives an
   independent box.**  1e-2 admits no CORRECT row the round-2 absolute closure
   did not already admit, on my box as on the fix's; the CONJUNCTION attributes
   0 of 737 arbitrated correct rows at every fraction from 1e-1 to 1e-3; the
   library produced 0 false positives over 2,304 rows and its decision matches
   the analytic criterion on 859 / 859 arbitrated rows.  The three false
   refusals a DIRECTED scan does find (**V-4**) are attributed identically by
   round 2's absolute closure and are refused on the round-3 branch point as
   well, so they are not a property of this change.
4. **The defect it targets is broader than the fix claimed, and the fix covers
   it.**  Nine mounts across five mechanisms land in the D-5 band; 1e-2
   recovers 100 of 102 rows (98.0 %).
5. **Both builds agree on every decision** -- including the three V-4
   refusals -- with the largest meaningful cross-build spread 1.44e-02 against
   a smallest decision margin of 0.789 over the ladder box's 859 arbitrated
   rows (54.6x).

Of the four defects raised, three -- V-1, V-2, V-3 -- are documentation,
message wording and probe methodology, and none changes a number the library
returns.  The fourth, **V-4**, does change a decision: three CORRECT answers
are refused.  It is nevertheless not a reason to hold this release, for one
measured reason: round 2 refuses exactly the same three rows, on the same
mount, at the same wall steps, with the same evidence dict, on the round-3
branch point.  Shipping round 3 leaves that behaviour untouched; NOT shipping
it leaves the D-5 class returning wrong answers as well.  The two open items
carried forward (R3-B, R3-C) are pre-existing, present in round 1, and each
already has a pinning test.

**Conditions.**  None blocking.  Three things should be recorded before the
release notes are written:

* **open V-4 as the next round's subject.**  It is the first measured case in
  this campaign of the guard refusing an answer that is correct by its own
  rule, it is present in the shipped tree AND in round 2, and the mechanism is
  named on both arms: the closure is vacuous whenever the snapped solve leaves
  the super-unity regime, and the move is a constant divided by a vanishing
  `w_wide` whenever the prescribed snap collapses a multi-slice wall set
  instead of perturbing it, so the conjunction of two degenerate tests fires
  on a correct answer.  It does not block 5.45.0 -- shipping
  round 3 leaves the behaviour on those rows exactly as round 2 left it -- but
  the release notes should not repeat "0 false positives" without the
  qualification "on a four-step wall ladder at degrees 6-10";
* correct V-1's three published bounds in the round-3 fix report (a
  corrections table, as round 3 itself did for round 2's D-4);
* state R3-A with the family floor (3.669) rather than the sample floor
  (49.107), and note that one of its rows is returned with NO warning at all
  because its `R+T` sits between the trigger and `_STACK_SUPERUNITY_BAR`.

**For the round after.**  V-4 is the first item, and it is not a constant
either: both of its arms degenerate for structural reasons (a vacuous closure
when the snapped solve leaves the super-unity regime, and a move normalised by
a `w_wide` that vanishes while the snap's real geometry change does not).  The
rest of this paragraph is the second item.  The single highest-value change
this campaign points at is not a constant.  It is that `R + T` is both the detector and the
attribution's denominator: R2-A bounds the guard below by the trigger, R3-A
bounds the closure below by the mount's own truncation floor, and S4.4 shows a
row that falls through both and is returned silently at 741x the wall shift.
A detector that does not go through `R+T` -- the conditioning of the interface
mode-match, or the `1/w^2` spurious-`|q|` predictor the within-layer arm
already computes -- would close both bands at once.  R3-B is the second
priority, and this verification raises it: its disarm window sits at
physically realistic feature widths (a 2 nm liner against a 37 pm wall
mismatch), and at those widths the WITHIN-LAYER arm the fix's R3-B entry
cites as the caller's fallback ("the caller is not left blind") is silent as
well -- the owned feature is far too wide to trip `_SLIVER_Q_EXCESS`.  The
fallback holds only in the pico regime the round-2 reproducer uses.

---

## S10. The runs

| probe | rows / fixtures | Windows | WSL |
|---|---|---|---|
| `v1_bitid.py` (AFTER) | 66 fixtures | 15.8 s | 11.9 s |
| `v1_bitid.py` (BEFORE, `f2371e0`) | 66 fixtures | 14.7 s | 11.8 s |
| `v2_closure.py` | 576 mounts, 2,304 rows | 302.5 s | 298.4 s |
| `v3_d5.py` | 24 screened, 9 mounts, 270 rows | 34.0 s | 31.3 s |
| `v4_move.py` | 4 resonance hunts + 14 mounts x 3 degrees x 22 wall steps, 1,284 rows | 3,055.7 s | 3,004.2 s |
| `v5_testbars.py` | every bar of the round-3 test file | 3.9 s | 4.3 s |
| `v6_open_items.py` | R3-B, R3-C, contracts | 7.9 s | 7.2 s |
| `v7_crossbuild.py` | the WIN/WSL comparison | < 1 s | -- |
| `v8_attack.py` | 6 mounts x 2 degrees x 30 wall steps, 313 rows | 56.2 s | 49.7 s |
| `v9_falserefusal.py` | V-4 taken apart, incl. the `f2371e0` arm | 7.4 s | 6.8 s |

All probe runs had `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1` on the command line, and the box was shared with three
other agents throughout, so the wall times are ratios and not benchmarks.

---

## S11. What this verification could NOT establish

* **The fix's own 576-configuration box was not re-run.**  Its 36.611 finite
  drop envelope, its 21 infinite-drop rows and its 79.032 move envelope are
  taken as published; what is verified is that an INDEPENDENT box of the same
  size reaches the same DECISIONS and gives the sizing rule the same answer.
  Where my box disagrees numerically (S3.3, S5.2) I report both.
* **The 615-row / eight-device flip census of `s3_dropgap.py` was not
  re-run.**  My flip census is 66 fixtures over seven devices, which is a
  smaller sample; it agrees in character (all flips one-way, all WRONG) and is
  more severe in degree.
* **No adjudication by an independent solver.**  The round-2 verification
  scored 14 rows against an `RCWAStack` oracle; this round classifies only by
  the exact `delta -> 0` limit of the same solver and by the prescribed-grid
  answer.  Those two references are independent of each other but not of the
  PMM implementation, so a systematic PMM error common to both would not be
  visible here.
* **The `unknown` branch and the sweep / prepared wavelength plumbing** were
  exercised only through R3-C.  Round 2's verification covered them (5e, 5f)
  and round 3 does not touch them.
* **Long-run stability.**  Every number here is a single measurement per
  build; the campaign's flake discipline was satisfied by cross-build
  agreement rather than by repetition.
* **The extent of the V-4 class.**  Three false refusals were found on ONE
  mount at ONE degree by a directed sweep of six mounts.  How large that class
  is -- which combinations of sub-unity sliver-free solve, device slope and
  wall step reach it -- was not mapped, and the search was not exhaustive in
  degree (only degrees 4 and 6 were swept on those mounts, and only the mounts
  the ladder box had already flagged).  The population statistic that matters
  for a repair is the joint distribution of `su_snapped == 0` and
  `move / w_wide > 100` over correct rows, and this verification measured it
  on 313 rows, not on thousands.

---

## S12. Commits

`verify/sliver-round3`, explicit-path `git add` only:

COMMIT-SHAS

# FIX ROUND 3 -- the `PMMStack` sliver arbiter's CLOSURE becomes RELATIVE

**Date** 2026-09-11 · **Branch** `fix/pmmstack-sliver-guard-round3` (from
`wave2/pmm2d` @ `f2371e0`, i.e. round 1 + its verification + round 2 + the
independent verification of round 2) · **Worktree** `C:/tmp/lum_sliver3` ·
**Scope** `lumenairy/elements/pmm/stack.py` only

**Subject** defect **D-5** of
`docs/audits/VERIFY_PMMSTACK_SLIVER_ROUND2_2026_09_11.md` (MEDIUM-HIGH, and
the one place round 2 is worse than round 1), plus its **D-3** (test-only) and
**D-4** (doc-only) siblings.

**Probes** `validation/probe_fix_sliver_round3/` -- four probes and a fixture
module, with the JSON of every table below committed for BOTH builds.

**Tests** `tests/unit/test_fix_pmmstack_sliver_round3.py` (6 new), plus the
verification's D-5 pinning test RE-PINNED against the repair it asked for.

**Binding** `docs/TESTING_STANDARDS.md`.

---

## S0. Terms

The guard, the screen and the arbiter are round 1's and round 2's; this report
only changes one expression inside the arbiter, so the terms are stated once
here and then used without further definition.

| term | meaning |
|---|---|
| **manufactured cell** | a cell of the shared union grid whose two walls share no owning layer -- geometry no single layer asked for.  `_cross_layer_sliver` flags one when it is at least `_SLIVER_OWN_SCALE_RATIO` = 100x finer than the finest wall spacing the input geometry does ask for |
| **the trigger** | `_SLIVER_TRIGGER_BAR` = 1e-3.  The arbiter runs only on a stack that carries a manufactured cell, is provably passive, and reads `max(R+T) - 1` above this |
| **the prescribed grid** | the grid `min_feature = 2 * w_wide * period` produces, where `w_wide` is the WIDEST manufactured cell -- i.e. the grid the refusal's first remedy would give the caller.  The arbiter's one extra solve is taken there |
| **the violation** | `max(R+T) - 1` on the solve about to be returned.  A theorem violation, not a tolerance: the stack is provably passive with a lossless propagating incidence medium |
| **`su_snapped`** | `max(R+T) - 1` on the prescribed grid, floored at 0 |
| **the DROP factor** | `violation / su_snapped` -- how much of the violation the prescribed snap REMOVES.  Infinite when the snapped solve reads at or below unity.  This is the statistic round 3's constant is sized on |
| **the move** | `move / w_wide`: the largest per-order efficiency difference between the returned solve and the prescribed-grid solve, over the orders they share and over BOTH incident polarizations, in units of the widest manufactured cell.  Unchanged by this round |
| **CORRECT / WRONG** | the campaign's fitted-constant-free continuity rule against the exact `delta -> 0` solve of the same stack: `err <= 10 delta` is RIGHT, `err > 100 delta` is WRONG, between is grey.  `err` is the polarization-1 distance, the convention every probe in this family classifies with |
| **arbitrated** | a row on which the arbiter actually runs: the screen fires AND the violation is above the trigger |

---

## S1. The defect, restated from measurement

Round 2's arbiter attributes the violation to the sliver when

```
su_snapped <= _SLIVER_ATTRIB_CLOSURE      (= 1e-5, ABSOLUTE)
        AND  move > _SLIVER_MOVE_FACTOR * w_wide
```

The first conjunct is a bar on an ABSOLUTE residue.  On any stack whose
SLIVER-FREE truncation super-unity already sits above 1e-5, the snapped solve
cannot read below 1e-5 either -- the sliver is gone but the truncation is not
-- so the conjunct **can never be met, however completely the snap restores
the answer**.  The arbiter then returns `truncation`, the wrong number is
RETURNED, and the warning says the prescribed remedy will not help.

The class is reachable and ordinary: a guided-mode grating in a
dense-superstrate grazing mount (period 1.0 um, `n_sup` 2.4, `n_sub`
1.45+0.05i, `wl` 0.93 um, theta 1.22 rad, degree 8).  Its sliver-FREE degree
ladder is clean truncation -- `R+T-1` = 2.078e-04 / **3.7295e-05** / 9.832e-06
/ 2.961e-07 / 3.153e-09 at degrees 6 / 8 / 10 / 12 / 14 -- so its degree-8
floor lands BETWEEN the closure bar and the trigger, and the stack is provably
passive, so `R+T <= 1` is a theorem there.

Re-measured here (`s2_d5.py`, Windows; the WSL column is in S6):

| `delta` | `R+T-1` returned | `R+T-1` on the prescribed grid | DROP | `move / w_wide` | `err/delta` returned | `err/delta` snapped | ROUND 2 | ROUND 3 |
|---|---|---|---|---|---|---|---|---|
| 1.1943e-05 (control) | +3.9167e-05 | +3.7294e-05 | 1.050 | 0.334 | **0.0324** | 0.00193 | returned, silent | **returned, silent** |
| 6.8726e-06 | **+1.9267e-01** | +3.7294e-05 | **5,166** | 13,665 | **1,811** | 0.00193 | `truncation`, RETURNED | **REFUSED** |
| 5.2134e-06 | **+1.9323e-01** | +3.7295e-05 | **5,181** | 18,251 | **2,399** | 0.00193 | `truncation`, RETURNED | **REFUSED** |
| 3.0000e-06 | **+2.3157e-02** | +3.7295e-05 | **621** | 3,501 | **3,501** | 0.00193 | `truncation`, RETURNED | **REFUSED** |

Every digit in that table is IDENTICAL on WSL (`s2_d5_wsl.json` against
`s2_d5_win.json`): the same four verdicts, the same 1.050 / 5,166 / 5,181 /
621 drops and the same 0.00193.

The control row is the two-sided half: its answer tracks the sliver-free
reference to 0.032x the physical wall shift, its violation is below the
trigger, and it is silent before and after.  The three defect rows are off by
1,811x-3,501x the wall shift at `R+T` = 1.19, and the prescribed `min_feature`
puts every one of them back on the sliver-free reference to `err/delta` =
0.00193 -- five decimal places.

---

## S2. What ships

One expression, in `_sliver_arbiter`:

```python
violation = max(worst - 1.0, 0.0)
closure = max(_SLIVER_ATTRIB_CLOSURE, violation * _SLIVER_CLOSURE_FRACTION)
drop = (violation / su) if su > 0.0 else float("inf")
attributed = (su <= closure and move > _SLIVER_MOVE_FACTOR * w_wide)
```

with `_SLIVER_CLOSURE_FRACTION = 1.0e-2` -- "the snap must remove **100x** of
the violation", i.e. 99 % of it -- and the round-2 absolute value kept as the
LOWER arm of the `max`.

That lower arm is the whole reason the change is safe to reason about: the
closure the arbiter applies is never SMALLER than round 2's, so **no solve
round 2 attributed can stop being attributed**, and the only verdict transition
this change can produce is `truncation -> sliver`.  A stack whose truncation
floor is below 1e-5 is arbitrated exactly as round 2 arbitrated it.  On the
live path the relative arm is the binding one whenever it is the larger of the
two, which the trigger makes true above `violation` = 1e-3 (`1e-3 * 1e-2` is
exactly the absolute bar, so the two coincide at the trigger and the relative
arm is larger everywhere above it); the absolute arm binds only when
`_sliver_arbiter` is called directly with a `worst` below the trigger, which is
what the round-2 tests do.

`_sliver_arbiter`'s evidence dict gains three keys -- `closure`, `drop`,
`violation` -- and both messages are rewritten to quote them (S5).  Nothing
else in the file changes: the screen, the trigger, the move criterion, the
passivity test, the within-layer arm, the `unknown` branch, the probe's
side-effect contract and `PMM_SLIVER_GUARD` are all untouched.

---

## S3. Sizing the constant

### S3.1 Why the DROP factor

When the relative arm binds, `su <= violation * f` is exactly `drop >= 1 / f`.
So the constant is a bar on the drop factor, and there are two populations it
has to be placed against.

**BELOW it, the CORRECT population.**  A solve whose answer already tracks the
exact `delta -> 0` limit has little for the snap to remove: its violation is
the mount's truncation floor before the snap and after it.  This is the
population a false refusal would come out of -- the defect round 2 exists to
fix.

**ABOVE it, the D-5 population.**  Rows whose snapped answer IS the
sliver-free reference but whose residue lands on the mount's own truncation
floor.  These are the rows the absolute bar rejects.

Neither is a clean envelope, and S3.2 and S3.5 say why: part of the correct
population has an INFINITE drop (its snapped solve leaves the super-unity
regime), and the D-5 population has no floor above the correct one.  The
sizing rule that survives measurement is narrower than "put the bar between
them", and it is stated in S3.2.

### S3.2 The CORRECT population, and why it does not size the bar the obvious way

The obvious sizing is "put the bar above the correct population's drop".  That
is wrong here, and the measurement says so.

Over the 291 arbitrated CORRECT rows of the 648-configuration census box the
drop factor runs **0.9023 .. 2.2893**, which would leave room for almost any
bar.  But the census box is one period, one wavelength and one ridge pair.
`s4_lower_envelope.py` widens it to 576 configurations -- three periods
(0.9 / 1.2 / 1.6 um), two wavelengths (0.85 / 1.05 um), two superstrate indices
(2.4 / 3.2), three lossy substrates, two angles (1.22 / 1.38 rad), two degrees
(6 / 8), two ridge permittivities (8.0 / 14.0), two slice counts and four wall
steps -- and adds the D-5 devices themselves at coarse wall steps.  Of its
**1,078** arbitrated CORRECT rows:

| | Windows | WSL |
|---|---|---|
| arbitrated CORRECT rows | **1,078** | 1,078 |
| with a FINITE drop | 1,057 | 1,057 |
| their drop envelope | **36.61106834650884** | 36.611068342943376 |
| with an INFINITE drop (the snapped solve reads at or below unity) | **21** | 21 |
| of those, how many the ROUND-2 ABSOLUTE closure also admits (`su_snapped <= 1e-5`) | **21 -- all of them** | 21 |
| the correct population's `move / w_wide` envelope | **79.03246375926788** | 79.03246269653049 |
| the same among the rows the closure admits | **37.006809804** | 37.006809857 |

The 21 infinite-drop rows are the point.  A correct answer whose truncation
super-unity happens to fall BELOW unity on the snapped grid satisfies ANY
closure, absolute or relative -- round 2 says the same thing about its own bar
("the closure test has to be ONE-SIDED ... which is exactly why 6 rows survive
it ... the move criterion is what separates those 6").  So the closure has
never been a separator on its own, and the question this constant has to answer
is narrower and answerable: **does the relative arm admit any CORRECT row that
the absolute arm did not already admit?**

| `_SLIVER_CLOSURE_FRACTION` | the drop it demands | CORRECT rows the closure admits | of which the CONJUNCTION attributes |
|---|---|---|---|
| (round 2, absolute `1e-5`) | -- | **21** | 0 |
| 1e-1 | 10 | **22** | 0 |
| 3e-2 | 33.3 | **22** | 0 |
| **1e-2 (SHIPPED)** | **100** | **21 -- the same 21** | **0** |
| 3e-3 | 333 | 21 | 0 |
| 1e-3 | 1,000 | 21 | 0 |

1e-2 is the COARSEST value on the ladder that adds no correct row at all.  The
22nd row, which 3e-2 and 1e-1 admit, is a real correct answer with a finite
drop of **36.611** (period 0.9 um, `wl` 0.85 um, `n_sup` 2.4, `n_sub`
3.4+1.7i, theta 1.38 rad, degree 8, four slices, `delta` = 1e-4, `err/delta` =
5.32, `R+T-1` = 1.4455e-03, `su_snapped` = 3.9483e-05, `move / w_wide` =
40.71).  A bar at 33.3 would sit 1.10x
BELOW it, i.e. inside the population; 100 sits **2.73x** above it.

### S3.3 The D-5 population, measured

`s3_dropgap.py` scans five D-5 mounts at degrees 6 / 8 / 10 and 26 log-spaced
wall steps, and keeps the rows the ABSOLUTE bar rejects but whose answer is
WRONG and whose move is past the move bar -- i.e. exactly the rows D-5 is
about:

| | Windows | WSL |
|---|---|---|
| rows | **88** | **88** |
| drop floor | **49.106731312403326** | 49.106731307447994 |
| drop ceiling | **5,304.6085904956335** | 5,304.608590367727 |
| their `err/delta` | 517.7 .. 14,696 | identical to 9 figures |

### S3.4 The ladder, and what 1e-2 costs

| `_SLIVER_CLOSURE_FRACTION` | the drop it demands | margin over the CORRECT population's FINITE envelope (36.611) | CORRECT rows newly admitted vs round 2 | D-5 rows recovered | worst row left returned |
|---|---|---|---|---|---|
| 1e-1 | 10 | **0.27x -- the bar is INSIDE the population** | +1 | 88 / 88 | -- |
| 3e-2 | 33.3 | **0.91x -- still inside** | +1 | 88 / 88 | -- |
| **1e-2 (SHIPPED)** | **100** | **2.73x** | **0** | **85 / 88** | `err/delta` = **2,990** |
| 3e-3 | 333 | 9.10x | 0 | 60 / 88 | `err/delta` = **6,439** |
| 1e-3 | 1,000 | 27.3x | 0 | 32 / 88 | `err/delta` = **11,582** |

1e-2 is the point on the ladder where the two demands cross: it is the
coarsest fraction that clears the correct population's finite envelope, and
therefore the one that recovers the most of the D-5 class without admitting a
correct row the round-2 closure did not already admit.

The round-2 verification proposed "anywhere in 1e-2 .. 1e-3".  That range was
derived from the three D-5 rows it had, whose drops are 621 .. 5,181, and from
the census box's 291-row correct population, whose drop tops out at 2.29.  A
wider correct population puts the finite envelope at 36.611 and an 88-row D-5
population reaches down to 49.107 -- which is 1.34x apart, i.e. the two
populations are ADJACENT on this statistic and not separated by decades.  That
is why 1e-2 is the value and not 1e-3 (which would leave 56 of the 88 rows
returned, the worst off by 11,582x the physical wall shift), and why the
sizing rule is "admit no new correct row" rather than "clear the correct
population by decades".

### S3.5 The conjunction is where the decision lives -- and the upper side is COVERAGE

**The conjunction.**  Over **1,369** arbitrated CORRECT rows on two
independent boxes -- 291 (census box, `s1_census.py`) and 1,078 (wider box,
`s4_lower_envelope.py`) -- plus the 615 further arbitrated rows of
`s3_dropgap.py`'s eight devices, of which the continuity rule calls none
correct, the number of correct rows attributed is **0 at every fraction from
1e-1 to 1e-3, on both builds**, because the MOVE criterion holds every one
out.  Its own margin is the thinner
of the two and is worth stating: the correct population's `move / w_wide`
envelope over the wider box is **79.032** against the 100x bar (**1.27x**), and
among the rows the closure admits it is **37.007** (**2.70x**).  That is
consistent with defect D-4's finding that this bar's published margins were
sample-scoped; it is the bar to watch, and it is not the one this round moved.

**The upper side.**  The D-5 population has **no floor above the correct one**,
and the reason is structural rather than sampling.  A row's drop is

```
drop  =  violation / su_snapped  >  _SLIVER_TRIGGER_BAR / (the mount's own truncation floor)
```

because the arbiter only runs above the trigger.  A mount whose sliver-free
floor is 3.73e-05 (degree 8 here) therefore cannot produce a fully-restoring
row below drop ~27; a mount whose floor is 2.08e-04 (degree 6 here) cannot
produce one below drop ~4.8, which is INSIDE the correct population's range.
The three rows 1e-2 leaves returned are exactly that: drop 49.107 at degree 6,
`err/delta` 1,514 .. 2,990.  On a mount whose truncation floor is within
`_SLIVER_CLOSURE_FRACTION` of the trigger, no setting of this constant can
attribute the violation, and the solve is returned with the plain super-unity
warning.

That is the same shape of limit as round 2's own R2-A (the trigger cannot go
below the correct population's envelope, so the wrong rows underneath it are
unreachable), one level deeper, and it is recorded as open item **R3-A**.

---

## S4. The censuses, re-run

Both censuses were re-run end to end on both builds, once on the tree still
carrying round 2 and once on the shipped tree.  Every row records the
library's actual decision AND the two criteria scored analytically from the
same measured `(worst, su_snapped, move, w_wide)`, so the round-2 column is
verifiable rather than remembered.

### S4.1 FALSE POSITIVES -- the 648-configuration box

| | round 1 | round 2 | **round 3** |
|---|---|---|---|
| configurations | 648 | 648 | 648 |
| CORRECT by continuity | 648 | 648 | 648 |
| **refused although correct** | **77** | **0** | **0** |
| arbitrated rows | 291 | 291 | 291 |
| rows carrying the truncation note | -- | 77 | **77 -- exactly the rows round 1 refused** |
| returned answers BIT-identical to the unguarded solve | -- | 648 / 648 | **648 / 648** (0 broken) |

Nothing moves, and the reason is two-sided rather than lucky: the box's
arbitrated correct rows have `move / w_wide` <= **13.553**, so the MOVE
criterion holds every one of them out on its own, AND their drop factor is at
most **2.2893**, so the widened closure does not reach them either.

The verdicts are also scored ANALYTICALLY on the same rows, from the recorded
`(worst, su_snapped, move, w_wide)`, so this is a check and not a restatement:
the analytic ROUND-2 verdict equals the library's decision on 291 / 291
arbitrated box rows and 447 / 447 arbitrated grid rows on the pre-change tree,
and the analytic ROUND-3 verdict at 1e-2 equals it on the same 291 and 447 on
the shipped tree.  Both builds.

### S4.2 FALSE NEGATIVES -- the 660-row grid

| | round 1 | round 2 | **round 3** |
|---|---|---|---|
| rows | 660 | 660 | 660 |
| RIGHT / grey / WRONG | 209 / 3 / 448 | same | same |
| arbitrated | 447 | 447 | 447 |
| WRONG rows returned unwarned | 5 | **1** | **1** |
| false positives | 0 | 0 | **0** |

This row counts only the rows the continuity rule calls WRONG.  The round-2
report's headline "8 -> 4" counts the grey rows as well (5 wrong + 3 grey ->
1 wrong + 3 grey), which is the same census read one class wider.

The one WRONG row that remains is the same row round 2 leaves: degree 10,
`delta` = 1.8854e-06, `err/delta` = 176.3, whose `R+T-1` = **+5.1383e-04** is
BELOW the trigger, so the arbiter never runs on it.  That is R2-A, unchanged
by this round.

### S4.3 The FLIP census -- what round 3 actually moves

`s3_dropgap.py`, 615 arbitrated rows over eight devices (five staircases with
continuity slopes 0.47 .. 31.4, five D-5 mounts, four points on a guided-mode
resonance flank where the device's own `dR/d(duty)` is 128-174):

| | Windows | WSL |
|---|---|---|
| arbitrated rows scanned | 615 | 615 |
| verdicts that change, `truncation -> sliver` (at the shipped 1e-2) | **85** | **85** |
| verdicts that change the other way | **0** | **0** |
| of the 85: CORRECT by the absolute continuity rule | **0** | 0 |
| of the 85: CORRECT by the slope-NORMALISED rule (`err <= 3 s delta`) | **0** | 0 |
| of the 85: WRONG by both rules | **85** | 85 |
| the mildest flipped row's `err/delta` | **517.7159682141083** | 517.7159810552637 |
| flips on the ordinary staircase family | **0** | 0 |
| flips on the resonant (steep `dR/dx`) family | **0** | 0 |
| the same at 1e-1 / 3e-2 / 3e-3 / 1e-3 | 88 / 88 / 60 / 32, all WRONG | identical |

The resonant family is the one R2-D says the move criterion is attackable on,
and it is worth stating separately that round 3 does not touch it: its rows
either already met the absolute closure (their snapped solve reads at or below
unity, so they were refused under round 2 as well -- and an independent
`RCWAStack` adjudication in the round-2 verification found the guard right on
all 14 rows it scored) or they fail the MOVE criterion, which round 3 does not
change.

---

## S5. The two messages

### S5.1 The `truncation` note

Round 2 ended it, unconditionally, with

> *"... the super-unity SURVIVES at 1+3.73e-05 and the answer moves only ...x
> the cell width.  Raising min_feature will silence nothing here -- reduce
> n_slices or raise degree."*

On the D-5 class that sentence is false by 621x-5,181x.  It is also false on
any row where the snap DOES remove the violation and the verdict turns on the
move criterion instead -- the note asserted both halves of a conjunction that
only one half of had failed.

The note now names WHICH criterion was not met and quotes what was measured.
Measured on a census-box row (`nsub` 1.45+0.08i, `nsup` 2.4, theta 1.22,
degree 6, `delta` 1e-3):

> *"A near-coincident-wall SLIVER (0.001 of a period) IS present on the union
> grid but is NOT what moved this answer: re-solved on the min_feature=2.4e-09
> grid that removes it, the super-unity SURVIVES at 1+0.0309 -- a 0.9847x drop
> from the 1+0.0304 here, against the 100x an attribution asks for -- and the
> answer moves 3.79x the widest manufactured cell (bar 100x).  Raising
> min_feature leaves 1+0.0309 standing -- reduce n_slices or raise degree."*

and on a row held out by the MOVE criterion alone the second half reads
instead *"the super-unity DOES fall away there -- to 1+..., a ...x drop -- but
the answer moves only ...x the widest manufactured cell against the 100x an
attribution asks for, so this number does not depend on that cell.  Raising
min_feature removes the cell without moving the answer"*.  Both are statements
about this call's measurement; neither promises anything about `min_feature`
that was not measured.

The phrase `is NOT what moved this answer` is kept verbatim, because three
test files and four probes use it as the truncation-note detector.

### S5.2 The refusal's ATTRIBUTION paragraph

It quoted the absolute bar, which is no longer the bar that was applied.  It
now quotes the drop, the violation it is a drop from, and the closure actually
used:

> *"ATTRIBUTION, MEASURED ON THIS CALL (one extra solve): re-solved on the
> min_feature=1.375e-11 grid the cell is gone and the super-unity GOES WITH IT
> -- max R+T there is 1+3.73e-05, i.e. a 5166x drop from the 1+0.193 this
> solve reads (bar: the snap must remove 100x of the violation, or reach
> 1e-05 outright, whichever is the weaker demand -- here 0.00193) -- while the
> answer MOVES 0.0939 in per-order efficiency, 1.366e+04x the widest
> manufactured cell (bar 100x).  So the SLIVER moved this answer, not degree /
> n_slices: this refusal is an attribution, not a guess."*

A snapped solve that reads at or below unity gives an infinite drop; the
sentence then reads *"ALL of the 1+... this solve reads"* rather than printing
`inf`.

---

## S6. Both builds

| | Windows | WSL (Ubuntu) |
|---|---|---|
| python | 3.14.6 | 3.12.3 |
| numpy | 2.4.4 | 2.4.6 |
| BLAS | scipy-openblas, Haswell kernel | scipy-openblas, SkylakeX kernel |
| threads | `OMP/OPENBLAS/MKL_NUM_THREADS = 1`, set before numpy is imported | same |

Every DECISION in this report is identical on the two builds, and every
statistic agrees to at least 9 significant figures:

| statistic | Windows | WSL |
|---|---|---|
| census box, arbitrated CORRECT rows' drop envelope | 2.2892515262885182 | 2.289251619603185 |
| census box, arbitrated CORRECT rows' `move / w_wide` envelope | 13.553113356416588 | 13.553107774128623 |
| false-negative grid, arbitrated WRONG rows' drop floor | 177,582.47948419364 | 177,582.54976331317 |
| D-5 population's drop floor | 49.106731312403326 | 49.106731307447994 |
| D-5 population's drop ceiling | 5,304.6085904956335 | 5,304.608590367727 |
| the wider box's CORRECT rows, FINITE drop envelope | 36.61106834650884 | 36.611068342943376 |
| the wider box's CORRECT rows, `move / w_wide` envelope | 79.03246375926788 | 79.03246269653049 |
| flips at the shipped 1e-2 | 85, all WRONG | 85, all WRONG |
| the mildest flipped row's `err/delta` | 517.7159682141083 | 517.7159810552637 |

---

## S7. What ELSE this round did

### S7.1 D-3 (test-only) -- the sample-scoped move bar

`test_the_arbiter_separates_the_two_causes_on_this_build` asserted
`min(sliver move) >= 3 * _SLIVER_MOVE_FACTOR` (= 300) over five hand-picked
rows.  Those rows read 4,789.5 / 287,271.7 / 15,956.0 / 15,956.5 / 47,742.9,
so the bar held with 16x to spare ON THAT SAMPLE -- and the FAMILY reaches
**147.411** (polarization 1 alone **134.171**), i.e. 2.0x BELOW what the line
demanded.

**Already landed in the base of this branch** (commit `667f416`, merged at
`f2371e0`).  The assertions now read

```python
assert min(s[1] for s in sliver) > ps._SLIVER_MOVE_FACTOR, sliver
assert max(t[1] for t in trunc) < ps._SLIVER_MOVE_FACTOR, trunc
assert min(s[1] for s in sliver) > 10.0 * max(t[1] for t in trunc)
```

-- each population on its own side of the constant (the DECISION the bar
makes, which the family DOES support: the family floor 147.411 carries
**1.47x** over the bar), plus a decade of separation between the two
populations the test itself measures (measured separation 4,789.5 / 3.451 =
**1,388x**, so the demanded decade carries 139x).  The family numbers and the
date are in the comment above them.  Verified present and green in this
branch's runs (S8); no further change was needed.

### S7.2 D-4 (doc-only) -- four sample-scoped numbers

Corrected in `docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND2_2026_09_11.md`, in
a new **S0.1 CORRECTIONS** table plus an inline `[D-4: ...]` marker at each of
the ten sites where one of them appears, three `(D-4, S0.1)` rows added to
that report's own both-builds table, and R2-D re-stated in place:

| published | corrected | measured on |
|---|---|---|
| the trigger clears the correct population by **9.11x** | **8.03x** -- envelope 1.24481e-04 | 826 correct rows over five fixtures |
| the WRONG population's `move / w_wide` floor is **466.2** (bar carries 4.66x) | **147.411**, pol-1 **134.171** (bar carries **1.47x**) | a 2,250-row grid over five fixtures, slopes 0.47 .. 31.4 |
| the CORRECT population moves at most **26.58** | **833.78** on a guided-mode-resonance grating with degree-stationary `dR/d(duty)` = 169.2 | 2,028 rows.  The move bar is a DECISION criterion whose correctness rests on the CLOSURE arm, not on a universal `dR/dx` = O(1) |
| ONE sub-unity WRONG row at **-2.8048e-04** | at least **11**, worst **-1.13511e-03**; 29 of 878 wrong rows at or below the trigger | the same 2,250-row grid |

None of the four changes a decision.  The third is the one worth carrying
forward as prose rather than as a number: R2-D's premise is confirmed (a
device whose own `dR/dx` far exceeds the move bar exists and is buildable) and
its published bound is refuted by 31x, but the DECISION survived independent
adjudication -- on all 14 rows an `RCWAStack` oracle scored, the guard refuses
answers RCWA also calls wrong, and on 12 of 14 the prescribed `min_feature` is
1.8x-7.1x closer to the truth.  So the move criterion keeps its bar and loses
its published margin, and R2-D is RE-STATED rather than closed.

### S7.3 The D-5 pinning test, re-pinned

`test_the_closure_criterion_is_absolute_not_relative` in
`tests/unit/test_verify_pmmstack_sliver_round2.py` pinned the defect and said
in its own docstring: *"If a relative closure ships, this test fails -- that
failure is the gate working; re-pin it against the improvement, do not relax
it."*  It is re-pinned, not relaxed: renamed
`test_the_closure_criterion_is_relative_not_absolute`, the premise assertions
kept verbatim (the mount is provably passive, its degree ladder is monotone,
its degree-8 floor is above the absolute bar and below the trigger, the snap
lands ABOVE the absolute bar), and the verdict assertions inverted to the
repaired decision -- `sliver`, REFUSED, the message naming `min_feature`, and
the false sentence absent.

---

## S8. Tests and runs

### S8.1 The new file

`tests/unit/test_fix_pmmstack_sliver_round3.py`, six tests.  Every bar in it is
measured on the running build; the fixed numbers are the library's own
constants and the geometry.

| test | what it asserts | measured here |
|---|---|---|
| `test_the_d5_rows_are_refused_and_the_refusal_names_min_feature` | the D-5 reproducer as a DECISION: the three rows are REFUSED, the refusal carries the sliver message, names the `min_feature` that was measured to restore the answer, and no longer says the remedy will silence nothing | premise: the ladder is monotone and its degree-8 floor is 3.73e-05, 3.7x above the absolute bar and 27x below the trigger; rows: drop 5,166 / 5,181 / 621 against a demanded 100, `move / w_wide` 13,665 / 18,251 / 3,501 against 100, `err/delta` 1,811-3,501 returned and 0.00193 snapped |
| `test_the_correct_rows_of_the_staircase_box_are_still_returned_bit_identical` | the other side of the same decision: 24 configurations of the staircase box, every one CORRECT by continuity (asserted), every one RETURNED, every returned `R` and `T` bit-identical to the unguarded solve | 24 rows, 14 of them arbitrated; `err/delta` 0.34-8.04 |
| `test_the_closure_fraction_separates_the_two_drop_populations` | the two-sided bar: each population on its own side of `1 / _SLIVER_CLOSURE_FRACTION`, plus a decade of SEPARATION between the two populations the test itself measures | correct rows 0.98 .. 2.29 (22 of them) against a bar of 100 -- 43.7x; D-5 rows 621 .. 5,181 -- 6.2x; separation 621 / 2.29 = **271x** against a demanded 10x |
| `test_the_relative_closure_can_only_widen_the_absolute_one` | the structural claim, on real rows: the applied closure is never below `_SLIVER_ATTRIB_CLOSURE`, and `round2 attributed => round3 attributes` | 3-4 rows from both populations |
| `test_the_round2_fixtures_arbitrate_identically_under_the_relative_closure` | the round-2 fixtures' verdicts and numbers are unchanged: five SLIVER rows still `sliver`, three TRUNCATION rows still `truncation` and held out by the MOVE arm, their returned `R`/`T` bit-identical | sliver rows' `su_snapped` <= 1e-6 and drop above the bar; truncation rows' `move / w_wide` <= 3.45 against 100 |
| `test_the_truncation_note_states_the_measured_drop_and_promises_nothing` | the message contract: the note quotes the measured drop and the residue to the library's own formatting, names WHICH criterion was not met, and the sentence D-5 showed to be false is gone from the library (comments excluded) and from every warning the test collects | the note on a census-box row reads "a 0.9847x drop from the 1+0.0304 here, against the 100x an attribution asks for" |

Two shapes are deliberately avoided.  The D-5 arm CONTINUES past a row that
does not meet its premise (`err > 100` returned and `err_snapped < 1` snapped)
and asserts an existence of at least two of three, so a build that moves one
row out of the band does not fail it.  The bar test asserts the SEPARATION
between the two populations it measured rather than a multiple of the
constant, which is the restatement D-3 asked for one file over.

### S8.2 The runs

All with one BLAS thread pinned on the command line.  That matters here: with
the thread count left to the wheel, the same five-file set took **8 minutes**
on a box already running another agent's 24-thread job, against **65 s**
pinned -- the sliver fixtures are near-degenerate eigenproblems and the
oversubscription cost is not a rounding error.

| run | Windows | WSL |
|---|---|---|
| `test_fix_pmmstack_sliver_round3.py` (new, 6 tests) | **6 passed**, 3.64 s | **6 passed**, 3.21 s |
| the four sliver files + the new one (65 tests) | **65 passed**, 1 warning, 64.92 s | **65 passed**, 1 warning, 66.88 s |
| `tests/unit/test_pmm*.py tests/unit/test_fix_pmm*.py tests/unit/test_verify_pmm*.py` (22 files) | **496 passed**, 48 warnings, 1,535.66 s (25:35) | -- |
| the census / walker / dispatcher-pin / public-api / doc-consistency sweep (26 files) | **1,282 passed, 12 skipped**, 4 warnings, 203.69 s | -- |
| `ruff check lumenairy/ tests/ validation/probe_fix_sliver_round3/` | -- | **All checks passed!** |

The one warning in every sliver run is the pre-existing
`test_return_owners_is_additive_and_warn_false_is_silent`'s deliberate
`_pmm_union_grid` snap warning, which that test raises on purpose.  The 48 in
the 22-file run are that one plus the pre-existing `_pmm_union_grid` snap
warning of `test_pmm_m3_efficiency.py` and the two-dimensional mortar files'
own deliberate `RuntimeWarning`s; none is new in this round.  The four in the
sweep are the pre-existing HFPI under-sampling warnings of the dispatcher-pin
files.

### S8.3 Probe wall times

| probe | Windows | WSL |
|---|---|---|
| `s1_census.py` (648 + 660 rows), BEFORE arm | 135.1 + 137.6 s | 146.2 + 141.5 s |
| `s1_census.py`, AFTER arm | 124.8 + 125.7 s | 129.5 + 133.1 s |
| `s2_d5.py` | 9 s | 8 s |
| `s3_dropgap.py` (615 arbitrated rows of 8 devices) | 128.9 s | 137.6 s |
| `s4_lower_envelope.py` (576 configurations + the D-5 devices) | 414.9 s | 392.2 s |

`.test_durations` gains the new file's six measured Windows timings and the
renamed D-5 entry (12,633 -> 12,639; an 8-line diff, nothing reformatted).

---

## S9. Open items

| | |
|---|---|
| **R3-A -- the relative closure has a floor of its own, and it is structural** | A row's drop is at least `_SLIVER_TRIGGER_BAR` divided by the mount's own sliver-free truncation floor, so on a mount whose floor is within `_SLIVER_CLOSURE_FRACTION` of the trigger no setting of the constant can attribute the violation.  Measured: at degree 6 the D-5 mount's floor is 2.078e-04, which bounds its fully-restoring rows below at drop ~4.8 -- INSIDE the correct population's own range (0.90 .. 2.29).  The three D-5 rows the shipped 1e-2 leaves returned (drop 49.107 at degree 6, `err/delta` 1,514-2,990) are the visible edge of this band, and they cannot be recovered by loosening the constant: 3e-2 would recover them but would also admit a CORRECT row the round-2 closure did not admit (S3.2), and a mount one degree coarser would produce rows 3e-2 cannot reach either.  This is R2-A one level deeper: the guard's floor is a property of the theorem it uses as a detector, not a choice.  A detector that does not go through `R+T` is what would close it, and this campaign has not found one |
| **R3-B (was D-1, MEDIUM, pre-existing) -- one thin OWNED feature disarms the cross-layer refusal for the WHOLE stack** | `_cross_layer_sliver` computes the own-scale as the GLOBAL minimum wall spacing over all layers, and flags a manufactured cell only when `own / w >= _SLIVER_OWN_SCALE_RATIO`.  A single sliver-thin feature that ONE layer legitimately owns therefore lowers `own` for every manufactured cell in the stack and the screen goes silent on genuine cross-layer slivers.  Measured (verification S8 D-1, `w8_lc_exact.py` arm B, both builds): the O-11 sliver at `delta` = 3e-5 reads `own / w` = 9,287.3, the screen fires and the solve is REFUSED at `R+T` = 2.17 (degree 12) / 23.4 (degree 14); adding a 1e-6-of-a-period liner ONE layer owns drops `own` to 1e-06 and `own / w` to **0.03**, the screen returns `None`, and the same stack is RETURNED at `R+T` = **23.30**.  Reproducer: `tests/unit/test_verify_pmmstack_sliver_round2.py::test_an_owned_liner_anywhere_disarms_the_cross_layer_refusal`.  NOT fixed here, deliberately: the repair is to score `own` per flagged cell against the finest spacing of the layers that own that cell's neighbours, which needs the per-cell owner sets `_pmm_union_grid` already returns but is a change to the SCREEN, i.e. to the input of every census in rounds 1-3, and would have to be re-measured against the 27,904-cell ordinary-geometry population that sizes `_SLIVER_OWN_SCALE_RATIO`.  It is not a one-line per-layer own-scale.  Round 1 behaves identically, so it is inherited and not a regression, and the caller is not left blind -- the within-layer warning fires on the same stack, though it names the liner rather than the cross-layer pair |
| **R3-C (was D-2, LOW, pre-existing) -- a KEYED `prepare()` stack is outside the guard entirely** | `_segment_passive` returns False for a `str` payload, so `_stack_provably_passive` is False for any stack carrying material KEYS -- which is the case `prepare()` exists for.  Measured (verification S8 D-2, both builds): a keyed prepared stack with the O-11 sliver at `delta` = 3e-5 solves to `R+T` = **23.42** with the screen never reached, **0** probes and only the plain super-unity warning.  Reproducer: `tests/unit/test_verify_pmmstack_sliver_round2.py::test_a_keyed_prepared_stack_is_outside_the_guard_entirely`.  NOT fixed here: `_PreparedPMMStack.solve` resolves `materials` before it solves, so handing the RESOLVED tensors to the guard is available without materialising anything, but it changes which stacks the guard reaches and needs its own bit-identity and census arms.  This is round 2's R2-C, now with a measurement attached |
| **R2-A, R2-B, R2-D, R2-E, R2-F** | unchanged from round 2, except that R2-D's published bound is corrected (S7.2) and R2-A now has a sibling in R3-A |

---

## CORRECTIONS 2026-09-11 from the independent round-3 verification

Source: `docs/audits/VERIFY_PMMSTACK_SLIVER_ROUND3_2026_09_11.md` (own
576-mount / 2,304-row box, six D-5 mounts over four mechanisms, both builds;
every decision identical on WIN and WSL).  The decisions in this document
stand; three of its published numbers are properties of its samples:

| published here | verification's reading | consequence |
|---|---|---|
| the D-5 population's drop floor is 49.107 (S3, R3-A) | **3.669** over 102 rows on four mechanisms | the R3-A band reaches INSIDE the correct population's drop range (0.64 .. 18.09); no setting of the fraction separates them, as R3-A already says.  One such row (drop 5.205, `err/delta` 741) is returned at `R+T` = 1.00235, BELOW the plain warning bar -- silently |
| the correct population's `move / w_wide` envelope is 79.032 (1.27x under the 100x bar) | **161.073** on the verification's ladder box; **906.555** on a tapered family (`move` saturates at 4.3e-3 while `w_wide` vanishes) | the move bar is INSIDE the correct population.  The move arm is not what holds correct rows out; the closure arm does.  The verification's decision, with numbers, is NOT to re-derive the bar for 5.45.0: the statistic (a move in units of a vanishing cell width), not its value, is the problem, and no fixed value is safe |
| the correct population's finite drop envelope is 36.611 (2.73x) | 18.090 on the verification's box (5.53x) | this document's larger envelope stays binding |

Recovery rate: 100 / 102 D-5 rows at 1e-2 (this document: 85 / 88), 88 at
3e-3, 77 at 1e-3.  Flip census on the verification's fixtures: 11 flips, all
`truncation -> sliver`, all wrong (`err/delta` 52,048 .. 7.6e+07) and all
restored by the prescribed grid.

**New open item V-4 (MEDIUM, INHERITED -- round 2 refuses the identical
rows).**  On one mount (period 1.02 um, `wl` 0.633 um, `n_sup` 3.10, `n_sub`
2.90+1.10i, theta 1.35, eps 12.25, 3 slices, **degree 4**) at `delta` =
1.662e-5 / 1.269e-5 / 7.395e-6 three CORRECT answers (`err/delta` 1.051 /
1.065 / 0.979) are REFUSED: `su_snapped` is exactly 0 (so the absolute bar
admits them as well), `move / w_wide` = 114.7 / 150.0 / 256.5 -- both arms
degenerate at once -- and the prescribed remedy moves the answer slightly
AWAY from a degree-16 reference (0.0592457 returned vs 0.0592724 snapped).
Degree 4 is supported, and the "degree too low" guard does not fire.  A
false REFUSAL, not a wrong answer; the subject of a later round.  R3-B was
reproduced and sharpened (a 2 nm liner plus a 37 pm wall mismatch, own/w =
54.5, silences the screen: `R+T` = 2.92, `err/delta` 2,573, returned under
the generic warning; non-monotone in `delta`), and R3-C reproduced
(`R+T` = 2.7598 returned).

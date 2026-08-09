# The null control conditioned on the GEOMETRY while the repair conditions on the ROUND-OFF -- 2026-08-09

One test failure was referred on `main` after the v5.33.0 merge (GitHub CI,
py3.10, shard 1, run 31307754759):

```text
tests/unit/test_pmm_m2_window_contract.py::
  test_min_feature_is_the_accuracy_lever_on_the_per_layer_path_too

ns=2 min_feature=5e-10: an ABOVE-threshold cell moved when the
forward-growth repair was switched -- the repair is only allowed to touch
solves that put a growing mode in the forward set, and a cured cell has none

assert 0.04886 == 0.0
```

The IDENTICAL tree was green on branch CI and is green on both mounts at every
thread count here, so the axis is the runner's CPU / OpenBLAS kernel.  That is
the campaign's standing disease and this is a new instance of it -- but not in
a bar.  It is in a control's PREMISE.

**The library is not changed.  The repair's behaviour on that cell was
CORRECT; the test's assumption about where it is allowed to act was
per-environment.**

---

## S1.  The diagnosis: the control and the repair partition different sets

Section (5) of that test is the M2 NULL FLOOR.  It asserted that every cell
with `min_feature` ABOVE M2's `min(off, |coat - off|)` threshold is
BIT-IDENTICAL with `PMM_FORWARD_GROWTH_REPAIR` on or off, tolerance 0.0, and
its message states the reasoning: *"the repair is only allowed to touch solves
that put a growing mode in the forward set, and a cured cell has none"*.

The second clause is the mistake, and it is a category error rather than a
tolerance:

```text
the control conditions on   min_feature > threshold      a fact about GEOMETRY
the repair  conditions on   prop & grow & (near|passive) a fact about ROUND-OFF
```

`_forward_growth_flip` acts on modes whose z-flux -- a round-off quantity for
an evanescent mode, which carries exactly zero z-power -- crosses
`_mass_flux_threshold`.  WHICH modes those are is decided by the BLAS reduction
order, on the same eigenvalues (`FIX_UNION_GRID_2THREAD_2026_08_06` S2.1
measures the two members of one double root at 1.87x and 0.096x the cut at one
thread and 2.21x / 0.063x at two).  A cured GEOMETRY removes the near-zero-width
cross-layer cell that makes such modes plentiful; it does not make the cut
un-crossable.  On the ubuntu py3.10 runner an above-threshold cell still
produced two of them, the repair redirected them, and the answer came back
right -- and the control called that a violation.

**Redirecting them was correct.**  The invariant `_forward_growth_flip` enforces
-- a forward mode of a passive layer may not grow along +z -- is a statement
about the mode, not about the geometry that produced it.  It applies wherever
such a mode appears.

So the control is re-stated on the INSTRUMENT, where it was always true.

---

## S2.  Reproduced on this build, to four figures

A test cannot change the runner's OpenBLAS kernel.  It CAN put the cut where
that kernel's round-off put it, which is `FIX_CI_ROUND2_PMM_2026_08_08` S6's
NEAR-CUT INJECTOR -- `_mass_flux_threshold` scaled by 1/3, pulling the cut down
under evanescent modes whose round-off flux sits just below it.  Nothing about
the geometry, the materials, the operator or the eigenvalues moves; only the
cut, which is the one thing a BLAS kernel is entitled to move.

The ORIGINAL test, unmodified (`git show origin/main:...`), run under that
injector [M, Windows, 1 thread]:

```text
ns=2 min_feature=5e-10: an ABOVE-threshold cell moved when the forward-growth
repair was switched -- the repair is only allowed to touch solves that put a
growing mode in the forward set, and a cured cell has none

assert 0.04885933506106029 == 0.0
  array([0. , 0. , 0. , 0.04885934, 0. ])
  ladder ON  [0.11087971 0.11060686 0.11052811 0.11049899 0.11048606]
  ladder OFF [0.11087971 0.11060686 0.11052811 0.06163965 0.11048606]
```

-- the referred failure verbatim, same cell, same rung (index 3 of
`(6, 8, 10, 12, 14)`, i.e. the 4th), same magnitude to four figures against the
CI log's 0.04886.  The CI failure is therefore not a mystery about ubuntu: it
is what this test does whenever the cut lands one notch lower than it does
here.

### S2.1  What the census says on the same cell

`_MODE_CUT_CENSUS`, summed over every modal row of the solve, cured cells,
degrees 6..14 [M, Windows 1 thread]:

```text
                        cut as this build finds it   cut scaled by 1/3
ns  mf      degree      raw  post  moved             raw  post  moved
 2  0.5 nm  6..10        0    0    0.0                0    0    0.0
 2  0.5 nm  12           0    0    0.0                2    0    4.886e-02
 2  0.5 nm  14           0    0    0.0                0    0    0.0
 3  1.5 nm  10           0    0    0.0                1    0    5.482e-01
 6  3.0 nm  6..14        0    0    0.0                0    0    0.0
```

`raw` is `n_grow` (the pre-repair DIAGNOSIS -- `_record_mode_cut` is
deliberately called on the raw `prop`/`q`, so this column does not move with
the switch), `post` is `n_grow_post` (the RESIDUAL, what the shipped selector
left growing), `moved` is `|R0(on) - R0(off)|`.

**Every cured rung reads `raw` = 0 with the cut where this build's round-off
puts it, in all five (mount x thread-count) environments:**

```text
mount    OPENBLAS_NUM_THREADS   raw per rung (15 rungs)   max move   all passive
Windows  1                      all 0                     0.0        yes
Windows  2                      all 0                     0.0        yes
Windows  24                     all 0                     0.0        yes
WSL      1                      all 0                     0.0        yes
WSL      2                      all 0                     0.0        yes
```

That is the measurement that makes the fix a re-statement and not a
relaxation: on every environment reachable here, the reconditioned control
takes the bit-identity branch on every one of the 15 rungs and is the ORIGINAL
assertion, verbatim, on every cell.

### S2.2  WHICH rung the injector lands on migrates with the pool

The same lesson the four-name adjudication and `FIX_CI_ROUND2` S8 record, one
level down.  Injector at 1/3, cured cells:

```text
mount / OPENBLAS_NUM_THREADS   rung(s) that move
Windows 1                      ns=2 deg 12 (raw 2), ns=3 deg 10 (raw 1)
Windows 2                      ns=3 deg 12 (raw 1)
Windows 24                     ns=2 deg 12 + 14, ns=3 deg 10 + 14
WSL 1                          ns=2 deg 12, ns=3 deg 10
WSL 2                          ns=3 deg 12
```

So the fail-before SCANS the family and never names a rung.

---

## S3.  The fix, at test level

`tests/unit/test_pmm_m2_window_contract.py` only.  No library file is touched;
no bar is moved; sections (1), (2), (3), (4) and (6) of the referred test, and
every other test in the file, are unchanged.

### S3.1  The control, re-stated on the instrument

Section (5) becomes a call to `_score_null_control(cells, refs)`, which
partitions rung by rung on the CENSUS:

```text
census reads ZERO raw growing modes
    -> nothing is in the repair's mask, so the two selectors are the SAME
       ARRAY and the answer must be BIT-IDENTICAL (tolerance 0.0).
       THE TRUE NULL, and the original assertion.

census reads a growing mode
    -> the rung MAY move, and then owes both of:
         n_grow_post == 0             the shipped forward set no longer grows
         |on - ref| <= |off - ref|    it ended closer-or-equal to the answer
                                      the reference names
```

`refs` is the RCWA 141-order anchor this test already carries (`RCWA_R0`),
which exists for `ns` = 2 and 6.  Neither claim is a tolerance: one is a mode
COUNT, the other an ORDERING between two distances.

Two preconditions are asserted per cell rather than assumed:

* `n_grow` is IDENTICAL with the switch off and on.  That is a design property
  of `_record_mode_cut` (it reads the pre-repair `prop`/`q`) and it is the
  whole reason a control may condition on it, so it is measured, not trusted;
* every modal row is recognised PASSIVE.  The moved-rung claims rest on "a
  forward mode of a PASSIVE layer cannot grow"; on a row where passivity is not
  proven that invariant does not apply and the claims would prove nothing.

### S3.2  The mechanics that had to move with it

* `coated_ladder` now delegates to `_ladder_rec`, which solves each rung with
  the T3-4 census armed and returns `(R0, n_grow, n_grow_post, non-passive
  rows)`.  Arming the census changes no number -- it decides only whether the
  instruments are computed -- and the columns are READ OFF `_MODE_CUT_CENSUS`
  rather than re-derived, so the control cannot drift from the shipped
  instrument.  `coated_ladder`'s signature and return value are unchanged, so
  its other five call sites are untouched.
* `_LADDER_CACHE`'s key gains the injector scale, for the same reason it
  already carries both switches: a cache that ignored it would serve an
  un-injected ladder to an injected measurement.
* `near_cut_injector(scale)` -- `FIX_CI_ROUND2` S6's injector, previously in
  `scratchpad/faultinject.py`, now a context manager in the test file so the
  fail-before ships with the claim it supports.

### S3.3  The fail-before

`test_the_null_control_conditions_on_the_census_not_the_threshold` (new).
Under the injector it scores the reconditioned control on the same three cured
cells and then:

```text
(a) the reconditioned control PASSES, and for the right reason -- every rung
    it let move carries a raw growing mode, ends with none growing, and ends
    no further from the cured answer;
(b) the ORIGINAL control -- bit-identity on every above-threshold rung -- is
    re-run VERBATIM on that same table and FAILS.  That is the CI failure,
    pinned in-tree and permanently: `pytest.raises(AssertionError,
    match="ABOVE-threshold cell moved")`;
(c) the null half is not vacuous -- the injector leaves rungs whose census
    reads ZERO, and those are still scored BIT-IDENTICAL.
```

The reference for (a) is the cell's own UN-INJECTED shipped ladder -- the
answer the library gives when the cut sits where this build's round-off puts
it -- rather than `RCWA_R0`, because `RCWA_R0` has no `ns` = 3 entry and the
pools that move only the `ns` = 3 cell (Windows 2, WSL 2) would leave the
closer-or-equal claim unexercised.

The injector scale is a LADDER, `(1/3, 1e-2, 1e-4)`, walked until one of the
cells scored carries a raw growing mode.  The SCAN is repair-ON only -- the raw
census is the pre-repair diagnosis and does not move with the switch -- so a
scale that carries nothing costs one ladder, not two.  Where no scale
reproduces, the test FAILS with its cell list and instructions to widen the
ladder.  **It never skips.**

### S3.4  A measurement that is PRINTED and not asserted

With the repair ON, the injected answer is BIT-IDENTICAL to the un-injected one
on all 15 cured rungs -- `1/3` scale, tolerance 0.0, full 17 digits.  There is
an argument for why: lowering the cut can only re-classify round-off-flux
(purely imaginary `q`) modes as propagating, and the repair hands exactly those
back to the `Im(q)` rule, which is what they had as evanescent.

It is not asserted, because the argument has a hole that is a per-build fact:
a genuinely propagating mode with a weak but REAL flux sitting in the
`(thr/3, thr]` band would be re-classified too, and the two rules do NOT agree
on it.  None exists on this device.  That is a property of this device's
spectrum, i.e. exactly the class of claim this document exists to stop
asserting.

---

## S4.  Fault injection -- every branch, in three environments

`scratchpad/faultinject_null.py`, run on Windows at 1 and 2 threads and on WSL
at 1.  All fifteen cases behave as designed in every cell (K..O are S5's
extension; the scorer and the fail-before body are SHARED, so A..J score them
too):

```text
case                                                       want  outcome
A  the shipped injector test: the scale ladder finds 1/3,  PASS  PASS
   the control passes and the original claim is pinned
   failing on the same table
B  the scale ladder FALLS THROUGH: scales (1.0, 1/3),      PASS  PASS
   where 1.0 moves nothing and 1/3 does
C  no scale reproduces (scales = (1.0,)): the test FAILS   FAIL  FAIL
   with its table and never skips
D  a FLOODING scale (1e-3) puts a growing mode on every    FAIL  FAIL
   cured rung: the non-vacuity guard refuses, because
   the bit-identity branch was then never scored
E  the passivity widening defeated under a deep injector   FAIL  FAIL
   (scale 0.03): a beyond-decade survivor stays in the
   forward set and n_grow_post > 0 on a moved rung
F  the reference SPOOFED to the broken answer, per rung    FAIL  FAIL
   on every cell: closer-or-equal refuses the move
G  the RAW census made switch-dependent (_mode_cut_growth  FAIL  FAIL
   stubbed to 0 when the repair is on)
H  passivity NOT PROVEN on the rows (_grid_is_passive      FAIL  FAIL
   stubbed False): the moved-rung claims are refused
   before they are made
I  the SHIPPED section (5) scored under the injector       PASS  PASS
J  the ORIGINAL section (5) under the same injector        FAIL  FAIL
K  the UNCOATED fail-before with the ladder truncated to   FAIL  FAIL
   (1/3), i.e. above that device's condition: it FAILS
   loudly rather than passing vacuously
L  the SHIPPED uncoated control under its deep injector    PASS  PASS
M  the ORIGINAL uncoated claim under the same              FAIL  FAIL
N  the threshold-rule fail-before                          PASS  PASS
O  the uncoated fail-before                                PASS  PASS
```

F needed the family form.  Its first draft spoofed the reference on the `ns` = 2
cell alone and PASSED at 2 threads -- where that cell does not move and only
`ns` = 3 does.  The migration again, inside the injector rig this time.

---

## S5.  The two SIBLING null controls, re-pointed

The first draft's OPEN item 3 recorded two more tests in this file carrying the
referred premise on different cells.  Neither had failed on any runner; both
carried the identical exposure, and "has not failed yet" is not a state this
campaign leaves a control in.  Both are now scored through the same
`_score_null_control`, and each gains its own injector-driven fail-before.

```text
test                                          cells               device
test_min_feature_threshold_rule_              (3, 1.5 nm)         coated
  predicts_stationarity                       (6, 3.0 nm)         taper
test_threshold_rule_holds_on_a_               (6, 3.0 nm)         uncoated
  SINGLE_REGION_uncoated_taper                (12, 1.5 nm)        taper
```

The re-pointing is the same in both places and nothing else in either test is
touched: the original bit-identity assertion is replaced by the scorer, which
takes the bit-identity branch on every rung whose census reads zero.  Measured
UN-INJECTED, all five (mount x thread-count) cells, both families:

```text
raw n_grow per rung   [0]*10 on both
sum n_grow_post       0
non-passive rows      0
```

-- so on every build reachable here both are the assertion they replace,
verbatim.  The threshold-rule pair keeps `RCWA_R0` as its reference (it exists
for `ns` = 6); the uncoated device has no anchor in this file, so its moved
half is scored on `n_grow_post` alone, which is the claim that needs no
reference.

Two mechanical changes came with it and neither is a claim: `_ladder_rec` gains
a device builder (`_coated_stack` / `_uncoated_stack`, which reproduce each
test's own builders exactly) and its cache key gains the device; the uncoated
test's private ladder cache is replaced by that shared one, same builders, same
degrees.

### S5.1  Per-test injector evidence

`_injected_null_control` walks the scale ladder for the cells IT is given.

**Threshold-rule pair** -- reaches the condition at `FIX_CI_ROUND2`'s own 1/3,
and on a DIFFERENT rung from the one CI found, because this pair contains no
`ns` = 2 cell:

```text
mount / OPENBLAS_NUM_THREADS   moved at 1/3        raw  post   move
Windows 1                      ns=3 deg 10          1    0     0.5482
Windows 2                      ns=3 deg 12          1    0     0.4627
Windows 24                     ns=3 deg 10 + 14     1,1  0     0.5482 / 0.4627
WSL 1                          ns=3 deg 10          1    0     0.5482
WSL 2                          ns=3 deg 12          1    0     0.4627
```

with 9 (or 8) of its 10 rungs still scored BIT-IDENTICAL in the same run.

**Uncoated taper** -- and this is a finding rather than a calibration.  It does
NOT reach the condition at 1/3, nor at 1e-1, 3e-2, 1e-2, 3e-3, 1e-3 or 3e-4.
It takes **1e-4**: the cut has to fall four decades before any mode's round-off
flux crosses it [M, Windows 1 thread, degrees 8..16]:

```text
scale     rungs with a raw growing mode   moved
1         0 / 10                          0
1/3       0 / 10                          0
1e-2      0 / 10                          0
1e-3      0 / 10                          0
1e-4      3 / 10                          3
```

That is what a SINGLE REGION with no conformal coat buys: there is no
coat/offset collision to inject near-zero-width cross-layer cells, so the
spectrum carries no round-off-flux mode anywhere near the cut.  **The uncoated
null control was the least exposed of the three by four decades, and that was
not knowable before it was measured** -- it is the same device M5 escalated as
showing the same silent-wrong scatter, so "simplest member of the class" did
not imply "safe" and does not here either; what it implies is a much larger
margin.  At 1e-4 the moved rungs are IDENTICAL in all five environments
(`ns=6` degrees 10 and 16, `ns=12` degree 10, raw 1/2/1, post 0 everywhere) --
no migration, because that cut position is nowhere near a round-off boundary.

Case K in S4 is the guard on this: with the ladder truncated to `(1/3,)` --
above this device's condition -- the uncoated fail-before FAILS loudly instead
of passing vacuously, in all three fault-injection cells.

### S5.2  Where the reference comes from in the injector tests

All three fail-befores use each cell's UN-INJECTED SHIPPED LADDER as the
reference, which is the only one that exists for all three devices.  It is a
real reference and not a tautology, but the measurement behind it is worth
recording: **with the repair on, the answer is BIT-IDENTICAL to the un-injected
one on every cured rung of both devices** -- 15/15 coated at scale 1/3 and
10/10 uncoated at 1e-4, tolerance 0.0, full 17 digits.  Case F (the reference
spoofed to the broken answer, per rung on every cell) shows the claim still
refuses a move in the wrong direction.

S3.4's caveat applies unchanged: this is printed, not asserted, because a
genuinely propagating mode with a weak but REAL flux inside the injected band
would break it, and none exists on either device.

---

## S6.  Result

`-q -p no:randomly`.  Windows = python 3.14.6 / numpy 2.4.4 / scipy 1.17.1,
scipy-openblas 0.3.31 Haswell, MAX_THREADS 24.  WSL = python 3.12.3 / numpy
2.4.6 / scipy 1.17.1, same OpenBLAS release, SkylakeX, MAX_THREADS 64.

```text
mount    OPENBLAS_NUM_THREADS   result
Windows  1                      19 passed    45 s
Windows  2                      19 passed    50 s
Windows  default (24)           19 passed   108 s
WSL      1                      19 passed    42 s
WSL      2                      19 passed    42 s
```

The count rises from 16 to 19: one fail-before per null control (S5).  COST,
measured at one thread against the referred tree (`git show origin/main:...`
run under the same interpreter): **39.4 s -> 44.9 s** for three injector-driven
fail-befores and the census columns.  The census is cheap (it computes no eig),
the scale scan is repair-ON only, and every ladder -- coated and uncoated,
injected and not -- now shares one `_LADDER_CACHE`.

Blast radius, Windows at 1 thread -- the four named files of the two preceding
rounds, `m2` + `m3` + `m1` + `v5_13_0_pmm_tapered`: **98 passed, 117 s** (the
+3 over the referred tree is this file's own added tests; `m3`, `m1` and the
tapered suite are unmodified and unaffected -- no library file is touched).
`ruff check tests/unit/test_pmm_m2_window_contract.py`: clean.

---

## S7.  Open

1. **Channel B, the T3-4 guard's DISARMED state, and X-1 are all untouched.**
   Nothing here changes the library.
2. **The MIRROR-IMAGE misclassification** (`FIX_CI_ROUND2` S8 item 2) is still
   untouched and still judged rare, and this round adds a small piece of
   evidence for that judgement: under the near-cut injector, which
   re-classifies a whole band of modes at once, the residual read 0 on every
   moved rung in every environment.  A second mechanism would have had many
   chances to show up there.
3. **The fourth null control of this file is NOT re-pointed.**
   `test_conforming_and_untapered_stacks_are_immune_to_both_knobs` asserts
   bit-identity on a CONFORMING stack and on a vertical untapered staircase.
   It is structurally safe rather than empirically safe: those stacks share
   walls, so their windows have no near-zero-width cross-layer cell to inject
   at all, and the knobs it varies are `window_halfwidth` and `min_feature`
   rather than the forward-growth switch.  Left alone deliberately.  (S5
   re-points the two that DID carry the referred premise.)
4. **THE PROCESS FINDING, and it cost one refutation inside this round.**  The
   first draft of the new test's claim (c) partitioned the ladder on "did the
   rung MOVE" instead of on "what does the census READ".  Those coincide on
   every environment this box can produce at scale 1/3 -- and come apart at
   1e-3, where a rung carries 2 raw growing modes and still does not move
   (ns=2, degree 6).  Such a rung was scored by the moved-rung branch, and the
   draft would have called it a null.  **A test that partitions on the OUTCOME
   instead of on the CONDITION is the same error as the one being fixed, one
   level down** -- and it was caught only because the flooding scale was run as
   a fault-injection case rather than reasoned about.

---

# VERIFY -- the `PMMStack` sliver guard's ARBITER, round 2, independently re-measured

**Date** 2026-09-11 · **Subject**
`docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND2_2026_09_11.md` (branch
`fix/pmmstack-sliver-guard-round2`, 13 commits, merged at `24651c8`) ·
**Worktree** `C:/tmp/lum_vsliver2`, branch `verify/sliver-round2` ·
**Without-arm** a read-only `git worktree add C:/tmp/lum_prer2 bb0527a`

**Probes** `validation/probe_verify_sliver_round2/` -- eleven probes and a
fixture module written from scratch, on both builds.  Nothing in this report
is read from the fix's own JSON; every number below was re-measured here.

**New tests** `tests/unit/test_verify_pmmstack_sliver_round2.py` (6), plus one
bar of the round-2 file restated.  No `lumenairy/` file was touched.

**Binding** `docs/TESTING_STANDARDS.md`.

---

## S0. Verdict summary

| # | claim | verdict | the number, both builds |
|---|---|---|---|
| 1 | bit-identity of every untouched path, RETURN **and** warning set | **CONFIRMED** | 31 / 31 fixtures identical, 0 warning-set differences, Windows and WSL |
| 2a | the trigger 1e-3 clears the correct population's envelope by 9.11x | **BOUNDED** | envelope **1.24481e-04** over 826 correct rows / 5 fixtures -> **8.03x**, not 9.11x |
| 2b | the closure bar 1e-5 splits WRONG (<= 1.54e-06) from TRUNCATION (>= 3.86e-05) | **CONFIRMED, with more room than claimed** | wrong 0 .. **1.4585e-10**; truncation best **6.580e-04** -- a 6.7-decade gap, the bar 66x below the truncation population |
| 2c | the move bar 100 splits CORRECT (<= 26.6) from WRONG (>= 466) | **half CONFIRMED, half REFUTED** | correct side: no correct row is arbitrated at all, and the arbitrated (correct) rows of the census box reach **13.55** -> 7.4x. Wrong side: the wrong population reaches DOWN to **147.41** (pol-1 **134.17**), i.e. **1.47x**, not 4.66x |
| 2d | the own-scale ratio 100 sits 3.93x above ordinary non-conforming geometry | **CONFIRMED** | max **25.4765** over 27,904 manufactured cells from 6,000 random 2-4-layer stacks -> **3.93x**; 0 cells over the bar |
| 2e | the 16-ULP passivity deadband is 20.8x the worst round-off and 9 decades below real gain | **CONFIRMED** | worst spurious **0.628 ULP** / 200,000 lossless directors (**25.5x**); 0 / 50,000 lossy refused; `Im(n)` = 1e-6 reads **6.167e+09 ULP** and is refused; the detection floor is `Im(n)` ~ **1e-14** |
| 2f | the q-excess bar 1e6 sits between ordinary owned cells (<= 3.32e3) and breaking liners (>= 8.79e6) | **CONFIRMED exactly** | **2.931 .. 3.322e+03** and **8.794e+06 .. 3.322e+07** |
| 3a | false positives 110 / 648 -> 0 / 648 | **CONFIRMED on my own box** | **77 / 648 -> 0 / 648** (my box's specific values differ; the character is identical: `err/delta` 0.336 .. 8.037, `R+T` 1.01006 .. 1.16463) |
| 3b | false negatives 8 / 660 -> 4 / 660 | **CONFIRMED, row for row** | **8 -> 4**, every one of the eight rows reproducing the fix's `err`, `err/delta` and `R+T-1` to 4 significant figures |
| 3c | the four residuals, and which half of the guard holds each out | **CONFIRMED and COMPLETED** | the one row the fix left "not measured" reads `move / w_wide` = **21.6** -- the MOVE criterion's floor, not the trigger's |
| 3d | a WRONG row exists below unity, which no super-unity bar can see | **CONFIRMED, and deeper** | **-1.1351e-03** (the fix reports -2.8048e-04); **11** sub-unity wrong rows and **29** wrong rows at or below the trigger, out of 878 |
| 4a | R2-D: the move bar assumes `dR/dx` = O(1); no device was found that breaks it | **mechanism CONFIRMED, published bound REFUTED, decision HOLDS** | a guided-mode-resonance grating with degree-stationary `dR/d(duty)` = **169.2** puts a CORRECT row's `move / w_wide` at **833.78** vs the published "<= 26.58" (31x). But on all 14 rows adjudicated by an independent RCWA oracle the refusals are of answers RCWA also calls wrong, and on 12 of 14 the prescribed remedy is 1.8x-7.1x closer to the truth.  A separate 1,458-configuration directed scan of the census box's mount produced **2,252 refusals and 0 false ones** |
| 4b | the other direction -- a sliver-WRONG solve returned as `truncation` | **REFUTED -- found, and it is a round-2 behaviour change** | see **D-5**: the snap removes a **5,181x** super-unity and restores the answer to `err/delta` = 0.002, and the arbiter still says `truncation` because the residue lands on the mount's own 3.73e-05 truncation floor, above the ABSOLUTE 1e-5 closure |
| 5a | the probe never mutates the caller's stack | **CONFIRMED** | the ONLY attribute a guarded solve writes is `_modal`, identical to a guard-disarmed control; `_src` and `_layers` objects and values unchanged; no `_sliver_probe` left behind; a prepared object's caches unchanged |
| 5b | it never recurses | **CONFIRMED** | exactly 1 probe per refusal, 0 on a returned solve; a stack marked `_sliver_probe` has `_sliver_screen` = `None` and `_within_layer_hazard` = `None` |
| 5c | cost 0.20x (Windows) / 0.27x (WSL) of the guarded solve | **BOUNDED** | **0.260x** (67.0 -> 17.4 ms) / **0.254x** (61.1 -> 15.5 ms) |
| 5d | fires on 0 of 600 converged correct rows | **CONFIRMED** | 600 rows, 599 RIGHT, arbiter fired **1** time and **0** times on a RIGHT row, both builds |
| 5f | the sweep and prepared paths arbitrate at their OWN wavelength, at any worker count | **CONFIRMED** | with a stale `set_source(4.0e-07)` and a sweep at 8.5e-07 the probe is handed **`[8.5e-07]`** at `max_workers` 1 / 2 / 4 and the solve is REFUSED at all three; a HEALTHY sweep is probed **0x** and its `R`/`T` buffers are **byte-identical** across worker counts; the key-free `prepare()` path is handed `[8.5e-07]` and REFUSED |
| 5g | the move is taken on the CENTRED order overlap, which matters | **CONFIRMED, and more than published** | the report says the order count differs on 359 of 637 rows; on my census box it differs on **69 of 69** arbitrated rows and on my clean fixtures on **0 of 100**.  `_sliver_answer_move` returns the centred distance (3.0 on synthetic `(2,5)` vs `(2,3)`) and `None` on a parity mismatch |
| 5e | the `unknown` branch keeps round 1's decision and says so | **CONFIRMED** | dispersive sweep: `provably_passive` False, screen never reached, 0 probes, plain warning at `R+T` = 23.4. No resolved source: verdict `unknown`, REFUSED above the round-1 bar, message contains "could NOT be run on this path" |
| 6a | the anisotropic classes are now reached; the non-Hermitian control is not | **CONFIRMED, to the digits published** | in-plane 45 deg `move` **38.86 / 290.36** (report: 38.9 / 290.4); out-of-plane 30 deg **22.18 / 316.46** (22.2 / 316.5); gyrotropic refused at err **685x / 2680x** (685 / 2680); non-Hermitian `provably_passive` False and returned at `R+T` = 1.093 |
| 6b | the pol-0 evidence: pol 1 reads 0.01x the shift, pol 0 reads 316x, and the arbiter uses BOTH | **CONFIRMED, decisive, and BROADER than published** | on the out-of-plane director: pol 1 **0.01055x**, pol 0 **316.4x**; `move` on pol 1 alone = **0.0106** (would NOT refuse), on both = **316.46** (refuses).  And the same asymmetry appears on ORDINARY dense-superstrate grazing mounts with no anisotropy at all: 11 of 2,252 refusals in the 1,458-configuration directed scan read pol-1 `err/delta` of 5.9-13.3 and pol-0 `err/delta` of **153x-434x** (S5.3) |
| 6c | the within-layer arm warns on the breaking widths and is silent on benign geometry | **CONFIRMED, with a sharper floor** | the ladder reproduces exactly; silent on 1e-2 .. 1e-5 AND on 1e-6 (`err` = **1.058e-03** at `R+T` = **0.999221**), and at 1e-7 it is silent on 1 of 4 degrees because that degree reads `R+T` = **0.5706**, SUB-unity |
| 6d | can an OWNED liner be REFUSED through the cross-layer path? | **NO -- but the converse is a defect** | see **D-1**: an owned liner ANYWHERE disarms the cross-layer refusal |

**Defects raised**: **`D-5` (MEDIUM-HIGH, a round-2 behaviour change)**,
`D-1` (MEDIUM, pre-existing), `D-2` (LOW, pre-existing), `D-3` (LOW,
test-only), `D-4` (LOW, doc-only).  Only `D-5` is new in round 2.

---

## S1. The two builds, and the arms

| | Windows | WSL (Ubuntu) |
|---|---|---|
| python | 3.14.6 | 3.12.3 |
| numpy | 2.4.4 | 2.4.6 |
| BLAS | scipy-openblas 0.3.31.188.0, **Haswell** kernel | scipy-openblas 0.3.31.188.0, **SkylakeX** kernel |
| threads | `OMP/OPENBLAS/MKL_NUM_THREADS = 1`, set before numpy is imported | same |

`lumenairy.__file__` is asserted by every probe.  The WITHOUT arm is a full
read-only worktree at `bb0527a`, not a file swap.  That is legitimate for this
subject: `git diff bb0527a 24651c8 -- lumenairy/` touches exactly three files
-- `elements/pmm/stack.py` (this work, +584 / -76), `elements/_lens_traced.py`
and `propagators/carrier.py` (two unrelated branches merged into the same
integration branch) -- so within the PMM package **`stack.py` is the only file
that differs**, and `elements/pmm/_core.py` is byte-identical between the two
arms.

---

## S2. Bit-identity (task 1)

`w1_bitid.py` hashes sha256 over `(dtype, shape, raw buffer)` of every array
each fixture returns AND records the set of warning messages.  31 fixtures,
written here rather than reused: shared and per-layer grids (`window_halfwidth`
1 and 2), classical / conical / single-slant / MIXED-slant, in-plane,
out-of-plane and gyrotropic tensors and a non-Hermitian payload, lossy layers,
a lossy substrate, an ABSORBING superstrate, Bragg dedupe, tapers with the snap
active and inactive, `stabilize='slices'`, `retain_internal` +
`internal_field`, `layer_absorption`, `per_order_amplitudes`,
`solve_vs_wavelength` at 1 / 2 / 4 workers, a DISPERSIVE sweep, `prepare()`
with a material key, a stack carrying a manufactured sliver below the trigger,
and three fixtures built specifically so the warning half of the claim is not
vacuous.

| | Windows | WSL |
|---|---|---|
| identical | **31 / 31** | **31 / 31** |
| differing | 0 | 0 |
| errors | 0 | 0 |
| warning-set differences | **0 / 31** | **0 / 31** |

The warning half is **not** vacuous here, which is the one thing the fix's own
39-fixture ledger could not say (all 39 of its fixtures are silent on both
trees).  Two of my fixtures warn on BOTH arms with byte-identical text:

* `w15_taper_snap_active` -- `_pmm_union_grid: snapped 14 pair(s) of
  NEAR-COINCIDENT cross-layer walls closer than min_feature=0.00214 ...`
* `w29_superunity_no_sliver` -- a dense-superstrate grazing mount with
  IDENTICAL walls in every layer (so no cell is manufactured and the screen
  cannot fire) reading `R+T` = **1.0389**: the PLAIN super-unity warning, and
  round 2 appends nothing to it.

and `w30_between_trigger_and_bar` is the fixture that proves the LOWERED
trigger creates no new report on its own: `R+T` = **1.00225**, above the
round-2 trigger and below the warning bar, silent on both arms.

The two builds' hashes DIFFER from each other (`w01` reads
`cad4e3292332c92e...` on Windows and `72da461acd93daa8...` on WSL), which is
the control proving 31/31 within a build is evidence rather than luck.

**Every `_warn_stack_energy` call site, before and after.**  Nine in
`stack.py` on both arms, in the same order, every one still passing
`stack=self` (or `stack=self._st` on the prepared path); exactly THREE gained
a `src=` argument -- the two `solve_vs_wavelength` stores and the prepared
path, i.e. the three whose wavelength is not `stack._src`.  No 1-D caller lost
its `stack=` and none gained `stack=None`.

**The 1-D and 2-D callers that pass `stack=None`.**
`git diff bb0527a 24651c8 -- lumenairy/elements/pmm/stack2d.py
lumenairy/elements/pmm/stack2d_pure.py lumenairy/elements/pmm/twod.py` is
EMPTY, and `stack2d.py:1628` still reads `_warn_stack_energy(R_eff, T_eff)`
with no `stack=`.  Open item R2-E stands exactly as written.

---

## S3. The bars, re-derived (task 2)

`w2_populations.py`: FIVE fixtures x 150 log-spaced deltas (3e-3 .. 1e-6) x
degrees 10 / 14 / 18 = **2,250 rows**, the guard disarmed throughout.  Only one
fixture (`O11`) reuses the fix's geometry; the other four are mine, chosen to
span the quantity R2-D says the move criterion assumes is O(1) -- the device's
own continuity slope `dR/dx`, measured at degree 14 in the smooth regime:

| fixture | period / wavelength / theta | slope `err/delta` |
|---|---|---|
| `C_nir` | 1.05 / 0.98 um / 0.42 | **0.47** |
| `O11` | 1.2 / 0.85 um / 0.15 | 1.15 |
| `B_vis` | 0.74 / 0.53 um / 0.31 | 3.15 |
| `D_tele` | 1.8 / 1.31 um / 0.11 | 10.2 |
| `S_steep` | 0.74 / 0.53 um / 0.31 | **31.4** |

Class counts, IDENTICAL on the two builds: 826 RIGHT, 878 WRONG, 546 grey,
852 arbitrated.  Every statistic below agrees between the builds to 5-6
significant figures.

### S3.1 The trigger (`_SLIVER_TRIGGER_BAR` = 1e-3) -- BOUNDED

| | fix | Windows | WSL |
|---|---|---|---|
| correct population's super-unity envelope | 1.0979e-04 (600 rows, 3 fixtures) | **1.24481e-04** (826 rows, 5 fixtures) | 1.24481e-04 |
| headroom at 1e-3 | 9.11x | **8.03x** | 8.03x |
| per-fixture envelopes | -- | `C_nir` 9.130e-05 (281 rows), `O11` 1.564e-05 (236), `B_vis` **1.2448e-04** (293), `D_tele` 1.718e-05 (16) | identical |
| `S_steep` | -- | **no CORRECT rows at all** -- its own slope (31.4) is above the campaign's 10x "RIGHT" cutoff, so the absolute continuity rule cannot label it | identical |

The claim "9.11x" is a property of the fix's three fixtures.  Adding two more
raises the family envelope by 13 % and drops the headroom to 8.03x.  The
`S_steep` row is worth noticing on its own: a device whose physical `dR/dx` is
31.4 has NO rows the campaign's absolute `err <= 10 delta` rule calls RIGHT,
at any delta -- the classification rule the whole census rests on is itself a
`dR/dx = O(1)` assumption, which is the same assumption R2-D raises about the
move criterion (S5.1).  The
DECISION the trigger encodes is unaffected -- it still clears the correct
population, on both builds, by most of a decade -- so this is BOUNDED rather
than REFUTED, but the report's "9.11x" should be read as a lower bound on the
population, not a margin.

### S3.2 The closure (`_SLIVER_ATTRIB_CLOSURE` = 1e-5) -- CONFIRMED with more room

| population | fix | mine |
|---|---|---|
| WRONG, snapped super-unity | 0 .. 1.5368e-06 | **0 .. 1.4585e-10** (849 arbitrated wrong rows) |
| TRUNCATION, best snapped super-unity | 3.860e-05 | **6.580e-04** (291 arbitrated rows of the census box, every one of them CORRECT) |
| the bar's two-sided margin | 6.5x / 3.9x | **6.9e+04x / 66x** |

The gap is not merely two-sided by decades -- it is **6.7 decades wide** on my
grids, and the shipped bar sits inside it with four decades above the wrong
population and 1.8 decades below the truncation one.  Both builds identical.

### S3.3 The move (`_SLIVER_MOVE_FACTOR` = 100) -- one side CONFIRMED, one REFUTED

| population | fix | Windows | WSL |
|---|---|---|---|
| CORRECT, `move / w_wide` | 0.1218 .. **26.58** | **no correct row reaches the trigger at all** on the 2,250-row grid; on the 648-configuration census box, where 291 CORRECT rows ARE arbitrated, the range is **0.122 .. 13.553** | identical |
| WRONG, `move / w_wide` | **466.2** .. 6.04e+07 | **147.411** .. 1.011e+07 | **147.411** .. 1.011e+07 |
| WRONG, pol 1 only | 338 | **134.171** | 134.171 |
| GREY, `move / w_wide`, arbitrated | (not reported) | **108.59 .. 230.84** (3 rows, all REFUSED) | identical |
| the bar's margins | 3.76x / 4.66x | **7.4x / 1.47x** | same |

**REFUTED**: `move / w_wide >= 466.2` is not a property of the wrong
population.  The lowest-move wrong rows on my grid:

| fixture | deg | delta | err/delta | `R+T-1` | `su_snap` | `move/w` | pol-1 `move/w` |
|---|---|---|---|---|---|---|---|
| `S_steep` | 18 | 5.88980e-06 | 151.9 | +1.605e-03 | 1.42e-14 | **147.41** | 147.41 |
| `B_vis` | 18 | 3.83194e-06 | 166.7 | +1.235e-03 | 7.08e-14 | 165.55 | 165.55 |
| `C_nir` | 14 | 5.28966e-06 | 133.8 | +1.250e-03 | 2.80e-14 | 178.40 | **134.17** |

so the bar carries **1.47x** below the population it must sit under, not
4.66x.  That is inside the cross-build spread's own order of magnitude but
NOT inside it in practice: both builds read 147.411 to six figures, so the
guard's DECISION on these rows is stable (all 849 arbitrated wrong rows are
attributed on both builds).  What is refuted is the published margin, and with
it the implicit claim that the move criterion has a decade of room -- it has
0.17 of one.

The correct side is confirmed and is stronger than published: **849 / 849**
arbitrated wrong rows attributed, **0 / 826** correct rows arbitrated at all,
**0 / 291** arbitrated census-box rows attributed.

**The CENTRED-OVERLAP contract is load-bearing, and more so than published.**
The report says the snapped grid resolves a different number of orders on
"359 of 637" arbitrated rows.  On my two clean fixtures it never does (0 of
100).  On the census box -- a dense superstrate at 1.22-1.44 rad, where the
snap changes which orders propagate -- it does on **69 of 69**, i.e. every
arbitrated row.  So the contract matters on 100 % of one realistic population
and 0 % of another, and `_sliver_answer_move` handles both: on synthetic
`(2, 5)` vs `(2, 3)` inputs it returns the centred-overlap distance
(**3.0**, correct), and on a parity mismatch `(2, 5)` vs `(2, 4)` it returns
`None`, which the arbiter reads as `unknown` rather than comparing
mismatched orders.

### S3.4 The own-scale ratio (`_SLIVER_OWN_SCALE_RATIO` = 100) -- CONFIRMED

`w3_bars.py` re-draws the population with a different RNG stream, 2-4 layers
(the fix used two) and 6,000 stacks, scoring `own / w` on EVERY manufactured
cell unbarred:

| | fix (4,192 two-layer stacks) | mine (6,000 stacks, 27,904 cells) |
|---|---|---|
| max | 25.43 | **25.4765** |
| p99.9 | 21.76 | 19.014 |
| p99 | 15.32 | 12.484 |
| median | 2.496 | 1.375 |
| cells at or above the bar | 0 | **0** |
| headroom | 3.93x | **3.93x** |

Identical to three significant figures on both builds.  CONFIRMED.

### S3.5 The passivity deadband (16 ULP) -- CONFIRMED, and the floor located

| | fix | mine |
|---|---|---|
| worst spurious `-lam_min(A) / max\|eps\|` over 200,000 random rotated uniaxial directors | 1.7115e-16 = **0.77 ULP** | **0.6279 ULP** |
| headroom at 16 ULP | 20.8x | **25.5x** |
| lossless directors REFUSED | 0 | **0 / 200,000** |
| lossy directors REFUSED | 0 | **0 / 50,000** (their `lam_min(A)` is strictly POSITIVE) |
| gain at `Im(n)` = 1e-6 | -1.0e-06 = 5.8e+09 ULP | **6.167e+09 ULP**, `_segment_passive` = False |

and the two-sided statement the fix does not make -- the deadband's blind
band, measured as a ladder (identical on both builds):

| `Im(n)` (gain) | ULP | `_segment_passive` |
|---|---|---|
| 1e-16 | 0.674 | **True** (inside the deadband) |
| 1e-15 | 6.175 | **True** |
| **1e-14** | **61.67** | **False** |
| 1e-12 | 6.167e+03 | False |
| 1e-6 | 6.167e+09 | False |

so the deadband is 25.5x above the worst round-off it exists for and 3.9x
below the smallest gain it must reject: the band it hides is `Im(n)` below
~**2.6e-15**, nine decades below the `Im(n)` = 1e-6 the report quotes and far
below anything a material model produces.  CONFIRMED, two-sided.

### S3.6 The q-excess (1e+6) -- CONFIRMED exactly

| population | fix | mine |
|---|---|---|
| ordinary OWNED cells, 0.001-0.30 of a period, degrees 8-16 | 2.931e+00 .. 3.322e+03 | **2.931e+00 .. 3.322e+03** |
| liners that BREAK and read super-unity (1e-7 of a period) | 8.793e+06 .. 3.322e+07 | **8.794e+06 .. 3.322e+07** |
| headroom | 2.5 decades / 0.94 decades | identical |

and the SOLVED ladder underneath it reproduces the fix's S3.6 table to three
significant figures on every cell (both builds) -- see S7.

---

## S4. The censuses (task 3)

`w5_census.py`: my own 648-configuration box and my own 660-row grid, both
inside the stated ranges but with DIFFERENT specific values from the fix's.

### S4.1 False positives -- my own box, 648 configurations

3 lossy substrates (1.45+0.08i, 2.0+0.35i, 3.4+1.7i) x 2 superstrate indices
(2.4, 3.2) x 3 angles (1.22, 1.33, 1.44 rad) x degrees 6/8/10 x 2 ridge
permittivities (10.5, 8.0) x 2 or 4 slices x delta in {3e-3, 1e-3, 3e-4},
i.e. wall steps of 0.36-3.6 nm on a 1.2 um period.

| | round 1 | **round 2** |
|---|---|---|
| configurations | 648 | 648 |
| CORRECT by continuity | **648** (grey 0, wrong 0) | 648 |
| **refused although correct** | **77 (11.9 %)** | **0** |
| their `err / delta` | **0.3357 .. 8.0369** | -- |
| their `R+T` | **1.010060 .. 1.164625** | -- |
| rows carrying the round-2 TRUNCATION note | -- | **77** -- exactly the rows round 1 refused |
| returned rows BIT-identical to the unguarded answer | -- | **648 / 648** (0 broken) |

The fix reports 110 / 648 -> 0 / 648 on its own box; mine reads 77 / 648 -> 0
/ 648 with the same character (`err/delta` 0.35-8.8 there, 0.34-8.04 here;
`R+T` 1.01008-1.12328 there, 1.01006-1.16463 here).  The absolute count is a
property of which substrates and angles a box happens to contain; the
DECISION -- a double-digit percentage of correct answers refused under round
1, none under round 2 -- reproduces.

The bit-identity line is worth stating on its own: **every one of the 648
returned answers is bit-for-bit the unguarded answer.**  Round 2 changes what
is SAID about these solves and nothing about the numbers.

On this box both arbiter criteria are two-sided with room: over the 291
arbitrated rows, `su_snap` runs **6.580e-04 .. 1.655e-01** (66x above the 1e-5
closure) and `move / w_wide` runs **0.122 .. 13.553** (7.4x below the 100
move bar).  Neither criterion is close to firing on a correct row here.

### S4.2 False negatives -- my own 660-row grid, and the fix's eight rows

120 log deltas (3e-3 .. 1e-6) x degrees 10/14/20, plus 60 log deltas
(3e-5 .. 1e-6) x degrees 8/10/12/14/16, on the O-11 fixture.

| | round 1 | **round 2** |
|---|---|---|
| rows | 660 | 660 |
| RIGHT / grey / WRONG | 209 / 3 / 448 | same |
| unwarned WRONG + grey | **8** (5 wrong + 3 grey) | **4** (1 wrong + 3 grey) |
| false positives | 0 | 0 |
| correct population's envelope | **9.8668e-05** | same |

**8 -> 4 CONFIRMED**, and the eight rows are the fix's eight, to four
significant figures:

| deg | delta | kind | `err` (fix) | `err` (mine) | `err/delta` (fix / mine) | `R+T-1` (fix / mine) |
|---|---|---|---|---|---|---|
| 20 | 1.7130e-06 | wrong | 2.8003e-03 | **2.7989e-03** | 1634.8 / **1633.9** | +7.1425e-03 / **+7.1420e-03** |
| 14 | 1.5859e-06 | wrong | 1.3763e-03 | **1.3781e-03** | 867.8 / **868.9** | +6.8637e-03 / **+6.8647e-03** |
| 12 | 2.2413e-06 | wrong | 1.0906e-03 | **1.0905e-03** | 486.6 / **486.5** | +1.7138e-03 / **+1.7138e-03** |
| 8 | 1.1888e-06 | wrong | 7.7156e-04 | **7.7153e-04** | 649.0 / **649.0** | +1.9903e-03 / **+1.9903e-03** |
| 10 | 1.8854e-06 | wrong | 3.3282e-04 | **3.3247e-04** | 176.5 / **176.3** | +5.1388e-04 / **+5.1383e-04** |
| 14 | 4.7421e-06 | grey | 2.1929e-04 | **2.1940e-04** | 46.2 / **46.3** | +3.3818e-04 / **+3.3823e-04** |
| 10 | 4.6995e-06 | grey | 9.6224e-05 | **9.6211e-05** | 20.5 / **20.5** | +3.7309e-05 / **+3.7302e-05** |
| 8 | 2.3743e-06 | grey | 3.1738e-05 | **3.1741e-05** | 13.4 / **13.4** | +4.1990e-05 / **+4.1985e-05** |

The first four are REFUSED by round 2; the last four remain.

### S4.3 The four residuals -- and the one the fix could not measure

| the four | `move / w_wide` | `su_snap` | which half holds it out | fix |
|---|---|---|---|---|
| deg 10, 1.8854e-06, wrong, 176x | **423.3** | 0.0 | the **TRIGGER** (its `R+T-1` = 5.14e-04) | 423.3, agrees |
| deg 14, 4.7421e-06, grey, 46.3x | **83.3** | 2.44e-14 | the **MOVE** criterion -- no trigger reaches it | 83.3, agrees |
| deg 10, 4.6995e-06, grey, 20.5x | **21.6** | 0.0 | the **MOVE** criterion | *"not measured"* |
| deg 8, 2.3743e-06, grey, 13.4x | **120.4** | 1.12e-08 | the **TRIGGER** | 120.4, agrees |

So the fix's open item R2-A2 ("one of the four is the MOVE criterion's floor")
is really **two** of the four: the row it left unmeasured moves 21.6 cell
widths, 4.6x below the bar, so lowering the trigger would not reach it either.
Two are the trigger's floor, two are the move criterion's.  That does not
change the recommendation -- it makes the report's own accounting exact.

### S4.4 The sub-unity floor (R2-A) -- CONFIRMED, and deeper

The fix reports one WRONG row at `R+T-1` = **-2.8048e-04**.  On my
2,250-row grid:

| | fix | mine (both builds) |
|---|---|---|
| lowest WRONG-row `R+T-1` | -2.8048e-04 | **-1.13511e-03** (`C_nir`, Windows) / -1.13470e-03 (WSL) |
| WRONG rows that are SUB-unity | 1 named | **11** |
| WRONG rows at or below the 1e-3 trigger | 4 of 660 | **29 of 878** (3.3 %) |

R2-A is real, it is not a single row, and its magnitude is four times the one
the report quotes.  A detector for that band remains unfound.

---

## S5. The resonant counter-fixture (task 4, open item R2-D)

R2-D, in the round-2 report's words: *"`move > 100 * w_wide` compares an
efficiency difference with a period fraction, i.e. it silently assumes
`dR/dx = O(1)` ... a device with `dR/dx` above ~50 could in principle move a
correct answer past the bar.  No such device was found; a resonant fixture
would be the way to attack it."*  This section builds those fixtures.

### S5.1 What the move criterion actually tests

The prescribed snap merges each colliding cross-layer pair to its MIDPOINT, so
it displaces each wall by at most `w_wide` of a period.  For a solve the
sliver has NOT corrupted, the two answers therefore differ by the device's own
response to that displacement:

```
move  <=  O(1) * (dR/dx) * w_wide       =>       move / w_wide  <=  O(dR/dx)
```

`move > 100 * w_wide` is therefore, to a factor of order one, a test of
`dR/dx > 100`.  That is why it separates so cleanly from the campaign's
`err > 100 delta` WRONG rule -- **the two are nearly the same statistic**,
with the snapped solve standing in for the unavailable `delta -> 0`
reference.  It also says exactly where the criterion is attackable: a device
whose OWN physical `dR/dx` exceeds ~100.

The relation is a BOUND and not an equality, and the bound is loose in the
safe direction: under the snap the two slices' walls move in OPPOSITE
directions, so their responses can cancel.  Measured on the ordinary O-11
fixture at degree 12, `move / w_wide` is **0.2794** against a device slope of
**4.6242** -- 16.5x smaller, not equal
(`test_the_move_criterion_is_bounded_by_the_devices_own_dR_dx`).  So a steep
device is a NECESSARY condition for the criterion to misfire, not a
sufficient one, which is why the search below had to be directed rather than
random.

This is the structural statement behind R2-D, and it is stronger than the
report's own phrasing ("silently assumes `dR/dx = O(1)`"): the assumption is
not incidental, it is the whole content of the criterion.

### S5.2 The fixtures built

`w6_resonant.py` and `w9_r2d_attack.py`.

**GMR** -- a weak-contrast (`eps` 3.6 / 4.0) grating 0.30 um thick on a 0.10 um
`eps` = 4.0 slab, period 1.0 um, `n_sub` = 1.45, theta = 0.10.  The guided-mode
resonance sits at `wl` = 1.74900 um with a linewidth of about **0.5 nm**;
`dR/d(duty)` on its flank, measured by central difference and STATIONARY in
degree over 14 / 16 / 20 / 24 to four significant figures:

| `wl` (um) | `R_0` | `dR/d(duty)` at degree 14 / 16 / 20 / 24 |
|---|---|---|
| 1.748800 | 0.25084 | 72.3 / 72.3 / 72.3 / 72.3 |
| 1.748880 | 0.38504 | 106.5 / 106.6 / 106.5 / 106.6 |
| 1.748920 | 0.47251 | 124.6 / 124.6 / 124.6 / 124.6 |
| **1.749000** | 0.68134 | **145.7 / 145.8 / 145.8 / 145.8** |
| 1.749040 | 0.78993 | 138.2 / 138.3 / 138.2 / 138.5 |
| 1.749320 | 0.85506 | 85.3 / 85.3 / 85.3 / 85.4 |

so a **degree-converged `dR/dx` of 145.8** is reached and exceeds the move
bar -- the report's "a device with `dR/dx` above ~50 could in principle move a
correct answer past the bar" is CONFIRMED as a real, buildable regime.

**FP** -- two `eps` = 6.25 gratings around a 0.62 um air spacer, period 0.9 um.

**The scan** (`w6_resonant.py`, 2,028 rows): 17 GMR configurations (the
resonance flank sampled at eight wavelengths, the same device pushed into the
census box's super-unity mount at two dense superstrates, and 3- and 4-slice
staircases) plus 6 Fabry-Perot ones, x degrees 8 / 12 / 14 / 18 x 26 deltas.
Each row carries the device's OWN measured smooth-regime slope `s`, a
degree-stationarity flag on that slope, the shipped arbiter's two quantities,
and the library's actual verdict; a row is a candidate false refusal when it
is REFUSED and `err / (s * delta) <= 3`.

**The adjudication** (`w10_rcwa_oracle.py`) is what settles it, because
`err / (s * delta)` is a within-family statistic and the family is the thing
under test.  It re-solves the same 14 devices with `RCWAStack` and ANALYTIC
rectangle form factors -- a different package, a different discretization, no
union grid and no wall-collision pathology at all -- with its own `n_orders`
31 / 41 / 51 / 61 convergence ladder printed per row.

**The directed attack** (`w9_r2d_attack.py`) is the exhaustive version of the
same question, because a false refusal needs THREE things at once: `dR/dx`
above ~100, an ORDINARY truncation super-unity above the 1e-3 trigger, AND
that super-unity CLOSING on the snapped grid.  It scans 1,458 configurations
of the census box's mount (dense superstrate 2.4 / 3.2, theta 1.22-1.44,
lossy substrate, degrees 6 / 8 / 10) crossed with three periods, three
wavelengths, three duty geometries, three ridge permittivities and two slice
thicknesses, 40 deltas each, keeping only rows the shipped arbiter REFUSES.
**Result: 1,458 configurations, 2,252 refusals, and -- once the correctness
statistic is the one the LIBRARY uses -- ZERO false refusals.**  Wall 2,647 s.
The scan's own scoring flagged 11 candidates with `err / (s * delta)` between
**0.93 and 1.45** on ORDINARY devices (measured slopes 1.42-3.11, no
resonance), which would be a false refusal on an ordinary device and a much
worse finding than R2-D.  They are not.  Scored per polarization:

| config | deg | `delta` | `err/delta` pol 1 | `err/delta` pol 0 | `err/delta` of the SNAPPED answer | `move/w` | `R+T` refused / snapped / reference |
|---|---|---|---|---|---|---|---|
| P 1.6 um, `n_sup` 2.4, theta 1.22, `eps` 12 | 6 | 2.791e-06 | **5.93** | **433.95** | **1.20** | 433.9 | 1.025469 / 1.002319 / 1.002321 |
| the same | 6 | 4.208e-06 | 6.53 | **286.50** | 1.20 | 286.4 | 1.025466 / 1.002319 / 1.002321 |
| the same | 6 | 7.791e-06 | 6.38 | **153.31** | 1.20 | 153.2 | 1.025464 / 1.002317 / 1.002321 |
| P 1.6 um, `n_sup` 3.2, theta 1.33, `eps` 16 | 8 | 4.208e-06 | 13.31 | **279.29** | 1.85 | 279.8 | 1.121191 / 1.104969 / 1.104971 |

Read that: on polarization 1 -- the campaign's own `err` convention, and the
statistic this probe scored with -- the answer looks correct to 6-13x the wall
shift.  On polarization 0 it is off by **153x to 434x**.  The SNAPPED answer
tracks the exact `delta -> 0` reference to **1.20x** the wall shift and its
`R+T` agrees with the reference to six digits (1.002319 vs 1.002321), so the
prescribed remedy is right and the refused answer is wrong.

**So `w9`'s 11 "false refusals" are an artefact of MY statistic, not a defect
in the guard** -- and they are an independent, ORDINARY-device confirmation of
the round-2 design's least obvious decision: taking the move on BOTH
polarizations.  A pol-1-only move would have called 11 of 2,252 refusals
false, on dense-superstrate grazing mounts that have nothing to do with
liquid crystals.  See S7.2.

### S5.3 Outcome -- the mechanism is REAL, the published bound is REFUTED, the DECISION holds

**(a) A device with `dR/dx` far above the move bar exists and is buildable.**
The GMR grating reaches a degree-stationary `dR/d(duty)` of **169.2**, and on
its sliver rows the CORRECT population's `move / w_wide` reaches **833.78**
(`w6_resonant.py`, 2,028 rows, 936 of them on devices with slope > 20).
Against the round-2 report's "CORRECT rows move at most 26.6", that is a
**31x** refutation of the published bound.  R2-D's premise is confirmed.

**(b) 27 rows LOOK like false refusals by the slope-normalised rule.**  Rows
the shipped guard REFUSES whose error is 0.61x to 2.9x the device's own
measured smooth-regime slope times the wall step -- i.e. rows whose answer
tracks this device's physical wall shift, with `slope_stationary` True on
every one.  The cleanest: `nl3`, degree 14, `delta` = 4.7547e-05, slope 165.0,
`err / (s * delta)` = **0.61**, `su_snap` = 0.0, `move / w_wide` = 247.6.

**(c) An INDEPENDENT oracle says the guard is RIGHT anyway.**
`w10_rcwa_oracle.py` re-solves the same 14 devices with `RCWAStack` and
ANALYTIC rectangle form factors -- a different package, no union grid, no
spectral-element Jacobian, no wall-collision pathology at all -- converged to
1.7e-06 .. 6.1e-06 between `n_orders` 51 and 61, with `|R+T-1|` <= 4.6e-08:

| device | deg | `delta` | `move/w` | error of the REFUSED answer | error of the PRESCRIBED remedy | remedy / refused |
|---|---|---|---|---|---|---|
| `nl3` | 14 | 4.7547e-05 | 247.6 | 7.069e-03 (0.90x the physical shift) | **1.184e-03** (0.15x) | **0.17** |
| `wl1.748920e-06` | 14 | 2.7360e-05 | 115.8 | 3.828e-03 (1.06x) | **6.588e-04** (0.18x) | **0.17** |
| `nl4` | 8 | 1.0892e-04 | 110.4 | **1.474e-03** (0.08x) | 2.534e-03 (0.14x) | 1.72 |
| `nl3` | 18 | 1.0892e-04 | 123.5 | 9.440e-03 (0.53x) | **2.714e-03** (0.15x) | 0.29 |
| `nl4` | 14 | 1.4359e-04 | 127.5 | **2.766e-03** (0.12x) | 3.337e-03 (0.14x) | 1.21 |
| `nl4` | 8 | 8.2627e-05 | 194.8 | 3.443e-03 (0.25x) | **1.923e-03** (0.14x) | 0.56 |
| `wl1.749080e-06` | 8 | 2.0755e-05 | 137.6 | 3.318e-03 (1.01x) | **4.620e-04** (0.14x) | 0.14 |
| `nl4` | 12 | 1.8929e-04 | 223.0 | 9.679e-03 (0.31x) | **4.393e-03** (0.14x) | 0.45 |
| `wl1.749040e-06` | 14 | 4.7547e-05 | 114.3 | 4.146e-03 (0.52x) | **1.290e-03** (0.16x) | 0.31 |
| `wl1.749040e-06` | 18 | 4.7547e-05 | 144.1 | 8.140e-03 (1.01x) | **1.290e-03** (0.16x) | 0.16 |
| `nl4` | 14 | 1.0892e-04 | 431.1 | 1.312e-02 (0.74x) | **2.533e-03** (0.14x) | 0.19 |
| `wl1.748920e-06` | 18 | 6.2679e-05 | 118.2 | 5.906e-03 (0.71x) | **1.502e-03** (0.18x) | 0.25 |
| `nl4` | 12 | 8.2627e-05 | 511.8 | 1.217e-02 (0.90x) | **1.922e-03** (0.14x) | 0.16 |
| `wl1.748920e-06` | 12 | 2.7360e-05 | 140.2 | 3.176e-03 (0.88x) | **6.589e-04** (0.18x) | 0.21 |

On **12 of 14** the prescribed `min_feature` is **1.8x to 7.1x CLOSER** to the
cross-package truth than the answer the guard refuses, and on the remaining
two it is within a factor 1.7 and both answers are of comparable quality
(0.08x-0.14x of the physical shift).  The remedy's error is a flat
**0.14x-0.18x the physical shift** on every one -- which is the geometric
perturbation the snap describes, exactly as the refusal message says.

**So: R2-D is a real mechanism and a refuted BOUND, but I could not turn it
into a HARMFUL false refusal.**  Every refusal the resonant counter-fixture
produced is a refusal of an answer an independent RCWA oracle also calls
wrong, and the remedy the refusal prescribes is closer to the truth.  The
correct statement to carry forward is: *the move criterion is bounded by the
device's `dR/dx` (measured 833.8 on a resonant device against a published
26.6), and on the devices where it fires early the guard's advice is still
sound.*  The published bound should be corrected; the criterion should not be
loosened on this evidence.

**(d) The OTHER direction did produce a defect.**  The hunt for a
sliver-WRONG solve whose snapped super-unity SURVIVES -- a false
`truncation` -- succeeded, and it is not about steepness at all.  See
**D-5**.

---

## S6. The arbiter's side effects (task 5)

`w4_side_effects.py`, both builds.

**A -- no mutation.**  The stack's attributes are fingerprinted individually
before and after a guarded solve that fires the arbiter, and compared against
a CONTROL run of the same solve with `PMM_SLIVER_GUARD = False`.  On both
builds and both deltas the changed set is `['_modal']` **with** the guard and
`['_modal']` **without** it -- i.e. the only thing written is what the solve
itself writes.  `id(st._src)` and `id(st._layers)` are unchanged, `st._src`
compares equal to its pre-solve copy, `st.min_feature` is unchanged, and
`hasattr(st, '_sliver_probe')` is False afterwards.  A prepared object solved
through the same path keeps `len(_eig_cache)` = 2 and `len(_mats_cache)` = 1,
the same as a healthy prepared solve.

**B -- no recursion.**  Spying on `_sliver_probe_solve`: `delta` = 3e-3 ->
returned, **0** probes; 1e-4 -> REFUSED, **1** probe; 3e-5 -> REFUSED, **1**
probe.  A stack with `_sliver_probe = True` set by hand has
`_sliver_screen(...)` = `None` and `_within_layer_hazard(...)` = `None`, so a
probe can neither arbitrate itself nor emit the within-layer warning.
Identical on both builds.

**C -- cost.**

| | fix | Windows | WSL |
|---|---|---|---|
| guarded solve (degree 14, 2 layers), median of 7 | 86.2 / 74.1 ms | **66.99 ms** | **61.11 ms** |
| the arbiter's extra solve, median of 7 | 16.8 / 20.0 ms | **17.40 ms** | **15.52 ms** |
| ratio | 0.20x / 0.27x | **0.260x** | **0.254x** |

BOUNDED: the claim "one extra solve, cheaper than the solve it guards" holds
on both builds; the specific 0.20x is not reproduced (0.26x here, on a box
carrying other work).  The ratio is a timing measurement and should not be
pinned in a test.

**D -- the firing rate.**  Three fixtures x degrees 12/14/16/18 x 50 deltas in
the CONVERGED band (3e-3 .. 1e-4) = **600 rows**, 599 of which the continuity
rule calls RIGHT:

| | Windows | WSL |
|---|---|---|
| rows | 600 | 600 |
| RIGHT rows | 599 | 599 |
| arbiter fired | **1** | **1** |
| arbiter fired on a RIGHT row | **0** | **0** |
| worst super-unity among the RIGHT rows | 1.1231e-06 | 1.1232e-06 |

CONFIRMED: "0 of 600 converged correct rows".

**E -- the `unknown` branch.**

| case | measured (both builds) |
|---|---|
| DISPERSIVE `solve_vs_wavelength` (a callable `eps`) | `_stack_provably_passive` = **False**, `_sliver_screen` = **None**, probe runs **0x**; the solve RETURNS at `R+T` = **23.42** under the plain super-unity warning -- round 1's behaviour and pre-round-1's, unchanged |
| no resolved source | `_sliver_arbiter` returns verdict **`unknown`**, probe runs **0x**, the solve is REFUSED because `R+T` = 2.173 is above `_STACK_SUPERUNITY_BAR`, and the message contains **"could NOT be run on this path"** |
| a KEYED `prepare()` stack | not `unknown` but INERT: `_stack_provably_passive` = **False** (a `str` payload), `_sliver_screen` = **None**, **0** probes, the solve returns at `R+T` = **23.42** under the plain warning -- see **D-2** |

**F -- the `src` wiring the arbiter must re-solve at.**  Re-measured by spying
on `_sliver_probe_solve` with a deliberately STALE `set_source(4.0e-07)` on the
stack and a sweep at 8.5e-07 / 9.0e-07 (both builds):

| path | measured |
|---|---|
| `solve_vs_wavelength`, `max_workers` = 1 / 2 / 4 | REFUSED at every worker count, and the probe is handed **`[8.5e-07]`** each time -- never the stale 4.0e-07 |
| a HEALTHY sweep (delta 3e-3) at 1 / 2 / 4 workers | probed **0x**, and the returned `R` and `T` buffers are **byte-identical** across the three worker counts |
| `prepare().solve` with concrete `eps` | REFUSED, probe handed **`[8.5e-07]`** from the call (the stack's own `_src` is `None`) |

CONFIRMED: the sweep and prepared paths arbitrate at their own physics, and
the thread pool changes nothing -- `_store` runs on the calling thread in both
branches, so the extra solve is a main-thread solve.

**G -- the fail-before switch still disarms everything round 2 added.**  With
`PMM_SLIVER_GUARD = False`, the cross-layer fixture at `delta` = 3e-5 and the
1e-7 within-layer liner both RETURN, the probe runs **0x**, and each emits
exactly **one** warning -- the plain super-unity one -- with no truncation
note and no within-layer paragraph.  With the switch on, the same two stacks
are REFUSED / returned-with-two-warnings.  CONFIRMED.

---

## S7. The LC class and the within-layer arm (task 6)

### S7.1 The three tensor classes, on the report's OWN director convention

`w8_lc_exact.py` copies the round-2 test file's `_uniaxial` so the published
numbers are checked as numbers.  Degree 14, the O-11 stack, both builds:

| tensor | exactly Hermitian? | `_segment_passive` | delta 1e-4 | delta 3e-5 | sliver-free control |
|---|---|---|---|---|---|
| in-plane director, 45 deg | **Windows yes / WSL no** (`lam_min/scale` = -4.263e-17 = 0.192 ULP on WSL) | yes | returned, `move` = **38.86** (report 38.9) | **REFUSED**, `move` = **290.36** (report 290.4), err **288.4x**, `R+T` **1.01054** | returned, `R+T` = 1 |
| out-of-plane director, 30 deg | **no**, `lam_min/scale` = **-2.994e-17** = 0.135 ULP | yes | returned, `move` = **22.18** (report 22.2) | **REFUSED**, `move` = **316.46** (report 316.5), `R+T` **1.01330** | returned, `R+T` = 1 |
| gyrotropic `eps_xy = -eps_yx = 0.35i` | yes | yes | **REFUSED**, err **685.4x** (report 685x) | **REFUSED**, err **2680x** (report 2680x) | returned, `R+T` = 1 |
| LOSSY in-plane director, `kappa` = 0.2 | no, `lam_min/scale` = **+0.227** | yes | returned, `R+T` = 0.8809 | **REFUSED**, err **8005x** | returned, `R+T` = 0.8807 |
| NON-Hermitian `eps_xy` = 0.2 | no, `lam_min/scale` = **-2.500e-02** | **no** | returned (unchanged), `su_snap` **6.31e-04** | returned (unchanged), `R+T` = 1.09311 | returned |

Every published figure reproduces.  The negative control does too: the
non-Hermitian payload is not provably passive, the screen never fires, and it
keeps the behaviour it had at `R+T` = 1.093.  The LOSSY director -- which the
report asserts is in the class but does not tabulate -- is reached.

**The build split the report anticipated is real, and it is the same one.**
The 45-degree in-plane director tests EXACTLY Hermitian on Windows
(`np.array_equal(M, M.conj().T)` is True) and NOT on WSL, where its
anti-Hermitian part's smallest eigenvalue reads **-4.263e-17 = 0.192 ULP**.
Both builds read `provably_passive` = True, because what ships is the
semi-definiteness test with its round-off deadband; an exact-equality test
would have shipped a guard that fires on one build and not the other.  Every
other number in the table above agrees between the builds to 6+ figures
(`move` 290.3504 vs 290.3611, 316.4555 vs 316.4555, 685.5719 vs 685.5719).

### S7.2 The pol-0 evidence -- CONFIRMED, and it changes the decision

The out-of-plane director at `delta` = 3e-5, scored against the exact
`delta -> 0` limit:

| | fix | measured here |
|---|---|---|
| pol 1, `err / delta` | "0.01x" | **0.010551** |
| pol 0, `err / delta` | "316x" | **316.4** |
| `move / w_wide`, pol 1 only | -- | **0.01055** -- 9,500x BELOW the bar |
| `move / w_wide`, both pols | 316.5 | **316.46** -- 3.2x above it |
| verdict | REFUSED | **REFUSED** |

and on my OWN out-of-plane director (a different rotation convention) the same
structure appears at a different delta: at 1e-5, pol 1 reads 6.05x, pol 0
reads 145.2x, `move` on pol 1 alone is 6.06 and on both is 145.26, and the
library REFUSES.  A pol-1-only move would have left the refusal unfired in
both cases.  **CONFIRMED: the arbiter uses both polarizations, and it is
load-bearing.**

**And the class is wider than the report claims.**  The round-2 report
presents the both-polarizations move as an anisotropic-stack refinement ("this
class is where taking the move on BOTH polarizations earns its keep").  The
1,458-configuration directed scan found the same asymmetry on ISOTROPIC
stacks: at period 1.6 um, `n_sup` = 2.4, theta = 1.22 rad, `eps_ridge` = 12,
degree 6, `delta` = 2.791e-06, polarization 1 is off by **5.93x** the wall
shift while polarization 0 is off by **433.95x**, and the snapped answer
tracks the exact limit at **1.20x**.  Eleven such rows appeared in 2,252
refusals.  A pol-1-only move would have called every one of them a false
refusal, on a mount with no birefringence anywhere in it.  This is the
strongest single piece of evidence for the shipped design that this
verification found.

### S7.3 The within-layer ladder

`w7_lc_within.py`, degrees 8 / 12 / 14 / 16, reference exact (the liner
vanishes as `d -> 0`):

| liner `d` | q-excess @ deg 14 | `err`, degrees 8 / 12 / 14 / 16 | `R+T` range | library WARNS? |
|---|---|---|---|---|
| 1e-2 | 2.565e+02 | 1.303e-02 (flat) | 1 | no |
| 1e-3 | 2.565e+03 | 8.793e-04 (flat) | 1 | no |
| 1e-4 | 2.565e+04 | 8.47e-05 (flat) | 1 | no |
| 1e-5 | 2.565e+05 | 1.53e-05 / 1.23e-05 / 2.69e-06 / 9.11e-06 | 0.999998 .. 1.000008 | no |
| **1e-6** | 2.565e+06 | 5.64e-05 / 7.44e-04 / **1.058e-03** / 3.34e-04 | **0.999221 .. 1.000355** | **no** |
| **1e-7** | 2.565e+07 | 5.29e-02 / 6.60e-01 / 6.54e-01 / **1.054** | **0.570636 .. 4.185373** | 3 of 4 degrees |

Every cell reproduces the report's S3.6 table to three significant figures,
including the floor it states: at `d` = 1e-6 the answer is **1.058e-03** wrong
while `R+T` reads **0.999221**, SUB-unity, and nothing fires.

The two-sided half is confirmed -- the arm is silent on 1e-2 .. 1e-5 -- and
the floor is SHARPER than the report says: at the catastrophic 1e-7 width the
warning is silent at degree 14 as well, because that degree reads `R+T` =
**0.5706**.  So the arm is silent on one of the four degrees at the width
where the answer is off by 100 %.  That is the same theorem-shaped floor, one
level worse than stated, and it should be written down as such.

### S7.4 Can an OWNED liner be REFUSED through the cross-layer path?

**No.**  Three constructions at widths 1e-3 / 1e-5 / 1e-6 / 1e-7 -- the liner
alone in every layer; the liner plus an ordinary non-conforming wall 1 % of a
period away in the last layer; two layers whose liners are OFFSET by `d/2` --
all return, and `_cross_layer_sliver` reports **no hit** in every one of the
twelve.  The ownership rule and the own-scale ratio between them keep the
caller's own geometry out of the refusal path, exactly as designed.

The converse, however, is a defect: see **D-1**.

---

## S8. Defects

| | severity | new in round 2? | one line |
|---|---|---|---|
| **D-5** | **MEDIUM-HIGH** | **yes** | the closure bar is ABSOLUTE, so a snap that removes a 5,181x super-unity and restores the answer exactly is still read as `truncation` -- three wrong answers round 1 refused are returned with a message that is measurably false |
| D-1 | MEDIUM | no | one thin OWNED feature lowers the GLOBAL own-scale and disarms the cross-layer refusal for the whole stack |
| D-2 | LOW | no | a KEYED `prepare()` stack is not `provably_passive`, so the screen is never reached |
| D-3 | LOW (test-only) | yes | one bar in the round-2 test file is a SAMPLE property (the family reaches 147.41 against a demanded 300) |
| D-4 | LOW (doc-only) | yes | three published numbers are sample-scoped: 9.11x -> 8.03x, `move` >= 466 -> 147.41, one sub-unity wrong row -> 11 |

### D-1 (MEDIUM, pre-existing, round 2 inherits it) -- one thin OWNED feature disarms the cross-layer refusal for the whole stack

`_cross_layer_sliver` computes `own` as the **global** minimum wall spacing
over ALL layers, and flags a manufactured cell only when `own / w >= 100`.  A
single sliver-thin feature that ONE layer legitimately owns therefore lowers
`own` for every manufactured cell in the stack, and the screen goes silent on
genuine cross-layer slivers that would otherwise be refused.

Measured (`w8_lc_exact.py` arm B, Windows; WSL identical to 6 figures).  The
screen is pure geometry, so the `own` and `own / w` columns are
degree-independent; the `R+T` columns are the solve's.

| stack | `own` | manufactured cells | `own / w` | screen | `R+T` deg 12 | `R+T` deg 14 | verdict |
|---|---|---|---|---|---|---|---|
| the O-11 sliver, delta 3e-5 | 0.2786 | 3.000e-05 x2 | **9287.3** | fires | 2.1726 | 23.4225 | **REFUSED** at both |
| the same, plus a 1e-6-of-a-period liner ONE layer owns | **1e-06** | 3.000e-05, 2.900e-05 | **0.03** | **silent** | **23.3037** | 3.5356 | **returned** at both |

At degree 12 the lined stack is 10.7x FURTHER from the theorem than the bare
one the guard refuses (23.30 vs 2.17) and is returned; at degree 14 both are
catastrophic and only the bare one is refused.  Which of the two reads worse
is a per-degree accident -- the point is that the refusal is switched off by a
feature that has nothing to do with the cross-layer pair.

Reproducer, one segment added to the existing fixture:

```python
segs = [(a, EH), (b - a, eps), (liner, 9.0), (1.0 - b - liner, EH)]  # liner=1e-6
# ... the same two-slice staircase, delta = 3e-5, degree 12
ps._cross_layer_sliver([L[1] for L in st._layers], 1e-12)   # -> None
st.solve()                                                   # returns R+T = 23.30
```

The caller is not left blind -- the round-2 WITHIN-LAYER warning fires on the
same stack -- but the message names the liner, not the cross-layer sliver, and
the solve returns a number that violates the theorem by a factor of 23.
Round 1 behaves identically (the screen is unchanged), so this is not a
regression and it does not block the release.  It should be stated as an open item next to R2-B,
because S3.6's "never refused -- it is the geometry the caller asked for"
reads as if the within-layer arm only ADDS a warning, and in fact its presence
also REMOVES a refusal.

The obvious repair -- score `own` per flagged cell against the finest spacing
of the layers that own that cell's neighbours, rather than globally -- is a
library change and is out of scope here.

### D-2 (LOW, pre-existing) -- a KEYED `prepare()` stack is outside the guard entirely

`_segment_passive` returns False for a `str` payload, so
`_stack_provably_passive` is False for any stack carrying material KEYS --
which is the case `prepare()` exists for ("prepare() swaps material KEYS").
Measured: a keyed prepared stack with the O-11 sliver at delta 3e-5 solves to
`R+T` = **23.42** with `screen` = False, `provably_passive` = False, **0**
probes, and only the plain super-unity warning.

The report's S4.4 row `prepare().solve` -> "REFUSED with a measured
attribution" is therefore true only of a KEY-FREE prepared stack, and its test
`test_the_prepared_path_arbitrates_at_the_wavelength_it_was_given` uses
concrete `eps` (`_stack(1e-4, 14)`).  R2-C already covers keyed stacks in
principle; the S4.4 table should say which of the two prepared cases it
measured.  `_PreparedPMMStack.solve` resolves `materials` before it solves, so
handing the resolved tensors to the guard is available without materialising
anything -- again, a library change, out of scope.

### D-5 (MEDIUM-HIGH, NEW IN ROUND 2) -- the closure criterion is ABSOLUTE, so a snap that restores the answer completely is still not believed

`_SLIVER_ATTRIB_CLOSURE` asks the snapped super-unity to fall below a FIXED
`1e-5`.  On any stack whose SLIVER-FREE truncation super-unity already sits
ABOVE that bar, the criterion **can never be met**, however completely the
snap restores the answer -- so the arbiter says `truncation`, the wrong number
is RETURNED, and the message tells the caller the opposite of what is true.

Reproducer: `validation/probe_verify_sliver_round2/w11_closure_absolute.py`,
and `test_the_closure_criterion_is_absolute_not_relative`.  A guided-mode
grating in a dense-superstrate grazing mount (period 1.0 um, `n_sup` = 2.4,
`n_sub` = 1.45+0.05i, theta = 1.22, `wl` = 0.93 um, degree 8) whose
sliver-FREE super-unity is ordinary truncation -- the degree ladder is clean,
`R+T-1` = 2.078e-04 / **3.7295e-05** / 9.832e-06 / 2.961e-07 / 3.153e-09 at
degrees 6 / 8 / 10 / 12 / 14 -- and whose degree-8 floor therefore lands
BETWEEN the closure bar and the trigger.  The stack is `provably_passive`, so
`R+T <= 1` is a theorem.

| `delta` | `R+T-1` returned | `R+T-1` on the prescribed grid | drop | `move/w_wide` | `err/delta` returned | `err/delta` snapped | verdict |
|---|---|---|---|---|---|---|---|
| 6.8726e-06 | **+1.9267e-01** | +3.7294e-05 | **5,166x** | 13,665 | **1811** | **0.00193** | `truncation`, RETURNED |
| 5.2134e-06 | **+1.9323e-01** | +3.7295e-05 | **5,181x** | 18,251 | **2399** | **0.00193** | `truncation`, RETURNED |
| 3.0000e-06 | **+2.3157e-02** | +3.7295e-05 | **621x** | 3,501 | **3501** | **0.00193** | `truncation`, RETURNED |

Every digit printed above is IDENTICAL on WSL
(`w11_closure_absolute_wsl.json`): the same three verdicts, the same 5,166x /
5,181x / 621x drops, the same 0.00193, and a degree ladder agreeing to 10
significant figures.  This is not a knife-edge.

Read that row by row: the returned answer violates the energy theorem by
**19 %** and is off by **1,811x to 3,501x the physical wall shift**; the
prescribed `min_feature` puts it back on the sliver-free reference to
`err/delta` = **0.00193**, i.e. to five decimal places; the move is 35x-183x
past the move bar.  Every piece of evidence the arbiter needs is present.  The
arbiter nevertheless returns `truncation`, and the warning says

> *"the super-unity SURVIVES at 1+3.73e-05 ... Raising min_feature will
> silence nothing here -- reduce n_slices or raise degree."*

which is measurably false: raising `min_feature` is exactly what fixes it.

**This is a round-2 behaviour change, not an inherited floor.**  Round 1
refuses all three rows (the screen fires and `R+T-1` = 0.19 is above
`_STACK_SUPERUNITY_BAR`).  Round 2 converts three correct refusals into three
returned wrong answers carrying an actively misleading diagnosis.

**The repair is measured and is one line.**  Make the closure RELATIVE -- the
snap must remove most of the violation rather than reach a fixed floor.  The
two populations separate by five decades on that statistic:

| population | super-unity DROP factor on the snapped grid |
|---|---|
| the 291 arbitrated CORRECT rows of the false-positive census box | **0.9023 .. 2.289** (median 1.011); 0 rows above 100x |
| the 448 arbitrated WRONG rows of the false-negative grid | **>= 1.776e+05** (p1 1.046e+08, median 1.833e+14); 0 rows below 1e3 |
| the three D-5 rows | 621x .. 5,181x |

so a criterion of the shape

```python
attributed = (su <= max(_SLIVER_ATTRIB_CLOSURE,
                        (worst - 1.0) * _SLIVER_CLOSURE_FRACTION)
              and move > _SLIVER_MOVE_FACTOR * w_wide)
```

with `_SLIVER_CLOSURE_FRACTION` anywhere in 1e-2 .. 1e-3 (i.e. "the snap must
remove 99 % to 99.9 % of the violation") catches all three D-5 rows with
6.2x-52x of margin while refusing NONE of the 291 correct rows -- 44x-437x of
margin on that side.  That is a two-sided gap of five decades, against the
absolute bar's 66x.  Not applied here: this is a verification, and no
`lumenairy/` file was touched.

**Severity.**  MEDIUM-HIGH rather than HIGH because the caller is still
warned that `R+T` = 1.19 and "the result is unreliable" -- the plain
super-unity warning fires as it always did, so nothing goes silently wrong.
What is wrong is the ATTRIBUTION and the remedy the message names, which is
the entire subject of round 2.

### D-3 (LOW, test-only) -- one bar in the round-2 test file is a SAMPLE property, not a family property

`test_the_arbiter_separates_the_two_causes_on_this_build` asserts

```python
assert min(s[1] for s in sliver) >= ps._SLIVER_MOVE_FACTOR * 3.0     # >= 300
```

over five hand-picked rows.  Re-measured, those five rows read `move / w_wide`
= 4789.5 / 287271.7 / 15956.0 / 15956.5 / 47742.9, so the assertion holds with
16x to spare **on that sample**.  The FAMILY reaches **147.41** (S3.3), i.e.
2.0x below the bar the test demands.  The test cannot flake today -- its rows
are deterministic -- but its docstring presents the five rows as "the two
populations the two arbiter bars must separate", and that reading is not
supported by the family.  This is the same shape the round-1 verification
raised against `_LADDER`, one level milder.

**RESTATED in this branch** (commit `667f416`): the two move assertions
now read

```python
assert min(s[1] for s in sliver) > ps._SLIVER_MOVE_FACTOR, sliver
assert max(t[1] for t in trunc) < ps._SLIVER_MOVE_FACTOR, trunc
assert min(s[1] for s in sliver) > 10.0 * max(t[1] for t in trunc)
```

-- each population on its own side of the bar (the DECISION), plus a decade of
separation between the two populations the test itself measured (a property of
the populations, not of the constant).  The family numbers and the date are in
the comment above them.  Measured after the restatement: the separation is
4789.5 / 3.451 = **1,388x**, so the decade demanded carries 139x.

The other three bars in that test are sound with room: the sliver rows'
`su_snap` <= **2.465e-13** against a 1e-6 demand (6.6 decades), the truncation
rows' `su_snap` >= **2.543e-03** against 1e-4 (25x), and their `move` <=
**3.451** against 33.3 (9.7x).

### D-4 (LOW, doc-only) -- three published numbers are sample-scoped

* the trigger's "9.11x" headroom is 8.03x once two more fixtures are added
  (S3.1);
* `move / w_wide >= 466.2` for the wrong population is 147.41 on a denser grid
  (S3.3) -- the published margin of 4.66x is really 1.47x;
* the sub-unity WRONG row at -2.8048e-04 is one of at least **11**, the worst
  reading -1.1351e-03 (S4.4).

None changes a decision; all three make the guard's floors look further away
than they are.

---

## S9. Test durability (task 7)

Re-timed on both builds (one BLAS thread):

| file | tests | fix's Windows / WSL | mine, Windows / WSL |
|---|---|---|---|
| `test_fix_pmmstack_sliver_walls_round2.py` (one bar restated here) | 19 | 24.66 / 27.94 s | **28.02 / 26.17 s** (19 passed, both) |
| the three sliver files together | 53 | -- | -- / **64.43 s** (53 passed) |
| all four with `test_m1_conditioning_guard.py` | 80 | 76.03 / 77.05 s | **82.91 s** (80 passed, 1 warning) |
| all FIVE, i.e. with `test_verify_pmmstack_sliver_round2.py` | 86 | -- | **77.78 / 76.55 s** (86 passed, 1 warning, both) |
| `test_verify_pmmstack_sliver_round2.py` alone (NEW) | 6 | -- | **4.05 / 3.41 s** (6 passed, both) |

The one warning in every run is the pre-existing
`test_return_owners_is_additive_and_warn_false_is_silent`'s deliberate
`_pmm_union_grid` snap warning, which that test raises on purpose.  WSL also
prints the two pre-existing `** On entry to DLASCL parameter number 4 had an
illegal value` lines from `test_m1_conditioning_guard.py`, exactly as the
round-2 report records; none of the four sliver files produces any.

### S9.1 Constant-by-constant

| where | constant | origin stated? | re-measured here | gap both sides |
|---|---|---|---|---|
| `_STACK_SUPERUNITY_BAR` = 1e-2 | pre-existing since v5.14 | yes | correct envelope 1.245e-04, wrong population reaches +5.14e-04 | one-sided only -- this bar is now the WARNING bar, not a decision bar |
| `_SLIVER_TRIGGER_BAR` = 1e-3 | 720 rows / 3 fixtures | yes | **1.2448e-04 over 826 rows / 5 fixtures -> 8.03x**; below it, 29 of 878 wrong rows | two-sided, 8.0x above / 1.0x below (the wrong population touches it) |
| `_SLIVER_ATTRIB_CLOSURE` = 1e-5 | 1,133 arbitrated rows | yes | **6.9e+04x above the wrong population, 66x below the truncation one** | two-sided by decades |
| `_SLIVER_MOVE_FACTOR` = 100 | the campaign's `err > 100 delta` rule | yes | **7.4x below the arbitrated correct rows, 1.47x below the wrong ones** | two-sided, but the lower side is 0.17 of a decade |
| `_SLIVER_OWN_SCALE_RATIO` = 100 | 4,192 random stacks | yes | **3.93x** over 27,904 cells | two-sided, 0.6 decades |
| `_PASSIVE_ANTIHERM_DEADBAND` = 16 ULP | 200k + 50k directors | yes | **25.5x** above round-off, **9 decades** below `Im(n)` = 1e-6 gain | two-sided by decades |
| `_SLIVER_Q_EXCESS` = 1e+6 | ordinary vs breaking liners | yes | **2.5 decades / 0.94 decades** | two-sided |
| test: `min(sliver move) >= 300` (BEFORE) | five hand-picked rows | yes | sample 4789.5, family **147.41** | **sample-scoped -- D-3** |
| test: the same, RESTATED here (`667f416`) | the two populations the test measures | yes | separation 4789.5 / 3.451 = **1,388x** against a demanded 10x | two-sided, 139x |
| test: `max(sliver su_snap) <= 1e-6` | same five | yes | 2.465e-13 | 6.6 decades |
| test: `min(trunc su_snap) >= 1e-4` | three rows | yes | 2.543e-03 | 25x |
| test: `max(trunc move) <= 33.3` | three rows | yes | 3.451 | 9.7x |
| test: `envelope < _SLIVER_TRIGGER_BAR` | 24 rows, re-derived at runtime | yes | family 1.2448e-04 -> 8.0x | one-sided by design; sound |
| test: `_PASSIVE_ANTIHERM_DEADBAND < 1e-13` | -- | implicit | 3.553e-15, 28x | sound |

### S9.2 Shapes S1-S5

| shape | present? | where |
|---|---|---|
| S1 magnitude-ratio defect pin | **yes, once -- FIXED in this branch** | D-3 (`min(sliver move) >= 300`), restated in `667f416` as a population separation |
| S2 pre-fix-referencing arm | no -- `test_fail_before_round1_...` forces round 1's `unknown` branch through the CURRENT code path (`st._src = None`), so it does not read a prior build | |
| S3 env-dependent precondition | no -- the thread-pool test SETS `max_workers` rather than checking the box, and never skips | |
| S4 floor bar | no bar in the three files is closer than 8x to its population, D-3's included after the restatement | |
| S5 exact count / set of nondeterministic machinery | **no** -- `len(fired) == 3`, `len(calls) == 1` and `seen == [_WL]` are all counts of deterministic loop bodies and spy hits, not of a mode census; every population statement is an inequality (`>= 40`, `>= 8`, `>= 4`, `>= 10`, `<= max(1, wrong // 5)`) | |

No `pytest.skip` on a resource check anywhere in the three files.

### S9.3 The new file's own bars

`tests/unit/test_verify_pmmstack_sliver_round2.py` (6 tests) is held to the
same standard.  Every numeric bar in it and its two-sided margin, measured
on both builds 2026-09-11:

| assertion | measured | margin |
|---|---|---|
| `own / w >= _SLIVER_OWN_SCALE_RATIO` on the bare fixture | **9287.3** | 92.9x |
| the lined stack's screen is `None` | its `own / w` is **0.03** | 3,300x below |
| `r_lined[3] > r_bare[3]` (the lined stack is no better) | **23.3037** vs **2.1726** | 10.7x |
| the keyed prepared stack's `R+T` above `_STACK_SUPERUNITY_BAR` | **23.42** | 2,300x |
| `slope < 10` (the move test's premise) | **4.6242 / 4.7102** | 2.1x |
| `ratio <= 3 * slope` (the move BOUND) | **0.2794 / 0.2865** against 13.87 / 14.13 | 50x |
| `ratio < _SLIVER_MOVE_FACTOR` | **0.2794** | 358x |
| `err >= 100` and silent (the within-layer arm) | 4 qualifying (width, degree) pairs of 8, best `err/w` = **6.5e+06** | 65,000x |
| the D-5 ladder is monotone and its degree-8 floor is between the closure and the trigger | 2.078e-04 > **3.7295e-05** > 9.832e-06; 3.7x above 1e-5, 27x below 1e-3 | 3.7x / 27x |
| the D-5 drop `> 100` | **621x .. 5,181x** | 6.2x |
| the D-5 `move / w_wide > _SLIVER_MOVE_FACTOR` | **3,501 .. 18,251** | 35x |
| `err > 100` and `err_snapped < 1` on the D-5 rows | 1,811-3,501 and **0.00193** | 18x / 518x |
| `round1_would >= 4` in the box arm | 12 of 24 | 3x |

Two shapes were deliberately avoided.  The D-5 arm CONTINUES past a row that
does not meet its premise (`err > 100 and err_snapped < 1`) and asserts only
`len(found) >= 2` of three, so a build that moves one row out of the band does
not fail it -- the third delta of the four in the probe already behaves that
way here.  The within-layer arm scans 8 (width, degree) pairs and asserts an
EXISTENCE, not a census, for the same reason.

---

## S10. Ship recommendation for 5.45.0

**SHIP round 2, and land the D-5 one-line repair first.**

Round 2 does what it says.  Its central claim -- that super-unity DETECTS but
does not ATTRIBUTE, and that one extra solve supplies the attribution -- is
confirmed on every axis I could re-measure: bit-identity 31/31 on both builds
with a non-vacuous warning half; the false-positive census reproduces
(77/648 -> 0/648 on my own box, all 648 returned answers bit-identical to the
unguarded ones); the false-negative census reproduces row for row (8 -> 4,
every one of the eight rows to four significant figures); the arbiter costs
0.25-0.26x of the solve it guards, fires 0 times on 600 converged correct
rows, mutates nothing, and never recurses; the anisotropic extension reaches
the whole liquid-crystal class with its negative control intact, and its
both-polarizations move is load-bearing (a pol-1-only move reads 0.0106 where
both read 316.46, on the row that must be refused).  Two of the three bars are
two-sided by decades on populations four times denser than the ones that set
them.

Three things should change before or alongside the tag:

1. **D-5 (blocking-ish).**  The absolute closure bar converts three refusals
   that round 1 made correctly into returned wrong answers -- off by 1,811x to
   3,501x the physical wall shift, at `R+T` = 1.19 -- carrying a message that
   says the prescribed remedy will not help, when measurement says it restores
   the answer to `err/delta` = 0.002.  This is the one place round 2 is worse
   than round 1.  The repair is a relative closure whose two populations
   separate by five decades (drop factor 0.90-2.29 correct vs >= 1.78e+05
   wrong), it costs no extra solve, and it is a one-line change to an
   expression that already exists.  I would land it, re-run the three sliver
   files and the census, and then tag.  Shipping without it is defensible --
   the plain super-unity warning still fires and nothing goes silently wrong
   -- but it ships a message that is known to be false on a reachable class.

2. **D-4 (documentation).**  Correct the three sample-scoped numbers in the
   round-2 report: the trigger's headroom (9.11x -> 8.03x over five
   fixtures), the wrong population's `move / w_wide` floor (466.2 -> 147.41,
   so the bar carries 1.47x and not 4.66x), and the sub-unity wrong row
   (one at -2.80e-04 -> at least 11, worst -1.14e-03).  Also correct the
   published "CORRECT rows move at most 26.58": on a resonant device a
   correct row reaches **833.78**.  None of these changes a decision; all of
   them make the guard's floors look further away than they are, which is the
   defect the round-1 verification raised in the first place.

3. **D-3 (test).**  Restate `min(sliver move) >= 300` in
   `test_the_arbiter_separates_the_two_causes_on_this_build` as a separation
   between the two populations it measures, or record in the comment that 300
   is a property of its five rows and the family reaches 147.41.

`D-1` and `D-2` are pre-existing, are inherited rather than introduced, and
should be added to the open-items table (R2-B and R2-C respectively) rather
than fixed in this release.

**R2-D should be re-stated, not closed.**  The counter-fixture the report
asked for exists: a guided-mode-resonance grating with a degree-stationary
`dR/d(duty)` of 169.2, on which the CORRECT population's `move / w_wide`
reaches 833.78.  The published bound is refuted by 31x.  The DECISION
survived: on all 14 rows adjudicated against an independent RCWA oracle, the
answers the guard refuses are wrong, and on 12 of 14 the prescribed
`min_feature` is 1.8x-7.1x closer to the truth.  So the criterion should keep
its bar and lose its published margin.

---

## S11. The runs

All with `PYTHONPATH` on the worktree and one BLAS thread.

| run | Windows | WSL |
|---|---|---|
| `tests/unit/test_verify_pmmstack_sliver_round2.py` (new, 6 tests) | **6 passed**, 4.05 s | **6 passed**, 3.41 s |
| `tests/unit/test_fix_pmmstack_sliver_walls_round2.py` after the D-3 restatement | **19 passed**, 28.02 s | **19 passed**, 26.17 s |
| the three sliver files | -- | **53 passed**, 1 warning, 64.43 s |
| the three sliver files + `test_m1_conditioning_guard.py` | **80 passed**, 1 warning, 82.91 s | -- |
| all FIVE (those four plus the new file) | **86 passed**, 1 warning, 77.78 s | **86 passed**, 1 warning, 76.55 s |
| every test file importing `PMMStack` (`grep tests/ --include='*.py' -l PMMStack`, **43** files incl. this round's) | see below | -- |
| `ruff check lumenairy/ tests/ validation/probe_verify_sliver_round2/` | **All checks passed** | -- |

**REGRESSION.**  `grep tests/ --include='*.py' -l PMMStack` reads **43** files
on this branch (the round-2 report's 42 plus
`tests/unit/test_verify_pmmstack_sliver_round2.py`); the list is in
`validation/probe_verify_sliver_round2/_pmmstack_test_files.txt`.

**COMPLETED AFTER THIS REPORT WAS CLOSED (addendum 2026-09-11): 819 passed,
1 skipped, 0 failed, 76 warnings in 2628.77 s (43:48)** -- the full transcript
replaces the partial one at
`validation/probe_verify_sliver_round2/_regression_win.txt`.  The count is
ONE MORE than the 818 the paragraph below arrives at by arithmetic; it is a
collection difference (one more test collected on the final run than the
812 + 6 sum assumed), not a decision -- zero failures, and the single skip is
the same pre-existing `test_niche_audit_m4_m5_m6_rcwa.py:387` threadpoolctl
skip.  The paragraph below is kept as it was written.

The 43-file run was launched on this branch and was still executing when this
report was written.  The reason is the box, not the tests: 24 logical CPUs
shared with several other agents' pytest suites and probe sweeps (one of them
holding 14 GB), so the run accumulated 1,324 s of CPU over 3 h of wall time --
about 12 % of one core.  Nothing failed; it did not finish.  Its partial output is committed as
`validation/probe_verify_sliver_round2/_regression_win.txt` -- **53 % of the
collected tests, one `s`, zero `F` and zero `E`** at the point this report was
closed, the single skip being the pre-existing
`test_niche_audit_m4_m5_m6_rcwa.py:387` threadpoolctl skip that the round-1,
verification and round-2 regressions all record.

What IS green on this branch, on BOTH builds:

| | Windows | WSL |
|---|---|---|
| the three sliver files + `test_m1_conditioning_guard.py` + the new file | **86 passed**, 1 warning, 77.78 s | **86 passed**, 1 warning, 76.55 s |
| `test_fix_pmmstack_sliver_walls_round2.py` after the D-3 restatement | **19 passed**, 28.02 s | **19 passed**, 26.17 s |

and for the wider gate the reference point is the round-2 report's own run on
the same library (**812 passed, 1 skipped, 0 failed in 43:19**, 42 files),
which this branch cannot have moved: `git diff 24651c8 HEAD -- lumenairy/` is
EMPTY, so the only file this branch adds to that gate is
`tests/unit/test_verify_pmmstack_sliver_round2.py`, whose 6 tests pass on both
builds in 4.05 / 3.41 s, and the one restated assertion in
`test_fix_pmmstack_sliver_walls_round2.py`, whose file passes on both.  The
expected reading is therefore **818 passed, 1 skipped** -- but I did not
observe it, and it should be observed before the tag.

The rerun command, for the record:

```
cd /c/tmp/lum_vsliver2
PYTHONPATH=/c/tmp/lum_vsliver2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1   MKL_NUM_THREADS=1 python -m pytest   $(cat validation/probe_verify_sliver_round2/_pmmstack_test_files.txt)   -q --no-header -p no:randomly
```

`.test_durations` spliced with the six measured Windows timings of the new
file (12,612 -> 12,618 entries; a 6-line diff, nothing reformatted).

### S11.1 Probe wall times

| probe | Windows | WSL |
|---|---|---|
| `w1_bitid.py` (x2 arms) | ~40 s | ~35 s |
| `w2_populations.py` (2,250 rows) | 289 s | 304 s |
| `w3_bars.py` (250,000 tensors + the liner ladder) | 33 s | 38 s |
| `w4_side_effects.py` (600-row firing box) | 161 s | 150 s |
| `w5_census.py` (648 + 660 rows) | 367 s | 402 s |
| `w6_resonant.py` (2,028 rows) | 1,004 s | -- |
| `w7_lc_within.py` | 27 s | 25 s |
| `w8_lc_exact.py` | 21 s | 19 s |
| `w9_r2d_attack.py` (1,458 configurations x 40 deltas) | 2,647 s | -- |
| `w10_rcwa_oracle.py` (14 rows x 5 RCWA solves) | 233 s | -- |
| `w11_closure_absolute.py` | 6 s | -- |

---

## S12. Commits (branch `verify/sliver-round2`)

| | |
|---|---|
| `f50bf23` | `probe(verify sliver r2)` -- the independent re-measurement suite: 31 bit-identity fixtures and the probes `w1`-`w8`, both builds |
| `57abd4d` | `test(pmm)` -- `tests/unit/test_verify_pmmstack_sliver_round2.py`, six tests, four of them pinning a known limitation with the re-pin instruction; `.test_durations` spliced (12,612 -> 12,618) |
| `64e8e94` | `probe(verify sliver r2)` -- the resonant counter-fixture (`w6`), its RCWA adjudication (`w10`), the directed R2-D scan (`w9`) and the D-5 reproducer (`w11`) |
| `667f416` | `test(pmm)` -- the round-2 file's move bar restated as a population SEPARATION (D-3) |
| `dc95864` | `docs(audits)` -- this report |
| `dae9f58` | `docs(audits)` -- D-5 reproduces digit for digit on the second build; the call-site diff, the fail-before arm, the centred-overlap measurement |
| `6ceaaa1` | `docs(audits)` -- the runs, the ship recommendation, and the two items this verification did not complete |

No `lumenairy/` file was changed on this branch -- `git diff 24651c8 HEAD --
lumenairy/` is EMPTY -- and nothing was merged, pushed, tagged or
version-bumped.

### S12.1 What this verification did NOT complete

| | why | what stands in for it |
|---|---|---|
| the 43-file `PMMStack` regression | starved on a shared 24-CPU box (12 % of one core over 3 h) at the time of writing; **it finished after the report closed: 819 passed, 1 skipped, 0 failed in 43:48** (S11 addendum) | the 86-test five-file set green on both builds, and `git diff 24651c8 HEAD -- lumenairy/` EMPTY; now also the completed run itself |
| an RCWA adjudication of the 11 `w9` candidates | the two packages disagree by 9.7e-02 on this mount's `delta` = 0 reference itself (a dense superstrate at 1.22 rad, where the PMM family's `kz_inc = Re(kz_sup)` normalization and the RCWA one need convention work I did not do) | the internal adjudication is decisive anyway: the SNAPPED answer tracks the exact `delta -> 0` limit to **1.20x** and its `R+T` matches the reference to six digits, so the refused answer is the wrong one |
| the report's "the snapped grid resolves a different order count on 359 of 637 arbitrated rows" | not reproducible without their exact grids | measured on mine instead: 0 of 100 on clean fixtures, **69 of 69** on the census box -- the CONTRACT is confirmed, the count is theirs |
| the report's own 110 / 648 false-positive count | their box's specific substrates and angles | my own 648-configuration box: **77 / 648 -> 0 / 648**, same character |
| the arbiter's 0.20x Windows cost | a timing measurement on a contended box | **0.260x / 0.254x**, i.e. the claim "cheaper than the solve it guards" holds |
| `w6` / `w10` on the second build | Windows only (each is a ~20-minute run) | the D-5 reproducer `w11` IS cross-build, digit for digit |

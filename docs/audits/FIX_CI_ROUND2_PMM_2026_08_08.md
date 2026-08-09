# CI round 2 -- the repair had a REACH, and the guard was reading the wrong state -- 2026-08-08

Seven CI failures were referred after `FIX_CI_M1_T34_2026_08_06` and
`FIX_UNION_GRID_2THREAD_2026_08_06` landed.  All seven pass on the Windows dev
box and in WSL at `OPENBLAS_NUM_THREADS` 1, 2 and default.

```text
A  test_pmm_m3_efficiency.py::test_t34_guard_is_silent_on_the_cured_ladder
   test_pmm_m3_efficiency.py::test_t34_guard_is_silent_where_the_answer_is_right[2-16-5e-10]
   test_pmm_m3_efficiency.py::test_t34_guard_is_silent_where_the_answer_is_right[2-8-None]
B  test_pmm_m2_window_contract.py::test_min_feature_is_the_accuracy_lever_on_the_per_layer_path_too
C  test_m1_conditioning_guard.py::test_anisotropic_cascade_is_not_falsely_refused
   test_m1_conditioning_guard.py::test_the_step_down_is_not_a_re_solve_the_census_says_there_is_none
```

A and B are ONE library defect with two faces, and both faces were named as
OPEN in the 2026-08-06 documents.  C is two test claims that were still
stronger than the invariants they stated.  Everything here is measured; the
CI-only branches are driven by fault injection with the CI log's own numbers
(S6).

---

## S1.  A and B are the same defect: the repair had a REACH, the guard read the wrong state

### S1.1  The referral's premise, and why it was wrong

The referral read the A failures as "the shipped `_forward_growth_flip` repair
evidently did NOT reclassify them (else they would not be in the forward
set)".  That inference does not hold, because
`FIX_UNION_GRID_2THREAD_2026_08_06` S4 made a deliberate choice:

> `_record_mode_cut` is deliberately still called with the pre-repair
> `prop`/`q`, so `_MODE_CUT_CENSUS` and the T3-4 verdict keep measuring the
> DIAGNOSIS -- what the bare selector would have done -- and not the repaired
> state.

So the guard's channel A was reporting what the BARE selector would have done.
On CI's round-off the bare selector would have grown 2-4 modes on those cells;
the repair redirected them; the answer came back right (`rel` = 0.00352 and
0.00573 against the RCWA 141-order anchor, which are this file's own
cured-ladder readings) and the guard spoke anyway.  The mask's three conjuncts
are bit-identical to `_mode_cut_growth`'s, so a nonzero raw reading is exactly
the set the repair acted on.

**A is therefore a FALSE ALARM produced by the instrument's choice of state,
not a coverage gap.**  It is the R-1b failure mode the campaign rates worse
than silence, arriving through the back door.

### S1.2  B is a coverage gap, and it reproduces locally two rungs up

B is real.  Instrumenting the flip itself on the M2 coated taper at
`n_slice` = 2, library default, per-layer grids [M, Windows, 1 thread] --
`raw grow` is `_mode_cut_growth`'s reading, `still growing` counts modes whose
forward `q` still has `Im(q) < -1e-6 |q|` AFTER the shipped selector:

```text
degree  R0 (shipped)  raw grow  still growing (cut ratio of the survivors)
 6      0.1109998     0         0
 8      0.1107229     0         0
10      0.1106425     0         0
12      0.1106127     8         0
14      0.1105995     4         0
16      0.1105928     7         0
18      0.0616661     8         2   at 23.6 x the cut
20      0.6233873     6         4   at 15.8 / 17.1 x the cut
```

Those last two rows are the two cells `FIX_UNION_GRID_2THREAD_2026_08_06` S9
item 3 recorded as open ("not fully closed by classification alone at high
degree on the ns=2 ladder ... whether those are a second mechanism or the same
one past the `_MODE_CUT_MARGIN_WARN` decade is not determined here").

**It is determined here: it is the same mechanism, one factor of 1.6-2.4
outside the bar.**  Every mode the repair leaves behind was measured with
`|Im q| / |q| = 1.0` EXACTLY -- `q` purely imaginary, so the mode carries
exactly zero z-power and its flux is round-off no matter how many multiples of
the cut that round-off happens to reach.

On CI the same survivors land inside degrees 10..14, which is the range B's
ladder scores, so B collapses there while it does not here.

### S1.3  Why the decade conjunct existed, and what replaces it

`_mode_cut_growth`'s docstring says it plainly:

> `within the cut's decade` -- this is what makes the statistic safe on a GAIN
> medium, where a forward mode legitimately grows.

That is the ONLY reason.  It approximates "this layer cannot amplify" by a
distance from the classification cut, and it buys that safety by capping the
repair's reach.  The exemption is now stated directly instead:

```python
grow = qf.imag < -_MODE_GROWTH_REL * xp.abs(qf)
near = xp.abs(flux) < _MODE_CUT_MARGIN_WARN * thr
bad  = prop & grow & (near | passive)
```

`passive` comes from `_grid_is_passive(mats)` (patterned layer, read off the
element table) or `_eps_is_passive(eps)` (uniform half-space, the scalar
permittivity it is actually solved on).  The accepted sufficient condition is
DIAGONAL and lossy-or-lossless:

```text
exy = eyx = exz = ezx = eyz = ezy = 0   exactly
Im(exx), Im(eyy), Im(ezz)  >=  0        (PUBLIC convention: Im(n) < 0 is GAIN,
                                         the same one _require_propagating_
                                         incidence already refuses under)
```

Verified against the shipped assemblers: `eps = (3.48+0j)**2` reaches
`elem_bnds` as `exx = eyy = ezz = 12.1104+0j`, and `(2-0.1j)**2` reaches it as
`3.99-0.4j` -- the public sign survives assembly, so the test is on the number
the user typed.

**No constant is added.**  `near | passive` either widens the mask to the whole
`grow` set or leaves it exactly where it was; there is no third state and
nothing to calibrate.  Where passivity is NOT PROVEN -- a gain medium, an
off-diagonal tensor, an unknown element payload, an empty table -- `passive`
is False and the function is bit-identical to the 2026-08-06 selector.

### S1.4  The guard's channel A moves to the RESIDUAL

`_mode_cut_verdict`'s channel A now reads `_mode_cut_growth_post` -- the
growing-mode count of the forward set the cascade was ACTUALLY assembled from
-- instead of `_mode_cut_growth`'s pre-repair diagnosis.  The census carries
BOTH (`n_grow` and the new `n_grow_post`), so every calibration table written
against the diagnosis still reads as written, and `_solve_screened` in the M2
suite and `_conical_cell` in the M3 suite are untouched.

The reason this is an improvement and not a relaxation is a measurement.  With
the passivity widening OFF, so that both populations are present, on the M2
coated taper, guard armed, per-layer, library default [M, Windows 1 thread]:

```text
ns  deg  mf     R0         rel      raw  post   RAW channel A   RESIDUAL channel A
 2    6  -      0.1109998  0.00825   0    0     quiet           quiet
 2    8  -      0.1107229  0.00573   0    0     quiet           quiet
 2   10  -      0.1106425  0.00500   0    0     quiet           quiet
 2   12  -      0.1106127  0.00473   8    0     FIRES  (wrong)  quiet
 2   14  -      0.1105995  0.00461   4    0     FIRES  (wrong)  quiet
 2   16  -      0.1105928  0.00455   7    0     FIRES  (wrong)  quiet
 2   18  -      0.0616661  0.43987   8    2     FIRES           FIRES
 2   20  -      0.6233873  4.66242   6    4     FIRES           FIRES
 6   10  -      0.1116612  0.00497   2    0     FIRES  (wrong)  quiet
 6   12  -      0.1116278  0.00467   1    0     FIRES  (wrong)  quiet
 6   14  -      0.1116137  0.00454   2    0     FIRES  (wrong)  quiet
 6   16  -      0.1116069  0.00448   4    0     FIRES  (wrong)  quiet
 6   18  -      0.1116032  0.00445   7    0     FIRES  (wrong)  quiet
 6   20  -      0.1116010  0.00443   7    0     FIRES  (wrong)  quiet
 2  6..16 0.5nm 0.1104795  0.0035    0    0     quiet           quiet
 6  6..16 3.0nm 0.1108157  0.0004    0    0     quiet           quiet
```

`rel` is against the RCWA 141-order anchor; "(wrong)" marks a firing on a cell
whose answer is right.

**The residual partitions this family EXACTLY**: `post > 0` on precisely the
two cells with `rel > 0.05` and on no other, in all five (mount x
thread-count) environments measured, while the raw reading false-positives on
NINE of them.  That is the mount-invariant discriminator
the T3-4 block comment has been asking for since the four-name adjudication,
and it costs nothing: it is the same `Im(q_forward) < -_MODE_GROWTH_REL |q|`
test, applied to the forward set the solve actually used.

### S1.5  Result at the library level

With both changes, the whole scanned family is right and nothing grows:

```text
ns  degrees   R0 range                  rel to RCWA        post
 2  6 .. 20   0.1105867 .. 0.1109998    0.00449 .. 0.00825   0
 6  6 .. 20   0.1116010 .. 0.1120871    0.00443 .. 0.00880   0
```

Cured cells (`min_feature` above M2's threshold) are **bit-identical** with
the widening on or off -- the repair touches nothing there, because there is
nothing to touch.  Measured: no mode enters the mask at all on any cured cell.

Sites changed (all five that route through the selector, so the JAX twins keep
their parity -- the traced assemblers carry a TRACED 0-d passivity flag rather
than falling back, so the two paths accept exactly the same media):

```text
_core.py      _sem_modes_tensor          passive = _grid_is_passive(mats)
_core.py      _sem_modes_uniform         passive = _eps_is_passive(eps)
_core.py      _jpmm_sem_modes_tensor     passive = mats["passive"]   (traced)
_core.py      _jpmm_sem_modes_uniform    passive = _jeps_is_passive(eps, jnp)
_jax_stack.py _jstack_modes_uniform      passive = _jeps_is_passive(eps, jnp)
_core.py      _jpmm_assemble_tensor      -> mats["passive"] = _jtensor_is_passive
_jax_stack.py _jstack_assemble           -> mats["passive"] = _jtensor_is_passive
```

**Fail-before switch:** `PMM_FORWARD_GROWTH_PASSIVE` (default `True`).
`False` restores the 2026-08-06 mask bit for bit; with
`PMM_FORWARD_GROWTH_REPAIR` `False` as well, the pre-2026-08-06 selector.  The
two switches together reproduce either shipped selector exactly.

---

## S2.  A, at test level

`test_t34_guard_is_silent_where_the_answer_is_right` and
`test_t34_guard_is_silent_on_the_cured_ladder` are unchanged in their
assertions and each gains one:

```python
assert cell["post"] == 0     # the invariant BEHIND the silence, asserted
assert not cell["fired"]     # verbatim
```

The silence claim is now build-independent BY CONSTRUCTION rather than by
calibration -- the shipped forward set of a passive stack cannot grow -- and
the added residual assertion means the control cannot be satisfied by a guard
that has merely gone quiet.  Nothing was weakened: `assert not fired` is the
same assertion it was, on a guard that no longer has a build-dependent reason
to speak there.

---

## S3.  B, at test level, and the wrong-cell scan

`test_min_feature_is_the_accuracy_lever_on_the_per_layer_path_too` keeps all
five of its sections verbatim and gains a sixth, which pins the CI mechanism
ON THIS BUILD -- no injection needed, because the survivors are two rungs
further up the same ladder:

```text
ns = 2, library default, degrees (10, 12, 14, 18, 20), widening OFF [M, Windows]

  OPENBLAS_NUM_THREADS   spread      survivors
  1                      3.75        degrees 18 (2 modes) and 20 (4)
  2                      2.15        degree 20 only (0.6233872 -> 0.1105868)
  24                     5.047e-04   NONE
```

**The survivor rung migrates with the POOL, and the first draft of this
section asserted it did not.**  It read `assert s_narrow > 0.5` and was refuted
at 24 threads on this box within the hour -- 5.047e-04, i.e. the 2026-08-06
mask already covers everything this pool's round-off produces on this ladder.
Exactly the disease, one more level down, in a section written to pin it.

So the fail-before half is ADJUDICATED -- printed with its reading when the
pool does not carry a survivor -- and the CURE half (`spread < 1e-2` on the
shipped ladder, and `< 5 %` from the RCWA anchor) is asserted on every pool.
The GUARANTEED fail-before for the widening is
`test_t34_guard_fires_on_every_silent_wrong_cell_of_this_build`, which scans
BOTH `n_slice` families and widens its degree set until it finds a wrong cell;
it passed at every thread count here, including the 24 where this one ladder
alone carries nothing.

`test_t34_guard_fires_on_every_silent_wrong_cell_of_this_build` had to move,
and its own assertion message asked for exactly this ("If it was FIXED, re-pin
this against the fix"): with the shipped selector the family contains no
silent-wrong cell at all.  It is re-pinned the way the M2 file's three tests
were re-pinned on 2026-08-06 --

* the four original claims are made VERBATIM (same family, same 5 % partition,
  same 20x separation, same 1e-5 closure bar) with the passivity widening OFF,
  so the defect is still REPRODUCED on every build;
* **(d) added**: `n_grow_post` > 0 on every WRONG cell -- the safety half of
  the partition, and a claim no bar can be tuned into because it is a mode
  count.  It could not be made before: the raw reading read >= 1 on nine cells
  that were right.  The OTHER half (`== 0` on every right cell) is measured
  exact in all five (mount x thread-count) environments but is PRINTED here
  rather than asserted -- the population it scores is the uncured ladder,
  whose cells migrate with the build, and a growing forward mode need not
  produce a large error on every build.  The no-false-positive claim is
  asserted where it IS build-independent: on the cured ladder and on the
  right-answer control, both of which now assert `n_grow_post == 0` directly
  (S2).  Same discipline as S4's (e);
* **(e) added**: the CURE -- the same scan with the shipped selector has no
  wrong cell and `n_grow_post` = 0 everywhere.

`test_t34_guard_ships_disarmed_and_arming_it_moves_no_number` pinned `(ns=2,
degree=12)` as a cell the guard speaks on.  Since channel A no longer speaks
about the repaired diagnosis, that rung's firing now rests on channel B alone
-- the build-fragile half.  The cell is FOUND instead of listed: the family is
scanned for a rung the guard speaks on, with the fail-before switch as the
backstop (with the widening off the family provably still contains a cell
whose forward set grows, and channel A cannot be quiet on one of those).
Total silence in both states FAILS with the table; it never skips.

`test_t34_guard_warnings_come_out_in_WAVELENGTH_ORDER_on_the_sweep` has the
same exposure for the same reason -- it owns the deferred-warning ORDERING
contract and needs a device that speaks -- and gets the same backstop: if the
sweep is silent with the shipped selector it is re-run with the widening off,
which provably speaks, and every assertion below is unchanged.  It fires
without the backstop on both mounts at 1 and 2 threads, so the branch is
inert today; it exists so a build where channel B goes quiet cannot turn an
ordering test into a failure about something else.

`test_the_forward_set_cannot_grow_on_the_union_grid_conical_staircase` (the
ninth-name pin) now scores its (b) half WITHOUT the `|flux| < 10 thr`
conjunct, i.e. against the invariant itself.  The bar is REMOVED from the
claim, not moved.  It also asserts that every row of that lossless staircase is
recognised PASSIVE, so the widened branch is proved to have been exercised.

One test was ADDED,
`test_the_passivity_widening_is_a_null_floor_and_an_invariant`, and it is the
one pin in this round that owes nothing to any environment -- no device, no
BLAS, no solve.  Over 400 random `(flux, q, thr)` spectra, half of them with a
PURELY IMAGINARY `q` (the pathological population -- zero z-power, so the flux
is round-off wherever the cut sits):

```text
claim                                                        measured
passive=False is bit-identical to a VERBATIM re-write of     0 / 400 differ
  the 2026-08-06 mask (tolerance 0.0, not array_equal)
passive=True leaves a GROWING mode in the forward set        0 / 400
... and the SAME spectra do under the 2026-08-06 mask        399 / 400
  (so the invariant claim is a measurement, not a
   tautology about the generator -- asserted, bar 100)
```

Random spectra are the right instrument precisely because the CI failures were
about round-off landing where the dev box's does not: this samples the whole
plane instead of one build's corner of it.  It runs in 1.3 s.

**Every remaining assertion that REQUIRES the guard to speak is now either
scored with the fail-before switch (where channel A's residual is guaranteed
non-empty on this family) or self-calibrating with that switch as its
backstop.**  The three that require SILENCE -- the two false-positive controls
and the legitimate-count-spread control -- are the ones the residual made
build-independent.

---

## S4.  C1 -- the anchor rung was not an anchor either

`test_anisotropic_cascade_is_not_falsely_refused` failed with

```text
a refined rung closes at 1.312e-02, worse than 5x the unconverged M=5 rung's
1.937e-05
assert 0.013115583930066954 < (5.0 * 1.9368069018810274e-05)
 + where 0.013115583930066954 = max([0.013115583930066954, 0.0010456702385113203,
                                     9.992007221626409e-15, 0.0011105502533808043])
```

The v5.33.1 fix had replaced an absolute `5e-3` bar with a comparison against
the ladder's own "deliberately unconverged" M = 5 rung.  On CI that rung closed
at 1.937e-05 -- three decades TIGHTER than the 2.020e-02 it reads here -- so
`5 x loose` became a 1e-04 bar and the CI ladder walked through that one too.
Twelve decades of spread inside one run.

**Why no rung can be the anchor, measured.**  Holding code, geometry and rung
fixed and varying ONLY the BLAS pool and the mount:

```text
rung   |R+T-2|                                              J00
       Win 1 thr   Win 2 thr   Win 24 thr  WSL 1 thr   WSL 2 thr
M= 3   4.2577e-03  4.2577e-03  4.2577e-03  4.8709e-02  4.8709e-02   =
M= 5   2.0200e-02  2.0200e-02  2.0200e-02  4.4875e-03  4.4875e-03   =
M= 7   4.4301e-04  4.4301e-04  4.4301e-04  5.4675e-04  5.4675e-04   =
M= 9   2.0581e-03  2.0581e-03  2.0581e-03  3.9565e-03  3.9565e-03   =
M=11   8.6799e-04  2.1803e-03  2.1803e-03  9.7311e-03  2.2228e-03   =
M=15   1.2593e-04  5.4012e-05  1.2443e-03  1.7622e-03  4.4269e-04   =
M=17   1.6645e-03  1.2471e-03  4.4796e-03  1.0558e-03  5.0917e-04   =
M=19   1.5227e-04  2.4035e-04  1.1724e-04  5.4986e-04  2.8244e-04   =
M=21   6.4260e-04  1.4535e-04  2.4854e-03  2.9880e-03  2.3940e-04   =
M=25   2.0413e-03  7.0078e-05  3.8106e-03  1.2089e-03  2.0599e-03   =
M=29   2.5313e-14  4.2829e-04  6.2761e-04  2.0662e-03  2.0028e-04   =
```

`J00` is IDENTICAL to every printed digit in all eleven rungs of all five
cells -- across two pythons (3.14.6 / 3.12.3), two numpys (2.4.4 / 2.4.6) and
two OpenBLAS kernels (Haswell / SkylakeX) -- while the SAME rung's closure
moves by up to 54x with the pool and 11x with the mount, and the sequence is
non-monotone in M in every column.  On a LOSSLESS cascade `R + T = 2` is very
nearly tautological -- the campaign's standing caution -- so what this residual
measures is the round-off of the near-cancelling deep-evanescent star
denominators, not the truncation.  **A quantity with no systematic content
cannot carry a convergence claim under ANY bar, absolute or relative.**

**The restructure.**  (a), (b) and (c) are unchanged.  The energy comparison is
replaced, and the load-bearing claim is moved to the quantity that actually
converges:

```text
(d) J00 is CAUCHY: |J00(M) - J00(M_finest)| strictly DECREASES rung by rung
    1.1357e-04 -> 3.6167e-05 -> 1.0151e-05 -> 4.244e-06   (~3x a step, and
    IDENTICAL on both mounts at every thread count)
(e) the closure ladder is PRINTED, never asserted on -- see below
(f) the one thing it CAN carry: |R+T-2| < 0.5 on every rung.  Worst measured
    anywhere is 2.02e-02 (25x headroom); the mis-assembled-cascade magnitude
    this class produces when it does go wrong is |R+T-1| = 21 (the ninth
    name), 40x the other side.  The bar sits inside a three-decade gap.
```

(d) is strictly stronger than what it replaces: an energy bar cannot see a
cascade that drifts to a wrong-but-unitary answer, which is the exact failure
mode this campaign exists for.

**(e) is a finding, and it was earned the hard way.**  The first draft of this
fix asserted the premise instead of quoting it -- "the closure spans >= 10x
across the ladder, so it is round-off" -- with 16x headroom on the four cells
then measured (160x / 374x / 173x locally, 1.3e12 on CI).  The three-environment
discipline killed it immediately: **WSL at one thread reads 8.16x**
(`['4.487e-03', '3.956e-03', '1.762e-03', '5.499e-04', '1.209e-03']`).  That is
the disease one more level down -- a per-build magnitude asserted as universal,
inside the very test written to disqualify that magnitude.  The evidence that
the closure carries no truncation information is a CROSS-POOL, CROSS-MOUNT
measurement, and a test cannot vary the BLAS pool in-process (the same argument
the ninth-name pin makes).  So the ladder is printed with its reason, the
measurement lives here, and no single-process statistic is asserted on it.

---

## S5.  C2 -- "every draw" was the same coin flip one level down

`test_the_step_down_is_not_a_re_solve_the_census_says_there_is_none` failed with

```text
seed 5 n 24: refinement read 9.7681e-04 against LU's 8.8175e-04
             -- the one candidate the guard tried stopped helping
```

ratio 1.108.  The v5.33.1 fix had already restructured this test's FIRST claim
away from a per-draw strict ordering, for exactly this reason (the two
residuals agree to a few percent, so the inequality is decided by round-off).
Its SECOND claim kept one: `resid(ir) < resid(lu)` on each of 15 draws.

The docstring states the decision-relevant claim itself: *"no direct
alternative is even 2x better ... refinement is the one route that helps"*.
Per-draw strict improvement is a STRONGER statement than that; it is not what
the guard's design rests on and it is round-off on any draw where the two
residuals coincide.

**Aligned with the claim it states**, in the SAME 2x unit claim (1) uses:

```text
claim                                          worst measured   bar    headroom
resid(ir) < 2 x resid(lu), every draw          1.108 (CI)       2.0    1.8x
median(resid(ir) / resid(lu))                  0.80             0.95   1.19x
a MAJORITY of draws improve outright           14/15 local      >7.5
```

The first is a real assertion, not a formality: fault-injection case F shows a
draw at ratio 2.5 still fails.  Nothing the guard's design rests on was
relaxed; the per-draw strict ordering was never that.

---

## S6.  Fault injection -- every new branch, driven with the CI log's numbers

The CI images cannot be run locally.  Two injectors were used:

* **the near-cut injector** -- `_mass_flux_threshold` scaled by 1/3, which
  pushes evanescent modes whose ROUND-OFF flux sits just under the cut just
  over it.  That is exactly the CI condition ("classified PROPAGATING on a
  z-flux ... within a factor 1.05 / 1.44 of the cut") produced on a build whose
  own round-off does not do it;
* **the CI-ladder spoof** -- `rcwa_jones_1d` wrapped so `|R+T-2|` takes the CI
  log's value per rung without touching `J00`.

`scratchpad/faultinject.py`, run on Windows at 1 and 2 threads and on WSL at
1; all thirteen behave as designed in every cell (case B is the one that
needed the family form -- see its row):

```text
case                                                        want  outcome
A   CI near-cut condition on a CURED cell: raw diagnosis     PASS  PASS
    non-empty, residual 0, guard quiet, answer moved < 1e-6
A'  ... with channel A restored to the RAW diagnosis         PASS  PASS
    (the 2026-08-06 policy) the same cell FIRES -- the CI
    failure verbatim
B   beyond-decade survivor: WHERE the narrow mask leaves     PASS  PASS
    one (18 and 20 at 1 thread, 20 only at 2, NEITHER at
    24) it is 44 % / 466 % wrong at 15.8-23.6 x the cut,
    and the widened mask leaves none on either rung and is
    0.45 % from RCWA.  Adjudicated, not asserted per rung:
    the first draft asserted BOTH rungs and was refuted at
    2 threads; the second asserted "at least one" and was
    refuted at 24.  The migration IS the finding
C   gain (Im eps < 0), off-diagonal, empty and 2-tuple       PASS  PASS
    element tables are NOT widened; a mode 100 x past the
    cut is repaired on a passive layer and untouched
    otherwise
D   the M1 energy ladder driven with the CI numbers          PASS  PASS
D'  ... the v5.33.1 energy bar on the SAME ladder            FAIL  FAIL
E   the M1 step-down family incl. the CI draw (1.108)        PASS  PASS
E'  ... the v5.33.1 per-draw ordering on the SAME draw       FAIL  FAIL
F   a MATERIALLY worse refinement (2.5x) still fails         FAIL  FAIL
G   the M3 (e) CURE branch with the widening defeated        FAIL  FAIL
    ("... [(2, 18, 0.4399), (2, 20, 4.6624)] are still
    wrong ... a surviving wrong cell is a SECOND mechanism")
H   the disarmed pin FALLS BACK when channel B is silenced   PASS  PASS
    (_MODE_CUT_MARGIN_WARN = 0) and finds its cell with the
    widening off
I   ... and FAILS (never skips) when the verdict is stubbed  FAIL  FAIL
    to total silence
J   the M2 ladder at degrees (10, 14, 18, 20): narrow        PASS  PASS
    spreads 3.75 at 1 thread, widened 3.73e-03
```

---

## S7.  Result

`-q -p no:randomly`.  Windows = python 3.14.6 / numpy 2.4.4 / scipy 1.17.1,
scipy-openblas 0.3.31 Haswell, MAX_THREADS 24.  WSL = python 3.12.3 / numpy
2.4.6 / scipy 1.17.1, same OpenBLAS release, SkylakeX, MAX_THREADS 64 -- the
identical numpy/scipy pair CI py3.11 carries.

### The three named files

```text
mount    OPENBLAS_NUM_THREADS   result
Windows  1                      89 passed           94 s
Windows  2                      89 passed          132 s
Windows  default (24)           WIN24
WSL      1                      88 passed, 1 skip  132 s
WSL      2                      88 passed, 1 skip  154 s
WSL      default (64)           WSLDEF
```

The count rises from 88 to 89 because of the added property pin (S3).

RUNTIME NOTE.  The two default-thread cells are much slower than the 1/2-thread
ones on this box, and it is threading overhead rather than work: the added
sections solve small dense eigs (degrees 18 and 20 on a 4-row stack), where a
24- or 64-thread pool costs more in synchronisation than it saves.  CI runners
carry 2-4 cores, so the numbers CI will see are the 1/2-thread column.

(The WSL skip is `test_the_guard_is_numpy_only_and_the_traced_path_is_unchanged`
-- `pytest.importorskip("jax")`, and that venv has no jax.  Unrelated and
unchanged by this work.)

### Blast radius, Windows at 1 thread

```text
suites                                                        result
8 PMM classification-path suites (per_layer_grids, tapered,   228 passed  209 s
  stack, stack_conical, conical, audit_fixes, w9_pmm_taper,
  w7_pmm)
test_v5_12_0_pmm_jones_autodiff                                34 passed  128 s
test_v5_14_2_jax_stacks + test_audit_w3_pmm_jax_guards         31 passed  169 s
```

The JAX side is the part this round could most easily have broken, because the
traced assemblers now carry a passivity flag through `mats` -- so it was
checked on its own rather than only inside a sweep.

TWO NAMES NOT RE-CHECKED HERE: `test_v5_20_13_pmm_jones_2d_fff_nv` and
`test_v5_20_0_pmm_rcwa_upgrades` are being edited concurrently by another
change (`FIX_FFF_NV_THREAD_BARS_2026_08_08`), so their state is not this
change's to report; both were already referred as standing failures by
`FIX_UNION_GRID_2THREAD_2026_08_06` S9 item 6, and both were measured there to
be byte-identical with the forward-growth repair on or off.

---

## S8.  Open

1. **Channel B is unchanged and is still the build-fragile half.**  It fires on
   nine cells of the classical family whose answers are right (S1.4) and on
   the whole conical family.  Channel A's move to the residual does not touch
   it; the T3-4 guard therefore still ships DISARMED, still for the conical
   false positive, and arming is still gated on the two-degree consensus probe.
   What HAS changed: on the classical family the residual is now a bar-free
   discriminator that partitions exactly, so a future arming decision has one
   channel it can trust.
2. **The MIRROR-IMAGE misclassification is not touched, and is judged rare.**
   This round repairs a mode called PROPAGATING whose flux is round-off.  The
   other direction -- a genuinely propagating mode called EVANESCENT, whose
   direction then comes from `Im(q)` on ~1e-16 noise -- is untouched.  It
   cannot produce a growing forward mode (a real `q` flipped either way still
   decays nowhere), so no invariant catches it, and it requires a mode with
   essentially zero real z-power AND a real `q`, i.e. one sitting exactly at
   cut-off.  Nothing in this family was measured doing it (the residual
   partitions the classical family exactly, which it could not if a second
   mechanism were producing wrong answers here).  Named so the next round
   does not have to rediscover it.
3. **The passivity test is a SUFFICIENT condition, not the exact one.**  A
   tensor with a nonzero off-diagonal is not accepted even when it is passive;
   the exact statement there is that `(eps - eps^H) / 2i` is positive
   semi-definite, which costs a 3x3 eigen-solve per element and does not trace.
   Those layers keep the 2026-08-06 mask, i.e. exactly what they had.  Closing
   it is a contained follow-up.
4. **`bor/_jax_bor.py`** still carries a structurally similar
   `where(propagating, Pz < 0, Im q < 0)` selector on a different flux
   definition.  Not examined, as in the 2026-08-06 round.
5. **The PMM conical cascade still has no energy check**
   (`FIX_UNION_GRID_2THREAD_2026_08_06` S9 item 1).  Unchanged.
6. **X-1 remains open**, untouched.
7. **Two standing failures elsewhere, NOT caused by this change** and already
   referred by `FIX_UNION_GRID_2THREAD_2026_08_06` S9 item 6:
   `test_v5_20_13_pmm_jones_2d_fff_nv::test_pmm_fff_nv_matches_rcwa_fff_nv`
   (an absolute `|R+T-2| < 1e-3` bar) and
   `test_v5_20_0_pmm_rcwa_upgrades::test_stack_sweep_geom_cache_is_transparent_and_reused`
   (an `np.array_equal` bit-identity).  Both are per-build magnitudes asserted
   as universal, and both are owned by a concurrent change.
8. **THE PROCESS FINDING, and it cost three refutations inside one day.**
   Every claim this round REMOVED and every claim it ADDED had to survive the
   same test: is it a statement about the PHYSICS, or about one pool's
   round-off?  Three of my own new claims failed it and were caught only
   because all six (mount x thread-count) cells were run --

   ```text
   claim as first written                       refuted by         became
   "the closure spans >= 10x" (M1 (e))          WSL 1 thr, 8.16x   printed
   "BOTH rungs 18 and 20 carry a survivor"      Win 2 thr          adjudicated
   "the ns=2 ladder collapses over _REACH"      Win 24 thr         adjudicated
   ```

   The invariant claims -- `n_grow_post == 0` on a passive stack, the null
   floor, `J00` Cauchy -- survived every cell unchanged, because they are not
   magnitudes.  That is the rule this campaign keeps rediscovering: **a bar on
   a round-off quantity is a coin flip no matter how much headroom it looks
   like it has.**

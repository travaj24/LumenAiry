# PMM four-name cross-build adjudication -- 2026-08-05

Four deterministic test failures were referred as "pass on Windows/MKL, fail on
WSL/OpenBLAS".  This document records what they actually are, what was changed,
and the measurement on every build and thread count.

Names:

1. `test_pmm_m2_window_contract.py::test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band`
2. `test_pmm_m3_efficiency.py::test_the_star_keeps_its_inverse_because_a_solve_breaks_a_shipped_identity`
3. `test_pmm_m3_efficiency.py::test_t34_guard_fires_on_the_silent_wrong_cells[2-12-0.4]`
4. `test_pmm_m3_efficiency.py::test_t34_guard_fires_on_the_silent_wrong_cells[6-10-4.0]`

---

## S1.  The axis is the BLAS THREAD COUNT, not the BLAS build

The referral's premise does not survive measurement, and it matters, because a
fix aimed at "MKL vs OpenBLAS" would have been aimed at nothing.

Neither mount is MKL, and both run the SAME OpenBLAS release:

```text
mount    python   numpy    BLAS
Windows  3.14.6   2.4.4    scipy-openblas 0.3.31.188.0 USE64BITINT
                           DYNAMIC_ARCH, Haswell kernels, MAX_THREADS=24
WSL      3.12.3   2.4.6    scipy-openblas 0.3.31.188.0 USE64BITINT
                           DYNAMIC_ARCH, SkylakeX kernels, MAX_THREADS=64
```

(`threadpoolctl.threadpool_info()` returns `[]` on the Windows interpreter, so
the library's own BLAS cap is inert there; the thread count is whatever
OpenBLAS picks.)

Measured, pre-fix, on the two named files, `-q -p no:randomly`:

```text
mount    OPENBLAS_NUM_THREADS   result
Windows  default                3 failed, 58 passed   (names 1, 3, 4)
Windows  1                      61 passed
WSL      default                4 failed, 57 passed   (names 1, 2, 3, 4)
WSL      1                      61 passed
```

So three of the four names fail on WINDOWS too, at the default thread count,
with numbers identical to the WSL ones to 9 significant figures:

```text
name 1  |dJ| = 6.7097951948866e-01 (Windows)  6.7097951945444e-01 (WSL)
name 3  r0   = 0.1106127253540     (Windows)  0.1106127244880     (WSL)
name 4  warns = []                 on both
```

The hidden axis is the BLAS reduction order set by the THREAD COUNT.  At one
thread all four pass on both mounts; at N threads three fail on both and the
fourth (name 2) fails on WSL only.  Everything below is therefore measured in
all four (mount x thread-count) cells, and the fixes are required to hold in
all four.

---

## S2.  Name 1 -- the halfwidth contract was measured on a device carrying the very defect it must exclude

**Root cause.**  `test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band`
runs two devices and asserts `dJ < 3e-4` on both.  Its own preamble asserts
that the `min_feature` snap is INERT on them, to remove the geometry confound
M2 found.  But "the snap is inert at the library default" means the device's
tightest cross-layer separation SURVIVES:

```text
uncoated ns = 3    off = (310/3) tan(2 deg) = 3.6085 nm   harmless
25 nm coat ns = 8  off = (310/8) tan(2 deg) = 1.3532 nm   A SLIVER
```

The second device is therefore guaranteed to carry the T3-4 silent-wrong
classification defect, and `dJ` there is not a window residual at all -- it is
the difference between a sound and a flipped classification.  `6.710e-01` is
the silent-wrong magnitude, not a tolerance miss.  The test's own docstring
already said the campaign's numbers were "max over the HEALTHY cells of two
devices"; the code did not screen for healthy.

Measured `n_growing` (S4) per cell, and the resulting `dJ`:

```text
device      deg   Win N     Win 1     WSL N     WSL 1     dJ (Win N)
uncoated3    6    0,0       0,0       0,0       0,0       1.2977e-04
uncoated3    8    0,0       0,0       0,0       0,0       3.1232e-05
uncoated3   10    0,0       0,0       0,0       0,0       6.3825e-06
fatcoat8     6    1,1       0,1       1,1       0,1       4.4048e-05
fatcoat8     8    1,2       0,1       1,1       0,2       6.7098e-01
fatcoat8    10    3,1       0,3       3,1       0,3       1.0673e+00
```

(pairs are halfwidth 1, halfwidth 2.)  The uncoated device is sound in all
four cells at every degree; the coated one is unsound in all four cells at
every degree.

**Fix (test only).**  Screen each cell with the SHIPPED T3-4 growth instrument
read off `_MODE_CUT_CENSUS` (`_solve_screened`), skip the unsound cells -- they
are exactly the coverage names 3/4 pin -- and assert the contract on the sound
ones.  The bar is now COMPARATIVE, with no calibrated constant: the window
residual must be under half the DEGREE-REFINEMENT residual at the same rung,
and must still decay spectrally.  Measured on the uncoated device, identical to
five digits in all four cells:

```text
degree   |dJ| window    |dJ| degree -> degree+2   ratio   bar
6        1.2977e-04     1.2317e-03                0.105   0.5
8        3.1232e-05     4.8489e-04                0.064   0.5
10       6.3825e-06     2.3257e-04                0.027   0.5
```

Worst-case headroom on the bar: 4.7x.  A non-vacuity assertion requires at
least one device to deliver a complete unscreened 3-rung ladder.

---

## S3.  Name 3 -- the wrong-cell list is a per-build fact asserted as universal

**Root cause.**  `test_t34_guard_fires_on_the_silent_wrong_cells` hard-coded
`(ns, degree, rel_min)` triples.  The DEVICE and the DEFECT are
build-independent (M2's threshold rule fixes the sliver width, and the degree
ladder collapses somewhere above degree 8 on both mounts) but WHICH degree
collapses is not -- it is decided by whether a round-off-flux mode lands above
or below the classification cut.  Cell `(2, 12)` is 44 % wrong at one thread
and 0.47 % (i.e. correct) at N.

**Fix (test only).**
`test_t34_guard_fires_on_every_silent_wrong_cell_of_this_build`
SCANS `ns in {2, 6} x degree in {10, 12, 14, 16}`, partitions on this build by
the RCWA-anchored reference (`0.1100920` / `0.1111090`, 141 orders), and
asserts (a) the family contains a wrong cell here -- widening to degrees 18/20
if it does not, never skipping; (b) `|R+T-1| < 1e-5` on every wrong cell, i.e.
conservation is blind to all of them; (c) the guard fires on every wrong cell.
The 5 % partition bar is not a calibration -- the two populations are
0.03-0.9 % and 42-466 %, three decades apart, in all four cells -- and the test
asserts the separation explicitly (`min(wrong) > 20 x max(right)`; measured
ratio 84x at N threads, 88x at 1).

The false-positive control is `test_t34_guard_is_silent_on_the_cured_ladder`:
raise `min_feature` above M2's threshold `min(off, |coat - off|)` and no sliver
remains, so the guard must be silent at every degree.  Cured means 0.5 nm at
ns = 2 (threshold 0.4127 nm) and 3.0 nm at ns = 6 (threshold 1.8042 nm); the
0.5 nm ns = 6 column is deliberately NOT used, because it is below that
threshold and cures nothing.  Measured silent, 12/12 rungs, in all four cells.

---

## S4.  Name 4 -- a real detection gap: the stack-level half is a coin flip, and the fix is a physical invariant

**Root cause.**  The T3-4 verdict was a conjunction of

* `spread` -- the propagating-mode COUNT differs between patterned layers;
* `at-risk` -- some layer carries a near-cut mode whose two direction criteria
  disagree.

`spread` is not a property of the device.  On the referred cell -- ns = 6,
degree = 10, default `min_feature` -- the answer is `0.5683670` against a
reference `0.1111090`, i.e. 411 % wrong at `|R+T-1| = 5.0e-07`, and it is the
SAME wrong number at 1 and at N BLAS threads.  The instrument is not:

```text
mount / threads   r0          counts per patterned layer   spread  verdict
Windows N         0.5683670   [4, 4, 4, 4, 4, 4]           0       SILENT
Windows 1         0.5683670   [4, 5, 4, 4, 4, 4]           1       fires
WSL N             0.5683670   [4, 4, 4, 4, 4, 4]           0       SILENT
WSL 1             0.5683670   [4, 5, 4, 4, 4, 4]           1       fires
```

Whether a round-off-flux mode lands one side of the cut or the other moves the
count by one and changes NOTHING about the answer.  A statistic that reads
differently on the same wrong answer cannot be the detector.

**What is actually wrong on that cell.**  The selector is

```python
prop = np.abs(flux) > thr
flip = np.where(prop, flux < 0.0, q.imag < 0.0)
```

For every mode it calls EVANESCENT the `Im(q)` rule guarantees `Im(q) >= 0`
after the flip: the mode decays along +z.  For the ones it calls PROPAGATING it
uses the flux sign instead.  A genuinely propagating mode has a real `q`, so
both rules agree and nothing can go wrong.  A really-evanescent mode whose
ROUND-OFF flux crosses the cut is handed the flux rule, and if that sign points
against `Im(q)` the mode is flipped into a GROWING one and put in the forward
set.  Dumped from the ns = 6 cell, the SUBSTRATE half-space row (solved on the
same sliver-bearing nodal grid):

```text
degree 8  (right, 0.1117556):  q = +0.717264j   |flux|/thr = 0.0023  prop=0
degree 10 (wrong, 0.5683670):  q = -0.717264j   |flux|/thr = 1.2671  prop=1
```

At degree 10 that mode is classified propagating on a flux 1.27x the noise
floor, its flux is positive so it is not flipped, and `q = -0.717264j` -- a
mode that GROWS along +z -- enters the forward set.  Note it is a HALF-SPACE
row: the old conjunction compares patterned layers only and could not have seen
it at any thread count.

**Fix (library).**  `_core._mode_cut_growth` adds channel A to
`_mode_cut_verdict`:

```text
n_growing = # modes with  prop
                          AND within _MODE_CUT_MARGIN_WARN of the cut
                          AND Im(q_forward) < -_MODE_GROWTH_REL |q|
```

scored over EVERY row, patterned or not.  Each conjunct is load-bearing:

* `prop` -- an evanescent-classified mode is repaired by the `Im(q)` rule by
  construction and can never grow;
* `within the cut's decade` (the existing `_MODE_CUT_MARGIN_WARN = 10`) -- this
  is what keeps the instrument safe on a GAIN medium, where a forward mode
  legitimately grows: an amplifying mode carries real power and sits ~1e8 x the
  cut, while every pathological mode measured here sits at 1.00-3.47x it;
* `Im(q) < -rel |q|` -- a ratio, so scale- and unit-free.  `_MODE_GROWTH_REL =
  1e-6` sits in the middle of a six-decade gap: genuinely propagating modes
  read `|Im q| / |q| <= 5e-10` and misclassified evanescent ones read exactly
  `1.0`.  Any bar in `[1e-8, 1e-3]` gives the identical verdict on every cell
  below.

The old conjunction is RETAINED as channel B, so no cell the calibrated guard
used to speak on goes quiet; the verdict fires on A or B.  The guard still
ships DISARMED: the conical false positive that disarmed it is NOT closed by
channel A (those three correct cells carry 1-3 growing forward modes at
1.07-2.87x the cut), which is now recorded in `_core.py` on a second
instrument.

**Fail-before / passes-after**, with `_mode_cut_growth` stubbed to `(0, inf)`
(channel A off = the pre-fix guard), running the two new T3-4 tests:

```text
mount / threads   channel A off                       channel A on
Windows N         FAIL ns=2 deg=14 rel=4.663 silent   PASS
Windows 1         PASS                                PASS
WSL N             FAIL ns=2 deg=14 rel=4.663 silent   PASS
WSL 1             (not run; Windows 1 passes both)    PASS
```

The pre-fix guard fails at N threads on BOTH mounts -- the regime the failure
was reported in -- and the growth channel fixes it there.  (It fails at
`ns = 2, deg = 14` rather than `ns = 6, deg = 10` only because the scan reaches
it first; both read `spread = 0` at N threads.)

**Separation of the shipped verdict**, default `min_feature`, `.` = correct,
`W` = wrong (`rel > 5 %`), `g` = `n_growing`, `s` = `spread`:

```text
cell        Windows N        Windows 1        WSL N            WSL 1
ns=2 d=6    . g0 s0 quiet    . g0 s0 quiet    . g0 s0 quiet    . g0 s0 quiet
ns=2 d=8    . g0 s0 quiet    . g0 s0 quiet    . g0 s0 quiet    . g0 s0 quiet
ns=2 d=10   . g0 s0 quiet    . g0 s0 quiet    . g0 s0 quiet    . g0 s0 quiet
ns=2 d=12   . g3 s1 FIRE     W g8 s2 FIRE     . g3 s1 FIRE     W g8 s2 FIRE
ns=2 d=14   W g7 s0 FIRE     W g4 s1 FIRE     W g7 s0 FIRE     W g4 s1 FIRE
ns=2 d=16   W g7 s2 FIRE     W g7 s2 FIRE     W g7 s2 FIRE     W g7 s1 FIRE
ns=6 d=6    . g0 s0 quiet    . g0 s0 quiet    . g0 s0 quiet    . g0 s0 quiet
ns=6 d=8    . g0 s0 quiet    . g0 s0 quiet    . g0 s0 quiet    . g0 s0 quiet
ns=6 d=10   W g1 s0 FIRE     W g2 s1 FIRE     W g1 s0 FIRE     W g2 s1 FIRE
ns=6 d=12   W g3 s1 FIRE     . g1 s1 FIRE     W g3 s1 FIRE     . g1 s1 FIRE
ns=6 d=14   . g1 s1 FIRE     . g2 s1 FIRE     . g1 s1 FIRE     . g2 s1 FIRE
ns=6 d=16   W g4 s2 FIRE     W g4 s3 FIRE     W g5 s2 FIRE     W g4 s3 FIRE
```

* every WRONG cell fires, 5 of 5, in all four (mount x threads) cells -- the
  `spread`-only guard is silent on `ns=2 d=14` and `ns=6 d=10` at N threads on
  both mounts;
* every CURED cell is quiet, 12 of 12, in all four cells (not shown: the
  ns=2 / 0.5 nm and ns=6 / 3.0 nm ladders, degrees 6..16, all `g0 s0`);
* the F2 alternating-grating control (correct, legitimate count spread
  `[4, 2, 4, 2]`) reads `g0` and stays quiet in all four cells;
* the lossy conforming null control (a `6.25 + 0.1j` layer) reads `g0`
  (Windows, both thread counts) -- the check that the growth channel does not
  fire merely because `q` is complex.

The three uncured cells that fire while landing within 2 % of the reference
(`ns=2 d=12` at N, `ns=6 d=12` at 1, `ns=6 d=14` at both) are NOT counted as
false positives, and the reason is measurable rather than rhetorical: each of
them is catastrophically wrong on the other thread count or at the neighbouring
degree.  A growing mode in the forward set is a mis-assembled cascade whose
error happened to be small; the guard's claim is about the classification, not
about the size of this rung's error.  The build-independent false-positive
control is the cured ladder, which is silent everywhere.

---

## S5.  Name 2 -- an MKL-specific bit-motion fact asserted as universal

**Root cause.**  The test justified keeping the explicit inverse in the
Redheffer star by substituting the backward-stable RIGHT solve and asserting
`moved > 0.0` on the R block.  Measured max |difference| against the shared
twin:

```text
mount / threads              R block      Jones block
Windows OpenBLAS-Haswell N   5.551e-17    2.001e-16
Windows OpenBLAS-Haswell 1   5.551e-17    8.327e-17
WSL OpenBLAS-SkylakeX    N   0.0          5.551e-17   <- the failure
WSL OpenBLAS-SkylakeX    1   1.388e-17    5.551e-17
```

The substitution IS separable from the inverse on every build; the R block
merely coincided on one of them.

**Fix (test only).**  Read the max over the R and Jones blocks; treat an exact
coincidence as a PASS (`solve` returning the inverse's bits does not break
anything); and add the build-independent half the bits-moved count cannot
stand in for -- a call counter proving the substituted star actually RAN
(`>= 3` calls on the 4-layer stack; measured 4).  The shipped-identity
assertion (`per-layer == shared` at tolerance 0.0) is untouched, and the
envelope is now `moved <= 64 eps = 1.42e-14` (measured worst 2.0e-16, 71x
headroom) instead of a bare `1e-9`.

---

## S6.  Result

`python -m pytest tests/unit/test_pmm_m2_window_contract.py
tests/unit/test_pmm_m3_efficiency.py -q -p no:randomly`

```text
mount    OPENBLAS_NUM_THREADS   before                after
Windows  default                3 failed, 58 passed   60 passed
Windows  1                      61 passed             60 passed
WSL      default                4 failed, 57 passed   60 passed
WSL      1                      61 passed             60 passed
```

61 -> 60 collected because the three parametrized cases of
`test_t34_guard_fires_on_the_silent_wrong_cells` became two tests
(`..._fires_on_every_silent_wrong_cell_of_this_build` and
`..._is_silent_on_the_cured_ladder`).  The 56 tests this work did not touch all
still pass on both mounts at both thread counts; the 57th previously-passing
test on WSL was the `[2-16-4.0]` parametrization, whose coverage now sits
inside the scan -- degree 16 is in `_T34_FAMILY` and is measured
wrong-and-firing in all four cells.

Regression, same mount, default threads:
`test_pmm_per_layer_grids.py`, `test_niche_audit_m2_m3_m9_pmm_guards.py`,
`test_niche_audit_w7_pmm.py`, `test_niche_audit_w9_pmm_taper.py`,
`test_v5_21_pmm_threaded_sweep.py`, `test_v5_13_0_pmm_tapered.py` ->
191 passed, 1 failed; plus `test_v5_20_0_pmm_conical.py`,
`test_v5_20_0_pmm_stack_conical.py`, `test_v5_11_0_pmm_stack.py`,
`test_v5_8_0_pmm.py` -> 59 passed.

The one failure,
`test_v5_13_0_pmm_tapered.py::test_sweep_matches_perwavelength`, is NOT caused
by this change and is out of scope for the four names: it never arms the census
or the guard, so `_record_mode_cut` -- the only function this change touches on
a solve path -- is never called on it.  It is the same axis: it fails at the
default thread count and passes at `OPENBLAS_NUM_THREADS=1`.  Referred as a
fifth name.

---

## S7.  Files changed

```text
lumenairy/elements/pmm/_core.py
  + _MODE_GROWTH_REL, _mode_cut_ratio, _mode_cut_growth
  ~ _mode_cut_verdict   (channel A / channel B)
  ~ _record_mode_cut    (rows carry n_grow, grow_ratio; census likewise)
  ~ the DISARMED block comment (records the spread coin flip)
tests/unit/test_pmm_m2_window_contract.py
  + _solve_screened
  ~ test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band
tests/unit/test_pmm_m3_efficiency.py
  ~ test_the_star_keeps_its_inverse_because_a_solve_breaks_a_shipped_identity
  - test_t34_guard_fires_on_the_silent_wrong_cells  (3 params)
  + _T34_FAMILY, _T34_WIDEN, _T34_REF, _T34_WRONG_REL, _t34_scan
  + test_t34_guard_fires_on_every_silent_wrong_cell_of_this_build
  + test_t34_guard_is_silent_on_the_cured_ladder
```

No public API changed.  The guard still ships DISARMED and the census still
ships off, so the default solve path is byte-for-byte what it was.

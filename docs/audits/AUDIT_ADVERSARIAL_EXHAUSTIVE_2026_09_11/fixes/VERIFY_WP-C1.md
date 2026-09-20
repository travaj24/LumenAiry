# VERIFY WP-C1 -- `apply_aperture(edge='gray')` as the default

Independent adversarial verification of `feat/c1-gray-edge-default`
(8 commits `8fbe7185`..`d73ee534` on `49ddf4bd`), report
[`WP-C1_GRAY_EDGE_REPORT.md`](WP-C1_GRAY_EDGE_REPORT.md).

Everything below was **re-measured**, never read off the report.  Where WP-C1
measured, this verification measured again on a DIFFERENT optic and a
DIFFERENT fixture family; where WP-C1 asserted, this verification mutated the
library and watched what went red.  Both builds, every time: Windows py3.14 /
numpy 2.4.4 / jax 0.11.0 / CuPy 14.0.1 on a device, and WSL py3.12 / numpy
2.4.6 / jax 0.10.2, both scipy-openblas, both with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line and `lumenairy.__file__` printed by every probe.  The PRE tree is this
verification's own `git archive 49ddf4bd` extraction at `C:/tmp/lum_vc1_base`,
built fresh; nothing is compared against a working copy.

Probes and JSON: [`validation/probe_verify_c1/`](../../../../validation/probe_verify_c1/).
Decision tests: `tests/unit/test_verify_c1_gray_edge.py` (28 ids, 26 pass +
2 strict xfail, 30 s, identical on both builds).

**SHIP, with four defects to close first** -- one of them (D1) a cross-backend
contract divergence the work package's own consolidation was meant to remove,
two documentation/Migration gaps (D2, D4) and one overclaim (D3).  None is a
wrong field.  The default move itself is correct, correctly measured, and
exactly reversible.

---

## 1. Verdict table

| # | claim | verdict | this verification's numbers |
|---|---|---|---|
| 1 | the default arm is bit-identical to `'gray'`, and explicit `'hard'` is bit-identical to the parent's default, on both kernels | **CONFIRMED** | PRE tree: `default is bit-identical to ['hard']` on RS and HF; branch: `['gray']`.  The `hard` and `gray` columns are unchanged in every digit PRE vs branch.  Fixtures: 25/27 (WIN) and 24/26 (WSL) aperture fixtures byte-identical PRE-default vs branch-`hard`; the 2 that move are the JAX eager route and are claim 6's pre-existing defect |
| 2 | the convergence ladder (grey 2nd order; the hard arm rising 53-54 % on the last refinement) | **CONFIRMED on WP-B11's optic; the RISE is FIXTURE-SPECIFIC** (defect **D3**) | all sixteen table entries and all twelve orders reproduced to the last printed digit by independent code (RS hard `8.3008e-03 3.3548e-03 3.4207e-04 5.2718e-04`, orders `1.307 / 3.294 / -0.624`; rise +54.1 % RS, +52.9 % HF; gains 21.06x / 8.33x).  On MY optic the hard arm FALLS at every step (orders `1.68 / 0.20 / 1.36` RS, `1.67 / 0.20 / 1.34` HF) -- the rate gap survives, the non-monotonicity does not |
| 3 | `edge_samples=4` is the knee from both sides | **CONFIRMED, wider margin on my optic** | N = 512, my optic: `2.3998e-03 / 5.0011e-04 / 1.0924e-04 / 1.1556e-04 / 1.1652e-04` at 1/2/4/8/16.  2 -> 4 gains **4.58x** (WP-C1 measured 2.04x); 4 -> 8 gains **0.945x** (5.8 % worse); 8 -> 16 0.992x.  `edge_samples=1` == `edge='hard'` bit for bit |
| 4 | all five in-library callers meant "unapodized", not "staircase" | **CONFIRMED for the five**; three binary-mask consumers found elsewhere | no in-library consumer of an `apply_aperture` RESULT assumes a binary mask, so no shipped answer is wrong.  But `optimize/wrapper_merits.py:266`, `analysis/plotting.py:1096/1491/1724/1757` and `ui/chebyshev_fit_dock.py:144` all boolean-cast a caller-supplied aperture array, and `plotting.py:1757` uses `count_nonzero` as a PIXEL COUNT.  Measured dilation of a grey mask under that cast: **+168 px (+1.37 %) on a 12281-px disc** |
| 5 | byte identity archive-to-archive | **CONFIRMED on my own 27 + 22 fixture set** | way back: **25/27** aperture and **22/22** non-aperture identical (WIN); **24/26** and **22/22** (WSL).  the move: **27/27** and **26/26** aperture moved, **0/22** non-aperture moved, both builds |
| 6 | the pre-existing JAX eager defect, its repair, jit==eager, grad finite, element keys validated | **CONFIRMED, except the key validation** (defect **D1**) | PRE eager on my 128x128 chain fixture: **3033** negative-zero real and **3066** negative-zero imaginary parts; a non-finite field left **3** non-finite values OUTSIDE the stop plus 778/853 negative zeros.  jit and NumPy: 0 and clean.  Branch: all three routes 0 negative zeros, all finite, **bit-identical on both arms on both builds**; `jax.grad` finite on default / gray / hard and under `jit`.  An unknown `edge` VALUE raises on all three routes -- but `edge_samples=2.5` and `'4'` are **accepted by the jit'd route and refused by the other two** |
| 7 | the area-vs-power restatement and the POWER band | **CONFIRMED; band re-derived and two-sided** | every reading reproduced exactly on both builds: circle AREA `+0.0746 % -> +0.0074 %`, POWER `-0.4893 %`, rim 364 px, bound 0.7415 %; rectangle AREA `+0.6400 % -> +0.0000 %`, POWER `-1.9900 %`, rim 176 px, bound 2.3467 %; deficit ladder `0.4967 / 0.2585 / 0.1297 %`.  Band checked two-sided with strictly positive slack on five of my own apertures; the deficit is **42-69 % of the bound** |
| 8 | the four re-pinned tests are decisions with derived bars; the two vacuous ones are non-vacuous | **THREE-AND-A-HALF of four** (defect **D2**) | no bar loosened, two tightened to exact.  `..._zeroes_a_blocked_pixel_even_if_it_is_not_finite` IS non-vacuous (all 3 params red under a mutated hard arm).  `..._beats_hard_at_anamorphic_and_offset_rims` is only PARTLY de-vacuumed: its `e_g4 <= e_hard` is satisfied by EQUALITY and **still passes when `edge='hard'` is mutated to return the grey mask**, while 15 other ids go red |
| 9 | the 17 new tests and their mutation matrix | **CONFIRMED by real library mutations on both builds; one survivor** | WP-C1's three rows reproduced as real source mutations (13 / 2 / 4 ids red).  Four of my own: rim off by one sub-sample **5 red**, grey mask applied twice **11 red**, chain `'edge'` key ignored **1 red**, hard arm scaling instead of selecting **8 red**.  The fifth, **the jit'd JAX kernel dropping the element's `edge_samples`, SURVIVES the entire 61-file / 2796-id sweep** |
| 10 | the Migration paragraph names every public entry point whose answer moves | **SEVEN OF EIGHT** (defect **D4**) | `lumenairy.evaluate` is missing, moves (`e7b1f67b` -> `59115e8e`, both builds) and has **no way back a caller can reach**.  Also unnamed: the GUI coronagraph dock and the `analysis.ao` module docstring's worked example |
| 11 | `Migration-Guide.md` has no 5.48.0 section | **CONFIRMED** | the section headers jump `## 5.47.0` -> `## 5.49.0`.  What 5.48.0 moved is listed in section 5 below |
| 12 | the three reds that were WP-C1's own are green, and the strip deleted no statement of record | **CONFIRMED, and 19 is the gate's own count** | the gate run against a `git archive b27a7af8` reports exactly **`19 shipped source line(s)`** (a raw grep counts 20 lines; one is exempted by the scanner, so the report's 19 is right and a grep would have been wrong).  Green on the branch.  A17: baselines **unchanged** (`apertures.py` 0, `elements.py` 15, `polarization.py` 38, `system.py` 25, total 688; the baseline JSON has no diff vs `49ddf4bd`), gate 5 passed.  Citations: exactly **6** CHANGELOG lines changed, all `file:line`, walker green.  **No record lost**: only 56 lines are deleted from `lumenairy/` in the whole branch, and the single deleted `Measured ...` line is a rewrap whose numbers (0.386 / 0.044 / 0.031 / 0.0041 %) survive verbatim two lines later |

---

## 2. What was measured, and how

### 2.1 An independent optic

WP-B11 sec. 2.9 and WP-C1 both measure lambda = 633 nm, a = 100 um, window
512 um, z = 16 mm (RS) / 5 mm (HF).  Re-running that geometry re-reads their
number.  `validation/probe_verify_c1/probe_ladder_v.py` uses

    lambda = 1064 nm,  a = 62.5 um,  window = 400 um,
    z_RS = 4.0 mm  (Fresnel number a^2/(lambda z) = 0.918),  z_HF = 2.5 mm

against the same closed form `U(0,0,z) = e^{ikz} - (z/r_a) e^{ik r_a}`, which
depends on no discretisation.  The RS alias threshold `2 W^2 / (N lambda)` is
largest at the coarsest grid (2.35 mm at N = 128) and z_RS clears it at every
N; the probe records that rather than assuming it.

| N | RS hard | RS gray | RS default | HF hard | HF gray | HF default |
|---|---|---|---|---|---|---|
| 128 | 8.8476e-03 | 1.8030e-03 | 1.8030e-03 | 1.9027e-02 | 6.2004e-03 | 6.2004e-03 |
| 256 | 2.7620e-03 | 6.6308e-04 | 6.6308e-04 | 5.9630e-03 | 1.8600e-03 | 1.8600e-03 |
| 512 | 2.3998e-03 | 1.0924e-04 | 1.0924e-04 | 5.1852e-03 | 4.2403e-04 | 4.2403e-04 |
| 1024 | 9.3571e-04 | 3.2122e-05 | 3.2122e-05 | 2.0526e-03 | 1.3795e-04 | 1.3795e-04 |
| step order | 1.68 / 0.20 / 1.36 | 1.44 / 2.60 / 1.77 | (= gray) | 1.67 / 0.20 / 1.34 | 1.74 / 2.13 / 1.62 | (= gray) |
| ladder gain | **9.46x** | **56.1x** | | **9.27x** | **44.9x** | |
| mean order | **1.08** | **1.94** | | **1.06** | **1.83** | |

Run against the PRE archive the last column reads `default is bit-identical to
['hard']` and the hard and gray columns are unchanged in every digit; run
against the branch it reads `['gray']`.  Windows and WSL agree to twelve
significant figures.

So the RATE claim reproduces on an optic two octaves away in Fresnel number:
grey is second order, hard is first order with an erratic step.  **What does
not reproduce is the non-monotonicity.**

### 2.2 The hard arm's rise is one optic's behaviour -- defect D3

On WP-B11's optic the hard arm's error rises 54.1 % (RS) and 52.9 % (HF) on
the last refinement, giving a negative order.  I reproduced all sixteen of
that table's entries to the last digit with my own code, so the measurement is
right.  But the CHANGELOG, `Migration-Guide.md` and `apply_aperture`'s own
docstring state the CONSEQUENCE without qualification:

> a circle's staircase area error does not shrink monotonically, so that arm
> has no usable order at all

On my optic it does shrink monotonically -- every hard step order is positive
(1.68 / 0.20 / 1.36 and 1.67 / 0.20 / 1.34).  The defensible general statement
is the one my ladder measures: the grey arm is second order and the hard arm
is first order with erratic steps, a mean-order gap of 0.86 (RS) and 0.77
(HF).  The negative order is a property of one (lambda, a, window, z).

This matters because the shipped test
`test_c1_the_hard_arm_is_the_arm_without_an_order_not_merely_a_worse_one`
asserts `min(hard step orders) < 0`, which is correct on ITS fixture and would
be wrong as a library property; a future re-pinning could adopt it as one.
`tests/unit/test_verify_c1_gray_edge.py::test_verify_c1_the_hard_arms_non_monotonicity_is_fixture_specific`
pins the scope from the other side.

### 2.3 Byte identity on my own fixtures

`validation/probe_verify_c1/probe_fixtures_v.py` builds **27 aperture** and
**22 non-aperture** fixtures (26 + 22 under WSL -- no CuPy there, recorded in
`unavailable` rather than skipped), with different grids, shapes, seeds and
entry points from WP-C1's set: an ODD grid, a decentred anamorphic annulus, an
anamorphic rectangle on an anamorphic grid, a complex64 call, a field carrying
NaN and +-inf, a non-default `edge_samples`, three `Aperture` operator shapes,
two `apply_lyot_stop` shapes, both JAX chain routes on two shapes, the CuPy
arm, `compute_psf` of an apertured pupil, and `lumenairy.evaluate`.

| comparison | build | aperture | non-aperture |
|---|---|---|---|
| PRE default vs branch `edge='hard'` (**the way back**) | Windows | **25 / 27** identical | **22 / 22** identical |
| | WSL | **24 / 26** identical | **22 / 22** identical |
| PRE default vs branch default (**the move**) | Windows | **27 / 27** moved | **0 / 22** moved |
| | WSL | **26 / 26** moved | **0 / 22** moved |

The two way-back exceptions on both builds are `v_chain_jax_eager` and
`v_chain_jax_eager_annular`, and they are claim 6's defect, not the flip: at
`49ddf4bd` the eager route read `386c993dff648a07` while its OWN jit'd kernel
and the NumPy chain both read `a04e643236c6bb2d` on the same input; on the
branch with `edge='hard'` it reads `a04e643236c6bb2d`.  It moved from
disagreeing with itself to agreeing with the parent commit's NumPy answer.

On the branch's default arm all three routes read `a209a3ddc2647625`.

### 2.4 The JAX repair, and the one contract that is still split -- defect D1

`validation/probe_verify_c1/probe_jax_v.py`, run against both trees on both
builds, reads identically on Windows and WSL:

| | PRE eager | PRE jit | PRE numpy | branch (all three) |
|---|---|---|---|---|
| negative-zero real / imag, 128x128 chain fixture | **3033 / 3066** | 0 / 0 | 0 / 0 | **0 / 0** |
| non-finite values left outside the stop | **3** | 0 | 0 | **0** |
| negative zeros on that fixture | 778 / 853 | 0 / 0 | 0 / 0 | **0 / 0** |
| three routes bit-identical | **no** | | | **yes, on both arms** |

`jax.grad` through the grey mask is finite (1709.9665764887113 on default and
gray, 1786.015029757675 on hard) and finite under `jax.jit`.

**D1.**  `_system_element_signature` builds the jit'd kernel's static signature
with `edge = str(edge_kw['edge'])` and `n_sub = int(edge_kw['edge_samples'])`.
The `int()` coercion means the jit'd route ACCEPTS element dicts that
`apply_aperture` -- and therefore the NumPy chain and the eager JAX route --
refuse:

| element key | `apply_aperture` / NumPy chain | JAX eager | JAX jit |
|---|---|---|---|
| `{'edge': 'soft'}` | raises | raises | raises |
| `{'edge': None}` | raises | raises | raises |
| `{'edge_samples': 0}` / `-2` | raises | raises | raises |
| `{'edge_samples': 4.0}` | accepted | accepted | accepted |
| **`{'edge_samples': 2.5}`** | **raises** | **raises** | **ACCEPTED, silently uses 2** |
| **`{'edge_samples': '4'}`** | **raises** | **raises** | **ACCEPTED** |

Measured on both builds.  WP-C1's claim is "one implementation, one default and
one way back"; that is true of the FIELD and was verified bit for bit, and it
is not yet true of the REFUSAL.

Unknown element KEYS (`{'edge_sample': 1}`, `{'gray': True}`) are silently
ignored by all three routes.  That is the chain's general convention for every
element type, so it is recorded as an observation rather than a defect -- but a
typo'd `edge_sample` now silently selects the new default where before the flip
it silently selected the old one, which is the first release in which that
silence costs anything.

### 2.3b The Migration note's own numbers, and the cost claim

Both are user-facing, so both were re-measured rather than carried over.

**How far the answer moves.**  RS spatial at z = 16 mm on WP-B11's optic,
grey against hard, relative L2 of the propagated field and worst pixel
normalised by the peak:

| N | rel. L2 (published / measured) | worst pixel (published / measured) |
|---|---|---|
| 256 | 7.678e-03 / **7.6856e-03** | 3.660e-03 / **3.6595e-03** |
| 512 | 2.930e-03 / **2.9301e-03** | 1.251e-03 / **1.2513e-03** |
| 1024 | 1.237e-03 / **1.2371e-03** | 5.434e-04 / **5.4345e-04** |

Five of six reproduce to every published digit; the N = 256 L2 differs in the
fourth digit (7.686 against 7.678, 0.1 %), which is a normalisation choice in
the denominator, not a disagreement about the move.

**What it costs.**  Rim-pixel counts on the same optic, exactly as published:
**312** boundary pixels at N = 256 (**0.476 %** of the grid) and **1196** at
N = 1024 (**0.114 %**), so the `edge_samples**2 = 16` extra indicator
evaluations are **0.0762x** and **0.0182x** of one full-grid pass.

What was NOT re-measured is the peak-memory claim in the source comment
("measured 6.0 float64 grids at N = 2048, against 5.0 for the hard edge");
that needs an allocator trace this verification did not run.

### 2.4b The claimed failure, proved out

WP-C1 says the existing cross-backend guard would NOT have caught a silent rim
divergence: *"its bar is 'fewer than 5 % of pixels mismatched', and a rim on a
64-pixel disk is about 1.5 %"*.  That is a claimed FAILURE, so it was proved
out rather than taken on trust, on that test's own fixture
(`test_audit_misc.py` `field`: N = 64, dx = 5 um, complex64, D = 97.5 um):

    mismatched pixels, edge='hard' vs edge='gray':  48 / 4096 = 1.17 %
    the test's bar:                                 5.00 %
    slack:                                          4.27x

So the claim holds with room -- a whole-rim divergence sits four times inside
that bar.  The measured figure is 1.17 %, not "about 1.5 %"; the report's
number is the only reading in it that this verification could not reproduce,
and it is conservative in the direction that matters.

### 2.5 The POWER band, re-derived

For a unit-amplitude field and a mask `f` in [0, 1]:

* `f**2 <= f` pointwise, with equality exactly at f = 0 and f = 1, so
  **POWER <= AREA**, and strictly less whenever any rim pixel exists;
* `f - f**2 <= 1/4`, attained at f = 1/2, and it is exactly 0 off the rim, so
  `AREA - POWER = sum(f - f**2) dx dy <= n_rim dx dy / 4`, i.e.
  **POWER >= AREA - n_rim dx dy / 4**.

Both sides are bounds on the mask's own range, not tolerances.  Checked
two-sided on five apertures WP-C1's restatement does not use (N = 512,
dx = 400/512 um), Windows and WSL byte-identical:

| fixture | AREA rel err (hard -> gray) | POWER rel err | rim px | bound / area | deficit / bound |
|---|---|---|---|---|---|
| centred circle | -1.65e-03 -> **-9.60e-06** | -4.14e-03 | 484 | 6.02e-03 | 0.69 |
| decentred circle (3.37, -1.83 px) | -2.58e-04 -> **+1.53e-05** | -3.98e-03 | 487 | 6.06e-03 | 0.69 |
| anamorphic rect, dy = 1.6 dx, decentred | +5.28e-03 -> **+3.02e-03** | -3.82e-03 | 588 | 9.12e-03 | 0.42 |
| anamorphic rect, centred | **+7.55e-04** -> +3.02e-03 | +1.32e-03 | 146 | 2.26e-03 | 0.55 |
| decentred annulus | +2.55e-04 -> **+3.98e-06** | -5.59e-03 | 649 | 8.24e-03 | 0.68 |

`AREA - POWER == sum(f - f^2) dx dy` to better than 1e-12 relative on every
row.  The band is never attained (0.42-0.69 of it), so it is a real bound with
room, and never violated on either side.

Two scope notes worth carrying, both visible in the table:

* on the CENTRED axis-aligned rectangle the HARD arm's area is 4x better than
  the grey arm's (+7.55e-04 against +3.02e-03).  That is the behaviour
  `test_verify_a8_e7_gray_edge_is_not_advertised_for_axis_aligned_rims`
  already pins, and it is correct -- box supersampling quantises an axis-
  aligned rim to 1/n_sub of a pixel -- but it means "the grey AREA is always
  better" is false and only the CURVED-rim form of that claim should be made.
* the grey AREA error is not monotone in N either: on the decentred circle it
  reads -6.06e-04 / +2.84e-06 / +1.53e-05 / -1.81e-05 at N = 128/256/512/1024.
  The second-order convergence in section 2.1 is a FIELD property; the area
  reading is a sub-sample quantisation residual that oscillates about zero.
  Nothing in the branch claims otherwise, but a reader could infer it from the
  validation restatement's "10x better on the circle", which is one fixture.

### 2.6 Mutation, by mutating the library

Each mutant is a fresh `git archive verify/c1-gray-edge` extraction with one
edit, run from its own tree so `lumenairy` cannot bind to the worktree.

| mutation | ids red (Windows) | ids red (WSL, WP-C1 file only) | caught by |
|---|---|---|---|
| M7 the signature default reverts to `'hard'` | 13 | 11 | `test_c1_the_default_edge_is_gray_...` + 12 others |
| M8 the `edge_samples` default moves to 8 | 2 | 2 | `test_c1_edge_samples_default_is_four_and_four_is_the_measured_knee` |
| M9 the sub-sample lattice is corner-anchored (`arange(n)/n - 0.5`) | 4 | 4 | `test_c1_a_rim_through_a_pixel_centre_reads_exactly_one_half` + the separable-area identity |
| M1 the rim fraction is off by one sub-sample (`(arange(n)+1)/n - 0.5`) | 5 | 4 | as M9, plus `test_verify_a8_e7_gray_edge_beats_hard_...[63-2.5-0.13]` |
| M2 the grey mask is applied twice (`frac**2`) | 11 | 5 | both ladders, the rim tests, `test_e7_aperture_gray_edge_removes_the_area_quantisation`, `test_circular_aperture_renders_the_disk_by_pixel_area`, all three `beats_hard` params |
| M4 the chain's `'edge'` / `'edge_samples'` keys are dropped | 1 | 1 | `test_c1_the_system_chain_takes_the_same_default_on_both_backends` |
| M5 `edge='hard'` returns the GREY mask | 15 | 11 | the way-back tests, both ladders, `test_e7_aperture_hard_edge_is_one_keyword_away_and_unchanged`, `..._is_not_advertised_for_axis_aligned_rims` -- **but NOT `..._beats_hard_at_anamorphic_and_offset_rims`** (defect D2) |
| M6 the hard arm SCALES by the boolean mask instead of selecting | 8 | 5 | `test_verify_a8_e7_gray_edge_zeroes_a_blocked_pixel_even_if_it_is_not_finite` (all 3 params) + the way-back tests |
| **M3 the jit'd JAX kernel drops the element's `edge_samples`** (`n_sub = None` in `_system_element_signature`) | **0 of 2796** | **0 of 17** | **nothing** |

M3 is a real divergence, not a no-op: with `{'edge_samples': 8}` in the element
dict the NumPy chain and the eager JAX route read `d3ca82780ee20bbd` and the
jit'd kernel reads `5608c91cd1d4ebae`.  The 61-file sweep (2782 passed, 14
skipped) is green under it on Windows, and the WP-C1 file is 17/17 green under
it on WSL.  Closed by
`test_verify_c1_the_jit_kernel_honours_the_elements_edge_samples`, which goes
red on all four `edge_samples` values under M3.

One weak row in the shipped matrix, recorded rather than filed:
`test_c1_mutation_edge_samples_moved_off_the_knee_is_caught` does not run the
named test's check function against a mutant the way the other two rows do; it
patches `apply_aperture.__defaults__` and asserts that a re-written one-line
`inspect` assertion raises.  That is close to a tautology.  M8 above is the
real mutation and the named test does catch it, so the claim is true -- the
row just is not the evidence for it.

### 2.7 The binary-mask misapplication risk

The brief asks for a downstream consumer that assumes a binary mask.  There is
none on any path an in-library `apply_aperture` result reaches, so **no shipped
answer is wrong**.  There are three places that boolean-cast an aperture array a
CALLER supplies, and before this release the natural way to build such an array
-- `apply_aperture(np.ones(...), ...)` -- satisfied the cast exactly:

| site | what it does | contract today |
|---|---|---|
| `lumenairy/optimize/wrapper_merits.py:266` | `mask = np.asarray(aperture, dtype=bool)` in the wrapper-merit grid cache, comment *"Custom user-supplied aperture array; assume boolean-coercible"* | not documented as boolean |
| `lumenairy/analysis/plotting.py:1096, 1491, 1724` | `astype(bool)` / `dtype=bool` for the NaN mask, the radial-RMS mask and the extent | documented `aperture : ndarray bool` |
| `lumenairy/analysis/plotting.py:1757` | `n_in_ap = int(np.count_nonzero(np.asarray(aperture, dtype=bool)))` -- a mask sum used as a **pixel COUNT**, feeding `_auto_n_bins` | same |
| `lumenairy/ui/chebyshev_fit_dock.py:144` | `weight = (finite & mask.astype(bool)).astype(np.float64)` -- a fit weight | GUI |

Measured: boolean-casting a grey circular mask (N = 256, dx = 4 um, D = 0.5 mm)
gives **12449** true pixels against **12281** for the hard mask and 12271.8 for
the analytic disc -- the whole rim joins the "inside" set, **+1.37 %**.  On
`plotting.py:1757` that changes `_auto_n_bins`; on `wrapper_merits.py:266` it
changes which pixels a merit integrates over.

A related inconsistency, not a defect: `lumenairy/optimize/merit_terms.py:604`
renders a prescription's `aperture_diameter` with its OWN pixel-centre mask
(`mask = (X*X + Y*Y) <= (ap/2)**2`), so after this release the optimizer's input
pupil and a chain's `'aperture'` element render the same stop two different
ways, differing by a rim.

No shipped example is affected: every `aperture=` in `examples/` is a scalar
diameter, not an array.

---

## 3. Defects

### D1 -- the jit'd JAX route accepts `edge_samples` values the other two routes refuse (P2)

**Reproducer** (both builds):

```python
import numpy as np, jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from lumenairy.propagators.system import (propagate_through_system,
                                          propagate_through_system_jax)
E = np.ones((64, 64), dtype=complex)
el = [{'type': 'aperture', 'shape': 'circular',
       'params': {'diameter': 5.3e-5}, 'edge_samples': 2.5}]
propagate_through_system(E, el, 1064e-9, dx=1.25e-6)               # ValueError
propagate_through_system_jax(jnp.asarray(E), el, 1064e-9, 1.25e-6,
                             verbose=True)                         # ValueError
propagate_through_system_jax(jnp.asarray(E), el, 1064e-9, 1.25e-6) # returns a field
```

Same with `'edge_samples': '4'`.

**Root.**  `lumenairy/propagators/system.py`, in `_system_element_signature`'s
`'aperture'` branch:

```python
        edge_kw = _aperture_edge_kwargs(elem)
        edge = (str(edge_kw['edge']) if 'edge' in edge_kw else None)
        n_sub = (int(edge_kw['edge_samples'])
                 if 'edge_samples' in edge_kw else None)
```

The coercions happen before `apply_aperture` ever sees the value, and
`apply_aperture`'s guard is exactly `n_sub != edge_samples`, which the
coercion has already made true.

**Requested edit** -- validate in `_aperture_edge_kwargs`, which is the ONE
place both backends read the element, so the signature can keep coercing
without changing what is legal:

```python
    kw: Dict[str, Any] = {}
    if 'edge' in elem:
        kw['edge'] = elem['edge']
    if 'edge_samples' in elem:
        kw['edge_samples'] = elem['edge_samples']
    if kw:
        # WP-C1 / VERIFY-C1 D1: validate HERE, so the jit'd kernel's static
        # signature (which coerces with ``int()`` / ``str()`` to stay
        # hashable) cannot accept an element the NumPy chain and the eager
        # JAX route refuse.  One reading of the element, one verdict.
        elements_mod.apply_aperture(
            np.zeros((1, 1), dtype=complex), 1.0, 'circular',
            {'diameter': 0.0}, **kw)
    return kw
```

(or, if a probe call is unwelcome, duplicate the two guards from
`apply_aperture` verbatim with a comment naming them as the same check).
Either way `_system_element_signature` then keeps its `int()` / `str()`
because by then the value is known to be an exact integer and one of the two
legal strings.

**Test that closes it.**
`tests/unit/test_verify_c1_gray_edge.py::test_verify_c1_all_three_chain_routes_agree_on_an_edge_element`,
with the two divergent rows as `xfail(strict=True)`.  `xfail_strict = true` is
on in `pyproject.toml`, so both rows FAIL the moment the coercion is removed
and the markers must go with it.

### D2 -- "grey BEATS hard" is asserted with `<=`, which equality satisfies (P3)

`tests/unit/test_audit2609_a8_verify.py:614`:

```python
    assert e_g4 <= e_hard, (e_g4, e_hard)
```

WP-C1 correctly restored `e_hard = abs(_area(edge='hard') / analytic - 1.0)`
after the default move turned it into `e_g4 <= e_g4`.  But the restored
assertion still cannot distinguish "grey beats hard" from "grey IS hard":
under M5 (`edge='hard'` mutated to return the grey mask) fifteen ids go red and
**this is not one of them**, on either build.

**Measured margins** on its own three fixtures (N = 512, dx = 1 um; identical
on both builds -- a mask sum is an integer count over `n_sub**2` with no BLAS
in it):

| fixture | `e_hard` | `e_g4` | `e_hard / e_g4` |
|---|---|---|---|
| D/dx = 37, dy/dx = 1.0, offset 0.37 px | 4.8456e-03 | 8.3480e-04 | **5.80** |
| D/dx = 63, dy/dx = 2.5, offset 0.13 px | 7.2029e-04 | 8.1703e-05 | **8.82** |
| D/dx = 145, dy/dx = 0.4, offset 0.29 px | 6.0777e-05 | 3.0498e-05 | **1.99** |

**Requested edit** (`tests/unit/test_audit2609_a8_verify.py`, replacing line
614, and one sentence in the docstring's "Bars" paragraph):

```python
    # VERIFY-C1 D2: a RATIO, not ``<=``.  ``e_g4 <= e_hard`` is satisfied by
    # EQUALITY, so it passes when the two arms are the same array -- measured:
    # with ``edge='hard'`` mutated to return the grey mask, 15 ids go red and
    # this was not one of them.  MEASURED 2026-09-20 on both builds:
    # e_hard / e_g4 = 5.80 / 8.82 / 1.99 on the three fixtures, so the 1.5x
    # bar sits 1.33x below the smallest and decisively above the 1.0 where
    # "grey IS hard" lives.
    assert e_hard / e_g4 >= 1.5, (d_px, dy_ratio, offset, e_hard, e_g4)
```

Closed independently here by
`test_verify_c1_grey_strictly_beats_hard_by_a_measured_ratio`, which goes red
on all three params under M5.

### D3 -- the "no order at all" claim is stated as a library property (P3, documentation)

The rise is measured on one optic.  On lambda = 1064 nm, a = 62.5 um, window
400 um, z = 4 mm / 2.5 mm the hard arm falls at every refinement.

**Requested edits** (three places, same sentence):

* `CHANGELOG.md`, `## [Unreleased]`, the paragraph beginning "The hard arm's
  error RISES on the last refinement": append
  *"-- on this optic.  The RATE gap is the general claim and reproduces
  elsewhere (an independent verification on lambda = 1064 nm, a = 62.5 um,
  window 400 um, z = 4.0 / 2.5 mm reads 1.94 and 1.83 for the grey arm against
  1.08 and 1.06 for the hard arm over the same three halvings); whether the
  staircase error actually RISES depends on where the rim falls on the lattice
  at each N."*
* `Migration-Guide.md` section 5.49.0, "### Why": change *"the hard rim's error
  does not even shrink monotonically -- it RISES 54 %..."* to *"the hard rim's
  error is first order at best and its step orders are erratic; on the
  reference optic it actually RISES 54 % from N = 512 to N = 1024 on the
  Rayleigh-Sommerfeld spatial kernel and 53 % on the Huygens-Fresnel OPL
  quadrature."*
* `lumenairy/elements/elements.py`, `apply_aperture` docstring: change *"The
  hard edge has **no convergence order at all** -- a circle's staircase area
  error does not shrink monotonically, hence the negative last step"* to *"The
  hard edge is first order at best and its step orders are erratic -- a
  circle's staircase area error need not shrink monotonically, and on this
  optic it does not, hence the negative last step -- while the grey edge is
  second order."*

Pinned from the other side by
`test_verify_c1_the_hard_arms_non_monotonicity_is_fixture_specific`.

### D4 -- `lumenairy.evaluate` moves and has no way back, and is not in the Migration table (P2)

`lumenairy.evaluate(prescription, source)` is the documented one-call entry for
a `.zmx` prescription.  `_prescription_to_elements` routes the Zemax-loader
shape through `io.codegen._decompose_prescription`, which emits an
`{'type': 'aperture', 'shape': 'circular', 'params': {'diameter': D}}` step for
every `is_stop=True` surface (`system.py:1421-1427`), with **no `edge` key**.
So `evaluate` takes the new default, its answer moved, and unlike every entry
point in the Migration table it has **no route back**: `evaluate` accepts no
`edge` argument and builds its element list internally.  The only way back is
to call the private `_prescription_to_elements`, inject the key and drive
`propagate_through_system` by hand.

**Reproducer**: `validation/probe_verify_c1/probe_fixtures_v.py`, fixture
`v_evaluate_prescription_stop` -- `e7b1f67b9b19d547` at `49ddf4bd`,
`59115e8eb2b2b0d0` on the branch, both builds.

**Requested edits**:

1. `Migration-Guide.md` section 5.49.0, "What moves" table -- add a row:

   | entry point | exposes `edge=`? | the way back |
   |---|---|---|
   | `lumenairy.evaluate` on a prescription with a STOP surface | no | re-record, or drive `propagate_through_system` with the element list yourself and add `'edge': 'hard'` to the `'aperture'` element |

2. `CHANGELOG.md`, the Migration paragraph's list of entry points -- insert
   `lumenairy.evaluate` after "`propagate_through_system_jax` (both its jit'd
   and its slow route)", with the same note.

3. Recommended, not required: give `evaluate` an `aperture_edge=None` keyword
   that `_prescription_to_elements` stamps onto every `'aperture'` element it
   emits, so the way back is reachable without a private call.  That is a new
   argument and therefore the maintainer's call, not this verification's.

Two more entry points move and are unnamed; neither has a way back either, and
both are reached only through code that is itself documented elsewhere, so
they are recorded rather than filed:

* `lumenairy/ui/coronagraph_dock.py:370` calls `apply_lyot_stop`, so Stop 3's
  displayed field moves.  `GUI_CHANGELOG.md` has no entry for this release.
* `lumenairy/analysis/ao.py:33`, inside the module docstring's worked AO loop,
  builds its pupil with `la.apply_aperture(np.ones((N, N), dtype=complex), ...)`.
  The example is still correct (it multiplies the pupil into a field, which is
  exactly what a grey amplitude mask is for), but its printed numbers move.

---

## 4. The observations that are not defects

* **The five named in-library callers are the right five.**  A grep for
  `apply_aperture` across `lumenairy/` finds exactly those five executing call
  sites plus the two JAX helpers, which are the same chain element.  Every one
  of the five docstrings was restated; the reasoning ("hard-edge" meant
  unapodized) is correct in each case.
* **One docstring was missed.**  `lumenairy/algebra/apertures.py` gained its
  Notes paragraph but its class summary line still reads
  *"Hard amplitude aperture with selectable shape."* and the section banner
  above it still reads *"Aperture (hard amplitude mask)"*.  After this release
  "hard" is a keyword VALUE, which is precisely why WP-C1 restated
  `elements.py`'s module header from "hard-edge amplitude masks" to
  "sharp-edged (unapodized)".  Suggested edits: *"Sharp-edged (unapodized)
  amplitude aperture with selectable shape."* and *"Aperture (sharp-edged
  amplitude mask)"*.
* **`.test_durations` carries no entry for `test_c1_gray_edge_default.py`.**
  The staleness gate does not require one for a new file (4 passed), and
  WP-C1 says so, but the 17 ids are invisible to `pytest-split`'s balancing and
  will land in whichever shard collects them.  This verification's 28 ids ARE
  spliced (16 596 -> 16 624 entries, re-parsed as JSON, gate still 4 passed).
* **The report's one remaining Windows red is gone.**
  `test_public_api.py::test_installed_metadata_version_matches_source_version`
  now passes: the box's editable install reads `5.48.1` against a `5.48.1`
  source.  `tests/unit/test_public_api.py` is 9 passed.
* **`np.array_equal(...) is not has_partial_pixels`** in the re-pinned
  `test_e7_aperture_hard_edge_is_one_keyword_away_and_unchanged` is sound --
  numpy 2.4 returns a Python `bool`, not `np.bool_`, so the identity
  comparison is not silently always-true.  Checked, because that shape is a
  classic vacuity.

---

## 5. Item 11 -- what a 5.48.0 Migration section would have to say

`Migration-Guide.md`'s section headers go `## 5.47.0 -- adversarial audit
remediation, Wave 4` straight to `## 5.49.0`.  5.48.0 moved two answers and one
schedule; its CHANGELOG block carries three `**Migration.**` paragraphs, whose
substance the guide is missing:

1. **FGA reference plane.**  Every field returned by `apply_real_lens_fga`,
   `apply_real_lens_fga_vector`, `apply_real_lens_universal(method='fga')` and
   `apply_real_lens_auto` when it dispatches to FGA changes on any prescription
   whose LAST surface is curved, as does every `_caustic_zone` estimate on such
   a prescription and therefore some routing decisions near a caustic-zone
   edge.  Wrong to right (0.07-0.53 fidelity to 0.999-class against an
   independent diffraction oracle).  **No keyword restores the old answer**; a
   FLAT last surface is bit-identical.  Callers who pinned FGA digests on a
   curved-last-surface prescription re-record.
2. **GBD per-surface projection.**  Every field returned by
   `apply_real_lens_gbd`, `apply_prescription_persurface_to_beamlets`,
   `propagate_gbd_through_prescription(per_surface=True)` and
   `apply_real_lens_universal(method='gbd')` moves on any prescription whose
   LAST surface is an even asphere, a biconic, a freeform, a field-frame
   decentred / tilted surface or a mirror.  A CONIC last surface moves in the
   fifth decimal (5.5e-05 relative in `Q`, 6.9e-06 in amplitude, 8.9e-06 rad of
   relative phase; base-ray positions and optical paths bit-identical); a FLAT
   last surface and the `world_output_plane` branch are bit-identical.  **No
   keyword restores the old answer.**
3. **New refusals, no answer moved.**  The per-surface GBD local branch now
   refuses an immersed exit medium and a mirror-terminated prescription.  No
   prescription the library serves today is affected (byte-identical
   archive-to-archive); a caller who really terminates in a medium adds the
   immersion medium as an explicit last element.
4. **The deprecation horizon slips 5.48 -> 5.50.**
   `NEXT_REMOVAL_VERSION` moves to `'5.50'` and `REMOVAL_SCHEDULE` gains
   `{'5.48': '5.50'}`, so the three GBD aliases `gbd_field_to_asm`,
   `asm_field_to_gbd`, `match_global_phase` and the `CarrierField`
   attribute-assignment freeze all keep working and their warnings say
   "rescheduled from v5.48".  Nothing is removed in 5.48.0.

5.48.1 moved nothing (one test repair), so it needs no section.

---

## 6. The runs

All with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the
command line, `-p no:randomly --capture=sys -q`, `PYTHONPATH` naming the tree
under test.

| what | Windows py3.14 | WSL py3.12 |
|---|---|---|
| the 61-file sweep -- 34 aperture-touching files + census + walkers + dispatcher pins + public API + doc consistency + A17 + `test_audit_except_budget.py` + both C1 files | **2782 passed, 14 skipped, 0 failed** in 29:36 | **2771 passed, 21 skipped, 4 failed** in 31:05, all four premise-gated (below) |
| `tests/unit/test_verify_c1_gray_edge.py` | **26 passed, 2 xfailed** in 30.1 s | **26 passed, 2 xfailed** in 30.9 s |
| `tests/unit/test_c1_gray_edge_default.py` (inside the sweep; also standalone under every mutant) | 17 passed | 17 passed |
| `tests/unit/test_audit2609_a17_history_lint.py` | **5 passed** | (in sweep) |
| `tests/unit/test_public_api.py` | **9 passed** | (in sweep) |
| `tests/unit/test_audit2609_a15a_durations_staleness.py` (after splicing 28 ids) | **4 passed** in 1:46 | -- |
| `validation/elements/test_elements.py` | **31/31** | **31/31**, every reading byte-identical |
| `validation/elements/test_doe.py` | **16/16** | -- |
| `validation/propagators/test_hfpi.py` | **12/12** | -- |
| `ruff check lumenairy/ tests/ scripts/` (WSL) | -- | **All checks passed!** |
| `python -m mypy` (no args) | **Success: no issues found in 33 source files** | -- |
| `python scripts/record_history_fingerprints.py --check` | **OK: every history document matches its module** (rc 0) | -- |

**The four WSL reds, premise-gated, none a library finding** -- and note that
the Windows lane is now 0 failed, because the box's editable install has since
been refreshed (`test_public_api.py` is 9 passed there and
`importlib.metadata.version('lumenairy')` reads `5.48.1`):

* `test_public_api.py::test_installed_metadata_version_matches_source_version`
  -- the WSL venv's editable install reads **5.11.0** against a 5.48.1 source.
  It fails identically on a clean `git archive 49ddf4bd` extraction under WSL,
  so it predates this work package; the remedy is `pip install -e .` in
  `~/lumvenv`, which is the box's to do.
* `test_v5_3_2_walker_source_line_citation.py::test_v18_5_the_5_47_0_block_citations_name_the_right_lines`
  and `::test_v18_5_companion_reanchor_tool_exists_and_covers_the_cited_files`
  -- both shell out to `git`, which from WSL cannot resolve this worktree's
  `.git` file (it points at a Windows path).  Both fail identically on the base
  tree under WSL and both are green on the Windows lane on this commit.
* `test_v5_2_3_walker_changelog_content.py::test_v16_synthetic_fabrication_is_caught`
  -- same root: the walker returns **rc = 2** ("the git plumbing failed")
  where the test expects rc = 1 ("fabrication flagged").  Green on Windows.
  Worth a separate, pre-existing note: this id's failure message reads *"This
  means the walker is silently passing fabrications -- a regression in the V16
  contract itself"*, which is wrong for rc = 2; the id conflates "git failed"
  with "fabrication not flagged", and only the rc value distinguishes them.
  It is not WP-C1's to fix.

---

## 7. What could not be measured

* **CuPy under WSL** -- absent.  The CuPy arm is measured on Windows only, and
  the probe records the absence in `unavailable` rather than skipping.
* **The JAX slow path on a GPU device.**  Measured on the JAX CPU backend on
  both builds.  The signed-zero difference is an XLA rewrite difference between
  jit and eager and is not expected to be device-specific, but that is
  reasoning, not a measurement.
* **A generated script executed end to end.**  `io/codegen.py` emits an
  unkeyworded `la.apply_aperture(...)` and `test_audit_io.py` pins the TEXT;
  this verification did not run an emitted script against both trees either.
* **The whole 14 666-id unit suite.**  What was run is the 61-file / 2796-id
  sweep, the two C1 files under nine mutants, and four validation files.  A
  release gate needs the full matrix.
* **Whether the 5 %-of-pixels bar in
  `test_audit_misc.py::test_jax_aperture_matches_numpy` should be tightened.**
  The two backends are now byte-identical, so that bar is satisfied with the
  whole margin; tightening it is B1-1's decision, and this verification's
  `test_verify_c1_the_jit_kernel_honours_the_elements_edge_samples` asserts the
  exact claim on the chain instead.
* **Whether any external caller passes an `apply_aperture` result into
  `plot_wavefront(aperture=...)`, `plot_opd_summary(aperture=...)` or a wrapper
  merit's array `aperture`.**  The dilation is measured (+1.37 %) but the
  exposure is not: nothing in `lumenairy/`, `validation/` or `examples/` does
  it, and what users do cannot be measured from here.

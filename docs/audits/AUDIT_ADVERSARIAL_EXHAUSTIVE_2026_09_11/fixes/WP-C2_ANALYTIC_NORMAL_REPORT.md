# WP-C2 -- the ray tracer's `sphere_normal` and `renormalize` defaults (5.49.0)

Brief: make `sphere_normal='analytic'` and `renormalize='exit'` the defaults
of `trace` / `trace_world`, on the maintainer's decision in
`MAINTAINER_DECISIONS_2026_09.md` section 1.3, after restating the two pins
WP-B9 section 5 items 2 and 3 named.  The domain clamp in `_sphere_normal`
STAYS: VERIFY-B9 3.2 measured that the closed form is NOT well-conditioned at
the rim, so dropping it is a separate vignetting decision nobody has taken.

Base: `49ddf4bd` (`origin/main`; the local `main` ref in this checkout was
stale at `e995f00e`, one commit behind, and the branch was reset onto
`49ddf4bd` before any work).  Branch `feat/c2-analytic-normal-default`,
worktree `C:/tmp/lum_c2`.  Not pushed.

Builds: Windows py3.14.6 / numpy 2.4.4 and WSL py3.12.3 / numpy 2.4.6, every
probe on both, every probe in a child process with `LUMENAIRY_ROOT` on
`sys.path` and `lumenairy.__file__` asserted inside it, all with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line.

---

## 0. Headline

1. **Both B9 reports' explanation of the two knife-edge pins is measurably
   wrong, in both cases.**  Neither pin's quantity is bimodal and neither has
   a saddle-basin flip in it.  The ModalAsymptotic arms are at the
   *cancellation floor* of a field whose conditioning this build can measure
   (`kappa = 6.24e+06`, so `eps * kappa = 1.39e-09`, and the two routines sit
   at 7.3x to 9.0x it).  The `w6_a2` root is *genuinely off centre*: a
   60-digit `decimal` Newton on the same polynomial system reproduces the
   library's answer to 4.5e-22, and the offset is `H^-1 r(v_c)` where
   `r(v_c)` is the least-squares fit's own asymmetry, resolved eight decades
   above its rounding floor.  `1e-15` was a bound on how asymmetric a fit
   happened to come out.  Both pins are now decisions with bars the running
   build derives, and are green under all four default combinations on both
   builds.
2. **`sphere_normal='analytic'` is the better route over the working aperture
   and is now the default.**  1.75 ULP against a 60-digit oracle out to
   `h = 0.95 |R|` (generic: 2.00 Windows, 2.25 WSL), never worse there by more
   than 1 ULP at any of 672 points, unit to 1.5 ULP by construction.  Above
   `0.95 |R|` neither route dominates -- that is stated in the code, in the
   CHANGELOG and in an arm of the test file, because it is the honest half of
   the claim.
3. **Timing: 1.08x to 1.44x, and the measurement's own resolution is +-7 %.**
   The controls (prescriptions with no pure sphere, where the switch cannot
   change anything) read 0.93x to 1.03x.  The box carried other agents' load
   throughout: 67 % to 100 % CPU, 10 to 21 concurrent python processes.  The
   contention-immune number is the profile share of the normal block:
   16.1 % -> 9.8 % (Windows), 19.6 % -> 10.4 % (WSL).
4. **`renormalize='exit'` is NOT measurable as a speed-up on either build.**
   Range 0.93x to 1.13x, median 1.00x (Windows) and 0.99x (WSL) -- inside the
   same +-7 % resolution the controls set.  WP-B9's 1.03x-1.10x does not
   reproduce here.  See section 4 for what was done about it.
5. **Vignetting: one rim band moves, nothing else.**  A directed `nextafter`
   walk constructs the straddle point VERIFY-B9 3.3 predicted and it reaches
   `_refract`'s `alive` flag and `trace`'s error code.  360 000 traced rays
   over twelve prescription and field-angle combinations, three of them
   shipped builders, move ZERO alive flags and ZERO error codes.  No shipped
   fixture's vignetting count changes, so there is no moved vignetting fixture
   to re-pin -- and the Migration note names the band anyway.
6. **The JAX tracer has no such switches and never did.**  It has always used
   a closed-form sphere normal and has no per-surface rescale to hoist, so
   this flip moves the CPU tracer TOWARD it.  CPU/JAX parity is 3.5e-18 m in
   position and 3.1e-17 m in OPL under ALL FOUR CPU settings, identical to the
   last digit, with the alive masks equal, on both builds.

---

## 1. Item 1 -- the two pins, restated (commit `bdb3ae73`)

### 1.1 What WP-B9 and VERIFY-B9 said, and what is actually there

Both reports describe one mechanism for both pins: "one knife-edge pixel
changing saddle basin", "the quantity is bimodal between ~0 and ~1e-8",
VERIFY-B9's "4.0 % of margin on a bimodal quantity is not a pin, it is a
coin".  The measurements are in `validation/probe_c2_analytic_normal/`.

**ModalAsymptotic (`modal_mechanism.py`, `modal_conditioning.py`).**

| measurement | Windows | WSL |
|---|---|---|
| saddle location, batch vs scalar, worst over 1024 pixels | 7.29e-18 | 6.62e-18 |
| the same, as a fraction of the pupil half-range | 8.70e-17 | 7.90e-17 |
| pixels whose saddle moved basin | **0** | **0** |
| correlation, saddle disagreement vs field disagreement | 0.063 | -0.003 |
| pixels with field disagreement > 1e-12 relative (of 1024) | 996 | 1000 |
| median / p99 / max field disagreement, relative | 2.1e-9 / 8.1e-9 / 1.03e-8 | 2.2e-9 / 8.7e-9 / 1.11e-8 |
| FD conditioning `kappa` (4-point ladder, 1e-13..1e-10) | 6.2423e+06 .. 6.2639e+06 | 6.2423e+06 .. 6.2638e+06 |
| `eps * kappa` | 1.3866e-09 | 1.3866e-09 |
| reading / floor | 7.4x | 8.0x |

So the quantity is continuous and dense, not bimodal; the two implementations
differ by a handful of roundings on a field that amplifies its own inputs by
six decades.  `max|new - cold_ref|` and `max|new - warm_ref|` agree to five
figures, i.e. the two references are themselves at the same floor (2.93e-09
apart), so the warm/cold distinction the pin's history is about does not
produce the reading either.

**`w6_a2` (`w6a2_oracle.py`, `w6a2_asymmetry.py`, `w6a2_resolution.py`).**

| measurement | Windows | WSL |
|---|---|---|
| `|v2*|` at the fit centre, four default combinations | 4.97e-16 .. 8.64e-16 | 4.09e-16 .. 6.40e-16 |
| 60-digit `decimal` Newton on the same polynomial system | agrees to **3.1e-22** | agrees to **4.5e-22** |
| oracle self-consistency, two independent starts | 1.0e-65 | 4.1e-66 |
| `|r(v_c)|` in float64 | 4.84e-08 .. 8.42e-08 | 4.40e-08 .. 8.71e-08 |
| the same at 60 digits, relative agreement | 3.3e-07 | 8.8e-07 |
| `sigma_min(H)` | 9.7413e+07 | 9.7413e+07 |
| `|offset|` vs the single Newton step `H^-1 r(v_c)` | **9.9e-32** | 0 .. 9.9e-32 |
| scalar solver's spread over 45 independent Newton starts | 2.1e-21 | 1.8e-21 |

The solver finds the same point from every start in the basin to 2e-21, and
that point is 5e-16 from the origin -- five decades apart, so the offset is not
solver noise.  It is the root of the computed system, and the computed system's
root is not at the centre because the *fitted coefficients* are not symmetric.
`1e-15` bounded that asymmetry.

### 1.2 The restatements

`tests/unit/test_audit_propagation.py` -- `_conditioning_bar` measures `kappa`
with a 3-point FD ladder over the fit's phase coefficients, asserts the ladder
is linear (so it is conditioning and not a threshold), asserts `kappa > 1e3`
(the premise that makes `eps * kappa` the right bar at all), and sets
`bar = 100 * eps * kappa`.  `_assert_at_conditioning_floor` asserts the reading
is under the bar AND that an injected drift -- sized from the measured `kappa`
to land two decades above it -- is still refused.  That is the fail-before
demonstration on the running build.

`tests/unit/test_niche_audit_w6_asymptotic.py` -- two decisions: the returned
expansion point equals the model's own one-Newton-step root from the pupil
centre (bar `4 eps x half-range = 4.49e-18`, the float64 resolution of `v2` on
its own scale; reading 0 to 9.9e-32), and that root is the pupil centre at the
fit's own resolution (reading 2.2e-14 to 4.4e-14 of the normalised box against
a 1e-10 decision bar, with the derived `|r|/sigma_min(H)` bound asserted
alongside and the premise that `r(v_c)` is resolved asserted separately).

### 1.3 Green under either default -- the gate WP-B9's order required

`pins_restated.py` runs all three arms under all four
`(renormalize, sphere_normal)` combinations, flipping `trace.__defaults__` /
`trace_world.__defaults__` on the function objects exactly as VERIFY-B9 section
4 did (and clearing the `lru_cache`d, TRACED `w6._fit` between combinations --
forgetting that silently reports the first combination's number four times).

| build | combination | lg00 | 4-mode | w6_a2 | reading | bar | margin |
|---|---|---|---|---|---|---|---|
| Windows | surface/generic | PASS | PASS | PASS | 1.0331e-08 | 1.3862e-07 | 13.4x |
| Windows | surface/analytic | PASS | PASS | PASS | 1.2478e-08 | 1.3862e-07 | 11.1x |
| Windows | exit/generic | PASS | PASS | PASS | 1.1015e-08 | 1.3862e-07 | 12.6x |
| Windows | exit/analytic | PASS | PASS | PASS | 1.0421e-08 | 1.3862e-07 | 13.3x |
| WSL | surface/generic | PASS | PASS | PASS | 1.1062e-08 | 1.3862e-07 | 12.5x |
| WSL | surface/analytic | PASS | PASS | PASS | 1.2433e-08 | 1.3862e-07 | 11.1x |
| WSL | exit/generic | PASS | PASS | PASS | 1.0789e-08 | 1.3862e-07 | 12.8x |
| WSL | exit/analytic | PASS | PASS | PASS | 1.0146e-08 | 1.3862e-07 | 13.7x |

Note the first row: at the SHIPPED defaults this box reads 1.0331e-08, i.e. the
pre-WP-B9 `1e-8` bar was already red here, independent of any default.  That is
what "a coin" looks like from the other side.

---

## 2. Item 2 -- the `sphere_normal` flip

### 2.1 The oracle ladder, on WP-C2's own sphere set

`sphere_oracle.py`: eight radii of both signs (2 mm, 34.5 mm, 51.5 mm, 80 mm,
120 mm, 500 mm, 1 m), eleven heights from the vertex to `0.99994 |R|`, six
azimuths, each set evaluated twice -- once refracting, once with the surface
declared a MIRROR -- 1056 points per build.

| quantity | Windows | WSL |
|---|---|---|
| closed form, worst ULP out to `h = 0.95 |R|` (672 points) | **1.75** | **1.75** |
| generic route, same | 2.00 | 2.25 |
| points where the closed form is worse by > 1 ULP, `h <= 0.95 |R|` | **0** | 0 |
| closed form, worst ULP over the whole set | 57.47 | 57.47 |
| generic route, same | 76.30 | 76.30 |
| unit-vector defect of the closed form, worst | 1.0 ULP | 1.5 ULP |
| closer / further / tie | 596 / 112 / 348 | 588 / 118 / 350 |
| normal depends on `glass_after` (mirror vs refracting) | no | no |
| domain-gate disagreements on this grid | 0 | 0 |

Per height (Windows, worst over radii and azimuths):

| `h/|R|` | 0 | 0.05 | 0.5 | 0.95 | 0.99 | 0.999 | 0.9999 | 0.99994 |
|---|---|---|---|---|---|---|---|---|
| closed form, ULP | 0.00 | 0.03 | 0.50 | 1.75 | 5.38 | 13.06 | 47.55 | 57.47 |
| generic, ULP | 0.00 | 0.50 | 1.00 | 2.00 | 8.88 | 17.34 | 58.32 | 76.30 |

At 22 of the 1056 points (all at `h >= 0.999 |R|`) the closed form rounds
WORSE, by up to 35 ULP.  That is VERIFY-B9 3.2's finding reproduced: above
`0.95 |R|` both routes are at the conditioning limit of `sqrt(1 - u)` and
neither dominates point by point.

### 2.2 Timing

`timing.py`, five prescriptions, `N = 100 000` rays, 20 traces per timing
sample, 9 repeats with the arm order alternating each repeat, minimum over
repeats, wall clock and process CPU time both recorded.  Windows' CPU clock
ticks at 15.6 ms, so an unbatched 0.25 s trace is quantised at 6 % and the
no-sphere CONTROLS read a 20 % "speed-up" off one tick -- the batching is not
optional.

Windows (CPU time; load recorded at 67 % -> 100 % CPU, 10 python processes):

| prescription | surfaces | generic | analytic | speed-up |
|---|---|---|---|---|
| spherical, 7 surfaces | 7 | 0.0570 s | 0.0508 s | 1.123x |
| Cooke-like triplet | 6 | 0.0523 s | 0.0477 s | 1.098x |
| Cassegrain, 2 spherical mirrors | 3 | 0.0227 s | 0.0180 s | 1.261x |
| conic, no sphere (CONTROL) | 3 | 0.0336 s | 0.0336 s | 1.000x |
| aspheric, no sphere (CONTROL) | 3 | -- | -- | contention spike, discarded |

WSL (CPU time; controls 0.975x and 1.014x):

| prescription | generic | analytic | speed-up |
|---|---|---|---|
| spherical, 7 surfaces | 0.1055 s | 0.0945 s | 1.116x |
| Cooke-like triplet | 0.1148 s | 0.0797 s | 1.441x |
| Cassegrain, 2 spherical mirrors | 0.0492 s | 0.0414 s | 1.189x |

**Range: 1.08x to 1.44x; medians 1.12x (Windows) and 1.19x (WSL).**  The
controls set the resolution at +-7 %, and one aspheric-control sample on
Windows read 1.66x on a run where the same arm's absolute time quadrupled --
a contention spike, reported rather than averaged away.  The structural
measure, which contention cannot move because both arms are profiled under it:
the normal block's share of `trace`'s own tottime falls from **16.081 % to
9.844 %** (Windows) and **19.632 % to 10.443 %** (WSL).

### 2.3 Vignetting

`vignetting.py`.

* **The straddle exists.**  A directed `nextafter` walk over eight radii and
  six azimuths, 60 steps each side of `h = |R| sqrt(0.9999)`, finds it:
  `R = -0.12 m`, `x = 0.0544288129262979`, `y = 0.10693953582952408`,
  `h/|R| = 0.9999499987499374`.  There the closed form's gate says VALID and
  the generic route's says out of domain.
* **It reaches the answer.**  Through `_refract`: `alive=False, code=4`
  (`RAY_NAN`) generic against `alive=True, code=0, L=0.3381, N=0.6667`
  analytic.  Through the public `trace` on a two-surface stack with the
  downstream aperture opened wide: `code=4` generic against `code=1`
  (`RAY_TIR`) analytic -- dead either way there, but for an honest physical
  reason instead of an arithmetic fault.
* **Nothing that is not aimed at it lands in it.**  360 000 rays over twelve
  combinations: a sphere with its clear aperture opened to `0.99999 |R|`; the
  seven-surface spherical stack at 0, 3 and 8 degrees; the two-mirror
  Cassegrain at 0 and 2 degrees; and three shipped builders
  (`make_singlet` 20/-20, `make_singlet` 51.5/inf, `make_doublet`
  51.7/-34.5/-120) at 0 and 5 degrees.  **Zero** alive flags moved, **zero**
  error codes moved, on both builds.

**So no shipped fixture's vignetting count changes, and there is no moved
vignetting fixture to re-pin.**  The Migration note names the band anyway,
because the change is real even though no current fixture samples it.

### 2.4 The clamp

Unchanged, per the ledger.  `test_c2_the_domain_clamp_stays_where_the_ledger_left_it`
LOCATES the threshold by 80-step bisection on the running build (so a change to
the expression, not only to the literal, is caught) and asserts a ray at
`0.99995 |R|` still dies `RAY_NAN` through the generic route.

---

## 3. Item 4 -- the JAX tracer

`jax_trace` has **neither switch**, and the reason is structural rather than an
oversight:

* `_refract_jax` has always used a closed-form sphere normal, `(x, y, z - R)/R`
  taken at the intersection point its own Newton returned, for every surface
  with a finite radius and no conic or aspheric term (`jax_trace.py:417-420`).
  There is no `sphere_normal` to flip.
* it has no per-surface renormalisation to hoist: the shared `refract_snell`
  core returns a unit vector from a unit normal and the JAX body never
  rescales.  There is no `renormalize` to flip.

So the CPU flip moves the CPU tracer TOWARD the JAX one.  `jax_parity.py`, four
prescriptions x two field angles x five CPU settings, both builds:

| CPU setting | worst `|dx|` | worst `|dopd|` | alive masks equal |
|---|---|---|---|
| generic / surface | 3.469e-18 m | 3.123e-17 m | yes |
| analytic / surface | 3.469e-18 m | 3.123e-17 m | yes |
| generic / exit | 3.469e-18 m | 3.123e-17 m | yes |
| analytic / exit | 3.469e-18 m | 3.123e-17 m | yes |
| the library default | 3.469e-18 m | 3.123e-17 m | yes |

Identical to the last digit on both builds: the parity floor is set by the JAX
Newton's own intersection arithmetic, not by the normal route, so the flip
moves no parity pin.  The default arm is byte-equal to the explicit
`analytic` arm, which is the same claim the signature makes, asked from a call.

---

## 5. Item 6 -- byte identity, archive to archive

`byte_identity.py`, run four times (two roots x two modes) per build.  Source
of truth: `git archive 49ddf4bd` extracted read-only into the scratchpad; each
side runs in its own child process with cwd AND `PYTHONPATH` set to its root
and `lumenairy.__file__` asserted inside it before any other import.  Never
under pytest.

Coverage: 1008 arrays, 1 630 399 values -- five stacks (spherical 7-surface,
conic, aspheric, 2-mirror Cassegrain, biconic) at three field angles under both
`output_filter` modes with every history bundle recorded, a DOE-kicked trace,
`trace_world`, and then the consumers: `ray_fan_data`, `opd_fan_data`,
`trace_prescription`, `spot_rms`, `spot_geo_radius`, `refocus`,
`seidel_coefficients`, `first_order_data`, `compute_pupils`,
`system_abcd_prescription`, `find_paraxial_focus`, `through_focus_rms`,
`enumerate_ghost_paths` and `ghost_analysis`, on two prescriptions at two
fields; plus seven NON-trace fixtures (`angular_spectrum_propagate`, the
Chebyshev Vandermondes, the glass indices).

| claim | Windows | WSL |
|---|---|---|
| both old keywords passed explicitly | **938 / 1008 identical** | 938 / 1008 |
| ... of which direct `trace` / `trace_world` arrays | **all identical, 0 moved** | 0 moved |
| ... the 70 that moved | `trace_prescription` 25, `refocus` 25, `ray_fan_data` 8, `opd_fan_data` 8, `spot_rms` 3, `through_focus` 1 | same |
| non-trace fixtures, no keyword | **7 / 7 identical** | 7 / 7 |
| defaults vs defaults (the full blast radius) | 547 / 1008; 461 moved | 547 / 1008; 461 moved |
| ... worst absolute / relative move | 1.796e-11 / 3.366e-13 | same |

The 70 exceptions are **the finding**: `trace_prescription`, `raytrace_system`,
`ray_fan_data`, `opd_fan_data`, `through_focus_rms` and the result-consuming
`spot_rms` / `spot_geo_radius` / `refocus` all trace internally and expose no
`sphere_normal` keyword, so there is no one-keyword way back through them.
Threading the two keywords through five public functions is an API expansion
the ledger did not ask for, so it is documented in the Migration note and
raised as a follow-up (section 8) rather than done here.

---

## 9. What could not be measured

1. **A clean absolute timing number.**  The box carried another agent's work
   for the whole work package (67 % to 100 % CPU, 10 to 21 concurrent python
   processes, recorded in every `timing_*.json`).  The controls bound the
   method at +-7 % and one control sample spiked to 1.66x.  The speed-up
   RANGE and the profile SHARE are sound; a single number is not available
   from this box today.
2. **Whether the rim band matters to any real design.**  It is about 1 ULP of
   `h` wide, so sampling cannot find it -- 360 000 rays found none -- and the
   only evidence that it is reachable at all is a directed `nextafter` walk.
   Whether a user's prescription works rays there is not something this work
   package can answer; the Migration note tells them how to find out.
3. **Cross-build identity.**  Deliberately not claimed anywhere: every byte
   identity comparison is archive-to-archive on the SAME build, and every bar
   in the restated pins is derived on the running build.

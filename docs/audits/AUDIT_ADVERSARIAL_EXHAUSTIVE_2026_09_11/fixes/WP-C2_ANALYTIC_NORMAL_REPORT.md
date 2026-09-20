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
line.  Nothing here claims cross-build identity: every byte-identity
comparison is archive-to-archive on the SAME build, and every restated bar is
derived on the running build.

---

## 0. Headline

1. **Both B9 reports' explanation of the two knife-edge pins is measurably
   wrong, in both cases.**  Neither pin's quantity is bimodal and neither has
   a saddle-basin flip in it.  The ModalAsymptotic arms sit at the
   *cancellation floor* of a field whose conditioning this build can measure
   (round 2: the TRUE worst case over directions is `kappa = 2.9059e+07`, so
   `eps * kappa = 6.4527e-09`, and the two routines read **1.57x to 1.93x**
   it -- essentially AT the floor.  The 6.24e+06 first published here was
   drawn along one random direction; see D2 in section 1.2).  The `w6_a2` root is *genuinely off centre*: a 60-digit
   `decimal` Newton on the same polynomial system reproduces the library's
   answer to 4.5e-22, and the offset is `H^-1 r(v_c)` where `r(v_c)` is the
   least-squares fit's own asymmetry, resolved eight decades above its
   rounding floor.  `1e-15` was a bound on how asymmetric a fit happened to
   come out.  Both pins are now decisions with bars the running build derives,
   green under all four default combinations on both builds.
2. **`sphere_normal='analytic'` is the better route over the working aperture,
   and is now the default.**  **1.50 ULP** against an 80-digit oracle with
   EXACT input conversion out to `h = 0.95 |R|` (generic: **1.75**, both
   builds), never worse there by more than 1 ULP at any of 672 points, unit to
   1.5 ULP by construction.  Above `0.95 |R|` neither route dominates --
   stated in the code, in the CHANGELOG and in an arm of the test file,
   because it is the honest half of the claim.  (Round 2, defect D1: the
   numbers first published here -- 1.75 against 2.00/2.25 -- came from an
   oracle whose own input conversion was not exact.  Section 2.1 carries the
   corrected sweep; the conclusion is unchanged and the margin widens.)
3. **Timing: 1.08x to 1.44x, and the measurement's own resolution is about
   +-7 %.**  The controls (prescriptions with no pure sphere, where the switch
   cannot change anything) read 0.93x to 1.03x.  The box carried other agents'
   load throughout: 67 % to 100 % CPU, 10 to 21 concurrent python processes.
   The contention-immune number is the profile share of the normal block:
   16.081 % -> 9.844 % (Windows), 19.632 % -> 10.443 % (WSL).
4. **`renormalize='exit'` is NOT measurable as a speed-up on either build.**
   0.95x to 1.13x, medians 1.00x (Windows) and 0.99x (WSL) -- inside the same
   resolution the controls set.  WP-B9's 1.03x-1.10x does not reproduce; it
   is not refuted either, it is simply smaller than this method can see.  The
   default moved on the ledger's structural argument and this report, the
   CHANGELOG and the Migration note all say so.
5. **Vignetting: one rim band moves, nothing else.**  A directed `nextafter`
   walk constructs the straddle point VERIFY-B9 3.3 predicted and it reaches
   `_refract`'s `alive` flag and `trace`'s error code.  360 000 traced rays
   over twelve prescription and field-angle combinations, three of them
   shipped builders, move ZERO alive flags and ZERO error codes.  No shipped
   fixture's vignetting count changes, so there is no moved vignetting fixture
   to re-pin -- and the Migration note names the band anyway.
6. **The JAX tracer has no such switches and never did.**  It has always used
   a closed-form sphere normal and has no per-surface rescale to hoist, so the
   flip moves the CPU tracer TOWARD it.  CPU/JAX parity is 3.469e-18 m in
   position and 3.123e-17 m in OPL under ALL FOUR CPU settings, identical to
   the last digit, with alive masks equal, on both builds.
7. **One documented bound was a reading and is exceeded.**  The pre-5.49.0
   `trace` docstring promised `| |d| - 1 | <= 1e-15` on the intermediate
   history bundles under `'exit'`.  Measured 1.8e-15 at 13 surfaces -- about
   `0.6 * n_surfaces * eps` -- so it is exceeded by the eighth surface.
   (ROUND 2, defect D3: both halves of that sentence are corrected in section
   3.2.  `n_surfaces * eps` is the bound, the coefficient runs 1.000 to 0.615
   and is not constant, and `1e-15` is first exceeded at the SEVENTH
   surface.)  The
   docstring now carries the derived form and the new test asserts it.

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
| FD conditioning `kappa` ALONG ONE RANDOM DIRECTION (4-point ladder, 1e-13..1e-10; superseded -- see round 2) | 6.2423e+06 .. 6.2639e+06 | 6.2423e+06 .. 6.2638e+06 |
| **`kappa`, TRUE worst case over directions (round 2)** | **2.9059e+07** | **2.9059e+07** |
| `eps * kappa` (true) | 6.4527e-09 | 6.4527e-09 |
| reading / floor (true) | **1.62x** | **1.57x** |

The quantity is continuous and dense, not bimodal: the two implementations
differ by a handful of roundings on a field that amplifies its own inputs by
six decades.  `max|new - cold_ref|` and `max|new - warm_ref|` agree to five
figures (the two references are 2.93e-09 apart), so the warm/cold distinction
the pin's history is about does not produce the reading either.

**`w6_a2` (`w6a2_oracle.py`, `w6a2_asymmetry.py`, `w6a2_resolution.py`).**

| measurement | Windows | WSL |
|---|---|---|
| `abs(v2*)` at the fit centre, four default combinations | 4.97e-16 .. 8.64e-16 | 4.09e-16 .. 6.40e-16 |
| 60-digit `decimal` Newton on the same polynomial system | agrees to **3.1e-22** | agrees to **4.5e-22** |
| oracle self-consistency, two independent starts | 1.0e-65 | 4.1e-66 |
| `abs(r(v_c))` in float64 | 4.84e-08 .. 8.42e-08 | 4.40e-08 .. 8.71e-08 |
| the same at 60 digits, relative agreement | 3.3e-07 | 8.8e-07 |
| `sigma_min(H)` | 9.7413e+07 | 9.7413e+07 |
| offset vs the single Newton step `H^-1 r(v_c)` | **9.9e-32** | 0 .. 9.9e-32 |
| scalar solver's spread over 45 independent Newton starts | 2.1e-21 | 1.8e-21 |

The solver finds the same point from every start in the basin to 2e-21, and
that point is ~5e-16 from the origin -- five decades apart, so the offset is
not solver noise.  It is the root of the computed system, and that root is not
at the centre because the FITTED coefficients are not symmetric.  `1e-15`
bounded that asymmetry.

### 1.2 The restatements

`tests/unit/test_audit_propagation.py` -- `_conditioning_bar` measures
`kappa` on the running build, asserts the response is linear (so it is
conditioning and not a threshold), asserts `kappa > 1e3` (the premise that
makes `eps * kappa` the right bar at all), and sets the bar from it.
`_assert_at_conditioning_floor` asserts the reading is under the bar AND that
an injected drift -- sized from the measured `kappa` to land two decades above
it -- is still refused.  That is the fail-before demonstration, on the running
build, in the test.

**ROUND 2 (VERIFY-WP-C2 defect D2) replaced the sampled `kappa` with the true
one.**  The first version drew ONE RANDOM DIRECTION in coefficient space, and
`kappa` is directional: the shipped seed reads 6.2427e+06 and another
1.6406e+06 on the same fixture and build, a 3.8x spread, so the bar's value --
and the arm's strictness -- was a property of a seed.  `kappa` is now the full
finite-difference Jacobian of the field in the fit's 70 phase coefficients,
reduced to the induced `inf <- 2` operator norm (the largest per-pixel response
to a unit-2-norm relative coefficient perturbation, maximised over ALL
directions; for a complex field and a real perturbation that is the largest
singular value of `[Re J_row; Im J_row]`, maximised over rows).  Measured
**2.9059e+07, identical to five digits on both builds and both fixtures**, i.e.
4.65x the shipped seed's draw, with the response along the attaining direction
linear to 1.0001 over three decades.

With the true `kappa` the two implementations sit essentially AT the floor --
1.57 to 1.93 `eps * kappa` over the eight readings -- so the bar moved from
100x the floor to **10x**: at 100x it would have been a 62x margin, a ceiling a
fiftyfold degradation would pass.  And the bar is now bracketed on BOTH sides
on the running build: one injected drift two decades above it must be refused,
and a second a decade BELOW it must be accepted, so the arm is not satisfiable
by a bar of infinity.

| build / defaults | reading | floors | bar | margin |
|---|---|---|---|---|
| Win surface/generic | 1.0331e-08 | 1.60 | 6.4524e-08 | 6.2x |
| Win surface/analytic | 1.2478e-08 | 1.93 | 6.4524e-08 | 5.2x |
| Win exit/generic | 1.1015e-08 | 1.71 | 6.4524e-08 | 5.9x |
| Win exit/analytic (shipped) | 1.0421e-08 | 1.62 | 6.4524e-08 | 6.2x |
| WSL surface/generic | 1.1062e-08 | 1.71 | 6.4524e-08 | 5.8x |
| WSL surface/analytic | 1.2433e-08 | 1.93 | 6.4524e-08 | 5.2x |
| WSL exit/generic | 1.0789e-08 | 1.67 | 6.4524e-08 | 6.0x |
| WSL exit/analytic (shipped) | 1.0146e-08 | 1.57 | 6.4524e-08 | 6.4x |

All eight pass, and the injected drift reads 100.0x-100.1x the bar on every
one of them.

`tests/unit/test_niche_audit_w6_asymptotic.py` -- two decisions: the returned
expansion point equals the model's own one-Newton-step root from the pupil
centre (bar `4 eps x half-range = 4.49e-18`, the float64 resolution of `v2` on
its own scale; reading 0 to 9.9e-32), and that root is the pupil centre at the
fit's own resolution (reading 2.2e-14 to 4.4e-14 of the normalised box against
a 1e-10 decision bar, with the derived `abs(r)/sigma_min(H)` bound asserted
alongside and the premise that `r(v_c)` is resolved asserted separately).

### 1.3 Green under either default -- the gate WP-B9's order required

`pins_restated.py` runs all three arms under all four
`(renormalize, sphere_normal)` combinations, flipping `trace.__defaults__` /
`trace_world.__defaults__` on the function objects exactly as VERIFY-B9 section
4 did (and clearing the `lru_cache`d, TRACED `w6._fit` between combinations --
forgetting that silently reports the first combination's number four times).

Re-run in round 2 with the corrected `kappa` (D2).  The readings are
unchanged -- they are properties of the library, not of the bar -- and the bar
moved from `100 * eps * kappa_sampled = 1.3862e-07` to
`10 * eps * kappa_true = 6.4524e-08`, so the margins are the honest ones:

| build | combination | lg00 | 4-mode | w6_a2 | reading | floors | bar | margin |
|---|---|---|---|---|---|---|---|---|
| Windows | surface/generic | PASS | PASS | PASS | 1.0331e-08 | 1.60 | 6.4524e-08 | 6.2x |
| Windows | surface/analytic | PASS | PASS | PASS | 1.2478e-08 | 1.93 | 6.4524e-08 | 5.2x |
| Windows | exit/generic | PASS | PASS | PASS | 1.1015e-08 | 1.71 | 6.4524e-08 | 5.9x |
| Windows | exit/analytic | PASS | PASS | PASS | 1.0421e-08 | 1.62 | 6.4524e-08 | 6.2x |
| WSL | surface/generic | PASS | PASS | PASS | 1.1062e-08 | 1.71 | 6.4524e-08 | 5.8x |
| WSL | surface/analytic | PASS | PASS | PASS | 1.2433e-08 | 1.93 | 6.4524e-08 | 5.2x |
| WSL | exit/generic | PASS | PASS | PASS | 1.0789e-08 | 1.67 | 6.4524e-08 | 6.0x |
| WSL | exit/analytic | PASS | PASS | PASS | 1.0146e-08 | 1.57 | 6.4524e-08 | 6.4x |

The injected ABOVE-bar drift reads 100.0x-100.1x the bar on all eight, and the
BELOW-bar drift round 2 added is accepted on all eight.

Note the first row: at the SHIPPED defaults this box read 1.0331e-08, i.e. the
pre-WP-B9 `1e-8` bar was already red here, independent of any default.  That is
what "a coin" looks like from the other side.

---

## 2. Item 2 -- the `sphere_normal` flip (commit `4c29ec44`)

### 2.1 The oracle ladder, on WP-C2's own sphere set

`sphere_oracle.py`: eight radii of both signs (2 mm, 34.5 mm, 51.5 mm, 80 mm,
120 mm, 500 mm, 1 m, both signs), eleven heights from the vertex to
`0.99994 |R|`, six azimuths, each set evaluated twice -- once refracting, once
with the surface declared a MIRROR -- 1056 points per build.

**The numbers below are the ROUND 2 re-run** with the exact input conversion
defect D1 asked for (`decimal.Decimal(float(x))` rather than
`ctx.create_decimal(repr(float(x)))`) at 80 digits.  The superseded readings
are kept in the last column so the correction is visible rather than silently
swapped.

| quantity | Windows | WSL | superseded (repr-converted, prec 60) |
|---|---|---|---|
| closed form, worst ULP out to `h = 0.95 R` (672 points) | **1.50** | **1.50** | 1.75 / 1.75 |
| generic route, same | **1.75** | **1.75** | 2.00 / 2.25 |
| points where the closed form is worse by > 1 ULP there | **0 of 672** | **0 of 672** | 0 |
| closed form, worst ULP over the whole set | **45.86** | **45.86** | 57.47 |
| generic route, same | **91.49** | **91.49** | 76.30 |
| unit-vector defect of the closed form, worst | 1.0 ULP | 1.5 ULP | same |
| closer / further / tie | 596 / 122 / 338 | 582 / 132 / 342 | 596 / 112 / 348 |
| points where the closed form is worse by > 1 ULP (whole set) | 14 | 18 | 22 |
| worst deficit | 22.83 ULP | 35.34 ULP | 35 ULP |
| normal depends on `glass_after` (mirror vs refracting) | no | no | no |
| domain-gate disagreements on this grid | 0 | 0 | 0 |
| `prec = 80` vs `prec = 120`, every summary field | identical | identical | -- |

Per height (worst over radii and azimuths; the two builds agree to the last
digit except at `0.999` and `0.9999`, where both are given):

| `h/R` | 0 | 0.05 | 0.5 | 0.95 | 0.99 | 0.999 | 0.9999 | 0.99994 |
|---|---|---|---|---|---|---|---|---|
| closed form, ULP | 0.00 | 0.00 | 0.50 | **1.50** | 2.75 | 7.09 | 33.13 / 36.17 | 45.86 |
| generic, ULP | 0.00 | 0.50 | 1.00 | **1.75** | 4.75 | 17.12 / 12.25 | 45.12 / 33.24 | 91.49 |

At 14 of the 1056 points on Windows and 18 on WSL -- all at `h >= 0.99 R` --
the closed form rounds WORSE by more than one unit, by up to 22.83 / 35.34
ULP.  That is VERIFY-B9 3.2 reproduced: above `0.95 R` both routes are at the
conditioning limit of `sqrt(1 - u)` and neither dominates point by point.

Why the correction matters and why it does not change the decision: `repr` of
a float is the shortest ROUND-TRIPPING decimal, not the float's exact value,
and `nz = sqrt(1 - u)` amplifies a relative input perturbation by
`u / (2 (1 - u))`.  The old conversion therefore contributed about 1.00 of
this probe's own ULP unit at `h = 0.95 |R|` -- exactly the number it then
reported for the closed form -- and about 41.6 at the clamp.  With the exact
conversion BOTH routes read better and the gap between them widens, so the
conclusion strengthens.

### 2.2 Timing

`timing.py`, five prescriptions, `N = 100 000` rays, 20 traces per timing
sample, 9 repeats with the arm order alternating each repeat, minimum over
repeats, wall clock and process CPU time both recorded.  Windows' CPU clock
ticks at 15.6 ms, so an unbatched 0.25 s trace is quantised at 6 % and the
no-sphere CONTROLS read a 20 % "speed-up" off one tick -- the batching is not
optional.

Windows (CPU time; load 67 % -> 100 % CPU, 10 python processes):

| prescription | surfaces | generic | analytic | speed-up |
|---|---|---|---|---|
| spherical, 7 surfaces | 7 | 0.0570 s | 0.0508 s | 1.123x |
| Cooke-like triplet | 6 | 0.0523 s | 0.0477 s | 1.098x |
| Cassegrain, 2 spherical mirrors | 3 | 0.0227 s | 0.0180 s | 1.261x |
| conic, no sphere (CONTROL) | 3 | 0.0336 s | 0.0336 s | 1.000x |
| aspheric, no sphere (CONTROL) | 3 | 0.2102 s | 0.1266 s | 1.660x -- contention spike |

WSL (CPU time; controls 0.975x and 1.014x):

| prescription | generic | analytic | speed-up |
|---|---|---|---|
| spherical, 7 surfaces | 0.1055 s | 0.0945 s | 1.116x |
| Cooke-like triplet | 0.1148 s | 0.0797 s | 1.441x |
| Cassegrain, 2 spherical mirrors | 0.0492 s | 0.0414 s | 1.189x |

**Range: 1.08x to 1.44x; medians 1.12x (Windows) and 1.19x (WSL).**  The
controls set the resolution at about +-7 %, and the one aspheric-control
sample that read 1.66x did so on a run where that arm's own absolute time
quadrupled -- a contention spike, reported rather than averaged away.  The
structural measure, which contention cannot move because both arms are
profiled under it: the normal block's share of `trace`'s own tottime falls
from **16.081 % to 9.844 %** (Windows) and **19.632 % to 10.443 %** (WSL).

### 2.3 Vignetting

`vignetting.py`.

* **The straddle exists.**  A directed `nextafter` walk over eight radii and
  six azimuths, 60 steps each side of `h = R sqrt(0.9999)`, finds it:
  `R = -0.12 m`, `x = 0.0544288129262979`, `y = 0.10693953582952408`,
  `h/R = 0.9999499987499374`.  There the closed form's gate says VALID and the
  generic route's says out of domain.
* **It reaches the answer.**  Through `_refract`: `alive=False, code=4`
  (`RAY_NAN`) generic against `alive=True, code=0, L=0.3381, N=0.6667`
  analytic.  Through the public `trace` on a two-surface stack with the
  downstream aperture opened wide: `code=4` generic against `code=1`
  (`RAY_TIR`) analytic -- dead either way there, but for an honest physical
  reason instead of an arithmetic fault.
* **Nothing that is not aimed at it lands in it.**  360 000 rays over twelve
  combinations: a sphere with its clear aperture opened to `0.99999 R`; the
  seven-surface spherical stack at 0, 3 and 8 degrees; the two-mirror
  Cassegrain at 0 and 2 degrees; and three shipped builders (`make_singlet`
  20/-20, `make_singlet` 51.5/inf, `make_doublet` 51.7/-34.5/-120) at 0 and 5
  degrees.  **Zero** alive flags moved, **zero** error codes moved, on both
  builds.

**So no shipped fixture's vignetting count changes, and there is no moved
vignetting fixture to re-pin.**  The Migration note names the band anyway,
because the change is real even though no current fixture samples it.

### 2.4 The clamp

Unchanged, per the ledger.
`test_c2_the_domain_clamp_stays_where_the_ledger_left_it` LOCATES the threshold
by 80-step bisection on the running build (so a change to the expression, not
only to the literal, is caught) and asserts a ray at `0.99995 R` still dies
`RAY_NAN` through the generic route.

---

## 3. Item 3 -- the `renormalize` flip (commit `44151397`)

### 3.1 The ladder

`renorm_ladder.py`: spherical and conic stacks of 3, 5, 7, 9, 11 and 13
surfaces built from one repeated cemented pair, 4000 rays each, traced both
ways under BOTH normal routes so the two switches' effects are never confused.
Identical to the last digit on Windows and WSL.

| quantity | value |
|---|---|
| `max abs(dx)` between the two modes | 6.592e-17 m |
| `max abs(dopd)` | 1.665e-16 m |
| `max abs(dL)` | 7.216e-16 |
| `alive` masks equal on every rung | yes |
| error codes equal on every rung | yes |
| difference / derived `n_surfaces * eps * abs(t)` envelope | 0.109 .. 0.386 |
| `_normalize_directions` calls, `'surface'` / `'exit'` | 0 / 1 |

The ratio to the envelope is 0.189 at 3 surfaces, peaks at 0.386 at 7, and
FALLS to 0.109 by 13, so the drift does not accumulate with surface count --
the envelope grows faster than the difference does.  That is the claim the
hoist rests on, and it is a ladder rather than a reading.

The call count has a trap in it: `trace` imports `_normalize_directions` BY
NAME, so a probe that patches `intersection._normalize_directions` sees
nothing and reports 0 under both settings -- which looks exactly like a pass.
The probe and the test both patch in `trace`'s own namespace.

### 3.2 One documented bound was a reading, and it is exceeded

The pre-5.49.0 docstring promised `abs(abs(d) - 1) <= 1e-15` on the
INTERMEDIATE history bundles under `'exit'`.  Measured on the same ladder:

| surfaces | 3 | 5 | 7 | 9 | 11 | 13 |
|---|---|---|---|---|---|---|
| history, worst | 6.7e-16 | 8.9e-16 | 1.2e-15 | 1.3e-15 | 1.3e-15 | **1.8e-15** |
| final bundle | 2.2e-16 | 2.2e-16 | 2.2e-16 | 2.2e-16 | 2.2e-16 | 2.2e-16 |

**ROUND 2 (VERIFY-WP-C2 defect D3) corrects both readings above.**  Measured
on the shipped test's own ladder, identical to the last digit on both builds:

| surfaces | 3 | 5 | 7 | 9 | 11 | 13 |
|---|---|---|---|---|---|---|
| history drift, worst | 6.66e-16 | 8.88e-16 | **1.22e-15** | 1.67e-15 | 1.67e-15 | 1.78e-15 |
| as a fraction of `n_surfaces * eps` | **1.000** | 0.800 | 0.786 | 0.833 | 0.682 | **0.615** |

`n_surfaces * eps` is the BOUND and it holds on every rung; the coefficient in
front of it runs from **1.000 at three surfaces to 0.615 at thirteen**, so
"about 0.6" is a long-end reading and a consumer sizing a tolerance from it is
40 % under on a triplet -- the commonest case.  And `1e-15` is first exceeded
at the **SEVENTH** surface (1.22e-15), not the eighth: five surfaces read
8.88e-16.  The docstring and the test now state the bound, the measured range
and the seventh-surface crossing; the test asserts the coefficient FALLS
(1.63x from three surfaces to thirteen, against a 1.25x bar) so it cannot be
restated as a constant.

Making `'exit'` the default makes this contract
load-bearing for every history consumer, so the docstring now carries the
`n_surfaces * eps` form with both measurements, and the test asserts the
envelope, asserts the drift GROWS with surface count (so the envelope is the
right shape and not an accident), and asserts the final bundle is unit to
4 eps regardless.

### 3.3 Timing -- the claimed speed-up does not reproduce

| build | range over five prescriptions (CPU) | median | wall range |
|---|---|---|---|
| Windows | 0.986x .. 1.095x | 1.000x | 0.90x .. 1.15x |
| WSL | 0.951x .. 1.125x | 0.987x | 0.98x .. 1.16x |

WP-B9 reported 1.03x-1.10x.  The method's resolution here is about +-7 %, so
an effect of that size is not resolvable on this box under this load: the
reading neither confirms nor refutes WP-B9's, it is not evidence.  Unlike
`sphere_normal`, the hoist applies to EVERY prescription, so there is no
control arm available for it -- which is itself worth recording.

The default moved anyway, on the ledger's structural argument (one rescale
instead of N, with the per-surface fault diagnosis unmoved).  The CHANGELOG
and the Migration note both say so in those words, so that a maintainer who
weighs the accounting differently can revert one commit.

---

## 4. Item 4 -- the JAX tracer

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

| CPU setting | worst `abs(dx)` | worst `abs(dopd)` | alive masks equal |
|---|---|---|---|
| generic / surface | 3.469e-18 m | 3.123e-17 m | yes |
| analytic / surface | 3.469e-18 m | 3.123e-17 m | yes |
| generic / exit | 3.469e-18 m | 3.123e-17 m | yes |
| analytic / exit | 3.469e-18 m | 3.123e-17 m | yes |
| the library default | 3.469e-18 m | 3.123e-17 m | yes |

Identical to the last digit on both builds: the parity floor is set by the JAX
Newton's own intersection arithmetic, not by the normal route, so the flip
moves no parity pin.  The default arm is byte-equal to the `analytic / exit`
arm and different from `generic / surface`, which is the signature's claim
asked from a call.

Note the two closed forms are not the same arithmetic -- NumPy substitutes the
near-branch sag, JAX uses the intersection's own `z` -- so the agreement above
is a measurement, not an identity.

---

## 5. Item 5 -- blast radius, measured

### 5.1 The run

Selection: a grep over `tests/unit` and `tests/integration` for every file
that reaches the tracer or a consumer of it (`trace(`, `trace_world`,
`trace_prescription`, `raytrace_system`, `ray_fan`, `opd_fan`, `spot_rms`,
`spot_geo`, `seidel`, `ghost_`, `apply_real_lens`, `trace_jax`, `jax_trace`,
`ray_transfer_jacobian`, `refocus`, `through_focus`, `compute_pupils`,
`first_order_data`, `system_abcd`, `exit_vertex`, `surfaces_from_prescription`,
`make_singlet`, `make_doublet`, `find_paraxial_focus`, `differential`) UNION
the lens family's `traced` / FGA / GBD / multibranch files.  **368 files**, the
list committed as
`validation/probe_c2_analytic_normal/trace_touching_files.txt`.

Run sharded (the box carried another agent's work throughout, and a single
process was projecting well past a working day), each shard
`python -m pytest <files> -q -p no:randomly --capture=sys -rf` with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line.  Seven parts, all completed:

| part | result | wall |
|---|---|---|
| 0 | 3 failed, 2388 passed, 5 skipped | 1:33:22 |
| 1 | 10 failed, 3155 passed, 49 skipped | 1:26:24 |
| 2 | 1 failed, 3401 passed, 4 skipped, 2 deselected | 1:36:32 |
| 3a | 639 passed | 0:45:42 |
| 3b | 895 passed, 1 skipped | 0:55:49 |
| 3c | 740 passed | 0:21:36 |
| 3d | 940 passed, 6 skipped | 0:33:29 |
| **total** | **14 failed, 12 158 passed, 65 skipped** | |

(Part 3 was one shard that had to be split four ways mid-run: it was
projecting past twelve hours behind three heavy traced-lens and Maslov
quadrature files.  `py-spy dump` confirmed it was computing, not hung, before
it was split.)

### 5.2 Every test that moved, classified

Nothing was loosened to make a reading pass; where a bar moved, it moved to a
quantity that can carry one.

**A. Genuine contracts that pinned the OLD default -- restated, not relaxed
(5 tests, `tests/unit/test_audit2609_b9_raytrace_perf.py`).**

| test | what it pinned | what it pins now |
|---|---|---|
| `test_b9_i1_default_is_the_per_surface_renormalise` | both public and private defaults in one assertion | the PRIVATE-layer contract only (`_refract` / `_reflect` / `_surface_normal`), which is what makes every direct caller unchanged by construction; the public defaults move to `test_c2_...` |
| `test_b9_i2_the_default_trace_is_bit_identical_and_analytic_is_bounded` | default == `'generic'`, bit for bit | default == `'analytic'`; `'generic'` is the way back, same derived envelope, arms swapped |
| `test_b9_i2_named_bit_equal_consumers_see_the_default` (renamed `test_b9_i2_the_switch_reaches_every_surface_and_only_by_asking`) | that the default keeps handing the generic route to two downstream pins | that the switch is honoured at EVERY surface and is never partial -- its old premise (those pins go red on a 1e-16 move) is the one item 1 measured and found false |
| `test_b9_i1_surface_mode_leaves_every_history_bundle_unit_length` | took its `'surface'` arm from the DEFAULT | names its mode, so the claim stays about the two modes; the `4 n eps` drift bound is unchanged |
| `test_b9_i1_exit_mode_calls_the_single_pass_exactly_once` | same | same |

**B. Repository gates that fired correctly on this change (2 tests).**

| test | why it fired | what was done |
|---|---|---|
| `test_v5_3_2_walker_source_line_citation.py::test_v18_5_the_5_47_0_block_citations_name_the_right_lines` | the docstring expansions moved eight `[5.47.0]` source-line citations, and four of those cited lines did not MOVE -- their CONTENT changed, which the re-anchor tool had no path for and correctly reported as NEEDS A HUMAN | four re-anchored mechanically; the other four answered with an explicit, guarded `EDITED_IN_PLACE` map in `scripts/reanchor_citations.py` naming each base coordinate, its new coordinate and the release that edited it |
| `test_public_api.py::test_no_shipped_source_claims_a_version_the_package_has_not_reached` | 16 docstring lines said "5.49.0" while `__version__` reads 5.48.1 | the docstrings describe the change instead; the CHANGELOG and Migration Guide carry the number, which is where the rule says it belongs |

**C. Pre-existing reds, verified identical at 49ddf4bd (9 tests).**  Each was
re-run against a read-only `git archive 49ddf4bd` and fails there with the
same message.

| test | reason |
|---|---|
| `test_audit2609_a9_ui.py::test_u6i_workers_honour_request_interruption` | Qt thread interruption timing |
| `test_audit2609_a9_ui.py::test_u7_file_new_keeps_display_preferences` | `libshiboken` QObject double-init |
| `test_audit2609_a9_ui.py::test_u7_matplotlib_is_not_imported_by_the_dock_modules` | the harness itself pulls matplotlib on this box |
| `test_audit2609_a9_verify_ui.py::test_u4_the_real_worker_restores_the_globals_on_every_exit_path` | same Qt worker family |
| `test_audit2609_a9_verify_ui.py::test_followup_86_interrupted_optimizer_reports_cancelled` | same |
| `test_niche_audit_w3_infra.py::TestA6EstimateAsmMemory::test_est_bounds_measured_first_call_peak[512-complex128]` | a measured peak-memory bound on a loaded box |
| ... `[1024-complex128]` | same |
| `test_public_api.py::test_installed_metadata_version_matches_source_version` | the editable install reports 5.47.0 against a source `__version__` of 5.48.1 -- a stale `pip install -e .` |
| `test_v4_15_agent_e.py::TestUI6and7PsfMtfDockRayAccumulation::test_no_last_write_wins` | a source-text grep for `np.add.at` in `ui/psf_mtf_dock.py::_load_from_raytrace`, which is not there on either tree |

**E. Genuine movers that are bars on a draw -- restated against the
fixture's own last-bit noise (2 tests, `tests/unit/test_niche_d3_guards.py`,
commit `788edd2c`).**  These two pass at 49ddf4bd and failed here, and they
are the only tests in the sweep of which that is true.

Both are measured with niche C6's stationary-phase launch ENGAGED on a
MULTIPLEXED 2x2 order fan.  That file's own
`test_c13_makes_the_d3_separation_build_independent` already documents the
state: the C6 residual-eikonal fit "explains NONE of its own data at EVERY
degree 1-6", returns `|grad a| = 974` against a physical maximum of 1, and
"perturbing ONLY that fit's coefficients by a relative 1e-12 ... moves
`|mux|` by 163x".  That sibling's CONDITION was moved on 2026-08-08 for
exactly this reason.

Measured (`validation/probe_c2_analytic_normal/d3_guard_draw.py`):

| configuration | `bad6` | `good6` | `bad4` | `bad4/bad6` | `moved` |
|---|---|---|---|---|---|
| 5.49 ray-tracer defaults | 1.0069 | 0.00831 | 1.177 | 1.17x | 0.836 |
| pre-5.49 defaults forced | 1.6462 | 0.00831 | 115.249 | 70.01x | 22.553 |
| 5.49 defaults, input nudged 1 ULP up | 1.0095 | 0.00831 | 1.177 | 1.17x | 0.836 |
| 5.49 defaults, input nudged 1 ULP down | 1.0087 | 0.00831 | 1.177 | 1.17x | 0.836 |
| pre-5.49 defaults, input nudged 1 ULP up | 1.5974 | 0.00831 | 115.248 | 72.15x | 22.553 |
| the file's own recorded numbers, Windows | -- | -- | 19.085 | 15.14x | 39.830 |

Three readings decide it.  Forcing the pre-5.49 keywords back does NOT
restore the recorded numbers -- it gives `bad4 = 115.25` where the docstring
recorded 19.08 and `moved = 22.55` where it recorded 39.83, so the magnitude
is not reproducible at fixed arithmetic between the recording date and today.
A one-ULP nudge of the INPUT envelope moves it by under 0.3 %, so it is not
input noise -- it is specifically the traced landings' last bits, amplified.
And `good6` is identical to three figures in every row, so the
well-conditioned arm is stable and it is the launch-ON multiplexed magnitude
alone that draws.  Claim 1 of the separation test (`bad6 > 5 * good6`) reads
121x at the shipped defaults and 198x at the old ones and keeps its bar
untouched.

Restated, not loosened: the exact half of the attribution (launch OFF -> the
two degrees return byte-identical fields) is unchanged and still passes; the
launch-ON half now compares the degree's effect against the SAME quantity's
one-ULP-input floor, measured in process.  Readings after the restatement:
degree effect 0.1689 of `bad6` against a 0.0026 floor (65x, bar 10x), and
`moved` 0.836 against a 0.1404 field-norm floor (5.95x, bar 3x).  The first
version of this restatement compared the linearity-error effect against the
FIELD-norm floor and correctly refused -- 0.1689 against 0.1404 -- which is
why the comparator is now like for like.

**Summary.**  368 files, 12 158 passed, 14 failed.  The only tests that moved
because of this work package are the **5** in block A, the **2** gates in
block B and the **2** in block E -- nine in total, every one restated as a
decision or a re-anchored fact, none by loosening a bar to fit a reading.
**Nine** of the fourteen reds are pre-existing (block C, each re-run against a
read-only archive of 49ddf4bd and failing there identically) and **four** were
an artefact of a shard reading the source while the second flip was being
committed (block D, green on a clean re-run).

**D. An artefact of this work package's own mid-run edits, not a mover (4
parametrised ids).**  `test_audit2609_a17_history_relocation.py`'s AST and
token-stream arms for `lumenairy.raytrace.trace` and `.world_trace` went red
inside a shard that was reading the source while the second flip and the
version-token correction were being committed and the fingerprints
re-recorded.  Re-run clean on the final tree: **752 passed**.

---

## 6. Item 6 -- byte identity, archive to archive

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
| both old keywords passed explicitly | **934 / 1008 identical** | **935 / 1008** |
| ... of which direct `trace` / `trace_world` arrays | **all identical, 0 moved** | 0 moved |
| ... the 73-74 that moved | `trace_prescription` 27, `refocus` 26, `ray_fan_data` 8, `opd_fan_data` 8, `spot_rms` 4, `through_focus` 1 | same, less `through_focus` |
| non-trace fixtures, no keyword | **7 / 7 identical** | 7 / 7 |
| defaults vs defaults (the full blast radius) | 413 / 1008; 595 moved | 414 / 1008; 594 moved |
| ... worst absolute / relative move | 2.363e-11 / 3.366e-13 | same |

The 73-74 exceptions are **the finding**: `trace_prescription`,
`raytrace_system`, `ray_fan_data`, `opd_fan_data`, `through_focus_rms` and the
result-consuming `spot_rms` / `spot_geo_radius` / `refocus` all trace
internally and expose neither keyword, so there is no one-keyword way back
through them.  Threading two keywords through five public functions is an API
expansion the ledger did not ask for, so it is documented in the Migration note
and raised as a follow-up (section 8) rather than done here.

---

## 7. Item 7 -- the tests

`tests/unit/test_c2_analytic_normal_default.py`, 19 tests, whole file 1.2 s --
every test three orders of magnitude inside the 60 s budget.

| test | what it decides |
|---|---|
| `..._defaults_are_what_the_ledger_decided_from_the_signature` | both defaults, and the private helpers' NON-movement, from `inspect.signature` |
| `..._defaults_are_what_a_call_actually_takes` | the same asked from a call: the `_surface_normal` spy sees `analytic_sphere=True` at every surface, and `_normalize_directions` runs exactly once (a revert that keeps the signature fails here and not above) |
| `..._way_back_is_the_pre_5_49_0_arithmetic_exactly` | `'generic'` / `'surface'` forced against a `_surface_normal` that ignores the keyword: byte-identical on three stacks |
| `..._control_without_a_sphere_is_byte_identical_either_way` | the switch is confined to what the predicate selects |
| `..._closed_form_is_the_better_route_over_the_aperture` (x5 radii) | the 60-digit oracle decision, two-sided (generic measured alongside), bar 4 ULP |
| `..._closed_form_is_a_unit_vector_by_construction` | the identity `abs(n)^2 = 1`, bar 4 ULP |
| `..._above_0p95_R_neither_route_dominates_and_the_test_says_so` | the honest other half: BOTH routes must leave the 4 ULP band before the clamp |
| `..._domain_clamp_stays_where_the_ledger_left_it` | the threshold LOCATED by bisection on the running build, plus the `RAY_NAN` kill |
| `..._rim_band_is_the_one_discontinuous_difference` | the straddle point CONSTRUCTED by a directed `nextafter` walk and shown to reach `alive` |
| `..._no_bundle_that_is_not_aimed_at_the_rim_band_lands_in_it` | and nothing else moves |
| `..._spherical_mirror_reflects_about_the_outward_normal` | the sign, against the oracle's law of reflection |
| `..._predicate_is_what_selects_the_closed_form` | the accepted set, six surface kinds |
| `..._history_bundles_are_not_unit_under_the_new_default` | the `n_surfaces * eps` drift contract, derived and laddered, with the way back restoring the old one |
| `..._exit_hoist_does_not_accumulate_with_surface_count` | the envelope claim, as a ladder |
| `..._mutation_matrix_is_stated_and_each_arm_is_named` | the table below, asserted so the names cannot rot |

**Mutation matrix.**

| mutation | caught by |
|---|---|
| either default silently reverted (signature) | `..._defaults_are_what_the_ledger_decided_from_the_signature` |
| either default reverted inside the loop only | `..._defaults_are_what_a_call_actually_takes` |
| `'generic'` / `'surface'` stop being the old arithmetic | `..._way_back_is_the_pre_5_49_0_arithmetic_exactly` |
| the closed form loses accuracy | `..._closed_form_is_the_better_route_over_the_aperture` |
| it stops being a unit vector | `..._closed_form_is_a_unit_vector_by_construction` |
| the `0.9999` domain clamp moves or goes | `..._domain_clamp_stays_where_the_ledger_left_it` |
| the rim band stops being the only difference | `..._no_bundle_that_is_not_aimed_at_the_rim_band_lands_in_it` |
| the normal's SIGN flips (mirror) | `..._spherical_mirror_reflects_about_the_outward_normal` |
| the selection predicate widens or narrows | `..._predicate_is_what_selects_the_closed_form` |
| the exit rescale runs twice, or not at all | `..._defaults_are_what_a_call_actually_takes` |
| the history drift contract changes shape | `..._history_bundles_are_not_unit_under_the_new_default` |
| the hoist starts accumulating with surfaces | `..._exit_hoist_does_not_accumulate_with_surface_count` |

---

## 8. Requested changes outside this work package's ownership

1. **The six entry points with no way back.**  `trace_prescription`,
   `raytrace_system`, `ray_fan_data`, `opd_fan_data`, `through_focus_rms` and
   the result-consuming `spot_rms` / `spot_geo_radius` / `refocus` trace
   internally and expose neither `sphere_normal` nor `renormalize`, so a caller
   who needs the pre-5.49.0 arithmetic through them has to drop to `trace`.
   Threading two keywords through five public signatures is an API expansion
   the ledger did not ask for, so it is documented rather than done.  Measured:
   exactly those 73-74 of 1008 recorded arrays move with the old keywords
   passed.  Owner: whoever owns `raytrace/trace.py`'s public consumers.
2. **`analysis.ghost` now uses a different normal from `trace`.**  It calls
   `_refract` / `_reflect` directly, and those keep
   `sphere_normal='generic'` -- the right default for a caller that owns its
   own loop -- so the ghost path and the main trace no longer agree in the
   last bit on a spherical prescription.  Nothing measured moves (ghost's
   recorded arrays are byte-identical either way).  The fix, if it is wanted,
   is one keyword at `ghost.py:934,955`.  Owner: `analysis/ghost.py`.
3. **WP-B9's and VERIFY-B9's account of the two knife-edge pins should be
   corrected in place.**  Both documents state a mechanism -- a saddle-basin
   flip on a bimodal quantity -- that is measurably not present in either pin
   (section 1).  The numbers in both reports reproduce; the explanation does
   not, and anyone reading them for a future default flip will reach for the
   wrong tool.
4. **`scripts/reanchor_citations.py` had no path for a cited line whose
   CONTENT changed** (as opposed to one that moved).  A default flip is
   exactly that case, and the tool correctly said "NEEDS A HUMAN".  The
   human's answer is now written down as an explicit `EDITED_IN_PLACE` map
   naming each base coordinate, its new coordinate and the release that edited
   it, guarded so the override refuses unless the current line still begins
   with the same leading token.  Whoever owns the citation gate should decide
   whether that shape is what they want long-term.

---

## 9. What could not be measured

1. **A clean absolute timing number.**  The box carried another agent's work
   for the whole work package (67 % to 100 % CPU, 10 to 21 concurrent python
   processes, recorded in every `timing_*.json`).  The controls bound the
   method at about +-7 % and one control sample spiked to 1.66x.  The speed-up
   RANGE and the profile SHARE are sound; a single number is not available
   from this box today.
2. **Whether `renormalize='exit'` is faster at all.**  Its claimed effect
   (1.03x-1.10x) is smaller than this method's resolution, and unlike
   `sphere_normal` it has no control arm, so nothing here settles it either
   way.
3. **Whether the rim band matters to any real design.**  It is about 1 ULP of
   `h` wide, so sampling cannot find it -- 360 000 rays found none -- and the
   only evidence that it is reachable at all is a directed `nextafter` walk.
   Whether a user's prescription works rays there is not something this work
   package can answer; the Migration note tells them how to find out.
4. **Cross-build identity.**  Deliberately not claimed anywhere: every byte
   identity comparison is archive-to-archive on the SAME build, and every bar
   in the restated pins is derived on the running build.


---

# Round 2 (VERIFY-WP-C2) -- 2026-09-20

The twelve defects, the tautology and the release-text follow-ups from
[`VERIFY_WP-C2.md`](VERIFY_WP-C2.md) (verdict SHIP after D11, D1, D3, D4, D7
and D12 are actioned and D2, D5, D6, D8, D9, D10 are filed), closed on
`feat/c2-analytic-normal-round2` off `verify/c2-analytic-normal` (`61ffe596`).

Everything below was **re-measured in this round**, never read off the
verification.  Both builds every time, with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line and `lumenairy.__file__` printed by every probe: **Windows py3.14.6 /
numpy 2.4.4 / jax 0.11.0** and **WSL py3.12.3 / numpy 2.4.6 / jax 0.10.2**.
The PRE tree is this round's own `git archive 49ddf4bd` extracted to
`C:/tmp/lum_c2b_pre`, and every archive comparison runs each side in its own
process with its own `sys.path`.  New evidence:
`validation/probe_c2_round2/` (six probes, JSON per build).

## Closure table

| defect | verdict | this round's numbers (Windows / WSL) |
|---|---|---|
| **D4** (P2) sixteen entry points with no way back | **CLOSED** | all sixteen take `sphere_normal=` / `renormalize=`, default `None`; **742 / 742 arrays byte-identical** archive to archive on BOTH builds, with **16 of 16** entry points shown to move at the default and `None` byte-identical to omitted on all 742 |
| **D11** (P1) four private docstrings say the generic route ships | **CLOSED** | all four rewritten; the new census test FAILS on a `git archive eadc67ba` tree naming **4 of 4** stale sentences, on both builds, and passes here |
| **D1** (P2) the oracle's input conversion is not exact | **CLOSED** | `Decimal(float(x))`, prec 60 -> 80; closed form **1.50 ULP** to `0.95 |R|` against the generic route's **1.75** (was 1.75 / 2.00-2.25), whole set **45.86 vs 91.49** (was 57.47 / 76.30), **0 of 672** worse by > 1 there; prec 80 == prec 120 on every summary field |
| **D12** (P2) the release text's byte-identity counts | **CLOSED** | 934 of 1008 (Win, 74 move) / 935 (WSL, 73 move), derived from the committed JSON by a test; the stale "938" / "the 70" are refused by name |
| **D7** (P2) `EDITED_IN_PLACE` accepts a reverted default | **CLOSED** | each entry pins a SHA-256 of the expected content and the release it was recorded for; the three abuses are now REFUSED with both lines printed, the two it already refused still are, and the map refuses one patch past its release |
| **D2** (P2) `kappa` is drawn from a random direction | **CLOSED** | `kappa` is the true induced `inf <- 2` norm of the field's Jacobian: **2.9059e+07**, identical to five digits on both builds and both fixtures, **4.65x** the shipped seed's draw; bar 100x -> **10x** the floor; eight readings 1.57-1.93 floors, margins **5.2x-6.4x**; the bar is now bracketed above AND below |
| **D3** (P3) "about 0.6 n eps" | **CLOSED** | the ratio runs **1.000 at 3 surfaces to 0.615 at 13**; `1e-15` first exceeded at the **SEVENTH** (1.22e-15 against 8.88e-16 at five), not the eighth |
| **D5** (P3) `analysis.ghost` refracts off a different normal | **CLOSED** | the ghost leg asks `_library_trace_default('sphere_normal')`; RMS spot radius moves **9.663e-13 mm / 5.400e-13 mm**, transmittance and ray counts unmoved; a spy sees `analytic_sphere=True` from ghost AND from `trace` |
| **D8** (P3) the d3 arm-2 floor's spread | **CLOSED** | floors over four one-ULP directions spread **3.22x / 4.79x** (arm 2) and **2.07x / 26.99x** (arm 1); both arms now bar against the MAX, margins 4.69x / 8.59x and 65.3x / 74.4x; arm 2 gains its lower half (the same degree twice reads **exactly 0.0**) |
| **D9** (P3) the Migration note's rim paragraph | **CLOSED** | the two gates bisect to the SAME float `0.9999499987499374` at all eight radii on the meridian, both builds; a ball lens or hemisphere loses **3024 of 60 000** rim-packed rays (5.04 %) to `RAY_NAN` on BOTH routes |
| **D6** (P3) JAX has no domain clamp | **CLOSED as a DECISION** | ledger section 1.10; 1962 of 40 000 rays past the clamp, CPU keeps **0** under all four settings, JAX keeps **1962**, identical on both builds; no clamp is added to JAX |
| **D10** (P3) the new ids are not in `.test_durations` | **CLOSED** | see "The recorded items" |
| the `w6_a2` tautology | **CLOSED** | `norm_offset <= 2 * bound` reads **0.99991 / 0.7159** of its own theorem and is retired; the replacement is that a SECOND Newton step is **1.94e-07 / 8.77e-08** of the first, against a 1e-4 bar |

## D4 -- the way back through all sixteen, archive to archive

The design is the one WP-C1 used for `evaluate(aperture_edge=)`: each entry
point takes the tracer's OWN two keywords, same names and same accepted
values, defaulting to `None`, and `trace._way_back_kwargs` turns `None` into
"do not name it at all".  An entry point that defaulted to today's `'exit'` /
`'analytic'` would freeze this release's default into every call site the day
the library's default moves again, which is the failure the campaign rule
exists to prevent.

`validation/probe_c2_round2/r2_wayback_entrypoints.py` runs each tree in its
own process with `lumenairy.__file__` asserted inside it, digests every
returned array with SHA-256 over dtype + shape + raw bytes, and compares four
arms: the PRE tree with no keyword, this tree with
`renormalize='surface', sphere_normal='generic'`, this tree at the defaults,
and this tree with both keywords `None`.

| entry point | module | way back (identical / total) | arrays that move at the default | `None` == omitted |
|---|---|---|---|---|
| `trace_prescription` | `raytrace.trace` | 36 / 36 | 24 | 36 / 36 |
| `raytrace_system` | `raytrace.trace` | 45 / 45 | 29 | 45 / 45 |
| `ray_fan_data` | `raytrace.ray_fan` | 4 / 4 | 2 | 4 / 4 |
| `ray_fan_data_world` | `raytrace.ray_fan` | 4 / 4 | 2 | 4 / 4 |
| `opd_fan_data` | `raytrace.ray_fan` | 4 / 4 | 2 | 4 / 4 |
| `opd_fan_data_world` | `raytrace.ray_fan` | 4 / 4 | 2 | 4 / 4 |
| `through_focus_rms` | `raytrace.ray_fan` | 3 / 3 | 1 | 3 / 3 |
| `paraxial_focus_world` | `raytrace.world` | 2 / 2 | 1 | 2 / 2 |
| `ray_transfer_jacobian` | `raytrace.differential` | 7 / 7 | 6 | 7 / 7 |
| `caustic_diagnostic` | `analysis.aberration` | 8 / 8 | 2 | 8 / 8 |
| `eval_image_plane_wfe` | `analysis.image_plane_wfe` | 12 / 12 | 1 | 12 / 12 |
| `plot_lens_layout` | `analysis.plotting` | 18 / 18 | 9 (7 on WSL) | 18 / 18 |
| `fit_canonical_polynomials` | `propagators.asymptotic_canonical_fit` | 298 / 298 | 8 (6 on WSL) | 298 / 298 |
| `fit_hf_polynomials` | `propagators.asymptotic_canonical_fit` | 295 / 295 | 5 (3 on WSL) | 295 / 295 |
| `apply_real_lens_traced` | `elements` | 1 / 1 | 1 | 1 / 1 |
| `apply_real_lens_maslov` | `elements` | 1 / 1 | 1 | 1 / 1 |
| **total** | | **742 / 742** | **16 of 16 entry points move** | **742 / 742** |

Identical on Windows and WSL.  The middle column is what makes the first one
mean something: every one of the sixteen really does produce different bytes
at the shipped defaults, so a 742/742 way back cannot be a keyword that
reaches nothing.

`spot_rms`, `spot_geo_radius` and `refocus` take no keyword because they do
not trace -- they consume a `TraceResult`, and their answers move only because
their input does.  The five JAX entry points take none because `trace_jax` has
neither switch by design.

**The census that keeps it true.**
`test_c2_every_entry_point_that_traces_carries_both_keywords` walks the
package's AST (a bare NAME counts as well as a call -- `ray_fan_data` PASSES
`trace` to `_trace_fan_set` rather than calling it -- and an ATTRIBUTE call
counts too), finds 20 exported directly-tracing functions, and requires the
four that lack a keyword to be EXACTLY the jax twins.  Its fail-before arm
reads a mutant whose `ray_fan_data` signature has lost `sphere_normal` while
its body still mentions it -- the shape a grep census misses -- and a real
mutant tree confirms it: on a fresh `git archive` of the round-2 tree with
that one edit, the two census arms and VERIFY-WP-C2's own are **3 failed** on
BOTH builds.

## D2 -- the conditioning bar, and why the margin got smaller

`kappa` was measured along one random direction in coefficient space, and
`kappa` is directional: `default_rng(20260920)` reads 6.2427e+06 and
`default_rng(7770001)` 1.6406e+06 on the same fixture and build.  It is now
the full finite-difference Jacobian of the field in the fit's 70 phase
coefficients, reduced to the induced `inf <- 2` operator norm -- for a complex
field and a real perturbation, the largest singular value of
`[Re J_row; Im J_row]` maximised over rows, computed in closed form for all
1024 pixels and confirmed by an SVD of the winning row.

| quantity | Windows | WSL |
|---|---|---|
| `kappa`, true worst case over directions | **2.9059e+07** | **2.9059e+07** |
| `kappa` along the shipped random direction | 6.2427e+06 | 6.2427e+06 |
| response along the attaining direction, deltas 1e-12 / 1e-11 / 1e-10 | 2.9062e+07 / 2.9059e+07 / 2.9059e+07 | same |
| floor `eps * kappa` | 6.4527e-09 | 6.4527e-09 |
| reading, shipped defaults | 1.042e-08 (**1.62 floors**) | 1.015e-08 (**1.57 floors**) |

With the correct `kappa` the two implementations sit essentially AT the floor,
so the 100x bar would have been a 62x margin -- a ceiling a fiftyfold
degradation would pass.  The bar is now **10x** the floor, and it is
BRACKETED: one injected drift two decades above it must be refused (reads
100.0x-100.1x the bar on all eight combinations) and a second a decade below
it must be accepted.  All eight combinations pass with margins 5.2x-6.4x
(section 1.3's table, re-run).

## D5 -- the ghost leg, and why it asks rather than names

`retrace_ghost_path` resolves the route once per call through the new
`raytrace.trace._library_trace_default('sphere_normal')` and passes it to both
`_reflect` and `_refract`.  It asks the LIBRARY rather than writing
`'analytic'` down, because a literal there would pin this release's default
into the ghost path for every release after it.  `renormalize` stays at the
private `True`: the ghost loop has no exit pass to hoist a single rescale to.

| quantity, three 2-bounce paths of a spherical doublet, 256 rays | Windows | WSL |
|---|---|---|
| RMS spot radius, worst change | 9.663e-13 mm | 5.400e-13 mm |
| FWHM, worst change | 0.0 | 2.842e-14 mm |
| total transmittance / energy fraction | unmoved | unmoved |
| `analytic_sphere` seen by a spy, ghost vs `trace` | `{True}` vs `{True}` | same |

At the refraction step itself the route the ghost leg asks for and `trace`'s
default are byte-identical, and the generic route on the same bundle is not --
so the identity is not vacuous.

## What could not be measured

1. **The 368-file blast radius was not re-run.**  It cost the WP-C2 agent
   about seven hours across seven shards.  What was re-run is in "The runs"
   below.
2. **A clean absolute timing number**, unchanged from the work package's own
   answer.  The element-op count replaces it for both switches, but it is a
   COUNT with a stated first-order cost model, not a time.
3. **`mpmath` on WSL**, unchanged: the 60-digit cross-check of the `w6_a2`
   Newton step is a Windows-only reading.
4. **The 1008-array byte-identity census was not re-run.**  D12 makes the
   release text quote the committed JSON rather than a retyped number, and the
   742-array sixteen-entry-point census supersedes its conclusion, but the
   1008 counts themselves are still the WP-C2 measurement.

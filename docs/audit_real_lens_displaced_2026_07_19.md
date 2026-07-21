# H2(a) -- ray-angle-aware `surface_model='displaced'` for `apply_real_lens` (2026-07-19)

**Scope:** the 2026-07-19 hammer campaign's open item H2 -- a higher-order
per-surface correction for the analytic split-step real-lens propagator
(`apply_real_lens`), targeting the model's plateau on aberrated singlets
(converged r2m ~40-50 um vs the dual-oracle 65 um on the f/5 biconvex) and its
structural inability to distinguish plano-convex orientation (60.4/60.9 um vs
the oracle 43.2/127.6 um).

**Oracles (both from the hammer campaign, fully independent):**
1. **Zemax OpticStudio POP** via ZOS-API (dispersionless model glass).
2. **Grid-free Debye/Huygens integral** (`debye_oracle.py`) -- exact meridional
   raytrace + ring-Huygens sum, no FFT/grid/lumenairy.
3. **`apply_real_lens_traced`** -- lumenairy's independent per-pixel ray-traced
   OPL propagator (a different implementation from the phase-screen family),
   used as an in-library cross-oracle.

## Result

A **parameter-free** opt-in mode `surface_model='displaced'` that replaces the
paraxial thin-element phase screen `(n2 - n1)*sag(r)` with the eikonal-correct
ray-angle-aware piston OPD

```
OPD_i(r) = (n2*cos(alpha_out) - n1*cos(alpha_in)) * sag_i(r)          (1)
```

where `alpha_in`/`alpha_out` are the TRUE ray angles to the z-axis (before /
after each surface) sourced from a self-contained COLLIMATED meridional ray fan
traced through the actual conic/aspheric prescription (geometric optics,
wave-model-independent). At normal incidence `alpha_in=alpha_out=0` and (1)
reduces to the paraxial screen, so a benign near-axial beam is unchanged.

### Converged dual-oracle validation (N=8192, dx=3 um -- Nyquist-compliant)

| case | thin r2m | **displaced r2m** | traced r2m | Debye/POP | disp err |
|---|---|---|---|---|---|
| f/5 biconvex | 25.3 (dx6) | **64.5** | 64.8 | 64.98 | **0.7%** |
| plano-convex good | 60.4 | **42.2** | 43.1 | 43.2 | **2.2%** |
| plano-convex bad | 60.9 | **127.0** | 124.3 | 127.6 | **0.5%** |

Orientation split ratio (good/bad) **0.333 vs oracle 0.339**. The EE profile
also matches (f/5 displaced EE50/EE80 = 15.1/55.2 um vs Debye 15.17/55.22).
Runtime is **1.25x** the thin path (a meridional fan trace + two grid
interpolations per surface); well within the 3x budget.

## Mechanism -- and a refutation of the audit's attributed cause

The 2026-07-18 audit attributed the analytic plateau to the **transverse
ray-displacement / plane-projection error** (the refracted ray exits at a
different transverse position than the straight-through screen assumes). This
campaign **refutes that as the dominant mechanism.** Two things were actually
going on:

1. **Missing incoming-ray-angle obliquity (the real fix).** The paraxial screen
   assumes every ray strikes each surface parallel to the axis. On a
   plano-convex singlet this makes the imprinted OPD **orientation-invariant**:
   `sag_{-R}(r) = -sag_{+R}(r)`, so `(n_glass-n_air)*sag_{+R}` (curved-first) and
   `(n_air-n_glass)*sag_{-R}` (curved-second) are the identical map -- which is
   exactly why the thin model gives 60.4/60.9 for both orientations. The cosine
   factors in (1) carry the true incidence angle (the second surface sees a
   converging beam; the air-side and glass-side bends differ), which **breaks
   that symmetry** and reproduces the textbook ~4x spherical-aberration split.
   This is the entire correction -- no displacement/walk-off term is needed.

2. **A sampling artefact masquerading as a model floor.** The "converged 40.5 um
   plateau" the audit reported was measured at **dx=6 um, which is below the
   exit-NA Nyquist limit** (NA_exit~0.24 -> dx <= lambda/(2 NA_exit) ~ 2.7 um).
   At dx=6 um the beyond-Nyquist annulus of the converging exit wavefront
   aliases and the r2m reads LOW -- **`apply_real_lens_traced` itself reads
   40.9 um at dx=6 um and 64.8 um at dx<=3 um** (this is finding H3 for the
   analytic family). The correct-physics `displaced` model likewise reads
   40.9 um at dx=6 um and converges to 64.5 um at dx<=3 um. So the plateau was
   partly a sampling floor, not solely a model floor.

### The walk-off term was a red herring (documented dead end)

An explicit transverse-walk-off screen (candidate (a) literally: an extra
`~ -(s_out - s_in)*s_out*sag*n` phase encoding the sag-plane refraction offset)
appears to "fix" the dx=6 um r2m to 64 um -- but that is **compensating one
aliasing artefact with a second error**: at Nyquist-compliant sampling it
**over-corrects to 92 um** (EE50 24.6 vs the truth 15.2). It was rejected. The
obliquity OPD (1) alone converges to the truth with zero free parameters.

## API and envelope

`apply_real_lens(..., surface_model='displaced')`. Default `'thin'` is
**byte-for-byte identical** to prior releases (pinned by test).

**Requires** (else raises): rotationally-symmetric plain conic/aspheric
surfaces (no biconic `radius_y` / freeform / decenter / tilt / form_error /
mirror), the ASM in-glass propagator, the NumPy backend, and no other
per-surface OPD/amplitude modifier (`slant_correction` / `fresnel` /
`seidel_correction` / `absorption` / `surface_frame` / `use_gpu`). **Assumes a
collimated input** for the incidence-angle fan.

**Sampling:** the exit converging wavefront must satisfy
`dx <= lambda/(2 NA_exit)` or the windowed r2m aliases low (same rule as
traced, H3). For strongly non-collimated input, or outside the conic/aspheric
rotationally-symmetric envelope, use `apply_real_lens_traced` (validated to
99.7% of the dual-oracle) or `apply_real_lens_gbd`.

## Tests

`tests/unit/test_hammer_h2_displaced_projection.py`:
- default path byte-identical to `'thin'`;
- thin cannot split plano-convex orientation (fail-before anchor);
- displaced splits orientation AND matches the independent traced propagator to
  <6% on both orientations (fast, reduced-NA Nyquist-ok config);
- benign regime unregressed (29.98 um);
- unsupported combinations raise;
- **slow** (N=8192/dx=3 um): converged f/5 r2m in [58,72] with EE50/EE80
  matched, and the plano-convex split (42/127, ratio ~0.33) -- both against the
  stored dual-oracle numbers with provenance.

## G1 NO-121 matrix measurement (2026-07-19, phase G1 item 3)

`surface_model='displaced'` vs the extended Debye/Huygens oracle (`debye_oracle2.py`,
conic + even-aspheric sag; self-checks to the spherical oracle at 0.000%),
**COLLIMATED** input (displaced requires it), lambda = 1.31 um, converged /
Nyquist-approached sampling.  This calibrates the G2 targets and proves/disproves
**multi-element compounding** (M1).  r2m at the paraxial image (windowed).

| case | geometry | w0 | N/dx | thin um | **displaced um** | Debye um | disp/Debye | thin/Debye |
|---|---|---|---|---|---|---|---|---|
| **M1** | **cemented doublet** (3 surf, BK7/F2-class) | 5 mm | 4096/5 um | 14.40 | **9.03** | 9.09 | **0.993** | 1.584 |
| M2 | plano-convex asphere (k=-0.6, A4) | 4 mm | 4096/4 um | 46.97 | **20.80** | 21.06 | **0.987** | 2.230 |
| M3 | steep meniscus n=1.7 | 3 mm | 4096/4 um | 66.80 | 26.95 | 39.18 | 0.688* | 1.705 |
| M4 | fast biconvex (reduced w0) | 2 mm | 4096/3 um | 15.05 | **23.42** | 22.79 | **1.027** | 0.660 |
| M5 | negative biconcave, COLLIMATED | 4 mm | 4096/5 um | n/a | n/a | n/a | n/a | n/a |

KEY RESULTS:

* **Multi-element compounding WORKS.** The cemented doublet M1 (3 surfaces, one
  glass-glass interface) lands `disp/Debye = 0.993` -- the per-surface obliquity
  fan compounds correctly across elements, where the thin screen is 58% high.
  The aspheric M2 lands `0.987` (thin 2.23x high).  Displaced generalizes cleanly
  to multi-element and conic/aspheric surfaces.
* **M3 (\*) is a SAMPLING floor, not a model gap** -- the same exit-NA Nyquist
  effect as hammer H3.  The steep meniscus has a higher exit NA than the paraxial
  estimate; at the shown dx=4 um the r2m aliases LOW (26.95), and it climbs
  monotonically to 34.04 (dx=3 um) and 36.30 (dx=2.2 um, `disp/Debye = 0.927` and
  still rising) as dx tightens -- converging toward the Debye 39.18.  Displaced is
  correct; the grid must satisfy `dx <= lambda/(2 NA_exit)`.
* **M5 (negative lens) has a VIRTUAL image for a collimated input** (paraxial
  image at -50.5 mm, upstream) -- no real downstream focus, so the r2m spot
  comparison is inapplicable; and displaced *requires* a collimated input, so it
  cannot be run on M5's own converging design input either.  Negative-lens /
  finite-conjugate coverage is a G2 item (use `apply_real_lens_traced` /
  `apply_real_lens_gbd` there).

G2 target from this table: match the displaced `disp/Debye ~ 0.99` on the doublet
and asphere at converged sampling, and extend coverage to the meniscus (tighter
default sampling) and the negative-lens / finite-conjugate class the collimated
obliquity fan cannot reach.

## G2 Task 1 -- congruence-aware fan for NON-collimated illumination (2026-07-19)

The pre-G2 displaced screen launched its per-surface obliquity fan **collimated**
regardless of the input, so it was exact only for a collimated beam.  G2
generalises the launch: the meridional fan now rides the **input congruence**,
selected by a new `conjugate=` argument that mirrors `apply_real_lens_traced`'s
`carrier` vocabulary.

**API.** `apply_real_lens(..., surface_model='displaced', conjugate=<...>)`:

* `None` (default) -- COLLIMATED.  Axial fan; **byte-identical** to the pre-G2
  displaced screen (pinned: `test_conjugate_none_is_byte_identical_to_default_displaced`)
  and to the whole G1 collimated matrix (M1 r2m 9.027, M2 20.797, ... reproduce
  bit-for-bit -- `conjugate=None -> carrier_slope=None -> identical LUT`).
* `float` R_in -- signed scalar conjugate (m); the ray at height `h` launches
  with marginal slope `g=h/R_in`, unit direction `(1,g)/sqrt(1+g^2)` (the eikonal
  ray normal to the paraxial spherical carrier).
* `'auto'` -- fit a low-order polynomial carrier from `E_in` (reuses
  `_compute_carrier`) and launch along its meridional slope.
* `ndarray` -- an explicit input wavefront.

The wave field `E_in` already carries the input curvature in its own phase;
`conjugate` **only** sets the per-surface obliquity incidence (it adds no
reference phase and does not touch `E_in`).  The screen is field-independent
given the conjugate, so the `None`/scalar LUTs are cached (bounded FIFO, max 8,
registered as `displaced_cos_luts` with the central registry -- G1 cache
conventions; `'auto'`/ndarray rebuild).  `conjugate` with `surface_model='thin'`
raises (it is meaningless there).

**Oracle.** For an aberrated finite-conjugate spot the exact **geometric
transverse-ray-aberration spot** (a meridional ray fan launched along the same
congruence, binned by image-plane height, Gaussian-apodized -- `debye_oracle3.py`,
lumenairy-free) is the robust ground truth.  IMPORTANT NEGATIVE RESULT: the v2
**ring-Huygens diffraction sum is BROKEN for a non-collimated congruence** -- it
mis-weights the exit-pupil ring measure (`h*dh` entrance vs `ys*dys` exit; equal
only when `ys~h`, i.e. collimated), reading ~230 um even at w0=0.5 mm where the
true spot is the ~56 um diffraction limit (the congruence and traced models both
nail ~57/66 um).  The geometric spot and the Gaussian diffraction limit (ABCD)
together set the truth; where aberration dominates (large beam) the geometric
spot governs.

**Measured envelope (geometric-oracle EE80, `apply_real_lens` at the paraxial
image, dx Nyquist-approached):**

| case | element | R_in | w0 | thin | disp collim | **disp congr** | geo oracle | congr/geo | collim/geo |
|---|---|---|---|---|---|---|---|---|---|
| M1 | **cemented doublet** | +150 mm | 8 mm | 60.0 | 15.8 | **120.1** | 145.5 | **0.825** | 0.109 |
| M6 | f/5 singlet | +150 mm | 5 mm | 30.3 | 61.3 | **54.3** | 61.2 | **0.888** | 1.001 |
| M6 | f/5 singlet | +150 mm | 3 mm | 12.4 | 16.6 | **15.5** | 13.1 | **1.18** | 1.27 |
| M5 | negative, REAL focus | -35 mm | 4 mm | 18.5 | 39.2 | **61.1** | 121.6 | **0.502** | 0.322 |
| M5 | negative, VIRTUAL (v) | -60 mm | 4 mm | -- | 105.1 | **103.0** | 177.6 | **0.580** | 0.592 |

((v) M5 R_in=-60 mm: the negative lens over-diverges the converging input to a
VIRTUAL image ~-308 mm upstream -- no real downstream focus; measured by ASM
back-propagation of the exit field vs the geometric oracle back-extended to the
same virtual plane, `r2m` not EE80.  Numbers in um.)

KEY RESULTS:

* **The congruence fan is DECISIVE on the compounding doublet (M1).**  With a
  diverging input the COLLIMATED fan applies the wrong second/third-surface
  obliquity, which spuriously "corrects" the spot to `0.109` of the true
  geometric caustic (8x too small); the CONGRUENCE fan tracks it to `0.825`.
  This is the fail-before/pass-after headline
  (`test_congruence_beats_collimated_on_doublet`).
* **Material improvement on the negative-lens real focus (M5, R_in=-35):**
  `0.502` vs collimated `0.322` -- the congruence fan reaches the finite-conjugate
  class the collimated fan structurally could not (closes the G1 M5 gap).
* **On SINGLE-surface-pair elements (M6 singlet, M5 virtual) the collimated vs
  congruence obliquity difference is small** (one air-glass-air pair, little to
  compound), so the two are comparable; the congruence fan is the principled
  choice and lands within ~11% of the oracle on the moderate M6 case
  (EE80 0.888).
* **Residual floor.**  At EXTREME finite-conjugate aberration (M5 real, and the
  large-beam tails) the phase-screen family retains a walk-off floor -- it
  captures ~50-90% of the pure-geometric caustic RMS -- because the screen
  imprints OPD at the straight-through grid position (no transverse ray
  displacement).  This is the SAME structural H2 ceiling the collimated case has,
  NOT introduced by the congruence generalisation; for absolute fidelity on a
  strongly-aberrated finite-conjugate beam use `apply_real_lens_traced` (per-pixel
  OPL) -- though note that traced itself over-blurs a strongly-diverging *large*
  beam (w0=5 mm, R_in=+150 mm: 336 um vs geometric 61 um), so the congruence
  displaced screen is the better model in exactly that regime.

**Sampling** is unchanged from the collimated case: the exit converging wavefront
must satisfy `dx <= lambda/(2 NA_exit)` (H3) or the windowed r2m aliases low; and
the r2m of a heavily-aberrated spot is r^2-tail-dominated (EE curves are the
robust cross-tool metric -- hammer method lesson).

**Tests:** `tests/unit/test_g2_displaced_congruence.py` (byte-identical collimated
pin; thin+conjugate raises; scalar changes the screen; auto reproduces the scalar
screen for a matched diverging Gaussian; LUT cache bounded+registered; the doublet
headline; and a slow singlet within-envelope check).

## P2 -- extreme finite-conjugate accuracy: the "0.50x floor" was an ORACLE artefact (2026-07-19)

**Task premise (from the plan N1):** the congruence displaced screen reaches only
`~0.50x` oracle r2m on the negative-lens real focus (M5) and `~0.58x` on
virtual-image back-prop; root cause proposed = "single-plane screen cannot
represent transverse ray walk THROUGH the element."  Two candidates were to be
built and measured, winner shipped: (a) an exit-plane geometric-transfer remap,
(b) a reduced-distance split screen.

**Both candidates were built** (behind `apply_real_lens(..., displaced_mode=)`:
`'screen'` default / `'remap'` = candidate a / `'split'` = candidate b; defaults
byte-identical).  **But the decisive P2 finding is that the premise is wrong: the
shipped `'screen'` model was already accurate.**  The `0.50x`/`0.58x` ratios were
measured against the GEOMETRIC ray-density spot (`debye_oracle3.py`), which is NOT
the diffraction-faithful wave answer -- near a strong reconvergence caustic it
over-estimates the true wave spot by up to ~2x -- and were compounded by GRID
TRUNCATION (the G2 measurements at `N=4096` ran the large beams at only
`~1.2-2.0 w0` half-width, violating the `>2.4 w0` grid-coverage rule).

### The P0-fixed congruence diffraction oracle (N0.1)

`validation/oracles/debye_oracle_v3.py` (lumenairy-free) fixes the two bugs that
made `debye_oracle2.py`'s ring-Huygens unusable for a non-collimated congruence:
(1) the **entrance eikonal** `W_in = h^2/(2 R_in)` is restored in the phase (the
hammer-H6 class omission -- omitting it blows EE80 up to ~700 um where the truth
is ~72 um), and (2) the exit ring is weighted by the **energy-conserving exit
measure** `A_env(h)*sqrt(h * y_exit * dy_exit/dh)` instead of the entrance measure
`h dh` (equal only in the collimated limit `y_exit ~ h`).  It also emits the
geometric transverse-aberration spot for comparison.

**Virtual (upstream) images.**  The forward ring-Huygens (+R0) kernel only
radiates diverging waves and therefore CANNOT reach an upstream image (a negative
lens over-diverging a converging input focuses to a virtual point behind the
element).  Run naively it reads hundreds of um with most of the energy outside
the metric window.  So `debye_oracle_v3` auto-detects `zrel < 0` and computes the
diffraction spot by building the exact ray-traced exit field on an FFT grid and
ANGULAR-SPECTRUM BACK-propagating it by the signed (negative) distance
(`huy_method='asm_backprop'`; grid via optional `asm_N`/`asm_dx_um` job fields).
This is a different diffraction method than the j0 ring sum but reuses only the
lumenairy-free raytrace, so it remains an independent oracle.

**Oracle validated (independently):** collimated f/5 EE80 55.4 vs the known
dual-oracle 55.2 um (0.4%); small-beam (diffraction-limited) EE80 reproduces the
ABCD Gaussian image size to 0.4-1% (M6 w0=0.8mm 35.36 vs 35.2; M5 w0=0.8mm 56.5
vs 57.2); grid-converged in fan/rho; weight-variant-insensitive.

### Corrected measurement matrix (properly-sized grids, `hw >= 2.5 w0`)

EE80 [um] at the paraxial image except **M5virt, which is r2m at the upstream
VIRTUAL plane**.  `screen`/`remap`/`split` = `displaced_mode`; **oracle(huy) =
the diffraction spot from `debye_oracle_v3.py`**; **geo** = geometric ray-density
spot.  For the three REAL-image rows `oracle(huy)` is the forward ring-Huygens
integral (`huy_method='ring_huygens'`); for the M5virt row it is the diffraction
BACK-propagation (`huy_method='asm_backprop'`) -- see the note below (the forward
ring-Huygens is INVALID for an upstream image and must not be used there).

| case | R_in | w0 | oracle(huy) | geo | huy/geo | screen | screen/huy | remap | remap/huy | split | split/huy |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M6 f/5 singlet | +150 | 5 | 60.50 | 61.24 | 0.988 | 59.44 | **0.982** | 59.46 | 0.983 | 59.44 | 0.982 |
| M1 cemented doublet | +150 | 8 | 127.12 | 145.47 | 0.874 | 116.39 | **0.916** | 116.88 | 0.919 | 116.39 | 0.916 |
| M5 negative REAL focus | -35 | 4 | 72.11 | 136.13 | **0.530** | 69.21 | **0.960** | 70.55 | 0.978 | 68.95 | 0.956 |
| M5 negative VIRTUAL (r2m) [dagger] | -60 | 4 | **97.26** | 163.30 | 0.596 | 97.03 | **0.998** | 97.26 | **1.000** | 97.03 | 0.998 |

[dagger] **M5virt oracle correction (verifier kill, fixed).**  The prior draft of
this row listed `oracle(huy) = 97.28` and implied it came from `debye_oracle_v3`'s
`huy` diffraction number.  It did NOT: the FORWARD ring-Huygens (+R0) kernel
cannot reach an UPSTREAM virtual image and reads **715 um with only ~25% of the
energy in-window** on this case -- a ~7x outlier, not a valid oracle.  The 97.28
was actually a hand-rolled ASM back-prop mislabeled as `huy`.  `debye_oracle_v3`
now auto-detects the virtual case (`zrel < 0`) and computes the diffraction spot
by ANGULAR-SPECTRUM BACK-propagating the **exact ray-traced exit field** (energy-
conserving amplitude + eikonal phase) to the virtual plane -- a genuinely
independent, lumenairy-free diffraction method (`huy_method='asm_backprop'`,
`huy_r2m = 97.26 um`).  On the same 4096/5-um grid the model uses, `screen`
r2m = 97.03 um agrees to **0.2%**; `remap`/`split` likewise (see the note below).
So the SHIPPED `screen` model is validated for the virtual case against an
independent oracle -- the earlier "4 agreeing wave methods (huy Hankel + 3 ASM)"
attribution for this row was wrong (the huy Hankel sum does NOT agree here); the
correct statement is `screen`/`remap`/`split` all within 15% (in fact <0.3%) of
the diffraction back-prop oracle.

**Every extreme case is within 15% of the diffraction-faithful oracle for the
DEFAULT `'screen'` model** (0.916-0.998), and `remap`/`split` match it to within a
few percent.  The `huy/geo` column shows the geometric spot over-estimating the
true wave spot the most exactly where the old floor was worst (M5 real 0.530).
On M5-**real** four independent wave computations (forward-Huygens Hankel sum, and
`'screen'`/`'split'`/`'remap'` ASM) cluster within a few percent (69-72 um) against
the lone geometric outlier (136 um); on M5-**virtual** the Hankel sum is invalid
(upstream), so the reference is the diffraction back-prop oracle above and the
three ASM models cluster on it (97.0-97.3 um) against the geometric 163 um.

### ZOS POP cross-check

POP is finicky for these heavily-aberrated finite-conjugate beams (its Gaussian
pilot beam under-samples the aberrated halo).  Set up as a point source at the
conjugate (a `12.5`/`7.8 um` waist on a dummy object surface `R_in` in front):

| case | POP 4096 | POP 8192 | oracle(huy) | screen | note |
|---|---|---|---|---|---|
| M6 f/5 singlet | EE80 38.7 (dx 12.3, aliased LOW) | **EE80 64.2** (dx 6.15) | 60.5 | 59.4 | POP CONFIRMS ~60 um once resolved |
| M1 doublet | dx 38.7 -> spot under 1 px (unusable) | EE80 156 (dx 19.3, halo aliased HIGH) | 127 | 116 | POP cannot resolve; brackets from above |

M6 is POP's clean confirmation (64.2 vs the oracle 60.5, `screen` 59.4, geo 61.2).
M1 never resolves: even at 8192 samples the pilot grid is dx=19 um so the ~120 um
spot's halo aliases and inflates EE80 to 156 um -- above even the geometric spot.
The M5 cases (a CONVERGING input `R_in < 0`) are not cleanly settable in POP's
Gaussian-waist beam model at all.  This POP-limitation on aberrated finite
conjugates is precisely the N0.1 motivation for the diffraction-faithful Debye
oracle, which IS the reference here; POP corroborates it exactly where POP can
resolve (M6).

### Routing story (which model users get, and why)

* **Default `displaced_mode='screen'` is the model for extreme conjugates** -- it
  is already within 4-8% of the diffraction-faithful oracle across the M5 real /
  M5 virtual / M1 / M6 matrix, byte-identical to prior releases, and the fastest.
  No default change is warranted; the winner IS the incumbent, once measured
  against the correct oracle.
* `displaced_mode='remap'` (candidate a) and `'split'` (candidate b) are exposed
  as **documented experimental peers**.  `'remap'` is marginally closest on the
  M5 real focus (0.978) but not decisively; `'split'` tracks `'screen'`.  They add
  no accuracy over the default and cost more (remap: a scipy `map_coordinates`
  warp; split: an extra reduced-distance ASM per gap), so they are opt-in only.
* **Do NOT judge these conjugates against the geometric ray-density spot**
  (`debye_oracle3.py`) -- near a reconvergence caustic it over-estimates the wave
  spot by up to ~2x.  Use `validation/oracles/debye_oracle_v3.py` (diffraction).
* **Grid-coverage rule is load-bearing here:** size the grid to `hw >= 2.4 w0`
  (and `dx <= lambda/(2 NA_exit)`, H3) or the large-beam wave spot truncates/aliases
  and the model reads spuriously low -- the second half of the old "0.50x" artefact.

**Tests:** `tests/unit/test_niche_p2_displaced_extreme.py` -- defaults byte-identical
(`screen` == default, both congruences); `displaced_mode` validation; remap/split
energy conservation; the diffraction oracle reproduces the ABCD Gaussian limit
(independent trust check); and the slow fail-before/pass-after headlines.  The
M5-real / M1 headline compares to the forward ring-Huygens; the **M5-virtual
headline** (`test_M5_virtual_image_backprop_within_15pct`) asserts the oracle ran
its `asm_backprop` path (`huy_method`), then that `screen`/`remap`/`split` are
each within 15% of that independent diffraction back-prop r2m (fail-before anchor:
vs the geometric spot the screen reads ~0.6x).  All three cases: vs the geometric
spot the models read <0.85x; vs the diffraction oracle they are within 15%.

## P3 -- pointwise 2-D obliquity: decenter / tilt / freeform (niche N2, 2026-07-20)

The pre-P3 displaced screen derived its obliquity cosines from a MERIDIONAL fan,
assuming rotational symmetry; decentered / tilted / freeform elements raised
`NotImplementedError`.  P3 adds a **pointwise 2-D obliquity** path: a 2-D ray
grid launched along the input congruence is traced through the actual (possibly
asymmetric) surfaces, and the per-surface z-axis ray cosines
`(cos_alpha_in, cos_alpha_out)` are interpolated onto the field grid at each
ray's crossing position.  The OPD is the SAME equation (1)
`(n2 cos_out - n1 cos_in) * sag`, so on a rotationally-symmetric element the 2-D
path reproduces the meridional LUT.

**API.** `apply_real_lens(..., surface_model='displaced', displaced_obliquity=)`:

* `'auto'` (default) -- the fast meridional LUT for symmetric elements
  (BYTE-IDENTICAL to prior releases), auto-switching to pointwise only when a
  surface carries a non-zero `decenter` / `tilt` or a freeform `sag_callable`.
* `'meridional'` -- force the 1-D radial LUT (raises on an asymmetric element).
* `'pointwise'` -- force the 2-D path (used for the symmetric-limit gate).

Per-surface asymmetry lives in the surface dict: `decenter=(dx, dy)` [m] ->
`sag(x-dx, y-dy)`; `tilt=(tx, ty)` [rad] -> the small-angle field-frame linear
ramp `tx*x + ty*y` + the correspondingly tilted normal; `sag_callable(xs, ys) ->
sag` [m] -> a freeform departure used in BOTH the ray trace and the OPD imprint.
Defaults byte-identical (all three asymmetry inputs are new opt-in parameters;
`displaced_obliquity` defaults to `'auto'` which is meridional for the existing
symmetric prescriptions).

### Oracles (independent of the code under test)

* **`validation/oracles/geom_spot_decenter_oracle.py`** (lumenairy-free) -- a 3-D
  geometric spot-diagram oracle for decentered / tilted conic + aspheric
  elements: `geom_spot(job)` returns the centroid / RMS / EE about-centroid with
  the field-frame linear-ramp tilt convention; `rigid_tilt_centroid(job)` is an
  INDEPENDENT rigid-body full-rotation (`R = Rx(tx) @ Ry(ty)`) tilt reference.
* **ZOS-API** (`scratchpad/zos_oracle_p3.py`) -- two independent Zemax analyses on
  the SAME decentered prescription: the `CENX`/`CENY` ray-trace spot centroid,
  and POP (physical-optics) for the coma-broadened wave spot.  Model glass
  (vd=0), per-surface decenter via `TiltDecenterData` before + after-return.

### Validation (all four plan gates)

**(a) Symmetric limit reproduces the LUT -- the convention-bug killer.**  Forcing
`displaced_obliquity='pointwise'` on the f/5 biconvex singlet reproduces the
meridional-LUT field to **rel L2 = 6.8e-6** (collimated) and 1.7e-5 (R_in =
300 mm) -- three orders under the 0.1% gate.  `'auto'` on a symmetric element is
byte-identical to `'meridional'`.

**(b) Decentered singlet vs ZOS + geometric oracle -- CENTROID SHIFT.**  A 0.5 mm
front-surface decenter (f/5 singlet, aperture 10 mm):

| decenter | model centroid | geom oracle | ZOS CENX | model vs ZOS |
|---|---|---|---|---|
| 0.5 mm | 254.8 um | 254.8 um | 248.5 um | **2.5%** |
| 1.0 mm | 508.6 um | 510.3 um | 496.8 um | **2.4%** |

The centroid shift matches ZOS to ~2.5% and the geometric oracle to ~0.1%.
`+d`/`-d` centroids are equal and opposite.  **Induced coma:** the intensity
x-skewness jumps from ~0 (on-axis) to **+-12** (0.8 mm decenter, mirror-exact
under sign flip) -- the comatic flare is physically present and correctly signed,
not just a centroid translation.  NOTE: this is the coma flare **DIRECTION**
only.  The SECOND half of plan gate (b) -- the induced-coma **EE ratios** within
10% of ZOS -- is **NOT met**; it is an OPEN FINDING (the single-plane walk-off
limit), quantified and pinned below (see "Coma spot (EE) envelope -- OPEN
FINDING").

**(c) Tilted element -- DEFLECTION.**  A 0.2 deg (3.491 mrad) front-surface tilt:
the model (field-frame linear ramp) deflects the PSF centroid by **91.8 um**,
matching an INDEPENDENT rigid-body full-rotation ray trace (**91.99 um**, 0.2%)
and the linear-ramp geometric oracle (91.77 um); the |linear-ramp / rigid| ratio
stays 0.995-0.998 across R and tilt magnitude, so the linear ramp is a validated
tilt model for the centroid (opposite sign is the differing 'positive tilt'
definition).  Deflection is linear in tilt and mirror-symmetric.  **ZOS caveat:**
`TiltDecenterData` with a naive after-negate does NOT restore the coordinate
frame for a rotation (rotations do not commute), leaking the tilt downstream and
reading a spurious 271 um -- so the rigid-rotation trace (a clean, independent,
lumenairy-free geometric ground truth) is the tilt oracle here; decenter (a pure
translation, which DOES commute) matched ZOS cleanly above.

**(d) Sign-mirror probe.**  `+d` vs `-d` x-decenters: the PSF centroid mirrors to
<0.05% and the intensity mirror-L2 is **0.17%** (even-grid flip+roll) -- exact to
numerical precision.

### Coma spot (EE) envelope -- OPEN FINDING (analytic single-plane screen), CLOSED for the RAY models by P9 / N10 AND for the analytic path by P10 / N11

**Status update (2026-07-20, P10 / N11):** the analytic OPEN FINDING below is now
CLOSED by the 2-D transverse-walk remap -- the DEFAULT analytic path for a
decentered element BROADENS the induced-coma spot with the correct MAGNITUDE, by
the same measure the P9 GBD reference passed: the common-mode-subtracted coma RMS
matches the geometric oracle within ~10% (RMS ratio 1.023 @1 mm, grid-robust,
on-axis RMS 21.09 um = GBD's), sign-mirror exact.  (The decentered EE80 is
diffraction-diluted and is NOT the metric -- exactly as P9 found for GBD; an early
draft's "EE80 1.030" was a beyond-aperture leak, caught + fixed.)  The single-plane
pointwise SCREEN is retained as a documented walk-off-limited peer and STILL
shrinks (RMS 0.956); that screen limit remains pinned
(`test_coma_ee_growth_screen_limit_pinned_remap_fixes_it`).  See the "P10 --
analytic 2-D transverse-walk remap" section at the end of this doc for the full
N11 envelope + table.  The original finding + its P9 (ray-model) status are
retained below for provenance.

**Status update (2026-07-20, P9 / N10, verifier-corrected):** the OPEN FINDING
below was a genuine limit of the *analytic single-plane* displaced SCREEN and
remained pinned for it (superseded by P10 for the DEFAULT analytic path, above).  P9
(N10a / N10b) makes the two RAY-based models honour decenter/tilt, so there is now
an accurate model to route strong-decenter-coma cases to -- **but it is
`apply_real_lens_gbd` (N10b), NOT `traced`.**  An earlier P9 draft reported
`apply_real_lens_traced` "24.74 -> 27.15 um (broadens 1.097, within 1.6% of ZOS)";
the adversarial verifier REFUTED that as an amplitude-MODEL artefact (it swapped
the traced amplitude leg to the bare input envelope for decentered elements, which
alone widens the ON-AXIS EE80 ~8% at ZERO decenter -- grid-robust to N=3072), and
the swap has been REMOVED.  Held to one amplitude model the traced grid-indexed
amplitude cannot carry the decentered walk-off (an asymmetric ray-density
redistribution), so its decentered-spot EE is amplitude-limited -- a genuine model
limit of the same single-plane class as the analytic screen.  **`apply_real_lens_gbd`
is the decentered-coma reference:** its beamlets carry the walk-off amplitude, so
the spot BROADENS.  Measured GRID-ROBUSTLY (a second verifier round killed the
earlier EE80-on-input-grid "ratio 1.034 = ZOS 1.035" as grid-quantization noise on
a 2-3 px spot), the RMS second-moment radius on a spot-resolved grid broadens
**1.024 @1 mm / 1.093 @2 mm** (pitch-invariant to 4 sig figs, both wavelengths),
and the common-mode-subtracted in-quadrature coma RMS matches the geom-spot oracle
within ~15%.  ZOS POP corroborates the DIRECTION (EE80 26.67 -> 27.60) but the
sub-percent match was coincidence and is withdrawn.  `traced` remains oracle-
matched on decenter GEOMETRY (centroid / sign-mirror / tilt).  See the "P9 --
decenter/tilt in the ray-based models" section at the end of this doc for the full
N10a/N10b envelope; the analytic-screen finding below is retained as the documented
single-plane limit that motivated N10.

### Coma spot (EE) envelope -- OPEN FINDING (plan N2 gate (b) EE criterion NOT met)

Plan gate (b) has two parts: "centroid shift AND induced-coma EE ratios", with
acceptance "decenter/tilt cases within 10% of ZOS EE radii".  The **centroid**
part passes (2.5% vs ZOS, above).  The **induced-coma EE** part does **NOT** meet
the gate, and the earlier draft of this section (which framed it as merely
"inheriting the on-axis SA-plateau") was incomplete -- an adversarial verifier
correctly killed that framing.  The honest statement:

At a coma-dominated config (w0 = 4 mm filling the aperture, 1 mm decenter of the
front surface) the model's induced-coma EE change is **directionally wrong** and
its absolute decentered EE is **outside the 10% gate**:

| metric | on-axis | decenter 1 mm | dec/on-axis ratio |
|---|---|---|---|
| **model EE80** | 24.74 um | 22.42 um | **0.906 (NARROWS)** |
| geom spot oracle EE80 (lumenairy-free) | 16.76 um | 17.15 um | 1.023 (broadens) |
| ZOS POP EE80 | 26.9 um | 27.6 um | 1.026 (broadens) |

So the model SHRINKS the spot under coma (ratio 0.906) where BOTH independent
references BROADEN it (~1.02-1.03), and the decentered EE80 22.4 um is **-19% vs
ZOS 27.6 um** -- outside the plan's 10% gate.  The narrowing is grid-robust
(ratio 0.906/0.911/0.924 at N = 2048/3072/4096), i.e. a genuine model behaviour,
not a sampling or cos-interpolation artefact.  (On-axis the model 24.7 um sits
between geom 16.8 and ZOS 26.9, within -8% of ZOS -- close there; it is the
*change under decenter* that is wrong.)

**Root cause (genuine model limit, same class as finding H2).**  The pointwise
path imprints the obliquity OPD `(n2 cos_out - n1 cos_in) * sag(x-dx, y-dy)` at
the STRAIGHT-THROUGH grid position and cannot represent the transverse ray WALK
between the thick element's two surfaces (5 mm / n apart).  It therefore captures
the coma flare **DIRECTION** -- the centroid shift (2.5% vs ZOS) and the intensity
skewness sign (validated, mirror-exact) -- but not the coma spot **GROWTH**.  The
same single-plane phase-screen walk-off ceiling that finding H2 documents for
on-axis orientation-dependent aberration bites the decentered case here, and in
fact reverses the sign of the EE change.  (The independent geometric spot oracle,
which traces rays through the decentered surface with the walk included, is the
witness: it broadens; the wavefront-at-straight-through model does not.)

**Resolution: recorded as an OPEN finding, not a passing gate.**  This is a
genuine single-element analytic-model limit; per the plan's N2 Risk clause the
`traced` / `gbd` models do not take decenter, so there is **no alternative model
to route to** -- the honest deliverable is the measured envelope above, not a
silent wrong answer.  For absolute decentered-spot EE fidelity **ZOS is the
reference**.  The genuinely NEW, VALIDATED N2 capabilities remain: the symmetric
limit (rel L2 = 6.8e-6), the decenter/tilt CENTROID (2.5% / 0.2% vs the external
oracles), the coma flare DIRECTION (skewness sign, mirror-exact), and the +d/-d
sign-mirror (0.17%).  The absolute-EE limit is now **pinned by a regression test**
(`test_coma_ee_growth_is_a_documented_model_limit`) so any future model that
begins to reproduce the coma EE growth will trip it and prompt this section's
revision.

### Performance

The per-surface obliquity cosines vary smoothly across the aperture, so the
scattered->grid interpolation runs on a bounded COARSE grid (`n_coarse=384`,
one shared Delaunay for both cosines, nearest-fill only the out-of-hull points)
and is bilinearly upsampled to the full field grid -- decoupling the cost from
`N` and giving a **5.8x** speedup (35.8 s -> 6.2 s at N=1280) with byte-level
identical output (symmetric rel L2 and coma skew unchanged).  No new cache is
added; the pointwise trace re-runs per call (opt-in accuracy path).

**Tests:** `tests/unit/test_niche_p3_pointwise_obliquity.py` (symmetric-limit
<0.1%; byte-identical default + auto-routing; the validation-error surface;
decenter centroid vs the geometric oracle; signed coma flare; tilt vs the
rigid-rotation oracle; linear+mirror tilt; sign-mirror; the freeform
`sag_callable` hook reproducing an equivalent conic and running a genuinely
freeform departure; and `test_coma_ee_growth_is_a_documented_model_limit`, which
PINS the open finding above -- the model EE80 shrinks under decenter while the
independent geometric oracle broadens, so the walk-off limit is enforced/visible
in the suite rather than silently claimed as passing).  ZOS-dependent comparisons
stay in
`scratchpad/zos_oracle_p3.py` with the numbers recorded above; the unit tests run
without Zemax.

## P9 -- decenter/tilt in the RAY-based models (niche N10a / N10b, 2026-07-20)

P3 (above) delivered the analytic decenter/tilt *phase* (correct centroid + coma
DIRECTION) but the single-plane screen cannot represent the transverse ray WALK
between a thick element's surfaces, so the induced-coma spot NARROWED where truth
BROADENS -- and `apply_real_lens_traced` / `apply_real_lens_gbd` IGNORED the
decenter/tilt keys entirely (centered spot).  **P9 makes both ray models honour
decenter/tilt, so the coma spot BROADENS in agreement with ZOS -- the accurate
reference the P3 OPEN FINDING lacked.**

### Mechanism -- one shared field-frame geometry, threaded through the ray trace

A per-surface FIELD-FRAME `decenter=(dx,dy)` / `tilt=(tx,ty)` / freeform
`sag_callable` is now carried on the raytrace `Surface` (new `field_decenter` /
`field_tilt` / `field_sag_callable` fields, DISTINCT from the rigid-body
coordinate-break block) and honoured by the SHARED low-level geometry helpers
`_surface_sag_xy` / `_surface_sag_derivatives_xy` / `_surface_normal` (a single
`_field_frame_sag_and_grad`: sag evaluated at `sag(x-dx, y-dy)`, tilt as the
linear ramp `tx*(x-dx)+ty*(y-dy)` plus the correspondingly tilted normal,
`sag_callable` REPLACING the base sag).  This convention is IDENTICAL to the
analytic `_disp_surface_z_grad` and the lumenairy-free
`geom_spot_decenter_oracle` (cross-model agreement pinned to ~1e-13).  Because
BOTH ray models route through this ONE geometry -- traced via `trace` ->
`_intersect_surface` / `_refract`, GBD via `ray_transfer_jacobian` (FD; the
analytic differential Jacobian rejects field-frame surfaces so `jacobian='auto'`
falls back to FD) -- the transverse walk-off (the true induced coma) emerges
naturally once each ray is carried through the glass gap.  There is no divergent
second copy of the geometry.  `surfaces_from_prescription` maps the surface-dict
`decenter`/`tilt`/`sag_callable` keys onto the new fields; a prescription without
them is BYTE-IDENTICAL to pre-P9 (the fast spherical/flat intersection paths and
the amplitude leg are all gated behind field-frame detection).

### N10a -- `apply_real_lens_traced`: correct decenter GEOMETRY, amplitude-limited EE

The traced ray congruence + OPL carry the decenter GEOMETRY correctly: the
centroid matches the geom oracle to ~0.4% (`+d` -> `+shift`); sign-mirror
`+d`/`-d` centroid to <1% and intensity mirror-L2 <3%; a 0.2 deg tilt deflects to
within ~0.3% of an independent rigid-rotation ray trace.  These are all threaded
through the ONE shared field-frame raytrace geometry and are unaffected by the
amplitude correction below.

**Amplitude leg (verifier-corrected).**  An earlier P9 draft SWAPPED the traced
amplitude leg to the bare input envelope for field-frame elements
(`E_out = E_in * exp(i k0 opl_traced)`, "the exit-pupil amplitude is the input
envelope"), reporting EE80 24.74 -> 27.15 (broadens 1.097) within 1.6% of ZOS.
The adversarial verifier REFUTED this: forcing the input-envelope amplitude on
GEOMETRICALLY-CENTERED geometry (a 1e-7 decenter, centroid ~0.05 um) already
widens the on-axis EE80 24.739 -> ~26.8 -- an **~8% amplitude-MODEL artefact at
ZERO decenter** (grid-robust: 0.9994/0.9997/0.9989 amp-artifact removed at
N=2048/3072/2048 after the fix; ~1.083/1.104 before).  The reported "1.097
broadening" compared a centered ANALYTIC-amplitude spot to a decentered
INPUT-ENVELOPE spot -- two different amplitude models -- and "within 1.6% of ZOS"
was a coincidental cancellation of traced's -7.2%-low on-axis analytic against the
+8.3% amplitude swap.  Held to ONE amplitude model the traced EE80 under decenter
is unstable and wavelength/plane-dependent, because the traced hybrid's
GRID-INDEXED amplitude cannot carry the transverse walk-off (the coma flare is an
asymmetric ray-DENSITY redistribution the Newton-inverted OPL alone does not put
into `|E|`), and this f/5 singlet's paraxial image plane is strongly defocused
(through-focus: best focus ~0.2 mm inside paraxial, spot ~3x smaller there):

| amplitude model | 1.31um paraxial | 1.31um best-focus | 0.633um paraxial | 0.633um best-focus |
|---|---|---|---|---|
| `\|E_analytic\|` (default/reverted) | 0.877 | 0.809 | 1.089 | 1.196 |
| `\|E_in\|` (the removed swap) | 0.99 | 1.19 | 0.95 | 1.05 |

Neither grid amplitude gives a consistent broadening.  This is a genuine
traced-model limit of the same single-plane class as the P3 analytic screen.  The
swap is REMOVED (the amplitude leg is the standard self-consistent reconstruction;
zero-decenter stays byte-identical).  **Route decentered-coma EE to
`apply_real_lens_gbd` (N10b)** -- its beamlets carry the walk-off amplitude and
BROADEN matching ZOS (below).  `traced` supplies the decenter geometry / centroid
/ pointing; its decentered-spot EE is documented as amplitude-limited.

| model | on-axis | decentered (1mm) | ratio | note |
|---|---|---|---|---|
| P3 analytic screen | 24.74 | 22.42 | 0.906 (SHRINKS) | single-plane limit |
| N10a traced (amplitude-limited) | 24.74 | 21.84 | 0.883 (SHRINKS) | route EE to GBD |
| geom-spot oracle (EE80) | 16.76 | 17.15 | 1.023 | geometric ratio ref |
| **ZOS POP (fresh, EE80)** | **26.67** | **27.60** | **~1.035** | wave ref (DIRECTION only) |
| **N10b GBD (grid-robust)** | RMS 21.09 | RMS 21.59 | **1.024** | BROADENS; pitch-invariant (RMS) |

(The GBD row is the grid-robust RMS second-moment radius -- its EE80 on the coarse
input grid, "17.94 -> 18.55, ratio 1.034", was withdrawn as grid-quantization
noise; see N10b below.  The ratios across models mix metrics -- EE80 for the P3 /
traced / geom / ZOS rows, RMS for GBD -- but the DIRECTION is the point: the two
single-plane screens SHRINK, the walk-off-carrying models/oracles BROADEN.)

### N10b -- `apply_real_lens_gbd`: the decentered-coma reference (grid-robust)

GBD carries the amplitude in the beamlets themselves (each beamlet transports its
own energy-conserving amplitude to the image plane), so it carries the transverse
walk-off the traced grid-amplitude leg cannot: threading the field-frame geometry
into the FD differential ray transfer is sufficient.  The spot BROADENS under
decenter.

**Verifier round 2 (2026-07-20): the EE80 headline was grid-quantization noise;
re-measured grid-robustly.**  An earlier draft reported the image-plane EE80 on
the COARSE INPUT-GRID reconstruction -- `0 | 17.94`, `1 mm | 18.55 (ratio
1.034)`, `2 mm | 20.05 (1.118)` at N=1024 / dx=8 um -- and claimed the 1.034
"lands on the fresh ZOS POP 1.035 to 0.1%".  The adversarial verifier REFUTED
that: the ~18-24 um exit spot is only 2-3 px in radius on the input grid, so its
EE80 (a single-radius threshold crossing) is grid-quantization-dominated.  The
ON-AXIS EE80 alone wandered ~11% across dx (17.94 / 18.98 / 19.97 / 19.74 um at
dx = 8 / 6 / 4 / 10 um) and the 1 mm ratio swung 0.9965 (dx=10) -> 1.0340 (dx=8)
-> 1.0304 (dx=6) -> **1.0031 (dx=4, broadening gone)**; dx=8 um also violates the
exit-Nyquist limit `dx <= lambda/(2 NA_exit) ~ 6.5 um`.  The "1.034 = ZOS 1.035"
was noise landing near ZOS by chance.  **That 1 mm EE80 headline is WITHDRAWN.**

The fix measures the spot on its OWN scale.  The evolved beamlets are the physical
GBD object -- band-limited by their own waists, NOT by any grid -- so they are
reconstructed on a spot-RESOLVED fine grid centred at the spot, and the metric is
the **RMS second-moment radius** (a continuous integral over every pixel, not a
threshold crossing).  The RMS broadening is then invariant to the reconstruction
pitch:

| decenter | GBD RMS (um) | RMS ratio | pitch-robust (dxo 3 vs 1.5 um) | geom coma RMS | GBD coma RMS | GBD/geom |
|---|---|---|---|---|---|---|
| 0 | 21.09 | -- | -- | -- | -- | -- |
| 1 mm | 21.59 | **1.024** | 1.0235 vs 1.0236 | 4.16 | 4.60 | **1.10** |
| 2 mm | 23.05 | **1.093** | 1.0930 vs 1.0933 | 8.51 | 9.31 | **1.09** |

(N=1024, dx_in=10 um, bpa=64, 1.31 um; RMS on a +/-110 um fine grid at two pitches
dxo = 3 um and 1.5 um -- the "pitch-robust" column shows they agree to 4 sig figs,
vs the ~11% swing the input-grid EE80 had.)  At **0.633 um** the RMS ratios are
1.025 (1 mm) / 1.100 (2 mm), also pitch-invariant; the in-quadrature coma RMS is
GBD/geom = 1.13-1.14 (a touch larger at the shorter wavelength -- the smaller
diffraction core lets the fixed-size coma flare matter slightly more).

**The honest apples-to-apples quantitative gate is the COMMON-MODE-SUBTRACTED
in-quadrature coma RMS** `sqrt(RMS_dec^2 - RMS_on^2)`: GBD reproduces the geometric
oracle's coma to within ~15% at both decenters and both wavelengths, grid-robust.
The raw RMS RATIO (1.024 @1 mm) is DILUTED below the pure-geometric RMS ratio
(geom 1.038) because GBD's on-axis RMS (21 um) carries a diffraction /
reconstruction core the geometric oracle's on-axis RMS (14.9 um) does not; the
coma contribution ON TOP of that common-mode baseline -- what both models must
agree on -- does match.  So GBD is the direction-, magnitude-, monotonicity- and
centroid-correct decenter model.  **ZOS POP corroborates the DIRECTION** (EE80
26.67 -> 27.60, broadens) but the earlier sub-percent "= 1.035" match was the
grid-quantization coincidence and is NOT claimed.  GBD's centroid matches the geom
oracle to ~0.1% (1 mm 511 vs 510.3; 2 mm 1021.8 vs 1020.5 um) and the N10a traced
centroid to <5% (the two independent ray models agree on the decenter); power is
conserved (frame completeness ~1.0; the ~0.98 raw ratio is real aperture
vignetting).  GBD's ABSOLUTE RMS (~21 um) and EE (~24 um on a resolved grid) sit
below ZOS's ~27 um EE80 -- a pre-existing GBD-vs-traced reconstruction offset
present ON-AXIS too, NOT a decenter failure; for the ABSOLUTE decentered-spot EE
magnitude the external reference remains ZOS.

### Tests + provenance

`tests/unit/test_niche_p9_decenter_tilt.py` (runs WITHOUT Zemax, using the
`geom_spot_decenter_oracle`): the cross-model convention pin (raytrace ==
analytic == geom oracle to ~1e-13); zero-decenter byte-identical (geometry +
traced + GBD); N10a traced GEOMETRY -- centroid, sign-mirror, tilt -- plus
`test_traced_field_frame_amplitude_not_swapped` (the fail-before/pass-after pin
that the removed amplitude swap no longer widens the on-axis EE80 at zero
decenter); N10b GBD as the decenter reference --
`test_gbd_decenter_broadens_grid_robust_rms` (the GRID-ROBUST replacement for the
killed `..._ee80_two_wavelengths`: reconstructs the evolved beamlets on a
spot-resolved fine grid and asserts the RMS broadening is > 1, invariant across
two reconstruction pitches to <0.5%, monotonic in decenter, and within ~15% of the
geom-oracle coma RMS -- at 1.31 + 0.633um), centroid + power + the GBD-vs-traced
cross-oracle + sign-mirror.  The verifier's round-1 repro scripts
(`scratchpad/p9_isolate_amp.py` / `p9_confirm.py` / `p9_grid_refine.py` /
`p9_633_grids.py`) show the amplitude artefact removed (amp-artifact 0.999 across
grids, was 1.083-1.104); the round-2 grid-fragility repro is
`scratchpad/vf_gbd_traced_grid.py` (on-axis EE80 wanders ~11% across dx; 1 mm
ratio swings 0.997 -> 1.034 -> 1.003) and the grid-robust re-measurement is
`scratchpad/p9fix_decouple.py` / `p9fix_calib.py` (RMS pitch-invariant to 4 sig
figs).  The ZOS comparisons above were run fresh with `scratchpad/zos_oracle_p3.py`
(per-surface `TiltDecenterData` with after-return; AUTO POP beam sampling -- a
fixed `width_mm` clips the 4mm waist and pixel-limits the grid, the P3-documented
POP finickiness).

## P10 -- analytic 2-D transverse-walk remap: the decentered coma spot broadens (niche N11, 2026-07-20)

P3's pointwise obliquity SCREEN imprints the refraction OPD at the
STRAIGHT-THROUGH grid position, so it captures the coma flare DIRECTION (centroid
+ skewness) but CANNOT represent the TRANSVERSE ray walk between a thick element's
two surfaces -- the induced-coma spot NARROWED (~0.91x) where the geometric-spot
oracle and ZOS both BROADEN (~1.02-1.03x).  P9 (N10) made the two RAY models
honour decenter/tilt (GBD became the accurate reference).  **P10 generalises P2's
rotationally-symmetric exit-plane remap to the full OFF-AXIS 2-D case, so the
ANALYTIC default now broadens the decentered spot correctly -- the P3 open finding
is CLOSED for the default analytic path (the remap CLOSED it; the pre-registered
N11 routing fallback was not needed).**

### Mechanism -- one geometric transfer, no free parameters

`_build_displaced_ray_map_2d` launches a REGULAR 2-D square congruence fan
(`n_side=181`, spanning +-1.03*r_max so the aperture disk has interior neighbours
for the Jacobian) against the decentered/tilted/freeform surfaces via the SHARED
`_disp_surface_z_grad` geometry (identical field-frame convention to the P3
pointwise screen, the P9 traced/GBD ray models, and the lumenairy-free
`geom_spot_decenter_oracle`), and builds the exit ray map `(x_out,y_out)(x_in,
y_in)` + total OPL (entrance eikonal + per-segment `n*path`, exactly as the 1-D
remap).  `_apply_displaced_remap_2d` then (a) samples the input amplitude envelope
`|E_in|` at each ray's entrance position, (b) transports it to the exit position
with the energy-conserving 2-D Jacobian factor
`1/sqrt(|det d(x_out,y_out)/d(x_in,y_in)|)` (finite-differenced on the regular
launch grid, dead rays nearest-filled first), and (c) interpolates the amplitude
and the OPL SEPARATELY (phase-safe -- the eikonal is smooth) onto the field grid,
combining as `amp * exp(i k0 (opl - opl_ref))`.  The transverse walk is carried by
`(x_out,y_out) != (x_in,y_in)`; the Jacobian carries the ray-density
redistribution the single-plane screen cannot.  The aperture is enforced on the
fan's ENTRANCE footprint (`r0 <= r_max`); the 3%-wider launch supplies only
Jacobian neighbours -- this closes a beyond-aperture Gaussian-tail LEAK found in
review (see below).  It is the DEFAULT for asymmetric elements
(`displaced_obliquity='auto'`) and is also selectable via `displaced_mode=
'remap'`; explicit `displaced_obliquity='pointwise'` keeps the single-plane SCREEN
peer.  No cache is added (opt-in accuracy path, re-runs per call, like the P3
pointwise trace).

### The honest metric is RMS / coma-RMS, NOT EE80 (a caught compensating error)

A first draft of this section reported an EE80 broadening ratio of 1.030 @1 mm
"matching ZOS 1.035".  An adversarial re-check (mirroring the P9 verifier's GBD
correction) REFUTED that number as a **beyond-aperture Gaussian-tail LEAK**: the
fan was contributing rays outside the true pupil (the exit-plane remap bypasses
the per-surface stop mask), and those high-aberration tail rays inflated the EE80.
With the aperture correctly enforced (energy now 0.9995 vs the leaky 1.025) the
decentered EE80 barely moves (~1.00) -- because the wave PSF carries a large
diffraction core (on-axis RMS ~21 um vs the geom ray-density ~15 um) that DILUTES
the ~4 um coma in the EE80 threshold-crossing.  **This is exactly what P9 found
for GBD** (whose EE80 headline was withdrawn as noise/dilution).  The honest,
grid-robust metric is the RMS second-moment radius + the common-mode-subtracted
in-quadrature coma RMS `sqrt(RMS_dec^2 - RMS_on^2)` -- and by THAT metric the remap
broadens correctly and matches the geom oracle.

### Decenter accuracy table (analytic-remap vs pointwise-screen vs geom, RMS metric)

f/5 biconvex singlet (R=+-51.68 mm, t=5 mm, n=1.5168, aperture 10 mm), COLLIMATED
Gaussian w0=4 mm (fills the aperture), 1.31 um, at the paraxial image; WINDOWED
RMS second-moment radius (110-um window) about the intensity centroid, N=3072 /
dx=4 um (sub-exit-Nyquist), grid-robust to 4 sig figs.  `geom` = the lumenairy-free
`geom_spot_decenter_oracle` (geometric ray-density spot).  Baselines use a
NEGLIGIBLE (1e-9 m) decenter so the on-axis and decentered rows are the SAME model.

| model | on-axis RMS | 1 mm RMS | 1 mm ratio | 2 mm RMS | 2 mm ratio |
|---|---|---|---|---|---|
| pointwise SCREEN (P3 peer) | 21.1 | 20.20 | **0.956 (SHRINKS)** | 18.73 | 0.886 |
| **2-D remap (P10 DEFAULT)** | 21.09 | **21.59** | **1.023 (BROADENS)** | **23.01** | 1.091 |
| geom-spot oracle (RMS) | 14.91 | 15.48 | 1.038 | 17.16 | 1.151 |

Common-mode-subtracted in-quadrature coma RMS `sqrt(RMS_dec^2 - RMS_on^2)` -- the
honest apples-to-apples gate (P9 method): **remap coma RMS 4.58 (1 mm) / 9.18
(2 mm) um vs the geom oracle 4.16 / 8.51 = 1.10 / 1.08** (within ~10%), grid-robust.

KEY RESULTS:

* **The remap BROADENS, closing the P3 shrink -- the SAME result P9 got for GBD.**
  Its on-axis RMS 21.09 um MATCHES the GBD reference (21.09); the RMS ratio
  broadens **1.023 @1 mm / 1.091 @2 mm** (grid-robust to 4 sig figs: 1.0229 vs
  1.0233 across dx=4.8/4.0 um), MONOTONIC, and the common-mode-subtracted coma RMS
  matches the geom oracle within ~10% at both decenters.  The retained single-plane
  SCREEN, by contrast, SHRINKS (RMS 0.956 @1 mm / 0.886 @2 mm; its coma RMS is
  imaginary) -- the documented walk-off limit + fail-before anchor.
* **The decentered EE80 is DIFFRACTION-DILUTED (not the metric to use).**  The
  wave PSF core (~21 um) dwarfs the ~4 um coma, so the aperture-correct remap's
  EE80 barely moves (~1.00) -- exactly as P9 found for GBD (EE80 headline withdrawn).
  The RMS / coma-RMS metric above is the honest one; the EE80 ratio is not gated.
  (The earlier "EE80 1.030" was a beyond-aperture leak, now fixed.)
* **Centroid / sign-mirror / tilt all correct.**  Decenter centroid 510.8 vs geom
  510.3 um (~0.1%; the wave-intensity centroid runs ~2.7% past the geom
  ray-density centroid at the tighter w0=3 mm config -- the coma tail pulls it);
  +d/-d PSFs mirror to numerical precision (EE80 equal, centroid mirror <0.1%,
  intensity mirror-L2 <1%); a 0.2 deg tilt deflects to within 0.2% of an
  independent rigid-rotation ray trace; the coma-flare skewness is >3, sign-exact.
* **Energy + phase.**  With the aperture enforced, exit power is within **0.05%**
  of the aperture-transmitted input (the geometric transfer is lossless; the
  scattered->grid interpolation surplus, ~2.5% before the aperture fix, is gone);
  the exit field is PHASE-CONTINUOUS (max adjacent-pixel step 1.6-2.1 rad < pi at
  sub-Nyquist sampling), so it focuses coherently.

### Grid-robustness + sampling

The exit NA is ~0.1, so the exit-Nyquist limit is `dx <= lambda/(2 NA_exit) ~
6.5 um` at 1.31 um.  The RMS second-moment radius (a continuous integral, not a
single-radius threshold crossing) is grid-robust: on-axis 21.10/21.09/21.09 and
the 1 mm ratio 1.0229/1.0233/1.0233 across dx=4.8/4.0/3.0 um (the EE80 threshold
crossing, by contrast, is quantization-noisy on the ~4 px spot -- the P9 lesson,
which is why RMS is the gated metric).  **At 0.633 um** the exit-Nyquist limit
tightens to ~3.2 um: at dx=4 um the metric aliases, at **dx=3 um the RMS is clean**
(on-axis 21.04, 1 mm ratio **1.022**, matching the 1.31-um 1.023).  The unit tests
gate 1.31 um at dx<=4.8 um and assert the 0.633-um broadening DIRECTION at dx=3 um.

### Routing story + the pre-registered N11 risk

The plan N11 pre-registered that the remap was NOT guaranteed to close the coma
gap (P2 had measured remap == screen on the ON-AXIS symmetric conjugate cases,
where the walk is a radial rescale the screen already captures), and specified an
honest fallback: keep the screen for centroid/pointing and route strong-decenter-
coma to the now-decenter-capable ray models (P9).  **The remap DID close it** by
the same measure the P9 GBD reference passed -- the common-mode coma RMS matches
the geom oracle within ~10% (GBD: ~10-15%), where the single-plane screen shrinks.
So the primary outcome holds: a user who decenters an element and uses the default
`surface_model='displaced'` no longer gets a silently-wrong NARROW spot (the
unacceptable outcome the plan warned against) -- the spot now broadens with the
correct coma magnitude.  The ray models (P9 traced/GBD) remain available and the
N8 gate can still route to them.  HONEST CAVEAT: like GBD, the ABSOLUTE spot size
carries a diffraction/reconstruction core, so for absolute decentered-spot EE
magnitude vs an external wave reference, ZOS remains the arbiter; the remap's
contribution is the correct coma DIRECTION + MAGNITUDE (coma RMS), grid-robust.

### Tests + provenance

`tests/unit/test_niche_p10_transverse_walk_remap.py` (runs WITHOUT Zemax, using
the `geom_spot_decenter_oracle`): the SYMMETRIC-limit byte-identical pin
(`displaced_mode='remap'` on a symmetric element == the P2 1-D remap internals
bit-for-bit); zero-decenter byte-identical (symmetric default == meridional
screen); default-routing (decentered default == `displaced_mode='remap'`, !=
pointwise screen); split-rejected-for-asymmetric; centroid vs geom (<5%); signed
coma flare + sign-mirror (exact); tilt deflection vs rigid rotation (<10%);
phase-continuity + energy (<5%); freeform `sag_callable` via the remap; and the
SLOW headlines -- `test_decenter_broadens_grid_robust_rms` (RMS broadens,
grid-robust across two sub-Nyquist grids to <0.5%, monotonic, on-axis RMS ~21 um,
and the fail-before SCREEN shrink), `test_remap_coma_rms_matches_geom_oracle`
(common-mode coma RMS within 20% at 1 + 2 mm -- the same gate + tolerance the P9
GBD reference passed), and `test_decenter_broadens_second_wavelength` (0.633 um RMS
direction).  The P3 `test_coma_ee_growth_screen_limit_pinned_remap_fixes_it` pins
the SCREEN shrink AND that the default remap's EE80 ratio sits well above the
screen's collapse (the true broadening magnitude lives in the P10 RMS gate).  The
verifier repro of the caught EE80 leak is `scratchpad/p10_final.py` /
`p10_proto.py`; the ZOS numbers are the fresh P9 POP run; the unit tests do not
require Zemax.

## P11 -- traced ray-density (Jacobian) amplitude mode (niche N12, 2026-07-20)

P9 (N10a) documented that `apply_real_lens_traced`'s exit AMPLITUDE is the
single-plane analytic-screen leg `|E_analytic|`, so the traced decentered-spot EE
is amplitude-limited (route EE to GBD).  N12 adds an **opt-in**
`apply_real_lens_traced(amplitude_model='ray_density')` (default `'screen'` =
current, byte-identical): the exit magnitude becomes the geometric ray-tube
energy-conserving amplitude `|E_in(x_in)| / sqrt(|det J|)`,
`J = d(x_out,y_out)/d(x_in,y_in)` the exit ray-map Jacobian (analytic gradient of
the entrance->exit fit), placed at the exit ray position with the traced OPL
phase.  The complex dtype is preserved; the assembly keeps the screen-mode phase
(its unit phasor) and swaps only the magnitude.

### Mechanism + caustic handling

`_ray_density_amp_grid` reuses the SAME entrance->exit fits + Newton inverse the
OPL phase uses (so amplitude and phase share exit positions): for each exit pixel
Newton returns the entrance `(xe, ye)`, `det J` is the analytic gradient of the
forward-map fit there, and `|E_in|` is bilinearly sampled at the entrance.  The
aperture stop is enforced at the ENTRANCE (a ray blocked by the stop carries no
energy), which makes the ray-density power exactly the aperture-transmitted input
power.  **Caustic handling (mandatory):** `det J -> 0` at a fold, so the amplitude
is DETECTED near-caustic (an absolute floor `1e-3 * median(|det J|)`, a `|det J|`
dynamic-range `> 30x`, OR a det J sign change between adjacent ray cells), CAPPED
(never inf/nan), and a one-time `RuntimeWarning` steers to
`apply_real_lens_gbd` / `apply_real_lens_fga`.  **This phase does NOT implement
the multi-branch KMAH/Maslov sum** -- GBD/FGA remain the caustic reference; the
single-branch mode's job is an honest, finite, flagged envelope at a fold.

### What was validated (measured, adversarial)

| gate | result |
|---|---|
| DEFAULT byte-identical | `amplitude_model='screen'` (default) == prior releases, `np.array_equal` pinned |
| ENERGY closure (< 0.5%, away from folds) | ray-density power / aperture-transmitted input = **0.999** at the exit vertex, and **decenter-STABLE** (0.999 at 0 / 1 / 2 mm) where the screen `apply_real_lens` amplitude LEAKS ~9% at a 2 mm decenter -- no silent renormalisation |
| CAUSTIC (`caustic_fold_ref`, fold DOWNSTREAM of the exit vertex) | exit-vertex ray-density field is FINITE + no fold warning (the vertex is single-valued); ASM to the fold plane (a wave method that handles the multi-valued caustic) matches the dense direct-RS reference: **r2m 0.1% / EE50 0.9% / EE80 3.0%** -- NO blow-up |
| CAUSTIC-AT-OUTPUT (traced output placed AT the focus) | `det J -> 0` DETECTED (warns, steers to GBD/FGA), output FINITE (never inf/nan); single-branch energy is NOT conserved there (expected, flagged) |
| SIGN-MIRROR | +d/-d ray-density PSFs mirror: centroid < 0.2%, intensity mirror-L2 **1.1%** |
| COLLIMATED unaberrated (slow lens) | det J ~ const -> ray-density reduces to a scaled input envelope, reproducing the screen (Airy-limited) field to < 10% over the bright support |
| ON-AXIS aberrated vs Debye | ray-density is oracle-CONSISTENT: EE tracks the validated screen leg to < 5% and, like screen, trends toward the `debye_oracle_v3` diffraction EE as the exit-NA-Nyquist sampling tightens (H3-limited at coarse dx) |

### The decenter premise -- REFUTED for the traced output plane (honest limit)

N12's premise was that `det J` (of traced's ray map) IS the coma redistribution
the screen leg lacks.  **Measured, this is false at the traced OUTPUT plane (the
exit VERTEX).**  The exit-vertex ray map is nearly the identity: `det J` median
**0.933**, spread only **0.007** (centered) -> **0.011** (2 mm decenter), i.e. an
amplitude modulation of ~0.3%.  So `ray_density` ~ `screen` there, and BOTH
broaden the decentered image-plane spot (RMS ratio **~1.06 @1 mm / ~1.10 @2 mm**,
grid-robust, monotonic -- never a shrink; comparable to the geometric oracle's
1.038/1.151 and the P9 GBD reference), tracking each other to within ~3%
(RMS-ratio).  (Note: the EE80 ratio is grid-fragile and is deliberately NOT the
gated metric -- the P9/P10 lesson -- so the earlier "traced shrinks" reading was
substantially an EE80 quantization artifact on an undersampled spot; with the
grid-robust RMS second-moment, traced broadens.)  The coma redistribution is a
DOWNSTREAM effect: at the IMAGE plane `det J` spans orders of magnitude and hits
~0 (min ~3e-15 -- a caustic at focus), so it is carried by the PHASE + propagation,
and single-branch ray density AT the image is a caustic (flagged, unreliable).

**Conclusion:** an exit-plane amplitude model cannot carry an aberration that
develops in propagation to the image.  `ray_density` makes the traced decentered
spot BROADEN (killing any shrink) and is energy-/caustic-correct, but it does NOT
make the focal-plane decentered PSF match the geometric oracle / GBD to 15% --
**`apply_real_lens_gbd` (N10b), whose beamlets carry the IMAGE-plane ray density,
remains the decentered-coma reference (unchanged from P9).**  (The raw coma-RMS /
geom ratio is window-fragile -- the P9/P10 lesson -- so it is NOT gated; the robust
pinned statement is that `ray_density` TRACKS `screen` at the exit vertex.)

### Where ray_density genuinely helps

`ray_density` is the physically-principled, ENERGY-CONSERVING geometric ray-tube
amplitude -- a decenter-STABLE alternative to the screen `apply_real_lens`
amplitude (which leaks energy under decenter), a smooth geometric envelope free of
the screen's exit-aperture Fresnel edge-ripple, and CAUSTIC-SAFE (detect + cap +
warn).  It differs materially from screen only where the traced OUTPUT-plane
Jacobian carries real ray-density structure; when the user traces to a plane AT a
focus that structure is a caustic (flagged), and at the exit vertex it is
near-trivial.

### Tests + provenance

`tests/unit/test_niche_p11_ray_density_amplitude.py` (runs WITHOUT Zemax, using
the lumenairy-free Debye oracle + the `caustic_fold_ref.npz` ground truth):
default byte-identical; finite + complex64-preserving; the validation-error
surface (bad model / `return_screen` / non-Newton inversion raise); energy
closure + decenter stability; collimated slow-lens Airy limit; sign-mirror;
on-axis aberrated vs Debye; the SLOW `test_caustic_fold_no_blowup_matches_reference`
(exit-vertex ray-density + ASM within ~8% of the fold ground truth, no inf/nan,
no false vertex warning); `test_caustic_at_output_plane_detected_and_finite` (the
focus caustic warns + stays finite); and the SLOW
`test_decenter_broadens_grid_robust_but_tracks_screen` (ray-density RMS broadens
grid-robustly + monotonically but TRACKS the screen within 3% -- the honest
exit-vertex limit).  Verifier repros: `scratchpad/rd_detj.py` (det J ~const at the
exit vertex, -> 0 at the image), `scratchpad/rd_energy_caustic.py` (energy +
caustic fold), `scratchpad/rd_focusprobe.py` (the focus caustic).

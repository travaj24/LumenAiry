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

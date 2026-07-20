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

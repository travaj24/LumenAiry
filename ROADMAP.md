# Roadmap — Known Limitations and Planned Improvements

This document catalogues remaining limitations in `lumenairy`
(currently 4.8.0) and outlines planned improvements.  Items are grouped
by module and roughly prioritised within each section.  Items that have
shipped since the original 3.0 roadmap was written are recorded in the
"Resolved since 3.0" appendix at the bottom of this file; the per-
section limitations below have been pruned to reflect *current* gaps,
not historical ones.

---

## Resolved since 3.0 (highlights)

- 4.4 — **Field-resolved analysis suite** (distortion grid, footprint
  per surface, spot diagram vs field, sensitivity ranking, Petzval
  radius, field-aberration sweep, relative illumination) lifted out of
  the GUI and into `lumenairy.analysis.field`.
- 4.4 — **World-frame ray tracing** for folded designs
  (`paraxial_focus_world`, world-frame trace + Seidel) in
  `lumenairy.raytrace.world`.
- 4.4 — **Adaptive optics module** (`DeformableMirror`, `apply_dm`,
  slope-to-modal reconstruction, `LeakyIntegrator`) lifted from
  validation scripts to `lumenairy.analysis.ao`.
- 4.5 — **Diffractive lens trio** (`create_diffractive_lens`,
  `create_diffractive_lens_kinoform`,
  `create_diffractive_lens_fresnel_zone_plate`) in
  `lumenairy.elements.doe`.
- 4.5 — **Strehl variants** (`strehl_ratio`, `strehl_marechal`,
  `strehl_phase_integral`) consolidated and exported flat.
- 4.6 — **Wiki overhaul**: 12-chapter physics textbook
  (Physics-I-Foundations.md … Physics-XII-Coronagraphy.md) plus
  function-reference manual, plus decision-tree Mermaid diagrams
  for propagator + lens selection.
- 4.7 — **Module rename**: `lumenairy.analysis.analysis` →
  `lumenairy.analysis.core` (back-compat shim retained).
- 4.7 — **Input validation**: `_validate_propagator_inputs` on all
  propagators (catches wavelength = 0 / negative / unit-confused
  values, dx = 0 / negative, wrong-rank inputs, empty arrays);
  `validate_prescription` on `surfaces_from_prescription` (catches
  empty / under-specified / NaN-containing prescriptions before
  they reach the trace).
- 4.7 — **Glass registry** supports user-supplied dispersion
  callables; ~30 additional bundled Sellmeier glasses
  (Schott N-FK51A, N-PSK53A, N-LAK33A/B, N-SF57, S-LAH64, etc.).
- 4.7 — Module-level `__all__` exports across all 9 analysis
  submodules; PyPI `Changelog`/`Releases` URLs in `pyproject.toml`.

---

## 1. Lens modelling (`lenses.py`)

### Current limitations

- **Uniform-slab approximation in `apply_real_lens`** — each glass
  region is treated as a uniform slab between vertex planes.  Rays
  through cemented doublets actually cross the cemented interface at
  `z = sag(h)`, not at a single vertex plane.  This is the dominant
  error source for multi-surface curved-interface systems (hundreds of
  nm RMS on cemented doublets at full aperture).
  `apply_real_lens_traced` eliminates this error but is 3-10x slower.

- **No polarization-dependent Fresnel coefficients** — the Fresnel
  transmission option (`fresnel=True`) uses an s/p scalar average.
  It does not produce polarization-split fields or couple into the
  Jones calculus framework.

- **No dichroic or wavelength-selective surfaces** — every surface is
  either fully transmitting or fully reflecting (mirror).  Dichroic
  beam splitters, notch filters, and bandpass coatings are not
  modelled inline (the `coatings.py` TMM module exists but is not
  integrated into the lens pipeline).

- **`apply_real_lens_traced` assumes collimated input** — each pixel
  ray is launched parallel to z.  Converging, diverging, or tilted
  input beams are not supported; fall back to `apply_real_lens` for
  those cases.

- **`apply_real_lens_maslov` (3.1.7) is experimental** — the
  Chebyshev-polynomial phase-space propagator is physically correct
  on the cases tested (Design 51 L1 with collimated Gaussian input,
  sub-1 % RMS intensity error) but has not been exercised across the
  full breadth of use cases:
    * Only L1 tested; L2, L3, L4 not yet verified.
    * Only collimated input tested; converging / diverging / tilted
      not yet verified.
    * Not yet integrated into a multi-element forward simulation
      pipeline.
    * Caustic output-plane regime (its headline use case) not yet
      exercised.
    * The `'local_quadrature'` integration method is broken when
      combined with `collimated_input=True` (samples hit the 1-D-
      valid subspace of the polynomial fit).
  Use `apply_real_lens_traced` as the reference for production runs
  until the coverage is broader.

- **No deformable or adaptive surfaces** — prescriptions are static.
  There is no mechanism for real-time surface perturbation (e.g.,
  deformable mirror actuator influence functions).

### Planned improvements

- Integrate `coatings.py` TMM reflectance/transmittance into the
  per-surface phase-screen model so that AR coatings, dichroic
  surfaces, and partial reflectors can be specified per surface in
  the prescription dict.
- Support converging/tilted input in `apply_real_lens_traced` by
  launching rays with direction cosines derived from the local
  wavefront slope.
- Investigate full WPM (Wave Propagation Method) as an alternative
  to the thin-element model for sharp-interface propagation through
  thick glass.  Estimated ~1 week of work for a basic implementation.
- Extend `apply_real_lens_maslov` validation: test on L2-L4 of
  Design 51; run against user-provided validation lenses; integrate
  into a forward-path simulation and compare end-to-end against
  `apply_real_lens_traced`.
- Fix `'local_quadrature'` for the `collimated_input=True` regime
  (sample along the 1-D ray curve rather than in a 2-D box around
  `v2*`).
- GPU path (`use_gpu=True`) for `apply_real_lens` and
  `apply_real_lens_traced` — currently CPU-only even when
  `propagation.py` dispatches FFTs to CuPy.  Explicit CuPy branches
  for the phase-screen multiplies and surface-sag computations would
  unlock 5-10x speedup on ASM-dominated sections.

---

## 2. Ray tracing (`raytrace.py`)

### Current limitations

- **Sequential surfaces only** — all surfaces share a common optical
  axis (z).  Folded systems, off-axis parabolas, and arbitrary 6-DOF
  surface placement require manual coordinate transforms that are not
  built into the trace engine.

- **No gradient-index (GRIN) ray bending** — GRIN lenses are available
  as a thin-element phase screen (`apply_grin_lens`), but the ray
  tracer does not curve rays through a continuously varying index
  profile.  This limits GRIN accuracy to low-NA, thin-element
  approximations.

- **No edge diffraction at apertures** — vignetting is binary
  (alive/dead at the semi-diameter boundary).  Physical-optics edge
  effects (Fresnel diffraction at aperture stops) are handled by the
  wave model, not the ray tracer.

- **No polarization ray tracing** — the ray tracer tracks OPL but not
  the Jones vector along each ray.  Polarisation-dependent effects
  (birefringent crystals, stress-optic, Fresnel phase splits) are
  not captured.

- **Threading limited by GIL** — `RectBivariateSpline.ev()` and the
  Newton inversion hold the GIL on stable CPython.
  `ThreadPoolExecutor` serialises and adds overhead.

### Planned improvements

- Non-sequential / 6-DOF surface placement using homogeneous
  coordinate transforms (4x4 matrices) per surface.
- GRIN ray bending via Runge-Kutta integration of the ray equation
  in a continuously varying index field.
- Free-threading Python 3.13t/3.14t support — the existing threaded
  Newton code would get near-linear speedup once the GIL is removed.
- `ProcessPoolExecutor` fallback for current CPython — requires
  refactoring the Newton closure to a module-level picklable function.
- CuPy GPU port of the Newton inversion (embarrassingly parallel).

---

## 3. Propagation (`propagation.py`)

### Current limitations

- **Fully coherent fields only** — all propagators assume a single
  monochromatic, spatially coherent field.  Broadband or partially
  coherent propagation requires the user to call the propagator
  multiple times and incoherently sum intensities.  The `coherence.py`
  module provides `extended_source_image` and `mutual_coherence` but
  these are separate workflows, not integrated into the propagator.

- **No Beam Propagation Method (BPM)** — there is no split-step
  Fourier or finite-difference BPM for waveguide or GRIN propagation.
  Simple BPM was prototyped and failed for sharp air-glass interfaces
  (the slowly-varying envelope approximation breaks at delta-n = 0.5).
  It would work for actual GRIN media (delta-n ~ 0.01) but is not
  implemented.

- **No nonlinear propagation** — no Kerr effect, self-phase
  modulation, self-focusing, or harmonic generation.

- **Fresnel propagator changes grid spacing** — `fresnel_propagate`
  returns `(E, dx_out, dy_out)` with a different pixel pitch at each
  plane.  Users must resample before applying subsequent phase screens.
  `propagate_through_system` handles this transparently, but the raw
  API is a footgun.

- **Inconsistent return types** — `angular_spectrum_propagate` returns
  a bare array, `fresnel_propagate` returns a 3-tuple, and
  `fraunhofer_propagate` returns a 3-tuple.  This forces users to
  remember which function returns what.

### Planned improvements

- Split-step Fourier BPM for GRIN media (delta-n < 0.05) with
  automatic step-size selection.
- Unified return type across all propagators (consider a `PropResult`
  namedtuple with `.field`, `.dx`, `.dy` attributes).
- Broadband propagation helper that loops over wavelengths and sums
  intensities, with optional spectral weighting.

---

## 4. Glass catalogue (`glass.py`)

### Current limitations

- **Sellmeier dispersion only** — no Cauchy, Conrady, Herzberger,
  Drude, or polynomial dispersion models.  Custom materials must be
  added via `GLASS_REGISTRY` with a Sellmeier-compatible callable.

- **No thermal dispersion (dn/dT)** — refractive indices are at a
  single temperature (typically 20-25 C).  There is no mechanism to
  specify or apply thermal coefficients.

- **Limited extinction data** — `get_glass_index_complex` returns
  absorption for some materials, but many Schott glasses in the
  registry have kappa = 0 (no absorption data).

- **No nonlinear optical coefficients** — no n2 (Kerr), chi(2) (SHG),
  Raman gain, or Brillouin coefficients.

- **No stress-optic coefficients** — no photoelastic effect modelling
  for mechanically loaded optics.

- **Limited programmatic introspection** — typos in glass names
  produce a `KeyError` with a long list of valid keys.  A future
  release will add `list_glasses()` / `search_glasses(pattern)`
  / `glass_info(name)` helpers.

### Planned improvements

- `list_glasses()` and `search_glasses(pattern)` helpers for
  discoverability.
- Thermal dispersion model: `get_glass_index(name, wavelength, T)`
  with dn/dT coefficients for common Schott glasses.
- Additional dispersion formula support (Cauchy, Conrady) for legacy
  glass data and user-defined materials.

---

## 5. Polarization (`polarization.py`)

### Current limitations

- **Jones calculus only** — no Mueller matrix support.  Depolarising
  media (rough surfaces, scattering, birefringent diffusers) cannot
  be modelled.

- **Not integrated with system propagation** — `propagate_through_system`
  does not accept `JonesField` inputs.  Users must manually extract
  Ex/Ey, propagate each, and re-wrap.

- **No birefringent material library** — uniaxial crystal plates are
  supported via the generic Jones matrix, but there is no built-in
  database of crystal properties (calcite, quartz, BBO, KDP, etc.).

- **No Faraday rotation** — magneto-optic effects (optical isolators,
  Faraday rotators) are not modelled.

- **Scalar propagation per component** — `JonesField.propagate()`
  calls the scalar propagator on Ex and Ey independently.  This is
  correct for isotropic media but incorrect for anisotropic or
  nonlinear media where the two components couple.

### Planned improvements

- Mueller matrix framework alongside Jones for partially coherent /
  depolarising scenarios.
- `JonesField` integration into `propagate_through_system` as a
  first-class element type.
- Built-in birefringent crystal database (calcite, quartz, BBO, KDP,
  LiNbO3) with wavelength-dependent ordinary/extraordinary indices.

---

## 6. Optimizer (`optimize.py`)

### Current limitations

- **No nonlinear constraints** — only parameter bounds are supported.
  Scipy's `SLSQP` and `trust-constr` support nonlinear constraint
  functions, but `design_optimize` does not expose them.
  `MinThicknessMerit` and `MaxFNumberMerit` exist as penalty-based
  soft constraints, but hard inequality constraints would be more
  robust.

- **No linked parameters** — cannot constrain two parameters to be
  equal (e.g., symmetric doublet where R1 = -R4) or related by a
  formula.

- **No multi-objective optimisation** — merit terms are combined via
  weighted linear sum only.  No Pareto front calculation or
  multi-objective evolutionary algorithm.

- **No sensitivity / gradient shortcuts** — every merit evaluation
  requires a full ray trace or wave propagation.  Analytic gradients
  (e.g., from ABCD matrix derivatives for focal length) are not
  exploited.

- **No automatic variable selection** — the user must manually specify
  which prescription parameters are free.  No helper to "make all
  radii free" or "make all thicknesses free" with a single call.

### Planned improvements

- Expose scipy nonlinear constraints through `design_optimize`.
- Linked-parameter support (equality and formula-based constraints).
- Analytic gradient computation for geometric merit terms (EFL, BFL,
  Seidel) via ABCD matrix differentiation.
- Auto-variable helper: `parameterization.free_all_radii()`,
  `.free_all_thicknesses()`, etc.

---

## 7. Sources (`sources.py`)

### Current limitations

- **All sources at waist** — Gaussian, HG, and LG modes are returned
  at the beam waist with flat phase.  There is no option to specify
  a waist location (producing a converging or diverging beam) or to
  propagate the mode to a given z before returning.

- **No mode decomposition** — given an arbitrary field, there is no
  function to decompose it into HG or LG mode content (overlap
  integrals).

- **No partially coherent source model** — all sources are fully
  spatially coherent.  Gaussian Schell-model beams, thermal sources
  with finite coherence area, and LED angular/spatial distributions
  beyond `create_led_source` are not available.

- **Inconsistent return types** — `create_gaussian_beam` returns
  `(E, X, Y)`, while some other creators return `(E, X, Y)` and
  others return just `E`.

### Planned improvements

- `waist_location` parameter for Gaussian/HG/LG modes.
- HG/LG mode decomposition: `decompose_hg(E, dx, w0, max_order)`.
- Consistent return types across all source functions.

---

## 8. System propagation (`system.py`)

### Current limitations

- **No polarization elements** — the system element list does not
  support polarizers, waveplates, or Jones matrices.

- **No inline coating model** — surfaces in the system list are
  either fully transmitting or fully reflecting.  There is no way to
  specify per-surface R/T from the coatings module.

- **No deformable mirrors or SLMs** — active optical elements require
  the user to construct a custom phase mask externally and insert it
  as a `'mask'` element.

- **Fresnel grid-scaling is silent** — when a Fresnel propagation step
  changes dx/dy, subsequent lens phase screens are automatically
  resampled, but no warning is emitted about the interpolation error
  this introduces.

- **No metadata on intermediates** — when `save_intermediates=True`,
  the returned list contains bare field arrays with no tags indicating
  which element or distance each corresponds to.

### Planned improvements

- Polarization-aware system propagation accepting `JonesField` and
  Jones matrix elements.
- Intermediate metadata: return a list of `(field, element_info)` tuples.
- Warning when Fresnel resampling exceeds a configurable error
  threshold.

---

## 9. Detector model (`detector.py`)

### Current limitations

- **Monochromatic only** — the detector integrates a single-wavelength
  intensity pattern.  No spectral response curve (QE vs wavelength).

- **No pixel cross-talk** — each detector pixel integrates
  independently.  Charge diffusion, optical cross-talk, and MTF
  degradation from pixel geometry are not modelled.

- **No readout electronics model** — no ADC quantisation, gain
  non-uniformity, or fixed-pattern noise.

### Planned improvements

- Wavelength-dependent QE curve.
- Pixel MTF model (sinc-based or measured) applied before sampling.
- ADC quantisation and gain non-uniformity.

---

## 10. Performance

### Current bottlenecks

- **`apply_real_lens_traced`** is 3-10x slower than the analytic model
  due to per-pixel Newton inversion.  `ray_subsample=4` reduces this
  to ~15x faster than unsubsampled, but the function is still the
  slowest single call in a typical simulation.

- **Large-grid FFTs** — at N = 8192-16384, FFT cost dominates.  SciPy
  FFT with `workers=-1` is the current default (2-4x speedup over
  single-threaded NumPy).  pyFFTW is opt-in but uses ~2x memory.

- **No GPU acceleration for lens models** — `apply_thin_lens` supports
  CuPy but `apply_real_lens` and `apply_real_lens_traced` are
  CPU-only.

- **Sequential ray trace is single-threaded** — `trace()` processes
  surfaces sequentially (inherent) and rays in parallel via NumPy
  vectorisation, but all computation is on a single CPU core.

### Planned improvements

- CuPy GPU port of `apply_real_lens` and `apply_real_lens_traced`.
- Free-threading CPython support (3.13t/3.14t) for the Newton
  inversion and spline evaluation.
- Investigate JAX or Numba JIT for the Newton inversion hot loop.
- Memory-mapped intermediate fields for through-focus scans that
  exceed RAM.

---

## 11. Validation and testing

### Current state

- ~670 `t_*` validation functions across 34 files in `validation/`
  (propagators, ray-trace, lenses, sources, analysis, optimization,
  prescriptions, storage, asymptotic, DOE, field-resolved analyses,
  world-frame trace, adaptive optics, coherence, folded designs).
- 49-test unit suite in `tests/unit/` (runs in <1 s, catches API-
  contract regressions on a fresh checkout, no external deps).
- Inter-library cross-checks against Zemax, rayoptics, Optiland,
  and OPDPy documented in [[Validation and Accuracy]].
- Pytest wrapper in `tests/integration/` runs the validation
  scripts as subprocesses (`-m integration`).
- GitHub Actions CI runs the unit suite on Python 3.11 / 3.12 /
  3.13 on Ubuntu + Windows on every push.
- PyPI Trusted Publishing OIDC auto-release on tagged commits.

### Planned improvements

- Comparison against published Zemax/CODE V benchmark prescriptions
  (currently we validate vs Zemax for the prescriptions we own; a
  curated set of textbook prescriptions would broaden coverage).
- Performance regression tracking (wall-clock time per case).
- Coverage expansion for the 4.5 DOE module (currently has a small
  validation file, but no cross-library comparison).

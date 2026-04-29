# `apply_real_lens` — change log, design, and validation

This document tracks the evolution of the `apply_real_lens` family
in `lumenairy.lenses`, what each feature does, and what the
validation data says about when each approach helps.

---

## TL;DR — which function to use

| Need | Function | Typical accuracy | Typical cost |
|---|---|---|---|
| Fast analytic model, most lenses | `apply_real_lens(…)` (defaults) | 10 nm – 100 µm RMS | 1–10 s |
| Same, with Fresnel/absorption | `apply_real_lens(…, fresnel=True, absorption=True)` | same | +50 % |
| Cemented doublets or multi-surface curved systems, still closed-form | `apply_real_lens(…, seidel_correction=True)` | 3-5× better than default on doublets, can be worse on singlets | +5 % |
| Sub-nm OPD agreement with geometric ray trace | `apply_real_lens_traced(…, ray_subsample=8)` | sub-nm | ~15 s on 4k² grid (3.1.7 default) |
| Caustic-safe output plane; analytically differentiable model for design optimization | `apply_real_lens_maslov(…, integration_method='stationary_phase')` | ~1% RMS intensity on smooth refractive singlets | comparable to traced at sub=4 |
| Anamorphic / cylindrical / biconic elements | Any of the three, with `radius_y` etc. in prescription | same | same |

`apply_real_lens` is the default; reach for `apply_real_lens_traced`
when you need high accuracy (optimisation, tight tolerancing); reach
for `apply_real_lens_maslov` when the output plane sits on/past a
geometric image (caustic regime) or you need a differentiable model
for gradient-based design optimisation.

---

## Public API summary

```python
apply_real_lens(E_in, lens_prescription, wavelength, dx,
                bandlimit=True,
                fresnel=False,
                slant_correction=False,
                absorption=False,
                seidel_correction=False,
                seidel_poly_order=6)

apply_real_lens_traced(E_in, lens_prescription, wavelength, dx,
                       bandlimit=True,
                       ray_subsample=8,                 # 3.1.7: bumped 1 -> 8
                       n_workers=None,
                       preserve_input_phase=True,       # 3.1.2
                       tilt_aware_rays=False,           # 3.1.4
                       parallel_amp=True,               # 3.1.3
                       parallel_amp_min_free_gb=48.0,   # 3.1.3
                       newton_amp_mask_rel=1e-4,        # 3.1.3
                       newton_mask_dilate_coarse_px=2,  # 3.1.3
                       fast_analytic_phase=False,       # 3.1.7
                       newton_fit='polynomial',         # 3.1.7 'polynomial' (default) | 'spline'
                       newton_poly_order=6,             # 3.1.7 (used if newton_fit='polynomial')
                       use_gpu=False)                   # 3.1.7 GPU via CuPy (needs newton_fit='polynomial')

apply_real_lens_maslov(E_in, lens_prescription, wavelength, dx,   # 3.1.7
                       ray_field_samples=16, ray_pupil_samples=16,
                       poly_order=4,
                       n_v2=32,
                       output_subsample=1,
                       extract_linear_phase=True,
                       chunk_v2=64,
                       use_numexpr=None,
                       integration_method='quadrature', # | 'stationary_phase' | 'local_quadrature'
                       stationary_newton_iter=12,
                       stationary_newton_tol=1e-10,
                       local_n_samples=8,
                       local_window_sigma=3.0,
                       collimated_input=False,
                       normalize_output='power')        # 'power' | 'peak' | 'none' | scalar

# Sampling check (use before large runs)
check_opd_sampling(dx, wavelength, aperture, focal_length)

# Wavefront extraction with Nyquist warnings
wave_opd_1d(E, dx, wavelength, axis='x', aperture=None, dy=None,
            focal_length=None, f_ref=None)
wave_opd_2d(E, dx, wavelength, aperture=None, dy=None, f_ref=None,
            focal_length=None)

# Biconic / anamorphic surface support
surface_sag_biconic(X, Y, R_x, R_y=None, conic_x=0.0, conic_y=None, ...)
make_cylindrical(R_focus, d, glass, axis='x'|'y', aperture, name)
make_biconic(R1_x, R1_y, R2_x, R2_y, d, glass, ...)
```

Propagation module flags (for all wave-propagation functions):
- `USE_SCIPY_FFT = True` (default) — multithreaded scipy.fft
- `SCIPY_FFT_WORKERS = -1` (default) — use all cores
- `USE_PYFFTW = False` — optional 10-20 % faster but ~2× memory

---

## 1. Analytic `apply_real_lens` — thin-element phase-screen model

Core model: for each refracting surface, apply an instantaneous phase
screen at the vertex plane, then ASM-propagate through the glass to
the next vertex plane.

### 1a. Paraxial OPD formula (`slant_correction=False`, default)
```
phase_k(x, y) = -k0 · (n2 - n1) · sag_k(x, y)
```
Exact in the small-angle limit.  Empirically equal-or-better than
slant-correction on every validation case because ASM propagation
between surfaces encodes most of the obliquity physics naturally.

### 1b. Slant-corrected OPD (`slant_correction=True`, opt-in)
```
phase_k(x, y) = -k0 · [n2·sag / cos(θ_t) - n1·sag / cos(θ_i)]
```
Accurate at any angle of incidence via Snell's law.  Useful in a few
geometries (asymmetric meniscus, steep aspheres) but not a universal
improvement — can be worse than paraxial on well-corrected systems.

### 1c. Fresnel / absorption (`fresnel=True`, `absorption=True`)
S/p-averaged Fresnel amplitude transmission per surface and bulk
attenuation through each glass region.  Small cost; use when 4 % per
uncoated air-glass loss matters or when modelling IR-absorbing glass.

### 1d. Seidel correction (`seidel_correction=True`, opt-in, default OFF)
Post-hoc radially-symmetric polynomial OPD correction derived from a
41-ray geometric fan through the prescription.  Even-power
polynomial fit in normalised pupil coordinate `ρ = r/r_pupil`, order
6 by default.  Empirical results:

| Case | Default off | `seidel_correction=True` |
|---|---|---|
| Plano-convex R=50 BK7 | 21 nm RMS | 750 nm RMS (worse) |
| AC254-100-C at ap=10 mm | 20.5 µm RMS | **4.6 µm RMS (4.5× better)** |
| AC254-050-C at ap=10 mm | 41 µm RMS | **24 µm RMS (1.7× better)** |

Useful for cemented doublets and other multi-surface curved-interface
systems; on well-corrected singlets the correction injects
polynomial-fit artefacts that exceed the already-small baseline
residual.  A 50 nm RMS internal threshold is applied to skip the
correction when the fit magnitude is too small to be worthwhile —
still, results are case-dependent, which is why it's off by default.

For uniformly-high accuracy across all geometries, use
`apply_real_lens_traced` instead.

### 1e. Per-surface geometry (prescription dict keys, all optional)
- `clear_aperture` — per-surface mechanical aperture [m] (vignettes)
- `decenter` — `(dx, dy)` lateral offset [m]
- `tilt` — `(tx, ty)` small-angle tilt [rad]
- `form_error` — 2D ndarray additive sag perturbation [m]
- `stop_index` — which surface holds the aperture stop

### 1f. Anamorphic / biconic surfaces (optional keys)
- `radius_y` — y-axis radius of curvature [m] (presence triggers biconic)
- `conic_y` — y-axis conic constant
- `aspheric_coeffs_y` — y-axis aspheric coefficients

When `radius_y` is present the surface is treated as a biconic
(independent x and y curvatures), covering cylindrical, toroidal,
and freeform-biconic elements.  All downstream consumers (ray tracer,
Seidel, ABCD, OPD fans) handle biconic surfaces transparently.

Helpers: `make_cylindrical(R_focus, d, glass, axis='x'|'y')` and
`make_biconic(R1_x, R1_y, R2_x, R2_y, d, glass)`.

---

## 2. `apply_real_lens_traced` — hybrid wave/ray phase screen

For sub-nm accuracy on cemented doublets and multi-surface systems,
the analytic thin-element model has a hard ceiling because it treats
each glass region as a vertex-to-vertex uniform slab, while real rays
cross interior interfaces at `z = sag(h)`.  The traced variant
bypasses this by computing the exit-pupil phase from a geometric
ray trace.

### Algorithm
1. Launch rays on a regular square entrance grid.
2. Trace through prescription using the sequential ray tracer;
   record each ray's exit `(x, y, OPL)`.
3. Fit cubic `RectBivariateSpline`s for
   `x_out(x_in, y_in)`, `y_out(x_in, y_in)`, `OPL(x_in, y_in)`.
4. For each wave-grid exit pixel, Newton-invert
   `(x_out, y_out) = (X_w, Y_w)` to find the entrance that lands
   there; evaluate OPL spline at the converged entrance.
5. Return `|E_amp| · exp(i · k · OPL)` where `|E_amp|` comes from
   `apply_real_lens` itself (amplitude only — the full ASM-through-
   glass pipeline gives physically correct diffraction, vignetting,
   and Fresnel effects).  The phase from `apply_real_lens` is
   discarded and replaced by the geometrically exact ray-traced OPL.

### Advantages
- Smooth C² interpolation (no Delaunay edge artifacts).
- Handles caustic-free regions of fast lenses correctly.
- Agrees with geometric ray trace to sub-nm RMS when properly
  sampled.

### Critical sampling rule
Same Nyquist rule as for any OPD extraction:
```
dx ≤ λ · f / aperture
```
Violating this causes `np.unwrap` to lose cycles near the pupil edge,
giving catastrophically wrong OPD values there.  `check_opd_sampling`
helps you check before spending an hour on a mis-sampled run.

### Parallelism status
- `n_workers` accepted for API compatibility but currently ignored.
- Scipy's `RectBivariateSpline.ev` empirically does NOT release the
  GIL, so `ThreadPoolExecutor` gives no speedup on stable CPython.
- Real parallelism paths left as future work:
  - Free-threading Python 3.13t+ / 3.14t (current thread code would
    work as-is)
  - `ProcessPoolExecutor` with refactored module-level helper
  - CuPy GPU port (the spline + Newton is embarrassingly parallel)

### `ray_subsample` — main speed knob
OPL is a very smooth function of pupil position, so sub-sampling
the ray bundle and bilinearly interpolating is essentially free:

| `ray_subsample` | Speedup | Fidelity loss |
|---|---|---|
| 1 (default) | 1× | none |
| 2 | ~4× | < 0.5 nm |
| 4 | ~15× | < 1 nm |

`ray_subsample=4` is the recommended production setting.

### 2b. Robustness & performance refinements (3.1.3)

Practical fixes that surfaced during production N=32768 runs with
post-DOE inputs (multi-mode superposition of diffraction orders).
All default-on, all numerically equivalent to prior behaviour on
inputs the 3.1.2 code handled correctly.

**Multi-mode-safe tilt extraction**: `_sample_local_tilts` now
amplitude-weights a Gaussian low-pass (`smooth_sigma_px`, default 4)
on the extracted `(L, M)` tilt field before clipping to `max_sin`.
On smooth single-mode inputs (plane wave, Gaussian, MLA beamlets)
the smoothing is effectively a no-op — validated: plane-wave tilt
preserved to 0.995×, MLA tilts to 1.004×.  On multi-mode inputs
(post-DOE 144-order field) the aliased pixel-to-pixel gradients
average to the mean (near zero for a balanced order set), so the
ray launch naturally degenerates to the classical collimated case
where the 3.1.2 tilt-aware feature would have injected chaotic
directions and produced an all-NaN OPL map.  Verified on the
plane_09 plane from an actual TX Design 36 run: max `|L|` dropped
from 0.5 (clipping active, chaotic) to 0.22 (no clipping, spline
well-behaved) at `smooth_sigma_px=4`.  Optional
`multimode_diagnostic` dict parameter logs raw/smoothed RMS and the
smoothing ratio.

**Amplitude-masked Newton** (`newton_amp_mask_rel=1e-4`,
`newton_mask_dilate_coarse_px=2`): skip Newton evaluation on
coarse-grid pixels where the analytic amplitude is below
`newton_amp_mask_rel * amp.max()`, since the final assembly
`E_analytic * exp(1j*delta_phase)` is essentially zero there
anyway.  Mask dilated by 2 coarse pixels so bilinear interpolation
near boundaries has real data on both sides.  Self-disables if
the mask would include >95 % of pixels (overhead not worth it) or
<1 % (pathological).  Biggest benefit on post-DOE fields at
downstream lenses (L3/L4 in the TX Design 36 sim), where only the
diffraction-order pixels carry meaningful amplitude.

**Parallelised amp + amp(pw) pass** (`parallel_amp=True`,
`parallel_amp_min_free_gb=48.0`): the two internal
`apply_real_lens` calls that build the amplitude envelope and
extract the analytic-lens phase are data-independent and now run
concurrently on a 2-worker `ThreadPoolExecutor`.  FFT execution
inside each thread serialises through the plan-cache lock in
`propagation._fft2`, but the non-FFT work (sag, phase screens,
numexpr-fused multiplies, glass-interval setup) overlaps.
Measured ~1.56x on the combined amp step at N=4096.  Auto-disables
when available RAM is below `parallel_amp_min_free_gb` since the
per-call transient peak at N=32768 approaches 80 GB per thread.

**Numexpr-fused phase-screen multiply in `apply_real_lens`**: the
`E * np.exp(-1j * k0 * opd)` per-surface step is now fused through
`ne.evaluate(..., out=E)` when numexpr is available and the field
is at least 2²⁰ elements.  Eliminates three complex128 NxN temporaries
(~50 GB of transient allocations at N=32768) and multi-threads.
Measured 1.66x on `apply_real_lens` at N=4096; numerically identical
to the numpy path (`max |diff| = 0`).  Numpy fallback preserved when
numexpr is not installed or the field is too small.

**Decenter-aliased entrance grids**: for surfaces with
`decenter == (0, 0)` (the common case), `Xs`, `Ys`, and `h_sq` are
aliased to the axis-centred grids instead of materialising three
fresh float64 NxN arrays per surface (~24 GB at N=32768).  Safe
because downstream code reads without mutating.

**Complex-dtype preservation end-to-end**: `apply_real_lens` and
`apply_real_lens_traced` no longer force `complex128` internally.
The caller's `E_in.dtype` is preserved throughout; complex64 mode
halves memory and roughly doubles FFT/phase-screen throughput,
with the `angular_spectrum_propagate` kernel's phase argument
reduced modulo 2 pi in float64 before casting to float32 — so the
~4e5-rad phase magnitudes typical of large air gaps don't degrade
into the ~0.02-rad float32 precision floor.  Validated
end-to-end: N=512 two-lens chain shows 4.9e-7 relative error between
complex128 and complex64 runs, all within the expected complex64
FFT round-off floor, and power conservation matches to 5.4e-7.

---

## 2b. `apply_real_lens_maslov` — phase-space / canonical-map propagator (3.1.7)

Third lens propagator, complementing the analytic (phase-screen) and
traced (Newton-on-spline) paths.  The method follows the
Kravtsov-Orlov / Forbes-Alonso phase-space formulation: instead of
representing the lens's effect in object-space coordinates (`s1`) or
pupil coordinates (`s1 -> s2` map), it fits a smooth polynomial to
the symplectic canonical map `(s2, v2) -> s1` and uses that chart to
evaluate a diffraction integral that is single-valued across
geometric-image caustics.

### Algorithm

1. Trace rays on a Chebyshev-node (`h`, `p`) grid between the entrance
   and exit planes.  Default `12 x 12 x 12 x 12 = 20736` rays before
   pupil-disc masking.
2. Fit a 4-variable Chebyshev tensor-product polynomial (total degree
   <= `poly_order`, default 4 -> 70 coefficients) to the back-map
   `s1x(s2, v2)`, `s1y(s2, v2)`, and `OPD(s2, v2)`.  On Design 51 L1
   with the paraxial-EFL NA scaling the fit residual is
   `~10^-8 waves RMS` — essentially exact.
3. Optional linear-phase extraction for diffractive surfaces at
   nonzero orders (`extract_linear_phase=True`, default on).  Pre-fits
   a 5-parameter linear plane in `(s2, v2)`, stores it separately, and
   drops it from the diffraction integrator.  The dropped linear
   piece is a rigid output translation already absorbed by
   chief-ray grid centering, and keeping it inside the integrator
   would alias catastrophically on nonzero grating orders.  No-op for
   zeroth-order / refractive-only prescriptions.
4. Evaluate the Maslov integral by one of three methods:

   - `'stationary_phase'` (recommended for single-lens /
     collimated-input / per-pixel single-ray regime): Newton-iterate
     to locate `v2*` where `grad_v2 OPD = 0`; evaluate the saddle-
     point formula `|det J_s1v2| * E_obj(s1*) * exp(2 pi i OPD*) *
     exp(i pi sig/4) / sqrt(|det H_v2v2|)`.  No v2 quadrature, no
     critical-sampling constraint, analytically differentiable.
   - `'quadrature'` (extended-source regime): Tukey-windowed uniform
     Riemann sum over a `n_v2 x n_v2` v2 grid.  Requires source waist
     `w_s >= D_s1 / n_v2` (quadrature-validity bound); fails silently
     below that (outer PSF rings vanish).
   - `'local_quadrature'`: Hessian-oriented uniform sampling in a
     small window around `v2*`.  Captures asymptotic corrections
     beyond leading stationary phase.  Extended-source regime only.

### Speedups

All four of the OPDGPU `maslov_zemax_merit.tex` paper's acceleration
tricks and one CPU-specific one are implemented:

1. **Precompute s2-basis** `G[s2, j] = T_{k1_j}(u_s2x) T_{k2_j}(u_s2y)`
   once over the output grid.  Each per-v2-sample polynomial evaluation
   becomes a `G @ h` matrix-vector product instead of rebuilding the
   Vandermonde over the full grid.
2. **Batched BLAS GEMM** across chunks of v2 samples (default
   `chunk_v2=64`).  7 quantities per chunk reduce to 7 GEMM calls
   rather than `7 * n_v2^2` matvecs.
3. **Vectorised weight-vector assembly** via NumPy fancy-indexing of
   the multi-index arrays, eliminating a Python-level `for j, (k1, ..)
   in enumerate(mi)` loop.
4. **numexpr fused integrand** for
   `Eobj * exp(2 pi i opd) * |det J| * weight`, falling back to plain
   NumPy when numexpr isn't importable.  `use_numexpr=None` (auto)
   picks numexpr when available.
5. **Cubic amplitude-phase upsampling** when `output_subsample > 1`:
   interpolates `|E|` and `unwrap(angle(E))` independently (order-3
   zoom on each) rather than bilinear on the complex field directly,
   which would alias the rapidly-varying phase.

On a 512² output grid at N_v2 = 24, the combined speedup is ~30×
over the naive per-sample Python loop (514 s -> 17 s on CPU).

### Amplitude normalisation

The raw stationary-phase formula omits the Huygens-Fresnel global
prefactor (~`-i k / (2 pi z)`) and therefore the absolute amplitude is
miscalibrated by a constant.  `normalize_output='power'` (default)
rescales so total `|E|^2` matches the input (lossless lens limit);
`'peak'` matches max `|E|`; `'none'` returns the raw saddle-point
output; passing a scalar multiplies by it.

### Tested regimes (3.1.7 initial release)

- Design 51 L1 (singlet, F/52, collimated Gaussian input,
  N=1024):  stationary_phase error = 0.17 % RMS intensity at
  `output_subsample=1`, 0.4 % at sub=2 (8.5 s), 1.2 % at sub=4
  (2.2 s).  At sub=4 Maslov is faster than
  `apply_real_lens_traced`.
- 3-source collimated Gaussian (same lens):  stationary_phase =
  1.3 % RMS vs traced.  Quadrature = 12.4 % (partial convergence
  with extended sources; still not as good as SP for this
  geometry).
- Coarse-grid robustness: at `dx` 2.4× above
  `apply_real_lens_traced`'s nominal critical-sampling rule
  (`dx > lambda f / aperture`), both methods give identical output
  on Gaussian-input geometries (the critical rule is only a hard
  wall on aperture-filling wavefronts).

### Not yet validated (3.1.7)

- Other Design 51 lenses (L2, L3, L4) — only L1 exercised in detail.
- Diverging / converging input fields — only collimated tested.
- Use inside a forward-path simulation (interactions with
  band-limited ASM, COORDBRK masks, prescription aperture handling).
- Caustic output-plane regime (the use case Maslov is theoretically
  designed for) — not exercised.
- Quadrature on true multi-emitter extended-source geometry.

Treat Maslov as a research-grade third option for now; use
`apply_real_lens_traced` as the reference for production runs.

---

## 2c. 3.1.7 `apply_real_lens_traced` speedup kwargs

### `fast_analytic_phase` — skip the ASM reference pass

The `preserve_input_phase` path runs `apply_real_lens` twice: once on
`E_in` (to get the amplitude envelope including diffraction,
vignetting, and Fresnel effects) and once on a unit plane wave (to
extract the lens-only analytic phase for the
`delta_phase = k0 * opl_traced - phase_analytic_lens` subtraction).
When `parallel_amp=False` these run sequentially and each takes
~300 s on an N=32768 L1 call.

`fast_analytic_phase=True` replaces the plane-wave ASM pass with a
direct analytic evaluation of the geometric phase:

```
phase_analytic(x, y) = -k * sum_i (n_{i+1} - n_i) * sag_i(x, y)
                        + k * sum_i n_mid_i * t_i     (piston)
```

which is a per-pixel sum of sag phase screens with no ASM-through-
glass FFTs.  The omitted physics is the Fresnel-like diffractive
correction accumulated by ASM through each glass interval — magnitude
scales as `t * k_perp^2 / (2 k)` and is well under 10 nm OPL for
Design 51 lenses (F/7 and slower).

Empirical: on Design 51 L1 (N=1024, `parallel_amp=False`),
`fast_analytic_phase=True` produces 0.000 % intensity change at the
exit plane, 9.1 nm RMS phase difference in the pupil, 24 % wall-time
savings.  The 0 % intensity change is because exit-plane intensity
doesn't carry phase information; downstream propagation does, where
9 nm RMS corresponds to Strehl ~0.998 (λ/145 RMS wavefront error).

Default `False` — physical accuracy preserved by default; flip on
explicitly when the speed matters and you've validated the phase
error is acceptable for your downstream use.

### `newton_fit` — spline vs polynomial

`newton_fit='polynomial'` is the **3.1.7 default**.  `newton_fit='spline'`
is the pre-3.1.7 behaviour (SciPy `RectBivariateSpline`) and remains
available for prescriptions with non-polynomial surface features.

The polynomial path uses a new `_Cheb2DEvaluator` class: a 2-D
Chebyshev tensor-product polynomial fit with exactly the `.ev(x, y,
dx=0/1, dy=0/1)` API used by the Newton loop (so the inversion code
is untouched).  For smooth refractive lens prescriptions this is
strictly at-or-above spline accuracy — all Seidel and higher-order
aberrations are polynomials in the entrance coordinates by
definition, and Chebyshev fits capture them exactly at their native
degree with no cubic-truncation error.  Analytic derivatives are
bit-exact from the polynomial recurrence; spline derivatives carry a
small interpolation error.

Default order 6 (tunable via `newton_poly_order`) captures up to the
8th Seidel (spherical, coma, astigmatism, distortion, and their
higher-order variants).  For surfaces with sharp non-polynomial
features (kinoforms, metasurfaces, freeforms with very high-order
Zernike content), stay on `'spline'`.

Empirical: on Design 51 L1 (N=2048, `parallel_amp=False`),
`newton_fit='polynomial'` produces 0 % intensity change and ~6 %
wall-time savings.  Bigger wins on larger grids where Newton is a
larger fraction of total runtime.

### Default `ray_subsample` = 8

Previously `ray_subsample=1`.  The internal safety floor
(`min_coarse_samples_per_aperture=32`, giving ~85 nm RMS phase error)
is already enforced via `on_undersample='error'` — dropping below
the floor raises a clear error with a recommended safe value.  At
N=32768 on Design 51 lenses, `ray_subsample=8` gives ~2000-3800
samples across each aperture — far above the floor — with ~0.005-
0.02 nm RMS projected phase error.  Users on small grids (N <~ 512)
get a clean error pointing them to `ray_subsample=4` or smaller.

The Newton inversion and ray-trace steps scale as `1/ray_subsample^2`,
so bumping the default from 1 to 8 gives up to 64× speedup on those
phases (they are ~30-40 % of total runtime on large-grid runs).

---

## 3. Exit-vertex OPL correction (critical bug — now fixed)

The `trace()` function leaves rays at the **sag** of the last
surface (`z = sag(h)`), not at the flat exit **vertex** plane
(`z = 0`).  For lenses with curved rear surfaces the on-axis ray
ends at `z = 0` while off-axis rays end at `z = sag(h) ≠ 0`.
Computing `OPL(off-axis) − OPL(on-axis)` from these mis-aligned
z-positions injected a systematic defocus error equal to
`n_exit · sag(h)` — enough to shift the implied focal length by
**43 %** for the AC254-100-C doublet (BFL 80 mm, measured 56 mm).

**Fix** (in `apply_real_lens_traced` and in the validation's
`compute_geometric_opd`): after `trace()`, transfer each ray from
the last sag to the vertex plane in the exit medium using the
**signed** parametric distance:

```python
t_to_vertex = -final.z / final.N
final.opd += n_exit * t_to_vertex   # signed, NOT abs()
final.x  += final.L * t_to_vertex
final.y  += final.M * t_to_vertex
final.z   = 0
```

The sign is critical: for **concave** rear surfaces (`sag < 0`,
ray behind vertex) the ray goes forward (`t > 0`) and OPL is
added.  For **convex** rear surfaces (`sag > 0`, ray ahead of
vertex) the ray goes backward (`t < 0`) and OPL is subtracted.
Using `abs(t)` forces the wrong sign for convex exits, producing
200,000× worse residuals on negative meniscus lenses.

| Metric | Before fix | After fix |
|---|---|---|
| Doublet focus error (AC254-100-C) | 10 mm | **0.000 mm** |
| Singlet focus error (plano-convex) | 0 mm (unaffected) | 0 mm |
| Traced OPD RMS (AC254-100-C, 15 mm ap) | 0.77 nm | **0.54 nm** |
| Traced OPD RMS (negative meniscus) | 33,742 nm | **0.17 nm** |
| Traced OPD RMS (equi-convex) | 0.81 nm | **0.41 nm** |
| Traced OPD RMS (biconcave) | 1.35 nm | **0.44 nm** |

Lenses with flat rear surfaces (plano-convex) were never affected.
The validation suite didn't catch the original bug because both the
"traced" and "geometric truth" used the same ray tracer — the
exit-vertex error cancelled in their difference.

---

## 4. Raytrace OPL bookkeeping fix (critical bug — now fixed)

The single biggest accuracy improvement in this round came from
fixing a subtle bug in `raytrace._intersect_surface` that silently
produced wrong OPL for any off-axis ray through a curved surface.

**Bug**: `_transfer` accumulated OPL by vertex-to-vertex axial
distance, while `_intersect_surface` then moved the ray from the
vertex plane to the actual sag intersection **without accumulating
OPL for that motion**.  Axial rays had `sag = 0` and the bug was
invisible; off-axis rays through curved surfaces could miss
millimetres of OPL.

**Fix**: `_intersect_surface` now takes `n_medium` and adds
`n_medium · |parametric_path|` to `opd`.  One-line change, but it
cut singlet wave-vs-geom residuals by 17×–130× and doublet
residuals by 5×.  Every consumer of `raytrace.trace` benefits
automatically.

---

## 5. Sampling and Nyquist tooling (new)

`check_opd_sampling(dx, wavelength, aperture, focal_length)` reports
the Nyquist margin (`dx_max / dx`) for clean OPD extraction from a
converging wavefront, plus concrete recommendations when marginal.

`wave_opd_1d` and `wave_opd_2d` accept `focal_length` and emit a
`RuntimeWarning` when sampling is marginal, and accept `f_ref` to
subtract a reference sphere before unwrap (for users who want
coarser grids).  All OPD extraction in the library now protects
against the Nyquist-edge failure mode.

---

## 6. SciPy FFT as default

`USE_SCIPY_FFT = True`, `SCIPY_FFT_WORKERS = -1` are defaults.  All
wave-propagation functions now run multithreaded by default with
zero extra memory (pocketfft shares buffers across threads).
Typical 2–4× speedup on modern multicore.

---

## 7. What doesn't work (explored and documented)

Two improvements were prototyped during this work but did not
deliver their promised accuracy; both are documented as **do not use**
and explained here for future reference:

### Sub-slicing BPM through the interface region
Single-reference-medium BPM requires sub-wavelength axial slabs to
handle sharp step-discontinuous air-glass interfaces.  Realistic
interface thicknesses (~0.2 mm) would need ~1000s of slabs, making
the method slower than `apply_real_lens_traced` for worse accuracy.
**Code was reverted.**  A proper full WPM implementation would work
but is ~1 week of work; not currently prioritised.

### Simple ThreadPoolExecutor on scatter/spline interpolators
`scipy.interpolate.LinearNDInterpolator.__call__` and
`RectBivariateSpline.ev()` both hold the GIL during evaluation on
stable CPython.  Thread pools serialise and actually slow things
down.  `ProcessPoolExecutor` works in principle but the Newton
helper would need to be refactored out of its closure to be
picklable.  Marked as future work.

---

## 8. Validation methodology

`validation/real_lens_opd/run_validation.py` runs three methods for
a suite of 21+ reference lenses:

1. `apply_real_lens(slant_correction=False)` — paraxial thin-element
2. `apply_real_lens(slant_correction=True)` — slant-corrected
3. `apply_real_lens_traced(ray_subsample=4)` — per-pixel ray-traced

Each compared against the geometric ray-traced OPL at the exit
pupil.  Report tables show RMS of the high-order aberration residual
(piston + tilt + defocus removed), in nanometres.  See
`validation/real_lens_opd/results/report.md` for current numbers,
and `zemax_prescriptions/` for Zemax-compatible LDE + `.zmx` files
for each case so results can be cross-verified in OpticStudio.

---

## 9. Backward compatibility

- `apply_real_lens(..., slant_correction=False, seidel_correction=False)`
  (all defaults) reproduces the original paraxial phase-screen model
  bit-for-bit when all other optional flags are off.
- All existing call sites continue to work; no prescription changes
  needed; new optional keys (`radius_y`, `conic_y`, etc.) are
  skipped when absent.
- New functions (`apply_real_lens_traced`, `check_opd_sampling`,
  `make_cylindrical`, `make_biconic`, `surface_sag_biconic`) are
  additive.
- `USE_SCIPY_FFT` default change is purely a performance
  improvement; results are bit-identical to NumPy's FFT for any
  supported grid.

---

## 10. Recommended call patterns

```python
# Most lenses, fastest mode
E = op.apply_real_lens(E_in, prescription, wavelength, dx)

# Cemented doublets / multi-surface / critical accuracy
E = op.apply_real_lens_traced(E_in, prescription, wavelength, dx,
                               ray_subsample=4)

# Intensity-accurate: Fresnel + absorption + slant
E = op.apply_real_lens(E_in, prescription, wavelength, dx,
                        fresnel=True, absorption=True,
                        slant_correction=True)

# Anamorphic cylindrical element
pres = op.make_cylindrical(R_focus=50e-3, d=3e-3, glass='N-BK7', axis='x')
E = op.apply_real_lens(E_in, pres, wavelength, dx)

# Pre-flight sampling check
from lumenairy.raytrace import (
    surfaces_from_prescription, system_abcd)
_, efl, bfl, _ = system_abcd(
    surfaces_from_prescription(prescription), wavelength)
samp = op.check_opd_sampling(dx, wavelength, aperture, bfl)
if not samp['ok']:
    raise ValueError(f'dx too coarse: {samp["recommendations"]}')
```

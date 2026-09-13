# Subsystem contract -- the `apply_real_lens` family

**Status:** living document.  This is the *current* contract, invariants, measured
accuracy envelopes and known limits of the real-lens propagators.  It is the first of
the per-subsystem documents recommended by
[`AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11`](../audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md)
section 14 (V7) -- audit *reports* are a historical record and are archived; this file
is updated by every round that touches the family and is the thing to read first.

Last updated: v5.46 (audit 2026-09-11 remediation).  Every number below was measured;
the script or test that produced it is named.  Numbers without a citation do not belong
in this file.

---

## 1. The family at a glance

| entry point | model | use it when | measured envelope |
|---|---|---|---|
| `apply_real_lens` | analytic per-surface phase screen + in-glass propagation | low NA, single-valued, screen-scale departures | 0.03-1.2 nm rms exit OPL against the wave model at converged sampling (section 6) |
| `apply_real_lens_traced` | real ray trace -> polynomial / spline / Newton inverse map -> wave field | anything the screen cannot do: real aberration, finite NA, form error, vignetting | exit-vertex transverse error 1.7e-12 um vs an independent vector-Snell trace |
| `..._traced(caustic='wave')` | traced field at the exit vertex + one band-limited ASM leg | **any plane where the ray map is multi-valued** -- folds, cusps, the axial focus | `P_out/P_in` 0.999999; peak and EE(5/10/25 um) match an independent `apply_real_lens`+ASM oracle to 4 digits |
| `..._traced(caustic='multibranch')` | branch enumeration + KMAH index | a fold you want resolved branch by branch | 0.0003 rad rms against the single-valued path on a curved rear (was 1.7984 rad) |
| `..._traced(caustic='uniform')` | Ludwig / Airy uniform asymptotics across one fold | rotationally-symmetric single fold | same exit-vertex fix; dark fill restricted to `r_c + 20 l_airy` (30x) |
| `apply_real_lens_maslov` | Maslov canonical-chart integral | through-focus, multivalued, moderate NA | chart on the exit vertex (was 31 waves of defocus on an f/19 singlet) |
| `apply_real_lens_gbd` | Gaussian beamlet decomposition | smooth, well-sampled pupils | Gouy/Collins phase no longer conjugated |
| `apply_real_lens_fga` | Frozen Gaussian approximation | caustics; exact on free space, correct Gouy phase | see the convergence caveat in section 7 |
| `apply_real_lens_universal` / `_auto` | router over the above | you do not want to choose | router is now frame-invariant (section 5) |
| `prepare_real_lens*` / `Prepared*Lens` | one-time build, many fields | sweeps, optimiser inner loops | -- |

JAX twins: `apply_real_lens_traced_jax`, `apply_real_lens_maslov_jax`.  **Both require
`jax.config.update('jax_enable_x64', True)` and refuse loudly without it** (audit S7):
with x64 off they returned complex64 for a complex128 input and the phase screen
differed from the float64 one by 1.71e-05 waves rms.

---

## 2. The one invariant that binds the family

`raytrace.trace()` leaves every ray at its intersection with the **last surface**, i.e.
at `z = sag(rho)` -- *not* on the exit-vertex plane.  Reading `image_rays.opd / .x / .y`
without correcting for that is the single most productive bug class in this subsystem:
the audit found six hand-written corrections and five consumers that had forgotten it
(audit section 15.1).

**There is now exactly one way to read exit state:**

```python
import numpy as np
import lumenairy as la
from lumenairy import raytrace

pres = la.make_singlet(R1=0.05, R2=-0.05, d=3e-3, glass='N-BK7', aperture=8e-3)
surfaces = raytrace.surfaces_from_prescription(pres)
bundle = raytrace.make_fan(semi_aperture=3.5e-3, n_rays=9, wavelength=633e-9)

res = raytrace.trace(bundle, surfaces, wavelength=633e-9)
exit_rays = res.at_exit_vertex()   # or exit_vertex_transfer(bundle, n_exit)

print('max |z| on the last surface : %.6f mm'
      % (1e3 * np.nanmax(np.abs(res.image_rays.z))))     # 0.118077 mm of sag
print('max |z| after at_exit_vertex: %.6f mm'
      % (1e3 * np.nanmax(np.abs(exit_rays.z))))          # 0.000000 mm
```

`vertex_plane_transfer_t` is the shared kernel; `exit_vertex_transfer_jax` is the JAX
twin; `resolve_exit_index` picks `n_exit`.  All four are public
(`lumenairy.__all__`).  A grazing ray (`abs(N) <= EXIT_VERTEX_GRAZING_TOL = 1e-30`,
finite) **dies** with `RAY_MISSED_SURFACE` rather than being teleported -- the previous
hand-written copies produced `t = -z/1e-30`.

Measured on a plano-convex with `R2 = -25 mm` over a 20 mm aperture against an
independent vector-Snell trace (`repro/TR-SIBLINGS/repro_vertex.py`): transverse error
**476.8 um -> 1.7e-12 um**, OPD **3 501 waves -> 4.4e-12 waves**.

Pinned by `tests/unit/test_audit2609_a1_exit_vertex.py`,
`tests/unit/test_audit2609_a3_caustic_siblings.py`.

> **Fixture rule.**  Every ray-based propagator test needs a fixture with a **curved
> last surface**.  The entire multibranch / uniform / Maslov / Seidel corpus used
> plano-rear singlets, where `sag == 0` and this whole class is invisible -- which is why
> the pre-fix code passed 8/8.

---

## 3. `apply_real_lens` -- the analytic model

Per surface: a phase screen `-k0 (n2 - n1) * z_surface`, then propagation through the
gap in the current medium.  28 keyword arguments; the ones that change physics:

### 3.1 `surface_model`

| value | what it does now |
|---|---|
| `'thin'` (default) | the per-surface screen at the vertex plane |
| `'displaced'` | screen + a transverse remap of the field, `displaced_mode={'screen','remap','split'}` |
| `'tangent_facet'` | route-3 momentum accumulator; the transverse walk is referenced away |
| `'tangent_facet_remap'` | the same, with the walk applied as a remap |

`'displaced'` used to resample `abs(E)` and discard the input field's phase (L3); it now
demodulates, resamples the complex residual once, and re-modulates -- for a real
non-negative input with `conjugate=None` this reduces to the old amplitude sample
exactly.  The remap is smooth-residual transport: the closer `conjugate` is to the
field's actual congruence, the better.  A wildly mismatched pair (a 20-wave-per-pixel
residual) is aliased, not refused.  The 2-D remap carries **no in-glass diffraction** --
documented in `displaced_mode`.

The 2-D remap's transverse resolution is the LAUNCH pitch `2 r_aperture / (displaced_n_side - 1)`,
not `dx`: it is a geometric transfer, so input structure finer than that pitch is smoothed to
the lattice, and the call warns (naming the `displaced_n_side` that would clear it) whenever the
launch pitch is coarser than twice the field pitch.  Default 257 rays a side; the inversion of the
launch->exit map is structured (Newton on the lattice's bilinear interpolant), so the exit field
carries no triangulation-hull holes inside the illuminated pupil.

### 3.2 Per-surface modifiers

* **`fresnel=True`** applies the POWER transmittance
  `T = (n2 cos theta_t)/(n1 cos theta_i) |t|^2`, not `|t|^2` (L13, CONVENTIONS section 7).
  Measured against the closed form `4 n1 n2/(n1+n2)^2` to 1.1e-16 on AIR->N-BK7 (0.958057),
  AIR->N-SF11 (0.921480), N-BK7->N-SF11 (0.993599) and an absorbing `n = 1.8 + 1e-3 i`
  (0.918367).  An element that starts and ends in air at normal incidence is unchanged.
  `theta_i` is the AOI of an **axial** ray -- no path has both a true local AOI and
  Fresnel, and the docstring says so.
* **`slant_correction=True`** implements the axial-translation identity
  `n2 cos(theta_i - theta_t) - n1`.  As a coefficient it is **287x (R = 20 mm) to 4276x
  (R = 100 mm)** closer to the exact eikonal than the paraxial screen; through the whole
  wave model 400x to 5215x on the same four faces.  It is **exact only for a collimated
  input** -- on a thick symmetric element the bundle at the second surface is already
  converging, and on the f/5 hammer fixture the old (wrong) form scored closer by
  cancellation.
* **`seidel_correction=True`** fits a rho^4-and-up residual from a 41-ray fan launched
  across `+-0.999 * r_pupil`, and **clamps** the fitted screen at the last radius the fan
  lands at (`rho_fit = max|x_model|/r_pupil`) rather than extrapolating it.  Measured exit
  OPD rms: 8 mm cemented doublet **173.6 -> 1.05 nm (165x)**, 4 mm doublet 10.9 -> 2.6 nm
  (4.1x), f/2 biconvex 402.6 -> 1.50 nm (268x), immersed rear 2.400 -> 0.023 nm (105x).
  The gate SKIPS when the fitted rho^4+ rms is below 5 nm (plano-convex: 1.349 nm).
  Refused together with `slant_correction=True` (L20) -- both replace the same coefficient.
* **`absorption=True`** attenuates along the local ray column, not the axial gap
  (apodisation error 1.52e-2 -> 4.2e-4).
* **`surface_frame=True`** treats the surface as a rigid body: the field grid is mapped
  through the inverse rigid-body transform and the phase is the **rotated surface's
  field-frame height** `z_f = R_zx x_s + R_zy y_s + R_zz g(x_s, y_s)` -- not the bare
  surface-frame sag.  Against the exact rotated-sphere geometry (R = 50 mm, +-2 mm):
  1.7 / 10.3 / 80.1 nm at 1 / 5 / 20 mrad, from 2.002 / 10.011 / 40.077 um pre-fix.  It is
  **not** the more accurate branch for a simple tilt (the default field-frame ramp reads
  1.64 / 9.10 / 53.4 nm on the same fixtures); it is the one that means *rigid body*.
* **`stop_index`** out of range now RAISES (it used to remove all aperture clipping);
  `-1` means the last surface.

### 3.3 The `tilt` key

`tilt = (t0, t1)` **IS** the linear sag ramp `t0*(x - dcx) + t1*(y - dcy)` added to the
surface's z-departure, with the correspondingly tilted normal.  As a right-hand rotation
pair that is `theta_x = t1`, `theta_y = -t0`.  Five consumers read it this way -- the
field-frame ramp, the surface-frame branch, `_disp_surface_z_grad`,
`raytrace/surface.py::_field_frame_sag_and_grad` and the lumenairy-free
`geom_spot_decenter_oracle` -- and the cross-model agreement is pinned by
`tests/unit/test_niche_p9_decenter_tilt.py`.  Measured end to end: `tilt = (5 mrad, 0)`
gives ray `(L, M) = (-2.575435e-03, 0)` and both wave branches -2.574742e-03 /
-2.574699e-03 against the exact thin prism 2.575446e-03; branch to branch **4.3e-08 rad**.
See CONVENTIONS section 7.

---

## 4. `apply_real_lens_traced` -- the ray model

48 keyword arguments.  The physics-affecting ones and their current behaviour:

* **`inversion_method={'newton','fit','backward_trace'}`**, **`newton_fit`**,
  **`newton_poly_order`**, **`decentred_fit_poly_order`** -- how the exit ray map is
  inverted onto the output grid.  `newton_fit='spline'` with a vignetted ray used to
  return an identically-zero field (T3); it does not now.
* **`amplitude_model={'screen','ray_density'}`** and **`caustic={None,'single',
  'multibranch','uniform','wave'}`**.
* **`form_error` is honoured** (T5).  The traced model used to cancel it out of the
  answer.  `caustic='multibranch'`/`'uniform'` REFUSE a `form_error` prescription: they
  are a pure ray construction with no analytic leg to re-apply the screen onto.
* **Vignetting is in the answer**: a per-surface `semi_diameter` / `clear_aperture` that
  kills rays now reduces the delivered power instead of being fitted over.  Nothing moves
  when no ray dies.  The CuPy branch (`use_gpu=True`, polynomial only) keeps the
  historical extrapolating behaviour.
* **The exit-NA undersample guard** prices the exit medium: it needs
  `n_exit sin(theta)` resolved, so a prescription ending in glass is asked for a grid
  `n_exit` times finer than before (measured 12.216 um advised against 6.957 um
  required -- 1.76x too coarse).  Air-ending prescriptions are bit-identical.
  `on_undersample='silent'` acknowledges it.
* **`carrier=<ndarray>`** is sampled and differentiated with its own per-axis pitch
  (used to be nearest-neighbour).
* **`fast_analytic_phase=True`** works (it raised `AttributeError` on every refracting
  prescription).  Measured cost of the approximation: ~7 nm rms OPL **per mm of glass**,
  not the flat 10 nm the UI tooltip claims.
* **`on_noncollimated='delegate'`** announces the model swap instead of doing it
  silently.

### 4.1 Choosing a caustic mode

`caustic='wave'` is the recommended mode for new work at any plane where the ray map is
multi-valued: it takes the single-valued traced field at the exit vertex -- where
geometric optics is exact -- and finishes with one band-limited ASM leg.  No branch
enumeration, no KMAH index, no `1/sqrt|J|`, no Ludwig swap, and it carries the
exponentially-decaying dark-side tail the branch sum drops to exactly zero.  This is the
commercial-POP pattern (Zemax POP / CODE V BSP).

Measured (f ~ 25 mm biconvex, N = 512, dx = 4 um, w0 = 0.30 mm, BFL = 24.4833 mm): at
`d = 0` it is **bit-identical** to the ordinary traced call (`np.array_equal` True); at
the paraxial focus `P_out/P_in = 0.999999`, peak 8.5781, EE(5/10/25 um) =
0.0408 / 0.1609 / 0.6348 -- matching an independent `apply_real_lens` + ASM oracle to four
digits, **where `caustic='multibranch'` returns nothing at all**.  Away from any caustic
it tracks the multibranch to 1.7 % over the bright core.

It is **not** the default for `output_plane_distance != 0`: that would move every pinned
multibranch number.

---

## 5. Routing (`apply_real_lens_universal`, `apply_real_lens_auto`)

The router is now **frame-invariant** (S10).  Both of its discriminators were blind to a
global tilt: the caustic-zone scorer measured each exit ray's crossing with the optical
*axis* (a tilt moves the focus off axis by ~`f theta`), and the collimation escape hatch
asked `_carrier_residual_rms` on the tilted field, which returns exactly the tilt
magnitude for a pure tilt.  They now score against the **chief ray** and the **de-tilted**
field respectively.

Measured on an f = 1.2 mm N-BK7 biconvex: the caustic zone's centre moved **+534.1 %**
under a 0.05 rad tilt before the fix and **-0.25 %** after, against a ray oracle's own
-0.27 %.  Routing: tilt 0.02 `'traced' -> 'fga'`; tilt 0.05 `'phase_screen' -> 'fga'`.
The four regimes the audit found correctly routed are bit-identical.

**What this does not claim:** the router is frame-invariant, not more accurate.  On that
fixture `phase_screen` and `fga` disagree by 3.9 um in centroid and a factor 3.9 in
intensity-rms width -- *by the same factor at tilt 0* -- which is the members' own
accuracy question (section 7), not a tilt artefact.

---

## 6. Measurement recipe (read this before re-measuring anything above)

Exit-OPL measurements on an apertured fixture need

```
dx ~ 1.45 * aperture / 2048
```

**not** the `0.3 lambda / NA` a carrier-Nyquist rule gives.  At N = 512 a meniscus reads
`model - wave = 557.8 nm` and an air-spaced doublet 949.2 nm of **pure hard-edge aliasing
through the in-glass ASM**, converging to 0.03 nm by N = 2048 (the `|E|` ripple over the
same window goes 0.640-1.130 -> 1.052-1.057).  Any re-measurement must check that
convergence first; two false alarms in VERIFY-A2 came from skipping it.

```python
import lumenairy as la

aperture = 4e-3                       # clear aperture [m]
N = 2048
dx = 1.45 * aperture / N              # the converged pitch for an exit-OPL measurement
print(f"N = {N}, dx = {dx * 1e6:.3f} um, window = {N * dx * 1e3:.2f} mm")
print("alias-free RS distance:",
      la.rs_alias_free_distance(N, dx, 633e-9), "m")
```

At that sampling, `model - wave` (the analytic model's exit OPL against the wave model's
own unwrapped exit phase, no ray oracle in the loop) reads:

| fixture | model - wave | ray - wave |
|---|---:|---:|
| plano-convex 4 mm | 0.037 nm | 0.848 nm |
| meniscus R = 20/25, 4 mm | 0.032 nm | 5.213 nm |
| air-spaced doublet 6 mm (4 surfaces) | 0.036 nm | 29.828 nm |
| f/2 biconvex R = +-8.24 mm, 4 mm | 0.635 nm | 807.7 nm |
| cemented doublet 8 mm | 1.222 nm | 173.395 nm |

Through-focus comparisons need the scan refined in two passes around the marginal-ray
crossing with a step <= 0.1 DOF; a single coarse scan is not enough to call a peak shift.

---

## 7. Known limits

1. **Seidel on an under-filled pupil.**  `seidel_correction=True` normalises its radial
   screen to `r_pupil`.  A fast, thick element walks its rays inward, so the outer pupil
   carries only the diffractive tail of the geometric field -- and a radial screen must
   put *something* on that annulus.  Measured on an f/2 biconvex (N = 4096,
   dx = 0.708 um): exit wavefront **268x better** inside `|x| <= 0.85 r_pupil`, and the
   focal peak **~37 % lower** with best focus 8.2 DOF away.  Central-row energy is
   94.59 % inside `rho <= 0.85`, 5.14 % in 0.85-0.90, 0.21 % beyond.  `model - wave` over
   the FULL pupil is 3662.8 nm against 1.5 nm over `rho <= 0.85`.
   *No guard ships for this*: the obvious candidate (`rho_fit < 0.95`) fires on the 8 mm
   cemented doublet where `rho_fit = 0.783` and the correction is excellent (165x, peak
   -0.24 %), so `rho_fit` does not separate the good case from the bad one.  The
   distinguishing quantity is whether the exit field still FILLS the pupil.  Use the
   traced or `caustic='wave'` members on fast elements; use the flag for the wavefront,
   not for the peak.
2. **Fast, symmetric elements** belong on the traced / caustic members, not on the
   analytic screen.  The screen is a per-surface plane projection; the router steers off
   `'phase_screen'` above `na_threshold` for this reason, and warns when it is forced.
3. **FGA convergence through a real singlet is unverified.**  On the f = 1.2 mm fixture
   FGA's intensity-rms width is 3.9x `phase_screen`'s and 3.7x the real-ray geometric fan
   (12.721 um against 3.269 / 3.44 um), at tilt 0 as well as under tilt.  FGA is exact on
   free space with the correct Gouy phase; its convergence knobs (`w0_factor`, `nsig`,
   `prune_frac`, `separable`, `momentum_sampling`) are exercised by exactly one test file
   each.  Treat FGA output through a refracting element as un-calibrated until that is
   measured.
4. **`slant_correction` is a collimated-input identity.**  On a thick symmetric element
   the bundle at the second surface is already converging, and the correction's own
   residual is then comparable to what it removes.
5. **`displaced` remaps carry no in-glass diffraction** in 2-D, and transport a complex
   residual that must be smooth on the grid.
6. **The analytic Fresnel factor is evaluated at the axial AOI.**  There is no path in
   the family that has both a true local AOI and Fresnel.
7. **CuPy twins have drifted**: the traced vignetting rejection kernel is NumPy, so
   `use_gpu=True` keeps the historical extrapolating behaviour.  CuPy is not installed on
   the calibration box and these paths are desk-checked only.

---

## 8. Which test pins which claim

| claim | pin |
|---|---|
| 2-D remap: structured inversion, mirror stability vs the launch lattice, `displaced_n_side` | `tests/unit/test_audit2609_b2_displaced_remap_inversion.py` |
| exit-vertex transfer, grazing policy, JAX twin | `tests/unit/test_audit2609_a1_exit_vertex.py` |
| caustic siblings on a **curved rear** | `tests/unit/test_audit2609_a3_caustic_siblings.py` |
| `caustic='wave'` == traced at `d = 0`; focus vs an independent oracle | `tests/unit/test_audit2609_a3_verify_traced.py` |
| Seidel: fan at the exit vertex, gate, clamp (not extrapolation) | `tests/unit/test_audit_glass.py`; `tests/unit/test_audit2609_a2_verify_lens_analytic.py::TestVerifyL1SeidelScreenIsClampedNotExtrapolated` |
| slant coefficient isolated from the field | `tests/unit/test_hammer_h1_slant_obliquity.py::test_h1_slant_moves_toward_the_oracle_without_overshooting` |
| Fresnel power transmittance vs `4 n1 n2/(n1+n2)^2` | `tests/unit/test_audit2609_a2_analytic_lens.py` |
| tilted surface deviates the beam; both branches, both axes | `tests/unit/test_v5_2_off_axis_conic_surface_frame.py::TestTiltedSurfaceDeviatesTheBeam`, `::TestFieldFrameTiltRampConvention` |
| cross-model tilt / decenter agreement (wave vs two ray models) | `tests/unit/test_niche_p9_decenter_tilt.py` |
| `displaced` phase transport, cache keys, exit-leg index | `tests/unit/test_audit2609_a2_displaced_models.py` |
| `form_error` honoured; multibranch/uniform refuse it | `tests/unit/test_audit2609_a3_verify_traced.py` |
| `stop_index` validation across the family | `tests/unit/test_audit2609_a2_analytic_lens.py`, `tests/unit/test_audit2609_a4_maslov_gbd.py::test_a2_*` |
| Maslov exit-vertex chart, Van Vleck amplitude, `'auto'` routing | `tests/unit/test_audit2609_a4_maslov_gbd.py::test_s3_*`, `::test_s4_*` |
| router frame-invariance (tilt) against a ray oracle | `tests/unit/test_audit2609_a4_fga_s10.py` |
| kwarg **combination** coverage (pairwise array + default round-trip) | `tests/unit/test_audit2609_a15a_lens_covering_array.py` |

---

## 9. Maintenance rule

The audit's process finding was that each round added a new `test_niche_*` file, a new
audit document and a new history block -- and never an update to the existing test for
the same kwarg, a combination test, or a deletion.  For this subsystem:

1. A round that changes behaviour updates **this file** and the existing test for that
   kwarg.  A new test file needs a reason.
2. Every opt-in kwarg needs at least one end-to-end oracle test on a fixture with a
   **curved rear surface** and a **non-collimated** input.
3. Numeric claims carry their oracle, its error floor and the measured value.  A claim
   without a citation gets deleted, not re-derived.
4. Audit history goes to `docs/audits/`, not into this file and not into the source.

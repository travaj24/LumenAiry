# WP-A7 changelog text -- analysis metrics (`lumenairy/analysis/`)

Findings A1-A7 of `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §8.

### Fixed -- analysis/opd: `wave_opd_2d` slipped whole waves on any aberrated pupil (A1, P0)

`wave_opd_2d` took `np.angle` over the WHOLE grid -- zero-amplitude exterior
included, where `np.angle(0) == 0` -- and unwrapped `axis=1` then `axis=0`.
The row pass anchored at column 0 (outside the pupil) and the column pass
re-anchored INDEPENDENTLY IN EVERY COLUMN, so each column picked up its own
`2*pi*k`. Masking happened only afterwards. The failure needed no
under-sampling: it appeared at phase gradients 10-25x below Nyquist, on
smooth, single-valued, simply connected wavefronts, while
`check_opd_sampling` reported SAFE. `wave_opd_2d` is the library's only 2-D
OPD extractor and feeds `optimize/driver.py` -> `RMSWavefrontMerit` /
`ZernikeCoefficientMerit` / `MatchTargetOPDMerit`, the GUI wavefront docks and
the cookbook recipe, so an optimiser driven by it converged on a wrong
prescription.

The unwrap now runs on the PUPIL MASK: the wrapped gradient is integrated
along horizontal runs of valid samples (one `cumsum`) and the runs are linked
to each other through their vertical neighbours by whole waves, so no
integration path ever crosses the exterior and the result is independent of
the path. Measured on a flat circular pupil (N = 512, dx = 1 um, aperture
400 um, 633 nm) carrying pure primary coma and no defocus:

| coma [waves rms] | max abs OPD error before | after | fitted OSA c8 before -> after (true) |
|---|---|---|---|
| 0.20 | 1.000 waves (5.2 % of the pupil wrong) | 0.000 | +0.1024 -> +0.2000 (0.2000) |
| 0.50 | 1.000 waves (55.2 %) | 0.000 | +0.5489 -> +0.5000 (0.5000) |
| 1.00 | 4.000 waves (76.6 %) | 0.000 | +0.9979 -> +1.0000 (1.0000) |
| 5.00 | 19.000 waves (95.2 %) | 0.000 | +4.8492 -> +5.0000 (5.0000) |

and the orchestrator's own cross-verification (`repro/orch/verify_analysis.py`,
0.5 waves rms coma at 0.27 rad/sample) goes from `max error 1.000 waves,
53.2 % of pupil wrong` to `0.000 waves, 0.0 %`.

`wave_opd_2d` also gained a `unwrap=` kwarg (`'itoh'`, the default described
above, or `'reliability'` -- the quality-guided Herraez, Burton, Lalor &
Gdeisat, *Appl. Opt.* **41** (2002) 7437 unwrap for noisy maps), and the
kernel is exported as the new public `lumenairy.analysis.unwrap_phase_2d`.

Files: `lumenairy/analysis/opd.py`, `lumenairy/analysis/__init__.py`.
Tests: `tests/unit/test_audit2609_a7_opd_unwrap.py` (18).

### Added -- analysis/opd: `wave_opd_2d` reports when no unwrap can succeed (A1)

The returned map is checked against the data it came from: every in-mask
neighbour link a single-valued map must satisfy is re-derived, and the largest
one it could not is warned about with its size in waves. Measured: 1.0 waves
for a charge-1 optical vortex, 1.0-7.0 for 5-30 waves rms of aliased coma,
11.0 for uniformly random phase, 0.0000 for every well-sampled pupil. A
second `RuntimeWarning` fires when the pupil support has disconnected
regions, whose relative piston is undetermined. The documented blind spot --
a radially symmetric aliased wavefront wraps onto the exactly self-consistent
phase of a lower-frequency wavefront and has no residue at all (measured
0.0000 waves at 16.5 / 32.9 / 98.8 rad per sample) -- is covered by the
pre-existing `focal_length` sampling gate instead, and both are now spelled
out in the docstring.

### Changed -- analysis/opd: `wave_opd_2d` anchors its piston at the pupil centre (A1)

An unwrap fixes the phase only up to one additive whole wave per connected
region. The map is now anchored on the principal value of the valid sample
NEAREST `x = y = 0` rather than on whatever sample the integration started
from, so a pupil carrying a known defocus comes back with the right INTEGER
WAVE COUNT (a converging wavefront is stationary at the pupil centre).
Measured on a 51.8-wave defocus pupil: absolute error 17.0000 waves before,
0.0000 after. The docstring states the condition: it is a CENTRED pupil that
comes back absolutely anchored, because that is where the anchor sample and
the wavefront's stationary point coincide. A pupil that does not straddle the
grid origin is anchored on its rim instead and its piston is an arbitrary --
but always EXACT -- whole number of waves (measured on a 1.2-waves-rms coma
pupil decentred by (+60, -40) and (+110, +90) um: +1.0000 and +2.0000 waves,
against 0.0000 centred). **Migration:** any caller that stored an absolute
`opd_map` piston from a strongly curved pupil may see it move by a whole
number of waves; the piston it stored was arbitrary. Shape, PV, RMS, Zernike
coefficients above piston, and every nearly-flat map are unaffected.

### Fixed -- analysis/through_focus: the Strehl denominator was a PARAXIAL reference (A2, P0)

`diffraction_limited_peak` built its aberration-free reference from the
paraxial quadratic phase `exp(-i k r^2 / 2 f)` while
`angular_spectrum_propagate` -- the propagator the reference and the
measurement both go through -- is exact. The quadratic differs from the
sphere by a real spherical-aberration term
`W040 = (D / lambda) / (128 (f/#)^3)` waves, so the "diffraction-limited"
denominator was itself aberrated, its peak depressed by its own Strehl, and
every ratio taken against it inflated by `1 / S_ref`. It is the denominator
for `through_focus_scan` / `_jax`, `single_plane_metrics`,
`tolerancing_sweep`, `monte_carlo_tolerancing`, `polychromatic_strehl`,
`polychromatic_psf`'s `per_wavelength_strehl` and the optimiser's
`StrehlMerit`.

The reference is now the exact sphere
`-k0 * sign(f) * (sqrt(x^2 + y^2 + f^2) - abs(f))`, which reduces to the
quadratic in the paraxial limit and costs one `sqrt` per sample. Measured
end-to-end on a PERFECT, aberration-free exact-spherical converging wavefront
at f/2 (D = 614 um, f = 1.228 mm, 600 nm, N = 4096):

```
diffraction_limited_peak   1.342173e+04  ->  1.519335e+05
peak of the perfect sphere at z = f      = 1.519335e+05  (unchanged)
through_focus_scan best Strehl           11.3200  ->  1.0000
```

Across f/#, the Strehl reported for an aberration-free pupil (D = 200 um,
600 nm, N = 2048):

| f/# | W040 [waves] | Strehl before | after |
|---|---|---|---|
| f/50 | 0.0000 | 1.0000 | 1.000000000 |
| f/10 | 0.0026 | 1.0000 | 1.000000000 |
| f/5 | 0.0208 | 1.0015 | 1.000000000 |
| f/3.9 | 0.0439 | 1.0067 | 1.000000000 |
| f/2.5 | 0.1667 | 1.0988 | 1.000000000 |
| f/2 | 0.3255 | 1.4292 | 1.000000000 |

f/50 is unchanged to 1e-6 relative, so the change is confined to the
non-paraxial regime it is about. **Migration:** any recorded Strehl from an
f/# faster than ~f/10 was inflated by `1 / S_ref` and will now read lower --
correctly; a Strehl above 1 for a diffraction-limited pupil is no longer
produced. The "Strehl can exceed 1 in some edge cases" note in
`plot_through_focus` described this defect, not a methodology consequence.

Files: `lumenairy/analysis/through_focus.py`.
Tests: `tests/unit/test_audit2609_a7_strehl_reference.py` (11).

### Fixed -- analysis/detector: the Shack-Hartmann `wavefront` was exactly half (A3, P1)

`shack_hartmann`'s reconstruction averaged two ONE-SIDED cumulative
integrals: `wf_x[i, j]` is `W(x_j, y_i) - W(x_0, y_i)` and `wf_y[i, j]` is
`W(x_j, y_i) - W(x_j, y_0)`, so for any wavefront separable in x and y --
tilt, defocus, astigmatism, i.e. essentially every use -- their average is
`(W - W_00) / 2`. Anchoring removed each half's piston but not the factor.
`slopes_x` / `slopes_y` (what the AO stack consumes) were correct; only the
documented third return value was wrong, and anyone budgeting from it was 2x
optimistic.

It is now a zonal least-squares solve of the co-located Southwell geometry
(*JOSA* **70** (1980) 998), which uses both slope components at every lenslet
and is correct for wavefronts that are not separable. Measured against the
analytic truth scaled by the sensor's own slope gain (N = 256, dx = 5 um,
pitch = 32 dx, f = 5 mm, 632.8 nm):

| input | ratio before | after | slope gain |
|---|---|---|---|
| tilt 0.20 mrad | 0.4741 | 0.9481 | 0.9481 |
| tilt 0.50 mrad | 0.4744 | 0.9488 | 0.9488 |
| tilt 1.00 mrad | 0.4725 | 0.9450 | 0.9450 |
| defocus, 1 um edge | 0.4626 | 0.9340 | -- |

`max abs wf` on the defocus case: 5.4263e-07 m -> 1.4351e-06 m against a
truth of 1.5000e-06 m. The docstring's "cumulative trapezoidal integration"
was also inaccurate (it was a rectangle-rule `cumsum`); both reconstructions
now really are trapezoidal.

### Changed -- analysis/detector: `shack_hartmann` gains `reconstruction=`, and unmeasured lenslets are NaN (A3)

`reconstruction='southwell'` (new default) or `'itoh'` (the single path
integral, down column 0 then along each row). **Migration:** a lenslet whose
slopes are the NaN out-of-bounds sentinel now returns NaN in `wavefront` as
well, instead of being integrated through as if it had measured zero slope --
the NaN pattern of `wavefront` matches `slopes_x` exactly. Callers that
reduce the map should mask (`np.nanmax`, `np.isfinite`).

Files: `lumenairy/analysis/detector.py`.
The `'itoh'` path does NOT get the NaN exclusion, and the docstrings now say
so on all three sites (`shack_hartmann(reconstruction=)`,
`_reconstruct_wavefront`, `_itoh_wavefront`): a path integral has one route to
each lenslet, so an un-measured lenslet on that route is substituted with zero
slope and integrated through, and everything beyond it on the path inherits
the error. Measured on an annular lenslet mask with exact analytic slopes:
`'itoh'` off by 0.49 of the wavefront span where `'southwell'` is exact to
8.7e-15 of it.

Tests: `tests/unit/test_audit2609_a7_detector_sh.py` (12);
`tests/unit/test_niche_s12_shack_hartmann_reference.py::TestTiltOracle::test_reconstruction_integrates_the_uniform_slope`
and `tests/unit/test_v4_16_1_agent_a.py::_measured_pitch_from_wf` had the
factor 0.5 / 2.0 baked into their oracles and were pinning the defect; both
are corrected in place (measured ptp 2.2680e-06 m post-fix against the oracle
2.2680e-06, ratio 1.0000; pre-fix 1.1340e-06, ratio 0.5000).

### Added -- analysis/image_plane_wfe: `object_distance = inf` (A4, P1)

`eval_image_plane_wfe` required a finite `object_distance > 0` and launched
the bundle at the object, so an infinite conjugate could only be approximated
by a large finite number -- and the object-side ray-surface intersection then
cancels catastrophically in float64 (`|P - C|^2 ~ object_distance^2` against
`R^2`, with the two roots separated by only `~2 abs(R)`). Measured on a
biconvex N-BK7 singlet (R = +-50 mm, d = 3 mm, D = 10 mm, 587.6 nm, EFL
48.87 mm, BFL 47.875 mm):

| object_distance | chief ray z0 (must be 0) | PV [waves] | RMS [waves] |
|---|---|---|---|
| 1e3 m | +0.0006 um | 3.4675 | 1.0430 |
| 1e4 m | +0.0238 um | 3.6912 | 1.0649 |
| 1e5 m | -5.34 um | 47.2747 | 10.7591 |
| 1e6 m | +589.41 um | 213.5098 | 61.9051 |
| **inf (new)** | **0 exactly** | **3.5944** | **1.0686** |

`prescription['object_distance'] = float('inf')` (or `None`) now launches a
collimated bundle on a plane wavefront a few aperture widths before surface 0,
with the object-side OPL carried exactly for any field angle; the paraxial
image distance comes out at the BFL (47.87519 mm) to 1e-9 relative. Verified
against an independent transverse-ray-aberration integral (PV 3.6586 waves,
-1.8 %) and the traced longitudinal spherical aberration (LSA -0.8144 mm =>
W040 3.651 waves, -1.6 %). `image_plane='best_rms'` on the same fixture moved
the reference sphere to 96.6435 mm on a 47.9 mm-BFL lens at 1e6 m; at infinity
it lands at 47.4773 mm with PV 0.9425 / RMS 0.2723 waves.

A new `field_max_rad` kwarg (and `prescription['field_max_rad']`) carries the
off-axis field for an infinite conjugate, where a field point is a DIRECTION
rather than a height; a non-zero `field` at infinity without it is refused,
and a `field_max_rad` at or beyond 90 deg is refused on the ANGLE (a guard on
the direction cosine `N = sqrt(1 - sin^2 x - sin^2 y)` cannot catch it: `sin`
is not monotonic past `pi/2`, so 2.0 rad = 114.6 deg would fold back silently
to 65.4 deg). **`field` keeps its object-position sense at BOTH conjugates**:
`field = (0, +1)` is an object above the axis either way, so the chief travels
towards `-y` and the direction cosines are `L = -sin(Hx * field_max_rad)`,
`M = -sin(Hy * field_max_rad)` -- the NEGATIVE of `raytrace.ray_fan`'s
`field_angle`, which is a ray-direction angle. Measured on the singlet above:
`object_distance = inf` at `field = (0, +1)` agrees with `object_distance =
1e3 m` at the same `field = (0, +1)` to 0.0025 / 0.0033 / 0.0033 waves
(<= 0.07 % of the map's span) at 0.5 / 1 / 2 deg, where the opposite sign
differs by 0.854 / 1.706 / 3.404 waves (23 / 40 / 59 % of span). PV and RMS
are sign-blind, so only a per-ray comparison sees it.
The collimated launch measures its OPL from the incident wavefront through
`raytrace.trace.seed_entrance_eikonal` (R2's helper, which carries the `N*z`
term an off-`z=0` launch needs); the finite point-source launch keeps
`opd_seed='plane'`, which is the correct zero for a bundle whose rays all
leave one point.

The read of `image_rays` at `image_plane_wfe.py:508` is **not** in the
exit-vertex bug class and was left alone: the rays are genuinely left on the
last surface's sag (measured `image_rays.z` over -242.084 … 0.000 um on a
prescription that does not end in an image plane), but `_ray_sphere_opd`
carries that `s2z` into the ray-sphere quadratic, so the OPD computed from the
sag state and from `res.at_exit_vertex()` agree to 1.774e-11 waves over a PV of
3.5944 waves.

### Added -- analysis/image_plane_wfe: a warning when a finite `object_distance` has lost the surface sag (A4)

Gated on the mechanism rather than a round number: the intersection error is
`~eps * object_distance^2 / min|R|`, and the warning fires when that exceeds a
tenth of a wave. On the fixture above it is silent at 1 m, 1e2 m and 1e3 m --
where the reported PV is flat at 3.465-3.468 waves -- and fires at 1e4 m,
where PV has already moved to 3.691, and above.

Files: `lumenairy/analysis/image_plane_wfe.py`.
Tests: `tests/unit/test_audit2609_a7_image_plane_wfe.py` (13).

### Fixed -- analysis/phase_retrieval: `gerchberg_saxton`'s reported error was off by N_pix (A5, P2)

`far_field` is an unnormalised DFT (`sum |F|^2 = N_pix * sum |field|^2`) and
was compared against a target rescaled to the SOURCE power, leaving a
hard-wired factor `N_pix` in the mean-square error, which therefore could not
reach zero even for an exact solution. On a target built as
`|FFT(source * exp(i phi0))|` with `phi0` supplied as `initial_phase` -- an
exact solution to 2.132e-14 -- the reported error was **3.775380e+02**, 97 %
of the target energy 3.896184e+02, and flat over 50 iterations; it is now
**3.26e-29**. The retrieved PHASE is unaffected (both amplitude-replacement
steps are scale invariant), which is why this survived. The JAX twin
`gerchberg_saxton_jax` never rescaled at all, so the two backends' `err`
differed by `N_pix` despite documenting the same physics; both now use
`sqrt(N_pix * source_power / target_power)`.

### Fixed -- analysis/phase_retrieval: `gerchberg_saxton_jax`'s `dtype` handling (A5 follow-up)

Two defects behind the "same physics" claim, both found in verification:

* A COMPLEX `dtype` (`np.complex128`) fell through the "unrecognised real
  dtype" branch, made `src` / `tgt` complex, and killed the call in
  `float(err)` with `TypeError: float() argument must be a string or a real
  number, not 'complex'`. A complex request now names the iteration's FIELD
  type and the amplitude / phase arrays take its real counterpart, so `err` is
  always a real float.
* `dtype=None` resolved to float32 unconditionally, so a caller who had
  enabled `jax_enable_x64` still got a single-precision answer. It now follows
  the flag, as `jax.numpy`'s own default float type does. Measured with x64
  on, 10 iterations at N = 32: the two backends' `err` now agree to every
  digit float64 carries (1.5027588862e-01 both, previously 2.0e-05 relative
  apart), and on an exactly-solvable target the JAX floor drops from 2.3e-12
  to 5.2e-29. **Migration:** a caller who wants the historical single
  precision under x64 passes `dtype=np.float32` explicitly.

Files: `lumenairy/analysis/phase_retrieval.py`.
Tests: `tests/unit/test_audit2609_a7_misc.py` (4);
`tests/unit/test_audit_analysis.py::_gs_reference` and
`test_3a_return_history_matches_reference` re-implemented the old scaling in
their own oracles and are corrected in place.

### Fixed -- analysis/plotting: `plot_stokes` used the x extent for both axes, and two half-pixel conventions disagreed (A5 / A7)

`plot_stokes` had no `dy` and took `_auto_extent(Nx, dx)` for both axes, so
any `Ny != Nx` or anamorphic Jones field got a mislabelled y axis --
invisibly, since `_auto_extent(64, 1e-6)` and `_auto_extent(32, 2e-6)` are
identical. It now takes `dy` (defaulting to `jones_field.dy`) and builds the
y half from `Ny`. Open since AUDIT_V4_13_0 (S2.1).

Separately, `_auto_extent` returned `(-N/2*dx, +N/2*dx)` while the sample grid
is `(arange(N) - N/2) * dx`; since `extent` addresses pixel EDGES, every drawn
pixel centre sat `+dx/2` from the sample it displayed, and `plot_psf`'s
`(x[0], x[-1])` form had the opposite half-pixel error -- so the same field
plotted two ways landed a whole pixel apart. Both now address the same edges,
`(-(N/2 + 1/2) dx, +(N/2 - 1/2) dx)`. Five sites that resolved the y unit with
a second independent `'auto'` call (which can land on a different unit and
silently mix units on one figure) now inherit the x axis's resolved unit.

### Changed -- analysis/plotting: the lens-layout runaway-focus census re-measured (A7 follow-up)

The measured envelope behind `_LAYOUT_FOCUS_RUNAWAY_RATIO = 10.0` recorded
`LA1509-C 0.9883`, taken before the `LA1509-C` catalogue radius was corrected
(R1 103.29 -> 51.5 mm, EFL 199 -> 99.652 mm). Re-measured on the six catalogue
rows: LA1050-C 0.9733 -> 0.9729, **LA1509-C 0.9883 -> 0.9762**, LA1301-C
0.9912 -> 0.9910, AC254-050-C 0.8677 -> 0.8664, AC254-100-C 0.9511 -> 0.9508,
AC254-200-C 0.9806 -> 0.9805. Max over the whole census 0.9992, so the
envelope claim and the bar are unchanged; the comment now carries both
measurement dates and names the rows that were re-measured.

Files: `lumenairy/analysis/plotting.py`.

### Performance -- analysis/through_focus: 202.3 -> 128.8 ms/plane, 36 % of a through-focus scan (A6)

Two changes, measured together on a 21-plane scan at N = 1024 complex128
(interleaved medians, `OPENBLAS_NUM_THREADS=1`):

* `through_focus_scan` rebuilt `exp(1j * kz * z)` over the whole K-grid every
  plane -- 52.9 of the plane body on this machine, 185.7 of 264.5 ms on the
  auditor's. For a uniformly spaced z it is a recurrence,
  `H(z + dz) = H(z) * H(dz)`: one complex multiply, measured 11.0 ms/plane,
  **4.8x**. Gated on z uniformity measured to better than 1e-12 rad of phase
  on the actual `z_values`, so a non-uniform scan keeps the direct `exp` and
  stays bit-identical to a per-plane `angular_spectrum_propagate`. The
  band-limit mask is now two 1-D broadcasts instead of a full N^2 boolean per
  plane (bit-identical), and the mask multiply is in place.
* `single_plane_metrics` built `|E|^2` three times (once itself, once inside
  `beam_centroid`, once inside `beam_d4sigma`) and the centroid twice.
  The default path now shares one intensity map, one meshgrid and one set of
  moment sums through a new `beam_stats._whole_grid_moments` /
  `_centroid_and_d4sigma`: 81.4 -> 52.4 ms/plane, **1.55x**, bit-identical
  (`peak_I`, `centroid_x/y`, `d4sigma_x/y` compare `==` against
  `beam_centroid` / `beam_d4sigma`). The ISO 11146 `background` / `aperture`
  paths, and anything that is not a plain 2-D NumPy field, are routed to the
  unchanged two-call form.

**Accuracy note (documented tolerance, not bit-identity).** Both the direct
`exp` and the recurrence evaluate the same exact function and both round the
ARGUMENT at `|kz z| eps / 2`, so their difference is bounded by
`2 |kz z|_max eps` and neither is the more accurate one. Measured
`max |H_rec - H_dir|`: 6.2e-12 at N = 1024 / 21 planes / `max|kz z|` =
1.09e4 rad, and 1.1e-10 at 2.23e5 rad, against a predicted 4.8e-12 / 9.8e-11.
The modulus of H is preserved to 1.8e-15, so no energy drifts. The worst
relative metric drift measured against a per-plane
`angular_spectrum_propagate` reference is 9.5e-12, and
`tests/unit/test_perf_v4_12_0_through_focus.py`'s agreement bar moves from
1e-12 to a derived `HOIST_RTOL = 1e-9` with that derivation recorded in the
file.

The JAX twin `through_focus_scan_jax` deliberately keeps its per-plane
`exp(1j kz z)`: it is the reference evaluation the NumPy path is checked
against (`TestThroughFocusScanMatchesJAXTwin`), and re-deriving it per plane
is what makes that check independent. The recurrence is a NumPy-path
optimisation only.

### Performance -- analysis/polychromatic: `radial_power_bands` stops growing with the band count (A6)

The mask-and-sum loop built a full `Ny x Nx` boolean and re-summed the whole
grid once per radius. Above a measured crossover (`_RADIAL_SORT_CROSSOVER =
96` bands; the crossover sits at 65-82 across N = 256...2048, almost
independently of N) it now sorts the pixels by radius once and reads every
band off the cumulative sum. At N = 1024, including the shared meshgrid and
`|E|^2` setup:

| n_radii | before | after |
|---|---|---|
| 1 | 29.6 ms | 29.8 ms (bit-identical, loop kept) |
| 8 | 35.1 ms | 34.1 ms (bit-identical) |
| 64 | 87.8 ms | 78.3 ms (bit-identical) |
| 256 | 387.8 ms | 129.8 ms (**2.99x**) |
| 1024 | 1245.7 ms | 129.3 ms (**9.63x**) |

Below the crossover the result is bit-identical to every earlier release --
`single_plane_metrics(bucket_radius=...)` asks for ONE radius per plane. Above
it the value differs only by float64 summation order (a sequential `cumsum`
over a radius-ordered permutation instead of numpy's pairwise reduction),
bounded by `O(Ny*Nx * eps)` = 7e-12 relative at N = 1024 and measured at
2.1e-13.

One query the two paths would otherwise answer differently is a **NaN
radius**: `R2 <= nan` is False everywhere, so the masked loop returns 0, while
`searchsorted` sorts NaN ABOVE every finite key and the sorted path handed
back the whole grid's power (measured 2.513e-09 against 0.0). The sorted path
now reproduces the masked loop's answer, so the crossover is invisible to the
caller. `+/- inf` needs no special case -- `r*r` is `+inf` on both paths and
both return the total.

Files: `lumenairy/analysis/through_focus.py`, `lumenairy/analysis/beam_stats.py`,
`lumenairy/analysis/polychromatic.py`.
Tests: `tests/unit/test_audit2609_a7_misc.py` (9);
`tests/unit/test_audit2609_verify_a7.py` (the crossover, NaN and inf pins).

### Fixed -- analysis: the P3 row (A7)

* `ghost.retrace_ghost_path` reported a "50 % encircled-energy" FWHM from the
  MEDIAN RAY radius. `make_rings` puts the same ray count on every ring, so
  the areal sampling density falls as `1/r` and the median ray is not the
  median of the ENERGY: on a uniform disc it reads 0.500 R for every ring
  count where the truth is `1 / sqrt(2) = 0.7071 R`, 29 % low. Rays are now
  weighted by the pupil area each stands for and the radius is the 50 % point
  of the resulting encircled-energy curve: 0.667 / 0.667 / 0.708 R at 6 / 12 /
  24 rings, i.e. within one ring spacing.
* `aberration.caustic_diagnostic` silently clamped a complex-conjugate
  eigenvalue pair to `tr / 2` via `max(0.25 tr^2 - det, 0)`, reporting two
  spurious coincident real eigenvalues and a wrong Maslov index for any system
  whose transverse map rotates. Those planes are now counted and warned about
  (an axisymmetric system never reaches the branch).
* `interferometry.simulate_interferogram` built the y tilt ramp on `dx`; it now
  takes `dy`. Its docstring promised "values in [0, 1]" where the
  implementation returns `background * (1 + V cos phi)`; the real range is
  documented.
* `ao_closed_loop`, `make_shack_hartmann_wfs` and `coronagraph_contrast_curve`
  live in `analysis/` but were reachable only from the package root;
  `clear_meshgrid_cache`, `meshgrid_cache_bytes` and
  `zernike_basis_cache_bytes` were exported nowhere. All are now in
  `lumenairy.analysis.__all__`, alongside the new `unwrap_phase_2d`.
* `strehl_phase_integral` and `strehl_ratio` disagree by three orders of
  magnitude on a tilted wavefront (measured 0.00069 vs 0.99996 at 1 wave rms
  of pure tilt) because the phase integral does not remove piston or tilt.
  Both conventions are legitimate; the docstring now says so.
* `zernike_basis_matrix`'s cache key sampled `X.flat[N * N / 2]`, which is
  `X[N / 2, 0]` = `x[0]` = `X.flat[0]` for the row-repeating meshgrid every
  caller passes -- no information for X at all. It now samples 16 points along
  the grid DIAGONAL, which moves in both axes; the key stays O(1) to build so
  a cache hit is still cheap.

Files: `lumenairy/analysis/ghost.py`, `lumenairy/analysis/aberration.py`,
`lumenairy/analysis/interferometry.py`, `lumenairy/analysis/zernike.py`,
`lumenairy/analysis/strehl.py`, `lumenairy/analysis/__init__.py`.
Tests: `tests/unit/test_audit2609_a7_misc.py` (7).

# WP-A1 changelog text (ray tracing + the shared exit-vertex helper)

Assembled by the orchestrator into `CHANGELOG.md`.  Finding IDs are from
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §4 and §15.1.

### Added -- raytrace: one shared exit-vertex transfer (audit §15.1 / F-O3)

`raytrace.trace()` leaves every ray at its intersection with the LAST surface,
i.e. at `z = sag(rho)`, not on that surface's vertex plane.  The audit's
exit-vertex census found SEVEN consumers reading `image_rays.opd / .x / .y`
there -- six re-deriving the correction by hand, five of them getting some part
of it wrong, and the two JAX copies disagreeing with the NumPy ones on grazing
rays.  There is now exactly one implementation:

* `TraceResult.at_exit_vertex(n_exit=None)` -- returns a NEW `RayBundle`
  transferred to the exit vertex plane; `n_exit` defaults to the index of
  `surfaces[-1].glass_after` at the trace wavelength (`glass_before` when the
  last surface is a mirror).
* `raytrace.exit_vertex_transfer(bundle, n_exit)` -- the functional form for
  callers that hold a bare bundle.
* `raytrace.jax_trace.exit_vertex_transfer_jax(state, n_exit)` -- the
  JAX-traceable twin (`jax.grad`-safe at the grazing boundary).
* `raytrace.EXIT_VERTEX_GRAZING_TOL` (`1e-30`) -- the shared `|N|` tolerance.

Semantics: signed `t = -z/N` on ALIVE rays only; `opd += n_exit*t`;
`x, y += (L, M)*t`; `z = 0`.  Grazing rays (`|N| <= 1e-30`) are KILLED with
`RAY_MISSED_SURFACE` instead of being teleported to the vertex plane with zero
optical path; dead rays are frozen exactly as they were.  The operator is
idempotent and never mutates its input.  Measured against the analytic
vertex-plane OPL `n_exit*(-sag/N)` on sphere / parabola / hyperbola / oblate
ellipsoid exit surfaces: `max |delta| = 0.0e+00 m`.  On a curved-rear singlet
the term the consumers were missing is a pure defocus: `-1.117e-03 m` of
`rho^2` with `|c4/c2| = 0.064`.

`intersection._intersect_surface` (flat branch), `intersection._transfer` and
`ray_fan.refocus` now share the same `vertex_plane_transfer_t` kernel, so the
four "advance to a `z = const` plane" primitives use identical arithmetic and
one definition of "grazing".

Migration note for the other work packages: replace every hand-written
`t = -z/N; opd += n*t; x,y += (L,M)*t; z = 0` block with
`result.at_exit_vertex()` (or `exit_vertex_transfer(bundle, n)` /
`exit_vertex_transfer_jax(state, n)`).  The helper changes behaviour in two
places relative to those copies: grazing rays die rather than teleport, and
dead rays keep their pre-transfer state.

### Fixed -- raytrace: `opd_fan_data` had no reference sphere (R1)

`opd_fan_data` / `opd_fan_data_world` returned `(image_rays.opd - opd_chief)`,
the OPL to each ray's OWN intercept, which differs from the wavefront error at
FIRST order in the transverse aberration (`W_plane - W_true = eps sin(theta')`).
Each ray is now referenced to a reference sphere centred on the chief ray's
image point and passing through the exit pupil (Welford §4; the convention
`analysis.eval_image_plane_wfe(sphere_tangent='exit_pupil')` already used).

Measured at `rho = 1` against an independent exact singlet trace (ray/sphere
intersection + vector Snell + straight leg to the Gaussian image point, no
library code):

| system | before | after | oracle |
|---|---|---|---|
| f/4 plano-convex (convex first) | **+36.194 w** | -11.720 w | -11.714 w |
| f/4 plano-convex (flat first) | **+151.789 w** | -45.063 w | -45.054 w |
| f/20 plano-convex | +0.0545 w | -0.01813 w | -0.01813 w |
| f/50 plano-convex | +0.0014 w | -0.000464 w | -0.000464 w |
| spherical mirror R=-200 mm, h=25 mm | **+64.502 w** | -20.611 w | -20.774 w (Seidel -S1/8) |

i.e. the wrong SIGN and 3.1x the magnitude at f/4, now agreeing with the oracle
to 0.0067 waves (the residual is the second-order difference between the
exit-pupil-tangent and last-surface-tangent reference-sphere conventions).  The
aberration-free control is unchanged: a parabolic mirror gives PV 7.0e-11 waves
before and after.

New keyword `reference_sphere_radius` (default `None` = derive from the exit
pupil; `np.inf` selects a reference PLANE; a positive float overrides).  When
the prescription does not end at or near an image plane there IS no image point
to reference to; the function now emits a `RuntimeWarning` naming the geometry
and falls back to the reference-plane limit instead of returning NaN and
~1400-wave garbage.

### Fixed -- raytrace: off-axis OPD fans carried a launch-plane tilt (R2)

`make_fan` / `make_ray` launch on the `z = 0` PLANE with `opd = 0`, which for a
field-angle bundle is not a wavefront: every ray at height `y` has already
travelled `y sin(theta)` relative to the common incident wavefront.  The OPD
fans now seed the **entrance eikonal** `L*x + M*y` on both the fans and their
chief rays.  Fitted linear term of the fan on an f/4 plano-convex singlet
(25 mm pupil, 587.6 nm):

| field | before | after |
|---|---|---|
| 0.5 deg | **-185.64 w** | +0.01 w |
| 2.0 deg | **-744.06 w** | +0.06 w |
| 5.0 deg | **-1879.86 w** | +0.15 w |

against 37.6 waves of real (quartic) aberration -- a 50:1 contamination at
5 deg.  Bit-identical on axis.

`_make_bundle` gains an opt-in `opd_seed={'plane', 'eikonal'}` (default
`'plane'`, i.e. unchanged) and `raytrace.trace.seed_entrance_eikonal(rays)` is
exported as the functional form, including the `N*z` term for bundles launched
off the `z = 0` plane.  The default is deliberately NOT changed: `_make_bundle`
is the shared launcher for ~20 consumers, two of which (`_lens_traced.py`'s
v5.25.1 H6 `_carrier_W_fn`, and the asymptotic canonical fit) add their own
entrance eikonal downstream and would double-count it.

### Fixed -- raytrace: `seidel_coefficients` ignored conic + aspheric terms (R3)

The per-surface loop read only `radius` / glasses / `thickness` / `is_mirror`;
`grep -n "conic\|aspheric" seidel.py` returned zero hits.  The Welford §8.5
aspheric contribution `dS_I = 8 (n2 - n1) A4_eff h^4` with
`A4_eff = conic/(8 R^3) + A4` is now added to S1, and its
`(y_chief/y_marginal)^{1,2,3}` scalings to S2 / S3 / S5 (S4, Petzval, is
curvature-only and unaffected), in all three branches (mirror, curved refractor,
flat-base asphere -- a Schmidt plate is the last of those).

| case | S1 before | S1 after | real-ray rho^4 oracle |
|---|---|---|---|
| mirror R=-200 mm, k=0 | +9.765625e-05 | +9.765625e-05 | -12.2047 um (-S1/8 = -12.2070) |
| mirror R=-200 mm, k=-0.5 | **+9.765625e-05** | +4.882813e-05 | -6.1023 um (-S1/8 = -6.1035) |
| mirror R=-200 mm, k=-1 (parabola) | **+9.765625e-05** | **1.355e-20** | 0.0000 um |
| mirror R=-200 mm, k=-1.5 | **+9.765625e-05** | -4.882812e-05 | +6.1016 um (-S1/8 = +6.1035) |
| singlet A4 = -500 m^-3 | **+5.320645e-05** | +2.737848e-06 | -0.3326 um (-S1/8 = -0.3422) |
| singlet A4 = -2000 m^-3 | **+5.320645e-05** | -1.486679e-04 | +18.6130 um (-S1/8 = +18.5835) |

Agreement with the real-ray rho^4 fit is 0.14-0.29 % across
A4 in {0, +-250, +500, -1000, -2000} m^-3 (the residual is genuine fifth order)
and exact (1.4e-20) for the parabola at every k.  A parabolic mirror at infinite
conjugate is now reported as aberration-free, and an A4 = -500 aplanatised
singlet as 0.34 um rather than 6.65 um.

The docstring now states what IS and is NOT included, and a `RuntimeWarning`
names any surface the rotationally-symmetric third-order expansion cannot
represent at all (biconic `radius_y` / `conic_y` / `aspheric_coeffs_y`,
`freeform`, the field-frame decenter/tilt block, and a power-2 aspheric
coefficient, which changes the paraxial curvature).  A6/A8 are documented as
out of scope -- they generate fifth- and higher-order aberration.

### Fixed -- raytrace: conic surfaces falsely reported RAY_MISSED_SURFACE (R4)

`intersection._intersect_surface`'s Newton branch seeded from the ray-SPHERE
quadratic and used ITS discriminant as the miss test.  A sphere of radius `R`
only exists for `h <= |R|`, so any ray beyond that on a paraboloid, hyperboloid
or flattened ellipsoid was killed although it genuinely hits the conic.  Both
JAX kernels (`_intersect_jax`, `_intersect_jax_param`) carried the identical
test.  All three now use the EXACT conic quadratic
(`F = c(x^2+y^2) - 2z + (1+k) c z^2 = 0`) as both seed and miss test, solved in
the Spencer & Murty (JOSA 52, 672 (1962)) stable form `t = e/q` with
`q = -(b + sign(b) sqrt(disc))/2` -- which is also the near root by
construction, preserving the v5.4.1 direction-aware behaviour for
backward-propagating (post-mirror) rays.

Thorlabs-class aspheric condenser R = 10.84 mm, k = -0.6 (conic valid to
h = 17.14 mm), collimated axial rays:

| h [mm] | before | after | true sag [mm] |
|---|---|---|---|
| 10.83 | alive, t = 6.095530 | alive, t = 6.095530 | 6.095530 |
| 10.90 | **dead (err 3), t = 0** | alive, t = 6.186249 | 6.186249 |
| 11.40 | **dead (err 3), t = 0** | alive, t = 6.863647 | 6.863647 |
| 15.00 | **dead (err 3), t = 0** | alive, t = 13.988555 | 13.988555 |

and a parabola R = 50 mm at h = 60 mm returns t = 36.000000 mm where it
previously returned `alive=False, t=0`.  Agreement with the closed-form conic
sag is `max |dz| = 8.7e-19 m` over h/|R| up to 1.8 including concave
hyperbolas.  A 41-ray fan through the public `trace()` API on the condenser goes
from 39/41 to 41/41 alive.  Genuine misses still die: an oblate ellipsoid
(k = +2, domain h < |R|/sqrt(1+k)) kills rays outside its domain exactly as
before.

Unchanged: the pure-spherical fast path (`conic == 0`, no aspherics) keeps its
own branch on both backends -- `max |dt| = 3.331e-16 m` vs an independent
Spencer-Murty root over the audit's 480k-random-ray sweep, the same figure as
before.  Anamorphic / freeform / field-frame surfaces keep the legacy sphere
seed, since a conic discriminant is not a valid miss test for them.

### Fixed -- raytrace: the diffraction-order kick omitted the medium index (R5)

The grating equation conserves the tangential wavevector,
`n2 L' = n1 L + m lambda_vac / Lambda`, so the kick applied to the
post-refraction direction cosines carries a `1/n2`.  All four sites
(`trace`, `trace_world`, `jax_trace._apply_doe_kick_jax`,
`apply_doe_phase_traced`) applied `m lambda / Lambda` directly: exact in air,
high by exactly `n2` at any interface into glass.  Measured on a grating on the
glass side of an air -> N-BK7 interface (Lambda = 5 um, lambda = 1.31 um,
m = 1): `L = 0.26200000` before vs `0.17425045` after, ratio 1.503583 = n(N-BK7)
exactly -- a 50 % direction error (15.2 deg instead of 10.0 deg).

The grating's own phase-screen OPL term `m lambda_vac x / Lambda` is
index-INDEPENDENT (its transverse gradient must equal `n2 L' - n1 L`) and is
unchanged; the two quantities are now computed separately so the identity holds.
`apply_doe_phase_traced` gains `n_medium: float = 1.0` (default reproduces the
pre-fix, air-correct behaviour); the in-trace kicks resolve the index from the
surface's `glass_after` automatically.

### Fixed -- raytrace: `rays_from_field` aliased at half the grid Nyquist (R6)

`_angle_complex_gradient` used the two-pixel phase ratio
`arg(E[j+1] conj(E[j-1])) / (2 dx)`, unambiguous only for
`|L| < lambda/(4 dx)` -- HALF what the grid supports -- and wrapping silently
above it.  It is now the symmetrised ONE-pixel circular mean
`arg(u(E[j+1] conj(E[j])) + u(E[j] conj(E[j-1]))) / dx` (each phasor normalised
before summing), exact to the full grid Nyquist.

At lambda = 1 um, dx = 2 um (grid Nyquist |L| <= 0.25):

| L_true | before | after |
|---|---|---|
| 0.150 | **-0.100000** | +0.150000 |
| 0.200 | **-0.050000** | +0.200000 |
| 0.240 | (aliased) | +0.240000 |
| 0.260 | -0.240000 | -0.240000 (beyond the GRID Nyquist) |

Boundary columns/rows now use a one-sided one-pixel difference instead of the
clipped self-reference, so edge rays get the full direction cosine: measured
ratio to truth 0.5000 before, 1.0000 after, over 128 edge rays.  The verified
sign convention (5e-16) and the converging-wave focus (0.001 nm rms at +f) are
preserved.  The old docstring claim that the two-pixel form "can detect
evanescent rays" is removed -- measured, an evanescent `L = 0.49` came back as a
benign `L = -0.01`.

### Fixed -- raytrace: `_transfer` teleported grazing rays (R6)

`_transfer` masked `t` to 0 for `|N| <= 1e-30` but reset `rays.z` to the next
vertex plane unconditionally, so a ray parallel to the axis-normal planes was
moved one gap downstream with ZERO optical path and stayed `alive=True,
error_code=0` -- the "immortal phantom" that R-4 removed from
`_intersect_surface`'s flat branch but not from here, and a reachable state
because `trace`'s DOE branch keeps an `N == 0` order alive by design.  Measured
on a bundle at `z = 1e-4` with `N = 0` through `_transfer(10 mm, n=1)`:
`z=[0 0], alive=[T T], opd=[0 0], error_code=[0 0]` before;
`z=[1e-4 1e-4], alive=[F F], opd=[0 0], error_code=[3 3]` after.  Dead rays
(including the newly-killed grazing ones) now keep their `z`, matching the
`_transfer_jax` freeze-dead-rays policy adopted in S3-12.

### Performance -- raytrace: in-place position / OPL updates in the hot path (R6)

`_intersect_surface`'s position+OPL block and `_transfer` allocated two
N-sized temporaries per update line (~8 per surface).  Both now write through a
single reusable buffer via `np.multiply(..., out=)` / `np.add(..., out=)`,
guarded so a non-float64 or read-only bundle falls back to the allocating form.
Output is BIT-IDENTICAL (the two roundings are the same two roundings):
measured `max |dx| = max |dy| = max |dopd| = 0.0` on a 7-surface trace.

Medians of 7 interleaved runs on a 7-surface prescription,
`output_filter='last'`, box shared with other jobs:

| N | before | after | speedup | tracemalloc peak before -> after |
|---|---|---|---|---|
| 300 000 | 2806 ns/ray | 2676 ns/ray | 1.049x | 70.7 -> 65.8 MiB |
| 1 000 000 | 2932 ns/ray | 2841 ns/ray | 1.032x | 235.6 -> 219.4 MiB |

### Fixed -- raytrace: the P3 bundle (R7)

* `intersection._refract` -- the `RAY_TIR` stamp is now genuinely
  first-failure-wins (`newly_tir & (error_code == RAY_OK)`), as its own comment
  had promised since the matching aperture-block fix; the `np.where` was
  unconditional and relabelled any code a ray already carried.
* `raytrace.rays_from_field` -- `opd` is seeded from `np.angle(E)`, which is
  WRAPPED into `(-lambda/2, +lambda/2]`.  This is now documented at three
  levels (module, parameter, `Returns`) and a new `opd_phase={'wrapped',
  'unwrapped'}` keyword (default `'wrapped'`, unchanged) unwraps the phase over
  the sampled support for consumers that treat `RayBundle.opd` as a geometric
  path.  Measured on a converging wave with 0.8 waves of true spread: returned
  `opd` covered the whole [-494.6, +498.3] nm wrap interval.
* `raytrace.trace_summary` -- the loss breakdown now includes
  `evanescent=` (and an `unclassified=` column), so it sums to the reported
  lost count when a `surface_diffraction` order goes evanescent.
* `raytrace.trace._register_fixed_index` -- the `GLASS_REGISTRY` /
  `_glass_cache` read-modify-write is now made under `glass._GLASS_CACHE_LOCK`,
  the lock the GL-2 comment says exists to serialise exactly that.  The
  import-time `'__thin_lens__'` registration is kept (it is load-bearing and
  pinned by `tests/unit/test_audit_p1_glass_registration.py`) and documented as
  deliberate, including the unbounded growth of content-derived index names.
* `raytrace.raytrace_system` -- CLONES the last surface with the image distance
  instead of mutating it in place, the pattern `trace_prescription` moved away
  from in v4.13.2 (audit P1-NEW-J).
* `raytrace.layout` -- the module docstring now states plainly that it contains
  no layout GEOMETRY and points at `analysis.plotting.plot_lens_layout` and
  `raytrace.spot_diagram`; the name is a v5.1.0 split artefact.
* `differential._adrt_jax` -- walked the whole prescription TWICE
  (`jacfwd(_state)` for the Jacobian, then `vmap(_full)` for the state and OPL).
  One `jax.jacfwd(_full, has_aux=True)` pass now returns all three; the Jacobian
  matches the NumPy dual backend to 3.3e-16 and the OPL/position exactly.
* `raytrace.field_of_view` -- the finite-conjugate branch returned
  `arctan((aperture/2)/object_distance)`, the object-space APERTURE half-angle,
  which does not depend on the sensor and is a numerical aperture, not a field
  of view.  With `sensor_half_height_m` it now returns the Gaussian-imaging
  field of view (`m = -f/(s_obj - f)`, `h_obj = h_img/|m|`); without it the
  legacy proxy is kept but emits a `RuntimeWarning` saying what it actually is.

### Changed -- tests that pinned the old behaviour

* `validation/raytrace/test_raytrace.py::t_opd_fan_small_for_singlet` passed a
  bare singlet with no image surface, so the "OPD" it measured was the lens's
  own converging curvature at its rear face (6.0 waves at f/40) rather than any
  aberration -- it could not have failed for an aberrated lens.  It now appends
  a flat image surface at the paraxial focus and the bar drops from 10 waves to
  1.0 (measured 0.0001).
* `validation/raytrace/test_raytrace.py::t_field_of_view_finite_conjugate`
  asserted the aperture-half-angle formula, i.e. it pinned R7 as correct.  It
  now pins the sensor-driven field of view and separately checks that the legacy
  proxy still returns the old number behind its warning.

### Added -- tests

* `tests/unit/test_audit2609_a1_exit_vertex.py` (17 tests) -- the exit-vertex
  helper against the analytic vertex-plane OPL on sphere / parabola /
  hyperbola / oblate-ellipsoid CURVED-REAR fixtures, alive masking, grazing
  kill, first-failure-wins, idempotence, non-mutation, `refocus(0)` equivalence,
  `n_exit` inference (including the mirror case), and JAX parity +
  grad-safety.
* `tests/unit/test_audit2609_a1_raytrace.py` (48 tests) -- R1..R7, each against
  an independent oracle and each asserting it is no longer the pre-fix value.

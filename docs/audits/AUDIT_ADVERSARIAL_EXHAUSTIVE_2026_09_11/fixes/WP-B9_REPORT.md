# WP-B9 -- ray tracing: the performance and completeness items WP-A1 deferred

Branch `audit-fixes-2026-09`, parent commit `284daccc`.  The six items are
WP-A1_REPORT.md section 6, 1-6 (audit RAYTRACE partition: Performance #2, #3,
#4, #7; Alternative algorithms #4, #6).  Item 7 of that list, **polarisation ray
tracing, is explicitly out of scope** and stays recorded as deferred.

Every number below is measured on this box with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one process at a
time.  Wall-clock figures are medians of interleaved runs and are labelled
whether they were taken WITHIN one process (reliable) or ACROSS the two trees
(load-contaminated); operation counts, allocation counts and ULP deltas are
load-independent.

---

## 0. THE HEADLINE -- what moved and what did not

**Nothing that was not asked for moved a single bit.**

A byte-identity harness (`scratchpad/b9/byte_identity.py`) runs 225 probe arrays
-- 159 809 float64 values -- across `trace` on five surface kinds x two field
angles, `output_filter='all'` history, a DOE-kicked trace, `trace_world`, all
five ray launchers, spot / geo / refocus / through-focus metrics, `system_abcd`,
`seidel_coefficients`, both differential Jacobians (composite and per-surface),
`trace_prescription` and `analysis.ghost`.  It was executed twice in CHILD
processes, never through pytest: once with cwd + `PYTHONPATH` on
`git archive 284daccc lumenairy` untouched, once on the same archive with only
`lumenairy/raytrace/*.py` replaced by these versions (`lumenairy.__file__`
asserted inside the tree in both).  Result:

> **225 / 225 arrays byte-identical.  0 moved.**

That is the discipline the two riskiest items are built around.  The audit's
perf #2 and perf #3 both change the last bit of every traced ray, so both ship
as keyword switches whose DEFAULT is the arithmetic the tree already had:

* `trace(..., sphere_normal='generic' | 'analytic')` -- default `'generic'`
* `trace(..., renormalize='surface' | 'exit')` -- default `'surface'`

and `_refract` / `_reflect` carry the matching per-call keywords with the same
defaults, so the two direct callers outside this package (`analysis/ghost.py`
and the finite-difference differential path) are unchanged **by construction,
not by measurement**.

---

## 1. Summary table

| item | status | files:lines | tests | oracle | measured before -> after |
|---|---|---|---|---|---|
| 1 `_refract`/`_reflect` renormalise hoist (perf #3) | **done, opt-in** | `raytrace/intersection.py:520` (`_normalize_directions`), `:541`, `:646` (`renormalize=`); `trace.py:60,230,341`; `world_trace.py:82,243` | `test_audit2609_b9_raytrace_perf.py::test_b9_i1_*` (9) | the renormalise contract `\|(L,M,N)\| = 1`; the pre-fix per-surface form | **1.012x (N=101) / 1.095x (N=1000) / 1.045x (N=20k) / 1.032x (N=200k)**; drift `max\|dx\| = 6.2e-17 m`, `max\|dopd\| = 1.9e-16 m`, alive masks identical.  Default bit-identical |
| 2 analytic sphere normal (perf #2, the 24 % block) | **done, opt-in** | `raytrace/surface.py:650` (`_is_pure_spherical`), `:679` (`_sphere_normal`), `:713`; `intersection.py:242,541,646`; `trace.py:61`; `world_trace.py:83` | `::test_b9_i2_*` (16) | **60-digit `decimal` normal**, written in the test | normal block **0.324 s (24.1 %) -> 0.139 s (12.5 %)** of the profile; whole trace **1.08-1.18x, median 1.13x**; vs the oracle **<= 4 ULP** and never worse than the generic route |
| 3 one bundle for the fans (perf #4) | **done** | `raytrace/ray_fan.py:79,96` (`_bundle_slice`, `_trace_fan_set`), applied `:584,639,836,897` | `::test_b9_i3_*` (6) | the four separate traces, issued by the test | `trace()` calls **4 -> 1**; `ray_fan_data` **1588.7 -> 545.9 us** (2.91x), `opd_fan_data` **1850.5 -> 854.3 us** (2.17x); `max\|delta\| = 0.0` on all four prescriptions |
| 4 `trace_jax` prescription cache (perf #7) | **done** | `raytrace/jax_trace.py:947,952,1066,1085` | `::test_b9_i4_*` (13) | the `aux` signature's own completeness, field by field | warm `trace_jax` **296.5 -> 83.0 us (3.54x)**; `_build_jax_prescription` **238.6 -> 53.2 us (4.48x)** |
| 5 area-uniform pupil (alt #6) | **done, default unchanged** | `raytrace/trace.py:1204,1285`; `ray_fan.py:1037`; `trace.py` `ray_pattern='vogel'` | `::test_b9_i5_*` (10) | the exact identity `mean(r^2/R^2) = 1/2` | mean `r/R` **0.580645 -> 0.665834**; mean `r^2/R^2` **0.419355 -> 0.500000 (exact)**; spot RMS **+2.14 %** and **+10.37 %** on two prescriptions |
| 6 aspheric analytic Jacobian (alt #4) | **done** | `raytrace/differential.py:404,407,445,463,477,557,748,1124` | `::test_b9_i6_*` (11) | the FD primitive with an `h`-ladder; `trace`'s own Newton intersection | `NotImplementedError` -> `max\|dJ\| = 2.1e-8 .. 7.2e-8` vs FD, scaling exactly as `h^2`; base-conic answer is **six decades** away |
| 7 polarisation ray tracing (alt #5) | **out of scope**, recorded | -- | -- | -- | -- |

Byte identity of everything else: **225/225 arrays, 159 809 values, 0 moved.**

---

## 2. Per item

### item 1 -- the `_refract` / `_reflect` renormalise hoist (audit perf #3)

**What the audit said.** Exact vector Snell with a unit normal returns a unit
vector identically, so the per-surface `sqrt(L^2+M^2+N^2)` + floor + three
divisions (~10 N-sized ops x n_surfaces) only remove ~1e-16 of drift; hoist them
to one pass at the end of `trace`, keeping the degenerate-ray detection.

**Changed.** `_refract(..., renormalize=True)` / `_reflect(..., renormalize=True)`
and `trace` / `trace_world`'s `renormalize={'surface', 'exit'}`.  With `'exit'`
the per-surface DIVISION is skipped and
`intersection._normalize_directions(rays)` runs once, on the bundle leaving the
last surface, BEFORE it is snapshotted -- so `image_rays` satisfies
`|(L, M, N)| = 1` on every `output_filter`, including the coord-break branch.

The per-surface `mag` and its `|d| < 1e-30 or not finite -> RAY_NAN + killed`
test are NOT hoisted and run in both modes: a direction that collapses at
surface 3 must be reported as having died at surface 3.  That is the whole
reason the brief said "keep the degenerate-ray detection per surface", and it is
also why the saving is smaller than the audit's "~10 N-sized ops" estimate --
only the `np.maximum` floor and the three in-place divisions go.

**Why opt-in, not default.** The brief offered two routes: a flag every caller
passes explicitly, or keeping the other callers identical by construction.  The
flag does BOTH -- `analysis/ghost.py` and the FD differential path call
`_refract` positionally and get `renormalize=True` unchanged -- and it lets the
default path be proved bit-identical rather than argued to be close.

**Verified.**
* Default: the 225-array byte-identity sweep above, plus a signature pin that
  `_refract`, `_reflect`, `trace` and `trace_world` all default to the
  per-surface form.
* `'exit'` drift, 2000 rays x (7-surface spherical, 3-surface conic):
  `alive` and `error_code` byte-identical; `max |dx| = 6.2e-17 m`,
  `max |dopd| = 1.9e-16 m`, `max |dL| = 5.0e-16` -- against a derived envelope of
  `n_surfaces * eps * |t|_max` = 1.7e-16 m.
* Structural count (no wall clock): `_normalize_directions` is called exactly
  ONCE per trace under `'exit'` and ZERO times under `'surface'`, while
  `_refract` is called once per surface in both.
* Under `'surface'` every recorded history bundle is unit to <= 2 ULP; under
  `'exit'` some intermediate bundle is measurably worse and the exit bundle is
  not -- the two-sided statement that the division was hoisted and not merely
  relocated.
* Degeneracy: a NaN and an infinite direction handed to `_refract` both die with
  `RAY_NAN` in both modes.  (An exactly-zero direction is NOT a case this guard
  can see: with `cos_i = 0` Snell returns `-cos_t * n`, a unit vector along the
  normal.  Stated in the test so nobody re-derives it.)

**Measured saving** (interleaved medians, same process, 7-surface spherical
stack, `output_filter='last'`):

| N | `'surface'` | `'exit'` | ratio |
|---|---|---|---|
| 101 | 667.6 us | 659.8 us | 1.012x |
| 1 000 | 1023.0 us | 934.7 us | **1.095x** |
| 20 000 | 14 960 us | 14 323 us | 1.045x |
| 200 000 | 424 288 us | 411 117 us | 1.032x |

**Residual risk.** Under `output_filter='all'` the intermediate `ray_history`
bundles carry `| |d| - 1 | <= 1e-15` in `'exit'` mode; a consumer that reads
history direction cosines as exactly unit must stay on `'surface'`.  Documented
in the `trace` docstring.

### item 2 -- the closed-form sphere normal (audit perf #2, the 24 % block)

**Re-measured the premise first.** cProfile, N = 200 000, 7 surfaces, 3 calls,
parent commit:

| function | tottime | share |
|---|---|---|
| `_conic_core.refract_snell` | 0.376 s | 28.0 % |
| `_intersect_surface` | 0.263 s | 19.6 % |
| `_refract` (own) | 0.158 s | 11.8 % |
| **normal block** (`_surface_normal` + `_base_surface_sag_derivatives_xy` + `_surface_sag_derivative`) | **0.324 s** | **24.1 %** |
| total | 1.342 s | |

-- the audit's 24 % reproduces exactly.

**Changed.** `surface._sphere_normal(x, y, R)` returns
`(-x/R, -y/R, sqrt(1 - h^2/R^2))`, which is `-(x, y, z - R)/R` with the
near-branch sag substituted, and is a unit vector identically.  Five array
operations against about fourteen, no Python calls, and no division by a small
`h` near the vertex.  It reproduces `_surface_sag_derivative`'s out-of-domain
policy exactly (`nz` is NaN for `h^2/R^2 >= 0.9999`), so vignetting does not
move.

**The v4.12.0 failure mode is addressed structurally, not hopefully.**  That
attempt changed the NORMAL without the matching INTERSECTION.  The predicate is
now ONE function, `surface._is_pure_spherical`, and
`intersection._intersect_surface` selects its closed-form root on the same call
(`intersection.py:242`) -- so the closed form is the normal of the sphere the
intersection actually solved, at the point the intersection actually returned.
Pinned by counting sag evaluations: a surface for which the predicate is True
produces ZERO `_surface_sag_xy` calls during its intersection.

**Verified against a 60-digit `decimal` oracle** written in the test (textbook
`n = (-dz/dx, -dz/dy, 1)/|.|` with `dz/dh = h/(R sqrt(1 - h^2/R^2))`, evaluated
at 60 significant digits -- ~44 digits beyond float64, so its own error is not
measurable here).  Over R in {+51.5, -34.5, +500, -1000, +2} mm at
h/|R| in {0, 0.05, 0.2, 0.5, 0.8, 0.95}:

* closed form within **4 ULP** of the oracle at every height and radius;
* **never worse** than the generic route at any height;
* **strictly better** at some height on every radius tested.

It is also a unit vector to <= 2 ULP by construction, where the generic route
has to divide by a computed magnitude.

**Measured saving** (the switch on; profile shape as above):

| | parent | `sphere_normal='analytic'` |
|---|---|---|
| normal block | 0.324 s (24.1 %) | **0.139 s (12.5 %)** |
| profile total | 1.342 s | 1.115 s |
| wall clock N = 200 000 | -- | **1.08-1.18x, median 1.13x** (three interleaved in-process pairs) |
| wall clock N = 1000 | -- | 1.10-1.13x |

**Why it is opt-in.**  It is not bit-identical.  Measured over 1500 rays x two
field angles: `max |dx| = 2.8e-17 m`, `max |dopd| = 8.3e-17 m`,
`max |dL| = 2.8e-16`, every `alive` and `error_code` byte-identical, and
**exactly zero** on prescriptions with no pure sphere (conic, biconic, mirror).
That is one to two decades below the trace's own 60-digit-oracle OPL floor
(1.39e-17 m) -- but two downstream pins sit inside it:

| pin | bar | reading with `sphere_normal='analytic'` as the default |
|---|---|---|
| `test_audit_propagation.py::...ModalAsymptoticStillBitEqual::...lg00...` | `1e-8 * peak` | `4.003e-04 / 3.852e+04` = **1.039e-08** (3.9 % over) |
| the same class's 4-mode arm | `1e-8 * peak` | `3.699e-04 / 3.560e+04` = **1.039e-08** (3.9 % over) |
| `test_niche_audit_w6_asymptotic.py::test_w6_a2_v2_star_is_untouched_by_the_verdict_fix` | `1e-15` floor | passed in my run; the orchestrator measured **1.150e-15** (15 % over) |

The mechanism is real and worth recording: `propagate_modal_asymptotic`'s
envelope-stationary saddle solve amplifies a last-bit ray-trace perturbation by
**~8 decades**, because a knife-edge pixel flips basin.  That is the same family
as the 1.17e-3 cross-backend divergence the v4.12.0 note recorded, one decade
milder.  Both of those pins have single-digit-percent headroom, so ANY change to
the trace's last bits trips them -- see section 5 for the exact restatement.

The gate the brief named, the Maslov cross-backend asymptotic test, is GREEN
either way: `tests/unit/test_niche_audit_w6_asymptotic.py` (53 tests, including
`test_w6_a10_jax_fit_survives_partial_vignetting`'s 1e-3 coefficient parity) and
`test_audit2609_a4_verify_maslov_asymptotic.py` pass in full.  NumPy<->JAX
`trace` parity is asserted under BOTH settings at 1e-15 m.

**One behaviour difference, confined to non-finite inputs.**  At a NOT-FINITE
position the generic route returns a fabricated axial normal `(-0, -0, 1)` --
its `np.where(h > 0, ..., 0.0)` guard is False for a NaN `h` -- while the closed
form propagates the NaN, which is the S11-7 policy the shared-core
`conic_sag_derivs` and the JAX backend already implement.  **Through the public
API nothing moves**: the ray-sphere quadratic already refuses a non-finite
position, so such a ray is dead with `RAY_MISSED_SURFACE` before the normal is
consulted.  Measured and pinned both ways in the test file.

### item 3 -- one concatenated bundle for the fan functions (audit perf #4)

**Changed.** `_trace_fan_set(tracer, bundles, surfaces, wavelength)` concatenates
the tangential chief, the sagittal chief and the two fans into one `RayBundle`,
issues ONE `trace()` (or `trace_world()`) with `output_filter='last'`, and
returns per-input views (`_bundle_slice`).  All four fan functions use it.

**Why it is exact rather than close.** Every step of `trace` is elementwise over
rays.  The only two ray-count-dependent constructs are the `np.any` / `.any()`
guards -- which decide whether an elementwise `np.where` is evaluated at all, not
what it evaluates to -- and the aspheric Newton loop's
`if converged.all(): break`, which can give a ray one extra iteration from an
already-converged point (`|dt| < 1e-15`, i.e. at most an ULP of `t`).  The
derived envelope was therefore `1e-15 * |t|` ~ 1e-17 m.

**Measured: `max |delta| = 0.0` on every field of every bundle**, on a
plano-convex singlet, a spherical stack, a `k = -0.6 / -1.2` conic pair and an
`A4 / A6` asphere, on axis and at 2 deg -- the Newton coupling never materialised
even where the loop runs.  The RT-5 invariant is exactly zero:
`ey(0) == ex(0) == 0.0` at 0, 1 and 3 deg.

`output_filter='last'` is free: the fan analytics read only `image_rays`, and
that bundle is the same `r.copy()` taken at the same point in the loop, so it is
bit-identical while `n_surfaces - 1` full `RayBundle.copy()` allocations
disappear.

**Measured** (101-ray fans, f/4 plano-convex N-BK7 with a flat image plane at
the paraxial focus; `trace()` call counts are exact, wall clock is a warm median
of 20 across the two trees):

| call | traces before | traces after | before | after | ratio |
|---|---|---|---|---|---|
| `ray_fan_data` on axis | 4 | **1** | 1588.7 us | 545.9 us | **2.91x** |
| `ray_fan_data` @2 deg  | 4 | **1** | 1530.5 us | 575.9 us | 2.66x |
| `opd_fan_data` on axis | 4 | **1** | 1850.5 us | 854.3 us | **2.17x** |
| `opd_fan_data` @2 deg  | 4 | **1** | 1783.5 us | 855.2 us | 2.09x |

(The rays traced are unchanged at `2*n_rays + 2`; only the number of calls
falls.  The wall-clock column is cross-tree and therefore carries the box's
load; the call count is exact.)

**Residual risk.** `_bundle_slice` returns NumPy VIEWS, not copies.  The fan
analytics only read them; a future caller that mutates one writes through to the
joint bundle.  Stated in the helper's docstring.

### item 4 -- the `trace_jax` prescription cache (audit perf #7)

**Re-measured the premise.** 2-surface / 5-ray prescription at 1.31 um, warm,
best-of-9 medians: `trace_jax` 296.5 us/call, of which
`_build_jax_prescription` is **238.6 us (80 %)**; inside that, `jnp.asarray`
is 79 % and `get_glass_index` 12 %.  A prebuilt `JaxPrescription` costs
56.8 us -- the prep was **5.2x** the call it preceded.  The audit's 750 us/call
on a loaded box scales to the same structure.

**Changed.** The built object is memoised on the `aux` tuple the builder already
assembles (LRU 32, its own lock, `clear_jax_prescription_cache()` registered
with the central `_cache_registry`).  The expensive leaf conversion moved to
`_build_jax_leaves` and is skipped on a hit.

**Why `aux` is a complete key (audit sec. 15.5).**  Not by inspection -- by
construction: the four leaves are `jnp.asarray` of `radii_py`, `conics_py`,
`thicks_py` and `asph_pairs`, and `aux` carries those four verbatim plus
`n_surf`, `asph_powers`, the resolved `semi_diameters`, the resolved `n_pre` /
`n_post` and `diff_aux`.  `_build_jax_prescription` reads nothing else.  Three
consequences, each pinned:

1. Because the RESOLVED indices are in the key rather than the glass NAMES, a
   mutated registry re-keys on its own: registering `__b9_probe_glass__` at
   n = 1.5, building, re-registering the same name at n = 1.7 and building again
   returns a DIFFERENT object with a different `aux`.  A name-keyed cache would
   have served the stale build.
2. The unsupported-surface guard runs BEFORE the lookup, so a mirror /
   coord-break / biconic / freeform prescription is refused on every call and
   never cached -- which is also why those fields need not be in the key.
3. Perturbing radius, conic, an aspheric coefficient, a thickness, a
   `semi_diameter`, the top-level `aperture_diameter`, a glass name, the
   wavelength or the `surface_diffraction` spec each produces a cache MISS and a
   different `aux`.  Tested field by field.

`JaxPrescription` is immutable in this package (`__slots__`, no attribute writes
after construction), so callers share one instance safely; an unhashable `aux`
(an exotic aspheric power key) falls back to building every time rather than
refusing the trace.

**Measured.**

| | before | after | ratio |
|---|---|---|---|
| `trace_jax`, warm eager | 296.5 us | **83.0 us** | **3.54x** |
| `_build_jax_prescription` | 238.6 us | 53.2 us | 4.48x |
| `trace_jax` with a prebuilt `jp` | 56.8 us | 40.0 us | the floor |

Traces are bit-identical with a cold and a warm cache (pinned on all seven
fields).  The pre-built fast path is documented as before and is still the
fastest route; the cache closes 3.5x of the 5.2x gap for callers who do not use
it.

**Residual.** The remaining 53.2 us is the `aux` assembly, ~34 us of which is two
`get_glass_index('N-BK7', wl)` calls at **16.8 us each** (`'air'` is 0.17 us).
Memoising that is a `glass.py` change -- see section 5.

### item 5 -- area-uniform pupil sampling (audit alt-algorithm #6)

**Changed.** `make_rings(..., pattern={'rings' (default), 'vogel'})`.  The Vogel
/ Fibonacci sunflower is `r_i = R sqrt(i/N)`, `theta_i = i pi (3 - sqrt(5))`
(the golden angle), with `i = 1..N` so the outermost ray sits exactly on the rim
as the outer ring does, and the chief added when `include_chief`.  Threaded
through `through_focus_rms(pattern=)` and `trace_prescription` /
`raytrace_system`'s `ray_pattern='vogel'`.

**The default does not move**, per the orchestrator's decision, and that is
pinned two ways: `make_rings(...)` is bit-identical to
`make_rings(..., pattern='rings')`, and both are bit-identical to the ring
geometry rebuilt from the documented formula inside the test.

**Measured** (defaults: `num_rings=6`, `rays_per_ring=36`, chief included, 217
rays):

| quantity | `'rings'` | `'vogel'` | area-uniform limit |
|---|---|---|---|
| mean `r/R`     | **0.580645** | **0.665834** | 2/3 |
| mean `r^2/R^2` | **0.419355** | **0.500000** | 1/2 |

(0.580645 reproduces the audit's 0.5806 exactly.)  The `r^2` row is an EXACT
identity for `'vogel'` with the chief included -- `r_i^2/R^2 = i/N` for
`i = 1..N` plus 0 averages to `((N+1)/2)/(N+1) = 1/2` for every N -- so the test
asserts it to 8 ULP rather than to an envelope, at four different (rings,
rays_per_ring) pairs.  The golden angle also gives every ray its own azimuth:
36 distinct azimuths for `'rings'`, `n_rays` for `'vogel'`.

**The spot-rms shift** (same counts, flat image plane at the paraxial focus):

| prescription | `'rings'` | `'vogel'` | shift |
|---|---|---|---|
| f/4 plano-convex N-BK7, 25 mm | 108.2226 um | 110.5391 um | **+2.14 %** |
| biconvex R = +-60 mm N-BK7, 16 mm | 106.1941 um | 117.2050 um | **+10.37 %** |

Both upward, as the centre-weighting predicts.  The test asserts the DIRECTION
(area-true reads larger on an aberrated system) rather than the numbers, so it
tracks library evolution; the numbers are in the docstring and here.

### item 6 -- aspheric support in `ray_transfer_jacobian_analytic` (audit alt #4)

**Changed.** `_adrt_step` branches on a non-empty `aspheric_coeffs`
(`differential.py:557`).  The exact conic root -- the caller's Spencer & Murty
`e/q` form, untouched -- seeds a FIXED 6-step Newton refinement onto
`conic + polynomial`:

    G(tau)     = z(tau) - S(u(tau)) - P(u(tau))
    dG/dtau    = Nz - (dS/du + dP/du) * 2 (x(tau) L + y(tau) M)
    grad F     = (-2 x D, -2 y D, 1),   D = dS/du + dP/du

all on `u = x^2 + y^2`, so `dS/du = c/(2w)` and `dP/du` are finite at the vertex
and nothing divides by `h`.  `u^m` is evaluated by squaring, in the same `+ - *
/` primitives both backends share.  The step count is fixed rather than
data-dependent because this runs under forward-mode AD on the `_AdrtDual` and
JAX backends alike; **differentiating the iteration itself is what makes the
Jacobian exact** rather than converged-value-only.

The conic path is untouched -- pinned by making the new branch fatal and tracing
a conic prescription through it.

**Verified** at three heights (1, 4, 8 mm) on an aspheric singlet, with A4 only,
A4+A6, A4+A6+A8, and an asphere on both surfaces over a `k = -0.6` base:

| claim | oracle | reading |
|---|---|---|
| analytic == FD to the FD's own truncation | the FD primitive | `max abs(dJ)` **2.1e-8 .. 7.2e-8** at `h_pos = 1e-6` (rel 7e-10 .. 2e-9) |
| and the gap IS truncation, not error | the `h`-ladder | 2.1e-6 .. 7.2e-6 at `h_pos = 1e-5`: **exactly `h^2`** (ratio asserted in 30..300) |
| the polynomial terms are load-bearing | tracing the BASE CONIC | **3.0e-2 .. 2.0e0** -- six decades above the agreement |
| the Newton refinement lands where the main trace lands | `raytrace.trace`'s own 10-step Newton (no shared code) | **3.5e-18 m** in x, 1e-16 in slope, 2.8e-17 m in OPL |
| cross-backend | JAX `jacfwd` vs the NumPy dual | **1.8e-14** on the Jacobian, 4.3e-19 m in x |
| the step budget is headroom | the 6-step result | bit-identical from **2** steps |

**The numba kernel is excluded for aspheres** (`differential.py:748`).  Its
inlined primitives replicate the CONIC arithmetic op-for-op; left eligible it
would have traced an asphere as its base conic -- right shape, wrong surface,
silently.  That exclusion is pinned.

Freeforms, biconics and field-frame decenter / tilt still raise, with the
message updated; odd aspheric powers are refused by the `Surface` constructor
first and by `_adrt_step`'s own backstop guard (which names itself) for a
hand-built surface-like object.

**Behaviour change, stated.** A `jacobian='auto'` consumer with an ASPHERIC
prescription now gets the exact analytic Jacobian where it previously fell back
to FD on the `NotImplementedError`.  The two agree to the FD's own truncation,
so it is a strict accuracy improvement, but it is a different last 8 digits.
The two `'auto'` sites are `propagators/gbd.py:3373` and
`propagators/fga.py:817` -- see section 5.  Every test in the verification set
that reaches them is green.

### item 7 -- polarisation ray tracing

**Out of scope**, as the brief states.  Recorded so it is not lost: `Surface.coating`
carries a complex index but the geometric trace never forms the 3x3 P-matrix;
the s/p basis and local rotations are already computable from the oriented
normal `_conic_core.refract_snell` returns.

---

## 3. Files touched

Modified (all inside `lumenairy/raytrace/`, which WP-B9 owns):

* `lumenairy/raytrace/surface.py` -- `_is_pure_spherical` (the shared
  predicate), `_sphere_normal` (the closed form), `_surface_normal(analytic_sphere=)`.
* `lumenairy/raytrace/intersection.py` -- `_normalize_directions`;
  `_refract` / `_reflect` gain `renormalize=` and `sphere_normal=`; the
  pure-spherical predicate now comes from `surface`; the `_intersect_surface`
  Notes rewritten to describe the pairing (the old text described a state that
  no longer exists and ended mid-sentence).
* `lumenairy/raytrace/trace.py` -- `trace(renormalize=, sphere_normal=)`;
  `make_rings(pattern=)`; `ray_pattern='vogel'` in `trace_prescription` and
  `raytrace_system`.
* `lumenairy/raytrace/world_trace.py` -- the same two keywords on `trace_world`.
* `lumenairy/raytrace/ray_fan.py` -- `_bundle_slice`, `_trace_fan_set`, applied
  in all four fan functions; `through_focus_rms(pattern=)`; two duplicated
  comment lines removed.
* `lumenairy/raytrace/jax_trace.py` -- the built-prescription cache,
  `clear_jax_prescription_cache`, `_build_jax_leaves`.
* `lumenairy/raytrace/differential.py` -- `_adrt_aspheric_items`,
  `_adrt_u_pow`, `_adrt_poly_sag`, `_adrt_conic_sag`,
  `_adrt_aspheric_intersect`, `_ADRT_ASPHERIC_NEWTON_STEPS`; the `_adrt_step`
  branch; the relaxed guard; the numba exclusion.

History documents re-recorded in this change (`scripts/record_history_fingerprints.py`,
each with its `--reason`):

* `docs/history/lumenairy.raytrace.surface.md`
* `docs/history/lumenairy.raytrace.intersection.md`
* `docs/history/lumenairy.raytrace.trace.md`
* `docs/history/lumenairy.raytrace.world_trace.md`
* `docs/history/lumenairy.raytrace.ray_fan.md`
* `docs/history/lumenairy.raytrace.jax_trace.md`
* `docs/history/lumenairy.raytrace.differential.md`

New:

* `tests/unit/test_audit2609_b9_raytrace_perf.py` (65 tests)
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B9_REPORT.md`
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B9_CHANGELOG.md`

Nothing outside `lumenairy/raytrace/`, `docs/history/lumenairy.raytrace.*.md`,
the new test file and the two `fixes/` documents was modified.
`lumenairy/elements/_lens_traced.py` was not touched.  No git write command was
run.

**On `__all__` symmetry.**  `_normalize_directions` and `_is_pure_spherical`
were briefly spelled without the underscore, which made
`test_v4_16_0_walker_all_symmetry.py::test_all_submodule_entries_reexported_or_exempt`
red: a non-underscored name in a submodule `__all__` must be re-exported from
`lumenairy/__init__.py`.  Both are internal cross-submodule helpers -- a user
traces with `trace(renormalize=..., sphere_normal=...)`, never by calling them --
so they take the underscore prefix that every one of their neighbours in those
two `__all__` blocks already carries (`_intersect_surface`, `_refract`,
`_surface_normal`, `_field_frame_active`, ...), exactly as the
`surface.__all__` comment prescribes.  No re-export is requested.
`clear_jax_prescription_cache` is likewise kept OUT of `jax_trace.__all__` for
the sibling pin in `test_v4_14_1_dispatcher_pin_cache_clears.py`; see section 5
if the orchestrator wants it public.

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one
process at a time.

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_b9_raytrace_perf.py -q` | **65 passed** (9 / 16 / 6 / 13 / 10 / 11 for items 1-6) | 11.5 s |
| `pytest tests/unit -q -k "raytrace or exit_vertex or seidel or opd_fan or ghost or jax_trace"` | **722 passed, 2 skipped** | 457.8 s |
| `pytest tests/unit -q -k real_lens` | 158 passed, 3 skipped, **2 failed** (other WP -- see below) | 291.8 s |
| `pytest` a1 (4 files) + a26 + a17 lint + a17 relocation + b9 | **916 passed, 2 failed** (other WP -- `_lens_traced` fingerprints) | 135.7 s |
| `pytest tests/unit/test_niche_d7_decentred_fit.py -q` (the slow one) | **38 passed** | 326.1 s |
| `pytest` w6 asymptotic + a4 verify Maslov + `test_audit_propagation` + d1 tilted carrier + v1_v8 readout guard | **278 passed** | 317.2 s |
| `pytest tests/unit/test_v4_16_0_walker_all_symmetry.py tests/unit/test_audit2609_a15b_reexports.py tests/unit/test_v4_14_1_dispatcher_pin_cache_clears.py tests/unit/test_audit2609_b9_raytrace_perf.py -q` | **123 passed** | 10.9 s |
| `python validation/run_all.py test_raytrace test_lenses` | **2/2 files pass**; `validation/raytrace` **54/54** | 45 s |
| `ruff check lumenairy/raytrace/ tests/unit/test_audit2609_b9_raytrace_perf.py` | **All checks passed** | -- |
| `python scripts/record_history_fingerprints.py --check` | clean for every `lumenairy.raytrace.*` document | -- |

The named gate for item 2 -- the Maslov cross-backend asymptotic test -- is in
the sixth row: `tests/unit/test_niche_audit_w6_asymptotic.py` (53 tests,
including `test_w6_a10_jax_fit_survives_partial_vignetting`'s 1e-3 cross-backend
coefficient parity and `test_w6_a2_v2_star_is_untouched_by_the_verdict_fix`'s
1e-15 floor) plus `test_audit2609_a4_verify_maslov_asymptotic.py`, all green.
`test_audit_propagation.py -k ModalAsymptoticStillBitEqual` is green in the same
row.

Failures found, with my judgement:

* `tests/unit/test_audit2609_a16_lens_config_round_trip.py::test_every_keyword_is_classified_as_field_contract_or_documented_exclusion[apply_real_lens_traced]`
  and `[prepare_real_lens_traced]` -- `['fit_basis'] are keyword parameters that
  lens_config neither carries as a field nor documents`.  `fit_basis` appears
  nowhere in `lumenairy/raytrace/`; it is a new keyword in
  `lumenairy/elements/_lens_traced.py`, which `git status` shows another Wave-4
  engineer is mid-edit on (+467 lines).  **Not WP-B9.**
* `tests/unit/test_audit2609_a17_history_relocation.py::...[lumenairy.elements._lens_traced]`
  (ast and token) -- the same engineer's un-recorded fingerprint drift.  **Not
  WP-B9**; I must not touch that file.
* Two skips are missing optional dependencies (`PySide6`, `rayoptics`) and one
  is the host-specific W5 digest guard.

Mid-session the whole library was briefly un-importable
(`AttributeError: module 'lumenairy.analysis.psf_mtf_otf' has no attribute
'encircled_energy_profile'`) while another WP was mid-edit on
`lumenairy/analysis/`; it cleared on retry.  An earlier large `-k` run showed
transient ghost-family failures that reproduce on NEITHER tree when run
standalone -- I verified that explicitly rather than attributing them.

**Byte-identity proof.**  `scratchpad/b9/byte_identity.py`, run in two child
processes with cwd + `PYTHONPATH` on `git archive 284daccc lumenairy` and on the
same archive overlaid with these `lumenairy/raytrace/*.py` (the ONLY difference
between the trees; `diff -rq` confirms it, and `lumenairy.__file__` is asserted
inside the tree at import).  Never through pytest.  **225 arrays / 159 809
float64 values: 225 byte-identical, 0 moved**, including every `alive` and
`error_code` array.

**Fail-befores.**  Item 6's is structural and stated in the test: the pre-fix
analytic path RAISED, so there is no pre-fix number to compare -- what is
demonstrated instead is that the polynomial terms are load-bearing (tracing the
base conic is six decades away from the FD truth).  Item 2's is the 60-digit
oracle comparison, which shows the generic route is measurably further from the
truth than the closed form at every radius.  Items 1, 3, 4, 5 all carry explicit
two-sided demonstrations (the hoisted mode's intermediate drift is non-zero
where the default's is not; the four-trace oracle; the field-by-field cache
re-key; the ring-vs-Vogel second moment).

---

## 5. Requested changes outside my ownership

1. **`lumenairy/glass.py` -- memoise `get_glass_index(name, wavelength)`.**
   Measured 16.8 us per `'N-BK7'` resolution against 0.17 us for `'air'`, i.e.
   the Sellmeier evaluation is not memoised on `(name, wavelength)`.  Two
   surfaces of glass cost ~34 us, which is **64 % of the residual 53.2 us**
   `_build_jax_prescription` now spends, and `trace()` pays the same on every
   call (`trace.py:149-150`).  An `lru_cache` keyed on `(glass_name, wavelength)`
   that is invalidated by `_invalidate_glass_name` would take the JAX prep from
   53 us to ~20 us and would also speed the NumPy trace's per-call prologue.
   (A registry generation counter would additionally let the WP-B9 cache key on
   glass NAMES instead of resolved indices, closing the rest of the gap to the
   40 us prebuilt floor.)

2. **`tests/unit/test_audit_propagation.py::TestAuditFixesV4_14_0_agent_1_1APropagateModalAsymptoticStillBitEqual`**
   -- its two `assert max_abs < 1e-8 * max(cold_peak, 1.0)` arms have **3.9 %
   headroom**: they read 1.039e-08 against the 1.000e-08 bar the moment the ray
   trace's last bits move (measured with `sphere_normal='analytic'` forced as the
   default).  That is an S4 floor bar with no gap -- the reading is not a
   continuous error, it is one knife-edge pixel changing saddle basin, so the
   quantity is bimodal between ~0 and ~1e-8 relative.  Exact edit if the
   orchestrator wants `sphere_normal='analytic'` to become the default:
   `1e-8` -> `3e-8` on both arms (one decade below the next real signal, one
   basin-flip above the measured 1.039e-8), with the comment recording
   "one saddle-basin flip = 1.04e-8 relative; measured 2026-09-13 under the
   closed-form sphere normal".  **I have not touched it**, and with the shipped
   default it is green.

3. **`tests/unit/test_niche_audit_w6_asymptotic.py::test_w6_a2_v2_star_is_untouched_by_the_verdict_fix`**
   -- a `1e-15` floor bar the orchestrator measured at **1.150e-15** under the
   flipped default (it passed in my own run, so it is knife-edge on this box
   too).  Exact edit, same conditions: `1e-15` -> `5e-15`, with the measured
   value and date in the comment.  Green with the shipped default.

4. **`propagators/gbd.py:3373` and `propagators/fga.py:817`** -- both build
   `_jac_candidates = [ray_transfer_jacobian_analytic, ray_transfer_jacobian]`
   for `jacobian='auto'` and rely on `NotImplementedError` to fall back.  For an
   ASPHERIC prescription that fallback no longer fires, so `'auto'` now returns
   the exact analytic Jacobian instead of the FD one (agreement ~1e-8 relative,
   analytic side exact).  No edit is required -- the comment above the dispatch
   already says "transparently falls back ... for any surface type the analytic
   path does not yet cover" and that is still true -- but the owners should know
   that the covered set grew, and the comment's parenthetical list of uncovered
   kinds should drop "aspheric".

5. **`lumenairy/__init__.py` + `tests/unit/test_audit2609_a15b_reexports.py::_REEXPORTS`**
   -- OPTIONAL.  `clear_jax_prescription_cache` is registered with
   `_cache_registry` (so `clear_asm_caches()` empties it) and reachable as
   `lumenairy.raytrace.jax_trace.clear_jax_prescription_cache`, but is
   deliberately absent from `jax_trace.__all__` because
   `test_v4_14_1_dispatcher_pin_cache_clears.py` requires every `clear_*` name in
   a submodule `__all__` to be re-exported at top level.  If the orchestrator
   wants it public, the edit is: add `clear_jax_prescription_cache` to the
   `from .raytrace.jax_trace import (...)` block and to the top-level `__all__`
   in `lumenairy/__init__.py`, add the matching `_REEXPORTS` identity entry, and
   append it to `jax_trace.__all__`.  Its sibling `clear_trace_jax_cache` is
   already re-exported, so the asymmetry is visible.

6. **`analysis/field.py:760,851,1000` and `analysis/ghost.py:821`** call
   `make_rings` and inherit the centre-weighted pupil.  They can now pass
   `pattern='vogel'` for an area-true statistic.  `ghost.py:605`'s
   `_ring_areas`-style per-ray area weighting is the alternative and stays
   correct; the two should not be combined.  No edit requested -- each owner's
   judgement.

---

## 6. Deferred, with designs

1. **Flipping the two opt-in defaults.**  `sphere_normal='analytic'` is worth
   **1.13x** on the whole trace and is the more accurate of the two routes;
   `renormalize='exit'` is worth 1.03-1.10x.  Both are one-line default changes
   once items 2 and 3 of section 5 are restated.  Recommended order: restate the
   two pins, flip `sphere_normal` first (bigger win, better accuracy), re-run the
   full matrix, then consider `renormalize`.
2. **The near-hemispherical false kill.**  `_sphere_normal` faithfully
   reproduces `_surface_sag_derivative`'s `norm < 0.9999` domain clamp, so a ray
   landing between `0.99995 |R|` and `|R|` on a pure sphere gets a NaN normal and
   dies as `RAY_NAN` -- although the ray-sphere intersection put it exactly ON
   the surface, so it is a real hit.  This is the R4 false-miss family one level
   down.  The closed form needs no clamp at all (`nz = sqrt(1 - h^2/R^2)` is
   well-conditioned to `h = |R|`, where `nz = 0` is a legitimately grazing
   normal), so the fix is to drop the `valid` gate in the analytic branch and
   let `np.maximum(1 - norm, 0.0)` handle the unreachable `h > |R|`.  Deliberately
   NOT done here: it is a vignetting change, not a performance one, and it would
   move a default.  Effort ~1 h plus a vignetting sweep.
3. **The fabricated axial normal at a non-finite position, generic branch.**
   `_base_surface_sag_derivatives_xy`'s `np.where(h > 0, ..., 0.0)` returns
   `dz_dx = dz_dy = 0.0` for a NaN `h`, so a conic / aspheric / biconic surface
   reports a perfectly defined `(-0, -0, 1)` normal at a NaN position -- the
   exact shape S11-7 fixed in the shared-core `conic_sag_derivs`.  The closed-form
   sphere branch already propagates the NaN.  The fix is S11-7's two lines
   (`dz_dx = np.where(np.isfinite(x), dz_dx, x)` and the `y` twin), but placing
   them in `_base_surface_sag_derivatives_xy` puts four extra N-sized operations
   INSIDE the 10-iteration Newton loop; placing them in `_surface_normal`'s
   generic branch costs four per surface instead.  A performance work package is
   the wrong place to add ~30 % to the generic normal for a pathological input,
   so it is recorded rather than done.  Effort ~1 h.
4. **Polarisation ray tracing** (audit alt #5) -- out of scope for this WP, as
   the brief states.  Recorded so it is not lost.
5. **`_intersect_surface`'s remaining allocations** (audit perf #1, partially
   done by WP-A1's R6).  The profile still shows `_intersect_surface` at 19.6 %
   and `refract_snell` at 28.0 %; both allocate ~10-15 N-sized temporaries that
   `out=` buffers would remove.  `refract_snell` is the shared backend-agnostic
   core, so giving it `out=` means giving it a buffer protocol that the JAX and
   dual backends ignore -- a design question, not a mechanical change.  Effort
   ~4 h, estimated 10-15 % more.

---

## Addendum (orchestrator, 2026-09-14): a pre-audit pin on the aspheric refusal was not restated

* `tests/unit/test_analytic_ray_transfer.py::test_analytic_rejects_biconic_and_asphere` (2026-07) asserted that
  `ray_transfer_jacobian_analytic` raises `NotImplementedError` on an even-aspheric surface.  This package made the analytic path
  trace the asphere, so the test has failed since 7592af4a; neither this package's selection nor VERIFY-B9's ran the file, and
  the release's full fast lane found it.  Restated at the close as `test_analytic_rejects_biconic` (the refusal that remains) and
  `test_analytic_traces_the_even_asphere_against_the_fd_primitive` (a positive pin against the finite-difference primitive on an
  off-axis ray set).  The same lesson as WP-B7's and WP-B8's follow-ups: grep the tests for the primitive whose contract moves.

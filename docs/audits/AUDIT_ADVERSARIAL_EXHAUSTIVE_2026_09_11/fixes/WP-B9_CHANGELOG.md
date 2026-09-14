# WP-B9 changelog text (ray tracing -- the performance and completeness
# items WP-A1 deferred)

Assembled by the orchestrator into `CHANGELOG.md` for 5.47.0.  Item IDs are
from `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A1_REPORT.md`
section 6 (1-6) and the audit's RAYTRACE partition (Performance #2/#3/#4/#7,
Alternative algorithms #4/#6).

**No default moves in this release.**  Every one of the six items either keeps
the arithmetic it found (proved byte-identical below) or is reachable only
through a new keyword whose default is the old behaviour.

### Performance -- raytrace: both fan functions trace once instead of four times (audit RAYTRACE perf #4)

`ray_fan_data`, `ray_fan_data_world`, `opd_fan_data` and `opd_fan_data_world`
each issued FOUR `trace()` calls -- a tangential chief, a sagittal chief, a
tangential fan and a sagittal fan -- through the same surfaces at the same
wavelength, paying the per-call glass resolution and the Python surface loop
four times over on bundles of 1, 1, `n_rays` and `n_rays` rays.  They now
concatenate the four launches into one bundle, trace it once and slice the
result (`raytrace/ray_fan.py:96` `_trace_fan_set`, `:79` `_bundle_slice`;
applied at `:584`, `:639`, `:836`, `:897`).

This is EXACT, not an approximation: every step of `trace` is elementwise over
rays, so a ray's trajectory cannot depend on what it is bundled with.  The two
ray-count-dependent constructs are the `np.any` guards (which only decide
whether an elementwise `np.where` is evaluated -- same values either way) and
the aspheric Newton loop's `if converged.all(): break`, which can give a ray one
extra iteration from an already-converged point (`|dt| < 1e-15`).  Measured
against the four separate traces on a plano-convex singlet, a cemented-class
spherical stack, a `k = -0.6 / -1.2` conic pair and an `A4 / A6` asphere, on
axis and at 2 deg: **`max |delta| = 0.0` on every field of every bundle**.  The
RT-5 invariant `ey(0) == ex(0) == 0` is exactly zero as before.

The single trace also passes `output_filter='last'`, since the fan analytics
read only `image_rays`; that drops one full `RayBundle.copy()` per surface and
leaves `image_rays` bit-identical (it is the same `r.copy()`, taken at the same
point in the loop).

Measured (101-ray fans, f/4 plano-convex N-BK7 singlet with a flat image plane
at the paraxial focus, warm medians of 20, this box):

| call | before | after | ratio |
|---|---|---|---|
| `ray_fan_data` on axis  | 1588.7 us | 545.9 us | 2.91x |
| `ray_fan_data` @2 deg   | 1530.5 us | 575.9 us | 2.66x |
| `opd_fan_data` on axis  | 1850.5 us | 854.3 us | 2.17x |
| `opd_fan_data` @2 deg   | 1783.5 us | 855.2 us | 2.09x |

`trace()` calls per fan function: **4 -> 1**, on the same 2*`n_rays`+2 rays.

### Performance -- raytrace: `trace_jax` no longer rebuilds its prescription on every call (audit RAYTRACE perf #7)

`trace_jax`'s eager cache saved the XLA compile but not the Python prep: every
call re-ran `_build_jax_prescription` -- glass lookups, `_resolve_semi_diameters`
and a `jnp.asarray` of every pytree leaf -- before reaching the kernel lookup.
Instrumented on a 2-surface / 5-ray prescription at 1.31 um (warm, best-of-9
medians): **296.5 us per warm call, of which `_build_jax_prescription` was
238.6 us**, and 79 % of THAT was the five `jnp.asarray` leaf conversions.
Handing `trace_jax` an already-built `JaxPrescription` cost 56.8 us, i.e. the
prep was 5.2x the work of the call it preceded.

The built `JaxPrescription` is now memoised (`raytrace/jax_trace.py:961`, LRU of
32, guarded by its own lock, cleared by `clear_jax_prescription_cache()` at
`:952` and registered with the central `_cache_registry` so `clear_asm_caches()`
reaches it).  Leaf construction moved to `_build_jax_leaves` (`:1085`); the
lookup happens after the cheap Python-float work at `:1066`.

**The key is the `aux` tuple the builder assembles anyway, and it is complete by
construction** (audit sec. 15.5): the leaves are `jnp.asarray` of `radii_py` /
`conics_py` / `thicks_py` / `asph_pairs`, and `aux` carries those four verbatim
plus `n_surf`, `asph_powers`, the resolved `semi_diameters`, the resolved
`n_pre` / `n_post` and `diff_aux`.  Nothing else is read.  Because the RESOLVED
glass indices (not the glass NAMES) are in the key, a mutated glass registry --
`register_fixed_glass`, `trace._register_fixed_index` -- re-keys on its own and
can never serve a stale build; that is pinned.  The unsupported-surface guard
runs BEFORE the lookup, so a mirror / coord-break / biconic / freeform
prescription is refused on every call and never cached.

| call | before | after | ratio |
|---|---|---|---|
| `trace_jax`, warm eager        | 296.5 us | 83.0 us | **3.54x** |
| `_build_jax_prescription`      | 238.6 us | 53.2 us | 4.48x |
| `trace_jax` with a prebuilt jp | 56.8 us  | 40.0 us | (the floor) |

The residual 53.2 us is the `aux` assembly, of which ~34 us is two
`get_glass_index('N-BK7', wl)` calls at 16.8 us each -- see the requested
change in the WP-B9 report.

### Added -- raytrace: `trace(renormalize='exit')`, the hoisted direction rescale (audit RAYTRACE perf #3)

Exact vector Snell with a unit normal returns a unit vector identically, so the
per-surface `sqrt` + floor + three divisions in `_refract` / `_reflect` only
remove ~1e-16 of rounding drift.  `trace` and `trace_world` gain
`renormalize={'surface' (default), 'exit'}` (`raytrace/trace.py:60`,
`world_trace.py:82`); `_refract` / `_reflect` gain the matching
`renormalize: bool = True` (`intersection.py:541`, `:646`), and the single-pass
form is `intersection._normalize_directions` (`:520`), applied once to the
bundle leaving the last surface (`trace.py:230`, `:341`, `world_trace.py:243`).

The degenerate-direction DIAGNOSIS is not hoisted: the per-surface
`|d| < 1e-30 or not finite -> RAY_NAN + killed` test runs in both modes, because
a direction that collapses at surface 3 must be reported as having died at
surface 3.

`'exit'` is NOT bit-identical -- the surviving drift enters the next surface's
ray-sphere quadratic, which assumes `a = |d|^2 = 1`.  Measured on a 2000-ray
7-surface spherical stack and a 3-surface conic stack: identical `alive` masks
and error codes, `max |dx| = 6.2e-17 m`, `max |dopd| = 1.9e-16 m`,
`max |dL| = 5.0e-16`.  Under `output_filter='all'` the INTERMEDIATE
`ray_history` bundles carry `| |d| - 1 | <= 1e-15`; only the final bundle is
rescaled.

Measured saving (interleaved medians, same process, 7-surface spherical stack):
**1.012x at N=101, 1.095x at N=1000, 1.045x at N=20 000, 1.032x at N=200 000**.
The default stays `'surface'`: a 3-9 % saving does not pay for moving every
traced number in the library by 1e-16.

### Added -- raytrace: `trace(sphere_normal='analytic')`, the closed-form sphere normal (audit RAYTRACE perf #2, the 24 % block)

The surface-normal block (`_surface_normal` -> `_base_surface_sag_derivatives_xy`
-> `_surface_sag_derivative`) is **24 % of `trace`'s own time** -- re-measured
here at 0.324 s of a 1.342 s profile (cProfile, N = 200 000, 7 surfaces, 3
calls).  For a pure sphere it computes `sqrt(x^2+y^2)`, a `np.where(h > 0, ...)`
guard, two divisions by `h`, a second `sqrt`, and then a normalising `sqrt` plus
three divisions: about fourteen N-sized array operations and three Python calls
for a vector that is available in closed form as

    n = -(x, y, z - R) / R = (-x/R, -y/R, sqrt(1 - h^2/R^2))

-- five array operations, no Python calls, and no division by a small `h` near
the vertex.  `raytrace/surface.py:679` `_sphere_normal`; selected by
`_surface_normal(..., analytic_sphere=True)` (`:713`) and reachable from
`trace(sphere_normal='analytic')` / `trace_world(...)` (`trace.py:61`,
`world_trace.py:83`) and from `_refract` / `_reflect`'s `sphere_normal=` keyword.

**The v4.12.0 attempt failed because the analytic NORMAL was applied without the
matching INTERSECTION.**  Both now select on ONE predicate,
`surface._is_pure_spherical` (`surface.py:650`), which
`intersection._intersect_surface` also uses for its closed-form root
(`intersection.py:242`) -- so the closed form is the normal of the sphere the
intersection actually solved, at the point the intersection actually returned.
Verified against a 60-digit `decimal` oracle on R = +-2 mm .. 1 m at heights up
to 0.95|R|: the closed form is within **4 ULP** of the oracle, is never worse
than the generic route at any height, and is strictly better at some height on
every radius tested.

Profile with the switch on (same run shape): the normal block falls from
**0.324 s (24.1 %) to 0.139 s (12.5 %)** of the trace, and the whole call is
**1.08-1.18x faster (median 1.13x over three interleaved in-process pairs)** at
N = 200 000, 1.10-1.13x at N = 1000.

It is OPT-IN because it is not bit-identical: measured over 1500 rays x 2 field
angles, `max |dx| = 2.8e-17 m`, `max |dopd| = 8.3e-17 m`, `max |dL| = 2.8e-16`,
with every `alive` and `error_code` byte-identical, and EXACTLY ZERO on
prescriptions with no pure sphere (conic, biconic, mirror).  Two downstream pins
sit inside that: `propagate_modal_asymptotic`'s two `1e-8`-relative comparisons
read **1.039e-08 against a 1.000e-08 bar** (3.9 % over) with the switch on,
because the modal-asymptotic saddle solve amplifies a last-bit ray perturbation
by ~8 decades.  Flipping the default is an orchestrator decision that requires
restating those pins; see the WP-B9 report.

### Added -- raytrace: `make_rings(pattern='vogel')`, area-uniform pupil sampling (audit RAYTRACE alt-algorithm #6)

`make_rings` is equal-radius / equal-count, so the pupil areal sampling density
falls off as `~1/r` and every unweighted `spot_rms` built on it is centre-biased
small.  It gains `pattern={'rings' (default), 'vogel'}` (`raytrace/trace.py:1204`,
generator at `:1285`): the Vogel / Fibonacci sunflower `r_i = R sqrt(i/N)`,
`theta_i = i pi (3 - sqrt(5))`, with `i = 1..N` so the outermost ray sits exactly
on the rim as the outer ring does.  Threaded through
`through_focus_rms(pattern=)` (`ray_fan.py:1049`) and
`trace_prescription` / `raytrace_system`'s `ray_pattern='vogel'`.

Measured at the defaults (`num_rings=6`, `rays_per_ring=36`, chief included, 217
rays):

| quantity | `'rings'` | `'vogel'` | area-uniform limit |
|---|---|---|---|
| mean `r/R`     | 0.580645 | 0.665834 | 2/3 |
| mean `r^2/R^2` | 0.419355 | 0.500000 | 1/2 |

The `r^2` row is EXACT for `'vogel'`: `r_i^2/R^2 = i/N` for `i = 1..N` plus the
chief's 0 averages to `((N+1)/2)/(N+1) = 1/2` for every N.

**The default does not move.**  Every spot number the library has ever published
carries the ring weighting; measured at the same counts, `spot_rms` reads
**+2.14 %** on an f/4 plano-convex N-BK7 singlet (108.2226 -> 110.5391 um) and
**+10.37 %** on a biconvex R = +-60 mm (106.1941 -> 117.2050 um).  `'vogel'` is
the opt-in for an area-true statistic.

### Added -- raytrace: the analytic ray-transfer Jacobian handles aspheres (audit RAYTRACE alt-algorithm #4)

`ray_transfer_jacobian_analytic` raised `NotImplementedError` for any
`aspheric_coeffs`, so every `jacobian='auto'` consumer silently fell back to the
finite-difference primitive there -- 9 traced rays per base ray and ~4e-8 of
truncation.  `_adrt_step` now carries the even-power polynomial departure
(`raytrace/differential.py:557`): the exact conic root seeds a FIXED 6-step
Newton refinement onto `conic + polynomial`
(`:477` `_adrt_aspheric_intersect`, `:445` `_adrt_poly_sag`, `:463`
`_adrt_conic_sag`), and the normal comes from the implicit `F = z - S(u) - P(u)`
with `grad F = (-2x D, -2y D, 1)`, `D = dS/du + dP/du`, all on `u = x^2 + y^2` so
nothing divides by `h` at the vertex.  The step count is fixed because this runs
under forward-mode AD on both backends; differentiating the iteration itself is
what makes the Jacobian exact rather than converged-value-only.

Verified against the FD primitive on an aspheric singlet at three heights
(1, 4, 8 mm), with A4 only, A4+A6, A4+A6+A8, and an asphere on both surfaces over
a `k = -0.6` base:

| quantity | reading |
|---|---|
| `max abs(J_analytic - J_FD)`, `h_pos = 1e-6` | 2.1e-8 .. 7.2e-8 (rel 7e-10 .. 2e-9) |
| the same at `h_pos = 1e-5` | 2.1e-6 .. 7.2e-6 -- **exactly `h^2`**, i.e. the gap IS the FD truncation |
| tracing the BASE CONIC instead | 3.0e-2 .. 2.0e0, **six decades** away |
| exit state vs the FD base ray (`trace`'s own 10-step Newton) | 3.5e-18 m in x, 1e-16 in slope, 2.8e-17 m in OPL |
| JAX backend vs the NumPy dual backend | 1.8e-14 on the Jacobian, 4.3e-19 m in x |
| bit-identical from | 2 Newton steps (shipped budget: 6) |

The numba forward-AD kernel is EXCLUDED for aspheric surfaces
(`differential.py:748`): its inlined primitives replicate the CONIC arithmetic
only, so left eligible it would have traced an asphere as its base conic --
right shape, wrong surface, silently.  Freeforms, biconics
(`radius_y` / `conic_y` / `aspheric_coeffs_y`) and field-frame decenter / tilt
still raise (`:1124`), with the message updated to say so.

Migration note: a `jacobian='auto'` consumer with an ASPHERIC prescription now
gets the exact analytic Jacobian where it used to get the FD one.  The two agree
to the FD's own truncation (~1e-8 relative), so this is a strict accuracy
improvement, but it is a different last 8 digits -- `propagators/gbd.py` and
`propagators/fga.py` are the two `'auto'` sites.
* Restated at the release close: the pre-audit pin
  `tests/unit/test_analytic_ray_transfer.py::test_analytic_rejects_biconic_and_asphere` demanded the
  refusal this entry removes and had failed since the package landed (no selection of the package or
  its verifier ran the file); it is now `test_analytic_rejects_biconic` plus
  `test_analytic_traces_the_even_asphere_against_the_fd_primitive`, a positive pin against the
  finite-difference primitive on an off-axis ray set.

### Fixed -- raytrace: two duplicated docstring lines

`ray_fan.py`'s RT-5 comment repeated `# heights` and `opd_fan_data`'s Notes
repeated the `W_plane - W_true = eps sin(theta')` line; `through_focus_rms`
repeated its "naming neither this function nor the offending argument" comment.
Text only.

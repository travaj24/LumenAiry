# VERIFY-WP-B9 -- release text

Three `lumenairy/raytrace/*.py` files changed.  No default moved, no
public signature changed, and the change is byte-neutral: the
VERIFY-WP-B9 probe (1542 arrays / 200 904 values, child processes on
extracted `git archive` trees) reads identically with and without it.

### Fixed -- `_trace_fan_set` preserves a dead launch ray's own error code

`raytrace.ray_fan._trace_fan_set` built the concatenated bundle's
`error_code` as `np.zeros(...)` for any input whose `error_code` was
`None`.  `RayBundle.__post_init__` synthesises `error_code` from `alive`
(a dead row with no code becomes `RAY_TIR`) so no input ever reaches that
branch today, but had it fired it would have relabelled an
already-dead launch ray `RAY_OK` -- the opposite of the first-failure-wins
contract the rest of the tracer keeps.  The joint bundle now reads each
input's own `error_code` array, which is what every caller already
supplies.  Byte-identical on every prescription measured.

### Fixed -- three source comments that claimed more than the code delivers

Comments describe current behaviour, so three that overstated it are now
measured statements:

* `raytrace.surface._sphere_normal` claimed "no cancellation" and that it
  "reproduc[es] the out-of-domain policy of `_surface_sag_derivative`
  exactly".  Neither holds at the edges.  `nz = sqrt(1 - h^2/R^2)`
  cancels as `h -> |R|`: against a 60-digit `decimal` oracle both the
  closed form and the sag-derivative route are within 4 ULP up to
  `h = 0.95 |R|` and both leave that bar above it, reaching ~1.5e-14
  (68 ULP) at `h = 0.99994 |R|`, with the closed form 4.3x further from
  the truth than the generic route at `R = -34.5 mm` there.  And the two
  domain gates are evaluated from different expressions -- `(x*x + y*y) /
  (R*R)` against `(1 + conic) * sqrt(x*x + y*y)**2 / R**2` -- so within
  ~1 ULP of `h^2 = 0.9999 R^2` they can straddle the threshold: at
  `R = 51.68 mm`, `x = y = 0.036541451242116801 m` the closed form
  refracts the ray and the generic route kills it `RAY_NAN`.  Reachable
  only under `sphere_normal='analytic'`; the shipped default is unmoved.
* `raytrace.surface._surface_normal`'s `analytic_sphere` parameter no
  longer says the closed form is "the more accurate of the pair" without
  qualification.
* `raytrace.jax_trace`'s built-prescription cache claimed
  `JaxPrescription` is "immutable ... so callers share one instance
  safely".  `__slots__` blocks new attribute NAMES, not writes to
  declared ones: `jp.radii = None` on a cached instance succeeds and is
  then served to every later caller.  The note now states the real
  contract (the entry is shared; treat it as read-only), records that
  `aux` is hashable by construction so the `TypeError` fallback is
  defensive only, records that a NaN radius re-read from the same
  prescription dict HITS on object identity, and records the three
  inputs the key deliberately does not cover because the built object
  does not read them (the trace wavelength beyond the indices it
  resolves, the trailing thickness, and an `aperture_diameter` a
  per-surface `semi_diameter` shadows).
* `raytrace.ray_fan._trace_fan_set`'s exactness argument said the extra
  Newton iteration "moves `t` by at most an ULP" by pointing at the
  loop's `|dt| < 1e-15` test.  That test is an ABSOLUTE metre tolerance,
  not an ULP of `t`; the ULP statement is true because a converged ray's
  own residual step is `~eps |t|`, and the note now says so and records
  the 90-case sweep across `|t| = 1e-15/eps = 4.5 m` that found no
  coupling.

### Added -- ten VERIFY-WP-B9 pins

`tests/unit/test_audit2609_b9_raytrace_perf.py` gains ten tests, each
with its own oracle: the 60-digit `decimal` normal (domain-gate straddle;
the conditioning limit above `0.95 |R|`), four separate `trace()` calls
(the fan set with dead rays in a sub-fan; the fan set across the
absolute-Newton-tolerance scale), `jax.jacfwd` through the independent
`trace_jax` kernel (the aspheric analytic Jacobian to 1e-12, measured
1.1e-14 .. 5.7e-14), and the cache's own behaviour (shared-and-rebindable
instance, hashable-by-construction `aux`, the key's exact scope, NaN-radius
keying, and a zero aspheric coefficient selecting the Newton branch and
losing numba eligibility).

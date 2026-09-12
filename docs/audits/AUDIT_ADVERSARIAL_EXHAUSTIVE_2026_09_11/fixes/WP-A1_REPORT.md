# WP-A1 — Ray tracing (`lumenairy/raytrace/`) + the shared exit-vertex helper

Branch `audit-fixes-2026-09`.  Findings R1–R7 (report §4) and the exit-vertex
bug class (§15.1 / ORCHESTRATOR F-O3).  Every number below is measured, not
quoted from the audit, except where explicitly labelled "pre-fix" — those are
from the audit's own repro scripts re-run on HEAD before any change.

---

## 0. THE EXIT-VERTEX HELPER — contract for the next wave

Read this section instead of my diff.

### Import paths

```python
from lumenairy.raytrace import exit_vertex_transfer, EXIT_VERTEX_GRAZING_TOL
from lumenairy.raytrace.jax_trace import exit_vertex_transfer_jax
# and, on any TraceResult:
result.at_exit_vertex(n_exit=None)
```

`exit_vertex_transfer` and `EXIT_VERTEX_GRAZING_TOL` are in
`lumenairy.raytrace.__all__`; `exit_vertex_transfer_jax` is exported from
`lumenairy.raytrace` and from `lumenairy.raytrace.jax_trace.__all__`.  The
implementation module is `lumenairy/raytrace/exit_vertex.py` (new), star-imported
by `raytrace/core.py`, so `from lumenairy.raytrace.core import
exit_vertex_transfer` resolves too.

### Signatures

```python
TraceResult.at_exit_vertex(n_exit: Optional[float] = None) -> RayBundle
exit_vertex_transfer(bundle, n_exit, *, fn_name='exit_vertex_transfer') -> RayBundle
exit_vertex_transfer_jax(state: JaxRayState, n_exit) -> JaxRayState
# low-level, shared by the four "advance to a z = const plane" primitives:
vertex_plane_transfer_t(z, N, alive, *, z_target=0.0,
                        tol=EXIT_VERTEX_GRAZING_TOL) -> (t, graze)
resolve_exit_index(surfaces, wavelength, *, fn_name, n_exit=None) -> float
```

### Semantics (all three forms)

| property | behaviour |
|---|---|
| operator | `t = -z/N`; `opd += n_exit*t`; `x += L*t`; `y += M*t`; `z = 0` |
| sign | SIGNED `t`, never `abs(t)` — a convex exit (`sag > 0`) SUBTRACTS OPL |
| mask | ALIVE rays only |
| grazing (`abs(N) <= 1e-30`) | KILLED: `alive=False`, `error_code=RAY_MISSED_SURFACE` (first-failure-wins), state FROZEN — no teleport, no `-z/1e-30` |
| dead rays | frozen exactly (`x, y, z, opd, error_code` unchanged) |
| mutation | none — a new bundle/state is returned; `image_rays` is untouched |
| idempotence | exact: applying twice is bit-identical to applying once |
| directions | `(L, M, N)` untouched — this is a transfer, not a refraction |
| `n_exit` default | `get_glass_index(surfaces[-1].glass_after, result.wavelength)`; `glass_before` instead when `surfaces[-1].is_mirror` or `glass_after` is a `'MIRROR'` marker; raises `ValueError` naming the function if the name cannot be resolved — it never guesses 1.0 |
| `n_exit` validation | must be positive and finite (scalar or per-ray).  A mirror's Welford `n2 = -n1` paraxial sign must NOT be passed: the traced OPL is a physical path length and the signed `t` already carries the propagation direction |
| JAX form | `JaxRayState` has no `error_code`, so the grazing kill shows only in `alive`; double-`where` throughout, so `jax.grad` is finite at `N = 0` |

### Differences from the six hand-written copies you are replacing

1. Grazing rays **die** instead of being teleported (NumPy copies) or getting
   `t = -z/1e-30` (the two `_lens_jax` copies).  If a consumer relied on a
   grazing ray surviving, it will now see `alive=False`.
2. Dead rays keep their pre-transfer state instead of being moved to `z = 0`.
3. `n_exit` is resolved once, from the prescription, with an explicit failure
   mode.

Everything else is bit-identical to the correct copies: measured against the
analytic vertex-plane OPL `n_exit*(-sag/N)` on sphere / parabola / hyperbola /
oblate-ellipsoid exit surfaces, `max |delta opd| = max |delta x| =
max |delta y| = 0.0e+00 m`.

### Migration sketch

```python
# before (any of the six copies)
final = res.image_rays
with np.errstate(divide='ignore', invalid='ignore'):
    t = np.where(final.alive & (np.abs(final.N) > 1e-30), -final.z / final.N, 0.0)
final.opd = final.opd + n_exit * t
final.x = final.x + final.L * t
final.y = final.y + final.M * t
final.z = np.zeros_like(final.z)

# after
final = res.at_exit_vertex()            # n_exit inferred
# or, when you already hold the index / a bare bundle:
final = exit_vertex_transfer(res.image_rays, n_exit)
```

For `lenses_maslov.py` / `_lens_traced_uniform.py` /
`asymptotic_canonical_fit.py`, which need a plane at `output_plane_distance = d`
rather than the vertex: `refocus(result, d)` is the same kernel with the
`(d - z)/N` leg already correct (that is exactly the `t = d/N` vs `(d - z)/N`
error F-O4 measured), or call `at_exit_vertex()` first and then propagate by `d`.

---

## 1. Summary table

| ID | status | files:lines | tests | oracle | measured before → after |
|---|---|---|---|---|---|
| §15.1 helper | **fixed (new API)** | `raytrace/exit_vertex.py` (new, 253 L); `raytrace/surface.py:320`; `raytrace/jax_trace.py:1335`; wired into `intersection.py:218,620`, `ray_fan.py:872` (`refocus`) | `tests/unit/test_audit2609_a1_exit_vertex.py` (17) | analytic vertex-plane OPL `n·(−sag/N)`, written in the test | copies: 6 hand-written, 2 backends disagreeing on grazing → 1 shared operator, `max|Δ| = 0.0e+00 m` vs the closed form; grazing `t = −z/1e-30` → killed |
| R1 | **fixed** | `raytrace/ray_fan.py:115-255` (new `_reference_sphere_radius` / `_reference_sphere_leg` / `_opd_fan_wfe`), `:642` `opd_fan_data`, `:763` `opd_fan_data_world` | `test_audit2609_a1_raytrace.py::test_r1_*` (6) | independent exact plano-convex trace (`_exact_plano_convex_wfe`) + Seidel `−S1/8` | f/4 `W(ρ=1)`: **+36.194 w → −11.720 w** (oracle −11.714); f/20 +0.0545 → −0.01813 (oracle −0.01813); parabola control PV 7.1e-11 → 7.0e-11 w |
| R2 | **fixed** | `raytrace/trace.py:807` (`_make_bundle(opd_seed=)`), `:885` `seed_entrance_eikonal`, applied at `ray_fan.py:78` for both fans | `test_audit2609_a1_raytrace.py::test_r2_*` (4) | the fan's own chief-referenced linear term must vanish | fitted linear term @5°: **−1879.86 w → +0.15 w** (real aberration 37.6 w); @0.5° −185.64 → +0.01 |
| R3 | **fixed** | `raytrace/seidel.py:1490` (warning), `:1517` `_aspheric_seidel`, `:1785` (applied in all 3 branches), docstring Notes | `test_audit2609_a1_raytrace.py::test_r3_*` (13) | real-ray ρ⁴ fit of the OPL to the paraxial focus | parabolic mirror `S1`: **+9.765625e-05 → 1.355e-20**; A4 = −500 singlet `−S1/8`: **6.651 µm → 0.342 µm** (real −0.333); agreement 0.14–0.29 % |
| R4 | **fixed** | `raytrace/intersection.py:317-370`; `raytrace/jax_trace.py:241-320` (`_intersect_jax`), `:1496-1548` (`_intersect_jax_param`) | `test_audit2609_a1_raytrace.py::test_r4_*` (7) | closed-form conic sag | condenser R=10.84, k=−0.6: **alive [T T T F F] → [T T T T T]**, `t` at h=10.90 mm **0 → 6.186249 mm** (true 6.186249); `max|Δz| = 8.7e-19 m` up to h/\|R\| = 1.8 |
| R5 | **fixed** | `raytrace/trace.py:213-243` + `:1264,1296` (`apply_doe_phase_traced`), `raytrace/world_trace.py:169-205`, `raytrace/jax_trace.py:436-560` + 3 call sites | `test_audit2609_a1_raytrace.py::test_r5_*` (4) | the grating equation `n₂L′ = n₁L + mλ/Λ` | L inside N-BK7: **0.26200000 → 0.17425045**, ratio to truth 1.503583 → 1.000000 |
| R6 | **fixed** | `raytrace/from_field.py:844-900` (estimator), `raytrace/intersection.py:594-640` (`_transfer`), `:39-92` (in-place kernel) | `test_audit2609_a1_raytrace.py::test_r6_*` (9) | plane wave / grid Nyquist; closed-form transfer | L_true 0.15: **−0.100 → +0.150**; edge-ray ratio **0.5000 → 1.0000**; grazing `_transfer` **alive[T T], z=[0 0] → alive[F F], err 3, z preserved**; perf 1.03–1.05×, peak 235.6 → 219.4 MiB at N=1e6, bit-identical |
| R7 | **fixed** (8 of 8) | `intersection.py:494`; `from_field.py:105,438`; `layout.py:4,100`; `trace.py:1727,1842`; `paraxial.py:112`; `differential.py:1070` | `test_audit2609_a1_raytrace.py::test_r7_*` (6) + validation | each item's own | see §2.7 |

Verified-correct items re-measured and unchanged: OPL vs the 60-digit oracle
**1.388e-17 m** (audit: 1.39e-17); spherical fast path **3.331e-16 m** (audit:
3.3e-16); coordinate breaks exact inverses; `system_abcd` closed forms 0.00e+00;
spherical Seidel ratios −0.9982 / −0.9992; JAX parity **6.939e-18 m / 2.776e-17 m**
(audit: 6.9e-18 / 2.8e-17); `jax.grad` vs FD **1.378e-9** (audit: 1.4e-9);
differential Jacobians 1.7e-9 … 1.0e-7; `rays_from_field` sign 5e-16 and
converging-wave focus 0.001 nm.

---

## 2. Per finding

### §15.1 — the exit-vertex helper (done first, per the WP)

**Wrong.** `trace()` leaves rays at `z = sag(ρ)` of the last surface.  Seven
consumers read `image_rays.opd/.x/.y` there; six re-derived the correction by
hand and the two JAX copies clamped `N` to `1e-30` where the NumPy copies used
`t = 0` — a cross-backend divergence in the same primitive.  The missing term is
`n_exit·sag(ρ)/N`, a pure ρ² (defocus) contribution that downstream polynomial
fits absorb silently.

**Changed.** New module `raytrace/exit_vertex.py` with the operator, the shared
low-level `vertex_plane_transfer_t` kernel and `resolve_exit_index`;
`TraceResult.at_exit_vertex` in `surface.py`; `exit_vertex_transfer_jax` in
`jax_trace.py`.  The three in-package hand-written copies the WP named now route
through the shared kernel: `intersection._intersect_surface`'s flat branch
(`intersection.py:214-218`), `intersection._transfer` (`:617-620`) and
`ray_fan.refocus` (`:868-874`).  Full contract in §0.

Note on what "route through the helper" means at those three sites: they share
the **`vertex_plane_transfer_t` kernel** (same arithmetic, same grazing
definition) rather than the bundle-level `exit_vertex_transfer`, because each
has its own legitimate post-conditions — `_intersect_surface` continues into the
aperture gate, `_transfer` targets `z = thickness` rather than 0, and `refocus`
must not mutate the caller's alive mask during a focus sweep.  Sharing the
kernel keeps the arithmetic bit-identical; sharing the bundle-level operator
would not have been correct.

**Verified.**
* Analytic vertex-plane OPL on sphere / parabola / hyperbola / oblate
  ellipsoid: `max |Δopd| = max |Δx| = max |Δy| = 0.0e+00 m`.
* Curved-rear fixture (the class is invisible on the plano-rear fixtures the
  whole traced-propagator corpus uses): the missing term fits to
  `c2 = −1.117e-03 m` of ρ² with `|c4/c2| = 0.064`.
* Grazing: `alive → False`, `error_code = 3`, `z` and `opd` unchanged.
* Idempotence: bit-identical on all seven fields.
* JAX parity: `1.7e-18 m` (position) / `5.2e-18 m` (OPL), `alive` and `z` equal;
  `jax.grad` finite at `N = 0`.
* `n_exit` inference through glass recovers `n(N-BK7)` to `rtol 1e-9`.

**Residual risk.** A consumer that relied on a grazing ray surviving the
transfer will now see it dead; that is the intended fix but it is a behaviour
change at the boundary.  `refocus` deliberately keeps the old
grazing-stays-alive policy so focus sweeps cannot mutate the caller's mask —
the two therefore differ on grazing rays only, which is documented in both
docstrings and pinned by
`test_matches_refocus_zero_on_alive_rays`.

### R1 — `opd_fan_data` has no reference sphere

**Wrong.** Both fan functions returned `(img.opd − opd_chief)/λ`, the OPL to
each ray's own intercept.  The wavefront error is the OPL difference to a
COMMON point; the two differ at first order in the transverse aberration.

**Changed.** Each ray is referenced to a reference sphere centred on the chief
ray's image point and passing through the EXIT PUPIL — the Welford §4 / rayoptics
/ Zemax convention that `analysis.eval_image_plane_wfe(sphere_tangent=
'exit_pupil')` already implements.  The sphere crossing is solved in the
cancellation-free form

```
leg = (|v|² − 2R(v·d)) / (R − v·d + sqrt(R² + (v·d)² − |v|²))     # = t + R
W   = ((opd − opd_chief) + n_img·leg) / λ
```

with `v = P − C`.  `R = |xp_z / N_chief|` from `first_order_data`; `R = inf`
(the reference-PLANE limit, which keeps the exact first-order term) is the
documented fallback when the exit pupil is unavailable.  New keyword
`reference_sphere_radius` overrides.

**Verified** against an INDEPENDENT exact singlet trace written in the test
(ray/sphere intersection, vector Snell, straight leg to the Gaussian image
point; no library code) at `ρ = 1`:

| system | before | after | oracle | \|after − oracle\| |
|---|---|---|---|---|
| f/4 plano-convex | +36.194 w | −11.720 w | −11.714 w | 0.0067 w |
| f/4 flat-first | +151.789 w | −45.063 w | −45.054 w | 0.0087 w |
| f/20 | +0.0545 w | −0.018134 w | −0.018134 w | 1.4e-6 w |
| f/50 | +0.0014 w | −0.000464 w | −0.000464 w | 3e-10 w |

The residual is the second-order difference between the exit-pupil-tangent
sphere and the oracle's last-surface-tangent leg; choosing the reference PLANE
instead would leave 0.422 w at f/4, so the exit-pupil sphere is 63× better.
Cross-check against the independent Seidel relation: `−S1/8 = −11.319 w` vs the
ray value `−11.720 w`, gap 0.402 w — which is the genuine fifth order (the
oracle's own ρ⁶ coefficient is −0.41 w).  Sign now matches; pre-fix it did not.
Control: the aberration-free parabolic mirror gives PV `7.012e-11` waves (was
`7.085e-11`).

**Residual risk.** When the caller's surface list does not end at/near an image
plane there is no image point, and the sphere is ill-posed.  This is now
DETECTED (`|v| >= R` for some alive ray) → `RuntimeWarning` naming the geometry
+ fallback to the reference plane, instead of the NaN / 1416-wave garbage an
unguarded solve produces.  One existing validation check used exactly that
degenerate fixture; it is fixed (see §4).

### R2 — off-axis launch-plane tilt

**Wrong.** `make_fan`/`make_ray` launch on the `z = 0` plane with `opd = 0`;
for a field-angle bundle that plane is not a wavefront, so the fan carries a
linear term of exactly `−y·sinθ`.

**Changed.** The OPD fans seed the entrance eikonal `L·x + M·y` on the fans AND
their chief rays.  `_make_bundle` gains `opd_seed={'plane','eikonal'}`
(default `'plane'` = unchanged) and `trace.seed_entrance_eikonal(rays)` is the
functional form (it also carries the `N·z` term, for bundles launched off
`z = 0`).

**Why not in `_make_bundle` by default** — the audit's suggested site.  I
measured the blast radius: `_make_bundle` is the shared launcher for ~20
consumers across `analysis/`, `elements/`, `propagators/`, `ui/` and
`raytrace/differential.py`.  Two of them already add their own entrance eikonal
(`elements/_lens_traced.py:9606`'s v5.25.1 H6 `_carrier_W_fn`, and the canonical
generating function in `propagators/asymptotic_canonical_fit.py`), so seeding it
in the launcher DOUBLE-counts there; and two others overwrite `bundle.z` after
construction (`analysis/image_plane_wfe.py:506`,
`asymptotic_canonical_fit.py:404`), for which the `z = 0` eikonal is incomplete.
I confirmed this empirically: with the seed as a default, 12 tests in
`tests/unit/test_niche_audit_w3_oracles.py` (LG aberration tensor / canonical
fit, another WP's area) and 1 in `test_analytic_ray_transfer.py` failed.  Per
COMMON rule 1 the fix is applied where the defect was measured and offered as an
opt-in everywhere else; see §5 for the requested follow-ups.

**Sign note.** The audit's §4 row and the RAYTRACE partition both write the fix
as `opd = −(x·L + y·M)`.  That sign is wrong — it doubles the artefact.  The
audit's own measurement demands the `+` sign (fan linear term `−185.64 w` vs
`y_max·sinθ/λ = +185.64 w`, ratio `−1.0000`), and `+` is what the library's own
H6 entrance-eikonal precedent adds.  Implemented as `+(L·x + M·y)` and verified:

| field | before | after | `y_max·sinθ/λ` |
|---|---|---|---|
| 0.5° | −185.64 w | +0.01 w | +185.64 w |
| 2.0° | −744.06 w | +0.06 w | +742.42 w |
| 5.0° | −1879.86 w | +0.15 w | +1854.06 w |

The real (quartic) content survives: the fan's ρ⁴ term stays at the on-axis
level (−11.9 w) instead of the pre-fix +37.2 w (which was itself contaminated by
R1).  Bit-identical on axis (`L = M = 0`).

**Residual risk.** The `~0.15 w` residual at 5° is the genuine higher-order
coupling between the eikonal and the aberration, not a leftover artefact — it
grows as θ³ and is 4 decades below the pre-fix value.

### R3 — Seidel ignores conic and aspheric coefficients

**Wrong.** Zero occurrences of `conic` or `aspheric` in `seidel.py`; the sums
returned were those of the base sphere, silently.

**Changed.** `_aspheric_seidel` (`seidel.py:1517`) implements Welford §8.5:
`dS_I = 8(n₂−n₁)A4_eff·h⁴` with `A4_eff = k/(8R³) + A4`, and
`dS_II/III/V = dS_I·(y_c/y_m)^{1,2,3}`; S4 (Petzval) is curvature-only and
untouched.  Applied in ALL THREE branches (mirror, curved refractor, flat base
— a Schmidt plate is `R = inf` with a pure A4).  `y_m = 0` (surface at an
internal image) returns zeros rather than ±inf.

**Note on the S3 exponent.** The audit's snippet writes
`S3[i] += dS * (y_c/y_m)` — the same power as S2.  That is a typo: the Welford
sums carry successive powers of the Abbe-invariant ratio (`S_I ~ A²`,
`S_II ~ A·A_c`, `S_III ~ A_c²`), so S3 takes `(y_c/y_m)²`.  The repro validated
S1 only; I used the Welford scaling and say so here.

**Verified** against a real-ray ρ⁴ fit of the OPL to the paraxial focus:

| case | S1 before | S1 after | `−S1/8` after | real-ray a4 | rel |
|---|---|---|---|---|---|
| mirror k = 0 | +9.765625e-05 | +9.765625e-05 | −12.2070 µm | −12.2047 µm | 0.019 % |
| mirror k = −0.5 | +9.765625e-05 | +4.882813e-05 | −6.1035 µm | −6.1023 µm | 0.020 % |
| mirror k = −1 | +9.765625e-05 | **1.355e-20** | −0.0000 µm | −0.0000 µm | exact |
| mirror k = −1.5 | +9.765625e-05 | −4.882812e-05 | +6.1035 µm | +6.1016 µm | 0.031 % |
| A4 = 0 | +5.320645e-05 | +5.320645e-05 | −6.6508 µm | −6.6393 µm | 0.17 % |
| A4 = −250 | +5.320645e-05 | +2.797215e-05 | −3.4965 µm | −3.4865 µm | 0.29 % |
| A4 = −500 | +5.320645e-05 | +2.737848e-06 | −0.3422 µm | −0.3326 µm | 2.90 %¹ |
| A4 = −1000 | +5.320645e-05 | −4.773075e-05 | +5.9663 µm | +5.9785 µm | 0.20 % |
| A4 = −2000 | +5.320645e-05 | −1.486679e-04 | +18.5835 µm | +18.6130 µm | 0.16 % |
| A4 = +500 | +5.320645e-05 | +1.036750e-04 | −12.9594 µm | −12.9414 µm | 0.14 % |

¹ the near-aplanatic point, where the ρ⁴ term has almost cancelled: 0.0096 µm
absolute, the smallest error in the table.

**What is and is not included** is now in the docstring: radius (full spherical
sums); conic + `aspheric_coeffs[4]` (Welford §8.5, as above); A6/A8/… NOT (they
generate fifth and higher order — third-order theory cannot represent them);
`aspheric_coeffs[2]`, biconic, freeform and the field-frame block NOT, and these
now raise a `RuntimeWarning` naming the surfaces, because no rotationally
symmetric third-order expansion exists for them.

**Residual risk.** The new warning fires on prescriptions that previously
returned silently; it is a `RuntimeWarning`, so it does not break callers, but
noisy suites with `-W error` will see it.  I checked the owned + adjacent test
files and none trip it.

### R4 — conic false miss for `h > |R|`

**Wrong.** The Newton branch seeded from the ray-SPHERE quadratic and used its
discriminant as the miss test.  Both JAX kernels carried the identical test.

**Changed.** For rotationally symmetric surfaces the seed AND miss test come
from the exact conic quadratic
(`F = c(x²+y²) − 2z + (1+k)c z² = 0`), solved in the Spencer & Murty stable form
`t = e/q`, `q = −(b + sign(b)√disc)/2` — which is the near root by construction
(`|e/q| <= |q/a|`), so the v5.4.1 direction-aware behaviour for
backward-propagating rays is preserved without an explicit `min(|t1|,|t2|)`.
Newton still runs and refines the polynomial departure for aspheres.
Anamorphic / freeform / field-frame surfaces keep the legacy sphere seed, since
a conic discriminant is not a valid miss test for them.  In `_intersect_jax` the
pure-spherical case (`conic == 0`, no aspherics) keeps its own closed-form
branch so the audited bit-level NumPy↔JAX parity is untouched — `conic` and
`asph_items` are Python-static there, so the selection is a trace-time branch
with no runtime cost.

**Verified** against the closed-form conic sag:

| R [mm] | k | h [mm] | before | after | true sag [mm] |
|---|---|---|---|---|---|
| 10.84 | −0.6 | 10.83 | alive, 6.095530 | alive, 6.095530 | 6.095530 |
| 10.84 | −0.6 | 10.90 | **dead (err 3), 0** | alive, 6.186249 | 6.186249 |
| 10.84 | −0.6 | 11.40 | **dead (err 3), 0** | alive, 6.863647 | 6.863647 |
| 10.84 | −0.6 | 15.00 | **dead (err 3), 0** | alive, 13.988555 | 13.988555 |
| 10.84 | −0.6 | 17.00 | **dead (err 3), 0** | alive, 23.648913 | 23.648913 |
| 50 | −1 | 60.00 | **dead (err 3), 0** | alive, 36.000000 | 36.000000 |
| 50 | −2 | 120.00 | **dead (err 3), 0** | alive, 80.000000 | 80.000000 |

`max |Δz| = 8.674e-19 m`.  Through the public `trace()` API a 41-ray fan on the
condenser goes from 39/41 to **41/41** alive.  `trace_jax` now agrees:
`alive = [T T T T T]` (was `[T T T F F]`), matching the analytic ADRT, which
always had the exact quadratic.  Genuine misses still die — an oblate ellipsoid
(k = +2, domain `h < |R|/√(1+k)`) gives `[T T F F]` with `error_code = 3`.
Unchanged: the spherical fast path, `max |dt| = 3.331e-16 m` vs an independent
stable root (audit baseline 3.3e-16).

**Residual risk.** The conic discriminant is now also the miss test for a conic
+ POLYNOMIAL asphere, where a large polynomial departure could in principle
extend the surface past the base conic's domain.  That is a strict improvement
over the sphere test it replaces (the conic domain always contains the sphere
domain), but it is not exact for such a surface.  Documented in the code.

### R5 — the diffraction-order kick omits the medium index

**Wrong.** All four sites applied `ΔL = mλ/Λ` to the direction cosines AFTER
refraction into `glass_after`.  The grating equation conserves the tangential
wavevector, so `ΔL = mλ/(n₂Λ)`.

**Changed.** All four sites now separate two quantities: the grating's
PHASE-SCREEN gradient `gL = mλ_vac/Λ` (index-independent, used for the OPL) and
the direction kick `dL = gL/n₂`.  This is load-bearing — the OPL term's
transverse gradient must equal `n₂L′ − n₁L = mλ/Λ`, so dividing it by `n₂` as
well would have introduced a new error.  `apply_doe_phase_traced` gains
`n_medium: float = 1.0` (validated, default = pre-fix behaviour);
`_apply_doe_kick_jax` gains the same parameter and all three JAX trace bodies
pass `n2`.

**Verified** (Λ = 5 µm, λ = 1.31 µm, m = 1, into N-BK7): library `L` inside the
glass **0.26200000 → 0.17425045**, which is `mλ/(n₂Λ)` to `< 1e-12` relative;
the pre-fix ratio to truth was exactly `n(N-BK7) = 1.503583`.  The OPL term is
unchanged and equals `mλx/Λ` to `< 1e-15` relative.  Air behaviour is
bit-identical (`n₂ = 1`).

**Residual risk.** The docstring's old claim that the form "neglects the cosine
factor" was itself inaccurate and is corrected: for in-plane diffraction the
direction-cosine form is exact; the remaining idealisation is the thin-screen
model (no thickness, no Bragg selectivity, no order-dependent efficiency), now
stated.

### R6 — `rays_from_field` aliasing, edge rays, `_transfer` grazing, in-place arithmetic

**Estimator.** `arg(E[j+1]·conj(E[j−1]))/(2dx)` → symmetrised ONE-pixel circular
mean `arg(û(E[j+1]conj(E[j])) + û(E[j]conj(E[j−1])))/dx`.  Measured at λ = 1 µm,
dx = 2 µm (grid Nyquist 0.25): `L_true` 0.150 → **−0.100 before, +0.150 after**;
0.200 → −0.050 / +0.200; 0.240 → +0.240; 0.260 aliases on both, because the GRID
cannot represent it.  Normalising each phasor before summing is load-bearing: an
amplitude-weighted sum degraded the audit's verified converging-wave focus from
0.001 nm to 16.3 nm rms; with the circular mean it stays at **0.001 nm**.  The
sign convention is preserved to 5e-16.

**Edge rays.** The clipped self-reference is dropped, leaving a one-sided
one-pixel difference.  Edge-column ratio to truth **0.5000 → 1.0000** over 128
edge rays.

**`_transfer` grazing.** Before: `z=[0 0], alive=[T T], opd=[0 0],
error_code=[0 0]` — teleported one gap downstream with zero OPL and still alive.
After: `z=[1e-4 1e-4], alive=[F F], opd=[0 0], error_code=[3 3]`.  Dead rays now
keep their `z` (the `_transfer_jax` S3-12 policy); ordinary rays are
bit-identical.

**In-place arithmetic.** `_intersect_surface`'s position+OPL block and
`_transfer` write through one reusable buffer (`_advance_along_rays`), guarded
on float64 + writeable with an allocating fallback.  Output is **bit-identical**
(`max |dx| = max |dy| = max |dopd| = 0.0`) because the two roundings are the
same two roundings.  Medians of 7 interleaved runs, 7 surfaces,
`output_filter='last'`, on a box shared with the other WPs:

| N | before | after | speedup | tracemalloc peak |
|---|---|---|---|---|
| 300 000 | 2806.3 ns/ray | 2676.1 ns/ray | 1.049× | 70.7 → 65.8 MiB |
| 1 000 000 | 2931.7 ns/ray | 2840.9 ns/ray | 1.032× | 235.6 → 219.4 MiB |

Deferred (with designs) in §6: the `_refract` renormalise hoist and the analytic
sphere normal — the latter is the 24 % block but a v4.12.0 attempt at it broke a
cross-backend test.

### R7 — the P3 bundle (all 8 items)

1. **`_refract` first-failure-wins** (`intersection.py:494`) — the `np.where`
   was unconditional despite the comment.  Verified with a hand-built bundle
   carrying `RAY_APERTURE` into a TIR: code stays 2 (was relabelled to 1).
2. **Wrapped `opd` in `rays_from_field`** — documented at module, parameter and
   `Returns` level, plus a new `opd_phase={'wrapped','unwrapped'}` (default
   unchanged).  Measured: wrapped spread ≤ λ; unwrapped 0.82 µm = 0.82 waves of
   true path.
3. **`trace_summary` evanescent** — the breakdown now prints `evanescent=` and
   an `unclassified=` column so it always sums to the lost count.  Verified on a
   grating whose order goes evanescent: `evanescent=3`, no `unclassified`.
4. **Registry lock** — `_register_fixed_index`'s two global writes are now under
   `glass._GLASS_CACHE_LOCK`.  The import-time `'__thin_lens__'` registration is
   KEPT (load-bearing: `surfaces_from_elements`' `'lens'` branch emits it,
   `glass.py` lists it as a sentinel, and
   `tests/unit/test_audit_p1_glass_registration.py` pins it on a minimal
   install) and documented as deliberate, along with the unbounded growth of
   content-derived index names.
5. **`raytrace_system` mutation** — `surfaces[-1]` is now CLONED with the image
   distance via `_surface_copy_with`.  Verified: two calls on the same element
   list give identical thicknesses and identical traced `y`.
6. **`layout.py` naming** — module docstring states plainly there is no layout
   geometry here and points at `analysis.plotting.plot_lens_layout` /
   `raytrace.spot_diagram`.
7. **`_adrt_jax` double trace** — one `jax.jacfwd(_full, has_aux=True)` pass.
   Jacobian matches the NumPy dual backend to 3.331e-16, `opd` and `x` exactly,
   `ux` to 1.4e-17.
8. **`field_of_view`** — the finite-conjugate branch now uses the sensor via the
   Gaussian magnification; without `sensor_half_height_m` the legacy aperture
   proxy is kept behind a `RuntimeWarning` that says it is a numerical aperture,
   not a field of view.

---

## 3. Files touched

Modified (all inside `lumenairy/raytrace/`, which WP-A1 owns):

* `lumenairy/raytrace/__init__.py` — export `exit_vertex_transfer`,
  `EXIT_VERTEX_GRAZING_TOL`, `exit_vertex_transfer_jax`.
* `lumenairy/raytrace/core.py` — star-import the new module.
* `lumenairy/raytrace/surface.py` — `TraceResult.at_exit_vertex`; `image_rays`
  docstring.
* `lumenairy/raytrace/intersection.py` — R4 conic quadratic; R6 `_transfer`
  grazing kill + in-place kernel; R7 `_refract`; shared-kernel wiring.
* `lumenairy/raytrace/trace.py` — R2 `opd_seed` + `seed_entrance_eikonal`; R5
  inline kick + `apply_doe_phase_traced(n_medium=)`; R7 registry lock +
  `raytrace_system` clone.
* `lumenairy/raytrace/world_trace.py` — R5 world twin.
* `lumenairy/raytrace/ray_fan.py` — R1 reference sphere (+ ill-posed guard), R2
  eikonal seeding, `refocus` shared kernel.
* `lumenairy/raytrace/seidel.py` — R3 aspheric terms + warning + docstring.
* `lumenairy/raytrace/jax_trace.py` — R4 (both kernels), R5, JAX exit-vertex twin.
* `lumenairy/raytrace/from_field.py` — R6 estimator + edges; R7 `opd_phase`.
* `lumenairy/raytrace/layout.py` — R7 evanescent breakdown + docstring.
* `lumenairy/raytrace/paraxial.py` — R7 `field_of_view`.
* `lumenairy/raytrace/differential.py` — R7 `_adrt_jax` single pass; R2 note.
* `validation/raytrace/test_raytrace.py` — two checks that pinned the OLD
  behaviour (see §4).

New:

* `lumenairy/raytrace/exit_vertex.py`
* `tests/unit/test_audit2609_a1_exit_vertex.py`
* `tests/unit/test_audit2609_a1_raytrace.py`
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A1_REPORT.md`
* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A1_CHANGELOG.md`

Nothing outside `lumenairy/raytrace/`, `tests/unit/test_audit2609_a1_*`,
`validation/raytrace/` and the `fixes/` directory was modified.  No git write
commands were run.

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1`.

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_a1_exit_vertex.py tests/unit/test_audit2609_a1_raytrace.py -q` | **65 passed** (17 + 48) | 11.9 s |
| `pytest` over the 11 raytrace/seidel unit files — BASELINE, before any change | 250 passed, 1 skipped | 81.7 s |
| the same 11 files after the changes | 250 passed, 1 skipped | 69.2 s |
| `pytest` over 17 raytrace + ray-consumer files (fga, gbd, merits, chain_multi, fga_dual, walker, …) | **362 passed, 3 skipped** | 1148 s |
| `pytest` over 16 adjacent files (analytic_ray_transfer, glass_registration, w3_oracles, v4_15_*, v5_1_0_split, v5_2, v5_21*) | 889 passed, 1 skipped, **2 failed** (other WP — see below) | 134 s |
| `pytest` over 23 files: the 11 raytrace/seidel + the 2 new + 10 adjacent | **762 passed, 1 skipped** | 188.9 s |
| final confirmation after the import cleanup (8 files incl. both new) | **380 passed, 1 skipped** | 82.5 s |
| `python validation/run_all.py test_raytrace test_seidel_field test_world_surfaces test_folded_designs` | 4/4 files pass; `validation/raytrace/test_raytrace.py` **54/54** | ~22 s |

Pre-existing / other-WP failures found, with my judgement:

* `tests/unit/test_niche_audit_w4_input_kind.py::test_wired_site_declares_expected_input_kind[beam_stats.py::beam_d4sigma(E)->field]`
  and `::test_all_sixty_eight_sites_are_wired` — these check
  `_validation._check_2d_scalar_field` wiring in `lumenairy/analysis/beam_stats.py`,
  a file another WP is editing concurrently (`git status` shows it modified).
  **Unrelated to WP-A1**: nothing in `raytrace/` is in the 68-site list, and the
  failures reproduce with `raytrace/` untouched.
* `docs/.../repro/RAYTRACE/p2_opl.py` fails at `assert len(idx) > 0, 'no root at
  surface 0'` INSIDE the auditor's own 60-digit `decimal` oracle, both before and
  after my changes.  Its sibling `p2b_opl.py` (the one the audit actually quotes,
  same oracle, more systems) runs clean and reports **1.388e-17 m** worst OPL
  error, matching the audit's 1.39e-17 m — so the OPL bookkeeping is verified;
  the failing script is a bug in the repro, not in the library.
* The whole suite was briefly un-importable mid-session
  (`AttributeError: module 'lumenairy.elements.pmm.twod' has no attribute
  'pmm_2d_order_drift'`) while another WP was mid-edit on `elements/pmm/`.  It
  cleared on retry.

I did NOT run the full 3.65 h suite (per COMMON rule 5).

**Confirming the new tests fail on the pre-fix code.**  A scratch harness
(`scratchpad/wpa1/prefix_check.py`) restores each pre-fix implementation in
process (monkeypatch) and re-runs the matching test function.  Result: every
regression test targeting a fix FAILS on the pre-fix code — R1 f/4 and f/20, the
R1 Seidel cross-check, R2 at 0.5° and 5° (and again with R1 already fixed, to
show the two are independent), R3 for k = −1, k = −0.5, A4 = −2000 and the
near-aplanatic case, R6 at L = 0.15, R6 edge rays, R6 `_transfer` grazing, and
R7 `_refract` — while the three designated CONTROLS (aberration-free parabola,
L = 0.05 below the old band limit, ordinary `_transfer`) correctly pass on both.
R4 and R5 are demonstrated analytically in the same harness (the sphere
discriminant is negative at h = 10.90/11.40/15/17 mm on the condenser and at
h = 60/100 mm on the parabola where the conic discriminant is positive; the
pre-fix DOE kick violates the 1e-12 relative bar by 5.0e11×).

---

## 5. Requested changes outside my ownership

1. **`elements/_lens_real.py`** (F-O1, L1) — the `seidel_correction=True` fan
   reads `res_fan.image_rays.opd` at the sag.  Replace with
   `res_fan.at_exit_vertex().opd`.  Expected: correction RMS 34 221 nm → ~1 118 nm,
   ρ² coefficient −8.96e-5 m → +3.0e-6 m (ORCHESTRATOR's measurement).
2. **`elements/lenses_maslov.py:1709-1735`** (F-O4, S3) — `exit_rays = tr.image_rays`.
   Use `tr.at_exit_vertex()`; and make the `output_plane_distance` leg
   `(d − z)/N`, i.e. `refocus(tr, d)`.
3. **`elements/_lens_traced_uniform.py:223-228, 587-592`** — `t = output_plane_distance/Nz`
   → `refocus(result, output_plane_distance)`.
4. **`elements/_lens_traced_multibranch.py`** (S1) — same.
5. **`propagators/asymptotic_canonical_fit.py:408-422, 915+`** — `(final.x, final.opd)`
   at the sag → `at_exit_vertex()`.
6. **`elements/_lens_traced.py:6673, 9581, 11104` and `elements/_lens_jax.py:556, 829`**
   — six correct hand-written copies; replace with `at_exit_vertex()` /
   `exit_vertex_transfer_jax()` so the grazing policy stops diverging between
   backends.  Note the two JAX ones currently produce `t = -z/1e-30` for a
   grazing ray; the helper kills it.
7. **`analysis/image_plane_wfe.py:508-511`** — depends on whether `surfaces`
   ends in an image plane (ANALYSIS auditor to confirm); if not,
   `at_exit_vertex()`.
8. **Entrance eikonal (R2) for the other launchers.**  I left `_make_bundle`'s
   default alone deliberately (see §2 R2).  Consumers whose OPL differences
   across a TILTED bundle are meant to be a wavefront error should call
   `raytrace.trace.seed_entrance_eikonal(bundle)` (or pass
   `opd_seed='eikonal'`) — and must NOT if they already add their own eikonal.
   Concretely: `analysis/aberration.py:442`, `analysis/field.py:417,629,1194,1201`,
   `analysis/plotting.py:2092` look like candidates;
   `elements/_lens_traced.py:9546` and `propagators/asymptotic_canonical_fit.py:397`
   look like they must NOT (they already do it, or fit a generating function
   whose Legendre branch the eikonal changes).  Each needs its own owner's
   judgement — I have not touched any of them.
9. **`analysis/plotting.plot_opd_fan`** — its PV/RMS labels now report a genuine
   wavefront error; any hard-coded axis limits or thresholds calibrated on the
   old (3.1× too large, wrong sign) numbers need re-checking.

---

## 6. Deferred, with designs

1. **`_refract` / `_reflect` renormalise hoist** (audit perf #3, ~10 N-sized ops
   per surface).  Exact vector Snell with a unit normal returns a unit vector
   identically, so the per-surface `sqrt(L²+M²+N²)` + 3 divides + 3 `np.where`
   only remove ~1e-16 of drift.  Design: keep the degenerate-ray detection
   (`mag < 1e-30 | ~isfinite` → `RAY_NAN` + kill) per surface, since that is a
   real diagnostic, but move the DIVISION to a single pass at the end of `trace`
   / `trace_world`.  Risk: `trace` is not the only caller of `_refract`
   (`analysis/ghost.py` and the differential FD path reach it), so the hoist
   must live behind a flag or every caller must be updated.  Effort ~3 h
   including a bit-identity sweep.
2. **Analytic sphere normal for `is_pure_spherical`** (audit perf #2, the 24 %
   block).  `(x, y, z−R)/R` instead of the generic
   `sqrt(x²+y²)` + `np.where(h>0,…)` + divide + second sqrt.  The v4.12.0 attempt
   failed because it was applied WITHOUT the matching intersection change and
   compounded a 1.17e-3 cross-backend error in the Maslov asymptotic test; now
   that the intersection paths are aligned it should be retried, but it needs
   that specific test as its gate.  Effort ~4 h including the cross-backend
   sweep.
3. **`ray_fan_data` / `opd_fan_data` issue four separate `trace()` calls**
   (audit perf #4).  One concatenated bundle would do; the obstacle is that the
   chief rays and the two fans have different `ep_off` offsets and different
   field-tilt axes, so the concatenation needs care to keep the RT-5 invariant
   (`ey(0) == ex(0) == 0`).  Effort ~2 h.
4. **`trace_jax` per-call prep** (audit perf #7): ~750 µs/call is
   `_build_jax_prescription`, not the XLA kernel.  Design: cache the built
   `JaxPrescription` on `id(prescription)` plus a cheap content hash, or accept a
   pre-built one (already supported) and document it as the fast path.  Effort
   ~2 h.
5. **Area-uniform pupil sampling** (`make_rings`, audit alt-algorithm #6).  The
   equal-radius/equal-count bias is documented and measured (mean `r/R` 0.5806 vs
   0.6667) and every `spot_rms` consumer inherits it.  A Vogel/sunflower
   generator is a one-liner but changes every spot number in the library, so it
   belongs behind a `pattern=` keyword with the current default — a deliberate
   API decision for the orchestrator, not a silent fix.
6. **Aspheric support in `ray_transfer_jacobian_analytic`** (audit alt #4): add
   the polynomial terms to `_adrt_step`'s implicit `F` and its gradient so the
   analytic path stops raising `NotImplementedError` for `aspheric_coeffs`.
   Effort ~4 h; needs its own FD cross-check.
7. **Polarisation ray tracing** (audit alt #5) — out of scope for a remediation
   pass; recorded so it is not lost.

---

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A1_CHANGELOG.md`

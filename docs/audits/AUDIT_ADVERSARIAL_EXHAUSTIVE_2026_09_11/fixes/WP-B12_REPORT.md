# WP-B12 -- the FGA reference plane: the differential transfer gains an explicit
# output plane, and the four `fga.py` sites ask for the exit-vertex one

Wave 5 item A.  Branch `fix/wp-b12-fga-reference-plane`, base `96cb2096` (the
5.47.0 release commit plus the Wave-5 plan).  Brief:
`briefs/WP-B12.md`; the finding it acts on is `fixes/VERIFY_WP-B7b.md`
sections 3.3, 3.5 and 8 (requests R-1 and R-2), and handoff section 4.1.

---

## 0. Terms used here

* **Last surface** -- the final refracting or reflecting surface of a
  prescription.  `lumenairy.raytrace.trace` stops every ray at its
  intersection with that surface, i.e. at `z = sag(rho)` in the surface's local
  frame.  `TraceResult.image_rays` reports that state.
* **Exit-vertex plane** -- the plane `z = 0` through that surface's vertex.
  `TraceResult.at_exit_vertex()` transfers a ray bundle to it along each ray,
  and `lumenairy/raytrace/exit_vertex.py` is the library's ONE implementation of
  that operator (the 2026-09-11 audit created it to replace seven hand-written
  copies).
* **Sag along the ray** -- the separation between those two planes for one ray:
  `t = -z/N`, so `x_v = x - sag*ux`, `y_v = y - sag*uy` and
  `opd_v = opd - n_exit*sag*sqrt(1 + ux^2 + uy^2)` with `ux = L/N` the unreduced
  slope.  It vanishes identically when the last surface is flat.
* **`z_image`** -- the image-side free-space leg `fga.py` adds by hand after the
  differential trace.  It is an axial distance measured **from the exit-vertex
  plane**; that is what a back focal distance is, and it is what
  `apply_real_lens_fga`'s `output_plane_distance` means.
* **Fidelity** -- `|<a,b>|^2 / (<a,a><b,b>)` between two complex fields on the
  same grid; 1 means the same field up to a global complex scale.
* **Sag-screen aberration estimate** -- `_sag_screen_aberration_rad`, the cheap
  routing-time bound on the analytic thin-screen model's error, in radians of
  wavefront.  The universal dispatcher's `aberrated` condition is this estimate
  against the 2.0 rad `_ABERRATION_MAX_RAD` budget.

---

## 1. The defect, and the mechanism

`ray_transfer_jacobian` and `ray_transfer_jacobian_analytic` returned the base
ray's state and Jacobian **on the last surface** -- `res.image_rays`, with no
`at_exit_vertex()` step.  Four sites in `lumenairy/propagators/fga.py` consumed
that state as if it were on the exit-vertex plane:

| site | the line that assumes the vertex plane |
|---|---|
| `_fga_core` | `xv = dt.x + z_image*uxo`, `opd_tot = dt.opd + z_image*sqrt(1+u^2)` |
| `_fga_coarse`'s coarse trace (`_trace`) | the same two lines |
| `_fga_coarse` (the vector path) | the same two lines |
| `_caustic_zone` | `z = -x/u`, the axial crossing measured from `dt.x` |

Adding a leg that starts at `z = 0` to a state that sits at `z = sag(rho)`
double-counts the sag: every beamlet leaves with a spurious optical path of
`n_exit * sag(rho) * sec(theta)` and a spurious transverse offset of
`sag(rho) * u`.  Because `sag(rho) ~ rho^2/(2R)` to leading order, the spurious
phase is a defocus-plus-spherical term that grows quadratically with pupil
radius -- exactly the shape a beamlet sum cannot absorb into a global piston.

**Measured, on six fixtures** (`validation/probe_wp_b12/probe_a_mechanism.py`;
801-ray collimated fan, both backends, against the library's own
`at_exit_vertex()`).  Identical to every digit printed on both builds
(Windows py3.14 / numpy 2.4.4, WSL py3.12 / numpy 2.4.6):

| fixture | last surface | NA | sag at the rim | `|dx|` | `|d(opd)|` |
|---|---|---|---|---|---|
| WP-B7b's N-SF11 biconvex, R = +/-1.6 mm, 633 nm | R2 = -1.6 mm | 0.160 | 7.72 waves | 0.6616 um | **7.786 waves** |
| mine: N-BAF10 biconvex R = +/-2.10 mm, 1.064 um | R2 = -2.10 mm | 0.134 | 6.72 waves | 0.8406 um | **6.770 waves** |
| mine: N-LAK22 bent singlet R1 = +2.05 / R2 = -1.40 mm, 1.55 um | R2 = -1.40 mm | 0.143 | 5.32 waves | 1.0101 um | **5.362 waves** |
| mine: N-SF6 converging meniscus R1 = +1.30 / R2 = +9.0 mm, 850 nm | R2 = +9.0 mm | 0.083 | 1.05 waves | 0.0764 um | **1.058 waves** |
| the H2 f/5 dual-oracle biconvex, R = +/-51.68 mm, 1.31 um | R2 = -51.68 mm | 0.102 | 173.90 waves | 22.7096 um | **174.761 waves** |
| mine: N-LASF9 plano-convex, curved side FIRST, 850 nm (**control**) | flat | 0.150 | 0.00 waves | 0.0 (exactly) | **0.0 (exactly)** |
| VERIFY-B7b's fixture V, w0 = 190 um (**control**) | flat | 0.150 | 0.00 | 0.0 (exactly) | **0.0 (exactly)** |
| the same, w0 = 205 um (**control**) | flat | 0.150 | 0.00 | 0.0 (exactly) | **0.0 (exactly)** |

The defect is the sag, to three digits, on every row -- and it is exactly zero,
bit for bit, on all three flat controls.  The largest reading is on the very
fixture VERIFY-B7b nominated to decide the caustic route, the H2 f/5 biconvex:
**174.8 waves** of spurious optical path at its rim, which is why scoring the
route on it before the repair would have measured the defect (section 6.3).

### 1.1 The oracle, and what it is worth

`validation/probe_wp_b12/b12_common.py` imports nothing from `lumenairy` for its
physics: Schott Sellmeier coefficients typed in, intersection by Newton on the
implicit sag `F(t) = (z + t uz) - z_v - sag(|x + t ux|)` (the closed-form
quadric of `test_audit2609_b7b_caustic_routing.py` is a deliberately different
algorithm), refraction by vector Snell with the normal oriented AGAINST the
incident ray, and a brute-force Rayleigh-Sommerfeld-I sum
`E(P) = (1/i lambda) SUM_k A_k e^{ikr} z/r^2` over (exit ray) x (azimuth).

| control | measured (both builds, 2026-09-14) |
|---|---|
| my Sellmeier vs `get_glass_index`, 5 glasses (N-SF11 633 nm, N-BAF10 1.064 um, N-LAK22 1.55 um, N-SF6 850 nm, N-LASF9 850 nm) | **0.0e+00** on all five |
| my exit-VERTEX state vs `TraceResult.at_exit_vertex()` | height **5.4e-20 m**, slope **8.3e-17**, OPL **1.5e-18 m** |
| my LAST-SURFACE state vs `ray_transfer_jacobian` (default) | height **8.1e-20 m**, OPL **2.8e-18 m** |
| the slope-vs-direction-cosine trap (`max|u - L|`) | **3.1e-04 .. 1.7e-03**, i.e. 1e13x the agreements above -- the comparison is real |
| RS radial readout grid | 1/20 of an Airy radius out to 12 Airy radii, then one grid pitch |

A shared-model caveat, stated once: the oracle builds the exit-plane boundary
field from geometrical optics and then propagates it exactly, which is the model
`'traced'` implements, so an oracle-vs-member comparison is not neutral between
`'traced'` and the others.  It does not affect anything below, because the
decisive readings are model-free: the defect is measured directly against the
library's own `at_exit_vertex()` with no oracle involved, and the fidelity
differences at issue are between 0.07 and 0.9998.

---

## 2. Where the projection belongs, and why

Two designs were on the table.  **(a)** the primitives return the vertex-plane
state and every consumer follows; **(b)** the four `fga.py` sites project.
What shipped is neither literally: the primitives gained an explicit
`reference='surface' | 'exit_vertex'` keyword, defaulting to `'surface'`, and
the four sites pass `reference='exit_vertex'`.  The reasons, in order of weight:

1. **There is exactly one other consumer, and it would be silently
   double-corrected by (a).**  `grep -rn ray_transfer_jacobian lumenairy/` finds
   two modules that CALL the primitives: `propagators/fga.py` (the four sites)
   and `propagators/gbd.py`
   (`apply_prescription_persurface_to_beamlets`), which already carries its own
   in-line vertex correction -- a v5.22 fix that folds `-sag` into its image
   leg, `t = (z_image - _sag)/Nz2`.  Moving the primitives' default would make
   GBD subtract the sag twice.  Every other lens-family module reaches the
   vertex plane through `TraceResult.at_exit_vertex()` on a trace of its own
   (`_lens_traced.py`, `_lens_traced_multibranch.py`, `_lens_traced_uniform.py`,
   `_lens_real.py`, `lenses_maslov.py`) and never touches these primitives.
2. **The default keeps the raytrace package's single documented convention.**
   `trace` stops on the last surface; `image_rays` says so; `at_exit_vertex` is
   the one supported way off it.  A primitive of the same package that silently
   returned a different plane would be the only exception, and
   `DifferentialTransfer`'s 5.47.0 docstring (which states the last-surface
   convention) would have to become false rather than more precise.
3. **It makes the reference plane visible at the call site.**  The defect was
   an implicit assumption at four call sites; the repair is the same assumption
   written down at those four call sites, where a reader of `xv = dt.x +
   z_image*uxo` can check it.
4. **It is one implementation, not four.**  The projection lives in one private
   function, `differential._project_to_exit_vertex_plane`, used by the
   finite-difference backend, the numba analytic kernel, the `_AdrtDual` NumPy
   path and the JAX path.  The brief's suggested edit -- an inline `conic_sag`
   block copied to each `fga.py` site -- would have been four copies and an
   eighth hand-written vertex transfer in a library that spent WP-A1 removing
   seven of them; it would also have been wrong on three surface classes
   (section 2.1).

### 2.1 What the shared projection does that an inline `conic_sag` copy does not

* **Sag from the package's general surface kernels.**  `_surface_sag_xy` and
  `_surface_sag_derivatives_xy` (`raytrace/surface.py`) cover every surface type
  the finite-difference tracer supports -- conic, even asphere, biconic,
  freeform, and field-frame decenter / tilt / `field_sag_callable` -- so the
  projection is exact wherever the primitive is.  A `conic_sag(radius, conic)`
  copy silently drops the aspheric departure, the biconic y-branch and the
  whole field-frame class.  That is not hypothetical: `gbd.py` contains exactly
  such a copy, and section 5.1 measures it **15.52 waves** wrong on an A4/A6
  last surface (1.651e-05 m against a true sag of 2.309e-05 m).  On JAX
  the helper uses `conic_sag` / `conic_sag_derivs` with `xp=jnp`, whose
  rotationally-symmetric conic + even-aspheric domain is exactly the analytic
  backend's own.
* **The exit-medium index.**  `resolve_exit_index` reads
  `surfaces[-1].glass_after` (or `glass_before` for a mirror) and raises rather
  than guessing 1.0; the brief's edit hard-coded `sqrt(1+u^2)`, i.e. `n = 1`.
* **The propagation sign.**  A mirror reverses the outgoing ray, and the
  unreduced slope `u = L/N` does not record that (it flips with `N`).
  `_exit_direction_sign` recovers it from the prescription -- a deterministic
  property of the optic, not of the data -- so `t = -z/N` keeps its sign.
  Pinned in `test_a_mirror_terminated_prescription_projects_with_the_right_sign`.
* **The Jacobian.**  See section 3.
* **A structural flat-surface short-circuit.**  `_last_surface_sag_vanishes` is
  a property of the SURFACE (flat conic base, no aspheric coefficients, no
  biconic y-branch, no freeform, no field frame), so a flat-last-surface
  prescription returns the input object unchanged and the field is bit-for-bit
  what 5.47.0 produced.  A data-dependent `if not np.any(sag)` test would have
  been a decision that moves with the data.

---

## 3. Does the JACOBIAN need the projection?  Yes, and it is free

The brief asked this explicitly.  The transfer to the vertex plane is a
free-space propagation by `-sag` **along each ray**, and `sag` is itself a
function of where the ray lands, so the derivative of the composed map carries
two extra blocks:

```
J_vertex = P @ J_surface ,    P = [[ I - u (x) grad s , -s I ],
                                   [        0        ,   I  ]]
```

Both are implemented (`_project_to_exit_vertex_plane`), and the claim that they
are the right blocks is pinned against an INDEPENDENT finite difference of the
vertex-plane state in
`test_the_projected_jacobian_is_the_derivative_of_the_projected_map`: with a
per-row normalisation and a measured step ladder, the projected Jacobian matches
the FD reference at 2.1e-10 where the un-projected one sits at 2.1e-05.

What it buys in the FIELD is a fifth-decimal effect, measured on both planes of
every fixture (`probe_b_ladder`, the `proj_state` vs `proj_state_jac` arms):

| fixture | plane | state only | state + Jacobian |
|---|---|---|---|
| b7b biconvex (N = 256) | vertex | 0.99947649 | **0.99948793** |
| b7b biconvex | focus | 0.99981633 | **0.99981921** |
| b7b biconvex (N = 192) | focus | 0.99981144 | **0.99981760** |
| N-BAF10 biconvex | focus | 0.99981266 | **0.99981841** |
| N-LAK22 bent | focus | 0.99968105 | **0.99969526** |
| N-SF6 meniscus | focus | 0.99973978 | 0.99973881 |
| flat control | both | identical | identical |

Ten of twelve rows improve, two move down by 1e-6.  **The answer to the brief's
question is: the Jacobian leg changes the field in the fifth decimal, but it
costs nothing (four row updates on an `(N,4,4)` array already in cache) and it
is the derivative of the map that was actually applied, so it ships.**  Dropping
it would leave the transfer and its Jacobian on two different planes -- the same
class of defect this package repairs.

---

## 4. The oracle ladder -- FGA before and after

`validation/probe_wp_b12/probe_b_ladder.py` scores `apply_real_lens_fga`
against the section-1 oracle at the exit vertex and at each fixture's own
traced best focus, in four arms of the SAME process:

| arm | what it is |
|---|---|
| `pre_b12` | both primitives forced onto `reference='surface'` and nothing projected -- exactly what the tree did before this package |
| `native` | the tree as it stands |
| `proj_state` | the probe's OWN projection of the state only (its own sag formulae, not the library's) |
| `proj_state_jac` | the probe's own projection of state AND Jacobian |

The `pre_b12` arm is not a claim, it is checked: it reproduces the separate
pre-repair run of this probe (taken on the tree at `96cb2096`, before any
library edit) to every printed digit -- 0.0737 / 0.1252 on the WP-B7b fixture
and the zone `[925.335, 935.543] um`.  So the before/after below is a
like-for-like measurement in one process, and it is confirmed
archive-to-archive in section 8.

### 4.1 Windows build (py3.14, numpy 2.4.4, scipy 1.17.1)

`sha` is SHA-256 of the returned field's exact bytes.

| fixture | plane | `pre_b12` | `native` | `native` == `proj_state_jac` | field bytes |
|---|---|---|---|---|---|
| WP-B7b N-SF11 biconvex (N = 256, dx = 1.4 um) | exit vertex | 0.0737 (P 1.002) | **0.9995** | yes (0.999488) | changed |
| the same | focus 926.0 um | 0.1252 (P 1.001) | **0.9998** | yes (0.999819) | changed |
| the same (N = 192, dx = 1.8 um -- the pinned tests' grid) | exit vertex | 0.0802 (P 0.239) | **0.9993** | yes | changed |
| the same (N = 192) | focus | 0.1032 (P 0.182) | **0.9998** | yes | changed |
| N-BAF10 biconvex, 1.064 um | exit vertex | 0.0745 (P 0.366) | **0.9995** | yes | changed |
| the same | focus 1479.2 um | 0.0953 (P 0.274) | **0.9998** (P 0.978) | yes | changed |
| N-LAK22 bent singlet, 1.55 um | exit vertex | 0.1338 (P 0.293) | **0.9990** | yes | changed |
| the same | focus 1244.7 um | 0.1798 (P 0.206) | **0.9997** (P 0.992) | yes | changed |
| N-SF6 meniscus, 850 nm | exit vertex | 0.5363 (P 0.952) | **0.9993** | yes | changed |
| the same | focus 1495.3 um | 0.5306 (P 0.956) | **0.9997** (P 0.982) | yes | changed |
| **N-LASF9 plano-convex, FLAT last (control)** | exit vertex | 0.9979 | 0.9979 | yes | **IDENTICAL** |
| the same | focus 1430.3 um | 0.9982 | 0.9982 | yes | **IDENTICAL** |
| **VERIFY-B7b fixture V, w0 = 190 um (flat last)** | exit vertex | 0.9985 | 0.9985 | yes | **IDENTICAL** |
| the same | focus | 0.9972 | 0.9972 | yes | **IDENTICAL** |
| **fixture V, w0 = 205 um (flat last)** | exit vertex | 0.9978 | 0.9978 | yes | **IDENTICAL** |
| the same | focus | 0.9958 | 0.9958 | yes | **IDENTICAL** |
| the H2 f/5 biconvex, on its OWN 100 um grid | exit vertex | 0.0001 (P 4.442) | 0.2607 (P 4.504) | yes | changed |
| the same | focus 48.41 mm | 0.0745 (P 0.315) | 0.5653 (P 0.829) | yes | changed |

Every curved-last-surface row moves from 0.07-0.54 to 0.999-class; every
FLAT-last-surface row is **byte-identical**, same SHA-256, at both planes, on
all three flat fixtures.  Power ratios move with the fidelities (the WP-B7b
fixture at its test grid goes P = 0.182 -> 0.999; the bent singlet 0.206 ->
0.992).

The H2 f/5 rows are the exception that proves section 6.3's point: the repair
improves them by three decades at the vertex (1e-04 -> 0.26) and by a factor of
eight at the focus, but the residual is the GRID, not the model -- that
fixture's Airy radius is 8.07 um on a 100 um pitch, so no member can represent
its focus there.  Its defect is also the largest measured anywhere in this
package: **174.8 waves** of spurious optical path and 22.7 um of height at the
rim (section 1's probe, ninth fixture).

### 4.2 WSL build (py3.12, numpy 2.4.6, scipy 1.17.1)

The same probe, same four arms, on the second build
(`probe_b_ladder_linux_312.json`).  Reported in section 8 with the run tail.

### 4.3 The caustic zone

`_caustic_zone` estimates the geometric caustic from the axial crossings
`z = -x/u` of a 25-ray meridional fan.  Read from the last surface instead of
the vertex plane, every crossing is short by about the sag.  The reference here
is the SAME estimator evaluated on the probe's own exact trace -- the
definition is shared (it is what is being scored), the trace and the reference
plane are not:

| fixture | oracle `[z5, z95]` (um) | `pre_b12` | `native` |
|---|---|---|---|
| WP-B7b biconvex | [921.586, 935.509] | [925.335, 935.543] | **[921.586, 935.509]** |
| the same (N = 192) | [921.669, 935.510] | [925.396, 935.544] | **[921.669, 935.510]** |
| N-BAF10 biconvex | [1473.427, 1492.132] | [1478.760, 1492.180] | **[1473.427, 1492.132]** |
| N-LAK22 bent | [1238.338, 1259.541] | [1244.499, 1259.596] | **[1238.338, 1259.541]** |
| N-SF6 meniscus | [1493.625, 1499.241] | [1492.942, 1499.235] | **[1493.625, 1499.241]** |
| flat-last control | [1429.114, 1446.170] | [1429.208, 1446.273] | [1429.208, 1446.273] (unchanged) |

The `native` column agrees with the independent estimator to the last printed
digit on every curved fixture.  The near-edge shift removed is **0.41 %** of
the focal distance on the WP-B7b fixture, **0.50 %** on the bent singlet and
**0.36 %** on the N-BAF10 biconvex -- the brief's "~0.5 %".  On the meniscus
the near edge moves the OTHER way (+0.05 %), because that surface's sag is
positive: the sign follows the curvature, which is what a sag correction must
do and a fudge factor would not.  On the flat control both arms are identical
and the residual 1e-4 against the oracle is the estimator's own 25-ray
discretisation, unchanged by this package.

---

## 5. Blast radius

| consumer | what it asks for | effect |
|---|---|---|
| `fga._fga_core` | `reference='exit_vertex'` | **moves** on a curved last surface |
| `fga._fga_coarse` (scalar `_trace` and the vector path) | `reference='exit_vertex'` | **moves** |
| `fga._caustic_zone` | `reference='exit_vertex'` | **moves** |
| `gbd.apply_prescription_persurface_to_beamlets` | nothing (keeps `'surface'`) | **unchanged, bit for bit** |
| `raytrace.ray_transfer_jacobian_jax` | no state, no keyword | unchanged; its docstring's "its output is at the last vertex" corrected |
| every other lens module | calls `TraceResult.at_exit_vertex()` on its own trace | untouched |

Public entry points that move: `apply_real_lens_fga`,
`apply_real_lens_fga_vector`, `apply_real_lens_universal(method='fga')` and
`apply_real_lens_auto` when it dispatches to FGA -- on any prescription whose
LAST surface is curved.  A flat last surface is bit-identical.

### 5.1 GBD, measured rather than assumed

`validation/probe_wp_b12/probe_d_consumers.py` runs `apply_real_lens_gbd` on a
CURVED-last-surface singlet twice -- once as shipped, once with both primitives
forced back onto `reference='surface'` -- and SHA-256s the returned field:

```
GBD curved-last-surface field bit-identical to the forced-surface arm: True
```

So GBD is untouched, as the default guarantees.  The same probe compares GBD's
own in-line vertex correction (the `_Rl` / `_kl` block of
`apply_prescription_persurface_to_beamlets`, which folds `-sag` into its image
leg) against the sag the shared projection uses:

| last surface | GBD's in-line sag vs the shared sag | on a true sag of |
|---|---|---|
| conic (R = -2.10 mm) | **1.6e-19 m** (0.000 waves) | 6.947e-06 m |
| the same + A4 = 4.0e8, A6 = -8.0e17 | **1.651e-05 m = 15.52 waves** | 2.309e-05 m |

GBD's copy is exact on a conic last surface and wrong by 71 % of the sag on an
even-aspheric one, because it evaluates only the conic base and drops the
polynomial departure.  That is a **pre-existing GBD defect**, independent of
WP-B12 (it has been there since the v5.22 fix that added the correction), and
it is not fixed here: `gbd.py` is outside this package's ownership, the repair
would move every per-surface-GBD field on an aspheric-last-surface prescription,
and it needs its own oracle ladder.  It is section 9's first open item, and the
shared `_project_to_exit_vertex_plane` is the thing it should be replaced by.

---

## 6. The caustic route, re-scored, and the `aberrated` condition MEASURED

The route is **not changed** by this package; handoff section 4.7 reserves it
for the maintainer.  What follows is the measurement it should be decided on.
`validation/probe_wp_b12/probe_c_route.py` and `probe_e_envelope.py`, both
scored against the section-1 oracle at each fixture's own traced best focus.

### 6.1 The three members at the caustic, on nine fixtures

`cost` is wall clock on a heavily loaded box and is reported, never asserted
(`docs/TESTING_STANDARDS.md`); the ratio between the two columns is the
durable part.

| fixture | NA | sag-screen est. | route taken | `fga` | `phase_screen` | `traced` | `fga` / screen cost |
|---|---|---|---|---|---|---|---|
| WP-B7b N-SF11 biconvex, N = 256 | 0.160 | 0.074 rad | `phase_screen` | **0.9998** | 0.9991 | 0.9996 | 15.74 s / 0.53 s |
| the same, N = 192 (the pinned tests' grid) | 0.160 | 0.074 | `phase_screen` | **0.9998** | 0.9991 | 0.9996 | 1.61 s / 0.02 s |
| N-BAF10 biconvex, 1.064 um | 0.134 | 0.061 | `phase_screen` | **0.9998** | 0.9994 | 0.9996 | 7.24 s / 0.05 s |
| N-LAK22 bent singlet, 1.55 um | 0.143 | 0.047 | `phase_screen` | **0.9997** | 0.9991 | 0.9989 | 7.61 s / 0.11 s |
| N-SF6 meniscus, 850 nm | 0.083 | 0.088 | `phase_screen` | **0.9997** | 0.9996 | 0.9996 | 6.06 s / 0.04 s |
| N-LASF9 plano-convex (flat last), N = 256 | 0.150 | 1.404 | `phase_screen` | **0.9982** | 0.9650 | 0.9981 | 2.25 s / 0.03 s |
| VERIFY-B7b's fixture V, w0 = 190 um | 0.150 | 1.536 | `phase_screen` | 0.9972 | 0.9646 | **0.9983** | 6.39 s / 0.16 s |
| fixture V, w0 = 205 um (over budget) | 0.150 | 2.028 | `fga` | 0.9958 | 0.9550 | **0.9975** | 8.90 s / 0.22 s |
| the H2 f/5 dual-oracle biconvex | 0.102 | 20.355 | `fga` | not scorable -- see 6.3 | | | |

The `traced` column is read with `ray_subsample=2`, the DENSEST setting, so it
cannot be the reason `traced` loses where it loses.  At its shipped default of
8 it refuses five of these fixtures outright -- fewer than 32 coarse samples
across a sub-millimetre aperture -- which is correct behaviour and is open item
3.  `fga` and `phase_screen` read the same on both builds to every printed
digit; the `traced` column was taken on the WSL build.

Two readings reproduce VERIFY-B7b's independently: fixture V at w0 = 190 um
gives `fga` 0.9972 / `phase_screen` 0.9646 / `traced` 0.9983 against their
0.9967 / 0.9639 / 0.9978, from a different oracle.

### 6.2 The `aberrated` condition, measured as a ladder

The condition is `sag_screen_aberration_rad > _ABERRATION_MAX_RAD` (2.0 rad);
above it the caustic branch keeps `'fga'` instead of the screen.  Walking ONE
optic (fixture V, whose last surface is FLAT, so nothing on this ladder moves
with WP-B12 and the reading is about the SCREEN's model error) across the whole
envelope by beam radius alone:

| `w0` | sag-screen est. | route | `fga` | `phase_screen` | `traced` | screen infidelity |
|---|---|---|---|---|---|---|
| 60 um | 0.016 rad | `phase_screen` | 0.9995 | **0.9999** | 0.9997 | 1.0e-04 |
| 100 um | 0.121 | `phase_screen` | **1.0000** | 0.9995 | 0.9998 | 5.0e-04 |
| 140 um | 0.466 | `phase_screen` | **0.9999** | 0.9923 | 0.9998 | 7.7e-03 |
| 165 um | 0.893 | `phase_screen` | **0.9993** | 0.9802 | **0.9993** | 2.0e-02 |
| 190 um | 1.536 | `phase_screen` | 0.9972 | 0.9646 | **0.9983** | 3.5e-02 |
| 205 um | 2.028 | **`fga`** | 0.9958 | 0.9550 | **0.9975** | 4.5e-02 |

Three things this says, none of which was visible before the repair:

1. **The condition earns its keep.**  At the one rung above the budget the gate
   leaves the screen, and the screen is the WORST member there by 4.2e-02 of
   fidelity.  VERIFY-B7b's "do not drop the `aberrated` condition" holds, now
   measured on a post-repair ladder rather than on a fixture where FGA was
   broken.
2. **The budget is loose by about a decade.**  The screen's infidelity crosses
   1e-3 between 0.12 and 0.47 rad and 1e-2 between 0.47 and 0.89 rad, so by the
   time the estimate reaches the 2.0 rad budget the screen is already 35x worse
   than the members the gate is protecting the caller from.  If the envelope is
   meant to mean "the thin screen is accurate here", the measured boundary is
   nearer **0.3-0.5 rad**.
3. **There is a genuine crossover at the bottom.**  Below ~0.1 rad the screen
   is the best member (0.9999 against FGA's 0.9995): FGA's beamlet
   discretisation floor, not the screen's model error, is the limit there.  So
   the branch should not simply prefer FGA everywhere.

### 6.3 Why the H2 f/5 fixture cannot be scored at its caustic

VERIFY-B7b named the H2 f/5 dual-oracle biconvex (R = +/-51.68 mm, t = 5 mm,
n = 1.5168, 1.31 um, w0 = 5 mm, image at 49.163 mm) as the fixture that should
decide the condition after the repair.  Its ROUTING quantities are reproduced
here -- NA 0.1017, sag-screen estimate **20.355 rad**, route `'fga'` at the
caustic and `'traced'` at the vertex -- but its caustic cannot be scored
against a diffraction oracle on any grid this probe can run, and the reason is
arithmetic rather than effort:

* the beam is 5 mm, so the grid must span at least +/-12.8 mm;
* the Airy radius is `0.61 lambda / NA` = **8.07 um**, so resolving the focal
  structure needs `dx <~ airy/3` = 2.7 um;
* that is **N = 9515** (the probe computes it; at the fixture's actual
  100 um pitch, `dx / airy = 12.4`, i.e. the focal spot is a twelfth of a
  pixel).

An FGA call on a 9515^2 grid with a 13x13 momentum swarm is three orders of
magnitude past this probe's budget, and scoring the members on the fixture's
own 100 um grid measures the grid, not the models: FGA's beamlet width there is
`w0_factor * dx` = 500 um, sixty Airy radii, so every member returns a smoothed
blob and the comparison ranks nothing.  The estimate-ladder of 6.2 is the
tractable substitute, and it reaches 2.03 rad on a resolvable caustic.

### 6.4 The recommendation (measured; not taken here)

1. **Keep the `aberrated` condition.**  Two-sided evidence: above the budget
   the screen is the worst member by 4.2e-02 (6.2), and the over-budget class
   includes the H2 f/5 regime at 20.4 rad where the analytic model is 58-123 %
   wrong by the 2026-07-19 dual oracle.
2. **The caustic branch's screen-vs-FGA choice is now a COST choice, and on the
   current evidence the screen should stay** -- but for a different reason than
   5.47.0 gave.  Inside the low-estimate class the screen costs 7e-04 of
   fidelity for a 30x to 150x speed-up.  That is a defensible default; what is
   no longer defensible is the sentence that FGA is "the measurably worse
   member", and the comment at the branch now says so with the table.
3. **Consider tightening `_ABERRATION_MAX_RAD` from 2.0 rad to ~0.5 rad**, or
   splitting the branch so that an estimate above ~0.1-0.3 rad takes `'traced'`
   or `'fga'` rather than the screen.  Measured cost of leaving it as it is:
   2.0e-02 to 3.5e-02 of fidelity on a caustic plane whose estimate sits
   between 0.9 and 2.0 rad -- a band the gate currently sends to the screen.
   This is a default move with its own Migration note and is NOT taken here.
   Both builds read the ladder identically.
4. **`traced` deserves its own look at a caustic**: it is the best member on
   every row where its sampling guard lets it run (0.9983 and 0.9975 against
   FGA's 0.9972 and 0.9958), which is WP-B7b's own escalation (handoff 4.6)
   reproduced after the repair.

---

## 7. Files changed

| file | what |
|---|---|
| `lumenairy/raytrace/differential.py` | `reference` / `n_exit` on both primitives; `_project_to_exit_vertex_plane`, `_last_surface_sag_vanishes`, `_exit_direction_sign`, `_validate_reference`, `_finish_analytic`; module and `DifferentialTransfer` docstrings; the `ray_transfer_jacobian_jax` note |
| `lumenairy/propagators/fga.py` | the four call sites pass `reference='exit_vertex'`; the `_universal_route` caustic comment and the `apply_real_lens_universal` `'fga'` bullet re-measured |
| `docs/history/lumenairy.raytrace.differential.md` | fingerprints re-recorded with the reason |
| `tests/unit/test_audit2609_b12_fga_reference_plane.py` | NEW -- 14 tests (1 `slow`) |
| `tests/unit/test_audit2609_a4_fga_s10.py` | `test_s10_untilted_regimes_are_unchanged` restated as a reference-plane DECISION against a second tracer; `_zone_from_a_second_tracer` added |
| `tests/unit/test_audit2609_b7b_caustic_routing.py` | the member test restated (both members reproduce the oracle; `'fga'` is closer, as an infidelity ratio); two docstrings re-measured |
| `tests/unit/test_fga.py`, `test_fga_h4_h5.py`, `test_g1_gate_generality.py`, `test_niche_audit_w9_dispatch2.py`, `test_niche_p8_capstone.py`, `test_niche_p7_seidel_gate.py` | dated WP-B12 paragraph each, saying why their claims survive the move |
| `CHANGELOG.md` | a `## [Unreleased]` block above `## [5.47.0]` with the `### Fixed -- FGA: ...` entry and its Migration note |
| `docs/.../fixes/WP-B12_REPORT.md`, `WP-B12_CHANGELOG.md` | this report and the release text |
| `.test_durations` | 14 new ids for the new file (pytest-split `--store-durations`, merged in place); valid JSON, 16 200 entries |
| `validation/probe_wp_b12/` | `b12_common.py` (fixtures + oracle), `probe_a_mechanism.py`, `probe_b_ladder.py`, `probe_c_route.py`, `probe_d_consumers.py`, `probe_e_envelope.py`, `archprobe.py` and their JSON on both builds (the `.log` files are gitignored under `validation/**/*.log`) |

---

## 8. Tests run

Every invocation carried `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`
and `--capture=sys -p no:randomly -X faulthandler`, from `C:\tmp\lum_b12`, on
2026-09-14.  The box was running eight to twelve other heavy python jobs
throughout (three sibling Wave-5 agents and the maintainer's own runs), so
every wall clock below carries contention; they are reported, never asserted.

### 8.1 The two builds

| | Windows | WSL |
|---|---|---|
| python | 3.14.6 | 3.12.3 |
| numpy / scipy | 2.4.4 / 1.17.1 | 2.4.6 / 1.17.1 |
| numba / jax | 0.65.1 / 0.11.0 | 0.65.1 / 0.10.2 |

### 8.2 pytest

| selection | build | result | duration |
|---|---|---|---|
| the eight pinned files, BEFORE restating them | Windows | **2 failed, 178 passed** -- `test_audit2609_a4_fga_s10.py::test_s10_untilted_regimes_are_unchanged` (the three literal caustic-zone readings) and `test_audit2609_b7b_caustic_routing.py::test_b7b_phase_screen_is_the_closer_member_at_the_caustic` (`assert f_fga < 0.5`, now 0.9998).  Nothing else in the eight files moved. | 1609.3 s |
| `test_audit2609_a4_fga_s10.py` after restating | Windows | **16 passed** | 52.9 s |
| `test_audit2609_b7b_caustic_routing.py` after restating | Windows | passed (with s10, 25 passed) | 165.6 s |
| `test_audit2609_b12_fga_reference_plane.py` (NEW, 14 ids incl. 1 `slow`) | Windows | **14 passed**; slowest id 11.3 s, so every test is inside the 60 s budget | 33.2 s |
| the eight pinned files + the new one, all restatements in | Windows | **194 passed, 0 failed** | 1571.7 s |
| the same nine files | WSL | **194 passed, 0 failed** -- the same 194 ids, the same verdict, a different interpreter and a different LAPACK | 3070.1 s |
| `test_raytrace.py`, `test_analytic_ray_transfer.py`, `test_gbd_feature_complete.py`, `test_audit2609_a1_exit_vertex.py`, `test_audit_raytrace.py`, `test_audit_w3_raytrace_parity.py`, `test_audit_w5_raytrace_bundles.py`, `test_audit_w6_raytrace.py`, `test_niche_audit_w3_raytrace_sources.py`, `test_audit2609_a1_raytrace.py`, `test_audit2609_b9_raytrace_perf.py`, `test_v5_4_1_raytrace_mirror_backward_ray.py`, `test_v5_4_6_wave8_raytrace.py` | Windows | **388 passed, 0 failed** | 806.7 s |
| the census / walker / dispatcher-pin / public-API / doc-consistency / history / `test_audit_except_budget.py` sweep (32 files), first run | Windows | 4 failed, 1623 passed, 11 skipped -- see 8.3 | 1248.9 s |
| the same sweep, after re-recording the history fingerprints | Windows | **1627 passed, 11 skipped, 0 failed** | 635.3 s |
| the same sweep, in a `git archive 96cb2096` tree (the parent) | Windows | **1626 passed, 12 skipped, 0 failed** | 1094.2 s |

There is no `tests/unit/test_*differential*.py`: the two differential-transfer
primitives are covered by `test_analytic_ray_transfer.py`,
`test_gbd_feature_complete.py`, `test_audit2609_a1_raytrace.py` and
`test_audit2609_b9_raytrace_perf.py`, all of which are in the 388 above.

### 8.3 The sweep, and one order-dependent red that is not this package's

The first sweep run read **4 failed, 1623 passed, 11 skipped** in 1248.9 s.
Three were `test_audit2609_a17_history_relocation.py` (both parametrisations
for `lumenairy.raytrace.differential`) and
`test_audit2609_a22_history_fingerprint_tool.py::test_every_committed_history
_document_matches_its_module` -- the history fingerprints, which had not yet
been re-recorded at that point; `python scripts/record_history_fingerprints.py
lumenairy/raytrace/differential.py --reason "..."` clears them and
`--check` then reports "every history document matches its module".

The fourth was `test_public_api.py::test_installed_metadata_version_matches
_source_version`, which read `importlib.metadata.version('lumenairy') ==
'3.7.8'` against `__version__ == '5.47.0'` -- a version string that exists
nowhere in this tree, in site-packages (which holds one
`lumenairy-5.47.0.dist-info`) or in the shared editable install's `egg-info`
(also 5.47.0).  It is a TRANSIENT, not a WP-B12 effect, and four measurements
say so: the same id passes alone in 0.3 s; it passes with the three
`sys.path`-mutating files of the selection run ahead of it; the whole selection
is green in a `git archive 96cb2096` tree (1626 passed, 12 skipped); and the
same selection re-run in this worktree is green (1627 passed, 11 skipped).  No
file WP-B12 touches is in that selection, and nothing in `differential.py` or
`fga.py` can reach `importlib.metadata`.  It is recorded here because a
metadata read that returns a version nobody can locate is worth someone's
attention -- it belongs beside the other order-dependent one-shot reds of
handoff section 1, not in this package.

### 8.4 The other gates

| gate | result |
|---|---|
| `wsl ruff check lumenairy/ tests/` | **All checks passed** |
| `wsl ruff check validation/probe_wp_b12/` | **All checks passed** |
| `python scripts/record_history_fingerprints.py --check` | **OK: every history document matches its module** |
| `python scripts/check_source_line_citations.py` (V18) | **ok=107, drift=0, total=107** |
| `python scripts/check_doc_identifiers.py` | **OK**; 621 API-claiming identifiers, 0 unresolved |
| `.test_durations` | 16 200 entries, valid JSON, 14 new ids for the new file, largest 32.4 s |

### 8.5 The archive-to-archive byte-identity proof

`git archive 96cb2096 | tar -x` into one scratch tree (the parent) and the
package's `lumenairy/` into another (the head), then
`validation/probe_wp_b12/archprobe.py` run in a CHILD process in each, with
`cwd` and `PYTHONPATH` set to that tree and `lumenairy.__file__` asserted to
live under it -- never through pytest, never against a shared tree:

| case | parent digest | head digest | verdict |
|---|---|---|---|
| flat last surface, at the exit vertex | `517818e86df635fc...` | `517818e86df635fc...` | **IDENTICAL** |
| flat last surface, at its focus | `c5dd570bf17b7023...` | `c5dd570bf17b7023...` | **IDENTICAL** |
| flat last surface, `_caustic_zone` | `[1429.207520666257, 1446.2730275843817] um` | the same to 16 digits | **IDENTICAL** |
| WP-B7b biconvex at its focus | `0916059cfa5b6f0e...` | `b278341cad84da48...` | changed |
| the same, `_caustic_zone` | `[925.3958338294063, 935.5435756141574] um` | `[921.668644366447, 935.509880287378] um` | moved |
| N-BAF10 biconvex at the exit vertex | `54179a51ff853adf...` | `7b846080ae11b4af...` | changed |
| the same, `_caustic_zone` | `[1478.6766505115946, 1492.179217333539] um` | `[1473.3111197389695, 1492.1307011541158] um` | moved |

The head tree is a copy of this worktree's `lumenairy/`, taken after the last
code edit; this worktree is exclusive to this package (no other agent writes in
it), and `git archive HEAD lumenairy` is verified against that copy after the
commit.

### 8.6 Probes

All under `validation/probe_wp_b12/`, each run on both builds with
`PYTHONPATH` pinned to the tree and `lumenairy.__file__` printed:

| probe | what | outputs |
|---|---|---|
| `probe_a_mechanism.py` | the mechanism and its controls on nine fixtures; the repair against `at_exit_vertex()` | `probe_a_mechanism_win32_314.json`, `..._linux_312.json` |
| `probe_b_ladder.py` | the four-arm oracle ladder at two planes per fixture, plus the caustic zone | `probe_b_ladder_win32_314.json`, `..._linux_312.json` |
| `probe_c_route.py` | the three members at the caustic on nine fixtures, with the routing quantities | `probe_c_route_win32_314.json`, `..._linux_312.json` |
| `probe_d_consumers.py` | GBD's blast radius and its in-line sag copy | `probe_d_consumers_win32_314.json`, `..._linux_312.json` |
| `probe_e_envelope.py` | the `aberrated` condition as a ladder | `probe_e_envelope_win32_314.json`, `..._linux_312.json` |
| `archprobe.py` | the archive-to-archive digests (8.5) | run in the scratch trees |

**Cross-build agreement.**  Probe A is identical to every printed digit on the
two builds.  Probe B's 72 fidelity readings (nine fixtures x two planes x four
arms) agree to a maximum of **8.7e-15**, and the six byte-identity verdicts on
the flat fixtures agree exactly.  Probe C's 52 readings agree to **5.6e-16**
and every one of its eighteen routing decisions is the same on both builds;
probe E's ladder agrees to **6.7e-16** with the same six routes.  Probe D is
identical to every printed digit.  Nothing in this package's evidence sits
inside a cross-build spread.

---

## 9. Open items

1. **`gbd.py`'s in-line vertex correction should become a call to the shared
   projection** (section 5.1).  Measured: exact on a conic last surface, 15.52
   waves wrong on an A4/A6 aspheric one, i.e. 71 % of the sag.  The replacement
   is `reference='exit_vertex'` on the `ray_transfer_jacobian` call plus
   deleting the `_Rl` / `_kl` block and the `- _sag` in `t`; it would move
   per-surface GBD fields on every aspheric-last-surface prescription (a
   correction) and is not bit-identical even on a conic one (different order of
   operations), so it needs its own package, oracle ladder and Migration note.
2. **The caustic route and `_ABERRATION_MAX_RAD`** -- section 6.4, three
   measured recommendations left for the maintainer (handoff 4.7): keep the
   `aberrated` condition; keep the screen inside the low-estimate class as a
   cost choice; consider tightening the 2.0 rad budget to ~0.5 rad or sending
   the 0.9-2.0 rad band to `'traced'`.
3. **`traced` at a caustic.**  On every row where its sampling guard lets it
   run, `traced` is the best member (0.9983 / 0.9975 against FGA's 0.9972 /
   0.9958).  That is WP-B7b's own escalation (handoff 4.6) and it now stands on
   post-repair numbers.  Its `ray_subsample=8` default also refuses five of the
   nine fixtures here outright; the refusal is correct (fewer than 32 coarse
   samples across the aperture) but it means `method='traced'` is not reachable
   on a sub-millimetre aperture without a keyword.
4. **A non-air exit medium.**  The projection resolves `n_exit` from the
   prescription, but `fga.py`'s own image-side leg is `z_image *
   sqrt(1 + u^2)` with no index, i.e. it assumes the exit medium is vacuum.
   On an immersed prescription the two now disagree by `(n_exit - 1) * sag *
   sec`.  Pre-existing and separate; every fixture in this library's FGA suite
   ends in air, so it is not measured here.  A caller in that situation should
   be refused rather than silently served -- worth a guard.
5. **`ray_transfer_jacobian_jax` has no `reference` keyword.**  It returns only
   a Jacobian (no state, no OPD), so nothing can misread its reference plane;
   its docstring's "its output is at the last vertex" is corrected to say the
   last SURFACE.  A caller needing the vertex-plane Jacobian on JAX uses
   `ray_transfer_jacobian_analytic(..., reference='exit_vertex')`, whose JAX
   branch is exercised in section 8.
6. **Not measured: a freeform or field-frame last surface through FGA.**  The
   projection takes its sag from `_surface_sag_xy` /
   `_surface_sag_derivatives_xy`, which cover those classes, and
   `test_an_aspheric_last_surface_projects_with_its_polynomial_departure`
   covers the even asphere against `at_exit_vertex()`.  A freeform-last-surface
   FGA field was not scored against a diffraction oracle: the probe's oracle is
   a rotationally-symmetric conic trace and cannot represent one.  The
   projection is correct there by construction (it uses the same sag kernels
   the tracer does), but "correct by construction" is not "measured".
7. **Not measured: the cross-build spread of the fidelity numbers.**  Both
   builds here are the same box; the CI mix of EPYC 9V74 and 7763 with older
   wheels was not sampled.  Every bar in the new test file is derived at
   runtime from a quantity the running build measures, and the two decisions
   that carry fixed constants (the 1e-14 m plane bars and the 0.99 fidelity
   bar) sit four to seven decades from anything a BLAS kernel can move.

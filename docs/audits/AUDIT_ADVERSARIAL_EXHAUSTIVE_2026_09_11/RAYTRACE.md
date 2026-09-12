# RAYTRACE audit — `lumenairy/raytrace/` (sequential ray tracer, paraxial/Seidel analytics, differential + JAX backends, field→rays bridge)

All repro scripts live in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/RAYTRACE/`.
Nothing in the repository was created, modified or deleted.

## Scope read (line by line)

| file | lines read | notes |
|---|---|---|
| `_conic_core.py` | 1–319 (all) | |
| `intersection.py` | 1–653 (all) | |
| `surface.py` | 1–671 (all) | |
| `trace.py` | 1–1708 (all) | |
| `world.py` | 1–369 (all) | |
| `world_trace.py` | 1–240 (all) | |
| `paraxial.py` | 1–267 (all) | (`system_abcd` is **not** here — it is in `seidel.py`) |
| `seidel.py` | 1–400, 733–1060, 1274–1742 | **not read: 400–733** (`LensInfo`/`lens_abcd`/`PupilInfo` dataclasses), **1060–1274** (`compute_pupils` body, `find_lenses`) — read their docstrings/contracts only |
| `seidel_analysis.py` | 185–405 | **not read: 1–185** (`seidel_field_sweep` body) |
| `differential.py` | 1–260, 298–560, 856–1060 | **not read: 560–856** (the numba dual kernel builder) |
| `jax_trace.py` | 96–835, 1040–1170, 1494–1644 | **not read: 835–1040** (`_build_jax_prescription`, `_trace_body_*` bodies), **1170–1494** (`trace_jax` body, `_intersect_jax_param` / `_refract_jax_param` bodies) |
| `from_field.py` | 87–423, 678–824 | **not read: 423–678** (the three `_place_*` samplers) |
| `ray_fan.py` | 40–793 (all) | |
| `bundles.py`, `layout.py`, `core.py`, `__init__.py` | all | |

Also read for cross-checks: `elements/_lens_traced.py` (exit-vertex blocks), `elements/_lens_jax.py` (ditto), `ui/model.py` (tilt grep only).

---

## Findings

### [P1] `opd_fan_data` / `opd_fan_data_world` do not compute wavefront error — the reference sphere is missing — `ray_fan.py:460`, `ray_fan.py:523`

**What is wrong.** Both functions return `(img.opd - opd_chief) / wavelength` where `img.opd` is the OPL accumulated to each ray's **own intercept on the final surface**. The wavefront aberration is the OPL difference measured to a **common point** (a reference sphere centred on the Gaussian image point). The two differ at **first order** in the transverse aberration: `W_plane − W_true = ε·sin θ′`, where `ε` is the ray's transverse aberration and `θ′` its exit angle. There is no reference-sphere term anywhere in `raytrace/`.

**Evidence** (`p8_opdfan.py`, `p9_opdfan_confirm.py`; oracle = OPL to the paraxial image point, independently cross-checked against `−S1_code/8` from `seidel_coefficients`):

| system | `opd_fan_data` @ρ=1 | true W @ρ=1 | ε @ρ=1 | `\|ε·sinθ′\|` | measured diff |
|---|---|---|---|---|---|
| f/4 plano-convex (convex first) | **+36.194 w** | −11.714 w | −222.8 µm | 48.33 w | 47.91 w |
| f/4 plano-convex (flat first) | **+151.79 w** | −45.05 w | −913.8 µm | 203.67 w | 196.84 w |
| f/20 plano-convex | +0.0545 w | −0.0181 w | −1.71 µm | 0.0726 w | **0.0726 w** |
| f/50 plano-convex | +0.0014 w | −0.0005 w | −0.11 µm | 0.0019 w | **0.0019 w** |
| spherical mirror R=−200 mm, h=25 mm | **+64.50 w** | ±20.7 w | — | — | — |

The exact agreement of the residual with `ε·sinθ′` at f/20 and f/50 identifies the mechanism unambiguously. Control: an aberration-free parabolic mirror (ε≡0) gives `opd_fan` PV = 7.1e-11 waves — the function is right exactly when it does not matter. Cross-check: the reference-sphere oracle (−11.71 w) agrees with the independently-computed Seidel `−S1/8` (−11.30 w, the remainder being genuine 5th order), so the oracle, not the library, is right.

**Impact.** `opd_fan_data` is in `lumenairy.__all__` and feeds `analysis/plotting.plot_opd_fan`. For an f/4 singlet it reports the wrong **sign** and 3.1× the magnitude of the true WFE. Anything reading it as wavefront error (RMS WFE, Strehl proxy, Zernike fit, merit function) is wrong by that factor for any system with non-negligible transverse aberration.

**Fix.** Reference each ray to the image point:
```python
z_img = 0.0            # last surface is the image plane; else use bfl
seg = np.hypot(np.hypot(img.x - x_img, img.y - y_img), z_img - img.z)
opl = img.opd + n_img * np.sign(z_img - img.z) * seg
```
i.e. add the leg from each intercept to the chief ray's intercept (Welford §4, "the optical path from the object to the Gaussian image point"). `refocus()` already knows the image-space index and the sag→plane correction; this is a 4-line addition to both fan functions.

---

### [P1] `opd_fan_data` at non-zero field carries a spurious `y·sin(θ)` launch-plane tilt — `ray_fan.py:503`, `trace.py:788` (`_make_bundle` seeds `opd = 0`)

**What is wrong.** `_make_bundle` sets `opd = 0` for every ray at the `z = 0` launch plane. For a field-angle bundle the incident wavefront is **not** the `z = 0` plane: a ray launched at height `y` with direction `(0, sinθ, cosθ)` has already accumulated `y·sinθ` of path relative to a common incident wavefront. The OPD fan therefore carries an uncorrected linear term of exactly `−y·sinθ`.

**Evidence** (`p8_opdfan.py`, f/4 plano-convex, 25 mm pupil):

| field | fitted linear term of the OPD fan | `y_max·sinθ/λ` | ratio | real (quartic) aberration |
|---|---|---|---|---|
| 0.0° | +0.00 w | 0 | — | +37.17 w |
| 0.5° | **−185.64 w** | +185.64 w | **−1.0000** | +37.02 w |
| 2.0° | −744.06 w | +742.42 w | −1.0022 | +37.11 w |
| 5.0° | −1879.86 w | +1854.06 w | −1.0139 | +37.65 w |

At 5° the fan's PV is 3717 waves of which ~1880 are pure launch-convention artefact against 38 waves of real aberration — a 50:1 contamination.

**Impact.** Every off-axis OPD fan is dominated by the artefact. Note the library already knows about this class of error: `elements/_lens_traced.py:9585` ("v5.25.1 hammer audit H6: carrier entrance eikonal ... The ray tracer accumulates OPL only from the ENTRANCE plane forward") adds the entrance eikonal on the real-lens path — the fix was never propagated to `opd_fan_data`.

**Fix.** Seed `opd = -(x·L + y·M)` (the eikonal of the incident plane wave at the launch plane) in `_make_bundle`, or subtract the fitted linear term inside `opd_fan_data` (and document which). Seeding in `_make_bundle` is bit-identical on axis (`L = M = 0`).

---

### [P1] `seidel_coefficients` silently ignores the conic constant and every aspheric coefficient — `seidel.py:1454-1650`

**What is wrong.** The per-surface loop reads only `surf.radius`, `glass_before/after`, `thickness`, `is_mirror`. `grep -n "conic\|aspheric" lumenairy/raytrace/seidel.py` returns **zero hits**. There is no aspheric Seidel contribution and no warning; the sums returned for a conic/aspheric surface are those of its base sphere. The docstring does not disclose this.

**Evidence** (`p4_seidel.py`, `p5_seidel_asph.py`):

* Mirror R = −200 mm, h = 25 mm: `S1 = +9.765625e-05` for **k = 0, −0.5, −1.0, −1.5 alike**. A parabolic mirror (k = −1) at infinite conjugate is exactly aberration-free — measured ray spread at the focus **0.000 µm** and OPL constant across the pupil to **1.4e-17 m** — yet the library reports `−S1/8 = −12.207 µm` of spherical.
* Singlet R1 = 51.68 mm, N-BK7, 25 mm pupil: `S1 = +5.320645e-05` for **A4 = 0, −250, −500, −1000, −2000 m⁻³ alike**, while the real-ray ρ⁴ coefficient swings −6.639 µm → −0.333 µm → +18.613 µm (a sign change). At A4 = −500 the lens is nearly aplanatic (0.33 µm) and the library still reports 6.65 µm.

**Fix (validated numerically).** In the library's own sign convention (`code = −S_Welford`), add to each refracting/mirror branch:
```python
A4_eff = (conic / (8.0 * R**3) if np.isfinite(R) else 0.0) \
         + (surf.aspheric_coeffs or {}).get(4, 0.0)
dS = 8.0 * (n2 - n1) * A4_eff * y_val_m**4
S1[i] += dS
S2[i] += dS * (y_val_c / y_val_m)
S3[i] += dS * (y_val_c / y_val_m)**2
S5[i] += dS * (y_val_c / y_val_m)**3      # S4 (Petzval) is unaffected
```
Measured agreement of `−(S1+dS)/8` with the real-ray ρ⁴ fit: **0.14–0.29 %** across A4 ∈ {0, ±250, ±500, −1000, −2000} (the residual is genuine 5th order, consistent with the fitted ρ⁶ term), and **1.4e-20** (exact cancellation) for the parabolic mirror at every k ∈ {0, −0.5, −1, −1.5}. Reference: Welford, *Aberrations of Optical Systems*, §8.5 (aspheric contribution `ΔS_I = 8 Δn A4 h⁴`, with the higher sums scaling as powers of `h_chief/h_marginal`).

---

### [P1] Conic surfaces falsely report `RAY_MISSED_SURFACE` for any ray whose closest approach to the centre of curvature exceeds `|R|` — `intersection.py:266`, mirrored in `jax_trace.py:292`

**What is wrong.** The Newton branch seeds from the ray–**sphere** quadratic and takes its discriminant as the miss test:
```python
missed_init = (disc < 0) & rays.alive        # intersection.py:266
```
A sphere of radius `R` only exists for `h ≤ |R|`; a paraboloid (k = −1), hyperboloid (k < −1) or moderately-flattened prolate ellipsoid extends well beyond that. The ray genuinely hits the conic, but the *sphere* seed has no real root, so the ray is killed with `RAY_MISSED_SURFACE`. `jax_trace._intersect_jax` has the identical test (`miss |= ~disc_ok`), so the two backends are consistently wrong.

**Evidence** (`p1b_intersect.py`, `p12_conicdomain.py`). Thorlabs-class aspheric condenser geometry R = 10.84 mm, k = −0.6 (conic valid to h = R/√(1+k) = 17.14 mm), clear semi-diameter 11.5 mm:

| h [mm] | alive | err | t returned [mm] | true sag [mm] |
|---|---|---|---|---|
| 10.00 | True | 0 | 5.090684 | 5.090684 |
| 10.83 | True | 0 | 6.095530 | 6.095530 |
| **10.90** | **False** | **3** | 0.000000 | **6.186249** |
| **11.40** | **False** | **3** | 0.000000 | **6.863647** |

Threshold measured at `h/|R| = 1.0517` for the first dead ray of a 41-ray fan through the public `trace()` API (2/41 rays killed). Parabola R = 50 mm at h = 60 mm: library returns `alive=False, t=0` where a brentq root-find on the sag gives t = 36.000000 mm.

Cross-backend: `trace` and `trace_jax` both kill these rays; **`differential.ray_transfer_jacobian_analytic` keeps them all alive** (measured `alive = [T T T T T]` on the bundle where the other two give `[T T T F F]`) because `_adrt_step` (`differential.py:424-433`) solves the *exact* implicit conic quadratic `F = c(x²+y²) − 2z + (1+k)c z² = 0`. So the library already contains the correct primitive.

**Impact.** Any fast/large-aperture conic or asphere whose clear aperture exceeds its base radius silently loses its outer rays — exactly the rays that carry the aberration content, and exactly the geometry aspheric condensers and fast parabolic mirrors have. Vignetting is reported as `RAY_MISSED_SURFACE`, so it does not even look like an aperture problem.

**Fix.** Replace the sphere seed + sphere-disc miss test with the exact conic quadratic that `_adrt_step` already uses (it is also the Spencer & Murty 1962 / Welford stable form):
```python
a = c*(L*L + M*M) + (1+k)*c*N*N
b = 2.0*(c*(x*L + y*M + (1+k)*z*N) - N)
e = c*(x*x + y*y + (1+k)*z*z) - 2.0*z
disc = b*b - 4*a*e                       # THIS is the correct miss test
q = -0.5*(b + np.sign(b)*np.sqrt(disc))  # no cancellation near the vertex
t = np.where(np.abs(a) < 1e-14, -e/b, e/q)
```
For a pure conic this is exact (no Newton at all); for a conic + polynomial asphere it is the correct starting point and the correct miss test, with Newton refining only the polynomial departure.

---

### [P1] The diffraction-order kick omits the medium index — `trace.py:215`, `world_trace.py:181`, `trace.py:1172` (`apply_doe_phase_traced`), `jax_trace.py:485`

**What is wrong.** All four sites apply `ΔL = m·λ_vac/Λ` to the direction cosines **after** refraction into `glass_after`. The grating equation conserves the *tangential wavevector*: `n₂ L′ = n₁ L + m λ_vac/Λ`, so the correct post-refraction kick is `ΔL = m λ_vac / (n₂ Λ)`. In air (`n₂ = 1`) the library is exact; at any interface into glass it is high by a factor `n₂`.

**Evidence** (`p11_misc.py`): a grating on the glass side of an air→N-BK7 interface, Λ = 5 µm, λ = 1.31 µm, m = 1:
```
library L inside the glass       = 0.26200000
grating eq.  m*lam/(n2*Lambda)   = 0.17425045
ratio                            = 1.503583   ==  n(N-BK7) exactly
```
A 50 % direction error (θ = 15.2° instead of 10.0°).

**Impact.** Any DOE/grating registered on a surface whose `glass_after` is not air. Also note the docstring's stated approximation ("neglects the cosine factor that distinguishes `sin` from the direction cosine") is itself inaccurate: for in-plane diffraction the grating equation *is* exact in direction cosines; the real approximation is the missing `1/n₂`.

**Fix.** Divide by `n2` at all four sites (`n_post[i]` is already resolved in both trace loops); for `apply_doe_phase_traced`, add an `n_medium: float = 1.0` keyword.

---

### [P2] `rays_from_field`'s phase-gradient estimator aliases above **half** the grid's own Nyquist angle — `from_field.py:767`

**What is wrong.** `kx = angle(E[j+1]·conj(E[j-1])) / (2 dx)` is unambiguous only for `|kx·2dx| < π`, i.e. `|L| < λ/(4 dx)` — half of what the grid itself supports (`λ/(2 dx)`). Above that it silently wraps.

**Evidence** (`p7_fromfield.py`, λ = 1 µm, dx = 2 µm ⇒ grid Nyquist |L| ≤ 0.25, estimator valid |L| < 0.125):

| L_true | L_recovered |
|---|---|
| 0.100 | +0.100000 |
| **0.150** | **−0.100000** |
| **0.200** | **−0.050000** |
| **0.300** | +0.050000 |
| **0.490** | −0.010000 |

It also falsifies the docstring's justification for choosing this form over `Im(∇E/E)`: it claims the form "can detect evanescent rays whose tangential k exceeds π/Δx because the central-difference output rolls back through zero" — measured, an evanescent `L = 0.49` comes back as a benign `L = −0.01`, i.e. no evanescent ray is ever flagged.

**Fix.** Use a symmetrised **1-pixel** difference, which is exact to the full grid Nyquist and still centred:
`kx = angle(E[j+1]·conj(E[j]) + E[j]·conj(E[j-1])) / dx`.

### [P2] `rays_from_field` gives edge rays exactly **half** the correct direction cosine — `from_field.py:738-741`

`ix_plus = clip(ix+1, 0, Nx-1)` collapses to `ix` on the last column (and `ix_minus` to `ix` on the first), so the 2-pixel baseline becomes 1 pixel while the divisor stays `2 dx`. Measured (`p7_fromfield.py`, uniform-amplitude tilted plane wave, `L_true = 0.05`, 64×64): inner pixels `L = +0.050000`, **edge columns `L = +0.025000`, ratio 0.5000 exactly**, over 128 edge rays. Fix: divide the boundary columns/rows by `dx`, not `2 dx` (or drop boundary pixels from placement).

### [P2] `_transfer` has no grazing-ray guard — the "immortal phantom" the R-4 fix removed from `_intersect_surface` still exists here — `intersection.py:492-509`

`t` is forced to 0 when `|N| ≤ 1e-30`, but `rays.z = np.zeros_like(rays.z)` runs unconditionally, so the ray is **teleported** to the next vertex plane with zero OPL and stays `alive=True, error_code=0`. Measured (`p11_misc.py`): a bundle at `z = 1e-4` with `N = 0` comes out of `_transfer(10 mm, n=1)` as `z=[0. 0.], alive=[True True], opd=[0. 0.], error_code=[0 0]`. This is precisely the case `intersection.py:143-158` (R-4) documents as an "IMMORTAL PHANTOM" and fixes for the flat-intersection branch — and `trace.py:153` explicitly keeps `N == 0` DOE orders alive *by design*, so the state is reachable. Fix: apply the same `graze` kill (`RAY_MISSED_SURFACE`) in `_transfer`, or leave `z` untouched for grazing rays.

### [P2] Five hand-rolled copies of the exit-vertex transfer; no shared helper, and the JAX copies differ — `elements/_lens_traced.py:6673,9581,11104`, `elements/_lens_jax.py:556,829`

`trace()` leaves rays at `z = sag(h)` of the last surface (documented at `ray_fan.py:600`). Every consumer that needs the exit **vertex** plane re-implements `t = -z/N; opd += n_exit*t; x += L*t; y += M*t; z = 0`. Five copies, three of them with a long explanatory comment repeating the same derivation. The NumPy copies mask on `alive & (|N| > 1e-30)`; the two JAX copies mask on `alive` only and clamp `N` to `1e-30`, so a grazing ray gets `t = -z/1e-30` there instead of 0 — a cross-backend divergence in the same primitive.

`raytrace.refocus(result, 0.0)` is already exactly this operator (it handles the sag→plane correction and the signed OPL, and resolves `n_exit` from `surfaces[-1].glass_after`). Fix: export a `transfer_to_exit_vertex(rays, n_exit)` helper from `raytrace/` and route all five sites through it.

### [P2] No in-place arithmetic anywhere in the trace hot path — ~4 full bundles of transient memory per call

Measured (`p10_perf.py`, `tracemalloc`, N = 1e6 rays, 7 surfaces): peak **257.5 MiB** with `output_filter='last'` and **510.2 MiB** with `'all'`, where one `RayBundle` is ~65 MiB. Every statement of the form `rays.x = rays.x + rays.L * t` allocates two full-grid temporaries; `_intersect_surface` alone allocates `t, dx, dy, dz, b, c, disc, disc_safe, sqrt_disc, t1, t2, h_sq, inside, clipped` plus the three position updates — ~15 N-sized arrays per surface. `_refract` adds another ~10 (`np.where` ×3, `mag`, 3 divisions) for a renormalisation that is a mathematical no-op (exact vector Snell returns a unit vector identically).

cProfile (N = 200k, 7 surfaces, 3 calls, box under heavy parallel load):

| function | tottime | share |
|---|---|---|
| `_intersect_surface` | 1.068 s | 32 % |
| `_conic_core.refract_snell` | 0.684 s | 20 % |
| `_base_surface_sag_derivatives_xy` + `_surface_normal` + `_surface_sag_derivative` | 0.80 s | 24 % |
| `_transfer` | 0.181 s | 5 % |

Fixes, in order of payoff: (a) `out=`/`+=` in `_intersect_surface`/`_transfer`/`_refract` (≈2× memory, ~15 % time); (b) hoist the renormalise out of `_refract` to once per trace (~7 N-sized ops × n_surfaces); (c) use the analytic sphere normal `(x, y, z−R)/R` for `is_pure_spherical` surfaces instead of the generic `sqrt(x²+y²)` + `np.where(h>0, …)` dispatch — that is the ~24 % block, and the v4.12.0 attempt noted at `intersection.py:76-81` failed only because it was applied *without* the matching intersection change.

---

### [P3] `_refract`'s "first-failure-wins" comment contradicts its code — `intersection.py:407-409`

```python
if newly_tir.any() and rays.error_code is not None:
    # First-failure-wins: RAY_TIR overwrites only RAY_OK entries.
    rays.error_code = np.where(newly_tir, RAY_TIR, rays.error_code)
```
The `np.where` is unconditional — exactly the defect the aperture block 50 lines below (`intersection.py:356`) documents having fixed ("the pre-fix line promised this in its comment but wrote `np.where(clipped, ...)` unconditionally"). Currently harmless because `alive ⇒ error_code == RAY_OK` holds by construction, but it is a live trap. Fix: `first_failure = newly_tir & (rays.error_code == RAY_OK)`.

### [P3] `rays_from_field` seeds `opd` with the **wrapped** phase — `from_field.py:392`

`opd_init = np.angle(E)[iy, ix] / k0` ∈ (−λ/2, λ/2]. Measured on a converging spherical wave with 0.8 waves of true OPL spread: returned `opd` range [−494.6, +498.3] nm at λ = 1 µm (i.e. the full wrap interval). Consumers that exponentiate (`bundles.ray_to_beamlet`: `exp(1j k0 opd)`; HFPI) are unaffected mod 2π; any consumer that treats `RayBundle.opd` as a geometric path (OPD fan, wavefront fit, `np.unwrap`, differencing across rays) sees a sawtooth. The docstring advertises `opd` as "[m]" with no wrap warning. Fix: document it, or unwrap the phase over the sampled support before seeding.

### [P3] `trace_summary`'s loss breakdown silently omits `RAY_EVANESCENT` — `layout.py:83-88`

The printed `[TIR=…, aperture=…, miss=…, nan=…]` never sums to the reported lost count when a `surface_diffraction` order goes evanescent (`RAY_EVANESCENT = 5`).

### [P3] Import-time global-registry mutation and an unlocked write — `trace.py:1564-1594`

`_register_fixed_index('__thin_lens__', 1.5, 550e-9)` runs at module import, mutating `glass.GLASS_REGISTRY` and `glass._glass_cache`. `_register_fixed_index` itself calls the lock-correct `_invalidate_glass_name` and then writes both globals **without** `_GLASS_CACHE_LOCK` — the read-modify-write the GL-2 comment three lines above says the lock exists to serialise. The registry also grows one entry per distinct `n_lens` value ever passed to `surfaces_from_elements` (content-derived names are idempotent, but unbounded in the number of distinct indices).

### [P3] `raytrace_system` mutates a caller-visible `Surface` in place — `trace.py:1683`

`surfaces[-1].thickness = image_distance` is the exact pattern `trace_prescription` was changed away from at `trace.py:1327` ("v4.13.2 audit P1-NEW-J: clone … instead of mutating it in place"). Currently safe only because `surfaces_from_elements` always returns fresh objects.

### [P3] `layout.py` contains no layout geometry

The module docstring and the v5.1 ROADMAP both promise "2-D layout figures"; the file only holds two `print()` summaries. There is no surface-arc / aperture drawing code in `raytrace/` at all (the plotting lives in `analysis/plotting.py`).

### [P3] `_adrt_jax` traces the whole system twice — `differential.py:1052-1053`

```python
jac = jax.vmap(jax.jacfwd(_state), ...)(s4)
st, opd = jax.vmap(_full, ...)(s4)
```
`jax.jacfwd(_full, has_aux=True)` returns both in one forward pass — a free ~2× on the JAX ADRT path.

### [P3] `field_of_view`'s finite-conjugate branch is not a field of view — `paraxial.py:76-91`

It returns `arctan((aperture/2)/object_distance)`, which is the object-space *aperture* half-angle, independent of the sensor. Self-described as "a rough proxy", but the name and return contract say otherwise.

---

## Performance opportunities

Measured on a box running ~20 parallel auditors, so absolute wall-clock is inflated; ratios and allocation counts are load-independent.

1. **In-place arithmetic in `_intersect_surface` / `_transfer` / `_refract`** — peak transient 257 MiB for 1e6 rays × 7 surfaces (one bundle = 65 MiB). Converting the position/OPL updates to `+=` and giving `np.where`/`np.sqrt` an `out=` buffer should roughly halve peak memory and buy ~10–15 % time (temporary allocation + first-touch page faults dominate at these sizes).
2. **Analytic sphere normal for `is_pure_spherical` surfaces** — the normal block (`_surface_normal` → `_base_surface_sag_derivatives_xy` → `_surface_sag_derivative`) is 24 % of `trace`'s tottime and computes `sqrt(x²+y²)`, a `np.where(h>0,…)`, a divide, and a second `sqrt` per ray, where `(x, y, z−R)/R` is 4 ops. Estimated 15–20 % end-to-end.
3. **Hoist the renormalise out of `_refract`/`_reflect`** — exact vector Snell with a unit normal returns a unit vector identically; the per-surface `sqrt(L²+M²+N²)` + 3 divides + 3 `np.where` (≈10 N-sized ops × n_surfaces) only removes ~1e-16 of drift. Do it once at the end of `trace`, keeping the degenerate-ray detection.
4. **`ray_fan_data` / `opd_fan_data` issue four separate `trace()` calls** (chief_y, chief_x, fan_y, fan_x) where one concatenated bundle would do — 4× the per-call glass resolution and Python surface-loop overhead.
5. **`output_filter='all'` costs one full `RayBundle.copy()` per surface** (510 MiB at N=1e6 vs 257 MiB for `'last'`); `trace_world` hard-codes `history = []` and appends unconditionally in the `'all'` branch, so it has no `'last'` memory benefit beyond the final copy.
6. **`_adrt_jax` double trace** (above), free ~2×.
7. **`trace_jax`'s eager cache saves only the XLA compile, not the Python prep.** Instrumented (`p13_jaxdomain.py`, 5-ray 2-surface prescription): first call **395 ms** (compile), then **750 µs/call warm** with `len(_TRACE_JAX_CACHE)` pinned at 1 — the cache is genuinely hit, there is no re-JIT. But `trace_jax` still runs `_build_jax_prescription` (glass lookups, `_resolve_semi_diameters`, `jnp.asarray` of every leaf, pytree flatten) on **every** call before the lookup, and that is what the 750 µs is. The docstring's "~20 µs for a cached jit'd kernel" is the kernel cost, not the call cost; a user sees ~40× that. (A second, earlier measurement on the same box gave 2377 µs/call for `trace_jax` vs 1381 µs/call for the NumPy `trace` on a 9-ray bundle — the box was running ~20 parallel jobs, so treat the absolutes as upper bounds, but the *structure* is load-independent.) Fix: cache the built `JaxPrescription` on `id(prescription)` + a cheap content hash, or accept a pre-built one.

---

## Alternative algorithms / methods

1. **Spencer & Murty (JOSA 52, 672 (1962)) generalised intersection** for the conic/aspheric path: solve the exact conic quadratic in the stable form `t = e/q`, `q = −½(b + sign(b)√disc)` (already implemented in `differential._adrt_step`), then iterate only on the polynomial departure with the Spencer–Murty normal-direction step. This fixes the `h > |R|` false miss (P1 above), removes the `disc` seed entirely for pure conics (no Newton at all), and typically cuts the aspheric iteration count from 10 fixed to 2–3.
2. **Welford/Hopkins aspheric Seidel terms** (Welford §8.5; Hopkins, *Wave Theory of Aberrations*, ch. 5) — the four-line fix above; no new machinery needed.
3. **Reference-sphere OPD** (Welford §4, or Hopkins' "optical path to the Gaussian image point") for `opd_fan_data`; ~4 lines, and it makes `opd_fan_data` consistent with `seidel_wfe`, which is already on the correct convention.
4. **Forward-mode AD for the whole differential path.** `ray_transfer_jacobian` (9 traced rays per base ray, central FD) carries ~4e-8 truncation and 9× the work; `ray_transfer_jacobian_analytic` already does it exactly with duals. The FD primitive should become a fallback only, and the analytic one should grow aspheric support (it currently raises `NotImplementedError` for any `aspheric_coeffs` — see `differential.py:975`) by adding the polynomial terms to `_adrt_step`'s implicit `F` and its gradient. Reference: Volatier, *JOSA A* 34, 1146 (2017); Stone & Forbes, *JOSA A* 14, 2824 (1997).
5. **Polarisation ray tracing (Chipman, *Polarized Light and Optical Systems*, ch. 9-11)** is absent from `raytrace/`: `Surface.coating` carries a complex index but the geometric trace never forms the 3×3 P-matrix. The s/p basis and local coordinate rotations are already computable from the oriented normal that `_conic_core.refract_snell` returns — adding a per-surface `P = O_out · J_fresnel · O_in` accumulation would give diattenuation/retardance for free on the existing trace. (`propagators.gbd._fresnel_jones_matrix_per_beamlet` does a per-beamlet version, so the physics exists but not in the ray path.)
6. **Area-uniform pupil sampling.** `make_rings` is equal-radius/equal-count (measured mean `r/R` = 0.5806 vs 0.6667 for area-uniform; mean `r²/R²` = 0.4194 vs 0.5), which the docstring documents but every `spot_rms` consumer silently inherits. A Vogel/sunflower disk (`r_i = R√(i/N)`, `θ_i = i·2π/φ²`) is a one-line generator that removes the bias.

---

## Code organization observations

* **`seidel.py` is 1742 lines carrying five unrelated responsibilities** (ABCD, per-lens characterisation, pupils, first-order report, Seidel sums). The module docstring argues they form "an unbreakable dependency chain", but `compute_pupils`/`_pre_stop_abcd`/`_post_stop_abcd` depend only on `system_abcd`, and `seidel_coefficients` depends on `_pre_stop_abcd` — a clean 3-file split (`abcd.py`, `pupils.py`, `seidel.py`) has no cycle.
* **Comment-to-code ratio is inverted in the hot path.** `system_abcd` is 100 lines of docstring + 100 of code; `_pre_stop_abcd` is 70 lines of docstring for 18 of code; `intersection._intersect_surface` has 90 lines of audit narrative interleaved with 120 of logic. The narrative is genuinely valuable (it is how I confirmed several conventions) but it is *inside* the functions rather than in a `docs/` convention note, and it actively hides control flow — the `flat_in_y` / `is_pure_spherical` dispatch is 4 lines of code inside 35 lines of comment.
* **`trace` and `trace_world` duplicate the DOE-kick block verbatim** (`trace.py:200-245` vs `world_trace.py:166-202`, ~35 lines each, identical including the R-13 comment). Two copies of a physics kernel that the P1 index fix above will have to be applied to twice.
* **`surfaces_from_elements`'s `'lens'` branch** (`trace.py:1391-1434`) carries 25 lines of dead reasoning ("This is too hacky", "Better: just store the focal length…") before the four lines that actually run.
* **`bundles.py`'s converters were dead on arrival** for ~2 versions (AttributeError on the first call, per its own docstring) and are still exported but have no internal caller.
* `raytrace/core.py` is a pure star-import shell with a 20-line comment explaining why it deliberately has no `__all__`; combined with `raytrace/__init__.py` re-exporting the same names, there are three import surfaces for every symbol.

---

## Unverified suspicions

* **JAX without `jax_enable_x64`.** `trace_jax_with_params` calls `jnp.float64(R)` unconditionally (`jax_trace.py:1617`); with x64 off JAX downcasts to float32 with a warning. `_newton_residual_tol` scales the convergence tolerance by the eps ratio, so the Newton path adapts, but the OPL accumulation itself would carry ~1e-7 relative error (metres-scale OPLs → ~0.1 µm), i.e. the traced phase would be meaningless at 1.31 µm. I verified parity only **with** x64 (2.8e-17 m); the float32 path is desk-checked only.
* **`trace_world` does not guard `surf.is_coordbrk`.** A coord-break Surface passed to `trace_world` would be intersected and refracted as a flat air→air surface rather than transformed. `world_surfaces_from_prescription` never emits one, so this is only reachable via hand-built lists — not tested.
* **`_intersect_surface`'s `min(|t1|, |t2|)` root pick.** Correct for every case I tested (including concave surfaces, post-mirror backward rays, and 480k random rays), but it is a heuristic, not a sheet test: for a ray that starts *beyond* the whole sphere it selects the far side. I could not construct a reachable sequential prescription that triggers it.
* **`trace_jax` eager-cache speed** (see Performance #7) — the measurement is load-contaminated.

---

## Checked and found correct (brief, so coverage is known)

* **OPL bookkeeping is exact.** `p2b_opl.py` compares `trace()`'s per-ray `opd` against a 60-digit `decimal` oracle that uses the implicit conic form, its exact gradient and exact vector Snell, sharing no code with the library. Worst error over 5 systems × 3–4 heights: **1.39e-17 m (1.1e-11 waves at 1.31 µm)**, positions to 8e-18 m. Systems: biconvex singlet on-axis; the same at 5° field; a concave-exit meniscus (R1=+25, R2=+40 — exercises the backward-`t` branch); a deep biconvex with overlapping sags (edge thickness negative); a k=−1/k=−0.5 conic singlet at 3° field. The `_transfer` + `_intersect_surface` signed-leg convention (REAL_LENS_CHANGES §3/§4) telescopes exactly.
* **Spherical fast path.** vs a numerically stable Spencer–Murty-form root (`t = F/(G + sign(G)√(G²−F))`): max `|Δt| = 3.3e-16 m` over 480k random rays across R ∈ {±0.002 … 1.0 m} and h up to 0.95|R| (relative to the edge sag: ≤4.6e-15). vs the Newton path on the same bundles: 3.0e-17 m, identical alive masks. The near-vertex cancellation in `(−b − √disc)/2` is real but sub-femtometre — not worth fixing on its own.
* **Coordinate breaks.** `world._apply_coord_break` puts the new local +z at world (0,−1,0) for `tilt_x = +90°` as CONVENTIONS §7 says; `intersection._apply_coord_break` leaves a +z-going ray at local (0,+1,0); the two are exact inverses over **200 random (tx,ty,tz,dx,dy, PARM6∈{0,1})** cases — position round-trip 6.9e-18 m, direction 3.3e-16. PARM6 order 0 puts the decenter in the old frame, order 1 in the new one, as documented. `differential._adrt_coordbreak` replicates the same transpose op-for-op. `ui/model.py` has no tilt-matrix copy left.
* **`system_abcd`.** EFL/BFL/FFL match the thick-lens closed forms `1/f = (n−1)(c1−c2+(n−1)t c1 c2/n)`, `BFL = f(1−(n−1)t c1/n)`, `FFL = f(1+(n−1)t c2/n)` to **0.00e+00** on five singlets (biconvex, both plano-convex orientations, meniscus, bi-concave); the two-thin-lens combo gives EFL 66.666667 / BFL 33.333333 mm exactly; a single concave R=−100 mm mirror gives +50 mm.
* **Seidel for spherical surfaces.** Real-ray reference-sphere ρ⁴ fit / (S1/8) = **−0.9982** and **−0.9992** for the two plano-convex orientations; the classic ~4× split reproduced (26.27 µm / 6.64 µm = **3.96**). `seidel_wfe`'s `−S_Welford` ingestion sign is right (the sign of the real-ray a4 matches `−S1/8`, not `+S1/8`), and its `(1/4)S4 H²ρ²` Petzval scaling is dimensionally and conventionally consistent with `S4_code = −c(n′−n)/(n n′)`.
* **Mirrors / negative post-mirror thickness.** A k=−1 mirror (R=−200 mm) with `thickness = −100 mm` focuses a collimated bundle to `y = 0` exactly at heights 0/5/15/25 mm, with `N` flipped negative and the OPL constant across the pupil to **1.39e-17 m**. The Welford `n′ = −n` parity bookkeeping in `system_abcd` and `seidel_coefficients` is consistent with that trace.
* **JAX parity (x64 on).** `trace_jax` vs `trace` on a 9-ray fan through a biconvex singlet: `max|Δy| = 6.9e-18 m`, `max|Δopd| = 2.8e-17 m`, `max|ΔL| = 0`. `jax.grad(mean exit OPL)/∂R1` vs central FD: **1.4e-9 relative**. `_transfer_jax`'s frame-shift form (`x += L·thickness`, `z += N·thickness − thickness`) is genuinely equivalent to the NumPy `t = (thickness−z)/N` form — both land on the same ray line and the signed OPL telescopes identically, as its docstring claims.
* **Differential Jacobians.** `ray_transfer_jacobian_analytic` vs an independent central-difference Jacobian built by calling `trace()` directly: max rel error **1.7e-9 … 1.0e-7** across 3 ray states × {with, without} a `tilt_x=3°, tilt_y=−2°, decenter=(0.3,−0.2) mm` coordinate break (the CB branch is correct, i.e. the pre-v5.29 inversion is genuinely fixed here too). The FD primitive `ray_transfer_jacobian` lands at 2e-8 … 6e-7 (its truncation floor). Both agree with `trace()`'s own OPL to ≤2.8e-17 m.
* **`rays_from_field` sign and focusing.** `E = exp(+i k(Lx+My))` recovers `(L, M)` to 5e-16; a converging `exp(−ikR)` wave produces rays that cross `z = +f` with **rms radius 0.001 nm**, while `exp(+ikR)` diverges (104 µm at `+f`) — the `exp(−iωt)`/`exp(+ikz)` convention of CONVENTIONS §7 is honoured.
* **`refocus`** correctly handles the `z = sag → plane` correction with a signed OPL and the image-space index (read; and `through_focus_rms` builds on it correctly, including the all-non-finite guard).
* **`_conic_core.refract_snell` / `reflect_mirror`**: normal orientation (`fl = where(d·n > 0, −1, +1)`), `cos_i = −d·n`, `d′ = η d + (η cos_i − √disc) n`, TIR mask `disc < 0` — all standard and correct, including the grazing case `d·n = 0` (which correctly TIRs for `η > 1`).
* **`_intersect_surface` aperture gate** uses the NaN-safe polarity (`inside = h_sq <= sd²`) and splits `RAY_NAN` from `RAY_APERTURE`; `validate_prescription` / `Surface.__post_init__` / `check_even_aspheric_powers` consistently reject odd aspheric powers at all three entry points.
* **`make_fan`'s per-axis field convention** (y-fan tilts in `M`, x-fan in `L`) is deliberate and is what makes `ey(0) = ex(0) = 0` in `ray_fan_data`; `make_rings`' area bias is documented and measured as documented (mean `r/R` = 0.5806).
* **The conic out-of-domain sag divergence is benign at the trace level.** `_conic_core.conic_sag` returns `+0.0` outside the conic domain while `elements.lenses.surface_sag_general` (what the NumPy trace calls) returns `nan` — measured at R = 20 mm, k = +2, h = 11.6 mm (valid limit 11.547 mm). `conic_sag`'s docstring declares this "intentionally distinct"; I checked whether the JAX Newton can converge onto the resulting phantom flat surface and survive the post-loop residual check. It cannot: on a k = +2, R = 20 mm surface with rays at h = 12/15/18 mm (all inside `|R|`, all outside the conic domain) **both** backends kill the rays — NumPy `alive=[T T F F F], err=[0 0 3 3 3]`, `trace_jax` `alive=[T T F F F]`. The mitigation is adequate.

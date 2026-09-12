# VERIFY-WP-A1 — adversarial re-verification of the ray-tracing work package

Verifier: VERIFY-A1 (independent; did not write the fixes).
Subject: commit `f602b72c` on `audit-fixes-2026-09` (diff base `f602b72c~1` = `0067d63b`).
Every number below is **my own measurement on the current working tree**, not a quotation from
`WP-A1_REPORT.md`. All python run with `OPENBLAS_NUM_THREADS=1`. No git write commands were run.

---

## 1. Verdict summary

| item | verdict | my independent oracle | my measurement |
|---|---|---|---|
| §15.1 exit-vertex helper | **VERIFIED** | 60-digit `decimal` trace built from the CLOSED FORM of the conic (exact quadratic, exact gradient normal, exact vector Snell) on curved-rear **immersed** fixtures the WP never used | `max abs(d opd) = 4.3e-18 m`, `max abs(d y) = 1.7e-18 m` over concave/convex/oblate rears with `n_exit` = n(N-SF11), n(N-BK7), 1.0 |
| §15.1 on a MIRROR | **VERIFIED** | Decimal exact reflection off a parabola + OPL-to-focus invariance | `max abs(d opd) = 4.3e-19 m`; OPL(launch→focus) constant to `1.4e-17 m` over h = 0…25 mm; immersed mirror uses `n(glass_before)` exactly |
| R1 reference sphere | **VERIFIED** | (a) my own Decimal reference-sphere solve on a **fast meniscus (f/2.4, virtual image)** and a **cemented doublet**; (b) the slope identity `eps_y = +(R/n) dW/dy_p`, derived here from scratch; (c) Seidel `-S1/8` | on-axis `max abs(W_lib - oracle) = 6.5e-11 / 4.7e-11 waves` on 45-wave fans; slope identity 0.23–1.66 %; `-S1/8` agrees to 4 digits at f/20 and f/50 |
| R2 entrance eikonal (SIGN) | **VERIFIED — the WP is right, the audit text is wrong** | first principles + a 3-way side-by-side measurement | pre-fix linear `-1894.63 w`; `+(Lx+My)` → `-40.57 w` (all R1 residual); `-(Lx+My)` (audit text) → **`-3748.69 w`, i.e. 2x the artefact** |
| R3 aspheric Seidel, **S3 exponent** | **VERIFIED — square is right, the audit snippet is a typo** | re-derivation of `dn*A4*abs(y_m rho + y_c H)^4` against the library's OWN `seidel_wfe` basis, then **real rays with a DISPLACED STOP** (tangential-minus-sagittal rho² fit = `-(1/2)S3`) | `dS3_lib/dS3_rays = 0.981…0.989`; the **first power predicts 3.75…3.78**. `dS2` power-1 → 0.988–0.990. `dS4 = 0` → residual 0.2–0.6 % of `dS3` |
| R4 exact conic quadratic | **VERIFIED** (one doc defect) | `brentq` on the exact implicit sag, plus NumPy↔JAX | hyperbola to h = 4·abs(R): `1.4e-17 m`; parabola to h = 6·abs(R): **0.0**; conic+A4 = −3e4: `1.7e-18 m`; genuine misses still die; JAX alive masks identical everywhere |
| R5 DOE kick / n2 | **VERIFIED** | the grating equation `n2 L' = n1 L + m lam/Lambda` | `L` in N-BK7 = `0.17425045`, ratio to truth `1.000000000000` (pre-fix ratio was exactly `n = 1.503583`); JAX twin agrees to `1e-15` |
| R6 estimator / edges / `_transfer` / in-place | **VERIFIED-WITH-NOTES** | plane waves at odd N, even N and **anamorphic dy != dx**; closed-form transfer; an in-process allocating-form arm | exact to the FULL grid Nyquist (worst `8.4e-15` at 0.95 Nyquist); edge rays `2.8e-16`; in-place **bit-identical** (`max abs(d) = 0.0` on x, y, z, opd at N = 1e3/5e4/3e5), 1.10x faster. **Memory claim does not reproduce** — see §5.1 |
| R7 (8 items) | **VERIFIED** | each item's own | see §4.7 |
| audit's "checked and found correct" list | **UNCHANGED** | the audit's own repro scripts re-run on HEAD | §6 |

Nothing is **NOT FIXED** and nothing is a **REGRESSION**. Ten open items follow in §7,
all P2/P3 and none blocking; two of them are factual errors in the WP's report/changelog text.

---

## 2. The exit-vertex helper (§15.1 / F-O3)

### 2.1 Contract confirmed as reported

I re-read the shipped signatures rather than the report:

```
TraceResult.at_exit_vertex(self, n_exit: Optional[float] = None) -> RayBundle
exit_vertex_transfer(bundle, n_exit, *, fn_name: str = 'exit_vertex_transfer')
exit_vertex_transfer_jax(state, n_exit)
vertex_plane_transfer_t(z, N, alive, *, z_target: float = 0.0, tol: float = 1e-30)
resolve_exit_index(surfaces, wavelength, *, fn_name='at_exit_vertex', n_exit=None) -> float
EXIT_VERTEX_GRAZING_TOL = 1e-30
```

`exit_vertex_transfer`, `EXIT_VERTEX_GRAZING_TOL` and `exit_vertex_transfer_jax` are all in
`lumenairy.raytrace.__all__`; `exit_vertex_transfer_jax` is in `jax_trace.__all__`;
`from lumenairy.raytrace.core import exit_vertex_transfer` resolves to the same object.
The report's §0 is accurate. (It is **not** in the top-level `lumenairy` namespace — §7.9.)

### 2.2 Independent 60-digit oracle — curved rear, **glass_after != air**

The WP's own §15.1 test compares `at_exit_vertex()` against `img.opd + n*(-img.z/img.N)`,
i.e. the helper's own formula re-evaluated on the helper's own inputs, and every one of its
fixtures exits into **air**. I wrote a `decimal` trace at 60 significant digits that never
reads a traced quantity — exact conic quadratic, exact gradient normal, exact vector Snell —
and gave the last surface a real exit medium:

| fixture (R1 = 30 mm, k = −0.7, t = 6 mm) | n_exit inferred | `max abs(d opd)` | `max abs(d y)` |
|---|---|---|---|
| rear R2 = **+40 mm** (concave, sag > 0 ⇒ **t < 0**, OPL SUBTRACTED), exit into **N-SF11** | 1.747969361427 | **1.7e-18 m** | 1.1e-19 m |
| rear R2 = −35 mm, k = −2.5 (convex, sag < 0 ⇒ t > 0), exit into air | 1.0 | **1.7e-18 m** | 1.7e-18 m |
| rear R2 = +60 mm, k = **+2** (oblate ellipsoid), exit into **N-BK7** | 1.503582905410 | **4.3e-18 m** | 1.7e-18 m |

Sag at ρ = 1 runs 1.36 mm (concave) to −1.62 mm (convex), and the transferred OPL term is
1.37 mm / 1.68 mm — 15 decades above the residual. **The sign is right in both directions**
and `resolve_exit_index` genuinely picks up the glass (using 1.0 instead of n(N-SF11) would
move the OPL by ~1.0e-3 m).

### 2.3 Mirrors

Concave parabolic mirror R = −200 mm, k = −1, `glass_after='air'`, `is_mirror=True`:

```
h [mm]   z_sag [um]   N (lib)        N (Decimal)     opd_vertex (lib)   oracle           d
 0.00       0.0000    -1.000000000   -1.000000000    +0.000000000e+00   -0.000000000e+00  0.0
 5.00     -62.5000    -0.998750781   -0.998750781    -1.250781739e-04   -1.250781739e-04  0.0
15.00    -562.5000    -0.988812927   -0.988812927    -1.131363922e-03   -1.131363922e-03  0.0
25.00   -1562.5000    -0.969230769   -0.969230769    -3.174603175e-03   -3.174603175e-03  4.3e-19
```

`t = -z/N` is correctly **negative** here (a backward-going ray reaching the vertex plane by
back-extrapolation), and the total OPL from the launch plane to the parabola's focus is
constant across the pupil to **1.388e-17 m** — the independent stigmatism check.
An *immersed* mirror (`glass_before='N-BK7'`, `glass_after='air'`, `is_mirror=True`) resolves
`n_exit = n(N-BK7) = 1.503583` exactly (`max abs(d) = 0.0e+00` vs the hand-computed transfer),
confirming the `glass_before`-when-mirror branch.

Note: `glass_after='MIRROR'` as a literal marker is unreachable through `trace()` — the trace
loop resolves every `glass_after` through `get_glass_index` first and raises
`ValueError: Glass 'MIRROR' not in registry`. The helper's `'MIRROR'`/`'__MIRROR__'` string
branch is therefore dead defensive code; the `is_mirror` branch is the live one. Harmless.

### 2.4 Rays that are dead before the transfer, grazing rays, and everything else

41 assertions, all pass (`scratchpad/verify_a1/v_ev_edge.py`):

* **dead rays frozen on all seven fields** (`x, y, z, opd, L, M, N`) and `error_code` preserved,
  while alive rays land on `z == 0.0` exactly.
* **grazing** (`N = 0`): `alive → False`, `error_code = 3`, `z`, `opd`, `x`, `y` all unchanged —
  no teleport. `abs(N) == EXIT_VERTEX_GRAZING_TOL` counts as grazing (the `<=` boundary);
  `abs(N) = 2e-30` survives and is transferred to `x = -5.0e+25` (the documented tolerance
  boundary, inherited from `_transfer`, not new).
* **first-failure-wins on the kill**: a ray already carrying `RAY_APERTURE` (1) that also grazes
  keeps code 1.
* **idempotent** bit-for-bit on all nine fields; the source bundle is untouched.
* `n_exit` validation rejects `0.0`, `-1.5`, `nan`, `inf` with the §2 `fn_name:` prefix;
  a per-ray `n_exit` array is honoured exactly; an unresolvable glass name raises
  `ValueError` naming `TraceResult.at_exit_vertex`.
* **non-contiguous** (stride-2 view) bundles give bit-identical results.
* `error_code=None` bundles kill via `alive` alone.
* Empty bundle, `alive` supplied as a Python list, and `N = nan` all behave (the NaN ray is
  killed and does not poison its neighbour — see §7.5 for the error-code nit).
* `at_exit_vertex()` on a prescription whose last surface is a **coord break** runs and returns
  finite OPL.
* `at_exit_vertex() == refocus(res, 0.0).image_rays` **bit-identically** on x, y and opd for an
  alive fan (they now share `vertex_plane_transfer_t`), and `refocus(res, 0.0)` lands on `z = 0`.

### 2.5 JAX twin

With `jax_enable_x64`, against the NumPy helper on a curved-rear immersed fixture with one
vignetted ray: `max abs(d x) = 0.0`, `abs(d y) = 8.7e-19 m`, `abs(d z) = 0.0`,
`abs(d opd) = 2.6e-18 m`, `alive` identical (including the dead ray, which the JAX twin also
freezes). Grazing rays are killed rather than given `t = -z/1e-30`
(`alive = [False False]`, `z` preserved at 1e-4, `x` unchanged) — the cross-backend divergence
the audit found is closed. `jax.grad` of the exit OPL w.r.t. `N` is finite at `N = 0`
(double-`where` holds). The JAX transfer is idempotent.

---

## 3. R1 — the reference sphere, tested against fixtures the WP never used

### 3.1 Is exit-pupil-tangent the right choice?

**Against the documented Zemax convention:** OpticStudio's OPD is referenced to the chief ray,
with the reference sphere **centred on the chief ray's intercept with the image surface** and a
**radius equal to the exit-pupil distance**. That is exactly what
`_reference_sphere_radius` + `_opd_fan_wfe` implement. One deliberate refinement:
the library uses `R = abs(xp_z) / abs(N_chief)` (the chief's *slant* path from the XP plane to the
image) rather than the axial `abs(xp_z)`, so the sphere passes through the chief's actual
crossing of the XP plane. I measured the size of that refinement (§3.2): 0.065 % of the WFE at
3° on a wildly aberrated meniscus, and zero on axis. Defensible and, if anything, more correct.

(Nit: the parameter is named `n_chief` although the value passed is `ref_y.N[0]`, a direction
cosine, not an index — §7.3.)

### 3.2 My own exact solve, on a fast meniscus and a doublet

Decimal (60-digit) exact trace + a reference sphere built from **my own** paraxial y-nu
computation of the exit-pupil position (not `first_order_data`):

| system | field | `max abs(W_lib - my exact sphere)` | fan PV |
|---|---|---|---|
| **fast meniscus** R1 = 20 mm / R2 = 15.5 mm, t = 8 mm N-BK7, f/2.4, **virtual image** (BFL = −292.096 mm), XP behind the lens | 0° | **6.5e-11 waves** | 44.84 w |
| same | 3° | 8.8e-02 waves | 135.69 w |
| **cemented doublet** 33.3 / −22.28 / −291.07 mm, N-BK7 + N-SF5, 11 mm semi-aperture | 0° | **4.7e-11 waves** | 45.33 w |
| same | 2° | 8.1e-03 waves | 52.79 w |

My independently computed XP-to-image distances (`287.603898 mm` meniscus, `86.679643 mm`
doublet) match `first_order_data.xp_z` in magnitude to 6 digits (the library reports a
magnitude; it disagrees in *sign* on the meniscus, but only the magnitude is used).
The off-axis residual is entirely the `/N_chief` refinement: at 3° with
`abs(v) = 4.1 mm` and `R = 287.6 mm` the second-order term is 46 waves, and 0.14 % of it is
0.065 waves — which is what I measure.

### 3.3 The SIGN, from first principles

I derived the transverse-aberration/wavefront-slope identity from scratch (a pure defocus `Dz`
puts the wavefront at radius `R + Dz y_p^2/(2R^2)` from the image point, giving
`W = n Dz y_p^2/(2R^2)`, while the ray from pupil height `y_p` sits at `eps = y_p Dz / R` in the
plane of `C`), which yields

> **`eps_y = +(R/n) dW/dy_p`** with the library's sign convention.

Measured against `ray_fan_data` (audited exact) at three f-numbers on the audit's plano-convex:

| | W(ρ=1) | eps(ρ=1) measured | eps(ρ=1) predicted from dW | max rel dev (interior) |
|---|---|---|---|---|
| f/4 | −11.72043 w | −222.76 µm | −201.36 µm | **1.656 %** |
| f/20 | −0.01813 w | −1.71 µm | −1.58 µm | **0.234 %** |
| f/50 | −0.00046 w | −0.11 µm | −0.10 µm | **0.296 %** |

The sign is right (the endpoint gap is the one-sided gradient plus 5th order). Seidel
cross-check `-S1/8`: −11.3186 vs −11.7204 w at f/4 (the 0.40 w gap is the genuine ρ⁶ content),
and **agreement to 4 significant digits at f/20 and f/50**.

### 3.4 Controls and guards

* Aberration-free parabolic mirror: PV = **7.09e-11 waves** (audit's pre-fix control 7.1e-11) —
  unchanged, as it must be.
* ρ⁴ scaling: halving the aperture twice gives W(1) ratios **16.042** and **16.011** — the fan
  is a genuine 4th-order wavefront, not a leftover ε·sinθ′ term.
* Prescription with **no image plane**: two `RuntimeWarning`s naming the geometry, fallback to
  the reference plane, all values finite. The message correctly states the result "is NOT a
  wavefront aberration".
* `reference_sphere_radius`: `None` → −11.72043, `inf` (plane) → −12.13579 (the 0.42 w the
  report quotes), `0.0965` → −11.70536, and `0.0` / `-1.0` / `nan` are rejected with the §2 prefix.
* **Afocal** Galilean expander (XP at infinity): `R` falls back to `inf`, output finite, no crash.
* `opd_fan_data_world` carries the identical fixes — world vs sequential fans agree to
  **0.00e+00 waves** at 0° and 2°.
* A **vignetted chief ray** (20° field through a 0.3 mm stop) makes the entire fan `NaN` rather
  than silent garbage. Acceptable; unchanged from pre-fix.

---

## 4. R2–R7

### 4.1 R2 — the eikonal sign, decided independently

**From first principles.** For a collimated bundle of direction `d`, the incident wavefront
through the origin is `{P : P·d = 0}`. A ray launched at `P0` on `z = 0` has already travelled
`P0·d = L x0 + M y0 (+ N z0)` of optical path from that wavefront. The seed is therefore
**`+(L x + M y + N z)`**. The audit's §4 row and `RAYTRACE.md` both prescribe `-(x L + y M)`,
which **adds** the artefact instead of removing it.

**By measurement** (f/4 plano-convex, 5° field, fitted linear term of the raw chief-referenced
OPD, no reference sphere so the three seeds are directly comparable):

| seed | fitted linear term | |
|---|---|---|
| none (pre-fix) | **−1894.63 w** | `y_max sinθ/λ = +1854.06 w` |
| `+(Lx+My)` (shipped) | **−40.57 w** | the remainder is R1's ε·sinθ′, which the reference sphere then removes → +0.15 w through the full `opd_fan_data` |
| `-(Lx+My)` (audit text) | **−3748.69 w** | ≈ **2×** the pre-fix artefact |

**Decisive independent discriminator**: a *paraxial* aperture (semi-aperture 0.2 mm) at a *large*
field (2° and 5°), where the genuine aberration is ~1e-3 waves. Fitted linear term −0.0000 w
against an artefact of 11.88 w / 29.67 w — five decades of separation between the three
hypotheses. On axis (`L = M = 0`) the seed is exactly 0, so `_make_bundle`'s default is
bit-identical. `_make_bundle(opd_seed='eikonal')` equals `seed_entrance_eikonal(plane bundle)`
to the last bit; the default is still `'plane'`; a bad `opd_seed` is rejected with the §2 prefix.
`seed_entrance_eikonal` correctly carries the `N·z` term (verified on a bundle at `z = −5 mm`,
`d = 0.0e+00`) and is documented as non-idempotent (measured: applying twice doubles it exactly).

Re-run of the audit's own `repro/RAYTRACE/p8_opdfan.py` on HEAD: fitted linear term
+0.01 / +0.06 / +0.15 waves at 0.5° / 2° / 5°, ratio to the artefact +0.0001 — matching the WP's
table exactly.

### 4.2 R3 — the S3 exponent, decided independently

**Derivation.** An aspheric departure `z = K h^4` acts as a thin plate of
`OPD = dn K h^4` evaluated at the paraxial ray height `h = y_m ρ cosθ (+ y_c H in y)`, so

```
W_asph = dn K (y_m^2 rho^2 + 2 y_m y_c rho H cos + y_c^2 H^2)^2
       = dn K [ y_m^4 rho^4 + 4 y_m^3 y_c H rho^3 cos
                + 2 y_m^2 y_c^2 H^2 rho^2 + 4 y_m^2 y_c^2 H^2 rho^2 cos^2
                + 4 y_m y_c^3 H^3 rho cos + ... ]
```

The library's own wavefront basis (`seidel_analysis.seidel_wfe`) is
`W = -[S1 r^4/8 + S2 r^3 cos/2 + S3 r^2 cos^2/2 + S3 r^2/4 + S4 H^2 r^2/4 + S5 r cos/2]`,
i.e. the field-curvature DC term is `(1/4)(S3 + S4 H^2)`. Matching the `cos 2θ` half gives
`dS3 = dS1 (y_c/y_m)^2`; the DC half is then satisfied **with `dS4 = 0`**. The first power
satisfies neither half. So **the WP's square is right, `dS4 = 0` is right, and the audit's
snippet `S3[i] += dS * (y_val_c / y_val_m)` is a typo** — and the two choices are coupled: a
first-power `S3` would also force a non-zero `S4`.

**By measurement, with the stop DISPLACED** (a flat stop 30 mm ahead of the lens, so
`y_c/y_m = +0.2618` at the aspheric surface; every R3 fixture in the WP's own test set has the
stop AT surface 0, where `y_c ≈ 0` and the exponents are indistinguishable). Oracle: real rays.
The tangential-minus-sagittal ρ² coefficient of the wavefront is `-(1/2) S3` exactly (defocus,
Petzval and the S4 term are rotationally symmetric and cancel), so `S3_rays = -2(a2_t - a2_s)`;
differencing the aspherised system against its base sphere isolates `dS3`:

| perturbation | `dS3` library | `dS3` real rays | lib/rays | **first-power hypothesis would give** |
|---|---|---|---|---|
| A4 = −4000 m⁻³ | −1.468972e-06 | −1.484584e-06 | **0.98948** | **3.77955** |
| A4 = +4000 m⁻³ | +1.468972e-06 | +1.491928e-06 | **0.98461** | **3.76095** |
| conic k = −3 | −6.375745e-07 | −6.445877e-07 | **0.98912** | **3.77816** |
| conic k = +3 | +6.375745e-07 | +6.501259e-07 | **0.98069** | **3.74597** |

The residual 1–2 % is genuine 5th order. The same rig confirms **`dS2` scales as the first
power** (lib/rays 0.98851–0.98955) and **`dS4 = 0`** (`-4 d a2_sag - dS3` is 0.20–0.60 % of
`dS3`, i.e. at the 5th-order floor).

`dS5` (third power) I could **not** measure: distortion is referenced away by a chief-referenced
wavefront, so `-2 d a1` reads ~1e-10 regardless. The derivation above fixes it
(`4 y_m y_c^3 → dS5 = dS1 (y_c/y_m)^3`); I record it as derived, not measured, and note that
`S5` is (correctly) added directly rather than through the Schwarzschild
`(A_c/A_m)(S3 + H^2 S4)` recurrence, because the aspheric scaling is a *height* ratio and
`A_c/A_m != y_c/y_m` in general.

**S1 numbers reproduce exactly** (audit's `p5_seidel_asph.py` re-run on HEAD):
parabolic mirror k = −1 → `S1 = 1.355253e-20`; k = 0 / −0.5 / −1.5 →
`+9.765625e-05 / +4.882813e-05 / -4.882812e-05`; A4 sweep
`+5.320645e-05 / +2.797215e-05 / +2.737848e-06 / -4.773075e-05 / -1.486679e-04 / +1.036750e-04`
for A4 = 0 / −250 / −500 / −1000 / −2000 / +500 — every value matches the report's "after"
column. Orchestrator `verify_rt_glass.py` likewise: `S1 = 1.3552527e-20` for k = −1.

### 4.3 R4 — exact conic quadratic

Oracle: `scipy.optimize.brentq` on the exact implicit residual with the closed-form sag written
locally, bracketed by a 20 001-point scan so the near root is the one found.

| geometry | rays | alive / err | `max abs(dz)` |
|---|---|---|---|
| prolate k = −0.6, R = 10.84 mm (domain 17.1395 mm) | h = 10.83, 10.90, 11.4, 15, 17.0 mm | all alive, err 0 | **1.4e-17 m** |
| same, h = 17.2 mm (**outside** the conic domain) | | **dead, err 3** | genuine miss preserved |
| **parabola** k = −1, R = 50 mm | h = 60, 100, **300 mm** (6·abs(R)) | all alive | **0.0e+00** |
| **hyperbola** k = −2, R = 50 mm | h = 60, 100, 120, **200 mm** (4·abs(R)) | all alive | **1.4e-17 m** |
| **oblate ellipsoid** k = +2, R = 20 mm (domain 11.547 mm) | h = 5, 11, 11.5 mm | alive | **3.5e-18 m** |
| same, h = 11.6, 15, 19 mm | | **dead, err 3** | correct |
| conic + **large polynomial** k = −0.6 + A4 = −3e4 (departure ≈ sag/3 at h = 15 mm) | h = 5…15 mm | alive | **1.7e-18 m** |
| oblate k = +2 + A4 = +5e4 crossing the domain | h = 11.6, 13 mm | dead, err 3 | (sag is NaN there in both the pre- and post-fix code) |
| **flat base + pure A4** (Schmidt plate) | h = 5, 15, 30 mm | alive | **0.0e+00** |
| **backward** rays (N = −1) starting at z = +40 mm on a hyperbola | h = 10, 60, 100 mm | alive | **0.0e+00** |
| sphere-as-conic, N = −1, z₀ = +30 mm, with an asphere | h = 10, 30, 45 mm | alive | **0.0e+00** |
| **degenerate a = 0**: axial ray on a parabola (L = M = 0, k = −1) | h = 0, 1e-9, 1e-3 m | alive, exact sag | Spencer–Murty `t = e/q` reduces correctly to the linear root `-e/b` |

**NumPy ↔ JAX parity on all of the above**: alive masks identical in every case (including the
genuine oblate misses), `max abs(dz) <= 7.7e-18 m`, `max abs(d opd) <= 2.8e-17 m`, with and
without a polynomial asphere. The audit's `p1b_intersect.py` and `p12_conicdomain.py` both
reproduce (`41/41` alive through the public API; `dz = +0.000e+00` at h = 10.90 / 11.40 mm).

**Doc defect found — §7.2.** The shipped comment in `intersection.py` and the report both assert
"the conic domain always contains the sphere domain". That is **false for k > 0**: the oblate
domain is `abs(R)/sqrt(1+k) < abs(R)`. I measured it (row 5/6 above) — the new miss test is
*stricter* there, not looser — but it produces no behaviour change, because
`elements.lenses.surface_sag_general` returns NaN outside the conic domain, so the pre-fix
Newton path killed those rays too. The conclusion ("no regression") stands; the justification
does not.

### 4.4 R5 — the index in the diffraction-order kick

Λ = 5 µm, λ = 1.31 µm, m = 1, grating on a surface whose `glass_after` is N-BK7:

* NumPy `L` inside the glass = `0.17425045`, `m λ/(n2 Λ) = 0.17425045`,
  **ratio = 1.000000000000** (pre-fix ratio was exactly `n(N-BK7) = 1.503583`).
* The **split is load-bearing and correct**: `n2 L' - n1 L` must equal `m λ/Λ`, so the OPL
  screen gradient `_gL = m λ/Λ` must stay index-independent while only the direction kick takes
  `1/n2`. Dividing the OPL term as well would have introduced a new error of the same size.
  Verified by construction and by the exit direction: after refracting back out to air the ray
  carries `L = 0.26200000 = n2 * 0.17425045`, exactly Snell.
* **JAX twin matches NumPy to 1e-15** on the same fixture.
* Air (`n2 = 1`) is bit-identical.
* `apply_doe_phase_traced(n_medium=1.0)` default preserves pre-fix behaviour.

### 4.5 R6 — estimator, edges, `_transfer`, in-place

**Estimator** — plane waves, `rays_from_field`, worst `abs(dL)`/`abs(dM)` over pupil fractions
0.30 / 0.70 / **0.95** of the grid Nyquist and a (0.50, 0.90) mixed case:

| grid | dx, dy | worst dev |
|---|---|---|
| **odd 65×65** isotropic | 2, 2 µm | 2.6e-15 |
| even 64×64 isotropic | 2, 2 µm | 2.6e-15 |
| **odd 51×37, ANAMORPHIC** | 2, 5 µm | 1.5e-15 |
| even 40×96, **ANAMORPHIC** | 4, 1 µm | 8.4e-15 |

Exact to the **full** grid Nyquist, at odd N and with `dy != dx`. Audit repro `p7_fromfield.py`:
`L_true` 0.150 → **+0.150000**, 0.200 → **+0.200000**, 0.240 → **+0.240000** (pre-fix −0.100,
−0.050); 0.260 and above alias because the *grid* cannot represent them.

**Edge rays**: 60 edge-column rays at `L_true = 0.05`, max deviation **2.8e-16** (pre-fix ratio
0.5000). **Evanescent detection now works**: a grid with Nyquist `L = 1.25` carrying
`L_true = 1.10` returns 100 % `alive=False`, `error_code = 5`, median `abs(L) = 1.1000`.

**float32 / complex64**: a complex64 plane wave gives finite directions to `7.5e-10`
(float32 eps); a complex64 field with a **hard aperture (exact zeros)** also gives finite
directions, with one `RuntimeWarning` from the pre-existing `np.maximum(abs_v, 1e-300)` clamp
(1e-300 underflows to 0 in float32). Pre-existing, not introduced here, and it degrades to a
warning rather than NaN.

**`_transfer` grazing**: `z=[1e-4 1e-4], alive=[F F], opd=[0 0], error_code=[3 3]` — killed and
frozen (pre-fix `z=[0 0], alive=[T T], err=[0 0]`). Ordinary rays are **bit-identical** to the
closed form `(thickness - z)/N` on x, y, opd and z.

**In-place kernel**: I monkeypatched the allocating form back in as an in-process pre-fix arm:

| N | `max abs(dx)` | `max abs(dy)` | `max abs(dz)` | `max abs(d opd)` | alive |
|---|---|---|---|---|---|
| 1 000 | 0.0 | 0.0 | 0.0 | 0.0 | identical |
| 50 000 | 0.0 | 0.0 | 0.0 | 0.0 | identical |
| 300 000 | 0.0 | 0.0 | 0.0 | 0.0 | identical |

**Bit-identical, confirmed.** Timing on a 7-surface trace, medians of 7 interleaved runs at
N = 2e5: 2951.2 vs 3259.6 ns/ray, **1.105×**. Memory: see §5.1 — the report's number does not
reproduce. Safety checks: `trace()` does not mutate the caller bundle, `output_filter='all'`
history snapshots are distinct objects with distinct contents, a float32 bundle takes the
allocating fallback (no `ValueError`), and read-only input arrays trace fine.

### 4.6 R6 behaviour changes not in the report's summary table

1. `_advance_along_rays` writes **through** `rays.x/.y/.z/.opd`. Pre-fix those attributes were
   *rebound* to new arrays, so a caller holding the original array saw nothing. I demonstrated
   the change: `_transfer` on a bundle whose fields are views into a caller-owned buffer now
   writes into that buffer. `trace`/`trace_world` copy first so the public path is safe, but
   `_intersect_surface`/`_transfer` are in `intersection.__all__` and `analysis/ghost.py` plus
   the FD differential path reach them directly. §7.8.
2. `_transfer` now leaves **dead** rays at their z instead of zeroing it, so
   `image_rays.z` of a vignetted ray is its death-point sag, not 0 (measured: `z = [0, 7e-4, 7e-4]`).
   This is arguably more consistent (`x`/`y` were already frozen), and I confirmed no consumer
   outside `raytrace/` reads `image_rays.z`. §7.7.

### 4.7 R7 — all eight items

| # | check | measurement |
|---|---|---|
| 1 | `_refract` first-failure-wins | a bundle carrying `RAY_APERTURE` (1) into a TIR comes out `[2 1]` — the pre-existing code (1) survives, the fresh TIR gets 2 |
| 2 | `opd_phase={'wrapped','unwrapped'}` | default unchanged (wrapped, `p7_fromfield.py` still measures the full [−494.6, +498.3] nm interval); bad value rejected with the §2 prefix |
| 3 | `trace_summary` evanescent | `[TIR=0, aperture=0, miss=0, nan=0, evanescent=5]` on a grating whose m=1 order is evanescent; `unclassified=1` appears when a ray carries an unknown code, and does **not** appear otherwise |
| 4 | registry lock | desk-checked: both `GLASS_REGISTRY` / `_glass_cache` writes in `_register_fixed_index` are now under `glass._GLASS_CACHE_LOCK`; the import-time `'__thin_lens__'` registration is kept deliberately (it is load-bearing for `surfaces_from_elements`) |
| 5 | `raytrace_system` clone | two calls on the same element list give identical thicknesses `[0.0, 0.12, 0.0]` and identical traced `y`; a caller-held `surfaces_from_elements` list is untouched |
| 6 | `layout.py` naming | module docstring now states plainly there is no layout geometry and points at `analysis.plotting.plot_lens_layout` / `raytrace.spot_diagram` |
| 7 | `_adrt_jax` single pass | JAX vs the NumPy dual backend on a conic singlet: `opd` 6.9e-18, `x`/`y` **0.0**, `ux` 0.0, `uy` 1.4e-17 |
| 8 | `field_of_view` | sensor-driven branch matches an independently computed Gaussian magnification `m = -f/(s-f)` to **<1e-12** and emits **no** warning; the no-sensor finite-conjugate path returns the legacy aperture half-angle behind a `RuntimeWarning` that says it is a numerical aperture |

---

## 5. Discrepancies between the WP report and my measurements

### 5.1 R6 memory saving — **does not reproduce** (P3, report/changelog accuracy)

`WP-A1_REPORT.md` §2.6 and `WP-A1_CHANGELOG.md` line 251 claim `tracemalloc` peak
**235.6 → 219.4 MiB** at N = 1e6 (and 70.7 → 65.8 MiB at N = 3e5). I measure, on the same
7-surface / `output_filter='last'` shape, medians of 3 interleaved runs **after** warming both
paths:

```
N = 1 000 000   shipped (in-place) : 178.3 MiB   [178.3, 178.3, 178.3]
                allocating          : 178.3 MiB   [178.3, 178.3, 178.3]
N =   200 000   shipped 35.7 MiB    allocating 35.7 MiB
```

i.e. **no measurable peak-memory difference**. (A single un-warmed first call reads 190.1 MiB
for the in-place arm, *higher* — that is import/warm-up noise, which is probably what the WP's
un-interleaved number captured.) The mechanism supports this: the allocating form's temporaries
are freed as each attribute is rebound, so the instantaneous peak differs by ~1 array, not by 8.
The **bit-identity and the 1.03–1.10× speedup do reproduce**; only the memory claim should be
dropped or re-derived before it reaches `CHANGELOG.md`.

### 5.2 R4 residual-risk justification is wrong (P3) — see §4.3 and §7.2.

### 5.3 Everything else in the report's summary table reproduces

Spot-checked and confirmed: R1 `+36.194 → −11.720` vs oracle `−11.714` (repro reports
`−11.7204` / `−11.7137`, difference `−0.0067`); R2 `−1879.86 → +0.15`; R3's whole S1 table; R4's
`dz` and alive masks; R5's `0.26200000 → 0.17425045` and ratio `1.503583 → 1.000000`; R6's
`L_true 0.15 → +0.150`, edge ratio `0.5000 → 1.0000`, grazing `alive[T T] → [F F] err 3`.

---

## 6. Collateral damage and the "checked and found correct" list

**Tests run** (all `OPENBLAS_NUM_THREADS=1`, `-q -p no:cacheprovider --no-header`):

| command | result | duration |
|---|---|---|
| 16 raytrace/seidel/new unit files (`test_audit2609_a1_exit_vertex`, `test_audit2609_a1_raytrace`, `test_raytrace`, `test_audit_raytrace`, `test_audit_w3_raytrace_parity`, `test_audit_w5_raytrace_bundles`, `test_audit_w6_raytrace`, `test_niche_audit_r2_seidel_wfe_sign`, `test_niche_audit_w3_raytrace_sources`, `test_niche_c3_gap_paraxial_guard`, `test_niche_p7_seidel_gate`, `test_s3_10_conic_core_shared`, `test_seidel_ground_truth`, `test_v5_4_1_raytrace_mirror_backward_ray`, `test_v5_4_6_wave8_raytrace`, `test_v5_2_off_axis_conic_surface_frame`) | **372 passed, 2 skipped** | 173.9 s |
| `python validation/run_all.py test_raytrace` | **54/54 passed**, file PASS | 41.2 s |
| the three `test_audit2609_a1_*` files incl. my new one | **72 passed** | 36.9 s |

The 2 skips are optional-dependency cross-checks (`rayoptics`, `Optiland` not installed) —
pre-existing, not resource preconditions on library behaviour.

One transient: mid-run the whole library became un-importable with
`NameError: name 'lru_cache' is not defined` from `elements/lenses_maslov.py:195` while another
WP was mid-edit. It cleared on retry; unrelated to WP-A1 (nothing in `raytrace/` imports it).

**Audit's verified-correct baselines, all re-measured on HEAD and unchanged:**

| quantity | audit | my re-measurement |
|---|---|---|
| OPL vs the 60-digit `decimal` oracle, 5 systems | 1.39e-17 m | **1.388e-17 m** (`p2b_opl.py`) |
| spherical fast path vs a stable Spencer–Murty root, 480k rays | 3.3e-16 m | **3.331e-16 m** (`p1b_intersect.py`) |
| fast path vs Newton | 3.0e-17 m | 7.752e-18 / 7.481e-18 / **2.971e-17 m** (`p11_misc.py`) |
| coordinate breaks, 200 random cases | exact inverses | position **6.939e-18 m**, direction **3.331e-16** |
| spherical Seidel real-ray ratios | −0.998 / −0.999 | **−0.9982 / −0.9992**; 4× plano-convex split reproduced |
| JAX parity (x64) | 6.9e-18 / 2.8e-17 m | **6.939e-18 / 2.776e-17 m**, `max abs(dL) = 0` |
| `jax.grad` vs FD | 1.4e-9 | **1.378e-9** |
| differential Jacobians | 1e-9 … 1e-7 | **1.698e-9 … 1.005e-7**; analytic-vs-FD `opd` to 2.8e-17 |
| `rays_from_field` sign | 5e-16 | **2.98e-16**; converging-wave focus **0.001 nm rms** |

---

## 7. Open items for the orchestrator

| # | sev | item |
|---|---|---|
| 1 | **P3** | **`WP-A1_CHANGELOG.md` line 250-251 carries a memory saving that does not reproduce** (§5.1). Drop the `tracemalloc peak` column or re-derive it with interleaved, warmed runs before it reaches `CHANGELOG.md`. The speedup column and the bit-identity claim are sound. |
| 2 | **P3** | **Factual error in shipped source and in the report**: `intersection.py`'s R4 residual-risk comment (and `WP-A1_REPORT.md` §2 R4) says "the conic domain always contains the sphere domain". False for k > 0 (oblate: `abs(R)/sqrt(1+k) < abs(R)`). Measured: no behaviour change, so this is a one-line comment fix, not a code fix. |
| 3 | **P3** | `ray_fan._reference_sphere_radius`'s third parameter is named `n_chief` but receives a **direction cosine** (`ref_y.N[0]`), not an index. Rename to `cos_chief` / `N_chief`. Also worth stating in the docstring that the library deliberately uses `abs(xp_z)/abs(N_chief)` where Zemax documents `abs(xp_z)` — I measured the difference at 0.065 % of the WFE at 3° on an f/2.4 meniscus, 0 on axis. |
| 4 | **P3** | **float32 dtype inconsistency in the helper** (relates to audit §15.6): `exit_vertex_transfer` on a float32 bundle returns `opd` as **float64** while `x`/`y`/`z` stay float32, because `n_exit_arr = np.asarray(n_exit, dtype=np.float64)` is a 0-d array and therefore non-weak under NEP 50. Fix: validate in float64, then `n_exit_arr = n_exit_arr.astype(out.opd.dtype, copy=False)`. |
| 5 | **P3** | A ray with `N = NaN` is classified as **grazing** by `vertex_plane_transfer_t` (`abs(nan) > tol` is False) and killed with `RAY_MISSED_SURFACE`. Behaviour is safe (frozen, no poisoning of neighbours — measured), but `RAY_NAN` is the honest code. Same in `_transfer` and the flat `_intersect_surface` branch (pre-existing there). |
| 6 | **P3** | **R3 disclosure gap**: `seidel_coefficients` still ignores `aspheric_coeffs[6]/[8]/…` **silently** (measured: `A6 = 1e9` produces zero warnings and an unchanged `S1`). The numbers are right — A6 generates 5th order, which third-order theory cannot carry — but silence is exactly the pattern that let R3 survive for years. Adding A6+ to the existing `_unrepresentable` warning list is a 2-line change. |
| 7 | **P2** | **Contract detail the next wave needs, absent from the report's §0 table**: `_transfer` now freezes a DEAD ray's `z` instead of zeroing it, so `image_rays.z` of a vignetted ray is its death-point sag. Consumers that mask on `alive` are fine; any that assumed `z == 0` for all rays are not. Should be added to the migration note the next wave reads. |
| 8 | **P3** | **In-place mutation contract**: `_advance_along_rays` writes *through* `rays.x/.y/.z/.opd` (pre-fix the attributes were rebound). `trace`/`trace_world` copy first, so the public path is safe, but `_intersect_surface`/`_transfer` are exported in `intersection.__all__` and reached directly by `analysis/ghost.py` and the FD differential path. Their docstrings should say "the arrays you pass are modified in place, not just the attributes". |
| 9 | **P3** | `exit_vertex_transfer` is in `lumenairy.raytrace.__all__` but **not** in the top-level `lumenairy` namespace, while `opd_fan_data` is. If it is to be "the ONLY way a consumer reads exit OPL", the top-level export helps discovery. `lumenairy/__init__.py` is the orchestrator's file. |
| 10 | **P3** | Two of the audit's own repro scripts now print **stale narrative** alongside correct post-fix numbers: `p5_seidel_asph.py` adds `dS1_pred` to an `S1_lib` that already contains it (so its `S1_fixed` column double-counts — read `S1_lib`), and `p11_misc.py` §A/§B print pre-fix conclusions under post-fix measurements. The measured numbers are right in both. Not WP-A1's files; worth a note so nobody re-reads those conclusions as current. |

### Not findings, recorded so the next verifier does not redo them

* `glass_after='MIRROR'` is unreachable through `trace()` (the trace loop resolves every
  `glass_after` through `get_glass_index` first), so `resolve_exit_index`'s `'MIRROR'` string
  branch is dead defensive code. The live branch is `is_mirror`, and it is correct.
* `opd_fan_data` with a vignetted chief ray returns an all-NaN fan, not silent garbage.
* `abs(N) = 2e-30` (just above the tolerance) survives the transfer and is moved to `x = -5e25`.
  That is the documented tolerance boundary, shared with `_transfer`; not new.
* One R1 test uses an S2-shape pre-fix-referencing arm
  (`assert abs(lib - prefix_value) > 0.5 * prefix_error`). The pre-fix values are macroscopic
  (47.9 waves at f/4), decades outside any cross-build spread, so it is not build-fragile.

---

## 8. Independent fail-before re-confirmation

The WP claims every regression test fails on the pre-fix code. I re-did this myself rather than
trusting its harness, by monkeypatching each pre-fix implementation back in-process and re-running
the WP's own test functions (`scratchpad/verify_a1/v_failbefore.py`):

```
R1   remove the reference sphere        3 arms fail, 1 passes  (the designated parabola CONTROL)
R2   remove the entrance eikonal        2 arms fail, 1 passes  (the designated on-axis CONTROL)
R2   seed -(Lx+My)  [the audit's text]  2 arms fail            <-- the test WOULD have caught the audit's sign
R3   strip conic + aspheric_coeffs      3 arms fail
R6   restore the 2-pixel estimator      2 arms fail, 2 pass    (L = 0.05 and the converging-wave CONTROLS)
R6b  restore the _transfer teleport     1 arm fails, 1 passes  (the ordinary-transfer CONTROL)
R7-1 restore the unconditional stamp    1 arm fails
-----------------------------------------------------------------------------------
14 / 19 arms fail on pre-fix code; the 5 that pass are exactly the designated controls.
```

The WP's fail-before claim is **confirmed independently**, and the R2 arm additionally shows the
test would have rejected the sign the audit's own text prescribes.

---

## 9. What I added

One new test file (permitted by `VERIFY_TEMPLATE.md`; no existing file was modified, no git
write commands were run):

**`tests/unit/test_audit2609_a1_verify_oracles.py`** — 7 tests, 6.2 s, all passing. It closes the
two gaps I found in WP-A1's own regression set:

1. `test_at_exit_vertex_matches_a_60_digit_closed_form_oracle[3 fixtures]` — the in-set test
   `test_at_exit_vertex_is_the_analytic_vertex_plane_opl` compares the helper against
   `img.opd + n*(-img.z/img.N)`, its own formula on its own inputs, and every fixture exits into
   air. This one runs a 60-digit `decimal` trace from the **closed form of the conic**, reads no
   traced quantity, and gives two of the three fixtures `glass_after != 'air'` so
   `resolve_exit_index` is actually exercised on a refracting exit. Bar 1e-16 m with the
   derivation and the 4.3e-18 m measurement in the docstring.
   *Fail-before (verified in process):* forcing `abs(t)` fails it, forcing `n_exit = 1.0` fails
   it, and omitting the transfer fails it. (The convex/air fixture is a control for the first
   two: `t > 0` there, and its `n_exit` is already 1.)
2. `test_r3_aspheric_s3_uses_the_squared_chief_marginal_ratio[3 perturbations]` — pins the S3
   exponent against **real rays with a displaced stop** (`y_c/y_m = 0.2618` at the asphere), the
   configuration no existing test creates. Bar `abs(ratio - 1) < 0.10` with the 5th-order
   derivation; it also asserts the two hypotheses are separated by a **factor**, not a tolerance.
   *Fail-before (verified in process):* substituting the audit snippet's first power fails both
   arms.
3. `test_r3_aspheric_petzval_s4_is_untouched` — pins `dS4 = 0` exactly (the other half of the
   same expansion; the square and `dS4 = 0` stand or fall together).

Scratch scripts with every measurement in this report:
`scratchpad/verify_a1/{v_exitvertex,v_ev_edge,v_r1,v_r1b,v_r2,v_r3,v_r3b,v_r3c,v_r4,v_jax,v_r6r7,v_r7,v_r7b,v_perf,v_break,v_failbefore,v_newtests_failbefore,v_rulings}.py`.

---

## 10. Open-item resolution (post-coordinator rulings)

The coordinator ruled on §7 and asked for the changes to land in the raytrace
files and the WP-A1 docs. All nine assigned items are implemented, measured and
pinned. Item 9 (top-level export) was reassigned to the tests/CI package and is
untouched here.

| # | ruling | what changed | pin | re-measured |
|---|---|---|---|---|
| 1 | drop the tracemalloc column | column removed from `WP-A1_CHANGELOG.md` (perf table) and `WP-A1_REPORT.md` §2 R6, each replaced by a short paragraph stating the interleaved re-measurement and why the mechanism predicts it | n/a (TESTING_STANDARDS forbids timing/memory assertions) | medians of 3 interleaved post-warm-up runs: **178.3 MiB for both** implementations at N = 1e6, **35.7 MiB for both** at N = 2e5. Speedup 1.03–1.10x and bit-identity retained |
| 2 | fix the R4 oblate claim | `intersection.py`'s R4 block now states the domain is `h < abs(R)` for k = 0, unbounded for k <= −1, `h < abs(R)/sqrt(1+k)` for k > −1 — so for **k > 0 the conic domain is SMALLER** and the test is stricter, not looser — and gives the real no-regression reason (`surface_sag_general` is NaN out of domain, so Newton never converged there pre-fix). `WP-A1_REPORT.md` §2 R4 carries the same correction, flagged as a correction | existing `test_r4_true_misses_are_still_killed` already pins the oblate kills | R = 20 mm, k = +2 (domain 11.547 mm), A4 = +5e4: h = 11.6 / 13 mm dead (err 3) before and after; h = 5 / 11 mm exact to **8.7e-19 m** |
| 3 | rename `n_chief`, document the refinement | parameter is now `N_chief` with a `Parameters` block saying it is a direction cosine, not an index; a `Notes` block states Zemax's documented `abs(xp_z)` vs the shipped `abs(xp_z)/abs(N_chief)`, the measured size of the difference, and how to reproduce Zemax (`reference_sphere_radius=abs(fod.xp_z)`). Same note added to `WP-A1_REPORT.md` §2 R1 | signature pinned in `v_rulings.py`; the R1 value tests already pin the numbers | `inspect.signature` third parameter is `N_chief`; **f/4 W(1) still −11.720430 w** (unmoved). Difference vs Zemax's axial radius: 0.088 w of 135.7 (0.065 %) at 3° on the f/2.4 meniscus |
| 4 | float32 `opd` | `exit_vertex_transfer` validates `n_exit` in float64, then casts it to `bundle.opd`'s dtype when that is floating; the `z` reset also carries the bundle's dtype | **`test_float32_bundle_keeps_its_dtype_on_every_field`** (new, `test_audit2609_a1_exit_vertex.py`) | all seven fields come back **float32**; float32 vs float64 agree to **6.89e-8 relative** (float32 eps 1.19e-7); float64 `opd` still float64; bad `n_exit` still rejected |
| 5 | `RAY_NAN` for non-finite `N` | `_kill_grazing` → **`_kill_unreachable`** (old name kept as an alias), which splits the single "cannot reach the plane" mask into `RAY_NAN` where `~isfinite(N)` and `RAY_MISSED_SURFACE` otherwise. Applies at all three sites sharing `vertex_plane_transfer_t`; the kernel's second return is renamed `unreachable` and documented | **`test_non_finite_direction_cosine_is_RAY_NAN_not_RAY_MISSED`** (exit-vertex file) and **`test_transfer_and_flat_intersect_split_nan_from_grazing`** (raytrace file) | `[NaN, grazing, healthy]` → `error_code [4, 3, 0]`, `alive [F, F, T]` at **all three** sites; first-failure-wins still holds on the NaN branch; the healthy ray is untouched |
| 6 | warn on A6/A8 | `seidel_coefficients`' `_unrepresentable` list now names any non-zero `aspheric_coeffs[p]` with `p >= 6`, with a reason string saying third-order theory has no term for them so they contribute exactly 0. Warning preamble widened to cover both classes; docstring updated | **`test_r3_a6_does_not_change_the_sums`** (new) pins the numbers; the existing `test_r3_higher_order_and_non_rotational_geometry_warns` arm that asserted A4/A6 must **not** warn is **inverted** (it pinned the silence the ruling removes) and gains two silence controls (pure A4, plain conic) | A6 = 1e9 emits a `RuntimeWarning` naming `aspheric_coeffs[6]`; **all five sums bit-identical** with and without it (`S1 = 1.121423e-06` both ways); pure A4 and a plain conic stay silent |
| 7 | document dead-ray `z` | stated in `exit_vertex.py`'s module docstring, `exit_vertex_transfer`'s `Returns`, `TraceResult.at_exit_vertex`'s `Returns`, and `_transfer`'s docstring; added to `WP-A1_REPORT.md` §0 (bold row in the contract table) and to the changelog migration note as item 3 of 4 | **`test_dead_rays_keep_their_death_point_z_through_transfer`** (new) — asserts a vignetted ray sits on surface 0's closed-form conic sag, exactly, and that `at_exit_vertex` leaves it there | dead rays at `z = [1.07242e-3, 2.42951e-3]` m (the h = 8 / 12 mm sags of the R = 30 mm, k = −0.7 front face), alive rays at exactly 0 |
| 8 | document in-place arrays | `.. warning::` blocks on `_advance_along_rays`, `_transfer` and `_intersect_surface` naming the `out=` write-through, who is safe (`trace`/`trace_world` copy first) and who must own their bundle (`analysis/ghost.py`, the FD differential path); new "In-place mutation of the three kernel sites" subsection in `WP-A1_REPORT.md` §0, plus a paragraph in the changelog migration note | covered by the existing aliasing probes in `v_r6r7.py` (trace does not mutate the caller bundle; history snapshots stay distinct) | unchanged behaviour — this ruling is documentation only |
| 10 | annotate the stale repro scripts | one "POST-FIX READING NOTE" paragraph at the top of `repro/RAYTRACE/p5_seidel_asph.py` (its `S1_fixed` column double-counts; read `S1_lib`) and of `p11_misc.py` (sections A and B print pre-fix conclusions under post-fix numbers). Both annotated, neither rewritten — they are audit evidence | n/a | both still run; `p8_opdfan.py` re-run as the canonical R1/R2 check (below) |

### Re-run after the rulings

| command | result | before the rulings |
|---|---|---|
| the 17 raytrace/seidel/verify unit files | **384 passed, 2 skipped** (177.7 s) | 372 passed, 2 skipped (+7 from the new oracle file, +5 new pins) |
| `python validation/run_all.py test_raytrace` | **54/54 passed**, file PASS (16.1 s) | 54/54 |
| `repro/RAYTRACE/p8_opdfan.py` (canonical R1/R2) | f/4 `W(1) = −11.7204 w` vs oracle `−11.7137`; 5° fitted linear term **+0.15 w** | identical |
| `scratchpad/verify_a1/v_rulings.py` (13 ruling assertions) | **13 ok, 0 failed** | — |

The 2 skips are the same optional-dependency cross-checks (`rayoptics`,
`Optiland`). No new warnings appear in the suite.

### One false alarm, recorded so it is not re-raised

My first post-ruling spot check reported the 5° R2 linear term as `−1.41 w`
rather than `+0.15 w`. That was **my estimator, not a regression**: I fitted a
degree-1 line, whereas the audit's `p8_opdfan.py` reads the `rho` coefficient of
a **degree-4** fit. The 5° fan carries −18.1 waves of `rho^2` and −12.2 of
`rho^4`, so a straight-line fit is dominated by their odd residual and reads
−3.2 w even on a perfectly-seeded fan (measured at both 12.5 mm and 25 mm
semi-diameters, i.e. with and without vignetting). With the degree-4 estimator
the value is **+0.1524 w**, matching the canonical repro exactly.

### Files changed by this resolution

Source (all within `lumenairy/raytrace/`, WP-A1's ownership):
`exit_vertex.py`, `intersection.py`, `ray_fan.py`, `seidel.py`, `surface.py`.
Tests: `tests/unit/test_audit2609_a1_exit_vertex.py` (+3),
`tests/unit/test_audit2609_a1_raytrace.py` (+2, and one pre-existing arm
inverted). Docs: `WP-A1_REPORT.md` (§0 contract table + in-place subsection,
§2 R1, §2 R4, §2 R6), `WP-A1_CHANGELOG.md` (migration note, R3 warning
paragraph, R6 perf table), `repro/RAYTRACE/p5_seidel_asph.py` and
`p11_misc.py` (header annotations only). No git write commands were run.

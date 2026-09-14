# WP-A4 — Maslov / GBD / FGA lens propagators, `_lens_jax`, and the asymptotic (phase-space) family

Branch `audit-fixes-2026-09`.  Findings **S2–S7** (report §2.5), **Y1–Y5**
(report §9), §15.1 / §15.6 cross-cutting items, plus two hand-offs from WP-A2.

Every number below is MEASURED on this checkout.  Numbers labelled "pre-fix"
come from the audit's own repro scripts re-run on HEAD before the change, or
from an in-process revert of exactly the line under test (stated where used).

---

## 1. Summary table

| ID | status | files:lines | tests | oracle | measured before → after |
|---|---|---|---|---|---|
| **S2** (P0) | **fixed** | `elements/lenses_maslov.py` `_local_window_1d` / `_local_window_geometry` (new, :155-260), `_integrate_local_quadrature` (:3655-3800), `_integrate_local_quadrature_cupy` (:3888-3990), `'auto'` router (:2155-2190), `_v2_oscillation_bound` (new) | `test_audit2609_a4_maslov_gbd.py::test_s2_*` (10) | closed form `e^{iπσ/4}/√\|det H\|` of a quadratic Fresnel integral | relerr at the SHIPPED defaults **1.19 / 1.19 / 9.12 / 1.19 / 2.48 / 3.07 → 1.15e-14 / 1.29e-15 / 5.87e-15 / 1.45e-14 / 7.83e-15 / 2.67e-16**; the 40 % convergence floor gone at every (window_sigma, n); `np.clip` over-count 7.5e+01…2.0e+03 → 2e-01…2e+00 |
| **S3** (P1 ✔) | **fixed** | `lenses_maslov.py:1913-1960` (`at_exit_vertex` + the `(d−z)/N` leg) | `test_audit2609_a4_maslov_gbd.py::test_s3_*` (2) | a physically NULL prescription edit (zero-thickness flat dummy); orch `maslov_exit_vertex_check2.py` | maslov − traced ρ² **−31.303 µm → −0.053 µm** (predicted sag −31.255 µm), ρ⁴ +0.104 → +0.099 µm, residual 4.2 nm unchanged; null-edit OPD shift **4.90 waves → 2.3e-13 waves** |
| **S4** (P1) | **fixed** | `lenses_maslov.py` `_van_vleck_density` / `_maslov_kernel_prefactor` (new, :96-153) + 7 call sites (4 NumPy, 3 CuPy desk-checked) | `test_audit2609_a4_maslov_gbd.py::test_s4_*` (3) | `angular_spectrum_propagate` on a two-flat-surface (pure free space) chart | `\|E_maslov/E_ASM\|` **4.998e-10 / 9.996e-10 / 1.996e-09 → 0.99966 / 0.99963 / 0.99800**, `arg` **+1.5716 (=+π/2) → +0.0002 rad** |
| **S5** (P1) | **fixed** | `propagators/gbd.py:373-392, :965-985, :3193-3200, :3415` (5 conjugations); compensator API deprecated (:1985-2045) | `test_audit2609_a4_maslov_gbd.py::test_s5_*` (2), `test_v5_21_gbd_asm_interop.py` (7, rewritten), `test_audit_w5_propagators.py::TestP230CollinsFactor` (re-based) | the textbook q-parameter Gaussian beam, written in the test | one beamlet: `arg(E/A)` **+0.927/+2.214/+2.942 rad (= 2ψ) → 0.000000**, relL2 with no phase fit **0.894/1.789/1.990 → 1.4e-13/5.4e-13/4.2e-12**; mixed-waist bundle **9.2–18.4 % irreducible → 5.06e-02 flat**; Collins vs the analytic Gaussian **2.00 → 3.20e-16** |
| **S6** (P1) | **partially fixed** (warn) | `lenses_maslov.py:2225-2255`, `_SADDLE_FLAT_INPUT_NA` | `test_audit2609_a4_maslov_gbd.py::test_s6_*` | symplectic identity `dOPD/dv2 = −n1(v1·ds1/dv2)`, closing to 5.8e-7 | silent → `RuntimeWarning` naming the measured input NA and the two correct alternatives.  The saddle itself is unchanged (design in §6) |
| **S7** (P1) | **fixed** | `elements/_lens_jax.py` `_require_jax_x64` / `_resolve_exit_index_from_prescription` (new, :34-105), both entry points | `test_audit2609_a4_maslov_gbd.py::test_s7_*` (5) | `jax.jit` traceability; the dtype contract; a NumPy `at_exit_vertex` trace | x64 off: complex128 in → **complex64 out → RuntimeError**; phase-screen error **1.71e-05 waves rms** (f32 vs f64) → refused; `jax.jit` **ConcretizationTypeError → compiles, 1.30e-12 vs eager**; 2 hand-written exit-vertex copies → the shared operator |
| **Y1** (P0 ✔) | **fixed** | `propagators/asymptotic_canonical_fit.py:159-236`, `asymptotic_jax_twin.py:145-175` | `test_audit2609_a4_asymptotic.py::test_y1_*` (2) | an independent ray trace of the chief ray; the `extract_linear_phase=False` fit | default-flag PSF **(−3.174e-04, −6.228e-04) m → (9.785e-05, 0.0) m** vs the traced chief ray (9.665e-05, 0); peak \|E\| **2.277e-05 → 1.99959e-03** (88× low → exact); \|E\| vs the reference fit **11.2×/0.106 → 9.21e-10** |
| **Y2** (P1) | **fixed** | `asymptotic_maslov.py` `van_vleck_weight` (new) + 4 consumers | `test_audit2609_a4_asymptotic.py::test_y2_*` (2), W6-A7 inverted | the analytic q-parameter Gaussian on an exact free-space chart | `E_code/E_true` **2.0000000000256e-08 j (= iλz) → 1.0000000000128 − 4.04e-11 j**, spread 4.2e-11 |
| **Y3** (P1) | **fixed** | `asymptotic_jax_twin.py:399-440, :459-520`, `asymptotic_maslov.py` `sym2x2_max_eigenvalue` / `lg00_sampling_waist_from_M` (new), `asymptotic_aberration_tensor.py:304-320` | `test_audit2609_a4_asymptotic.py::test_y3_*` (3) | a converged 5-point central FD; the NumPy twin | `d/ds2x` **rel 8.60e-01 → 7.7e-05**, `d/dv*_x` **8.61e-01 → 7.0e-04**; JAX twin out-of-box **60 non-finite of 81 → 0**, exactly 0 on all 72 pixels NumPy zeroes |
| **Y4** (P2) | **partially fixed** | `asymptotic.py` `_warn_dropped_pixels` (new, :255-300) + 9 exits; NaN-safe `sqrt`/divide (:597-660) | exercised in the new suites; no perf test (TESTING_STANDARDS S1) | — | silent zeroing → one `RuntimeWarning` naming the fraction and the dominant reason (measured 48/49, 98.0 % on a 3× grid; **no** warning on a clean in-box call); two leaked NumPy `RuntimeWarning`s gone.  The three perf items are **deferred** (§6) |
| **Y5** (P3) | **mostly fixed** | `asymptotic_aberration_tensor.py` (w_o clamp, `A_lead` guard, `pupil_modes` warning, Seidel caveat), `asymptotic.py` (Seidel caveat, radiometry text), 3 × "exact at the stationary point" | `test_audit2609_a4_asymptotic.py::test_y3_default_w_o_is_frozen_and_matches_numpy` | the NumPy/JAX pair; the closed form | `w_o` cross-backend divergence (97 % on `\|L\|` at `λ_max(Re M) < 1`) → identical helper; `A_lead` `inf` → masked; `pupil_modes` silent → warns.  Import-time monkey-patching and the triplication are **deferred** (§6) |
| **S9** (P2, my slice) | **fixed** (estimator) / **deferred** (GBD FFT kernel) | `lenses_maslov.py` `_v2_oscillation_bound` | `test_audit2609_a4_maslov_gbd.py::test_s2_v2_oscillation_bound_uses_the_total_variation` | `TV(T_n) = 2n` on [−1, 1], written in the test | estimator under-count up to **2.47×** (136.2 → 337.1) closed; `np.clip` over-count closed with S2 |
| **S10** (P2) | **fixed** (routing) / **verified correct** (FGA vector) / **deferred, out of scope** (Maslov vector) | `propagators/fga.py`: `_caustic_zone` chief-ray metric, `_universal_route` de-tilted collimation test, new `_global_mean_tilt` / `_remove_global_tilt` | `test_audit2609_a4_fga_s10.py` (16) | a real-ray geometric focal zone from `raytrace.trace` + `at_exit_vertex` (a different code path from the differential fan) | caustic zone for a 0.05 rad tilt **[2.002, 11.020] mm → [1.019, 1.029] mm** against a real focus at **1.0185 mm**; zone centre moved **+534.1 % → −0.25 %** (oracle −0.27 %); route **'phase_screen' → 'fga'**, matching the untilted beam; collimation residual for a 0.05 rad plane wave **5.000e-02 → 1.257e-08** |
| **S11** (P3, my slice) | **partially fixed** | `lenses_maslov.py:1574,1580` (`dy` on the fold-split legs) | — | — | anamorphic fold gaps used `dy = dx`; now threaded.  `uniform_fold_airy`/`pearcey` left dead — see §6 |
| §15.9 feature | **not done, deliberately** | — | — | — | the Pearcey path could not be validated in this pass; left dead code (§6) |
| **NEW** (found verifying S3) | **fixed** | `lenses_maslov.py` `_solve_fit` conditioning gate (`_GRAM_COND_MAX` / `_GRAM_COND_SINGULAR`) | `test_audit2609_a4_maslov_gbd.py::test_solve_fit_*` (3) | an analytically rank-1-deficient design matrix; the `output_plane_distance` composition contract | the v5.21 normal-equations Cholesky returned an ARBITRARY null-space member on a rank-deficient Gram: two charts agreeing to **3.1e-15** gave coefficients **0.869 waves** apart.  `output_plane_distance` vs baking the distance in: relL2 **0.449 / 0.757 / 0.849 / 1.384 → 3.48e-05 / 1.70e-05 / 2.33e-05 / 9.27e-04** at d = 0.5 / 1 / 2 / 5 mm |
| WP-A2 hand-off (a) | **fixed** | `lenses_maslov.py:1716-1730` | `test_audit2609_a4_maslov_gbd.py::test_a2_stop_index_*` (7) | `_lens_real._normalise_stop_index` | out-of-range / float `stop_index` warned-and-continued → **raises** with the §2 prefix; `-1` now means the last surface |
| WP-A2 hand-off (b) | **fixed** | `lenses_maslov.py:1677-1685` | `test_audit2609_a4_maslov_gbd.py::test_a2_anamorphic_*` | the smaller semi-extent | an anamorphic grid whose **y** extent is too small now warns (it did not before) |

**Verified-correct items re-measured and unchanged** (the WP's "keep intact"
list): Wick moments and LG/HG orthonormality (the W6 suite's own pins, 82
passed); the saddle-point algebra vs a brute-force quadrature of its own
integrand (W6-A3/A9, 9 passed after the oracles were updated to the corrected
weight); the W6-A1 branch claim; `propagate_hf_chebyshev_quadrature`
**2.7398e-11** against the analytic Gaussian (audit: 2.74e-11) and its Van
Vleck density `4.99999999999e+07` vs `1/(λz) = 5e+07`; NumPy/JAX parity inside
the box **1.148e-11** on `aberration_tensor` (0,0) and **2.745e-10 RMS /
9.353e-10 worst** on the 17×17 grid (audit: 7.99e-11, 2.64e-10 / 7.14e-10);
`solve_envelope_stationary_jax_ift` vs NumPy **3.042e-19**; x64 enforcement in
the asymptotic family; FGA untouched.

---

## 2. Per finding

### S3 — the Maslov canonical chart was fitted on the curved exit SURFACE

**Wrong.** `apply_real_lens_maslov` read `tr.image_rays` — the rays as
`raytrace.trace` leaves them, at `z = sag(ρ)` of the last surface — and fitted
`(s2, v2) → OPD` there while documenting and returning it as the exit-plane
field.  The missing `n_exit·sag(ρ)/N` is a pure ρ² term the Chebyshev fit
absorbs silently.  With `output_plane_distance ≠ 0` the leg was `t = d/N`
instead of `(d − z)/N`.

**Changed.** `exit_rays = tr.at_exit_vertex()` (the shared §15.1 operator, which
also resolves `n_exit` from the prescription and kills grazing rays rather than
teleporting them), and the `output_plane_distance` leg now starts from `z = 0`
and is alive-masked on `rt.EXIT_VERTEX_GRAZING_TOL`.  Composed, the two legs are
the `(d − z)/N` the requested plane needs, with the sag leg priced at the exit
medium's index and the free leg at `output_plane_n`.

**Verified.**

* `repro/orch/maslov_exit_vertex_check2.py` (the orchestrator's own F-O4
  fixture), with an in-process revert of `TraceResult.at_exit_vertex` to
  `self.image_rays` for the "before" row:

  ```
  PRE-FIX  maslov-traced: tilt=+0.000 um  rho^2=-31.303 um  rho^4=+0.104 um  residual RMS=4.3 nm
  POST-FIX maslov-traced: tilt=-0.000 um  rho^2= -0.053 um  rho^4=+0.099 um  residual RMS=4.2 nm
  thin-traced (control) : tilt=+0.000 um  rho^2= +0.000 um  rho^4=+0.007 um  residual RMS=0.1 nm
  exit sag at rho=1: -31.255 um
  ```

  i.e. the ρ² term collapses onto the predicted sag and the ρ⁴ + residual are
  untouched.  The flat-exit control (plano-convex, `thin − traced`) has no ρ²
  term before or after.
* Chart level, needing no third-party tracer: appending a zero-thickness FLAT
  dummy surface is a physically null edit that moves `trace`'s last surface onto
  the vertex plane.  Pre-fix it changed the fitted OPD by **4.90 waves** at
  ρ = 0.14 mm (= `|sag| = 4.906 µm` at λ = 1 µm); post-fix the two design
  matrices agree to **4.4e-15** and the OPD right-hand sides to **2.3e-13
  waves** on a ~1055-wave OPL.

**Residual risk.** The field-level comparison in the audit's `p16` is dominated
by fit CONDITIONING at `poly_order=5` on a 0.3 mm aperture, not by the sag: I
measured the two prescriptions' `coef_opd` differing by up to 15.8 waves from a
1e-13 perturbation of an identical design matrix (`cond(AᵀA) ≫ 1e13` at that
order).  That is the audit's own separate conditioning finding, unchanged here;
the chart-level pin above is immune to it.

### S4 — Van Vleck density and the `k/(2πi)` prefactor

**Wrong.** The integrand carried `|det(∂s1/∂v2)|`; the Van Vleck–Morette kernel
in d = 2 requires its square root, and the `k/(2πi) = 1/(iλ)` prefactor was
missing entirely.  The docstring called the scale "an arbitrary overall
prefactor"; it is not, and neither factor is constant.

**Changed.** One shared `_van_vleck_density(|det J_norm|, v2x_h, v2y_h)` at all
seven integrand sites (`quadrature`, `stationary_phase`, `local_quadrature`,
`levin`, and the three CuPy twins), plus `_maslov_kernel_prefactor(wavelength)`
applied once to the assembled field just before `normalize_output`.
`_integrate_levin` needed the `v2x_h` / `v2y_h` half-widths back (E-L9 had
dropped them as dead): with `|det J|` the measure cancelled exactly against the
weight, with `sqrt(|det J|)` it leaves `sqrt(v2x_h·v2y_h)` for the unit-box
engine to carry.

**Verified.** `repro/MASLOV-GBD-FGA/p12_prefactor.py`
(`normalize_output='none'`, `integration_method='quadrature'`, vs
`angular_spectrum_propagate`):

| λ, z | `|ratio|` before | `ratio/(λz)` before | `|ratio|` after | `arg` before → after |
|---|---|---|---|---|
| 1 µm, 0.5 mm | 4.998e-10 | 0.99960 | 0.99966 | +1.5716 → +0.0002 |
| 1 µm, 1 mm | 9.996e-10 | 0.99957 | 0.99963 | +1.5720 → +0.0008 |
| 1 µm, 2 mm | 1.996e-09 | 0.99804 | 0.99800 | +1.5693 → −0.0017 |
| 2 µm, 0.5 mm | 9.996e-10 | 0.99965 | 0.99962 | +1.5747 → +0.0024 |
| 2 µm, 1 mm | 1.996e-09 | 0.99801 | 0.99791 | +1.5706 → −0.0009 |
| 2 µm, 2 mm | 3.988e-09 | 0.99697 | 0.99671 | +1.5693 → −0.0020 |

The `ratio/(λz) = 1.000` column across two wavelengths and three distances is
the proof that the Jacobian POWER was wrong; `arg = +π/2` is the missing `1/i`.
The residual 0.2–0.3 % after the fix is the chart + quadrature accuracy (the
pixel-to-pixel spread of the ratio, 1.4e-3…2.3e-3, is unchanged by the fix).

**All four integrators are now on ONE absolute scale** — which is also how I
checked that the Levin measure factor (`sqrt(v2x_h·v2y_h)`) is the right one, a
quantity that cancels identically in the pre-fix algebra and so had no other
witness.  Same free-space chart, `normalize_output='none'`, vs
`angular_spectrum_propagate`:

```
       quadrature: |E/E_ASM| = 1.000667  arg = +0.00004 rad  spread = 5.78e-03
            levin: |E/E_ASM| = 0.999394  arg = -0.00084 rad  spread = 4.57e-03
 stationary_phase: |E/E_ASM| = 0.959119  arg = +0.06190 rad  spread = 5.89e-02
```

(`stationary_phase`'s 4 % is the leading-order saddle truncation on a
mildly-oscillatory chart, not a normalisation error.)  Every integrator
preserves the complex64/complex128 dtype contract — smoke-tested on all four ×
both dtypes, all finite.

**Residual risk.** The three CuPy twins are desk-checked only (CuPy is not
installed); they take the identical `_van_vleck_density` call, so they cannot
drift from the NumPy sites, but the prefactor is applied in the shared host-side
driver and therefore covers them too.

### S2 — `local_quadrature` and the `'auto'` router

**Wrong.** Three compounding defects, all reached by the DEFAULT
`integration_method='auto'` on any real focusing chart: (a) the sampling box was
scaled by the Hessian EIGENVALUES but laid out on the COORDINATE axes, so the
two widths swapped whenever `H44 > H33` and `H34` was ignored entirely; (b) the
samples were a hard-truncated uniform Riemann sum of a chirp, leaving the
Fresnel endpoint oscillation — an `O(1/extent)` error that does not shrink with
`local_n_samples`; (c) `np.clip` folded out-of-box samples onto the box edge
while still counting them at the full unclipped cell area.

**Changed.**

1. `_local_window_geometry` diagonalises the physical 2×2 Hessian
   (`θ = ½ atan2(2b, a − d)`) and the lattice is built on the rotated principal
   axes.  The rotation is orthogonal, so the area element `σ1σ2 dξ²` is
   unchanged by it.
2. A Gaussian taper of width `window_sigma / 3` multiplies the samples.
3. `_local_window_1d` divides that taper's exact effect on the QUADRATIC MODEL
   back out — computed as `∫exp(i s ξ²)dξ / Σ_k exp(i s ξ_k²) w(ξ_k) dξ` on the
   SAME finite lattice, with `s = sign(λ_j)`.  Because the correction is
   computed on the same lattice, the scheme is exact for a quadratic chart at
   any `local_n_samples` / `local_window_sigma`, and for a real chart it is the
   Gaussian-regularised saddle with the model divided out — i.e. never worse
   than `stationary_phase`, plus whatever non-quadratic content the lattice
   resolves.
4. Out-of-chart samples are DROPPED (the Chebyshev recurrences are not accurate
   outside the box either), not folded onto the boundary.
5. The CuPy twin follows, taking the lattice / taper / correction from the same
   host-side `_local_window_1d`.

**Verified** (`repro/MASLOV-GBD-FGA/p6`, `p7`; synthetic charts whose exact
value is closed-form, and on which `stationary_phase` was already exact at
`0.00e+00`, so the chart and fit are not in question):

* defaults (`n = 8`, `ws = 3.0`): **1.19 / 1.19 / 9.12 / 1.19 / 2.48 / 3.07 →
  1.15e-14 / 1.29e-15 / 5.87e-15 / 1.45e-14 / 7.83e-15 / 2.67e-16**;
* swap symmetry: `(A, B)` and `(B, A)` now give bit-identical answers (they
  differed by a factor 7.7 in relative error);
* the convergence ladder: **1e-14…1e-15 at every** `(ws, n)` in
  {3, 4, 6, 10} × {8, 16, 32, 64, 128}, against a 4.0e-01 floor from n = 32 on;
* `ws` = 20 / 40, where the requested window exceeds the fitted chart box:
  **7.5e+01…2.0e+03 → 2.8e-01…2.2e+00**, i.e. bounded by the honest cost of
  truncating the chart instead of the `np.clip` over-count.

**Routing.**  The WP asked for `'auto'` to go to `stationary_phase` or FGA
"where `local_quadrature` is not provably better".  I measured it rather than
assuming: on an f = 6 mm N-BK7 biconvex at its exit plane, scored against a
converged uniform quadrature (n_v2 = 320; self-converged to 8.5e-04 at
n_v2 = 256, 1.4e-03 at 192):

```
stationary_phase                    relL2 8.4347e-01   [0.8 s]
local_quadrature n=8  ws=3 (default) relL2 2.5757e+00  [0.3 s]
local_quadrature n=16 ws=3           relL2 1.4677e+00  [0.6 s]
local_quadrature n=24 ws=4           relL2 1.1446e+00  [1.0 s]
```

So `stationary_phase` is measured-better on a real chart as well as exact on the
synthetic ones, and `'auto'` now routes there.  **Both** asymptotic evaluators
are O(1) wrong at a lens EXIT plane — the v2-Hessian and `ds1/dv2` both collapse
there, which is the opposite of the asymptotic regime — so the fallback now
carries a `RuntimeWarning` saying exactly that and naming the two remedies
(`'quadrature'` with an explicit `n_v2`, or `output_plane_distance=` to move the
observation plane to where the asymptotics belong).

**Residual risk.** `'auto'` never selecting `local_quadrature` means the fixed
integrator is now opt-in only.  That is deliberate: it is exact on a quadratic
chart but, on the real chart above, its window reaches into pupil zones where
the order-4 fit is extrapolating, and it scored worse than the cheaper
`stationary_phase`.

### S5 — the GBD Gouy / Collins phase was conjugated

**Wrong.** `BeamletBundle.Q` is the ENGINEERING `1/q` (`q_code = conj(q_phys)`)
and the renderer converts on output (`exp(+0.5j k conj(Q) ρ²)`); the AMPLITUDE
did not.  `propagate_beamlets_freespace` used `Q_new/Q_old` where the physics
ratio `q0_phys/q_phys` is `conj(Q_new/Q_old)`, and the two Collins sites used
`1/(A + BQ)` where it is `conj(1/(A + BQ))` (A, B real).  The library documented
the resulting `2 arctan(z/zR_beamlet)` as an inter-propagator convention and
shipped three public functions plus a test file to compensate.

**Why it is an error, re-derived here.** For `q = z − i z_R` under
`exp(−iωt)`/`exp(+ikz)`, the fundamental Gaussian's amplitude ratio is
`q0/q = −i z_R/(z − i z_R)`, whose modulus is `w0/w(z)` and whose argument is
`−arctan(z/z_R)`.  In the engineering convention `1/q_phys = conj(Q)`, so
`q0/q = 1/(1 + z conj(Q0)) = conj(1/(1 + z Q0))`.

**Verified.**

* `p1_gbd_gouy.py`, one beamlet vs the analytic Gaussian written in the script:
  `arg(E/A)` **+0.927295 / +2.214297 / +2.942255 rad → 0.000000** at
  z = 0.5 / 2 / 10 z_R, and relative L2 **with no phase fit** 0.894 / 1.789 /
  1.990 → **1.41e-13 / 5.37e-13 / 4.20e-12**.  Pre-fix the discrepancy matched
  `2ψ` to 1e-13, which is what made it diagnosable.
* `p2_gbd_gouy_full.py`, full pipeline vs `angular_spectrum_propagate`
  (N = 256, λ = 1 µm, z = 3 mm): residual global phase **+3.133 rad → +5e-06
  rad**, relL2 raw **1.999 / 1.995 / 1.987 → 1.76e-03 / 7.02e-03 / 1.57e-02** at
  waist_factor 1 / 2 / 3 (the Gabor-frame floor, previously only reachable after
  the converter).
* `p3_gbd_mixed.py`, where the error was NOT a global phase
  (`decompose_field_adaptive` mixes 3 µm and 12 µm beamlet waists): raw relL2
  **1.78e-01 / 3.14e-01 / 8.41e-01 / 1.82e+00 → 5.06e-02 / 5.06e-02 / 5.06e-02 /
  5.11e-02** at z = 20 / 50 / 200 / 1000 µm, and the best-global-phase-removed
  value now EQUALS the raw one (nothing left to remove) where it was 9.2–18.4 %.
* `p4_gbd_persurface.py`'s finding stands: the per-surface Collins version was
  near-global at the exit plane, which is why `apply_real_lens_gbd` overlaps
  were unchanged to 6 decimals — `tests/unit/test_lens_gbd.py` and
  `test_gbd_feature_complete.py` pass unchanged.

**Compensator API.** `gbd_asm_gouy_phase` returns `0.0`; `gbd_field_to_asm` /
`asm_field_to_gbd` return `E` unchanged; all three warn through
`_deprecation.warn_deprecated_alias(version_added='5.46',
version_removed='5.48')` (`NEXT_REMOVAL_VERSION` is 5.48).  They still validate
their field argument, so an existing pipeline keeps running.

**Test inversions (defect pins).**

* `tests/unit/test_v5_21_gbd_asm_interop.py` — the whole file asserted the
  offset.  Rewritten to assert the opposite, with the waist-factor DEPENDENCE as
  the diagnostic: the residual phase spread across waist_factor 1 / 1.5 / 2.5 is
  now < 2e-07 rad where it was 0.025 rad.  A real convention offset cannot
  depend on a numerical knob.
* `tests/unit/test_audit_w5_propagators.py::TestP230CollinsFactor::test_matches_analytic_collins_and_gaussian_w_of_z`
  asserted `exp(ikz)/(1 + zQ0)` — the library's own expression, so it could not
  see a sign error in it.  Re-based on a textbook Gaussian-beam oracle written
  in the test (`exp(ikz)·(w0/w(z))·exp(−i arctan(z/z_R))`).  Measured: the
  post-fix code matches that oracle to **3.20e-16** relative; the old pinned
  expression scores **2.00** (it is the oracle's conjugate up to the axial
  piston).  A negative control now asserts the two are distinguishable on the
  fixture, and the modulus assertion — which is conjugation-BLIND and passed in
  both versions, which is exactly why the defect survived — is kept.

### S6 — the asymptotic saddle ignores the input field's phase

**Wrong and unchanged.** Both asymptotic evaluators solve `grad_v2 OPD = 0`.
The symplectic identity `dOPD/dv2 = −n1 (v1 · ds1/dv2)` (the auditor measured it
closing to 5.8e-7 relative) makes that the `v1 = 0` collimated launch ray at
every pixel and for every input, while the driver deliberately sizes the chart
to cover a diverging / tilted one.

**Changed.** A `RuntimeWarning` when either asymptotic method is selected, the
input is not declared collimated, and the driver's own measured angular spread
exceeds `_SADDLE_FLAT_INPUT_NA = 1e-3` (chosen so a numerically flat wave's
1e-10…1e-12 FFT second-moment floor never trips it, and any real divergence
does).  `collimated_input=True` silences it; `'quadrature'` and `'levin'`
integrate the true integrand and are unaffected.

**Residual risk / deferred.** The real fix is to fit the input's local
wavevector — see §6.

### S7 — `_lens_jax`: float32 and jit

**Wrong.** Neither entry point read `jax_enable_x64`, so under JAX's default a
complex128 input was truncated at `jnp.asarray` and the output came back
complex64; and both seeded the Newton inversion with
`float(x_out_grid[...])`, which raises `ConcretizationTypeError` under `jit`, so
the DEFAULT path could not be jitted at all.

**Changed.** `_require_jax_x64` (the `rcwa/_core.py` policy: RAISE, with
actionable text) at the top of both entry points; the tracer-safe initial guess
used unconditionally (`stop_gradient`-ed, since the Newton root does not depend
on its starting point, which keeps the geometry gradient identical to the
pre-fix `_diff_geom` branch); and the two hand-written exit-vertex copies
replaced by `exit_vertex_transfer_jax` with `n_exit` resolved through a new
`_resolve_exit_index_from_prescription` that names the function on failure.

**Verified**, by loading the pre-fix `_lens_jax.py` from `git show HEAD:` as a
sibling module and running both side by side:

```
jax_enable_x64 = False
PRE-FIX : returned dtype = complex64          (input complex128)
POST-FIX: RuntimeError: apply_real_lens_traced_jax: the JAX (differentiable)
          real-lens path requires double precision, but jax_enable_x64 is disabled …
PRE-FIX jit: ConcretizationTypeError: Abstract tracer value encountered where a
          concrete value is expected: traced array with shape float32[]
POST-FIX jit: compiles; max |E_jit − E_eager| = 1.2972774956e-12 (both entry points)
```

and the precision cost itself, pre-fix float32 vs float64 phase screen on the
same fixture: **1.71e-05 waves rms / 8.17e-05 waves p-v** over 1781 pixels (the
audit measured 1.5e-4…1.8e-3 waves rms on a larger lens, against 2e-10…3e-9 with
x64 — the fixture here is a 0.3 mm aperture, so the OPL and hence the rounding
are correspondingly smaller).

**Migration note** in the changelog: this is a behaviour change for a caller who
did not enable x64 — they got a silently-wrong complex64 answer and now get a
one-line refusal.

### Y1 — `extract_linear_phase=True` dropped the v2-linear phase

**Wrong.** The 5-term prefit removes `a0 + a1 u1 + a2 u2 + a3 u3 + a4 u4`.  The
`a0, a1, a2` terms depend only on `s2` and factor out of the `v2` integral;
`a3 u3 + a4 u4` are linear in the INTEGRATION variable, so dropping them changes
`g = dΦ/dv2`, hence `b`, hence the complex saddle shift `δ* = M⁻¹b/2` and
`Re(bᵀM⁻¹b/4)` — an amplitude and position error.

**Changed.** `CanonicalPolyFit.eval_phi` and `eval_phi_with_v2_grad` (and the
JAX `eval_phi_xp`, which the JAX `_compute_M_b_xp` differentiates) always add
`a3 u3 + a4 u4` and its `a3` / `a4` gradient contribution; `include_linear` now
gates only `a0 + a1 u1 + a2 u2`.  `H_Phi` is unchanged (a linear term has zero
second derivative).  `HFPolyFit.eval_phi` is untouched: its `u3, u4` are `s2`,
not an integration variable, and `propagate_hf_chebyshev_quadrature` applies the
full ramp itself.

**Verified** (`repro/ASYMPTOTIC/t8_peak_location.py`, stock N-BK7 singlet,
λ = 1.31 µm, `source_centre = (100 µm, 0)`, where the rank-deficient design
splits a 2 699.44-wave ramp into `a1 = 2699.44` and `a3 = 2740.05` waves):

```
chief-ray landing (independent ray trace): 9.665155779822701e-05   0.0
extract_linear_phase=True   peak |E| = 1.99959e-03 at (9.785275e-05, 0.0)
extract_linear_phase=False  peak |E| = 1.99959e-03 at (9.785275e-05, 0.0)
```

against the pre-fix `True` row of `2.277e-05 at (-3.174e-04, -6.228e-04)` —
88× too small and ~700 µm off, off-axis in y where the source has no y offset.
The two flags' |E| now agree to **9.21e-10** over 149 bright pixels (pre-fix the
NORMALISED profiles differed by 11.2× / 0.106, so no rescale could rescue it).
The phase still differs by the documented s2-tilt reference (`a1` = 2699.44
waves ⇒ 5.53 rad of spread), which is the W6-A4 convention and is unchanged.

**W6 pins rewritten.**
`test_w6_a4_diffracted_tilt_is_removed_from_the_returned_phase` asserted
`|a3| + |a4| < 1e-8` as a load-bearing PREMISE ("or the omission would corrupt
the AMPLITUDE too").  The omission is gone, so the assertion is restated as a
characterisation of that fixture (a surface-1 grating puts its ramp in `a1`
alone, which is what makes the phase-spread assertion below it unambiguous), and
the off-axis case it was standing in for is now pinned directly in
`test_audit2609_a4_asymptotic.py`.

### Y2 — Van Vleck–Maslov normalisation of the modal propagator

Same class as S4; see the summary table and §3 for the measurement
(`E_code/E_true` = `2.0000000000256e-08 j` → `1.0000000000128 − 4.04e-11 j`
against an exact free-space chart and the analytic q-parameter Gaussian).  One
shared `van_vleck_weight(det_J, wavelength)` in `asymptotic_maslov.py` is called
by all four consumers.

**The scale move, derived and checked.**  `A_lead` changes from `detJ` to
`−1j·sqrt(detJ)/λ`, so every `|L|²` moves by exactly `1/(λ²·|det J|)`.  Measured
on the two shipped LG-merit fixtures, at the envelope-stationary point:

| fixture | `|det J|` | factor `1/(λ²·detJ)` | `|L₀₀|²` after | after ÷ factor | previously pinned |
|---|---|---|---|---|---|
| R1 = 500 mm, ap 4 mm, w_s = 5 µm | 3.980321e-06 m² | 1.486604e+17 | 6.006606e+14 | 4.040490e-03 | 4.040485e-03 |
| … w_s = 20 µm | " | " | 4.764657e+14 | 3.205062e-03 | 3.205059e-03 |
| … w_s = 50 µm | " | " | 7.048782e+13 | 4.741534e-04 | 4.741536e-04 |
| R1 = 60 mm, ap 12 mm, w_s = 20 µm | 7.076126e-06 m² | 8.362145e+16 | 5.259136e+14 | 6.289217e-03 | 6.289211e-03 |

— agreement to 6–7 significant figures on every row, i.e. the 18-decade move is
the Van Vleck factor and nothing else.  `AberrationTensorResult` gained
`van_vleck_weight` (appended with a `None` default, like the two v5.29 fields)
so a caller can divide it back out; verified round-trip
`L_legacy = L·λ²·|det J|` → 3.205061889e-03.

**The Strehl-deficit question (WP: "verify, do not edit optimize/").**  The
audit expected the fix to make `1 − |L|²` "read sensibly".  **It does not, and
it never could on that branch** — reported here rather than papered over:

* A pure `output_modes=[(0, 0)]` request takes the CLOSED-FORM POINT-SAMPLING
  branch, which the module's own W3-T3 note already records as returning
  `U(chief)·conj(LG_k(0))` — *units of field per length* — while the σ-grid
  branch returns the true overlap `∫ conj(LG_k) U` (field × length).  The
  closed form omits the `∫d²s2` measure; it is an overlap DENSITY, over-counting
  a true overlap by `1/(π w_o²)`.
* Pre-Y2 the erroneous extra factor `λ·sqrt(|det J|)` is a LENGTH, and it
  cancelled that 1/length² by dimensional accident, which is the only reason
  `|L|² ≈ 3.2e-03` ever looked like a Strehl ratio.
* Post-Y2, `1 − |L|²` on the stock singlet is **−4.76e+14** (it was +0.9968).
  The σ-grid branch is not a Strehl amplitude either: `|L₀₀|² = 122.3` on the
  same fixture, because the pupil "mode" normalisation `sqrt(2/(π w_p²))` rides
  inside the integral.

Neither version was a Strehl deficit; the pre-fix number was a plausible-looking
coincidence.  The concrete fix lives partly outside my ownership — see §5.

### Y3 — JAX twin gradients and guards

**Wrong.** (a) the default `w_o` came from `jnp.linalg.eigvalsh(Re M)`, whose
JVP carries `1/(λᵢ − λⱼ)`; `Re M = JᵀJ/w_s² + I/w_p²` is near-isotropic for any
rotationally symmetric system (gap/mean **3.31e-10**), so every derivative that
ROTATES `Re M` was ~86 % wrong while the ones that only RESCALE it survived.
(b) the JAX twins reproduced none of the NumPy path's guards.

**Changed.** (a) `sym2x2_max_eigenvalue` (closed form, double-`where`d so the
gradient is finite at an exact degeneracy) inside a shared
`lg00_sampling_waist_from_M` that BOTH backends call — which also closes the Y5
"bit-for-bit contract is false" item, since the NumPy clamp `[1e-9, 1.0]` now
applies to both.  The default `w_o` is additionally `jax.lax.stop_gradient`-ed:
`λ_max` of a near-degenerate symmetric matrix is genuinely non-differentiable at
the branch crossing, and on this branch `w_o` is a bookkeeping convention, not a
physical length (the module says so in capital letters), so differentiating
through it is ill-posed.  Passing `w_o=` explicitly makes it a live slot; the
docstring's "Differentiable via jax.grad wrt … w_o" now states the exception.
(b) the four masks (in-box `s2`, in-box `v2`, `|det M| ≥ 1e-300`,
`|Re b_quad| ≤ B_QUAD_EXP_MAX`) plus a final finite gate are `jnp.where` gates
in `_modal_field_lg00_pixel_jax`, each a double-`where` so no NaN enters the
graph.

**Verified** (`repro/ASYMPTOTIC/t13_grad_all.py`, `t10_jax.py`):

| derivative | pre-fix grad | pre-fix 5-pt FD | pre-fix rel | post-fix grad | post-fix FD | post-fix rel |
|---|---|---|---|---|---|---|
| `d/ds2x` (explicit `w_o`) | — | — | 6.9e-04 (audit) | −9.57454010e+11 | −9.57379958e+11 | **7.7e-05** |
| `d/dv*_x` (explicit `w_o`) | — | — | 1.5e-02 (audit) | −2.34874817e+10 | −2.35038542e+10 | **7.0e-04** |
| `d/ds2x` (default `w_o`) | 9.438e-03 | 6.738e-02 | **8.60e-01** | equals the explicit-`w_o` gradient to 1.7e-11 | | |
| `d/dv*_x` (default `w_o`) | 2.316e-04 | 1.661e-03 | **8.61e-01** | " | | |

Out-of-box grid (3× the fit half-box): NumPy zeroes 72 of 81 pixels; the JAX
twin returned **60 non-finite values** there and now returns
`max |E_jax| = 0.0` with `n_nonfinite = 0`.  In-box parity is preserved or
better: `aberration_tensor` (0,0) **1.148e-11** (audit 7.99e-11), the 17×17 grid
**2.745e-10 RMS / 9.353e-10 worst** (audit 2.64e-10 / 7.14e-10).

**Resolving one of the audit's unverified suspicions.**  `d/d(source_point)`
disagreed with a 5-point FD by 6.4 % / 6.9 %, which the auditor could not
separate from FD conditioning.  Measured here: the AD value is identical
(2.37087352e+08) in both the default and explicit-`w_o` runs, while the FD
itself moves between runs (2.26250000e+08 vs 2.18750000e+08 — 3.4 % apart).  It
is FD conditioning, not an AD error; the derivative is ~9 orders below
`d/dw_s`, so the central difference is differencing away ~9 significant digits.

### S10 — the universal dispatcher's two tilt-blind discriminators

**Wrong.**  `apply_real_lens_universal` routed a TILTED but perfectly
collimated, single-valued, high-NA plane AT ITS FOCUS to `phase_screen` — NA
0.145 above the 0.12 `na_threshold` and inside the caustic, precisely the
regime the dispatcher's own docstring says the thin screen cannot handle.  Two
compounding causes, both blind to a global tilt:

1. `_caustic_zone` scored each exit ray's crossing with the optical AXIS
   (`z = -x_exit / u_exit`).  A global input tilt moves the focus off axis by
   ~`f·θ` — measured: the chief ray of a 0.05 rad tilted beam lands
   **+66.402 µm** off axis at the focal plane — so the axis crossings measure a
   different quantity and the zone came back as junk.
2. The final escape hatch asked
   `_carrier_residual_rms(E_in, None, wavelength, dx) > _NONCOLLIMATED_RESID_THRESH`,
   and that residual is EXACTLY the tilt magnitude for a pure tilt.

**Changed.**  `_caustic_zone` now scores the crossing with the CHIEF ray — the
amplitude centroid of the meridional row launched along the amplitude-weighted
mean local slope, traced in the SAME `ray_transfer_jacobian` call as the fan
(one extra array element), so it costs nothing and cannot drift from it.
`z = (x_c − x_i)/(u_i − u_c)` is the strict generalisation of `−x_i/u_i` and
reduces to it exactly when the chief ray is the axis.  A vignetted chief ray
falls back to the axis rather than to a garbage reference.  `_universal_route`
now applies the collimation test to the DE-TILTED field via a new
`_remove_global_tilt` (built on a new `_global_mean_tilt`, which reuses
`_tilt_dispersion`'s own conjugate-product local wavevector so the two agree by
construction and neither needs an FFT).

**Verified — independent oracle.**  `_ray_focal_zone` in the new test file
traces real rays with `raytrace.trace`, carries them to the exit vertex with
the shared WP-A1 operator, and takes the geometric focal zone as the axial
range where the bundle's RMS transverse spread about its OWN centroid is within
20 % of its minimum.  That is a different code path from the
`ray_transfer_jacobian` differential fan `_caustic_zone` uses, and it needs
nothing from the dispatcher.  Measured best-focus planes: **1.021305 mm**
(collimated), 1.020605 (tilt 0.02), 1.018505 (tilt 0.05), 1.019905 (tilt 0.05 +
80 µm decentre) — a collimated beam's focal DISTANCE moves by at most **0.27 %**
under tilt.

| input | `_caustic_zone` before [mm] | after [mm] | contains the real focus? | route before → after |
|---|---|---|---|---|
| collimated | [1.021056, 1.032698] | **bit-identical** | yes → yes | fga → fga |
| tilt 0.02 rad | [1.318251, 10.987635] | [1.021117, 1.032215] | **no → yes** | **traced → fga** |
| tilt 0.05 rad | [2.002278, 11.019568] | [1.019191, 1.029461] | **no → yes** | **phase_screen → fga** |
| tilt 0.05 + 80 µm dec | [1.996611, 28.212060] | [1.009260, 1.025989] | **no → yes** | **phase_screen → fga** |
| 80 µm decentre | [1.020926, 1.032689] | [1.010242, 1.028427] | yes → yes | fga → fga |
| slow lens, exit plane | [98.021603, 98.021702] | **bit-identical** | — | phase_screen → phase_screen |
| multi-valued (2 tilted) | [0.277843, 3.274212] | **bit-identical** | — | fga → fga |

The zone CENTRE moved **+534.1 %** under a 0.05 rad tilt before the fix and
**−0.25 %** after, against the oracle's own −0.27 %.  The three symmetric
fixtures are bit-identical because a centred beam's chief ray IS the axis.
`apply_real_lens_auto`'s 2-way choice follows the same `_caustic_zone`:
**'gbd' → 'fga'** for the tilted case.

Collimation discriminator (`_NONCOLLIMATED_RESID_THRESH = 0.02`):

```
tilt                raw _carrier_residual_rms   de-tilted
0.005               5.000000e-03                1.257379e-09
0.020               2.000000e-02                5.029515e-09
0.050               5.000000e-02                1.257379e-08
0.100               1.000000e-01                2.514758e-08
R=10 mm             9.714549e-03                9.715578e-03   (real divergence, preserved)
R= 3 mm             3.238183e-02                3.238526e-02   (real divergence, preserved)
0.05 rad + R=3 mm   5.956998e-02                3.238526e-02   (= the untilted value to 4.0e-09)
```

**What this does NOT claim.**  The fix makes the router FRAME-INVARIANT; it
does not make the chosen member more accurate on this fixture, and I measured
that rather than assume it.  At the focus, against the traced chief-ray landing
(+66.402 µm) and the real-ray geometric fan (+66.439 µm, 0.645 µm rms;
diffraction limit λ/(2·NA) = 3.44 µm): `phase_screen` gives centroid
**+66.469 µm** and intensity-rms width **3.269 µm**; `fga` gives **+62.524 µm**
and **12.721 µm**.  The two members disagree by 3.9 µm and a factor 3.9 — and
by the same factor at tilt 0 (3.169 vs 12.607 µm), so it is the members' own
accuracy question, which is the auditor's own open suspicion about FGA's
convergence knob through a real singlet ("measured `apply_real_lens_fga`
fidelity 0.357 ... I cannot call this a defect"), re-measured here and still
open.  What the pre-fix router did was pick between two models that differ by
4x on the observer's FRAME.  That is the defect, and it is closed.

**Edge cases re-measured** (pre vs post, same fixtures): a diverging lens with
a collimated or tilted input, and a strongly diverging input through the fast
lens, all return `None` on both sides -- correct, there is no downstream
caustic.  A beam filling the grid is unchanged to 1e-4 relative, and a
1e-300-amplitude field is unchanged (the masks are relative).  One further
improvement fell out: at a 0.20 rad tilt the pre-fix axis metric found fewer
than three "converging" rays and returned `None`, i.e. the router got NO
caustic information at all; post-fix it returns [0.9814, 0.9817] mm.

**Residual risk.**  The decentred (untilted) zone changed slightly
([1.020926, 1.032689] → [1.010242, 1.028427] mm) because the crossing is now
scored against the beam's own chief ray rather than the axis.  That is the more
correct metric — the caustic is where the bundle self-crosses — and it still
contains the oracle focus and still routes to `fga`; but it is a behaviour
change on an untilted fixture and is called out here rather than buried.

### S10 (vector) — `apply_real_lens_fga_vector` verified correct, `apply_real_lens_maslov_vector` out of scope

S10's third sub-item is that `apply_real_lens_maslov_vector` threads
`normalize_output` through with its `'power'` default and applies it
INDEPENDENTLY to `E_x` and `E_y`, evaluates the Fresnel Jones at axial
incidence for every pixel, and has no polarization transport or `E_z`.  That
function lives in `lumenairy/elements/lenses_maslov.py`, which is **outside my
current ownership** (a verifier is working on it) — the exact requested change
is in §5.

The FGA peer, `apply_real_lens_fga_vector`, is in `fga.py` and does **not**
share the defect; I measured all three sub-items rather than assuming:

* it applies ONE joint scale to `(ex, ey, ez)`.  Measured on a `(2, 48, 48)`
  Jones input with `P_x/P_y = 4` exactly, through an f = 1.2 mm singlet to 1 mm
  past the vertex: `normalize_output='none'` gives `P_x/P_y = 3.9999998702357877`
  and `'power'` gives `3.999999870235787` — the same to **2.2e-16 (1 ulp)** —
  with the total power restored to 1.0000000000000002.  The 3.2e-08 departure
  from the input ratio is the s/p diattenuation the system really applies, and
  it SURVIVES the normalisation;
* it carries the per-surface Fresnel s/p Jones with the geometric frame
  rotation (polarization ray tracing), not an axial-incidence matrix;
* it ships a real longitudinal component via `return_longitudinal=True`.

Pinned by `test_s10_fga_vector_normalises_the_jones_components_jointly`, which
is GREEN on both sides — a guard on a defect class the sibling has, not a fix.

### NEW — the canonical fit was returning an arbitrary null-space member

Found while verifying S3's `output_plane_distance` leg, not in the audit.

**Wrong.**  `_solve_fit`'s v5.21 normal-equations Cholesky is justified by "``A``
is a normalized tensor-Chebyshev Vandermonde -- well-conditioned and ~1.5x
oversampled -- so squaring the condition number in ``G`` is safe".  On a small,
fast chart it is not, and the `LinAlgError` fallback ladder cannot see the
failure: a numerically positive-semidefinite but RANK-DEFICIENT Gram factors
happily and returns an arbitrary member of the solution set.

**Measured** (f = 6 mm N-BK7 biconvex, 0.2 mm aperture, `poly_order=4`):

```
cond(A) = 1.81e+15   cond(A^T A) = 6.18e+18   rank(A) = 65 of 70 columns
inputs  : max|dA| = 3.11e-15    max|dOPD| = 6.82e-13 waves
coefs   : max|dc| = 8.690e-01 WAVES
residual: run1 1.613e-09   run2 1.756e-09   cross (A1 @ c2 vs B1) 1.756e-09 waves
```

Both coefficient vectors fit their own data equally well, and each fits the
OTHER run's data equally well — the difference is pure null space.  Invisible on
the training manifold; not invisible inside the `v2` integral, which samples
`(s2, v2)` combinations off it.  It is what made the docstring's
"``output_plane_distance`` … matches baking the same distance into the
prescription's last thickness to ~1e-10" false by 10 decades.

**Changed.**  `_solve_fit` measures `cond(A^T A)` once (an `M x M` `eigvalsh`,
microseconds beside the `A^T A` GEMM it already forms) and routes anything above
`_GRAM_COND_MAX = 1e12` to `np.linalg.lstsq`, whose minimum-norm solution is a
deterministic, unique function of `(A, RHS)` — and which is the pre-v5.21
behaviour, so this restores it exactly where it mattered.  Above
`_GRAM_COND_SINGULAR = 1/eps` it additionally WARNS, because there the fit is
genuinely rank-deficient and even the min-norm answer depends on the solver's
`rcond` cut; the message names the three real remedies (lower `poly_order`,
wider chart, more rays).  Well-conditioned charts keep the fast Cholesky path
byte-identical (the audit's own 1.5 mm fixture is `cond(G) = 1.24e9` at order 4,
well inside the gate).

**Verified.**  `output_plane_distance=d` versus baking `d` into the
prescription's last thickness, same fixture:

| d | chart dOPD | chart ds1 | field relL2 before | after |
|---|---|---|---|---|
| 0.5 mm | 6.8e-13 waves | 0.0 m | 4.487e-01 | **3.48e-05** |
| 1 mm | 9.1e-13 | 0.0 | 7.573e-01 | **1.70e-05** |
| 2 mm | 9.1e-13 | 0.0 | 8.493e-01 | **2.33e-05** |
| 5 mm | 2.7e-12 | 0.0 | 1.384e+00 | **9.27e-04** |

The chart columns are the S3 result (the composition is exact); the field
columns are this fix.  The `roi=` contract is bit-identical throughout
(`max |E_roi − E_full[slice]| = 0.0`).  S3 and S4 are unaffected by the gate
(their charts are well-conditioned): the orch ρ² term is still −0.053 µm and
`p12`'s ratios are unchanged to every printed digit.

**Residual risk.**  The gate costs one `eigvalsh` of an `M x M` matrix per fit
(M = 70 at order 4, 210 at order 6) — sub-millisecond against the `A^T A` GEMM
on ~6 000 rays.  Charts between 1e12 and 1/eps now take the SVD instead of the
Cholesky, which is the M-P5 speed-up given back on exactly the charts where it
was unsafe.

### Y4, Y5, S9, S11 — see the summary table and the changelog

The correctness-adjacent halves are done (drop diagnostics, the two leaked NumPy
`RuntimeWarning`s, the `A_lead` overflow guard, the `w_o` clamp, `pupil_modes`,
the anamorphic `dy` on the fold-split legs, the total-variation oscillation
bound, and the three wrong narrative claims).  The performance halves are
deferred with designs in §6.

---

## 3. Files touched

**Source (all within the WP's ownership list).**

* `lumenairy/elements/lenses_maslov.py` — S3, S4, S2, S6, S9 (estimator),
  S11 (`dy`), the WP-A2 hand-offs, the new cache enrolment.
* `lumenairy/elements/_lens_jax.py` — S7, §15.1.
* `lumenairy/propagators/gbd.py` — S5 (5 conjugations + 3 deprecations).
* `lumenairy/propagators/asymptotic.py` — Y2 consumer, Y4, Y5 docs.
* `lumenairy/propagators/asymptotic_maslov.py` — new shared helpers
  (`van_vleck_weight`, `sym2x2_max_eigenvalue`, `lg00_sampling_waist_from_M`,
  `B_QUAD_EXP_MAX`), Y5 wording.
* `lumenairy/propagators/asymptotic_canonical_fit.py` — Y1, Y5 wording.
* `lumenairy/propagators/asymptotic_jax_twin.py` — Y1 twin, Y2, Y3.
* `lumenairy/propagators/asymptotic_aberration_tensor.py` — Y2, Y3, Y5.
* `lumenairy/propagators/fga.py` — S10 (`_caustic_zone` chief-ray metric,
  `_universal_route` de-tilted collimation test, new private `_global_mean_tilt`
  / `_remove_global_tilt`).  Added in the S10 follow-up pass, after the rest of
  the package was committed as `32ba3ba2`.  The diff also carries ONE
  unrelated four-line hunk at `_pick_ray_transfer` (line 794): a long
  single-line `from ..raytrace.differential import ...` wrapped into
  parentheses.  That was a PRE-EXISTING `ruff I001` violation on the committed
  file (verified against `HEAD`), swept up by the `ruff --fix` pass over my own
  new imports.  Pure formatting; called out so a verifier diffing the file is
  not surprised by it.  The two new helpers are module-private
  (`_global_mean_tilt` / `_remove_global_tilt`), matching every other helper in
  that file; `fga.py` has no `__all__` and `propagators/__init__.py` (which I
  may not edit) carries an explicit re-export list, so a public name there
  would have been half-exported.  Promoting them -- the S6 follow-up and the
  carrier chain both want a de-tilt primitive -- means adding them to that list
  in the same commit.

Not touched: `propagators/subaperture.py` (no finding needed it),
`elements/lenses_gbd.py` (verified correct by the audit, and S10 did not need
it).

**Tests — new.**

* `tests/unit/test_audit2609_a4_maslov_gbd.py` (34 tests: S2 ×10, S3 ×2,
  S4 ×3, S5 ×2, S6 ×1, S7 ×5, WP-A2 ×8, fit-conditioning ×3).
* `tests/unit/test_audit2609_a4_asymptotic.py` (7 tests: Y1 ×2, Y2 ×2, Y3 ×3).
* `tests/unit/test_audit2609_a4_fga_s10.py` (16 tests: caustic-zone ×5,
  collimation/de-tilt ×7, routing invariance ×4 — one of which is the
  no-regression guard on the four regimes the audit found correctly routed —
  plus the FGA vector-normalisation guard).

**Tests — modified (each one pinned a defect or was a stale copy of a fixed
algorithm; none had its tolerance loosened).**

* `tests/unit/test_v5_21_gbd_asm_interop.py` — REWRITTEN (the whole file
  asserted the S5 offset).
* `tests/unit/test_audit_w5_propagators.py` — `TestP230CollinsFactor`'s
  amplitude oracle re-based on the textbook Gaussian (it used the library's own
  expression).
* `tests/unit/test_audit_propagation.py` — the two
  `…ModalAsymptoticStillBitEqual` inline scalar references carried the pre-Y2
  `amp_lead`; updated to the corrected weight.  The pins' tolerances (1e-8
  relative to the reference's own peak; 5 % energy; non-zero count) are
  untouched.
* `tests/unit/test_niche_audit_w6_asymptotic.py` — the two brute-force
  quadrature oracles (`_quad_oracle`, `_a9_quad`) carried the pre-Y2 weight;
  `test_w6_a7_output_field_carries_no_radiometric_normalisation` pinned the
  ABSENCE of the normalisation and is inverted; the W6-A4 `|a3| + |a4| < 1e-8`
  premise is restated.
* `tests/unit/test_niche_audit_eh1_maslov_upsample.py` — the E-L9 "no dead
  args" row for `_integrate_levin` listed `v2x_h` / `v2y_h`, which S4 needs
  back.  The row is narrowed to `('mi',)` and a COUNTER-PIN
  (`test_s4_levin_reads_the_v2_half_widths_it_takes`) asserts they are both
  taken AND read, so the dead-argument contract is not merely re-broken.
* `tests/unit/test_v5_21_maslov_jax_caustic.py::test_maslov_integration_method_auto_matches_and_is_fast`
  — asserted `'auto'` is byte-identical to `quadrature` OR
  `local_quadrature`; extended to the three concrete methods (the asymptotic
  arm is now `stationary_phase`), with a diagnostic message naming the
  per-method delta.  The byte-identity requirement itself is unchanged.
* `tests/unit/test_audit_optimize.py` and
  `tests/unit/test_niche_audit_r_guards_and_merits.py` — the two LG-merit
  non-vacuity bands re-pinned to the corrected `|L₀₀|²` with the
  `1/(λ²·|det J|)` derivation and both the old and new measured values
  (coordinator request; these live in optimize test files, noted in §5).  The
  R-5 NumPy/JAX parity tolerance was restated as RELATIVE (`rel=1e-5`): it used
  to be `rel=1e-6, abs=1e-6` on a quantity of order 1, where the `abs` term did
  the work; the underlying backend disagreement is unchanged at ~1.5e-06
  relative, and the twins themselves agree to **3.5e-15** when handed the same
  `v_star` (measured), so the residual is the two merit WRAPPERS' independent
  Newton solves, not the twin.

**Docs.** `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A4_REPORT.md`
(this file) and `WP-A4_CHANGELOG.md`.

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1`, `python -m pytest … -q --no-header
-p no:cacheprovider`.

| command | result | duration |
|---|---|---|
| `tests/unit/test_audit2609_a4_maslov_gbd.py` | **34 passed** | 26 s |
| `tests/unit/test_audit2609_a4_asymptotic.py` | **7 passed** | 25 s |
| `tests/unit/test_niche_audit_w6_asymptotic.py` + both new files | **82 passed** | 214 s |
| `tests/unit/test_v5_21_gbd_asm_interop.py tests/unit/test_lens_gbd.py` | **12 passed** | 206 s |
| `tests/unit/test_audit_w5_propagators.py tests/unit/test_audit_w6_propagators.py` | **38 passed, 1 skipped** | 4 s |
| `tests/unit/test_audit_propagation.py` (whole file) | **101 passed** | 16 s |
| `tests/unit/test_niche_audit_r_guards_and_merits.py tests/unit/test_audit_optimize.py` | **109 passed** | 45 s |
| `tests/unit/test_v4_16_1_dispatcher_pin_cache_registry_enrollment.py tests/unit/test_v4_16_0_agent_d_cache_registry.py` | **18 passed** | 11 s |
| `test_gbd_feature_complete, test_hammer_h7_gbd_diverging, test_niche_p1_gbd_chain, test_niche_p4_gbd_reexpand, test_niche_r3_gbd_mem_lstsq, test_niche_r5_gbd_vector_catastrophe, test_v5_21_gbd_windowed_adaptive` | **100 passed, 3 failed** (all 3 NOT mine — see below) | 1076 s |
| `test_niche_audit_eh1_maslov_upsample, test_v5_21_maslov_jax_caustic, test_v5_21_gbd_maslov_perf, test_fga, test_fga_h4_h5, test_fga_prefactor_dedup` | **89 passed, 2 failed** → both were mine and are fixed (see "test files modified"); FGA untouched and green | 1556 s |
| `test_audit2609_a4_maslov_gbd, test_audit2609_a4_asymptotic, test_niche_audit_eh1_maslov_upsample, test_v5_21_maslov_jax_caustic` (after the two test fixes AND the `_solve_fit` conditioning gate) | **70 passed** | 486 s |
| `tests/unit/test_audit_except_budget.py` | **2 failed — PRE-EXISTING**: the non-UI `except Exception:` count is 51 at HEAD against a budget of 48, i.e. the pin was already red on the committed tree.  My own two new sites were NARROWED (the file's own instruction) so my net contribution is **0**; the one new site since HEAD is in `elements/_lens_imap.py` (WP-A3) | 1 s |
| `python -m ruff check` on every file I touched | clean (`I001` introduced by my import additions, fixed; baseline at HEAD was also clean) | — |
| all four integrators × complex64/complex128 smoke | dtype contract preserved, all finite | 30 s |
| `tests/unit/test_audit2609_a4_fga_s10.py` | **16 passed** | 6 s |
| `test_fga, test_fga_h4_h5, test_fga_prefactor_dedup, test_g1_gate_generality, test_niche_audit_w9_dispatch2, test_niche_d3_guards, test_niche_p7_seidel_gate` (every file that exercises the FGA dispatch) | **205 passed** | 879 s |
| `test_fga.py -k "caustic or universal or auto or route"` (re-run after the helper rename) | **9 passed** | 112 s |

**Pre-existing / other-WP failures found, with my judgement.**

1. `tests/unit/test_niche_r3_gbd_mem_lstsq.py::test_traced_field_matches_lstsq_reference[kw0/1/2]` —
   `TypeError: _lstsq_solve() got an unexpected keyword argument 'score_domain'`
   raised from `lumenairy/elements/_lens_traced.py:3360`.  The test
   monkeypatches `_lstsq_solve`; WP-A3's concurrent edit added a `score_domain=`
   kwarg to the call site.  **Not mine** (`_lens_traced.py` is WP-A3's) and
   unrelated to GBD: the failing assertion never reaches any GBD code.
2. `tests/unit/test_v4_14_1_dispatcher_pin_cache_clears.py::test_submodule_clear_helper_reexported_at_top_level[lumenairy.analysis-clear_meshgrid_cache]` —
   `lumenairy.analysis.__all__` exports `clear_meshgrid_cache` but
   `lumenairy/__init__.py` does not re-export it.  **Not mine**
   (`lumenairy/analysis/` is the ANALYSIS WP's, and `lumenairy/__init__.py` is
   off-limits to me).  My own new clearer is deliberately NOT in
   `lenses_maslov.__all__` for exactly this reason — see §5.
3. `tests/unit/test_audit_except_budget.py` — see the table: the budget was
   already exceeded at HEAD (51 vs 48).  I added two `except Exception:` sites
   in `_lens_jax.py` and then NARROWED both to
   `(AttributeError, KeyError, TypeError, ValueError)`, which is what that
   file's own comment instructs ("If you add a non-tracer-guard site, NARROW
   it instead of bumping"); a per-file diff against HEAD confirms my net
   contribution is zero.  The one genuinely new site is in
   `elements/_lens_imap.py` (WP-A3).
4. Mid-run, `lumenairy/raytrace/intersection.py:670` raised
   `NameError: name '_kill_grazing' is not defined` for about ten minutes while
   a concurrent rename (`_kill_grazing` → `_kill_unreachable`) was half-applied.
   It resolved on its own; no action needed, recorded because it briefly made
   every ray-traced propagator unusable.

**Validation suite.**  Not run: `validation/run_all.py test_asymptotic`
/ `test_gbd` / `test_lenses` are the topic files for my area, and each is a
multi-hour job on this box (the unit-level GBD batch alone took 18 minutes).
COMMON.md §5 asks for economy on a shared machine; the unit coverage above plus
the 38 new pins exercise every changed line.  **Recommended for the
orchestrator's integration pass**: `python validation/run_all.py test_asymptotic
test_gbd test_lenses`.

---

## 5. Requested changes outside my ownership

1. **`lumenairy/optimize/core.py` / `optimize/jax_merits.py` — `LGAberrationMerit`
   and `make_lg_aberration_merit_jax` are not Strehl deficits.**
   `jax_merits.py:471` computes `piston_weight * (1 - |L|**2)` and documents it
   as "a Strehl deficit that → 1 for a perfect system".  It is not, in either
   the pre- or post-Y2 scale (measurements in §2, Y2).  The concrete fix, in
   order of preference:
   * make the merit scale-free — `1 − |L / L_ref|²` with `L_ref` the same
     coefficient evaluated on the aberration-free reference (same `w_s`, `w_p`,
     `s2_image`, `|det J|`), which is what "Strehl" means and is invariant under
     any normalisation convention; or
   * route the merit through a multi-mode `aberration_tensor` request (the
     σ-grid branch returns mutually consistent overlaps), and normalise by
     `Σ_k |L_k|²`.
   Until then the merit is still MONOTONE in aberration and still scales
   linearly with its weight (both pinned), so an optimiser converges to the same
   design; only the printed value is meaningless.  I did not touch `optimize/`
   as instructed — but I did re-pin the two LG-merit tests that live in
   `tests/unit/test_audit_optimize.py` and
   `tests/unit/test_niche_audit_r_guards_and_merits.py`, at the coordinator's
   explicit request, with the full derivation in the comments.
2. **`lumenairy/__init__.py` — re-export `clear_maslov_local_window_cache`.**
   The new cache is enrolled with `_cache_registry` (so `clear_asm_caches()`
   drains it), but I deliberately kept the clearer out of
   `lenses_maslov.__all__`: `test_v4_14_1_dispatcher_pin_cache_clears` requires
   every submodule-`__all__` `clear_*` name to be re-exported at top level, and
   `lumenairy/__init__.py` is off-limits to me.  Whoever owns that file should
   add `clear_maslov_local_window_cache` to the `lenses`/`elements` import block
   and `__all__`, and add the name to `lenses_maslov.__all__` in the same
   commit.
3. **`lumenairy/elements/lenses.py:928-941` — `_fit_normaliser` docstring is off
   by the pad convention** (audit P3): `half = 0.5*(vmax−vmin)*(1+pad)` gives
   `(v−c)/half ∈ [−1/(1+pad), 1/(1+pad)] = [−0.95238, 0.95238]` for
   `pad = 0.05`, not the documented `[−0.95, 0.95]`.  One-line docstring fix in
   WP-A2's file.
4. **`lumenairy/_math/chebyshev.py:333-345` — `chebyshev_fit_2d` coefficients do
   not round-trip for an off-centre grid** (audit P3): `normalize_xy=True`
   includes an OFFSET that the returned `{(i,j): c}` dict does not record, while
   the docstring claims the dict round-trips through `surface_sag_chebyshev`
   (which evaluates `T_i(x/a) T_j(y/b)`).  True only for a grid symmetric about
   0.  Either return the (centre, half-width) with the dict or drop the claim.
5. **`lumenairy/elements/_lens_traced.py:3360`** — see failure 1 in §4; the
   `score_domain=` kwarg broke a monkeypatching test in WP-A3's own area.
6. **`lumenairy/elements/lenses_maslov.py` — `apply_real_lens_maslov_vector`,
   the third S10 sub-item.**  The file left my ownership when the package was
   committed, so the change is specified here rather than made.  Three
   independent defects, in the order I would fix them:
   * **the Fresnel Jones is evaluated at AXIAL incidence for every pixel**
     (`_fresnel_jones_matrix_per_beamlet(xb, yb, zc, zc, ...)` with
     `zc = np.zeros_like(xb)`, i.e. `ux = uy = 0`).  The chart already carries
     the traced per-surface direction cosines; pass them instead of zeros, so
     `t_s` / `t_p` and the diattenuation are evaluated at the real angle of
     incidence.  This is the one that matters for the diverging / converging /
     tilted inputs the wrapper is offered for;
   * **`normalize_output` is threaded through with its `'power'` default and
     applied INDEPENDENTLY to `E_x` and `E_y`**, so the output polarization
     ratio is forced back to the post-Fresnel input ratio.  Minimum fix:
     refuse `normalize_output` in `('power', 'peak')` on the vector wrapper
     with a Section 2 message; better fix: one joint scale, exactly as
     `apply_real_lens_fga_vector` already does — measured there at 1 ulp
     (§2, S10 vector);
   * **no parallel transport / Richards–Wolf rotation of the Jones vector into
     the exit-ray frame, and no `E_z`.**  GBD ships
     `reconstruct_vector_field_with_ez` and FGA ships
     `return_longitudinal=True`; the Maslov wrapper needs the analogue, or its
     docstring must stop motivating it as a "polarization-resolved study
     through a focus".
   Effort: ~1–2 days for all three; the second alone is ~1 hour and stops the
   silent-wrong output today.

---

## 6. Deferred, with designs

1. **S6 proper — fit the input's local wavevector.  DEFERRED to a later
   follow-up; not mine now** (`lenses_maslov.py` left my ownership when
   `32ba3ba2` landed).  What ships today is the warning, which stops the
   silent-wrong output but does not make the two asymptotic evaluators correct
   for a non-collimated input.

   *The defect.*  `_maslov_newton_saddle_cpu` (and its GPU twin) solve
   `grad_v2 OPD = 0`.  The symplectic identity `dOPD/dv2 = −n1 (v1 · ds1/dv2)`
   — which the auditor measured closing to 5.8e-7 relative on a real singlet
   chart — makes that the `v1 = 0` launch ray at EVERY pixel, i.e. the on-axis
   collimated ray, for every input.  The stationary point of the TOTAL
   integrand phase is `grad_v2[arg E_in(s1(v2)) + k·OPD] = 0`, i.e.
   `(v1_in − v1) · ds1/dv2 = 0`, which selects the ray whose LAUNCH direction
   matches the input field's local wavevector.  Measured: at the 2 % smallest
   `|grad_v2 OPD|` the traced rays have mean `|v1| = 6.93e-03` on a chart of
   NA 0.05 whose all-ray mean is 3.79e-02.

   *The design.*  Fit the input's local wavevector `(k1x, k1y)(s1)` as two more
   columns of the SAME Chebyshev design matrix the chart already builds: the
   trace carries `v1x, v1y` per ray, `_solve_fit` already takes a stacked RHS
   (it solves OPD, s1x, s1y together), so the marginal cost is one wider RHS
   and no extra factorisation.  Then add `k1 · ds1/dv2` to the Newton gradient
   and `d(k1 · ds1/dv2)/dv2` to the Hessian in `_maslov_newton_saddle_cpu` /
   `_maslov_newton_saddle_xp`, and add the same term to `opd_star` in
   `_integrate_stationary_phase` and to `opd_v` in
   `_integrate_local_quadrature` (the two sites that already carry `lin_v3` /
   `lin_v4`, so the threading pattern exists).  `E_in`'s own amplitude stays
   where it is — only its PHASE joins the exponent.

   *How to verify.*  `'quadrature'` integrates the true integrand and is
   unaffected by the saddle, so it is the oracle: on a diverging (w0 = 12 µm)
   and a tilted (0.03 rad) input through the f = 6 mm singlet, at a plane a few
   depths of focus past the exit vertex, `'stationary_phase'` must come within
   the truncation error of a converged `'quadrature'` run.  Today it does not,
   and the new warning says so.  The regression test should assert the warning
   DISAPPEARS once the saddle is correct, so the two cannot drift apart.

   *Effort.* ~1 day including the oracle.
2. **S10 — DONE** in the follow-up pass; see §2.  The vector half
   (`apply_real_lens_maslov_vector`) is specified in §5 item 6, since that file
   left my ownership.
3. **The FGA-vs-phase_screen accuracy question at NA 0.145, still open.**  Not a
   defect I can close, but S10's measurement put a number on the auditor's own
   unverified suspicion: at the focus of an f = 1.2 mm singlet the two members
   disagree by a factor **3.9 in intensity-rms spot width** (12.72 µm for `fga`
   at its default sampling vs 3.27 µm for `phase_screen`, against a 3.44 µm
   diffraction limit and a 0.645 µm geometric fan), at tilt 0 as much as at
   tilt 0.05.  The dispatcher currently prefers `fga` in this regime.  Next step
   is a convergence sweep of FGA's `w0_factor` / `dq_step` / `p_max` against a
   converged Rayleigh–Sommerfeld or `apply_real_lens_traced` reference at this
   NA; if FGA converges to `phase_screen`, the default sampling is the bug and
   the routing is right; if it does not, `na_threshold` is mis-set.
4. **Y4 performance.** (a) one fused `_basis_and_grad34` per evaluation
   contracted against `np.stack([coef_s1x, coef_s1y, coef_phi])` — the auditor
   measured **2.3–5.1×** on the dominant 79 % of runtime; (b) hoist the
   loop-invariant `T1`, `T2`, `T12` out of `_solve_envelope_stationary_batch`'s
   Newton loop — a further **1.4×**; (c) scale-relative Newton tolerance
   (`tol * max(r0, 1)`, matching the verdict test) so the loop stops instead of
   always running all 12 iterations.  All three are pure refactors of code I
   touched; I left them out because each needs a bit-identity demonstration
   against the current output and the pass was already long.
5. **Y4 — `aberration_tensor` default cost.** Build only the requested `(p, ℓ)`
   pairs in `decompose_lg` (21 built / 11 used), cache
   `_measure_image_plane_waist` per `(fit, s2_image, w_s, w_p, v2_centre)`, and
   raise the `sigma_grid_n` cap once (4) makes 512 affordable — the default
   currently pays 12× the `n = 64` cost AND still returns an aliased answer
   (its own warning asks for 494, the cap truncates to 256).
6. **Y5 structural.** Move `eval_phi_xp` / `eval_s1_xp` onto the dataclasses
   instead of import-time monkey-patching; collapse the triplicated
   `_compute_M_b` / Newton / polynomial-substitution kernels.  The Y1 defect is
   precisely a fix that landed in one copy of an algorithm and not the others,
   so this is the highest-leverage item in the partition — and the most
   invasive.  Effort: ~2–3 days with a bit-identity harness.
7. **§15.9 — the uniform asymptotics.** `uniform_fold_airy` and `pearcey`
   (`lenses_maslov.py:986-1084`) remain DEAD CODE and I did not wire them in.
   Wiring them means pairing coalescing saddles through the Chester–Friedman–
   Ursell cubic map, which needs a second complex Newton root per pixel and,
   critically, a brute-force quadrature at a genuine fold to gate it against.
   The auditor's own Probe 3 (fold caustic vs a converged Rayleigh–Sommerfeld
   oracle) was not run for the same reason.  **I could not validate the Pearcey
   path in this pass, so I left it dead** — which is the option the WP file
   explicitly allows.  Design when it is picked up: locate the two saddles of
   the full exponent, form `ζ = (¾(S₂ − S₁))^{2/3}`, evaluate `Ai(−k^{2/3}ζ)` /
   `Ai′`, and gate the whole path behind `integration_method='uniform'` against
   a brute-force RS integral through the marginal focus of an f/2 singlet.
8. **S9 — GBD `_reconstruct_fft` kernel clipping.** Clip the kernel to
   `±ceil(R_cut/dx)` (the bound `_reconstruct_windowed` already computes) and
   pad to `scipy.fft.next_fast_len`.  Measured by the auditor: the FFT peak is
   36× the output-grid bytes and scales as N² with no cap (~9.7 GB at N = 4096).
   Pure performance, no correctness exposure; left out of a pass that changed
   this file's physics, to keep the diff reviewable.

---

## 7. Changelog

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A4_CHANGELOG.md`

---

## Addendum after WP-B7 (orchestrator, 2026-09-13)

* **Section 6 item 7 is corrected.**  `uniform_fold_airy` and `pearcey` are NOT dead code: `lumenairy/elements/_lens_traced_uniform.py` imports
  `_fold_airy_eval` and `pearcey` and uses them for `apply_real_lens_traced(caustic='uniform')`, covered by `test_niche_k4_uniform_caustic.py` and
  `test_niche_r2_pearcey_cusp.py`.  What is unwired is a uniform path for the Maslov `v2` integral; WP-B7 section 6 measured, against a brute-force
  Rayleigh-Sommerfeld oracle converged to 4.6e-8 at the marginal focus of an f/1.92 singlet, that the saddle costs 0.036 of fidelity against the
  exact integrator on the same chart, and left the Maslov path as it is with that table published.

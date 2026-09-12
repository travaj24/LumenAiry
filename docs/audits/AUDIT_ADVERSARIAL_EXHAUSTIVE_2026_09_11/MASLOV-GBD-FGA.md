# MASLOV-GBD-FGA audit — Maslov / GBD / FGA lens propagators + `_math` primitives

All repro scripts live in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/MASLOV-GBD-FGA/`
(`p1_gbd_gouy.py` … `p20_auto.py`). No repository file was created, modified or deleted.

## Scope read (files + line ranges actually read; what I did NOT get to)

Read line-by-line:
* `lumenairy/_math/chebyshev.py` — 1-404 (whole file).
* `lumenairy/_math/levin.py` — 1-356 (whole file).
* `lumenairy/elements/lenses_gbd.py` — 1-606 (whole file).
* `lumenairy/elements/lenses.py` — 820-941 (`_multi_indices_total_degree`,
  `_evaluate_polynomial_4d(_and_grad34)`, `_fit_normaliser`).
* `lumenairy/elements/lenses_maslov.py` — 1-125, 742-1087 (Newton saddle, Tukey,
  `_solve_fit`, `_select_poly_order_auto`, `uniform_fold_airy`, `pearcey`),
  1087-2460 (the whole `apply_real_lens_maslov` driver + `apply_real_lens_maslov_vector`),
  2461-2680 (`_integrate_quadrature`), 2836-3060 (`_integrate_stationary_phase`,
  `_warn_levin_over_tolerance`), 3412-3540 (`_integrate_local_quadrature`).
* `lumenairy/propagators/gbd.py` — 1-1050 (module header, `BeamletBundle`,
  2x2 helpers, `decompose_field_to_beamlets`, `recommend/converge_gbd_sampling`,
  `decompose_field_adaptive`, `propagate_beamlets_freespace`,
  `apply_thin_lens_to_beamlets`), 1047-1830 (`apply_aperture_to_beamlets`,
  `reconstruct_field_from_beamlets`, `_reconstruct_windowed`, `_reconstruct_fft`),
  1940-2062 (`gbd_asm_gouy_phase` / converters / `match_global_phase`),
  3294-3550 (`apply_prescription_persurface_to_beamlets`, world reframe).
* `lumenairy/propagators/fga.py` — 1-100, 384-620 (`_hk_prefactor`, swarm lattice,
  adaptive momentum nodes, Gabor analysis), 1119-1520 (`_fga_coarse`,
  `_fga_through_lens`), 1510-1640 (`apply_real_lens_fga` signature/doc),
  2226-2380 (`_caustic_zone`, `apply_real_lens_auto`), 2745-3078
  (`_universal_route`, `apply_real_lens_universal`).
* Cross-checked dependencies: `elements/_lens_traced.py:6677-6690, 9581-9590`
  (vertex-plane correction), `propagators/asm.py:583-594` (signature),
  `docs/audits/AUDIT_MASLOV_2026_07_09.md` (whole), `tests/unit/test_v5_21_gbd_asm_interop.py`,
  `tests/unit/test_lens_gbd.py`.

NOT read / not reached: the CuPy twins beyond their signatures
(`_integrate_quadrature_cupy`, `_integrate_stationary_phase_cupy`,
`_integrate_local_quadrature_cupy`, `_fga_*_cupy`) — desk-check only, no GPU
available; the numba kernel bodies in `lenses_maslov.py:125-740` and
`fga.py:110-372` (I verified the NumPy references they claim to mirror);
`_integrate_levin`'s batched quadtree (`lenses_maslov.py:2996-3412`);
`gbd.py:2063-3294` (spectral / vector / CSP / ghost / thin-film blocks);
`fga.py:1765-2226` (`fga_memory_estimate`, `_fga_vector_through_lens`,
`apply_real_lens_fga_vector`) and `2378-2745` (`_tilt_dispersion`,
`_sag_screen_aberration_rad`, `_seidel_sa_wfe_rad`) beyond their routing role.
Probe 3 (fold-caustic brute-force Rayleigh–Sommerfeld comparison) was not run —
see *Unverified suspicions*.

---

## Findings

### [P0] `local_quadrature` — the integrator `integration_method='auto'` (the DEFAULT) selects for any real focusing chart — mis-assigns its sampling window and never converges
`lumenairy/elements/lenses_maslov.py:3455-3521` (and the CuPy twin
`_integrate_local_quadrature_cupy`, same algebra).

Two independent defects:

**(a) The window is scaled by the Hessian EIGENVALUES but laid out on the
COORDINATE axes.** Lines 3458-3467 compute `lam1 = tau/2 + sqrt(disc)` (always
the *larger* eigenvalue), `lam2 = tau/2 - sqrt(disc)`, then
`sigma1_norm = 1/sqrt(pi*|lam1|)/v2x_h` is applied to `u_v2x` and `sigma2_norm`
to `u_v2y` (3478-3481). There is no rotation into the eigenbasis, and the two
widths are tied to the eigenvalue *ordering*, not to the axes. For a diagonal
Hessian with `H44 > H33` the widths are exactly swapped: the narrow-curvature
axis gets the narrow window and vice versa.

**(b) No window function.** The local samples are a hard-truncated uniform
Riemann sum (`contrib.sum(axis=1) * w2d_phys`, 3516-3521). Truncating
`int exp(i*pi*A*u^2) du` at a finite range leaves the Fresnel endpoint
oscillation, an `O(1/window_sigma)` error that does **not** decrease with
`local_n_samples`.

**Evidence** (`p6_localquad_axes.py`, `p7_localquad_defaults.py`): synthetic
chart `OPD = 0.5*A*u3^2 + 0.5*B*u4^2 + C*u3*u4`, `s1 = u3/u4`, `E_in == 1`, whose
exact value is `exp(i*pi*sigma/4)/sqrt|det H|`. `_integrate_stationary_phase` on
the same charts is exact (relerr `0.00e+00`), so the fit and chart are not the
issue.

| A | B | H34 | relerr at the DEFAULTS (`n=8`, `ws=3.0`) |
|---|---|---|---|
| 40 | 40 | 0 | **1.19** |
| 40 | 4 | 0 | **1.19** |
| 4 | 40 | 0 | **9.12**  ← the eigenvalue/axis swap |
| 100 | 10 | 0 | 1.19 |
| 40 | 40 | 30 | 2.48 |
| 60 | 20 | 25 | 3.07 |

Convergence sweep (A=B=40, isotropic — defect (b) alone):

```
window_sigma= 3.0: n=8:1.2e+00  n=16:4.7e-01 n=32:4.1e-01 n=64:4.0e-01 n=128:4.0e-01
window_sigma= 4.0: n=8:6.1e+00  n=16:6.1e-01 n=32:3.2e-01 n=64:2.9e-01 n=128:2.9e-01
window_sigma= 6.0: n=8:2.4e+01  n=16:5.7e+00 n=32:6.1e-01 n=64:2.2e-01 n=128:1.9e-01
window_sigma=10.0: n=8:4.6e+01  n=16:2.4e+01 n=32:1.4e+01 n=64:3.7e+00 n=128:1.8e-01
window_sigma=20.0: 7.5e+01 … 1.2e+02     <- np.clip(u_v2,-1,1) over-counts (see [P2] below)
```

i.e. a 40 % floor at the default `window_sigma=3` no matter how many samples.

**This is a default path.** `p20_auto.py`, `apply_real_lens_maslov(...,
verbose=True)` on an f = 6 mm N-BK7 biconvex, 0.15 mm aperture, collimated
input, `poly_order=4` prints:

```
maslov  integrate   59.5%  auto -> local_quadrature (need n_v2~19433)
```

so the shipped default (`integration_method='auto'`) routes an ordinary
focusing singlet straight into this integrator.

**Caveat, stated honestly:** my synthetic chart has a constant integrand
amplitude. In a real chart `E_in(s1(v2))` decays across the window, which
partially suppresses the endpoint error (how much depends on
`|ds1/dv2| * window_sigma * sigma` vs the beam radius). Defect (a) — the
eigenvalue/axis swap — is unconditional.

**Fix.** (i) Diagonalise the 2x2 Hessian and sample on the rotated lattice
(`R @ [sigma1*Xlin, sigma2*Ylin]`), so the box follows the true principal axes;
(ii) multiply the local samples by a smooth window (Gaussian
`exp(-xi^2/(2*ws^2))` or the same Tukey the uniform path uses) and divide by its
discrete integral, or replace the real-axis window with a steepest-descent
(complex-`v2`) contour, which converges geometrically; (iii) renormalise
`w2d_phys` when `np.clip` bites.

---

### [P1] Maslov integrand uses `|det(ds1/dv2)|` where the Van Vleck amplitude is its SQUARE ROOT, and the `k/(2*pi*i)` prefactor is missing
`lenses_maslov.py:2621-2623` (quadrature), `2921-2944` (stationary_phase),
`3511-3521` (local_quadrature), plus the three CuPy twins and `_integrate_levin`.

The semiclassical kernel is
`K = (k/(2 pi i))^{d/2} |det d^2S/ds1 ds2|^{1/2} e^{ikS}`; substituting
`d^2 s1 = |det(ds1/dv2)| d^2 v2` and
`|det d^2S/ds1 ds2|^{1/2} = |det(ds1/dv2)|^{-1/2}` gives, for d = 2,

```
E(s2) = (k / (2 pi i)) * INT E_in(s1(s2,v2)) * |det(ds1/dv2)|^{1/2} * e^{ikS} d^2v2
```

The code computes `abs_J = |det_J|/(v2x_h*v2y_h)` — the **full** Jacobian — and
applies no prefactor. (`apply_real_lens_maslov`'s own docstring, line 1155,
states the integrand as `|det(ds1/dv2)|`, and line 1253 asserts "the Maslov
integral itself carries an arbitrary overall prefactor". It does not: the
prefactor is `k/(2 pi i)`, and the Jacobian power is not a prefactor at all.)

**Evidence** (`p12_prefactor.py`): a free-space chart (two flat air surfaces,
thickness z) where `ds1/dv2 = -z I` exactly, so `J = z^2` and
`sqrt(J) = z`. If the amplitude were `sqrt(J)` the ratio to the exact ASM field
would be `i*lambda` (z-independent); if it is `J`, the ratio is `i*lambda*z`.
Measured (N=64, dx=4 um, Gaussian w0=30 um, `normalize_output='none'`,
`integration_method='quadrature'`, n_v2=64):

```
lam=1.0um z=0.50mm : |ratio|=4.998e-10  arg=+1.5716  ratio/(lam*z)=0.99960  ratio/lam=5.0e-04
lam=1.0um z=1.00mm : |ratio|=9.996e-10  arg=+1.5720  ratio/(lam*z)=0.99957  ratio/lam=1.0e-03
lam=1.0um z=2.00mm : |ratio|=1.996e-09  arg=+1.5693  ratio/(lam*z)=0.99804  ratio/lam=2.0e-03
lam=2.0um z=0.50mm : |ratio|=9.996e-10  arg=+1.5747  ratio/(lam*z)=0.99965
lam=2.0um z=1.00mm : |ratio|=1.996e-09  arg=+1.5706  ratio/(lam*z)=0.99801
lam=2.0um z=2.00mm : |ratio|=3.988e-09  arg=+1.5693  ratio/(lam*z)=0.99697
```

`ratio/(lambda*z) = 1.000` across two wavelengths and three distances, and
`arg = +1.571 = +pi/2`. So `E_maslov = i * lambda * z * E_exact` exactly — the
z-dependence *proves* the wrong Jacobian power, and the `pi/2` proves the
missing `1/i`.

**Impact.** With the default `normalize_output='power'` only the constant part
is removed; the residual is the spatial/angular variation of `sqrt(J)`, which
re-weights the angular spectrum *inside* the integral. Measured on benign
charts this is small — 0.44 % / 0.42 % `|E|`-profile spread for a converging
input through free space (`p13_amp_profile.py`), 0.30 % at NA 0.13
(`p15_highNA.py`, where the exact free-space `J = z^2/N^4` makes the error the
`1/N^2` apodisation) — but it is unbounded on an aberrated chart where `J`
varies with pupil zone, and `roi=` runs are *forced* to `normalize_output='none'`
(`lenses_maslov.py:1910-1924`) and therefore return an absolutely wrong scale.
This is why I rate it P1 rather than P0.

**Fix.** `abs_J_c = np.sqrt(np.abs(det_J_c) / (v2x_h * v2y_h))` at all three (six
with CuPy) sites, and multiply the assembled field once by
`1.0 / (1j * wavelength)` before Step 6. Then `normalize_output='none'` returns
a physically-scaled field and `'power'` becomes a diagnostic rather than a
crutch. The `AUDIT_MASLOV_2026_07_09.md` coverage statement explicitly lists
"the interior quadrature-node bookkeeping of each integrator (the
oscillatory-integral weight assembly)" as **not line-verified**, so its
"The Maslov physics is correct" verdict does not cover this.

---

### [P1] GBD beamlet amplitude carries the Gouy / Collins phase with the WRONG SIGN (conjugated); the library has documented the error as an inter-propagator "convention"
`propagators/gbd.py:979-981` (scalar free space, `qratio = Q_new / Q_old`),
`:976-977` and `:385-391` (tensor, `prod_i 1/sqrt(1 + t*lam_i)`), `:3193`
(`apply_abcd_to_beamlets`), `:3415` (`amp = amp / np.sqrt(_det2x2(ABQ))` per
surface).

`BeamletBundle.Q` is documented (`:260-268`) as the *engineering* `1/q`
(`q_code = conj(q_physics)`) and the renderer correctly converts on output:
`exp(+0.5j k conj(Q) rho^2)` (`:1416-1420`, `:1397-1401`). The amplitude
factor does **not** convert: the correct physics amplitude is
`q0_phys/q_phys = conj(Q_new/Q_old)`, but the code uses `Q_new/Q_old`.

**Evidence 1** (`p1_gbd_gouy.py`) — one beamlet, free space, versus the analytic
Gaussian under exp(-i w t) (`q = z - i zR`, `E = (w0/w) e^{-r^2/w^2}
e^{i(kz + k r^2 z/(2(z^2+zR^2)) - psi)}`):

```
z= 0.50 zR: |E|/|A|=1.000000  arg(E/A)=+0.927295  2*psi=+0.927295  diff=+6.7e-14
z= 2.00 zR: |E|/|A|=1.000000  arg(E/A)=+2.214297  2*psi=+2.214297  diff=+9.9e-14
z=10.00 zR: |E|/|A|=1.000000  arg(E/A)=+2.942255  2*psi=+2.942255  diff=-1.8e-14
  (relative L2 after removing that scalar: 1.3e-13 / 5.3e-13 / 4.2e-12)
```

The transverse amplitude AND curvature phase are exact to 1e-12; the whole
discrepancy is `exp(+2*i*psi)` — exactly `(exp(-i psi))^{-2}`, i.e. the Gouy
phase applied with the opposite sign.

**Evidence 2** (`p2_gbd_gouy_full.py`) — full pipeline vs
`angular_spectrum_propagate`, N=256, lam=1 um, z=3 mm:

```
                       measured global phase   library formula 2*atan(z/zRb)   relL2 raw   after phase removal
waist_factor=1.0             +3.133220                 +3.133215                1.999        1.76e-03
waist_factor=2.0             +3.108090                 +3.108085                1.995        7.02e-03
waist_factor=3.0             +3.066235                 +3.066230                1.987        1.57e-02
-- with qratio -> conj(qratio) --
waist_factor=1.0        residual global phase = +4.71e-06   relL2 raw (NO phase fit) = 1.76e-03
waist_factor=2.0        residual global phase = +4.68e-06   relL2 raw (NO phase fit) = 7.02e-03
waist_factor=3.0        residual global phase = +4.62e-06   relL2 raw (NO phase fit) = 1.57e-02
```

Conjugating the ratio drops the residual global phase from ~pi to 4.7e-6 rad.

**Why this is an error, not a convention.** `gbd_asm_gouy_phase`
(`gbd.py:1954-1994`), `gbd_field_to_asm`, `asm_field_to_gbd` and
`tests/unit/test_v5_21_gbd_asm_interop.py` exist to remove
`phi0 = 2*arctan(z/zR_beamlet)`, and the docstring claims GBD "carries" a Gouy
phase that ASM "does not". A Gabor frame of exact Gaussian-beam solutions,
propagated exactly and summed, reproduces the ASM field with **no** residual
phase — free-space propagation is linear and each beamlet is an exact solution.
The residual depends on `waist_factor`, a purely numerical knob, which is the
definition of an error.

**Non-global consequence** (`p3_gbd_mixed.py`) — `decompose_field_adaptive`
(a public API, default arguments) mixes two beamlet waists (measured 3 um and
12 um), so the wrong Gouy is no longer a common factor:

```
z=  20 um  as-is   relL2 raw=1.78e-01  best-global-phase-removed=1.18e-01 | conj-fix 5.06e-02
z=  50 um  as-is   relL2 raw=3.14e-01  best-global-phase-removed=1.75e-01 | conj-fix 5.06e-02
z= 200 um  as-is   relL2 raw=8.41e-01  best-global-phase-removed=1.84e-01 | conj-fix 5.06e-02
z=1000 um  as-is   relL2 raw=1.82e+00  best-global-phase-removed=9.16e-02 | conj-fix 5.11e-02
```

i.e. 9-18 % residual error that *no* global phase fit can remove, versus 5.1 %
(the frame floor) with the fix.

At the lens EXIT plane the per-surface Collins version of the same error is
near-global and therefore nearly invisible: `p4_gbd_persurface.py` measured
`apply_real_lens_gbd` vs `traced`/`analytic` overlaps unchanged to 6 decimals
with and without conjugating `_det2x2`/`_eigvals2x2`. It will matter at
`output_plane_distance != 0` and in any coherent combination.

**Fix.** `qratio = xp.conj(Q_new / Q_old)`; tensor
`qratio = xp.conj(xp.prod(1/xp.sqrt(1 + t*lam), axis=1))` (both in
`propagate_beamlets_freespace` and `_freespace_tensor_moebius_np`);
`amp = amp / np.conj(np.sqrt(_det2x2(ABQ)))` in
`apply_prescription_persurface_to_beamlets` and `apply_abcd_to_beamlets`. Then
`gbd_asm_gouy_phase` should return 0 (or be deleted with its converters) and
`test_v5_21_gbd_asm_interop.py` inverted (`_relerr(Gz, Az) < 5e-3` with no
converter).

---

### [P1] `apply_real_lens_maslov` references its exit chart to the CURVED last surface, not the last-surface vertex plane
`lenses_maslov.py:1709-1746`. `tr = rt.trace(rays, surfaces, wavelength)` leaves
`exit_rays` on the last surface (`exit_rays.z == sag(r)`), and the code uses
`ex_x / ex_y / ex_opd` **directly** as the exit-plane chart. Every sibling
applies the correction: `_lens_traced.py:6679-6687` and `:9583-9589`
(`t_to_vertex = -final.z/final.N`, `opd += n_exit*t_to_vertex`), `_lens_jax.py`,
and `gbd.py:3443-3463` — whose v5.22 comment states *"Every OTHER consumer of the
ray-transfer OPL applies the same SIGNED (not abs) `-sag/N` vertex correction …
this path was the sole omission."* `apply_real_lens_maslov` is the remaining
omission.

**Evidence** (`p11_sag_ref.py`, f = 6 mm N-BK7 biconvex, 1.5 mm aperture,
lam = 1 um, 17 638 alive rays):

```
exit z: min=-4.29e-05  max=-3.54e-07  RMS=2.79e-05 m  -> 27.9 waves of path
vertex-plane correction: transverse shift RMS = 2.43 um, OPD shift RMS = 28.08 waves, PV = 42.70 waves
symplectic identity dOPD/ds2 = [n2 v2 - n1 v1.ds1/ds2]/lam   (PLANE reference) : rel residual 4.86e-01
symplectic identity with the curved-surface term + n2 N2 dsigma/ds2           : rel residual 5.71e-08
```

The identity closes to 5.7e-8 only with the curved-surface term.

**Self-consistency check with no third-party oracle** (`p16_dummyflat.py`,
plano-convex with `R2 = -2 mm`, 0.30 mm aperture): appending a
**zero-thickness FLAT dummy surface** — a physically null prescription edit —
moves the chart to the vertex plane and changes the answer:

```
original    : exit z = [0, -3.06e-07, -1.23e-06, -2.76e-06, -4.91e-06]  opd = [1055.25 …  1047.86] waves
+dummy flat : exit z = [0, 0, 0, 0, 0]                                   opd = [1055.25 …  1052.76] waves
```

The OPD difference at r = 0.14 mm is 1052.76 - 1047.86 = **4.90 waves**, exactly
`|sag| = 4.906 um` at lam = 1 um. The resulting Maslov *fields* differ by up to
~5 waves of radial phase (radial cut: measured +4.74 waves at r = 136 um vs the
predicted `-sag/lambda = +4.63` waves) and by 116 % in the `|E|` ratio spread.

**Impact.** The returned array is the field *on the curved exit surface*
indexed by transverse position, not the exit-plane field the docstring
promises; and it compounds with `output_plane_distance` (`:1730-1735`), whose
leg starts from `z = sag(r)`, so the requested observation plane is offset by
`sag(r)` — tens of microns for a fast lens, comparable to its depth of focus.

**Fix**, immediately after `exit_rays = tr.image_rays`:

```python
_t0 = np.where(np.abs(exit_rays.N) > 1e-30, -exit_rays.z / exit_rays.N, 0.0)
ex_x   = exit_rays.x   + _t0 * exit_rays.L
ex_y   = exit_rays.y   + _t0 * exit_rays.M
ex_opd = exit_rays.opd + n_exit * _t0        # n_exit = index after the last surface
```

then apply the existing `output_plane_distance` leg on top. Mirror in
`_lens_jax.apply_real_lens_maslov_jax`.

---

### [P1] The Maslov asymptotic evaluators solve for the saddle of the OPD ALONE, ignoring the input field's phase — silently valid only for a collimated input
`lenses_maslov.py:789-847` (`_maslov_newton_saddle_cpu`), `:742-786` (GPU twin),
consumed by `_integrate_stationary_phase` (`:2906`) and
`_integrate_local_quadrature` (`:3448`).

The stationary point of the *total* integrand phase is
`grad_v2[ arg E_in(s1(v2)) + k*OPD ] = 0`, i.e. `(v1_in - v1) . ds1/dv2 = 0` —
it selects the ray whose *launch* direction matches the input field's local
wavevector. Solving `grad_v2 OPD = 0` alone selects `v1 = 0` (because
`dOPD/dv2 = -n1 v1 . ds1/dv2`, verified below), i.e. the on-axis collimated ray,
for **every** pixel and every input.

**Evidence** (`p10_fit_symplectic.py`, same singlet, chart NA 0.05):

```
symplectic identity (1)  dOPD/dv2 = -n1 (v1 . ds1/dv2)/lam :
   v2x: rel residual 5.81e-07      v2y: rel residual 5.81e-07     <- identity confirmed
saddle of OPD alone: at the 2 % smallest |grad_v2 OPD| the traced rays have
   mean |v1| = 6.93e-03  (chart NA 0.05; all-ray mean |v1| = 3.79e-02)
```

So the saddle sits where `v1 -> 0`. The driver nevertheless sizes the pupil
chart specifically to cover a diverging / tilted input
(`na_proxy = na_lens + na_input`, `:1618-1667`, plus the `input_na` kwarg and
its coverage warning), and `collimated_input` only affects the chart size —
nothing warns that `stationary_phase` / `local_quadrature` (hence
`integration_method='auto'` at any realistic NA) then evaluate the wrong saddle.
There is no docstring caveat.

**Fix.** Either fit the input's local wavevector `(k1x, k1y)(s1)` alongside
`s1x/s1y` and add `k1 . ds1/dv2` to the Newton gradient/Hessian, or detect a
non-flat input (the driver already measures `_na_meas` from the angular
spectrum) and refuse / warn for the two asymptotic methods.

---

### [P2] `apply_real_lens_universal` routes a TILTED but perfectly collimated, single-valued high-NA plane at its focus to `phase_screen`
`propagators/fga.py:2799-2817` (`_universal_route`), `2226-2281`
(`_caustic_zone`).

Two compounding causes:

1. `_caustic_zone` traces a `+x` meridional half-fan and keeps only rays that
   cross the **axis** (`conv = alive & (xo*uo < 0)`, line 2274), then reports the
   5th–95th percentile of `z = -x/u`. With a global input tilt the focus is
   off-axis (`~ f * tilt`), so the axis crossings are meaningless and the zone
   is junk.
2. The final escape hatch (line 2815) `if spread > _NONCOLLIMATED_RESID_THRESH
   and not aberrated: return "phase_screen"` uses
   `_lens_traced._carrier_residual_rms(E_in, None, wavelength, dx)`, which
   returns **exactly the tilt magnitude** for a pure tilt.

**Evidence.** `p19_dispatch.py` (routing decisions only, no propagation),
f = 1.2 mm biconvex, 0.30 mm aperture, NA = 0.145, plane at the focus
(1.027 mm), `na_threshold=0.12`:

```
collimated / fast lens, AT FOCUS  : NA=0.1452 tiltdisp=0.0000 caustic=[1.021,1.033]mm -> universal='fga'          auto(2way)='fga'
tilted input (0.05 rad), same plane: NA=0.1452 tiltdisp=0.0007 caustic=[2.002,11.020]mm -> universal='phase_screen' auto(2way)='gbd'
decentred beam,          same plane: NA=0.1452 tiltdisp=0.0000 caustic=[1.021,1.033]mm -> universal='fga'          auto(2way)='fga'
multi-valued (2 tilted beams)      : NA=0.1452 tiltdisp=0.3257 caustic=[0.278,3.274]mm -> universal='fga'          auto(2way)='fga'
collimated / slow lens, exit plane : NA=0.0312 aberr=0.0 rad                            -> universal='phase_screen' auto(2way)='gbd'
```

and the discriminator itself:

```
_NONCOLLIMATED_RESID_THRESH = 0.02
tilt=0.000 rad -> carrier residual rms = 0.00000e+00
tilt=0.005 rad -> 5.00000e-03      tilt=0.020 rad -> 2.00000e-02  (== threshold)
tilt=0.050 rad -> 5.00000e-02  -> "NON-collimated"
tilt=0.100 rad -> 1.00000e-01  -> "NON-collimated"
curv R=10 mm   -> 1.21670e-02      curv R=3 mm    -> 4.05565e-02
```

A 0.05 rad (2.9 deg) tilt reads *more* "non-collimated" than an R = 3 mm
converging wavefront. The plane therefore goes to the thin phase-screen model
at NA = 0.145 > `na_threshold` = 0.12 and inside the caustic — precisely the
regime the dispatcher docstring says the phase screen cannot handle.
`apply_real_lens_auto` (the 2-way router) also drops to `'gbd'` for the same
`_caustic_zone` reason, i.e. the non-caustic member at a caustic plane.

**Fix.** Remove the intensity-weighted mean tilt from `E_in` (a global linear
phase) before both `_caustic_zone` and the collimation test, and score the
caustic on the CHIEF-ray crossing rather than the axis crossing (or on the
transverse ray-fan caustic, `det(dx_out/dx_in) = 0`).

### [P2] `local_quadrature` over-counts when `np.clip(u_v2_samp, -1, 1)` bites
`lenses_maslov.py:3482-3483` clips the sample coordinates into the fit box, but
`w2d_phys = sigma1_phys*sigma2_phys*dxi**2` (line 3493) still uses the
*unclipped* cell area, so every clipped sample is counted at full weight while
sitting on top of its neighbours. Measured (`p7`, A=B=40, `window_sigma`=20/40):
relative error 7.5e+1 … 2.0e+3. With the default `window_sigma=3.0` the clip
fires whenever `3/sqrt(pi*|lambda|) > v2_h`, i.e. on any weakly-curved
(near-caustic) chart — exactly where the method is reached for.

### [P2] The `n_v2` auto-resolution estimator and the under-resolution warning under-count the v2 oscillations by up to ~2.5x
`lenses_maslov.py:1944-1953` (`auto` dispatch), `1962-1971` (auto `n_v2`),
`2124-2139` (the warning). All three use
`v2_osc = sum |coef_opd[k]| over terms with k3>0 or k4>0`, justified as
"Chebyshev polynomials are bounded by 1 … the sum … upper-bounds the OPD
excursion in WAVES = cycles along v2". The *excursion* is not the oscillation
count: the number of cycles along `v2` is bounded by the TOTAL VARIATION, and
`TV(T_n)` on [-1,1] is `2n`, so the correct bound is
`sum |c_k| * max(k3,k4)`.

Measured (`p18_perf.py`, real fitted charts):

```
f=6mm biconvex 1.5mm ap  order=4: sum|c|= 285.6 -> n_v2=1143 | TV-bound  312.3 -> 1250  (1.09x)
                         order=6: sum|c|= 737.9 -> n_v2=2952 | TV-bound 1233.9 -> 4936  (1.67x)
                         order=8: sum|c|= 634.0 -> n_v2=2537 | TV-bound 1539.7 -> 6159  (2.43x)
f=2mm biconvex 0.3mm ap  order=8: sum|c|= 136.2 -> n_v2= 545 | TV-bound  337.1 -> 1349  (2.47x)
```

Since `poly_order='auto'` can select 8 (`_MZ_POLY_AUTO_MAX`), the auto `n_v2`
and the "want n_v2 >~ N" advice can be ~2.5x low, and the
`auto -> quadrature/local_quadrature` decision can pick `quadrature` for a chart
that actually needs more than the 256 cap.

### [P2] `apply_real_lens_maslov_vector` — polarization model is weaker than advertised in three specific ways
`lenses_maslov.py:2378-2447`.

* The Fresnel Jones is evaluated with `ux = uy = 0` for every pixel
  (`_fresnel_jones_matrix_per_beamlet(xb, yb, zc, zc, ...)`, line 2433, with
  `zc = np.zeros_like(xb)`), i.e. **axial incidence everywhere**. For the
  diverging / converging / tilted inputs this wrapper is offered for, the
  per-surface angle of incidence — and hence `t_s`/`t_p` and the diattenuation —
  is wrong.
* There is no parallel transport / Richards–Wolf rotation of the Jones vector
  from the entrance transverse frame into the exit-ray frame, and no `E_z`
  component. The docstring motivates the function as "polarization-resolved
  study through a focus"; the geometric polarization rotation and the
  longitudinal field that dominate a high-NA focus are both absent. (GBD ships
  `reconstruct_vector_field_with_ez`; the Maslov wrapper has no analogue.)
* `**maslov_kwargs` carries `normalize_output` through with its default
  `'power'`, applied **independently** to `E_x` and `E_y`
  (lines 2439-2444 -> `lenses_maslov.py:2342-2347`). Any differential
  transmission, vignetting or inbox-clipping between the two components is
  normalised away, so the output polarization ratio is forced back to the
  post-Fresnel input ratio.

### [P2] GBD `_reconstruct_fft` builds a full-size kernel regardless of beamlet width
`gbd.py:1774-1826`. The kernel is `(2*Ny-1) x (2*Nx-1)` complex128 and
`_fftconv_same` transforms at `(3*Ny-2) x (3*Nx-2)` — independent of the actual
Gaussian support `R_cut`, which `_reconstruct_windowed` already computes as
`n_sigma/sqrt(0.5*k*lambda_min(-Im Q))`. Non-power-of-2 sizes too (766, 1534).

Measured (`p18_perf.py`, `tracemalloc` peak, spread bundle after 1 mm of free
space):

```
N=256  fft-conv     :     253.3 ms  peak alloc  38.1 MB   (output grid 1.0 MB)  -> 38x
N=256  windowed     :  267847.8 ms  peak alloc 511.8 MB                          (1058x slower)
N=512  fft-conv     :    1560.1 ms  peak alloc 152.6 MB   (output grid 4.2 MB)  -> 36x
N=512  windowed     : 1324665.2 ms  peak alloc 520.8 MB                          ( 849x slower)
```

The FFT path is a genuine ~850-1060x win over the windowed scatter here
(corroborating the "~2000-3700x" claim in the code comment) and is *also*
cheaper in memory at these sizes (152.6 MB vs 520.8 MB at N=512, because the
windowed path is pinned at its `mem_budget_mb=512` tile). But the FFT peak is
~36x the output-grid bytes and scales as N^2 with no budget cap at all:
~9.7 GB at N = 4096 and ~155 GB at N = 16384, where the windowed path's
`mem_budget_mb` would have held it flat.
**Fix:** clip the kernel to `+-ceil(R_cut/dx)` (matching `window`) and pad to
`scipy.fft.next_fast_len`; for a localized bundle that is a (R_cut/(N dx))^2
reduction in both the kernel and the transform.

### [P3] `fold_split` free-space legs drop `dy`
`lenses_maslov.py:1369` and `:1375` call
`angular_spectrum_propagate(E, _din, wavelength, dx)` with no `dy`, so an
anamorphic grid silently uses `dy = dx` on the gaps around each fold, while the
rest of the function is anamorphic-aware (`:1446-1463`, `:1856-1857`,
`:1994`).

### [P3] The four Maslov integrators integrate DIFFERENT integrands
`quadrature` multiplies by a Tukey(alpha=0.2) window over the whole v2 box
(`lenses_maslov.py:1977-1983`, `2549`), whose integral is 1.8 rather than 2.0 —
a 10 % apodisation of the angular spectrum. `stationary_phase`,
`local_quadrature` and `levin` apply no window at all. Because
`integration_method='auto'` switches between them on a threshold, the returned
field changes discontinuously with the chart. (The Tukey *normalisation* itself
is fine: measured `sum(w)*du` = 1.80040642 / 1.80005564 / 1.79999324 / 1.80000000
at n_v2 = 32/64/128/256 vs the exact 1.8.)

### [P3] `chebyshev_fit_2d` returns coefficients that do not round-trip for an off-centre grid
`_math/chebyshev.py:333-345`: `normalize_xy=True` maps
`x -> 2*(x - x.min())/span - 1`, which includes an **offset** when `x` is not
symmetric about 0, but the returned `{(i,j): c}` dict carries no record of the
(centre, half-width). The docstring (lines 262-266, 300-304) claims the dict
"round-trips through the library's own evaluator" `surface_sag_chebyshev`, which
evaluates `T_i(x/a) T_j(y/b)` — true only for a grid symmetric about 0.

### [P3] `_fit_normaliser` docstring is off by the pad convention
`elements/lenses.py:928-941`: `half = 0.5*(vmax-vmin)*(1+pad)` gives
`(v-c)/half` in `[-1/(1+pad), 1/(1+pad)]` = `[-0.95238, 0.95238]` for
`pad=0.05`, not the documented `[-(1-pad), (1-pad)] = [-0.95, 0.95]`.

---

## Performance opportunities (estimated gain; how measured/estimated)

| # | Where | Gain | Basis |
|---|-------|------|-------|
| 1 | `gbd._reconstruct_fft` kernel clipped to `R_cut` + `next_fast_len` | up to `(N*dx/(2*R_cut))^2` on memory AND FLOPs; at N=512 the peak drops from 152.6 MB toward the ~4-10 MB grid scale | measured `tracemalloc` peaks 38.1 MB (N=256) / 152.6 MB (N=512) vs 1.0 / 4.2 MB grids (`p18_perf.py`) |
| 2 | `gbd.reconstruct_field_from_beamlets` default `window=None` | 1058x on a spread bundle | measured 253 ms (fft) vs 267 848 ms (windowed scatter) at N=256 (`p18_perf.py`). The internal callers already pass `window=5.0`; the *public* default is still the slow one, guarded only by a warning above `1e8` work units |
| 3 | `lenses_maslov._integrate_quadrature` 7 separate `_factor_contract` einsums per chunk (`:2604-2610`) | ~7 -> 1 einsum by stacking `Hh_*` on a leading axis; the contraction is the inner loop | code inspection: identical `Tyb`/`Tx_1d` operands, only the `Hh` tensor differs |
| 4 | `lenses_maslov._integrate_local_quadrature` `np.broadcast_to(...).ravel()` (`:3486-3487`, `3500-3501`) | removes a `(PX_CHUNK, n_s2)` float64 copy of a constant-along-axis array per chunk | code inspection; `_opd_vd3` could take `u_s2*` un-tiled |
| 5 | `_math/levin.levin1d_adaptive` rebuilds `_cheb_D(k, aa, bb)` and runs a full SVD per interval | `_cheb_D` is a fixed `[-1,1]` matrix times `2/(bb-aa)`; hoist it. Measured ~0.45-0.58 s per 1-D integral at omega up to 1e4 (`p17_math.py`) | measured timings |
| 6 | `gbd._reconstruct_windowed` `np.bincount(..., minlength=Ny*Nx+1)` twice per (bucket, chunk) (`:1669-1672`) | allocates 2 x full-grid float64 per inner iteration; `np.add.at` on a preallocated buffer avoids it | code inspection, not measured separately |

**Confirmed-correct performance claims:** the Maslov quadrature `G` matrix is
genuinely never materialised — `_QUAD_FACTORIZE` (Kronecker scatter, `:2524-2547`)
plus `_QUAD_ROW_BAND` banding make the 451 GB-at-N=16384 figure obsolete, so the
brief's "is `G[s2,j]` chunked?" is answered *yes*, two ways.

## Alternative algorithms / methods

1. **Wire the already-present uniform asymptotics into the saddle integrators.**
   `uniform_fold_airy` and `pearcey` exist in `lenses_maslov.py:986-1084` but are
   dead code — no integrator calls them. Pairing coalescing saddles through the
   Chester–Friedman–Ursell cubic map would make `stationary_phase` finite across
   a fold instead of `1/sqrt(max(|det H|, 1e-300))` ~ 1e150, at the cost of a
   second Newton root per pixel. Refs: Chester, Friedman & Ursell,
   *Proc. Camb. Phil. Soc.* **53**:599 (1957); Ludwig, *Comm. Pure Appl. Math.*
   **19**:215 (1966); Kravtsov & Orlov, *Caustics, Catastrophes and Wave Fields*,
   2nd ed., Springer 1999, ch. 4.
2. **Complex-source-point / complex-ray Gaussian-beam summation** for the caustic
   band: replace the real saddle with a complex one; stays finite and needs no
   catastrophe classification. Refs: Deschamps, *Electron. Lett.* **7**:684
   (1971); Heyman & Felsen, *JOSA A* **18**:1588 (2001). *Practical note:* FGA is
   already this method in disguise, and I measured it to be **exact** on the
   free-space oracle (fidelity 1.000000, complex scale 1.0027 + 0.0054i, power
   ratio 1.005 at z = 0.5 z_R, `p9_fga.py`) — including the absolute Gouy phase
   that GBD gets wrong. The right move is therefore to route caustic work to FGA
   (as `apply_real_lens_universal` already does) and to repair/retire Maslov's
   `auto -> local_quadrature` path rather than add another asymptotic layer.
3. **Wigner / phase-space transport.** Propagate the Wigner distribution along
   rays and reconstruct; Liouville makes the transport Jacobian exactly 1, so the
   `|J|` vs `sqrt|J|` ambiguity of the present chart cannot arise, and there is no
   Maslov index to track. Refs: Bastiaans, *JOSA* **69**:1710 (1979); Alonso,
   *Adv. Opt. Photon.* **3**:272 (2011). Cost: 4-D transport, and interference
   requires care.
4. **Flux-tube + Huygens hybrid (the commercial-POP pattern).** Trace a
   differential ray tube, take the amplitude as `sqrt(dA_0/dA)` (the ray-density
   Jacobian — exactly the `sqrt(J)` missing from the Maslov integrand), and
   finish the last leg with an explicit Rayleigh–Sommerfeld integral over the
   exit pupil. `apply_real_lens_traced(amplitude_model='ray_density')` already
   implements the amplitude side, so a Maslov-vs-traced *amplitude* cross-check
   would have caught the P1 Jacobian finding. Refs: Greynolds, *Proc. SPIE*
   **560**:33 (1985); Zemax POP / CODE V BSP documentation.
5. **Batch the Levin engine.** `_math/levin.py` is the only genuinely
   caustic-uniform evaluator here and I verified it to 1e-13…1e-15 *with* a
   stationary point (below); the blocker is ~0.5 s per 1-D integral. All pixels
   share the same collocation structure, so a batched TSVD (one `np.linalg.svd`
   over a `(n_px, k, k)` stack) would make full-grid caustic charts tractable.

## Code organization observations

* `apply_real_lens_maslov` is **1 290 lines** (1087-2376) with six phases inline
  (fold-split, chart sizing, trace, fit, integrate, upsample/normalise). The
  chart sizing (1590-1700), the ROI block (1859-1925) and the linear-phase
  re-application (2268-2337) are each independently testable units that are not
  reachable except through the whole driver.
* Comment-to-code ratio in `lenses_maslov.py` and `gbd.py` is extreme and almost
  entirely historical ("v5.22 FIX", "audit E-L9", "byte-identical"). The
  45-line E-H1 comment at `2193-2227` annotates 9 lines of code. Meanwhile the
  two facts a reader most needs — *which convention `Q` is in* and *which plane
  the Maslov output lives on* — are stated nowhere near the code that decides
  them (and the second is wrong, see P1 above).
* `gbd.py` ships three public functions plus a dedicated test file
  (`gbd_asm_gouy_phase`, `gbd_field_to_asm`, `asm_field_to_gbd`,
  `test_v5_21_gbd_asm_interop.py`) whose entire purpose is to compensate for the
  conjugation bug in `propagate_beamlets_freespace`. Fixing the sign deletes the
  API.
* NumPy/CuPy twins are maintained by hand and have already drifted (the `E-L9`
  comments document dead-argument drift between `_integrate_quadrature` and
  `_integrate_quadrature_cupy`). Every physics finding above has to be applied
  twice, with no shared kernel to guarantee it.
* `_integrate_stationary_phase` (`2924-2932`) and `_integrate_local_quadrature`
  (`3455-3459`) duplicate the Hessian de-normalisation block verbatim; a shared
  helper (like the `_maslov_newton_saddle_cpu` extraction already done) would
  keep the two in step.
* `_fft_reconstruct_applicable` swallows every `Exception` except `MemoryError`
  (`gbd.py:1710-1715`) to detect a JAX tracer. A `TypeError` from a genuinely
  malformed bundle silently changes the arithmetic route — the same
  silent-route-change the docstring says the library refuses.

## Unverified suspicions

* **Probe 3 (fold caustic vs brute-force Rayleigh–Sommerfeld) not run.** The
  box was heavily loaded by sibling audits and a converged RS oracle plus three
  Maslov integration methods past the marginal focus of an f/2 singlet did not
  fit the session budget. From the code I expect: `stationary_phase` diverging
  as `1/sqrt(max(|det H|, 1e-300))` at the fold; `local_quadrature` inheriting
  the P0 defects (its `sigma = 1/sqrt(pi |lambda|)` blows up as `lambda -> 0`,
  driving the `np.clip` over-count of the second P2); `quadrature` finite; and
  `levin` the only uniformly valid one. Confirming needs the RS integral and a
  phase-jump-sign check across the fold.
* **FGA through a real singlet.** `p9_fga.py` measured `apply_real_lens_fga`
  fidelity 0.357 vs `apply_real_lens` where `apply_real_lens_gbd` got 0.9985,
  with power ratio 0.694. My chart was aggressive (0.15 mm aperture, 60 um
  waist, `dq_step=2`, auto `p_max`/`n_p`) and heavily clipped, and the module
  documents an "energy caveat" for near-axial inputs through strong focusing, so
  I cannot call this a defect. It deserves a dedicated run with an explicit
  `p_max` at the field content and `normalize_output='power'`.
* **FGA slope->direction-cosine monodromy factors.** `fga.py:1223-1224` and
  `1492-1493` use `go = (1+u^2+v^2)^-1.5` on the output block and
  `gi = (1+u^2+v^2)^{+1.5}` on the input block. The `O(u^2)` isotropic
  approximation is documented (audit S2-16), but I did not verify that the input
  factor should be the *reciprocal* form rather than the same sign; confirming
  needs an analytic two-surface monodromy in both parameterisations.
* **`_reconstruct_windowed` bincount churn** (perf item 6) — inferred from the
  code, not measured in isolation.

## Checked and found correct (brief)

* `_math/chebyshev.py`: `T_n`, `T'_n = n U_{n-1}`, and the differentiated
  `T''_{n+1} = 2u T''_n + 4 T'_n - T''_{n-1}` recurrence all match
  `numpy.polynomial.chebyshev` on `[-1,1]` to max abs 8.1e-15 / 3.4e-12 /
  1.1e-9 for n <= 32 (relative ~3e-15 given `T''_32 ~ 3.5e5`) — `p17_math.py`.
  *Caveat:* outside `[-1,1]` the recurrences lose accuracy fast (1.4e-2 at n=32,
  |u| <= 1.5). The Maslov code masks those pixels via `inbox`, so it is not
  reached, but it is a latent trap for a new consumer. `chebyshev_fit_2d`'s
  `chebvander2d` packing / `reshape(n_x+1, n_y+1)` order is right.
* `_math/levin.py`: `levin1d_adaptive` vs a 4 000 001-point Simpson reference —
  **without** a stationary point (`g = x`): relerr 4.3e-12 / 1.5e-13 / 3.2e-13 at
  omega = 1e2 / 1e3 / 1e4; **with** an interior stationary point
  (`g = (x-0.5)^2`): 3.0e-15 / 3.7e-14 / 1.7e-14. The TSVD really does select
  the slowly-varying solution, so the classic Levin stationary-point failure IS
  guarded, as the module header claims. `_cheb_D` reproduces Trefethen's
  `cheb.m` (reversed for ascending nodes, scaled `2/(b-a)`); `_cc_weights` is the
  standard Clenshaw–Curtis formula; `_residual_est`'s `chebgrid2d` index order
  (`C2[n_t, m_s]` -> `x = u_t`) matches its comment.
* Maslov `_integrate_stationary_phase`'s saddle-point evaluator: for
  `INT A e^{2 pi i OPD_w} d^2v` the prefactor `(2 pi / lambda_large)^{n/2}` with
  `lambda_large = 2 pi` is 1, the Hessian de-normalisation `H/v_h^2` is right,
  and the KMAH phase `exp(i pi sigma/4)` with `sigma` = Hessian signature
  (+2 / 0 / -2 from det & trace) is the correct **sign** under
  exp(-i w t) / exp(+i k S). Verified exact (relerr `0.00e+00`) against the
  closed-form Gaussian integral on six synthetic charts (`p6`). The formula is
  self-consistent with the quadrature integrand — the amplitude error (P1 above)
  is in the integrand, not the evaluator.
* Maslov `_maslov_newton_saddle_cpu`/`_xp`: the 2x2 Newton step and the MSL-1
  sign-preserving `det_safe` floor are correct; the CPU active-subset and GPU
  freeze-converged forms are equivalent.
* Maslov quadrature measure: `abs_J = |det(ds1/du)|/(v2x_h*v2y_h)` correctly
  converts to `|det(ds1/dv2)|`, and `weight = tukey * du^2 * v2x_h * v2y_h` is
  the correct `d^2 v2`. Tukey normalisation converges to `int w = 1.8`
  (2.3e-4 rel at n_v2=32, 3.6e-15 at 256).
* Maslov canonical-map fit: residuals on an f = 6 mm biconvex at order 6 are
  OPD 2.09e-07 waves, `s1x`/`s1y` 4.70e-6 um, `v1x`/`v1y` 5.6e-9 (`p10`). The
  symplectic identity `dOPD/dv2 = -n1 (v1 . ds1/dv2)/lambda` closes to 5.8e-7
  relative. Conditioning of the 4-D total-degree Chebyshev design matrix on the
  pupil-masked ray grid (17 638 rays):
  `cond(A)` = 3.52e4 / 9.12e6 / 2.55e9 / 7.78e11 and `cond(A^T A)` =
  1.24e9 / 8.33e13 / 4.09e17 / 1.41e19 at `poly_order` 4 / 6 / 8 / 10. The
  normal-equations Cholesky in `_solve_fit` is therefore **safe at order 4**
  (the default) and **marginal at 6**, but `cond(A^T A)` at order 8 — which
  `poly_order='auto'` may select — is 4e17, past float64. The `cho_factor`
  -> `np.linalg.solve` -> `lstsq` fallback ladder exists, but it triggers only on
  a raised `LinAlgError`, not on silent loss of digits. Worth a documented
  order cap or an explicit condition check.
* `uniform_fold_airy` / `pearcey`: I re-derived the quartic moment
  `INT t^{2k} e^{i t^4} dt = 1/2 Gamma((2k+1)/4) e^{i pi (2k+1)/8}` and the CFU
  cubic mapping; both correct (agrees with `AUDIT_MASLOV_2026_07_09.md`).
* GBD `apply_thin_lens_to_beamlets`: `Q -> Q - 1/f` is right in the engineering
  convention; the screen phase `exp(-i k rho^2/(2f))` matches CONVENTIONS §7
  (converging for `f>0` under exp(-i w t)/exp(+ikz)); the base ray IS deflected
  (slope kick `u -= x/f`, then re-normalised to direction cosines); no extra
  amplitude factor is correct (Collins `A=1, B=0` -> `1/sqrt(det(A+BQ)) = 1`).
* GBD `decompose_field_to_beamlets` amplitude `E * (step*dx)^2/(pi w0^2)`
  reproduces the field in the Poisson-summation limit (residual ripple
  `2 exp(-pi^2 w0^2/dq^2)` ~ 1e-4 at overlap 1); the anisotropic branch's
  diagonal tensor `Q = diag(-i/zRx, -i/zRy)` and `pixel_area/(pi wx wy)` are the
  consistent generalisation.
* GBD `_reconstruct_fft` gating (uniform Q, uniform direction, on-grid to 1e-4 px)
  and the zero-pad / `mode='same'` alignment are correct.
  `_reconstruct_windowed`'s `R_cut` from the *smallest* eigenvalue of `-Im Q`
  (the widest axis) is the right bound, and the separable outer-product path is
  the correct factorisation for a non-skew `Q`.
* `lenses_gbd._carrier_referenced_bundle`: `Q += 0.5*(Wxx+Wyy)` has the right
  SIGN (a diverging carrier `W = rho^2/(2R)` gives `Re Q += 1/R`, matching the
  renderer's `exp(+i k rho^2/(2R))`); the anamorphic early-return, the launch
  direction `grad W`, and the `exp(+i k0 W)` piston restoration are consistent.
  `_fft_upsample`'s `* (ky*kx)` scaling and DC alignment are right (modulo the
  standard even-N Nyquist-bin asymmetry, negligible for a band-limited field).
* `apply_real_lens_gbd`'s `jacobian` / `reexpand` validation and the
  `z_image=None`-vs-`0.0` mapping on the `per_surface=False` branch
  (`lenses_gbd.py:531-548`) are correct as documented.
* FGA: the Herman–Kluk assembly `Z = (A+D) + i(k w0^2 C - B/(k w0^2))` maps
  correctly onto `R = sqrt(det[(1/2)(M_qq + M_pp - i hbar gamma M_qp +
  (i/(hbar gamma)) M_pq)])` with `hbar = 1/k`, `gamma = 1/w0^2`; the
  `Re a >= 0` branch is the continuous one; the normalisation
  `C = (k/2pi)^2 * dq^2 * dp^2 / 2` is exactly `1/(2 pi hbar)^d` times the
  phase-space measure with the `a(0) = 2^{d/2}` identity correction. Measured
  free-space FGA vs the analytic Gaussian at z = 0.5 z_R: fidelity 1.000000,
  complex scale 1.00266 + 0.00543i, power ratio 1.00535, relL2 after the scalar
  4.20e-4 (`p9_fga.py`) — FGA carries the Gouy phase *correctly*, which is the
  cleanest independent confirmation that GBD's is wrong.
* `apply_real_lens_universal` / `apply_real_lens_auto` routing is defensible in
  4 of the 5 regimes I probed (collimated slow lens -> `phase_screen`;
  collimated fast lens at focus -> `fga`; decentred beam at focus -> `fga`;
  two-beam multi-valued -> `fga`); only the tilted case fails (P2 above).
  `angular_spectrum_propagate(exitf, opd, wavelength, dx, dy)` at
  `fga.py:3075` passes `dy` in the right positional slot (checked against
  `asm.py:583-594`).

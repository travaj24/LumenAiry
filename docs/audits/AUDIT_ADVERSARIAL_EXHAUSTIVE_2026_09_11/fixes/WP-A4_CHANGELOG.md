# WP-A4 changelog text (Maslov / GBD / FGA + the asymptotic family)

Findings S2–S7 (report §2.5) and Y1–Y5 (report §9) of
`AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11`.

---

### Fixed -- Maslov propagator: the canonical chart is built on the exit VERTEX plane (audit S3)

`apply_real_lens_maslov` read `TraceResult.image_rays` directly, i.e. the rays
as `raytrace.trace` leaves them -- ON the last surface, at `z = sag(rho)` -- and
fitted the canonical map `(s2, v2) -> OPD` there while documenting and returning
it as the exit-plane field.  The missing `n_exit * sag(rho) / N` term is a pure
`rho^2` (defocus) contribution that the Chebyshev fit absorbs silently.  With
`output_plane_distance != 0` the free-space leg was `t = d/N` instead of
`(d - z)/N`, so the requested observation plane was offset by the sag as well.

The driver now transfers through the shared `TraceResult.at_exit_vertex()`
(§15.1) before anything is fitted, and the `output_plane_distance` leg starts
from `z = 0`.

* `lumenairy/elements/lenses_maslov.py` (`apply_real_lens_maslov`, trace block).
* Measured (`repro/orch/maslov_exit_vertex_check2.py`, biconvex R = ±100 mm
  N-BK7, t = 3 mm, 5 mm pupil, λ = 1 µm, N = 512, dx = 14 µm): the
  maslov − traced `rho^2` term **−31.303 µm → −0.053 µm** against the predicted
  `n_exit * sag_exit(rho=1) = −31.255 µm`; `rho^4` +0.104 → +0.099 µm and the fit
  residual 4.2 nm rms, both unchanged.  Appending a zero-thickness FLAT dummy
  surface -- a physically null prescription edit -- used to change the fitted OPD
  by **4.90 waves** at ρ = 0.14 mm (`= |sag|`); it now changes it by 2.3e-13
  waves.
* Tests: `tests/unit/test_audit2609_a4_maslov_gbd.py::test_s3_*` (2).

### Fixed -- Maslov propagator: Van Vleck amplitude and the k/(2πi) prefactor (audit S4)

The integrand carried `|det(ds1/dv2)|` where the Van Vleck–Maslov kernel
requires its SQUARE ROOT, and the `k/(2 pi i) = 1/(i lambda)` prefactor was
absent.  On a free-space chart, where `ds1/dv2 = -z I` exactly, the returned
field was `i * lambda * z` times the true one -- wavelength- AND
distance-dependent, not the "arbitrary overall prefactor" the docstring claimed.
`normalize_output='power'` (the default) hid the constant part but not the
spatial variation of `sqrt(J)`, which re-weights the angular spectrum inside the
integral; `roi=` runs are forced onto `normalize_output='none'` and returned an
absolutely wrong scale.

Fixed at all seven sites through one shared `_van_vleck_density` helper (the
four NumPy integrators `quadrature` / `stationary_phase` / `local_quadrature` /
`levin`, plus the three CuPy twins -- CuPy is not installed here, so those are
desk-checked), with the `1/(i lambda)` prefactor applied once to the assembled
field.  `_integrate_levin` regained the `v2x_h` / `v2y_h` half-widths that
E-L9 had dropped as dead: the square root no longer cancels against the
`d^2 v2` measure, so the unit-box Levin engine has to carry
`sqrt(v2x_h * v2y_h)` explicitly.

* `lumenairy/elements/lenses_maslov.py`.
* Measured (`repro/MASLOV-GBD-FGA/p12_prefactor.py`, `normalize_output='none'`
  vs `angular_spectrum_propagate`): `|E_maslov / E_ASM|`
  **4.998e-10 / 9.996e-10 / 1.996e-09 → 0.99966 / 0.99963 / 0.99800** at
  λ = 1 µm, z = 0.5 / 1 / 2 mm, and `arg` **+1.5716 rad (= +π/2) → +0.0002 rad**;
  the same at λ = 2 µm.  `ratio/(λz)` was 0.998–1.000 on every row before, which
  is the proof that the Jacobian power was wrong.
* Tests: `tests/unit/test_audit2609_a4_maslov_gbd.py::test_s4_*` (3).
* **Migration.**  `apply_real_lens_maslov(..., normalize_output='none')` now
  returns a physically-scaled field; its absolute amplitude changes by
  `1/(i * lambda * |det ds1/dv2|^{1/2})` relative to v5.45.  The default
  `normalize_output='power'` is unaffected except for the (small, real) shape
  change from the corrected density: 0.44 % / 0.42 % of the `|E|` profile on a
  benign converging chart, 0.30 % at NA 0.13, unbounded on an aberrated one.

### Fixed -- Maslov `local_quadrature`: principal axes, a window, and the window divided back out (audit S2, P0)

`integration_method='auto'` -- the default -- routed every real focusing chart
into `_integrate_local_quadrature`, which had three compounding defects: the
sampling box was scaled by the Hessian EIGENVALUES but laid out on the
COORDINATE axes (so the two widths swapped whenever `H44 > H33`, and the cross
term `H34` was ignored entirely); the samples were a hard-truncated uniform
Riemann sum of a chirp, leaving the Fresnel endpoint oscillation -- an
`O(1/extent)` error that does not shrink with `local_n_samples`; and `np.clip`
folded out-of-box samples onto the box edge while still counting them at the
full unclipped cell area.

The integrator now diagonalises the 2×2 physical Hessian and samples on the
rotated lattice, tapers with a Gaussian window, divides that window's exact
effect on the QUADRATIC MODEL back out (computed on the same finite lattice, so
the scheme is exact for a quadratic chart at any `local_n_samples` /
`local_window_sigma`), and DROPS out-of-chart samples instead of piling them on
the boundary.  The CuPy twin follows, sharing the host-side lattice/taper/
correction so the two cannot drift.

* `lumenairy/elements/lenses_maslov.py` (`_integrate_local_quadrature`,
  `_integrate_local_quadrature_cupy`, new `_local_window_1d` /
  `_local_window_geometry`).
* Measured (`repro/MASLOV-GBD-FGA/p6`, `p7`; synthetic quadratic charts with a
  closed-form answer) at the shipped defaults `local_n_samples=8`,
  `local_window_sigma=3.0`:

  | A | B | H34 | before | after |
  |---|---|---|---|---|
  | 40 | 40 | 0 | 1.19 | 1.15e-14 |
  | 40 | 4 | 0 | 1.19 | 1.29e-15 |
  | 4 | 40 | 0 | **9.12** | 5.87e-15 |
  | 100 | 10 | 0 | 1.19 | 1.45e-14 |
  | 40 | 40 | 30 | 2.48 | 7.83e-15 |
  | 60 | 20 | 25 | 3.07 | 2.67e-16 |

  The 40 % convergence floor is gone: 1e-14…1e-15 at every
  `(window_sigma, n)` in {3, 4, 6, 10} × {8, 16, 32, 64, 128} (it was flat at
  4.0e-01 from n = 32 on).  Where the requested window exceeds the fitted chart
  box (`window_sigma` = 20 / 40) the error is now bounded at 2e-01…2e+00 -- the
  honest cost of truncating the chart -- instead of the 7.5e+01…2.0e+03 the
  `np.clip` over-count produced.
* Tests: `tests/unit/test_audit2609_a4_maslov_gbd.py::test_s2_*` (10).

### Changed -- Maslov `integration_method='auto'` falls back to `stationary_phase`, and warns (audit S2)

`'auto'` resolved to `'local_quadrature'` whenever uniform quadrature would need
more than `_N_V2_AUTO_MAX` samples.  It now resolves to `'stationary_phase'`
and emits a `RuntimeWarning`.

Measured on an f = 6 mm N-BK7 biconvex at its exit plane, scored against a
converged uniform quadrature (n_v2 = 320, self-converged to 8.5e-04 at
n_v2 = 256): `stationary_phase` relL2 **0.84**, `local_quadrature` **2.58** at
its defaults (1.47 at n = 16, 1.14 at n = 24 / ws = 4).  Both are exact on the
synthetic quadratic charts; the difference is that the local window reaches into
pupil zones where the order-4 fit is extrapolating.  Neither asymptotic
evaluator is valid at or near the exit plane of a focusing system -- the
v2-Hessian and `ds1/dv2` both collapse there -- which is what the new warning
says, together with the two remedies (`integration_method='quadrature'` with an
explicit `n_v2`, or `output_plane_distance=` to move the observation plane to
where the asymptotics belong).

* **Migration.** A call that relied on `'auto'` silently selecting
  `local_quadrature` now gets `stationary_phase` plus a warning.  Pass
  `integration_method='local_quadrature'` explicitly to keep the old routing
  (the integrator itself is now correct); pass `'quadrature'` for an
  exit-plane field.

### Fixed -- Maslov: the v2-oscillation estimator counts total variation, not excursion (audit S9/N2)

The `'auto'` integrator choice, the `'auto'` `n_v2` resolution and the
under-resolution warning all used `sum |c_k|` over v2-dependent Chebyshev terms,
justified as an upper bound on the OPD excursion.  The number of CYCLES along
v2 is bounded by the TOTAL VARIATION, and `TV(T_n)` on [-1, 1] is `2n`, so the
correct bound is `sum |c_k| * max(k3, k4)` -- now in one shared
`_v2_oscillation_bound` used at all three sites.  The audit measured the two
differing by up to **2.47×** on real fitted charts (f = 2 mm biconvex, order 8:
136.2 → 337.1; f = 6 mm, order 8: 634.0 → 1539.7), i.e. `auto` could pick
`quadrature` for a chart that then speckled, with the warning that would have
said so silenced by the same under-count.

### Added -- Maslov: the asymptotic evaluators warn on a non-collimated input (audit S6)

`stationary_phase` and `local_quadrature` solve `grad_v2 OPD = 0`.  The
symplectic identity `dOPD/dv2 = -n1 (v1 . ds1/dv2)` (verified to 5.8e-7 relative
on a real singlet chart) makes that the `v1 = 0` collimated launch ray at every
pixel and for every input -- while the driver deliberately sizes the chart to
cover a diverging / tilted one (`na_proxy = na_lens + na_input`).  There was no
warning and no docstring caveat.  A `RuntimeWarning` now fires when either
asymptotic method is selected and the measured input angular spread exceeds
`_SADDLE_FLAT_INPUT_NA = 1e-3`, pointing at `'quadrature'` / `'levin'` (which
integrate the true integrand) or `collimated_input=True`.

* Tests: `tests/unit/test_audit2609_a4_maslov_gbd.py::test_s6_*`.
* Deferred: fitting the input's local wavevector `(k1x, k1y)(s1)` and adding
  `k1 . ds1/dv2` to the Newton gradient/Hessian, which would make the two
  asymptotic methods correct rather than merely honest.

### Fixed -- GBD: the beamlet Gouy / Collins phase was conjugated (audit S5)

`BeamletBundle.Q` is the ENGINEERING `1/q` (`q_code = conj(q_physics)`) and the
renderer converts on output (`exp(+0.5j k conj(Q) rho^2)`); the AMPLITUDE did
not.  `propagate_beamlets_freespace` used `Q_new / Q_old` where the physics
amplitude ratio is `conj(Q_new / Q_old)`, and the two Collins sites used
`1/(A + B Q)` where it is `conj(1/(A + B Q))`.  The library had documented the
resulting `2 arctan(z / zR_beamlet)` as an inter-propagator "convention" and
shipped three public functions plus a test file to compensate for it -- but the
offset depended on `waist_factor`, a purely numerical knob, which is the
definition of an error.

Conjugated at five sites: the scalar and tensor branches of
`propagate_beamlets_freespace`, `_freespace_tensor_moebius_np`, both branches of
`apply_abcd_to_beamlets`, and `apply_prescription_persurface_to_beamlets`.

* `lumenairy/propagators/gbd.py`.
* Measured (`repro/MASLOV-GBD-FGA/p1`, one beamlet vs the analytic Gaussian):
  `arg(E/A)` **+0.927295 / +2.214297 / +2.942255 rad (= 2ψ exactly, to 1e-13) →
  0.000000**, and relative L2 with NO phase fit **0.894 / 1.789 / 1.990 →
  1.41e-13 / 5.37e-13 / 4.20e-12** at z = 0.5 / 2 / 10 z_R.
* Measured (`p2`, full pipeline vs `angular_spectrum_propagate`, N = 256,
  λ = 1 µm, z = 3 mm): residual global phase **+3.133 rad → +5e-06 rad** and
  relL2 with no phase fit **1.999 / 1.995 / 1.987 → 1.76e-03 / 7.02e-03 /
  1.57e-02** at waist_factor 1 / 2 / 3 (the Gabor-frame floor).
* Measured (`p3`, `decompose_field_adaptive` mixed-waist bundle, where the error
  was NOT a global phase): relL2 **1.78e-01 … 1.82e+00 raw, 9.2–18.4 % after the
  best global-phase fit → 5.06e-02 flat at z = 20 / 50 / 200 / 1000 µm**, with
  nothing left for a phase fit to remove.
* Tests: `tests/unit/test_audit2609_a4_maslov_gbd.py::test_s5_*` (2);
  `tests/unit/test_v5_21_gbd_asm_interop.py` REWRITTEN (it pinned the defect);
  `tests/unit/test_audit_w5_propagators.py::TestP230CollinsFactor::test_matches_analytic_collins_and_gaussian_w_of_z`
  re-based on a textbook Gaussian-beam oracle instead of the library's own
  expression (post-fix agreement **3.20e-16**, the old pinned expression scores
  **2.00**).

### Deprecated -- `gbd_asm_gouy_phase`, `gbd_field_to_asm`, `asm_field_to_gbd` (audit S5)

The three functions existed only to compensate the conjugated Gouy phase above.
They are now no-ops (`0.0` and the identity respectively), emit a
`DeprecationWarning` through `_deprecation.warn_deprecated_alias`, and are
scheduled for removal in v5.48.

* **Migration.** Delete the call: a GBD free-space field already matches
  `angular_spectrum_propagate` in absolute phase.  For a general
  propagator-agnostic reconciliation use `match_global_phase`, which is
  unchanged.

### Changed -- `_lens_jax` requires `jax_enable_x64` (audit S7)

`apply_real_lens_traced_jax` and `apply_real_lens_maslov_jax` never read JAX's
x64 setting, so under JAX's default a complex128 input was truncated at
`jnp.asarray` and the result came back complex64 -- silently.  Both now call a
new `_require_jax_x64` that RAISES with actionable text, adopting the
`elements/rcwa/_core.py` policy the audit identified as the right one (§15.6).

* Measured on this fixture (f = 6 mm biconvex, 0.3 mm aperture, λ = 1 µm): with
  x64 off the pre-fix code returned **complex64** for a complex128 input, and
  its phase screen differed from the float64 one by **1.71e-05 waves rms /
  8.17e-05 waves p-v**; the audit measured 1.5e-4 … 1.8e-3 waves rms on a larger
  lens against 2e-10 … 3e-9 with x64.
* **Migration.** `jax.config.update('jax_enable_x64', True)` once at import.
  The NumPy `apply_real_lens_traced` / `apply_real_lens_maslov` are unaffected.

### Fixed -- `_lens_jax`: the default path is `jax.jit`-able, and uses the shared exit-vertex operator (audit S7, §15.1)

Both JAX entry points seeded their Newton inversion with
`float(x_out_grid[...])`, which raises `ConcretizationTypeError` under
`jax.jit` -- so the default (static-geometry) path could not be jitted at all,
contradicting the docstring's "vmap+JIT replaces the pool".  The already-written
tracer-safe expression (with `stop_gradient`, since the Newton root does not
depend on its starting point) is now used unconditionally.  The two hand-written
exit-vertex copies, which clamped `N` to `1e-30` and so gave a grazing ray
`t = -z/1e-30` (~1e26 m of phantom OPL) where the NumPy copies gave `t = 0`, are
replaced by `raytrace.jax_trace.exit_vertex_transfer_jax`.

* Measured: `jax.jit` now compiles both entry points and agrees with the eager
  call to **1.30e-12** on a unit-peak field (pre-fix: `ConcretizationTypeError:
  Abstract tracer value encountered where concrete value is expected`).
* Tests: `tests/unit/test_audit2609_a4_maslov_gbd.py::test_s7_*` (5).

### Fixed -- asymptotic family: the v2-linear phase stays inside the integrand (audit Y1, P0)

`extract_linear_phase=True` (the DEFAULT) pre-fits and removes
`a0 + a1 u1 + a2 u2 + a3 u3 + a4 u4` from `Phi`.  The `a3 u3 + a4 u4` terms are
linear in the INTEGRATION variable `v2`, so dropping them moves the complex
saddle `delta* = M^-1 b / 2` and `Re(b^T M^-1 b / 4)` -- an amplitude and
position error, not a phase reference.  On an off-axis `source_centre` the
rank-deficient design splits a ~2 700-wave ramp roughly 50/50 between `a1` and
`a3`, so this is reached by `propagators/subaperture.py:524` for every
off-centre patch of the patch-decomposition propagator.  The identical defect
was found and fixed in the sibling `lenses_maslov` (audit N4) and never ported.

`CanonicalPolyFit.eval_phi` and `eval_phi_with_v2_grad` (and the JAX
`eval_phi_xp`) now always include `a3 u3 + a4 u4` and its `a3` / `a4`
contribution to the gradient; `include_linear` gates only `a0 + a1 u1 + a2 u2`,
which is constant in `v2` and factors out.  `H_Phi` needs no change.

* `lumenairy/propagators/asymptotic_canonical_fit.py`,
  `asymptotic_jax_twin.py`.
* Measured (`repro/ASYMPTOTIC/t8`, stock N-BK7 singlet, λ = 1.31 µm,
  `source_centre = (100 µm, 0)`), against an independent ray trace of the chief
  ray at **(9.665e-05, 0) m**: with the DEFAULT flag the peak moves
  **(-3.174e-04, -6.228e-04) m → (9.785e-05, 0.0) m** and the peak amplitude
  **2.277e-05 → 1.99959e-03** (88× low → identical to the
  `extract_linear_phase=False` reference).  The two flags' |E| now agree to
  **9.21e-10** over 149 bright pixels; on-axis fits are unaffected (ratio
  exactly 1.0 before and after), which is why every pin passed.
* Tests: `tests/unit/test_audit2609_a4_asymptotic.py::test_y1_*` (2);
  `tests/unit/test_niche_audit_w6_asymptotic.py` W6-A4 premise restated (it
  asserted `|a3| + |a4| < 1e-8` as a load-bearing premise).

### Fixed -- asymptotic family: Van Vleck–Maslov normalisation of the modal propagator (audit Y2)

Same class as S4.  The `v2` integrand carried `|det J|` where Van Vleck–Maslov
requires `-1j sqrt(|det J|) / lambda`, so the output was
`i * lambda * sqrt(|det J|)` times the true field -- wavelength- AND
field-point-dependent, not the constant the docstring described.  Fixed at all
four consumer sites through one shared `van_vleck_weight` helper
(`propagate_modal_asymptotic`, `aberration_tensor`, and the two JAX evaluators).

* `lumenairy/propagators/asymptotic_maslov.py` (new helper), `asymptotic.py`,
  `asymptotic_aberration_tensor.py`, `asymptotic_jax_twin.py`.
* Measured (`repro/ASYMPTOTIC/t16`, a synthetic fit encoding exact free-space
  Fresnel propagation, z = 20 mm, λ = 1 µm, vs the analytic q-parameter Gaussian
  beam): `E_code / E_true` **(8.09e-19 + 2.0000000000256e-08 j) →
  (1.0000000000128 − 4.04e-11 j)**, spatial spread of the ratio 4.2e-11.  The
  pre-fix value is exactly `i * lambda * z`.
* The correctly-normalised sibling `propagate_hf_chebyshev_quadrature` is
  UNCHANGED and re-verified at **2.7398e-11** against the same analytic beam
  (audit: 2.74e-11), so the two families are now on one scale.
* Tests: `tests/unit/test_audit2609_a4_asymptotic.py::test_y2_*` (2);
  `tests/unit/test_niche_audit_w6_asymptotic.py::test_w6_a7_output_field_carries_the_van_vleck_normalisation`
  INVERTED (it pinned the absence of the normalisation as a documentation gap);
  the two `test_audit_propagation.py` `…ModalAsymptoticStillBitEqual` inline
  scalar references and the two `test_niche_audit_w6_asymptotic.py` brute-force
  quadrature oracles updated to the corrected weight (they were stale COPIES of
  the algorithm, and their tolerances -- all relative to their own reference --
  are unchanged).
* **Migration.** Every `L` entry of `aberration_tensor` and every pixel of
  `propagate_modal_asymptotic` changes scale by `1 / (lambda^2 |det J|)` in
  `|.|^2`.  Measured on the two shipped test fixtures the factor is
  8.362145e+16 (R1 = 60 mm, ap 12 mm) and 1.486604e+17 (R1 = 500 mm, ap 4 mm),
  reproducing the previously-pinned `|L_00|^2` values to 6 significant figures
  when divided back out.  `AberrationTensorResult.van_vleck_weight` (new, with a
  `None` default) exposes the applied factor so a caller that needs the v5.45
  scale can recover it: `L_legacy = L * lambda**2 * |det J|` in magnitude.

### Fixed -- asymptotic JAX twin: degenerate `eigvalsh` gradient and the four missing guards (audit Y3)

`aberration_tensor_lg00_jax`'s default `w_o` came from
`jnp.linalg.eigvalsh(Re M)`, whose JVP carries `1/(lambda_i - lambda_j)`;
`Re M = J^T J / w_s^2 + I / w_p^2` is near-isotropic for any
rotationally-symmetric system (measured gap/mean **3.3e-10**), so `jax.grad` was
~86 % wrong for every parameter that ROTATES `Re M` (image point, saddle point,
decentre) while the ones that only rescale it (`w_s`, `w_p`) survived.

The eigensolve is replaced by the closed-form 2×2 largest eigenvalue
(`sym2x2_max_eigenvalue`, double-`where`d so the gradient is finite at an exact
degeneracy) in a shared `lg00_sampling_waist_from_M` that BOTH backends now
call, and the default `w_o` is `stop_gradient`-ed: `lambda_max` of a
near-degenerate symmetric matrix is genuinely non-smooth, and on this branch
`w_o` is a normalisation convention rather than a physical length, so
differentiating through it is ill-posed.  Pass `w_o=` explicitly to make it a
live differentiable slot.

Separately, the JAX twins reproduced none of the NumPy path's guards.  The four
masks (in-box `s2`, in-box `v2`, `|det M| >= 1e-300`, `|Re b_quad| <= 700`) plus
a final finite gate are now `jnp.where` gates in `_modal_field_lg00_pixel_jax`.

* `lumenairy/propagators/asymptotic_jax_twin.py`,
  `asymptotic_aberration_tensor.py`, `asymptotic_maslov.py`.
* Measured (`repro/ASYMPTOTIC/t13`, stock singlet): with an explicit `w_o`,
  `d/ds2x` **9.438e-03 vs a converged 5-point FD of 6.738e-02 (rel 8.60e-01) →
  -9.57454010e+11 vs -9.57379958e+11 (rel 7.7e-05)**, and `d/dv*_x` rel
  **8.61e-01 → 7.0e-04**.  The default-`w_o` gradient now equals the
  explicit-`w_o` one to 1.7e-11 relative.
* Measured (`repro/ASYMPTOTIC/t10`, a grid 3× the fit half-box): the JAX twin
  returned **60 non-finite values of 81** where NumPy returned zeros; it now
  returns **0 non-finite** and is exactly 0 on all 72 pixels NumPy zeroes.
  In-box parity is preserved: `aberration_tensor` (0,0) **1.148e-11** relative
  (audit 7.99e-11), the 17×17 grid **2.745e-10 RMS / 9.353e-10 worst** (audit
  2.64e-10 / 7.14e-10), and the IFT solver's `v*` **3.042e-19** (audit 7.3e-19).
* Tests: `tests/unit/test_audit2609_a4_asymptotic.py::test_y3_*` (3).

### Fixed -- asymptotic family: `w_o` cross-backend clamp, `A_lead` overflow, `pupil_modes`, and three wrong narrative claims (audit Y5)

* The "BIT-FOR-BIT cross-backend contract" for the default `w_o` was false: the
  NumPy side clamped to `[1e-9, 1.0]` and the JAX side did not, so for
  `lambda_max(Re M) < 1` they returned 1.000000 and 29.857 and `|L|` differed by
  **97 %**.  Both now call the same helper, clamp included.
* `aberration_tensor`'s scalar `A_lead` had no `|Re b_quad| <= 700` guard and
  overflowed to `inf` with a bare NumPy warning on inputs the batched path
  rejects cleanly.  Both now share `B_QUAD_EXP_MAX`.
* `pupil_modes` is very nearly inert -- the pupil CONTENT comes only from
  `pupil_amplitudes` -- so a caller who passed `pupil_modes=[(0,0),(1,0)]`
  without matching amplitudes silently got an LG_{0,0} pupil and a result object
  that claimed otherwise.  `aberration_tensor` now warns, naming the modes that
  carry no coefficient.
* "This is exact at the stationary point", said of the dropped Gauss–Newton
  Hessian term in three places, is wrong: at the stationary point
  `J^T(s1 - s_src)/w_s^2 = -(v - v_c)/w_p^2 != 0`, so the term does not vanish;
  only the residual does.  Reworded in all three.
* The Seidel/Zernike framing at the top of `asymptotic.py` and on
  `AberrationTensorResult` over-claimed what `L` measures (an image-plane LG
  overlap, not a pupil wavefront-error expansion; the module's own W4-T2 note
  records 5 of 6 sign flips of the "(2,0) spherical" channel across adjacent
  designs).  Caveated in both places.

### Fixed -- `propagate_modal_asymptotic` says how many pixels it zeroed, and stops leaking NumPy warnings (audit Y4)

Six independent gates dropped pixels to exactly 0, and five early exits could
return an all-zero array, with no warning and no diagnostic -- while the sibling
`propagate_hf_chebyshev_quadrature` warns when its grids leave the fit box.
`optimize/driver.py` and `propagators/dispatch.py` feed this field straight into
cross-propagator wave merits, where "mostly zero" is indistinguishable from
"dark".  A single `RuntimeWarning` now names the fraction dropped and the
dominant reason, from every exit.  Separately, `np.sqrt(det_M)` and
`math.pi / safe_sqrt` were applied to the whole array including the NaN entries
from out-of-box Newton results (the dtype-aware sentinel
`np.where(sqrt != 0, ...)` does not catch NaN, since `NaN != 0` is True), which
leaked `RuntimeWarning: invalid value encountered in sqrt` and `… in divide`
from routine calls; both are now computed on the valid entries only,
bit-identically.

### Fixed -- Maslov: the canonical fit no longer returns an arbitrary null-space member (NEW, found while verifying S3)

The v5.21 normal-equations Cholesky in `_solve_fit` is justified by "``A`` is a
normalized tensor-Chebyshev Vandermonde -- well-conditioned and ~1.5x
oversampled -- so squaring the condition number in ``G`` is safe".  That does
not hold on a small, fast chart, and the `LinAlgError` fallback ladder cannot
see the failure: a numerically positive-semidefinite but RANK-DEFICIENT Gram
factors happily and returns an arbitrary member of the solution set.

Measured on an f = 6 mm N-BK7 biconvex, 0.2 mm aperture, `poly_order=4`:
`rank(A) = 65` of 70 columns, `cond(A) = 1.81e+15`, `cond(A^T A) = 6.18e+18`.
Two runs of the SAME optic -- `output_plane_distance=d` versus baking `d` into
the prescription's last thickness -- whose design matrices agree to **3.1e-15**
and whose OPD right-hand sides agree to **6.8e-13 waves** came back with
coefficients **0.869 waves apart**, both with the same fit residual
(1.613e-09 vs 1.756e-09 waves, and 1.756e-09 cross-evaluated).  The difference
lives entirely in the null space: invisible on the training manifold, and not
invisible inside the v2 integral, which samples `(s2, v2)` combinations off it.

`_solve_fit` now measures `cond(A^T A)` once (an `M x M` `eigvalsh`, microseconds
beside the `A^T A` GEMM) and routes anything above `_GRAM_COND_MAX = 1e12` to
`np.linalg.lstsq`, whose minimum-norm solution is a deterministic, unique
function of `(A, RHS)` -- and which is also the pre-v5.21 behaviour, so this
restores it exactly where it mattered.  Above `1/eps` (~4.5e15) it additionally
warns, because there the fit is genuinely rank-deficient and even the min-norm
answer depends on the solver's `rcond` cut.

* Measured effect on the documented `output_plane_distance` contract
  ("matches baking the same distance into the prescription's last thickness"),
  same fixture at d = 0.5 / 1 / 2 / 5 mm: relL2 **0.449 / 0.757 / 0.849 /
  1.384 → 3.48e-05 / 1.70e-05 / 2.33e-05 / 9.27e-04**.  The docstring's
  "~1e-10" is corrected to the measured numbers, with the chart-identity
  (6.8e-13 waves in OPD, 0 m in s1) stated as the exact part.
* Well-conditioned charts keep the fast Cholesky path, byte-identical to
  v5.21 (`cond(A^T A) = 1.24e9` at poly_order 4 on the audit's own 1.5 mm
  fixture, well inside the gate).
* Tests: `tests/unit/test_audit2609_a4_maslov_gbd.py::test_solve_fit_*` (3).

### Fixed -- Maslov `fold_split` free-space legs honour `dy` (audit S11)

The two mirror-leg gaps called `angular_spectrum_propagate(E, d, wavelength,
dx)` with no `dy`, so an anamorphic grid silently propagated those gaps with
`dy = dx` while the rest of the driver is anamorphic-aware.

### Changed -- `apply_real_lens_maslov` validates `stop_index` and checks the anamorphic aperture extent (WP-A2 hand-off)

`prescription['stop_index']` now goes through `_lens_real._normalise_stop_index`,
so an out-of-range or non-integer value RAISES with the §2 prefix instead of
being `int()`-ed into the "non-entrance stop" warning path -- where a negative
index read as a mid-train stop and a float raised a bare `TypeError`.
`stop_index=-1` now means the LAST surface, as Python indexing does.  The
pre-flight aperture check passes the y extent (`N_y` / `dy`) so an anamorphic
grid is checked against the axis that truncates first.

* **Migration.** A prescription carrying an out-of-range `stop_index` now raises
  from `apply_real_lens_maslov` where it previously warned about a non-entrance
  stop and continued.
* Tests: `tests/unit/test_audit2609_a4_maslov_gbd.py::test_a2_*` (8).

### Fixed -- fga: the universal dispatcher's two tilt-blind discriminators (audit S10)

`apply_real_lens_universal` routed a TILTED but perfectly collimated,
single-valued, high-NA plane AT ITS FOCUS to `phase_screen` -- NA 0.145 above
the 0.12 `na_threshold` and inside the caustic, precisely the regime the
dispatcher's own docstring says the thin screen cannot handle.  Both
discriminators in that decision were blind to a global tilt:

* `_caustic_zone` scored each exit ray's crossing with the optical AXIS
  (`z = -x_exit / u_exit`).  A global input tilt moves the focus off axis by
  ~`f * theta`, so the axis crossings measure a different quantity entirely.
  It now scores the crossing with the CHIEF ray -- the amplitude centroid of
  the meridional row launched along the amplitude-weighted mean local slope,
  traced in the SAME `ray_transfer_jacobian` call as the fan (one extra array
  element), so it costs nothing and cannot drift from it.
* the final escape hatch asked `_carrier_residual_rms(E_in, None, ...)`, which
  returns EXACTLY the tilt magnitude for a pure tilt.  It now asks it on the
  DE-TILTED field (new `_remove_global_tilt` / `_global_mean_tilt`): a global
  linear phase is a change of reference direction, not a divergence.

* `lumenairy/propagators/fga.py` (`_caustic_zone`, `_universal_route`, new
  module-private `_global_mean_tilt` / `_remove_global_tilt`).  Both
  `_caustic_zone` call sites -- `apply_real_lens_auto` and `_universal_route`
  -- get the fix.
* **Independent oracle** (`tests/unit/test_audit2609_a4_fga_s10.py::_ray_focal_zone`):
  real rays through `raytrace.trace` + `TraceResult.at_exit_vertex` -- a
  different code path from the `ray_transfer_jacobian` differential fan -- with
  the geometric focal zone taken as the axial range where the bundle's RMS
  transverse spread about its own centroid is within 20 % of its minimum.
  Measured best-focus planes: **1.021305 mm** (collimated), 1.020605 (tilt
  0.02), 1.018505 (tilt 0.05), 1.019905 (tilt 0.05 + 80 um decentre) -- the
  focal DISTANCE of a collimated beam moves by at most **0.27 %** under tilt.
* Measured `_caustic_zone`, f = 1.2 mm N-BK7 biconvex, 0.30 mm aperture,
  lambda = 1 um, N = 192, dx = 2 um:

  | input | before [mm] | after [mm] | contains the real focus? |
  |---|---|---|---|
  | collimated | [1.021056, 1.032698] | **bit-identical** | yes -> yes |
  | tilt 0.02 rad | [1.318251, 10.987635] | [1.021117, 1.032215] | **no -> yes** |
  | tilt 0.05 rad | [2.002278, 11.019568] | [1.019191, 1.029461] | **no -> yes** |
  | tilt 0.05 + 80 um decentre | [1.996611, 28.212060] | [1.009260, 1.025989] | **no -> yes** |
  | 80 um decentre | [1.020926, 1.032689] | [1.010242, 1.028427] | yes -> yes |
  | slow lens, exit plane | [98.021603, 98.021702] | **bit-identical** | - |
  | multi-valued (2 tilted beams) | [0.277843, 3.274212] | **bit-identical** | - |

  The zone CENTRE moved **+534.1 %** under a 0.05 rad tilt before the fix and
  **-0.25 %** after, against the oracle's own -0.27 %.  The three symmetric
  fixtures are bit-identical because a centred beam's chief ray IS the axis.
* Measured collimation discriminator (`_NONCOLLIMATED_RESID_THRESH = 0.02`):
  raw `_carrier_residual_rms` **5.000000e-03 / 2.000000e-02 / 5.000000e-02 /
  1.000000e-01** at tilt 0.005 / 0.02 / 0.05 / 0.1 -- exactly the tilt --
  against **1.257379e-09 / 5.029515e-09 / 1.257379e-08 / 2.514758e-08**
  de-tilted.  Real divergence readings survive: R = 10 mm 9.714549e-03 ->
  9.715578e-03, R = 3 mm 3.238183e-02 -> 3.238526e-02.  A 0.05 rad TILTED
  R = 3 mm beam read **5.956998e-02** raw (84 % high, and over the gate for the
  wrong reason) and now reads 3.238526e-02, matching the untilted beam to
  4.0e-09 relative.
* Routing, same fixtures: tilt 0.02 **'traced' -> 'fga'**, tilt 0.05
  **'phase_screen' -> 'fga'**, tilt 0.05 + decentre **'phase_screen' -> 'fga'**.
  The four regimes the audit found correctly routed (collimated slow at the
  exit plane, collimated fast at focus, decentred at focus, multi-valued) are
  unchanged.  `apply_real_lens_auto`'s 2-way choice follows: **'gbd' -> 'fga'**
  for the tilted case.
* **What this does NOT claim.**  The fix makes the router frame-invariant; it
  does not make the chosen member more accurate on this fixture.  Measured at
  the focus against the traced chief-ray landing (+66.402 um) and the real-ray
  geometric fan (+66.439 um, 0.645 um rms; diffraction limit
  lambda/(2 NA) = 3.44 um): `phase_screen` gives centroid +66.469 um and
  intensity-rms width 3.269 um, `fga` gives +62.524 um and 12.721 um.  The two
  members disagree by 3.9 um and a factor 3.9 -- and by the SAME factor at
  tilt 0 (3.169 vs 12.607 um), so that is the members' own accuracy question
  (the auditor's open suspicion about FGA's convergence knob through a real
  singlet, re-measured here), not a tilt artefact.  What the pre-fix router
  did was pick between them on the observer's frame.
* Tests: `tests/unit/test_audit2609_a4_fga_s10.py` (16).

### Verified correct -- fga: the vector wrapper normalises the Jones components jointly (audit S10)

S10's third sub-item is that `apply_real_lens_maslov_vector` threads
`normalize_output` through with its `'power'` default and applies it
INDEPENDENTLY to `E_x` and `E_y`, so any differential transmission, vignetting
or in-box clipping between the two is normalised away and the output
polarization ratio is forced back to the post-Fresnel input ratio.
`apply_real_lens_fga_vector` does NOT share that defect and needed no change:
it applies ONE joint scale to `(ex, ey, ez)`, carries the per-surface Fresnel
s/p Jones with the geometric frame rotation, and ships a real `E_z` via
`return_longitudinal=True`.

Measured on a `(2, 48, 48)` Jones input with `P_x/P_y = 4` exactly, through an
f = 1.2 mm singlet to 1 mm past the vertex: `normalize_output='none'` gives
`P_x/P_y = 3.9999998702357877` and `'power'` gives `3.999999870235787` -- the
same to **2.2e-16 (1 ulp)** -- while the total power is restored to
1.0000000000000002.  The 3.2e-08 departure from the input ratio is the s/p
diattenuation the system really applies, and it survives the normalisation.
Pinned so it stays that way.

### Added -- cache registry: the `local_quadrature` sample-lattice cache

`_local_window_1d` (new, `functools.lru_cache`) is enrolled with
`_cache_registry.register_cache_clearer` under `'maslov_local_window'` and gets
a `clear_maslov_local_window_cache()` clearer, so it participates in
`clear_asm_caches()` / `lumenairy_context(clear_caches_on_exit=True)` like every
other module-level cache.  Entries are a few kB each at an LRU cap of 32, so no
byte-budget hook.

---

## VERIFY-A4 follow-up (independent re-verification pass)

The changes VERIFY-A4 made on top of `32ba3ba2` have their own changelog text
in `fixes/VERIFY_WP-A4_CHANGELOG.md` (same voice, same assembly).  They are:

* **`### Fixed -- optimize`** — the LG aberration merit is now a DIMENSIONLESS
  coupling `|L|^2 / |L_ref(0,0)|^2`, referenced to the aberration-free twin of
  the same optic, on both backends.  **Migration note:** the (0, 0)
  contribution moves from `-4.79e+14` (32ba3ba2) / `+0.9968` (v5.45) to a
  Strehl deficit in about `[-3e-03, 1]`; any absolute threshold, logged value
  or stopping tolerance tuned against either old scale must be re-derived.
  Relative channel weights keep their meaning (every channel is divided by the
  same per-field-point constant), so an optimiser converges to the same design.
  `LGAberrationMerit` gained `strehl_branch` (default `'sigma'`, a real Strehl;
  `'closed_form'` is the JAX-parity branch and warns that it is not one) and
  `sigma_grid_n` (default 64); it now costs two `aberration_tensor` calls per
  field point.
* **`### Fixed -- lenses_maslov`** — `apply_real_lens_maslov_vector` applies ONE
  joint `normalize_output` scale to the Jones pair (audit S10, third sub-item,
  requested in §5 item 6 of the report) and launches its polarization base rays
  along the input field's own local wavevector.
* **`### Fixed -- lenses_maslov`** — the S6 saddle warning is gated on the
  input's WAVEFRONT NA, not on the second moment of its angular spectrum; it
  used to fire on every collimated beam narrower than ~1 mm.
* **`### Changed -- tests`** — four oracles that imported
  `van_vleck_weight` from the module under test now derive it locally.
* **`### Changed -- validation`** — the three JAX cases in
  `validation/elements/test_lenses.py` enable x64 for their process, so
  `validation/run_all.py test_lenses` is green again (46/46).
* **`### Changed -- tests`** —
  `test_a1_auto_n_v2_resolves_demanding_default_quadrature` restated against a
  converged `n_v2 = 512` quadrature oracle.
* **`### Changed -- tests`** — WP-A15a §5 item 8: three wall-clock
  assertions (`test_audit_propagation.py` × 2, `test_audit_optimize.py` × 1)
  replaced by exact operation counts, and `pytestmark = pytest.mark.slow` on
  `test_audit_lens_models_2026_07.py`.
* **`### Added`** —
  `lumenairy.propagators.asymptotic.aberration_free_reference_fit` (the name is
  final; `lumenairy/__init__.py`'s eager re-export is safe).

A second block in the same file covers the VERIFY-A4 FOLLOW-UP (the
coordinator's rulings on the report's §5): the JAX LG merit's new sigma
branch, `local_quadrature`'s window-truncation warning, `_integrate_levin`
through the shared Van Vleck helper, the cached aberration-free reference,
the `clear_maslov_local_window_cache` re-export, and five pins in
`test_niche_audit_w3_oracles.py` / `test_v5_21_2_subsystem_audits.py`
carried through the audit-Y2 scale move.

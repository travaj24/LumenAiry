# VERIFY-WP-A4 — changelog text (append to `WP-A4_CHANGELOG.md`)

Written in the repository's CHANGELOG voice.  These are the changes VERIFY-A4
made on top of commit `32ba3ba2`.

### Fixed -- optimize: the LG aberration merit is a dimensionless coupling, not a raw `|L|^2`

`LGAberrationMerit` and `make_lg_aberration_merit_jax` reported the bare
`|L|^2` of the LG aberration tensor and documented `1 - |L|^2` as "a Strehl
deficit".  `L` is dimensional: the pure-`(0, 0)` request takes the closed-form
point-sampling branch, which returns `U(chief) * conj(LG_k(0))` — field per
length — and after audit Y2 put the Van Vleck weight `-1j sqrt(|det J|)/lambda`
into the integrand, `|L|^2` reads **4.79e+14** on the stock singlet, so
`1 - |L|^2` was **-4.79e+14**.  Before Y2 it read 3.2e-03 and only *looked*
like a Strehl ratio because the missing `lambda*sqrt(|det J|)` (a LENGTH)
cancelled the branch's 1/length² by dimensional accident.

Both merits now divide every channel by `|L_ref(0, 0)|^2`, the same
coefficient evaluated on the **aberration-free twin** of the same optic — new
`lumenairy.propagators.asymptotic.aberration_free_reference_fit`, which zeroes
every `Phi` coefficient of total degree >= 3 in the pupil variables and leaves
the geometry (`coef_s1x` / `coef_s1y`, hence `|det J|`), the boxes, the
extracted linear ramp and the wavelength untouched.  Every dimensional factor
cancels identically, and an already-aberration-free fit is returned *unchanged*
so the ratio is exactly `1.0` bit-for-bit.  The reference is evaluated at the
twin's own chief-ray landing, so moving the image point off the chief ray
LOWERS the coupling (measured merit `-2.84e-03 / +0.258 / +0.851 / +0.99994` at
`dy = 0 / 20 / 50 / 100 um`).

`LGAberrationMerit` gained `strehl_branch` (default `'sigma'`) and
`sigma_grid_n` (default 64).  The default routes through the sigma-grid
OVERLAP, where the normalised coupling is a real Strehl ratio: measured
`1.000000 / 0.901312 / 0.811431 / 0.659170 / 0.447292` as an f/2.5 N-BK7
plano-convex singlet's cubic-and-higher pupil phase is scaled by
`0 / 0.5 / 1 / 2 / 4`, and `0.815601` (curved side toward the source, 25.8
waves of cubic+ phase) vs `0.731874` (flat side toward it, 39.8 waves) on the
classic orientation pair.  `strehl_branch='closed_form'` keeps the branch the
JAX twin is restricted to; it warns, because a truncated leading-order saddle
expansion does not conserve energy and that ratio RISES with aberration
(`1.000000 -> 1.024914 -> 1.095510` on the same ladder).  The sigma ratio is
insensitive to the grid (identical to 6 significant figures from
`sigma_grid_n` 32 to 256), so the cheap default costs ~1.4 s per field point
against the closed form's ~0.15 s rather than the ~11 s the adaptive grid
would.

**Migration.** Any absolute threshold, weight calibration or logged value tuned
against the old merit must be re-derived: the (0, 0) contribution moves from
`-4.79e+14` (v5.46-pre) / `+0.9968` (v5.45) to a Strehl deficit in
approximately `[-3e-03, 1]`.  A composite merit's RELATIVE channel weights are
unchanged in meaning — every channel is divided by the same per-field-point
constant — so an optimiser converges to the same design; only the printed
numbers and any absolute stopping tolerance change.  The merit now costs two
`aberration_tensor` calls per field point instead of one.  Pass
`strehl_branch='closed_form'` to keep the cheap branch, at the documented cost
that it is not a descent direction.

Tests: `tests/unit/test_audit2609_a4_verify_maslov_asymptotic.py::test_ruling1_*`
(2), and the re-pinned
`tests/unit/test_niche_audit_r_guards_and_merits.py::test_r5_*` (4) /
`tests/unit/test_audit_optimize.py::...::test_piston_weight_scales_merit_linearly`.

### Fixed -- lenses_maslov: `apply_real_lens_maslov_vector` applies ONE joint normalisation to the Jones pair

The wrapper forwarded `normalize_output` (default `'power'`) to the two scalar
legs, which normalised `E_x` and `E_y` INDEPENDENTLY — forcing the output
polarization ratio back to the post-Fresnel input ratio and deleting the
diattenuation the wrapper exists to compute.  Both legs now run at
`normalize_output='none'` and one scale is applied to the pair, as
`apply_real_lens_fga_vector` does.  Measured on an f/3.3 N-BK7 biconvex with a
`P_x/P_y = 1.777777777778` linear input: the propagated ratio is
`1.777777663332023` under `'none'`, `'power'` and `'peak'` alike —
bit-identical, 0 ULP — and it departs from the input ratio by `-6.4376e-08`
(the s/p diattenuation, 5.154e+08 ULP, so the pre-fix behaviour is
distinguishable by 8 decades).  An unknown `normalize_output` now raises with
the Section 2 prefix.

The scale's reference is the POST-FRESNEL pair, i.e. the field the two scalar
legs are actually handed — the literal joint form of the scalar
`normalize_output='power'` contract — so the surface Fresnel transmission
stays in the absolute scale (`T1*T2 = 0.9219035` on the fixture above;
`'power'` restores that to `1.000000000000000` and `'none'` is 28.46x it).
`apply_real_lens_fga_vector` normalises to the RAW input pair instead, under a
lossless assumption; the difference is documented in the wrapper's Notes and
keeps `tests/unit/test_v5_21_maslov_jax_caustic.py::test_maslov_vector_polarization`
— which pins that `T1*T2` survives — passing unchanged.

The polarization base rays are also launched along the input field's own local
wavevector instead of axially (new `_local_direction_cosines` /
`_input_direction_cosines`, the same conjugate-product estimator the FGA
router uses).  A real, non-negative input gives EXACTLY `(0, 0)`, so a
collimated call is bit-identical to the pre-v5.46 behaviour; a 0.03 rad tilt is
recovered as `ux = 0.030000` and an f = 20 mm converging wavefront as `-x/f` to
`4.0e-04 = dx/(2f)`, the forward-difference bias.  The docstring's
"polarization-resolved study through a focus" claim is retracted: there is
still no `E_z` and no exit-frame transport of the Jones vector (audit S10's
remaining sub-item).

Tests: `test_audit2609_a4_verify_maslov_asymptotic.py::test_s10_vector_*` (3).

### Fixed -- lenses_maslov: the S6 saddle warning no longer fires on a collimated beam

The audit-S6 `RuntimeWarning` was gated on the second moment of
`|FFT(E_in)|^2`, which for a collimated beam of finite width is its
DIFFRACTION spread, not a divergence — measured 3-sigma NA `3.54e-03` /
`1.10e-03` for flat-phase Gaussians of waist 0.25 / 0.8 mm at 1.31 um, both
above the `_SADDLE_FLAT_INPUT_NA = 1e-3` threshold, so the warning fired on
every collimated beam narrower than ~1 mm.  It is now gated on the spread of
the input's LOCAL WAVEVECTOR (new `_wavefront_na`), which is EXACTLY zero for a
real, flat-phase field at any width and reproduces the physical number when the
input really is non-flat: `6.00e-03 / 3.00e-02` for tilts of 0.002 / 0.01 rad
and `8.42e-02 / 3.37e-02` for f = -20 / +50 mm, against `6.11e-03 / 3.00e-02`
and `8.42e-02 / 3.37e-02` spectrally.  The message still reports both numbers.

Tests: `test_audit2609_a4_verify_maslov_asymptotic.py::test_s6_*` (7).

### Changed -- tests: two oracles that imported the library helper they verify

`tests/unit/test_niche_audit_w6_asymptotic.py`'s two brute-force quadrature
oracles (`_quad_oracle`, `_a9_quad`) and
`tests/unit/test_audit_propagation.py`'s two inline `amp_lead` references were
changed by WP-A4 to call `lumenairy.propagators.asymptotic.van_vleck_weight`.
An oracle that reuses the library's own weight cannot detect a wrong weight —
the exact failure mode audit Y2 recorded, where every pre-fix pin passed with
`|det J|` to the first power.  All four now derive the weight in the test file
from the stationary-phase Fresnel integral (`-1j sqrt(|det J|) / lambda`), and
a new
`test_w6_verify_van_vleck_weight_matches_the_textbook_fresnel_form` is the one
place the two forms are compared (0 ULP over a 6 x 3 ladder of
`(|det J|, lambda)`).  All 53 W6 tests and all 101 `test_audit_propagation`
tests still pass.

### Changed -- validation: the JAX real-lens cases enable x64 for their process

`validation/run_all.py test_lenses` went red on three of the four
`apply_real_lens_traced_jax` / `apply_real_lens_maslov_jax` cases when audit
S7's x64 contract landed (`RuntimeError: ... requires double precision`).  The
three cases now call `jax.config.update('jax_enable_x64', True)`, the one-liner
the fourth case and the other JAX validation cases already use.  The refusal
itself stays pinned in
`tests/unit/test_audit2609_a4_maslov_gbd.py::test_s7_*`.
`validation/run_all.py test_lenses` is **46/46 passed** (27.7 s).

### Changed -- tests: `test_a1_auto_n_v2_resolves_demanding_default_quadrature` restated

Audit S2 changed the `'auto'` router: on a demanding tight-focus chart the
router reports `need n_v2 ~ 2248` against a cap of 256 and `'auto'` resolves to
`'stationary_phase'`, so `n_v2` is inert and the default call is
**byte-identical** to the historical `n_v2=32`.  The test's fail-before arm
(`il2(old32, truth) > 0.5`, measured 0.67 pre-S2) therefore measured
`1.325e-04`.  It is restated — not deleted — as the property that now holds:
both defaults track the well-resolved `local_quadrature` reference the test
already carried to `il2 = 1.325e-04` (bar 5e-3), `'auto'` is no worse than the
fixed default, and the two are asserted byte-identical so a router change
re-opens the question.  A dated one-off against a converged uniform quadrature
(`n_v2 = 512`, self-converged to `il2 = 2.14e-06` vs `n_v2 = 384`) is recorded
in the docstring — both defaults `5.844e-04`, the `local_quadrature` reference
`5.754e-04` — rather than re-run, because that oracle costs ~40 min on this
fixture.  The restated test runs in 1.64 s.

### Fixed -- optimize: the LG merit says so when its aberration-free reference collapses

The reference is a reference SPHERE only while the pupil phase it removes is a
perturbation.  On the validation suite's own fixture (51.5 mm N-BK7 singlet,
`object_distance = 200 mm`, `pupil_box_half = 0.02`) the fit carries
**4.206e+05 waves** of cubic-and-higher pupil phase; zeroing it builds a
different optic whose focus leaves `s2_image`, and the reference coupling
collapses (`|L_ref(0,0)|^2 = 6.16e-10` against `|L(0,0)|^2 = 0.799`, i.e. a
coupling of 1.30e+09).  `LGAberrationMerit` now emits one `RuntimeWarning`
naming the removed waves and stating that the channels are mutually consistent
and monotone but NOT Strehl-normalised on that chart.  Silent on a well-posed
chart (measured coupling 1.0029).  `validation/run_all.py test_asymptotic` is
48/48 passed (62 s).

### Fixed -- docs: `AberrationTensorResult.van_vleck_weight` round-trip

The field's docstring said `L_legacy = L * |det J| / van_vleck_weight`, "i.e.
`L * lambda**2 * |det J|` in magnitude".  The first form is right; the
restatement is the factor for `|L|^2`, not for `L` — in magnitude
`L_legacy = L * lambda * sqrt(|det J|)`.  Corrected.

### Changed -- tests: three wall-clock assertions replaced by operation counts, one slow marker

TESTING_STANDARDS S1, from WP-A15a's section 5 item 8.

* `test_audit_propagation.py::…::test_fused_path_does_the_fused_work` --
  `assert t_fused < 60.0` retired.  Its stated purpose (catch an accidental
  O(N^2), a dropped chunking, a per-pixel Python loop) is already covered by
  the per-chunk `np.exp`/`np.einsum`/`np.sum` counts the test carries; the one
  thing they missed -- GRID scaling -- is now a count too: the heavy-call
  counts on a DOUBLED grid must be identical.  Timing still printed.
* `test_audit_propagation.py::…::test_decompose_lg_cache_speedup` ->
  `::test_decompose_lg_cache_builds_no_modes_on_a_hit`.  The `>= 5x` floor
  (whose own docstring called it "a soft speedup floor ... timing on CI hosts
  is noisy") becomes an exact count on `asymptotic_modes._evaluate_poly2d`:
  **28** polynomials cold (`(p_max+1)(2 ell_max+1)` at `p_max = ell_max = 3`),
  **0** cached, **28** again after `clear_lg_mode_stack_cache()`, plus the
  v5.30 W6-A13 identity contract (the hit returns the SAME array).
* `test_audit_optimize.py::…::test_dual_annealing_terminates_quickly_when_cancelled`
  -- `assert dt < 180.0` retired; the two merit-evaluation-count assertions
  above it already pin "not stuck", exactly and machine-independently.
* `test_audit_lens_models_2026_07.py` gains `pytestmark = pytest.mark.slow`
  (641.0 s on the committed `.test_durations`; 206.7 s on this box after the
  ruling-2 restatement, still over the 2 min/file bar).  Verified: `-m "not
  slow"` deselects all 69 ids.

### Added -- `lumenairy.propagators.asymptotic.aberration_free_reference_fit`

Public helper returning the aberration-free twin of a `CanonicalPolyFit` (see
the first entry).  Idempotent, and returns the *same object* when there is
nothing to remove, so a caller that divides by the reference gets exactly 1.0.


---

## VERIFY-A4 follow-up (coordinator rulings on the report's section 5)

### Added -- optimize: `make_lg_aberration_merit_jax` gets a sigma-grid branch

The JAX LG merit was restricted to the closed-form point-sampling branch,
which is a leading-order saddle value and therefore not a descent direction
even after the aberration-free normalisation -- its coupling RISES with
aberration.  It now has the same `strehl_branch` choice as the NumPy
`LGAberrationMerit`, defaulting to `'sigma'`: a new
`_lg00_sigma_overlap_jax` builds the sigma grid, `jax.vmap`s
`solve_envelope_stationary_jax_ift` + `_modal_field_lg00_pixel_jax` over it
and projects onto LG_{0,0} analytically, all in `jnp` and differentiable.

It does NOT have to reproduce NumPy's `_measure_image_plane_waist` probe or
its adaptive `sigma_grid_n` ladder: the merit uses only the RATIO
`|L|^2/|L_ref|^2` and the basis waist, extent and grid are identical on both
sides of it, so they cancel.  MEASURED on an f/2.5 N-BK7 plano-convex
singlet (w_s = 20 um, w_p = 0.05, on-axis): JAX coupling **1.002678** against
the NumPy sigma branch's **1.002871**, i.e. 1.92e-04 relative, with the
residual identified as the basis-waist convention (moving the JAX `w_frac`
0.25 -> 0.5 moves that side by 6.3e-04 while `n_grid` 16 -> 32 moves it by
3e-09).  `jax.grad` through the whole thing is finite
(`d(1-S)/dw_s = -3.10e+01`).  Cost 11.7 s first call (XLA compile), 1.4 s
warm.  `strehl_branch='closed_form'` keeps the old path and warns.
`tests/unit/test_niche_audit_r_guards_and_merits.py::test_r5_numpy_lg_merit_matches_jax_twin`
now compares sigma to sigma at `abs=2e-03` on the coupling (2.9x the worst
measured 6.97e-04, 2.5 decades below an x-vs-(1-x) confusion).

### Fixed -- lenses_maslov: `local_quadrature` says when it truncates its window

The Gaussian-taper correction is computed on the FULL lattice while
out-of-box samples are DROPPED, so the scheme's exactness on a quadratic
chart holds only while the lattice fits inside the fitted Chebyshev box.
New `_warn_local_window_truncation` (NumPy and CuPy integrators both) emits
one `RuntimeWarning` naming the dropped fraction above a 1 % trigger.
MEASURED on an anamorphic quadratic chart against the closed-form Fresnel
value: 0/64 dropped and relerr 6.60e-15 at `window_sigma = 3.0`, 0/1089 and
3.41e-15 at 2.5; **44/121 (36.4 %) and 8.09e-02** at 5.0, **78/169 (46.2 %)
and 8.79e-02** at 7.5 -- and 8.17e-01 at the shipped defaults on a chart
whose small Hessian eigenvalue puts `sigma2_norm` at 1.785.  The module
comment and the function docstring gain the "while the tapered lattice fits
inside the chart box" qualifier.

### Changed -- lenses_maslov: `_integrate_levin` uses the shared Van Vleck helper

Both Levin integrand closures wrote `sqrt(|det J_norm|)` out by hand -- the
duplication `_van_vleck_density` exists to prevent.  They call it now, in the
association that is BIT-IDENTICAL (`np.array_equal` over 1e5 samples x 3
anamorphic half-width pairs): `_van_vleck_density(d, 1, 1)` is `d ** 0.5`,
NumPy's `** 0.5` is `sqrt` bit-for-bit, and the box Jacobian `sqrt(hx*hy)`
is applied as before.  Unit half-widths are correct here because the Levin
engine works in the normalised unit box end to end.

### Performance -- optimize: the LG merit's aberration-free reference is cached

`LGAberrationMerit` takes the reference from `ctx._canonical_fit_cache`,
keyed by `('lg_ref', <the fit's own key>, source_point, w_s, w_p, w_o,
strehl_branch, sigma_grid_n)`, and requests it on a FIXED minimal mode set
rather than the term's own `output_modes` -- only its `(0, 0)` entry is read,
and pinning it makes every channel of every term on one optic divide by the
same constant (which is what lets a `CompositeMerit` compare its channels)
as well as letting one cached reference serve them all.  MEASURED, three
terms on one fit (medians of 3): **4.365 s -> 2.807 s**, i.e. 1.455 ->
0.936 s per term, with exactly ONE `lg_ref` cache entry.  Values unchanged
(composite 1.829191e-01 before and after).  No timing is asserted anywhere.

### Added -- `clear_maslov_local_window_cache` is re-exported at top level

Added to `lenses_maslov.__all__` and imported / listed in
`lumenairy/__init__.py`, which is what
`test_v4_14_1_dispatcher_pin_cache_clears` requires of every
submodule-`__all__` `clear_*` name.  The cache was already enrolled with
`_cache_registry`, so `clear_asm_caches()` behaviour is unchanged.

### Changed -- tests: five pins carried through the audit-Y2 scale move

`tests/unit/test_niche_audit_w3_oracles.py` (x4) and
`tests/unit/test_v5_21_2_subsystem_audits.py` (x1) were red at `32ba3ba2`
from WP-A4's own Y2 change, in files its report did not list.  Re-pinned by
DERIVATION, not by re-baking: `L_new = L_old * (-1j)/(lambda sqrt|det J|)`
and `|L|^2_new = |L|^2_old / (lambda^2 |det J|)`, with `|det J|` read off
`AberrationTensorResult.van_vleck_weight`.  Measured at lambda = 1.31 um:
`|det J| = 3.908349e-02 / 3.931805e-02` on the two validation designs, so
the factor is 1.490953e+13 / 1.482059e+13.

* the closed-form constants `15.448+3.188j` / `-6.571-14.392j` become
  `1.2310011487876e+07-5.9649449469122e+07j` /
  `-5.5405030010922e+07+2.5295326982543e+07j`, and the test now ALSO asserts
  the identity that produced them (measured 3.6e-14 / 1.8e-13 relative);
* the sigma anchors `9.0968975e-14 / 7.1975598e-14` become
  `1.3564605658e+00 / 1.0671876291e+00` (measured ratios 1.491124e+13 /
  1.482708e+13, i.e. the predicted factor to 1.1e-04 / 4.4e-04 -- the
  residual is `det J`'s variation over the sigma grid);
* the LG-merit curvature-response values become 2.2006679213e+09 /
  9.9881936194e+04 with response 9.999546e-01 (the 2e-2 relative tolerance
  is UNCHANGED), and the docstring records that this fixture trips the
  reference-collapse guard;
* `test_opt1_lg_jax_merit_is_strehl_deficit_not_amplitude` keeps its claim
  but changes its driver: a waist mismatch is not an aberration and now
  cancels against the reference, so the test walks the IMAGE POINT off the
  chief ray instead -- merit -3.19e-03 / +1.70e-01 / +6.94e-01 / +9.91e-01
  at dy = 0 / 20 / 50 / 100 um, pinned as strict monotonicity plus
  `v(100 um) > 0.5` and `|v(0)| < 0.05`.

No bar was loosened; every changed one carries its derivation.

### Added -- tests: the mirror exit vertex and Y1 at a second field angle

`tests/unit/test_audit2609_a4_verify_maslov_asymptotic.py` gains six tests:
the O-1 truncation warning (fires / silent, with the fraction in the
message), the O-2 bit-identity, the mirror-last-surface exit vertex
(`n_exit = n(glass_before) = 1.503583` applied to 0.0; the "air" mistake
would be 7.783e-06 m = 5.9 waves), and Y1's chief ray at THREE field angles
-- (100, 0), (0, 150) and (-70, 70) um, which split the extracted ramp
`|a3| / |a4|` as 2.74e+03 / 2.6e-10, 8.5e-10 / 4.00e+03 and 1.89e+03 /
1.89e+03, so the `a4 u4` half of the Y1 fix is exercised for the first time.
PSF within 1.20 / 1.80 / 1.21 um of the independently traced chief ray on a
92.3 um pitch.

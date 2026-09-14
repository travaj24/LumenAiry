# WP-B7 changelog text (the asymptotic family: Y4 / Y5 / S9 + §15.9)

Release 5.47.0.  Findings **Y4**, **Y5** and **S9** of
`AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11` (WP-A4 report §6 items 3–8),
§15.9 for the uniform path, and VERIFY-B1's follow-ups **F1** and **F2**.

---

### Fixed -- Maslov propagator: the pupil chart is sized from the MEAN launch direction plus the SPREAD (VERIFY-B1 F2)

`apply_real_lens_maslov` sized its pupil chart as
`na_proxy = na_lens + 3 * sqrt(<v^2>)`, the second angular moment of
`|FFT(E_in)|^2` **about zero**.  For a uniform tilt `theta` that moment IS
`theta`, so the chart was sized to `na_lens + 3 theta` -- three times the
launch angle.  A tilt is a change of reference direction, not an angular
spread, and the chart is a box about `v = 0`, so what it has to reach is
`|mean| + 3 sigma_about_mean`.  The over-sized box is what the order-`poly_order`
entrance-coordinate fit then had to span, and above ~2x the lens NA it could
not: the S6 saddle rode on a chart that had misplaced the very coordinates the
term is contracted against, and `_S1_FIT_RESIDUAL_MAX` (VERIFY-B1 V1) refused
it, leaving the OPD-only answer.

The driver now measures the first angular moment as well and uses
`|mean| + 3 sigma_about_mean` whenever the mean is a real launch direction --
above `_NA_MEAN_MIN_FRACTION = 0.1` of the spread.  Below that bar the old
arithmetic runs verbatim, which is what keeps a centred input byte-identical:
the two forms are different float64s even at a mean of 1.7e-14.

* `lumenairy/elements/lenses_maslov.py` (`_NA_MEAN_MIN_FRACTION`, the
  `na_input` block, the S6 docstring and the chart half of the S6 warning).
* Measured on the f = 6 mm N-BK7 / 1.0 um chart and an f = 14.10 mm N-SF11 /
  1.55 um one, fidelity against the EXACT pointwise `'quadrature'` on the same
  chart (`stationary_phase` / `local_quadrature`):

  | chart, tilt | `na_input` | s1 fit residual | fidelity |
  |---|---|---|---|
  | A, 1.5x NA | 0.2239 -> 0.0791 | 1.29e-03 -> 6.1e-05 | 0.912/0.985 -> 0.922/0.984 |
  | **A, 2.0x NA** | 0.2986 -> **0.1040** | 3.98e-03 -> **1.17e-04** | **0.000/0.000 -> 0.911/0.979** |
  | **A, 4.0x NA** | 0.3405 -> **0.1180** | 6.90e-03 -> **1.63e-04** | **0.000/0.000 -> 0.864/0.959** |
  | **B, 4.0x NA** | 0.4257 -> **0.1461** | 1.97e-02 -> **2.34e-04** | **0.000/0.000 -> 0.703/0.858** |

  The bar is the geometric mean of a measured bracket: the largest
  `mean / sigma` a field with NO launch direction produces is **1.5e-02**
  (a uniform white-noise phase screen; a hard-edged aperture on an even grid
  reads 9.3e-05, a centred Gaussian 1.1e-11), and the smallest a field that
  HAS one produces is **5.5e-01** (a 1 mrad tilt, itself at
  `_SADDLE_FLAT_INPUT_NA`).
* **Migration.**  A call whose input carries a real mean launch direction --
  a tilted or decentred-source beam -- now builds a smaller, better-conditioned
  pupil chart and returns a different (more accurate) field.  Collimated,
  converging, diverging, speckled-about-zero and hard-apertured inputs are
  byte-identical.  To reproduce a 5.46 number on a tilted input, pass the old
  sizing explicitly: `input_na = 3 * sqrt(<v^2>)` of that field's own angular
  spectrum.
* Tests: `tests/unit/test_audit2609_b7_asymptotic.py` §5 (6 ids).

### Fixed -- Maslov propagator: the S6 fallback scores the fitted wavevector's SLOPE, not only its value (VERIFY-B1 F1)

The S6 term the two asymptotic evaluators add is `k1 . ds1/dv2`, and the
saddle's Newton consumes that term's `v2`-DERIVATIVE in its Hessian.
`_K1_FIT_RESIDUAL_MAX` scores the `k1` fit's VALUE residual, which is bounded
for a field whose fitted derivative is not: a degree-4 fit of a hard-edged
aperture -- where `_local_direction_cosines` reports a launch direction of 0 in
the dark and the true wavefront in the light -- has a value residual of
0.08-0.26, comfortably inside the 0.5 bar, and a slope error two decades worse
than any speckle.  Those inputs engaged the fitted saddle and were made worse
by it.

A second bar now scores the slope.  `k1` is refitted at `poly_order - 1` --
whose basis is the column subset of the SAME design matrix with total degree
below the cap -- and the two charts' `(dk1/du3, dk1/du4)` are compared at the
ray points, intensity-weighted and normalised by the RMS of `k1` itself so a
uniform tilt (constant `k1`, zero slope in both fits) scores 0 rather than 0/0.

* `lumenairy/elements/lenses_maslov.py` (`_K1_DERIV_RESIDUAL_MAX = 1.2`,
  `_k1_fit_derivative_error`, the fallback gate, a new `_why` branch, the
  progress line and the docstring).  `_K1_FIT_RESIDUAL_MAX = 0.5` is unchanged
  -- the statistic it scores has not changed.
* Measured on two charts, fidelity against the exact pointwise `'quadrature'`
  on the same chart, OPD-only saddle -> engaged:

  | input | value residual | **slope error** | engaging |
  |---|---|---|---|
  | clean tilt 0.5..4x lens NA | 1e-11 .. 1e-14 | **1.2e-10 .. 2.8e-08** | wins |
  | converging / diverging | 4e-06 .. 1.3e-05 | 2.0e-06 .. 5.6e-06 | wins |
  | speckle 0.002 .. 0.100 rad rms | 2.4e-03 .. 1.7e-01 | 4.8e-03 .. **5.6e-01** | wins |
  | **hard edge at 0.80 of the pupil** | 7.9e-02 | **2.8e+00** | **0.016 -> 0.000 LOSES** |
  | hard edge at 0.95 / 0.60 | 1.5e-01 / 2.6e-01 | 4.3e+00 / 3.9e+00 | LOSES |

  The bar is the geometric mean of the two-chart bracket 5.6e-01 (last input
  where engaging wins) .. 2.7e+00 (first where it loses) = 1.24.  Every case
  that engages today and wins still engages.
* Cost: one extra least-squares solve against a narrower column subset of the
  existing design matrix, plus two term-by-term accumulations over the traced
  rays, on the ENGAGED path only.  Nothing that does not engage pays anything.
* **Migration.**  A hard-edged or heavily speckled input above the new bar now
  keeps the OPD-only saddle and emits the S6 `RuntimeWarning` naming the
  mechanism and `integration_method='quadrature'`.  Pass
  `input_wavevector_saddle=True` to force the previous behaviour.
* Tests: `tests/unit/test_audit2609_b7_asymptotic.py` §6 (1 id).

### Fixed -- `apply_real_lens_maslov_jax` carries the chief-ray displacement of a non-collimated input

Despite its name this entry point is a thin-OPD geometric phase screen plus a
Maslov / Gouy index term, not the phase-space diffraction integral of the NumPy
`apply_real_lens_maslov`; it has no stationary-point solve, so audit S6 does not
apply to it (WP-B1 report §6 item 6 and §7 item 5 are struck; VERIFY-B1 §7).
Its own defect is elsewhere: the screen's OPL is indexed by the ENTRANCE point
of the ray that lands on each output pixel while `E_in` is sampled at the
output pixel, and for a non-collimated input those are different points,
because a ray walks across the element.  The input's own phase has to be
re-referenced to the entrance point.

The first-order term `k0 * k1 . (entrance - pixel)`, with
`k1 = (1/k0) grad arg E_in`, is now added.  It is EXACTLY zero for a real
non-negative input, so a collimated field is byte-identical.

* `lumenairy/elements/_lens_jax.py` (`_local_direction_cosines_jax`, the
  `input_wavevector_saddle` keyword and the phase assembly).
* Measured on an f = 14.10 mm N-SF11 singlet at 1.55 um, screen then
  `angular_spectrum_propagate` to a plane 0.30 mm past the focus, intensity
  centroid against an exact conic raytrace of the input's own ray fan:

  | input | oracle | screen | corrected |
  |---|---|---|---|
  | collimated | 0.000 um | -0.000 (0.00 %) | -0.000 (**byte-identical**) |
  | tilt 0.25x lens NA | 120.243 um | 117.041 (**-2.66 %**) | 120.013 (**-0.19 %**) |
  | tilt 0.50x | 240.511 um | 234.109 (**-2.66 %**) | 240.056 (**-0.19 %**) |
  | tilt 1.00x | 481.227 um | 468.143 (**-2.72 %**) | 479.947 (**-0.27 %**) |

  13.1 um at one lens NA is about 1.6 diffraction-spot radii on this optic.
* `input_wavevector_saddle=` takes the same three values, with the same
  meaning, as `apply_real_lens_maslov`'s keyword, so a caller can switch
  backends without changing which ray the answer is built on -- with the
  caveat, stated in the docstring, that this path has no saddle and what the
  keyword selects here is the displacement term.
* **Migration.**  A non-collimated input through this entry point returns a
  different (more accurate) field.  `input_wavevector_saddle=False` reproduces
  the 5.46 screen exactly.
* Tests: `tests/unit/test_audit2609_b7_asymptotic.py` §6b (1 id, JAX-gated).

### Performance -- the batched asymptotic kernels evaluate one Chebyshev basis per sweep instead of three (audit Y4)

`_compute_M_b_batch` contracted `coef_s1x`, `coef_s1y` and `coef_phi` against
the same point set through two `CanonicalPolyFit` methods that each rebuilt the
`(M, N)` basis tensors for themselves, then called `_phi_v2_hessian_batch`,
which rebuilt them again.  `_solve_envelope_stationary_batch` rebuilt the
basis's `s2`-only factor on every Newton sweep even though the iteration moves
`v2` alone.

`_basis_and_grad34` now builds `basis_f` / `basis_d3` / `basis_d4` once and
`CanonicalPolyFit.eval_s1_and_phi_with_v2_grad` contracts all three vectors
against them, optionally handing back `T1[K1] * T2[K2]` so the Hessian pass
reuses it; the Newton hoists that factor out of its loop and gathers the active
columns.

* `lumenairy/propagators/asymptotic_canonical_fit.py`,
  `lumenairy/propagators/asymptotic_maslov.py`.
* **Byte-identical**, proved archive-to-archive (`git archive b2baa505`
  vs this tree, both imported from child processes with `lumenairy.__file__`
  asserted): `np.array_equal` on all twelve of `v2x_star`, `v2y_star`,
  `converged`, `M`, `b`, `s1*`, `J`, `phi*`, `G0`, `detJ`, `H_phi` and the
  `propagate_modal_asymptotic` field.  Each output is still ONE
  `np.tensordot` of one coefficient vector against one basis tensor -- the
  single stacked `(3, M) @ (M, P)` GEMM the audit suggested was not taken,
  because a GEMM is entitled to reorder the reduction against a GEMV.
* Measured (stock N-BK7 singlet fit, order 6, M = 210, 41x41 raster,
  `w_s = 20 um`, `w_p = 0.02`, single-threaded, interleaved medians):
  `_solve_envelope_stationary_batch` **271.1 -> 145.9 ms** (1.86x),
  `_compute_M_b_batch` **50.0 -> 20.7 ms** (2.41x),
  `propagate_modal_asymptotic` **320.7 -> 191.0 ms** (1.68x); an earlier
  pre-and-post pair on a quieter box gave 1.78x / 2.88x / 1.58x.  Chebyshev
  table builds per call -- the build-free statement of the same claim --
  **26 -> 12** and **144 -> 50**.
* Tests: `tests/unit/test_audit2609_b7_asymptotic.py` §1 (3 ids), which assert
  `array_equal` and the operation counts -- no wall clock.
* **Private protocol.**  `_solve_envelope_stationary_batch` now reads the
  fit's basis directly (`basis_index_columns`, the box centres and
  half-ranges, `poly_order`, `coef_s1x` / `coef_s1y`) and no longer calls
  `eval_s1_with_v2_grad` per sweep, so the `fit` argument has to be a
  `CanonicalPolyFit` as the signature has always said -- a duck-typed
  object offering only that method raises `AttributeError`.  The two
  P1-NEW-3 contract tests in `tests/unit/test_audit_propagation.py` were
  such objects (found by WP-B11b's wide sweep after this package landed);
  they are rebuilt as genuine order-1 fits on unit boxes, whose
  coefficient on `u3` / `u4` is the Jacobian column and whose constant
  term is `s1` at the cold start, and a third test pins that construction
  through the fit's own evaluator.

### Added -- `_solve_envelope_stationary_batch(scale_relative_stop=)`, opt-in (audit Y4)

The convergence VERDICT has been scale-relative since v5.30 (the residual is
dimensional and O(1e7) on the library's own default waists, so the documented
`tol = 1e-12` was unreachable); the STOP was still the absolute test, so every
pixel ran all `max_iter` sweeps taking round-off Newton steps after it had
converged.  `_NEWTON_SCALE_RELATIVE_STOP` (module seam) and
`scale_relative_stop=` (per call) opt the stop into the same test.

It ships OFF because it MOVES THE ANSWER: a pixel that leaves the active set
early keeps the converged iterate rather than the one twelve round-off steps
later.  Measured on the same fixture: 1.23x on the Newton, 12 -> 11 sweeps,
`max |dv2|` 2.2e-11 on a 0.0851 pupil half-range, and the
`propagate_modal_asymptotic` field moves 9.1e-11 relative L2 -- under the 3e-8
the shipped bit-equality pins carry, which is precisely why it is a seam and
not a default.

* `lumenairy/propagators/asymptotic_maslov.py`.
* Tests: `tests/unit/test_audit2609_b7_asymptotic.py` §2 (1 id), two-sided.

### Performance -- `aberration_tensor` builds only the modes it reads, and memoises its image-plane waist probe (audit Y4)

`decompose_lg` built the whole `(p_max, ell_max)` rectangle -- the enclosing box
of the caller's mode list, 21 modes for a `(2,0) / (1,1) / (0,3)` selection that
reads 3 -- and `_measure_image_plane_waist` re-ran its coarse propagate on every
call that shared a fit, an image point and a pupil weighting.

* `decompose_lg(..., only=...)` and `_lg_mode_conj_stack(..., only=...)` build
  and return only the requested `(p, l)` pairs, in the same canonical order,
  with the set in the cache key; `aberration_tensor` passes its `output_modes`.
* `_measure_image_plane_waist` is memoised on
  `(fit fingerprint, s2_image, source_point, pupil_amplitudes, w_s, w_p,
  v2_centre, n, propagate)` -- the audit's proposed key omitted `source_point`
  and `pupil_amplitudes`, both of which reach the probe's own propagate call and
  change the width it measures.  The fit enters by a content fingerprint of its
  coefficient vectors, not by `id()`, which CPython reuses after collection.
  Bounded at 64 entries, FIFO, drained by `clear_image_plane_waist_cache()` and
  by `clear_asm_caches()` through the central registry.
* `lumenairy/propagators/asymptotic_modes.py`,
  `lumenairy/propagators/asymptotic_aberration_tensor.py`.
* Measured (validation singlet, order 6, 8^4 rays, six output modes spanning a
  21-mode rectangle, adaptive `sigma_grid_n` = 256): `aberration_tensor`
  **12 203 -> 7 439 ms** (1.64x) with `L`, `w_o` and `sigma_grid_n`
  **bit-identical** (max |dL| = 0.0).  93 % of the pre-change call was the two
  kernels the Y4 fusion above addresses; the mode-stack and waist-probe savings
  are the remainder.
* The adaptive `sigma_grid_n` cap stays at 256.  Measured: 512 is four times
  the sigma-grid pixels, i.e. ~30 s per call even after the fusion, so it is
  not affordable as a default; the existing warning keeps naming the `n` the
  chirp actually needs.
* Tests: `tests/unit/test_audit2609_b7_asymptotic.py` §3 (2 ids).

### Performance -- GBD FFT reconstruction clips its Gaussian kernel to the beamlets' own support (audit S9)

`_reconstruct_fft` built its kernel over the FULL `(2Ny-1, 2Nx-1)` linear
convolution offset range whatever the beamlet's actual decay, so the transform
was `(3Ny-2, 3Nx-2)` -- nine output grids per array with several alive, and no
cap: measured **37.5-37.7x the output-grid bytes at every N**, i.e. 158 MB at
N = 512 and the ~9.7 GB the auditor projected at N = 4096.

The kernel is now clipped to `+-ceil(R_cut / d)` per axis with
`R_cut = n_sigma / sqrt(alpha)` and `alpha = -0.5 k Im(Q)` -- the same
amplitude-decay coefficient `_reconstruct_windowed` computes, per axis here
because the applicability gate has already refused a skew `Q` -- and
`_fftconv_same` pads each transform to `scipy.fft.next_fast_len`.
`_FFT_KERNEL_N_SIGMA = 6.5`, wider than the windowed path's 5.0 because one
kernel serves the whole bundle: `exp(-6.5^2) = 4.5e-19` puts the truncation two
decades below float64 eps relative to the kernel peak.

* `lumenairy/propagators/gbd.py` (`_FFT_KERNEL_N_SIGMA`,
  `_kernel_half_width`, `_fft_len`, `_fftconv_same`, `_reconstruct_fft`).
* Measured (`tracemalloc` peak over output-grid bytes; beamlets decomposed at
  `waist_factor = 1`, so the propagation distance selects the regime):

  | N | z | kernel | peak xgrid | reconstruct |
  |---|---|---|---|---|
  | 512 | 0.05 mm | 10 / 511 | **37.73 -> 6.14** | **594.5 -> 47.3 ms** |
  | 512 | 0.50 mm | 65 / 511 | 37.73 -> 7.75 | 633.6 -> 55.2 ms |
  | 512 | 8.00 mm | 511 / 511 | 37.73 -> 37.80 | 491.2 -> **338.3 ms** |
  | 256 | 0.05 mm | 10 / 255 | 37.65 -> 6.42 | 140.1 -> 9.1 ms |
  | 128 | 0.05 mm | 10 / 127 | 37.50 -> 7.00 | 30.6 -> 2.1 ms |

  The last row of the 512 block is the other half of the contract: a beamlet
  that has genuinely spread to fill the grid keeps the full kernel and pays
  what it always did in memory (+0.2 %), while `next_fast_len` alone still buys
  1.45x on the transform.
* NOT byte-identical: the clip drops a tail and the transform length changes.
  Derived tolerance, measured against the windowed scatter-add at
  `n_sigma = 5 / 7 / 9` (whose own truncation at 9 is `exp(-81)`): **8.2e-16 ..
  2.3e-15** relative L2 in the clipped regime -- and the reading does not move
  with `n_sigma`, which is the proof that what remains is the transform's own
  round-off.  In the unclipped regime the field moves **5.6e-16 .. 8.3e-16**
  against the pre-change path.  For scale, the pre-change path agreed with the
  windowed sum at `n_sigma = 8` to 4.3e-15 .. 1.5e-14.
* Tests: `tests/unit/test_audit2609_b7_asymptotic.py` §4 (5 ids), two-sided on
  both the clipped and the full-support regime.

### Documentation -- `apply_real_lens_fga`'s analytic-Jacobian whitelist (WP-B9 request 4)

WP-B9 gave `ray_transfer_jacobian_analytic` even-power aspheric support.
`propagators/gbd.py`'s `jacobian='auto'` picks that up with no code change,
because it dispatches on the primitive's own `NotImplementedError`; `fga.py`'s
`_pick_ray_transfer` does not, because it gates on `_is_all_conic`, a whitelist
that still excludes `aspheric_coeffs`.  The three comments that said the
analytic form "does not handle" aspherics are corrected to state what the code
actually does -- the whitelist is FGA's, not the primitive's -- and point at
the `gbd` dispatch for the class it now covers.  No routing moved.

* `lumenairy/propagators/fga.py` (`_is_all_conic`, `_pick_ray_transfer`,
  `apply_real_lens_fga`'s `exact_jacobian` docstring).
* Tests: `tests/unit/test_audit2609_b7_asymptotic.py` §7 (1 id) pins that an
  aspheric prescription reaches the analytic primitive (it no longer raises),
  that a biconic one still does not, and that `gbd`'s `'auto'` orders its
  candidates analytic-first.
* **Migration (against the escalated widening, NOT this release).**  If
  `_is_all_conic` is later widened to the primitive's real coverage, an
  aspheric prescription's `apply_real_lens_fga(exact_jacobian=None or True)`
  switches from the finite-difference Jacobian to the exact analytic one: the
  two agree to better than 1e-6 relative on an A4 singlet (the FD central-
  difference truncation floor) and the analytic side is exact, so the change
  raises accuracy and drops the trace count 9N -> N.  Pass
  `exact_jacobian=False` to keep the FD answer.

### Investigated -- FGA vs `phase_screen` at NA 0.145: FGA does not converge (WP-A4 §6 item 3)

Run against a brute-force Rayleigh-Sommerfeld oracle built from an exact conic
raytrace of the input's own rays, converged to 0.3 % in the pupil sampling, on
the audit's own fixture (f = 1.2 mm biconvex N-BK7, 0.30 mm aperture,
NA 0.1452, focus 1.027 mm past the exit vertex, lambda = 1.0 um).

A fifteen-point sweep of `w0_factor` / `dq_step` / `p_max` / `n_p` moves the
FGA spot between 10.57 and 12.83 um against the oracle's 2.656 um and
`phase_screen`'s 3.169 um; `dq_step` and `n_p` are inert to four digits, so it
is not a sampling error.  Scored as a field: fidelity **0.3234** (defaults),
**0.3826** (best setting), against **0.9965** for `phase_screen`, with 92 % of
FGA's energy outside the 5 um core that holds 95 % of the oracle's.  Across an
NA sweep from 0.039 to 0.192, `phase_screen` is the closer member at EVERY NA,
by 10x to 46x, while the router flips to `fga` at NA 0.145 through its caustic
gate.

No default moved in this release.  The conclusion -- that this is an accuracy
defect in the `fga` member rather than a mis-set `na_threshold`, and that
`_universal_route`'s caustic branch should stop preferring `fga` over
`phase_screen` for a SINGLE-VALUED input -- is specified with its exact edit in
the WP-B7 report §8.3, because the change reds route pins in five test files
outside this package and belongs in one reviewed commit of its own.

### Investigated -- §15.9, the uniform asymptotics: the premise is out of date and the fold oracle now exists

`uniform_fold_airy` and `pearcey` are **not dead code**:
`lumenairy/elements/_lens_traced_uniform.py` imports `_fold_airy_eval` (the
closing Chester-Friedman-Ursell expression of `uniform_fold_airy`) and
`pearcey` and uses them for `apply_real_lens_traced(caustic='uniform')`, under
`tests/unit/test_niche_k4_uniform_caustic.py` and
`test_niche_r2_pearcey_cusp.py`.  What is unwired is a uniform path for the
Maslov `v2` integral.

The auditor's Probe 3 has now been run: an f/1.92 N-BK7 singlet whose marginal
focus sits 183 um inside its paraxial one -- a genuine fold -- read against a
brute-force Rayleigh-Sommerfeld oracle converged to `1 - fidelity = 4.6e-08`.
`apply_real_lens_maslov('stationary_phase')` scores **0.5921** there, the exact
pointwise `'quadrature'` on the SAME chart **0.6278**, and
`apply_real_lens_traced(caustic='uniform')` **0.0030**.  The saddle costs 0.036
of fidelity against the exact integrator; the canonical CHART costs the other
0.37, and raising it from order 6 to order 8 does not help.  A uniform Maslov
evaluator therefore cannot reach the oracle at this fold, so
`integration_method='uniform'` is left unwired and the table is published
(WP-B7 report §6).

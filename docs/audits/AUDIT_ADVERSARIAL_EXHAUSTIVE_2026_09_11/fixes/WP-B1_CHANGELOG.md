# WP-B1 changelog text (release 5.47.0) -- the Maslov asymptotic saddle follows the input field

Finding **S6** of `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11` (report §2.5), the
half WP-A4 deferred with a design in its report §6 item 1.  WP-A4 shipped a
`RuntimeWarning`; the saddle itself was unchanged.

---

### Fixed -- Maslov `stationary_phase` / `local_quadrature`: the saddle is the stationary point of the FULL integrand, not of the OPD alone (audit S6, P1)

The v2 integrand is
`E_in(s1(s2, v2)) |det ds1/dv2|^(1/2) exp(2 pi i OPD_waves)`, so the phase that
is stationary in `v2` is the TOTAL `arg E_in(s1(v2)) + 2 pi OPD_waves`.  Both
asymptotic evaluators solved `grad_v2 OPD = 0`, and the symplectic identity
`dOPD/dv2 = -n1 (v1 . ds1/dv2)` makes that the `v1 = 0` on-axis collimated
launch ray at **every** pixel and for **every** input -- while the driver sizes
the pupil chart specifically to cover a diverging / converging / tilted one
(`na_proxy = na_lens + na_input`).  So `'auto'` at any realistic NA, which
routes to `'stationary_phase'`, returned a leading-order expansion about the
wrong ray: for a uniformly tilted input it returned an intensity pattern that
does not move at all, because `|E_in|` does not change under a tilt and the
entrance point the saddle picked did not depend on the input's phase.

The driver now fits the input's local wavevector `k1 = (1/k0) grad arg E_in`
over the **same** tensor-Chebyshev chart coordinates as the OPD and entrance-
coordinate fits, sampled at the traced rays' own entrance points, and both
saddle solvers add `(k1 . ds1/dv2) / lambda` to the Newton gradient and
`d(k1 . ds1/dv2)/dv2` to the Hessian.  The saddle condition becomes
`(v1_in - v1) . ds1/dv2 = 0`: the ray whose LAUNCH direction is the input's own.
The corrected Hessian is also what `stationary_phase` uses for the
Gaussian-moment amplitude and the Maslov signature, and what
`local_quadrature` uses for its window's principal axes, widths and taper
correction.  `E_in` is still sampled as the complex field, so its phase enters
the answer exactly and the fit only places the saddle -- nothing is added to
`opd_star` / `opd_v` and the input phase is never double-counted.

* `lumenairy/elements/lenses_maslov.py`: `_input_phase_terms` /
  `_eval_input_phase_terms` / `_k1_fit_to_device` (new, :1109-1169),
  `_maslov_newton_saddle_xp` (:1172), `_maslov_newton_saddle_cpu` (:1232),
  `_integrate_stationary_phase` (:3813), `_integrate_local_quadrature`
  (:4437), the two CuPy twins (:4630, :4709), the driver S6 block
  (:2699-2825), `_wavefront_na_from_cosines` / `_sample_real_bilinear`
  (new, :3368-3402), `_SADDLE_FLAT_INPUT_NA` / `_K1_FIT_RESIDUAL_MAX` /
  `_S6_INPUT_WAVEVECTOR_SADDLE` (:96-162).
* Oracles: (a) the **symplectic identity itself**, which turns the converged
  saddle back into the launch direction of the ray it selected; (b) a
  **lumenairy-free** exact sequential-conic raytrace of the INPUT's own rays
  plus a direct Rayleigh--Sommerfeld surface sum (method, not code, from the
  inline oracle of `tests/unit/test_niche_d6_exact_tilted_leg.py`), grid-
  converged to `1 - fidelity ~ 4e-09` and agreeing with the library's own
  converged uniform `'quadrature'` to `1 - fidelity = 6.3e-05`.
* Measured (f = 6 mm N-BK7 biconvex singlet, 0.60 mm clear aperture -> the
  NA 0.05 chart the audit's census was taken on, lambda = 1 um, readout
  0.6 mm past best focus):

  | quantity | input | before | after |
  |---|---|---|---|
  | mean \|v1\| of the saddle ray | tilted | 7.0e-19 | 9.00e-03 (= the input's own) |
  | mean \|v1 - v1_in\| | tilted | 9.00e-03 | 1.1e-13 |
  | mean \|v1 - v1_in\| | converging | 8.48e-03 | 6.0e-15 |
  | field fidelity vs the oracle, `stationary_phase` | tilt 0.5x / 1x NA | 0.6189 / 0.2301 | 0.9101 / 0.9105 |
  | field fidelity vs the oracle, `local_quadrature` | tilt 0.5x / 1x NA | 0.9234 / 0.7298 | 0.9827 / 0.9829 |
  | focal centroid error, `stationary_phase` | tilt 0.5x / 1x NA | 14.657 / 26.329 um | 0.014 / 0.066 um |
  | focal centroid error, `local_quadrature` | tilt 0.5x / 1x NA | 6.228 / 12.154 um | 0.007 / 0.029 um |
  | EE(2 um) / EE(10 um) about the oracle centroid | converging f = +40 mm | 0.0286 / 0.5654 | 0.0052 / 0.1458 (oracle 0.0048 / 0.1366) |
  | field fidelity vs the oracle | converging f = +40 mm | 0.5432 / 0.9398 | 0.9967 / 0.9993 |

  The corrected fidelities equal the method's own **collimated** fidelity
  (0.910229 for `stationary_phase`, 0.982759 for `local_quadrature`) to
  2.4e-04, which is the right envelope: a leading-order expansion about the
  correct ray is exactly as accurate on a tilted input as on a flat one.  On
  the larger 1024-point grid `stationary_phase` returned **NaN** at tilt 4x the
  input NA before, and fidelity 0.9080 after.
* Tests: `tests/unit/test_audit2609_b1_maslov_input_wavevector.py` (25).
* **The collimated case is byte-identical.**  A real, non-negative `E_in` has
  `arg E_in == 0`, so its local wavevector is EXACTLY zero and the fit is never
  built.  Verified against the 5.46.0 module itself (loaded out of
  `git show HEAD:`) with `np.array_equal`: **24 of 24 rows**, covering both
  arms of the WP-A4 S6 fixture on all four integrators, the WP-B1 fixture's
  collimated field, `collimated_input=True`, sub-threshold wavefronts, and
  `'quadrature'` / `'levin'` on tilted and converging inputs.

### Changed -- the S6 warning is restated: it now marks the fallback, not the fixed case

5.46 warned whenever an asymptotic method met an input whose wavefront NA
exceeded `_SADDLE_FLAT_INPUT_NA = 1e-3`.  That is exactly the class 5.47
computes correctly, so the warning no longer fires there.  It is kept, with the
same `saddle of the OPD alone` wording, for the regime where the chart cannot
carry the input's local wavevector and the OPD-only saddle is retained:

* **The criterion.**  The intensity-weighted RMS residual of the `(k1x, k1y)`
  fit, as a fraction of their own intensity-weighted RMS, above
  `_K1_FIT_RESIDUAL_MAX = 0.5` -- "the fit explains less than 75 % of the local
  wavevector's power".  Measured on the f = 6 mm chart at order 4: pure tilt
  1.4e-11, converging / diverging 1.1e-05 / 1.3e-05, 10-20 waves of coma /
  astigmatism / trefoil / spherical 8.0e-05..5.7e-02, a hard-edged aperture at
  0.4-0.95 of the traced pupil 7.2e-02..2.9e-01, speckle at 0.002-0.05 rad rms
  1.1e-01..9.1e-01, at 0.1-0.6 rad rms 9.6e-01..9.8e-01, uniform white-noise
  phase 9.9e-01.  A speckle ladder scored against the oracle puts the point
  where the fitted saddle stops helping between residual 0.91 (fidelity
  0.191 -> 0.251) and 0.96 (0.191 -> 0.105); the bar is placed a factor 1.8
  below it.
* **The fallback** is the 5.46 saddle, byte-identically, plus the warning.
* The warning also fires when the module seam is set to `False`, and when the
  wavefront is flat across the traced aperture but not across the grid.

### Added -- `lenses_maslov._S6_INPUT_WAVEVECTOR_SADDLE`, a module-level A/B seam

`None` (default) is the decision above; `False` always solves
`grad_v2 OPD = 0` -- the 5.46 saddle with the 5.46 warning -- and `True` uses
the fitted wavevector whenever the input is not flat, faithful fit or not.  In
the style of the existing `_QUAD_FACTORIZE` / `_SP_PIXEL_CHUNK` seams in the
same module.

* **Migration.**  For a NON-COLLIMATED input with `integration_method` of
  `'stationary_phase'`, `'local_quadrature'`, or `'auto'` where it resolves to
  `'stationary_phase'`, the returned field CHANGES -- that is the fix.  A caller
  who needs to reproduce a 5.46 number sets
  `lumenairy.elements.lenses_maslov._S6_INPUT_WAVEVECTOR_SADDLE = False`
  (process-global, private, no stability guarantee) or pins 5.46.
  `collimated_input=True` also pins the old saddle but is NOT a way back to the
  old numbers: it re-sizes the pupil chart as well (`na_lens = 1e-5`,
  `na_input = 0`).  `'quadrature'` and `'levin'` integrate the true integrand
  pointwise, have no saddle, and are byte-identical to 5.46 for every input --
  they were already the correct choice for a non-collimated input and remain
  so where the fallback fires.  Collimated inputs are unchanged everywhere.
* The per-call spelling of the seam is an `input_wavevector_saddle=` keyword.
  It is NOT in this release: a new keyword-only parameter on
  `apply_real_lens_maslov` requires an entry in
  `lumenairy/elements/lens_config.py::KWARG_ONLY`, which belongs to another
  work package in this wave.  Requested in the WP-B1 report.

### Performance

The corrected Newton evaluates four more Chebyshev contractions per iteration
(the two entrance-coordinate fits and the two wavevector fits) and the driver
pays one extra least-squares solve against the already-built design matrix.
MEASURED (best of 3, WP-B1 fixture, 120x120 ROI, 16^4 rays, order 4):
`stationary_phase` **0.191 s -> 0.207 s (1.08x)**, `local_quadrature`
**0.519 s -> 0.571 s (1.10x)**; the extra `_solve_fit` at 40 000 rays x 70
terms is **10.8 ms**.  A collimated input pays none of it -- the fit is never
built (0.178 s / 0.529 s, unchanged).

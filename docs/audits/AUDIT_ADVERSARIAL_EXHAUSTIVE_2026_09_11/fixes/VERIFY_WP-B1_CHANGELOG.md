# VERIFY-WP-B1 changelog text (release 5.47.0) -- the S6 fallback scores both factors of the term

Independent adversarial re-verification of WP-B1 (commit 2871e92e) and its
follow-up (8dab7de5), finding **S6** of
`AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11`.  WP-B1 itself is verified; this is
the one defect found in it that is fixed here.  Full measurement in
`fixes/VERIFY_WP-B1.md`.

---

### Fixed -- Maslov S6 fallback: the chart has to carry BOTH factors of `k1 . ds1/dv2`, not just `k1` (audit S6, P1)

The term the two asymptotic saddle solvers add is the input's local wavevector
contracted with the chart's own entrance-coordinate map,
`(k1 . ds1/dv2) / lambda`.  The fallback criterion scored only the `k1` fit,
and a **uniform tilt fits `k1` perfectly by construction** -- it is a constant
-- however badly the chart carries the `ds1/dv2` it is contracted against.
Because `na_proxy = na_lens + na_input` sizes the pupil chart from the 3-sigma
*about zero* moment of the input's angular spectrum, a uniform tilt `theta`
contributes `3 theta`, so a large tilt inflates the box the order-`poly_order`
tensor-Chebyshev chart must span and the entrance-coordinate fit degrades.
The result was a silently wrong answer with a perfect fit score:

| tilt / lens NA | k1 fit | s1 fit | `stationary_phase` | `local_quadrature` | exact `'quadrature'` |
|---|---|---|---|---|---|
| 1.5 | 1.3e-14 | 1.3e-03 | 0.932 | 0.984 | 0.997 |
| **2.0** | 2.6e-14 | **4.1e-03** | **0.000** | **0.006** | 0.968 |

(f = 6 mm N-BK7 singlet, NA 0.05, lambda = 1 um, order 4, field fidelity
against an exact conic-raytrace + Kirchhoff oracle whose collimated floor is
0.931 / 0.984.  An independent f = 13.3 mm N-SF11 / 1.55 um chart collapses at
the same statistic: 0.919 / 0.969 at s1 fit 1.8e-03, 0.038 / 0.061 at
3.3e-03, with a **-27.7 um** centroid error at tilt 2x.)  Nothing warned.

The S6 fallback now measures the entrance-coordinate fit as well: its relative
RMS residual over the traced rays must be at or below
`_S1_FIT_RESIDUAL_MAX = 2.5e-3`, the geometric mean of the measured bracket
(1.8e-03 last good .. 3.3e-03 first collapse), which also sits a factor 1.9
above the other chart's last good row and 1.6 below its first bad one.  Above
it the OPD-only saddle is kept and the S6 `RuntimeWarning` fires naming the
two remedies that repair the chart -- a larger `poly_order`, or an explicit
`input_na` that stops `na_proxy` over-sizing the pupil box.  Both recover the
full answer on the same input (0.910 / 0.969 at order 6; 0.898 / 0.971 with
`input_na=theta`).

* `lumenairy/elements/lenses_maslov.py`: `_S1_FIT_RESIDUAL_MAX` (new, :168-197),
  the S6 driver block's fallback gate and its warning branch, the
  `S6 input-wavevector saddle` progress line (which gains an `s1 chart fit`
  field BEFORE the `k1 fit residual ... (engaged)` tail, so the existing
  parser in `test_audit2609_b1_maslov_input_wavevector.py` is unchanged), and
  the `apply_real_lens_maslov` docstring.
* **Nothing that works today stops working.**  Every input class that engages
  in 5.47 still engages: off-axis converging 6.0e-06, astigmatic 7.5e-06, a
  hard-edged aperture at 0.8 of the pupil 4.5e-04, 0.01 rad rms speckle
  4.5e-04, the WP-B1 report's converging f = +40 mm 5.5e-06 and diverging
  f = -25 mm 6.7e-06, and every tilt to 1.5x the lens NA on both charts.  The
  WP-A4 S6 fixture measures 1.1e-02 but is already refused by the `k1` bar
  (0.957), so its behaviour and its byte-identity are untouched.
* **Byte-identity is preserved**: 27 of 27 archive-to-archive comparisons
  against 2871e92e^ behave exactly as before the change (23 identical, 4
  correctly different), and all 27 are unchanged against 8dab7de5 itself.
* `input_wavevector_saddle=True` overrides the new gate, exactly as it
  overrides the `k1` one.
* **Migration.**  A call that today returns a silently misplaced field for an
  input tilted past ~1.5x the lens NA now returns the OPD-only answer and a
  `RuntimeWarning`.  Both are wrong; the warning names the two settings that
  make it right.  To keep the 5.47.0-as-shipped behaviour for such a call,
  pass `input_wavevector_saddle=True`.
* Tests: `tests/unit/test_audit2609_b1_maslov_input_wavevector.py` 28 -> 33.

### Changed -- the `_K1_FIT_RESIDUAL_MAX` derivation in the source is re-measured

The speckle rows of the ladder recorded beside `_K1_FIT_RESIDUAL_MAX = 0.5`
did not reproduce on the chart they name: measured with the shipped code, and
reading the residual the driver itself prints, speckle at
0.002 / 0.005 / 0.01 / 0.02 / 0.05 rad rms on a tilted carrier gives
**4.7e-03 / 1.2e-02 / 2.3e-02 / 4.6e-02 / 1.1e-01**, against the recorded
1.1e-01 / 2.6e-01 / 4.7e-01 / 7.1e-01 / 9.1e-01 -- a uniform factor ~20, and
in line with the analytic value `sigma sqrt(2) / (k0 dx theta)` for white
phase noise on a tilt.  The pure-tilt, converging / diverging and
hard-edged-aperture rows of the same ladder reproduce exactly.  Scored against
the exact pointwise `'quadrature'` rather than a geometric-optics oracle (which
is not a truth for speckle), the fitted saddle's fidelity is already down from
0.908 / 0.982 to **0.656 / 0.702 at residual 1.1e-02** and **4e-04 / 5e-04 at
5.3e-02** -- so the 0.5 bar does not fire until the answer is two decades
gone.  The mechanism is that the residual scores the fit's VALUE while the
saddle also consumes its two DERIVATIVES: the S6 Hessian term's RMS grows
**180x** (8.3 -> 1507) between a clean tilt and 0.05 rad rms speckle.

The constant is UNCHANGED -- re-founding it is a shipped-behaviour decision,
not a verification finding, and on a non-collimated carrier the OPD-only
fallback is never the better answer anyway, which is what makes the bar hard
to place.  What changed is the comment beside it, which now carries the
measured ladder, dated, says plainly what the statistic cannot see, and names
`integration_method='quadrature'` as the remedy.  A new pin fixes the
statistic's calibration against its analytic value to a factor of 2, so a
future drift of this size cannot pass unnoticed.

* `lumenairy/elements/lenses_maslov.py:122-166`;
  `test_verify_b1_the_k1_fit_residual_is_the_statistic_it_claims_to_be`.

### Performance

One `(n_rays x M) @ (M x 2)` GEMM and a reduction, on the ENGAGED path only
and once per call.  MEASURED at 16^4 rays x 70 terms: **5.7 ms**, against
0.078 s (`stationary_phase`) and 0.121 s (`local_quadrature`) for the whole
engaged call on a 40 x 40 ROI of the f = 13.3 mm chart -- 7 % and 5 %.  A
collimated input, a declared `collimated_input=True`, an input below the
engagement bar, and `'quadrature'` / `'levin'` never reach it and pay nothing.

# WP-B1 — Maslov S6 proper: the asymptotic saddle follows the INPUT field's local wavevector

Branch `audit-fixes-2026-09`, on top of HEAD `81d5b586` (release 5.46.0).
Finding **S6** (audit report §2.5), the half WP-A4 deferred with a design in
its report §6 item 1.  WP-A4 shipped a `RuntimeWarning`; the saddle itself was
unchanged and both asymptotic evaluators were still wrong for a non-collimated
input.

Every number below is MEASURED on this checkout.  "Before" means the same
build with `lenses_maslov._S6_INPUT_WAVEVECTOR_SADDLE = False`, which is the
5.46.0 saddle held side by side with the new one in one process; where a
comparison is against 5.46.0 *itself* it says so and names the mechanism (the
HEAD module loaded out of `git show HEAD:` as a sibling of the package).

> **One test in the verification set is RED, deliberately, and it is not mine
> to edit.**
> `tests/unit/test_audit2609_a4_verify_maslov_asymptotic.py::test_s6_saddle_warning_fires_only_on_a_non_flat_input`
> asserts that a 0.01 rad tilted input MUST trip the S6 warning.  That is
> exactly the case this work package now computes correctly, and the brief's
> own deliverable 2 says to "retire or restate the S6 warning: it fires today
> for the case you are now computing correctly".  So the assertion is
> superseded by the fix, and the file belongs to WP-A4, not to me.  The exact
> patch is in §7 item 2, with its measurements.  Everything else in the
> verification set is green — §8.

---

## 1. Summary table

| ID | status | files:lines | tests | oracle | measured before → after |
|---|---|---|---|---|---|
| **S6** (P1) | **fixed** | `elements/lenses_maslov.py`: `_input_phase_terms` / `_eval_input_phase_terms` / `_k1_fit_to_device` (new, :1109-1169), `_maslov_newton_saddle_xp` (:1172), `_maslov_newton_saddle_cpu` (:1232), `_integrate_stationary_phase` (:3813), `_integrate_local_quadrature` (:4437), the two CuPy twins (:4630, :4709), the driver's S6 block (:2699-2825), `_wavefront_na_from_cosines` / `_sample_real_bilinear` (new, :3368-3402), `_K1_FIT_RESIDUAL_MAX` / `_S6_INPUT_WAVEVECTOR_SADDLE` (:117-162) | `test_audit2609_b1_maslov_input_wavevector.py` (25) | (a) the symplectic identity `dOPD/dv2 = −n1 (v1·ds1/dv2)`, which reads the launch direction back off the converged saddle; (b) a lumenairy-free exact conic raytrace of the INPUT's own rays + a direct Rayleigh–Sommerfeld surface sum, grid-converged to `1−fid ≈ 4e-09` and agreeing with the library's converged `'quadrature'` to `1−fid = 6.3e-05` | saddle launch direction `|v1 − v1_in|` **9.00e-03 → 1.1e-13** (tilted), **8.48e-03 → 6.0e-15** (converging); field fidelity vs the oracle **0.6189 / 0.2301 → 0.9101 / 0.9105** (`stationary_phase`, tilt 0.5×/1× NA) and **0.9234 / 0.7298 → 0.9827 / 0.9829** (`local_quadrature`); focal centroid error **14.657 / 26.329 µm → 0.014 / 0.066 µm** and **6.228 / 12.154 µm → 0.007 / 0.029 µm**; converging EE(10 µm) **0.5654 / 0.2473 → 0.1458 / 0.1310** against the oracle's 0.1366 |
| **S6 warning** | **restated** | `lenses_maslov.py:2788-2824`, `_K1_FIT_RESIDUAL_MAX` (:148) | `test_b1_the_s6_warning_is_gone_exactly_where_the_fix_applies`, `test_b1_the_fallback_*` (3) | the speckle ladder scored against the same oracle | fired on every non-flat input (which is now the FIXED class) → fires only on the fallback regime, criterion `k1`-fit residual > 0.5 |
| **collimated byte-identity** | **proved** | — | `test_b1_a_collimated_input_is_byte_identical_*` (3), `test_b1_the_integrators_without_a_saddle_are_untouched` (2), `test_b1_a_declared_collimated_input_*` (2), `test_b1_a_sub_threshold_wavefront_*` | `np.array_equal` against the 5.46.0 module loaded from `git show HEAD:` | **24 of 24 rows `array_equal`**, including both arms of the WP-A4 S6 fixture on all four integrators (re-run to 22/22 on the final source, the two `levin` rows dropped for their 280 s / 450 s runtime) |

---

## 2. S6 — what was wrong, and what the fix is

### The defect

The v2 integrand is
`E_in(s1(s2, v2)) · |det ds1/dv2|^(1/2) · exp(2πi·OPD_waves)`, so the phase that
is stationary in `v2` is the TOTAL

```
Psi(v2) = OPD_waves(s2, v2) + arg E_in(s1(s2, v2)) / (2 pi).
```

Both asymptotic evaluators solved `grad_v2 OPD = 0`.  The symplectic identity
`dOPD/dv2 = −n1 (v1 · ds1/dv2)` makes that the `v1 = 0` on-axis collimated
launch ray at **every** pixel and for **every** input — while the driver sizes
the pupil chart specifically to cover a diverging / converging / tilted one
(`na_proxy = na_lens + na_input`).

Measured directly, by reading the launch direction back off the converged
saddle through that identity
(`v1 = −(λ/n1)·(ds1/du_v2)^{−T}·grad_u_v2 OPD_waves`), on the f = 6 mm N-BK7
singlet at 0.60 mm clear aperture — the NA 0.05 chart the audit's own census
was taken on, whose all-ray mean `|v1|` is **4.11e-02**:

| input | saddle | mean \|v1\| | mean \|v1_in\| | mean \|v1 − v1_in\| |
|---|---|---|---|---|
| collimated | 5.46 / 5.47 | 6.7e-19 | 0 | 6.7e-19 |
| tilt 0.5× NA | 5.46 | 7.0e-19 | 2.2508e-03 | 2.2508e-03 |
| tilt 0.5× NA | **5.47** | 2.2508e-03 | 2.2508e-03 | **8.7e-14** |
| tilt 1.0× NA | 5.46 | 7.2e-19 | 4.5016e-03 | 4.5016e-03 |
| tilt 1.0× NA | **5.47** | 4.5016e-03 | 4.5016e-03 | **4.5e-13** |
| converging f = +40 mm | 5.46 | 7.0e-19 | 8.4759e-03 | 8.4759e-03 |
| converging f = +40 mm | **5.47** | 3.4334e-03 | 3.4334e-03 | **5.9e-15** |

The pre-fix `|v1|` column is machine zero at every pixel: the audit's measured
6.93e-03 was the mean over the 2 % smallest `|grad_v2 OPD|` of the *ray* set,
which is a sampling of that same fact.  The two `|v1_in|` values for the
converging input differ between the rows because the two saddles land on
DIFFERENT entrance points, which sample different parts of a curved wavefront;
for a uniform tilt they cannot, and they do not.

The user-visible consequence for a tilted input is sharp: a uniform tilt is a
pure phase, so `|E_in|` does not change, and the entrance point the OPD-only
saddle picks does not depend on the input's phase — therefore the returned
INTENSITY pattern does not move at all, while the truth moves by ≈ `f·θ`.

### The fix

The input's local wavevector `k1 = (1/k0) grad arg E_in` — which is `n1` times
its direction cosine, and exactly what `_local_direction_cosines` already
returns — is fitted as two more Chebyshev fits over the SAME chart coordinates
as the OPD and entrance-coordinate fits, sampled at the traced rays' own
entrance points.  Then, in waves per unit of the normalised chart coordinate,

```
g3 += (k1x ds1x/du3 + k1y ds1y/du3) / lambda          # Newton gradient
a33  = (dk1x/du3 ds1x/du3 + k1x d2s1x/du3^2 + ...) / lambda   # and Hessian
```

and the saddle condition becomes `(v1_in − v1) · ds1/dv2 = 0`.  One shared
`_input_phase_terms` does the arithmetic for both saddle solvers and both
integrators, so the CPU and `xp` (CuPy / NumPy) twins cannot drift apart.

**One deviation from the WP-A4 design, deliberate.**  That design also said to
add "the same term" to `opd_star` in `_integrate_stationary_phase` and to
`opd_v` in `_integrate_local_quadrature`.  I did not, and the reason is that it
would double-count: both integrators already sample the COMPLEX `E_in` at the
saddle / at each window sample, and `arg E_in(s1*)` IS the input phase there.
Writing the leading-order formula out,

```
INT A e^{i Phi} d^2v ~ A(v*) e^{i Phi(v*)} e^{i pi sig/4} / sqrt|det Phi''|
   with A = |E_in| |det J|^{1/2},  Phi = arg E_in + 2 pi OPD_waves
   =>  A(v*) e^{i Phi(v*)} = E_in(s1*) |det J|^{1/2} e^{2 pi i OPD*}
```

which is the product the code already forms.  So only two things change: WHERE
the saddle is (the Newton gradient) and HOW SHARP it is (the Hessian, which
sets `amp_sp`, the Maslov signature `sig`, and `local_quadrature`'s window
axes / widths / taper correction).  This is also strictly more accurate than
the design's version — the input phase enters through the sampled field, at
full accuracy, instead of through a degree-`poly_order` fit — and it is what
makes the collimated case byte-identical rather than merely close.  The fitted
`k1` is used ONLY to place and shape the saddle.

**A second deviation, also deliberate.**  The design said to put `k1x`, `k1y`
on the existing stacked right-hand side (`_solve_fit(A, [opd, s1x, s1y])`),
"one wider RHS and no extra factorisation".  I used a separate `_solve_fit`
call against the same design matrix instead.  Widening a GEMM's right-hand
side is entitled to move the existing columns in the last bits, and those
columns feed `'quadrature'` and `'levin'` too — so the cheap version would
have made every uniform-quadrature and Levin answer in the library
non-reproducible across this release for any non-collimated input.  The extra
solve is a second `A^T A` + `eigvalsh` + Cholesky at M = 70 on ~40 000 rays:
measured **10.8 ms**, against 0.19 s for the smallest `stationary_phase` call in
this report.  It buys the byte-identity result in §4.

### The two thresholds

`_SADDLE_FLAT_INPUT_NA = 1e-3` (unchanged from 5.46, where it gated the
warning) now gates ENGAGEMENT, measured over the traced ray entrance points and
weighted by the input intensity there — the part of the wavefront the integral
actually samples.  Reusing it is what makes "the warning fired exactly where
the fix now engages" true by construction, and it keeps every sub-bar input
bit-for-bit 5.46.0.

It is a FLATNESS declaration, not a negligibility claim, and the source comment
says so with the measurement: the relative L2 by which the fitted saddle moves
the field is **4.3e-02 at NA_wf = 3.0e-04, 1.46e-01 at the bar, 4.7e-01 at
3.0e-03**, i.e. linear in the input tilt with no knee.  What the bar really
separates is a field that HAS a wavefront from one that is flat up to numerical
noise (a numerically real field measures EXACTLY 0; float64 phase dirt measures
~1e-16), which matters because fitting `k1` out of phase dirt would trip the
residual gate and warn about nothing.  Left as a residual risk in §6.

`_K1_FIT_RESIDUAL_MAX = 0.5` is the fallback criterion — see §3.

---

## 3. The fallback: when the chart cannot carry the input's wavevector

The brief asked for a measurement and a stated criterion for the case where
"the input phase is not smooth enough to differentiate".  The criterion is the
intensity-weighted RMS residual of the `(k1x, k1y)` fit as a fraction of their
own intensity-weighted RMS — i.e. `0.5` reads "the fit explains at least 75 %
of the local wavevector's power".  Above it the OPD-only saddle is kept and the
S6 warning fires.

Measured on the f = 6 mm chart at `poly_order=4`, with the field fidelity
against the lumenairy-free oracle for the OPD-only and the forced-fit saddles:

| input | residual | fid (5.46 saddle) | fid (fitted saddle) |
|---|---|---|---|
| pure tilt, 0.5×/1×/4× NA | 1.4e-11 … 1.2e-11 | 0.192 | **0.907** |
| converging / diverging (f = +40 / +15 / −25 mm) | 1.1e-05 … 1.7e-05 | 0.481 | **0.991** |
| 10–20 waves coma / astigmatism / trefoil | 8.0e-05 … 1.0e-03 | — | improves |
| defocus + 5 waves r⁴ / 20 waves r⁴ (order 4) | 1.8e-02 / 5.7e-02 | — | improves |
| hard-edged aperture at 0.8 / 0.95 / 0.6 / 0.4 of the pupil | 7.2e-02 / 1.5e-01 / 2.7e-01 / 2.9e-01 | 0.164 / 0.184 / 0.118 / 0.086 | **0.812 / 0.304 / 0.279 / 0.190** |
| speckle 0.002 / 0.005 / 0.01 / 0.02 / 0.05 rad rms | 1.1e-01 / 2.6e-01 / 4.7e-01 / 7.1e-01 / 9.1e-01 | 0.192 | **0.911 / 0.789 / 0.608 / 0.439 / 0.251** |
| speckle 0.1 / 0.2 / 0.3 / 0.6 rad rms | 9.6e-01 / 9.8e-01 / 9.8e-01 / 9.8e-01 | 0.191 / 0.191 / **0.572** / 0.223 | 0.105 / 0.011 / **0.000** / 0.000 |
| uniform white-noise phase | 9.9e-01 | 0.014 | 0.009 |
| 50 waves r⁶ (order 4) | 2.4e+00 | 0.394 | 0.410 |

So the fitted saddle helps up to residual ≈ 0.91 and hurts above ≈ 0.96 (at
0.3 rad rms it collapses to an all-zero field: the Newton is chased outside the
chart box and all 14 400 output pixels fail to converge and are correctly
zeroed, against 1 446 for the same fixture with no speckle).
The bar is placed at **0.5**, a factor 1.8 below the last case where engaging
still helps and 1.8 above the first where it does not.  That deliberately
refuses a band (0.5 … 0.91) in which the fit would still have helped: the
criterion has to be a property of the INPUT, because the library has no oracle
at run time, and "three quarters of the power is explained" is the strongest
statement the fit itself supports.  The refused band is reachable with the
seam, and it warns.

Two things the criterion does NOT catch, both stated in the source:

* **Aliasing.**  A wavefront steeper than the grid's own Nyquist angle
  `λ/(2dx)` makes the phase-difference estimator wrap to a direction that is
  perfectly smooth — measured residual **1.2e-10** at a 1.2× Nyquist tilt.
  That is not a saddle problem: the whole sampled field is the aliased one, and
  every other consumer of `E_in` sees the same thing.  The docstring says to
  sample finely enough that `max|grad arg E_in|·dx < π`.  The prior claim in a
  draft of this warning ("a wavefront steeper than Nyquist does this") was
  measured and found FALSE, and removed.
* **An input wavefront the fit ORDER cannot carry**, which is a different
  remedy: 20 waves of r⁴ scores 5.7e-02 at order 4 and 1.0e-03 at order 6; 50
  waves of r⁶ scores 2.44 at order 4 and 1.77 at order 6.  The warning names
  `poly_order` as one of the remedies.

**The WP-A4 S6 fixture lands in this regime**, which is why
`tests/unit/test_audit2609_a4_maslov_gbd.py::test_s6_*` is green, unchanged,
with its diverging arm still warning and its output byte-identical to 5.46.0.
Measured: ray-sampled NA 0.1998, `k1` fit residual **1.55**.  That fixture's
clear aperture (0.30 mm) is 3.1× its grid half-width (96 µm), so most traced
ray entrance points sit off the sampled field entirely, and its f = −0.5 mm
diverging wavefront passes the grid's Nyquist angle (0.125) at r = 62 µm.  An
order-4 chart cannot represent that, and the library says so instead of
guessing.

---

## 4. The collimated case is byte-identical — proof

A real, non-negative `E_in` has `arg E_in == 0`, so `_local_direction_cosines`
returns EXACTLY zero (pinned separately) and the driver never builds the fit:
the S6 term is not small, it is absent, and the arithmetic is literally
5.46.0's.

Proved by loading the pre-change `lenses_maslov.py` out of
`git show HEAD:lumenairy/elements/lenses_maslov.py` as a sibling module inside
`lumenairy.elements` (so its relative imports resolve) and running both side by
side with `np.array_equal`:

```
=== A4 S6 fixture (tests/unit/test_audit2609_a4_maslov_gbd.py) ===
  flat (collimated)    + stationary_phase        array_equal=True   S6warn 0->0
  flat (collimated)    + local_quadrature        array_equal=True   S6warn 0->0
  flat (collimated)    + quadrature              array_equal=True   S6warn 0->0
  flat (collimated)    + levin                   array_equal=True   S6warn 0->0   (280 s)
  flat + stationary_phase + collimated_input=True array_equal=True  S6warn 0->0
  diverging f=-0.5 mm  + stationary_phase        array_equal=True   S6warn 1->1
  diverging f=-0.5 mm  + local_quadrature        array_equal=True   S6warn 1->1
  diverging f=-0.5 mm  + quadrature              array_equal=True   S6warn 0->0
  diverging f=-0.5 mm  + levin                   array_equal=True   S6warn 0->0   (450 s)
  diverging + stationary_phase + collimated_input=True array_equal=True S6warn 0->0
=== WP-B1 fixture (f=6 mm singlet, NA 0.05, roi readout) ===
  collimated + stationary_phase / local_quadrature / quadrature(n_v2=48)  all True
=== non-collimated: the paths with no saddle must not move ===
  tilt 1x NA + quadrature(n_v2=48)               array_equal=True
  converging f=+40mm + quadrature(n_v2=48)       array_equal=True
=== non-collimated + collimated_input=True ===
  tilt 1x NA + stationary_phase / local_quadrature   both True
=== non-collimated, seam False ===
  tilt / converging x stationary_phase / local_quadrature   all True, S6warn 1->1
=== sub-threshold wavefront (NA_wf <= 1e-3) ===
  tilt 1.00e-04 / 3.00e-04 / 3.33e-04            all True
=== ENGAGED (must differ) ===
  tilt 1x NA + stationary_phase                  array_equal=False relL2=1.06  S6warn 1->0
  tilt 1x NA + local_quadrature                  array_equal=False relL2=1.25  S6warn 1->0
  converging + stationary_phase                  array_equal=False relL2=0.70  S6warn 1->0
  converging + local_quadrature                  array_equal=False relL2=0.21  S6warn 1->0

24/24 byte-identity rows PASS
```

`tests/unit/test_audit2609_a15a_lens_covering_array.py` has **no Maslov
cells** — the brief expected some; its covering arrays are
`apply_real_lens` (`analytic`) and `apply_real_lens_traced` (`traced`) only, so
`-k maslov` deselects all 45.  The A4 fixtures above cover the same ground, and
the A15a suite is green.

The durable form of this proof lives in the test file as byte-identity between
the shipped default and the seam forced to `False` on the same build, plus the
`_local_direction_cosines`-is-exactly-zero premise — nothing there pins a prior
release's numbers.

---

## 5. Files touched

**Source (within this WP's ownership).**

* `lumenairy/elements/lenses_maslov.py` — the whole fix; +453 / −63 lines.  No
  other module needed to change: the saddle lives entirely in this file, and
  `propagators/asymptotic*.py` implements a different (canonical-transform)
  family with its own `_compute_M_b` saddle, which finding S6 does not name and
  which I did not touch.

**History.**

* `docs/history/lumenairy.elements.lenses_maslov.md` — re-recorded with
  `scripts/record_history_fingerprints.py` in this change (reason on the
  header's `re_recorded:` line).  It is the only history document whose module
  I touched.

**Tests — new.**

* `tests/unit/test_audit2609_b1_maslov_input_wavevector.py` (25 tests).

**Tests — modified.**  None.  The WP-A4 S6 pins are green unchanged (§3), and
no existing test needed its tolerance touched.

**Reports.**

* `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B1_REPORT.md`
  (this file) and `WP-B1_CHANGELOG.md`.

---

## 6. Deferred / residual risk

1. **`input_wavevector_saddle=` as a per-call keyword.**  Shipped as the
   module seam `_S6_INPUT_WAVEVECTOR_SADDLE` instead, because a new
   keyword-only parameter on `apply_real_lens_maslov` fails
   `tests/unit/test_audit2609_a16_lens_config_round_trip.py::test_every_keyword_is_classified_as_field_contract_or_documented_exclusion[apply_real_lens_maslov]`
   until `lumenairy/elements/lens_config.py::KWARG_ONLY` classifies it — and
   that file belongs to WP-B2 in this wave.  The exact edit is in §7.
2. **`_SADDLE_FLAT_INPUT_NA` is an absolute angle, not a chart-relative one.**
   Below the bar the OPD-only saddle's launch direction is wrong by up to
   3.3e-04 rad, which is 2 µm at the focus of the f = 6 mm fixture and 0.3 mm
   on an f = 1 m system.  The physically right statement is a fraction of the
   chart's own pupil half-width (`|k1| / na_proxy`), which scales with the
   optic.  I did not change it because it would break the "the 5.46 warning
   fired exactly where 5.47 engages" symmetry and the sub-bar byte-identity,
   both of which are worth more this release than the extra decade.  Design if
   picked up: engage on `3·rms(k1) > eta · na_proxy` with `eta` derived from
   the saddle displacement `H^{-1} g_in` staying inside a fixed fraction of the
   chart box, and warn on a failed fit only above the present absolute bar so a
   numerically-flat input still cannot warn about nothing.
3. **The fallback refuses a band (residual 0.5 … 0.91) in which the fitted
   saddle would still have helped** — §3.  Closing it needs a criterion that
   is not a property of the fit alone; the obvious one is outcome-based: run
   the corrected Newton, and if the in-box non-convergence fraction rises
   sharply against the OPD-only run, fall back.  That is one extra Newton on
   the pathological path only, but it needs the convergence mask plumbed back
   from the integrators to the driver.
4. **Performance.**  The corrected Newton does five Chebyshev contractions per
   iteration where it did one (OPD, the two entrance-coordinate fits, the two
   wavevector fits), each rebuilding the same 4-variable basis.  MEASURED on
   the WP-B1 fixture's 120×120 ROI (best of 3): `stationary_phase`
   **0.191 s → 0.207 s (1.08×)** and `local_quadrature`
   **0.519 s → 0.571 s (1.10×)**, plus **10.8 ms** for the extra `_solve_fit`
   at 40 000 rays × 70 terms — small here because this ROI's 14 400 pixels
   converge in a few Newton steps, and it will grow towards 5× on a full-grid
   chart that runs all 12.  The WP-A4 report's own §6 item 4(a) — one fused
   `_basis_and_grad34` contracted against a stacked coefficient tensor — would
   collapse all five into one basis build and is worth 2.3–5.1× on the
   dominant part of the runtime; it is now worth more, not less.  Only the
   ENGAGED path pays anything: a collimated input measures 0.178 s / 0.529 s,
   unchanged.
5. **The GPU twins are desk-checked, not run.**  CuPy is not installed on this
   machine.  Both `_integrate_*_cupy` route their S6 terms through the same
   `_input_phase_terms` / `_eval_input_phase_terms` the CPU path uses, and the
   `xp` saddle solver is exercised with `xp = np` against the CPU one
   (byte-identical, with and without the term) in
   `test_b1_the_cpu_and_xp_saddle_solvers_agree_with_the_s6_term`.  The device
   upload of the four coefficient vectors goes through one new
   `_k1_fit_to_device`, matching how `coef_opd` / `coef_s1*` are already
   uploaded.
6. **`_lens_jax.apply_real_lens_maslov_jax` is NOT fixed.**  It is the JAX
   sibling of this propagator and carries the same OPD-only saddle, but
   `lumenairy/elements/_lens_jax.py` is outside this WP's ownership list.  The
   requested change is in §7.

---

## 7. Requested changes outside my ownership

1. **`lumenairy/elements/lens_config.py`** (WP-B2) — to let the seam become a
   per-call keyword.  Add to `KWARG_ONLY['apply_real_lens_maslov']`:

   ```python
        'input_wavevector_saddle':
            "per-call numerical-method policy: which stationary point the two "
            "asymptotic evaluators expand about (audit S6).  It is a property "
            "of the INPUT FIELD, not of the optic or the machine, so it "
            "cannot travel in a LensGeometry / LensNumerics / LensResources "
            "that is reused across fields.",
   ```

   and, in `lumenairy/elements/lenses_maslov.py`, add
   `input_wavevector_saddle: Optional[bool] = None` to the signature after
   `input_na`, forward it in the `fold_split` `_leg_kw` dict, and read it in
   place of the module seam (`_s6_mode = (_S6_INPUT_WAVEVECTOR_SADDLE if
   input_wavevector_saddle is None else input_wavevector_saddle)`).  I can make
   the `lenses_maslov.py` half in a follow-up once the `lens_config.py` entry
   lands; doing it now would have left the verification set red.

2. **`tests/unit/test_audit2609_a4_verify_maslov_asymptotic.py`** (WP-A4's
   file) — **THE ONE RED TEST IN THE VERIFICATION SET.**
   `test_s6_saddle_warning_fires_only_on_a_non_flat_input` asserts that a
   0.01 rad tilted input must trip the S6 warning.  That input is now computed
   correctly and is silent, which is deliverable 2 of the brief ("retire or
   restate the S6 warning").  The exact patch — measured on that file's own
   fixture (0.8 mm aperture biconvex, N = 96, dx = 10 µm, λ = 1.31 µm):

   ```python
   -    assert saddle_warnings(tilted, integration_method='stationary_phase'), (
   -        'a 0.01 rad tilted input must trip the S6 warning')
   +    # WP-B1 (audit S6): the saddle now carries the input's fitted local
   +    # wavevector, so a 0.01 rad tilt is the case that is COMPUTED
   +    # CORRECTLY and must be SILENT (measured k1 fit residual 3.61e-13 on
   +    # this fixture, against the 0.5 bar).  The warning marks the FALLBACK:
   +    # an input whose local wavevector the chart cannot represent.  The same
   +    # tilt carrying 0.6 rad rms phase noise measures 7.50e-01 and is
   +    # refused, so both arms of the restated gate are pinned here.
   +    assert not saddle_warnings(tilted,
   +                               integration_method='stationary_phase'), (
   +        'a 0.01 rad tilted input is now expanded about its OWN launch ray '
   +        'and must NOT warn')
   +    speckled = tilted * np.exp(
   +        1j * 0.6 * np.random.default_rng(4).standard_normal(tilted.shape))
   +    assert saddle_warnings(speckled,
   +                           integration_method='stationary_phase'), (
   +        'a speckled input, whose local wavevector an order-4 chart cannot '
   +        'fit, must still trip the S6 warning -- the fallback keeps the '
   +        'OPD-only saddle')
   ```

   The other three assertions in that test (flat is silent, `'quadrature'`
   never warns, `collimated_input=True` silences it) are unchanged and pass.
   Its docstring's "MEASURED: the collimated case's FFT second moment is
   3.54e-03" note stays true and stays relevant.

3. **`tests/unit/test_audit2609_a4_maslov_gbd.py`** (WP-A4's file) — green as
   written; I did not touch it.  Its `test_s6_*` diverging arm still warns
   because that fixture is in the FALLBACK regime (§3), which is worth saying
   in its docstring rather than leaving as a coincidence.  Suggested addition
   to the docstring, no assertion change:

   ```
   WP-B1 (5.47): the warning now marks the FALLBACK.  This fixture is in it:
   its 0.30 mm clear aperture is 3.1x the grid half-width, so most traced ray
   entrance points are off the sampled field, and its f = -0.5 mm wavefront
   passes the grid's Nyquist angle (0.125) at r = 62 um.  Measured k1 fit
   residual 1.55 against the 0.5 bar -- refused, and announced.
   ```

4. **`tests/unit/test_audit2609_a15a_lens_covering_array.py`** (WP-A15a's
   file) — it has no Maslov family at all, so the covering-array invariants
   (finite, shaped, no energy gain, default-passed == omitted) never reach this
   propagator.  Adding a `maslov` family with the existing
   `lens_covering_array_fixture` — which is already a DIVERGING spherical wave
   from 120 mm, exactly the S6 regime — would have caught the amplitude half
   of this finding on the "no energy gain" arm.  Not mine to add.

5. **`lumenairy/elements/_lens_jax.py`** (WP-A4's file) —
   `apply_real_lens_maslov_jax` carries the same OPD-only saddle.  The JAX
   Newton is the `_cheb_*` evaluator in that file; the same two terms apply,
   with `k1` fitted host-side by the caller of `_solve_fit` there.

---

## 8. Tests run

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` on every run.
Python 3.14.6, NumPy 2.4.6, Windows 11.

**The working tree holds several other Wave-4 packages' in-flight edits**
(`rcwa/*`, `pmm/*`, `analysis/*`, `raytrace/*`, `propagators/carrier.py`,
`elements/_lens_traced.py`, `propagators/hfpi.py`, …), so the headline runs
below are taken in an **isolated export**: `git archive HEAD` (read-only) into
a scratch tree, with only my four files copied over it
(`lumenairy/elements/lenses_maslov.py`,
`tests/unit/test_audit2609_b1_maslov_input_wavevector.py`,
`docs/history/lumenairy.elements.lenses_maslov.md`, and the two reports).
Every result is therefore attributable to this work package alone.  The
real-tree confirmations follow.

### 8.1 Isolated export (HEAD 81d5b586 + this WP's files)

| command | result | duration |
|---|---|---|
| `pytest tests/unit -k "maslov or asymptotic"` | **433 passed, 5 skipped, 1 failed** — the one failure is §7 item 2 | 785.7 s |
| `pytest tests/unit/test_audit2609_a17_history_lint.py test_audit2609_a17_history_relocation.py test_v5_4_7_walker_v20_cross_backend_parity.py test_audit2609_b1_maslov_input_wavevector.py test_audit2609_a4_maslov_gbd.py` | **810 passed** | 90.5 s |
| `python validation/run_all.py test_lenses` | **ALL 1 files passed** (46/46 assertions) | 27.3 s |
| `ruff check` (whole tree) | **All checks passed** | 1 s |
| `python scripts/record_history_fingerprints.py --check` | **exit 0**, 124 documents OK, no drift | 3 s |
| `pytest tests/unit/test_audit2609_a15a_lens_covering_array.py -k maslov` | 45 deselected (no Maslov cells; §4) | 0.1 s |

### 8.2 Working tree (with the other packages' edits present)

| command | result |
|---|---|
| `pytest tests/unit/test_audit2609_b1_*.py a4_maslov_gbd a4_asymptotic a4_fga_s10 v5_4_7_walker_v20_cross_backend_parity a15a_lens_covering_array` | **134 passed** (68.7 s) |
| `ruff check lumenairy/elements/lenses_maslov.py tests/unit/test_audit2609_b1_maslov_input_wavevector.py` | **All checks passed** |
| `python scripts/record_history_fingerprints.py lumenairy/elements/lenses_maslov.py --check` | **OK** — re-recorded in this change |
| `pytest tests/unit -k "maslov or asymptotic"` | 428 passed, 5 skipped, **6 failed** — attributed below |

The six working-tree failures, isolated by re-running each against
`git archive HEAD` + only `lenses_maslov.py`:

| failure | verdict |
|---|---|
| `test_audit2609_a17_history_relocation.py::…[lumenairy.elements.lenses_maslov]` ×2 | **mine, fixed** — the fingerprints were not yet re-recorded when that run started; `--check` is clean now |
| `test_audit2609_a4_verify_maslov_asymptotic.py::test_s6_saddle_warning_fires_only_on_a_non_flat_input` | **mine, intended** — §7 item 2 |
| `test_audit_propagation.py::…ModalAsymptoticStillBitEqual::test_lg00_single_mode_bit_equal` | **not mine** — passes on HEAD + only my file |
| `test_audit_propagation.py::…ModalAsymptoticStillBitEqual::test_lg_p0_4mode_prescription_bit_equal` | **not mine** — same |
| `test_niche_audit_w6_asymptotic.py::test_w6_a2_v2_star_is_untouched_by_the_verdict_fix` | **not mine** — same (and it is a TESTING_STANDARDS S4 floor bar: `1.150e-15 < 1e-15`, failing by 15 %) |

A separate note for whoever owns `test_audit2609_a17_history_lint.py`: the
ratchet caught two "pre-5.47" phrases in an early draft of my comments and it
was right to — a comment says what the code does now.  Both were reworded to
name the BEHAVIOUR (`the OPD-only saddle`) instead of a release, and the
migration statement lives in the changelog where it belongs.  The lint is
green without re-baselining.

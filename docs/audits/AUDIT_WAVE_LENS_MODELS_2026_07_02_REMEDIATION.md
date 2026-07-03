# Wave Lens Models — Audit Remediation — 2026-07-03 (v5.18.1)

Implementation record for `AUDIT_WAVE_LENS_MODELS_2026_07_02.md` (the field audit)
reconciled with `..._REVIEW.md` (the companion review). Branch:
`fix/lens-models-audit-remediation`. All findings were re-verified against the
current source before editing (a 9-agent verify pass); two of the review's own
claims were corrected in the process (see §3). Read the two source docs for the
finding definitions.

---

## 1. Implemented

| ID | Change | File | Tests |
|---|---|---|---|
| **N1** | Hoist the `(N_out², M)` Chebyshev design matrix `G` into the quadrature branch; drop the unused `G` param from `_integrate_stationary_phase` / `_integrate_local_quadrature`. Verified those two never read `G`. | `lenses_maslov.py` | `test_n1_all_integrators_run_full_resolution` |
| **N2** | Under-resolution warning for uniform `quadrature`: v2-oscillation bound from `Σ|coef_opd|` over v2-dependent terms; points at the asymptotic evaluators. | `lenses_maslov.py` | `test_n2_under_resolved_quadrature_warns` |
| **N3** | Size the pupil chart from `na_lens + na_input`, `na_input` = 3σ of the input angular spectrum (one FFT); new `input_na=` kwarg + coverage warning. **Clamp `na_proxy` < 1** + reject non-finite `input_na` (adversarial P2/B, §3a). | `lenses_maslov.py` | `test_n3_input_na_widens_chart`, `..._coverage_warning`, `..._clamps_na_proxy`, `..._not_clamped_when_physical`, `..._nan_input_na_raises_clear_error` |
| **N4** | Re-apply the orphaned fitted linear OPD term. Exact split: s2 part `(c0+c1u_s2x+c2u_s2y)` re-applied after dispatch — **piston as a scalar, slope on the FINE post-upsample grid** (load-bearing for freeform prescriptions, adversarial P2/A, §3a); v2 part `(c3u_v2x+c4u_v2y)` inside every integrator's OPD + saddle-point gradient (Hessian unchanged). | `lenses_maslov.py` | `test_n4_linear_phase_reapplied_offaxis` (True==False, all 3 integrators), `..._freeform_large_slope_is_subsample_invariant`, `..._wellresolved_output_is_subsample_invariant`, `..._piston_reapplied_as_global_phase` |
| **F3** | Route Maslov `_progress` through the suite `call_progress(stage, frac, msg)`; fixes the mid-lens `TypeError` on a standard callback. | `lenses_maslov.py` | `test_f3_suite_style_progress_callback` |
| **F4** | Gate the `tilt_aware_rays` recommendation on a **wrapping-safe** coherence ratio (nearest-neighbour complex-product increments, not `np.gradient(np.angle)`). | `_lens_traced.py` | `test_f4_single_tilt / two_beam_fringe / collimated` |
| **N7** | Docstring: `n_workers` is a no-op on the default polynomial-Newton path. | `_lens_traced.py` | — |
| **N8** | Docstring: analytic model's `sag·θ²` oblique validity boundary + symmetric-relay caveat. | `_lens_real.py` | — |
| **F5** | Revised model-selection decision guide + `carrier` / `on_noncollimated` docs. | `_lens_traced.py` | — |
| **§5.1** | **Carrier-referenced traced** (`carrier=<float>|<ndarray>|'auto'`): reference the correction to the beam's own smooth carrier congruence instead of a plane wave. + **F1** `on_noncollimated` guard (`'warn'|'delegate'|'off'`) on the carrier-residual angular spread. | `_lens_traced.py` | `test_carrier_*`, `test_f1_*` (6) |

19 targeted regression tests (`tests/unit/test_audit_lens_models_2026_07.py`), all
green; the full unit suite was run as the pre-push gate.

## 2. The carrier generalization — what it does and does NOT solve

The sponsor's question was whether the traced model can be generalized past its
collimated-input assumption. The mechanism is implemented and unit-validated:
`_carrier_residual_rms` drives a diverging beam's residual to ~0 with the
matching carrier; `'auto'` recovers a known conjugate (`grad W ≈ x/s`); the F1
guard warns / delegates correctly; `carrier=None` is a regression no-op.

**But the design-119 end-to-end test is honest and important:** `carrier='auto'`
does **not** sharpen the no-MLA 64-emitter imaging. A *single* carrier removes
the common divergence, but each emitter is its own congruence, so a per-lens
residual (**measured ~0.02–0.04 rad even with `carrier='auto'`**) survives and
the spots stay soft — the `on_noncollimated` guard **keeps firing with the
carrier set**, correctly directing the user to `apply_real_lens`. So:

* The carrier's validated win is a **single** divergent / tilted source
  (residual → 0), where it references the traced correction correctly.
* For the **multi-emitter TX case, `apply_real_lens` (analytic) remains the
  correct model** — unchanged from the original field investigation. The guard
  now flags this at runtime instead of blurring silently.
* K-carrier decomposition (one traced pass per congruence, summed coherently)
  is the future extension for multi-congruence fields; past a few carriers it
  re-derives the phase-space integral — at which point use Maslov.

## 3. Corrections to the review (verify-before-implement)

* **F4 discriminator.** The review proposed a coherence ratio from
  `np.gradient(np.angle(E))`. That spikes at 2π phase wraps and mis-classifies a
  *strong* single tilt as incoherent. Implemented with **nearest-neighbour
  complex-product phase increments** (`angle(E[i+1]·conj(E[i]))`), which stay in
  (−π, π] and are wrapping-safe. Validated across single / strong-tilt /
  two-beam-fringe / collimated.
* **N1 scope.** The review said hoisting `G` unblocks *both* non-quadrature
  integrators at full resolution. The memory validation showed
  `stationary_phase` OOMs at a **different** site — `_opd_and_derivs` builds the
  Chebyshev basis for **all in-box pixels at once** (~133 GB at N=16384,
  output_subsample=1), because unlike `local_quadrature` it is **not
  pixel-chunked**. So N1 truly unblocks only the pixel-chunked
  **`local_quadrature`** at scale (61.8 GB observed vs the 451 GB `G` OOM);
  `stationary_phase` needs pixel-banding too (§4).

## 3a. Adversarial-review follow-up (2026-07-03)

A refutation-first review of the remediation diff itself flagged three defects
in the newly-added N3/N4 code. All three were reproduced and fixed; two of the
review's severity claims were themselves corrected on investigation.

* **N3 — NA proxy can exceed 1 (P2, fixed).** A speckled / hard-aperture input
  at fine `dx` gives a 3σ angular-spread estimate `na_input > 1` (measured
  **2.7–5.4** for white-noise fields at `dx = 0.3–0.6 µm`). That made
  `na_proxy = na_lens + na_input > 1`, so **every** pupil ray had
  `v1x²+v1y² > 1` → `N_dir = 0` → the entire chart went grazing and the
  wide-angle content the term was meant to capture was silently dropped.
  **Fix:** clamp `na_proxy` to `0.999` (the physical horizon) with a
  `RuntimeWarning`. Verified end-to-end: `NA_proxy` clamps `2.786 → 0.999`,
  output stays finite with non-zero power. Test `test_n3_broadband_input_clamps_na_proxy`.

* **N3 — FFT temporaries not freed (P3, fixed).** The `input_na` estimate builds
  `fft2(E_in)`, `|·|²`, and two `meshgrid` frequency arrays (each `N²`). Only
  `_P` was being freed; `_v2 / _FX / _FY / _fx` lived until function exit.
  **Fix:** `del _v2` inside the `_Ptot > 0` branch and `del _P, _FX, _FY, _fx`
  after the estimate (guarded so an all-zero input does not `NameError`).

* **N4 — s2-tilt post-multiply aliased on the coarse grid (P2, fixed).** The
  review flagged that the s2 linear-OPD post-multiply ran on the **coarse** field
  *before* the cubic upsample, so a slope above the coarse Nyquist
  (`c1 > N_out_coarse/4`) would alias / flip. **Fix:** piston `_lin[0]` is now
  applied as a **scalar** global phase (grid-invariant, avoids an `N²`
  temporary); the slope terms `_lin[1]/_lin[2]` are applied on the **fine,
  post-upsample grid**, at the coordinate `scipy.zoom` actually sampled
  (reproduced by zooming the coarse axis with the same call — convention-
  independent, no `grid_mode=False` edge-stretch); a `|·| > 1e-6` gate skips the
  `N²` coordinate build when the slope is negligible (the ~17 GB-at-N=32768
  symmetric-lens case).

  **Reachability — corrected twice.** My *first* pass characterised this slope as
  "~0 for every realizable input / unreachable" because the trace ignores literal
  `decenter`/`tilt` keys. **A second, independent adversarial verification
  refuted that**: `surfaces_from_prescription` **does** honor **freeform** odd
  terms (`xy_polynomial`, `zernike`), and a wedge/prism then produces a genuine
  output-position OPL slope — measured `|_lin[1]|` from 0.5 up to **~9776 waves**.
  So the fine-grid placement is **load-bearing, not defensive**: a flat prism
  with `|_lin[1]| = 15.6` waves (≈2× the coarse Nyquist) stays sign- and
  magnitude-invariant between `output_subsample` 1 and 6 *because* the slope is
  applied on the fine grid; a coarse-grid multiply flips it (verifier Claim C:
  fine grid recovers 0.42 cyc/pix vs coarse-then-zoom 0.10). It remains true that
  `|_lin[1]| ~ 1e-10` for rotationally-symmetric prescriptions, and that when a
  large output tilt *also* has an in-integral (pupil / `_lin_v3/_v4`) component
  — a strongly-**powered** freeform lens, not a flat wedge — that component is
  coarse-resolved and aliases above the coarse Nyquist **regardless** of where
  this post-multiply runs (the N2 regime; reduce `output_subsample`). Tests:
  `test_n4_freeform_large_slope_is_subsample_invariant` (the reachable large-real-
  slope regime), `..._wellresolved_output_is_subsample_invariant`,
  `..._piston_reapplied_as_global_phase`.

* **N3 — NaN `input_na` escapes the clamp (found by the second verification,
  fixed).** `input_na = NaN` gave `na_proxy = NaN`; `NaN >= 1.0` is `False`, so it
  slipped past the clamp and reached the trace as `N_dir = NaN`, dying with a
  misleading *"0 rays survived"* TIR message. **Fix:** validate an explicit
  `input_na` is finite and non-negative up front (clear `ValueError`; also
  rejects `inf`), and change the clamp guard to `not (na_proxy < 1.0)` so any
  residual non-finite proxy is caught. Test `test_n3_nan_input_na_raises_clear_error`.

*The §3a fixes were themselves put through an independent, refutation-first
verification workflow (3 adversarial agents executing code); it refuted the N4
"unreachable" claim and the NaN path above, and could not break the N4 restructure's
physics (True==False intensity to 2.15e-4; piston `|E|`-invariant to 3e-12;
fine-grid slope Nyquist-correct at injected c1=20). The corrections here reflect
its findings.*

## 4. Deferred (documented follow-ups)

| Item | Why deferred |
|---|---|
| **N6** eigen-rotated, tapered `local_quadrature` window | Accuracy refinement for astigmatic (rotated-saddle) charts; needs a regression oracle; ~no benefit on centered relays; rewriting the production integrator's hot loop for it was not worth the risk in this pass. |
| **`stationary_phase` pixel-banding** | Its `_opd_and_derivs` is not pixel-chunked (§3); band it like `local_quadrature`'s `PX_CHUNK` loop to make it scale. |
| **F2** quadrature output-row-banding | Only needed for the exact-integral validation mode; `local_quadrature` (N1) is the production path. |
| **carrier K-decomposition** | The multi-congruence extension (§2). |
| **Performance** (T-P1 prepared-traced, T-P2 inverse-map fit, M-P2 G-factorization, M-P5/6 composite maps + ROI, A-P1 prepared screens, M-P4/7 numba/float32) | Optimizations; each warrants its own validation cycle. |

## 5. Consumer wiring (Reverse_Symmetric_ASM, not this repo)

`tx_design_study_sim.py` gained `LENS_TRACED_CARRIER`; `run_poc_119_120_v518.py`
gained `POC_TRACED_CARRIER` — used only for the design-119 carrier validation
above.

---

*Remediation performed single-context against lumenairy 5.18.1, 2026-07-03.*

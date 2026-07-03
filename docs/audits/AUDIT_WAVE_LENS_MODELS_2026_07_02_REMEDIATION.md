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
| **N3** | Size the pupil chart from `na_lens + na_input`, `na_input` = 3σ of the input angular spectrum (one FFT); new `input_na=` kwarg + coverage warning. | `lenses_maslov.py` | `test_n3_input_na_widens_chart`, `..._coverage_warning` |
| **N4** | Re-apply the orphaned fitted linear OPD term. Exact split: s2 part `(c0+c1u_s2x+c2u_s2y)` as an output post-multiply (constant in v2 → factors out of the integral); v2 part `(c3u_v2x+c4u_v2y)` inside every integrator's OPD + saddle-point gradient (Hessian unchanged). | `lenses_maslov.py` | `test_n4_linear_phase_reapplied_offaxis` (True==False to ~1e-7, all 3 integrators) |
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

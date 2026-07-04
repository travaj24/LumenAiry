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

## 4. Deferred-items program (2026-07-03, second pass)

The originally-deferred follow-ups were then worked through as their own
program.  Status below distinguishes **implemented + validated** from
**assessed** (design recorded, deferred with a concrete reason).

### 4.1 Implemented + validated

| Item | What shipped | Validation |
|---|---|---|
| **`stationary_phase` pixel-banding** | `_opd_and_derivs` evaluated in memory-budgeted pixel bands (seam `_SP_PIXEL_CHUNK`); fixes the ~133 GB full-grid basis build. | Byte-identical for realistic bands; 3.8e-9 (30x below f32 ULP) at the degenerate 1-pixel band (np.sum reduction shape only). |
| **F2 quadrature output-row-banding** | Pass the two per-axis Vandermondes instead of the full `(N_out^2, M)` `G`; build only a per-output-row-band `G` (seam `_QUAD_ROW_BAND`). Kills the 451 GB allocation. | Byte-identical (rel=0.0), numpy AND numexpr, all band sizes. |
| **M-P2 G-factorization** | `G = Ty (x) Tx` Kronecker: scatter `H`'s rows into a `(P,P,.)` tensor, one `einsum` per band -- no `G` at all, ~M/P fewer FLOPs (seam `_QUAD_FACTORIZE`). | Factorized vs explicit `G @ H` = 2.5e-9 (below f32 ULP), numpy + numexpr. |
| **T-P1 prepared-traced** | `prepare_real_lens_traced(...)` / `PreparedTracedLens`: caches the input-independent screen; each call = one `apply_real_lens` + one complex multiply. New public API, exported top-level. | Prepared == direct to 4.6e-16 (c128) / 1.5e-7 (c64); reuse on 2 different inputs matches 2 direct calls to <1e-12; **measured 55x per-call speedup**, break-even after <1 reuse; `carrier='auto'` rejected. |
| **Multi-emitter traced** (carrier K-decomp) | `apply_real_lens_traced_multi(emitter_fields, ..., carriers=, reuse_prepared=)`: applies the traced model PER emitter (each a single congruence) and coherently sums -- the correct way to use the non-linear traced model on a multi-emitter scene; the known emitters ARE the K congruences (no blind segmentation).  Reuses a T-P1 screen per shared carrier. New public API, exported. | Single-emitter `multi([E]) == traced(E)` to 2.1e-15; reuse == no-reuse == sum-of-direct (byte-identical); captures the large traced non-linearity (171% `traced(sum)` vs `sum traced` on an aberrated divergent-overlap case).  Honest regime guidance in the docstring: no benefit for the exactly-linear analytic model or well-corrected lenses; for the no-MLA multi-angle case analytic remains the right model. |
| **non-divisible-N** | End-to-end test + a forced-pad-guard test (shorten the 1-D axis zoom). | Proved `scipy.zoom` returns exactly N for all N=16..4096 x ss=2..16 (0 mismatches) -> the pad guards are defensive-only. |
| **JAX-x64 flake** | Made the `test_twins_raise_without_x64` subprocess env explicit (`JAX_ENABLE_X64=0`). Root cause: a sibling module's `os.environ.setdefault("JAX_ENABLE_X64","true")` at import leaks into the inherited subprocess env. | Passes under a deliberately polluted env. |

### 4.2 Assessed (design recorded; deferred with reason)

| Item | Assessment |
|---|---|
| **N6** eigen-rotated `local_quadrature` window | **Implemented, validated, REVERTED.** The rotation (align the window to the Hessian eigenvectors: `theta = 0.5 atan2(2 H34, H33-H44)`, offset `xi1*sigma1*e1 + xi2*sigma2*e2`, area element unchanged) is mathematically correct and reduces to the legacy path when axis-aligned. But a clean benefit could not be isolated against a dense-quadrature oracle: `local_quadrature` already diverges ~67% (intensity) from the oracle in the strong-astigmatism regime where window rotation would matter (partly aperture-vs-grid truncation), and rotation was neutral-to-worse in every tested case. Shipping it would be an unvalidated behavior change. Rotation math preserved here for a future cycle that first closes the `local_quadrature`-vs-oracle accuracy gap (which is the real issue). |
| **A-P1** prepared analytic screens | **IMPLEMENTED.** `prepare_real_lens(...)` / `PreparedAnalyticLens` caches the per-surface `exp(-i k0 (n2-n1) sag)` screens + entrance mask (input-independent) and reuses the ASM legs -- whose transfer functions are already cached inside `angular_spectrum_propagate`, so no ASM-math reimplementation was needed and the divergence surface is just the screen multiply. Scoped to the default path (NumPy / ASM / plain conic+aspheric); the factory raises `NotImplementedError` for decenter / tilt / freeform / biconic / clear-aperture / stop / mirror surfaces and the slant / fresnel / absorption / seidel / surface-frame / GPU / non-ASM modes. Validated: prepared == direct to **3e-15** (complex128) / 1.45e-6 (complex64, float32-ULP); reuse on two inputs matches two direct calls to <1e-12; **2.85x faster on an 8-surface lens (N=512)**; every unsupported config rejected. New public API, exported top-level. |
| **T-P2** inverse-map fit | **IMPLEMENTED (opt-in).** `inversion_method='fit'`: fit `opl(x_out, y_out)` by scattered total-degree Chebyshev lstsq from the traced ray samples, convex-hull-masked, evaluated on the exit grid -- no per-pixel Newton. Default stays `'newton'` (zero risk). Validated 2.6e-6 vs Newton; **2.42x faster at the default `ray_subsample`** (slower only at `ray_subsample=1`, where the optimized parallel Newton wins -- follow-up: skip the redundant Newton spline setup on the fit path). Not yet promoted to default pending the sub-nm cemented-doublet campaign. |
| **M-P4** numba 4-var Chebyshev kernel | **IMPLEMENTED.** A `@njit(parallel, fastmath)` kernel for the Maslov `_opd_and_derivs` (all six outputs via O(poly_order) stack recurrences); both integrator closures delegate to a shared `_opd6` dispatch (Numba default, NumPy fallback, `_MASLOV_USE_NUMBA` seam). Kernel vs NumPy ~1e-15 on all outputs; end-to-end byte-identical; **13.6x faster** (stationary_phase, N=256). CuPy twin deferred (needs integrator-wide GPU array support first). |
| **M-P5** prepared-Maslov / wavelength-rescale | **Per-call half IMPLEMENTED, rescale half declined.** The three fit solves (OPD, s1x, s1y) share one design matrix -- collapsed into a single multi-RHS `lstsq` (one SVD): **~2.9x fit**, ULP-close, universal. The full cache-and-rescale (reuse the trace/fit across wavelengths, `opd_waves = opl_metres/lambda`) is **dispersion-limited**: ignoring `n(lambda)` costs 0.15 waves over just 2 nm (1.47 over 20 nm, N-BK7), so caching is valid only over a ~2 nm window where the benefit is marginal (you refit every ~2 nm anyway). Declined in favour of the universal fit-stacking + the now-cheaper (M-P4-accelerated) re-trace. |
| **M-P6** ROI / image-plane evaluation | **IMPLEMENTED.** `roi=(cx, cy, half_width)` evaluates only a square window; the returned field is **byte-identical to the full-grid slice** (each output pixel integrates independently), on- and off-axis, at O(roi_n^2) instead of O(N^2) integrand evals. Measured 1.5x (N=256) -> 3.4x (512) -> **8.8x (1024)** for a 40x40 window, growing with N as the trace+fit floor shrinks. Follow-up: composing a free-space leg into the canonical map to place a downstream (focus-plane) ROI directly. |
| **M-P7** float32 quadrature arrays | Assessed and **declined**: the quadrature integrator is the exact-integral *validation reference*; trading its float64 precision for float32 bandwidth is counterproductive there. Float32 would make sense only on the production `local_quadrature` path, where it warrants its own power-normalized validation. |
| **carrier K-decomposition** | **Tractable form IMPLEMENTED** as `apply_real_lens_traced_multi` (§4.1): when the K congruences are the KNOWN emitters, no blind segmentation is needed -- propagate each per-emitter field through the traced lens and coherently sum. The remaining open piece is the AUTO variant: segmenting the K congruences from a single *blended* field's local-wavevector map (a clustering problem) when the emitters are not separately available. Past a few carriers it re-derives the phase-space integral -> use Maslov. |

## 5. Consumer wiring (Reverse_Symmetric_ASM, not this repo)

`tx_design_study_sim.py` gained `LENS_TRACED_CARRIER`; `run_poc_119_120_v518.py`
gained `POC_TRACED_CARRIER` — used only for the design-119 carrier validation
above.

---

*Remediation performed single-context against lumenairy 5.18.1, 2026-07-03.*

# Wave Lens Models — Companion Review, Fix Plan & Performance Ledger — 2026-07-02 (v5.18.1)

Companion to `AUDIT_WAVE_LENS_MODELS_2026_07_02.md` (the field-driven design-119 audit).
Scope of this review: full read of `lumenairy/elements/_lens_traced.py` (2,680 lines) and
`lumenairy/elements/lenses_maslov.py` (1,082 lines); targeted read of
`lumenairy/elements/_lens_real.py` (per-surface loop, `_propagate_through_glass`, banding
path).  Method: **code-level verification** of the original audit's claims plus independent
physics analysis of each model's kernel.  The design-119 end-to-end *measurements* in the
original report were NOT rerun here — they are accepted as reported; every verdict below is
grounded in the current source at the cited lines, not in new simulations.  Read-only — the
fixes below are recommendations.

Sponsor questions addressed: (1) is the traced model really collimated-only, and can it be
generalized? (2) is Maslov the universal answer, and what does it need? (3) is output-banding
Maslov a bad idea even though it is not the true blocker? (4) how can all three propagators
be made faster and more generally useful?

---

## 1. Executive Summary

The original audit's two headline conclusions **survive verification** — the traced model's
correction is collimated-referenced and unguarded (F1), and Maslov is the principled
universal form — but three load-bearing details change what the right fixes are:

* **The F1 mechanism section is wrong about *how* traced fails.**  Traced is not a bare
  "ray-traced phase mask": its default mode is
  `E_out = apply_real_lens(E_in) · exp(i·Δφ)` with
  `Δφ = k₀·OPL_ray,pw − angle(apply_real_lens(𝟙))` — the input's divergence/tilts ARE
  carried (by the analytic split-step leg); only the *differential correction* is
  plane-wave-referenced (`_lens_traced.py:2643-2650`, rays launched `L = M = 0` at
  `:1950-1951`).  The error on non-collimated input is therefore the **misapplied
  (ray − analytic) plane-wave discrepancy**, not "thickness·tanθ·(n−1), many waves" (that
  estimate describes the legacy `preserve_input_phase=False` mode, which did not run).  A
  second, unreported mechanism likely dominates on finite-conjugate relays: the plane-wave
  **reference field itself is ill-conditioned** there (N5).  The *conclusion* — traced is
  only valid when the input congruence ≈ collimated — stands.

* **Maslov's "unusable at scale" finding is three-quarters a misplaced allocation.**  The
  451 GB `G` design matrix is built **unconditionally** but consumed **only by the default
  `'quadrature'` integrator**; the `stationary_phase` and `local_quadrature` integrators
  receive `G` in their signatures and never touch it (N1).  Hoisting one allocation makes
  `local_quadrature` runnable at `output_subsample=1`, full 16384² resolution, in bounded
  memory, **today**.  Separately, quadrature mode at `n_v2=32` is *physically*
  under-resolved at the 119 NA regardless of memory (N2) — banding alone would have made it
  fit and still speckle.

* **Maslov has an unflagged input-coverage hole** that affects exactly the 119 workload: the
  pupil-direction chart is sized from the *lens* EFL (`na_proxy`), not from the *input
  field's* angular content, with no `input_na` parameter and no coverage guard (N3).  A
  0.1–0.2 rad divergent source can fall off the chart and return silently dim/wrong output
  at **any** resolution.

* **Banding (the original F2 fix) is still a good idea** — it is necessary for the
  quadrature integrator's per-chunk intermediates at full output resolution and composes
  with everything else — it is simply mis-sequenced as the *first* fix and insufficient
  alone.  See §4.1.

* **Generalization**: the traced model can be extended to the divergent-source / emitter-
  array case that triggered the investigation with a **carrier-referenced correction**
  (§5.1) at ~its current cost — no need to "become Maslov" for that regime.  Maslov remains
  the universal solver for genuinely multi-congruence / near-caustic fields; §5.2 gives its
  production path.

* **Performance**: the single biggest cross-cutting win is the library's own
  prepared-object pattern (cf. `PreparedRCWA2D`): for the traced model the *entire*
  correction Δφ is input-independent per (prescription, λ, grid) — a `prepare_…` variant
  reduces repeated calls (optimizers, tolerancing, multi-field sweeps) to **one
  `apply_real_lens` + one complex multiply** (§6.1).

---

## 2. Verification ledger for the original findings

| Finding | Verdict | Notes |
|---|---|---|
| **F1** (traced collimated-only, unguarded) | **CONFIRMED, mechanism corrected** | Ray launch `L=M=0` verified (`_lens_traced.py:1950-1951`); docstring limitation verified (`:1211-1213`).  But §2 of the original describes the `preserve_input_phase=False` mode; the default hybrid's error law is different (see §1 and N5).  The recommended guard is right; its discriminator should be the **carrier-residual** spread, not raw angular spread (§5.1), or it will wrongly reject emitter-array fields the carrier fix handles. |
| **F2** (Maslov O(N²·M) design matrix; no feasible setting) | **PARTIALLY REFUTED** | The allocation and its size are real (`lenses_maslov.py:498`; M = C(order+4,4): 70 at order 4, 210 at order 6).  But `G` is only used by `'quadrature'` (N1), so "no setting fits memory + Nyquist" holds only for that integrator — `local_quadrature` has no such dilemma once the allocation is hoisted.  The §4 memory table is correct arithmetic aimed at the wrong bottleneck; the observed speckle is equally consistent with v₂ under-resolution (N2). |
| **F3** (progress-callback signature crash) | **CONFIRMED** | `lenses_maslov.py:209-216`: `progress(phase=…, fraction=…, elapsed=…, note=…)` then 4-positional fallback; the suite convention (`call_progress`) is `(label, frac[, msg])`.  Fix as recommended (adapter or signature change). |
| **F4** (`tilt_aware_rays` catastrophic on multi-source fields) | **CONFIRMED, with root cause** | `_sample_local_tilts` (`_lens_traced.py:770-927`) derives *per-pixel* directions from `angle(E_shift·conj(E))` with σ = 4 px amplitude-weighted smoothing.  On a 64-emitter coherent interference pattern the raw gradient is fringe-aliased and 4 px of smoothing is far too little at N = 16384 — the "multi-source has no single direction" framing is imprecise: at each pupil point the emitter directions differ only by (array size)/(throw), i.e. the field IS quasi-single-congruence; the failure is the **estimator**, not the physics.  That is why the carrier fix (§5.1) works where per-pixel tilts do not. |
| **F5** (no model-selection contract) | **CONFIRMED** | Adopt the revised table in §5.4 (the original's table needs the analytic model's own validity caveat added — see N8). |
| §3 "on-axis astigmatism is physically impossible" | **NEEDS CAVEAT** | A symmetric relay + symmetric (wrong) Δφ acting on a rotationally-symmetric input cannot produce astigmatism.  It can on an **elliptical grating-coupler mode** (different NA_x/NA_y sample the angle-dependent error differently).  The measured 17.6 × 9.4 µm ellipticity is evidence of *source asymmetry × angle-dependent model error*, not of an extra numerics defect — do not chase a phantom anisotropy bug. |

---

## 3. New findings

### N1 (P1 — the actual Maslov scale blocker; one-line fix)
`G = np.empty((N_out_coarse², M))` is allocated **before** the integrator dispatch
(`lenses_maslov.py:498-502` vs the dispatch at `:511-547`) and is consumed **only** inside
`_integrate_quadrature` (`G @ H_*` at `:707-713`).  `_integrate_stationary_phase`
(`:768-914`) and `_integrate_local_quadrature` (`:917-1077`) accept `G` and never reference
it — they evaluate via `_opd_and_derivs` in `PX_CHUNK`-banded pixel chunks with bounded
memory.  Consequence: at N = 16384, `output_subsample=1`, the *non-quadrature* integrators
OOM on an allocation they never read.  **Fix:** build `G` inside the quadrature branch only
(and see M-P2 for eliminating it there too).

### N2 (P1 — quadrature v₂ resolution is physically insufficient at production NA)
The uniform Tukey-windowed v₂ quadrature uses `n_v2 = 32` samples (default) against an
integrand phase `2π·OPD(s₂, v₂)` that oscillates ~NA·aperture/λ times across the chart —
**~10³ cycles** at 0.2 rad over mm-scale apertures at λ = 1.31 µm.  No memory fix makes
32-point uniform quadrature converge there; the 12 % power loss + speckle in the original §3
Maslov row is consistent with this as much as with output undersampling.  **Fix:** at
production NA, `stationary_phase` / `local_quadrature` are the correct evaluators (that is
the standard asymptotics hierarchy); add an automatic estimate of v₂ oscillation count
(cheap: from the fitted `coef_opd` derivative range) that warns / auto-switches when
`n_v2` is under-resolved.

### N3 (P1 — Maslov chart does not cover the input's angular content)
The traced bundle's entrance directions span `|v₁| ≤ na_proxy` with
`na_proxy = r_aperture / max(|EFL|, 10·r_ap)` (or a thickness heuristic; `collimated_input`
→ 1e-5) — `lenses_maslov.py:303-330`.  The v₂ stationary point selects rays whose
*entrance* direction matches the field's local direction, so input angular content beyond
the traced range is simply **not represented**: the polynomial chart is extrapolated or the
stationary point clips at `u_v2 = ±1`, and the output is silently dim/wrong at any
resolution.  There is no `input_na` parameter and no coverage check.  **Fix:** size the v₁
sampling from the measured angular spectrum of `E_in` (second moment of |FFT(E_in)|², one
FFT) plus the lens-NA margin; expose `input_na=` for callers who know it; and flag bright
output pixels whose converged `u_v2*` sits at the clip boundary (the Newton results are
already in hand at `:850-853` / `:993-994`).

### N4 (P2 — `extract_linear_phase=True` permanently drops the linear OPD)
`linear_coeffs` is fitted and subtracted (`lenses_maslov.py:374-384`) and **never
re-applied** — not passed to any integrator, not restored on the output.  Benign for a
centered rotationally-symmetric lens (linear component ≈ piston); silently wrong (lost
output tilt, shifted stationary points) for decentered / tilted / off-axis configurations.
**Fix:** keep the subtraction (it is good fit conditioning) and add the linear term back
*analytically* inside each integrator's phase (exact, ~free): it is a closed-form
`c₀ + c₁u_s2x + c₂u_s2y + c₃u_v2x + c₄u_v2y` added to `opd_*` at evaluation.

### N5 (P2 — traced's plane-wave reference field is ill-conditioned on finite-conjugate relays)
`phase_analytic_lens = angle(apply_real_lens(ones))` (`_lens_traced.py:1702-1722`,
`:1754-1761`).  A relay designed for a divergent source, fed collimated light, focuses at
the wrong conjugate — the reference field can pass at/near focus inside or at the exit of a
group, where `|E_pw| ≈ 0` over most of the aperture and its `angle()` is numerical noise.
Δφ then injects garbage phase precisely where the real beam has light.  This is a
*reference-construction* failure, independent of the ray-launch direction, and any
generalization must fix both (the carrier reference in §5.1 does).  It also predicts that
F1's blur severity is design-dependent (worst where the pw conjugate lands near a plane of
interest) — consistent with the original's observation that some groups hurt more than
others.

### N6 (P3 — `local_quadrature` window: no Hessian rotation, hard edge)
The local window samples σ₁ along x and σ₂ along y (`lenses_maslov.py:1026-1031`) using the
Hessian *eigenvalues* (`:1006-1015`) but **not the eigenvectors** — for a rotated saddle
(H₃₄ ≠ 0, e.g. astigmatic charts) the window misaligns with the stationary ridge.  The
window is also a hard-edged uniform Riemann sum over ±`window_sigma` (`:1020-1024`) — the
oscillatory integrand's truncation error rings at the edge.  **Fix:** rotate the sample
frame into the Hessian eigenbasis (2×2, closed form) and apply a smooth (Gaussian/Tukey)
taper with the matching analytic weight correction.

### N7 (P3 — traced: worker pool silently unavailable on the default fit path)
`_invert_newton_parallel` forces the serial path for `newton_fit='polynomial'` — the
**default** — because the process worker only rebuilds SciPy splines
(`_lens_traced.py:2363-2369`; the in-code comment acknowledges it).  The numba kernel
mitigates on CPU; still, the documented `n_workers` knob is a no-op on the default path.
**Fix:** add a worker-side polynomial path (ship coefficients, not knots — cheaper to
pickle than splines), or document the limitation at the `n_workers` parameter.

### N8 (P3 — the analytic model's own validity boundary should be stated)
`apply_real_lens` applies per-surface **normal-projected** thin screens
(`(n₂−n₁)·sag(x,y)`, `_lens_real.py:971-984`) with exact homogeneous propagation between
surfaces.  Its oblique error per surface grows with sag × angle² (plus walk-off within the
sag zone) and is *design-dependent*; the sharp 5.4 µm result on 119 benefited from a
**symmetric relay** (even-order model errors partially cancel across the symmetric groups).
"Analytic is the correct default for divergent sources" is right for this family of designs
but is not a universality claim — the decision table (§5.4) should carry this caveat so the
next investigation doesn't over-trust analytic on fast, asymmetric trains.

---

## 4. Fix plan (priority order)

| # | Fix | Model | Effort | Effect |
|---|---|---|---|---|
| 1 | **N1**: hoist `G` into the quadrature branch | maslov | trivial | `local_quadrature`/`stationary_phase` run at full resolution today |
| 2 | **F1 guard, carrier-aware**: measure input angular spread (one FFT) — warn; with `on_noncollimated='delegate'`, fall back to `apply_real_lens`; once §5.1 lands, measure the **carrier-residual** spread instead | traced | small | turns the silent regression class into a one-line diagnostic without rejecting carrier-fixable inputs |
| 3 | **N3**: input-NA-aware chart + coverage flag | maslov | small | correctness for divergent sources at any resolution |
| 4 | **F3**: progress-signature adapter | maslov | trivial | API parity; no more mid-lens `TypeError` |
| 5 | **N4**: re-apply `linear_coeffs` analytically in all three integrators | maslov | small | off-axis / decentered correctness |
| 6 | **§5.1 carrier-referenced traced** (`carrier=` / `carrier='auto'`) | traced | medium | generalizes traced to the divergent-source / emitter-array regime; also resolves F4 and N5 |
| 7 | **F2 banding** of the quadrature per-chunk intermediates (+ M-P2 factorization) | maslov | small–medium | exact-integral mode at full resolution; see §4.1 |
| 8 | **N2**: v₂-resolution estimate → warn / auto-switch integrator | maslov | small | prevents silent quadrature speckle |
| 9 | **F4 guard**: deprecate raw `tilt_aware_rays` in favour of the carrier path; if kept, gate on a multimodality diagnostic and stop the `:1936` warning from recommending it unconditionally | traced | small | closes the misleading escape hatch |
| 10 | **N6**: eigen-rotated, tapered local window | maslov | small | accuracy on astigmatic charts |
| 11 | **F5 / N8**: publish the revised decision table (§5.4) in the three docstrings + designer guide | all | trivial | selection contract |
| 12 | **N7**: worker-side polynomial Newton (or doc fix) | traced | small | honest `n_workers` |

### 4.1 Is banding Maslov a bad thing, given it's not the blocker?

**No — banding is correct and should still land; it is just fix #7, not fix #1.**  Three
reasons it stays on the list:

1. Even after N1 (hoist) and M-P2 (kill `G` via factorization), the **quadrature**
   integrator's per-chunk working arrays are `(N_out², chunk_v2)` — at N = 16384,
   chunk = 64 that is ~137 GB *per array* (there are seven).  Row-banding the output is
   what bounds those: a band of R rows costs `(R·N_out · chunk)` per array (≈ 2 GB at
   R = 256), independent of total N.  So full-resolution *exact-integral* (non-asymptotic)
   Maslov needs banding regardless.
2. Banding is **exactly safe** here: every integrator is per-output-pixel independent (the
   v₂ integral/stationary evaluation never couples output pixels), so bands introduce no
   approximation and no seams — same argument as the validated `sag_chunk_rows` pattern in
   the traced/analytic models it mirrors.
3. It composes with everything else (factorized evaluation, GPU, float32) and gives the
   same memory ceiling to the `local_quadrature` pixel chunks if `PX_CHUNK` is ever raised
   for GEMM efficiency.

What banding does **not** do — and why it must not be sold as *the* fix — is address N2
(v₂ under-resolution: banded quadrature at n_v2 = 32 still speckles at 0.2 rad) or N3
(chart coverage).  Sequence: N1 → N3 → banding+factorization → N2 auto-switch.

---

## 5. Generalization roadmap ("make all of them as generally useful as possible")

### 5.1 Traced → carrier-referenced traced (the concrete generalization)

Physical basis: a multi-emitter field is *not* fundamentally multi-congruence at the lens.
At pupil point x, the ray from emitter j has direction ≈ (x − x_j)/d; the spread across
emitters is (array size)/(throw) — typically ≪ the common divergence.  The field is one
smooth **carrier** (the mean spherical wave) plus a small angular residual.  Design:

1. **Carrier**: accept `carrier=` as (a) a prescription-level conjugate ("point source at
   z = −s", the usual known quantity), (b) an explicit low-order wavefront, or (c)
   `'auto'`: robust low-order (Zernike/poly) fit of the intensity-weighted smooth phase —
   *never* per-pixel gradients (that is F4's failure).
2. **Rays**: launch the OPL bundle along carrier normals (the existing `tilt_aware`
   plumbing, fed by the fitted W(x) instead of `_sample_local_tilts`).
3. **Reference**: `Δφ = k·OPL_ray(carrier) − angle(apply_real_lens(exp(i·k·W)))` — the
   reference congruence matches the beam, and the reference field is well-conditioned at
   the exit plane because it focuses where the real beam does (fixes N5).
4. **Guard** (replaces the naive F1 guard): after carrier removal, the residual angular RMS
   × aperture / λ.  Warn / delegate above threshold.  This is cheap (one FFT of
   `E_in·exp(−ikW)`) and is the *right* discriminator: raw angular spread would reject the
   emitter-array case this fix handles.
5. **Validity boundary** (state it plainly in the docstring): one carrier fails for
   genuinely multi-congruence fields — comparable-power beams at well-separated angles
   (immediately post-DOE at large split angles), or planes at/near an intermediate focus /
   caustic.  K-carrier decomposition (K traced passes, summed coherently) extends it, but
   as K grows this re-derives the phase-space integral — at that point use Maslov, don't
   grow traced.

Cost: same order as today (the carrier reference replaces the plane-wave `amp(pw)` leg
one-for-one; the carrier fit is negligible).  Compatible with the prepared-object caching
in §6.1 (cache key gains the carrier parameters).

### 5.2 Maslov → the production universal solver

Sequence (mostly §4 items, restated as a path): **N1 hoist** → **N3 input-NA chart +
coverage flag** → make **`local_quadrature` the default at scale** (with N6's rotated,
tapered window) and keep `quadrature` as the exact-integral validation mode (banded +
factorized, §6.2) → **N4 linear-phase re-application** → **N2 auto-switch** → then the two
capability upgrades that change what Maslov is *for*:

* **ROI / image-plane evaluation** (M-P6): compose the free-space leg into the canonical
  map (the fit machinery is unchanged — trace to the image plane instead of the exit
  vertex) and evaluate only in caller-specified output windows.  For spot-imaging studies
  the output pixel count drops by 10³–10⁴× and the whole memory/Nyquist discussion becomes
  moot.  This is the highest-leverage change for the TX workflow.
* **Composite maps** (M-P5): fit one canonical map per contiguous prescription train
  (trace through all groups at once) instead of per-lens application with intermediate
  grids — fewer evaluations, no intermediate-plane sampling constraints, no per-lens error
  compounding.  (Segments remain split at DOEs / free-space elements that are not part of
  the prescription.)

### 5.3 Analytic → keep as the fast default, with its boundary stated

No structural change recommended.  Add the N8 caveat to its docstring, and the shared
angular-spread diagnostic as an *informational* note when per-surface sag × angle² error
estimates exceed λ/4 (computable from the prescription + measured input NA — the same
estimator as fix #2, reused).

### 5.4 Revised model-selection table

| Input / situation | Use | Why |
|---|---|---|
| Collimated / MLA-relayed input; thick or cemented optics; sub-nm OPD wanted | `apply_real_lens_traced` (as-is) | collimated reference assumption holds; exact geometric OPD |
| Divergent / converging / tilted source, or emitter array with modest residual spread, through a multi-element train | `apply_real_lens_traced(carrier=…)` **once §5.1 lands**; until then `apply_real_lens` | carrier-matched exact OPD; guard = residual spread |
| Same, when a fast *estimate* suffices, or the design is symmetric / well-corrected | `apply_real_lens` | handles all angles via ASM legs; per-surface thin-screen error is the (design-dependent) accuracy floor — see N8 |
| Genuinely multi-congruence fields, planes at/near a caustic, autodiff design loops, or "I need the answer at the image plane only" | `apply_real_lens_maslov` (after §5.2 items 1–3) | phase-space chart is caustic-safe and input-general once N1/N3 land |
| Aberration-free control / isolating model vs geometry | `thin` (ABCD) | paraxial-perfect reference |

Every row's precondition should be **machine-checked** by the shared angular-spread /
carrier-residual estimator (fixes #2, #9, N3) so a wrong choice raises or delegates at
runtime rather than blurring silently.

---

## 6. Performance ledger (all three propagators)

### 6.1 `apply_real_lens_traced`

* **T-P1 (largest; new): `prepare_real_lens_traced(prescription, wavelength, dx, N[, carrier])`.**
  The entire traced leg is **input-independent** per (prescription, λ, grid[, carrier]):
  the ray trace, the Chebyshev/spline fits, the Newton inversion, `phase_analytic_lens`,
  and hence the whole `Δφ` map and the exit-aperture/NaN masks.  A prepared object caches
  `Δφ` once; each call is then **one `apply_real_lens(E_in)` + one banded complex
  multiply** — ≥2× per call, and it deletes the trace/fit/Newton stages from optimizer /
  tolerancing / multi-field loops entirely (the workloads the persistent worker pool was
  built for, `_lens_traced.py:222-235`).  Note the input-dependent `newton_amp_mask_rel`
  optimization is skipped in prepared mode (compute Δφ full-grid once) — a one-time cost
  amortized by the first reuse.  This mirrors the library's `PreparedRCWA2D`/
  `PreparedPMM2D` precedent.
* **T-P2 (new): fit the inverse map directly and delete Newton.**  The forward trace
  already yields scattered exit samples (x_out, y_out, OPL); fitting Chebyshev polynomials
  of (OPL, x_in, y_in) **as functions of (x_out, y_out)** by scattered least squares — the
  exact pattern Maslov uses for its chart (`lenses_maslov.py:389-402`) — replaces the
  per-pixel Newton stage (~50 % of the non-amp budget) with one small lstsq + one
  polynomial evaluation.  Same smoothness argument as the existing forward fit; validate on
  the cemented-doublet sub-nm cases before switching the default.
* **T-P3 (exists; document as a recipe):** `fast_analytic_phase=True` (skips the second
  `apply_real_lens` leg, ~20 % at large N — validated ≤10 nm on ≥F/6 designs),
  `parallel_amp`, `use_gpu`/`amp_use_gpu`, `sag_chunk_rows` (banded assembly), the numba
  Chebyshev kernel.  These compose; a "fast production profile" note in the docstring
  would save users rediscovering them.
* **T-P4:** N7 (worker-side polynomial path) if Newton is retained.

### 6.2 `apply_real_lens_maslov`

* **M-P1:** N1 hoist (memory, not speed — but it is the enabler).
* **M-P2 (new): eliminate `G` structurally in quadrature mode.**  `G` is a Kronecker
  product: `G[(iy,ix), m] = Ty[k₂,iy]·Tx[k₁,ix]`.  `G @ H` therefore factorizes: scatter-add
  `H`'s rows over their (k₁,k₂) pairs into a (order+1)² grid, then two small matmuls
  (`Ty᳕ · Ĥ · Tx`) per chunk column — **~M/(order+1) ≈ 30× fewer FLOPs** (order 6) on the
  seven integration GEMMs and zero (N², M) allocation.  Combined with output row-banding
  (§4.1) this makes the exact-integral mode both fit and fast at full resolution.
* **M-P3:** banding (§4.1).
* **M-P4:** reuse the numba Chebyshev value+derivative kernel from
  `_lens_traced._get_cheb2d_val_grad_numba` (generalized to 4 variables) inside
  `_opd_and_derivs` — it currently rebuilds four Vandermondes + gathers per Newton
  iteration and per pixel chunk in pure NumPy (`lenses_maslov.py:796-821`, `:945-970`).
  GPU (CuPy) is equally mechanical: everything is GEMM/elementwise.
* **M-P5:** fit-once reuse: the traced rays and the s₁/OPD fits are geometry-only (OPD is
  stored in waves but derived from meters, `:353-354`) — a prepared-Maslov re-scales
  OPD/λ per wavelength and refits only when dispersion changes; plus composite maps per
  contiguous train (§5.2).
* **M-P6:** ROI / image-plane evaluation (§5.2) — the largest practical cost lever
  (10³–10⁴× fewer output pixels for spot studies).
* **M-P7:** float32 for the quadrature field/Jacobian arrays (phase stays float64 in the
  `exp(2πi·OPD)` argument): halves bandwidth on the GEMM-bound stage; validate against the
  power-normalized output.

### 6.3 `apply_real_lens` (analytic)

* **A-P1 (new): prepared screens.**  The per-surface sag/OPD screens are recomputed every
  call (`_lens_real.py:967-985`) but depend only on (prescription, λ, grid).  A prepared
  variant caches the per-surface OPD bands (float32-eligible via the existing `sag_dtype`
  machinery) or the fused complex screens; the per-call cost then reduces to the FFT legs
  + screen multiplies.  Biggest effect on many-surface prescriptions and optimizer loops;
  memory-bounded by storing banded/float32 OPD rather than complex screens.
* **A-P2 (exists):** GPU path, numexpr fusion, `sag_chunk_rows` banding, pyFFTW plan cache
  — already in place; the remaining FFT cost is irreducible for a split-step method.
* **A-P3 (shared):** when called from `apply_real_lens_traced` in prepared mode (T-P1), the
  duplicate `amp(pw)` leg disappears entirely — the two optimizations compound.

### 6.4 Shared

* The angular-spread / carrier-residual estimator (fix #2) costs one FFT and serves all
  three models' guards plus Maslov's N3 chart sizing — implement once in a shared helper.
* All prepared objects should enroll in `lumenairy._cache_registry` per the v4.16.0
  contract if module-level, or follow the `_PreparedPMMStack` instance-cache precedent if
  per-object.

---

## 7. Verification notes

* Code-verified at the cited lines: F1's launch/reference structure, F3's signature, F4's
  estimator, N1 (G unused by two integrators — no `G @` in either body), N2's n_v2 default
  and Tukey quadrature, N3's `na_proxy` derivation, N4's orphaned `linear_coeffs`, N6's
  axis-aligned window, N7's forced-serial polynomial path, N8's normal-projected screens.
* Physics-argued (not re-simulated): the corrected F1 error law (§1), the quasi-single-
  congruence property of emitter arrays (§5.1), the v₂ oscillation-count estimate (N2),
  the stationary-phase selection of input direction (N3).  The Maslov kernel itself was
  checked and found sound: the (s₂, v₂) chart is always a graph (the phase-space map is
  symplectic), `|det ∂s₁/∂v₂|·dv₂` is the correct pulled-back Huygens measure, and the
  stationary-phase mode carries the Hessian-signature (Maslov) phase
  (`lenses_maslov.py:883-887`).
* Not covered: `_lens_real.py` beyond the per-surface loop and glass propagation; the JAX
  twins (`_lens_jax.py`); re-running the design-119 measurements (accepted as reported).
  Recommended follow-up validation once fixes land: re-run the original §3 table with
  (a) carrier-referenced traced, (b) Maslov `local_quadrature` at `output_subsample=1`
  post-N1/N3 — the two rows the original could not measure.

---

*Companion review performed single-context against lumenairy 5.18.1, 2026-07-02.*

# Lumenairy v4.11.0 — Audit-Fix Verification (round 2)
Date: 2026-05-16
Baseline: v4.9.0 (commit `eb4aa97`) → HEAD (v4.11.0, commit `4f61656`).
Method: 7 parallel verification agents, one per physics domain plus a cross-cutting parity/tests
agent. Each finding from the round-1 report (`AUDIT_REPORT_2026_05_16.md`) was checked against
the actual v4.10/v4.11 code; agents also swept the diff for new bugs the fixes may have
introduced. Diff total: ~1950 LOC across 38 files.

Status legend: ✅ correctly fixed · ⚠️ partial · ❌ not fixed / regressed / dormant · 🆕 new bug
introduced · 📝 documented limitation / acceptable trade-off.

---

## TL;DR

About 70 of the 100-odd round-1 findings are correctly addressed end-to-end. The mirror-Seidel
fix is the headline win — verified end-to-end (`seidel_coefficients` → `total` dict →
`aberration_summary` → `seidel_wfe`). The propagator, glass, optimizer-sentinel, and most
analysis fixes also landed cleanly.

**However, six fixes are dead-on-arrival and one new bug class crept in:**

1. **C-OP-1 `MultiWavelengthMerit` is still a no-op.** The structural rewrite is correct, but
   `optimize/core.py:1927` calls `apply_real_lens` **positionally** while v4.7 made it
   keyword-only after `E_in`. Every call raises `TypeError`, silently caught by `except
   Exception: pass` at lines 1916/1947. The sub-context keeps the parent's single-wavelength
   `E_exit/opd_map/strehl_best`. **Chromatic optimisation is still broken.**
2. **M-LR-1 decentered-stop fix is inoperative.** `_lens_real.py:691-692` calls
   `getattr(surf, 'decenter_x_m', 0.0)` on a **dict**, which always returns the default. Even
   if `.get` were used, the actual key is `'decenter'` (a tuple), not `'decenter_x_m'`.
3. **C-PL-1 circular polarisation handedness is still inconsistent.** `create_circular_polarized`
   was flipped to `(1,-i)/√2` (good), but `apply_waveplate(QWP, 45°)` on `(1,0)` produces
   `(1,+i)/√2` (LHC), and `vector_diffraction.py:147` still hard-codes `(1,+i)/√2` for the
   circular pol option, and the new docstring says "S3 > 0 for right" while the actual
   `S3 = -2·Im(Ex·Ey*)` gives −1 for `(1,-i)/√2`. **The three sites disagree in three different
   ways.**
4. **H-PR-4 `create_point_source` central pixel.** The "fix" is only a `RuntimeWarning`; `r` is
   still clamped to `1e-30`, giving `|E| ≈ 1e30` on the singular pixel.
5. **H-RT-5 JAX intersect NaN guard.** Release notes claim "NaN/non-finite t masked into
   state.alive" — not implemented. Disc<0 still silently sets `t=0` and keeps the ray alive.
6. **M-RT-3 `_refract` / `_reflect` alive flag on degenerate.** Code comments say "flag the ray
   dead", but only `error_code` is written; `rays.alive` is unchanged.

**Five new bugs surfaced by the sweep:**

- 🆕 Richards–Wolf rim mask: `sin_theta` is clipped to `sin θ_max` **before** the mask
  `sin_theta ≤ sin θ_max` is built → every out-of-pupil pixel masquerades as a rim point.
- 🆕 `_sag_derivatives_param` (`jax_trace.py:659-660`) still lacks `sign(R)` — same root cause
  as the C-RT-3 fix that landed in `_sag_derivatives_jax`. Parametric path used by
  `fit_canonical_polynomials_jax` gives wrong transverse direction at concave surfaces.
- 🆕 `_intersect_jax_param` Newton step still uses the old single-where pattern
  (`jax_trace.py:715`) — NaN-poisons the parametric gradient.
- 🆕 `subaperture.py:281-285` `output_grid_xy = np.stack([OX, OY], axis=-1)` produces
  `ndim=3`; the `else: sgx, sgy = output_grid_xy` branch raises `ValueError: too many values
  to unpack` for any common grid.
- 🆕 `optimize/core.py:1940` and `:2034` use `np.argmax(scan.strehl)` rather than
  `np.nanargmax` — returns the index of the first NaN if any per-z Strehl is NaN.

**Three release-note overclaims:**

- ⚠️ "All phase-retrieval functions accept `seed=`" — `gerchberg_saxton_jax` accepts `seed=`
  but **ignores it** (line 438: `_ = seed`; init state is deterministic `E0 = src`). NumPy
  `error_reduction` and `hybrid_input_output` don't accept `seed=` at all.
- ⚠️ "Tilted-ASM band-limit centred on FX + fx0" — correctly applied in `propagation.py`, but
  the cited "library-wide grid consistency" for the `apply_fresnel_curvature` half-pixel fix
  isn't actually library-wide. `gbd.py`, `hf.py`, `subaperture.py` still use the `arange(N) -
  N/2 + 0.5` convention. Cross-method GBD/HF↔ASM/Fresnel interference now has a half-pixel
  offset.
- ⚠️ "Metrics inside JIT" in `through_focus_scan_jax` — the propagation kernel is genuinely
  vmapped, but the metrics loop at lines 907–944 is still host-NumPy `for i in range(n_z)`.

---

## Per-finding verification table

### CRITICAL (round-1)

| Round-1 ID | Topic | Status | Notes |
|---|---|---|---|
| C-VD-1 | Richards–Wolf rewrite | ⚠️ | `1/√cos θ` apod ✓, `-ikf/(2π)·exp(-ikf)` prefactor ✓; `dx_focal` user-supplied now silently overridden with `RuntimeWarning` (option (b) from round-1, accepted). 🆕 rim mask is degenerate (clip-before-mask). Likely missing `1/f²` in angular Jacobian — unit-test against Airy normalisation. |
| C-RT-1 / C-AB-1 | Mirror Seidel | ✅ | Welford form with n2=-n1 at `core.py:3086-3121`; end-to-end propagation through `aberration_summary` and `seidel_wfe` verified. ❌ No regression test added. |
| C-SC-1 | Tilted-ASM band-limit | ✅ | `propagation.py:1714-1715` mask uses `FX_shifted = FX + fx0`. Demod/remod carrier consistent. |
| C-AS-1 | aberration_tensor ℓ=0 closed-form | ✅ | `asymptotic.py:1566-1568` evaluates output polynomial at saddle `(s2x_img, s2y_img)`; on-axis multi-p RuntimeWarning at lines 1510-1523 with workaround guidance. |
| C-GB-1 | GBD reconstruction tilt phase | ✅ | `gbd.py:293-298` adds the per-beamlet linear ramp `exp(ik·(L_b·Δx + M_b·Δy))`. Gracefully no-ops for axial-only bundles. |
| C-AS-2 | HF Chebyshev `-i` Maslov | ✅ | `asymptotic.py:2361`: `out = out * (-1j)`. Matches Fresnel `1/(iλz)` paraxial limit. |
| C-AB-2 | Finite-conjugate Lagrange invariant | ✅ | `core.py:2953-2971` derives `y_m_init = u_obj·d_obj`, `nu_m_init = n_first·u_obj`, giving non-zero H for stop-at-front. |
| C-OP-1 | MultiWavelengthMerit no-op | **❌** | **Structural rewrite at `optimize/core.py:1880-1957` is correct in intent, but line 1927 calls `apply_real_lens(E_in_wl, ctx.prescription, wl, dx_pix)` positionally. `apply_real_lens` is keyword-only after `E_in` since v4.7. Every call raises `TypeError`, caught by `except Exception: pass` at lines 1916/1947 → sub-context keeps the parent's single-wavelength values → chromatic constraint still a no-op in practice.** Compare working call in `MultiFieldMerit.evaluate:2014-2015`. |
| C-OP-2 | design_optimize dtype restore | ⚠️ | Uses `_RestoreDtype` with `__del__` finaliser (`optimize/core.py:2497-2513`). Works on CPython for normal exceptions; unreliable at interpreter shutdown and against `KeyboardInterrupt`, and silently drops finaliser exceptions. Cleaner: `try/finally`. |
| C-AB-3 | Sagittal/tangential fan swap | ✅ | `field.py:872-887` builds bundles directly: chief +y-tilted, sag fan spreads in x, tan in y. Astigmatism sign matches Hopkins/Welford. |
| C-LR-1 | apply_real_lens Seidel sign flip | ✅ | `_lens_real.py:826-831` removes the negation. ⚠️ 50 nm RMS gate at line 843-844 now too coarse (residual SA after fix often <50 nm and silently skipped) — needs lowering or replacing. |
| C-RT-2 | JAX _transfer_jax thickness | 📝⚠️ | Documented limitation per release notes — but the "~1% for NA≤0.1" bound is optimistic. Per-surface error is ~0.5% (transverse, ~thickness·NA²/2), accumulates to ~2.5% over 5 surfaces. OPL error similar order. Documentation should read "0.5% per surface, accumulates". Double-where doesn't fix it; the gradient overflow is into `lstsq` normal equations in `fit_canonical_polynomials_jax` — try Tikhonov-stabilised solve. |
| C-RT-3 | JAX sag derivative sign(R) | ⚠️ | `_sag_derivatives_jax` fixed (`jax_trace.py:156-160`). 🆕 `_sag_derivatives_param` (lines 659-660) **still unfixed**; concave surfaces through `trace_jax_with_params` / `fit_canonical_polynomials_jax` still get wrong transverse refraction. |
| C-RT-4 | Coord-break order vs Zemax | ✅ | `core.py:823-828` order=0 = decenter-then-tilt. Comments and `prescriptions.py` Zemax loader consistent. |
| C-AB-4 | AO Zernike basis rim FD | ✅ | `ao.py:431-440` switches to one-sided FD at rim. |
| C-PR-1 | through_focus bucket units | ✅ | JAX path now returns absolute integrated `|E|²·dA`, matching NumPy `radial_power_bands`. ⚠️ minor `<` vs `≤` boundary off-by-one between paths. |
| C-PR-2 | hash() seed in MC tolerancing | ✅ | Fixed dict `{'decenter_x':1, ..., 'form_error':5}` replaces `hash(knob)`. Deterministic across processes. |
| C-PR-3 | Linearised Strehl non-quadratic | ✅ | Quadratic Maréchal: `S_pred = S_nom − Σ a_k·xi²` with `a_k = (S_nom − S(σ))/σ²` (`through_focus.py:1359-1374`). Mathematically right. |
| C-PL-1 | Circular polarisation handedness | **❌** | `create_circular_polarized('right')` returns `(1,-i)/√2` ✓. But: (a) the new docstring says "S3 > 0 for right" while `S3 = -2·Im(Ex·Ey*) = -1` for `(1,-i)/√2`; (b) `apply_waveplate(QWP@45°)` on `(1,0)` produces `(1,+i)/√2` (LHC, not RHC); (c) `vector_diffraction.py:147` still hard-codes `(1,+i)/√2`. **Three sites disagree in three different ways.** |
| C-OP-3 | JAX system aperture schema | ✅ | `system.py:542-586` now reads `shape`+`params` mirroring NumPy. ⚠️ But `propagate_through_system_jax` calls NumPy `angular_spectrum_propagate` at line 534 with implicit JAX→host conversion → `jax.grad` through a `'propagate'` element silently zeros gradient. |
| C-OP-4 | register_fixed_glass without refractiveindex | ✅ | Sentinel-tuple dispatch in `glass.py:419-435` handled before tuple→refractiveindex.info branch. |
| C-PR-4 | compute_psf 'power' area-Parseval | ✅ | `analysis/core.py:799-812` rescales by area products `sum(|pupil|²)·dx_pupil²` and `sum(psf)·dx_psf²`. Docstring matches code. |

### HIGH (selected)

| Round-1 ID | Topic | Status | Notes |
|---|---|---|---|
| H-SC-1 | apply_fresnel_curvature half-pixel | ⚠️ | The `+0.5` was removed from `apply_fresnel_curvature` ✓. 🆕 But the claimed "library-wide consistency" isn't real: `gbd.py:121-122`, `hf.py:101-102,276-277,333-337`, `subaperture.py:245-246` still use the offset. Cross-method interference has a half-pixel mismatch. |
| H-SC-3 | Cached H mutation | ✅ | `propagation.py:1284`: `H_returned = H.copy()`. |
| H-AS-1 | GBD axial OPL kwarg | ⚠️ | `apply_abcd_to_beamlets` accepts `axial_opl=` (`gbd.py:489-492`). But `propagate_gbd_through_prescription` at line 570 doesn't populate it. **Dormant fix.** CHANGELOG admits with "*should* pass". |
| H-AS-2 | Asymptotic Maslov index | ⚠️ | Branch tracking in `propagate_modal_asymptotic:1759-1824` only. Sibling sites (`aberration_tensor:1422`, `aberration_tensor_lg00_jax:2659`, `_modal_field_lg00_pixel_jax:2708`) still use principal sqrt. JAX-grad through caustic wrong-sign. No doc'd limitation. |
| H-AS-3 | asm_subdomain dx mismatch | ✅ | `mhs.py:376-382` raises on mismatch. |
| H-AS-4 | apply_thin_lens_to_beamlets slopes | ✅ | `gbd.py:218-227` converts `u=L/N`, applies kick, re-normalises. |
| H-HF-1 | HFPI Kirchhoff weighting | ⚠️ | Applied to `init_paths_from_field` and `init_paths_stratified` ✓; **`apply_aperture_diffraction` still missing the `1/(iλ)` + solid-angle factors** (`hfpi.py:197-256`). Cascaded HFPI through multiple apertures still unphysical. Stale docstring at `:343-360`. |
| H-HF-2 | apply_aperture_diffraction obliquity | ✅ | `(cos θ_in + cos θ_out)/2` symmetric Kirchhoff form (`hfpi.py:245-247`). |
| H-HF-3 | coatings absorbing layers / TIR | ⚠️ | Complex-Snell **not implemented** — `RuntimeWarning` emitted instead (`coatings.py:110-119`); accepted as documented limitation. `T = |t|²·Re(η_sub)/Re(η_amb)` is correctly applied ✓. |
| H-HF-4 | apply_jones_matrix shape validation | ✅ | Callable return validated; `(2,2,Ny,Nx)` and `(Ny,Nx,2,2)` both accepted. |
| H-RT-1 | RAY_MISSED_SURFACE stamping | ✅ | `core.py:518-568` per-ray `converged` flag + `disc<0` masking + first-failure-wins error-code stamp. |
| H-RT-2 | JAX trace unsupported surfaces raise | ✅ | `NotImplementedError` at `jax_trace.py:471-499` for mirrors, coord-breaks, biconic, freeform. ❌ No test. |
| H-RT-4 | _intersect_surface stuck rays | ✅ | `stuck = |dF_dt|≤1e-30`; convergence requires `(¬stuck OR |F|<1e-12)`. |
| H-RT-5 | JAX intersect NaN guard | **❌** | **Release notes claim fixed, code unchanged.** `jax_trace.py:177-247` still has no `isfinite(t)` mask, no `disc<0 → alive=False`. Line 214 just zeros `t` for ray-missed cases. Newton-stuck → alive=False also not implemented. |
| H-RT-6 | JAX DOE differentiability | ❌ | Identical to v4.9: `float(period_x)` still blocks `jax.grad`. |
| H-RT-7 | sqrt(max(disc,0)) gradient trap | ⚠️ | TIR sqrt got double-where in `_refract_jax` ✓ but `_intersect_jax:205` still has `sqrt(maximum(disc, 0))` with the disc=0 gradient singularity. And `_intersect_jax_param:715` Newton step still uses old single-where → 🆕 NaN-poisoned parametric gradient. |
| H-LR-1 | apply_aspheric_lens clamp | ✅ | NaN sentinel via `xp.where(valid, sag, NaN)` matches `surface_sag_general`. 📝 Caller must set aperture or NaN leaks. |
| H-LR-2 | get_glass_index missing import | ✅ | Imported in `_lens_thin.py:36`. |
| H-LR-3 | aplanatic 0+0j amplitude clip | ✅ | Out-of-domain returns `1+0j`. |
| H-LR-4 | tilt_aware_rays default | ✅ | One-shot RuntimeWarning when input tilt RMS > 1e-4 rad. |
| H-AB-1 | AO Zernike unit mismatch | ✅ | Gradient divided by `semi_aperture`. |
| H-AB-2 | AO docstring example | ✅ | `slopes_x, slopes_y, *_ = la.shack_hartmann(...)` + `column_stack`. ⚠️ No executable end-to-end test. |
| H-AB-3 | image_plane_wfe aim at EP | ✅ | Uses `fod.ep_z`, `fod.ep_radius`. |
| H-AB-4 | aberration_summary returns NaN | ✅ | `np.full(5, np.nan)` on exception + glass-warning hook. |
| H-AB-5 | MultiFieldMerit z-scan around on-axis BFL | ❌ | `optimize/core.py:2023-2025` still scans `bfl ± bfl/20`. Off-axis paraxial focus offset still ignored. |
| H-AB-6 | MinBackFocalLengthMerit BFL sentinel | ✅ | `ctx_is_valid('bfl')` checked. |
| H-AB-7 | SphericalSeidelMerit sentinel | ✅ | Validity guarded at `optimize/core.py:540-552`. |
| H-AB-8 | ChromaticFocalShiftMerit self-contained | ✅ | Takes `wavelengths=` kwarg, re-derives EFLs per wavelength. |
| H-PR-1 | Phase-retrieval seed= / dtype= | ⚠️ | JAX trio accepts both ✓. **`gerchberg_saxton_jax` accepts `seed=` but ignores it** (line 438: `_ = seed`). NumPy `error_reduction` and `hybrid_input_output` don't accept `seed=` at all. Release-notes claim "All six" is overstated. |
| H-PR-2 | through_focus_scan_jax vmap | ⚠️ | Real `jax.vmap(_asm_one_z)(z_jax)` ✓ but no `jax.jit` (re-traces every call) and metrics loop still host-NumPy. Release notes overstate "metrics inside JIT". |
| H-PR-3 | rms_radius definition parity | ✅ | Both backends use centroid-centred `√(σ_x² + σ_y²)` ≡ `d4σ/4·√2`. |
| H-PR-4 | create_point_source central pixel | ❌ | Still `r = max(r, 1e-30)` → `|E| ≈ 1e30`. Fix is only a `RuntimeWarning`. Docstring still doesn't match physics. |
| H-PR-5 | GS/ER/HIO seed parameter | ⚠️ | Same as H-PR-1. |
| H-GL-1 | Sellmeier validation | ✅ | Resonance and negative-radicand checks raise. |
| H-GL-2 | precision='single' wave-leg dtype | ⚠️ | Fixed in `MatchIdealSystemMerit` and `ToleranceAwareMerit` ✓. New C-OP-1 patch (`:1926`) and `MultiFieldMerit:2013` hard-code `np.complex128` → silently demotes precision. |

### MEDIUM / LOW

| Round-1 ID | Topic | Status | Notes |
|---|---|---|---|
| M-LR-1 | Decentered stop aperture | **❌** | **`_lens_real.py:691-692` calls `getattr(surf, 'decenter_x_m', 0.0)` on a dict** → always returns default. Even with `.get`, key should be `'decenter'` (tuple), not `'decenter_x_m'`/`'decenter_y_m'`. Behaviour identical to pre-4.10.2 despite release-note claim. |
| M-LR-2 | Fresnel amplitude vs intensity averaging | ✅ | `T_eff = 0.5·(|t_s|² + |t_p|²)`. |
| M-LR-4 | _lens_traced exit-vertex sign | ✅ | Signed `t_to_vertex` retained. |
| M-AB-1 | Petzval sign | ✅ | `field.py:957-963` returns `-1/inv_R` (Born & Wolf convention). |
| M-AB-2 | Chief-ray from alive | ✅ | Masks dead rays with `inf` before `argmin`. |
| M-AB-4 | seidel_wfe convention doc | ⚠️ | Warning text updated, but the main docstring at `seidel_analysis.py:158-170` still describes `(1/4)·S₄·σ²·ρ²` while the code uses `H²`. |
| M-PR-3 | dy on top-hat/annular/Bessel | ✅ | All three accept `dy=None` with `dx*dy` area normalisation. |
| M-PR-6 | 1/20 vs 1/10 Nyquist | ❌ | `through_focus.py:564`: `cutoff = 1/(20·dx)` is actually 1/10 of Nyquist; comment still says 1/20. |
| M-RT-3 | _refract/_reflect alive on degenerate | ❌ | `core.py:646-659, 692-702`: comments claim "flag the ray dead" but only `error_code` written; `rays.alive` unchanged. |
| RS docstring | RS kernel sign | ⚠️ | Code correctly uses `(1/r - ik)` (Goodman 3-43) ✓. **Docstring at `propagation.py:2663` still shows old `(ik - 1/r)` form.** |
| RS z≤0 guard | RS back-propagation | ❌ | Still no `z≤0` validation. ASM/Fresnel/Fraunhofer/SAS/MFT all guard; RS silently accepts negative z and returns forward-prop kernel with mis-signed prefactor. |

### NEW BUGS introduced in v4.10–v4.11

| # | File:Line | Severity | Issue |
|---|---|---|---|
| N1 | `optimize/core.py:1927` | **CRITICAL** | MultiWavelengthMerit calls keyword-only `apply_real_lens` positionally; TypeError swallowed by bare except → C-OP-1 still a no-op. |
| N2 | `vector_diffraction.py:118,137` | HIGH | `sin_theta` clipped to `sin θ_max` before mask built from `sin_theta ≤ sin θ_max` → every out-of-pupil pixel masquerades as rim pixel. |
| N3 | `jax_trace.py:659-660` | HIGH | `_sag_derivatives_param` still missing `sign(R)` (C-RT-3 fix only applied to non-param twin). Concave surfaces wrong through `fit_canonical_polynomials_jax`. |
| N4 | `jax_trace.py:715` | HIGH | `_intersect_jax_param` Newton step uses old single-where pattern → NaN-poisoned parametric gradient. |
| N5 | `subaperture.py:281-285` | HIGH | `output_grid_xy = np.stack([OX, OY], axis=-1)` is `ndim=3`; `else: sgx, sgy = output_grid_xy` raises for Ny > 2. Intent was `sgx, sgy = OX, OY`. |
| N6 | `optimize/core.py:1940, 2034` | MEDIUM | `np.argmax(scan.strehl)` should be `np.nanargmax` — returns index of first NaN if any per-z Strehl is NaN. |
| N7 | `optimize/core.py:1926, 2013` | MEDIUM | Hard-coded `np.complex128` in MultiWavelengthMerit and MultiFieldMerit reintroduces the precision='single' regression that H-GL-2 was meant to fix. |
| N8 | `optimize/core.py:1924` | MEDIUM | Aperture fallback uses `or` instead of `is None` check → silently overrides legitimate `aperture_diameter=0.0` sentinel with grid-arbitrary default. |
| N9 | `optimize/core.py:1945-1946` | MEDIUM | Bare `except Exception: pass` in MultiWavelengthMerit silently falls back to pre-4.10 broken behaviour on any propagation failure. |
| N10 | `detector.py:391` | LOW | Shack-Hartmann reference centroids computed with `np.fft.fft2`; measurements use `angular_spectrum_propagate`. Calibration mismatch — small but not zero. |
| N11 | Half-pixel grid convention | LOW | H-SC-1's "library-wide consistency" claim is wrong: `gbd.py`, `hf.py`, `subaperture.py` still use the `+0.5` offset. Cross-method interference is half-pixel mis-aligned. |

### Release-note overclaims

| Claim | Reality |
|---|---|
| "All phase-retrieval functions accept `seed=` and `dtype=` for reproducibility / backend parity." | `gerchberg_saxton_jax` accepts `seed=` but ignores it. NumPy `error_reduction` and `hybrid_input_output` don't accept `seed=`. |
| "through_focus_scan_jax now uses jax.vmap … metrics inside JIT" | vmap is real; metrics loop is host-NumPy. |
| "Apply Fresnel curvature half-pixel … library-wide consistency" | Three other modules still use the old convention. |
| "Tilted-ASM band-limit centred on FX + fx0" | ✓ (correctly applied). |
| "_transfer_jax accurate to ~1% for NA ≤ 0.1" | Actually ~0.5% per surface, ~2.5% over 5 surfaces. |

---

## Test coverage gap

The fix wave added **zero new test files** and modified **one existing test** (a tolerance
loosening in `test_propagation.py`). The release-notes claim "All 34 validation files (314
tests) pass" is consistent with the fixes but provides essentially no regression coverage for
the specific fixes. Notable gaps:

- **No mirror-Seidel regression test.** The top-billed fix has zero coverage.
  `validation/raytrace/test_seidel_field.py` exercises only refractive singlets.
- **No `MultiWavelengthMerit` chromatic-correctness test.** A simple assertion that wrapping a
  `StrehlMerit` in `MultiWavelengthMerit` over `[1.0 μm, 1.55 μm]` produces a chromatic-spread-
  dependent value would have caught the positional-call bug (N1) immediately.
- **No `NotImplementedError` test for `trace_jax`** on unsupported surface types.
- **No tilted-ASM band-limit regression test** asserting non-trivial-tilt output is non-zero.
- **No phase-retrieval backend-parity test** for the `seed=` / `dtype=` claim.

---

## Priority fix list for v4.11.1

1. **Patch `optimize/core.py:1927`**: change `apply_real_lens(E_in_wl, ctx.prescription, wl,
   dx_pix)` to `apply_real_lens(E_in_wl, prescription=ctx.prescription, wavelength=wl,
   dx=dx_pix)`. **One-line change, single biggest correctness restoration.** Restoring this
   fix also exposes N7-N9 inside the same merit; address them in the same patch.
2. **Patch `_lens_real.py:691-692`**: use `surf.get('decenter') or (0.0, 0.0)` instead of
   `getattr(...)` on a dict. The decentered-stop fix has been dormant since 4.10.2.
3. **Resolve C-PL-1 fully**. Pick a single convention (the optics-RHC = `(1,-i)/√2` under
   `exp(-iωt)`) and propagate it through `create_circular_polarized`, `apply_waveplate`,
   `stokes_parameters` docstring, and `vector_diffraction.py:147`. Add a single test that
   asserts `S3 > 0` for `create_circular_polarized('right')`, and that QWP@45° on `(1,0)`
   produces the same Jones vector.
4. **Fix N2** (Richards–Wolf rim mask). Build the mask before clipping `sin_theta`.
5. **Patch N5** (`subaperture.py:281-285`). Single-line fix: `sgx, sgy = OX, OY`.
6. **Fix N3 / N4** (parametric JAX twins). Apply C-RT-3 (`sign(R)`) to `_sag_derivatives_param`
   and the double-where Newton-step pattern to `_intersect_jax_param`.
7. **H-PR-4**: floor `|E|` rather than `r` in `create_point_source`.
8. **H-RT-5**: actually mask `~isfinite(t)` and `disc < 0` into `state.alive` in
   `_intersect_jax`.
9. **M-RT-3**: add `rays.alive = rays.alive & ~_degenerate` to both `_refract` and `_reflect`.
10. **N1+N2 hardening**: remove the bare `except Exception: pass` in MultiWavelengthMerit
    (lines 1916, 1945-1947). Replace with explicit exception types and a `warnings.warn`.
11. **H-AS-1**: have `propagate_gbd_through_prescription` actually populate `axial_opl=` when
    calling `apply_abcd_to_beamlets`.
12. **Lower the Seidel correction RMS gate** in `_lens_real.py:843-844` — after the C-LR-1
    sign fix, real residuals are often <50 nm and are silently skipped.
13. **Documentation**: fix the RS docstring at `propagation.py:2663` to match the new code;
    update `seidel_wfe` docstring to `H²` from `σ²`; update the `_transfer_jax` accuracy bound
    to "~0.5% per surface, accumulates".
14. **Add regression tests** for at least the five highest-impact fixes (mirror Seidel,
    MultiWavelengthMerit chromatic-spread, tilted-ASM band-limit non-zero output, `trace_jax`
    NotImplementedError, RS phase consistency with ASM at z>0).

---

## Cross-cutting themes (revisited)

### JAX backend status

Round 1 called the JAX backend a "second-class citizen" with five distinct regressions. After
this round:
- Sign(R) on sag derivatives: ✅ in main path, ❌ in parametric path (N3).
- `_transfer_jax`: 📝 documented limitation, accepted.
- `trace_jax` raises on unsupported surfaces: ✅.
- NaN guard in intersect: ❌ release notes overclaim.
- DOE differentiability: ❌ unchanged.
- `through_focus_scan_jax` vmap: ⚠️ partial.
- JAX system aperture: ✅ but gradients through `'propagate'` elements still silently zero
  (round-2 finding from cross-cutting agent).

JAX is no longer second-class for forward-only use but remains second-class for differentiable
optimisation. A "JAX correctness audit" milestone — including a NumPy↔JAX parity test suite —
would close the gap.

### Silent-failure pattern

Round 1 flagged this as a cross-cutting theme. Round 2 finds:
- Sentinel checks added in `MinBackFocalLengthMerit`, `MaxFNumberMerit`,
  `SphericalSeidelMerit`. ✅
- `aberration_summary` returns NaN instead of zero. ✅
- `RAY_MISSED_SURFACE` now stamped. ✅
- **But** the new `MultiWavelengthMerit` bare-except is a fresh instance of the same pattern
  (N9), and `_refract`/`_reflect` still fail to update `alive` on degenerate rays (M-RT-3),
  and `H-RT-5` is unfixed despite release notes. The pattern is recurring.

### Reproducibility

Round 1: GS/ER/HIO had no `seed=`, `monte_carlo_tolerancing_linearized` used `hash()`. Round
2: `hash()` fix landed ✓, `seed=` only partly delivered (NumPy ER/HIO still unseeded; JAX GS
accepts but ignores).

---

**Net summary**: about 70% of round-1 findings cleanly fixed; ~12% partial; ~12% landed-but-
inoperative or unchanged-despite-claims; ~5% net-new bugs from the fix wave. The mirror-Seidel
fix is a clean win. The MultiWavelengthMerit and decentered-stop fixes both ship as dead code
due to call-signature mistakes. The circular-polarisation handedness fix has gone from "one
sibling wrong" to "three siblings all disagreeing in different ways". A short v4.11.1 patch
addressing items 1-2 in the priority list above plus a handful of regression tests would close
most of the residuals.

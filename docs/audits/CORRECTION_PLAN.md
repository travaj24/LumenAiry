# LumenAiry v4.9.0 → v4.10.0 Correction Plan

Unifies findings from three audit sources:
1. **Round 1** (this assistant, 5 agents) — scalar propagators, ray+lens, sources, aberrations, vector/specialty
2. **Round 2** (this assistant, 5 more agents) — asymptotic/GBD/HFPI, propagation deep-dive, glass/coatings, system+optimize+Seidel, detector/plotting/coronagraph
3. **External** `AUDIT_REPORT_2026_05_16.md` (8-agent audit)

Findings overlap heavily. This plan eliminates duplicates and orders fixes by impact + file.

## Severity scheme

- **CRITICAL**: wrong physics in a default code path, or a hard crash mid-run.
- **HIGH**: wrong physics in a non-default code path, or silent failure that produces "good-looking" wrong values.
- **MEDIUM**: inconsistencies, latent bugs, edge cases.
- **LOW / DOC**: docstring drift, cosmetic, dead code.

## Fix order (by file, ordered by aggregate impact)

| # | File | Bugs addressed | Tier |
|---|------|----------------|------|
| 1 | `raytrace/core.py` | Mirror Seidel zeros (C-RT-1), exit-pupil radius inversion (Z-XP-1), coord-break order (C-RT-4), Lagrange invariant finite-conjugate (C-AB-2), _intersect stuck-ray (H-RT-4), RAY_MISSED never set (H-RT-1), flat-surface comment | CRITICAL |
| 2 | `raytrace/jax_trace.py` | _transfer wrong formula (C-RT-2), sag-deriv sign (C-RT-3), mirror/coord-break skip (H-RT-2), TIR gradient (H-RT-6/7), NaN guard (H-RT-5), DOE float-cast | CRITICAL |
| 3 | `raytrace/seidel_analysis.py` | S4 Petzval fallback should NaN-not-warn, dimension consistency | MEDIUM |
| 4 | `system.py` | JAX propagator drops apertures (C-OP-3), unused dy_new | CRITICAL |
| 5 | `propagators/propagation.py` | RS sign flip (Z-SC-1), apply_fresnel_curvature +0.5 offset (H-SC-1), tilted-ASM bandlimit (C-SC-1), return-by-ref H cache (H-SC-3), complex64 promotion (Z-SC-2), arange float64 on JAX, R=0 silent copy | CRITICAL |
| 6 | `propagators/vector_diffraction.py` | Richards-Wolf reimplementation: Bluestein-backed dx_focal, 1/sqrt(cos θ) Jacobian, -ikf/(2π)·exp(-ikf) prefactor (C-VD-1) | CRITICAL |
| 7 | `propagators/asymptotic.py` | ℓ=0 closed-form degeneracy (C-AS-1), missing -i Maslov prefactor (C-AS-2), sqrt(detM) Maslov tracking (H-AS-2), decompose_lg/hg 1D→2D, Newton stall check, Sigma dead code | CRITICAL |
| 8 | `propagators/gbd.py` | Missing per-beamlet tilt phase ramp (C-GB-1), axial OPL phase (H-AS-1), direction-cosine/slope conflation (H-AS-4) | CRITICAL |
| 9 | `propagators/mhs.py, hf.py, subaperture.py, hfpi.py` | dtype coercion (Z-HF-1), out_surface.dx ignored (H-AS-3), wrong kwargs (Z-SU-1), solid-angle norm (H-HF-1), obliquity reference (H-HF-2) | HIGH |
| 10 | `optimize/core.py` | evaluate() crash (Z-OP-1), MultiWavelengthMerit no-op (C-OP-1), complex_dtype try/finally (C-OP-2), MultiFieldMerit aperture+per-field BFL (H-AB-5/C-AB-2), FD step scaling, MinBFL sentinel (H-AB-6), JaxMerit abs default, MaxFNumber sentinel, MinThickness penalises air, source aperture mask | CRITICAL |
| 11 | `elements/_lens_real.py` | Seidel correction sign flip (C-LR-1), Fresnel amplitude→intensity (M-LR), geometric power factor, cos_safe warn, anamorphic gradient, decentered stop axis-centred h_sq | CRITICAL |
| 12 | `elements/_lens_thin.py` | apply_axicon get_glass_index import (H-LR-2), apply_aspheric clamp NaN (H-LR-1), apply_thin_lens aplanatic clip (H-LR-3) | HIGH |
| 13 | `elements/_lens_traced.py` | tilt-aware-rays default warning (H-LR-4), sign convention assertion | MEDIUM |
| 14 | `elements/coatings.py` | Snell complex-n (H-HF-3), TIR silent cap, T=1-R lossy bug, dead num/den code (H-HF-5), sign-convention docstring | HIGH |
| 15 | `elements/polarization.py` | create_circular_polarized handedness swap (C-PL-1), apply_jones_matrix shape guard (H-HF-4) | CRITICAL |
| 16 | `elements/lenses.py` | surface_sag_general conic-edge NaN/warn | MEDIUM |
| 17 | `elements/doe.py` | create_fresnel_zone_plate negative-f sign | LOW |
| 18 | `analysis/field.py` | field_aberration_sweep sag/tan fan swap (C-AB-3), petzval_radius sign | CRITICAL |
| 19 | `analysis/detector.py` | SH wavefront wavelength/(2π) factor, pixel area-integration via reduceat, SH 2D reconstruction, reference-centroid subtraction, OOB NaN | CRITICAL |
| 20 | `analysis/coherence.py` | mutual_coherence conjugate swap, Köhler high-NA doc | CRITICAL |
| 21 | `analysis/ghost.py` | TIS extra cos(θ), focus_z_estimate doc | HIGH |
| 22 | `analysis/image_plane_wfe.py` | Strehl pupil-weighted RMS, best-RMS exit-pupil radius, chief-ray pick, aim-at-EP (H-AB-3) | HIGH |
| 23 | `analysis/interferometry.py` | fringe formula visibility scaling | HIGH |
| 24 | `analysis/through_focus.py` | JAX vs NumPy parity (C-PR-1), monte_carlo hash() seed (C-PR-2), Strehl quadratic prediction (C-PR-3) | CRITICAL |
| 25 | `analysis/core.py` | compute_psf normalize='power' Parseval (C-PR-4), OSA Zernike table doc, dead `- cx_k * 0.0`, check_sampling NA | CRITICAL |
| 26 | `analysis/aberration.py` | aberration_summary bare-except (H-AB-4), caustic thicknesses length | HIGH |
| 27 | `analysis/ao.py` | zernike_modal_basis rim ρ>1 (C-AB-4), semi_aperture units (H-AB-1), broken docstring example (H-AB-2), pinv regularization | CRITICAL |
| 28 | `analysis/phase_retrieval.py` | JAX error_reduction support-after-IFFT (Z-PR-1), float32 cast (H-PR-1), seed param (H-PR-5), GS rescaling | HIGH |
| 29 | `analysis/coronagraph.py` | rms vs σ semantics | MEDIUM |
| 30 | `analysis/plotting.py` | dx vs dy axis labels, auto_crop padding per-axis, log floor guard | LOW |
| 31 | `sources/core.py` | point_source clamp warn (H-PR-4), use_gpu cupy loader (Z-GS-1), top_hat/annular/bessel dy support (M-PR), fiber NA doc | HIGH |
| 32 | `glass.py` | wavelength-range validation (H-GL-1), resonance-pole guard, kappa-missing warn | HIGH |
| 33 | `user_library.py` | load_phase_mask wavelength default (M-OP), eval() safety | MEDIUM |
| 34 | Validation suite run + regressions | — | — |
| 35 | Version bump 4.9.0 → 4.10.0, CHANGELOG, commit | — | — |

## Conventions

- **Sign convention**: `exp(-iωt)` time. Forward propagation kernel is `exp(+ikz·sqrt(1 − ...))`.
- **GFM/MathJax**: Wiki files use `\\_` for literal underscore in math text-mode blocks.
- **No `--no-verify`** on git commits.
- **Validation gate**: each major file group runs `python validation/run_all.py <test_module>` before move-on. Final pass runs the full suite.

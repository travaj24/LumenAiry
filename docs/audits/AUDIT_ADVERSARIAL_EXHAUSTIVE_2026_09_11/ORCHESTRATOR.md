# ORCHESTRATOR — independently verified findings on `apply_real_lens` (2026-09-11)

## F-O1 [P1, documented-as-recommended opt-in] `seidel_correction=True` injects ~90 µm·ρ² of spurious defocus
- `lumenairy/elements/_lens_real.py` Seidel block (`_apply_real_lens_impl`, ~L6480–6560): the 41-ray fan is traced with
  `raytrace.trace(...)` and `final_fan = res_fan.image_rays; opl_ray = final_fan.opd` — no exit-vertex transfer.
  `trace()` leaves rays at the last surface's SAG (measured `final.z` ∈ [−75.3 µm, 0] for R3 = −128.23 mm), so
  `delta_ray` is short by `n_exit·|sag_last(h)|` ≈ h²/(2|R3|), a pure ρ² term, which the even-polynomial fit absorbs
  and applies as a phase screen.  REAL_LENS_CHANGES §3 documents this exact bug class and fixed it in
  `apply_real_lens_traced` (L9583) and the validation — but NOT here.
- Measured (docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/orch/seidel_fan_replicate.py, AC254-100-like N-BK7/N-SF11 doublet, 10 mm pupil, 632.8 nm):
  correction RMS as shipped = 34 221 nm (gate 5 nm); ρ² coefficient −8.96e-5 m.  With the signed transfer
  `t = −z/N`: RMS 1 118 nm, ρ² +3.0e-6 m.
- Consequence (docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/orch/seidel_focus_shift.py, N=1024, dx=20 µm): on-axis peak moves from z = 114.0 mm
  (paraxial BFL 114.85 mm) to z = 62.0 mm.  The validation metric (piston+tilt+DEFOCUS removed) cannot see it,
  which is why the documented "4.5× better on AC254-100-C" survived.
- Fix: apply the signed exit-vertex transfer to the fan before reading `opd` (reuse the helper `_lens_traced`
  uses at L9583 — better: add `TraceResult.to_exit_vertex()` in `raytrace` and call it from every consumer).
  Secondary (P2): even with the transfer a +3 µm ρ² residual remains because the analytic reference
  `Σ(n2−n1)·sag` omits the in-glass propagation curvature that the ASM legs DO carry; the correction therefore
  over-corrects defocus by construction.  Either drop the ρ² term from the fit (correct only ρ⁴ and above) or
  reference against a 1-D wave pass.

## F-O2 [P1, opt-in] `surface_frame=True` drops the tilt ramp: a tilted flat face deviates the beam by 0
- `_lens_real.py` ~L5985–6010: `Xs = cy*dx_local + sx*sy*dy_local; Ys = cx*dy_local`, sag evaluated at (Xs, Ys),
  `z_s` discarded, and the linear ramp suppressed ("already encoded in the rotated (Xs, Ys)").  For a plane the
  rotated footprint changes nothing, so the field-frame z of the rotated surface, ≈ −ty·x + tx·y (+ cos·cos·sag),
  is lost.  A tilted refracting face is a thin prism; deviation must be ≈ (n−1)·θ.
- Measured (docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/orch/surface_frame_tilt_check.py, N-BK7 plate, θ = 5 mrad, λ = 1 µm): field-frame branch
  deviates 2.538 mrad (= (n−1)θ), `surface_frame=True` deviates 0.000 mrad on both axes.  For a curved tilted
  surface the same term is missing, so the exit beam direction is wrong by (n2−n1)·θ for any tilted element.
- Also: the two branches disagree on axis convention (field-frame `tilt[0]` ramps along x, i.e. a rotation about
  y; surface-frame `tilt[0]` is a rotation about x) — documented as "differing positive-tilt definition" but a trap.
- Fix: in the surface-frame branch compute the field-frame axial position of the rotated surface,
  z_f = R_zx·x_s + R_zy·y_s + R_zz·sag_s(x_s, y_s) (R = Rx(tx)·Ry(ty): R_z· = (−cx·sy, sx, cx·cy)), and use
  OPD = (n2−n1)·z_f; unify the tilt-axis convention across both branches (and `raytrace`).

## F-O3 [census] Exit-vertex bug class: `trace()` leaves rays at the last-surface sag; which consumers correct for it
`grep image_rays` across `lumenairy/` → 27 consumer files; only `_lens_traced.py` (L6679, L9583, L11107) and `propagators/gbd.py`
apply a `t = −z/N` transfer.  Consumers that read `.opd`/`.x` of `image_rays` WITHOUT it (verified by reading the code):
- `elements/_lens_real.py` Seidel fan (F-O1, measured).
- `elements/lenses_maslov.py:1710-1735` — canonical map `(s2, v2, OPD)` taken at the sag; with `output_plane_distance≠0` the
  free-space leg is `t = d/N` instead of `(d − z)/N`.  Numerical check: docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/orch/maslov_exit_vertex_check.py.
- `elements/_lens_traced_uniform.py:223-228, 587-592` — `t = output_plane_distance/Nz`, ignores `ex.z` (fold-ring geometry).
- `propagators/asymptotic_canonical_fit.py:408-422, 915+` — `(final.x, final.opd)` at the sag.
- `raytrace/ray_fan.py:503-518 opd_fan_data` — OPD at the sag minus the chief's OPD at z=0 (a spurious n·sag(h) "defocus"
  whenever the surface list does not end on a flat image surface); `refocus()` (L575+) documents the sag-to-vertex fix
  but `opd_fan_data` does not use it.
- `analysis/image_plane_wfe.py:508-511` — depends on whether `surfaces` ends in an image plane (ANALYSIS auditor to confirm).
Recommended structural fix: give `TraceResult` an explicit `at_exit_vertex()` (signed transfer, alive-masked) and make it
the ONLY way consumers read exit OPL; then delete the three hand-rolled copies.

## F-O4 [P1, opt-in propagator] `apply_real_lens_maslov` reports the field at the last-surface SAG as the exit-vertex field
- `lenses_maslov.py:1709-1735`: `exit_rays = tr.image_rays` (at the sag); with `output_plane_distance=0` no transfer; with
  `≠0` the leg is `t = d/N` (should be `(d − z)/N`).  The canonical map `(s2, v2) → OPD` is therefore a map onto the curved
  exit surface, evaluated as if on the vertex plane.
- Measured (docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/orch/maslov_exit_vertex_check2.py; biconvex R = ±100 mm N-BK7, t = 3 mm, 5 mm pupil, λ = 1 µm,
  N = 512, dx = 14 µm, Gaussian w = 1.6 mm; unwrapped radial cut, fit piston+tilt+ρ²+ρ⁴):
  maslov − traced: ρ² = **−31.303 µm**, ρ⁴ = +0.104 µm, residual 4.2 nm RMS; predicted n_exit·sag_exit(ρ=1) = −31.255 µm.
  thin − traced on the same cut: ρ² = 0.000 µm, residual 0.1 nm (so the traced reference is sound).
  Flat-exit plano-convex control (R = 50 mm / ∞): maslov − traced residual 7 nm RMS, no ρ² term.
- Impact: the added ρ² OPD equals ~10 dioptres here — the same order as the lens's own power — so any downstream focus /
  through-focus / PSF from the Maslov field is wrong for every curved-exit element.  REAL_LENS_CHANGES §2b validated Maslov
  by RMS INTENSITY at the output plane (phase-blind) on an F/52 lens, which is why it passed.
- Fix: transfer `exit_rays` to the vertex plane with the signed `t = −z/N` (OPL += n_exit·t; x,y += (L,M)·t) before building
  `(s2, v2, OPD)`; make the `output_plane_distance` leg `(d − z)/N`.  Same fix for `_lens_traced_uniform.py:225-228/589-592`
  and `asymptotic_canonical_fit.py:408-422` (both read `image_rays` at the sag).

## Cross-verification of TR-MAIN-2's P1s (docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/orch/verify_trmain2.py, independent fixture: f≈25 mm biconvex, N = 256)
1. `apply_real_lens_traced(E_real_float64)` → `AttributeError: type object 'numpy.complex128' has no attribute 'type'`
   (`_lens_traced.py:11697` builds a scalar TYPE, not a dtype); `apply_real_lens` on the same array returns complex128.  CONFIRMED.
2. `newton_fit='spline'` with one vignetting `semi_diameter`: polynomial P_out/P_in = 0.9754 (21 601 non-zero px);
   spline P_out/P_in = 0.0000, 0 non-zero px.  CONFIRMED (NaN samples poison the FITPACK spline; comment at :9711 is false).
3. `caustic='multibranch', output_plane_distance = BFL (25.478 mm)`: P_out/P_in = 0, 0 non-zero px, 0 warnings.  CONFIRMED.

## Orchestrator cross-verification log (independent fixtures unless noted)
| Auditor claim | Result | Script |
|---|---|---|
| PROP-CORE: `fresnel_propagate` 33 % error at z = 0.25·N dx²/λ, no guard | reproduced exactly (3.34e-1 / 3.43e-1 / 3.92e-2) | PROP-CORE/p11_fresnel_alias.py re-run |
| PROP-CORE: `backend.scipy.jv(v, x)` calls `scipy.special.jv(x, v)` | confirmed: jv(0,2.0)=0.000 vs 0.2239; jv(2,1.5)=0.4913 vs 0.2321; no in-library importer | orch (inline) |
| THIN: bundled N-BAF52 / N-LAK33A / N-LAK33B Sellmeier rows wrong | confirmed vs refractiveindex.info SCHOTT-optical: 1.637147 vs 1.608631; 1.754279 vs 1.753930; 1.755294 vs 1.755000 (N-BK7, N-SF11 exact) | orch/verify_rt_glass.py |
| THIN: `generate_turbulence_screen` variance 2× the lattice PSD sum | confirmed: ratio 1.976 ± 0.16 (N=256, 40 seeds) | orch/verify_turb.py |
| RAYTRACE: conic intersection kills rays with h > |R| | confirmed: R=10.84 mm, k=−0.6 → alive [T,T,T,F,F] at h = 8, 10, 10.8, 10.9, 11.4 mm, error_code 3 | orch/verify_rt_glass.py |
| RAYTRACE: `seidel_coefficients` ignores `conic` | confirmed: S1 = 9.765625e-05 for both k = 0 and k = −1 (parabola must be 0) | orch/verify_rt_glass.py |
| TR-INFRA: `fast_analytic_phase=True` raises AttributeError (`raytrace._surface_sag_xy`) | confirmed | orch/verify_trinfra.py |
| UI: `to_prescription()` drops `is_mirror`, mis-indexes semi-diameters | confirmed via the auditor's Qt stub: exported EFL 158.9 mm vs UI-internal 178.8 mm; mirror exported as air→air with sd 6 mm | UI/t2_mirror.py re-run |
| TR-INFRA: traced model silently cancels `form_error` | confirmed: 250 nm PV astig → analytic Δφ 0.397 rad, traced 4.85e-4 rad (~800× suppression) | orch/verify_trinfra.py |
| RL-MODELS: `displaced` remaps discard the input phase (default routing for asymmetric elements) | confirmed: flat vs 35-wave-defocused input → identical output (4.7e-16) on both 1-D and 2-D remaps; screens differ by 2π p-v | RL-MODELS/t5_phase_discard.py re-run |
| ANALYSIS: `diffraction_limited_peak` paraxial reference inflates Strehl | confirmed: perfect exact-sphere pupil at f/2.5 (W040 = 0.33 waves) → actual peak / reference = 1.468 (must be 1.000) | orch/verify_analysis.py |
| ANALYSIS: `wave_opd_2d` integer-wave slips on an aberrated pupil | confirmed: 0.5 waves rms coma, 0.27 rad/sample → max error 1.000 waves, 53.2 % of pupil wrong by > 0.4 waves | orch/verify_analysis.py |
| PROP-CORE-SUB1: point-sampled RS kernel creates energy in the near field | confirmed: P_out/P_in = 21.44 (N=64, dx=2 µm, z=50 µm), 5.31 (N=128, dx=1 µm); ASM on the same grid 1.0000; RS exact (3e-13) at z=300 µm | orch/verify_rs_alias.py |
| PROP-CORE-SUB1 vs PROP-CORE: Richards–Wolf `fft2` orientation | NOT a defect: the two auditors used different pupil-indexing conventions; the code is self-consistent with the aperture-coordinate convention (PROP-CORE's measured +u·λ·f focus shift for a pupil ramp) — reclassified as a P3 documentation hazard | reasoning from both measurements |
| IO-OPTIMIZE: CODE V `DIM M` read as metres | confirmed by source: `{'M': 1.0, 'MM': 1e-3, 'IN': 0.0254}`, tokens `('M','MM','IN')`; CODE V's own tokens are M (mm) / C / I | prescriptions_code_v.py:223–258 |
| POLAR: `coating_reflectance(polarization='te')` takes the p branch | confirmed by source: `if pol == 's': … else: eta = n/cos` | coatings.py:112, :216–219 |
| ASYMPTOTIC: default `include_linear=False` drops a3·u3 + a4·u4 inside the integrand | confirmed by source | asymptotic_canonical_fit.py:207–225 |
| PROP-HF: `richards_wolf_focus` returns E_z with the sign opposite to E_x/E_y | confirmed with an independently derived oracle (strength vector from the rigid rotation of the pupil polarisation, reduced to the radial Bessel form E_z/E_x = −2i·I₀₁/(I₀₀+I₀₂), scipy `quad`): code/oracle = −1.0006 / −0.9954 / −0.9994 / −0.9670 at x_f = 0.60 / 1.21 / 1.81 / 3.01 µm (x-pol, NA 0.5, f = 4 mm, Np = 256); E_z odd in x to 4.5e-16 and 3e-19 on the y-axis, E_x even to 2.4e-16 — magnitudes agree, sign flipped. Accepted as P0 (every call). Resolves the K10 disagreement: the geometric mapping is aperture-indexed, the polarisation vectors ray-direction-indexed | orch/verify_rw_ez.py |
| RL-CORE: `slant_correction` obliquity coefficient has the wrong sign | confirmed analytically: expanding the code's `n2·cosθt − n1·cosθi` and the module's own eq.(3) `n2·cos(θi−θt) − n1` to O(θ²) gives (n2−1) + θ²(n2−1)/(2n2) vs (n2−1) − θ²(n2−1)²/(2n2); the total-error ratio n2/(n2−1) = 2.941 for N-BK7 matches the auditor's measured 2.9414–2.9417 at four radii | derivation (this log) |
| RL-CORE: `fresnel=True` applies |t|², not the power transmittance | confirmed from source (`T_eff = 0.5(|t_s|²+|t_p|²)` with no n2cosθt/n1cosθi factor, both copies) and by arithmetic: (2/(1+n))² = 0.6323 (what the code returns for a single AIR→BK7 face) vs 4n/(1+n)² = 0.9581 (power transmittance) at n = 1.515089 | `_lens_real.py:6366`, `:5842` |

## Disclosure — side effect caused by one sub-auditor
The PROP-CORE-SUB1 sub-auditor reported that, to stop a runaway oracle script, it ran a blanket
`taskkill //F //FI "IMAGENAME eq python.exe"` which terminated FIVE python.exe processes on this host (not only its own).
Other auditors' experiments (and possibly an unrelated user process) may have been killed at that moment; no repository
file or git state was touched. It also observed an unrelated python.exe holding ~29.7 GB RSS during the audit, which is
why shell commands degraded to minutes at times.

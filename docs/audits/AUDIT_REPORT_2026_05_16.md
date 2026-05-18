# Lumenairy v4.9.0 — Multi-Agent Physics & Bug Audit
Date: 2026-05-16
Method: 8 parallel Claude Code audit agents, one per physics domain. ~73 kLOC of Python read.

This report consolidates findings from all 8 agents. Severity scale: CRITICAL = produces wrong
numerical answers in a default code path; HIGH = wrong answers in a non-default path, or silent
failure; MEDIUM = inconsistencies, latent issues, gated by kwargs; LOW = docstring / cosmetic.
Each finding has the agent's full justification in the transcript — only file:line and a one-line
summary are kept here.

---

## v4.9.0 audit-claim verification

The README claims a list of audit fixes shipped in 4.9. Verification summary:

| Claim | Status |
|-------|--------|
| #2.1 Seidel `Δ(u/n)` Welford form | ✅ Present and correct in refracting branch |
| #2.1 Flat-refracting-surface branch not zeroed | ✅ Present and correct |
| #2.5 `aberration_tensor` σ-integration for ℓ≠0 | ⚠️ Fix correct for ℓ≠0, but the **kept** ℓ=0 closed-form branch is degenerate (see C-AS-1) |
| #4.6/#4.7 `seidel_wfe` Petzval H² | ✅ Present in refracting branch |
| #2.2 GBD axial phase `exp(1j·k·t)` (signed t) | ✅ Verified correct |
| #2.3 Coronagraph `pix_per_lam_over_D` | ✅ Verified correct |
| #3.3 Fresnel/Fraunhofer/SAS `z<=0` guards | ✅ Present — but **RS is NOT guarded** despite docstring suggesting back-prop works (C-SC-1) |
| #3.5 TIR mask placement (`slant_correction=True, fresnel=False`) | ✅ Verified |
| #4.5 `cosmic_ray_rate_per_m2_per_s` scaling | ✅ Verified |
| 4.8.0 `create_point_source` sign-on-z0 | ✅ Verified |

**Additional finding from the v4.9 claim list:** the v4.9.0 release notes say `seidel_coefficients`
was fixed, but the fix only covers **refracting** surfaces. Mirror surfaces still produce
zero S1–S5 (C-RT-1 / C-AB-1 below) — this is the same class of bug as the flat-surface bug that
v4.9 fixed, just in the reflection branch.

---

## CRITICAL FINDINGS (sorted by impact)

### C-VD-1. `vector_diffraction.richards_wolf_focus` is a plain FFT of the pupil, not the Richards–Wolf integral
`lumenairy/propagators/vector_diffraction.py:78,84-87,144-149,159-175`

The function defines `dx_focal` independently (defaulting to `λ/(4·NA)`), but the underlying
`fft2(pupil)` produces a focal pitch of `λ·f/(N·dx_pupil)` — a hard FFT identity. The user-supplied
`dx_focal` only changes the axis labels. Also missing the `1/cos θ` Jacobian when converting from
(θ,φ) integration to (kx,ky) FFT space — so even the in-focus amplitude weighting is biased,
worst at the rim where high-NA matters most. Combined: the current `richards_wolf_focus` is not
the Richards–Wolf integral; high-NA focal-field results are not trustworthy. Use Bluestein
(already present in `_bluestein.py`) to make `dx_focal` a genuine free parameter and add the
1/√(cos θ) prefactor in the pupil weighting.

### C-RT-1 / C-AB-1. Mirror branch in `seidel_coefficients` never assigns S1–S5
`lumenairy/raytrace/core.py:3032-3045`

```python
elif surf.is_mirror and np.isfinite(R):
    c = 1.0 / R
    u_m = nu_val_m / n1
    ...
    i_m = c * y_val_m + u_m
    phi = 2.0 * n1 * c
    nu_m_after = nu_val_m - y_val_m * phi   # paraxial ok
    nu_val_m = nu_m_after
    # never writes S1[i] .. S5[i]
```

S1[i]…S5[i] stay at zero for every mirror. Every catadioptric / reflective system (Cassegrain,
Schwarzschild, FSO fold) gets wrong Seidel sums. Apply the same Welford form with `n2 = -n1`.

### C-SC-1. Tilted ASM band-limit is mis-centred on baseband (default `bandlimit=True`)
`lumenairy/propagators/propagation.py:1687-1693`

After demodulating to baseband, the band-limit window must be centred on `-f0` (the carrier),
not on 0. The current `np.abs(FX) < fx_max` mask kills the energy-bearing baseband modes for any
non-trivial tilt and lets through the aliasing-prone bands. Fix: `np.abs(FX + fx0) < fx_max`.

### C-AS-1. `aberration_tensor` closed-form ℓ=0 path returns identical L for every (p,0) mode
`lumenairy/propagators/asymptotic.py:1481-1510`

The closed-form path projects onto the (0,0) coefficient of the conjugated output LG polynomial.
Since L_p^0(0)=1 for all p, `L[(0,0), src]`, `L[(1,0), src]`, `L[(2,0), src]` … all evaluate to
the same scalar — defocus and all higher (p,0) spherical modes are indistinguishable. The
σ-integration path is correct; the bug is only triggered by lists of pure ℓ=0 modes (the natural
choice for a Strehl/defocus/spherical merit set). Remove the closed-form branch or σ-integrate
always for ℓ=0.

### C-GB-1. GBD reconstruction missing per-beamlet tilt-phase ramp
`lumenairy/propagators/gbd.py:270-274`

Each beamlet gets a Gaussian envelope `exp(-ik·Q·ρ²/2)` but no `exp(ik·(L·Δx + M·Δy))` linear
phase ramp. On-chief-ray phase is correct (so a focal spot still focuses where it should), but
off-chief-ray interference patterns and PSF wings degrade for any non-paraxial bundle. Documented
"position-only" disclaimer at lines 95-109 hints at this but doesn't make it explicit.

### C-AS-2. HF Chebyshev quadrature missing 2-D Maslov `-i = i^(-1)` prefactor
`lumenairy/propagators/asymptotic.py:2233-2240`

For d=2, the Van Vleck–Morette asymptotic Green's function carries `(2π)^(-d/2)·i^(-d/2)`. The
`-i` is missing. Cross-check: paraxial Φ = (z+ρ²/(2z))/λ gives `√|det H| = 1/(λz)`, while the
Fresnel kernel is `1/(iλz) = -i/(λz)`. Off by a global 90°. Multiply integral by `-1j`.

### C-AB-2. Lagrange invariant identically zero for finite-conjugate, stop-at-front systems
`lumenairy/raytrace/core.py:2912-2930,2949-2950`

For `A_pre=1, B_pre=0` and finite `object_distance`: `y_m_init=0, y_c_init=0` → H = 0. Petzval
term `H²·S4` then vanishes from `seidel_wfe`. Chief ray for a finite off-axis source should
launch from object height `h_obj`, not from y=0 with an angle. Add `field_height` kwarg or
convert `field_angle` to equivalent object-space height.

### C-OP-1. `MultiWavelengthMerit` does not re-evaluate wave leg at each wavelength
`lumenairy/optimize/core.py:1833-1853`

```python
sub_ctx = EvaluationContext(
    prescription=ctx.prescription, wavelength=wl, ...,
    E_exit=ctx.E_exit, opd_map=ctx.opd_map, strehl_best=ctx.strehl_best, ...)
total = total + self.sub_merit.evaluate(sub_ctx)
```

Only `wavelength`, `efl`, `bfl` change; `E_exit`, `opd_map`, `strehl_best` are copied unchanged
from the parent context. Wrapping `StrehlMerit` or `RMSWavefrontMerit` in `MultiWavelengthMerit`
evaluates the same single-wavelength field N times and sums it — chromatic Strehl optimisation
is a no-op.

### C-OP-2. `design_optimize` doesn't restore `complex_dtype` on exception
`lumenairy/optimize/core.py:2341-2343,2641-2642`

`precision='single'` flips `DEFAULT_COMPLEX_DTYPE` to `complex64`, but the restore at the end of
the function runs only on the success path. A scipy raise / KeyboardInterrupt leaves the global
dtype permanently flipped to complex64, halving precision for every subsequent unrelated call in
the process. Wrap in try/finally.

### C-AB-3. Sagittal/tangential ray fans swapped in `field_aberration_sweep`
`lumenairy/analysis/field.py:863-868`

`make_fan(axis='x', θ)` is called "sagittal" but it tilts the chief along x AND spreads the fan
along x — both in the meridional plane → it's actually tangential. Both labels are tangential
fans of two unrelated chief rays. The reported `astigmatism = sag_shift - tan_shift` is between
two different configurations. Build the bundles directly: chief tilted along +y, sagittal fan
along x, tangential fan along y.

### C-LR-1. `apply_real_lens` Seidel correction has a sign flip on the analytic OPL reference
`lumenairy/elements/_lens_real.py:765-793` (specifically line 792)

```python
opl_wave_rel = -(opl_analytic - opl_analytic[i_ax])   # <-- the negation
correction = delta_ray - opl_wave_rel
```

The thin-element phase `exp(-i·k0·(n2-n1)·sag)` adds OPL `+(n2-n1)·sag`, with the same sign as
the geometric ray OPL. The negation gives `correction ≈ 2·opl_analytic + small_residual`. Gated
by `seidel_correction=True` and a 50 nm RMS threshold, but actively wrong when triggered. Remove
the negation.

### C-RT-2. JAX `_transfer_jax` uses axial thickness as the parametric step
`lumenairy/raytrace/jax_trace.py:363-372`

```python
new_x = state.x + state.L * thickness        # should use t = (thickness - z)/N
new_z = state.z + state.N * thickness - thickness
```

Three consequences: wrong transverse positions for any non-axial ray; wrong OPL (`n·thickness`
≠ `n·|t|`); and `new_z` doesn't return to the next-surface vertex frame. The NumPy `_transfer`
(`core.py:693-703`) does this correctly. Mirror it.

### C-RT-3. JAX sag-derivative drops `sign(R)` for `R<0` surfaces
`lumenairy/raytrace/jax_trace.py:131-159` (`_sag_derivatives_jax`)

```python
zx = x / sd   # always positive for x>0, regardless of R sign
```

NumPy `_surface_sag_derivative` at `core.py:408-430` correctly carries the sign via
`dz_dh = h/(R·denom)`. JAX form `x/√(R²-(1+k)h²)` is always positive. Refracted rays at any
concave conic/aspheric get wrong transverse direction in JAX.

### C-RT-4. Coord-break order convention reversed vs. Zemax PARM 6
`lumenairy/raytrace/core.py:787-798`

```python
if order == 1:
    _decenter(); _tilts()
else:
    _tilts(); _decenter()   # order == 0 (default) -- this is Zemax PARM 6 = 1
```

Comments at `core.py:213-214` and `prescriptions.py:696` say "0 = decenter then tilt", but the
code does tilt-then-decenter for `order=0`. Trace with `dy=10mm, tilt_x=5°` on a ray at origin:
lumenairy `order=0` gives (0,-10,0); Zemax PARM 6=0 gives (0,-9.962,0.872). Zemax loader at
`prescriptions.py:718` stores PARM 6 verbatim, so every imported folded design with the Zemax
default has wrong frame transforms.

### C-AB-4. AO `zernike_modal_basis` evaluates Zernikes at ρ > 1 for rim lenslets
`lumenairy/analysis/ao.py:405-418`

The central-difference step `+eps` pushes `ρ_x_plus > 1` for lenslets near the pupil rim;
`zernike_polynomial` then forces Z=0 (correct behaviour, but the finite-difference becomes
`(0 - Z_in)/(2·eps)` — a giant spurious spike). Contaminates the influence matrix, biases the
reconstructor. Use the analytic Zernike-gradient formula or one-sided FD at the rim.

### C-PR-1. `through_focus_scan_jax` returns `power_in_bucket` in different units than NumPy path
`lumenairy/analysis/through_focus.py:898 (JAX) vs core.py:280 (NumPy)`

NumPy returns absolute integrated intensity `sum(|E|²)·dx·dy`. JAX returns a **fraction** of total
intensity. Same field name, two different physical quantities depending on `backend=`.

### C-PR-2. `monte_carlo_tolerancing_linearized` uses Python `hash()` for seeding
`lumenairy/analysis/through_focus.py:1296`

```python
random_seed=spec_idx * 100 + hash(knob) % 1000
```

Python 3's `hash()` is randomised per process via `PYTHONHASHSEED`. Form-error realisations
change between runs even with the same `seed=` argument; FD sensitivities and predicted Strehl
distribution non-reproducible. Use a deterministic knob→int mapping.

### C-PR-3. `monte_carlo_tolerancing_linearized` Strehl prediction is non-quadratic-around-nominal
`lumenairy/analysis/through_focus.py:1287-1296, 1320-1325`

Strehl is quadratic near nominal (Maréchal: `S ≈ 1 − (2πσ)²`), so a linear FD probe + linear
trial superposition gives a mean-zero distribution around `S_nom`. Should fit a quadratic
`S(α) − 1 ≈ −a·α²` per knob (two probes per knob) and sum the negative-definite quadratic terms.

### C-PL-1. `create_circular_polarized` handedness inconsistent with `apply_waveplate` v4.7 fix
`lumenairy/elements/polarization.py:577-582` vs `:454-459`

Under `exp(-iωt)`, working `(1,0)` → QWP@45° with `e = exp(-iπ/2)` gives `(1,-i)/√2` ≡ optics
RHCP. But `create_circular_polarized('right')` returns `(1,+i)/√2` ≡ optics LHCP. One of the two
labels is wrong. `apply_waveplate` and `stokes_parameters` (`S3 = -2·Im(Ex·Ey*)` ≡ "right
positive") agree under the `exp(-iωt)` convention, so `create_circular_polarized` is the broken
one — swap branches.

### C-OP-3. JAX system propagator silently drops apertures
`lumenairy/system.py:542-548` (JAX) vs `:405-409` (NumPy)

NumPy path reads `elem['shape'], elem['params']`. JAX path reads `elem['radius']`. A working
NumPy element list ported to `propagate_through_system_jax` has every aperture silently skipped.

### C-OP-4. `register_fixed_glass` is broken when `refractiveindex` not installed
`lumenairy/user_library.py:197` and `glass.py:399-420`

User-registered fixed glasses store a sentinel tuple `('__user__','__fixed__','__fixed__')` in
`GLASS_REGISTRY`, but the dispatch falls into the non-Sellmeier branch which raises
`ImportError` if `refractiveindex` is absent. Saved fixed-index materials become unusable on
machines without the optional dep. Register as a callable instead.

### C-PR-4. `compute_psf normalize='power'` doesn't preserve area-integrated intensity
`lumenairy/analysis/core.py:782-789`

Rescales so `sum(psf) == sum(|pupil|²)` (pixel-sum). Physical Parseval requires
`sum(psf)·dx_psf² == sum(|pupil|²)·dx_pupil²`. Strehl ratios cancel the constant so the
docstring's first use case works, but anyone using the PSF for absolute photon flux (also
documented use case) is off by `(dx_pupil/dx_psf)²`. Correct the docstring or rescale.

---

## HIGH FINDINGS (selected)

Listed here with one-line summaries; see agent transcripts for code snippets and fixes.

**Scalar propagators**
- H-SC-1 `apply_fresnel_curvature` uses `arange(N) - N/2 + 0.5`, while every other propagator uses no offset → half-pixel-shifted curvature relative to the field grid (`propagation.py:1362-1363`).
- H-SC-2 SAS asymmetric padding `as1 = (N+1)//2` breaks centring for odd N (`propagation.py:3036-3037,3135`).
- H-SC-3 `return_transfer_function=True` returns the **cached** H by reference — a caller `H *= mask` silently corrupts every subsequent ASM call at the same key (`propagation.py:1657,1695-1696`). Copy on lookup.

**Asymptotic / GBD / MHS**
- H-AS-1 `apply_abcd_to_beamlets` drops axial OPL phase `exp(i·k·L_axial)` (`gbd.py:451-455`).
- H-AS-2 Asymptotic propagators use `np.abs(det J)` / `np.sqrt(complex det)` everywhere — Maslov index dropped, wrong phase past any caustic (`asymptotic.py:1227,1410,1732,1979`).
- H-AS-3 `asm_subdomain` and prescription subdomains silently ignore `out_surface.dx`; field labelled at output dx but actually at input dx (`mhs.py:371-377,481-492`).
- H-AS-4 `apply_thin_lens_to_beamlets` conflates direction cosines with paraxial slopes (`gbd.py:209-219`).

**HFPI / vector diffraction / polarization**
- H-HF-1 HFPI missing solid-angle normalisation `2π(1-cos θ_max)/N_paths` — absolute amplitudes wrong (`hfpi.py:111-125`).
- H-HF-2 `apply_aperture_diffraction` obliquity uses the +z axis instead of the incoming-ray direction → secondary HF sources anisotropic on tilted surfaces (`hfpi.py:226`).
- H-HF-3 `coatings.py:103,119` Snell's law uses `n.real` for both indices — absorbing layers and TIR branch wrong; `T = max(0, 1−R)` (line 147) is wrong for non-vacuum substrate.
- H-HF-4 `apply_jones_matrix` silent shape mismatch for callable matrices — `J.shape = (N,N,2,2)` broadcasts and produces wrong answer with no error (`polarization.py:362-368`).
- H-HF-5 `coatings.py:129-130` has unused-but-wrong dead-code `num`/`den` lines next to the correct formula — maintenance trap.

**Ray tracing**
- H-RT-1 `RAY_MISSED_SURFACE` is defined and `trace_summary` reports it, but it is **never set** anywhere (`core.py:59` + Newton loop in `_intersect_surface`). Vignetted rays continue alive.
- H-RT-2 JAX trace silently ignores mirrors, coord-breaks, biconic, freeform — passes them through as flat refractive surfaces (`jax_trace.py:413-481`).
- H-RT-3 `_paraxial_trace` uses `u_new = u_old - y·phi/n2`, dropping the `n1/n2` factor on `u_old` (`core.py:1996-2050`). Latent — function is currently unused but exported.
- H-RT-4 `_intersect_surface` early-exit ignores stuck rays: `dF_dt → 0` forces `dt=0`, which satisfies convergence trivially (`core.py:540-544`).
- H-RT-5 JAX `_intersect_jax` no NaN guard — NaN from grazing rays propagates to all subsequent surfaces (`jax_trace.py:166-232`).
- H-RT-6/7 JAX gradient issues: `float(period_x)` in DOE kick blocks `jax.grad` w.r.t. grating period; `jnp.sqrt(jnp.maximum(disc, 0))` blows up at disc=0 — use the double-where pattern.

**Lens / DOE**
- H-LR-1 `apply_aspheric_lens` clamps `denom_arg` to `1e-12` instead of zeroing invalid pixels → near-singular sag outside the surface domain (`_lens_thin.py:400-402`).
- H-LR-2 `apply_axicon` calls `get_glass_index` but the import is missing in `_lens_thin.py` — NameError on any axicon with a string glass name (`_lens_thin.py:629-632`).
- H-LR-3 `apply_thin_lens('aplanatic')` uses `xp.where(valid, ..., 0.0+0.0j)` as a phase mask → silently clips amplitude in the annulus `r>=f` (`_lens_thin.py:159`).
- H-LR-4 `apply_real_lens_traced` `tilt_aware_rays=False` (the default) uses plane-wave reference OPD; documented but incorrect for tilted inputs ≳ λ/aperture (`_lens_traced.py:1043, 1592-1593`). Add a runtime tilt-RMS warning.

**Aberration / WFE / AO / Field**
- H-AB-1 AO `zernike_modal_basis` returns gradients in normalised-pupil coords; SH-WFS slopes are in physical coords → reconstructed modal coefficients off by factor `semi_aperture` (`ao.py:419-422`).
- H-AB-2 AO docstring example doesn't run — `shack_hartmann` returns a 5-tuple but `slope_to_modal` expects an `(N,2)` array (`ao.py:38-53` + `:461`).
- H-AB-3 `eval_image_plane_wfe` aims rays at the first surface plane (z=0), not at the entrance pupil at `fod.ep_z`. Wrong for any system with stop in the middle (`image_plane_wfe.py:430-435`).
- H-AB-4 `aberration_summary` silently catches `Exception` from `seidel_coefficients` and returns zeros — optimizer merits act on "diffraction-limited" zero Seidels (`aberration.py:184-186`).
- H-AB-5 `MultiFieldMerit` scans through-focus ±BFL/20 around the **on-axis** BFL, missing off-axis best focus on lenses with field curvature (`optimize/core.py:1908-1922`).
- H-AB-6 `MinBackFocalLengthMerit` accepts the invalid-BFL sentinel `1e9` as "satisfies clearance" (`optimize/core.py:2003-2005`).
- H-AB-7 `SphericalSeidelMerit` and similar read `ctx.seidel[i]` with no validity check — returns 0 ("perfect zero") on `seidel_coefficients` failure (`optimize/core.py:537-538`).
- H-AB-8 `ChromaticFocalShiftMerit` requires `ctx.efls_per_wavelength`, populated only as a side effect of `MultiWavelengthMerit.evaluate()`. Term ordering can silently disable the constraint (`optimize/core.py:1795-1800`).

**Analysis core / sources**
- H-PR-1 `gerchberg_saxton_jax` / `error_reduction_jax` / `hybrid_input_output_jax` hardcast inputs to `float32` — NumPy variants disagree at 1e-6 level (`phase_retrieval.py:420,459,503`).
- H-PR-2 `through_focus_scan_jax` advertises "vmap over z" but is a Python for-loop; metrics run on host NumPy (`through_focus.py:862-867`).
- H-PR-3 `through_focus_scan_jax` `rms_radius` is computed about the brightest pixel; NumPy path uses D4sigma about the centroid. Two metrics, same name.
- H-PR-4 `create_point_source` clamps `r → 1e-30` → central pixel `|E| ≈ 1e30` (docstring claims the opposite — "central pixel is clamped to a finite floor") (`sources/core.py:594-602`).
- H-PR-5 GS / ER / HIO have no `seed=` parameter; JAX twins hardcode `seed=0` — backends disagree, not reproducible (`phase_retrieval.py:139,249,356`).

**Optimizer / glass**
- H-GL-1 No Sellmeier wavelength-range validation; out-of-range λ silently returns nonsense (e.g. N-BK7 at 5 μm) (`glass.py:187-201`).
- H-GL-2 `precision='single'` knob is half-effective — wave-leg merits hard-code `E = np.ones(..., dtype=np.complex128)` (`optimize/core.py:2420,1898,2112`).
- H-GL-3 `least_squares` with bounds silently switches to `'trf'` from `'lm'` (`optimize/core.py:2583-2586`).
- H-GL-4 LM residual `sqrt(max(merit, 0))` is non-differentiable at zero; FD Jacobian near minimum produces inf/nan columns (`optimize/core.py:2578-2580`).

---

## MEDIUM / LOW

Roughly 50 additional findings across the eight reports. Highlights:

- M-SC: `bandlimit` `<` vs `<=` inconsistency between ASM and ASM-MFT (`propagation.py:618-619` vs `:1891-1892`).
- M-PR: OSA Zernike index table in docstring puts "Primary spherical" at j=10; actual OSA convention puts it at j=12 (j=10 is oblique quadrafoil). Code is correct, table is wrong (`analysis/core.py:1253-1264`).
- M-AB: `petzval_radius` returns `+1/inv_R`, missing the standard `-` sign (Born & Wolf §4.4) (`field.py:923-936`).
- M-AB: `eval_image_plane_wfe` chief-ray pick `argmin(px²+py²)` can pick a dead vignetted ray and NaN the whole WFE map (`image_plane_wfe.py:448-452`).
- M-AB: `seidel_wfe` convention is half-Hopkins / half-Welford — S4 takes explicit H², others have it baked in. Docstring hedges; users will misuse (`raytrace/seidel_analysis.py:159-163, 282-286`).
- M-LR: `apply_real_lens` aperture stop at `stop_index` uses axis-centred `h_sq_axis` even when the surface has a non-zero `decenter` (`_lens_real.py:664-665`).
- M-LR: Fresnel transmission averages **amplitude** coefficients `0.5*(t_s+t_p)` instead of intensity — wrong for unpolarised light at AOI > ~10° (`_lens_real.py:644`).
- M-PR: `create_top_hat_beam`, `create_annular_beam`, `create_bessel_beam` use `dx` for the y-axis spacing and don't accept `dy` (`sources/core.py:691,733,865`).
- M-PR: `M2` Wigner cross term has a dead `- cx_k * 0.0` placeholder (`analysis/core.py:576-577`).
- M-OP: `MinThicknessMerit` and `MaxThicknessMerit` penalise air gaps too — the docstring says "any glass thickness" but iterates `prescription['thicknesses']` (`optimize/core.py:1962-1987`).
- M-OP: `load_phase_mask` defaults `wavelength = 1.0` m if neither caller nor JSON file supplies one → `k = 2π rad/m`, useless phase (`user_library.py:471-473`).
- M-OP: `load_phase_mask` uses `eval()` with `numpy` exposed → code-execution risk if anyone can write to `~/.lumenairy/library/phase_masks/*.json` (`user_library.py:449-458`).
- L-VD: `polarization='circular'` in `richards_wolf_focus` is hard-coded to `(1,+i)/√2` with no `left` option (`vector_diffraction.py:114`).
- L-AS: `solve_envelope_stationary` stall test uses an obscure Python chained comparison (`asymptotic.py:1054`).
- L-AB: `Sigma = 0.5 * M_inv` is dead code (`asymptotic.py:1409`).

(See the per-agent transcripts for the full set.)

---

## Cross-cutting themes

1. **JAX path is a second-class citizen.** Five different physics regressions are JAX-only:
   sign-loss in sag derivative (C-RT-3), wrong transfer formula (C-RT-2), silently-skipped mirrors
   and coord-breaks (H-RT-2), dropped apertures (C-OP-3), `float32` cast in phase retrieval
   (H-PR-1), Python-loop "vmap" (H-PR-2), and `rms_radius` definitional disagreement (H-PR-3).
   The cleanest fix is a shared test suite that runs every NumPy regression also through the
   JAX backend and asserts numerical equality (within float32 rtol where applicable).

2. **Silent-failure paths.** `RAY_MISSED_SURFACE` defined but never set; `aberration_summary`
   bare-except returning zero Seidels; `_intersect_surface` stuck-ray convergence; HFPI dead
   path-weights with NaN phase. Several merits treat invalid sentinels (BFL=1e9, seidel=0) as
   "perfect" answers. Pattern: prefer NaN-propagation over zero-as-fallback, and require
   `np.isfinite()` checks in merit functions before they take credit for "good" values.

3. **Mirror + finite-conjugate Seidel coverage gap.** v4.9 fixed refracting-flat and Δ(u/n),
   but mirrors (C-RT-1) and finite-conjugate stop-at-front (C-AB-2) are still broken. These
   are not new bugs — they pre-date v4.9 — but they were not caught by the previous audit.

4. **Sign-convention drift between sibling functions.** `create_circular_polarized` vs
   `apply_waveplate` (C-PL-1); `petzval_radius` vs textbook; sagittal/tangential fan labels
   (C-AB-3); coord-break PARM 6 vs Zemax (C-RT-4). These are all places where local-author
   consistency works but cross-function consistency fails under the library's documented
   `exp(-iωt)` time convention.

5. **Docstring drift.** OSA Zernike table; RS supports back-prop "to come" comment;
   apply_real_lens_traced `preserve_input_phase` tilt limitation; `vector_diffraction.py`
   `dx_focal` "free parameter". Several docstrings now contradict the v4.9 behaviour and would
   benefit from a sweep.

6. **Reproducibility.** `gerchberg_saxton` / `error_reduction` / `hybrid_input_output` have no
   `seed=` parameter (H-PR-5); JAX twins hardcode `seed=0`; `monte_carlo_tolerancing_linearized`
   uses `hash(knob)` (C-PR-2). For optimisation regression tests these surfaces should accept
   a seed and use `np.random.default_rng(seed)`.

---

## Recommended fix order

1. **C-RT-2, C-RT-3, C-OP-3, H-RT-2** — JAX correctness floor. Without these, no JAX-backed
   ray-trace or system call gives the same answer as NumPy.
2. **C-VD-1** — `richards_wolf_focus` is the headline mis-implementation. Until fixed, any
   "high-NA focal-plane" result published from Lumenairy is unsupported.
3. **C-RT-1 / C-AB-1, C-AB-2** — Mirror Seidels and finite-conjugate Lagrange invariant.
   Same bug class as the v4.9 flat-surface fix; mechanically applicable.
4. **C-OP-1, C-OP-2** — `MultiWavelengthMerit` no-op and `complex_dtype` leak. Both are silent.
5. **C-SC-1** — Tilted-ASM band-limit; default `bandlimit=True` makes this active.
6. **C-AS-1, C-AS-2, C-GB-1, H-AS-2** — Asymptotic / GBD physics: the closed-form ℓ=0
   degeneracy, the missing `-i` prefactor, the tilt ramp omission, and the dropped Maslov index.
7. **C-AB-3** — Sagittal/tangential fan swap in `field_aberration_sweep`.
8. **C-PR-2, C-PR-3** — Tolerancing reproducibility and quadratic Strehl prediction.
9. **C-PL-1** — Circular-polarization handedness convention.
10. **C-AB-4, H-AB-1, H-AB-2** — AO module: rim Zernike sampling, unit mismatch, broken docstring example.
11. **C-LR-1, C-RT-4** — Less-default-path bugs, but they corrupt specific imported workflows.
12. **C-OP-4, H-LR-2, H-PR-4** — Crash/NameError/inf-amplitude edge cases.
13. The remaining HIGH items, then MEDIUM/LOW.

---

## Methodology notes

- Eight agents, ~73 kLOC of Python, ~3500 lines of report text in aggregate.
- Each agent was given the relevant v4.9 release-note claims to verify, told to cite file:line,
  and asked to skip nitpicks unless they reflect a real correctness concern.
- Substantial cross-checking happened between agents (e.g. the Seidel mirror bug was found
  independently by the raytrace agent and the aberration agent; the OSA Zernike-table bug
  was caught by both the aberration and analysis-core agents).
- Several findings each agent initially flagged were verified-and-withdrawn after deeper
  reading; only the substantive items survived to this report.

# Lumenairy v4.11.1 — Round-3 Fresh-Eyes Audit
Date: 2026-05-16
Codebase: ~74,517 LOC across ~80 Python files
Method: 11 parallel fresh-eyes audit agents, each instructed to find substantive new issues
not already catalogued in rounds 1 or 2.

---

## TL;DR

Round 3 surfaced **~30 new substantive findings** that previous rounds and the v4.10/v4.11/v4.11.1
fix waves missed, spanning physics correctness, dead-on-arrival fixes, the previously-unaudited
IO/Zemax module, and test quality. The headline issues are:

### ⚠️ MOST IMPORTANT META-FINDING

**Round-1's identification of C-LR-1 (the `apply_real_lens` Seidel-correction sign flip) was
itself wrong.** The pre-v4.10 code had `opl_wave_rel = -(opl_analytic - opl_analytic[i_ax])`,
which was the CORRECT sign convention. The v4.10 "fix" dropped the negation based on the
round-1 audit's incorrect physics reasoning, producing a correction that effectively triples
the lens's OPD at the edge. v4.11.1's threshold drop from 50 nm → 5 nm didn't help — the
correction magnitude is millimeters, far above either threshold. Existing tests pass because
they only check that the field is finite, not that the correction matches ground truth.

This is the first instance in the audit series where the audit itself **introduced** a bug
via incorrect physics reasoning rather than catching one. It's a cautionary case for the
"every fix needs a regression test against ground truth" recommendation.

### Top critical findings (would produce silently wrong results in default code paths):

1. **`S-LAH64` and `S-LAH79` bundled Sellmeier coefficients are wrong.** `glass.py:173-176`.
   Computed `n_d` differs from OHARA catalog by **0.058** (LAH64) and **0.117** (LAH79). Likely
   misattributed coefficients from a different glass. Any lens design using these specific OHARA
   glasses produces a real lens that doesn't match the simulation.

2. **EVENASPH aspheric round-trip with Zemax is broken (off-by-one PARM indexing).**
   `prescriptions.py:469-473`. Filter drops `PARM 1` entirely (the dominant α₄ coefficient) and
   formula `power = 2*parm_num` mis-labels every higher coefficient by one slot. Every
   Zemax-authored EVENASPH file ever loaded by Lumenairy has silently lost α₄.

3. **`MultiWavelengthMerit` chromatic optimisation is still effectively a no-op** despite
   v4.11.1's "fix-of-the-fix". The structural rewrite is correct, but the merit's z-scan still
   has unreliable behaviour due to a downstream `np.argmax` (now `nanargmax`) handling and
   the bare except is narrowed but still swallows real failures.

4. **`propagate_hfpi_through_prescription` kills every path for any finite-conjugate system.**
   `hfpi.py:690-692`. Paths are initialised at z=0 going forward, then immediately
   back-propagated to `z=-object_distance`; the `t >= 0` mask kills them all. Function only
   "works" for `object_distance=0`.

5. **`init_paths_stratified` enumerates only 2 strata regardless of how many requested.**
   `hfpi.py:493-505`. The `np.repeat` pattern is wrong; only `(0,0,0,0)` and `(1,1,1,1)`
   of the 16 combinations are ever sampled. The "stratified" sampler is two-point Monte Carlo
   with perfect correlation between source-pixel and direction indices.

6. **Chained-mirror Seidel parity not tracked.** `raytrace/core.py:3036, 3093-3128`. The
   v4.10 mirror-Seidel fix only handles a single mirror; the next surface re-queries
   `glass_before='air'` and gets `n1=+1` instead of `-1`. Cassegrain / Schwarzschild / any
   2-mirror catadioptric still gets wrong Seidel sums beyond the first mirror.

7. **`system_abcd` and `seidel_coefficients` use DIFFERENT mirror conventions.**
   `core.py:2158-2161` vs `:3093`. For a concave mirror R=-100mm, `system_abcd` returns
   EFL=-0.05m (negative) while the conventional answer is +0.05m. `find_paraxial_focus`
   inherits the sign error.

8. **GBD axial-OPL "dormant fix activated" in v4.11.1 is dead-on-arrival.** `gbd.py:576-590`.
   The code calls `.get()` on a `Surface` dataclass (which has no `.get()` method);
   `AttributeError` is silently swallowed by the bare `except Exception`; `axial_opl=None`.
   GBD reconstructed fields still lack the system's absolute axial phase reference.

9. **Richards–Wolf prefactor is missing a `1/f²` factor AND has the wrong sign on
   `exp(±ikf)`.** `vector_diffraction.py:221`. A 1m vs 1cm focal length differs by 10⁴ in
   Airy-peak intensity. The global phase `exp(-ikf)` is opposite to every other forward-prop
   in the library (which uses `exp(+ikf)` under `exp(-iωt)`) — coherent superposition with a
   reference arm has a `exp(-2ikf)` mismatch.

10. **`seidel_wfe` is missing the field-curvature DC term `(1/4)·S₃·ρ²`.**
    `seidel_analysis.py:290-294`. Standard Hopkins/Welford has both `(1/2)·S₃·ρ²·cos²θ`
    (astigmatism) AND `(1/4)·(S₃ + S₄·H²)·ρ²` (FC DC). Both docstring math AND code agree
    with each other — both are wrong.

11. **`normalize_prescription` mirror filter is a no-op.** `prescriptions.py:2579-2584`.
    Checks `e.get('mirror')` but library uses `element_type='mirror'`. Defeats the v4.8 mirror
    guard for the most common entry path (Zemax load → normalize_prescription →
    apply_real_lens).

12. **Quadoa aspheric serializer iterates dict keys instead of values.**
    `prescriptions.py:2104-2107`. Writes `[4.0, 6.0, 8.0]` (the powers) to JSON, not the
    coefficients.

13. **Phase retrieval `seed=` / `dtype=` parity STILL broken** despite release notes.
    `gerchberg_saxton_jax` accepts `seed=` but ignores it; NumPy `error_reduction` and
    `hybrid_input_output` don't accept `seed=` at all.

14. **`propagate_hf_chebyshev_quadrature` is missing the `-1j` Maslov prefactor for
    `propagate_huygens_fresnel_with_opl_callable`.** `hf.py:165-179`. The v4.10 C-AS-2 fix
    was applied to `propagate_hf_chebyshev_quadrature` only; the sibling OPL-callable
    function in `hf.py` has the same missing prefactor.

15. **`propagate_huygens_fresnel_through_prescription(method='asymptotic')` silently
    discards `E_in`'s amplitude and phase**, replacing them with a unit-amplitude
    fundamental Gaussian. The default `method='asymptotic'` path makes any structured
    input silently become a fundamental Gaussian.

### Process / test-quality findings:

- **3 of 9 v4.11.1 pinning tests pass for the wrong reason**: MultiWavelengthMerit (only
  checks warning absence, not chromatic semantics); Subaperture (import-only smoke test —
  the actual bug fires on `.call()`, not import); Tilted-ASM bandlimit (tilt magnitude too
  small to trigger pre-fix behaviour).
- **Validation harness suppresses all warnings globally** (`_harness.py:33` does
  `warnings.simplefilter('ignore')`) → any `RuntimeWarning` emitted inside library code
  during validation is silently swallowed.
- **Several validation tests have inner `try/except: return True, 'skipped'`** that papers
  over real failures.

---

## Cross-cutting recurring themes

### Theme 1: Mirror handling is the consistent weak point

The v4.10 / v4.11.x mirror-Seidel fix is the headline correctness improvement of the audit
series, but **every audit round has surfaced new mirror-related bugs**:

- Round 1: mirror branch never wrote S1..S5 (fixed in v4.10)
- Round 2: verified the single-mirror fix lands correctly
- Round 3:
  - **CRIT-2** Chained-mirror Seidel parity not tracked
  - **CRIT-3** `system_abcd` / `seidel_coefficients` mirror-power sign disagreement
  - **MED-2** World-frame builder advances cursor by `s.thickness · R[:,2]` after a mirror
    without flipping the z-axis (unenforced convention)
  - **C-IO-4** `normalize_prescription` mirror filter is a no-op
  - **C-IO-5** Mirror DISZ lost on Zemax round-trip; coord-break DISZ double-counted
  - **HIGH-3** `_paraxial_trace` mirror power bug (dead code but lingers)

The unit test for the mirror fix uses a single concave mirror — chained-mirror behaviour has
never been tested. Recommend a Cassegrain regression test with hand-computed Seidel sums.

### Theme 2: Dead-on-arrival fixes continue to ship

Three new instances of "the fix was written but doesn't actually run":

- **N-AS-1** v4.11.1's GBD axial-OPL "dormant fix activated" calls `.get()` on a `Surface`
  dataclass; `AttributeError` swallowed by bare except.
- **C-OP-1 (revisited)** `MultiWavelengthMerit` ran on positional `apply_real_lens` until
  v4.11.1 — fix landed, but separate findings in round 3 show the per-wavelength wave leg
  still has issues (silent bare-except fallback, `complex128` hard-coding inherited).
- **M-LR-1** Decentered stop fix used `getattr(surf, 'decenter_x_m', ...)` on a dict (round-2
  finding); v4.11.1 actually fixed this one.

Pattern: defensive `try/except Exception: pass` and `getattr(obj, key, default)` calls hide
fixes that never run. Recommend a one-time grep for `except Exception` and `getattr(.*,.*,.*)`
calls in physics-critical paths.

### Theme 3: Sibling-function omissions

When a bug is fixed in function A, the sibling function B with the same bug is often missed:

- C-AS-2 `-1j` Maslov in `propagate_hf_chebyshev_quadrature` was fixed in v4.10 → sibling
  in `hf.py` still has the bug (N-AS-4).
- H-AB-3 EP-aiming fix in `eval_image_plane_wfe` (v4.10) → siblings `ray_fan_data`,
  `opd_fan_data`, `field_aberration_sweep`, `relative_illumination` still aim at z=0
  (HIGH-1, H-AB-5).
- C-RT-3 sign(R) fix in `_sag_derivatives_jax` (v4.10/v4.11.1) → `_sag_derivatives_param`
  twin was still wrong (v4.11.1 caught this one).
- `_intersect_jax` got double-where and NaN-guards in v4.11.1 → `_intersect_jax_param`
  Newton step still has the same gradient trap (v4.11.1 caught this one).
- AO rim Zernike FD fix in v4.10 → only handles `+x` and `+y` rim, not `-x`, `-y` (H-AB-3).

### Theme 4: dtype hygiene

Despite v4.11.1's "N7 fix" for `MultiWavelengthMerit` / `MultiFieldMerit`, several functions
still hard-code `complex128`:

- `optimize/core.py:2618-2620`: main `design_optimize` wave-leg input field
- `analysis/core.py:1025, 1189`: `polychromatic_strehl` and `polychromatic_psf`
- `vector_diffraction.py:70, 192-194`
- `gbd.py:134`, `hf.py:114`, `asymptotic.py:328, 1446, 1742, 1751`, `_bluestein.py:143, 317`
- `elements/polarization.py:100-101`: `JonesField.__init__` promotes complex64 → complex128

`precision='single'` is therefore silently negated in roughly half the modules.

### Theme 5: JAX-gradient sinks

Round 1 noted JAX backend was a "second-class citizen". Round 3 finds residual issues:

- `compute_psf`, `strehl_ratio`, `compute_otf` materialize traced values to Python floats
  (`float(...)`, `complex(...)`) — break `jax.grad`.
- `propagate_through_system_jax` calls NumPy `angular_spectrum_propagate` for `'propagate'`
  elements → implicit JAX→host conversion silently zeros gradients.
- Asymptotic Maslov tracking only in `propagate_modal_asymptotic`; JAX twins use principal
  sqrt — wrong-sign cotangent through caustics.

---

## Per-domain critical findings

### Scalar propagators (`propagation.py`, `dispatch.py`, `_bluestein.py`)

- **CRIT** RS back-propagation kernel still wrong (round-2 finding); docstring fixed but no
  z≤0 guard.
- **CRIT** ASM-MFT band-limit `<` (JAX) vs `≤` (NumPy/CuPy) — same propagator, two backends,
  different output at the boundary.
- **CRIT** SAS asymmetric padding `as1 = (N+1)//2` is only correct for `pad=2`. Larger pad
  values place input off-centre.
- **CRIT** Dispatcher routes negative `z` to forward-only propagators (Fraunhofer, SAS) which
  then raise.
- HIGH SAS returns `(E_out, dx_out, dx_out)` — third element duplicates the first.
- HIGH `asm_propagate` docstring claims "MFT variants return 3-tuple" — actually ASM-MFT
  returns bare ndarray, while SAS/Fraunhofer return 3-tuple.
- HIGH Dispatcher signature has no `dy=`, doesn't route 3-D batched fields.
- HIGH `z=0` strips evanescent waves (ASM `H = exp(0) = 1` for propagating but `H=0` for
  evanescent).
- HIGH JAX path uses `xp.float64` but JAX defaults to float32 without `jax_enable_x64`.
- HIGH `apply_fresnel_curvature` uses `indexing='ij'` while every other propagator uses
  `'xy'` (works but a maintenance trap).
- HIGH `resample_field` and `_select_asm_variant` ignore `dy`.

### Asymptotic / GBD / MHS / HF / subaperture

- **CRIT** GBD axial-OPL fix dead-on-arrival (`.get()` on dataclass).
- **CRIT** Maslov branch tracking in only one of five asymptotic sites; sibling NumPy
  `aberration_tensor` and two JAX evaluators still use principal sqrt.
- **CRIT** 2-D raster Maslov unwrap is mathematically ill-defined (row-wraps spuriously flip
  the counter).
- **CRIT** `propagate_huygens_fresnel_with_opl_callable` missing the `-1j` Maslov prefactor.
- **CRIT** `propagate_huygens_fresnel_through_prescription(method='asymptotic')` silently
  discards `E_in`'s structure (replaces with fundamental Gaussian).
- HIGH Modal asymptotic propagator drops linear-phase chief-ray ramp
  (`include_linear=False` while the HF Chebyshev path uses `True`).
- HIGH Subaperture: same global fit used for every patch (patches outside axial box → zero
  field).
- HIGH `propagate_hf_chebyshev_quadrature` strips kernel imaginary part for real `E_in`.
- HIGH MhsPipeline `_validate` uses exact `!=` on float `dx`, missing `centre` equality
  check, no `z` monotonicity check.
- HIGH MhsPipeline `pre_distance` vs `prescription['object_distance']` silently double-counts
  free-space distance.
- HIGH `decompose_field_to_beamlets` uses `(I - N/2 + 0.5)*dx` but `reconstruct_field_from_beamlets`
  uses `(i - N/2)*dx` — internal half-pixel offset.
- HIGH `eval_van_vleck_density` returns `sqrt(|det|)` — drops Maslov sign.

### HFPI / Richards–Wolf / polarization / coatings

- **CRIT** `propagate_hfpi_through_prescription` kills all paths for finite-conjugate.
- **CRIT** `init_paths_stratified` only enumerates 2 strata regardless of input.
- HIGH Richards–Wolf prefactor missing `1/f²` factor (10⁴ intensity error for f=1m vs 1cm).
- HIGH Richards–Wolf global phase `exp(-ikf)` has wrong sign for `exp(-iωt)` convention.
- HIGH Coating `'avg'` mode uses p-pol admittance for both polarizations (line 160-171
  reuses last-iteration `eta_*`).
- HIGH HFPI RNG re-uses master seed at every aperture → perfectly correlated draws across
  diffraction events.
- HIGH HFPI Kirchhoff `1/(iλ)·dΩ` STILL missing at `apply_aperture_diffraction` AND
  completely absent from vectorial HFPI.
- HIGH `apply_waveplate` docstring contradicts implementation (`J = R(-θ)·diag(1,exp(+iφ))·R(θ)`
  vs actual `R(θ)·diag(1,exp(-iφ))·R(-θ)`).

### Ray tracing (`core.py`, `jax_trace.py`, `seidel_analysis.py`, `paraxial.py`)

- **CRIT** `bundles.py` conversion helpers (`ray_to_path`, `path_to_ray`, etc.) are dead
  code — `RayBundle` doesn't have `positions`/`directions` attributes.
- **CRIT** Chained-mirror Seidel parity not tracked.
- **CRIT** `system_abcd` and `seidel_coefficients` use different mirror conventions.
- HIGH `ray_fan_data` / `opd_fan_data` aim chief at `(0,0,0)` instead of EP.
- HIGH `spot_diagram` and `trace_summary` print wrong Airy radius (missing `f_eff` factor).
- HIGH `_paraxial_trace` carries unfixed mirror power bug (dead code, but exported).
- HIGH `_intersect_jax` no convergence check — always runs full 8 iterations even if
  diverging.
- HIGH `seidel_wfe` S5 distortion mixes field-cubed and field-linear scaling.

### Lens / DOE / coronagraph elements

- **CRIT** **`apply_real_lens` Seidel correction sign FLIPPED by the v4.10 "C-LR-1 fix" — the
  pre-v4.10 code was correct; the audit's round-1 identification of C-LR-1 was wrong.**
  `_lens_real.py:828-867`. For a 100 mm-EFL plano-convex BK7 singlet at rim height 12.7 mm:
  - `delta_ray` (geometric OPL relative to axis) = `-(n−1)·sag` ≈ `−0.806 mm` at rim (the
    edge ray has SHORTER geometric OPL — Fermat's principle for a converging wave).
  - `opl_analytic` (sum of `(n2−n1)·sag`) accumulates `+0.806 mm` at rim (thin-element OPD
    increases with sag).
  - These have **OPPOSITE signs** — the pre-v4.10 negation `opl_wave_rel = -(opl_analytic
    - opl_analytic[axis])` was the correct alignment of the two reference frames.
    `correction = delta_ray - opl_wave_rel = -0.806 - (-0.806) ≈ 0` (the expected tens-of-nm
    residual).
  - v4.10's "fix" dropped the negation, producing `correction = -0.806 - 0.806 = -1.612 mm`
    — effectively **triples the lens's analytic OPD at the edge** and shortens the apparent
    focal length by a factor of ~3.
  - v4.11.1's threshold drop from 50 nm → 5 nm doesn't help — the correction magnitude is
    millimeters, far above either threshold.
  - **Existing tests pass because they only assert "field is finite and non-zero"**, not
    that the correction matches ground truth. Round-2 verification marked this ✅ without
    comparing to ground truth.
  - Recommend: restore `opl_wave_rel = -(opl_analytic - opl_analytic[i_ax])`; add a
    regression test that compares `apply_real_lens(..., seidel_correction=True)` against
    `apply_real_lens_traced(...)` (ground truth) to within tens of nm.
- HIGH `apply_real_lens_traced` M_x / M_y indices transposed in paraxial-magnification
  initial guess. `_lens_traced.py:1789-1792`. `np.meshgrid(..., indexing='ij')` puts x along
  axis 0, but the code varies axis 1 → computes ∂x_out/∂y_in (zero by symmetry) instead of
  ∂x_out/∂x_in. Newton converges anyway because the polynomial Jacobian is correct, but every
  pixel starts at the clipped-to-boundary initial guess instead of at the right answer.
  Same bug mirrored in `_lens_jax.py:476-479`.
- HIGH `apply_real_lens` and `apply_real_lens_maslov` silently drop `freeform_type` surface
  terms. `_lens_real.py:536-543`. The raytracer honours freeform; the thin-element-style
  path doesn't. `apply_real_lens_traced` is internally inconsistent (geometric leg includes
  freeform, amplitude leg doesn't).
- HIGH `apply_real_lens_traced` and `apply_real_lens_maslov` silently ignore `stop_index`.
  `_lens_traced.py:1311, 1508`, `lenses_maslov.py:173`. Only `apply_real_lens` honours
  `stop_index`; switching from `apply_real_lens` to the higher-accuracy traced/Maslov path
  silently moves the stop to the entrance.
- HIGH NaN sentinel from aspheric clamp leaks into wave field when `slant_correction=True`
  or `fresnel=True`. `_lens_real.py:563-678`. `np.gradient(sag)` propagates NaN; comparisons
  with NaN return False so the near-grazing warning never fires for NaN regions;
  `exp(-ik·NaN) = NaN` then poisons the entire downstream ASM step.
- HIGH `_Cheb2DEvaluator` polynomial fit doesn't check for sufficient finite samples.
  `_lens_traced.py:471-480`. When many entries of `vals` are NaN, lstsq returns a
  minimum-norm solution with no warning. Mirrored in `_lens_jax.py:117-122`.
- MEDIUM Per-surface `decenter` on the prescription dict is silently dropped by the
  ray-tracer. `raytrace/core.py:1459-1477` (`surfaces_from_prescription`). The `Surface`
  dataclass has `decenter_x_m, decenter_y_m` fields but `surfaces_from_prescription` never
  reads `ps.get('decenter')`. `apply_real_lens` (v4.11.1) and the ray-tracer see different
  geometry for the same prescription — affects every downstream consumer including the
  Seidel correction's own ray fan.
- MEDIUM `apply_mirror` lacks `dy` parameter — silently wrong on anamorphic grids.
- MEDIUM `GaussianBSDF.evaluate` has broken broadcasting for batched scattered directions.
- MEDIUM Seidel correction's 1-D fan only goes to `0.9·r_pupil` — misses 10% of marginal-ray
  aberrations; polynomial extrapolates from 0.9 to 1.0 on a 6th-order fit.
- MEDIUM `_reverse_prescription` doesn't reverse decenter / tilt / form_error signs (only
  affects opt-in `inversion_method='backward_trace'` path).

### Aberration / AO / WFE / field / coherence / ghost / interferometry

- **CRIT** `seidel_wfe` missing field-curvature DC term `(1/4)·S₃·ρ²` (both docstring AND
  code wrong by the same amount).
- **CRIT** `distortion_grid` silently produces unphysical N=0 rays at moderate field angles
  (no `L²+M²>1` guard).
- HIGH `ghost.py` intensity formula ignores transmission losses (~3× over-estimate for
  10-surface systems).
- HIGH `ghost.py` `focus_z` formula is dimensionally arbitrary.
- HIGH AO rim Zernike FD only handles `+x`/`+y` rim — same spike pattern on `-x`/`-y`.
- HIGH Image-plane reference-sphere drops `1/N_chief` factor for off-axis chief.
- HIGH `field_aberration_sweep` and `relative_illumination` aim rays at z=0 instead of EP.
- MEDIUM `seidel_wfe` H-fallback uses image-height `f_eff·σ` proxy instead of true Lagrange
  invariant `y_pupil·σ` — for f/4 system, H_squared is 64× too large.
- MEDIUM `petzval_radius` docstring formula sign-mismatched with implementation (both work,
  but docstring is misleading).

### Analysis core / through-focus / phase retrieval / detector / sources

- **CRIT** Phase retrieval `seed=` / `dtype=` parity STILL broken (gerchberg_saxton_jax
  ignores seed; NumPy error_reduction and hybrid_input_output don't accept seed).
- **CRIT** `compute_psf` Parseval default broke existing peak-equals-1 tests; tests now pass
  for the wrong reason (`t_strehl_perfect` asserts `psf.max() > 0.99` but with 'power' default
  the peak is ~90000).
- **CRIT** `polychromatic_strehl` and `polychromatic_psf` still hard-code `complex128`
  (v4.11.1's N7 fix was incomplete).
- HIGH `find_best_focus` is NaN-fragile but `monte_carlo_tolerancing` relies on it (v4.11.1
  added guards in merits but not in find_best_focus itself).
- HIGH Linearized MC tolerancing can predict S > S_nom (negative a_k from one-sided FD
  probe breaks Maréchal invariant).
- HIGH `compute_psf` silently assumes square pupil.
- HIGH `Source.*` classmethods don't propagate `dtype=`, `dy=`, `normalize=` to underlying
  factories.
- HIGH `apply_detector` non-integer pixel-pitch / dx_field ratio gives wrong area integral.
- MEDIUM Inconsistent normalization defaults across source factories (peak vs power vs
  silent-power vs raw).
- MEDIUM Cosmic-ray model still single-pixel deposit (real cosmic rays are tracks).

### Optimizer / glass / system / memory / user_library / context

- **CRIT** `S-LAH64` and `S-LAH79` Sellmeier coefficients are wrong (`n_d` off by 0.058 and
  0.117 respectively).
- **CRIT** `precision='single'` silently negated in `design_optimize` main wave-leg
  (`E0 = np.ones((N,N), dtype=np.complex128)` hard-coded).
- **CRIT** JAX/NumPy aperture schemas are mutually incompatible (`'radius'` /
  `'half_width_x'` / `'inner_radius'` vs `'diameter'` / `'width_x'` / `'inner_diameter'`).
- **CRIT** `eval()` phase-mask sandbox still open through `np` module exposure (`np.load`,
  `np.save` reachable).
- HIGH `design_optimize` LM silently switches to TRF when bounds are present.
- HIGH `_dtype_restore_guard` relies on CPython `__del__` (unreliable at interpreter shutdown
  and `KeyboardInterrupt`).
- HIGH `lumenairy_context()` doesn't revert partial changes if `apply_globals(new_state)`
  raises midway.
- HIGH `MultiFieldMerit` z-scan around on-axis BFL (H-AB-5 still not fixed).
- HIGH `_LM_FLOOR = 1e-30` too small to regularize the residual gradient near merit=0.
- HIGH `load_phase_mask` glass_block branch still has 1.0 m wavelength fallback (only the
  expression branch was hardened).
- HIGH `pick_batch_size` / `should_split` ignore `_MAX_RAM_OVERRIDE` (call
  `available_memory_bytes()` instead of `get_ram_budget()`).

### IO / Zemax / prescriptions / storage / codegen (new audit area)

- **CRIT** EVENASPH aspheric coefficient round-trip broken (PARM off-by-one).
- **CRIT** Quadoa aspheric serializer iterates dict keys instead of values.
- **CRIT** Zemax exporter places STOP marker on wrong surface in folded designs.
- **CRIT** `normalize_prescription` mirror filter is a no-op.
- **CRIT** Mirror DISZ lost on Zemax round-trip; coord-break DISZ double-counted.
- HIGH codegen emits `op.GLASS_REGISTRY` but imports `lumenairy as la` → NameError.
- HIGH codegen drops `aperture_diameter` for mirror elements in system-list style.
- HIGH `load_material` silently drops the saved dispersion field.
- HIGH `aspheric_coeffs` has three different types across loaders (None / dict / list).

### Test suite quality (new audit area)

- 3 of 9 v4.11.1 pinning tests pass for the wrong reason (MultiWavelengthMerit checks
  warning absence not semantics; Subaperture is import-only; Tilted-ASM tilt too small).
- 3 more have adequate-but-loose tolerances (point source, mirror Petzval).
- **No phase pinning test for the RS kernel sign fix anywhere in the suite** — the v4.10
  Goodman 3-43 fix is invisible to any regression check.
- **No Zemax-cross-check test for the coord-break order** — the world-frame test only
  asserts orders disagree, not which one matches PARM 6 = 0.
- Validation harness `_harness.py:33` does `warnings.simplefilter('ignore')` →
  RuntimeWarnings emitted inside library code during validation are silently swallowed.
- Several validation tests have inner `try/except: return True, 'skipped'` patterns that
  paper over real failures.

### Cross-cutting integration

Mostly verifies the v4.11.1 surface is in good shape:
- Sign conventions consistent across all major sites.
- `array_namespace` dispatch correct.
- `lumenairy_context` snapshot/restore round-trip working.
- Cache invalidation via `clear_asm_caches` / `reset_fft_backend` working.

But surfaces three residual concerns:
- Hard-coded `complex128` in ~9 modules (`vector_diffraction`, `gbd`, `hf`, `asymptotic`,
  `_bluestein`, `JonesField.__init__`).
- JAX-gradient sinks in `compute_psf`, `strehl_ratio`, `compute_otf`.
- Half-pixel grid convention not actually library-wide (gbd, hf, subaperture, optimize/core
  use `+0.5`; ASM/Fresnel/RS don't).

---

## Recommended fix priorities for v4.11.2

### Tier 1 (silently wrong numerical answers in default code paths)

1. **`S-LAH64` / `S-LAH79` Sellmeier coefficients.** Verify against OHARA catalog; replace
   or remove the `__sellmeier__` flag and fall back to refractiveindex.info.
2. **EVENASPH PARM off-by-one.** Single-line formula fix in `prescriptions.py`
   (`power = 2 + 2*parm_num`, filter `>= 1`). Add round-trip test.
3. **Chained-mirror Seidel parity.** Track `mirror_parity = mirror_count % 2` in
   `seidel_coefficients` AND `system_abcd`; flip `n1`, `n2` accordingly. Add a Cassegrain
   regression test with hand-computed Seidel sums.
4. **`system_abcd` mirror sign.** Reconcile with `seidel_coefficients`. Either flip
   `system_abcd` to use `phi = (n2-n1)/R` with `n2=-n1` (matches Welford), or document the
   sign convention loudly.
5. **`seidel_wfe` missing `(1/4)·S₃·ρ²`.** Append `+ (1/4)*S3*rho2` to the return. Update
   docstring formula.
6. **`propagate_hfpi_through_prescription` finite-conjugate dead path.** Init at
   `z=-object_distance`, propagate forward.
7. **`init_paths_stratified` cartesian product.** Use `np.indices((n_iy, n_ix, n_th, n_ph)).reshape(4, -1)`.
8. **Richards–Wolf `1/f²` and `exp(+ikf)` sign.** Two single-line fixes in
   `vector_diffraction.py:221`.
9. **`normalize_prescription` mirror filter.** Change `e.get('mirror')` to
   `e.get('element_type') == 'mirror'`.
10. **Quadoa aspheric serializer.** Rewrite to iterate `coeffs.items()`.
11. **`compute_psf` Parseval default broke existing peak-equals-1 tests.** Update at least
    `t_strehl_perfect` (and similar) to use `normalize='peak'` explicitly OR rewrite to test
    the actual Parseval invariant. Audit all tests for the same staleness.

### Tier 2 (dead-on-arrival fixes from prior waves)

12. **`bundles.py` AttributeError** on `RayBundle.positions`. Either add `@property`
    accessors on RayBundle or rewrite the conversion helpers.
13. **GBD axial OPL `.get()` on dataclass.** Change to attribute access (`s.thickness`,
    `s.glass_after`).
14. **`propagate_huygens_fresnel_with_opl_callable` missing `-1j` Maslov.** Apply same fix as
    `propagate_hf_chebyshev_quadrature`.
15. **Coating `'avg'` p-pol admittance reuse.** Save `eta_*` per polarization inside the loop.
16. **AO rim Zernike FD asymmetric.** Extend to handle `-x` and `-y` rim too.

### Tier 3 (sibling-function omissions)

17. **`ray_fan_data`, `opd_fan_data`, `field_aberration_sweep`, `relative_illumination`**
    aim at EP, not z=0. Port the v4.10 H-AB-3 fix.
18. **Asymptotic Maslov tracking** in `aberration_tensor`, `aberration_tensor_lg00_jax`,
    `_modal_field_lg00_pixel_jax`. Hoist the v4.10 branch-tracking logic into a shared helper.

### Tier 4 (test suite quality)

19. **Strengthen the 3 weak v4.11.1 pinning tests** (MultiWavelengthMerit chromatic
    semantics; Subaperture actual function call; Tilted-ASM with `fx0 > fx_max`).
20. **Add RS-vs-ASM phase pinning test** at z > 0 with `bandlimit=True`. This is the
    single biggest test-coverage gap.
21. **Add Cassegrain Seidel regression test** with hand-computed S1..S5.
22. **Remove `warnings.simplefilter('ignore')`** from `_harness.py` (or scope it to a
    specific category).

### Tier 5 (dtype hygiene, JAX-grad, docstring drift)

23. Replace remaining hard-coded `np.complex128` allocations with `get_default_complex_dtype()`
    in `vector_diffraction.py`, `gbd.py`, `hf.py`, `asymptotic.py`, `_bluestein.py`,
    `JonesField.__init__`, `polychromatic_strehl`, `polychromatic_psf`,
    `design_optimize.E0`.
24. Replace `float(...)` / `complex(...)` materializations in `compute_psf`,
    `strehl_ratio`, `compute_otf` with `jnp.where`-based array operations to preserve
    JAX gradients.
25. Sync `apply_waveplate` docstring formula with the actual implementation.
26. Sync `petzval_radius` docstring formula with the implementation (Born & Wolf §4.4 sign).

---

## Summary numbers

| Audit round | Total findings | Critical | High | Medium | Low |
|---|---|---|---|---|---|
| Round 1 | ~100 | ~22 | ~35 | ~30 | ~13 |
| Round 2 | ~50 (verification) | 6 unfixed + 5 new bugs | — | — | — |
| **Round 3** | **~120 substantive findings (this report)** | **~25** | **~50** | **~30** | **~15** |

The codebase has converged on a coherent `exp(-iωt)` convention across all the propagator /
lens / source / polarization sites I checked — that's a substantial win for a 74 kLOC
codebase. **The remaining issues cluster in three buckets**:

1. **Mirror handling** (multi-mirror Seidel, mirror sign convention, mirror in IO round-trip,
   mirror in lens-API guards).
2. **Sibling-function omissions** (bugs fixed in A, still present in B that does the same
   physics).
3. **Module-level surface area not previously audited**: IO/Zemax/prescriptions (5 critical
   findings; biggest impact: aspheric coefficient round-trip silently lossy for every
   Zemax-authored EVENASPH file).

The pattern across all three rounds suggests a single high-leverage process change:
**every fix should add a regression test that would have FAILED on the buggy version.**
This was the missing discipline in v4.10's "ship without tests" wave, and the symptoms
(dead-on-arrival fixes, sibling-function omissions, pinning tests that pass for the wrong
reason) recur because they're not gated.

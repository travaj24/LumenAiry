# WP-A2 — `apply_real_lens` analytic model (`elements/_lens_real.py`, `elements/lenses.py`)

Read first: `COMMON.md`, then the partition reports `RL-CORE.md` (all of it), `RL-MODELS.md`, `ORCHESTRATOR.md`
(F-O1, F-O2, F-O3), and report sections §2.1 (rows L1–L20), the E4 row in §5, §14 rows V1–V2 and §15 in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`. Repro: `repro/RL-CORE/` (incl. the independent ray oracle
`oracle.py`), `repro/RL-MODELS/`, `repro/orch/seidel_*.py`, `repro/orch/surface_frame_tilt_check.py`.

## Files you own
`lumenairy/elements/_lens_real.py`, `lumenairy/elements/lenses.py`. Tests: the Seidel test in
`tests/unit/test_audit_glass.py:92–152` (only that test — WP-A8 owns the glass-catalogue material and will add a NEW
file), `tests/unit/test_v5_2_off_axis_conic_surface_frame.py`, any other test file that pins `_lens_real` / `lenses`
behaviour, and new files `tests/unit/test_audit2609_a2_*.py`. Do NOT touch `propagators/asm.py` (the fftshift-folding
perf item belongs to WP-A5 — note it in "requested changes") nor `raytrace/` (WP-A1).

## Findings to implement (all of them; the order below is the recommended order)
- **L12 (P1 ✔)** `slant_correction`: replace `(n2·cosθt − n1·cosθi)·sag` by the module's own axial-translation identity
  `(n2·cos(θi−θt) − n1)·sag` = `(n2·(cos_ti·cos_tt + sqrt(sin2_ti·sin2_tt)) − n1)·sag` in BOTH copies (`:6156`, `:5780`).
  Verify with `repro/RL-CORE/p5c_slant_sign.py` and `p5_slant.py` against `oracle.py` (expect 290–4000× better than
  paraxial on a single surface); rewrite the docstring guidance.
- **L13 (P1 ✔)** `fresnel=True`: multiply `T_eff` by `(n2r·cos_tt)/(n1r·cos_ti)` in both copies so the applied factor is
  the POWER transmittance under the library's Σ|E|²·dx² = power convention; state that convention in the docstring
  (and request a CONVENTIONS.md sentence in your report). Verify `repro/RL-CORE/p3_fresnel_energy.py`
  (single AIR→BK7 face 0.958, cemented BK7→SF11 0.994, plate unchanged 0.917873).
- **L14 (P1)** `stop_index`: normalise negatives, raise `ValueError` (§2 prefix) outside `[0, len(surfaces))`;
  make `prepare_real_lens` and `apply_real_lens` agree about the key.
- **L2 (P1 ✔)** `surface_frame=True` drops the tilt: compute the FIELD-frame height of the rotated surface,
  `z_f = (R @ (x_s, y_s, g(x_s, y_s)))_z` with `R = Rx(tx) @ Ry(ty)`, and use `opd = (n2−n1)·z_f`; the two existing tilt
  assertions in `test_v5_2_off_axis_conic_surface_frame.py` pin the DEFECT — re-derive them (V2): a tilted flat face must
  deviate the beam by (n−1)θ (measured from the exit-phase gradient / centroid), and delete the two
  `array_equal(no_kwarg, default_kwarg)` tautologies. Oracle: `repro/RL-CORE/p8b_surfaceframe_exact.py`.
- **L19 (P3, do together with L2)** the field-frame `tilt` convention is `(−θy, +θx)` while CONVENTIONS §7, `raytrace`
  and the surface-frame branch use right-hand rotations about +x/+y. Align the field-frame branch with §7 (ramp
  `−θy·x + θx·y`). This CHANGES results for existing `tilt=` users: document the migration
  (`new_tilt = (old_t1, −old_t0)`) in your changelog text, and check whether `raytrace.surfaces_from_prescription` reads
  the same `tilt` key from the same prescription dict — if so the fix also makes ray and wave paths agree (say so).
  Also L19's second item: `absorption` uses the axial thickness — use the local glass path `t − sag_i + sag_{i+1}`.
- **L3 (P1 ✔)** the displaced remaps discard the input phase: carry it (resample `|E|` and the residual phase
  `angle(E_in·exp(+ik·W_conj))` — or the input's unwrapped-free residual against the conjugate congruence — separately
  and re-apply it), or at the very least detect a non-conjugate input (residual gradient above a threshold) and warn
  loudly; fix the `conjugate` docstring. Verify with `repro/RL-MODELS/t5_phase_discard.py` (flat vs 35-wave-defocused
  input must now differ; the 150 mm diverging source must focus at 25 mm, not 21 mm).
- **L4 (P1)** `_screen_obliquity_angle_field` applies n1 to momenta that are already optical → double count in glass
  (×1.5168): apply n1 only on the `TiltedCarrier` / scalar-conjugate branches. Verify on the immersed-surface fixture.
- **L5, L6 (caches)** key `_DISPLACED_COS_GRID_CACHE` on a VALUE fingerprint of `sag_callable` (evaluate on a fixed
  stencil) and `_DISPLACED_LUT_CACHE` on the resolved index, not the glass NAME.
- **L7** displaced ray maps hard-code n_exit = 1: use `get_glass_index(surfaces[-1]['glass_after'])`.
- **L8** `tangent_facet_remap` fold / pull-back guards must score over the pupil / non-zero support, not the padding.
- **L9** scale the 2-D remap resolution and the pointwise cos-grid with the grid; break the Newton loop on the residual.
- **L10** documented approximation — docstring only (derive the AOI from the carrier momentum when available if cheap).
- **L15** `form_error`: shape/dtype validation with the §2 prefix; document that it is a FIELD-frame map.
- **L16 (perf)** `surface_sag_general` in place (`out=`), bit-identical (5.13 → 2.0 float64 grids, 4.5×).
- **L17 (perf)** flat-surface early-out on the default screen; cos/sin screen into a preallocated complex view;
  `E *= ph` and `np.multiply(out=)` / boolean assignment instead of fresh `xp.where` grids on the fresnel/slant/aperture
  path. Bit-identical where claimed (the cos/sin form is bit-identical to `exp` per the auditor's measurement; verify),
  measured with interleaved medians + tracemalloc.
- **L18** qualify the `sag_chunk_rows` "wall-clock neutral" sentence with the measured numbers.
- **L20** misc: `_warn_if_aperture_exceeds_grid` extent on anamorphic grids; `fresnel` refraction angle from the complex
  index and the AOI ceiling documented; delete/wire `_VALID_SCREEN_OBLIQUITY`; document the numexpr/numpy complex64
  1-ULP difference; reject `slant_correction=True` + `seidel_correction=True` in
  `_check_apply_real_lens_kwarg_combination` (or make them consistent).
- **L11** `assert` → `ValueError` for the thicknesses/surfaces contract; the `screen_obliquity` performance table numbers;
  `on_partial_aperture` reachability; dead `_split_mode` branch.
- **E4 (P1, `lenses.py:263–276`)** the numba aspheric kernel drops the whole polynomial for a non-C-contiguous `h_sq`
  (accumulates into a `ravel()` copy): `np.ascontiguousarray` and assign back; test with F-order and transposed input.
- **L1 (P1 ✔) — do this LAST.** `seidel_correction=True` is 8–3600× worse than off because (a) the fit basis contains ρ²,
  (b) the fan OPL is read at the last-surface sag instead of the exit vertex plane, (c) the "analytic" reference
  `Σ(n2−n1)·sag_i(h)` is not the split-step model's OPL (the in-glass ASM legs already carry the slab obliquity, so the
  correction double-counts it). Implement it CORRECTLY: (1) transfer the fan to the exit vertex plane — before you start,
  check whether `lumenairy/raytrace` already exposes `TraceResult.at_exit_vertex(...)` (WP-A1 is adding it concurrently;
  `grep -rn "at_exit_vertex" lumenairy/raytrace`); if present use it, otherwise implement the signed transfer
  `t = −z/N; opd += n·t; x,y += (L,M)·t` as a private function marked `# TODO(audit-2609/A1): replace with
  TraceResult.at_exit_vertex` for the orchestrator to swap; (2) build the model reference from a thin-screen RAY model
  of what the split-step actually does (at each surface deflect the ray by the screen gradient ∇[(n2−n1)·sag] at its
  transverse position without axial motion, then propagate straight through the glass gap accumulating n·t/cosθ — i.e.
  the eikonal of the ASM leg) evaluated on the same fan, both referenced to the exit vertex plane; (3) fit the residual
  from ρ⁴ upward (no ρ²), keep the 5 nm gate. Decision rule, measured against `repro/RL-CORE/oracle.py` with
  `p6_seidel.py` / `p6d_psf.py`: on the plano-convex (true residual 0.85 nm) the gate must now SKIP; on the 8 mm doublet
  (true residual 173 nm) the corrected exit OPD must improve by at least 3× AND the through-focus peak must not drop.
  If you cannot meet that rule, do NOT ship a partial correction: turn the flag into a `DeprecationWarning`-ed no-op
  pointing at `apply_real_lens_traced`, and say why in the report. Either way rewrite the docstring (it currently
  RECOMMENDS the option for AC254-class doublets) and replace the vacuous test in `test_audit_glass.py:92–152` (V1) with
  an unwrapped, curved-rear, focus-position assertion whose bar you derive.

## Verification specifics
- The default `thin` path's exit OPD vs `oracle.py` (0.848 / 1.18 / 1.83 / 10.9 nm rms on the four fixtures) must not
  regress; banded == whole-grid byte identity must hold after every change (`repro/RL-CORE/p10_banded_identity.py` —
  keep the three surface bodies consistent; if you find a way to collapse them into one body generator, do it, with the
  byte-identity matrix as the gate); `prepare_real_lens` byte-identical to `apply_real_lens`; complex64 6.7e-7.
- Re-run `repro/orch/seidel_focus_shift.py` and `repro/orch/surface_frame_tilt_check.py` after the fixes.
- Every default that changes (L13's per-interface factor, L19's tilt convention, L1's behaviour) goes into your
  changelog text with the measured before/after and a migration note.

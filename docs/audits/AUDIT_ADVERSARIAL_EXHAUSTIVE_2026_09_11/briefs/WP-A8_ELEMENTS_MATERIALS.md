# WP-A8 — Thin elements, DOEs, materials (`lumenairy/elements/*.py` top level, excluding the lens/solver families)

Read first: `COMMON.md`, then the partition report `THIN-ELEMENTS-GLASS.md`, the `ORCHESTRATOR.md` rows on the
glass rows / turbulence / conic, and report section §5 (rows E1–E3, E5–E7; E4 belongs to WP-A2) plus §15 in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`. Repro: `repro/THIN-ELEMENTS-GLASS/`,
`repro/orch/verify_rt_glass.py`, `repro/orch/verify_turb.py`.

## Files you own
`lumenairy/elements/glass.py`, `elements/elements.py`, `elements/_lens_thin.py`, `elements/bsdf.py`,
`elements/thin_grating.py`, `elements/doe.py`, `elements/zernike.py`, `elements/materials.py`, and any other top-level
`lumenairy/elements/*.py` EXCEPT: `_lens_real.py`, `lenses.py` (WP-A2), `_lens_traced*.py`, `_lens_imap.py`,
`_traced_flags.py` (WP-A3), `_lens_jax.py`, `lenses_maslov.py`, `lenses_gbd.py`, `fga.py` (WP-A4), `coatings.py`,
`berreman*.py` (WP-A11), and the `pmm/`, `rcwa/`, `eme/`, `bor/` subpackages. Tests: thin-element/glass test files
EXCEPT `tests/unit/test_audit_glass.py` (WP-A2 is rewriting a test in it — put your glass tests in a new file), plus new
`tests/unit/test_audit2609_a8_*.py`.

## Findings to implement
- **E1 (P0 ✔)** three bundled Sellmeier rows are a different glass: N-BAF52 (n_d 1.637147 vs 1.608631 — looks like a
  mis-copied N-KZFS11 row), N-LAK33A/B off by 3e-4. Source the correct SCHOTT coefficients (the `refractiveindex`
  package with its database IS installed here — use it to obtain/verify; cross-check n_d and V_d against the catalogue
  values), replace the rows, and add a VALUE cross-check (n_d, V_d for every bundled row against the package when it is
  importable, tolerance derived) to `_check_glass_registry_consistency` so this class cannot recur. Verify with
  `repro/orch/verify_rt_glass.py` (N-BK7, N-SF11 must stay exact).
- **E2 (P1)** `get_glass_index_complex` raises `NoExtinctionCoefficient` (not in the caught tuple) instead of the
  documented κ = 0 fallback for CaF2, fused silica, MgF2, Si with the recommended extra installed — catch the package's
  class via `getattr`/`Exception` and test the package-PRESENT path.
- **E3 (P1 ✔)** `generate_turbulence_screen` delivers 2× the requested phase variance (`sqrt(2·psd)`): drop the factor,
  add the subharmonic low-frequency correction as an option (default off unless you can show it is safe), verify the
  structure-function ratio → 1.00 as r → 0 over ≥ 40 seeds with a derived bar; document the default change.
- **E5 (P1)** `apply_grin_lens` uses the thin form f = 1/(n0·g²·d) while recommending quarter pitch (36 % wrong there):
  φ = −k·n0·g·sin(gd)·r²/2 (exact for the parabolic-index rod), warn for g·d > 0.2 only if the caller forces the thin
  form; verify the ASM focus tracks f_exact.
- **E6 (P2)** Harvey–Shack TIS closed form or u = sinθ grid (18 % under-read at l = 1e-3); `make_bsdf` key validation
  incl. the documented `A`/`B` aliases; `thin_grating_efficiency_1d` Klein–Cook guard; `create_microlens_array`
  separable phase (7× peak); vectorise `sample_scatter_rays`. (The `surface_sag_general` conic-branch memory item is
  WP-A2's.)
- **E7 (P3)** `elements.zernike` Noll pointer; `glass.py` docs of `'air'`/`'vacuum'`/`'__MIRROR__'` registry entries vs
  reality and the Edlén-air impossibility; `create_periodic_phase_mask` non-square cells; `surface_sag_general(R=0)` →
  raise (coordinate with WP-A2 if the fix must live in `lenses.py` — then request it); `apply_aperture` grey-pixel edge as
  an option; document the reference-plane difference between `apply_spherical_lens` and `apply_real_lens`.

## Verification specifics
- The audit verified correct: conic/aspheric sag to 1e-19 m, thin-lens sign/focus, axicon, thin-grating Fourier
  coefficients, Rytov 0th/2nd order, kinoform/FZP efficiencies, BSDF oblique normalisation, glass-name resolution,
  κ sign, Forbes-Q normalisation, Zernike recurrence/normalisation, the turbulence PSD shape — keep them.
- Re-run the repro scripts before/after and quote numbers.

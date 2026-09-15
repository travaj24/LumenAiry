# WP-A9 — Designer UI (`lumenairy/ui/`)

Read first: `COMMON.md`, then the partition report `UI.md` and report section §6 (U1–U7), §15 in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`. Repro: `repro/UI/` — PySide6 is NOT installed on this
machine; the auditor built a Qt stub there (find it and reuse it for every test you write — never `pytest.skip` on
PySide6 absence; the stub is the harness).

## Files you own
`lumenairy/ui/*.py`, `run_lumenairy_designer.py`, `GUI_README.md` (only if a finding needs it). Tests: UI test files
(under the stub) and new `tests/unit/test_audit2609_a9_*.py`. NOT `raytrace/trace.py` (U2's `surfaces_from_prescription`
fallback — WP-A1 finished its pass; if the `elements`-fallback indexing still mis-assigns semi-diameters when mirrors
are present, fix the UI side by emitting per-surface `semi_diameter`, and put the exact `trace.py` change in your report).
Write your changelog text for BOTH `CHANGELOG.md` (library-facing effects) and `GUI_CHANGELOG.md` (the UI keeps its own
changelog) into `fixes/WP-A9_CHANGELOG.md`, clearly separated.

## Findings to implement
- **U1 (P0 ✔)** `SystemModel.to_prescription()` strips `is_mirror` and emits Zemax-signed negative post-mirror
  thicknesses into the legacy `surfaces`/`thicknesses` keys — 18 call sites in 14 docks analyse a different system than
  the layout shows (EFL 158.9 vs 178.8 mm). Emit `is_mirror`, `semi_diameter`, `is_stop` per surface; for folded systems
  either refuse the legacy keys or use the library's `allow_unfolded_equivalent` handshake; re-run `repro/UI/t2_mirror.py`.
- **U2 (P0 ✔)** per-surface semi-diameters re-indexed onto the wrong surfaces whenever a mirror is present — emit
  `semi_diameter` per surface (see the ownership note).
- **U3 (P1)** the point-source ray bundle is degenerate (azimuth loop variable unused; marginal ray √2 too steep).
- **U4 (P1)** a default wave-optics run permanently disables `USE_PYFFTW`/`USE_SCIPY_FFT` process-wide from the worker
  thread; `set_max_ram` likewise — save/restore (`try/finally`) or the library's scoped API.
- **U5 (P1)** the lens-model router falls back to the crude per-surface ASM loop on ANY exception and never says so —
  report `lens_model_used`, honour `allow_unfolded_equivalent`, surface the exception.
- **U6 (P1)** the crash/λ-mismatch list: `_on_finished` slicing with the INPUT N; Insert ▸ Source presets
  `AttributeError`; source edits resetting λ to 1310 nm and dropping polarization / model wavelength never synced;
  `emitter_array` unusable; `_build_trace_surfaces_world` dropping inter-element air gaps; PSF/MTF "pupil from ray trace"
  using a non-existent `.opl` and image-plane coordinates; the optimizer worker mutating the live model from the worker
  thread; 14 of 16 QThread workers ignoring `requestInterruption()`.
- **U7 (P2/P3)** workers shadowing `QThread.finished`; dead controls; `power_in` plane; `File ▸ New` re-running
  `__init__`; `_apply_real_lens_asm_equiv` calibration; the blocking benchmark on the GUI thread; module-scope matplotlib
  imports (30 of 49 modules — make them lazy); a worker base class.

## Verification specifics
- Every model-level fix gets a stub-driven test (the stub is the only harness available); keep the coordinate-break
  math (verified equal to `raytrace/world.py`), singlet export byte-equivalence with `make_singlet`, aperture
  diameter/radius consistency, refraction phase-screen sign, Airy / f-number formulas intact.
- Do not import `lumenairy.ui` from anything outside `ui/` (the audit verified it is not on the `import lumenairy` path —
  keep it that way).

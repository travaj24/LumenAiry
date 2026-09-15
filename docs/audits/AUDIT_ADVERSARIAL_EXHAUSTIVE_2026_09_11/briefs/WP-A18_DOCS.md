# WP-A18 — Documentation: README / Migration-Guide / CONVENTIONS updates and a living subsystem document

Read first: `COMMON.md`, then `TESTS-ARCH.md` [P3-2] and the "Docs consolidation strategy", report §14 V7 and §15 in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`, and EVERY `fixes/WP-*_REPORT.md` and `fixes/WP-*_CHANGELOG.md`
(the doc changes other engineers requested: CONVENTIONS sentences for the Σ|E|²·dx² = power convention, the field-frame
`tilt` convention, the Richards–Wolf pupil coordinate, the §7.1 Jones wording (already done by WP-A12 — verify), the
CODE V `DIM` migration note, changed defaults).

## Files you own
`README.md`, `Migration-Guide.md`, `ROADMAP.md`, `CONVENTIONS.md` (all sections except what WP-A12 already changed in
§7.1 — read its diff first), `docs/*.md` except `docs/audits/`, new `docs/subsystems/real_lens.md`, `GUI_README.md`
(if the UI WP requested it). NOT `CHANGELOG.md` (the release step assembles it).

## Deliverables
1. Fix the non-resolving identifiers the audit listed (`_decompose_prescription`, `focus_fixed_sampling`,
   `_detect_backend`, `return_kind`, `opl_fn`, `image_centres`, `world_origin`, `world_R`, `rcwa_1d`,
   `_PROPAGATE_SYSTEM_JAX_CACHE`, `row_reset`, `fiber_mode`, `_spawn_rng`, `_fd_grad_pure`, …) — resolve each against the
   current package (`python -c "import lumenairy; ..."`), rewrite or remove; then run a resolver over all backticked
   identifiers in README/ROADMAP/Migration-Guide/CONVENTIONS and report the remaining count with a clean denominator
   (exclude exception names, parameter names, test citations).
2. Add every convention sentence requested by the other WPs to `CONVENTIONS.md` (with its "Changelog of this file"
   entry), and every migration note to `Migration-Guide.md` (tilt convention, CODE V units, `fresnel=True` per-interface
   factor, `seidel_correction` behaviour, `min_feature` default, HFPI `rng=None`, PMM 2-D `cascade` default, …) — take
   them from the WP changelog files; do not invent.
3. `docs/subsystems/real_lens.md`: the living contract of the `apply_real_lens` family — models, what each option does
   NOW, measured accuracy envelopes from the audit and the fix reports (cite the repro scripts), known limits, and which
   test pins each claim. Model it as the first of the per-subsystem documents the audit recommends; keep it under ~400
   lines.
4. README: split nothing large in this pass, but make the landing section point at the audit report, the subsystem doc
   and the migration notes; fix the cookbook snippets that reference changed defaults.

## Verification
- Run the repo's doc walkers/tests that read README/Migration-Guide (`grep -rln "README\|Migration" tests/unit`) after
  your edits; every code snippet you touch must execute (run them).

## Addendum (orchestrator ruling, 2026-09-12)
- Tilt convention: WP-A2 did NOT re-spell `tilt`; it unified both `apply_real_lens` branches on the convention `raytrace/surface.py::_field_frame_sag_and_grad` and the geometric spot oracle already use (pinned by `tests/unit/test_niche_p9_decenter_tilt.py`). Align `CONVENTIONS.md` §7 to the IMPLEMENTED convention (read `_lens_real.py`'s tilt block and WP-A2_REPORT.md §2 L2+L19 for the exact axis/sign statement) and remove the contradicting sentence; NO `new_tilt = (old_t1, -old_t0)` migration note is needed.
- Power convention sentence (WP-A2 L13): `sum |E|^2 dx dy` IS the power; an index step needs `T = (n2 cos theta_t)/(n1 cos theta_i) |t|^2` -- add to §7 with the L13 cross-reference.
- `docs/subsystems/real_lens.md` must carry VERIFY-A2's sampling recipe (report §3): measuring the analytic model's exit OPL against a ray oracle needs `dx ~ 1.45 * aperture / 2048`, not a carrier-Nyquist pitch -- at N = 512 a meniscus reads 558 nm of pure hard-edge aliasing through the in-glass ASM. Also record the L1 fast-element caveat and its clamp fix (VERIFY-A2 follow-up) once landed.

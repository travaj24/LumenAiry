# WP-A1 — Ray tracing (`lumenairy/raytrace/`) + the shared exit-vertex helper

Read first: `COMMON.md` (rules), then the partition reports `RAYTRACE.md` and `ORCHESTRATOR.md` (F-O1–F-O4 and the
exit-vertex census) and report sections §4 (rows R1–R7) and §15.1 in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`. Repro scripts: `repro/RAYTRACE/`, `repro/orch/`.

## Files you own
`lumenairy/raytrace/*.py` (all modules: trace, surface, intersection, ray_fan, seidel, seidel_analysis, from_field,
world, world_trace, jax_trace, differential, paraxial, _conic_core, layout, …). Tests: every `tests/unit/test_*raytrace*`,
`test_*seidel*`, `test_*ray_fan*`, `test_*from_field*` file, plus new files `tests/unit/test_audit2609_a1_*.py`.
You do NOT own the consumers of ray results (`elements/_lens_real.py`, `elements/lenses_maslov.py`,
`elements/_lens_traced*.py`, `propagators/asymptotic_canonical_fit.py`, `analysis/image_plane_wfe.py`) — other WPs will
switch them to your helper in the next wave, so its contract must be precise and documented in your report.

## Deliverables, in this order
1. **The exit-vertex helper (new API, do this FIRST and report its exact signature early in your final report).**
   `raytrace.trace()` leaves rays at their intersection with the LAST surface (z = sag(ρ)); seven consumers read
   `image_rays.opd/.x/.y` there. Add `TraceResult.at_exit_vertex(n_exit=None)` (returning a transferred ray bundle of
   the same type as `image_rays`, never mutating the original) implementing the signed transfer
   `t = −z/N; opd += n_exit·t; x += L·t; y += M·t; z = 0` on ALIVE rays only, with grazing rays (|N| below the
   tolerance `_transfer` already uses) killed with the existing error-code vocabulary rather than teleported.
   `n_exit` defaults to the index of the last surface's exit medium at the trace wavelength (find how `TraceResult`
   / the surfaces carry media and wavelength; if it cannot be inferred, require it and say so in the docstring).
   Also expose a functional form for callers that hold a bare bundle (e.g. `exit_vertex_transfer(bundle, n_exit)`),
   and a JAX-traceable equivalent in `jax_trace.py` for the `_lens_jax` consumers. Tests: exact against the analytic
   OPL on the vertex plane for a sphere and a conic (independent formula, not the tracer), a curved-REAR fixture
   (every existing consumer test uses plano-rear fixtures where sag ≡ 0 — that is why this class survived), alive
   masking, grazing handling, idempotence (applying twice is a no-op), JAX parity.
   Then replace the hand-written copies INSIDE `raytrace/` (`intersection.py:139`, `jax_trace.py:235`, and
   `ray_fan.py::refocus`'s sag-to-vertex logic) with the helper.
2. **R1** `opd_fan_data` / `opd_fan_data_world`: reference each ray to the image point (reference sphere), not to its own
   intercept — wrong sign and 3.1× at f/4 today. Oracle: the Seidel −S1/8 relation the auditor used and an
   independent exact sphere. Feeds `plot_opd_fan` PV/RMS.
3. **R2** off-axis fans seed `opd = 0` on the z = 0 launch plane: seed `opd = −(x·L + y·M)` (entrance eikonal) —
   1 880 waves of artefact at 5° today.
4. **R3** `seidel_coefficients` ignores `conic` and aspheric coefficients: add the Welford aspheric terms
   (`8·(n2−n1)·A4_eff·y⁴` with `A4_eff = k/(8R³) + A4`, validated 0.14–0.29 % vs real rays, exact for the parabola);
   check higher-order coefficient handling and document what is and is not included.
5. **R4** conic intersection seeds from the ray–SPHERE quadratic and uses ITS discriminant as the miss test → rays
   with h > |R| are killed on conics/parabolas. Use the exact conic quadratic (Spencer–Murty form; `differential._adrt_step`
   already holds it) as seed AND miss test in `intersection.py` and `jax_trace.py`; keep Newton for aspheres seeded
   from it. Orchestrator repro: `repro/orch/verify_rt_glass.py`.
6. **R5** the diffraction-order kick omits the medium index at four sites (`trace.py:215, :1172`, `world_trace.py:181`,
   `jax_trace.py:485`): ΔL = mλ/(n₂Λ).
7. **R6** `rays_from_field` aliasing above half Nyquist (symmetrised 1-px difference), edge rays half direction cosine,
   `_transfer` teleporting grazing rays with zero OPL and `alive=True` (kill instead), in-place arithmetic in the hot path.
8. **R7** the P3 bundle: `_refract` error-code overwrite vs its comment; `rays_from_field` seeding `opd` with WRAPPED
   phase; `trace_summary` omitting evanescent losses; import-time `GLASS_REGISTRY` mutation without the lock;
   `raytrace_system` mutating a caller's Surface; `layout.py` naming; `_adrt_jax` tracing twice; `field_of_view`'s
   finite-conjugate branch.

## Verification specifics
- Keep the things the audit verified CORRECT bit-identical or within documented tolerance: OPL to 1.4e-17 m vs the
  60-digit oracle, coordinate breaks exact inverses, `system_abcd`, spherical Seidel sums, JAX parity 7e-18 m,
  differential Jacobians. Re-run `repro/RAYTRACE/*.py` before/after.
- Every changed bar carries its derivation (TESTING_STANDARDS).
- Report the helper's exact signature, semantics and import path in a dedicated section so the next wave can adopt it
  without reading your diff.

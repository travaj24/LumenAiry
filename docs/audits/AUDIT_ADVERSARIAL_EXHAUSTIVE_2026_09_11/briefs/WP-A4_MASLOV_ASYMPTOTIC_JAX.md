# WP-A4 — Maslov / GBD / FGA lens propagators, `_lens_jax`, and the asymptotic (phase-space) family

Read first: `COMMON.md`, then the partition reports `MASLOV-GBD-FGA.md`, `ASYMPTOTIC.md`, `ORCHESTRATOR.md` (F-O4),
and report sections §2.5 (S2–S7), §9 (Y1–Y5), §15 in `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`.
Repro: `repro/MASLOV-GBD-FGA/`, `repro/ASYMPTOTIC/`, `repro/orch/maslov_exit_vertex_check*.py`.

## Files you own
`lumenairy/elements/lenses_maslov.py`, `elements/lenses_gbd.py`, `elements/fga.py`, `elements/_lens_jax.py`,
`lumenairy/propagators/gbd.py`, `propagators/asymptotic*.py` (all asymptotic modules), `propagators/subaperture.py`
(only if a finding needs it). Tests: the corresponding test files and new `tests/unit/test_audit2609_a4_*.py`.
NOT `_lens_traced*.py` / `_lens_imap.py` (WP-A3), NOT `raytrace/` (WP-A1 — its helper has landed).

## The exit-vertex helper (already landed by WP-A1)
`lumenairy/raytrace` now exposes `TraceResult.at_exit_vertex(...)` plus a JAX-traceable equivalent — read
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A1_REPORT.md` for the exact contract. Use it for S3 and
for the two hand-written copies in `_lens_jax.py` (old L556 / L829), bit-identically.

## Findings to implement
- **S3 (P1 ✔)** `apply_real_lens_maslov` reports the field on the curved exit SURFACE as the vertex-plane field
  (`lenses_maslov.py:1709–1735`: `exit_rays = tr.image_rays` at the sag; with `output_plane_distance≠0` the leg is
  `t = d/N` instead of `(d − z)/N`) — 31 waves of defocus on an f/19 singlet. Transfer with the helper before building
  `(s2, v2, OPD)`; verify with `repro/orch/maslov_exit_vertex_check2.py` (maslov − traced ρ² term → ~0 µm; flat-exit
  control unchanged). Same class in `asymptotic_canonical_fit.py:408–422, 915+` (the canonical map fitted on the sag).
- **S4 (P1 ✔)** the Maslov integrand uses |J| where Van Vleck requires √|J| (E = i·λ·z·E_exact on free space):
  fix the density and the prefactor; the sibling `propagate_hf_chebyshev_quadrature` IS correctly normalised — make them
  agree (Y2 is the same class in `asymptotic_maslov.py:277`, `asymptotic.py:646`, `asymptotic_aberration_tensor.py:230,
  :961`, `asymptotic_jax_twin.py:285, :421, :467`: `sqrt(detJ)` and a `−1j/λ` prefactor at all sites; downstream
  `jax_merits.py:471`'s Strehl deficit then reads sensibly — verify, do not edit optimize/).
- **S2 (P0)** `apply_real_lens_maslov(integration_method='auto')` routes real focusing charts to `local_quadrature`,
  whose window is laid out on the wrong axes and unwindowed (40 % floor, relerr 1.19 at defaults): fix
  `local_quadrature` (axes + window) if it can be made to converge on the auditor's charts, and route `'auto'` to
  `stationary_phase` (measured exact) or FGA where `local_quadrature` is not provably better; re-measure the auditor's
  convergence ladder.
- **S5 (P1)** GBD carries the Gouy phase conjugated and ships a public API to compensate: fix the sign, deprecate the
  compensator (via `_deprecation.py` patterns), verify against an analytic Gaussian through focus.
- **S6** — read the row; implement.
- **S7 (P1)** `_lens_jax` runs float32 unless the caller enabled x64 (10⁶× OPD accuracy loss) and cannot be `jit`-ed on
  its default path: adopt the `rcwa/_core.py` policy (`_require_jax_x64` — raise with a message) and make the default
  path traceable (no concretising `float(...)`/Python control flow on traced values); JAX↔NumPy parity test.
- **Y1 (P0)** `extract_linear_phase=True` (DEFAULT) drops `a3·u3 + a4·u4` — linear in the integration variable — from the
  integrand: keep them inside the integrand as `lenses_maslov.py:821–825, :2914` already do; rewrite the W6 pins that
  assert a3 ≈ 0 as a premise; verify with `repro/ASYMPTOTIC/t4, t6, t8, t9` (off-axis PSF back on the chief ray, amplitude
  within 2.2e-5 of the reference).
- **Y3 (P1)** `jax.grad` through `aberration_tensor_lg00_jax` ~86 % wrong for anything rotating Re M (degenerate
  `eigvalsh` JVP): closed-form 2×2 λ_max or `stop_gradient`; port the four NumPy guards into the JAX twins as `jnp.where`
  (no non-finite outputs); FD-vs-AD test at the 1e-4 gate on a rotating fixture.
- **Y4 (P2)** `propagate_modal_asymptotic` silent zeroing → warn/return diagnostics; one fused Chebyshev basis build
  (2.3–5.1×, +1.4× hoisting `T12`); per-pixel temporaries; `aberration_tensor` default cost vs its own aliasing warning
  (`sigma_grid_n` cap 256 vs requested 494); scale-relative Newton tolerance.
- **Y5 (P3)** `pupil_modes` inert; `A_lead` overflow guard; the Seidel/Zernike framing over-claim; "exact at the stationary
  point" wording; the cross-backend `w_o` contract; import-time monkey-patching; triplicated `_compute_M_b`/Newton kernels.
- **New feature (§15.9):** wire the already-present `uniform_fold_airy` / `pearcey` uniform asymptotics into the saddle
  integrators (they are dead code today) behind an explicit option, gated against a brute-force quadrature at a fold;
  if the Pearcey path cannot be validated in this pass, leave it dead and say so.

## Verification specifics
- Keep intact (re-check): Wick moments 9e-14, LG/HG orthonormality 3e-13, saddle-point algebra vs brute force 7e-6, the
  W6-A1 branch claim, `propagate_hf_chebyshev_quadrature` absolute correctness, NumPy/JAX parity 8e-11 inside the box,
  x64 enforcement in the asymptotic family, FGA exact on free space with the correct Gouy phase.
- Every fix measured before/after with the repro scripts; every bar derived.

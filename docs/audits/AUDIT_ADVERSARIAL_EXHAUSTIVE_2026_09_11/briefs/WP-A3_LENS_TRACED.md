# WP-A3 — `apply_real_lens_traced` and its siblings (`_lens_traced.py`, `_lens_imap.py`, multibranch, uniform, segmented)

Read first: `COMMON.md`, then the partition reports `TR-INFRA.md`, `TR-MAIN-1.md`, `TR-MAIN-2.md`, `TR-SIBLINGS.md`,
`ORCHESTRATOR.md` (the TR-MAIN-2 cross-verification and the exit-vertex census), and report sections §2.2 (T1–T11),
§2.3 (T12–T16), §2.5 (S1, S8–S11 — the sibling rows), §14 (V3's D3/D4/D5 analysis) and §15 in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`. Repro: `repro/TR-INFRA/`, `repro/TR-MAIN-1/`,
`repro/TR-MAIN-2/`, `repro/TR-SIBLINGS/`, `repro/orch/verify_trmain2.py`, `repro/orch/verify_trinfra.py`.

## Files you own
`lumenairy/elements/_lens_traced.py`, `_lens_imap.py`, `_lens_traced_multibranch.py`, `_lens_traced_uniform.py`,
any `_lens_traced_*.py` sibling (segmented / multi live inside `_lens_traced.py` — confirm), `_traced_flags.py`,
`lumenairy/_math/chebyshev.py` (T11 consolidation target). Tests: every traced-lens test file and new
`tests/unit/test_audit2609_a3_*.py`. NOT `_lens_real.py`, `lenses.py` (WP-A2), NOT `_lens_jax.py`, `lenses_maslov.py`
(WP-A4), NOT `raytrace/` (WP-A1 — its helper has landed; see below), NOT `propagators/carrier*.py` (WP-A6).

## The exit-vertex helper (already landed by WP-A1)
`lumenairy/raytrace` now exposes `TraceResult.at_exit_vertex(...)` (read WP-A1's report at
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A1_REPORT.md` for the exact signature and semantics).
Replace the three hand-written transfer copies in `_lens_traced.py` (around the old L6679 / L9583 / L11107 sites) with
it, bit-identically (prove with the traced fixtures), and use it to fix **S1**.

## Findings to implement
- **S1 (P0)** `caustic='multibranch'` and `'uniform'` evaluate on the last surface's SAG, not the exit plane
  (`_lens_traced_multibranch.py:172–182`; `_lens_traced_uniform.py:225, :589` use `t = output_plane_distance/Nz`,
  ignoring `ex.z`) — 3 501 waves of OPD error and 477 µm transverse error on a curved-rear singlet; every existing test
  fixture is plano-rear. Fix with the helper (`t = (d − z)/N`), add curved-rear fixtures, compare against the
  TR-SIBLINGS wave oracle (its 25-line Kirchhoff/ASM hand-off) and against the default traced path at d = 0.
- **T1 (P1 ✔)** real-dtype `E_in` crashes at four sites (`np.dtype(...)` vs a scalar type) — fix, test real float32/64 in.
- **T2 (P1 ✔)** `caustic='multibranch'` at/near focus returns an all-zero field silently: count skipped triangles and
  refuse (or route to the wave hand-off below), two-sided energy tripwire (4–8× too high in the band before focus).
- **T3 (P1 ✔)** `newton_fit='spline'` + one vignetted ray → all-zero field (NaN poisons FITPACK): refuse when
  `not alive.all()` or fill by nearest-finite; fix the false guard comment; the launch square's corners at 1.06× the
  aperture.
- **T4 (P1 ✔)** `fast_analytic_phase=True` → `AttributeError` (`raytrace._surface_sag_xy` not exported): import from
  `raytrace.surface`; add a REFRACTING test; fix the NaN leak outside the conic domain and the "under 10 nm" claim.
- **T5 (P1 ✔)** `form_error` silently cancelled (250–800×) — add it to `delta_phase` explicitly (both legs consistent)
  or refuse with a message; same for every phase-only feature the ray model lacks (enumerate them in the docstring).
- **T6 (P1)** `_reverse_prescription` must negate every sag term (aspheric, freeform, `tilt`, `sag_callable`), reverse
  the thickness pairing correctly under `len(thicknesses)==len(surfaces)`, carry `stop_index`/`elements`.
- **T7 (P2)** `_sample_local_tilts` wrap/half-pixel bias (midpoint construction like the siblings); `_compute_carrier`
  ndarray rounding (`np.rint`).
- **T8 (P2)** `apply_real_lens_traced_multi(reuse_prepared=True)` accepted-kwarg set from `inspect.signature`;
  `apply_real_lens_traced_segmented` square-pixel refusal / `dy` forwarding.
- **T9 (P2, memory)** chunk the `inversion_method='fit'` design matrix; `np.copyto(where=)` for the final masking;
  `vdot` instead of complex128 upcasts in `_ray_density_self_checks`; 1-D spectral windows; chunk the pure-NumPy
  Chebyshev fallback. tracemalloc before/after at N = 1024 (and extrapolated).
- **T10 (P2)** block the numba Chebyshev kernel's per-sample allocations (bit-identical, 1.63×); score the
  least-squares refinement over the evaluation domain; strengthen `_script_has_main_guard`.
- **T11 (P3)** three Chebyshev Vandermonde copies → `_math/chebyshev.py`; the `_geometric_lens_phase` and
  `_opl_by_backward_trace` claims; `_spectral_gap_cuts` docstring; `PreparedTracedLens` "same 8 kwargs" comment → a test.
- **T12–T16** (§2.3): read the rows (carrier=<ndarray> handling T12, …, the `np.abs(E_analytic[::sub, ::sub])` /
  cached `phase_analytic_lens` perf item T15, T16 P3) and implement.
- **S8 (P2)** `build_inverse_map` cache key must include the flags that change the arithmetic; the imap GRAM guard must
  not be priced against live free RAM (environment must not change physics).
- **S9 (P2, perf)** the multibranch rasteriser at the default `output_plane_distance=0` (144 s vs 0.85 s, 12.2 GB RSS
  → budget) and the uniform Airy dark fill restricted to r_c + 20 l_airy (25.4 s → ~1.5 s at N = 2048); measure.
- **S10, S11** — read the rows; implement.
- **New feature — wave hand-off for caustic planes (§15.9, first bullet).** Add a `caustic` mode (propose the name, e.g.
  `caustic='wave'`) that takes the traced exit-vertex-plane field and propagates it to `output_plane_distance` with the
  library's ASM (band-limited), instead of enumerating ray branches: exact through folds, cusps and the axial focus, and
  measured by TR-SIBLINGS to be 2× faster than multibranch and 15× faster than uniform at N = 4096 while reproducing the
  dark-side tail multibranch drops. Gate it against the multibranch/uniform paths away from caustics and against the
  independent Kirchhoff oracle at a fold. Consider making it the default for `output_plane_distance ≠ 0` only if every
  existing multibranch/uniform test still passes with a documented tolerance — otherwise keep the old default and
  recommend it in the docstrings.

## Verification specifics
- Keep intact (re-check): the final phase assembly wrapping in complex128, masked pixels exactly zero, energy
  1.000000 / 0.999934, multi coherent-sum piston 0.0004 rad, spectral partition of unity 2.2e-16, `PreparedTracedLens`
  factorisation exact, v5.44 banded assembly byte-identical across 7 configurations, Newton inversion 1e-15 m,
  deterministic solve byte-identical across thread counts, `_compute_carrier('auto')` exact on spheres.
- Add the covering fixtures the audit says are missing: curved rear surface + non-collimated input for every opt-in
  kwarg you touch (§14 V3: D3/D4/D5 — real-dtype input, multibranch AT the paraxial focus asserting on the FIELD, spline
  with a vignetted ray).

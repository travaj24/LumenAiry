# WP-A13 — PMM 2-D: hybrid, staggered, stacks, JAX twins

Read first: `COMMON.md`, then the partition report `PMM-2D.md` (all of it) and report sections §12 (G5–G13), §15 in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`. Repro: `repro/PMM-2D/`.

## Files you own
`lumenairy/elements/pmm/twod.py`, `twod_jones.py`, `stack2d.py`, `_stack2d_cache.py`, `_jax_twod.py`,
`_jax_twod_jones.py`, `_jax_stack2d.py`, `twod_staggered.py`, `stack2d_pure.py`. Tests: the PMM 2-D test files and new
`tests/unit/test_audit2609_a13_*.py`. NOT the 1-D modules / `pmm/_core.py` (WP-A12 — if `_core.py` needs a change,
request it), NOT `rcwa/_core.py` (WP-A14 owns `_sqrt_decay`, finding G11).

## Findings to implement
- **G5 (P1)** `PMM2DStackHybrid` never routes the per-slot Li operators: pass `EpnxF`/`EpnyF` in `_build_layer_modes`
  and `_symmetric_layer_specs` (two lines mirroring `twod.py:889–898`); gate with the two-orientation equality test
  (x-patterned vs y-patterned grating, T00 and the Jones equal to 1e-13) for `li` at several `n_orders`, `symmetry`
  auto/False.
- **G6 (P2)** `pmm_jones_2d(formulation='li')` docstring → the measurement (worse phase than `laurent`, `fff_nv` best on
  separable cells); default to `fff_nv` on separable in-plane cells if all existing tests still pass with documented
  tolerances (otherwise keep the default and document); wire the even-parity fold for `fff_nv` (it is only reachable on
  separable cells, where the fold's precondition holds).
- **G7 (P2)** tighten `_PASSIVE_TOL_2D` toward the measured clean floor or add an advisory threshold, AND add the
  consecutive-`n_orders` T00 drift check (`stabilize=True` already computes it) because closure is anti-correlated with
  the per-order error on the auditor's fixture; point `pmm_jones_2d`'s docstring at the no-floor staggered engine.
- **G8 (P2)** `PreparedPMM2D.solve` must call `_warn_lossless_energy_2d`, thread `symmetry` and `truncation` through
  `prepare_pmm_2d*` and the `*_vs_wavelength` helpers, and its docstring's "only delta" sentence must be true (measure).
- **G9 (P2)** the staggered family's `eps_cell` cost cliff: add a cost guard on `2·(Nx(M−1))²` naming the MERGED strip
  count (reuse `_cell_to_walls_tile`'s logic), and cross-reference PIXEL vs SEGMENT semantics in both families' docstrings.
- **G10 (P2, perf)** JAX twins: drop the dense Kronecker projector pair and express `_proj` as the two per-axis einsum
  contractions of `_sandwich_factorized`; no dense N×N `diag(1/Mdiag)`; diagonal GLL masses → `1/np.diag(M)` on the
  separable branches; `cascade='fused'` as the default (bit-identical to `'fast'`/`'tree'` — prove); cache the k0-free
  projected TENSOR operators the way scalar layers already are; extend circular truncation to `pmm_jones_2d`,
  `PMM2DStackHybrid` and the sweeps; surface `cache_stats()['eig']['refused']` when non-zero.
- **G12 (P3)** add `symmetry` to `_mode_key`; make `period_x`/`period_y` read-only after the first `add_layer` (or store
  walls as fractions) — test both stale-cache scenarios from the report.
- **G13 (P3)** `return_jones_transmission=True` on `pmm_jones_2d` (the amplitudes exist at `twod_jones.py:804`) with the
  waveplate retardance test from the report (Λ/λ = 0.2 grating: hybrid at n_orders 15 within 0.2° of the
  `rcwa_jones_1d` reference +100.066°; slow-axis `exp(+iδ)` convention; S₃ = −1 loop); a seam note pointing at the
  stack's `jones_transmission()`; `_assemble_2d` "reference implementation" note; delete the dead `noqa` re-exports.
- **G11** belongs to WP-A14 (rcwa/_core.py) — do not edit; but add a PMM-2D-side test that a near-zero evanescent
  `lam²` is not flipped once WP-A14's predicate lands (mark it to skip-with-reason if the predicate is still old? NO —
  per TESTING_STANDARDS never skip on environment; instead assert the CURRENT contract and note the dependency in your
  report).

## Verification specifics
- Keep intact (re-check): Jones basis vs TMM 1e-16, 90° covariance of `pmm_jones_2d` 5e-14, mirror/C4/reciprocity/
  inversion symmetries, even-parity fold 1.75e-13, `fff_nv` raises and its separable reduction, cache keys, JAX x64
  enforcement / parity / AD-vs-FD, frame-anchor gating, slant refusals, the staggered engine's n_orders invariance.
- The staggered engine's M = 10 solve takes ~12 min — do not run it repeatedly; use M ≤ 8 in tests with derived bars.

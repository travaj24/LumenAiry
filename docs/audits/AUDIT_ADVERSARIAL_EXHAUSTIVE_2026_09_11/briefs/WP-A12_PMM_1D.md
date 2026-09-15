# WP-A12 — PMM 1-D / stack / core (`elements/pmm/_core.py`, `stack.py`, `oned.py`, `conical.py`, `_jax_stack.py`) + CONVENTIONS §7.1

Read first: `COMMON.md`, then the partition report `PMM-1D.md` (all of it, including its provenance notes on the two
instrumentation bugs and the three failed timing probes) and report sections §11 (G1–G4), §15 in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`. Repro: `repro/PMM-1D/`.

## Files you own
`lumenairy/elements/pmm/_core.py`, `stack.py`, `oned.py`, `conical.py`, `_jax_stack.py`, `pmm/__init__.py`; and
`CONVENTIONS.md` §7.1 ONLY (the Jones-basis wording — coordinate the exact sentence with the measured facts in PMM-1D.md
and PMM-2D.md: both 1-D Jones solvers return the lab Cartesian (E_x, E_y) basis, index 0 = x ≡ tm, index 1 = y ≡ te,
with J_xx = −r_p in the Fresnel p convention at φ = 0). Tests: the PMM 1-D test files and new
`tests/unit/test_audit2609_a12_*.py`. NOT the 2-D modules (WP-A13), NOT `rcwa/` (WP-A14).

## Findings to implement
- **G1 (P1)** the differentiable `PMMStack.solve` twin returns before every NumPy-branch guard: hoist the concrete
  incidence guard (`_jpmm_concrete_incidence_guard`) above the traced dispatch; run the PURE-GEOMETRY sliver screen
  before the dispatch (it needs no traced value); after the twin returns run `_warn_stack_energy` when outputs are
  concrete (eager call). Reproduce the two silent failures (gain superstrate → negative efficiencies; manufactured
  sliver → ΣR+ΣT = 8.35) before, refusal/warning after; JAX parity and `grad` unchanged where they were correct.
- **G2 (P2)** the default `min_feature = period·1e-5` sits one decade below the measured hazard band: raise the default
  to `period·1e-3` (the auditor measured the cure: 0 of 11 rungs scatter at 1e-3·P, and cases the snap does not touch
  are bit-identical) — document the change and the collision-scale rule; keep the parameter.
- **G3 (P2)** memoise the sliver-arbiter verdict on its geometric key (walls, `min_feature`, `degree`, `wl`, `angle`) so a
  sweep pays it once, and make the two collapse solves lazy; carry the exactly-diagonal SEM masses as 1-D diagonals
  (`_safe_inv`/`_safe_solve` on diagonals → `1/d`; `A @ diag(v)` → `A * v`) — bit-identical A/B; compute `Q @ W2` once;
  key `_GEO_EIG_CACHE` on a blake2b digest instead of `B.tobytes()` and enrol it in `ByteBudgetedLRU`; `np.add.at` in
  `_sem_fourier_projection`; if feasible, the dimensionless-kx0 pencil so oblique wavelength sweeps reuse the eig.
- **G4 (P3)** correct the "eig is ~85 % of runtime" note (call counts / flop model; say the empirical split is
  unmeasured), the `elements_per_region>1, grade=True` "speed lever" claim (matched-DOF pessimisation), add the TM
  convergence caveat to `pmm_jones_1d` and `PMMStack`, `internal_field(pol=…)` accepting `'te'/'tm'/'s'/'p'`, raise on
  `angle=A, theta=T` with A ≠ T, extract the ~15 copies of the far-field order-budget block into one
  `_farfield_order_set(...)` helper (bit-identical gate over the existing tests), and record the slanted-TM super-unity
  floor (4–9e-5, decreasing with degree) in the slant solver's docstring.
- **Alternative algorithms (PMM-1D §"Alternative algorithms", optional after the above):** a Gegenbauer/ultraspherical
  basis option for the TM wall-corner singularity, or a genuine hp mesh (geometric grading with decreasing degree) — only
  if you can demonstrate the convergence gain on the auditor's Au/air TM fixture; otherwise document as deferred.

## Verification specifics
- Keep intact (re-check with the repro scripts): mode spectrum vs Botten roots ≤ 3.8e-13, energy ≤ 3e-12, reciprocity
  2e-11, unpatterned TMM parity 4e-14 in amplitude and phase, TE/TM labelling, S-matrix stability, JAX parity 6e-14 and
  `grad` 5.8e-8, conical limits, mortar identity, the round-4 sliver guard's 0 false negatives, cache key completeness,
  unit invariance over 13 decades, `layer_absorption` closure.
- Count solves, don't time them (the machine is shared); bit-identity for every "perf" change.

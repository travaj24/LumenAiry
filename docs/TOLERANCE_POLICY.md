# Test-tolerance policy (BLAS-gauge contracts) — 2026-06-10

Four runner-flavored test re-pins were needed in the v5.14.1 cycle alone.
Every one traced to the same physics: **the gauge inside a degenerate
eigen-pair is not specified by the math**, so LAPACK builds (MKL vs OpenBLAS
vs whatever microarchitecture a CI runner has) legitimately return different
mixtures, and any contract tighter than the physics is build-flavored
flakiness waiting to happen. This document is the decision rule; the named
constants live in `tests/unit/_tolerances.py`.

## Decision rule

Ask: *does the quantity under test pass through an eigendecomposition with
(near-)degenerate eigenvalues, or through `lstsq`/`solve` on a
(near-)singular system?*

1. **No (pure arithmetic / FFT / quadrature / a single well-conditioned
   solve):** bit-identity (`np.array_equal`) is allowed and preferred —
   e.g. builder refactors that must reproduce a legacy construction,
   prepared-vs-rebuild caches, sweep-vs-per-λ loops of the SAME code path.
2. **Yes, but the comparison is within ONE code path on ONE build** (same
   call repeated, cache-vs-direct of identical inputs): bit-identity is
   still fine — determinism within a build is guaranteed.
3. **Yes, and the comparison CROSSES code paths** (stack vs single-layer,
   generalized vs symmetric cascade, N-block vs 2N-block, scalar vs tensor
   promotion) **or will run on more than one build** (CI): use a PHYSICAL
   tolerance from `_tolerances.py`. Never bit-identity, never 1e-12.
4. **The comparison sits AT a degenerate limit point** (slant→0, off-plane→0,
   normal incidence on a C4 cell): the gauge is maximally free; use the
   LIMIT-class tolerance and document which limit.

## Named tolerance classes (tests/unit/_tolerances.py)

| constant | value | use |
|---|---|---|
| `GAUGE_CROSS_PATH` | 5e-6 | cross-path efficiency/Jones agreement through degenerate eigs (stack↔segments, promotion↔direct) |
| `GAUGE_AT_LIMIT` | 1e-4 | contracts evaluated AT a degenerate limit point (vanishing slant/off-plane) |
| `LOSSLESS_CLOSURE` | 1e-9 | R+T−1 on a provably-lossless single solve, clean geometry |
| `STAIRCASE_CLOSURE` | 1e-7 | R+T−1 through many-layer staircases (accumulated star roundoff; one CI runner measured 4.5e-8 on 8 slices) |
| `CONSENSUS_PER_ORDER` | 2e-4 | a stabilized answer vs the adjacent-truncation consensus |
| `CROSS_FAMILY_DIELECTRIC` | 2e-3 | PMM↔RCWA on converged dielectric problems |
| `CROSS_FAMILY_METAL` | 2e-2 | PMM↔RCWA on metal problems at matched practical truncation (document the convergence study if tighter) |

## Build-portable patterns (use instead of fixed (P, M) pins)

- **WHICH truncations are unstable is build-dependent.** Never pin "M=20
  blows up" or "M=11 is clean": probe a window, classify on the running
  build, then assert the *contract* (raise-or-clean-closure; stabilize
  returns consensus). Precedents: `test_large_period_energy_blowup_guarded`,
  `test_stabilize_is_bit_exact_noop_on_clean_geometry`,
  `test_silent_window_warns_and_stabilize_recovers`.
- **One realization is not the geometry.** Different staircase banding of
  the same trapezoid converges to one limit but differs at finite n_slices;
  pin references per realization (application feedback, 2026-06-10).
- When a bit-identity contract must be RETIRED (a degenerate-gauge path was
  made noise-robust), drop to `GAUGE_CROSS_PATH` and say why in the test
  docstring — the v5.14 dense-resonance fix is the template.

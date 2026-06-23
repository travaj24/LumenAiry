# BOR-PMM (axisymmetric / Body-of-Revolution PMM) — Phase 0 prototype

Standalone de-risking prototype for the cylindrical-coordinate PMM solver
(`docs/pmm_bor_axisymmetric_roadmap.md`, see the **ASSESSMENT + GROUNDED
FORMULATION + VALIDATED PHASE 0** section). Outside the package by design (the
roadmap's Phase-0 gate), validated against exact analytic oracles before any
library integration.

## Status (2026-06-19)

| milestone | what | oracle | status |
|---|---|---|---|
| **M1** | radial spectral-element eigensolve: cylindrical metric (`1/r`, `m²/r²`) + `r=0` axis BC | Bessel zeros `j_{m,n}` (TM) / `j'_{m,n}` (TE) + eigenfunctions `J_m(γr)` | **DONE — ~1e-13**, spectrally convergent |
| **M2 oracle** | open-cladding (`K_m`) fiber dispersion — the CLEAN guided-mode gate (`fiber_oracle.py`) | canonical Okamoto/Snyder–Love vector char. eq. | **DONE — exact match**; caught a cross-`eps` bug in the old PEC `stepindex_oracle` |
| **M2 solver** | coupled `(E_r,E_phi)` vector eigensolve: q² E_z-elimination + wall-normal inverse rule + consistent divergence-free filter (`coupled_radial_eigensolver.py`) | the fiber oracle above | **DONE — guided modes ~1e-4..1e-2 (FD floor), no spurious leakage**; 4 tests. Weakly-guided/large-box → SEM follow-on |
| **M3** | open radial boundary — radial PML (complex coordinate stretch `s=1+i sigma`), the roadmap's #1 risk | bound-mode q INVARIANT to `sigma_max`; radiation continuum absorbed (complex q) | **DONE** — q invariant to 1e-6 across `sigma_max=3..20`; bound mode = oracle (8.5e-5); continuum absorbed; 2 tests |
| **M4** | z-cascade S-matrix (`zcascade.py`): modal `W/V` + Redheffer; `r dr` only in flux/energy (interface is pointwise) | same-medium identity, slab Airy, energy `R+T=1` | **DONE** — GATE 0 `5e-11`, round-trip `1.7e-10`, slab `3e-10`, energy `1e-6`; 5 tests. Clean half-spaces + Cartesian limit → M5 |
| **M5a** | clean half-spaces — PEC/Dirichlet wall (`wall='pec'`): antisymmetric ghost → real-q box modes (`farfield.py` + `zcascade.py`) | same-medium identity, multi-mode Fresnel | **DONE** — identity `3e-12`, ≥8 (vs 1) propagating modes mean `2e-3` |
| **M5b far-field core** | Fourier-Bessel / Hankel decomposition → cylindrical orders + Parseval power (`farfield.py`) | round-trip, Parseval, kt→θ | **DONE** — Parseval `1.8e-10`; 3 tests |
| **CURE** | **Yee div-conforming grid** (`staggered=True`): E_r on faces, E_phi/E_z on nodes → de Rham `curl·grad=0` kills the spurious sea | spurious 365→0; structured cascade energy machine-precision, N-stable | **DONE** — `4e-13`@N=150…`2e-12`@N=450 (nodal floor was `3.8e-2`); 4 tests |
| M5 (rest) | structured-layer efficiencies (±1 vortex Hankel); full multi-order GATE 4 vs `pmm_efficiency_1d`; `BORStack` library port | per-order η vs planar; reciprocity | UNBLOCKED by the cure (machine-precision basis); pending |

M1 closes the roadmap's stated #2 risk (the axis singularity) to machine
precision. M2's analytic oracle is ready as the gate for the coupled operator —
the genuinely-hard, energy-invisible inverse-rule piece the critique requires be
validated per-quantity (not by energy conservation; cf. the "lossless trap").

## Files

- `radial_eigensolver.py` — the validated M1 radial spectral-element eigensolver.
- `stepindex_oracle.py` — the validated M2 analytic dispersion oracle (`6×6`
  Bessel boundary matching).
- `test_radial_eigensolver.py` — 11 tests (run from this directory:
  `pytest test_radial_eigensolver.py`).

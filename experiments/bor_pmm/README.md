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
| M3 | open radial boundary (PML / Hankel matching) — the #1 risk | bare half-space reflection ≈ 0 | pending |
| M4 | z-cascade + scattering (`r dr` flux split) | radially-uniform → Fresnel | pending |
| M5 | far-field + public API + integration | energy/m, reciprocity, Cartesian limit, FDTD-BOR | pending |

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

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
| **M2 oracle** | analytic step-index / coaxial dispersion (the per-quantity gate for the coupled `E_r` inverse-rule) | homogeneous reduction → TM+TE spectrum | **BUILT + validated ~1e-9** |
| M2 solver | coupled `m=1` vector eigensolve at one ring wall (TE↔TM coupling + `E_r` inverse rule + `r dr` W/V-orthonormality) | the M2 oracle above | next |
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

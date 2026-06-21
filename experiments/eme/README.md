# EME lateral cascade — 2-D Bloch layer modes from 1-D building blocks

**Status:** scalar layer-mode solver — validated + tested (prototype, `experiments/`).

This is the *"stack 1-D PMM solvers laterally to make a 2-D surface"* idea, done
rigorously. A doubly-periodic layer `eps(x, y)` (period `Lx × Ly`) that is
**piecewise uniform in y** — a stack of y-strips, each varying only in x — has
its full 2-D Bloch modes built **without a 2-D eigensolve**:

1. In each y-strip the field separates into that strip's **1-D-x eigenmodes**
   (`strip_x_modes`), each propagating laterally in y with `ky = sqrt(lam - qz^2)`.
2. The strips are joined by a **lateral S-matrix cascade** (Redheffer) carrying
   Bloch periodicity in y (`cell_smatrix`).
3. The 2-D modes are the `qz^2` at which the unit-cell Bloch condition is
   singular — found by a real-axis root-find on `sigma_min(M(qz^2))`
   (`dispersion` / `layer_modes`).

## Why it is not trivial (two failed first attempts, now cured)

| Pitfall | Symptom | Cure |
|---|---|---|
| **Transfer matrices** carry `exp(+|ky| h)` for evanescent strip modes | lateral cascade blows up ~1e5 vs analytic | **Redheffer S-matrices** — only ever propagate *decaying* exponentials |
| **Eigenvalue tracking** of the Bloch multiplier `mu → target` | modes sit at per-strip band edges where `mu` vanishes from the spectrum (branch transition) → missed | **`det(M(qz^2)) = 0`** via `sigma_min(M)`, which crosses zero *cleanly* through the band edge — no contour (Beyn) method needed |
| **Single-basis** mode matrix `M` lives in the *first* strip's basis | a mode deeply evanescent in that strip has sub-precision signature → not found (`sigma_min ≈ 0.98` at a real mode) | **multi-basis search** — rotate the strip list to start the cell in each strip, union the modes (each basis sees the modes that propagate in *its* strip), dedup by `merge_rtol` |

`M = -S12 t^2 + (I + S12 S21 - S22 S11) t - S21`, with `t = exp(i ky0 Ly)` — the
Bloch quadratic-eigenvalue residual of the cell S-matrix.

## Validation (`test_eme_2d.py`, 4 tests passing)

Validated against a **direct 2-D finite-difference eigensolve** (`ref_2d_modes`):

- **`test_uniform_matches_analytic`** — uniform layer → analytic
  `eps k0^2 - kx^2 - ky^2` (to x-FD accuracy).
- **`test_structured_2strip_converges_from_2dfd`** — the load-bearing test. The
  EME is **analytic in y**; the 2-D-FD is finite-difference in y, so the 2-D-FD
  **converges to the EME** as `Ny → ∞` (monotone, 2nd-order; converged residual
  `< 0.05`). The EME *is* the exact-y limit.
- **`test_high_contrast_pillar_multibasis`** — eps=6 pillar in eps=1 host
  (`[H, G, H]`), deeply-localized modes. The multi-basis search finds all 6 modes;
  every one matches the 2-D-FD (`< 0.5`, the y-FD error is ~0.1). Without
  multi-basis the localized modes are missed.
- **`test_no_duplicate_modes`** — modes surfaced in several strip bases are
  deduped; no two reported modes coincide.

Run: `cd experiments/eme && python -m pytest test_eme_2d.py` (≈2 min — each test
builds a converged 2-D-FD oracle).

## Scope and the path to a full surface solver

This module solves the **scalar Helmholtz layer modes** — the hard, novel core
(stable lateral cascade + robust band-edge mode-finding). It is *not yet* a full
diffraction-efficiency solver. Two documented extensions reuse these modes as the
layer basis:

1. **Diffraction (z-cascade + far-field).** Sandwich the structured layer between
   plane-wave half-spaces: project the EME modes `psi(x, y)` onto the 2-D Rayleigh
   plane-wave basis (overlap integrals → interface S-matrix), propagate
   `exp(i qz d)` through depth, cascade in z (reuse the same Redheffer star), and
   read the transmitted/reflected plane-wave amplitudes as diffraction
   efficiencies. This makes it directly comparable to `pmm_efficiency_2d` /
   `rcwa_efficiency_2d`.
2. **Vector (TE/TM).** The scalar strip field becomes a 2-component
   (Ez-/Hz-polarized) cross-section; the strip eigenproblem and the lateral
   interface conditions double in size. Required for an anisotropic / full-Jones
   surface (parity with `pmm_jones_2d` / `rcwa_jones_2d`).

## Known v1 limitation

Exactly **degenerate** modes (e.g. a 4-strip checkerboard with a symmetry-paired
`qz^2`) can cluster or merge under `merge_rtol`. A symmetry-aware multiplicity
count (or a small `ky0` perturbation to split the pair) is the fix — deferred.

## Files

- `eme_2d.py` — the solver (`strip_x_modes`, `cell_smatrix`, `dispersion`,
  `layer_modes`) + the FD oracle (`ref_2d_modes`, `strips_to_eps_xy`).
- `test_eme_2d.py` — the 4-test validation suite.

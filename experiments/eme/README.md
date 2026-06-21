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
| **Transfer matrices** carry `exp(+\|ky\| h)` for evanescent strip modes | lateral cascade blows up ~1e5 vs analytic | **Redheffer S-matrices** — only ever propagate *decaying* exponentials |
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

## Mode field reconstruction (`mode_field`)

`mode_field(strip_modes, qz2, ky0, Ly, Ny)` returns the mode's transverse field
`psi(x, y)` — the eigenvector, not just `qz²`. At a found mode the per-strip
modal amplitudes are the null vector of the **global lateral interface system**
(`_global_lateral_nullspace`): the strip-interface + Bloch-wrap conditions
assembled as one block matrix `G(qz²)`. Stable — each block carries only a
*single* strip's `exp(±i ky h)`, never the accumulated transfer-matrix product.
Validated (`test_mode_field_matches_2dfd_eigenvector`): the reconstructed field
overlaps the 2-D-FD eigenvector to >0.999 for non-degenerate modes.

## Diffraction efficiencies — a documented dead end (`eme_diffraction.py`)

The natural next step — turn the modes into a diffraction *surface* solver by
mode-matching to Rayleigh half-spaces — **does not work with these modes**, and
that is a real finding, not a bug:

- **The mode-matching math is exact.** A uniform layer reproduces the analytic
  scalar slab (Airy/Fabry-Pérot), energy `R+T=1` (machine precision at normal
  incidence; FD accuracy at oblique). `mode_match` is validated
  (`test_eme_diffraction.py`).
- **Structured layers do not converge.** Efficiencies wander (e.g. `T_00` swinging
  0.27 to 0.48) and energy strays from 1 as the order/mode count grows, for *every*
  mode selection tried (highest `qz²`; largest order-overlap; complete coarse
  grid). **Why:** a convergent modal grating method (Botten / classical lamellar)
  computes the layer modes *in the truncated N_pw-order Fourier space*, where
  there are exactly `N_pw` of them and they span it. The EME computes modes at
  full *real-space* resolution; truncating that set to `N_pw` is not a basis of
  the retained-order space. Efficiency truncation is a Fourier-space notion — the
  EME modes are real-space.
- **Use the right tool for efficiencies:** `lumenairy.rcwa_efficiency_2d`
  (Fourier-space modes) or `pmm_efficiency_2d`. The EME's niche is **modes / band
  structure**, not diffraction efficiencies.

(Note `rcwa_efficiency_2d` solves the full *vector* problem; this scalar model
reduces to it only in the y-uniform → 1-D-TE limit, not for a true 2-D crossed
grating.)

## The remaining vector extension (open)

A vector (TE/TM) layer-mode solver would make the scalar strip field a
2-component cross-section; the strip eigenproblem and lateral interface
conditions double in size. That extends the **mode** solver (parity with the mode
content of `pmm_jones_2d`), and is independent of the diffraction finding above.

## Known v1 limitation

Exactly **degenerate** modes (e.g. a 4-strip checkerboard with a symmetry-paired
`qz^2`) can cluster or merge under `merge_rtol`. A symmetry-aware multiplicity
count (or a small `ky0` perturbation to split the pair) is the fix — deferred.

## Files

- `eme_2d.py` — the solver (`strip_x_modes`, `cell_smatrix`, `dispersion`,
  `layer_modes`) + the FD oracle (`ref_2d_modes`, `strips_to_eps_xy`).
- `test_eme_2d.py` — the 4-test validation suite.

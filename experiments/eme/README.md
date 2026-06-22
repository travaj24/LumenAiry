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

## Vector (TE/TM) layer-mode solver (`eme_2d_vector.py`)

The full-Maxwell generalization of the scalar solver — the **vector** 2-D Bloch
modes (propagation constant `qz` along z, both polarizations) of a y-strip layer,
from 1-D-x vector strip modes.

- **Strip modes** (`strip_vector_modes`): each strip's 1-D-x **Berreman-in-y**
  4-field generator on the y-tangential state `[Ex, Ez, Hx, Hz](x)`, eigenvalue
  `ky`. Yee-staggered x. The operator is **qz-dependent** (conical TE/TM coupling
  `~ qz·dε/dx`); the ky² spectrum shifts rigidly `ky²(qz)=ky²(0)−qz²` but the
  eigenvectors rotate, so it's rebuilt per qz². **Perf:** the 4Nx generator is
  block-anti-diagonal `[[0,B],[C,0]]`, so its eig reduces *exactly* to the 2Nx
  `eig(B·C)` — **~7.5× cheaper** per strip (~3× the whole solver), the dominant
  cost.
- **TE/TM**: TE = E along the invariant z (Ez), TM = Hz. Decoupled in an x-uniform
  strip or at qz=0 (where the solver reduces **exactly** to two scalar `eme_2d`
  runs — validated byte-level); hybridized by the x-structure at conical qz.
- **Mode condition** (`layer_vector_modes`): the **global block-`G`** lateral
  interface matrix is singular at a mode (`σ_min(G)=0`). `G` (one block per strip,
  no accumulation) replaces the Redheffer cascade residual `σ_min(M)`: the cascade
  *physics is exact* (signs, `[W;-V]` backward, oracle independence all proven in
  adversarial review) but its star-product **loses conditioning** as the
  propagating-strip-mode count grows toward low qz², so it found only **2/16**
  modes; the block `G` is well-conditioned and recovers the full band. Acceptance
  uses a **degeneracy-agnostic rank-drop test** (a genuine k-fold mode shows a
  sharp `s_k ≪ s_{k+1}` gap; spurious dips decay smoothly).
- **Field reconstruction** (`mode_field_vec`): the per-strip amplitudes are the
  null vector of the same `G`; returns the tangential-E field `(Ex, Ez)`.
- **Oracle** (`ref_2d_modes_vector`): an independent **Yee-staggered 2-D vector
  FD** Maxwell solve — first-order generator on `[Ex,Ey,hx,hy]`, eigenvalue
  `qz²=−γ²`, spurious-free (cross-checked vs an independent Fourier-PWE solver).
  Dense `eig` by default (full spectrum, for degeneracy/count checks); pass `k=`
  for a **sparse shift-invert** that returns the `k` in-band physical modes
  **~145× faster** (the test-suite arbiter).

**Validation** (`test_eme_2d_vector.py`, 6 tests, ~33 s): qz=0 → scalar reduction
(byte-level); uniform oracle (doubly degenerate, spurious-free); structured →
the 2-D-FD oracle converges to the EME; **full-band completeness** (block-`G`
recovers **16/16** of the band the cascade missed at only 2/16); dedup; mode field.

**Validated regime & limitations.** The mode-finder is validated for **structured
layers** (TE/TM split): recall **16/16** on the reference 2-strip cell at Nx=20
with ~1 spurious near-threshold candidate — cross-check completeness-critical work
against the oracle. **High-degeneracy** layers (a uniform slab: `±ky × 2-pol`
4-fold-degenerate dense clusters) give unreliable mode-finding; use the oracle /
analytic dispersion there. Exactly-degenerate pairs may merge under `merge_rtol`.

**Universality features** (toward general-tool parity; all additive, the legacy
path is byte-identical):
- **Arbitrary geometry** (`eps_xy_to_strips`): rasterize an arbitrary `eps(x,y)`
  (callable or grid) into `S` y-strips — the lateral analog of RCWA z-staircasing
  (1st-order; refine `S` for slant/curve). The EME accepts an arbitrary cell, not
  only a hand-built strip list.
- **Mode multiplicity** (`return_multiplicity=True`): also returns each mode's
  degeneracy `k` (the rank-drop order).
- **Spurious verify** (`verify=True`): drops the ~1 residual spurious candidate via
  an FD-oracle cross-check → recall 16/16, **spurious 0** (oracle-assisted; a
  self-contained PDE-residual check fails on piecewise-y Gibbs noise).
- **Anisotropic ε** (tensor): `strip_vector_modes` accepts a per-node `(Nx,3,3)`
  permittivity tensor — **diagonal birefringence + out-of-plane `exz`/`ezx`
  coupling** (lossless). The tensor generator reduces **byte-exactly** to the
  scalar one for an isotropic tensor, and a uniform strip matches the role-swapped
  Berreman planar dispersion to machine precision (incl. asymmetric `exz≠ezx`). The
  block-anti-diagonal `eig(B·C)` speedup survives (a `bc_ok` guard falls back to
  full `eig` otherwise). The 2-D-FD *oracle* is tensor-aware for **diagonal**
  tensors (the `exz` oracle term has a residual Yee-stagger error → deferred; the
  strip generator's `exz` is validated independently via Berreman).
- **Lossy ε**: the FD oracle `ref_2d_modes_vector(return_complex=True)` solves
  lossy layers **exactly** (complex `qz²`; a uniform lossy slab's modal loss
  `Im(qz²)` matches analytic to ~1e-13). The strip-solver itself is lossless; a
  weak-loss first-order perturbation reproduces the modal loss to ~5% (the exact
  non-Hermitian PT — the eliminated `Ez ∝ 1/ε` term — is unfinished), so use the
  oracle for exact lossy modes.

*Documented as out of scope (research-grade):* high-degeneracy mode-finding (a
conditioning wall in `G`'s degenerate null space); **strong-loss / leaky modes**
and lossy-anisotropy (complex-`qz²` modes are off the real-axis scan → need a
contour / Beyn eigensolver); `eyz/ezy` tensor coupling.

**Performance.** Two exact/validated speedups: the block-anti-diagonal strip-eig
reduction (~3× the whole solver) and the sparse shift-invert oracle (~145×). A
third (reducing the qz²-scan resolution `n_scan`) was investigated and **rejected**
— it trades away recall (sharp σ_min dips need fine sampling to bracket). Net: the
test suite went 357 s → 33 s (~11×).

## Known v1 limitation (scalar)

Exactly **degenerate** modes (e.g. a 4-strip checkerboard with a symmetry-paired
`qz^2`) can cluster or merge under `merge_rtol`. A symmetry-aware multiplicity
count (or a small `ky0` perturbation to split the pair) is the fix — deferred.

## Files

- `eme_2d.py` — the scalar solver (`strip_x_modes`, `cell_smatrix`, `dispersion`,
  `layer_modes`, `mode_field`) + the FD oracle (`ref_2d_modes`).
- `eme_2d_vector.py` — the vector solver (`strip_vector_modes`,
  `layer_vector_modes`, `mode_field_vec`) + the Yee 2-D vector oracle
  (`ref_2d_modes_vector`).
- `eme_diffraction.py` — the documented diffraction dead end (validated slab math).
- `test_eme_2d.py` (scalar, 5) / `test_eme_2d_vector.py` (vector, 6) /
  `test_eme_diffraction.py` (diffraction finding, 3).

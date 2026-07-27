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
  permittivity tensor — the **full out-of-plane 3×3** (diagonal birefringence +
  `exz`/`ezx` **and** `eyz`/`ezy` coupling, lossless). The tensor generator reduces
  **byte-exactly** to the scalar one for an isotropic tensor (and to the diagonal+`exz`
  body when `eyz=ezy=0`). A uniform strip matches the role-swapped Berreman planar
  dispersion to machine precision for `exz` (incl. asymmetric `exz≠ezx`); for `eyz`
  the role-swapped Berreman is the **wrong** oracle (its z-axis maps onto the
  eliminated y-axis), so `eyz` is validated against the analytic **Christoffel
  determinant** `det(k kᵀ − |k|²I + k0²ε)=0` to ~1e-13 (symmetric, asymmetric
  non-reciprocal, and combined `exz+eyz`). The block-anti-diagonal `eig(B·C)` speedup
  survives for diagonal/`exz`; `eyz` breaks it (populates the E-E/H-H blocks) and the
  `bc_ok` guard falls back to full `eig`, so the **strip modes stay rigorous**. The
  eyz **layer** mode-finder is gated (raises — `[W;-V]` is no longer the exact
  backward mode; see the out-of-scope note). The 2-D-FD *oracle* is tensor-aware for
  **diagonal** tensors only (the `exz`/`eyz` oracle terms have a residual Yee-stagger
  error → deferred; the strip generator is validated independently via
  Berreman / Christoffel).
- **Magnetic media** (`mu_x` / `mu_xy`): scalar permeability `μ(x,y)` (curl E = i k0 μ h)
  in the strip generator and the oracle — dispersion `eps·μ·k0² − kx² − ky²`. `μ=1`
  byte-exact; lossy-`μ` modal loss exact (via `return_complex`); the `eig(B·C)`
  speedup survives. (Tensor-`ε` with `μ≠1` raises — deferred.)
- **Lossy / leaky `qz²`** (complex modes): the FD oracle `ref_2d_modes_vector(
  return_complex=True)` solves lossy layers **exactly** (uniform slab `Im(qz²)` to
  ~1e-13). For the *strip solver*, a **seeded Beyn refiner** (`beyn_refine_complex`
  / `layer_vector_modes_complex`) reaches the complex modes the real-axis `σ_min`
  scan structurally cannot: seed from the coarse complex oracle, refine to the
  EME's own complex mode (x-FD floor ~1e-2, one mode/contour, weak–moderate loss).
- **Fine-staircase speedup** (`solver="banded"`): `σ_min` via inverse-power on the
  block-tridiagonal `G` (O(S) vs the dense O(S³) SVD) — for the large-`S` slant/curve
  regime (same modes; 2.3× at S=8 → 3.7× at S=16, growing).

*Documented as out of scope (research-grade, premises probe-tested):*
- **High-degeneracy / uniform-slab mode-finding** — the floor is **O(h²) x-FD error
  on high-`|kx|` strip modes, NOT degeneracy** (the premise was refuted: a
  symmetry-reduced `G` inherits the identical floor); use the oracle / analytic there.
- **Autonomous complex-mode discovery** (vs the seeded Beyn above) and
  lossy-anisotropy — the near-pole satellite cloud blocks clean unseeded multiplicity.
- **A self-contained spurious filter** — the ~1 spurious is a *true* `det(G)`
  ghost-zero with a Maxwell-consistent field; no function of `G` alone separates it,
  so `verify=True` stays oracle-assisted.
- **`eyz/ezy` *layer* mode-finder** — the eyz **strip** operator/modes are shipped
  and rigorous (above); only the cascade `layer_vector_modes` is gated (raises),
  because `eyz` breaks the `[W;-V]` backward mode the global block-`G` hard-codes —
  the general non-`[W;-V]` 4-field backward cascade is deferred (and structured-`eyz`
  has no converging independent oracle for an end-to-end layer check).

**Performance.** Two exact/validated speedups: the block-anti-diagonal strip-eig
reduction (~3× the whole solver) and the sparse shift-invert oracle (~145×). A
third (reducing the qz²-scan resolution `n_scan`) was investigated and **rejected**
— it trades away recall (sharp σ_min dips need fine sampling to bracket). Net: the
test suite went 357 s → 33 s (~11×).

## Physics-interior audit W6 (2026-07-26) — what was wrong, and the oracles

This subsystem was the *honest coverage gap* of the 2026-07-25 adversarial audit
(never numerically validated there; the 2026-07-09 EME audit was read-only and
concluded "none above nit level"). Four independent oracles were built first and
now live in `tests/unit/test_niche_audit_w6_eme.py`:

1. **analytic symmetric 3-layer slab dispersion**, bisected from the textbook
   equation with no package code (`_slab_betas`). Because the EME is *analytic in
   y* and the discrete x-Laplacian is exact on `exp(i kx0 x)`, an x-uniform
   `[clad|core|clad]` cell must satisfy `qz² = β²_slab − kx0²` — measured to
   **2.4e-8 relative** on the confined modes;
2. **lossless power conservation** of the lateral cell S-matrix,
   `Σ_i Re(ky_i)(|S11[i,j]|² + |S21[i,j]|²) = Re(ky_j)` for every propagating
   input `j` (both S ports live in strip 0's basis);
3. the **analytic Airy / Fabry-Pérot slab**, lossless *and lossy*, for the
   mode-matching driver;
4. the package's own 2-D-FD oracles used as a **recall / spurious** cross-check.

**The `kx0 ≠ 0` path was broken (CRITICAL).** `strip_x_modes` keyed its solver on
`kx0 == 0`, but `A = D + diag(ε k0²)` is **Hermitian for a real ε at any real
`kx0`** — the wrap corners carry `exp(+i kx0 Lx)` and its conjugate. Sending it
to `eig` anyway (a) returned `lam` with roundoff imaginary parts of *arbitrary
sign*, and `np.sqrt`'s principal branch (`Re ≥ 0`, **not** `Im ≥ 0`) then put
8–11 of 16 strip modes on the exponentially **growing** propagator — the exact
pitfall row 1 of the table above says S-matrices cure — and (b) normalised with
the complex-*symmetric* bilinear form, which is the wrong metric for a Hermitian
operator (`max|Φ^H Φ − I| = 43.2`). Measured on the reference structured cell at
`kx0 = 0.37`, `ky0 = π`: **68 roots returned, 0/3 real modes recovered, all 68
spurious**, and power conservation violated by 1.5e-2 (up to 1.8e-1 at
`kx0 = 1.1`). `kx0 = 0` was correct throughout, which is why every shipped test
passed. Fixed by `eigh` for real ε at any `kx0`, an exactly-Hermitian wrap phase
(`conj(ph)`), and one shared **decaying-branch** selector `_ky_forward`
(`Im(ky) ≥ 0`, matching the vector sibling's `_strip_split_forward`) used by
`_wv`, `_global_lateral_nullspace` and `mode_field` — the two field
reconstruction sites had inlined the same unguarded `np.sqrt`. Post-fix: power
conserved to 1.3e-15, recall 3/3 with 0 spurious at every `kx0` tested.

**`mode_match` carried the same class of growing exponential in z (HIGH).** The
backward layer amplitudes were referenced at `z = 0`, so
`exp(−i qz depth) = exp(+|qz| depth)` entered the matched system for every
evanescent layer mode; `cond(A) ~ exp(2|qz|max·depth)` reached 8.7e38 and past
the `lstsq` cutoff the answer collapsed to `R_00 = 1`, `T_00 = 0` **for a
homogeneous index-matched medium** (exact `T = 1`) — with `energy = 1.000000`
masking it. The backward amplitudes are now referenced at `z = depth`
(algebraically identical, only the decaying `exp(i qz depth)` survives):
`cond(A) ≈ 2` and the analytic Airy slab is reproduced to ~5e-15 for
`depth = 0.2 … 16`.

**`diffraction_fd` did not absorb (HIGH).** It discarded `Im(qz²)`, so an
absorbing slab reported `energy = 1.000000` — at `n = 1.5 + 0.2j`, `depth = 4` it
claimed all the light emerged while 95% was absorbed. A complex `eps_xy` now
keeps the complex spectrum; the analytic lossy Airy R/T match to 8 decimals.

**The negative result above still stands.** W6-2 was a conditioning bug in the
*z* match; the structured non-convergence is a *basis* problem. With the stable
reformulation in place a structured layer still fails to converge and still warns.

**The vector finder had no band-edge guard (found by one of this wave's own
pins).** A vector strip mode's H-part is `(C U)/(i ky)`, which divides by zero
for a mode exactly on a band edge — and `qz² = 0` puts the reference cell's
uniform `ε = 2` strip precisely there (`min|ky| = 0.000e+00` at `Nx = 8`,
`k0 = 8`). The `NaN` then surfaced several frames later as
`ValueError: array must not contain infs or NaNs`, so the entirely natural
window `qz2_range = (0, …)` crashed; the same opaque error came from a `qz²` so
far outside the band that `exp(+|ky| h)` overflows. The scalar sibling has
skipped its analogous sample since audit P3-18. Both cases now raise a *named*
`LinAlgError` from one shared `_equilibrated_G` builder, which
`layer_vector_modes` skips just as `layer_modes` does.

Also fixed: the rasterizers now enforce the layer finders' `sum(h) == Ly`
contract (they used to leave uncovered y rows at `ε = 0` → `inf`/`NaN` in the
vector oracle); a junk `solver=` value is rejected instead of silently running
dense; the `layer_vector_modes` detection density is per unit of the
*dimensionless* `(hi − lo)·Ly²` (it was per unit of raw `qz²`, so one physical
cell asked for 3944 points in µm, 400 in nm and 3.94e9 — a 31.5 GB `linspace` —
in mm); `sigma` without `k` raises instead of being inert; the scalar sparse
oracle takes a fixed ARPACK `v0` like both its siblings.

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

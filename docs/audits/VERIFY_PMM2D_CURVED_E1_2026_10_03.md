# VERIFY -- curved cells for the pure staggered 2-D PMM, Phase E1 (out-of-plane tensors and slanted walls inside curved cells)

Date: 2026-10-03.  Verifier: Claude Opus 5.5 (model ID `claude-opus-5-5`),
independent adversarial verification.
Mount: worktree `C:/tmp/lum_vcurved_e1`, branch `verify/pmm2d-curved-e1`, on
the builder's tip `893d48df` (`feat/pmm2d-curved-e1-oop-slant`, commits
`46cf85f0` .. `893d48df` on the Phase D tip `eae470d9`, Phase C and D
verifier branches merged in).  PRE tree: the verifier's own `git archive
eae470d9` in `C:/tmp/vce1_pre_eae4` (not the builder's).
Builds: Windows 11 (CPython 3.14.6, numpy 2.4.4, scipy 1.17.1) and WSL
Ubuntu (CPython 3.12.3, numpy 2.4.6, scipy 1.17.1, `~/lumvenv`), BLAS pinned
to one thread on the command line, `lumenairy.__file__` asserted under the
tree being measured in every probe (`verify_e1/_ve1common.py`).  The box was
shared with two sibling agents (E2 builder, E3 verifier) and ran at 90-100 %
load for the whole verification, so every wall time below is an upper bound
and only same-run ratios mean anything; no accuracy number depends on load.
Evidence: `validation/probe_pmm2d_curved/verify_e1/` (probes + JSON; the
fixtures, the oracle and the engineered defects are in `_ve1common.py` and
`_ve1mut.py`).  Decision tests: `tests/unit/test_verify_pmm2d_curved_e1.py`.
Nothing under `lumenairy/` was edited.

---

## 0. Words used here

* **Map** `(x, y) = Phi(u, v)`, Jacobian `J`, `sg = det J`, `adj(J) = sg
  J^-1`.  **Composite frame** of a slanted layer under a map: `(x, y, z) =
  (Phi(u, v) + t w, w)`, Jacobian `Lambda = [[J, t], [0, 1]]`, `t` the
  INTERNAL shear (the negative of the public `slant`, rotated by the gauge
  sign).  **tau** `= J^-1 t` (the shear seen in `(u, v)`), **kappa** `=
  chi_t' tau`.
* **First-order generator**: the out-of-plane pencil `A x = q B x` on `x =
  [E1; E2; G1; G2]`; its **permeability blocks** are `R = C[chi_t]C` (on the
  B side of the two E rows), `K_tz` (E rows) and the `chi33`-weighted `G3`
  projection (G rows).
* **The two gauge constants**: `_OOP_ROT_SIGN = -1` (sign on the four
  out-of-plane tensor entries and on the slant vector) and `_OOP_H_GAUGE =
  -1j` (`G` to Eq.-25 `H`).
* **C2-broken cell**: a cell with no 180-degree rotation symmetry about z --
  the only kind on which a flipped rotation sign shows in the zeroth-order
  Jones matrix at normal incidence (a C2-symmetric cell's flipped device is
  the rotated original, whose zeroth-order Jones is identical).
* **The same device two ways**: a pattern whose material walls are all grid
  lines, solved unmapped and under a transfinite map whose moved vertices
  all sit inside ONE material, so the physical structure is identical and
  only the discretisation differs.
* **Own oracle** `eps_mu_slab`: an (eps, mu) Berreman slab, solved by the
  slab's eigen-decomposition with backward modes referenced at its bottom
  face (an 8 x 8 interface system, no matrix exponential) and analytic s / p
  half-space modes -- a different formulation from the builder's
  `expm`-based oracle.
* **Mutation / fail-before**: an engineered defect reached through the real
  code path and restored afterwards (`_ve1mut.vmutate`).

---

## 1. Verdict

| # | builder claim | verdict | the verifier's own measurement |
|---|---|---|---|
| E1-1 | no map = `eae470d9` bytes; NamedTuple bit-identical | **CONFIRMED** | own 199-key set (every OOP, slant, parity on / off, slanted-stack with the frame-anchor amplitudes, magnetic, tensor, per-layer and Phase C / D MAPPED fixture): 199 / 199 on Windows AND on WSL; the three E1 kernels raising: 199 / 199; `_chi_*` raising under a map: every unmapped key produced and equal, and the 10 unmapped suites 266 passed |
| (2a) | no 3 x 3 inverse; the G3 elimination exact | **CONFIRMED, scope stated** | a z-independent map cannot create `mu'^{3t}` from a block-form mu (exactly 0.0 over 2000 draws, and by the block-diagonal congruence); the composite frame creates one and the code carries exactly it (`tau` identity 2.7e-15); the OOP-mu refusal protects `chi'_tt = (mu^-1)_tt`, which the code does not form -- a correctness guard, loud at 10 / 10 entry points; its message is stale (V-E1-1) |
| (2b) | B Hermitian positive definite under a map | **CONFIRMED** | `-R` Hermitian to 1.1e-16; smallest generalised eigenvalue against the plain Gram 0.14 .. 0.30 on four circle variants (0.077 on a strong stretch), M = 6 .. 8, though `chi_t` reaches 4e-4 pointwise; the Cholesky never fails |
| (2c) | composite metric, `tau = J^-1 t` | **CONFIRMED, sharpened** | derived and measured; `tau` deviates from `t` by up to 5.7x (c3) / 12.4x (c5); the `tau = t` fail-before is 10-40x larger on a SLANTED PILLAR under the circle map (8.7e-3 .. 3.2e-2) than on the builder's slab (7.7e-4); `tau = J t` 2.3e-2 .. 8.0e-2 |
| E1-2 | identity map = shipped generator | **CONFIRMED** | the builder's ids pass on both builds; the unmapped `mu = 1` route through the general generator = the shipped one (A 1.0e-15, eigenvalues <= 7.1e-14) |
| (3) | both gauge constants map-independent | **CONFIRMED -- the argument needs no map symmetry** | flipped constants miss the same amounts mapped and unmapped on three NON-symmetric maps; on a C2-BROKEN cell at NORMAL incidence the same device mapped / unmapped converges 8.5e-4 -> 1.3e-5 under the shipped AND under the globally flipped rotation sign, while a map-only flip sits flat at 3.0e-2 |
| E1-3 / E1-3m | OOP slab under maps = Berreman; the (eps, mu) oracle | **CONFIRMED on own tensors, maps, oracle** | own oracle = `berreman_jones_1d` 4.6e-15, = Airy 4.4e-16, = the builder's 7.6e-15; general-director, non-reciprocal, lossy gyrotropic eps with a lossy gyrotropic mu, lossless gyro / symmetric lossy / anisotropic mu, on an asymmetric two-harmonic stretch, a 4 x 4 shear, c3, c5: round-off or the map's resolution, tracking the in-plane LC film; slanted and slanted-magnetic slabs round-off by M = 7; the out-of-plane mu refused loudly |
| E1-4 | OOP film under the circle map | **CONFIRMED** | c3 normal 1.4e-13 by M = 8; c5 2.0e-13 by M = 6; conical 9.0e-10 (c3 M = 8) / 1.3e-11 (c5 M = 6) |
| E1-5 | OOP pillar: two topologies, RCWA 1 / N, staircases | **CONFIRMED** on own pillar | topologies 3.7e-6 (normal) / 4.8e-5 (conical); exact-disk RCWA 1 / N on five points; staircases converge to their own devices (non-monotone in k on this fixture: D-E1-4) |
| E1-6 | slant under a map; slanted circle; mirror | **CONFIRMED, one wording note (D-E1-4)** | slanted slab null test round-off (shear) / spectral (circle: 5.8e-10 at M = 7, 8.3e-12 at M = 8); slanted circle topologies 2.7e-5; the z-staircase extrapolation is legitimate in N_z (second order, four points) and 1 / N in its envelope, landing 3.2e-4 .. 4.0e-4 from the curved answer (a 1e-4-class floor); the opposite slant is the exact mirror (1.2e-13) |
| E1-7 / E1-8 | reciprocity; the non-reciprocal twin is physics | **CONFIRMED -- Onsager decisive** | magnetisation reversed (`nr` fwd vs `nr^T` rev): 2.1e-4 -> 4.8e-5 -> 1.8e-6 (c3 M = 5 .. 7), 1.4e-6 (c5 M = 5), tracking the reciprocal control, while `nr` fwd vs `nr` rev stays FLAT at 6.4e-3 / 1.5e-2 |
| F-E1-5 / (6) | parity accelerator off under a map / with mu | **CONFIRMED as shipped; SCOPE for symmetric lossless cases, CORRECTNESS for a lossy mu** | lifted on a centred circle map it is exact (eigenvalues 1.8e-14, stack 1.3e-13) and 2.3x faster; with a lossy mu its structural test passes and only a Cholesky failure saved it |
| E1-9 / (7) | mutation matrix | **CONFIRMED + one gap CLOSED** | 10 own kinds, all caught by builder ids, but the WEAK-loss `qz_off` case (silent, 3.3 x Im mu) only by the new decision test |
| E1-10 | cost | **not re-measured** (saturated box); the same-run ratios of section 7 only | |
| (8) | docs | **CONFIRMED with four small notes** | 37 / 37 BUILD_E1 numbers agree with the JSON; the CHANGELOG example runs; mypy, WSL ruff, fingerprints clean; V-E1-1, D-E1-2 .. D-E1-4 |

---

## 2. E1-1 -- byte identity, the verifier's own key set

`v1_bytes.py`, run in BOTH trees with the same file.  Fixtures chosen
independently of the builder's: unmapped out-of-plane generators (a director
in a general direction at normal incidence; its non-reciprocal twin on
NON-UNIFORM walls at an oblique Bloch phase; a lossy gyrotropic tensor
rotated out of the plane at conical incidence; a centro-symmetric OOP cell),
each with its operators, its region modes, its region modes through the
PARITY reduction and the parity gauge itself; slanted scalar, in-plane-tensor
(conical) and out-of-plane cells; in-plane magnetic cells (gyrotropic mu with
an LC eps at conical incidence, a lossy scalar mu, a lossy tensor mu); the
`_homog_geom_cache` tuple and the half-space modes built from it (the
NamedTuple refactor); ten `pmm_jones_2d_staggered` calls (OOP auto / oblique
/ conical with the parity reduction off, the parity reduction ON and OFF on
a centro-symmetric cell, lossy gyrotropic OOP, slanted scalar at conical,
slanted OOP, tensor + gyro mu, scalar mu); eight stacks (an OOP mixed stack
with `symmetry="auto"` and `False`, a lossy OOP stack at conical incidence,
a globally sheared two-layer slanted stack at conical incidence with the
per-order transmitted amplitudes -- the frame-anchor phase --, a slanted
pattern under a vertical film, a slanted uniform layer, a magnetic mixed
stack, a per-layer-grid stack with an OOP layer, a rectangles-only shape
layer with an OOP tensor), every one with `layer_absorption` and the modal
amplitudes; and the Phase C / D MAPPED solves whose permeability blocks were
refactored into `_chi_R / _chi_Ktz / _chi_Gw` (an off-centre circle at
oblique incidence, an LC under a stretch, LC + gyro mu under a shear at
conical incidence, a lossy mu under the 5 x 5 circle, a shape circle with an
LC eps and a gyro mu, a shape ellipse with a uniform film).

| run | keys | identical to PRE |
|---|---|---|
| Windows, pre vs post | 199 | **199** |
| WSL, pre vs post | 199 | **199** |
| Windows post, the three E1 kernels (`_assemble_oop_general`, `_stag_map_slant_weights`, `_stag_scale_weight`) RAISING | 199 produced | **199** |
| Windows post, `_chi_R / _chi_Ktz / _chi_Gw` RAISING under a map (`chitrap`) | 153 produced (every unmapped key) | **153** -- the 46 missing are exactly the six mapped fixtures, which must reach the shared blocks |

(`v1_bytes_{pre,post,posttrap,postchitrap,wslpre,wslpost}.json`,
`v1_compare_*.json`.)  The NamedTuple refactor is bit-identical (the
`homog.tuple` and `homog.modes` keys).  The `chitrap` mutant over the
UNMAPPED suites (OOP, slant, magnetic, anisotropic, both block-eig files,
5.12 staggered, `test_staggered`, the per-layer slant verifier, non-uniform
walls): **266 passed** (`mut_logs/chitrap.txt`).

---

## 3. The derivation, checked independently

### 3.1 (a) No 3 x 3 inverse, and where `mu'^{3t}` can come from

The constitutive law is used in two block forms, (I) `G'_t = chi'_tt c_t +
chi'_t3 c^3` and (II) `G'_3 = (c^3 - mu'^{3t} G'_t) / mu'^{33}`; they are
the same law by the block-inverse identities `chi'_3t chi'_tt^-1 =
-mu'^{3t} / mu'^{33}` and the Schur complement `chi'_33 - chi'_3t
chi'_tt^-1 chi'_t3 = 1 / mu'^{33}` (both follow from `mu' chi' = I`
written in blocks; neither uses symmetry).  (I) needs
`chi'_tt` -- the tt block of the FULL inverse -- and `chi'_t3`; (II) needs
only `mu'^{33}` and `mu'^{3t}`.  So the elimination is exact for any
`mu'^{3t}` and no 3 x 3 inverse is ever needed PROVIDED `chi'_tt` is
available without one.  That proviso is what decides the scope:

* **A z-independent map cannot create `mu'^{3t}` from a block-form mu.**
  With `Lambda_m = blockdiag(J, 1)`, `M(mu) = sg Lambda_m^-1 mu
  Lambda_m^-T` has t3 block `sg J^-1 mu_t3 = 0` and 3t block `sg mu_3t
  J^-T = 0` when `mu_t3 = mu_3t = 0`: the congruence by a block-diagonal
  matrix preserves block-diagonal form.  Measured on 2000 random `J` (det >
  0) and random complex block-form `mu`: the out-of-plane blocks of `M(mu)`
  and of its inverse are **exactly 0.0** (`v2a_mu3t.json`).
* **The composite (slant) frame does create one**, and the shipped code
  handles exactly that one: `Lambda = [[J, t], [0, 1]]` gives `mu'^{3t} =
  -sg mu33 tau^T`, carried as `tau = -mu'^{3t} / mu'^{33}` (measured identity
  2.7e-15) and `kappa = chi'_t3 = chi'_tt tau` (5.8e-13 relative), with
  `chi'_tt = J^T mu_t^-1 J / sg` UNCHANGED by the shear and the Schur
  complement `1 / (sg mu33)` (same bound).  Here `chi'_tt` IS available
  without a 3 x 3 inverse because the material is block-form: the shear
  only adds the rank-one `t` column.
* **What the out-of-plane-permeability refusal protects.**  For a material
  `mu` with `mu_t3 != 0`, `chi'_tt` is the tt block of the full inverse,
  `(mu_tt - mu_t3 mu_3t / mu_33)^-1`, not the `mu_t^-1` that `_chi_maps` /
  `_stag_map_eff_tensor` compute, and `tau`, `kappa` would have to come from
  the material (per cell / per node) rather than from the shear alone.  The
  generator's STRUCTURE would carry it (the `tau` / `kappa` slots exist), but
  the weights would be silently wrong without the full split.  The refusal
  is therefore a correctness guard, and it holds: an out-of-plane `mu` is
  refused LOUDLY (`NotImplementedError` naming "OUT-OF-PLANE permeability")
  at all ten entry points tried -- the solver with / without a map / with a
  slant, the Jones entry with / without a map, uniform / mapped / slanted
  stack layers, a mapped `mu_cell` and a shape's `mu=` (`v2a_mu3t.json`).
  **Defect V-E1-1 (wording)**: five of those ten messages (all that go
  through `_require_inplane_mu`) still give the PRE-E1 reason, "the
  out-of-plane first-order generator has no mu blocks", which is false since
  this phase (section 9).

### 3.2 (b) B stays Hermitian positive definite under a map

`B = blkdiag(-R, Ggram2, Ggram1)`; the claim is that `-R = C[chi_t]C` is HPD
for pointwise HPD `chi_t`.  Measured on the assembled operator
(`v2b_hpd_6_7_8.json`), OOP disk, vacuum and a lossless gyrotropic mu in the
disk, M = 6 / 7 / 8:

| map | Hermiticity of `-R` (rel.) | smallest generalised eigenvalue of `-R` against the PLAIN Gram | pointwise smallest eigenvalue of `chi_t` at the nodes | Cholesky |
|---|---|---|---|---|
| 3 x 3 circle, r 0.30 p | <= 1.1e-16 | 0.30 / 0.27 / 0.23 (gyro 0.26 / 0.22 / 0.19) | 3.1e-3 .. 2.2e-3 | never fails |
| 3 x 3 circle, r 0.45 p | <= 9.3e-17 | 0.22 / 0.19 / 0.17 (gyro 0.18 / 0.16 / 0.14) | 7.9e-4 .. 3.9e-4 | never fails |
| 3 x 3 circle OFF-CENTRE | <= 9.9e-17 | 0.29 / 0.26 / 0.23 | 2.2e-3 .. 1.5e-3 | never fails |
| 5 x 5 circle | <= 1.1e-16 | 0.27 / 0.21 (M = 6 / 7) | 2.8e-3 .. 2.3e-3 | never fails |
| asymmetric two-harmonic stretch (strong) | <= 7.3e-17 | 0.10 / 0.086 / 0.077 | 5.0e-2 | never fails |

The Duffy-weighted `chi_t` reaches pointwise eigenvalues of 4e-4 near the
singular vertices, but the Galerkin operator's smallest eigenvalue RELATIVE
to the plain Gram stays at 0.14 .. 0.30 and falls only ~10 % per rung: the
nodes where `chi_t` is small carry a small share of every basis function's
mass.  The absolute condition number of `-R` (1.6e4 .. 3.4e5) is the plain
Gram's own.  The `_bgen_hermitian` flag reads True on every lossless case
and False on a lossy mu.  The whitening is safe.

### 3.3 (c) The composite frame and where `tau`'s variation matters most

`Lambda = A Lambda_m` with `A = [[I, t], [0, 1]]` (the shear) and
`Lambda_m = blockdiag(J, 1)`: `A Lambda_m = [[J, t], [0, 1]]`, `det = sg`.
The congruence of the shear's congruence is the map's congruence of `S(mu) =
A^-1 mu A^-T`; with `S(mu)^-1 = A^T mu^-1 A` one gets `chi'_t3 = J^T chi_t t
/ sg = chi'_tt (J^-1 t)`, i.e. `tau = J^-1 t = adj(J) t / sg` -- and the
measured identities of 3.1 confirm it numerically for random `J`, `t`, `mu`.

`tau` varies across a curved cell: `max |tau - t| / |t|` over the cells is
**5.7** on the 3 x 3 circle (both radii) and **12.4** on the 5 x 5 circle
(`v2c_tau_*.json`) -- `tau` blows up like `1 / sg` toward the singular
vertices.  That is where the "tau = t" fail-before (the shipped constant
slant blocks laid on a mapped cell) bites hardest: on a SLANTED PILLAR under
the circle map (public slant 0.15, M = 6; 36-vector change against the
correct arm):

| fixture | `tau = t` | `tau = J t` (the Jacobian instead of its inverse) | the slab null test (sheared map, M = 5) |
|---|---|---|---|
| eps 3.5 disk, c3, normal / conical | **8.7e-3 / 3.2e-2** | 2.3e-2 / 8.0e-2 | `tau = t`: 6.9e-4 normal, 2.2e-3 oblique; `tau = J t`: 2.8e-3 / 6.1e-3 |
| OOP disk, c3, normal / conical | **1.6e-2 / 2.7e-2** | 3.8e-2 / 6.2e-2 | |
| eps 3.5 disk, c5 (M = 5), normal | **1.7e-2** | 5.8e-2 | |

The pillar under the curved map is 10-40x more sensitive to the composite
coupling than the builder's slab null test (7.7e-4) -- the strongest place
to pin it, and both wrong couplings are caught there by orders of magnitude
above the discretisation's own rung change.

### 3.4 (d) The reduction to the shipped generator

Unmapped, the general generator is reachable with `mu_cell = ones`.
Against the shipped generator (`v2d_reduction.json`), vertical and slanted,
normal and conical Bloch phases: `A` relative 1.0e-15, `B` <= 8.7e-17,
eigenvalues <= 7.1e-14.  `chi = I`, `tau = t` reduce every block to the
shipped one, as section 1.3 of BUILD_E1 states.

---

## 4. The gauge constants are map-independent (the argument's weak point, tested)

**The argument, re-derived.**  `_OOP_ROT_SIGN` flips the four out-of-plane
entries.  As a matrix statement that is the congruence by `rho =
diag(-1, -1, 1)` (or equivalently by the z-mirror `diag(1, 1, -1)`; inversion
leaves a tensor unchanged), and `rho` commutes with `blockdiag(J, 1)` for
EVERY 2 x 2 `J`, pointwise -- no symmetry of the map is needed, because the
sign acts on the tensor at a point, never on the geometry.  The builder's
wording ("the rotation maps `u -> -u`, `x -> -x` together, so `J` is
invariant") suggests a symmetric map is required; it is not.  The weak point
is therefore not the commutation but whether the sign is applied to the
mapped weights at all, and whether some map-dependent quantity smuggles a
second sign in.  Tested three ways:

**(a) Films under NON-symmetric maps** (`v3_film_{none,th3,c3off,sh4}_M6.json`;
general-director tensor `DIRGEN`, M = 6, R/T / Jr / Jt against the own
oracle):

| arm | unmapped, oblique | asym. two-harmonic stretch, oblique | off-centre circle, conical | sheared 4 x 4, conical |
|---|---|---|---|---|
| shipped | 1.4e-10 / 6.3e-11 / 1.5e-10 | 4.6e-6 / 2.2e-6 / 5.4e-6 | 2.1e-7 / 7.5e-8 / 3.5e-7 | 1.2e-13 / 3.4e-14 / 7.0e-14 |
| rotation +1 | 2.0e-3 / 3.0e-2 / **0.27** | 2.0e-3 / 3.0e-2 / **0.27** | 6.4e-4 / 1.5e-2 / **0.13** | 6.4e-4 / 1.5e-2 / 0.13 |
| H gauge +1j | 2.0e-3 / 5.0e-2 / **1.3** | 2.0e-3 / 5.0e-2 / 1.3 | 6.4e-4 / 4.2e-2 / 1.3 | 7.3e-4 / 4.2e-2 / 1.3 |
| H gauge +1 | closure 2.0 | closure 2.0 | closure 2.5 | closure 2.5 |

The flipped constants miss by the SAME amounts mapped and unmapped, to two
digits, on every non-symmetric map, at every mount; the rotation sign is
blind at normal incidence on every film (2e-14 .. 1e-6, the map's own
resolution), as it must be (a film is C2-symmetric).

**(b) The same device two ways, C2-BROKEN, at NORMAL incidence**
(`v3_same_M{4,5,6}.json`; decision test
`test_ve1_gauge_constants_are_map_independent_on_a_c2_broken_cell`).  A
staircase triangle of `DIRGEN` on a 4 x 4 grid -- no 180-degree symmetry, so
the rotation sign shows in the zeroth-order Jones at normal incidence (Jr
7.5e-3) -- solved unmapped and under a non-symmetric 4 x 4 transfinite map
whose three moved vertices sit inside one material each (one in the tensor
triangle, so the out-of-plane congruence `adj(J) e_t3` acts there with a
non-diagonal `J`).  Largest difference over R, T (all orders) and both
Jones:

| arm | mapped vs unmapped, M = 4 / 5 / 6 | moves the unmapped answer by |
|---|---|---|
| shipped | 8.5e-4 / 7.3e-5 / **1.3e-5** (converging) | -- |
| rotation +1, globally | 9.7e-4 / 8.8e-5 / 1.2e-5 (converging together) | 2.9e-2 |
| rotation sign withheld from the MAPPED weights only (a map-dependent gauge) | **2.9e-2 / 3.0e-2 / 3.0e-2 (flat)** | 0 |
| H gauge conjugated on the MAPPED region only | 0.27 (M = 4); M = 5, 6 refused loudly by the conditioning guard | 0 |

A map-dependent rotation gauge would have to show here; it does not, and the
engineered one is caught at 2000x the shipped residual by M = 6.  (The
globally wrong H gauge `+1` is also a consistent -- wrong -- device: mapped
and unmapped agree to 3.7e-4 at M = 6 while both miss the shipped answer by
2.6 in R + T.)

**(c) The pillar against the unmapped staircase** (`v3_pillar_M5_k2.json`):
an off-centre OOP disk under its own 3 x 3 map vs the shipped OOP solver's
k = 2 staircase of the same disk, normal incidence.  Curved vs staircase is
2.3e-2 in every arm (the staircase's geometry error); the rotation flip moves
the curved and the staircase answers by 3.9e-2 / 3.3e-2 on the per-order
vector and by 6e-6 on the zeroth-order Jones -- an off-centre disk is still
C2-symmetric about its own centre, so the zeroth-order Jones is blind, as
predicted.  This comparison cannot discriminate a gauge at the 1e-2 level
(the staircase is a different device); (b) is the decisive one.

**Conclusion: both constants are map-independent, by the pointwise
commutation argument (which needs no map symmetry) and by measurement on a
non-symmetric map with a C2-broken cell at normal incidence.**

---

## 5. E1-3 / E1-3m -- uniform slabs on the verifier's own tensors, maps and oracle

Oracle validation first (`v0_oracle.json`): against `berreman_jones_1d` at
mu = I on the three own out-of-plane tensors at three mounts, R / T / Jr /
Jt <= **4.6e-15**; against the Airy formula of an isotropic (eps, mu) slab,
lossless and lossy, s and p, 0 / 30 / 55 deg: <= **4.4e-16**; against the
BUILDER's expm oracle with three permeabilities (gyro, lossy gyro,
anisotropic) on three tensors at three mounts: <= **7.6e-15**.

Tensors: `dirgen` (uniaxial, director polar 52 deg, azimuth 37 deg -- no
zero entry), `nrgen` (its non-reciprocal twin: `e13`, `e23` with imaginary
parts, Hermitian), `gyrol` (a LOSSY GYROTROPIC tensor rotated by three Euler
angles: fully populated, non-symmetric), with mu `gyro` (lossless
gyrotropic, Hermitian), `gyrol` (lossy gyrotropic: the QZ branch), `syml`
(symmetric lossy), `aniso`; and `lc` (the in-plane LC, the reference film).
Maps: `th3` / `th2b` (an ASYMMETRIC two-harmonic separable stretch, 3 x 3 /
2 x 2 strong), `sh4` (4 x 4 sheared transfinite, five interior vertices
moved), `c3`, `c5` (circle, r 0.3 p).  Mounts: normal, oblique (30, 0),
conical (22, 63 deg).  Worst of R/T, Jr, Jt (`v4_slab_*.json`,
`v4_summary.json`):

| combo, map (_s = public slant 0.17, -0.08) | normal | oblique (30, 0) | conical (22, 63) |
|---|---|---|---|
| dirgen_none | 1.5e-14 / 1.6e-14 / 2.7e-14 / 5.9e-14 / 1.2e-13 (M 4..8) | 6.0e-06 / 3.8e-08 / 1.5e-10 / 4.6e-13 / 1.3e-13 (M 4..8) | 4.6e-07 / 1.3e-09 / 2.3e-12 / 7.9e-14 / 1.4e-13 (M 4..8) |
| dirgen_th3 | 3.8e-04 / 3.0e-05 / 1.4e-06 / 4.1e-08 / 1.1e-09 (M 4..8) | 7.6e-04 / 7.5e-05 / 5.4e-06 / 4.3e-07 / 2.3e-08 (M 4..8) | 5.2e-04 / 5.0e-05 / 3.5e-06 / 1.7e-07 / 1.2e-08 (M 4..8) |
| dirgen_th2b | 2.2e-02 / 1.9e-03 / 1.2e-04 / 1.6e-05 / 4.8e-07 (M 4..8) | 1.5e-02 / 2.4e-03 / 4.9e-04 / 5.7e-05 / 9.0e-06 (M 4..8) | 2.7e-02 / 2.3e-03 / 1.4e-04 / 2.8e-05 / 1.1e-06 (M 4..8) |
| lc_th2b | 1.5e-02 / 3.1e-03 / 1.9e-04 / 2.7e-05 / 8.0e-07 (M 4..8) | 1.9e-02 / 3.4e-03 / 7.6e-04 / 8.7e-05 / 1.4e-05 (M 4..8) | 3.0e-02 / 4.2e-03 / 2.5e-04 / 5.0e-05 / 2.0e-06 (M 4..8) |
| dirgen_sh4 | 6.4e-14 / 9.6e-14 / 8.4e-14 / 1.4e-13 (M 4..7) | 3.3e-07 / 9.0e-10 / 1.5e-12 / 4.5e-13 (M 4..7) | 1.6e-08 / 1.7e-11 / 1.2e-13 (M 4..6) |
| dirgen_c3 | 4.9e-06 / 8.1e-08 / 2.7e-10 / 2.3e-12 / 1.4e-13 (M 4..8) | 4.3e-04 / 4.9e-05 / 2.9e-06 / 1.6e-07 / 6.8e-09 (M 4..8) | 1.0e-04 / 1.5e-05 / 5.6e-07 / 2.4e-08 / 9.0e-10 (M 4..8) |
| lc_c3 | 8.1e-06 / 1.3e-07 / 4.4e-10 / 3.5e-12 / 8.4e-14 (M 4..8) | 4.6e-04 / 4.4e-05 / 2.6e-06 / 1.4e-07 / 6.2e-09 (M 4..8) | 9.0e-05 / 1.4e-05 / 5.0e-07 / 2.2e-08 / 8.3e-10 (M 4..8) |
| dirgen_c5 | 1.3e-08 / 2.6e-11 / 2.0e-13 (M 4..6) | 6.8e-07 / 7.2e-09 / 6.0e-11 (M 4..6) | 1.6e-07 / 2.4e-09 / 1.3e-11 (M 4..6) |
| nrgen_none | 9.6e-15 / 2.8e-14 / 4.2e-14 / 1.1e-13 / 1.1e-13 (M 4..8) | 6.0e-06 / 3.8e-08 / 1.5e-10 / 4.4e-13 / 1.7e-13 (M 4..8) | 4.8e-07 / 1.3e-09 / 2.4e-12 / 4.9e-14 / 1.4e-13 (M 4..8) |
| nrgen_th3 | 3.9e-04 / 3.1e-05 / 1.4e-06 / 4.3e-08 / 1.1e-09 (M 4..8) | 7.3e-04 / 7.3e-05 / 5.2e-06 / 4.3e-07 / 2.2e-08 (M 4..8) | 5.4e-04 / 5.3e-05 / 3.7e-06 / 1.8e-07 / 1.3e-08 (M 4..8) |
| nrgen_th2b | 2.3e-02 / 2.0e-03 / 1.2e-04 / 1.7e-05 / 5.0e-07 (M 4..8) | 1.5e-02 / 2.3e-03 / 4.8e-04 / 5.5e-05 / 9.0e-06 (M 4..8) | 2.8e-02 / 2.4e-03 / 1.5e-04 / 2.9e-05 / 1.2e-06 (M 4..8) |
| nrgen_c3 | 5.2e-06 / 8.5e-08 / 2.8e-10 / 2.3e-12 / 4.4e-13 (M 4..8) | 4.3e-04 / 4.9e-05 / 2.9e-06 / 1.6e-07 / 6.8e-09 (M 4..8) | 1.0e-04 / 1.5e-05 / 5.5e-07 / 2.3e-08 / 9.0e-10 (M 4..8) |
| nrgen_c5 | 1.4e-08 (M 4..4) | - | - |
| gyrol_gyrol_none | 7.1e-15 / 1.6e-14 / 1.9e-14 / 5.3e-14 / 1.1e-13 (M 4..8) | 2.2e-05 / 1.3e-07 / 5.3e-10 / 1.5e-12 / 6.9e-14 (M 4..8) | 1.6e-06 / 4.5e-09 / 8.0e-12 / 7.3e-14 / 1.4e-13 (M 4..8) |
| gyrol_gyrol_th3 | 1.1e-03 / 8.7e-05 / 3.9e-06 / 1.2e-07 (M 4..7) | 2.1e-03 / 2.1e-04 / 1.5e-05 / 1.4e-06 (M 4..7) | 1.5e-03 / 1.5e-04 / 1.1e-05 / 5.3e-07 (M 4..7) |
| gyrol_gyrol_th2b | 2.1e-02 / 5.5e-03 / 3.5e-04 / 4.7e-05 / 1.4e-06 (M 4..8) | 3.5e-02 / 6.1e-03 / 1.6e-03 / 1.6e-04 / 3.1e-05 (M 4..8) | 3.6e-02 / 7.1e-03 / 4.4e-04 / 8.7e-05 / 3.4e-06 (M 4..8) |
| gyrol_gyrol_sh4 | 5.1e-15 / 2.0e-14 / 2.2e-14 (M 4..6) | - | - |
| gyrol_gyrol_c3 | 1.5e-05 / 2.2e-07 / 9.4e-10 (M 4..6) | - | - |
| dirgen_gyro_none | 2.0e-14 / 3.2e-14 / 3.6e-14 / 6.8e-14 / 1.2e-13 (M 4..8) | 1.5e-05 / 9.2e-08 / 3.7e-10 / 1.0e-12 / 1.1e-13 (M 4..8) | 1.3e-06 / 3.5e-09 / 6.3e-12 / 7.6e-14 / 2.3e-13 (M 4..8) |
| dirgen_gyro_th2b | 2.7e-02 / 4.6e-03 / 2.8e-04 / 3.9e-05 / 1.2e-06 (M 4..8) | 2.9e-02 / 5.3e-03 / 1.2e-03 / 1.3e-04 / 2.1e-05 (M 4..8) | 5.2e-02 / 6.1e-03 / 3.5e-04 / 7.4e-05 / 2.9e-06 (M 4..8) |
| dirgen_gyro_c3 | 1.3e-05 / 1.9e-07 / 7.9e-10 / 5.2e-12 / 2.6e-13 (M 4..8) | 5.3e-04 / 4.8e-05 / 2.8e-06 / 1.6e-07 / 6.7e-09 (M 4..8) | 1.6e-04 / 1.7e-05 / 6.3e-07 / 2.7e-08 / 9.9e-10 (M 4..8) |
| dirgen_gyro_c5 | 4.0e-08 / 6.4e-11 / 3.5e-13 (M 4..6) | 9.1e-07 / 6.5e-09 / 6.2e-11 (M 4..6) | 3.0e-07 / 3.9e-09 / 1.9e-11 (M 4..6) |
| dirgen_syml_th3 | 2.3e-04 / 1.8e-05 / 8.2e-07 / 2.5e-08 (M 4..7) | 7.7e-04 / 7.7e-05 / 5.6e-06 / 4.6e-07 (M 4..7) | 3.8e-04 / 3.6e-05 / 2.5e-06 / 1.2e-07 (M 4..7) |
| dirgen_none_s+0.17-0.08 | 1.7e-14 / 1.7e-14 / 8.9e-14 / 5.3e-14 (M 4..7) | 6.0e-06 / 3.9e-08 / 1.5e-10 / 4.3e-13 (M 4..7) | 4.5e-07 / 1.3e-09 / 2.2e-12 / 5.3e-14 (M 4..7) |
| dirgen_th3_s+0.17-0.08 | 3.9e-04 / 3.0e-05 / 1.4e-06 / 4.2e-08 (M 4..7) | 8.3e-04 / 7.5e-05 / 5.7e-06 / 4.5e-07 (M 4..7) | 5.4e-04 / 5.1e-05 / 3.6e-06 / 1.7e-07 (M 4..7) |
| dirgen_sh4_s+0.17-0.08 | 8.9e-10 / 1.3e-11 / 8.8e-14 / 1.4e-13 (M 4..7) | 3.4e-07 / 6.5e-08 / 8.2e-10 / 6.1e-11 (M 4..7) | 4.1e-08 / 3.7e-09 / 2.2e-11 / 1.1e-12 (M 4..7) |
| dirgen_c3_s+0.17-0.08 | 5.7e-06 / 2.7e-07 / 2.7e-09 / 5.8e-10 (M 4..7) | 4.1e-04 / 5.7e-05 / 4.0e-06 / 2.2e-07 (M 4..7) | 8.9e-05 / 1.2e-05 / 7.8e-07 / 9.1e-08 (M 4..7) |
| gyrol_gyrol_none_s+0.17-0.08 | 6.7e-15 / 1.4e-14 / 1.6e-14 / 5.1e-14 (M 4..7) | 2.1e-05 / 1.3e-07 / 5.3e-10 / 1.5e-12 (M 4..7) | 1.6e-06 / 4.5e-09 / 8.1e-12 / 6.7e-14 (M 4..7) |
| gyrol_gyrol_th3_s+0.17-0.08 | 1.1e-03 / 8.6e-05 / 3.8e-06 / 1.2e-07 (M 4..7) | 2.0e-03 / 2.0e-04 / 1.5e-05 / 1.4e-06 (M 4..7) | 1.5e-03 / 1.5e-04 / 1.1e-05 / 5.3e-07 (M 4..7) |
| gyrol_gyrol_sh4_s+0.17-0.08 | 6.3e-10 / 8.5e-12 / 1.8e-14 / 2.1e-14 (M 4..7) | 3.3e-07 / 4.3e-08 (M 4..5) | - |
| gyrol_gyrol_c3_s+0.17-0.08 | 1.4e-05 / 2.4e-07 / 3.4e-09 / 4.2e-10 (M 4..7) | 6.3e-04 / 4.7e-05 / 3.2e-06 / 1.6e-07 (M 4..7) | 1.7e-04 / 1.5e-05 / 6.9e-07 / 7.9e-08 (M 4..7) |
| dirgen_gyro_none_s+0.17-0.08 | 3.0e-14 / 2.6e-14 / 1.9e-14 / 7.1e-14 (M 4..7) | 1.4e-05 / 8.8e-08 / 3.5e-10 / 9.8e-13 (M 4..7) | 1.3e-06 / 3.6e-09 / 6.4e-12 / 7.3e-14 (M 4..7) |
| dirgen_gyro_sh4_s+0.17-0.08 | 1.0e-09 / 1.6e-11 / 2.3e-13 / 1.9e-13 (M 4..7) | 4.3e-07 / 5.9e-08 / 7.4e-10 / 5.6e-11 (M 4..7) | 5.4e-08 / 4.9e-09 / 1.8e-11 / 1.7e-12 (M 4..7) |
| dirgen_gyro_th3 | 9.0e-04 / 7.3e-05 / 3.3e-06 / 1.0e-07 / 2.6e-09 (M 4..8) | 1.8e-03 / 1.8e-04 / 1.3e-05 / 1.0e-06 / 5.4e-08 (M 4..8) | 1.3e-03 / 1.3e-04 / 9.2e-06 / 4.4e-07 / 3.3e-08 (M 4..8) |
| lc_th3 | 6.2e-04 / 4.9e-05 / 2.3e-06 / 6.8e-08 / 1.7e-09 (M 4..8) | 1.2e-03 / 1.2e-04 / 8.2e-06 / 6.7e-07 / 3.5e-08 (M 4..8) | 8.9e-04 / 9.0e-05 / 6.3e-06 / 3.0e-07 / 2.2e-08 (M 4..8) |

(Rows with fewer rungs or a dash: the 4 x 4 shear and lossy-mu (QZ) runs that hit the 30-minute job cap on the saturated box; each wrote its finished rungs. The slanted slab under the 3 x 3 circle map at normal incidence was pushed further (`v4d_slant_circle.json`): 5.8e-10 / 8.3e-12 / 1.7e-12 at M = 7 / 8 / 9 against 2.3e-12 / 1.4e-13 / 3.8e-13 vertical -- tau ~ 1 / sg at the singular vertices slows it, it does not stop it.)

Every ladder is spectral to round-off (unmapped, `sh4`, `c5`, the circle at
normal incidence) or to the map's own resolution (the stretches, the circle
off normal), and the out-of-plane slab tracks the IN-PLANE LC film under the
same map rung for rung (`th2b` oblique M = 8: OOP 9.0e-6 vs LC 1.4e-5;
`th3` 2.3e-8 vs 3.5e-8; `c3` 6.8e-9 vs 6.2e-9) -- the residual is the map,
not the out-of-plane machinery.  The slanted magnetic slab, unmapped and
under the shear (the slant x mu path the magnetic build refused), is a
physical no-op and reads round-off by M = 7.

The fully populated mu with an out-of-plane block (`MU_FULL_OOP`) is refused
loudly everywhere (3.1).

---

## 6. E1-5 / E1-6 / E1-7 -- the verifier's own pillars

Pillar (own): period 1.1, r 0.33, depth 0.45, lambda 1, air / 1.5.  OOP
director 30 deg out of the plane, azimuth 30 deg (`n_o` 1.52, `n_e` 1.78).
Slanted circle: isotropic eps 3.5, public slant `+-0.15`.  The 36-vector of
R / T over nine orders and both polarisations.

### 6.1 The OOP pillar (E1-5)

`v5_summary.json` (36-vector distances):

| family | knob | distance | rung change / closure |
|---|---|---|---|
| c3 curved | M = 4 / 5 / 6 / 7 / 8 (to c3 M = 9) | 3.8e-2 / 5.2e-3 / 3.2e-3 / 1.6e-4 / 8.1e-5 | last step 8.1e-5; closure 3.6e-3 .. 8.1e-8 |
| c5 curved (independent topology) | M = 4 / 5 / 6 / **7** (to c3 M = 9) | 2.8e-4 / 4.0e-5 / 9.6e-6 / **3.7e-6** | steps 2.4e-4 / 3.0e-5 / 1.3e-5; closure 1.5e-6 .. 3.7e-11 |
| shipped OOP solver, staircase k = 1 (4 steps) | M = 4 / 6 / 8 (to c5 top) | 4.0e-2 / 1.27e-2 / 1.31e-2 | converged in M |
| same, k = 2 (8 steps) | M = 4 / 5 / 6 | 1.85e-2 / 1.94e-2 / 1.95e-2 | converged in M |
| same, k = 4 (16 steps) | M = 3 / 4 | 4.9e-3 / 4.3e-3 | |
| exact-disk tensor RCWA (own Bessel form factor, Laurent; patch called 9x per solve, form factor = shipped one to 9.3e-17) | n = 9 / 11 / 13 / 15 / 17 (N = 2n + 1 = 19 .. 35) | 4.2e-3 / 3.5e-3 / 3.0e-3 / 2.6e-3 / 2.3e-3 | distance x N = 0.080 / 0.081 / 0.082 / 0.082 / 0.081; local rates vs the curved answer 0.93 / 0.94 / 1.03 / 1.06 (five points: 1 / N); closure <= 1.9e-13 |
| same, Richardson in 1 / N | pairs (9, 11) .. (15, 17) | 3.1e-4 / 2.0e-4 / 9.6e-5 / 1.7e-4 | wanders: a 1e-4-class corroboration |
| conical (22, 38 deg) | c3 steps M = 6 / 7 / 8; c5 M = 5 -> 6; c3 M = 8 vs c5 M = 6; RCWA n = 9 / 13 | 9.3e-3 / 1.7e-3 / 2.6e-4; 1.0e-5; **4.8e-5**; 2.7e-3 / 2.0e-3 (x N 0.051 / 0.053) | c5 closure 1.5e-8 |

The two topologies agree to 3.7e-6 at normal and 4.8e-5 at conical
incidence; the exact-disk RCWA's envelope falls like 1 / N toward the curved
answer on five points (rate 0.93 .. 1.06), and its Richardson pairs wander
at the 1e-4 level, as BUILD_E1 recorded for its own pillar.  The staircases
converge in M to their own devices; k = 4 is the closest (4.3e-3), but on
this fixture k = 1 (the circumscribed square) sits closer than k = 2 (1.3e-2
vs 1.9e-2): BUILD_E1's "strictly decreasing toward the curved answer" is a
property of its fixture, not of the staircase family (no consequence).
**CONFIRMED** on the verifier's own pillar.

### 6.2 The slanted circle (E1-6)

Public slant +0.15 (`v5_summary.json`, `v5z_analysis.json`):

| family | knob | distance | notes |
|---|---|---|---|
| c3 composite | M = 4 / 5 / 6 / 7 / 8 (to c3 M = 9) | 6.7e-2 / 1.5e-2 / 6.1e-3 / 5.7e-4 / 1.9e-4 | closure 4.7e-3 .. 7.4e-7 |
| c5 composite | M = 4 / 5 / **6** (to c3 M = 9) | 5.5e-4 / 1.0e-4 / **2.7e-5** | steps 4.5e-4 / 7.7e-5; closure 5.5e-5 .. 4.3e-7 |
| shipped slant solver, in-plane staircase k = 1 / 2 / 4 | top M (to c5 M = 6) | 3.1e-2 / 2.2e-2 / 6.4e-3 | each converged in M to its own device |
| exact-disk RCWA z-staircase (`RCWAStack`, `shapes=` disks at `c + slant z_mid`) | N_z = 4 / 8 / 16 / 32 x n = 9 / 11 / 13 / 15 (13 points) | 5.3e-3 .. 3.1e-3 raw | distance x N flat at N_z = 8 (0.094 / 0.096 / 0.096 / 0.096, four points); rates of the N_z differences 1.85 / 1.94 (n = 9) and 1.81 / 1.92 (n = 11), four points each: second order in N_z |
| same, Richardson in 1 / N (pair 9, 11) then in 1 / N_z^2 | (4, 8) / (8, 16) / (16, 32) | **4.0e-4 / 3.2e-4 / 3.2e-4** | |
| same, a separable least-squares fit `v_inf + a / N + b / N_z^2` over all 13 points | | **3.3e-4** | fit residual 3.7e-5 |
| the opposite slant (-0.15) | c3 M = 8, c5 M = 6 | 3.1e-2 from +0.15 | **the exact x-mirror: 1.2e-13 / 1.6e-13**; the RCWA z-staircase obeys the same identity to 5.2e-14 |
| conical (22, 38 deg) | c3 steps M = 6 / 7 / 8; c5 M = 5 -> 6; c3 M = 8 vs c5 M = 6 | 8.2e-3 / 9.9e-3 / 5.9e-4; 6.8e-5; 4.2e-4 | the 3 x 3 layout is slow off normal (F-B6, as BUILD_E1 says): budget the 5 x 5 layout |

**Legitimacy of the extrapolation.**  In N_z the z-staircase is cleanly
second order (four points per order count, rates 1.8 .. 1.9 on the
differences), so the 1 / N_z^2 step is sound.  In N the envelope against
the curved answer is 1 / N to +-1 % on four points (N_z = 8), but the
self-convergence of consecutive rungs is not settled (difference rates 1.80
then 1.48 at both N_z = 4 and 8, where a clean 1 / N gives ~2.0) -- the
max-abs metric mixes components that converge at different rates.  The
double extrapolation therefore carries a 1e-4-class floor, and the three
estimators (two consecutive-pair chains and the global fit) all land at
3.2e-4 .. 4.0e-4 from the curved answer, while the two curved topologies
agree to 2.7e-5.  BUILD_E1's "within 2.2e-4" (on three points per axis, no
rate shown) is the same class; the honest statement is "the exact-disk
z-staircase corroborates the curved composite answer at the 3e-4 level; the
two curved topologies fix it at ~3e-5".  **CONFIRMED with that wording
note** (D-E1-4).

### 6.3 Reciprocity and the Onsager check (E1-7 / E1-8)

Conical (22, 38 deg), channel incidence -> reflected order, against its
reversal; singular values of the power-normalised Jones block
(`v5_recip_*.json`).  `nr` = the non-reciprocal twin (`e13 = conj(e31)`
with +0.22i: a magneto-optic tensor), `nrT` = its transpose = the same
medium with the MAGNETISATION REVERSED:

| geometry | order | reciprocal control (fwd vs rev) | non-reciprocal fwd vs rev | **Onsager: nr fwd vs nrT rev** | wrong pairing |
|---|---|---|---|---|---|
| slanted OOP disk, c3 M = 5 / 6 / 7 | (-1, 0) | 2.3e-4 / 5.0e-5 / 2.1e-6 | 6.3e-3 / 6.4e-3 / 6.4e-3 (flat) | **2.1e-4 / 4.8e-5 / 1.8e-6** | 6.4e-2 |
| same | (0, -1) | 5.3e-4 / 6.5e-5 / 1.1e-5 | 1.5e-2 / 1.5e-2 / 1.5e-2 (flat) | **5.2e-4 / 6.6e-5 / 1.2e-5** | 6.6e-2 |
| slanted OOP disk, c5 M = 5 | (-1, 0) / (0, -1) | 1.2e-6 / 4.0e-7 | 6.4e-3 / 1.5e-2 | **1.4e-6 / 2.0e-7** | 6.4e-2 |
| vertical OOP disk, c3 M = 6 | (-1, 0) / (0, -1) | 3.4e-5 / 6.0e-5 | 6.2e-3 / 1.4e-2 | 3.3e-5 / 6.1e-5 | 7.8e-2 |

The Onsager partner tracks the reciprocal control rung for rung on both
orders and both topologies while the non-reciprocal residual is flat to two
digits: the non-reciprocity is physics, and the out-of-plane entries enter
with the right index order and rotation-gauge sign under the composite
frame.  **Decisive, CONFIRMED.**

---

## 7. The parity accelerator under a map or with mu (F-E1-5)

`v6_parity_M5.json` (M = 5, 3 x 3 cells, OOP disk, normal incidence; region
eig, best of 2-3 on the loaded box):

| case | shipped | reduction LIFTED (gauge computed as for an unmapped, nonmagnetic cell) |
|---|---|---|
| unmapped C2-symmetric cell (the shipped accelerator) | engaged, **2.9x**, eigenvalues 5.9e-14, stack 5.8e-13 | -- |
| CENTRED circle map (C2-symmetric) | refused | structural residuals 3.5e-14 / 2.6e-16, ACCEPTED, eigenvalues **1.8e-14**, full stack **1.3e-13**, **2.3x** |
| OFF-CENTRE circle map | refused | refused by the gauge's own wall-parity test (the walls are not mirror-symmetric) |
| unmapped, lossless gyrotropic mu | refused | residuals 3.7e-15 / 8.1e-17, ACCEPTED, eigenvalues 4.7e-14, stack 7.0e-13, **3.1x** |
| unmapped, LOSSY gyrotropic mu | refused (`_bgen_hermitian` False -> QZ, 5.4x the whitened eig at this size) | structural residuals pass (4.3e-15), the sector Cholesky FAILS (not PD) -> falls back |
| centred circle map + lossy mu | refused | residuals pass, sector Cholesky fails -> falls back |

**Conclusion.**  For a C2-symmetric map and any LOSSLESS material the
refusal is a COST / validation-scope decision: the lifted reduction is
correct to round-off and 2.3-3.1x faster on the region eig.  For a LOSSY mu
it is a CORRECTNESS decision: the structural test passes (it checks `R B R =
B`, which a non-Hermitian `B` satisfies) and only the accident that the
sector Cholesky of the lower triangle is not positive definite for these
losses kept it from returning a wrong answer -- for a WEAK loss it would not
fail (section 8, `qz_off`).  The integration that re-enables it for
symmetric maps must keep the `_bgen_hermitian` gate in front of it.  Decision
test `test_ve1_parity_reduction_under_a_centred_circle_map_is_scope_not_correctness`.

---

## 8. E1-9 and the verifier's own mutations

Fixtures (`v7_mutations.py`; slabs read the own oracle, worst of R/T, Jr,
Jt; pillars the 36-vector change against the correct arm; M = 5 unless
noted): S1 / S2 sheared map, `dirgen`, normal / oblique; S3 sheared, `dirgen`
+ gyro mu, oblique; S4 asymmetric stretch, `gyrol` + lossy gyro mu, conical;
S5 unmapped, same; S6 / S7 sheared, slanted (0.17, -0.08), normal / oblique;
S8 3 x 3 circle, oblique, M = 6; S9 strong stretch, `dirgen` + aniso mu,
normal, M = 6; S10 / S11 symmetric lossy mu, unmapped conical / sheared
oblique; P1 OOP disk c3 normal; P2 eps-3.5 disk slanted 0.15 c3 normal; P3
OOP disk with a gyro mu in the disk, c3 normal.  (`v7_summary.json`.)

| kind | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| correct (none) | 9.6e-14 | 9.0e-10 | 9.8e-10 | 1.5e-04 | 4.5e-09 | 1.3e-11 | 6.5e-08 | 2.9e-06 | 1.2e-04 | 8.5e-10 | 9.2e-10 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| tau_J | 9.6e-14 | 9.0e-10 | 9.8e-10 | 1.5e-04 | 4.5e-09 | 2.8e-03 | 6.1e-03 | 2.9e-06 | 1.2e-04 | 8.5e-10 | 9.2e-10 | 0.0e+00 | 2.2e-02 | 0.0e+00 |
| tau_t | 9.6e-14 | 9.0e-10 | 9.8e-10 | 1.5e-04 | 4.5e-09 | 6.9e-04 | 2.2e-03 | 2.9e-06 | 1.2e-04 | 8.5e-10 | 9.2e-10 | 0.0e+00 | 1.0e-02 | 0.0e+00 |
| qz_off | 9.6e-14 | 9.0e-10 | 9.8e-10 | RAISES | RAISES | 1.3e-11 | 6.5e-08 | 2.9e-06 | 1.2e-04 | RAISES | RAISES | 0.0e+00 | 0.0e+00 | 0.0e+00 |
| chi33_unmapped | 7.4e-14 | 1.0e-03 | 8.7e-04 | 1.3e-02 | 4.5e-09 | 1.9e-07 | 1.5e-03 | 7.7e-03 | 1.2e-04 | 8.5e-10 | 7.9e-04 | 1.6e-02 | 4.5e-02 | 2.5e-02 |
| chi_unmapped | 3.6e-02 | 3.5e-02 | 5.6e-02 | 3.4e-01 | 4.5e-09 | 3.7e-02 | 3.7e-02 | 1.4e-01 | 9.5e-01 | 8.5e-10 | 5.0e-02 | 6.1e-02 | 7.2e-02 | 7.6e-02 |
| oop_over_sg | 1.4e-04 | 2.4e-04 | 1.3e-04 | 3.0e-03 | 4.5e-09 | 9.4e-04 | 2.7e-03 | 1.7e-03 | 1.1e-02 | 8.5e-10 | 1.3e-04 | 9.8e-03 | 1.7e-02 | 1.3e-02 |
| oop_adjT | 3.3e-04 | 1.3e-03 | 1.3e-03 | 1.5e-04 | 4.5e-09 | 2.7e-03 | 5.7e-03 | 2.9e-03 | 1.2e-04 | 8.5e-10 | 1.2e-03 | 2.3e-05 | 9.0e-04 | 6.2e-05 |
| double_rot | 9.6e-14 | 9.0e-10 | 9.8e-10 | 1.5e-04 | 4.5e-09 | 1.3e-11 | 6.3e-01 | 2.9e-06 | 1.2e-04 | 8.5e-10 | 9.2e-10 | 0.0e+00 | 5.0e-02 | 0.0e+00 |
| flux_R | 9.6e-14 | 9.0e-10 | 9.8e-10 | RAISES | RAISES | 1.3e-11 | 6.5e-08 | 2.9e-06 | 1.2e-04 | RAISES | RAISES | 0.0e+00 | 1.7e-15 | 0.0e+00 |
| mu_blocks_off | 3.6e-02 | 3.5e-02 | 7.1e-01 | 3.8e-01 | 4.5e-09 | 3.6e-02 | 3.8e-02 | 1.4e-01 | 6.8e-01 | 8.5e-10 | 4.7e-01 | 6.1e-02 | 6.6e-02 | 3.3e-01 |

(S columns: worst error against the oracle; P columns: the 36-vector CHANGE against the correct arm, so the correct row reads 0.)

Readings:

* **`tau_J`** (the Jacobian instead of its inverse in the composite weights)
  and **`tau_t`** are invisible to every vertical fixture and caught by the
  slanted slab (2.8e-3 / 6.9e-4 at normal) and the slanted pillar (2.2e-2 /
  1.0e-2).
* **`qz_off`** (the QZ fallback forced off): for every lossy mu of the fixture
  set (Im 0.02 .. 0.08) the Cholesky RAISES (`LinAlgError: not positive
  definite`) -- loud.  For a WEAK loss it is SILENT (`v7b_qz_weak.json`):
  the Cholesky reads only the lower triangle and returns an answer wrong by
  3.3 x Im(mu): 3.3e-2 at Im 1e-2, 3.3e-3 at 1e-3, 3.3e-5 at 1e-5, 3.4e-8 at
  1e-8, with nothing raised -- against 9.4e-10 (unmapped) / 1.9e-11 (shear)
  correct.  The structural flag `_bgen_hermitian` (1e-12 relative) is what
  stands between a weakly lossy permeability and a silently wrong answer;
  its threshold sits where the error would be below 1e-11.  NO builder test
  carries a weakly lossy mu; decision test
  `test_ve1_the_qz_fallback_is_load_bearing_for_a_weakly_lossy_mu` closes it.
* **`chi33_unmapped`** (mu blocks present but `chi33 = 1 / mu33`, the
  material's, without `sg`): invisible at normal incidence on a uniform slab
  (7.4e-14: the longitudinal curl is zero there) and on the unmapped slab;
  caught off normal (1.0e-3 oblique) and on every pillar (1.6e-2 .. 4.5e-2).
* **`chi_unmapped`** (the map's metric dropped from all permeability blocks)
  and **`mu_blocks_off`**: caught everywhere a map is, 3.6e-2 .. 0.95.
* **`oop_over_sg`** (the in-plane `1 / sg` wrongly applied to the out-of-plane
  entries) and **`oop_adjT`** (the transposed cofactor): caught on the shear
  and the stretch (1e-4 .. 1.1e-2); `oop_adjT` is nearly invisible on the
  circle pillar (2.3e-5 at c3 M = 5, below that rung's own error) -- only
  the non-symmetric `J` of the shear separates `adj(J)` from `adj(J)^T`.
* **`double_rot`** (the rotation sign applied a second time to a slanted
  mapped cell's pre-rotated weights): invisible on the slanted slab at
  normal incidence (1.3e-11), caught at oblique (0.63) and on the slanted
  pillar (5.0e-2).
* **`flux_R`** (the flux split whitened with B's E rows `-R` instead of the
  plain Gram -- the E1 change reverted): bit-for-bit HARMLESS on every
  lossless fixture (the split's sign does not depend on the whitening), and
  load-bearing only for a LOSSY mu, where it RAISES (`-R` is not Hermitian).
  The E1 change is right, and its only observable consequence is that the
  lossy-mu QZ branch works at all.

Which unit test ids catch each kind (`mut_logs/`, the builder's 19 E1 ids +
the verifier's decision tests, `-n 4`):

| kind | builder E1 ids that fail (of 19) | verifier ids that fail (of 7) |
|---|---|---|
| none | 0 (25 passed, 1 xfailed = the V-E1-1 pin) | 0 |
| `tau_J` | 2: `e1_6_slanted_uniform_slab_under_a_map_is_the_slab`, `e1_6_slanted_circle_lands_on_its_converged_answer` | 0 |
| `tau_t` | 2: the same two | 0 |
| `qz_off` | 1: `e1_3m_oop_eps_with_mu_matches_the_eps_mu_oracle[lossy-None]` (it RAISES there) | 2: `ve1_the_qz_fallback_is_load_bearing_for_a_weakly_lossy_mu` (the SILENT case), `ve1_the_permeability_side_of_B_stays_hpd...` |
| `chi33_unmapped` | 5: `e1_3_..._sheared_map`, `e1_3m[gyro-shear]`, `e1_5`, both `e1_6` | 1: `ve1_gauge_constants_are_map_independent_on_a_c2_broken_cell` |
| `chi_unmapped` | 9 | 1 |
| `oop_over_sg` | 9 | 0 |
| `oop_adjT` | 7 | 0 |
| `double_rot` | 3: `e1_2[slant1]`, both `e1_6` | 0 |
| `flux_R` | 1: `e1_3m[lossy-None]` | 0 |
| `mu_blocks_off` | 11 | 2 |
| `chitrap` (`_chi_*` raise under a map) on the 10 UNMAPPED suites | -- | **266 passed** |

Every engineered defect is caught by at least one builder id, with one
qualification: `qz_off` is caught by the builder's suite only because a
STRONG loss makes the Cholesky raise; the weak-loss SILENT case is caught
only by the verifier's decision test (section 8, `v7b_qz_weak.json`).

---

## 9. Docs

* BUILD_E1 numbers against the builder's JSON: **37 / 37** spot-checked
  numbers agree at the doc's two significant figures (`v8_docs.json`:
  E1-1, E1-3 shear / stretch / arms, E1-4, E1-5, E1-6, E1-7, E1-9, E1-10).
  Provenance note: the four `e{5,6,7,9}_summary.json` aggregation files
  record unpinned BLAS threads (the aggregators were run without the pins;
  they only read per-run JSON, which are all pinned -- 364 of 364).
* CHANGELOG `[Unreleased]` E1 entry: the code example executes verbatim on
  this tree (closure 1.0021 / 1.0007 at `n_modes = 4`); the cost bullet says
  "0.8-1.0x" Phase D's mapped in-plane region while BUILD_E1 3.11 says
  "0.6-1.0x" (the slanted mapped region, 13.5 s vs 22.8 s, is the 0.6) --
  harmless, recorded (D-E1-3).
* Scope notes (module docstrings of `twod_staggered`, `stack2d_pure`,
  `shapes2d`, the solver and Jones-entry docstrings): consistent with the
  measured scope (out-of-plane mu refused; per-layer maps and JAX are E2 /
  E3; the parity reduction not used under a map or with mu).  Two stale
  sentences: V-E1-1 (the refusal message, 3.1) and D-E1-2 (the
  `_region_modes_oop` docstring's item 1 still says `B = blkdiag(Ggram1,
  Ggram2, Ggram2, Ggram1)` is a block Gram, true only on the shipped branch).
* No forward version token in any E1-authored file (the `5.50` hits in the
  tree are the pre-existing deprecation horizon and earlier verifier docs
  quoting V-D6).
* `python -m mypy`: `Success: no issues found in 33 source files`.  WSL ruff
  0.15.16 on `lumenairy/ tests/ scripts/` and `verify_e1/`: `All checks
  passed!`.  `scripts/record_history_fingerprints.py --check`: `OK: every
  history document matches its module.`

---

## 10. Defects, with exact edits

No physics defect was found.  Four documentation defects, the first pinned
by a strict xfail:

**V-E1-1 (P3, wording; pinned by the strict xfail
`test_ve1_the_out_of_plane_mu_refusal_states_the_current_reason`).**  The
out-of-plane-permeability refusal still gives the pre-E1 reason (five of the
ten entry points route through it).  In
`lumenairy/elements/pmm/twod_staggered.py`, `_require_inplane_mu`:

* docstring, replace

      Magnetic anisotropy is IN-PLANE only in this engine: the paper's
      ``R = C[chi_t]C`` / ``K_tz = C[chi_t][d2;-d1]`` / ``S_tt(chi33)`` route
      generalizes the SECOND-ORDER pencil, while the out-of-plane FIRST-ORDER
      generator (:meth:`Granet2DTransverseE._assemble_oop`) carries no ``mu``
      blocks at all (its ``G3``/``E3`` eliminations assume ``mu = 1``).  So an
      out-of-plane ``mu`` raises, at the SAME RELATIVE ``1e-12 * scale`` floor

  with

      Magnetic anisotropy is BLOCK-FORM only in this engine: the paper's
      ``R = C[chi_t]C`` / ``K_tz = C[chi_t][d2;-d1]`` / ``S_tt(chi33)``
      operators serve the SECOND-ORDER pencil and (since Phase E1) the
      out-of-plane FIRST-ORDER generator, and both read ``chi_t =
      [mu_t]^-1`` and ``chi33 = 1 / m33``.  An out-of-plane ``mu`` would need
      the full 3 x 3 constitutive split (``chi_tt = (mu^-1)_tt`` and ``tau =
      -mu^{3t} / mu^{33}`` from the material), so it raises, at the SAME
      RELATIVE ``1e-12 * scale`` floor

* message, replace

      f"magnetic route generalizes the SECOND-ORDER (2 q^2) block-form "
      f"pencil (Granet Eq. 6: [[m11, m12, 0], [m21, m22, 0], "
      f"[0, 0, m33]]), and the out-of-plane first-order generator has "
      f"no mu blocks.  Pass a BLOCK-FORM mu.")

  with

      f"magnetic route takes a BLOCK-FORM mu (Granet Eq. 6: [[m11, m12, "
      f"0], [m21, m22, 0], [0, 0, m33]]) in both the in-plane pencil and "
      f"the out-of-plane first-order generator; an out-of-plane mu needs "
      f"the full 3 x 3 constitutive split (chi_tt = (mu^-1)_tt, tau = "
      f"-mu^{{3t}} / mu^{{33}}), which is not implemented.  Pass a "
      f"BLOCK-FORM mu.")

  and remove the `xfail` marker of the pinning test.

**D-E1-2 (wording).**  `_region_modes_oop` docstring, item 1: "``B =
blkdiag(Ggram1, Ggram2, Ggram2, Ggram1)`` is Hermitian positive definite
(it is a block Gram)" -> "on the shipped branch ``B = blkdiag(Ggram1,
Ggram2, Ggram2, Ggram1)`` is a block Gram; on the permeability branch (a map
or a material ``mu``, Phase E1) its E rows hold ``-R = C[chi_t]C``,
Hermitian positive definite for a pointwise HPD ``chi_t`` and NOT Hermitian
for a lossy ``mu``, which takes ``sla.eig`` (QZ) instead".

**D-E1-3 (wording).**  CHANGELOG E1 cost bullet "(0.8-1.0x: ...)" ->
"(0.6-1.0x: ...)", as BUILD_E1 3.11 reads (the slanted mapped region is the
0.6).

**D-E1-4 (wording).**  BUILD_E1 3.7 (c) and the CHANGELOG ("an exact-disk
RCWA z-staircase both approach it"): "lands within 2.2e-4 of it" -> "...
corroborates it at the 2e-4 .. 4e-4 level -- the RCWA's own extrapolation
floor: the slice extrapolation is cleanly second order, the order one is
1 / N in its envelope but not settled rung to rung"; and BUILD_E1 3.6 "the
staircases converge in M to different devices, strictly decreasing toward
the curved answer" -> "..., the 16-step one closest" (on the verifier's
fixture the 4-step staircase sits closer than the 8-step one).

---

## 11. Ship recommendation

**SHIP Phase E1** (after the wording fixes V-E1-1 and D-E1-2 .. D-E1-4,
none of which moves a number; V-E1-1 flips the strict xfail).  The
derivation was re-derived and every load-bearing step measured on the
verifier's own tensors, maps, oracle and pillars; both gauge constants are
map-independent for a reason that needs no map symmetry, and the decisive
same-device test on a C2-broken cell at normal incidence pins it; the
Onsager check with the magnetisation reversed is decisive for the slanted
out-of-plane curved pillar; no shipped byte moved on either build.  The one
gap the mutation matrix found -- a weakly lossy permeability with the QZ
fallback disabled returns a silently wrong answer -- concerns a guard the
build already has and gets right; the new decision test pins it.

---

## 12. What the integration must carry

1. **Keep `_bgen_hermitian` in front of every whitened path**, including a
   future parity reduction for symmetric maps: a non-Hermitian `B` passes
   the reduction's structural test (`R B R = B`), and a weakly lossy `mu`
   passes a lower-triangle Cholesky silently (error 3.3 x Im mu).  Order:
   the Hermitian gate first, then the parity gauge.
2. **The parity reduction for C2-symmetric maps is ready to enable for
   lossless materials**: lifted on the centred circle map it is exact
   (eigenvalues 1.8e-14, full stack 1.3e-13) and 2.3x faster on the region
   eig; the gauge's own wall-parity test already refuses a non-symmetric
   map.  It needs its own two-sided gate (lossless symmetric map on, lossy
   mu off, slant off); `test_ve1_parity_reduction_under_a_centred_circle_map_is_scope_not_correctness`
   is its starting point.
3. **Budget the 5 x 5 circle layout for slanted pillars off normal
   incidence** (c3 still moves 5.9e-4 per rung at M = 8 conical; c5 6.8e-5
   at M = 6).
4. **An out-of-plane permeability** needs `chi_tt = (mu^-1)_tt` and per-node
   `tau`, `kappa` from the material; the generator's `tau` / `kappa` slots
   already carry the structure, so the extension is in the weights only --
   and must be gated against an (eps, mu) oracle on a mu with `mu_t3 != 0`
   (`_ve1common.delta` already takes a full mu).
5. **The E2 per-layer maps** inherit the composite frame per layer: the
   frame-anchor phase and `_check_stack_slant` assume one map for the stack;
   a per-layer map under a slanted stack must re-derive the interface match
   (the covariant tangential fields of two different `J` at one `(u, v)`).

---

## 13. Not measured

* **E1-10 cost, re-measured**: the box ran at 90-100 % load with sibling
  agents for the whole verification; only the same-run ratios of section 7
  (parity lift 2.3-3.1x; QZ 5.4x the whitened eig at dimension 576) are
  reported.
* **The probe ladders on WSL**: the byte-identity set and the unit tests
  (the builder's 19 + the verifier's 7) ran on both builds; the convergence
  ladders are one build's (Windows).
* **The 4 x 4 sheared map at M = 8 and some mapped lossy-mu ladders past
  M = 7**: the first attempts hit the 30-minute job cap on the saturated box;
  section 5 reports what finished.
* **The exact-disk z-staircase at N_z = 16 / 32 with n >= 13 / 15**: each
  needs > 30 minutes here and was stopped; the grid used has 13 points (four
  per axis on both axes).
* **An independent full-wave oracle for the pillars** (3-D FEM with a tilted
  director): as for the builder, none is available; the references are two
  PMM topologies, the exact-disk RCWA and the staircases.
* **The same-device gauge test at M = 7**: stopped by the job cap; M = 4 ..
  6 show the decisive shape (converging vs flat).

---

## 14. Test tails

* Windows (CPython 3.14.6, numpy 2.4.4, scipy 1.17.1), the decision tests
  serial: `tests/unit/test_verify_pmm2d_curved_e1.py` -> `6 passed, 1
  xfailed in 12.14s` (`ve1_tests_serial_win.txt`; the xfail is the strict
  V-E1-1 pin); durations spliced into `.test_durations` (7 added).
* Windows, the regression sweep, `-n 6`, 71 files
  (`verify_e1/suite_files.txt`: the builder's 70-file list -- every
  `*pmm2d*`, `*stack2d*`, `*stagger*`, `*curved*` file incl. Phases A-E1,
  the A / B / C verifier files, census, public API, every walker, doc
  identifiers, history lint / relocation / fingerprint tool, kernel
  consistency, re-exports, except budget -- plus the Phase D verifier's file
  and this one), in two halves: `1418 passed, 1 skipped in 884.18s`
  (`suite_win_a.txt`) and `258 passed, 7 skipped, 3 xfailed in 776.23s`
  (`suite_win_b.txt`) -- **1676 passed, 8 skipped, 3 xfailed, 0 failed**.
  The 8 skips are the pre-existing premise gates (the mortar round-2 LAPACK
  premise; seven changelog walkers with nothing to verify in the `[5.49.0]`
  block); the 3 xfails are the Phase C verifier's strict pins `test_vc4` /
  `test_vc5` (E2's) and V-E1-1.
* WSL Ubuntu (CPython 3.12.3, numpy 2.4.6, scipy 1.17.1, BLAS pinned,
  `lumenairy` from `/mnt/c/tmp/lum_vcurved_e1`): the builder's E1 file and
  this one -> `25 passed, 1 xfailed in 103.47s` (`suite_wsl_e1.txt`).
* The mutation matrix (`mut_logs/summary.txt`, section 8) and the `chitrap`
  unmapped suites `266 passed in 713.96s`.
* `python -m mypy`: `Success: no issues found in 33 source files`
  (`mypy_win.txt`).  WSL ruff 0.15.16 on `lumenairy/ tests/ scripts/` and
  `verify_e1/`: `All checks passed!` (`ruff_wsl.txt`).
  `scripts/record_history_fingerprints.py --check`: `OK: every history
  document matches its module.`

Reproduction (BLAS pinned on every command line, `PYTHONPATH` the tree
measured):

```
cd /c/tmp/lum_vcurved_e1/validation/probe_pmm2d_curved/verify_e1
python v0_oracle.py                                    # the oracle, three ways
python v1_bytes.py <tree> <label> [trap|chitrap]; python v1_compare.py ...
python v2_derivation.py a|b 6 7 8|c c3 6 eps35|d       # 3.1 .. 3.4
python v3_gauge.py film <map> 6 | same <M> | pillar 5 2
sh runjobs.sh jobs_v4*.txt N ; python summ_v4.py        # section 5
sh runjobs.sh jobs_v5{a,b2,c}.txt N ; python summ_v5.py ; python v5z_analysis.py
python v6_parity.py 5 ; sh runjobs.sh jobs_v7.txt 3 ; python summ_v7.py ; python v7b_qz_weak.py
python v8_docs.py ; sh run_mutmatrix.sh <kinds> ; sh run_wsl_bytes.sh ; sh run_wsl_tests.sh
```

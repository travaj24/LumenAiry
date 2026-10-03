# VERIFY -- curved cells for the pure staggered 2-D PMM, Phase A (independent re-measurement of the map protocol, the quadrature assembly, the three traps and the cofactor far field)

Date: 2026-10-02.  Verifier: Claude Opus 5.5 (model ID `claude-opus-5-5`),
independent of the builder.
Object under verification: branch `feat/pmm2d-curved-cells`, commits
`5651bec4 .. 539ce4a3` on the plan commit `47c1c3a0`; build doc
`docs/audits/BUILD_PMM2D_CURVED_A_2026_10_02.md`; plan
`docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md` sections 2, 3, 4.1.
Mount: worktree `C:/tmp/lum_vcurved_a`, branch `verify/pmm2d-curved-a`.
PRE tree for byte identity: `git archive 47c1c3a0` extracted to
`C:/tmp/va_pre_47c` (outside the worktree).  Mutant trees: `git archive
539ce4a3` copies under `C:/tmp/va_mut/<mutant>` (the worktree's `lumenairy/`
was never edited).
Builds: (W) Windows 11, CPython 3.14.6, numpy 2.4.4, scipy 1.17.1; (L) WSL
Ubuntu, CPython 3.12.3, numpy 2.4.6, scipy 1.17.1.  `OMP_NUM_THREADS =
OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` on every command line;
`lumenairy.__file__` asserted under the tree being measured in every probe
(`_vcommon.py`, `VA_ROOT`).
Evidence: every number below is read from a JSON (or, for three small
diagnostics, a text) file in `validation/probe_pmm2d_curved/verify_a/`,
suffix `_win` / `_wsl` for the build.  Decision tests:
`tests/unit/test_verify_pmm2d_curved_a.py` (6 ids).

---

## 0. Words used here

* **Coordinate map** `(x, y) = Phi(u, v)`: a smooth distortion of the plane;
  the solver keeps a straight rectangular wall grid in `(u, v)` and the map
  places it in the physical `(x, y)` plane.  One map serves every layer and
  both half-spaces.
* **Jacobian** `J = d(x, y)/d(u, v) = [[x_u, x_v], [y_u, y_v]]`; **metric**
  `g = J^T J`; `sqrt(g)` denotes `det J`, the local area magnification.
  **Shear** of a map: a non-zero off-diagonal metric entry `g12 = x_u x_v +
  y_u y_v`, i.e. the `(u, v)` grid lines are not perpendicular in `(x, y)`.
* **Covariant components** `E' = J^T E`: the field components measured along
  the `(u, v)` grid lines.  **Effective tensors**: the permittivity and
  permeability the solver sees in `(u, v)`.
* **Separable stretch**: `x = f(u)`, `y = h(v)` -- no shear.  **Sheared map**
  (the verifier's, section 3.3): `x = u + cx sin(k u) F(v)`, `y = v + cy
  sin(k v) G(u)` on walls `0, p/2, p`; every wall stays straight (the sine
  vanishes on it) while every cell interior is sheared, so the PHYSICAL
  device is the straight-walled one and exact oracles still apply.
* **Preimage walls**: `(u, v)` walls placed at `f^-1` of the physical
  material walls, so the stretch moves resolution but not the device.
* **Mutant**: a deliberately broken copy of the source, used to show that a
  test can see the defect it guards; a mutant no test fails is a **survivor**
  (a gap in the test set).
* **Oracle self-gap**: the change of an oracle between two of its own
  resolutions -- the floor below which a comparison against it means nothing.

Verifier's fixture (independent of the build's): wavelength 1.3, square period
1.0, depth 0.4, air above, `n = 1.52` below, features `eps = 6.25`; stripe
`x in [0.20, 0.65]` (exact oracle `pmm_efficiency_1d`, `stabilize=False`,
degree 40; self-gap vs degree 48: TE 9.7e-13, TM 1.1e-06); pillar
`x in [0.20, 0.65], y in [0.25, 0.70]`; film (exact oracle: Airy slab).
Verifier's stretches: sine `a = 0.02, 0.08, 0.12 p`; `harm_sym` (two
harmonics, odd); `harm_asym` (`f = u + 0.10 p sin(ku) + 0.04 p [sin(2ku +
0.9) - sin 0.9]`, no mirror symmetry, `min f' = 0.154`).

---

## 1. Verdict table

| item | builder's claim | verifier's measurement (own fixture, W / L) | verdict |
|---|---|---|---|
| A1 no map = parent bytes | 109 / 109 SHA identical | **181 / 181** keys identical PRE vs tip on BOTH builds (generic `vars()` walk of 8 solver flavours incl. tensor, two magnetic, out-of-plane, slant; region modes incl. OOP with / without parity; geometric cache + 6 homogeneous mode sets; 3 far projectors + projections; 6 single-polarisation and 11 Jones cases incl. odd / even M, oblique, conical, lossy; 5 stacks incl. lossy with absorption, a DBR dedupe, tensor + magnetic + OOP mix, the per-layer mortar with absorption, oblique + conical) -- `v1_compare.json`.  Structural: with all seven mapped-only functions booby-trapped by a pytest plugin (verified to fire under xdist on a mapped test), the 37-file unmapped pmm2d / stack2d / staggered sweep passes 623 / 623 (1 pre-existing skip) | CONFIRMED |
| A2 identity map | operators <= 5.5e-14, R/T/Jones <= 2.6e-14 | operators <= 7.7e-14 over M = 4..7, integer and non-uniform walls, normal and oblique Bloch phases; end-to-end R / T / Jones <= 9.1e-13 (W) / 9.0e-13 (L) incl. conical (20, 35 deg) -- `v2_ident_*.json` | CONFIRMED (bar 1e-11 holds) |
| A2 one kernel? | "ONE function for all 18 blocks" | TRUE inside the mapped path; but the mapped and the unmapped assembly are TWO implementations of the same bilinear form (section 2) -- the 1e-14 agreement is structural (both are exact Gauss rules on polynomial integrands), not luck | FINDING F-V1 (design note) |
| formulation | `eps' = sqrt g g^-1 eps`, 18 blocks, cofactor far field, plain-Gram H partner and flux | re-derived from the covariant curl (section 3); every SEPARABLE block re-built from scratch by 1-D factors and matched to 2.2e-13 .. 1.3e-12 (oracle self-gap 6e-14 .. 1.6e-13) at M = 4, 6, 8 -- `v2_sep_oracle_*.json`; the SHEAR blocks (never exercised by the build) verified on a sheared film vs Airy: 1.2e-3, 6.8e-6, 7.6e-7, 2.6e-8, 4.2e-9 at M = 4..8 and a sheared stripe vs the 1-D oracle (TE 2.5e-6, TM tracks the unmapped rim floor 2.2e-4 at M = 10) -- `v3_*.json` | CONFIRMED |
| cofactor vs J^T | cofactor; no-cofactor 0.22 | on the film vs Airy (M = 7): cofactor 5.2e-8 (stretch 0.12 p, oblique 25 deg) / 2.3e-11 (normal) / 4.8e-7 (shear, oblique) / 2.6e-8 (shear, normal); J^T 0.40 / 0.39 / 0.21 / 0.10; J^-T (no area element) 0.40 / 0.62 / 0.16 / 0.11; identity 0.40 / 0.17 / 0.13 / 0.064 -- `v3_proj_arms_*.json` | CONFIRMED |
| A3 film under stretch | 1.29e-12 at M = 6 | asymmetric stretch, preimage walls: TM 2.4e-2 .. 1.1e-10 (M = 4..10), TE <= 2.3e-13; on a uniform `(u, v)` lattice (sine 0.12) round-off by M = 6 -- `v4_ladder_film_asym_*.json`, `v7_walls_film*.json` | CONFIRMED (spectral) |
| A4 stripe vs 1-D | 4.48e-4 at a = 0.05 p, M = 8 | TE at M = 10: sine 0.02 7.0e-8 (unmapped 5.3e-8), sine 0.08 6.7e-6, sine 0.12 9.6e-5, harm_sym 5.0e-5, harm_asym 9.4e-4; TM tracks the unmapped rim floor at every stretch -- `v4_ladder_*.json` | CONFIRMED (converges; a strong / asymmetric stretch costs 3-6 rungs on this higher-index fixture) |
| F1 adaptive nq | fixed 2M + 8 wrong by 5.2e-4 at 0.15 p | operator error vs the independent oracle at 2M + 8: 2.7e-3 (M = 4), 2.0e-3 (M = 6), 9.1e-4 (M = 8); adaptive: 4.2e-13 / 7.0e-13 / 9.2e-13.  R/T vs 1024 nodes (harm_asym, M = 6): 7.8e-5 at 2M + 8, then a FLOOR of 1.5e-8 .. 2e-7 (both builds, non-monotone) from nq = 40 on -- the floor is the mapped solve's conditioning, not quadrature (section 4.2) | CONFIRMED necessary and sufficient; picks 2x more nodes than the floor needs |
| 256 cap | caps at 256 with a warning | never fires on any separable stretch with `min f' >= 0.05`; fires at `min f' = 0.01` with a correct, actionable message; the EFFECTIVE cap is the largest `2^k (2M + 8) <= 256` (128..256: 144 at M = 5, 160 at M = 6, 192 at M = 8), not 256 | CONFIRMED with doc defect D5 |
| F2 validation | lattice periodicity of Phi - id AND of J | position periodicity is needed; Jacobian periodicity is NOT (the seam is an ordinary grid line, across which the normal Jacobian column may jump).  The shipped `validate()` REFUSES the planning circle map (normal-column jump 0.055 at the seam) that converges spectrally once the check is bypassed (film 5.3e-7, 1.8e-8, 2.7e-11, 2.2e-13 at M = 4..7) -- `v11_circle_*.json` | DEFECT D1 (blocks Phase B) |
| F3 oblique / conical | film 9.6e-8 at M = 7 | sheared film oblique 25 deg 4.8e-7 / 4.1e-7 / 1.3e-8 / 5.4e-9 at M = 7..10 (odd / even pairs, independent of `n_orders` 2..4 to 2 %, `v3b_oblique_floor_win.json`), conical (25, 40 deg) 1.2e-7 at M = 7; stretched STRIPE oblique 25 deg vs the 1-D oracle at that angle: sine 0.08 TE 1.5e-5 / TM 1.2e-4 (the unmapped rim floor 1.0e-4), harm_asym 2.3e-3 / 9.8e-4 at M = 9, converging; pillar conical vs the unmapped limit 2.3e-3 / Jones 1.7e-3 (M = 8) -> 1.1e-3 / 8.7e-4 (M = 9); y-momentum on the y-uniform stripe: power in `m_y != 0` orders equals the UNMAPPED solve's to a few % at every M (2.5e-10 -> 4e-25) -- the map adds no violation -- `v6_*.json` | CONFIRMED |
| A5 split | <= 7.2e-16 | <= 6.5e-16 on sine 0.12, harm_asym and the SHEARED map (mixed tensor entries up to 0.77 live) -- `v5_split_*.json` | CONFIRMED |
| A6 H Gram | 2.3e-6 vs mixed 0.157 | stripe / pillar (harm_asym, M = 8): correct closure 1.5e-3 / 1.6e-3 (the discretisation error of a harder fixture), MIXED 0.18 / 0.080 closure and 0.17 / 0.18 on R/T; CONSISTENT-wrong (every region through `-R`): R/T unchanged to 1.8e-13 -- the plan's claim that the trap is invisible unless the helpers mix is CONFIRMED -- `v5_hgram_*.json` | CONFIRMED |
| A7 absorption | 1.99e-5 at M = 7; `-R` flat 0.070 | lossy stripe / pillar: harm_asym 1.2e-3 / 8.7e-4 (M = 8), shear 1.4e-5 / 2.7e-5; `-R` flux Gram 0.020 / 0.137 / 0.019 / 0.020 -- flat on three arms, FALLING on the harm_asym stripe (0.126 -> 0.020), still 17x above the correct arm -- `v5_absorb_*.json` | CONFIRMED |
| magnetic mapped region | (Phase D; refused) | API refusal confirmed.  Composition `mu' = sqrt g g^-1 mu` INJECTED through `_stag_map_weights`: magnetic film (eps 2.0, mu 1.8) vs the exact (eps, mu) slab 1.5e-7 (harm_asym) / 6.4e-9 (shear) at M = 8; eps <-> mu duality on the stripe with vacuum half-spaces falls 0.21 -> 3.4e-3 (harm_asym) and 0.042 -> 2.9e-3 (shear) over M = 5..8 against an O(0.1-0.7) no-swap control -- `v5_mag_*.json` | COMPOSES (Phase D may proceed on this formula) |
| F5 walls | preimage walls 3-4 decades slower | reproduced (sine 0.12 film: uniform lattice 2.7e-13 at M = 6, preimage walls 2.3e-8); EXPLAINED: one preimage cell spans the whole compression; adding two walls INSIDE that cell gives 1.3e-7 / 1.4e-10 / 5.6e-14 at M = 4..6, better than the uniform lattice -- `v7_walls_film*.json` | CONFIRMED + trap D6 |
| multi-layer mapped stack | not measured | stripe / uniform / lossy pillar under harm_asym: absorption closure 5.8e-2 -> 1.7e-3 (normal) and 0.14 -> 1.3e-4 (conical) over M = 4..8; the two lossless layers absorb <= 2.7e-11; R / T approach the unmapped limit (8.3e-3 / 1.1e-2 at M = 8); a top vacuum spacer is a no-op to the discretisation level by both the patterned and the geometric-eig route; per-layer + map raises naming Phase E -- `v8_stack_*.json` | CONFIRMED |
| tests | 13 ids, every gate two-sided | 17 mutants: 10 survive the 13 ids (one of them equivalent); all 9 non-equivalent survivors caught by the 6 new decision tests (`v9_mutation_matrix.json`) | GAPS CLOSED (section 9) |
| docs | -- | SEE section 11 | minor defects D5, D7 |

---

## 2. One kernel or two (house rule: one implementation per kernel)

`_stag_quad_weighted` is ONE function for all 18 mapped blocks (the
flavours `m` / `d` / `dL` are a switch), as claimed.  But the weighted
bilinear form `INT INT W(u, v) X(u) Y(v) du dv` now has TWO implementations
in `twod_staggered.py`:

* the shipped Kronecker path (`_eps_weighted` via `_global_pair_segmat`, and
  `_eps_dir.segmat`), which multiplies per-segment reference matrices
  (`Basis1D.m_ref` / `c_ref`, themselves a `2M + 4`-node Gauss rule) by a
  per-CELL constant weight and krons the two axes;
* the quadrature path (`_stag_quad_axis_factor` + `_stag_quad_weighted`),
  which contracts per-NODE weights with per-node basis values.

Likewise the far field has two projectors (`_stag_fourier_projection` /
`_far_projector_2d` and `_far_projector_mapped`).  The 1e-14 agreement under
the identity map is STRUCTURAL: for a constant weight the integrand is a
polynomial of degree `<= 2M - 2` per axis and both paths are exact Gauss
rules for it, so they differ only by summation order -- not luck.  The
verifier's identity-map extraction (`v2_ident_*.json`) reads <= 7.7e-14 on
every retained operator at M = 4..7, odd and even, integer and non-uniform
walls, normal and oblique.

Assessment.  This is a specialisation (piecewise-constant weight) next to a
generalisation (variable weight), not a backend copy, and it is what keeps
A1's byte identity possible; folding the kron path into the quadrature path
would move every shipped answer by round-off.  The two are held together by
the build's A2 kernel test (all 18 flavour / set pairings, constant weight)
and, after this verification, by the long-cell projector test (section 9).
Recommendation: keep the pair, but (a) state the duplication and its
guard in the `_stag_quad_weighted` docstring, and (b) Phase B must not add a
third implementation (the circle needs a corner-graded RULE, not a new
contraction).

---

## 3. The formulation, re-derived

### 3.1 Effective tensors

Time convention `exp(-i omega t)`: `curl E = i omega mu0 mu H`, `curl H =
-i omega eps0 eps E`.  Take the 3-D map `r = (Phi(u, v), w)` with Jacobian
`Lambda = d r / d(u, v, w) = blockdiag(J, 1)`.  For a covariant field
`E' = Lambda^T E` the curl transforms as a vector density:

    curl'(Lambda^T E) = det(Lambda) Lambda^-1 (curl E)

(componentwise, `(curl' E')^i = eps^{ijk} d_j (Lambda^m_k E_m)`; the
symmetric second-derivative terms cancel and the remaining
`eps^{ijk} Lambda^l_j Lambda^m_k` is the cofactor `det(Lambda)
(Lambda^-1)^i_n` contracted with `eps^{nlm}`).  Hence

    curl' E' = det Lambda Lambda^-1 (i omega mu0 mu H)
             = i omega mu0 [det Lambda Lambda^-1 mu Lambda^-T] (Lambda^T H)
             = i omega mu0 mu' H',     mu' = det Lambda Lambda^-1 mu Lambda^-T

and the same for `eps'`.  With `Lambda = blockdiag(J, 1)` and a scalar
`eps`: `eps' = eps blockdiag(det J (J^T J)^-1, det J) = eps blockdiag(sqrt g
g^-1, sqrt g)`, `mu'` likewise with `mu = 1`, so `chi_t = [mu'_t]^-1 = g /
sqrt g`, `chi33 = 1 / sqrt g`.  Because the map is z-independent, both stay
block-form.  This is the build's `_stag_map_weights` entry for entry:
`e11 = eps g22 / sqrt g`, `e12 = e21 = -eps g12 / sqrt g`, `e22 = eps g11 /
sqrt g`, `e33 = eps sqrt g`, `c_ij = g_ij / sqrt g`, `c33 = 1 / sqrt g`
(Weiss et al., Opt. Express 17, 8051 (2009); the same transformation-optics
form in Ward & Pendry 1996).  For a material `mu` the same congruence gives
`mu' = sqrt g J^-1 mu J^-T` per block -- the formula the verifier injected for
the magnetic film and the duality check (section 5.2).

### 3.2 Which operators carry the metric -- the 18 blocks

Granet's second-order pencil `L E = gamma^2 (-R) E` with `R = C[chi_t]C`,
`L = [eps_t] + S_tt - K_tz Meps33^-1 K_zt`, `S_tt = -Curl^H Gw^-1 Gw_chi
Gw^-1 Curl`, `K_tz = C[chi_t][d2; -d1]`, `K_zt = div(eps_t .)` contains the
material tensors in exactly: `R` (4 blocks: chi22, chi21, chi12, chi11),
`[eps_t]` (4: e11, e22, e12, e21), `Gw_chi` (1: chi33), `K_tz` (4 terms),
`Meps33` (1: e33), `K_zt` (4 terms) = **18**; `Curl`, `Gw`, the directed
derivatives and the Bloch glue are metric-free (they are incidence
operators -- Kreeft, Palha & Gerritsma, arXiv:1111.4304).  The build's list
is right.  The verifier re-built every SEPARABLE block from scratch (1-D
weighted masses / derivatives from the basis definition, 2000-node rule) and
re-assembled `R`, `[eps_t]`, `S_tt`, Schur and `L` with the module's formulas:
agreement 2.2e-13 .. 1.3e-12 relative at the adaptive rule (M = 4, 6, 8;
oracle self-gap 6.1e-14 .. 1.6e-13), `v2_sep_oracle_*.json`.  The SHEAR
blocks (the `e12 / e21 / c12 / c21` weights, the cofactor's off-diagonal
entries) are zero on every map the build gated; the verifier's sheared film
(3.3) is the first measurement of them.

### 3.3 The far field: which density the tangential field is

The shipped kernel extracts `a_m = (1/A) INT E_t(x, y) exp(+i k_m . r) dx dy`
with Cartesian `E_t`.  The solver holds the covariant `E'_t = J^T E_t`, i.e.
`E_t = J^-T E'_t` (a 1-form), and `dx dy = det J du dv`, so the integrand
pulled back is `det J J^-T E'_t = cof(J) E'_t` -- a 1-form times an area
density; `cof(J) = [[y_v, -y_u], [-x_v, x_u]]`.  `J^T` would be correct only
for a contravariant field without the area element; `J^-T` alone forgets the
area element.  On a separable stretch `cof(J) = diag(y_v, x_u)` while `J^T =
diag(x_u, y_v)`: they differ at any non-trivial stretch.  Measured on the
film vs Airy at M = 7 (`v3_proj_arms_*.json`): cofactor 5.2e-8 (stretch 0.12
p, oblique 25 deg), 2.3e-11 (stretch, normal), 4.8e-7 (shear, oblique),
2.6e-8 (shear, normal); `J^T` 0.40 / 0.39 / 0.21 / 0.10; `J^-T` 0.40 / 0.62
/ 0.16 / 0.11.  The shipped choice wins by 6-10 decades in every arm, both
builds.

The flux used by `layer_absorption` is `INT (E_t x H_t*)_z dx dy`.  Under the
map `E_t x H_t = det(J^-T) (E'_t x H'_t)` and `dx dy = det J du dv`: the two
factors cancel, so the flux form is the PLAIN `(u, v)` Gram -- as built.  The
Eq.-25 H partner (`gamma C [H1; H2] = [k^2 eps_t + S_tt] E`) comes from the
transverse part of `curl H = -i omega eps E`, which carries `chi33` (inside
`S_tt`) but not `chi_t`, so it is tested with the plain Gram -- as built.

---

## 4. Stretches, the quadrature rule, and the cap

### 4.1 Ladders (A3 / A4 on the verifier's fixture)

TE (incident `E_y`) error of the stripe against the 1-D oracle,
`v4_ladder_*_win.json` (the WSL readings agree to the printed digits on the
arms run there):

| M | unmapped | sine 0.02 | sine 0.08 | sine 0.12 | harm_sym | harm_asym |
|---|---|---|---|---|---|---|
| 6 | 4.8e-4 | 3.5e-5 | 9.2e-3 | 2.7e-2 | 1.1e-2 | 3.2e-2 |
| 8 | 9.4e-7 | 2.6e-6 | 4.8e-5 | 9.1e-4 | 2.3e-3 | 6.4e-3 |
| 10 | 5.3e-8 | 7.0e-8 | 6.7e-6 | 9.6e-5 | 5.0e-5 | 9.4e-4 |

Every mapped arm converges to the oracle; TM (incident `E_x`) tracks the
unmapped solver's rim-limited floor (9.5e-5 at M = 10) at every stretch.  A
stretch only redistributes resolution, and on this higher-index fixture a
strong or asymmetric one costs three to six rungs -- consistent with the
build's "four or more at 0.15 p".  The asymmetric stretch is no worse in KIND
than the symmetric one (no symmetry assumption hides in the assembly).

### 4.2 Is the adaptive node count necessary and sufficient?

Operators against the independent separable oracle (`v2_sep_oracle_*.json`,
harm_asym, `min f' = 0.154`):

| M | 2M + 8 | 2 (2M + 8) | 4 (2M + 8) | adaptive (nodes) | 512 |
|---|---|---|---|---|---|
| 4 | 3.3e-3 | 4.4e-6 | 2.2e-11 | 4.2e-13 (128) | 4.3e-13 |
| 6 | 2.0e-3 | 1.9e-6 | 7.0e-13 | 7.0e-13 (80) | 6.6e-13 |
| 8 | 9.1e-4 | 5.6e-8 | 9.2e-13 | 9.2e-13 (96) | 8.6e-13 |

R / T of the stripe against a forced 1024-node rule (`v4_nq_M6_*`,
`v4_nq_M8_win`): M = 6: 7.8e-5 at 2M + 8, then 1.0e-7, 9.2e-8, 1.2e-7 (the
adaptive 80), 6.5e-8, 5.2e-8, 4.8e-8, 8.6e-8 for nq = 40 .. 512 (W); the WSL
readings of the same arms scatter 1.5e-8 .. 1.8e-7.  M = 8: 1.5e-6 at 2M + 8,
then 5e-10 .. 2.2e-7.  So: the fixed rule is NOT sufficient (necessary to
adapt); the adaptive rule IS sufficient; the floor below it is not
quadrature -- R / T move 2e-8 .. 1e-7 and the Jones matrix 2-5e-6 between
999 / 1000 / 1024-node rules whose operators agree to ~1e-13, while the
identity map moves 3e-14 (`v4c_sensitivity_win.txt`): under a strong
non-polynomial map the solve amplifies operator perturbations by ~1e6 at
M = 6 (consistent with the incident-overlap ill-posedness of F-V2, section
6: the noise sits in the incident-`E_x` column of the Jones matrix, whose
covariant form is the non-polynomial one, and shrinks 20x with fewer retained
orders; it falls with M), which sets the practical floor for any test bar on mapped
quantities at these sizes: R / T bars >= 1e-6, Jones bars >= 1e-5, each
measured on two builds.

### 4.3 The cap

On separable stretches the cap never fires down to `min f' = 0.05`
(`v4_cap_*.json`: nq 144 / 160 / 192 at M = 5 / 6 / 8, converged).  At
`min f' = 0.01` it fires, with the message "the coordinate map's geometric
weights are not resolved to the quadrature tolerance 1e-13 within 144 Gauss
nodes per axis per cell (moment error 7.30e-09); the operators carry a
quadrature error of that order.  The map varies too sharply inside a cell --
add walls where it is steep, or soften it." -- accurate there (R / T vs 1024
nodes 1.2e-6 / 5.8e-8 / 1.1e-11 at M = 5 / 6 / 8, against a discretisation
error of 0.11 / 0.037 / 0.010).  Note the numbers 144 / 160 / 192: the loop
doubles from `2M + 8` and stops when the NEXT doubling would exceed 256, so
the effective cap is the largest `2^k (2M + 8) <= 256` (defect D5, doc).

---

## 5. Traps, and a magnetic mapped region

### 5.1 A5 / A6 / A7 -- see the verdict table.  Two observations beyond it

* The consistent-wrong arm (every region, patterned and homogeneous,
  recovers H through `-R`) leaves R / T unchanged to 1.8e-13 on both
  builds: the H-partner trap is real only when helpers MIX, exactly as the
  plan says; the build's mixed fail-before is the right one.
* The `-R` flux-Gram fail-before is not always flat: on the harm_asym stripe
  it falls 0.126, 0.054, 0.061, 0.025, 0.020 (M = 4..8) while the correct arm
  falls 0.058 -> 1.2e-3.  The build's statement "a wrong bilinear form, not a
  discretisation error" holds for the pillar and the sheared arms (flat
  0.137 / 0.019) but not universally; A7's bar remains two-sided on the
  build's fixture, where the build measured it.

### 5.2 Every mapped region is already magnetic -- does a PHYSICAL mu compose?

The API refuses `mu` under a map (Phase D), confirmed.  The composition was
injected (`v5_traps.py mag`): `chi_t -> g / (sqrt g mu)`, `chi33 -> 1 /
(sqrt g mu)` for a scalar cell `mu`.  A uniform magnetic film (eps 2.0, mu
1.8) matches the exact `(eps, mu)` slab: harm_asym 2.3e-2 .. 1.5e-7, shear
1.8e-3 .. 6.4e-9 at M = 4..8 (TE row exact to 1e-13 at every M on the
stretch, as for the non-magnetic film).  Duality with vacuum half-spaces
(R / T of `E_y` on the eps-stripe vs `E_x` on the mu-stripe, order by order):
harm_asym 0.21, 0.047, 0.049, 3.4e-3; shear 0.042, 0.039, 4.0e-3, 2.9e-3 at
M = 5..8, against a no-swap control of 0.45-0.74 / 0.09-0.24.  The plain-Gram
H partner stays right with a real `mu` on top of the metric's, and Phase D's
formula composes.

---

## 6. Oblique and conical incidence on PATTERNED cells

`v6_stripe_oblique_*.json`, `v6_pillar_conical_*.json`: SEE the verdict
table.  The pulled-back incident phase is right: the zero-order Jones of the
mapped pillar at conical (25, 40 deg) approaches the unmapped limit (7.4e-2,
2.9e-2, 1.3e-2, 6.5e-3, 1.7e-3, 8.7e-4 at M = 4..9) and the oblique stripe
approaches the 1-D oracle at the same angle.  y-momentum on the y-uniform
stripe under an x-only map is conserved to the SAME level as the unmapped
solve on the same walls (the residual power in `m_y != 0` orders is
2.5e-10 / 1.8e-10 at M = 4 and below 1e-19 from M = 6, mapped / unmapped).

FINDING F-V2 (incident overlap, documented-behaviour gap, not a code bug).
The stack builds the incident modal vector as `cinc = lstsq(Hsup, delta_00)`
over the RETAINED orders.  Without a map (and under the identity map) the
incident plane wave is an EXACT discrete half-space mode and `cinc` is one
propagating mode: the evanescent share of `cinc` is 1e-14 (`v8c_incident`).
Under a non-polynomial map it is not (the covariant `E'_u = x_u E_x` is not
a polynomial): the evanescent share is 2.8e-5 / 7.5e-5 (sine 0.02, M = 5,
`n_orders` 2 / 3) and 11 % / 26 % (harm_asym, M = 5), falling to 1.1 % /
2.2 % at M = 7.  Consequences, both converging with M and both ZERO without a
map: (a) R / T / Jones depend on `n_orders` (harm_asym Jones 6.0e-4 between
`n_orders` 2 and 3 at M = 6, 5.6e-5 at M = 8; sine 0.02 8.8e-8 -> 5.3e-10;
unmapped 7e-16, `v4b_orders_win.txt`); (b) a vacuum spacer on top of the
stack -- physically a no-op -- moves R / T by 5.3e-6 -> 2.4e-10 (sine 0.02,
M = 5 -> 7) and 2.3e-2 -> 3.0e-3 (harm_asym), identically whether the spacer
is a patterned or a uniform layer (`v8b_vacuum_ref_win.txt`); identity map
1.9e-15.  It is not the dominant error: restricting the overlap to the
propagating modes leaves the TE stripe error unchanged to 1e-15 and moves TM
by less than the discretisation error (`v8d_incident_propagating`).  It
should be documented next to `cmap=` and re-measured by Phase B's gates.

---

## 7. F5 -- where the `(u, v)` walls go

Reproduced and explained (`v7_walls_film*.json`, uniform film, sine 0.12, TM
vs Airy):

| M | uniform 3 x 3 `(u, v)` | preimage walls (stripe) | walls added outside the wide cell | walls added INSIDE the wide cell |
|---|---|---|---|---|
| largest `max f' / min f'` in one cell | 2.82 | 6.32 | 6.32 | 4.43 |
| 4 | 2.7e-6 | 3.5e-4 | 3.5e-4 | 1.3e-7 |
| 6 | 2.7e-13 | 2.3e-8 | 2.3e-8 | 5.6e-14 |
| 8 | 8.4e-14 | 1.1e-12 | 1.1e-12 | 1.5e-14 |

For a film there is no material boundary; what matters is how much of the
map's variation one cell must resolve.  The covariant field of a plane wave
carries `f'(u)` and the metric weights carry `1/f'`; the stripe's preimage
walls put one cell across `u in [0.12, 0.77]`, i.e. the whole compression
(`f'` from 1.75 to 0.25), and a polynomial basis on one wide cell resolves
that slowly; walls added outside that cell change nothing, two walls inside
it beat even the uniform lattice.  For a PATTERNED cell the preimage walls
are not optional: they are what makes a material boundary a basis break in
`(u, v)`.

DEFECT D6 (user-facing trap, P3).  `SeparableStretch(u_walls=..., ...)`
takes `(u, v)` walls; passing the PHYSICAL walls there silently builds a
different device whose walls sit at `f(x_wall)`: the stripe at `[0.20,
0.65]` under sine 0.08 becomes `[0.276, 0.585]`, 0.081 off the intended
device's oracle and 5.8e-7 from the built one at M = 8
(`v7_walls_trap_*.json`).  It is documented in the class docstring only.
Phase C's primitives must take PHYSICAL geometry and own both the preimage
walls and the subdivision of steep cells; until then the docstrings of
`pmm_jones_2d_staggered(cmap=)` and `PMM2DStackPure(cmap=)` should point at
`SeparableStretch.from_physical_walls`.

---

## 8. A mapped stack with two patterned layers and a lossy layer

`v8_stack_*.json`: [stripe eps 6.25, d 0.25] / [uniform eps 2.1, d 0.15] /
[pillar eps 3 + 0.5i, d 0.2] under harm_asym on the shared grid.

| M | normal: sum A vs 1 - R - T | normal: R/T vs unmapped M = 9 | conical (20, 35 deg): sum A vs 1 - R - T | conical: R/T vs unmapped M = 9 |
|---|---|---|---|---|
| 4 | 5.8e-2 | 0.50 | 0.14 | 0.16 |
| 6 | 3.8e-2 | 5.6e-2 | 2.5e-3 | 7.8e-2 |
| 8 | 1.7e-3 | 8.3e-3 | 1.3e-4 | 1.1e-2 |

(The unmapped per-layer solve on the same physical walls sits 8.6e-5 / 5.2e-5
from its own M = 9 value at M = 8, so the mapped stack converges to the same
limit, slowly under this hard asymmetric map.)  The two LOSSLESS layers'
`layer_absorption` stays at 1.8e-14 .. 2.7e-11 at every M and angle -- the
plain flux Gram is right in every layer of a multi-layer mapped cascade, not
only the lossy one.  A vacuum spacer (patterned or uniform) on top is a no-op
to 1.9e-15 under the identity map and to the discretisation level under a
stretch (section 6, F-V2): the patterned and the geometric-eig routes give
the SAME numbers to the printed digits, so the two routes agree under a map
(`v8_stack_vacuum_*.json`, `v8b_vacuum_ref_win.txt`); the zero-order Jones
of the spacer stack equals the stripe's times `exp(+2 i k0 d)` to the same
level (the `exp(-2 i k0 d)` arm misses by 0.81-0.86).

`layer_grids='per-layer'` + `cmap` raises `NotImplementedError` naming Phase
E ("per-layer grids under a map need a curved (non-separable) mortar between
the layers' partitions, which is Phase E of the curved-cell plan"),
`v8_stack_refuse_*.json`.

---

## 9. The tests: mutation matrix

17 source mutants (`v9_mutants.py`), each run against the 13 build ids and
the 6 verifier ids (`C:/tmp/va_mut/log13_*.txt`, `logV_*.txt`):

| mutant | what it breaks | build ids failing | verifier ids failing |
|---|---|---|---|
| m01 cofactor -> J^T | far field | a3, a4, a6 correct, a7 | sheared_film |
| m01b cofactor off-diagonals swapped | shear far field | **none** | sheared_film (0.114) |
| m02 half-space H partner through -R | trap H | a3, a4, a6 correct, a7 | sheared_film |
| m02b patterned layer H partner through -R | trap H | a3, a4, a6 correct, a6 fail-before | sheared_film |
| m02c flux Gram -R | absorption | a7 | -- |
| m03 nq pinned to 2M + 8 at the CALL SITE | F1 | **none** | independent_oracle (3.3e-3) |
| m03b nq function returns 2M + 8 | F1 | quadrature_is_adaptive | independent_oracle |
| m03c moment test on sqrt g only | F1 | quadrature_is_adaptive | independent_oracle |
| m04 sign of eps'_12 flipped | shear tensor | **none** | sheared_film (0.543) |
| m04b sign of chi_12 flipped | shear tensor | **none** | sheared_film (0.364) |
| m04c chi33 = sqrt g | S_tt | a4, a6 correct, a7 | independent_oracle (0.77) |
| m05 fingerprint ignores curve parameters | cache key | **none** | fingerprint_sees_the_curve_parameters |
| m05b per-solve eig key ignores the map | cache key | none | none -- EQUIVALENT (the dedupe cache lives for one solve with one map; nothing persistent consumes the fingerprint in Phase A) |
| m06 off-diagonal projector blocks dropped | shear far field | **none** | sheared_film (5.1e-4) |
| m07 solver's det J <= 0 refusal removed | protocol | **none** | solver_refuses_a_folding_map |
| m08 boundary-onto-itself check removed | protocol | **none** | validation_refuses_a_translated_map |
| m09 far projector at fixed 2M + 16 | projector rule | **none** | projector_keeps_the_phase_rule_on_a_long_cell (2.96e-5) |

Every non-equivalent survivor is now caught by a named id.  The six decision
tests take 0.01-9.5 s each (spliced into `.test_durations`); their bars,
measured on both builds: sheared film 2.64e-8 (W) / 2.66e-8 (L) under 1e-6;
independent oracle 4.26e-13 under 1e-10; long-cell projector 1.5e-15 under
1e-11.  The verifier did not mutate m05b further: once Phase B/C add a
persistent cache keyed by `fingerprint`, a cache-staleness test is needed.

---

## 10. Cost, and the regression sweeps

### 10.1 Cost

`v10_cost_*.json`: a fresh process per configuration builds one solver
(assembly, best of two) and runs its region eig; the verifier's 3 x 3 stripe
cell.  Windows readings (the box carried two other single-threaded probes
and a WSL batch at the time, so these are upper bounds):

| M | assembly: none / identity / sine 0.08 / harm_asym (nq) | region eig | peak RSS: none -> mapped |
|---|---|---|---|
| 6 | 0.075 / 0.092 / 0.103 / 0.119 s (20 / 40 / 80) | 1.9-2.2 s | 122 -> 129-144 MB |
| 8 | 0.46 / 0.48 / 0.41 / 0.43 s (24 / 48 / 96) | 20-26 s | 234 -> 260-281 MB |
| 10 | 1.24 / 1.26 / 1.10 / 1.13 s (28 / 56 / 112) | 52-85 s | 493 -> 566-597 MB |

Assembly costs 1.0-1.7x the shipped kron assembly on the two builds (the
build's 1.1-1.8x) and stays 1-6 % of the region eig; the eig is unchanged in size and its time
scatters with load, not with the map.  Peak resident memory grows 6-21 %:
the ten per-node weight arrays `(Nx, Ny, nq, nq)` -- five complex
(permittivity), five real (inverse permeability), 120 bytes per node, 14 MB
at nq = 112 on a 3 x 3 grid -- and the quadrature caches.  At the cap
(nq = 256) the weights alone are 71 MB on a 3 x 3 grid (the source comment's
"~94 MB" counts all ten as complex) and scale with `Nx Ny nq^2`: ~200 MB on
a 5 x 5 circle grid at the cap -- one more reason for D2.  WSL (`v10_cost_wsl.json`, a quieter box): assembly 0.049 / 0.059 /
0.081 / 0.083 s (M = 6), 0.25 / 0.29 / 0.30 / 0.31 s (M = 8), 0.94 / 1.05 /
1.05 / 1.18 s (M = 10) -- 1.1-1.7x; region eig 0.9-1.0 / 7.6-8.4 / 41-60 s;
peak RSS 102 -> 109-123, 220 -> 247-263, 503 -> 568-585 MB.

### 10.2 Sweeps

* The 13 build ids + the 6 verifier ids: Windows -- both files inside the
  50-file sweep below, all passed (the verifier's file alone: 6 passed in
  33 s); WSL -- 19 passed in 310 s (`wsl_pytest_curved.txt`).
* The regression sweep (Windows, xdist 8 workers, `--capture=sys -p
  no:randomly`): the 45 pmm2d / stack2d / staggered / census / public-API /
  doc-consistency / except-budget / history files plus
  `test_audit2609_a21_doc_identifiers`, `test_audit2609_a21_pmm_warning_filter`,
  `test_audit2609_b6_pmm_basis_and_tensor_cache`, `test_niche_c1_consolidation`
  and `test_pmm_per_layer_grids` -- 50 files: **1541 passed, 1 skipped** (the
  same pre-existing skip as in the booby-trapped run) in 26 min.
* The booby-trapped unmapped sweep (37 files): 623 passed, 1 skipped, no
  `VA-BOOM`.
* WSL ruff 0.15.16: `lumenairy/elements/pmm/`, both curved test files and
  every probe in `verify_a/` clean.  `python -m mypy`: no issues (33 files;
  see section 11).  `scripts/record_history_fingerprints.py --check`: OK.

---

## 11. Docs

* BUILD doc numbers vs its JSON: A1 (109 / 109), A2 (kernels <= 1.6e-14,
  operators <= 5.5e-14, eig 4.3e-12, the 1e-6 p gap 7.8e-6 .. 1.0e-5), F1
  (5.2e-4 / 7.8e-5 / 5.3e-6 and 0.0 / 0.0 / 3.1e-11), A5, A6 and the cost
  table all match their JSON.  One slip: "far projector <= 4.5e-15" -- the
  oblique arm of the same JSON reads 5.2e-15 (D7, trivial).
* CHANGELOG `[Unreleased]` entry: every claim matches a measurement.
* `_curvemap.py` read as an optical physicist: Jacobian, metric, covariant
  components, effective tensors, lattice periodicity, fingerprint, `det J`,
  the stretch's 33:1 range are all defined in plain words.  Undefined or
  internal: "variable-coefficient (quadrature) assembly" and "the magnetic +
  tensor route" (IdentityMap docstring) are solver-internal phrases; the
  `_stag_map_nodes` docstring's "Legendre moments" is undefined (the
  integrals of the weight against Legendre polynomials).  And the module's
  periodicity paragraph states the Jacobian must be periodic -- wrong per D1.
* No forward version token in the `lumenairy/` diff.
* `python -m mypy`: "Success: no issues found in 33 source files" -- but the
  strict file list does not include `lumenairy/elements/pmm/`, so the new
  code is not type-checked (pre-existing scope, noted).
* WSL ruff (0.15.16) on `lumenairy/elements/pmm/` and both curved test
  files: clean.  `scripts/record_history_fingerprints.py --check`: OK;
  `scripts/check_doc_identifiers.py`: OK.

---

## 12. Defects, with reproducers and the exact edits requested

**D1 (P2 -- blocks Phase B).  `CellMap.validate` requires a periodic
Jacobian; the solver needs only a periodic POSITION.**
Reproducer: `v11_circle.py nodes` -- the planning circle map (position and
tangential Jacobian periodic to 0.0, normal column jumping 0.055 / 0.062 at
the `u`-seam) is refused: "the map must be lattice-periodic ... the
Jacobian mismatch 5.503e-02"; with the check bypassed its film converges
spectrally (5.3e-7, 1.8e-8, 2.7e-11, 2.2e-13 at M = 4..7).  The seam is a
grid line like any other: `E'_v` (continuous across `u = const`) needs
`d Phi / dv` periodic, which position periodicity already gives; `E'_u` may
jump.  Requested edit (`_curvemap.py`, `_check_periodic` and its two
callers): compare the positions and only the TANGENTIAL Jacobian column --
for the `u`-seam `(x_v, y_v)` = indices 3, 5 of `geom`; for the `v`-seam
`(x_u, y_u)` = indices 2, 4 -- and drop the normal column from `djac`; fix
the module docstring ("and the Jacobian is periodic" -> "and so are the
derivatives ALONG the seam; the derivatives across it may jump, as across any
grid line") and BUILD F2.  Add a test: the circle map (or any map whose
normal derivative jumps at the seam) validates and its film converges.

**D2 (P2 for Phase B, P3 now).  The adaptive node rule's criterion fails on
maps with singular corners, warns falsely, and costs an order of magnitude.**
Reproducer: `v11_circle.py nodes|film`.  On the circle map the moment test
never converges (moment error 1.2-2.9 relative at M = 4..8), the cap fires
at every M with "the operators carry a quadrature error of that order"
(i.e. O(1)) -- false: the adaptive and the planning `2M + 8` rules give the
same film error to 2-4 % (5.26e-7 / 5.09e-7, ..., 2.2e-13 / 2.6e-13) -- and
the node search alone costs 15.8 s at M = 4 (moments up to 512 nodes), a full
solve 42.7 s vs 3.0 s.  Requested for Phase B: replace the moment criterion
on singular-corner maps (e.g. converge the ASSEMBLED blocks, or the moments of
the bounded products `sqrt g x weight`, or grade the rule toward the singular
vertex), and make the warning state the measured operator change rather than
the raw moment error.

**D3 (P3).  `_init_map` runs the node search before the det J check.**
A folding map first doubles to the cap with a misleading quadrature warning
("moment error 3.39e-02 within 256 nodes"), then raises "det J <= 0".
Requested edit: evaluate `det J` on the base rule (`2M + 8` nodes) before
`_stag_map_nodes`, or have `_stag_map_nodes` raise on `sg <= 0`.

**D4 (P3, documentation).  The incident overlap under a non-polynomial map**
(F-V2, section 6): document next to `cmap=` that R / T depend on `n_orders`
and on the reference plane at the discretisation-error level under a map;
Phase B's gates should measure `n_orders` independence at their gate M.

**D5 (P3, doc).  "Capped at 256"** in the `_stag_map_nodes` docstring and
BUILD 4.2: the effective cap is the largest `2^k (2M + 8) <= 256` (144 at
M = 5, 160 at M = 6, 192 at M = 8), and the search computes moments at up to
twice that.  Requested: state it.

**D6 (P3).  Wall trap** (section 7).  Requested: point the `cmap=`
docstrings at `from_physical_walls`; Phase C primitives own the walls.

**D7 (trivial).**  BUILD A2 "far projector <= 4.5e-15" -> 5.2e-15.

---

## 13. Ship recommendation, and what Phase B must carry

**Recommendation: SHIP Phase A to the feature branch, with the six decision
tests, and with D1 fixed before Phase B begins.**

* Nothing shipped moves: 181 / 181 own keys byte-identical to the parent on
  both builds, and the unmapped 37-file sweep passes with every mapped-only
  function booby-trapped.
* The formulation is right, and now verified on the parts the build could not
  reach: the shear terms (sheared film vs Airy, spectral to 4.2e-9; sheared
  stripe vs the 1-D oracle), the independent separable-factor oracle for all
  18 blocks (<= 1.3e-12), the cofactor against J^T / J^-T at oblique
  incidence, a magnetic mapped region by injection, multi-layer mapped
  stacks.
* D1 is not reachable by either shipped map (identity and separable stretch
  both have periodic Jacobians), so it does not block Phase A; it blocks the
  first transfinite map Phase B adds.  The requested edit is four lines.
* D3-D7 are small and may ride with Phase B's first commit.

What Phase B must carry:

1. D1: validate position periodicity and the TANGENTIAL Jacobian column only;
   a test with the planning circle map (whose normal column jumps 0.055 at the
   seam) validating and converging.
2. D2, the circle's 1/r corner weights vs the adaptive rule and the 256 cap:
   the moment criterion cannot converge on them (moment error 1.2-2.9 at every
   M), so as shipped EVERY circle solve hits the cap, prints a warning that
   claims an O(1) operator error, and pays 5-14x in the node search -- while
   the planning rule `2M + 8` gives the same film error (2.2e-13 vs 2.6e-13
   at M = 7).  Phase B must choose the circle's rule (corner-graded, or a
   criterion on the assembled blocks) and re-measure; the warning must report
   a measured operator change.
3. The shear terms are live on every circle / fillet map: keep the verifier's
   sheared-film decision test and add the circle film at oblique incidence
   (gate B10).
4. F-V2: measure `n_orders` independence and a vacuum-spacer no-op at each
   Phase-B gate M; document the behaviour next to `cmap=`.
5. Test bars on mapped R / T must sit above the mapped solve's noise floor
   (R / T 1.5e-8 .. 2e-7 and Jones ~1e-6 between converged quadrature rules at
   M = 6 on a strong map, build-dependent -- section 4.2).
6. F5 / D6: cells must not span a steep stretch of the map; the circle's
   3 x 3 grid puts the whole arc in one cell per side -- measure a subdivided
   grid; Phase C primitives own walls and subdivision.
7. When a persistent cache keyed by `fingerprint` appears, add a staleness
   test (m05b is equivalent today only because the dedupe cache lives for one
   solve).
8. One contraction: no third implementation of the weighted bilinear form or
   of the far projector (section 2).

---

## 14. What was not measured

* A second build for: the oblique / conical stripe and pillar (v6), the
  multi-layer ladder (v8 ladder), the incident-overlap diagnostics (v8c,
  v8d, v4b), the oblique floor ladder (v3b), the circle film (v11 film), the
  sine 0.02 / 0.12 / harm_sym ladders, and the M = 9, 10 rungs of every
  ladder -- Windows only.  Every other probe ran on both builds and agrees
  to the printed digits (section 1).
* The circle / fillet maps under the library beyond the film and the node
  count (Phase B's scope); no FEM oracle was run.
* A material `mu` under a map through the public API (refused; measured only
  by injection, section 5.2).
* Peak memory at M >= 12 (not attempted: the user's own processes held part
  of the RAM and the plan's sizes stop at M = 10).

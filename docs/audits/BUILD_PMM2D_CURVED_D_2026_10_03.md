# BUILD -- curved cells for the pure staggered 2-D PMM, Phase D (anisotropic and magnetic materials inside curved cells)

Date: 2026-10-03.  Status: BUILT + gated, not pushed.
Builder: Claude Opus 5.5 (model ID `claude-opus-5-5`).
Mount: worktree `C:/tmp/lum_curved_d`, branch `feat/pmm2d-curved-d`, built on
`607d43b0` (Phase C) with the Phase B verifier branch merged
(`verify/pmm2d-curved-b`, `5d2272fd`, as merge `074226d2`); Windows 11
(tesla-ryzen), CPython 3.14.6, numpy 2.4.4, scipy 1.17.1,
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` on the command
line, `lumenairy.__file__` asserted under the tree being measured in every
probe (`build_d/_common.py`; the BEFORE arm sets `LUM_TREE`).  PRE tree for
byte identity: `git archive 607d43b0` extracted to `C:/tmp/curved_pre_d/`.
The box was saturated for the whole build (24 to 30 single-threaded Python
processes on 24 logical cores, 100 % CPU, the maintainer's own scripts and
two sibling verifiers), so every WALL TIME below is an upper bound, several
times the idle figure; no accuracy number depends on the load.
Plan: `docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md` section 4.4,
approved with the planner's recommendations on all seven questions of
section 7.  Phases A, B, C: `docs/audits/BUILD_PMM2D_CURVED_{A,B,C}_*.md`;
Phase A verifier: `docs/audits/VERIFY_PMM2D_CURVED_A_2026_10_02.md`; Phase B
verifier: `docs/audits/VERIFY_PMM2D_CURVED_B_2026_10_02.md` (folded in here,
section 6).
Evidence: every number below is read from a JSON file in
`validation/probe_pmm2d_curved/build_d/` (the probe that wrote it is named in
each table; `_dcommon.py` holds the shared fixtures, the Berreman film oracle
and the engineered mutations).  Tests:
`tests/unit/test_pmm2d_staggered_curved_d.py` (19 tests).

---

## 0. Words used here

* **Coordinate map** `(x, y) = Phi(u, v)`, Jacobian `J = [[x_u, x_v], [y_u,
  y_v]]`, `sg = det J` (the local area magnification), metric `g = J^T J`.
  `adj(J) = sg J^-1 = [[y_v, -x_v], [-y_u, x_u]]`.
* **Covariant fields** `E' = J^T E`, `H' = J^T H`: the components along the
  `(u, v)` grid lines, the solver's unknowns under a map.
* **Block-form tensor**: `[[e11, e12, 0], [e21, e22, 0], [0, 0, e33]]`
  (in-plane anisotropy; the solver's second-order pencil).  An
  **out-of-plane** tensor has `e13`, `e23`, `e31` or `e32` non-zero.
* **LC**: a rotated in-plane uniaxial tensor (n_o 1.5, n_e 1.8, director in
  the plane; at 0.55 rad from x in the films -- the shipped G3 tensor -- and
  at 30 degrees, **LC30**, in the pillars).  **GYRO**: the gyrotropic tensor
  `[[2.25, 0.5i, 0], [-0.5i, 2.25, 0], [0, 0, 2]]` (Hermitian, lossless,
  `e12 = -e21`); its transpose is the opposite magnetization.
* **Maps used.**  `s05` / `s15`: a separable sine stretch of both axes
  (`x = u + a sin(2 pi u / p)`, `y = v - (a/2) sin(2 pi v / p)`, `a = 0.05 p`
  / `0.15 p`; at `0.15 p` the local stretch spans 33:1).  `shear`: a 3 x 3
  transfinite map with straight edges and the four interior vertices moved
  by 3-6 % of the period -- bilinear cells with `g12 != 0` and a
  NON-DIAGONAL `J`.  `c3` / `c5`: the 3 x 3 and 5 x 5 circle maps of Phase B
  (four singular vertices, the corner rule).
* **Fail-before / mutation**: a deliberately broken arm reached through the
  real code path (`_dcommon.mutate`: the ONE congruence kernel, the weight
  routine or the H-partner path patched with one specific defect and
  restored afterwards).

---

## 1. The formulation: the effective tensors of a block-form material

### 1.1 Derivation

Time convention `exp(-i omega t)`: `curl E = i omega mu0 mu H`,
`curl H = -i omega eps0 eps E`.  The 3-D map `r = (Phi(u, v), w)` has the
Jacobian `Lambda = blockdiag(J, 1)` (the map does not depend on `z`).  For
covariant components `E' = Lambda^T E` the curl transforms as a vector
density: `curl'(Lambda^T E) = det(Lambda) Lambda^-1 (curl E)` (the Phase A
verifier's section 3.1 writes the index computation out).  Hence

    curl' E' = det Lambda Lambda^-1 (i omega mu0 mu H)
             = i omega mu0 [det Lambda Lambda^-1 mu Lambda^-T] (Lambda^T H)

and the same for `eps` -- the transformation-optics congruence (Ward &
Pendry, J. Mod. Opt. 43, 773 (1996); Weiss et al., Opt. Express 17, 8051
(2009), Eqs. 7-12; Kuchenmeister, Opt. Express 22, 1342 (2014), for
anisotropic media in curvilinear coordinates):

    eps' = det Lambda  Lambda^-1  eps  Lambda^-T,      mu' likewise.

With `Lambda = blockdiag(J, 1)` and a BLOCK-FORM `eps` the result stays
block-form (no `w` dependence, no `e13` created):

    eps'_t  = sg J^-1 eps_t J^-T = adj(J) eps_t adj(J)^T / sg
    eps'_33 = sg eps_33
    chi_t   = [mu'_t]^-1 = J^T [mu_t]^-1 J / sg
    chi_33  = 1 / (sg mu_33)

Entry by entry `eps'_ij = adj_ik eps_kl adj_jl / sg`: `J^-1` acts on the ROW
index and `J^-T` on the COLUMN index of `eps`.  For a symmetric tensor the
order is immaterial; for a GYROTROPIC one (`eps != eps^T`) swapping the two
sides, or transposing `eps`, reverses the gyration -- which is why the gate
for it reads the Jones matrix (2.2).  For a scalar `eps` the formula is
`eps adj(J) adj(J)^T / sg = eps adj(g) / sg = eps sg g^-1`, the scalar
route of Phases A-C; for vacuum `mu = 1` it gives `chi_t = g / sg`,
`chi_33 = 1 / sg`, Phase A's metric weights.

`chi_t` is the POINTWISE inverse of `mu'` at every quadrature node, taken
before the quadrature.  The shipped magnetic route inverts `mu_t` per cell,
which is exact there because `mu` is constant per cell and the walls are
element boundaries; under a map `mu` is still constant per cell but `J` is
not, so `mu'` varies inside the cell and inverting a cell average of it is a
different operator (mutation `mu_after`, 2.9: 0.44 on the magnetic film).
`J^T [mu_t]^-1 J / sg` IS that pointwise inverse, computed in closed form.

### 1.2 As built

| piece | where | what |
|---|---|---|
| the congruence | `twod_staggered._stag_map_eff_tensor(eps, mu, xu, xv, yu, yv)` | the four formulas above, broadcast over any node array; `mu = None` is vacuum |
| the node weights | `_stag_map_weights(..., mu_cell=None)` -> `_stag_map_weights_tensor` | a scalar cell with no `mu` keeps the scalar route `_stag_map_eff` (the bytes of Phases A-C, D1); a block-form tensor or ANY `mu_cell` takes the Jacobian at every tensor-rule node and every corner-rule (Duffy) point and calls the one congruence; a scalar cell is promoted to `eps I` (`_stag_map_as33`) |
| the 18 blocks | `_stag_quad_weighted` (unchanged) | the solver's `_assemble` body is UNCHANGED: under a map it already runs with `tensor = True` and the chi maps from the node weights, so the four `[eps_t]` blocks (the two mixed V1 x V2 masses now carry a smoothly varying `e12'(u, v)`), `Meps33`, the two-term `K_zt`, the four `R` blocks, `Gw_chi` and the four `K_tz` terms flow through the SAME quadrature kernel with the effective tensor as the weight -- no tensor-flavoured copy |
| the solver | `Granet2DTransverseE` | lifts the tensor and `mu_cell` refusals under a map; evaluates the node weights after `mu_cell` is validated (`_init_map_weights`); keeps an OUT-OF-PLANE `eps` / `mu` and a `slant` refused under a map, naming Phase E |
| the stack | `PMM2DStackPure` | `_require_map_scope` refuses only out-of-plane tensors and slant (Phase E); a magnetic or tensor layer under a map takes its own region eig, deduped with the map fingerprint (unchanged code) |
| the shapes | `shapes2d` | every primitive takes `mu=` (scalar or block-form) next to `eps`; the merge paints a `mu` grid exactly like the `eps` grid (`_paint_mu`; a shape without `mu` paints 1, `background_mu` fills the rest); `compile_shapes(..., with_mu=True)` returns it as a fifth output and the four-output form REFUSES a magnetic layer (it would drop `mu` silently); the stack records a magnetic shape layer as an `add_layer(eps_cell=, mu_cell=)` layer (`kind = 'magnetic'`) so the lossless predicate, the Wood guard and the eig dedupe treat it as one; `add_layer(..., background_mu=)` and `pmm_jones_2d_staggered(..., background_mu=)` |

Not touched: `Basis1D`, the quadrature kernel, the corner rule, the far
projector, the incident decomposition, the eigensolver (QZ), the mortar, the
out-of-plane generator.

---

## 2. The gates

Every number is from this tree, 2026-10-03.  "Unit test" names the decision
in `tests/unit/test_pmm2d_staggered_curved_d.py`; ladders too expensive for a
unit test are measured here only.

### 2.1 D1 -- no map = today's bytes; scalar maps = Phase C's bytes

| claim | measured | bar | fail-before | JSON |
|---|---|---|---|---|
| no map = the bytes of `607d43b0`, Phase C's fixture set (operators of every dispatch branch incl. tensor, magnetic, out-of-plane, slant; region modes; the geometric cache; far projectors; every public entry; the mortar; absorption) EXTENDED by 13 MAPPED SCALAR keys (the circle solver's operators, the shape circle at normal and conical incidence, a stretched lossy stripe with absorption) | **122 / 122** SHA-256 identical | equality | the identity map through the quadrature path: 28 of 36 operator hashes differ | `d1_bytes_pre.json`, `d1_bytes_post.json`, `d1_compare.json` |
| the same on the Phase A VERIFIER's fixture set (every tensor and magnetic fixture of 5.43 / 5.44; `verify_a/v1_bytes.py` run unchanged) | **181 / 181** identical | equality | -- | `d1_v1bytes_{pre,post}.json`, `d1_v1compare.json` |
| no Phase D kernel is reached without a tensor / `mu` under a map | `_stag_map_eff_tensor`, `_stag_map_weights_tensor`, `_stag_map_as33` booby-trapped; an unmapped tensor and an unmapped magnetic Jones solve and a SCALAR shape circle run | no trap fires | -- | unit test `test_d1_unmapped_and_scalar_mapped_solves_never_reach_the_phase_d_code` |
| the two expressions of one formula agree: `eps I` through the tensor route = the scalar route | operators 1.1e-15 (shear) / 3.1e-15 (circle) relative, R / T / Jones 6.0e-15 / 1.3e-14 (M = 5) | `<= 1e-12` | the tensor route with the mixed weights negated: 0.13 / 0.97 (a scalar cell carries `e12' = -eps g12 / sg`) | `d2_scalar.json`, `d_unit_scalar_mixed.json`; unit test `test_d1_tensor_route_at_eps_times_identity_is_the_scalar_route` |

### 2.2 D2 -- a uniform tensor film under a map is the film (Berreman 4x4, Jones included)

A uniform LC or GYRO film (the G3 fixture: period 0.40 um, lambda 1 um,
depth 0.55 um, air over n = 1.5) as a uniform layer of the mapped stack,
against `berreman_jones_1d` -- R and T summed over orders per input, and the
complex 2 x 2 Jones matrix (`d2_film_<map>_<tensor>_<angle>.json`):

| map, tensor | M = 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| s05, LC (R,T / Jones) | 5.1e-06 / 1.4e-06 | 1.2e-07 / 3.0e-08 | 9.9e-10 / 2.8e-10 | 1.1e-11 / 2.6e-12 | **6.9e-14 / 1.1e-13** |
| s05, GYRO | 7.3e-06 / 2.1e-06 | 1.8e-07 / 4.7e-08 | 1.4e-09 / 4.2e-10 | 1.6e-11 / 4.1e-12 | 1.3e-13 / 1.2e-13 |
| s15 (33:1), LC | 4.1e-05 / 1.3e-05 | 3.2e-06 / 6.2e-07 | 7.6e-09 / 2.5e-09 | 2.8e-10 / 5.4e-11 | **5.6e-13 / 2.9e-13** |
| s15, GYRO | 5.9e-05 / 1.8e-05 | 4.8e-06 / 1.1e-06 | 1.1e-08 / 3.5e-09 | 4.2e-10 / 9.7e-11 | 2.0e-12 / 3.9e-13 |
| shear, LC | 3.7e-15 / 1.5e-15 | 1.5e-14 / 4.5e-15 | 1.9e-14 / 9.4e-15 | 5.1e-14 / 2.6e-14 | 1.2e-13 / 1.1e-13 |
| shear, GYRO | 6.8e-15 / 2.6e-15 | 3.6e-15 / 1.1e-14 | 2.3e-14 / 2.1e-14 | 3.2e-14 / 3.1e-14 | 1.4e-13 / 1.3e-13 |
| shear, LC, conical (25, 40) | 2.3e-09 / 9.4e-10 | 1.2e-12 / 4.9e-13 | 2.8e-14 / 2.7e-14 | 1.1e-13 / 2.7e-14 | 2.1e-14 / 5.8e-14 |
| shear, GYRO, conical | 1.6e-09 / 1.3e-09 | 8.8e-13 / 7.4e-13 | 1.8e-14 / 2.3e-14 | 1.1e-13 / 2.4e-14 | 6.2e-14 / 7.4e-14 |

**STOP condition 1 is not met**: the rotated-uniaxial film under a stretch
reaches 6.9e-14 (`a = 0.05 p`) and 5.6e-13 (`a = 0.15 p`) of Berreman at
`M = 8`, against the stop bar 1e-9.  The film under the stretch is spectral,
not exact at every M -- the covariant field `E'_u = x_u(u) E_x` of a plane
wave is not a polynomial under a sine stretch -- and the bilinear shear map
IS exact at every M (the covariant plane-wave field is a polynomial of degree
one per cell): 3.7e-15 .. 1.4e-13 from `M = 4`, at normal and conical
incidence.

The two-sided arms (`d2_arms.json`, `M = 7`):

| arm | s05 (R,T / Jones) | shear | c3 |
|---|---|---|---|
| GYRO, correct | 1.6e-11 / 4.1e-12 | 3.2e-14 / 3.1e-14 | 4.9e-13 / 1.3e-12 |
| GYRO, eps TRANSPOSED in the congruence | **1.6e-11 / 0.1496** | **5.7e-14 / 0.1496** | **5.4e-13 / 0.1496** |
| LC, correct | 1.1e-11 / 2.6e-12 | 5.1e-14 / 2.6e-14 | 6.4e-13 / 9.5e-13 |
| LC, SIDES swapped (`J^-T eps J^-1`) | 1.1e-11 / 2.6e-12 (identical: `J` diagonal) | **7.0e-4 / 2.6e-3** | **1.3e-3 / 4.4e-3** |
| LC, transposed | 1.1e-11 / 2.6e-12 (a no-op: LC is symmetric) | 3.5e-14 / 2.0e-14 | 5.9e-13 / 9.5e-13 |

`J01 / J10 = -1 + 5e-11` (s05), `-1 + 9e-14` (shear, c3) -- the gyrotropic
film's own signature (`J01 = J10` for a reciprocal film).

**The transpose gate is not blind, and R / T is**: the transposed gyrotropic
film moves the Jones matrix by 0.15 while its R / T error is the correct
arm's to 2e-14 -- an R / T gate would pass a transposed build.  STOP
condition 2 (the gyrotropic Jones gate cannot distinguish the transpose
conventions) is not met.

Unit tests: `test_d2_tensor_films_under_a_stretch_and_a_shear_match_berreman`
(s05, LC and GYRO: `M = 6` <= 1e-8, measured 1.44e-9 worst, 0.84 decades;
the `M = 4 -> 6` drop >= 3 decades, measured 3.7; shear `M = 4` <= 1e-11,
measured 6.8e-15; fail-before the side swap on the shear at `M = 4`, 7.0e-4 /
2.6e-3 >= 1e-4) and `test_d2_gyrotropic_jones_sees_the_transpose_and_rt_does_not`
(s05, `M = 5`: correct Jones 4.7e-8 <= 1e-7; transposed Jones 0.1496 >= 1e-2;
`|R,T error transposed - correct|` 1.9e-15 <= 1e-12; `|J01 / J10 + 1|` 6.0e-7
<= 1e-5).

### 2.3 D3 -- a tensor weight through the four singular vertices of the circle

**The films** under the circle maps (`d2_film_c3_*.json`; c5 in 2.3.1):

| tensor, angle | M = 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| c3, LC, normal (R,T / Jones) | 2.5e-06 / 2.9e-06 | 2.8e-08 / 3.5e-08 | 7.7e-11 / 1.7e-10 | 6.4e-13 / 9.5e-13 | 2.4e-13 / 9.5e-14 |
| c3, GYRO, normal | 2.0e-06 / 2.9e-06 | 2.3e-08 / 4.6e-08 | 4.4e-11 / 1.7e-10 | 4.9e-13 / 1.3e-12 | 1.4e-13 / 1.3e-13 |
| c3, LC, conical (25, 40) | 2.9e-05 / 1.6e-05 | 4.5e-06 / 2.1e-06 | 8.7e-08 / 3.8e-08 | 3.3e-09 / 1.5e-09 | 8.8e-11 / 3.9e-11 |
| c3, GYRO, conical | 3.3e-05 / 1.7e-05 | 6.5e-06 / 3.3e-06 | 8.1e-08 / 5.7e-08 | 4.5e-09 / 2.5e-09 | 9.9e-11 / 6.2e-11 |

Spectral to round-off at normal incidence; at conical incidence spectral and
four decades behind (the Phase B finding F-B6 -- the Bloch phase composed
with the map -- applies unchanged to tensors).

**The quadrature** (`d3_quad.py` -> `d3_quad_{c3_M6,c3_M8,c5_M5,stretch_M6,stretch_M8}.json`).
The corner rule now integrates the individual Jacobian products of
`adj(J) eps adj(J)^T / sg` and `J^T mu^-1 J / sg`, not only the five metric
combinations the node criterion measures.  The disk filled with LC30, arm
`epsmu` adds `mu = diag(2, 2, 1)` in the disk; node count forced, operators
(L, R) relative to the top rung:

| rule | n = 8 | 12 | 16 | 20 | 24 | 32 | 48 (top) |
|---|---|---|---|---|---|---|---|
| corner rule, c3, M = 6, eps | 9.5e-04 | 2.8e-09 | **7.2e-13** | 6.8e-13 | 6.7e-13 | 5.6e-13 | 0 |
| corner rule, c3, M = 6, eps + mu | 1.0e-03 | 5.7e-09 | **7.3e-13** | 6.8e-13 | 6.7e-13 | 5.7e-13 | 0 |
| corner rule, c3, M = 8, eps + mu | 1.8e-02 | 1.1e-05 | 2.1e-11 | 2.0e-12 | 2.7e-12 | 2.4e-12 | 0 |
| corner rule, c5, M = 5, eps + mu | 1.0e-03 | 5.6e-11 | 7.3e-14 | 5.6e-14 | 4.1e-14 | 2.8e-14 | 0 |
| PLAIN tensor rule, c3, M = 6, eps + mu | 0.33 | 0.16 | 0.097 | 0.063 | 0.045 | 0.026 | 0.012 |

**Yes, the corner rule still reaches round-off with a tensor and a magnetic
weight**, at the same `n = 16` as Phase B's scalar cells; the plain rule
converges only algebraically (`~n^-1.3` here; Phase B's scalar operators
6 %).  The adaptive node count picks `n0 = 2 M + 8` (20 / 24 / 18) and its
operators are 5.5e-13 (c3, M = 6), 2.7e-12 (c3, M = 8), 6.2e-14 (c5, M = 5)
from those at `2 n0` -- the five-weight criterion stays adequate for the
tensor products.  Under the sine stretch (where the criterion doubles:
`n0 = 20 / 80` at `a = 0.05 / 0.15 p`, `M = 6`; `24 / 96` at `M = 8`) the
tensor and magnetic operators at `n0` are within 1.5e-13 .. 2.6e-14 of those
at `4 n0`.

Unit tests: `test_d3_tensor_films_under_the_circle_map_are_spectral` (c3, LC
and GYRO, `M = 6` <= 1e-9, measured 1.74e-10 worst, 0.77 decades; drop
`M = 4 -> 6` >= 3 decades, measured 4.4 / 4.6) and
`test_d3_corner_rule_reaches_roundoff_with_tensor_and_magnetic_weights`
(`n = 16` vs 48 <= 1e-11, measured 7.3e-13; plain `n = 24` >= 1e-2, measured
4.5e-2; `n0 = 20`).

#### 2.3.1 The 5 x 5 film ladders

`d2_film_c5_{lc,gyro}_normal.json` (normal incidence, ladder to `M = 6`):
LC 5.5e-09 / 8.6e-09, 6.5e-12 / 1.1e-11, **2.0e-14 / 3.8e-14**; GYRO 3.8e-09 /
7.9e-09, 4.9e-12 / 1.6e-11, **6.7e-14 / 3.1e-14** at `M = 4 / 5 / 6` (R,T /
Jones) -- round-off by `M = 6`, as Phase B's scalar film on the 5 x 5 map.

### 2.4 D4 -- a tensor circular pillar against two oracle families

The LC30 disk (r = 0.36, director at 30 degrees to x) in air, period 1.2,
depth 0.5, lambda 1, air above, n = 1.45 below (`d4_pillar.py` ->
`d4_*.json`, `d4_summary.json`).  Distance = the largest of the 36 entries
(nine orders `|m|, |n| <= 1`, R and T, both inputs) to the c3 `M = 10` rung.

| family | knob | distance to c3 `M = 10` | rung change | closure | wall (s, loaded) |
|---|---|---|---|---|---|
| c3 | M = 4 / 5 / 6 / 7 / 8 / 9 | 4.5e-2 / 1.0e-2 / 5.2e-3 / 3.9e-4 / 1.7e-4 / 1.0e-5 | 3.5e-2 / 4.9e-3 / 4.8e-3 / 2.1e-4 / 1.6e-4 / 1.0e-5 | 1.9e-3 .. 2.2e-7 (M = 9), 1.0e-7 (M = 10) | 1 .. 409 |
| c5 (independent topology) | M = 4 / 5 / 6 / 7 / 8 | 3.3e-4 / 3.8e-5 / 1.3e-5 / 3.1e-6 / **2.5e-6** | 2.9e-4 / 2.6e-5 / 1.0e-5 / 5.0e-6 | 1.6e-5 / 1.1e-7 / 5.7e-9 / 1.5e-11 / 1.1e-12 | 44 .. 1337 |
| shipped 2-D tensor RCWA, EXACT disk form factor (Laurent) | 9 / 13 / 17 / 21 / 25 / 29 orders per axis | 9.1e-3 / 6.6e-3 / 5.2e-3 / 4.2e-3 / 3.5e-3 / 3.0e-3 | -- | <= 3e-13 | 11 .. 32 |
| same, Richardson in 1 / N | pairs (9, 13) .. (25, 29) | 8.9e-4 / 5.5e-4 / 2.2e-4 / 3.2e-4 / **5.6e-5** | -- | -- | -- |
| shipped staggered solver, staircase (no map) | 4 steps (k = 1), M = 4 / 6 / 8 / 10 | 4.8e-2 / 1.6e-2 / 1.7e-2 / 1.7e-2 | (converged in M: 1.7e-2) | 8.4e-11 (M = 10) | 1 .. 413 |
| same | 8 steps (k = 2), M = 5 / 6 / 7 | 1.37e-2 / 1.38e-2 / 1.38e-2 | (converged) | 1.3e-10 | 105 .. 1030 |
| same | 16 steps (k = 4), M = 3 / 4 | 6.1e-3 / 5.4e-3 | 7.3e-4 (pre-asymptotic) | 5.1e-6 | 55 / 695 |

The c5 `M = 8` rung (dof 2450) sits 2.5e-6 from c3 `M = 10` (dof 1458), its
own last rung change 5.0e-6: the two independent topologies agree at the
level of their own convergence.

* **The curved answer is fixed to ~4e-6**: the two INDEPENDENT topologies
  agree to 1e-6 .. 4e-6 (c3 `M = 10 .. 12` vs c5 `M = 8`; VERIFY_D s 4),
  inside the c3 ladder's own last rung changes (3.5e-6, 2.8e-6 at `M = 11 /
  12` -- the rate slows past `M = 10`, the pillar rim's edge singularity,
  plan 3.3), and both ladders fall onto it.
* **The RCWA family approaches it algebraically, as 1 / N.**  Distance x N
  = 0.082, 0.085, 0.088, 0.087, 0.086, 0.086 over N = 9 .. 29 (within +-4 %;
  the Phase B verifier's D-5 check), the local rate from successive
  differences 0.91, 0.73, 0.91, 1.19; the Richardson pairs fall toward the
  curved answer (a 1e-4-class corroboration: pairs wander 5e-5 .. 3e-4 to
  N = 33, a 1/N least-squares fit 7.7e-5 -- VERIFY_D s 4) and nowhere else
  -- a reference whose limit is the curved answer.  The patched route's
  self-check: the SAME exact-disk patch on a scalar disk equals the shipped
  `rcwa_efficiency_2d_shapes` to 1.9e-16 (9 orders) / 6.6e-14 (17 orders)
  (`d4_rcwa_scalar_n{4,8}.json`) -- the patch is the shipped Laurent
  operator with the exact form factor, component by component.
* **The staircases converge in M to DIFFERENT devices** (1.7e-2, 1.4e-2,
  5.4e-3 for 4, 8, 16 steps), strictly decreasing toward the curved answer;
  the 16-step reading is pre-asymptotic in M (7.3e-4 from M = 3 to 4).

How the RCWA was made exact: `rcwa_jones_2d(formulation='laurent')` builds
each tensor component's Toeplitz matrix in ONE function
(`rcwa.twod._eps_convolution_2d`); the probe (`d4_pillar.exact_disk`) patches
it for the duration of the call to return `bg_ij delta + (d_ij - bg_ij) F`,
with `F` the shipped analytic disk form factor (`_shape_form_factor`) and
`bg_ij` / `d_ij` read from the pixel cell (which only identifies the
component).  `fff_nv` was not used: it builds its own factorization from the
pixels, so a pixel cell would put the O(1/S) staircase error back in.

Unit test `test_d4_tensor_pillar_lands_on_its_converged_answer` (c5 `M = 4`,
dof 450, against the saved c3 `M = 10` rung: 3.3e-4 <= 1e-3, 0.48 decades;
fail-before the 4-step staircase at `M = 4`, 4.8e-2 >= 1e-2).

### 2.5 D5 -- the Li 2003 gyrotropic grating under the identity map and a stretch

Li, J. Opt. A 5:345 (2003), Example 1 (= Granet 2023 Table 2 with two
transcription errors): periods 2.4 x 1.4 lambda, a half-filled 2 x 2 cell of
`eps_b = [[2.25, -0.5i, 0], [0.5i, 2.25, 0], [0, 0, 2]]` in `eps_a = conj(eps_b)`,
depth lambda, substrate index `1 + 5i` (`eps = -24 + 10i`), incident E_x;
the table lists six REFLECTED orders; the second row is the cell with the
cross terms reversed (`d5_li2003.py` -> `d5_li_*.json`, `d5_li_summary.json`).

| arm | M | Li row 1 (max dev.) | row 2 | vs the unmapped solve at the same M (R,T / Jones) |
|---|---|---|---|---|
| unmapped (shipped) | 6 / 8 / 10 / 12 | 3.24e-4 / **8.74e-5** / 7.10e-5 / 7.12e-5 | 1.32e-2 | -- |
| IDENTITY TransfiniteMap (tensor route) | 6 / 8 | 3.24e-4 / **8.74e-5** | 1.32e-2 | 3.5e-14 / 4.5e-14; 8.7e-14 / 1.4e-13 |
| stretch (a = 0.05 p_x, -0.03 p_y; preimage walls) | 5 / 6 / 7 / 8 / 9 / 10 / 11 / 12 | 4.5e-3 / 2.8e-3 / 3.9e-4 / 1.4e-4 / 7.5e-5 / 7.3e-5 / 7.1e-5 / 7.1e-5 | 1.3e-2 | 5.9e-3 / 3.1e-3 / 1.6e-3 / 3.4e-4 / 1.1e-4 / 2.8e-5 / 4.1e-6 / **2.9e-6** |
| stretch, cross terms SWAPPED | 6 / 8 | (row 2) 2.8e-3 / 1.4e-4 | (row 1) 1.41e-2 / 1.32e-2 | -- |

* **Identity map**: the tensor route under an identity map reproduces the
  unmapped path to 4.5e-14 .. 1.4e-13 and Li's table to the same 8.739e-5
  at `M = 8` (8.739349639e-5 against the shipped 8.739349637e-5).
  BIT-FOR-BIT is not available (the brief asked for it): the quadrature
  reassociates the sums (plan P1, 5.5e-14), which is why "no map" is a
  DISPATCH; the shapes route with rectangles only (an identity merged map)
  IS byte-identical, because it runs the unmapped solver (Phase C, C16).
* **Stretch**: converges to the unmapped answer (2.9e-6 at `M = 12`, inside
  the unmapped ladder's own rung changes 8.0e-6 / 3.6e-6 / 1.1e-6 at
  `M = 9 -> 12`); the stretch costs one to two rungs, as Phase A measured.
  The gyrotropic sign survives the map: the swapped cell lands on Li's
  second row and misses the first by 1.32e-2.
* On a PATTERNED gyrotropic grating the transpose IS R / T-visible (Li's two
  rows differ only in the (1, -1) / (1, 1) pair): the transposed congruence
  under the stretch lands on row 2 (`d9_summary.json`, `li_stretch`), unlike
  on a uniform film (2.2).

Unit tests `test_d5_li2003_under_the_identity_map_is_the_unmapped_solve`
(`M = 6`: 4.5e-14 <= 1e-11) and
`test_d5_li2003_under_a_stretch_keeps_the_gyrotropic_sign` (`M = 8`: row 1
1.40e-4 <= 5e-4 -- the shipped G2 bar, Li's own four-decimal rounding -- and
row 2 missed by 1.32e-2 >= 1e-3; the swapped cell the other way round).

### 2.6 D6 -- a material permeability under the map

**The magnetic film** (eps 2.0, `mu = diag(1.8, 1.8, 1.3)`, depth 0.5 on
n = 1.45) against the exact (eps, mu) slab at normal incidence
(`d6_film_*.json`; Jones off-diagonals <= 1.0e-14 throughout):

| map | M = 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| c3 | 1.4e-06 | 1.7e-08 | 2.2e-11 | 3.1e-13 | 5.5e-14 |
| c5 | 2.8e-09 | 2.9e-12 | 2.1e-14 | 4.2e-14 | |
| s15 | 1.4e-06 | 4.4e-09 | 1.6e-12 | 2.3e-14 | 9.5e-14 |

**Duality** (`d6_dual_*.json`): the E_x input of the magnetic disk (eps 1,
`mu = diag(2, 2, 1)`) against the E_y input of the dielectric disk
(`eps = diag(2, 2, 1)`, mu 1), vacuum half-spaces, the same map -- two
different weight paths (the chi blocks against the eps blocks):

| map | M = 4 | 5 | 6 | 7 | 8 | 9 | no-swap control |
|---|---|---|---|---|---|---|---|
| c3, duality residual | 5.4e-3 | 8.9e-4 | 5.8e-4 | 8.6e-6 | 3.2e-5 | 1.2e-5 | 0.048 .. 0.052 |
| c3, the magnetic pillar's own rung change | 6.2e-3 | 1.0e-3 | 1.0e-4 | 2.6e-5 | 7.5e-6 | -- | |
| c5, duality residual | 2.3e-4 | 6.0e-5 | 2.6e-5 | 1.3e-5 | | | 0.048 |

Duality is not a discrete identity of this basis (E and H are placed in
different staggered spaces), so the residual falls with M at the
discretisation level -- non-monotonically on c3, tracking the rung change
-- and the no-swap control stays at 0.048.  The magnetic pillar's c3 and c5
answers agree to 1.0e-5 (c5 `M = 6` vs c3 `M = 9`).

**The staircase limit** of the magnetic pillar (the shipped magnetic solver,
no map; `d6_stair_*.json`), distance to the c3 `M = 9` magnetic pillar:
4 steps (k = 1) 1.96e-2 / 1.91e-2 / 1.91e-2 / 1.91e-2 at `M = 4 / 6 / 8 / 10`;
8 steps 1.25e-2 / 1.25e-2 / 1.24e-2 at `M = 5 / 6 / 7`; 16 steps 5.94e-3 /
5.94e-3 at `M = 3 / 4` -- each converged in M to its own staircase device,
strictly decreasing toward the curved answer, as the dielectric staircases
did (D4) and Phase B's scalar ones.

**The H-partner trap with a material mu** (plan 4.4: "the fail-before is -R
as the Eq.-25 Gram, which must now be visible even without mixing helpers --
measure it").  Measured (`d_unit_mag.json`, `d9_summary.json`): with the H
partner of EVERY mapped region recovered through `-R`, the magnetic film
moves by 1.9e-2 at `M = 4` AND at `M = 5` (flat: a wrong operator, not a
discretisation error; correct 1.7e-8 at `M = 5`), the duality by 5.7e-2 --
while every NON-magnetic gate is untouched (the LC film under the circle map
2.8147005e-8 -> 2.8147008e-8; the D4 pillar moves 2.7e-14).  Confirmed: a
material `mu` makes `-R` differ between regions, so the consistent-wrong
arm that Phase A measured invisible becomes visible.

Unit tests `test_d6_magnetic_film_under_the_circle_map_matches_the_airy_slab`
(`M = 5` 1.7e-8 <= 1e-7; `mu_after` 0.44 and `hgram_R` 1.9e-2 >= 1e-3),
`test_d6_magnetic_pillar_is_the_dual_of_its_dielectric_twin` (c3 `M = 5`:
8.9e-4 <= 3e-3; control 4.85e-2 >= 1e-2),
`test_d6_the_h_partner_trap_needs_a_material_mu` (the LC film through `-R`
within 1e-12 of the correct arm).

### 2.7 D7 -- a tensor AND a magnetic layer in one merged-map stack

Build-doc geometry (`d7_stack.py` -> `d7_*.json`): Phase C's C6 layout with
the materials changed -- layer 1 an LC30 circle (r = 0.2), layer 2 a
filleted square (0.9 x 0.9, r = 0.09) of eps 2.25 with `mu = diag(1.5, 1.5,
1.2)` -- ONE merged 7 x 7 map.  Unit-test geometry (`d_unit_stack.json`): an
LC30 circle (r = 0.3) inside a magnetic Rect (0.9 x 0.9) over a plain Rect
layer, one merged 5 x 5 map.

| quantity | 7 x 7, M = 3 | 7 x 7, M = 4 | 5 x 5, M = 3 | 5 x 5, M = 4 | fail-before |
|---|---|---|---|---|---|
| lossless closure | 3.4e-3 | 3.0e-4 | 2.3e-2 | 9.2e-4 | -- |
| absorption in the LOSSLESS layer(s): lossy LC (LC30 + 0.3i I) | 2.5e-15 | 1.0e-14 | 1.1e-15 | | `-R` as the flux Gram: 6.4e-3 / 4.7e-3 (7 x 7, M = 3 / 4), 2.0e-2 (5 x 5) |
| same, lossy mu (mu_t = 1.5 + 0.2i) instead | 2.6e-15 | 4.6e-15 | | | `-R`: 6.9e-3 / 6.6e-3 |
| sum of `layer_absorption` vs 1 - sum R - sum T (lossy LC / lossy mu) | 3.0e-3 / 2.5e-3 | 3.4e-4 / 2.5e-4 | 2.0e-2 | | `-R`: 1.4e-2 / 1.1e-2 (M = 3), 9.2e-3 / 7.6e-3 (M = 4) |
| vacuum painted with `mu = I` (tensor route) vs a uniform vacuum layer, same map | 6.6e-15 | | 3.1e-15 | | painted mu = 1.1 (5 x 5) / eps 1.1 (7 x 7): 2.3e-2 / 2.1e-2 |
| vacuum painted WITHOUT mu (scalar route) vs with `mu = I` | 4.6e-15 | | | | |

The sum-vs-balance column is the discretisation level (it tracks the
closure, as Phase C's C6 recorded); the per-layer reading of a LOSSLESS
layer is the sharp gate.

Unit tests `test_d7_tensor_and_magnetic_layers_share_one_map_and_conserve_energy`
(5 x 5; closure `M = 4` 9.2e-4 <= 5e-3; lossless layers 1.1e-15 <= 1e-10;
`-R` 2.0e-2 >= 1e-3) and `test_d7_vacuum_painted_with_mu_one_is_vacuum`
(3.1e-15 <= 1e-9; mu 1.1 2.3e-2 >= 1e-3).

### 2.8 D8 -- oblique and conical incidence through a tensor circle

The D4 pillar at 25 deg (phi 0) and conical (25 deg, phi 40 deg), and the
reversed channels (`d8_oblique.py` -> `d8_*.json`, `d8_summary.json`).
Reciprocity: the singular values of the power-normalized 2 x 2 Jones block
of (incidence -> reflected order) against its reversal -- an identity the
solver does not impose, valid for a reciprocal medium (the LC is real
symmetric):

| channel | map | M = 5 | 6 | 7 | 8 | wrong pairing |
|---|---|---|---|---|---|---|
| (25, 0) -> (-1, 0) | c3 | 1.0e-5 | 1.9e-6 | 1.9e-7 | 2.0e-8 | 0.13 |
| (25, 40) -> (-1, 0) | c3 | 1.2e-4 | 1.0e-5 | 4.2e-6 | 1.2e-7 | 0.12 |
| (25, 40) -> (0, -1) | c3 | 2.2e-4 | 2.1e-5 | 6.5e-6 | 6.0e-7 | 0.093 |
| (25, 0) -> (-1, 0) | c5 (M = 4 / 5 / 6) | 1.1e-7 / 1.6e-9 / 1.3e-10 | | | | 0.13 |
| (25, 40) -> (-1, 0) | c5 | 1.7e-6 / 5.6e-8 / 1.0e-9 | | | | 0.12 |
| (25, 40) -> (0, -1) | c5 | 2.5e-6 / 3.0e-8 / 5.9e-9 | | | | 0.093 |
| (25, 40) -> (-1, 0), GYROTROPIC disk (non-reciprocal control) | c3, M = 6 | -- | 1.6e-2 | | | 0.13 |
| (25, 40) -> (0, -1), GYROTROPIC disk | c3, M = 6 | -- | 1.8e-2 | | | 0.12 |

Lossless closure, c3 `M = 5 .. 8`: 3.1e-3 .. 1.1e-5 (phi 0), 2.5e-3 ..
5.4e-5 (phi 40), the reversed runs 8.1e-3 .. 3.3e-5 and 1.6e-2 .. 1.2e-4;
c5 `M = 4 / 5 / 6`: 6.4e-5 .. 9.2e-4, 3.0e-6 .. 3.4e-5, 9.5e-9 .. 6.8e-7.
Reciprocity holds one to four decades better than the closure at every rung
and falls spectrally; the GYROTROPIC disk breaks it at the physics level
(1.6e-2 / 1.8e-2 at c3 `M = 6`, against 1.0e-5 / 2.1e-5 for the LC disk at
the same rung, closure 3.7e-4 .. 1.1e-3).

**The non-reciprocal control**: the GYROTROPIC disk (eps != eps^T) at c3
`M = 5`: 2.1e-2 (`d_unit_oblique.json`), 2.2 decades above the reciprocal
LC's 1.2e-4 -- the identity sees the material, not only a pairing error.

Unit test `test_d8_conical_tensor_circle_is_reciprocal_and_a_gyrotropic_one_is_not`
(c3 `M = 5`: LC 1.2e-4 <= 1e-3; wrong pairing 0.12 >= 1e-2; gyro 2.1e-2 >=
3e-3; closure 8.1e-3 <= 3e-2).

### 2.9 D9 -- the mutation matrix

Every engineered defect against every gate at fixed M, the correct arm as
the reference row (`d9_mutations.py` -> `d9_<arm>.json`, `d9_summary.json`).
Cells: the gate's error; **bold** = caught (>= 10x the correct arm and
above the gate's bar).

| gate (M) | correct | transpose | side | no_sg_e33 | mixed_sign | mu_after | hgram_R |
|---|---|---|---|---|---|---|---|
| GYRO film, s05, Jones (6) | 4.2e-10 | **0.15** | 4.2e-10 | 4.2e-10 | **0.15** | 4.2e-10 | 4.2e-10 |
| GYRO film, s05, R / T (6) | 1.4e-9 | 1.4e-9 | 1.4e-9 | 1.4e-9 | 1.4e-9 | 1.4e-9 | 1.4e-9 |
| LC film, shear, R / T (6) | 1.9e-14 | 1.8e-14 | **7.0e-4** | 4.2e-14 | **4.8e-4** | 1.9e-14 | 3.3e-14 |
| LC film, c3, conical, R / T (6) | 8.7e-8 | 8.7e-8 | **2.0e-3** | **1.2e-4** | **7.0e-3** | 8.7e-8 | 8.7e-8 |
| magnetic film, c3 (6) | 2.2e-11 | 2.2e-11 | **1.3e-2** | 2.2e-11 | **0.86** | **0.48** | **1.9e-2** |
| magnetic duality, c3 (6) | 5.8e-4 | 5.8e-4 | **3.1e-2** | 4.8e-3 | **0.24** | **7.7e-2** | **5.7e-2** |
| Li 2003 under the stretch, row 1 (7) | 3.9e-4 | **1.3e-2** (lands on row 2) | 3.9e-4 | **1.8e-2** | **1.3e-2** | 3.9e-4 | 3.9e-4 |
| LC30 pillar, c3, change vs correct (6) | 0 | 4.2e-14 | **1.6e-2** | **1.5e-2** | **0.43** | 0 | 2.7e-14 |

Every defect is caught by at least one gate by two decades or more, and
each gate's blindness is the physics:

* **transpose**: only the GYRO Jones (film) and the Li rows (patterned) see
  it; every R / T of a symmetric tensor, and the gyrotropic film's R / T AT
  NORMAL INCIDENCE, are transpose-blind (at conical incidence the
  transposed gyrotropic film moves R / T by 9.9e-4: VERIFY_D s 2).
* **side** (`J^-T eps J^-1`): invisible under the stretch (diagonal `J`);
  every gate with a non-diagonal `J` (shear, circle) sees it.
* **no_sg_e33**: `eps'_33` enters only the div(D) = 0 Schur term, which a
  plane wave at NORMAL incidence never exercises (`E_z = 0`; the normal LC
  film under the circle map reads 2.81462e-8 mutated vs 2.81470e-8
  correct, `d_unit_no_sg_e33.json`) -- caught at oblique incidence (1.2e-4,
  flat in M, against 8.7e-8), by the patterned Li grating and the pillar.
* **mixed_sign**: on the LC film under a diagonal-`J` map it mirrors the
  director's in-plane angle -- R / T-blind (1.21966e-7 both arms at `M = 5`,
  `d_unit_mut.json`), Jones-visible (9.8e-3 against 3.0e-8).
* **mu_after**: visible exactly where a material `mu` sits under a
  non-constant `J` (the magnetic film 0.48, the duality 7.7e-2).
* **hgram_R**: visible exactly where a material `mu` is present (2.6).

Unit tests: `test_d9_no_sg_e33_is_caught_only_off_normal_incidence`,
`test_d9_mixed_sign_is_rt_blind_on_a_stretched_film_and_jones_visible`, and
the arms inside D2 (transpose, side) and D6 (mu_after, hgram_R).

### 2.10 D10 -- cost

Operator construction only (node count, weights, the 18 quadrature blocks,
the Schur term), best of three, 3 x 3 circle map, plus one QZ region eig
for scale (`d10_cost.py` -> `d10_cost_M{6,8,10}.json`; LOADED box, upper
bounds, only same-run ratios mean anything):

| M (pencil) | scalar mapped | tensor mapped | tensor + mu mapped | tensor unmapped | region eig (any arm) |
|---|---|---|---|---|---|
| 6 (450) | 0.55 s | 0.42 s (0.76x) | 0.50 s (0.91x) | 0.18 s | 6.3 .. 7.9 s |
| 8 (882) | 3.3 s | 4.1 s (1.24x) | 1.5 s (0.46x) | 1.2 s | 61 .. 77 s |
| 10 (1458) | 4.7 s | 4.6 s (0.99x) | 4.9 s (1.04x) | 3.1 s | -- |

A tensor or magnetic mapped cell costs what a scalar mapped cell costs: the
same pencil, the same eig, an assembly within the load noise of the scalar
one (0.46 .. 1.24x) and 1.5x the unmapped tensor assembly at `M = 10`; the
assembly is 1-6 % of a region eig.  A magnetic or tensor UNIFORM layer under
a map takes its own region eig (it cannot ride the eps-free geometric eig),
exactly as without a map.

---

## 3. Findings

### 3.1 F-D1 -- the gates the plan named are blind to three of the four defects in the configuration it named (formulation finding, measured)

The plan (4.4) gates the transformation on uniform films under the circle
map at normal incidence and the brief on a film under a stretch.  Measured
(2.9): under ANY separable stretch `J` is diagonal, so `J^-1 eps J^-T` and
`J^-T eps J^-1` coincide (the side swap is invisible); at NORMAL incidence
`eps'_33` never enters (`E_z = 0`); a mirrored director (the mixed-sign
flip) leaves a uniform film's R / T unchanged; and a transposed gyrotropic
tensor leaves a uniform film's R / T unchanged at NORMAL incidence (at
conical incidence it is R / T-visible, 9.9e-4: VERIFY_D s 2).  The gate set here therefore
adds a SHEARED bilinear map (non-diagonal `J`, still exact), OBLIQUE /
conical films, the JONES matrix on every film, and the patterned Li grating
and tensor pillar -- every defect is then caught by two decades or more
(2.9).

### 3.2 F-D2 -- the H-partner trap with a material mu is real without mixing helpers (the plan's prediction, confirmed)

2.6.  Without a material `mu` every mapped region's `-R` is the same
geometric operator and the consistent `-R` arm is invisible (2.7e-14 on the
pillar); with one it moves the magnetic film by 1.9e-2, flat in M.

### 3.3 F-D3 -- `geom[4]` serves TWO roles: the half-spaces' H partner and the incident decomposition (probe finding; no library change)

The first version of the `hgram_R` mutation replaced the inverse Gram in
the mapped `_homog_geom_cache` tuple, which Phase C's incident
decomposition (`_stag_incident_coeffs_mapped`) also reads (`geom[4]`).  That
arm moved the NON-magnetic Li grating by 1.1e-2 and the tensor pillar by
5.3e-2 -- not the H-partner trap but a broken incident projection.  The
mutation was corrected to change only the H partner (2.6 numbers are the
corrected arm).  Recorded because the same tuple slot carrying both roles is
a place a future edit could break one while meaning the other; both are
gated (the incident decomposition by Phase C's C9 and every D gate here).

### 3.4 F-D4 -- duality is not a discrete identity of this basis

2.6.  The magnetic / dielectric dual pair converges onto each other at the
discretisation level (1.2e-5 at c3 `M = 9`), not to round-off -- E and H sit
in different staggered spaces.  The gate is therefore a convergence gate
with a flat no-swap control, not a round-off identity.

### 3.5 F-D5 -- the node-count criterion stays adequate for tensor weights

2.3.  It measures only the five metric weights, while a tensor weight
carries the individual Jacobian products; measured adequate (5.5e-13 ..
2.7e-12 from `2 n0` on the circle, <= 2.6e-13 from `4 n0` under the
stretch).  Not changed.

---

## 4. What moved

Nothing shipped: D1, 122 / 122 and 181 / 181 SHA-256 identical against
`607d43b0` -- including 13 MAPPED SCALAR keys, so no Phase A-C answer moved
either.  New behaviour, reachable only with a tensor or a `mu` under a map
(which raised before): block-form tensors and permeabilities inside curved
cells.  The Phase B verifier's D-1 fix (section 6) refuses a malformed user
`EdgeCurve` that was accepted before -- a correct curve is unaffected.

---

## 5. Not measured

* **The probe READINGS on a second build.**  The unit gates were run on the
  WSL build (section 8); the ladders here are one build's.
* **An independent full-wave oracle for the TENSOR pillar** (the 3-D FEM
  runner of the plan is a scalar quarter-cell formulation with mirror walls,
  which a 30-degree director breaks).  The references are the two
  independent topologies (2.5e-6), the exact-form-factor tensor RCWA
  (Richardson to 5.6e-5) and the staircases.
* **Out-of-plane tensors under a map** (Phase E; refused).
* **Idle wall times.**

---

## 6. The Phase B verifier's report, folded in

Merged `verify/pmm2d-curved-b` (`5d2272fd`) as `074226d2` (`.test_durations`
dict-union).  Then:

| item | what | commit |
|---|---|---|
| D-1 (P2, silent wrong answer) | `TransfiniteMap.__init__` checks each edge curve's analytic derivative against a five-point central difference of its value and refuses a mismatch (the report's section-12 edit, verbatim); the verifier's strict xfail flipped to a pass and the marker was dropped (65 / 65 curved + verifier tests pass) | `9e14a88d` |
| N-1 | `singular_vertices` documents that a nearly antiparallel user corner is not listed and falls to the tensor rule, whose node criterion warns at its cap | `9e14a88d` |
| D-2 .. D-5 (documentation) | CHANGELOG Phase B entry and BUILD_B 4.4 (the floor 1e-9 .. 1e-8, the ~1e-7 window dependence, an overdetermined window is not a fix, Phase C's decomposition is); BUILD_B 3.6 (the three-point fit bounds nothing; `r^(2 lambda) ~ r^1.61`; five radii put the limit on the square to ~3e-5); the B6 mode test renamed `test_b6_fillet_modes_approach_the_square_as_a_power_of_r` with a docstring saying the [8, 32] window accepts both powers; BUILD_B 4.6 (budget oblique resolution by dof, not topology); BUILD_B 3.4 / 3.8 (the 16-step reading pre-asymptotic; the RCWA pairwise rate wanders, Richardson is a 1e-3-class reference) | `6a84c377` |
| N-3 | the mapped stack's missing `n_orders` cap is DELIBERATE: under a map the far field is a quadrature integral and the incident field the window-free modal decomposition, so the per-layer cap's reason (aliasing order slots inside the least-squares overlap) does not arise.  Measured: the nine low orders of the 3 x 3 circle at `n_orders` 2 vs 6 / 9 (`M = 4`, cap 4) and 7 / 10 (`M = 5`, cap 5) agree to <= 9.3e-16 (`d0_norders_cap.json`); documented at the parameter and the projector | `bb7f8c62` |
| the pre-existing walker red | `stack2d_pure.__all__` exports `material_key` -- added by the maintainer's commit `4ec402bc` (the pure-stack viewers) without a re-export or an exemption; registered in the walker's exemption registry with its reason (a viewer palette helper of one class, a generic name); the walker is green | `cba8410b` |

---

## 7. Documentation

Module scope notes of `twod_staggered.py` (the curved-cell paragraph) and
`stack2d_pure.py` (the module docstring, `cmap=`, `add_layer`); the
`shapes2d` module docstring (a new "Materials" section; the known limits now
name only Phase E), the `Shape2D` / `Rect` docstrings and `compile_shapes`
(`background_mu`, `with_mu`); the `Granet2DTransverseE` and
`pmm_jones_2d_staggered` `cmap=` / `shapes=` parameters; `CHANGELOG.md`
(`## [Unreleased]`); `docs/PMM_ROADMAP.md` (the 2-D curved row); a cookbook
example (a liquid-crystal-filled circular hole).  History fingerprints of
`stack2d_pure` re-recorded with the reason (section 8).

---

## 8. Reproduction and test tails

```
cd /c/tmp/lum_curved_d/validation/probe_pmm2d_curved/build_d
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved_d
# D1 (PRE tree = git archive 607d43b0 in C:/tmp/curved_pre_d)
(cd /c/tmp/curved_pre_d && PYTHONPATH=C:/tmp/curved_pre_d python C:/tmp/lum_curved_d/validation/probe_pmm2d_curved/build_d/d1_bytes.py C:/tmp/curved_pre_d pre)
python d1_bytes.py C:/tmp/lum_curved_d post idmap ; python d1_compare.py
(cd ../verify_a && VA_ROOT=C:/tmp/curved_pre_d PYTHONPATH="C:/tmp/curved_pre_d;." python v1_bytes.py phaseDpre)   # and phaseDpost on this tree; moved to d1_v1bytes_*.json
python d1_v1compare.py
python d0_norders_cap.py
python runjobs.py jobs_d2.txt 6 ; python d2_film.py arms ; python d2_film.py scalar
python runjobs.py jobs_d3.txt 5
python runjobs.py jobs_d4.txt 7 ; python d4_pillar.py summary
python runjobs.py jobs_d5.txt 3 ; python runjobs.py jobs_d56.txt 5 ; python d5_li2003.py summary ; python d6_magnetic.py summary
python runjobs.py jobs_d8.txt 3 ; python d8_oblique.py summary
python runjobs.py jobs_d9.txt 4 ; python d9_mutations.py summary
python runjobs.py jobs_d10.txt 1
python runjobs.py jobs_unit.txt 4         # the unit-test readings d_unit_*.json
cd /c/tmp/lum_curved_d && python -m pytest tests/unit/test_pmm2d_staggered_curved_d.py --capture=sys -p no:randomly
```

Test tails, both builds:

* Windows (CPython 3.14.6, numpy 2.4.4, scipy 1.17.1), serial, the box by
  then lightly loaded: `tests/unit/test_pmm2d_staggered_curved_d.py` ->
  `19 passed in 56.66s`; slowest 7.6 s (D2 films, D9 no_sg_e33, D3 films,
  D7 stack); durations spliced into `.test_durations` (19 added).  Earlier,
  saturated, `-n 4`: `17 passed in 111.70s`, slowest 40.6 s.
* Windows, the existing suites (67 files: every `test_*pmm2d*`,
  `*stack2d*`, `*stagger*`, `*curved*` file incl. Phases A / B / C / D and
  both verifiers' files, census, public API, every `__all__` / changelog
  walker, doc identifiers, dispatcher doc consistency, except budget,
  history relocation / lint / fingerprint tool, kernel consistency,
  re-exports), `-n 6`, saturated box: `1642 passed, 8 skipped, 96 warnings
  in 1092.84s` (`suite_win.txt`).  The 8 skips are pre-existing premise
  gates (the mortar round-2 LAPACK premise; seven changelog walkers with
  nothing to verify in the `[5.49.0]` block).  The `__all__` walker on
  `material_key` that was red on `607d43b0` is green (section 6).
* WSL Ubuntu (CPython 3.12.3, numpy 2.4.6, scipy 1.17.1, BLAS pinned,
  `lumenairy` from `/mnt/c/tmp/lum_curved_d`): Phases A + B + C + D, both
  verifiers' files and the `__all__` walker: `90 passed in 341.16s`
  (`suite_wsl.txt`).
* `python -m mypy` (the configured strict list): `Success: no issues found
  in 33 source files`.
* WSL ruff 0.15.16 on `lumenairy/ tests/ scripts/` and `build_d/`: `All
  checks passed!` (12 import-order fixes in the probes, applied).
* History fingerprints: `stack2d_pure` re-recorded with the reason;
  `record_history_fingerprints.py --check`: every document matches.

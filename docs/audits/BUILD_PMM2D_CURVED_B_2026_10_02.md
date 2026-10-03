# BUILD -- curved cells for the pure staggered 2-D PMM, Phase B (transfinite maps, the corner quadrature, the circle / fillet / ellipse / sinusoid gates, oblique and conical incidence)

Date: 2026-10-02.  Status: BUILT + gated, not pushed.
Mount: worktree `C:/tmp/lum_curved_b`, branch `feat/pmm2d-curved-b`, built on
`539ce4a3` (Phase A); Windows 11 (tesla-ryzen), CPython 3.14.6, numpy 2.4.4,
scipy 1.17.1, `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1`
on the command line, `lumenairy.__file__` asserted under the worktree in every
probe (`build_b/_common.py` derives the root from its own location).  PRE tree
for byte identity: `git archive 539ce4a3` extracted to `C:/tmp/curved_pre_b/`.
The box was shared with two other agents' jobs and up to ~35 single-threaded
processes on 24 logical cores for the whole build, so every WALL TIME below is
an upper bound; no accuracy number depends on the load.
Plan: `docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md` section 4.2, approved
with the planner's recommendations on all seven questions of section 7.
Phase A: `docs/audits/BUILD_PMM2D_CURVED_A_2026_10_02.md`.
Evidence: every number below is read from a JSON file in
`validation/probe_pmm2d_curved/build_b/` (the probe that wrote it is named in
each table).  Tests: `tests/unit/test_pmm2d_staggered_curved_b.py`.

---

## 0. Words used here

* **Transfinite (Gordon-Hall) map.**  The general curved map: some EDGES of
  the solver's rectangular `(u, v)` wall grid are declared to be curves (an
  arc, an ellipse arc, a piece of a sinusoid, or a straight line between two
  moved vertices); inside every cell the map is the bilinear blend of that
  cell's four edge curves.  Two cells sharing an edge share its curve, so the
  map is continuous across every grid line (`C0`); its derivative may jump
  there.  Where no edge is curved and no vertex moved it is the identity.
* **Singular vertex.**  A cell corner where `det J = 0`.  A smooth closed
  curve made of grid lines of a tensor grid must turn by 90 degrees in
  `(u, v)` where it does not turn at all in `(x, y)` -- the circle's four
  45-degree points, the fillet's four 45-degree points -- so the cell's two
  edge tangents there are antiparallel.  Nearby the effective tensors grow
  like `1 / distance` (integrable).
* **Corner (Duffy) rule.**  The quadrature this build adopts in every cell
  that owns a singular vertex: the cell (split into quadrants when it owns
  several) is cut into triangles with a vertex ON the singular corner, and
  each triangle is mapped from the unit square with one edge collapsed onto
  that corner (Duffy, SIAM J. Numer. Anal. 19, 1260 (1982)); the collapse's
  Jacobian cancels the `1 / distance`, so the transformed integrand is
  analytic and a Gauss rule on it converges spectrally.
* **Round-off floor.**  The spread of R / T under a random relative
  perturbation of the operator weights at the 1e-15 level: what a different
  BLAS build may legitimately move (section 4, F-B4).
* **Fail-before**, **rung**, **oracle**: as in the plan (0.1).

---

## 1. What was built

| item | file | what it does |
|---|---|---|
| edge curves | `lumenairy/elements/pmm/_curvemap.py` | `EdgeCurve` (value + analytic derivative on `s in [0, 1]`, `key()` for the fingerprint); `Line`, `Arc` (+ `Arc.through(P, Q, center)`, the shorter arc joining two vertex images, refused off the circle), `EllipseArc` (parametric angle, optional rotation), `Sinusoid` (`x = base + A sin(2 pi y / p + phase)` or the `y`-form) |
| `TransfiniteMap(u_walls, v_walls, vertex_images, curved_edges)` | same | the Gordon-Hall blend per cell with analytic Jacobians; endpoint check of every curve against its vertex images (a mismatch RAISES -- it would tear the map); `validate()` (lattice periodicity, boundary onto itself, `det J > 0` on a 12 x 12 probe grid of every cell); `singular_vertices` (corners where the two edge tangents are parallel to 1e-10); `geom_points` (pointwise evaluation, vectorised; the base class gets a generic fallback); fingerprint over the vertex images and every curve parameter.  Lifted from the planning probe's `TransfiniteMap` / `Line` / `Arc` (`validation/probe_pmm2d_curved/_curved_scratch.py` lines 123-210) |
| gate map builders (private) | same | `_circle_map_3x3`, `_circle_map_5x5`, `_fillet_map_5x5` (lifted from the planning probe's builders, lines 213-329), `_ellipse_map_3x3`, `_sine_stripe_map_3x3` (new).  Phase C wraps them as public shape primitives |
| `CellMap.validate` (amended) | same | lattice periodicity of the TANGENTIAL derivative only (F-B1) |
| the corner rule | `lumenairy/elements/pmm/twod_staggered.py` | `_stag_duffy_points` (reference points + weights), `_stag_map_singular_corners` (from the map's `singular_vertices`), `_StagMapQuad` (the shared tensor rule + a point rule per corner cell, in the axis-rule layout of `_stag_map_quad_rule`), `_StagNodeWeight` (tensor + point weights); `_stag_map_weights` evaluates the effective tensors on both (one formula, `_stag_map_eff`; `det J > 0` checked at EVERY node of both rules); `_stag_quad_weighted` sums a corner cell over its points through the SAME axis-factor kernel `_stag_quad_axis_factor` (cache keyed by rule and axis, F-B3); `_stag_map_nodes` measures each corner cell's moments on its own rule, with ONE scale per metric tensor (F-B5) |
| tests | `tests/unit/test_pmm2d_staggered_curved_b.py` | gates B1-B10 as decisions (section 3) |

Not touched: the shipped (no-map) assembly, `Basis1D`, the far projector
(Phase A already sizes it by the physical extent of each cell, plan 2.5 --
its integrand, the cofactor times the field, is bounded and analytic in a
corner cell, so it needs no corner rule), `stack2d_pure.py`, the eigensolver.

---

## 2. THE FIRST DESIGN DECISION -- the quadrature of the singular vertices

Phase A's finding F1 predicted that its adaptive rule (double `nq` from
`2 M + 8` until the Legendre moments of the five geometric weights agree to
1e-13) would run to its 256-node cap on the circle.  Measured first, before
any gate (`b1_quadrature.py`).

### 2.1 What the moment criterion demands (`b1_quadrature_demand_M6.json`)

Per cell, the moment self-gap between `n` and `2 n` nodes per axis, `M = 6`:

| cell | n = 20 | 40 | 80 | 160 | 320 | 640 | 1280 | demand |
|---|---|---|---|---|---|---|---|---|
| every regular cell (c3: 8 cells; c5: 21 cells), curved or not | 5.9e-15 .. 1.7e-14 | | | | | | | **20** (`2 M + 8`) |
| c3 disk cell (four singular corners) | 5.6e-3 | 1.4e-3 | 3.6e-4 | 9.1e-5 | 2.3e-5 | 5.7e-6 | 1.4e-6 | none (`~1e5` by extrapolation) |
| c5 corner cells (one singular corner each) | 1.3e-3 | 3.3e-4 | 8.4e-5 | 2.1e-5 | 5.3e-6 | 1.3e-6 | 3.3e-7 | none |

F1 is confirmed and sharpened: only the cells that OWN a singular vertex are
the problem, and there the moments converge ALGEBRAICALLY, exactly `n^-2` (a
factor 4 per doubling: the tensor Gauss rule on a homogeneous `1 / r` corner
singularity).  The criterion can never be met there.

### 2.2 What it costs in R / T (`b1_quadrature_ladder_*.json`, `b1_quadrature_summary.json`)

Full library solves with the per-axis node count FORCED (plain tensor rule in
every cell -- Phase A's family), distance to the corner-rule limit (the Duffy
top rung of 2.4), max over the nine orders of both inputs:

| nq | c3, M = 6 | c3, M = 8 | c5, M = 5 |
|---|---|---|---|
| 16 | 5.6e-07 | 8.2e-08 | 3.3e-09 |
| 24 | 2.8e-07 | 3.0e-08 | 1.5e-09 |
| 48 | 7.2e-08 | 6.9e-09 | 4.1e-10 |
| 96 | 1.8e-08 | 1.7e-09 | 1.6e-10 |
| 256 (the cap) | 2.7e-09 | 3.5e-10 | 1.0e-10 |
| 512 | 1.0e-09 | 6.7e-11 | 4.4e-11 |

**The planner's 2.8e-8 is reproduced exactly** -- `nq = 24` against `nq = 96`
at `M = 8` reads 2.80e-08 -- **and corrected in its reading**: it is not a
converged quadrature error but the difference of two members of an `n^-2`
family (the `nq = 96` rule is itself 1.7e-9 off).  R / T are far less
sensitive than the operators: the plain rule's OPERATORS (`L`, `R`) at
`nq = 2 M + 8` are 6.3e-2 (c3, `M = 6`) and 7.3e-2 (c3, `M = 8`) relative from
the converged ones (`b1_quadrature_duffy_*.json`, field
`op_plain_to_duffy_top`) while R / T move 3e-7 / 3e-8.  The singular part of
the weights acts on the field combination that a smooth physical field makes
vanish at the vertex (`adj(g) / det J ~ (c, 1)(c, 1)^T / r` with
`c E'_u + E'_v -> 0`), which is why the planner found it "harmless" -- but a
gate that read the operators (a Phase D tensor, an absorption integral) would
see 6 %.

### 2.3 The three options (`b1_quadrature_rules_M{6,8}.json`, `b1_quadrature_reparam_M6.json`)

Moment error in the singular cells against a Duffy rule at 200 x 200 nodes
per triangle (self-gap 1.2e-14 .. 2.0e-14 against 160), `M = 8`, by the
number of POINTS in the cell:

| rule | c3 disk cell | c5 corner cell |
|---|---|---|
| (a) plain tensor Gauss, 24^2 = 576 / 256^2 = 65536 / 512^2 points | 5.3e-3 / 4.8e-5 / 1.2e-5 | 1.3e-3 / 1.1e-5 / 2.8e-6 |
| (b1) geometrically graded tensor rule (ratio 0.15, 8-16 layers per end) | 1.4e-4 at 5184; 6.5e-9 at 41616; 2.0e-12 at 160000 | 2.7e-4 at 1600; 7.8e-9 at 11664; 2.6e-12 at 43264 |
| (b2) **Duffy corner rule**, n per direction per piece | 2.2e-5 at n = 12 (1152); **2.7e-13 at n = 16 (2048)**; 8.3e-15 at n = 20 (3200) | 2.0e-6 at n = 16 (512); **1.9e-14 at n = 20 (800)** |
| (c) re-parametrise the cell, `s = q(s')` per axis (plain Gauss self-gap n vs 2n, 20 .. 640) | `q' = 0` at the ends (cubic): 0.93 .. 0.94 -- NO convergence (the weights become non-integrable along whole edges); sqrt-type `q = (2/pi) asin s'`: 1.7e-2 .. 5.3e-4 (`n^-1`, slower than doing nothing, `n^-2`) | -- |

Why (c) cannot work: the metric weights of a per-axis re-parametrised map
scale like `q'(s) / q'(t)`, so any `q` that tames the vertex makes the weight
singular along the whole edge where `q'` vanishes (or blows up); a vertex
singularity becomes an EDGE singularity.  A conformal-type local map would
avoid that but makes `chi33 = 1 / det J ~ 1 / r^2`, non-integrable.  And the
`det J = 0` itself is topological (the plan, 2.6): no re-parametrisation of a
tensor grid removes it.

(b1) works but needs 40-160 thousand points per cell for 1e-9 .. 1e-12; (b2)
reaches 1e-13 with 2048 points (c3) and 800 (c5) and its integrand is
analytic, so the SAME adaptive moment criterion now passes at the first
check: the solver picks `nq = 2 M + 8` on every circle and fillet map
(recorded per rung in `b4_circle_*.json`, field `nq`).

### 2.4 The decision, and its R / T ladder (`b1_quadrature_duffy_*.json`)

**Decision: (b2), the Duffy corner rule in the cells that own a singular
vertex; the tensor rule everywhere else.**  The corner rule's own ladder
(node count forced; distance to its top rung; operators to their top rung):

| n | c3, M = 6: R / T / ops | c3, M = 8: R / T / ops | c5, M = 5: R / T / ops |
|---|---|---|---|
| 8 | 6.8e-08 / 1.5e-03 | 8.5e-07 / 2.6e-02 | 9.0e-10 / 1.6e-03 |
| 12 | 9.0e-10 / 2.8e-09 | 1.3e-11 / 1.8e-05 | 1.9e-10 / 3.3e-11 |
| 16 | 1.0e-09 / 7.0e-13 | 1.8e-11 / 9.9e-12 | 9.4e-11 / 9.2e-14 |
| 20 | 9.4e-10 / 8.9e-13 | 1.2e-11 / 3.4e-12 | 7.4e-11 / 9.8e-14 |
| 24 | 3.7e-10 / 4.2e-13 | 2.3e-11 / 3.1e-12 | 7.6e-11 / 7.1e-14 |
| 32 | 1.2e-09 / 5.5e-13 | (top) | 7.8e-11 / 7.5e-14 |
| top | 48 | 32 | 48 |

The operators converge to round-off by `n = 16`; R / T plateau at the
round-off floor of the mapped solve (section 4, F-B4: 6.8e-10 at `M = 6`,
1.4e-11 at `M = 8`, measured independently by perturbation), not at a
quadrature floor.  The plain rule's distance to this limit falls as `n^-2`
with no floor of its own (2.2), so the two families converge to the same
answer -- an independent cross-check of the corner rule.  **No floor above
1e-6 remains (stop condition 2 not met): the quadrature error at the shipped
`nq = 2 M + 8` is below the round-off floor.**

Cost: section 5 (`b1_quadrature_cost.json`).

---

## 3. The gates

The task's gate list (B1-B10 as numbered in the Phase-B brief; the plan's
table 4.2 uses a different numbering -- its geometry, C0, symmetry and
two-topology rows are folded into B1, B3 and B4 here).  Every number is from
this tree, 2026-10-02.  "Unit test" names the decision in
`tests/unit/test_pmm2d_staggered_curved_b.py`; "build doc" ladders are too
expensive for a unit test and are measured here only.

### 3.1 B1 -- no map = today's bytes; the transfinite identity is the identity map

| claim | measured | bar | fail-before / upper gap | JSON |
|---|---|---|---|---|
| no map = the bytes of `539ce4a3` | **109 / 109** SHA-256 identical (Phase A's fixture set: operators of every dispatch branch, region modes, geometric cache, far projectors, `pmm_efficiency_2d_staggered` / `pmm_jones_2d_staggered` / stack / mortar outputs and absorption) | equality | the identity map through the quadrature path: 28 of 36 operator hashes differ (the 8 equal ones are `None` attributes) | `b2_bytes_pre.json`, `b2_bytes_post.json`, `b2_compare.json` |
| the unit-test restatement: no Phase-B function is reached without a map | booby-trapped `_stag_duffy_points`, `_StagMapQuad`, `_stag_map_singular_corners`, `_stag_map_eff`, `_stag_map_geom5`, `_stag_quad_weighted`, `_stag_map_nodes`, `_far_projector_mapped`; a two-layer oblique stack with absorption solves | no trap fires | -- | unit test `test_b1_no_map_never_reaches_the_phase_b_code` |
| `TransfiniteMap(walls)` (no curve, identity vertices) = `IdentityMap` = the kron assembly | Phase A's A2 fixture (3 x 3 pillar, non-uniform walls): operators vs kron <= 9.9e-15 (`M = 5`), 4.8e-14 (`M = 7`); vs `IdentityMap` <= 5.0e-15 / 3.3e-14; R / T vs kron 6.4e-15 / 4.1e-14; node count 18 / 22 = `2 M + 8` for both maps, no corner cell | rel `<= 1e-11` (Phase A's A2 bar, 2.3 decades above) | one vertex image moved by 1e-6 p moves `L` by > 1e-9 (asserted) | `b2_identity.json` |
| the Gordon-Hall evaluation | a straight-edged map with four moved interior vertices against the closed-form bilinear map and its derivatives: 8.9e-16 | -- | -- | `b2_identity.json` (`bilinear_closed_form_max_abs`) |
| geometry | mapped disk area (3 x 3 and 5 x 5), ellipse, fillet pillar, whole cell: <= 5e-16 relative; C0: the shared edge (position and tangent) from both cells <= 4.4e-16; analytic Jacobian vs central difference <= 2e-10 | 1e-12 / 1e-14 / 1e-8 | a vertex image moved by 1e-9 p RAISES (endpoint check); a circle too large for its cell RAISES (`det J <= 0` inside) | unit test `test_b1_transfinite_geometry_and_construction` |

### 3.2 B2 / B8 -- a uniform film under the curved maps is exact, through the singular vertices

A uniform eps-4 film passed as a PATTERNED cell: the only thing that can be
wrong is the map.  Max over every order and both inputs against the Airy slab
(`b3_film_normal.json`, `b3_film_nocof_plain.json`):

| map | M = 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| circle 3 x 3 (corner rule) | 8.1e-07 | 2.8e-08 | 4.3e-11 | 3.3e-13 | 5.4e-14 |
| circle 3 x 3, Phase-A tensor rule everywhere | 8.1e-07 | 2.9e-08 | 4.2e-11 | 3.3e-13 | 6.4e-14 |
| circle 5 x 5 | 5.7e-10 | 9.2e-13 | 2.2e-14 | 4.2e-14 | |
| ellipse 3 x 3 (0.40 x 0.28) | 1.3e-06 | 3.0e-08 | 4.7e-11 | 2.9e-13 | |
| fillet 5 x 5 (r / side 0.2) | 1.1e-09 | 9.6e-13 | | | |
| no-cofactor far projector (fail-before), `M = 6` | | | 0.148 (c3), 0.110 (c5) | | |

Spectral on every map, down to round-off; the planner's film ladder (4.0e-6,
1.2e-7, 6.9e-10, 4.7e-12, 3.9e-14, plan 3.3) is matched in rate and floor and
is 1.6-16x larger at low M (the scratch solver differs from the library in
its eigensolver and half-space handling, not in the map).  B8: the film
cannot see the corner quadrature -- the corner rule and the tensor rule agree
to two digits -- because a uniform field makes the singular part of the
weights act on a vanishing combination (section 2.2); the operator-level
difference (6 %) is what the corner rule fixes, and the unit test gates THAT
(`test_b8_corner_rule_decision_on_the_circle`: corner-rule operators at
`n = 20` vs `n = 40` <= 1e-11, measured 8.9e-13; the plain rule's 6.3e-2
above the bar 1e-3; the plain criterion WARNS at its cap) together with the
ONE-kernel identity (`test_b8_corner_rule_is_the_tensor_kernel_on_smooth_weights`:
the corner rule on polynomial weights equals the tensor rule, all blocks,
bar 1e-12).

Unit test `test_b2_film_under_the_circle_map_is_exact_and_spectral`: `M = 6`
<= 1e-9 (measured 4.3e-11; 1.4 decades), the `M = 4 -> 6` drop >= 3 decades
(measured 4.3); no-cofactor at `M = 4` > 1e-2.

### 3.3 B3 -- the circular pillar against the saved FEM oracle

FEM oracle: `fem/summary.json` (NGSolve 6.2.2604, three meshes `h1.0_e20 p4`,
`h0.8_e30 p4`, `h1.0 p6`; its own spread 8.34e-6; `R + T - 1` = 2.7e-7) --
provenance asserted by the probe and the unit test, not re-run (no fixture
changed).  Largest per-order distance (E along y, 18 entries), closure, the
four-fold symmetry `te(m, n) - tm(n, m)`, and the distance to the planning
probe's scratch solver at the same rung (`b4_circle.json`):

| map, M | dof | nq | to FEM | rung change | closure | symmetry | vs planner scratch | wall (s) |
|---|---|---|---|---|---|---|---|---|
| c3, 6 | 450 | 20 | 6.21e-03 | 5.8e-03 | 1.0e-03 | 8.9e-10 | 3.5e-07 | 27 |
| c3, 7 | 648 | 22 | 8.18e-04 | 3.6e-04 | 3.0e-05 | 2.4e-10 | 1.4e-07 | 67 |
| c3, 8 | 882 | 24 | 4.58e-04 | 4.5e-04 | 2.6e-05 | 1.0e-11 | 3.0e-08 | 178 |
| c3, 9 | 1152 | 26 | 2.83e-05 | 1.3e-05 | 6.7e-07 | 1.9e-12 | 5.7e-10 | 388 |
| c3, 10 | 1458 | 28 | 1.68e-05 | 1.1e-05 | 2.7e-07 | 2.4e-12 | 3.2e-10 | 858 |
| **c3, 11** | 1800 | 30 | **7.77e-06** | 6.8e-06 | 5.7e-09 | 3.6e-12 | 1.5e-10 | 1524 |
| **c3, 12** | 2178 | 32 | **1.93e-06** | -- | 1.7e-09 | 7.1e-12 | 7.2e-11 | 2715 |
| c5, 4 | 450 | 16 | 4.63e-04 | 3.8e-04 | 6.4e-05 | 2.2e-09 | 7.8e-08 | 36 |
| c5, 5 | 800 | 18 | 8.40e-05 | 4.1e-05 | 1.7e-06 | 5.7e-12 | 4.0e-09 | 124 |
| c5, 6 | 1250 | 20 | 4.32e-05 | 2.2e-05 | 1.8e-08 | 1.6e-11 | 1.0e-09 | 554 |
| c5, 7 | 1800 | 22 | 2.16e-05 | 1.1e-05 | 1.2e-10 | 4.3e-14 | 4.6e-10 | 1614 |
| c5, 8 | 2450 | 24 | 1.02e-05 | -- | 1.9e-12 | 3.2e-13 | 2.3e-10 | 3958 |

**STOP condition 1 is not met**: at `M = 11` on the 3 x 3 map the circle is
7.77e-06 from the oracle, inside the FEM's own 8.3e-6 spread and 3.2x under
the stop bar 2.5e-5; at `M = 12` 1.93e-06.  The planner's P3 ladder is
reproduced rung for rung (6.2e-3, 8.2e-4, 4.6e-4, 2.8e-5, 1.7e-5, 7.8e-6,
1.9e-6); the library's own answer differs from the scratch solver's by the
scratch's tensor-rule quadrature error (3.5e-7 at `M = 6` falling to 7.2e-11
at `M = 12`, the `n^-2` family of section 2.2 at `nq = 2 M + 8`).  The node
count the adaptive criterion picks is `2 M + 8` on every rung (column nq): the
corner rule made the singular cell pass at the first check.  The two
topologies agree to 9.3e-06 (c3 `M = 12` vs c5 `M = 8`; the planner's 9.3e-6).  The four-fold symmetry holds to the round-off
floor (<= 8.9e-10, F-B4).

Unit test `test_b3_circle_lands_on_the_saved_fem_oracle` (c3, `M = 7`):
8.18e-04 <= 2.5e-3 (0.5 decades); fail-before the 4-step staircase at the
same `M`, measured 7.07e-2, > 10x the bar.

### 3.4 B4 -- the staircase limit

The shipped staggered PMM on the planner's 4k-step staircases (walls at
`c +- r i / k`, a cell filled when its centre is inside the circle; the
shipped code is byte-identical, so the planner's saved staircase JSON are
this tree's answers -- `p3_circle_stair_k{1,2,4}.json`): distance to the FEM
7.07e-02 (k = 1, `M = 10`), 5.36e-02 (k = 2, `M = 7`), 5.65e-03 (k = 4,
`M = 4`) -- the planner's 7.1e-2 / 5.4e-2 / 5.7e-3 reproduced
(`b4_circle.json`, `stair`).  Ordering: strictly decreasing.  Direction: the
cosine between each staircase step and (curved answer - previous staircase)
is 0.930 (k = 1 -> 2, `M = 10 / 7` vs c3 `M = 10`), 0.991 (k = 2 -> 4) and
0.997 (k = 1 -> 4 vs the FEM); at unit-test sizes (k = 1 at `M = 7`, k = 2
at `M = 5`, curved c3 `M = 7`) 0.933 (`b4_stair_direction.json`).
Two-sided: the curved answer sits within 2.5e-3 of the FEM while every
staircase stays >= 5.2e-2 from it (a curved solve that secretly staircased
would collapse onto them).  Unit test
`test_b4_staircases_converge_toward_the_curved_circle`: `d(k1) > d(k2) >=
1e-2`, cosine >= 0.8.

### 3.5 B5 -- the in-plane Bloch modes: spectral for the circle, stalled for a square

The leading four `n_eff^2` of the layer's own pencil (no rim, so only the
cross-section is seen), rung-to-rung change (`b5_modes_*.json`):

| cross-section | rung changes |
|---|---|
| circle 3 x 3 | 6->7 1.3e-02, 7->8 2.8e-03, 8->9 5.8e-04, 9->10 4.0e-05, 10->11 6.5e-06, 11->12 3.3e-07 |
| circle 5 x 5 | 4->5 2.2e-03, 5->6 2.1e-05, 6->7 1.5e-06, 7->8 4.7e-08, 8->9 2.5e-09 |
| square pillar (3 x 3, no map) | 6->7 5.0e-05, 7->8 5.9e-05, 8->9 1.5e-05, 9->10 3.6e-05, 10->11 5.9e-06, 11->12 1.6e-05 |
| y-uniform stripe (no corner) | 6->7 5.6e-06, 7->8 9.3e-08, 8->9 1.2e-09, 9->10 1.2e-11, 10->11 8.8e-13 |
| fillet r / side 0.2 (5 x 5) | 5->6 2.6e-04, 6->7 2.5e-06, 7->8 1.1e-06, 8->9 1.8e-07 |
| fillet r / side 0.05 (5 x 5) | 5->6 5.6e-03, 6->7 2.3e-05, 7->8 6.3e-06, 8->9 3.8e-06 |

The planner's ladders are reproduced to two digits (the planner ran the
tensor rule: the modes, like the film, do not see the corner quadrature).
DECISION on the RATE (unit tests `test_b5_circle_modes_converge_spectrally`,
`test_b5_square_pillar_modes_stall`): the circle's change falls >= 5x per
rung and is <= 1e-5 by `M = 6 -> 7` on the 5 x 5 map (measured 13.5x and
1.5e-6); the same decision FAILS for the square at `M = 8, 9, 10` (ratio
0.42: the change grows) -- the square is the fail-before.

### 3.6 B6 -- the fillet ladder, monotone in r, and the r -> 0 limit

Square pillar side 0.6 with its corners rounded to `r`, the 5 x 5 fillet map
(walls through the 45-degree points and the tangency points; four singular
vertices); the sharp square is the shipped solver on the 3 x 3 walls.
Input 'te' (R00 / T00 are polarization-independent here), each fillet at its
top rung against the sharp square at `M = 12` (`b6_fillet.json`, `limit`):

| r / side | top M | dR00 | dT00 | own rung change R00 / T00 |
|---|---|---|---|---|
| 0.01 | 7 | +2.4e-06 | -2.9e-05 | 4.6e-05 / 2.4e-04 |
| 0.02 | 7 | -4.7e-05 | +5.5e-05 | 4.1e-05 / 1.9e-04 |
| 0.05 | 8 | -2.63e-04 | +4.63e-04 | 7.6e-06 / 1.7e-05 |
| 0.1 | 8 | -7.51e-04 | +1.73e-03 | 7.2e-06 / 1.3e-05 |
| 0.2 | 8 | -1.73e-03 | +7.36e-03 | 8.5e-06 / 1.4e-05 |
| sharp square ladder (`M = 6 .. 12`) | | | | rung changes 1.4e-3, 8.8e-5, 2.1e-5, 2.5e-5, 5.3e-6, 1.1e-5 (all orders) |

The planner's P4 shifts (R00 -2.6e-4 / -7.5e-4 / -1.7e-3, T00 +4.6e-4 /
+1.7e-3 / +7.4e-3) are reproduced.  MONOTONE in r for r / side >= 0.05, each
step 3-50x above the fillets' own rung changes.  The r -> 0 LIMIT: the
constant of the fit `dT00 = c + a r^2 + b r^3` through r / side = 0.05, 0.1,
0.2 is +6.9e-05 (`dR00`: -5.4e-05) -- 0.9 % (3 %) of the r / side = 0.2
shift, four to five times the fillets' own top-rung changes (1.3e-5 ..
1.7e-5) and six times the sharp square's (1.1e-5): a three-point fit with
no `r^4` term, so it bounds the limit at the 5e-5 level rather than proving
it to 1e-5; the two
small fillets (r / side 0.01, 0.02, inside the sliver band of plan 4.3,
fillet segment 1.5e-3 and 2.9e-3 of the period) sit within their OWN rung
change of the square (|dT00| 2.9e-5, 5.5e-5 against 2.4e-4, 1.9e-4).  So the
limit is the sharp square to the fillets' convergence, which is slower than
the square's for small r -- the curvature of an arc of radius `r` must be
resolved inside a cell `r / sqrt 2` wide -- and the plan's "to the sharp
square's own convergence" is met only to within that factor.  STOP condition
(plan 4.2: a fillet with r / side >= 0.05 not below 1e-4 per rung by
`M = 8`): not met (the all-orders rung change at 7 -> 8 is 1.7e-5 / 1.3e-5 /
1.4e-5 for r / side 0.05 / 0.1 / 0.2).

Unit tests: `test_b6_fillet_moves_the_efficiencies_far_above_convergence`
(r / side 0.2 at `M = 5` vs the square at `M = 6`: R00 -1.80e-3 <= -5e-4,
T00 +7.65e-3 >= +2e-3) and `test_b6_fillet_modes_approach_the_square_as_r_squared`
(the leading Bloch `n_eff^2`, which sees no rim: square 3.0079446, r / side
0.05 3.0070140, 0.2 2.9901856 (`b5_modes_*.json`), monotone, and the
distance ratio 19 against r^2's 16, inside [8, 32] -- an offset limit would
read ~1, a linear one 4).

### 3.7 B7 -- a same-area square is not a fillet

The square shrunk to the filleted pillar's area, shipped solver
(`b6_fillet.json`, `eqarea`): R00 +3.2e-4 against the sharp square at `M = 7`
(r / side 0.2; +2.4e-4 at `M = 6`; +3.2e-4 at `M = 11` against the square
at `M = 11`), +1.2e-4 at r / side 0.1 and +3.1e-5 at 0.05 (`M = 11`) -- the WRONG direction (the fillet moves R00 by -1.7e-3 /
-7.4e-4) -- and it overshoots T00 (+1.0e-2 against +7.4e-3 at r / side 0.2,
`M = 7`).  Unit test `test_b7_same_area_square_is_not_a_fillet`: same-area
shift > +5e-5 (measured +2.4e-4 at `M = 6`) while the fillet's is negative,
and the two answers differ by >= 1e-3 in R00 (measured 2.04e-3).

### 3.8 B9 -- two new shapes: an ellipse and a sinusoidal wall

No planner numbers.  References: the shipped 2-D RCWA with the EXACT form
factor (Laurent; the ellipse is a shipped shape; the sinusoidal ridge is
added probe-side through its Jacobi-Anger form factor
`F(m, n) = J_-n(m G_x A) (e^{-i m G_x x2} - e^{-i m G_x x1}) / (-i m G_x p_x)`,
checked against a 4096^2 pixel FFT to 4.6e-6), and the staircase limit of
the shipped staggered PMM (`b9_shapes.json`):

| shape | curved ladder: rung change (M) | closure at top | RCWA, distance to the curved top rung (orders per axis) | 1 / N Richardson pair | staircases (k, M): distance |
|---|---|---|---|---|---|
| ellipse 0.40 x 0.28, 3 x 3 map | 7.5e-3 (6), 5.9e-4 (7), 5.9e-4 (8), 2.7e-5 (9), 2.3e-5 (10); top `M = 11` | 6.3e-9 | 1.29e-2 (9), 9.1e-3 (13), 7.0e-3 (17), 5.7e-3 (21), 4.8e-3 (25), 4.1e-3 (29) | 1.9e-3, 1.2e-3, 6.1e-4, 4.2e-4, 3.0e-4 | 9.1e-2 (1, 8), 5.3e-2 (2, 6), 6.1e-3 (4, 4) |
| sinusoidal ridge, A = 0.12, 3 x 3 map | 1.6e-3 (6), 9.9e-5 (7), 3.2e-5 (8), 1.8e-5 (9), 9.3e-6 (10); top `M = 11` | 2.6e-11 | 2.14e-2 (9), 1.45e-2 (13), 1.10e-2 (17), 8.8e-3 (21), 7.3e-3 (25), 6.3e-3 (29) | 1.2e-3, 6.7e-4, 4.6e-4, 3.2e-4, 2.4e-4 | 6.4e-2 (1, 6), 6.8e-2 (2, 6), 1.7e-2 (4, 4) |

Both behave like the circle: the curved ladder converges to the rim-capped
1e-5-per-rung level, the exact-form-factor RCWA approaches it algebraically
(and its Richardson extrapolation keeps falling toward it, with no sign of
converging elsewhere), the staircases approach it from far away -- for the
ellipse monotonically (9.1e-2, 5.3e-2, 6.1e-3), for the sinusoid only at the
8-row staircase (the 2-row and 4-row staircases sample the wall at +-A and
+-0.71 A and sit equally far, 6.4e-2 / 6.8e-2; 1.7e-2 at 8 rows).  The
ellipse's four-fold symmetry is genuinely broken (`te(m, n) - tm(n, m)` 0.11,
against <= 8.9e-10 for the circle), the plan's B4 fail-before.  Unit tests
`test_b9_ellipse_against_the_exact_form_factor_rcwa` (curved `M = 7`; RCWA at
9 / 13 orders 1.35e-2 > 9.8e-3 >= 6e-3; Richardson 2.5e-3 <= 6e-3; symmetry
broken >= 1e-2) and `test_b9_sinusoidal_wall_mirror_and_exact_form_factor`
(the mirror identity `R(m, n; A) = R(m, -n; -A)` <= 1e-7; RCWA 2.13e-2 >
1.45e-2 >= 6e-3; Richardson 2.7e-3 <= 6e-3).

### 3.9 B10 -- oblique (25 deg) and conical (25 deg, phi 40 deg) incidence on the circle

Not probed by the planner.  `b10_oblique.json` (from `b10_*.json` rungs):

| quantity | theta 25, phi 0 | theta 25, phi 40 |
|---|---|---|
| c3 rung change, `M = 6 -> 10` | 1.7e-2, 6.2e-3, 8.1e-4, 2.5e-4 | 3.0e-2, 5.7e-3, 1.7e-3, 2.4e-4 |
| c3 closure, `M = 6 .. 10` | 1.9e-3, 6.8e-4, 4.1e-5, 1.3e-5, 5.3e-7 | 5.4e-3, 4.8e-4, 2.0e-4, 1.3e-5, 2.9e-6 |
| c5 rung change, `M = 4 -> 7`; closure at 7 | 9.8e-4, 5.7e-5, 2.8e-5; 8.3e-9 | 1.3e-3, 3.0e-5, 3.6e-5; 2.6e-8 |
| the two topologies (c3 `M = 10` vs c5 `M = 7`), polarization-summed R per order | 1.3e-5 | 6.7e-6 |
| y-mirror `R(m, n) = R(m, -n)` | 5.2e-14 .. 1.4e-12 (`M <= 9`), 1.2e-8 (`M = 10`) | (no symmetry) |
| RECIPROCITY, channel -> reflected order (-1, 0) vs its reversal (singular values of the power-normalized Jones block) | c3: 3.1e-6, 8.3e-7, 4.1e-8, 9.7e-9, 3.5e-9 (`M = 6 .. 10`); c5: 8.8e-8 .. 2.6e-12 | c3: 3.9e-5, 9.4e-6, 6.7e-7, 1.3e-7, 2.7e-9; c5: 5.5e-6 .. 4.3e-11 |
| reciprocity, order (0, -1) | -- | c3: 3.7e-5, 1.7e-5, 1.9e-7, 2.4e-7, 7.3e-9; c5: 1.6e-5 .. 4.1e-9 |
| wrong pairing (the reversed run's specular channel) | 0.093 .. 0.11 | 0.094 .. 0.12 |
| engineered defect (no cofactor), `M = 8` | reciprocity 1.7e-4 (against 4.1e-8 correct), closure 0.12 | -- |
| RCWA, exact disk form factor, 9 .. 29 orders: distance to c3 `M = 10` (polarization-summed R) | 4.1e-3, 2.9e-3, 2.2e-3, 1.75e-3, 1.45e-3, 1.25e-3 | 2.3e-3, 1.7e-3, 1.3e-3, 1.1e-3, 9.0e-4, 7.8e-4 |
| film under the map vs Airy (s / p), c3, `M = 4 .. 8` | 7.2e-4, 6.8e-5, 3.5e-6, 2.1e-7, 1.0e-8 | 3.3e-4, 2.7e-5, 1.4e-6, 6.3e-8, 3.4e-9 |

Reverse angles (incidence along `-k_mn`): (24.2498 deg, 0) for (-1, 0) at
phi = 0; (35.2731 deg, -28.0614 deg) for (-1, 0) and (40.4136 deg,
119.9586 deg) for (0, -1) at phi = 40.  Reciprocity is an independent
physical identity (the solver does not impose it); it holds three to five
decades better than the rung-to-rung convergence and is broken by the
engineered defect by 3.6 decades.  The Bloch glue, the `alpha0` kernel and
the corner rule need nothing new at oblique or conical incidence under a
periodic curved map.  The FEM oracle was NOT run at oblique incidence: its
runner is a quarter-cell mirror-symmetry formulation (PMC / PEC walls) valid
only at normal incidence; an oblique run needs a Bloch-periodic full cell
(about 4x its 371k-413k unknowns), which this build did not attempt.  The
reference at oblique is therefore the exact-form-factor RCWA (floor: 1.25e-3
/ 7.8e-4 at 29 orders, algebraic) plus the two independent topologies
(1.3e-5 / 6.7e-6) and reciprocity.

Unit tests: `test_b10_oblique_circle_is_reciprocal_and_mirror_symmetric`
(`M = 6`: reciprocity <= 3e-5, wrong pairing >= 1e-2, mirror <= 1e-6),
`test_b10_conical_circle_is_reciprocal` (`M = 6`: <= 3e-4, wrong >= 1e-2),
`test_b10_film_under_the_circle_map_at_conical_incidence` (`M = 6` <= 1e-5,
`M = 4 -> 6` drop >= 1.5 decades).

---

## 4. Findings

### 4.1 F-B1 -- only the TANGENTIAL derivative must be periodic (formulation finding, fixed)

Phase A's `CellMap.validate` required all four Jacobian entries to agree on
the two sides of the periodic seam (its F2 recorded "lattice periodicity of
`Phi - id` and of `J`").  Every transfinite map violates that, because the
derivative ACROSS a side (`dPhi/du` on `u = 0, p_x`) is set by each side's
own cell blend: measured seam mismatch of the full Jacobian 6.21e-2 (3 x 3
circle), 2.53e-2 (5 x 5 circle), 2.13e-2 (fillet), 5.93e-2 (ellipse), 0.797
(sinusoidal stripe), while positions and the TANGENTIAL derivative agree to
<= 4.4e-16 (`b0_validate.json`).  What the solver needs is exactly the
tangential part: the unknown continuous across the side `u = 0 ~ p_x` is
`E'_v = E . dPhi/dv` (the staggered basis is broken in `u` for `E'_u`), the
same reason the map may be only C0 across every interior grid line.  The
check is now tangential-only; the circle maps it admits are the ones that
land on the FEM (3.3).

### 4.2 F-B2 -- the planner's "harmless" vertex quadrature is an `n^-2` family; R / T hide a 6 % operator error (measured, fixed by the corner rule)

Section 2.  The planner's 2.8e-8 (24 vs 96 nodes, `M = 8`) is reproduced
exactly (2.80e-08) and is the difference of two members of an algebraic
family, not a converged error; the plain rule's operators are 6.3-7.3 %
wrong at the default node count while R / T, the film and the Bloch modes
barely notice.  Any later gate that reads the operators directly (Phase D's
tensors, an absorption density) would have seen it.

### 4.3 F-B3 -- the corner rule's factor cache collided across axes when one basis serves both (build defect, fixed)

`_stag_quad_weighted` cached a corner cell's per-axis factors by cell; when
the SAME `Basis1D` object is passed for x and y, the y factor was served the
x factor built on the other axis' points: 0.43 relative on the 18-block
kernel identity (`test_b8_corner_rule_is_the_tensor_kernel_on_smooth_weights`,
first run).  The solver always builds two `Basis1D` objects, so no solve was
affected; the cache key now names the axis (commit `66624d61`).

### 4.4 F-B4 -- the curved solve's R / T carry a ~1e-9 round-off floor from the minimum-norm incident projection (formulation finding, measured, NOT changed)

A random relative perturbation of 1e-15 of the HALF-SPACES' effective
weights (the patterned layer untouched) moves the 3 x 3 circle's R / T by
6.8e-10 at `M = 6` and 1.4e-11 at `M = 8` (`b11_floor_pert_*.json`); the
same perturbation of the layer alone moves them 2.4e-14 and, of the
half-spaces under an identity map on the same walls, 8.4e-15
(`b11_floor_pert_M6_n3_layer_c3.json`, `b11_floor_pert_M6_n3_homog_id.json`).  The
cause is `cinc = lstsq(Hsup, rhs)` in `PMM2DStackPure.solve` (Phase A, plan
2.5): with the default far-field window (`n_orders = 3`, 98 equations) and
450 half-space modes the system is UNDERdetermined, and under a curved map
every half-space mode spreads over all orders, so the minimum-norm draw
depends on the basis inside the exactly degenerate plane-wave multiplets.
Widening the window until the system is overdetermined (`n_orders = 10`, 882
equations) drops the floor to 7.7e-15 (`b11_floor_pert_M6_n10.json`) -- the
draw is the cause.  The window also moves R / T by ~1e-7 at `M = 6`
(`b11_floor_window_M6.json`: 9.4e-8 .. 2.1e-7 across `n_orders = 2 .. 8`
against 10): the incident plane wave is not exactly representable by the
mapped half-space modes, and the least-squares fit sees only the window.
Both are far below the discretisation error at every `M` measured (3-4
decades under the rung-to-rung change), so nothing was changed here, and
every unit bar sits >= 2 decades above the floor.  Recommended for Phase C
(where the stack gains the shape layer): an exact modal decomposition of the
mapped incident field (its covariant L2 projection with the plain Gram, then
the half-space mode matrix) instead of the windowed least squares, or a
window sized to be overdetermined whenever a map is present.  The floor
falls spectrally with `M` (6.8e-10, 1.4e-11, 2.1e-13 at `M = 6, 8, 10`;
the window dependence 2.1e-7 at `M = 6`, 1.5e-9 at `M = 8`,
`b11_floor_window_M8.json`).  Not traced: the y-mirror residual of the
oblique circle jumps from 1.4e-12 (`M = 9`) to 1.2e-8 (`M = 10`, section 3.9),
above this normal-incidence floor; it is 2 decades under the unit bar and 3
under the rung change there.

### 4.5 F-B5 -- the node-count criterion held a tiny-shear map to round-off (build defect, fixed)

`_stag_map_nodes` normalised each of the five geometric weights by its own
moment scale, so a map with a tiny shear (one vertex moved by 1e-6 p,
`g12 ~ 1e-6`) was held to 1e-13 relative on round-off-sized `g12` moments:
144 nodes and a warning (`test_b1_transfinite_identity_is_the_identity_map`,
first run).  The three metric weights are the entries of ONE tensor and now
share its scale; Phase A's stretch counts are unchanged (36 / 40 / 24 / 28
and 144 / 160 / 192 / 112 at `a = 0.05 / 0.15 p`, `M = 5 / 6 / 8 / 10`, the
`a8_cost.json` values) (commit `66624d61`).

### 4.6 F-B6 -- the 3 x 3 circle map costs resolution at oblique incidence

A uniform film at 25 degrees under the 3 x 3 circle map reads 7.2e-4, 6.8e-5,
3.5e-6, 2.1e-7, 1.0e-8 (`M = 4 .. 8`) -- spectral, but four to six decades
behind normal incidence (4.3e-11 at `M = 6`), while the 5 x 5 map reads
7.8e-7, 7.9e-9, 7.2e-11, 6.7e-13 (`b3_film_oblique.json`).  A tilted plane
wave `exp(i k_x x)` is a polynomial-unfriendly function of `(u, v)` where the
map compresses, and the 3 x 3 disk cell squeezes its 45-degree corners
hardest.  This is Phase A's F6 (resolution follows the map) in a new guise;
the shape layer (Phase C) should prefer the 5 x 5 topology when the
incidence is oblique, or subdivide the disk cell.

---

## 5. What moved

Nothing shipped: B1, 109 / 109 hashes byte-identical against `539ce4a3`
over every dispatch branch of the family; `cmap=None` never reaches a
Phase-B function.  Inside the mapped path (reachable only with an explicit
map, an expert-level object until Phase C): maps with singular vertices now
use the corner rule (R / T of the circle move by the old rule's quadrature
error, 3.5e-7 at `M = 6` down to 7.2e-11 at `M = 12` against the planning
scratch, `b4_circle.json`); maps without (identity, stretches, the
sinusoidal stripe) are untouched except that a map with a TINY shear now
gets the node count its tensor needs (F-B5; Phase A's stretch counts
unchanged).  `CellMap.validate` admits maps whose normal derivative jumps at
the seam (F-B1); every map Phase A admitted is still admitted.
`stack2d_pure.py` is unchanged, so its history fingerprint is too.

Cost (`b1_quadrature_cost.json`, best of three, operator assembly incl. the
node-count criterion): corner rule 0.104 s against 0.057 s with the tensor
rule everywhere (c3, `M = 6`; 3200 corner points), 0.389 / 0.286 s (c3,
`M = 8`; 4608), 0.293 / 0.241 s (c5, `M = 5`; 4 x 648) -- 1.2-1.8x, against
a region eig of seconds to minutes.
The pencil, its size and the eig are unchanged; the corner rule adds
`8 n^2` (3 x 3 disk cell) or `2 n^2` (5 x 5 corner cells) points per corner
cell.

---

## 6. Not measured

* **The FEM at oblique / conical incidence** (B10): the planner's runner is a
  normal-incidence quarter cell; a Bloch-periodic full-cell run (~1.5 M
  unknowns) was not attempted.  The oblique reference is the
  exact-form-factor RCWA (algebraic, 1.25e-3 / 7.8e-4 at 29 orders), the
  second topology (1.3e-5 / 6.7e-6) and reciprocity.
* **The FEM for the ellipse, the sinusoid and the fillets** (no runner for
  those shapes); their references are RCWA with the exact form factor, the
  staircases, and (fillets) the sharp-square limit.
* **The fillet r -> 0 limit to the sharp square's OWN convergence** (1e-5):
  the small fillets converge more slowly; measured to their own rung
  change (3.6).
* **Phase A's other probes on the new maps** (absorption, mixed H partner)
  -- unchanged code, not re-run on curved maps.
* **The probe READINGS on a second build.**  The unit gates were run on a
  second build (WSL, section 7); the ladders here are one build's.
* **Unloaded wall times.**  The box was shared throughout; every time above
  is an upper bound.

---

## 7. Reproduction

```
cd /c/tmp/lum_curved_b/validation/probe_pmm2d_curved/build_b
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved_b
python b0_validate.py                                  # F-B1
python b1_quadrature.py demand 6                       # 2.1
python b1_quadrature.py ladder c3 6                    # also: c3 8, c5 5
python b1_quadrature.py rules 6                        # also: 8
python b1_quadrature.py reparam 6
python b1_quadrature.py duffy c3 6                     # also: c3 8 8,12,16,20,24,32 ; c5 5
python b1_quadrature.py summary ; python b1_quadrature.py cost
# B1 bytes: the PRE tree is `git archive 539ce4a3` in C:/tmp/curved_pre_b
(cd /c/tmp/curved_pre_b && PYTHONPATH=C:/tmp/curved_pre_b python C:/tmp/lum_curved_b/validation/probe_pmm2d_curved/build_b/b2_bytes.py C:/tmp/curved_pre_b pre)
python b2_bytes.py C:/tmp/lum_curved_b post idmap ; python b2_compare.py
python b2_identity.py
python b3_film.py normal ; python b3_film.py nocof plain ; python b3_film.py oblique
python b4_circle.py curved c3 <M>                      # M = 6 .. 12 ; c5: 4 .. 8
python b4_circle.py summary ; python b4_circle.py direction
python b5_modes.py circle3 6 12                        # circle5 4 9, rect3 6 12, stripe3 6 11, fillet0.2 5 9, fillet0.05 5 9
python b6_fillet.py fillet <r/side> <M> ; python b6_fillet.py square <M> ; python b6_fillet.py eqarea <r/side> <M> ; python b6_fillet.py summary
python b9_shapes.py curved ellipse <M> ; python b9_shapes.py rcwa sine <n> ; python b9_shapes.py stair ellipse <k> <M> ; python b9_shapes.py summary
python b10_oblique.py rung c3 <M> <theta> <phi> [nocof] ; python b10_oblique.py rcwa 25 40 <n> ; python b10_oblique.py summary
python b11_floor.py pert 6 3 ; python b11_floor.py pert 6 10 ; python b11_floor.py pert 6 3 layer c3 ; python b11_floor.py pert 6 3 homog id ; python b11_floor.py window 6
cd /c/tmp/lum_curved_b && python -m pytest tests/unit/test_pmm2d_staggered_curved_b.py --capture=sys -p no:randomly
```

Test tails, both builds:

* Windows (CPython 3.14.6, numpy 2.4.4, scipy 1.17.1), box nearly idle:
  `18 passed in 104.16s`; slowest 17.2 s (B6 efficiencies), 14.9 s (B4),
  14.8 s (B3); durations spliced into `.test_durations`.
* WSL Ubuntu (CPython 3.12.3, numpy 2.4.6, scipy 1.17.1, BLAS pinned,
  `lumenairy` from `/mnt/c/tmp/lum_curved_b`), together with Phase A's 13:
  `31 passed in 482.77s` (box loaded; slowest 76 s).
* Existing suites (44 files: every `test_*pmm2d*`, `*stack2d*`,
  `*stagger*` file incl. Phase A's, census, public-API, doc-consistency,
  except-budget, history relocation), Windows, `-n 4`:
  `1423 passed, 1 skipped` (the skip is the shipped premise gate of
  `test_fix_pmm2d_mortar_round2.py` -- numpy and scipy load different
  OpenBLAS builds on this box).
* `python -m mypy`: no issues (33 files); WSL ruff: clean.

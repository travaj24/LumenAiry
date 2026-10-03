# BUILD -- curved cells for the pure staggered 2-D PMM, Phase C (shape primitives, the stack-level map merge, the public API, the exact incident decomposition, curved viewers)

Date: 2026-10-02 / 03.  Status: BUILT + gated, not pushed.
Mount: worktree `C:/tmp/lum_curved_c`, branch `feat/pmm2d-curved-c`, built on
`91d00288` (Phase B) with the Phase A verifier branch merged
(`verify/pmm2d-curved-a`, `3f041b1c`); Windows 11 (tesla-ryzen), CPython
3.14.6, numpy 2.4.4, scipy 1.17.1, `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS =
MKL_NUM_THREADS = 1` on the command line, `lumenairy.__file__` asserted under
the tree being measured in every probe (`build_c/_common.py`; the BEFORE arms
set `LUM_TREE` to the PRE tree).  PRE tree for byte identity and for every
before/after: `git archive 91d00288` extracted to `C:/tmp/curved_pre_c/`.
The box was saturated by other agents' jobs for the whole build (up to 44
single-threaded Python processes on 24 logical cores, 96-100 % CPU), so every
WALL TIME below is an upper bound, several-fold the idle figure; no accuracy
number depends on the load.
Plan: `docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md` section 4.3, approved
with the planner's recommendations on all seven questions of section 7 (map
owned by the STACK and merged across layers; raise when two shapes' curved
regions overlap; the four `det J = 0` vertices accepted; fillets documented as
geometry fidelity; the whitened eig a separate later patch; NO maps on
`pmm_efficiency_2d_staggered`; fillet `r < 1.414e-3 p` refused in the
primitive; A-C ship as 5.50.0).  Phases A and B:
`docs/audits/BUILD_PMM2D_CURVED_A_2026_10_02.md`,
`docs/audits/BUILD_PMM2D_CURVED_B_2026_10_02.md`; Phase A verifier:
`docs/audits/VERIFY_PMM2D_CURVED_A_2026_10_02.md`.
Evidence: every number below is read from a JSON file in
`validation/probe_pmm2d_curved/build_c/` (the probe that wrote it is named in
each table).  Tests: `tests/unit/test_pmm2d_staggered_curved_c.py`.

---

## 0. Words used here

* **Shape primitive.**  A small object that takes PHYSICAL geometry --
  `Rect(cx, cy, w, h, eps)`, `FilletRect(cx, cy, w, h, r, eps)`,
  `Circle(cx, cy, r, eps, core=None)`, `Ellipse(cx, cy, a, b, eps,
  angle=0)`, `SinusoidalWall(axis, x0, amplitude, period_count=1, phase=0,
  eps=, width=None)` -- and knows its own wall layout: which `(u, v)` walls it
  needs, which grid edges carry which curve, where every corner, 45-degree
  point and tangency point sits physically, and which `(u, v)` cells it fills.
* **Hard edge.**  A piece of a shape's outline carried by one grid line (an
  arc, a straight side, a piece of a sinusoid).  The merged map reproduces
  every hard edge exactly.
* **The merge.**  `compile_shapes` (one layer) and
  `PMM2DStackPure.add_layer(shapes=...)` (every layer of a stack) take the
  UNION of all shapes' walls as the wall grid, give every grid vertex lying on
  a hard edge that edge's exact image (every other vertex stays where it is),
  every grid edge lying on a hard edge the matching piece of its curve (every
  other edge is straight between its vertex images), and blend each cell
  (Gordon-Hall).  Conflicts raise, naming both shapes and their layers.
* **Painting.**  Within one layer the shapes are painted in order onto
  `background_eps`; a later shape covers an earlier one.
* **Fail-before, rung, oracle**: as in the plan (0.1).

---

## 1. What was built

| item | file | what it does |
|---|---|---|
| shape primitives | `lumenairy/elements/pmm/shapes2d.py` (new; public through `lumenairy.elements.pmm`) | `Rect`, `FilletRect`, `Circle` (3 x 3, or the 5 x 5 layout with `core=`), `Ellipse` (axis-aligned, or rotated `|angle| < 45 deg` with the four disk-cell corners MOVED onto the ellipse), `SinusoidalWall` (a half-plane or, with `width=`, a ridge; `axis` 'x' or 'y'; integer `period_count`).  Each carries a scalar or `(3, 3)` `eps`, an analytic `signed_distance` / `contains` / `boundary_points` / `area` / `perimeter`, and its layout.  The lone `Circle`, `Circle(core=0.5)`, `FilletRect` and axis-aligned `Ellipse` reproduce Phase B's gate maps to the FINGERPRINT (same walls, curves and vertex images, the far walls written `2 c - wall` so a centred shape is Phase B's `P - wall` to the bit).  Shapes are IMMUTABLE after construction (section 4.6). |
| `compile_shapes(period_x, period_y, shapes, background_eps, *, grid_hint=None)` | same | `-> (eps_cell, x_walls, y_walls, cmap)`: the merge for one layer (`_merge`, shared with the stack); `grid_hint` = the minimum segment count per axis. |
| the merge `_merge` | same | union of walls (snapped within 1e-12 p, so two abutting shapes share a wall); the SLIVER check on the merged grid naming the walls' owners; vertex and edge CLAIMS from every hard edge (exact images at its own vertices, the curve evaluated at interior crossings, `piece()` sub-curves for cut edges); a conflict between two claims raises ("different physical positions for the same grid vertex", "different curves on one cell edge"); a physical CROSSING test of every pair of outlines with at least one curve (one outline's boundary points on both sides of the other's signed distance by more than 1e-9 p) raises before any of that; the `TransfiniteMap` is built with `_validate=False`, scanned for a FOLD (`det J <= 0` on a 12 x 12 Gauss grid per cell) naming every shape that owns a wall, a vertex or an edge of the folded cell, then validated as before; the grid is squared (`Nx == Ny`) and `grid_hint` honoured by halving the widest segment (the plan's rule) through `RefinedMap`; each layer is painted by `(u, v)` cell centre. |
| `RefinedMap(base, u_walls, v_walls)` | `lumenairy/elements/pmm/_curvemap.py` | the h-refinement of a map: every fine cell takes the base cell's own blend (a sub-rectangle of a Gordon-Hall cell IS the Gordon-Hall blend of its own edge images -- every term of the blend is linear in `s` or in `t`), singular vertices carried over; refuses a fine grid that drops a base wall. |
| `EdgeCurve.piece(s0, s1)` | same | sub-curves for `Line`, `Arc`, `EllipseArc`, `Sinusoid` (`piece(0, 1)` is the curve itself, so an uncut edge keeps its exact parameters; `Sinusoid.between(t0, t1)` cuts at exact running coordinates). |
| `TransfiniteMap(..., _validate=True)` | same | a private switch so the merge can locate and name a fold before the unchanged validation runs. |
| `PMM2DStackPure.add_layer(shapes=, background_eps=)` | `stack2d_pure.py` | records the layer and re-merges EVERY shape layer of the stack into one map (`_add_shapes_layer`, `_recompile_shapes`); a refusal rolls the stack back.  Rectangles only (an identity map) -> the SHIPPED UNMAPPED solver on the merged walls (`_shape_walls`; a tensor `eps` is accepted there); a curved map -> `self.cmap`.  Priced through `_validate_stag_cost` at `add_layer`.  Refused, naming the phase: a tensor under a curved map (Phase D), `mu` / slant / per-layer grids with shapes (Phase D / E), raw `eps_cell` layers mixed with shape layers, an explicit `cmap=` with shapes. |
| the incident field under a map (F-B4, verifier D4) | `twod_staggered.py` (`_stag_incident_load_mapped`, `_stag_incident_coeffs_mapped`), `stack2d_pure.py` | the exact L2 modal decomposition of the incident plane wave, renormalised on the specular order (section 3.9); unmapped solves keep the shipped least-squares overlap bit for bit. |
| `pmm_jones_2d_staggered(..., shapes=, background_eps=)` | `twod_staggered.py` | the one-layer stack, byte for byte; exclusive with `eps_cell` / `cmap` / `mu_cell` / `slant`. |
| `pmm_efficiency_2d_staggered(shapes=, background_eps=)` | same | raises, pointing at the Jones entry (plan Q5). |
| viewers | `stack2d_pure.py` (`_mapped_cell_outline`, `_plot_mapped_layer`, `_mapped_section`) | a mapped stack draws every cell's PHYSICAL image with no wall lines and the material boundaries as curves; `plot_section` reads the material along the physical cut (exact outlines for shape layers), every boundary bisected to round-off.  Phase A's refusal lifted. |
| Phase A verifier fold-ins | `twod_staggered.py`, docs | D3 (det J checked before the node search doubles), D5 (effective cap stated), D6 (the `cmap=` docstrings point at `from_physical_walls` and `shapes=`), D7 (BUILD_A 4.5e-15 -> 5.2e-15); D1 / D2 were already closed by Phase B (F-B1, the corner rule) and the verifier's six decision tests pass unchanged. |
| docs | module docstrings of `shapes2d.py`, `twod_staggered.py` (scope note rewritten), `stack2d_pure.py`; `CHANGELOG.md`; `docs/PMM_ROADMAP.md` (the 2-D curved row and its phase); `docs/cookbook.md` (a curved-cells entry) | -- |
| tests | `tests/unit/test_pmm2d_staggered_curved_c.py` | gates C1-C17 (section 3). |

Not touched: `Basis1D`, the assembly, the corner rule, the far projector, the
eigensolver (QZ), the mortar.

---

## 2. Two design decisions, measured

### 2.1 Per-edge claims, not "the macro-cell's blend on its sub-rectangles"

The plan (4.3) proposed that a wall crossing another shape's curved macro-cell
subdivide it, each sub-cell taking the macro-cell's blend.  That keeps the
first shape's map but BENDS the crossing wall inside the macro-cell: in a
3 x 3 circle (r = 0.36, p = 1.2) the image of a straight wall `u = u0` through
the circle's side cell bulges by `s0 (r - r / sqrt 2)` -- up to 0.105 at the
cell's arc -- so the OTHER layer's straight material boundary along that wall
would sit in the wrong place.  The merge built here instead gives every grid
vertex and edge the image its hard edges demand and leaves every other vertex
where it is: a straight wall of one shape stays straight, the other shape's
outline stays exact, and the curved shape's transition cells are confined
automatically by whatever walls are nearest.  A refinement that adds NO
material boundary (squaring the grid, `grid_hint`) does use the
sub-rectangle rule (`RefinedMap`), where bending the added line is harmless.
For a lone primitive the two constructions coincide, which is why the lone
primitives reproduce Phase B's maps to the fingerprint.

Consequences, all gated: two shapes in different layers may NEST (a small
square over a large disk) and two disjoint curves may share a transition cell
(a cell whose left edge is one circle's arc and whose right edge is its
neighbour's -- gate C13); what cannot be represented raises (crossing
outlines; two curves on one edge; a fold).

### 2.2 No automatic steep-cell subdivision (verifier recommendation 7, measured)

The Phase A verifier asked for a measured criterion to subdivide cells where
the map is steep.  Measured:

* Every lone primitive map is gentle: the spread (max / min over a 16 x 16
  Gauss grid) of the largest singular value of `J` is <= 1.37 in every
  non-singular cell of the 3 x 3 circle (`c4_walls_steep.json`), 1.72 for a
  sinusoidal ridge of amplitude 0.2 in p = 1.2 (`c4b_steep_A0.2_M4.json`);
  the cells that own a singular vertex are handled by the corner rule.
* WHERE the walls of a transfinite circle go does not move the rate at all:
  the same exact disk on the uniform-lattice walls (0.4, 0.8) with its corners
  moved onto the circle reads, for a film, 2.02e-6 / 1.28e-8 / 5.2e-12 /
  3.3e-13 against the primitive's 2.02e-6 / 1.11e-8 / 8.4e-12 / 3.2e-13 at
  M = 4 .. 7 (`c4_walls_film_M*.json`).
* Subdividing the STEEP cells of the A = 0.2 ridge instead of the widest
  (equal grid, 5 x 5) pays 2.2x at M = 4, 34x at M = 5, 4.3x at M = 6 on the
  film (`c4b_steep_A0.2_M*.json`: 1.1e-7 / 3.1e-10 / 2.5e-12 against 5.0e-8 /
  9.2e-12 / 5.7e-13); at A = 0.05 the two orderings are within 8x of each
  other at every M.

Decision: the default layouts stay as measured (no automatic splitting --
it would also change the maps the Phase B gates pin), and `grid_hint=` is the
user's knob.  Recommended follow-up: let `grid_hint` / the squaring split by
steepness x width rather than width alone (a measured 4-34x on strong
sinusoids); not done here because it changes merged layouts the gates read.

---

## 3. The gates

Every number is from this tree, 2026-10-02 / 03.  "Unit test" names the
decision in `tests/unit/test_pmm2d_staggered_curved_c.py`.

### 3.1 C1 -- no shape, no map: the shipped bytes; the shapes route IS an explicit map

| claim | measured | bar | fail-before | JSON |
|---|---|---|---|---|
| no map = the bytes of `91d00288`, Phase B's fixture set | **109 / 109** SHA-256 identical (operators of every dispatch branch, region modes, geometric cache, far projectors, `pmm_efficiency_2d_staggered` / `pmm_jones_2d_staggered` / stack / mortar outputs and absorption) | equality | the identity map through the quadrature path: 28 of 36 operator hashes differ (the 8 equal ones are `None` attributes) | `c1_bytes_pre.json`, `c1_bytes_post.json`, `c1_compare.json` |
| the same on the Phase A VERIFIER's fixture set (`verify_a/v1_bytes.py`, run unchanged) | **181 / 181** identical | equality | -- | `c1_v1bytes_{pre,post}.json`, `c1_v1compare.json` |
| no Phase C function is reached without shapes / a map | `_merge`, both incident helpers and `_recompile_shapes` booby-trapped; an oblique two-layer lossy stack with absorption, the Jones and the efficiency entries solve | no trap fires | -- | unit test `test_c1_no_shapes_never_reaches_the_phase_c_code` |
| the shapes route = `compile_shapes` + `cmap=` + `eps_cell=` | a slab with a circular hole: equal to the bit (`o, R, T, J`) | equality | a 1e-12-relative radius change changes the fingerprint | unit test `test_c1_shapes_route_is_the_explicit_map_route_byte_for_byte` |
| rectangles ride the UNMAPPED solver | identity map through the mapped path vs the unmapped route: 7.4e-15 / 1.8e-14 / 5.6e-14 / 5.1e-14 at normal incidence, M = 4 .. 7 | `<= 1e-11` (the A2 bar) | -- | `c1_identity_oblique_M*.json`, unit test `test_c1_rectangles_ride_the_unmapped_solver` |
| ... at OBLIQUE incidence (0.3, 0.2 rad) | 6.0e-5 / 8.0e-7 / 1.9e-8 / 1.8e-10 at M = 4 .. 7 -- the two INCIDENT treatments (unmapped least squares vs the mapped exact decomposition, 3.9) differ at the discretisation level and converge together spectrally | build doc | -- | `c1_identity_oblique_M*.json` |

### 3.2 C2 -- the circle primitive IS Phase B's map and lands on the FEM

`Circle(0.6, 0.6, 0.36)` -> Phase B's `_circle_map_3x3` fingerprint;
`Circle(..., core=0.5)` -> `_circle_map_5x5`; `eps_cell` equal.  The shapes
route against the explicit-map route: bytes equal at every rung checked (c3
M = 6, 7; c5 M = 4) (`c_circle_*.json`, field `explicit.bytes_equal`).
Distance to the saved 3-D FEM oracle (largest per-order, E along y; the
oracle's own spread 8.3e-6) and the difference of the 36-entry R / T vector to
Phase B's saved rung (`c_ladders_summary.json`):

| map, M | to FEM (this tree) | to FEM (Phase B) | R / T vs Phase B's rung | closure | wall (s, loaded) |
|---|---|---|---|---|---|
| c3, 6 | 6.214e-03 | 6.214e-03 | 5.2e-07 | 1.0e-03 | 37 |
| c3, 7 | 8.185e-04 | 8.185e-04 | 1.4e-08 | 3.0e-05 | 141 |
| c3, 8 | 4.584e-04 | 4.584e-04 | 3.3e-10 | 2.6e-05 | 150 |
| c3, 9 | 2.829e-05 | 2.829e-05 | 5.1e-12 | 6.7e-07 | 313 |
| c3, 10 | 1.685e-05 | 1.685e-05 | 1.5e-13 | 2.7e-07 | 684 |
| **c3, 11** | **7.768e-06** | 7.768e-06 | 2.7e-13 | 5.7e-09 | 1196 |
| c5, 4 | 4.639e-04 | 4.635e-04 | 1.1e-06 | 6.5e-05 | 32 |
| c5, 5 | 8.394e-05 | 8.395e-05 | 7.2e-09 | 1.7e-06 | 107 |
| c5, 6 | 4.318e-05 | 4.318e-05 | 1.3e-10 | 1.8e-08 | 708 |

The only difference to Phase B's numbers is this phase's incident fix (3.9):
5.2e-7 at M = 6 falling spectrally to round-off by M = 10 -- three decades and
more under the distance to the oracle at every rung.  Unit test
`test_c2_circle_primitive_is_the_phase_b_map_and_lands_on_the_fem`:
fingerprints, bytes at M = 5, the FEM distance at M = 7 <= 2.5e-3 (Phase B's
bar; measured 8.185e-4).

### 3.3 C3 -- the fillet primitive IS Phase B's map, and its ladder

`FilletRect(0.6, 0.6, 0.6, 0.6, r)` -> Phase B's `_fillet_map_5x5` fingerprint
for r / side = 0.05, 0.1, 0.2; `eps_cell` equal; bytes equal to the explicit
map at M = 4 (all three).  Input 'te', R00 / T00 (`c_ladders_summary.json`):

| r / side | M | R00 | T00 | vs Phase B's rung (R00 / T00) | closure |
|---|---|---|---|---|---|
| 0.05 | 4 / 5 / 6 | 0.036903 / 0.038342 / 0.038382 | 0.198345 / 0.215205 / 0.216292 | 2.9e-10 / 1.0e-11 / 9.1e-14 ; 8.4e-11 / 6.5e-11 / 6.6e-13 | 1.6e-3 / 5.3e-5 / 1.0e-5 |
| 0.1 | 4 / 5 / 6 | 0.037738 / 0.037888 / 0.037884 | 0.209324 / 0.217144 / 0.217622 | 4.0e-10 / 4.9e-11 / 1.3e-13 ; 4.7e-9 / 2.0e-10 / 7.3e-12 | 1.0e-3 / 2.0e-5 / 3.1e-6 |
| 0.2 | 4 / 5 / 6 | 0.037120 / 0.036935 / 0.036900 | 0.221866 / 0.223171 / 0.223269 | 4.3e-9 / 1.7e-10 / 2.9e-12 ; 5.9e-9 / 1.8e-10 / 5.9e-11 | 1.5e-4 / 3.2e-6 / 1.3e-7 |

The fillet shift at r / side = 0.2, M = 6, against the sharp square at
M = 12 (Phase B's `b6_square_M12.json`, R00 0.038608, T00 0.215953):
dR00 = -1.71e-3, dT00 = +7.32e-3 (Phase B at M = 8: -1.73e-3 / +7.36e-3).
Unit test `test_c3_fillet_primitive_is_the_phase_b_map_and_moves_the_device`
(`c_unit_c3fid.json`): r / side 0.2 at M = 4 against the sharp `Rect` at M = 5:
dR00 = -1.57e-3 <= -5e-4, dT00 = +8.8e-3 >= +2e-3 (Phase B's bars); the
same-area square at M = 5 moves R00 +2.8e-4 (the wrong way, B7); `r = 0` is
the `Rect` (no map).

### 3.4 C4 -- the F5 / D6 traps are closed by construction

* RATE: the uniform film under the primitive's circle map: 2.02e-6, 1.11e-8,
  8.4e-12, 3.2e-13 at M = 4 .. 7 (`c4_walls_film_M*.json`).  Unit bar: M = 5
  `<= 1e-7`, the M = 4 -> 5 drop `>= 1.5` decades (measured 2.3).
* SAME DEVICE on hand-built walls: the same exact disk laid by hand on the
  shipped uniform-lattice walls (0.4, 0.8) with the disk-cell corners moved
  onto the circle converges to the primitive's answer: pillar R / T differ by
  2.3e-5 / 4.1e-6 / 1.7e-6 at M = 5 / 6 / 7, against a rung change of
  5.4e-3 / 5.8e-3 (`c4_walls_pillar_M*.json`).  Unit bar 5e-4 at M = 5.
* FAIL-BEFORE: the boundary SNAPPED to the lattice (the circle through the
  lattice vertices, r = 0.283) is a different device, 0.13 from the
  primitive at M = 5 (`c_misc_c4.json`); asserted `>= 5e-3`.
* The verifier's D6 trap (physical walls given as `u_walls` of a
  `SeparableStretch` with a 0.08 sine) builds walls at 0.269 / 0.629 instead
  of 0.20 / 0.65 -- 6.9e-2 off (`c_misc_d6.json`); every primitive's outline
  vertices map to exactly the physical points asked for (round-off, bar
  1e-12; unit test `test_c4_a_primitive_never_takes_u_walls_for_physical_ones`).
* Section 2.2: wall placement does not move the rate of a transfinite circle;
  steepness inside a cell does, modestly, on strong sinusoids.

### 3.5 C5 -- refusals name the two shapes and their layers

Unit test `test_c5_refusals_name_both_shapes_and_layers`: crossing outlines
in two layers ("layer 1: Circle(...) and layer 2: Rect(...) CROSS in plan
view", the stack rolled back to its previous map); a sharp rectangle over a
rounded corner in another layer (a fold naming both); a straight edge laid on
a fillet's flat side (zero-height cell: "FOLDS ... bounded by FilletRect(...),
Rect(...)"); two edges 6e-4 apart ("SLIVER ... belong to Rect(...) and
Rect(...)"; a shared wall is accepted); a fillet radius 1.5e-3 below
1.414e-3 p (the sliver contract named, "use radius=0 (a sharp corner) or a
radius >= ..."; 1.8e-3 accepted); a tensor shape under a curved map (Phase
D); shapes with per-layer grids (Phase E); a raw `eps_cell` layer joining a
shape stack; `shapes=` with an explicit `cmap=` stack or with `eps_cell=` on
the Jones entry; `shapes=` without `background_eps`; a circle outside the
unit cell.

### 3.6 C6 -- two layers, two shapes, one merged map

A circle (r = 0.2, layer 1) nested inside a filleted square's footprint
(0.9 x 0.9, r = 0.09, layer 2): ONE merged 7 x 7 map (`c_unit_c6.json`,
`c6b_absorb_M3.json`):

| quantity | M = 3 | M = 4 | fail-before | unit bar (M = 3) |
|---|---|---|---|---|
| lossless closure (layer 2 eps 2.25) | 3.2e-3 | 2.8e-4 | -- | 3e-2 |
| absorption in the LOSSLESS circle layer (layer 2 eps 2.25 + 0.4i) | 7.7e-15 | 5.7e-15 | `-R` as the flux Gram: 1.7e-2 | 1e-10 |
| sum of `layer_absorption` vs 1 - sum R - sum T | 2.2e-3 | 1.1e-4 | `-R`: 1.6e-2 / 1.1e-2 | recorded only (discretisation level) |
| vacuum-painted layer 2 vs a uniform vacuum layer 2 on the SAME map | 2.2e-15 | 4.8e-15 | painting eps 1.1: 8.2e-3 | 1e-9 |

### 3.7 C7 / C8 -- the convenience entry; the efficiency entry refuses

`pmm_jones_2d_staggered(..., shapes=)` equals the one-layer stack to the bit
(an off-centre circle with a stripe below it, oblique incidence); 
`pmm_efficiency_2d_staggered(shapes=)` raises `NotImplementedError` naming
`pmm_jones_2d_staggered(... shapes=...)`.

### 3.8 C9 -- the incident field under a map (Phase B F-B4, verifier D4)

The shipped overlap `cinc = lstsq(Hsup, delta_00)` is under-determined under a
map (2 (2 n_orders + 1)^2 equations, 2 q^2 unknowns).  BEFORE = the PRE tree,
AFTER = this tree, the 3 x 3 circle centred (`c3`) and off-centre at
(0.55, 0.66) (`c3off`) (`c9_incident_{pre,post}_*_M*.json`):

| quantity | M | before (c3 / c3off) | after (c3 / c3off) |
|---|---|---|---|
| R / T moved by a 1e-15 random perturbation of the half-space weights | 5 | 1.1e-7 / 7.9e-9 | 5.1e-15 / 2.9e-15 |
| | 6 | **6.8e-10** / 4.9e-10 | **8.0e-15** / 9.1e-15 |
| | 7 | 6.6e-11 / 9.2e-11 | 2.3e-14 / 2.1e-14 |
| R / T at n_orders 2, 3, 5 vs 8 (max) | 5 | 1.3e-5 / 1.4e-5 | 8.3e-16 / 9.4e-16 |
| | 6 | 2.9e-7 / 4.4e-7 | 6.2e-16 / 5.6e-16 |
| | 7 | 1.2e-7 / 1.3e-7 | 3.3e-16 / 3.9e-16 |
| a 0.3 VACUUM spacer on top (a physical no-op) | 5 / 6 / 7 | 2.5e-5 / 2.0e-6 / 1.2e-6 | 2.6e-5 / 1.9e-6 / 1.2e-6 |

**How the decomposition was chosen** (`c9b_*.json`).  The bare L2 projection
`cinc = W0^-1 G^-1 b` (`b` the L2 load of the covariant incident field, `G`
the plain Gram, `W0` the shared geometric eigenvectors) was the first version;
Phase B's film gate (B2) caught it: its own far field is `delta_00` only to
the projection error, while the efficiencies are normalised to a unit incident
amplitude.  The shipped version renormalises it by the 2 x 2 matrix that makes
its ORDER-0 far field exactly the input (only order 0 enters):

| arm | film under the circle map vs Airy, M = 5 / 6 / 7 | pillar: n_orders dependence, M = 5 / 6 / 7 |
|---|---|---|
| least squares (shipped, the PRE behaviour) | 2.8e-8 / 4.3e-11 / 3.3e-13 | 1.3e-5 / 2.9e-7 / 1.2e-7 |
| bare L2 | 2.6e-6 / 8.8e-8 / 4.8e-9 | 1.1e-15 / 6.2e-16 / 4.4e-16 |
| **L2 renormalised on order 0** | **1.1e-8 / 8.4e-12 / 3.2e-13** | **8.3e-16 / 6.2e-16 / 3.3e-16** |

At OBLIQUE (25 deg, 0) and CONICAL (25 deg, 40 deg) incidence the
renormalised decomposition and the least squares agree in accuracy at every
rung on both the circle map and the identity map (`c9b_filmobl_M*.json`, film
vs Airy, circle 25 / 0: 7.2e-4 vs 7.7e-4, 6.8e-5 vs 7.1e-5, 3.5e-6 vs 3.6e-6,
2.1e-7 vs 2.1e-7 at M = 4 .. 7; 25 / 40: 3.3e-4 vs 2.7e-4, ..., 6.3e-8 vs
6.5e-8).

**What remains (verifier D4).**  A vacuum spacer still moves R / T at the
discretisation level, unchanged by the fix (2.0e-6 at M = 6, falling with M):
under a curved map the half-space modes are not plane waves, so a reflected
discrete mode spreads over several orders whose phases advance differently
through the spacer.  It is documented next to `cmap=` (both entries) and
`shapes=`.  Unit tests `test_c9_incident_decomposition_is_window_free_and_off_the_floor`
(window and floor <= 1e-12 at M = 5; fail-before the shipped overlap through
the real code, window 1.3e-5 >= 1e-9) and
`test_c9_incident_renormalisation_keeps_the_film_exact` (film M = 6 <= 1e-9;
the bare L2 >= 1e-8).

### 3.9 C10 -- oblique and conical incidence through the shapes route

At the five angles of Phase B's B10 (25 / 0, its reverse 24.2498 / 0, 25 / 40,
its two reverses) the shapes route equals the explicit map to the bit at
M = 6 (`c_oblique_*.json`).  Against Phase B's saved rungs, R / T move by
8.5e-5 / 5.3e-6 / 7.1e-7 (25 / 0) and 5.6e-5 / 2.4e-6 / 3.2e-7 (25 / 40) at
M = 6 / 7 / 8 -- the incident fix, falling about a decade per rung, 1.5 decades
under the rung change.  Reciprocity (singular values of the power-normalised
Jones block of a channel against its reversal, `c_ladders_summary.json`):

| channel | M = 6 | 7 | 8 | Phase B (6 / 7 / 8) | wrong pairing |
|---|---|---|---|---|---|
| 25 / 0, order (-1, 0) | 2.6e-6 | 8.0e-7 | 2.8e-8 | 3.1e-6 / 8.3e-7 / 4.1e-8 | 0.094 .. 0.113 |
| 25 / 40, order (-1, 0) | 4.5e-5 | 9.3e-6 | 5.7e-7 | 3.9e-5 / 9.4e-6 / 6.7e-7 | 0.099 .. 0.111 |
| 25 / 40, order (0, -1) | 3.6e-5 | 1.4e-5 | 3.4e-7 | 3.7e-5 / 1.7e-5 / 1.9e-7 | 0.094 .. 0.116 |

Unit test `test_c10_oblique_and_conical_shapes_route_is_the_explicit_map`.

### 3.10 C11 -- the viewers draw curves, not the wall grid

`plot_geometry` on a circle shape stack: the 256 drawn outline vertices sit on
the circle to 3.3e-16 (bar 1e-12); the `(u, v)` cell outline -- what drawing
the wall grid would show -- is 0.105 (0.29 r) off (fail-before, bar 0.05 r);
`plot_section` at y = 0.6 puts the boundaries at 0.24 and 0.96 to 5.6e-17
(bar 1e-10) (`c_misc_c11.json`).  A uniform layer draws no outline.

### 3.11 C12 -- every primitive's geometry is exact

Mapped area of the painted cells and length of the drawn outline against the
analytic values, min `det J` over interior Gauss nodes (`c_unit_c12.json`):
area <= 3.8e-15 relative, perimeter <= 4.4e-16 relative for `Rect`, `Circle`
(3 x 3 off-centre, 5 x 5), `FilletRect` (w != h), `Ellipse` (axis-aligned and
rotated 20 deg), `SinusoidalWall` (a ridge; a two-wave half-plane along y);
min `det J` 2.8e-3 (Gauss nodes approach the singular vertices).  Bars 1e-12
and > 0.

### 3.12 C13 -- a 2 x 2 circle array is the halved period (plan C3)

Four circles in the doubled period 2.4: a 5 x 5 merged map in which each
cell between two circles blends TWO arcs.  Every odd order of the doubled
period must vanish (`c_unit_c13.json`, `c_unit_c13_M4M5.json`):

| M | odd orders, four circles | three circles (fail-before) | even orders vs the single circle at M + 1 |
|---|---|---|---|
| 3 | 5.3e-3 | 4.2e-2 | 5.3e-3 |
| 4 | 6.5e-6 | 2.5e-2 | 5.6e-2 |
| 5 | 3.4e-7 | 2.2e-2 | 5.6e-3 |

The even orders agree with the single circle at the discretisation level of
the coarser map (the 2.4 cell at M carries about half the resolution per
circle of the 1.2 cell at M + 1).  Unit bar: odd orders at M = 4 <= 1e-3.

### 3.13 C14 -- four-fold symmetry (plan C4)

`te(m, n) - tm(n, m)` of the circle: 2.8e-14 (M = 5), 8.2e-14 (M = 6) --
Phase B read 8.9e-10 at M = 6 with the least-squares incident overlap; an
ellipse r x 1.02 r reads 5.8e-3 / 6.0e-3 (`c_unit_c14.json`).  Unit bar 1e-10.

### 3.14 C15 / C16 -- no stale map; tensors routed or refused

C15: a shape is immutable after construction (the stack compiles it at
`add_layer`; the eig dedupe is keyed by the map fingerprint), and a 1e-12
relative change of any parameter gives a new fingerprint.  C16: a TENSOR
rectangle rides the unmapped solver and equals the shipped integer-grid tensor
solve to 3.6e-15 (`c_unit_c16.json`); the same tensor on a circle raises
naming Phase D -- the effective tensor needs `J` itself: the best scalar-route
surrogate `s sqrt(g) g^-1` of `eps_t = [[4, 0.3], [0.3, 3]]` is 0.20 off the
true `sqrt(g) J^-1 eps_t J^-T` at the circle map's nodes (0.125 even where
`J = I`), and the map changes the tensor by up to 25x near the singular
vertices (`c_misc_c16.json`) -- Phase D work is genuinely required.

### 3.15 C17 -- the rotated ellipse obeys its mirror

The rotated `Ellipse` is the one primitive whose disk-cell corners are MOVED
onto the outline.  The ellipse at +20 deg is the y-mirror of the one at
-20 deg, so `R(m, n; +a) = R(m, -n; -a)`: 1.5e-14 (M = 4), 8.6e-13 (M = 5);
without the mirror the same comparison reads 2.6e-2 / 1.8e-2 (the device is
genuinely asymmetric); closure 1.4e-3 / 3.4e-3 (`c_misc_ell.json`).  Unit bar
1e-10.

---

## 4. Findings

### 4.1 F-C1 -- the bare L2 incident decomposition breaks the efficiency normalisation (build defect, caught by Phase B's B2, fixed)

Section 3.8.  The first version of the F-B4 fix moved the film under the
circle map from 4.3e-11 to 8.8e-8 at M = 6; renormalising on order 0 brought
it to 8.4e-12.  Phase B's `test_b2_film_under_the_circle_map_is_exact_and_spectral`
is the gate that saw it.

### 4.2 F-C2 -- the 3 x 3 sinusoid film converges non-monotonically (measured, no defect)

The film under `compile_shapes`' ridge map (v walls 0, 0.3, 0.6, 1.2 after
squaring) reads 2.7e-6, 1.7e-9, 1.9e-9, 2.1e-13, 2.9e-13 at M = 4 .. 8, while
Phase B's thirds layout reads 3.6e-8, 2.1e-9, 1.1e-11, 7.1e-14, 1.6e-13
(`c4c_plateau_M*.json`).  The `RefinedMap` and a directly built
`TransfiniteMap` on the same walls agree at every rung (2.7e-6 / 2.7e-6, ...,
2.1e-13 / 1.7e-13); forcing 4x the assembly nodes changes nothing; the least
squares reads the same family.  It is the discretisation of the wide
0.6-period row, not a quadrature or refinement defect.

### 4.3 F-C3 -- the oblique incident treatments differ at the discretisation level (measured, documented)

Section 3.1: the identity map through the mapped path and the unmapped
solver differ at oblique incidence by 6.0e-5 (M = 4) to 1.8e-10 (M = 7)
because the incident plane wave `e^{-i alpha0 x}` is not a polynomial EVEN
without a map; the unmapped path keeps its least-squares overlap (bytes), the
mapped path the decomposition.  Both converge to the same answer.

### 4.4 F-C4 -- a fillet's flat side lives on its 45-degree wall (design limit, documented)

The Phase B fillet layout (kept, so the gates' maps are reproduced) places
each flat side on the grid line through the fillet's 45-degree points and maps
it to the physical side.  Another shape's straight edge on that side closes a
zero-height cell and raises (C5); so does a sharp rectangle over a rounded
corner (the same cell holds two different outlines -- "different curves in one
cell", which the maintainer's rule refuses).  An alternative layout (walls on
the sides and the tangency points, the sharp-corner vertex moved onto the
45-degree point) would make the first case representable; not built.

### 4.5 F-C5 -- Phase A verifier D1-D7

D1 and D2 were closed by Phase B (tangential-only periodicity; the corner
rule); the verifier's six decision tests pass on this tree.  D3: a folding map
now raises the `det J` refusal on the base rule (the node search checks
`det J` at every node before doubling).  D5, D6, D7: documentation, done.

### 4.6 F-C6 -- a mutable shape would make the compiled map stale (verifier recommendation 7, fixed)

The stack compiles its shapes at `add_layer` and keys its eig dedupe by the
map fingerprint; a shape mutated afterwards would silently desynchronise
both.  Shapes now refuse attribute assignment (C15).

### 4.7 F-C7 -- Phase B's B2 fail-before arm re-derived (intentional algorithm change)

B2's no-cofactor arm read `> 1e-2` at M = 4 because the defective projector
entered twice (the incident overlap AND the outgoing projection).  With the
exact incident decomposition it enters once: a FLAT 4.77e-3 at M = 4 / 5 / 6
against 2.0e-6 / 1.1e-8 / 8.4e-12 correct (`c_b2_nocof_rederive.json`).  The
bar was re-derived once, dated, to `> 1e-3` (0.68 decades under the defect,
2.7 above the correct reading); the WSL build caught it.

### 4.8 F-C8 -- the rollback's broad except (fixed)

`_add_shapes_layer` first restored the stack with `except Exception: ...
raise`, one over the non-ui broad-except budget (51 > 50); it is now a
try/finally on a success flag.

---

## 5. What moved

Nothing shipped: C1, 109 / 109 and 181 / 181 SHA-256 identical against
`91d00288`; no default answer moved, so there is no `Migration-Guide.md`
entry.  Inside the mapped path (unreleased; Phases A-C ship together as
5.50.0): the incident decomposition (3.8) moves mapped R / T by the old
overlap's window effect (5.2e-7 at M = 6 on the circle, falling
spectrally), and the mapped viewers draw instead of raising.

Cost (loaded box, upper bounds): the merge itself is milliseconds
(`compile_shapes` + validation of a 7 x 7 map < 0.5 s); the incident load is
one quadrature pass per solve; a shape costs the wall grid it needs (a circle
3 x 3, a fillet 5 x 5, a circle and a fillet in two layers 7 x 7 -- the pencil
grows as the square of the segment count).

Unit tests (`tests/unit/test_pmm2d_staggered_curved_c.py`, 21 tests; the first full run had 20 -- C17 was added after it), on the
saturated box with `-n 4`: `20 passed in 218 s`; slowest 73 s (C6, 7 x 7 at
M = 3), 61 s (C2), 46 s (C3), 39 s (C13), 38 s (C9 film) -- each about 3-5x
its idle time on this box (Phase B's comparable tests read 15-17 s idle).

---

## 6. Not measured

* **The probe READINGS on a second build** (WSL): the unit gates were run on
  the second build (section 7); the ladders here are one build's.
* **Idle wall times.**  The box was saturated throughout.
* **A FEM oracle for the merged two-shape stack, the ellipse rotated, the
  sinusoid at large amplitude**: their references are geometric (area,
  perimeter, outline), physical identities (vacuum, 2 x 2 array, reciprocity,
  symmetry) and the Phase B oracles through the shared maps.
* **The steepness-aware split** (2.2): measured as an opportunity, not built.

---

## 7. Reproduction

```
cd /c/tmp/lum_curved_c/validation/probe_pmm2d_curved/build_c
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved_c
# C1 (PRE tree = git archive 91d00288 in C:/tmp/curved_pre_c)
(cd /c/tmp/curved_pre_c && PYTHONPATH=C:/tmp/curved_pre_c python C:/tmp/lum_curved_c/validation/probe_pmm2d_curved/build_c/c1_bytes.py C:/tmp/curved_pre_c pre)
python c1_bytes.py C:/tmp/lum_curved_c post idmap ; python c1_compare.py
(cd ../verify_a && VA_ROOT=C:/tmp/curved_pre_c PYTHONPATH="C:/tmp/curved_pre_c;." python v1_bytes.py phaseCpre)   # and phaseCpost on this tree; move to c1_v1bytes_*.json
python c1_v1compare.py ; python c1_identity_oblique.py <M>            # M = 4 .. 7
# C2 / C3 / C10 ladders (jobs2.txt lists every rung run here)
./runjobs.sh jobs2.txt 6 ; python c_ladders.py summary
# C4
python c4_walls.py film <M> ; python c4_walls.py pillar <M> ; python c4_walls.py steep
python c4b_steep.py <A> <M> ; python c4c_plateau.py <M>
# C9 (before: LUM_TREE=C:/tmp/curved_pre_c PYTHONPATH=C:/tmp/curved_pre_c)
python c9_incident.py <M> c3|c3off ; python c9b_normalisation.py <M> film|pillar|filmobl
# unit-test readings
python c_unit_readings.py c3fid|c6|c13|c14|c16|c12 ; python c6b_absorb.py 3
python c_misc_readings.py c4|d6|c11|c16
cd /c/tmp/lum_curved_c && python -m pytest tests/unit/test_pmm2d_staggered_curved_c.py --capture=sys -p no:randomly
```

Test tails, both builds:

* Windows (CPython 3.14.6, numpy 2.4.4, scipy 1.17.1), saturated box,
  `-n 4`: `tests/unit/test_pmm2d_staggered_curved_c.py` -> `20 passed in
  218.33s` (C17, added after: `2 passed` with C7, 8.7 s); slowest 73 s (C6).
* WSL Ubuntu (CPython 3.12.3, numpy 2.4.6, scipy 1.17.1, BLAS pinned,
  `lumenairy` from `/mnt/c/tmp/lum_curved_c`), Phases A + B + C + the Phase A
  verifier's tests: `1 failed, 56 passed in 505.33s` -- the failure was
  Phase B's B2 fail-before arm, re-derived (F-C7) and then `2 passed`
  (B2 and C17) on WSL.
* Existing suites, Windows, `-n 6` (every `test_*pmm2d*`, `*stack2d*`,
  `*stagger*`, `*curved*` file incl. Phases A / B / C and the verifier's,
  census, public-API, walker, doc identifiers, doc consistency,
  except-budget, history relocation and lint, kernel consistency,
  re-exports): `5 failed, 1502 passed, 1 skipped in 1505.94s`; the five:
  B2 (F-C7, fixed), three ids of the except budget (F-C8, fixed) and the
  `__all__` walker on `material_key` -- PRE-EXISTING on `91d00288` (it fails
  there identically: `stack2d_pure.__all__` exports the viewer helper
  `material_key`, added by `4ec402bc`, which is neither re-exported nor
  exempt; not touched here, flagged).  Re-run after the fixes: `1 failed, 776
  passed` (the walker, `material_key` only).
* `python -m mypy` (the configured strict file list): `Success: no issues
  found in 33 source files`; `lumenairy/elements/pmm/` is NOT on that list --
  `python -m mypy lumenairy/elements/pmm/_curvemap.py
  lumenairy/elements/pmm/shapes2d.py` reports 368 errors, every one
  `no-untyped-def` / `no-untyped-call` (12 genuine inconsistencies found
  that way were fixed); adding the package to the strict list is a
  maintainer decision.
* WSL ruff 0.15.16 on `lumenairy/ tests/ scripts/` and `build_c/`: `All
  checks passed!`

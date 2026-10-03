# BUILD -- curved cells for the pure staggered 2-D PMM, Phase E2 (a different curved map in every layer: the non-separable curved mortar)

Date: 2026-10-03.  Status: BUILT + gated + VERIFIED (verifier fold-in,
section 10), not pushed.
Mount: worktree `C:/tmp/lum_curved_e2`, branch `feat/pmm2d-curved-e2-perlayer`,
built on `eae470d9` (Phase D), with the Phase C verifier branch merged
(`verify/pmm2d-curved-c`, `eff64792`) and its defects folded in (section 6).
Windows 11 (tesla-ryzen), CPython 3.14.6, numpy 2.4.4, scipy 1.17.1,
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` on the command
line, `lumenairy.__file__` asserted under the tree being measured in every
probe (`build_e2/_common.py`; the BEFORE arms set `LUM_TREE` to the PRE
tree).  PRE tree for byte identity: `git archive eae470d9` extracted to
`C:/tmp/curved_pre_e2/`.  The box was shared with four sibling agents and
other jobs throughout (37 Python processes, 100 % CPU at the worst), so every
WALL TIME below is an upper bound; no accuracy number depends on the load.
Plan: `docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md` section 4.5
(Phase E, approved 2026-10-02; the maintainer split it into E1 = out-of-plane
+ slant under a map, E2 = this, E3 = the JAX twin).  Phases A-D:
`BUILD_PMM2D_CURVED_{A,B,C,D}_*.md`.  Evidence: every number is read from a
JSON file in `validation/probe_pmm2d_curved/build_e2/` (the probe that wrote
it is named in each table).  Tests: `tests/unit/test_pmm2d_staggered_curved_e2.py`.

---

## 0. Words used here

* **Layer map.**  The coordinate map `(x, y) = Phi(u, v)` a layer is solved
  on.  Phases A-D owned ONE map per stack; E2 gives every layer of a
  `layer_grids='per-layer'` stack its own (`add_layer(..., cmap=)`, or the
  layer's own `shapes=`).
* **Mortar.**  The shipped per-layer coupling of two layers on different
  element grids: tangential E is projected onto the lower layer's trace
  space and tangential H onto the upper layer's
  (`BUILD_PMM2D_STAGGERED_MORTAR_2026_09_11.md`).  Its only geometric
  ingredient is the **cross-mass** -- the overlap integral of one layer's
  basis functions against the other's.
* **Separable / non-separable.**  On straight wall grids the cross-mass is
  a Kronecker product of 1-D integrals (both bases live on rectangles of one
  plane).  On two different curved maps it is an integral over the
  PHYSICAL cell of one basis pulled back against the other: it does not
  factor.
* **Transition map.**  `Psi = Phi_a^-1 o Phi_b` takes a point in layer b's
  `(u, v)` to layer a's; its Jacobian `T = J_a^-1 J_b`.
* **Piece.**  The overlap of one cell of layer a with one cell of layer b.
* **Fast path.**  The Phase C stack-wide merged map, taken when the shapes
  of every layer fit one map.
* **Riding.**  A homogeneous layer adopting its neighbour's grid (section
  2.5).
* **Rung / fail-before / oracle**: as in the plan (0.1).

---

## 1. What was built

| item | file | what it does |
|---|---|---|
| the curved cross-mass kernel | `lumenairy/elements/pmm/_curvemortar.py` (new) | `curved_cross_mass(ga, gb, n)`: the ONE quadrature kernel (section 2.2); `curved_cross_mass_adaptive` (node count grown 1.5x until a 1e-12 relative change, capped at 96 with a warning); `cross_h_from_x`; `StagCrossOpsMapped` (the dense twin of `StagCrossOps`); `_MapView` (pointwise geometry, damped Newton inversion, point location, the singular vertices of a map, its interior edges). |
| the dense mortar branch | `_core._interface_smatrix_mortar_2d`, `_core._interface_smatrix_general_mortar_2d` | when `cr.dense`, the cross operators are applied as matrices; each side's own mass stays its separable plain Gram; the separable arithmetic is unchanged (E2-1). |
| `StagGridOps(..., cmap=None)` | `twod_staggered.py` | a grid carries its layer's map; the key includes the map fingerprint (`None` = the shipped key). |
| `SeparableStretch.geom_points` | `_curvemap.py` | a vectorised pointwise form (the base class looped over distinct `U`). Additive. |
| per-layer maps in the stack | `stack2d_pure.py` | `add_layer(..., cmap=)` on per-layer stacks (validation `_check_layer_map`); per-layer `shapes=` (`_recompile_shapes_perlayer`: every shape layer compiles its own map; the Phase C merge is then attempted as the fast path, `_perlayer_fast_ok`); `_solve_per_layer` solves each region, the geometric eig and the half-spaces under the grid's map, takes `StagCrossOpsMapped` between differently mapped grids, the cofactor far field and the unique L2 incident projection on MAPPED end grids, and caps `n_orders` only on unmapped ends; the q-matching default and the riding rule (`_perlayer_modal_counts`, `_perlayer_geometry`); the viewers draw each layer on its own map (`_layer_map`). |
| the Phase C verifier fold-in | `shapes2d.py`, docs, tests | section 6. |
| tests | `tests/unit/test_pmm2d_staggered_curved_e2.py` | gates E2-1 .. E2-8 and the API (12 tests, 0.9 .. 14.4 s each serially, 54 s total). |
| docs | module docstrings of `_curvemortar`, `stack2d_pure` (the per-layer section gains PER-LAYER MAPS; the `cmap=` / `add_layer` docs), `twod_staggered` (scope note); `CHANGELOG.md`; `docs/PMM_ROADMAP.md`; history fingerprints of `stack2d_pure` and `_core` re-recorded with the reason. | -- |

Not touched: `_assemble_oop`, slant (E1), any JAX file (E3), the walker test
file (E1's V-D6).

---

## 2. The design decision, measured

### 2.1 The formula (one bilinear form, two operators)

The shipped mortar imposes continuity weakly in each layer's own plain
`du dv` product, which is the metric-free flux pairing
`INT (E x h*) . z dx dy` (the 2-D cross product of covariant components
carries `det J`, which cancels the area element -- Phase A's finding).  Pulling
layer a's covariant field into b's coordinates is the 1-form pullback
`E'_b = T^T E'_a`, so the E-row cross operator is

    X_{beta alpha}[i, j] = INT conj(f^b_{beta,i}(q)) T_{alpha beta}(q) f^a_{alpha,j}(Psi(q)) du_b dv_b

(`beta`, `alpha` the component, 1 = along u in V1 = B x Btilde, 2 = along v in
V2) -- equivalently, in a's coordinates, the weight `adj(J_b^-1 J_a)` with
`du_a dv_a`.  The H-row operator is NOT a second integral: the same bilinear
form, written for the H placement of the Eq.-25 dual (H1 in V2, H2 in V1) and
the adjugate of `T`, gives

    CrossH = [[ X22^H, -X12^H],
              [-X21^H,  X11^H]]

which is the shipped `blkdiag(C2, C1)` when `T = I`.  The flux identity
`F_a(e_a, H_a) = F_b(E_b, h_b)` was checked coefficient by coefficient
(all four blocks); every other piece of the mortar is unchanged.

### 2.2 The three approaches the brief named

The brief asked for a measured choice among (i) a common fine parametrisation
on which both bases are evaluated by inverting one map, (ii) the union of the
two maps' grids with a composite map, each basis pulled back through it, and
(iii) a restriction to maps that agree on a shared coarse structure.

**The geometric fact that decides it.**  Inside one piece the integrand is
analytic (both bases polynomial on their own cells, both maps analytic inside
a cell); ACROSS the other layer's walls it is not -- `B` is discontinuous
across walls and `Btilde` kinks.  In either layer's coordinates the other
layer's walls are CURVES, and where the two outlines CROSS (the case E2
exists for) no tensor grid can contain both sets of curves as grid lines: a
composite map whose cells refine both layers' cells does not exist.  So
approach (ii), carried out literally, IS one tensor Gauss rule per cell of
one layer with the other layer's walls cutting through those cells; and
approach (iii) is exactly Phase C's merge, which refuses crossing outlines.

**What was built**: approach (i) with exact cuts -- an iterated Gauss rule on
every cell of the primary layer, CUT along the other layer's walls pulled back
into it:

1. the other layer's interior edges are pulled back (each smooth; two meet
   only at a grid vertex of the other layer, never cross);
2. an inner direction is chosen that minimises the curves' tangency points;
3. the outer coordinate is broken at every curve end, tangency and parallel
   curve; an interval ending at a tangency takes `s = s0 + L xi^2` (the
   crossing points move like `sqrt(s - s0)` there);
4. along every outer node the inner coordinate is broken at every crossing
   (a 2-D Newton on curve parameter and crossing point together -- no nested
   inversion), each sub-interval takes a Gauss rule, its other-layer cell is
   located once and every node is inverted there (damped Newton on that
   cell's analytic Gordon-Hall formula, nearest-sample start, bounding-box
   prefilter).
5. **Singular vertices.**  `det J = 0` at a closed curve's 45-degree points,
   where `Phi^-1` behaves like a square root.  In THAT map's coordinates the
   integrand stays analytic (`adj(J)` is bounded and its basis polynomial), in
   the other's it carries the root.  Every piece is integrated in the
   coordinates of the map whose singular vertex it touches; a piece touching
   both is refused, naming the cells, unless the two cells carry the same map
   (identical transition).

The three numbers the brief asked for (`e2_d_design_M4.json`, M = 4, the
circle / crossing-sinusoid pair):

| quantity | measured |
|---|---|
| Newton inversion per node | **31.5 us** (circle map), **21.7 us** (sinusoid), max error 9.8e-15 / 1.1e-15 over 36 000 / 16 000 random nodes; one Gordon-Hall evaluation 0.36 us / point.  At M = 8 the whole curved cross-mass is **1.1 s** of a 48 s per-layer solve (`e2_9_cost_M8.json`). |
| conditioning of the non-separable cross-mass | `cond(X)` **769** against **829** for the separable cross-mass of the same two grids unmapped and **878** for the plain Gram -- no conditioning penalty.  The mortar's solves see `X` only on the right-hand side (their operands are each side's own Gram times its modes, unchanged). |
| (ii) composite rule: exact or not | **NOT exact, algebraic**: relative error of the cross-mass 4.2e-2, 1.4e-2, 8.1e-3, 8.6e-4, 3.2e-4, 2.0e-4 at n = 8 .. 256 nodes per cell per axis (31 s at 256).  The cut rule: 1.4e-6, 3.2e-10, 5.5e-14, 6.8e-15, 1.0e-15 at n = 6, 9, 12, 18, 27 -- spectral. In the stack the composite rule at n = 64 moves the E2-4 device by 6.8e-5 at M = 5 (`e2_8_mutations_M5.json`, arm `nocut`): below that rung's discretisation level, but it is a floor that does not fall with M. |

**Decision: approach (i) with exact cuts, (iii) kept as the fast path.**  The
cut rule is the only one that converges spectrally on crossing outlines; it
costs 1-2 s per interface and no conditioning; the Phase C merge stays first
because it is cheaper per DOF and conforming (section 4.3).

### 2.3 Independent checks of the kernel (before any stack)

| claim | oracle | measured | JSON |
|---|---|---|---|
| the same map on both sides | the plain Gram | 2.5e-15 (tau = 1 and complex) | `e0_smoke_M4.json` |
| identity maps, different walls | the shipped separable `StagCrossOps` | 7.3e-16 | `e0_smoke_M4.json` |
| ONE integral: computed in either layer's coordinates (sinusoid-x / sinusoid-y) | itself, other node set and weight | 7.2e-16 at n = 24 | `e0b_consistency_M4.json` |
| two separable stretches | the 1-D factorisation `X11 = kron(Cy[Btilde], Cx'[B])` by brute-force 200-node Gauss | 1.3e-14 / 5.4e-15, `X12 = X21 = 0` exactly; without `psi'`: 0.42 | `e2_x_separable_M5.json` |
| circle / sinusoid in the SINUSOID's coordinates (the circle's root singularity) | the cut rule in circle coordinates | 6.9e-4 (n = 16), 1.8e-4 (n = 32): algebraic, as predicted -- the reason for rule 5 | `e0b_consistency_M4.json` |

### 2.4 The q-matching default (measured, adopted)

The mortar's accuracy is set by the COARSER of the two trace spaces.  A
sinusoidal wall compiles to a 2 x 2 map, a circle to 3 x 3, so at equal M the
sinusoid layer has 2/3 of the circle's per-axis count `q = N (M - 1)`.  On an
ALL-VACUUM two-layer stack across a sinusoid map and a circle map (an
exact Airy slab), at M = 6 (`e2_h_host_sin_circ_vac_*.json`):

| (M_sinusoid, M_circle) | error vs the exact slab |
|---|---|
| (6, 6) | 1.0e-6 |
| (6, 8) | 1.0e-6 |
| (8, 6) | 3.9e-9 |
| (8, 8) | 2.2e-10 |

The coarse side limits; the fine side does not.  (These are explicit
``cmap=`` films with ``n_modes`` given per layer, where the default itself
never applies -- they measure the mechanism, not the default; the default
is pinned two-sided by the verifier's ``test_ve2_6``.)  **Decision**: a shape layer
that does not name `n_modes` takes the finest shape layer's `q`
(`M_i = max(M, ceil(q_max / N_i) + 1)`), the shipped neighbour rule's analogue
for patterned layers.  Single-map references reach 1e-11 .. 1e-13 by M = 6 / 7
(`e2_h_host_shared_*`), so the mortar is the residual once the coarse side is
matched.

### 2.5 Riding: a homogeneous layer needs no map of its own

A layer whose material is homogeneous (a vacuum-painted shape layer; a
uniform `eps` layer without per-layer keywords next to a mapped one) rides
its neighbour's grid, so that interface is a square match instead of a
mortar.  It is what makes the vacuum-layer identity exact (E2-4: BIT-identical
to the same spacer on the shared map) and removes a needless curved mortar --
measured, keeping the vacuum layer on its own sinusoid map under the
circle's rim costs 2.1e-3 / 3.4e-3 / 2.7e-3 at M = 4 / 5 / 6
(`e2_4_vacuum_M*.json`, `noride_vs_alone`).  An all-unmapped per-layer stack
never rides (no mapped neighbour) -- the shipped bytes are untouched (E2-1).
`_e2_no_ride` is the test instrument.  A rider takes the nearest non-riding
layer ABOVE, else below; riding above vs below differs only at the
discretisation level (verifier V5).  A uniform layer that names
`n_modes` / `grid` / walls keeps its own grid, and naming `n_modes` on ANY
layer -- even the stack's own `M` -- takes a mergeable stack off the
merged-map fast path and exempts that layer from q-matching: measured 0.12 /
0.028 in R / T at M = 4 / 5 (verifier V5).  Documented in the
`stack2d_pure` module docstring and the CHANGELOG.

### 2.6 V-D3 (the Phase C verifier): a hybrid merge -- measured, deferred

The verifier found the per-edge merge OVER-REFUSES supercells (two circles of
radii 0.30 / 0.40 in one row; equal circles offset 0.02 .. 0.05 p in y; a
rectangle touching a circle at 30 deg) and recommended a hybrid: claims for
material boundaries, the macro-cell blend for foreign walls.  Measured here
(`e2_m_overrefusal_M{3,4,5}.json`):

| layout | one layer | two layers, shared | two layers, per-layer (E2) closure M = 3 / 4 / 5 |
|---|---|---|---|
| circles r 0.30 / 0.40 (2.4 x 1.2) | FOLD | FOLD | 3.7e-2 / 1.1e-2 / 1.6e-2 |
| equal circles, dy 0.05 | FOLD | FOLD | 7.5e-2 / 1.9e-2 / 9.6e-3 |
| rectangle touching a circle at 30 deg | FOLD | FOLD | (n_orders cap at M = 3) / 1.5e-2 / 1.6e-3 |

E2 solves the DIFFERENT two-layer devices (each shape in a layer of its
own); they are not the over-refused one-layer supercells and not a
workaround: against the one-layer macro-cell (t 0.5) the splits differ by
0.43-0.51 in R / T at M = 3..5 and do not converge to it (verifier V6; a
zero-thickness split is refused).  In ONE layer they still refuse:
the curved mortar couples layers, not shapes within a layer, so a same-layer
supercell needs the hybrid merge itself, which changes Phase C's map
construction and its pinned fingerprints.  **V-D3 stays DEFERRED, not closed** (not built), with the messages and
docstring fixed per both verifiers (sections 6 and 10: no "split the layer"
advice) and the Phase C verifier's `v10_macro.py` as the measured
prototype; recorded in the plan's open list.

---

## 3. The gates

Unit-test bars are derived from the measurements below, each stated in the
test's docstring with both gaps.  "Ladder" = build doc only.

### 3.1 E2-1 -- no map = today's bytes

| claim | measured | bar | fail-before | JSON |
|---|---|---|---|---|
| the full staggered fixture set (Phase D's 122 + 12 per-layer MORTAR fixtures: non-conforming pairs at normal / oblique / conical incidence with absorption, non-uniform walls with a lossy uniform layer, the generalized out-of-plane mortar, a forced conforming mortar, a taper staircase, the cross-mass factors) vs `eae470d9` | **134 / 134** SHA-256 identical (re-run after the Phase C verifier fold-in: still 134 / 134) | equality | the identity map through the quadrature path: 28 / 36 operator hashes change (the 8 equal ones are `None` attributes) | `e2_1_bytes_{pre,post}.json`, `e2_1_compare.json` |
| the unmapped per-layer path never reaches the new kernel | non-conforming lossy + uniform + out-of-plane + absorption with `curved_cross_mass` booby-trapped: no trap; the same trap fires on a mapped stack | -- | -- | unit test `test_e2_1_...` |

Bit-identity of the new kernel with the separable one is NOT possible (a
different quadrature arrangement, 7.3e-16 apart): the separable mortar stays
the pinned fast path for unmapped pairs.

### 3.2 E2-2 -- the same map through the curved mortar is the shared solve

A circle layer over a lossy film (eps 2.25 + 0.05i), both on the 3 x 3 circle
map (`e2_2_same_map_M4.json`):

| incidence | per-layer (square match) vs shared | FORCED curved mortar vs shared | H-swap off (fail-before) |
|---|---|---|---|
| normal | bit-identical | 1.4e-15 | 1.10 |
| (0.3, 0) | bit-identical | 1.3e-15 | 0.92 |
| (0.3, 0.7) conical | bit-identical | 3.1e-15 | 0.80 |

Bar 1e-10 (the brief's; 4.5 decades above the reading); fail-before >= 0.1.
The STOP condition (no approach exact to 1e-10 when the maps agree) is not
triggered.

### 3.3 E2-3 -- two different separable stretches

Kernel level: section 2.3 (1.3e-14 against the 1-D factorisation).  Stack
level: a pillar on x, y in 0.3 .. 0.9 stretched in x (0.06 p) over eps 2.25 on
x in 0.5 .. 0.8, y in 0.2 .. 0.7 stretched in y (0.08 p), against the same
device on its physical walls (`e2_3_stretches_M*.json`):

| M | mapped (curved mortar) vs unmapped per-layer (shipped mortar) | mapped vs the 5 x 5 union grid | unmapped vs the union grid | mapped closure |
|---|---|---|---|---|
| 4 | 2.1e-2 | 3.6e-2 | 3.6e-2 | 1.2e-2 |
| 5 | 2.0e-3 | 4.3e-3 | 2.8e-3 | 1.0e-3 |
| 6 | 6.0e-3 | 5.8e-3 | 6.9e-4 | 8.0e-5 |
| 7 | 5.2e-4 | 4.8e-4 | 3.3e-4 | 1.9e-5 |

The stretched stack converges to the physical device (5e-4 at M = 7), in the
same class as the shipped per-layer mortar on the same walls; both differ
from the union grid at the own-walls-only level (section 4.3).  The 2 x
cross-over at M = 6 is the non-monotone own-walls-only behaviour the shipped
mortar shows too.  Unit bar M = 5 <= 6e-3 and the M = 4 -> 5 drop >= 0.5
decades (measured 1.0).

### 3.4 E2-4 -- a circle over a sinusoidal wall that crosses it

Layer 1 (0.3) an eps-4 disk r = 0.36; layer 2 (0.25) the wall
`x = 0.6 + 0.12 sin(2 pi y / p)` between air and 2.25 -- through the disk.
The merge refuses ("... CROSS in plan view"), the per-layer maps solve.

| M | lossless closure (`e2_4_closure_M*.json`) |
|---|---|
| 4 | 6.5e-4 |
| 5 | 1.8e-4 |
| 6 | 2.4e-5 |
| 7 | 7.6e-7 |
| 8 | 9.7e-7 |

The STOP condition (closure <= 1e-4 by M = 8) is met at M = 6.  Fail-before:
the H-row swap off reads closure 0.10 / 0.12 at M = 5.

**Closure is NOT the accuracy.**  Against an independent conforming merged
map and RCWA (the Phase E2 verifier's V3, the two agreeing to 4.3e-4) the
R / T error of this device is 1.1e-1, 3.0e-2, 1.4e-2, 1.6e-3, 1.0e-3,
9.6e-4, 7.2e-4 at M = 4 .. 10 (conical 25 / 40 deg: 9.4e-2 .. 6.7e-3, not
monotone); closure understates it by 2-5 decades; the rung-to-rung change
tracks it within 2x.  The ACCURACY CLASS of the curved mortar on a
patterned / patterned interface whose outlines cut each other's cells is
therefore ~1e-2 at M = 6 and ~1e-3 at M = 8 .. 10 (section 4.3).

**Absorption** (disk lossy, eps 4 + 0.3i; `e2_4_absorb_M4.json`): the LOSSLESS
wall layer absorbs 1.5e-15 / 4.6e-16 (each layer's PLAIN Gram is its flux
form -- the Phase A rule carried across the mortar); with `-R` as the flux
form it reads 5.0e-3 / 4.7e-3.  The budget sum(A) against 1 - R - T: 7.5e-3 /
5.4e-3 at M = 4 (the discretisation level; recorded).

**The vacuum-layer identity** (a vacuum-painted wall layer on top of the disk
-- air above, a physical no-op; `e2_4_vacuum_M*.json`):

| M | per-layer vs the same spacer on the shared circle map | that spacer vs the circle alone (Phase C's spacer effect) | NO-RIDE arm vs the circle alone |
|---|---|---|---|
| 4 | **0 (bit-identical)** | 3.6e-4 | 2.1e-3 |
| 5 | **0** | 7.4e-5 | 3.4e-3 |
| 6 | **0** | 5.0e-6 | 2.7e-3 |

The identity is exact through riding (2.5).  "Phase B circle to round-off",
as the brief phrased it, is not attainable even on ONE map: a vacuum spacer
on a mapped circle moves R / T at the discretisation level (Phase C gate C9:
the half-space modes under a map are not plane waves) -- the middle column.

**Both ways where a merged map exists** (the wall at x0 = 0.12, A = 0.05, left
of the disk; `e2_4_merged_M*.json`): the per-layer stack takes the fast path
and is BIT-identical to the shared stack at every rung; forced through the
curved mortar (`_e2_per_layer_maps`):

| M | per-layer maps vs merged map | closure merged | closure per-layer |
|---|---|---|---|
| 4 | 9.9e-2 | 1.5e-3 | 1.5e-3 |
| 5 | 1.1e-2 | 1.4e-3 | 2.8e-4 |
| 6 | 9.7e-3 | 1.8e-4 | 2.8e-5 |
| 7 | 4.5e-4 | 8.5e-6 | 5.1e-6 |

At M <= 8 the merged map's own error dominates this column (merged vs
merged M10: 1.05e-1, 1.01e-2, 9.75e-3, 4.34e-4, 4.14e-4); the per-layer
limitation appears beyond M = 8 as a stall near 3e-4 while the merged map
reaches 1.8e-5 at M = 9 (verifier V4).  Unit bar M = 4 <= 0.2.

### 3.5 E2-5 -- the mortar's own convergence (quadrature vs node count, fixed M)

Relative error of the cross-mass against its own n = 3 x (adaptive) value
(`e2_5_quadrature_M{4,6}.json`):

| n | circle / sinusoid M = 4 | sinusoid-x / -y M = 4 | circle / sinusoid M = 6 | sinusoid-x / -y M = 6 |
|---|---|---|---|---|
| 4 | 1.5e-2 | 4.3e-3 | 1.4e-1 | 1.8e-1 |
| 8 | 3.7e-8 | 1.9e-6 | 1.6e-5 | 8.4e-5 |
| 12 | 9.3e-13 | 3.6e-10 | 6.9e-10 | 1.2e-7 |
| 16 | 4.6e-16 | 3.5e-14 | 1.7e-14 | 3.5e-11 |
| 20 | 3.8e-15 | 8.1e-16 | 1.3e-14 | 5.3e-15 |
| adaptive stop (n, last change) | 23, 1.3e-15 | 27, 8.2e-15 | 30, 9.7e-15 | 35, 2.0e-15 |

Spectral; the adaptive rule lands 2-3 decades under its 1e-12 tolerance.
Unit bar (circle / sinusoid, M = 4): n = 12 <= 1e-11, the adaptive result
<= 1e-13; fail-before the composite rule at n = 64 >= 1e-4 (measured 8.6e-4).

**Its limit, measured** (`e2_n_near_singular.json`): two CIRCLE maps whose
singular vertices approach -- r 0.36 against 0.30 / 0.34 / 0.355 / 0.3599 --
need n = 41 / 41 / 62 / 96 (the cap), all converged to <= 2.2e-13; at
-- superseded by the verifier (V-E2-D6): below 5e-5 .. 7e-5 apart (M = 4) a
node inversion fails, and at 7e-5 .. 3e-4 the n >= 112 rules already fail.
Concentric or near-tangent circles closer than the sliver contract are
refused by the merge, so a per-layer stack DOES route them through the
curved mortar (dr 1e-4 solves at n 96; dr 1e-6 raises from solve(), now
naming the two layers, the cells, the rung and the remedy).  Crossing
circles never reach this limit: they are refused up front (V-E2-D2).

### 3.6 E2-6 -- oblique and conical incidence through the curved mortar

The E2-4 device at (25 deg, 0) and (25 deg, 40 deg); reciprocity of
reflection order (-1, 0) -- singular values of the power-normalised Jones
block against the reversed channel's (`e2_6_angles_M*.json`):

| M | reciprocity (25, 0) | reciprocity (25, 40) | closure (max over both inputs, both directions) | wrong pairing (0, 0) |
|---|---|---|---|---|
| 4 | 1.4e-4 | 7.5e-4 | 5.8e-3 | 0.039 / 0.067 |
| 5 | 1.1e-5 | 1.1e-4 | 1.8e-3 | 0.041 / 0.050 |
| 6 | 7.7e-6 | 7.2e-6 | 2.3e-4 | 0.045 / 0.073 |
| 7 | 1.0e-6 | 7.3e-6 | 5.6e-5 | 0.038 / 0.077 |

Unit bars at M = 4: reciprocity <= 3e-3, closure <= 2e-2, wrong pairing
>= 1e-2.

### 3.7 E2-7 -- three layers, three maps

A disk (0.25), the x-wall through it (0.2), a wall along y crossing both
(`y = 0.5 + 0.1 sin(2 pi x / p)`, eps 1.7, 0.2) -- two curved mortars in one
cascade (`e2_7_three_M*.json`):

| M | closure | middle layer lossy: absorption of the two LOSSLESS layers | sum(A) vs 1 - R - T |
|---|---|---|---|
| 4 | 1.5e-3 | 2.5e-14 | 1.2e-3 |
| 5 | 2.1e-4 | 5.3e-14 | 8.1e-5 |
| 6 | 7.4e-5 | 2.6e-13 | 6.6e-5 |
| 7 | 2.9e-6 | 2.7e-13 | 1.4e-6 |

Unit bars at M = 4: closure <= 1e-2, lossless absorption <= 1e-10.

### 3.8 E2-8 -- the mutation matrix

On the E2-4 device at M = 5 (`e2_8_mutations_M5.json`, shipped closure
1.8e-4 / 9.0e-5) unless stated:

| mutation | effect | caught by |
|---|---|---|
| map-inversion Newton tolerance loosened 1e3x | R / T move 2.6e-15 | nothing -- by design: Newton converges quadratically past it; the load-bearing tolerance is the residual acceptance (`_CURVE_MORTAR_RES_TOL`), not the step |
| the cross-mass computed with BOTH maps ignored (the separable cross-mass on the two `(u, v)` grids) | R / T move 6.7e-2; closure 2.0e-4 (unchanged class) | E2-3's factorisation (0.42), E2-4 both ways (3.9e-2 against the shipped 1.1e-2 at M = 5), unit test E2-8 (>= 1e-2 at M = 4) -- closure alone cannot see it |
| the cross-mass on ONE layer's map (both bases through the upper map) | the geometry is inconsistent (the lower basis read on the wrong walls); E2-3's factorisation measures the class (no `psi'`: 0.42) | E2-3 |
| the flux Gram replaced by `-R` | the lossless layer absorbs 5.0e-3 / 4.7e-3 (M = 4) | E2-4 absorption (bar 1e-10 two-sided) |
| the composite (no-cut) rule, n = 64 | R / T move 6.8e-5 -- a floor that does not fall with M | E2-5 (the cross-mass 8.6e-4) |
| the H-row V1/V2 swap off | R / T move 0.51; closure 0.10 / 0.12 | E2-2, E2-4 |
| the fast path forced onto the crossing pair (the merge's crossing test disabled) | the merge's independent vertex-claim check refuses ("DIFFERENT physical positions for the same grid vertex"); the stack stays on the per-layer maps, R / T unchanged (0.0) | the merge itself; unit test E2-8.  E2-4's both-ways comparison is the gate that would see a merged map built anyway (the maps-ignored arm reads 3.9e-2 there) |

### 3.9 E2-9 -- cost

The NON-overlapping pair, where both routes exist (`e2_9_cost_M{6,8}.json`;
loaded box, upper bounds, ratios within one run):

| M | merged map: grid, pencil, wall | per-layer maps: M per layer, pencils, wall | curved cross-mass (n) | ratio per-layer / merged |
|---|---|---|---|---|
| 6 | 4 x 4, 800, 34.7 s | (6, 9), 450 / 512, 10.0 s | 1.18 s (30) | 0.29 |
| 8 | 4 x 4, 1568, 252 s | (8, 12), 882 / 968, 47.8 s | 1.11 s (36) | 0.19 |

The per-layer route is cheaper per solve (two smaller pencils instead of one
merged 4 x 4 one; the eig is cubic) and the curved cross-mass is 2-12 % of it (circle / sinusoid; circle /
circle spends most of the solve in it, verifier V3g).
It is LESS accurate per rung where an outline cuts the neighbour's cells
(3.4, 4.3), so the merged map stays the default whenever it exists.

---

## 4. Findings

### 4.1 F-E2-1 -- a symmetric curve's tangency sat on a sample (fixed during the build)

The tangency search skipped a derivative that was EXACTLY zero at a
Chebyshev sample (a concentric circle pulled back into a circle map is
symmetric about tau = 1/2): the tangency was missed, a sub-interval spanned
two cells, and a node inversion failed.  Sign changes are now taken between
consecutive NONZERO samples, a zero sample being the tangency.

### 4.2 F-E2-2 -- Newton started on a singular corner (fixed during the build)

The nearest-sample guess could be the cell's singular corner itself
(`det J = 0`: an infinite step).  The sample grid is now Gauss-interior; a
point exactly AT a singular vertex is returned as that corner; the Newton is
damped (halved steps until the residual falls) and stops an iterate that
leaves the cell by a full cell; points outside the cell's bounding box skip
Newton.

### 4.3 F-E2-3 -- own-walls-only mortars converge ALGEBRAICALLY where an outline cuts the neighbour's cells (the limitation)

The field just above or below a pillar carries the rim (edge) singularity
along the pillar's outline.  In the pillar's own map that outline is a grid
line; in a neighbour whose grid does not contain it, the outline cuts
through cells and the trace converges algebraically.  This is not specific
to the curved mortar -- the SHIPPED separable mortar shows it on the same
geometry:

| arm (`e2_v_spacer_shipped_M*.json`, `e2_w_shipped_baseline_M*.json`) | M = 4 | 5 | 6 | 7 |
|---|---|---|---|---|
| shipped: square pillar, vacuum spacer on ONE cell (the M rule) vs the pillar alone | 4.1e-3 | 3.0e-3 | 1.2e-3 | -- |
| shipped: same, spacer on 3 x 3 walls at 0.2 / 0.8 | 3.6e-3 | 2.1e-3 | 1.7e-3 | -- |
| shipped: same, spacer on the pillar's walls (conforming) | 4.7e-6 | 2.3e-7 | 7.7e-9 | -- |
| shipped: pillar over a stripe on other walls vs the union grid | -- | 2.8e-2 | 9.7e-3 | 5.6e-3 |
| ... the stripe layer q-matched | -- | 5.1e-3 | 8.6e-4 | 1.4e-3 |
| curved: vacuum layer on its own sinusoid map over the disk (no ride) | 2.1e-3 | 3.4e-3 | 2.7e-3 | -- |
| curved: disk over the non-crossing wall vs the merged map | 9.9e-2 | 1.1e-2 | 9.7e-3 | 4.5e-4 |

So the per-layer curved route answers to the 1e-2 .. 1e-4 level at practical
M on patterned-patterned interfaces, in the shipped mortar's class.  The
cause is CONFIRMED by the Phase E2 verifier (V4): the algebraic tail (rung
exponent p_M 1.8 .. 2.2 at eps 4) keeps its exponent at eps 1.1 and 1.02
while its size relative to the scattered amplitude falls in proportion to
delta-eps -- the rim singularity seen through a non-conforming cut.  The
merged map (Phase C), being conforming, converges like the lone circle
(rim-capped at ~1e-5 per rung, plan 3.4; slowed beyond M = 10 by the rim
edge, Phase C table 3.2: 1.7e-5 at M = 10, 7.8e-6 at M = 11 against the FEM).
Hence the design: the merge is tried first and taken whenever it exists; the
curved mortar is the route for what one map cannot carry.  Documented in
the `stack2d_pure` module docstring and the CHANGELOG.

### 4.4 F-E2-4 -- the uniform-film arms are NOT the brief's "round-off"

The vacuum identity is exact only because the homogeneous layer rides
(2.5).  A genuine two-map mortar has an energy-conserving error set by the
coarser trace space (2.4: 1e-6 at M = 6 equal-M on an all-vacuum stack,
3.9e-9 q-matched); single maps reach 1e-11 .. 1e-13 there.

### 4.5 F-E2-5 -- incident representation ("mode-pick"), recorded for after Phase E

The Phase C verifier measured that taking the incident wave as the discrete
(0, 0) eigenmode pair of the superstrate is equal or better than the
renormalised L2 projection on every measure (up to 200x), on the mapped path;
moving the UNMAPPED oblique path to a window-free projection would move
shipped bytes (a Migration-Guide entry).  Not done in E2 (the per-layer path
uses the shared path's mapped projection on mapped end grids and the shipped
least squares on unmapped ones); recorded as a post-E item.

---

## 5. What moved

Nothing shipped: E2-1, 134 / 134 SHA-256 identical to `eae470d9`; no
default answer of an existing call moved.  New behaviour exists only on
`layer_grids='per-layer'` stacks that use `cmap=` or `shapes=` (which raised
before).  The Phase C verifier fold-in (section 6) changes two Phase C
outputs by design: a rectangles-only merge whose vertex claims sit within
1e-13 p of the grid is now the identity (the unmapped solver -- it was the
mapped path, 6.0e-5 apart at oblique incidence on such layouts), and a
rotated `Ellipse` lays out its corners at the normal-angle points (the
parametric points folded for most of the documented range).  No E2-1 fixture
contains either (still 134 / 134 after the fold-in).

---

## 6. The Phase C verifier fold-in (`VERIFY_PMM2D_CURVED_C_2026_10_03.md` section 9)

Merged `verify/pmm2d-curved-c` (`eff64792`) with `--no-ff` (`.test_durations`
merged cleanly).  Applied:

| item | edit | evidence |
|---|---|---|
| V-D1 (P2) | `_VERTEX_SNAP = 1e-13`; near-grid vertex claims snapped in `_merge` | `test_vc4` flips (marker removed) |
| V-D2 (P2) | the rotated `Ellipse` corners where the outward normal points at 45 / 135 / 225 / 315 deg; docstring | `test_vc5` flips (marker removed) |
| V-D3 (P3) | the fold message and the shapes2d docstring name the arc-bulge mechanism; measured and deferred (2.6) | `e2_m_overrefusal_M*.json` |
| V-D4 | "exact modal decomposition" -> "unique L2 modal projection (window-free)" (docstrings, comments, CHANGELOG, roadmap); CHANGELOG examples `n_modes=4`; the roadmap names the PLAN's Phase D / E | -- |
| V-D5 | the fillet limit written `sqrt(2) x 1e-3 (1.4142e-3)` in the docstring and the message; the C5 regex follows | `test_c5` |
| V-D7 | float reprs in the sliver, vertex-claim and fillet messages | -- |
| V-D6 | NOT touched (E1's, the walker test file) | -- |
| `test_vc2` | reads `_merge`'s first five outputs (Phase D made it return six) | -- |

---

## 7. Not measured

* **An independent oracle for a crossing two-layer device** (a 3-D FEM run of
  the circle over the crossing wall): the references here are physical
  identities (closure, absorption of lossless layers, reciprocity, the vacuum
  identity), the merged map where it exists, and the shipped mortar on the
  same geometry.
* **M > 8 ladders** of the per-layer devices, and E2-6 / E2-7 beyond M = 7
  (cost on the loaded box).
* **A same-layer supercell** (two circles in one layer): still refused
  (2.6).
* **Idle wall times**: the box was saturated throughout.

---

## 8. Reproduction

```
cd /c/tmp/lum_curved_e2/validation/probe_pmm2d_curved/build_e2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved_e2
python e0_smoke.py 4 ; python e0b_consistency.py 4 ; python e2_x_separable.py 5
python e2_d_design.py 4                                   # the design measurement
(cd /c/tmp/curved_pre_e2 && PYTHONPATH=C:/tmp/curved_pre_e2 python C:/tmp/lum_curved_e2/validation/probe_pmm2d_curved/build_e2/e2_1_bytes.py C:/tmp/curved_pre_e2 pre)
python e2_1_bytes.py C:/tmp/lum_curved_e2 post idmap ; python e2_1_compare.py
python e2_2_same_map.py 4
python e2_3_stretches.py <M>                              # M = 4 .. 7
python e2_4_overlap.py closure|absorb|vacuum|merged <M>   # M = 4 .. 8
python e2_5_quadrature.py <M> ; python e2_n_near_singular.py
python e2_6_angles.py <M> ; python e2_7_three.py <M> ; python e2_8_mutations.py 5
python e2_9_cost.py <M>                                   # M = 6, 8
python e2_h_host.py <arm> <M> [theta phi]                 # E2H_VACUUM=1, E2H_MS=a,b
python e2_v_spacer_shipped.py <M> ; python e2_w_shipped_baseline.py <M>
python e2_m_overrefusal.py <M>
cd /c/tmp/lum_curved_e2 && python -m pytest tests/unit/test_pmm2d_staggered_curved_e2.py --capture=sys -p no:randomly
```

## 9. Test tails

* Windows (CPython 3.14.6, numpy 2.4.4, scipy 1.17.1), the E2 unit file
  serially: `12 passed in 54.43s` (slowest 14.4 s, E2-2).
* Windows full sweep after the Phase C verifier fold-in (every
  `pmm2d` / `stack2d` / `stagger` / `curved` / `per_layer` file incl. Phases
  A-D, E2 and the verifier files, census, public API, every walker, doc
  identifiers, doc consistency, except budget, history lint / relocation /
  fingerprint tool, re-exports, kernel consistency; 70 files, `-n 6`):
  `1672 passed, 8 skipped, 96 warnings in 1059.09s` -- no failure (the
  `material_key` walker red the Phase C build doc reported is green on this
  base).
* WSL Ubuntu second build (CPython 3.12.3, numpy 2.4.6, scipy 1.17.1, BLAS
  pinned, `lumenairy` from `/mnt/c/tmp/lum_curved_e2`): curved A-D + E2 +
  the A / B / C verifier files + the mortar, per-layer-grid and mortar-fix
  files, `-n 4`: `180 passed, 1 skipped in 537.82s`.  Kernel readings on
  WSL (`e2_x_separable_M5_wsl.json`, `e2_5_quadrature_M4_wsl.json`): the
  separable factorisation 1.27e-14 / 5.37e-15 (Windows 1.28e-14 / 5.43e-15),
  the adaptive node counts 23 / 27 identical to Windows.
* `python scripts/record_history_fingerprints.py --check`: every document
  matches its module (stack2d_pure and _core re-recorded in `ba22a010`).
* `python -m mypy` (the configured strict list): `Success: no issues found
  in 33 source files`.  `lumenairy/elements/pmm/` is not on that list;
  `mypy lumenairy/elements/pmm/_curvemortar.py` reports only
  `no-untyped-def` / `no-untyped-call` (the genuine annotation errors it
  found were fixed).
* WSL ruff 0.15.16 on `lumenairy/`, the touched tests and
  `validation/probe_pmm2d_curved/build_e2/`: `All checks passed!`

---

## 10. The Phase E2 verifier fold-in (`VERIFY_PMM2D_CURVED_E2_2026_10_03.md`)

Merged `verify/pmm2d-curved-e2` (`2fd951d5`, `--no-ff`, durations
dict-union).  Verdict there: SHIP after documentation edits, no P1.

| item | what was done | evidence |
|---|---|---|
| V-E2-D1 (P2, code) | `_cell_pieces` runs a GRAZING refinement (`_grazing_refine`, the verifier's prototype: every between-sample dip into, or excursion out of, the cell resolved on a 257-point sub-grid) before reading its runs; a piece end that meets the outer direction nearly tangentially (`tan < 0.1`) counts as a tangency in the inner-direction choice and takes the square-root substitution | section 10.1; `test_ve2_3` flips to a plain gate |
| V-E2-D2 (P2, known limit) | the both-singular refusal now states the limit (most crossing closed-curve pairs; splitting the cell is not implemented) with no inapplicable advice; CHANGELOG and plan record it | `test_ve2_4` stays a strict xfail (the pin) |
| V-E2-D3 (P2, docs) | closure is not the accuracy: the accuracy class stated (3.4, CHANGELOG); the CHANGELOG example uses `n_modes=7` (1.6e-3 on this device) with the reason | -- |
| V-E2-D5 / D6 (P3, code) | the adaptive rule compares only rungs >= 1.5x apart, keeps the last good rung when a finer one fails to invert (warning), and its warning names the real cause (no unreachable `grid_hint` advice); the inversion failure names both cells, the rung and the remedy, and `solve()` names the two layers; the scope sentence of 3.5 corrected | unit test `test_e2_v5_v6_...` |
| V-E2-D7 (P3, docs) | V-D3 recorded as deferred (2.6); the CROSS / FOLD / vertex-claim messages and the `shapes2d` known limits no longer advise splitting a layer | -- |
| V-E2-D8 (P3, code) | `_VERTEX_SNAP = _WALL_SNAP`; pinned in `test_vc4` (a claim 5e-13 p off) | `test_vc4` |
| V-E2-D9 (P3, docs) | the numbers (2-12 %, 36 000 / 16 000 nodes, the merged-map column, < 3e-13, 3e-6, "16 to 20 nodes", the roadmap's Phase E1 / E3, the history reason line, the riding docs) | -- |
| V-E2-D10 (P2, code) | `convergence_floor` isolates a shape layer on its OWN map (straight walls were another device, 0.14 off at M = 3) and an explicit per-layer `cmap=` layer under its map; no longer raises on the fast path | unit test `test_e2_v10_...` (the circle layer's floor equals the shape alone on its map to 1e-12) |
| V-E2-D11 (P3, code) | a both-singular cell pair whose two maps are the same function of `(u, v)` on the rectangles' overlap (one circle on a `grid_hint` map and on its plain map) is integrated (identity transition), not refused | `e2_v11_layouts.json`: 1.6e-15 against the separable cross-mass; overlap test off -> refused; unit test `test_e2_v11_...` |
| defaults, two-sided (verifier item 9) | riding above vs below, the `n_modes` exemption from q-matching and from the fast path -- documented (2.5); the 1e-6 vs 3.9e-9 claim restated as an explicit-`cmap=` film measurement (2.4) | verifier `test_ve2_6` |
| the rim cause (verifier item 8) | cited (4.3) | verifier V4 |

### 10.1 V-E2-D1, measured against the verifier's physical brute force

`validation/probe_pmm2d_curved/build_e2/e2_g_graze_{pre,post}.json`, M = 4
(the verifier's `verify_e2/v2_brute.py` as the oracle; its own floor on
circle pairs is ~5e-10, measured here 5.0e-10 between its n = 20 and 28 on a
non-grazing circle pair, where the kernel agrees with it to 5.2e-10):

| case (M = 4) | kernel vs brute, BEFORE | kernel vs brute, AFTER | what the fix adds (kernel with / without the refinement) |
|---|---|---|---|
| sinusoid crest, x-wall graze delta = 0 | -- | 8.7e-15 | -- |
| ... delta = 1e-7 | 1.8e-10 | 8.7e-15 | -- |
| ... delta = 1e-6 | **5.6e-9** (silent: adaptive change 7.7e-15) | **8.3e-15** | 5.6e-9 |
| ... delta = 3e-5 | 8.3e-15 | 8.3e-15 | -- |
| circle bottom, y-wall graze delta = 1e-7 | 6.8e-7 | 6.8e-7 | 1.2e-10 |
| ... delta = 1e-6 | 6.4e-7 | 6.5e-7 | 3.7e-9 |
| ... delta = 1e-5 | 4.7e-7 (warned at the cap, n 96) | 4.7e-7 (n 62, no warning) | 0 (the sliver is sampled) |
| circle, wall 1e-3 / 1e-6 BELOW / touching / 0.03 below / 0.03 above (no graze) | -- | 5.4e-10 / 5.4e-10 / 5.4e-10 / 5.2e-10 / 5.6e-10 | -- |

The sinusoid graze is fixed to round-off.  On the circle, the refinement
finds the sliver the 65-sample run logic dropped -- its share scales like
the sliver's area, 1.17e-10 at 1e-7 and 3.70e-9 at 1e-6, a ratio of
31.6 = 10^1.5 (`e2_g_prepost.json`; unit test `test_e2_v1_...`).  The
REMAINING ~6.5e-7 against the brute force is the ORACLE's, not the
kernel's: at that graze the brute force's own n = 20 vs n = 28 change is
4.2e-7 (against 5.0e-10 on a non-grazing circle pair; `e2_g_oracle_selfchange.json`), with the wall just
below or touching the circle (no cut) the two agree to its 5.4e-10 floor
(`e2_g_oracle_check.json`), and the kernel's cross-mass is continuous and
LINEAR across the graze (relative change 1.9196 x delta from delta = -1e-3
to +1e-3, no jump at 0; `e2_g_continuity.json`).  The verifier's reading that the circle needs a further square-root
substitution is therefore not confirmed; the near-tangency treatment was
built anyway (a piece end with `tan < 0.1` counts as a tangency in the
inner-direction choice and takes the substitution) and is gated by the
verifier's `test_ve2_5`.  Second build (WSL): section 10.2.

### 10.2 Tails after the fold-in (both builds)

* Windows (CPython 3.14.6, numpy 2.4.4, scipy 1.17.1), curved A-D + E2 + the
  A / B / C / E2 verifier files + the mortar, per-layer-grid and mortar-fix
  files + history relocation, doc identifiers, except budget, `-n 4`:
  `950 passed, 1 skipped, 1 xfailed in 311.50s` (the xfail: `test_ve2_4`,
  the V-E2-D2 pin; `test_ve2_3` passes as a plain gate).  The broad 70-file
  sweep run before the machine's restart, on the same code: `1681 passed,
  8 skipped, 1 xfailed in 1114.44s`.  The four new ids, serially: 20.4 s
  (`test_e2_v1_...`), 16.1 s (`_v11_`), 5.1 s (`_v10_`), 2.7 s (`_v5_v6_`).
* WSL Ubuntu second build (CPython 3.12.3, numpy 2.4.6, scipy 1.17.1): the
  same test set without the three hygiene files, `-n 4`: `189 passed,
  1 skipped, 1 xfailed in 293.34s`; the circle-graze shares on WSL equal the
  Windows ones to every printed digit (`e2_g_prepost_wsl.json`).
* E2-1 re-run after the fold-in: 134 / 134 still identical to `eae470d9`.
* `record_history_fingerprints.py --check`: OK.  `python -m mypy`
  (configured list): no issues in 33 files.  WSL ruff 0.15.16: all checks
  passed.


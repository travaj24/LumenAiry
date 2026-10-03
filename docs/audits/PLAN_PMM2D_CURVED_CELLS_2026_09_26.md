# PLAN -- curved in-plane boundaries (circles, ellipses, fillets, sinusoidal walls) for the PURE staggered 2-D PMM

Date: 2026-09-26.  Status: PLAN (binding brief for the build agents, pending
the maintainer's answers in section 7).
Branch: `feat/pmm2d-curved-cells` on `4ec402bc` (5.49.0 + the stack viewer).
Target release: 5.50.0 (MINOR -- a new solver capability) once Phases A-C
land; Phase D may follow as a PATCH.
Evidence: every number below is read from a JSON file in
`validation/probe_pmm2d_curved/` (the probe scripts, their raw outputs and
`summary.json`, produced by `summarize.py`).  The probes ran on
tesla-ryzen (py3.14.6 / numpy 2.4.4 / scipy 1.17.1, BLAS pinned to one
thread on the command line) while an unrelated 96 GB finite-element job
held most of the machine, so every WALL TIME below is an upper bound; no
accuracy number depends on the load.

---

## 0. Summary, and the decision this plan asks for

### 0.1 Words used throughout

* **The pure staggered 2-D PMM** is the no-floor crossed-grating solver
  `pmm_efficiency_2d_staggered` / `pmm_jones_2d_staggered` /
  `PMM2DStackPure` (`lumenairy/elements/pmm/twod_staggered.py`,
  `stack2d_pure.py`; Granet, JOSA A 40, 652 (2023)).  It divides the unit cell
  into a grid of rectangles (the **wall grid**: `Nx` columns by `Ny` rows,
  `Nx == Ny` required) and expands the field in each rectangle in Legendre
  polynomials up to `M` functions per axis (`M` is the `n_modes` / `degree`
  knob).  Today every material boundary must be one of the grid's straight
  walls, so a circle can only be approximated by a **staircase** of
  rectangles.
* **A coordinate map** `(x, y) = Phi(u, v)` is a smooth distortion of the
  plane.  The solver works on a straight rectangular grid in the new
  coordinates `(u, v)`; the map bends that grid in the real `(x, y)` plane so
  that the circle (or fillet, or sinusoid) is exactly one of the bent grid
  lines.  The map does not depend on the height `z` ("z-independent"), so
  every layer of a stack can share it.
* **The Jacobian** `J = d(x, y)/d(u, v)` is the 2x2 matrix of the map's
  local stretch and shear; `det J` is its local area magnification and the
  **metric** `g = J^T J` its local change of lengths and angles.  Where
  `det J = 0` the map pinches a direction to zero length (a **singular
  vertex**).
* **Covariant components** `E' = J^T E` are the field components measured
  ALONG the bent grid lines.  In `(u, v)` coordinates Maxwell's equations keep
  exactly their Cartesian form if the materials are replaced by the
  **effective tensors** `eps' = det J * J^-1 eps J^-T` and
  `mu' = det J * J^-1 mu J^-T` (transformation optics).  For an isotropic
  material this is `eps'_t = eps sqrt(g) g^-1` in the plane and
  `eps'_33 = eps sqrt(g)` along z, and -- this matters -- the vacuum
  permeability becomes a non-trivial in-plane tensor `mu' = sqrt(g) g^-1`.
* **Transfinite (Gordon-Hall) map**: inside each rectangle of the `(u, v)`
  grid, the map is the bilinear blend of the four edge curves (an arc, a
  straight line, a sinusoid ...).  Neighbouring rectangles share their edge
  curve, so the map is continuous across every grid line (C0) even though
  its derivative may jump there.
* **Oracle**: an independent calculation used as the truth.  **Fail-before**:
  a deliberately broken arm of a test that must fail, proving the test can
  see the defect it guards.  **Rung**: one step of a convergence ladder in
  `M`.

### 0.2 What the probes say, in one paragraph

The maintainer's sketch survives with four amendments.  A z-independent
covariant map turns a curved cell into a cell of smoothly varying BLOCK-FORM
permittivity AND permeability tensors, which the shipped solver's own
magnetic + tensor route already knows how to discretise; replacing its
piecewise-constant masses by 2-D Gauss quadrature reproduces the shipped
operators to 5.5e-14 (P1) and gives a solver that converges to the SAME
answer as the unmapped one under a pure 1-D stretch (P2), is exact for a
uniform film under a circle map (spectral, 3.9e-14 by `M = 8`), and on a
circular pillar lands on an independent 3-D finite-element oracle to 1.9e-6
per order -- inside that oracle's own 8.3e-6 mesh spread -- while the shipped
2-D RCWA with the EXACT disk form factor at 29 x 29 orders is still 3.3e-3
away and the 4- and 8-step staircases are 7.1e-2 and 5.4e-2 away (P3).  The amendments: (i) it is not only the masses -- 18 weighted
blocks change, including the two weighted derivative operators, and the map
makes every region MAGNETIC, which reopens the module's documented
"ONE TRAP" (the H-partner Gram) in two shipped helpers; (ii) there is no
"interface projector" to change -- under one stack-wide map the interfaces
stay square matches and the only map-aware far-field piece is the Rayleigh
extraction, which needs the cofactor `det J * J^-T` (dropping it is an
O(1e-1) error, P2b); (iii) any smooth closed curve built from grid lines of a
tensor grid has four singular vertices (`det J = 0`), which the probes show
to be harmless; (iv) rounding a pillar's corners (a fillet) makes the
IN-PLANE convergence fast but does NOT make the diffraction efficiencies
converge spectrally, because the pillar's top and bottom rims -- untouched
by a fillet -- carry their own edge singularity (P4).

### 0.3 The decision

**Approve the phased build of section 4 -- Phases A to D -- with the map
owned by the STACK (one map shared by every layer) and the tensor-grid
topology (four singular vertices per closed smooth curve accepted); defer
Phase E.**  Seven smaller questions are in section 7, each with a measured
recommendation.

### 0.4 The sketch, part by part

| # | the sketch said | verdict | evidence |
|---|---|---|---|
| 1 | z-independent map; Maxwell in `(u, v)` with `eps' = sqrt(g) g^-1 eps`, `mu' = sqrt(g) g^-1 mu`; block-form tensors the solver already handles | **CONFIRMED**, with the precision that a scalar cell becomes a full in-plane tensor (`e12 = -eps g12 / sqrt(g)`) AND a magnetic one (`mu' != 1` everywhere, vacuum included), so every mapped region runs the magnetic + tensor route | P2 (stretch converges to the unmapped limit), P3 film (3.9e-14) |
| 2 | core change = variable-coefficient segment masses by 2-D quadrature; DOF, pencil, parity gauge, eigensolver unchanged | **CONFIRMED, AMENDED**: 18 weighted blocks (the four `R`, four `[eps_t]`, `Meps33`, the `chi33` Gram of `S_tt`, four `K_tz`, four `K_zt` terms), not only masses; the curl / gradient incidence matrices stay metric-free; DOF and pencil unchanged; the pencil's right-hand matrix stays Hermitian positive definite (so a Cholesky-whitened eig applies, 5.6-10.9x faster than QZ at the same size, P5); the parity gauge is used only by the out-of-plane path (Phase E) | P1 (5.5e-14), P5 |
| 3 | half-spaces stay physical Rayleigh expansions; what changes is the INTERFACE projector (plane waves pulled back with the `J^T` factor) | **REFUTED as stated, CONFIRMED in substance**: the shipped half-spaces are not Rayleigh expansions -- they are the staggered basis's own homogeneous modes, and the Rayleigh projection happens ONCE, at the far field.  Under one stack-wide map the half-spaces are solved in the mapped frame (the shared eps-free geometric eig SURVIVES exactly: `-R == [eps'_t]/eps`, measured <= 3.1e-16 relative), every interface stays a square match with NO projector, and the pulled-back plane waves enter only the far-field projector and the incident overlap -- with the COFACTOR `det J * J^-T`, not `J^T` (the Jacobian of the area element is not optional: 0.21 error, P2b) | P2b, P2 `geom_split_rel` |
| 4 | per-cell Gordon-Hall blend of four edge curves, C0 across cells, identity where no curve, periodic | **CONFIRMED** (areas exact to 1e-15, P3/P4), with one unavoidable property: `det J = 0` at the four turning vertices of every closed smooth curve of grid lines -- MEASURED HARMLESS (quadrature 2.8e-8, smooth-field film spectral to 3.9e-14, circle Bloch modes spectral) | P3 quad, P3 film, P3/P4 modes |
| 5 | shape layer (`fillet_rect`, `circle` / `ellipse`, `sinusoidal_wall`) generating walls + edge curves | **CONFIRMED feasible** (the P3 / P4 map builders are exactly such generators); design constraints: every change of curve type (arc to line) must be a grid vertex; several shapes in one cell need disjoint curved macro-cells; a fillet's own segment is `r / sqrt 2` wide and meets the 1e-3-of-period sliver contract below `r = 1.414e-3 p` | P4, section 4.3 |

---

## 1. What exists today, and exactly where a map would enter

All line numbers are `4ec402bc`.

### 1.1 `lumenairy/elements/pmm/twod_staggered.py`

| site | lines | what it does today | what a map changes |
|---|---|---|---|
| module docstring, "Scope / limitations" | 168-176 | "Axis-aligned RECTANGULAR pillars only ... CURVED boundaries need Granet's transfinite curved-quad mapping (not implemented)"; "Corner-capped" | rewrite in Phase C; the corner-cap note gains the rim-edge finding of P4 |
| `Basis1D` | 1302-1583 | per-axis 1-D modified-Legendre sets `Btilde` (continuous) / `B` (broken), affine per segment, Bloch `tau` glue | **unchanged** -- it lives in `(u, v)`.  The de Rham property `d(Btilde) in span(B)` is a statement in `(u, v)` and holds for any map |
| `_global_matrix`, `_global_pair_segmat` | 1495, 1586 | per-segment 1-D mass / stiffness / mixed matrices, scaled by the affine `J_n` | unchanged; the quadrature path needs the per-segment VALUES of the local functions at Gauss nodes instead of these integrated matrices |
| `_stag_axis_masses`, `StagGridOps`, `StagCrossOps`, `_stag_cross_mass_1d`, `_stag_kron_apply` | 1611-1830 | the per-layer mortar's factored Kronecker masses | unchanged while the map is stack-wide (every layer shares `(u, v)`); a per-layer map would need a NON-separable cross-mass (Phase E) |
| `Granet2DTransverseE.__init__` | 1875-1990 | dispatch: scalar / tensor / magnetic / slant / out-of-plane | gains `cmap=None`; `None` = today's bytes; a map forces the in-plane magnetic + tensor route |
| `_axis_mats` | 1992 | the 1-D masses and directed derivatives | unchanged (metric-free) |
| `_eps_weighted` | 2009 | `sum_{sx,sy} w[sx,sy] kron(Gy[sy], Gx[sx])` -- the piecewise-constant weighted mass | **the central replacement**: a 2-D Gauss-Legendre quadrature per cell with a weight that varies inside the cell (`_stag_quad_weighted`, section 4.1) |
| `_chi_maps` | 2034 | pointwise 2x2 inverse of a piecewise-constant `mu` | becomes "pointwise at the quadrature nodes" (`chi_t = g / sqrt(g)`, `chi33 = 1/sqrt(g)` for vacuum) |
| `_assemble` | 2055-2327 | builds `R = C[chi_t]C`, `[eps_t]`, `S_tt = -Curl^H Gw^-1 Gw_chi Gw^-1 Curl`, `K_tz`, `Meps33`, `K_zt`, the Schur term | every weighted call (`R` x4, `[eps_t]` x4, `Gw_chi`, `K_tz` x4, `Meps33`, `K_zt` x4 = 18 blocks) routes to the quadrature helper; `Curl`, `Gw`, the directed derivatives stay |
| `_eps_dir` | 2547 | the weighted DERIVATIVE blocks (`K_tz`, `K_zt`) | same quadrature helper with the derivative on the test (`dL`) or trial (`d`) factor |
| `_assemble_oop` | 2329-2545 | first-order out-of-plane generator: nine weighted `ew(...)` masses (the full 3x3 `eps` tensor), unweighted curl / gradient krons, NO permeability blocks (it eliminates `G3` assuming `mu = 1`) | **Phase E only**: a map makes `mu' != 1`, so the out-of-plane generator needs permeability blocks it does not have |
| `_stag_fourier_projection`, `_far_projector_2d`, `_pmm2d_project_orders` | 2651, 2724, 2752 | SEPARABLE Rayleigh projector `kron(Ty, Tx)` per component (E1 -> Ex, E2 -> Ey) | replaced, when a map is present, by the pulled-back 2-D projector with the four cofactor blocks (`Eu -> Ex`, `Ev -> Ex`, `Eu -> Ey`, `Ev -> Ey`) -- P1 shows the identity map reproduces the shipped projector to 1.2e-15 |
| `_region_modes` | 2782-2857 | `eig(L, -R)` (QZ), forward branch, Eq.-25 H partner through `inv(-R)` (nonmagnetic) or the retained plain Gram `Ggram_blocks` (magnetic) | a mapped region is magnetic, so the plain-Gram branch (lines 2844-2851) is the one that must run -- trap H, P2b |
| `_homog_geom_cache`, `_homog_region_modes` | 3295, 3344 | the shared eps-free geometric eig for every uniform scalar region; H partner through `Ginv = inv(-R)` (line 3336) | the SPLIT survives a map exactly (`-R == [eps'_t]/eps`, `geom_split_rel` <= 3.1e-16), but the H partner must switch to the plain Gram -- this is the realistic trap (`Hmix`, 0.053-0.106 error, P2b) |
| `_stag_parity_gauge`, `_stag_block_eig` | 2931, 3019 | out-of-plane parity reduction, structure verified on the assembled pencil | out-of-plane only; untouched by Phases A-D |
| `_slant_congruence`, `_slant_rot_gauge` | 666, 700 | the constant shear's pointwise congruence | Phase E (slant x curved composes two Jacobians and needs the out-of-plane `mu` blocks) |
| `_validate_stag_cost`, `_MAX_STAG_PENCIL_DOF` | 491, 424 | the pencil-size guard | unchanged; shape primitives add walls, so their cost must be priced through it |
| `_STAG_MIN_SEG_FRAC` (sliver contract) | 1008 | refuses a segment narrower than 1e-3 of the period | the fillet primitive's `r / sqrt 2` segment walks into it (section 4.3) |
| `pmm_efficiency_2d_staggered` | 3372-3700 | single layer on an INTEGER uniform grid only | Phase C: raises on `shapes=` / `cmap=` with a pointer to the Jones entry (section 7, Q5) |
| `pmm_jones_2d_staggered` | 3705-3896 | builds a one-layer `PMM2DStackPure` and solves it | inherits the map through the stack |

### 1.2 `lumenairy/elements/pmm/stack2d_pure.py`

| site | lines | what a map changes |
|---|---|---|
| module docstring, "Scope" | 36-60 | rewrite in Phase C |
| `PMM2DStackPure.__init__` / `add_layer` | 731, 795-999 | `add_layer(..., shapes=...)`; the stack collects every layer's shapes into ONE map at `solve()` |
| `solve()` shared-grid path | 1776-2075 | the homogeneous solver `sol_h` (line 1831) and `_homog_geom_cache` are built on the map; `G_gram = (-sol_h.Rmat)` (line 1839) must become the PLAIN block Gram (the z-flux in the mapped frame is `INT (E'_u H'_v* - E'_v H'_u*) du dv`, metric-free); the far field (line 1973) takes the pulled-back projector |
| `_solve_per_layer` | 2120-2446 | mortar path; a stack-wide map keeps its separable `(u, v)` cross-masses valid provided every layer's grid contains the map's kink lines (the continuity is weak in the plain `du dv` product, which is also the metric-free flux form); different maps per layer do not (Phase E) |
| `_flux_at`, `layer_absorption` | 2473, 2517 | formula unchanged; it must be handed the plain Gram (see above) |

### 1.3 The 1-D precedent: what transfers and what does not

`lumenairy/elements/rcwa/oned.py` lines 150-260 (`_asr_metric_profile`,
`_asr_convolutions`) ship Granet's sine-stretch ("adaptive spatial
resolution") for the 1-D Fourier RCWA, with the load-bearing note that the
COVARIANT, metric-multiplied form (`[[f eps]]`) "converges to the WRONG value
at high N while still being bit-exact at eta = 0".  That is a Fourier
factorization defect: products of discontinuous functions do not commute with
truncation in a Fourier basis (the Li rules).  In this basis the walls sit on
element boundaries and every product is integrated exactly per element, so the
covariant form is simply the Galerkin form -- and P2 measures it converging to
the right value (section 3.2).  The 1-D ASR's other lesson DOES transfer: the
bridge between the stretched layer basis and the physical half-space basis
must run in the right direction; here that bridge is the far-field cofactor,
and P2b measures what the wrong one costs.

---

## 2. The formulation

### 2.1 The map and the frame

`(x, y) = Phi(u, v)`, `z = w`, one map for the whole stack.  `J = [[x_u,
x_v], [y_u, y_v]]`, `g = J^T J`, `sqrt(g) = det J > 0` except at isolated
vertices (2.6).  In the `(u, v, w)` frame the fields are the covariant
components `E' = J^T E`, `H' = J^T H` (component `E'_u = E . dPhi/du`, the
field ALONG the `u` grid line), and Maxwell's curl equations keep their
Cartesian form with

    eps'_t  = eps sqrt(g) g^-1 = (eps / sqrt g) [[g22, -g12], [-g12, g11]]
    eps'_33 = eps sqrt(g)
    mu'_t   =     sqrt(g) g^-1          ->  chi_t = [mu'_t]^-1 = g / sqrt(g)
    mu'_33  =     sqrt(g)               ->  chi33 = 1 / sqrt(g)

(Weiss et al., Opt. Express 17, 8051 (2009), Eqs. 7-12; Essig & Busch,
Opt. Express 18, 23258 (2010), Eqs. 5-8 -- read from the local copies in the
`PMM_Papers` folder).  Because the map does not depend on `z`, the tensors
stay BLOCK-FORM (`e13 = e23 = 0`), so the second-order `2 q^2` pencil of
Granet's Eqs. 23-25 still applies.  With a block-form tensor `eps` or `mu`
the same congruence `sqrt(g) J^-1 eps J^-T` applies pointwise (Phase D).

### 2.2 Which operators carry the geometry

Kreeft, Palha & Gerritsma (arXiv:1111.4304, Prop. 8 and the Coda) prove the
general statement for mimetic spectral elements: the topological
(incidence) matrices are invariant under a map; the metric enters only the
Hodge-star / mass matrices.  In `_assemble` that reads:

* metric-FREE (unchanged): the curl `Curl = [kron(dbt_y, Mbb_x),
  -kron(Mbb_y, dbt_x)]`, the `Vw` Gram `Gw`, the directed derivatives
  `dbt`, the Bloch glue, the pencil dimension;
* metric-WEIGHTED (replaced by quadrature): `R` (four blocks, weights
  `chi22, chi21, chi12, chi11`), `[eps_t]` (four blocks -- the two mixed ones
  are nonzero for a SCALAR material because `g12 != 0`), `Meps33`
  (`eps sqrt g`), `Gw_chi` (`chi33`), `K_tz` (four terms, `chi_t`), `K_zt`
  (four terms, `eps'_t`) -- 18 blocks.

The derivative blocks are why "masses" undersells it: `K_tz` and `K_zt` are
weighted DERIVATIVE operators (today's `_eps_dir`), so the quadrature helper
must carry the three flavours `m` (mass), `d` (derivative on the trial
function) and `dL` (on the test function).

### 2.3 Why the staggered C0 basis is the right space, and what is continuous

Across a grid line `v = const`, `dPhi/du` is the TANGENT of that shared edge
curve, identical from both sides because the map is C0 with shared edge
curves; so `E'_u = E . dPhi/du` is the tangential electric field and is
continuous across `v = const` lines, and may jump across `u = const` lines.
That is exactly the staggered placement `E1 in B(u) (x) Btilde(v)` (broken in
`u`, continuous in `v`), and symmetrically for `E'_v`, `H'_u`, `H'_v`.  The
normal flux densities `D'^u`, `B'^u` are continuous across `u = const` lines
and are carried weakly by the Galerkin form, as today.  Kreeft et al.
(Remark 32) state the same placement for 1-forms on curvilinear quads.  So
nothing about the basis changes: the map is C0, `J` may jump across a grid
line, and the covariant components are exactly the conforming unknowns.

### 2.4 The two identities that keep the shipped machinery

* **The shared geometric eig survives.**  For an isotropic homogeneous
  region, `det chi_t = det g / g = 1`, so `-R = adj(chi_t)^T = chi_t^-1 =
  sqrt(g) g^-1`, which is exactly `[eps'_t] / eps`; and `K_zt`, `Meps33`
  both scale with `eps`, so the Schur term is eps-free.  Hence
  `L(eps) = eps (-R) + L0` and ONE eig still serves every uniform isotropic
  layer and both half-spaces.  MEASURED: `max|[eps'_t]/eps - (-R)| / max|R|`
  = 0 (identity), <= 1.0e-16 (stretch), <= 3.1e-16 (circle and fillet maps)
  (`p2_stretch_*.json`, `p3_circle_*.json`, `p4_fillet_*.json`, field
  `diag.geom_split_rel`).
* **The H partner needs the PLAIN Gram.**  Eq. 25 (`gamma C [H1; H2] =
  [k^2 eps_t + S_tt][E1; E2]`) is tested with the plain `(u, v)` inner
  product; the pencil's `-R` is a different operator once `chi_t != I`.  The
  module's "THE ONE TRAP" paragraph already says this for magnetic
  materials; a map makes EVERY region magnetic, so `_homog_region_modes`
  (line 3336, `inv(-R)`) and the stack's retained flux Gram (line 1839,
  `-sol_h.Rmat`) must both switch.  The trap is subtle: if EVERY region
  makes the same mistake the R/T are unchanged (the same geometry-only
  operator multiplies every H partner -- measured 8.3e-16 and 1.1e-14,
  `p2b_failbefore.json`, arm `trap_H_minusR_gram`); it bites when the
  helpers MIX -- a patterned layer on the magnetic branch and half-spaces on
  `_homog_region_modes` -- which is exactly what reusing the shipped code
  unchanged would do (0.106 / 0.053 error, closure 0.157 / 0.133, arm
  `trap_Hmix_halfspaces_minusR`), and in `layer_absorption` whenever the
  flux Gram is `-R`.

### 2.5 The far field and the incident wave

The shipped kernel extracts order `m` as `a_m = (1/A) INT E(x, y)
exp(+i k_m . r) dx dy` (sign fixed by the order-mirror note in
`_stag_fourier_projection`).
Pulled back:

    a_m = (1/A) INT INT cof(J) E'(u, v) exp(+i k_m . Phi(u, v)) du dv,
    cof(J) = det J * J^-T = [[ y_v, -y_u], [-x_v, x_u]]

-- a NON-separable 2-D quadrature with four blocks, of which the two
off-diagonal ones vanish only for an axis-separable map.  The incident-wave
overlap (`_guarded_lstsq` on `Hsup`) inherits it with no further change, and
so does the per-order `rz`/`tz` flux formula (it runs in the physical
half-space).  The identity map reproduces `_far_projector_2d` to 1.2e-15 with
the off-diagonal blocks exactly 0 (`p1_identity.json`).  Dropping the
cofactor is loud: 0.215 / 0.064 on R/T, closure 0.417 / 0.394 (arms
`trap_F_no_cofactor` and `trap_J_no_area_element` -- the latter pulls the
field back correctly but forgets that `dx dy = det J du dv`).

The quadrature for this kernel is oscillatory (`k_m . Phi`); P3 used
`2 M + 16` nodes per axis per cell.  Phase B must size it by the shipped
per-segment phase rule (`_stag_quad_order`, line 2632) applied to the
physical extent of each curved cell.

### 2.6 Smoothness, periodicity, and the corners of the map

* **Inside each `(u, v)` cell the map must be analytic** (arcs, lines,
  sinusoids and their bilinear blends are), so that the weights are smooth
  and the per-cell Gauss quadrature is spectrally accurate.  Every point
  where the boundary curve changes type -- the tangency point where a fillet
  arc meets a straight side, where the curvature jumps -- must be a GRID
  VERTEX.  That is why P4's fillet uses a 5 x 5 grid (walls through the
  45-degree fillet points AND through the tangency points), and why the
  Weiss 2009 circle map's kinks "running across the whole cell" become
  element walls here.
* **Across a cell edge the map need only be C0.**  `J` may jump; the
  covariant placement of 2.3 absorbs it.
* **Periodicity.**  The primitives make the map the IDENTITY on the cell
  boundary, so `Phi(u + p_x, v) = Phi(u, v) + (p_x, 0)`; the Bloch glue
  `tau = exp(-i alpha0 p)` is then unchanged, and the far-field kernel
  carries `alpha0` exactly as today.  OBLIQUE AND CONICAL INCIDENCE WERE NOT
  PROBED (every probe here is at normal incidence); gate B10 closes that.
* **The singular vertices.**  At a vertex where a closed smooth boundary
  turns 90 degrees in `(u, v)` but not at all in `(x, y)` (the circle's four
  45-degree points), the two edge tangents are antiparallel and `det J = 0`.
  This is unavoidable for any smooth closed curve made of grid lines of a
  TENSOR grid (at least four such vertices); the non-tensor "O-grid"
  topology that avoids it (recommended by the literature digest) is
  incompatible with the global tensor-product basis `V1 = B(u) (x)
  Btilde(v)` and would be a new solver (section 5).  Near such a vertex the
  weights grow like `1/r` -- integrable -- and MEASURED harmless: the default
  quadrature is within 2.8e-8 of a 4x finer one (`p3_circle_quad_M8.json`),
  and a uniform film under the circle map converges SPECTRALLY to the exact
  Fresnel answer (4.0e-6, 1.2e-7, 6.9e-10, 4.7e-12, 3.9e-14 at
  `M = 4 .. 8`, `p3_circle_film.json`).
* **Where an arc meets a straight wall at a genuine corner** (a D-shape, a
  fillet with an unfilleted neighbour), the physical corner has the usual
  field singularity and the map is non-singular there (the parameter corner
  and the physical corner both turn); nothing new happens.

---

## 3. The probes and what they refute or confirm

Fixture of P2-P5 unless stated: lambda = 1, square period 1.2, a dielectric
eps = 4 (n = 2) feature in air, height 0.5, air above, n = 1.45 below,
normal incidence, both polarizations ('te' = E along y, 'tm' = E along x),
the nine orders `|m|, |n| <= 1` (the (+-1, +-1) orders are evanescent in air
and propagate in the substrate).  Errors are `max |dR, dT|` over the nine
orders and both polarizations unless stated.

### 3.1 P1 -- identity: the quadrature assembly reproduces the shipped kron assembly

`p1_identity.py` -> `p1_identity.json`.  3 x 3 centred pillar, integer grid
and a non-uniform wall array, `M = 5, 7`.

| quantity | max relative difference | bit-identical? |
|---|---|---|
| `[eps_t]`, `R`, block Gram vs `-R` | 7.3e-15 | no |
| `S_tt`, Schur, `L` | 5.5e-14 | no |
| pencil eigenvalues (nearest match) | 4.3e-12 | -- |
| far-field projector (P1/P2 blocks); off-diagonal blocks | 1.2e-15 abs; exactly 0 | no |
| full solve R/T vs shipped `pmm_efficiency_2d_staggered` | 8.1e-14 abs | no |

Confirms sketch item 2 at round-off.  It is NOT bit-identical (quadrature
reassociates the sums), so "no map = today's bytes" must be a DISPATCH
(`cmap is None` runs the shipped kron code), not "the identity map through
the new path".

### 3.2 P2 -- a pure 1-D stretch converges to the unmapped answer (the decisive gate)

`p2_stretch.py` -> `p2_stretch_{stripe,pillar}.json`.  Map
`x = u + a sin(2 pi u / p_x)`, `y = v`; walls placed at the PREIMAGES of the
physical walls; at `a = 0.15 p_x` the local stretch `x_u` runs from 0.058 to
1.94, a 33:1 range.  Stripe = y-uniform ridge (no in-plane corners; exact oracle
`pmm_efficiency_1d`, degree 40, self-gap 4.1e-7 TM / 1.2e-11 TE).  Pillar =
rectangular, oracle = the unmapped solve at `M = 10`.

Stripe, TE (the polarization whose 1-D oracle converges spectrally):

| M | a = 0 | a = 0.05 p_x | a = 0.15 p_x |
|---|---|---|---|
| 7 | 6.1e-05 | 9.3e-04 | 1.9e-02 |
| 9 | 1.4e-07 | 1.5e-05 | 3.9e-03 |
| 10 | 1.4e-07 | 9.7e-07 | 3.0e-03 |
| 11 | 2.6e-08 | 6.6e-07 | 3.1e-04 |

Stripe, TM: 1.5e-5 / 1.3e-5 / 6.1e-5 at `M = 11` for `a = 0 / 0.05 / 0.15`
(TM is capped by the ridge-corner singularity in BOTH solvers -- see 3.4).
Pillar, difference to the unmapped `M = 10` answer at `M = 10`: 6.7e-6
(`a = 0.05`), 5.3e-4 (`a = 0.15`, still falling: 4.97e-3 at `M = 7`); the
lossless closure falls to 1.1e-8 (`a = 0.05`) and 2.0e-5 (`a = 0.15`).

Verdict: the covariant map, the variable-coefficient masses, the mapped
half-space eig and the pulled-back projector together converge to the right
answer.  A map is not free, though: it costs resolution where it compresses
(`a = 0.05` lags `a = 0` by one to two rungs; the 33:1 stretch by four or
more).  For the shape layer that is a design rule -- keep `det J` within a
modest range away from the singular vertices (the P3 / P4 maps span
`0 .. 2.3` and `0 .. 2.7`).  Unlike the 1-D Fourier ASR (1.3), the covariant
multiplied form is the RIGHT one here.

**P2b, the fail-before arms** (`p2b_failbefore.py` -> `p2b_failbefore.json`,
`a = 0.05 p_x`, `M = 8`):

| arm | stripe error / closure | pillar error / closure |
|---|---|---|
| correct | 0 / 2.3e-6 | 0 / 1.0e-6 |
| H partner through `-R` in EVERY region | 8.3e-16 / 2.3e-6 | 1.1e-14 / 1.0e-6 |
| H partner mixed: layer plain Gram, half-spaces `-R` | 0.106 / 0.157 | 0.053 / 0.133 |
| far field without the cofactor | 0.215 / 0.417 | 0.064 / 0.394 |
| far field `J^-T` without the area element | 0.215 / 0.417 | 0.064 / 0.395 |

### 3.3 P3 -- a circular pillar

`p3_circle.py` -> `p3_circle_{curved3,curved5,quad_M8,film,stair_k*,rcwa}.json`.
Radius 0.36 (0.3 of the period).  **curved3**: 3 x 3 grid, the middle cell is
the disk (Gordon-Hall blend of four 90-degree arcs; all 16 grid vertices
identity-mapped; `det J` in `[0, 2.33]`, 0 at the four 45-degree points).
**curved5**: 5 x 5, the disk is the inner 3 x 3 block with a straight-edged
centre cell (`det J` in `[0, 2.71]`).  Mapped disk area = `pi r^2` to 5e-16
relative in both.

Efficiency ladder (successive differences `d(M -> M+1)`, and the lossless
closure):

| M (curved3) | dof | d(M -> M+1) | closure |
|---|---|---|---|
| 6 | 450 | 5.8e-3 | 1.0e-3 |
| 7 | 648 | 3.6e-4 | 3.0e-5 |
| 8 | 882 | 4.5e-4 | 2.6e-5 |
| 9 | 1152 | 1.3e-5 | 6.7e-7 |
| 10 | 1458 | 1.1e-5 | 2.7e-7 |
| 11 | 1800 | 6.8e-6 | 5.7e-9 |
| 12 | 2178 | -- | 1.7e-9 |

curved5: 3.8e-4, 4.1e-5, 2.2e-5, 1.1e-5 for `M = 4 -> 8` (dof 450 -> 2450),
closure 2.9e-12 at `M = 8`.  The two INDEPENDENT topologies agree to 9.3e-6
(curved3 `M = 12` vs curved5 `M = 8`).  Converged values (curved3,
`M = 12`): R00 = 0.023277, T00 = 0.153821, T(+-1, 0) = 0.101005,
T(0, +-1) = 0.130499, T(+-1, +-1) = 0.077408, sum R = 0.073540.

The discretization respects the circle's four-fold symmetry:
`te(m, n) = tm(n, m)` to 6.7e-12 (curved3, `M = 12`) and 2.1e-12 (curved5,
`M = 8`).

Distances of the other families to the curved3 top rung:

| family | setting | distance | wall time |
|---|---|---|---|
| curved5 | `M = 8` (dof 2450) | 9.3e-6 | 416 s |
| shipped RCWA, EXACT disk form factor (Laurent) | 9 / 13 / 17 / 21 / 25 / 29 orders per axis | 1.1e-2 / 7.8e-3 / 5.9e-3 / 4.6e-3 / 3.8e-3 / 3.3e-3 | 160 s at 29 x 29 |
| same, Richardson-extrapolated assuming error ~ 1/N | pairs (4,6) .. (12,14) | 2.4e-3 / 6.6e-4 / 4.2e-4 / 2.6e-4 / 1.8e-4 | -- |
| shipped RCWA, pixel map, Li rule | S = 128 / 256 / 512 pixels, 25 orders per axis | 2.9e-3 / 3.6e-3 / 3.2e-3 | 25-31 s |
| shipped staggered PMM on a staircase | 4 steps (k = 1, `M = 10`) / 8 steps (k = 2, `M = 7`) / 16 steps (k = 4, `M = 4`) | 7.1e-2 / 5.4e-2 / 5.7e-3 | 457 / 661 / 451 s |

The Fourier family converges algebraically (about `1/N`) TOWARD the curved
answer: its Richardson extrapolation closes from 2.4e-3 to 1.8e-4 as the
order pair rises, with no sign of converging anywhere else.  The staircase
family is not usable as a reference at affordable sizes (its areas are 27 %,
-4.5 % and +3.4 % off the disk's, and the 16-step staircase is already a
9 x 9 grid).  The honest oracle is a converged 3-D FEM of the same pillar
(3.3.1), and it lands on the curved answer.

**The rate.**  The circle's efficiency ladder falls about 15-35x per two rungs
between `M = 6` and `M = 10` and then slows to the same 1e-5-per-rung level
as the rectangular pillar's (3.4) -- the efficiencies are capped by the
pillar's RIM (3.4), not by the circle.  The in-plane content converges
spectrally, as the Bloch-mode ladder shows (`p34_modes.py` ->
`p34_modes_*.json`; `d` = rung-to-rung change of the four leading `n_eff^2`):

| cross-section | M: 8 -> 9 | 9 -> 10 | 10 -> 11 | 11 -> 12 |
|---|---|---|---|---|
| circle, 3 x 3 map | 5.8e-4 | 4.0e-5 | 6.5e-6 | 3.3e-7 |
| circle, 5 x 5 map (read at `M` 5 -> 6 / 6 -> 7 / 7 -> 8 / 8 -> 9) | 2.1e-5 | 1.5e-6 | 4.8e-8 | 2.5e-9 |
| square pillar (corner-capped) | 1.5e-5 | 3.6e-5 | 5.9e-6 | 1.6e-5 |
| y-uniform stripe (no corners) | 1.3e-9 | 1.2e-11 | 1.0e-12 | 1.7e-12 |

**The quadrature.**  At `M = 8`, `nq = 2M + 8 = 24` Gauss nodes per axis per
cell is within 2.8e-8 of `nq = 96` (`p3_circle_quad_M8.json`), three decades
under the rung-to-rung change -- the `1/r` weights at the singular vertices
need no special rule.

### 3.3.1 The FEM oracle

Independent 3-D finite-element solve of the same pillar, run for this plan
(`validation/probe_pmm2d_curved/fem/`: `fem_circle.py` builds and solves,
`results.jsonl` holds every run, `summary.json` the best estimate).  NGSolve
6.2.2604 -- the engine DynaMeta's `solve_fem` wraps; DynaMeta's own entry
returns only the zeroth order, so the script writes the same formulation out
(scattered field against the analytic Fresnel background, Nedelec elements,
complex-stretch PML half-spaces, a quarter cell with PMC / PEC mirror walls)
and extracts every order by exact parity-kernel quadrature over three slabs
plus an up/down two-wave fit.  Curved (high-order) geometry elements,
`mesh.Curve(p)`.  Its OWN error bar: three independent high-resolution meshes
(rim-refined 20 nm at p = 4; h x 0.8 with a 30 nm rim at p = 4; the coarse
mesh at p = 6; 371k-413k unknowns) agree to 8.3e-6 on every propagating
order; `R + T - 1` = 2.7e-7; the slab-averaged Poynting flux matches the
order sums to 1e-6.

| order (E along y) | FEM (+- max dev.) | curved3 `M = 12` | curved5 `M = 8` | RCWA exact disk, 29 x 29 | staircase 16 steps (`M = 4`) | staircase 4 steps (`M = 10`) |
|---|---|---|---|---|---|---|
| R (0,0) | 0.023276 (4.9e-6) | 0.023277 | 0.023283 | 0.023866 | 0.025303 | 0.022664 |
| R (+-1,0) | 0.016854 (6.5e-7) | 0.016854 | 0.016855 | 0.017041 | 0.016669 | 0.010349 |
| R (0,+-1) | 0.008277 (1.0e-6) | 0.008278 | 0.008278 | 0.007876 | 0.007092 | 0.002666 |
| T (0,0) | 0.153819 (6.7e-6) | 0.153821 | 0.153812 | 0.151972 | 0.152186 | 0.128410 |
| T (+-1,0) | 0.101004 (8.3e-6) | 0.101005 | 0.101014 | 0.104283 | 0.106304 | 0.115887 |
| T (0,+-1) | 0.130501 (6.4e-6) | 0.130499 | 0.130494 | 0.128022 | 0.124848 | 0.201185 |
| T (+-1,+-1) | 0.077408 (1.2e-6) | 0.077408 | 0.077406 | 0.077429 | 0.078167 | 0.047188 |

Largest per-order distance to the FEM value (`summary.json`, key
`P3.fem_oracle`):

| family | ladder |
|---|---|
| curved3 (dof 450 .. 2178, `M = 6 .. 12`) | 6.2e-3, 8.2e-4, 4.6e-4, 2.8e-5, 1.7e-5, **7.8e-6** (`M = 11`), **1.9e-6** (`M = 12`) |
| curved5 (dof 450 .. 2450, `M = 4 .. 8`) | 4.6e-4, 8.4e-5, 4.3e-5, 2.2e-5, 1.0e-5 |
| RCWA, exact disk, 9 .. 29 orders per axis | 1.1e-2, 7.8e-3, 5.9e-3, 4.6e-3, 3.8e-3, 3.3e-3 |
| RCWA, pixel map (Li), 128 / 256 / 512 px, 25 orders | 2.9e-3 / 3.6e-3 / 3.2e-3 |
| staircase, 4 / 8 / 16 steps | 7.1e-2 / 5.4e-2 / 5.7e-3 (the last at `M = 4` only) |

So the curved solve enters the FEM's own +-8.3e-6 band at `M = 11` (dof
1800) and sits 1.9e-6 from it at `M = 12`; the Fourier engine with the EXACT
disk form factor is 400x further away at a comparable problem size (29 x 29
orders, 160 s).  This is the P3 verdict: CONFIRMED against an independent
engine, per order, to the oracle's own error.

A side finding for the DynaMeta ledger (not this library): DynaMeta's
`solve_fem` on the same quarter cell gives the right zeroth order (R00
0.023335 against the script's 0.023298 at the same PML strength), but its
all-orders Poynting diagnostic reads `R_flux + T_flux = 1.051` on this
diffracting cell (0.0697 + 0.9817, against 0.0735 + 0.9264 slab-aligned;
`fem/dm_check.jsonl`, `fem/flux_mask_probe.py`); root cause not found.

### 3.4 P4 -- fillets

`p4_fillet.py` -> `p4_fillet_r{0,0.05,0.1,0.2}.json` and
`p4_fillet_equal_area_squares.json`.  Square pillar side 0.6, fillet radius
`r = 0, 0.03, 0.06, 0.12` (`r / side = 0, 0.05, 0.1, 0.2`); `r = 0` is the
3 x 3 rectangular cell, `r > 0` the 5 x 5 fillet map (walls through the
45-degree points and the tangency points; mapped area = the exact
`side^2 - (4 - pi) r^2` to 1e-16).

**How much the zeroth order moves** (normal incidence; R00 and T00 are
polarization-independent for these four-fold-symmetric pillars, measured
equal to 1e-12):

| r / side | R00 | T00 | change vs r = 0 (R00 / T00) | same-area SQUARE, `M = 11` (R00 / T00) |
|---|---|---|---|---|
| 0 (`M = 12`) | 0.038608 | 0.215953 | -- | -- |
| 0.05 (`M = 8`) | 0.038345 | 0.216416 | -2.6e-4 / +4.6e-4 | 0.038645 / 0.216591 |
| 0.1 (`M = 8`) | 0.037857 | 0.217681 | -7.5e-4 / +1.7e-3 | 0.038730 / 0.218522 |
| 0.2 (`M = 8`) | 0.036875 | 0.223311 | -1.7e-3 / +7.4e-3 | 0.038934 / 0.225953 |

The same-area square -- the obvious cheap surrogate, shrinking the square
until its area matches -- is NOT a surrogate for a fillet.  It moves R00 the
WRONG way (+3.1e-5, +1.2e-4, +3.2e-4 against the fillet's -2.6e-4, -7.5e-4,
-1.7e-3) and overshoots T00, leaving it 3.0e-4 / 8.7e-4 / 2.1e-3 from the
fillet in R00 and 1.8e-4 / 8.4e-4 / 2.6e-3 in T00 at r/side = 0.05 / 0.1 /
0.2 -- 20 to 260 times the convergence level.  A fillet acts through its
SHAPE, which is what a curved cell exists to represent.

**Does the convergence become spectral once r > 0?**  In-plane, largely yes;
for the diffraction efficiencies, NO -- and the reason is not the fillet:

* The Bloch-mode ladder (`p34_modes_fillet*.json`) at `M` 7 -> 8 -> 9:
  r/side 0.05: 6.3e-6, 3.8e-6; 0.1: 3.7e-6, 1.1e-6; 0.2:
  1.1e-6, 1.8e-7 -- falling, against the square's 5.9e-6 .. 3.6e-5 between
  `M = 8` and `M = 12`, with no downward trend (`p34_modes_rect3.json`).  Smaller fillets converge more slowly (the arc's
  curvature must be resolved inside a cell of size `r / sqrt 2`), and the
  curvature JUMP at the tangency point (a vertex, so it does not spoil the
  quadrature) still leaves a weak field singularity, so the in-plane rate is
  fast but not the clean exponential of the circle.
* The EFFICIENCY ladders all stall near 1e-5 per rung at `M` 7-11: square
  2.1e-5, 2.5e-5, 5.3e-6, 1.1e-5 (`M` 8 -> 12); fillets 1.7e-5, 1.3e-5,
  1.4e-5 (`M` 7 -> 8, r/side 0.05 / 0.1 / 0.2); circle 1.3e-5, 1.1e-5,
  6.8e-6.  The cause is the pillar's TOP and BOTTOM RIM, the edge where the
  flat face meets the side wall, which every finite-height pillar has and no
  in-plane rounding removes.  The y-uniform stripe isolates it: its in-plane
  modes are converged to 1e-12 by `M = 10`, yet its TM efficiencies are still
  1.5e-5 from the exact 1-D oracle at `M = 11`, while its TE efficiencies
  (the polarization with no strong wedge singularity) reach 2.6e-8
  (`p2_stretch_stripe.json`, `a = 0`).  The 1-D PMM needs degree 30-40 on the
  same stripe for the same reason.

So a fillet is a GEOMETRY-FIDELITY feature (it moves T00 by up to 7.4e-3 at
r/side = 0.2, far above the 1e-5 convergence level), not a convergence
accelerator.  What WOULD accelerate the efficiencies is resolving the rim --
an out-of-scope z-direction change (section 5).

### 3.5 P5 -- cost

`p5_cost.py` -> `p5_cost.json`.  One REGION solve (assembly, then the modal
eig and the Eq.-25 H partner) per child process, so each peak working set is
that configuration's alone; BLAS on one thread; the box was 72-99 % busy with
an unrelated job, so every time is an upper bound and only ratios measured
back to back mean anything.

| grid, `M`, pencil | assembly (s): shipped kron / scratch rectangle / scratch curved | modes (s): shipped QZ / scratch curved QZ / scratch curved WHITENED | peak RSS (MB): rectangle / curved |
|---|---|---|---|
| 3 x 3, 6, 450 | 0.09 / 0.14 / 0.12 | 3.7 / 2.8 / 0.5 | 118 / 121 |
| 3 x 3, 8, 882 | 0.43 / 0.62 / 0.55 | 23.0 / 27.3 / 2.5 | 223 / 226 |
| 3 x 3, 10, 1458 | 1.55 / 1.75 / 1.99 | 150.7 / 117.8 / 12.0 | 467 / 471 |
| 5 x 5, 6, 1250 (fillet) | -- / 1.40 / 1.29 | -- / -- / 7.5 (rect.), 7.2 (fillet) | 369 / 368 |
| 5 x 5, 8, 2450 (fillet) | -- / 6.9 / 17.7 | -- / -- / 53.3 (rect.), 60.3 (fillet) | 1173 / 1173 |

What it says:

* **A curved cell costs what a rectangular cell on the same wall grid
  costs.**  Same pencil, same eig time, same memory (to 1-3 %).  The
  variable-coefficient assembly is 1.1-1.6x the shipped kron assembly on
  3 x 3, i.e. 2-4 % of a QZ region solve and 17-24 % of a whitened one.  (The 5 x 5 `M = 8` assembly
  pair, 6.9 s against 17.7 s, does IDENTICAL block work -- the identity map
  runs the same 18 quadrature blocks -- so its spread is the box load; the
  build should still vectorise the transfinite evaluation, which the probe
  does in Python per cell.)
* **The real cost of a shape is the wall grid it needs.**  A circle fits the
  same 3 x 3 grid as a rectangular pillar.  A fillet needs 5 x 5, i.e.
  `(5/3)^2 = 2.8x` the pencil at equal `M` and about 20x the eig time; it
  buys back part of that in the in-plane rate (3.4) but not in the
  rim-capped efficiencies, so at equal accuracy a filleted pillar costs
  roughly the 5 x 5 / 3 x 3 ratio.
* **The eigensolver.**  The pencil's right-hand matrix `-R` is Hermitian
  positive definite in every in-plane region, mapped or not, so a Cholesky
  whitening turns the generalized eig into a standard one: 5.6-10.9x faster
  than QZ at `M = 6 .. 10` on the same pencil (12.6x against the shipped path
  at `M = 10`), with the same answers -- P1's full solve ran through the
  whitened path and matches the shipped QZ solve to 8.1e-14.  All the
  ladders in this document used it.  Adopting it in the library is a separate
  decision (section 7, Q4).

---

## 4. The phased build

Common rules for every phase (from `docs/TESTING_STANDARDS.md` and the
previous campaigns): decisions not readings; every bar derived, dated, with a
measured gap on both sides; every gate two-sided (a fail-before arm through
the shipped code path, or an engineered defect); no test > 40 s, no file
> 3 min single-threaded; grids <= 3 x 3 at `M <= 8` or 5 x 5 at `M <= 5` in
unit tests (the shipped QZ region solve measured 23 s at 3 x 3, `M = 8` on the
loaded box, P5; larger ladders are build-doc measurements, not tests); OMP /
OPENBLAS / MKL pinned at file top; every probe asserts its `lumenairy.__file__`.
Each phase writes `docs/audits/BUILD_PMM2D_CURVED_<PHASE>_<date>.md` with
the measurement tables its bars cite.

### 4.1 Phase A -- the variable-coefficient assembly, the map protocol, the three traps

**Scope.**  Scalar `eps` cells, in-plane only, the STACK-wide map passed as an
object; the only map shipped in this phase is a separable per-axis stretch
(`x = f(u)`, `y = h(v)`), used as the gate.

**Files.**
* `lumenairy/elements/pmm/_curvemap.py` (new): the map protocol
  `geom(sx, sy, U, V) -> X, Y, x_u, x_v, y_u, y_v` on a tensor of Gauss
  nodes; `IdentityMap`; `SeparableStretch`; a content fingerprint (walls +
  curve parameters) for caching; validation (`det J > 0` on every interior
  quadrature node, identity on the cell boundary).
* `twod_staggered.py`: `_stag_quad_weighted(bx, by, xspec, yspec, W)` --
  the 2-D quadrature generalisation of `_eps_weighted` / `_eps_dir` with the
  `m` / `d` / `dL` flavours, restricted to each cell's supported global
  functions (the probe's `_axis_factor` + `blk`); the effective-tensor
  weights (2.1); `Granet2DTransverseE(..., cmap=None)` routing all 18
  weighted blocks through it when a map is present and through the shipped
  kron code when not; `_region_modes` taking the plain-Gram branch for a
  mapped region; a mapped `_homog_geom_cache` that keeps `-R` for the pencil
  and the PLAIN Gram for the H partner; `_far_projector_2d(..., cmap=None)`
  with the cofactor projector.
* `stack2d_pure.py`: the map on the shared-grid `solve()` path (homogeneous
  solver, geometric cache keyed by the map fingerprint, plain flux Gram,
  far field).  Per-layer (`layer_grids='per-layer'`) + map RAISES in this
  phase.
* `tests/unit/test_pmm2d_staggered_curved_a.py`.

**Gates** (bars from this plan's measurements; the build re-measures and
re-dates them):

| gate | claim | oracle / arms | bar and its derivation |
|---|---|---|---|
| A1 | no map = today's bytes | SHA of R/T/Jones and of every operator on the full staggered fixture set vs a `git archive` of the parent | equality; fail-before = the identity map through the quadrature path, which is NOT bit-identical (P1: 5.5e-14) -- proves the hash can see a round-off change |
| A2 | identity map through the quadrature path | the shipped operators | rel. <= 1e-11: 2.3 decades above the measured 5.5e-14 (P1), and a map amplitude of 1e-6 p moves the operators by O(1e-6), 5 decades above the bar |
| A3 | uniform film under a stretch is exact | Airy slab | `p2_stretch_film.json`: TM 4.0e-7 / 5.3e-9 / 1.3e-12 at `M = 4 / 5 / 6` (`a = 0.15 p`), TE <= 1.4e-14 from `M = 4` -> bar 1e-10 at `M = 6`, and the TM decay `M = 4 -> 6` >= 4 decades (measured 5.5); fail-before = the no-cofactor projector |
| A4 | stretch self-consistency on the TE stripe | `pmm_efficiency_1d` (degree 40) | unit test at `a = 0.05 p`, `M = 8`: measured 4.5e-4 against the no-cofactor fail-before's 0.215 -> bar 5e-3 (1.0 decade above, 1.6 below); the build doc carries the ladder to `M = 11` (9.7e-7 at `M = 10`) |
| A5 | the geometric split | `-R` vs `[eps'_t]/eps` | measured <= 3.1e-16 -> bar 1e-12 (3.5 decades); fail-before = a uniform MAGNETIC region, where the split is genuinely broken (measure it) |
| A6 | the H-partner Gram | two-sided: correct arm closure <= 1e-5 at `M = 8`; the MIXED arm must exceed 1e-2 | P2b: 2.3e-6 vs 0.157 |
| A7 | absorption under a map | lossy film under a stretch: `sum layer_absorption == 1 - sum R - sum T`; fail-before = `-R` as the flux Gram | NOT measured in these probes -- the build measures both arms and derives the bar |

**Oracles.**  `pmm_efficiency_1d`, the Airy slab, the unmapped solver.
**Cost model.**  Assembly by quadrature adds `O(N^2 nq^2 M^4)` flops per
block; P5 measures its share (3.5).  The eig is unchanged at equal size.
**Risks and stop conditions.**  Stop if the TE stripe under `a = 0.05 p`
does not reach 1e-5 of the 1-D oracle by `M = 10` (it measured 9.7e-7), or if
A2 cannot be brought under 1e-11 (it measured 5.5e-14).
**Estimate.**  8-10 agent-hours build, 3 verify.

### 4.2 Phase B -- transfinite maps, the circle and fillet gates

**Files.**  `_curvemap.py`: `Line`, `Arc`, `EllipseArc`, `Sinusoid` edge
curves (value + derivative on `[0, 1]`); `TransfiniteMap(u_walls, v_walls,
vertex_images, curved_edges)` with endpoint checks, the boundary-identity
check and `det J > 0` on interior nodes (the singular VERTICES are allowed,
never a singular interior node); `twod_staggered.py`: the far-projector
quadrature sized by the physical extent of each cell (2.5).
`tests/unit/test_pmm2d_staggered_curved_b.py`.

| gate | claim | measured | bar |
|---|---|---|---|
| B1 | geometry exact | disk area 5e-16 rel, cell area 1.3e-15 rel, fillet area exact to 1e-16 | 1e-12 |
| B2 | the map is C0 with shared edges | edge images from both neighbours identical by construction | exact equality of the two evaluations; fail-before = a perturbed vertex image (the constructor must RAISE) |
| B3 | a smooth field under the singular circle map is spectral | film: 4.0e-6 .. 3.9e-14 at `M = 4 .. 8` | error at `M = 7` <= 1e-10 (measured 4.7e-12) AND the `M = 5 -> 7` drop >= 3 decades (measured 4.4); fail-before = the no-cofactor projector |
| B4 | four-fold symmetry of the circle | `te(m, n) - tm(n, m)` 6.7e-12 (`M = 12`) | 1e-9 at `M = 7`; fail-before = an ellipse with semi-axes `r, 1.02 r` (must exceed 1e-4 -- measure it) |
| B5 | two topologies agree | curved3 vs curved5 | build-doc ladder; unit test at curved3 `M = 8` vs curved5 `M = 5`, bar = 3x the measured difference with the measured next-rung change as the lower gap |
| B6 | the circle against an independent engine | FEM (3.3.1) and RCWA Richardson | build doc only (the oracles cost minutes to hours); bar from the oracle's OWN error estimate |
| B7 | quadrature adequacy | 2.8e-8 at `M = 8` | build doc, re-measured at `M = 6` and `M = 10` |
| B8 | fillet continuity in r | R00 moves -2.6e-4 at r/side = 0.05, monotone to 0.2 | the r -> 0 trend must approach the r = 0 value; measure r/side = 0.01, 0.02 (inside the sliver band; see 4.3) |
| B9 | in-plane spectral rate | circle modes 3.3e-7 by `M = 11 -> 12`, square stalls at 1.6e-5 | build doc (the unit-test sizes cannot separate them: at `M = 9 -> 10` both read ~4e-5) |
| B10 | oblique (25 deg) and conical (phi = 40 deg) incidence under the circle map | NOT probed | a uniform film under the map vs the Airy slab / `berreman_jones_1d` (exact) -- the B3 bars and decay claim at each angle; then the circular pillar at 25 deg against the FEM script (`fem/fem_circle.py` needs a Bloch-periodic full cell for that, build-doc only) |

**Risks and stop conditions.**  Stop if B3 does not decay spectrally (the
probe: 4.4 decades from `M = 5` to `M = 7`), or if a fillet with
r/side >= 0.05 has not reached a rung-to-rung change below 1e-4 by `M = 8`
(the probe: 1.3e-5 .. 1.7e-5 at `M = 7 -> 8`).  **Estimate.**  10-12 agent-hours build, 3 verify.

### 4.3 Phase C -- shape primitives and the public API

**Design.**  A shape is a small object that knows its curved MACRO-CELLS
(the P3 / P4 layouts, confined to its own bounding box plus a margin), its
walls, its edge curves and its material.  The stack merges the shapes of
ALL its patterned layers into ONE wall grid and ONE map at `solve()`:

* the union of every shape's walls (and of every rectangular layer's walls)
  is the global wall grid; a wall that crosses another shape's curved
  macro-cell subdivides it, and each sub-cell takes the macro-cell's blend
  evaluated on its sub-rectangle (still analytic inside);
* two shapes whose curved macro-cells OVERLAP raise, naming both (a common
  map for two intersecting curves is Phase E);
* `Nx == Ny` is restored by splitting the widest segment of the shorter
  axis (a legal h-refinement);
* the result is priced through `_validate_stag_cost` before any assembly.

Primitives: `circle(center, radius, eps)`, `ellipse(center, semi_axes,
angle, eps)`, `fillet_rect(center, size, radius, eps)`,
`sinusoidal_wall(x0, amplitude, eps_left, eps_right)` (a full-period
`x = x0 + A sin(2 pi y / p_y)` line).  API: `PMM2DStackPure.add_layer(
thickness, *, shapes=[...], eps_host=...)`; `pmm_jones_2d_staggered(...,
shapes=..., cmap=...)` (`cmap` = an explicit `TransfiniteMap` for power
users); `pmm_efficiency_2d_staggered` raises on either, pointing at the
Jones entry (section 7, Q5).

**The sliver contract.**  A fillet's own segment is `r / sqrt 2` wide, so
`_STAG_MIN_SEG_FRAC` (1e-3 of the period) refuses `r < 1.414e-3 p`.  The
primitive refuses first, naming "use radius=0 (a sharp corner) or a
radius >= ...".  The round-3 degradation band (3e-2 of the period) is a
MORTAR warning and does not fire on a shared grid.

**Gates.**  C1 byte identity of EVERY existing public call (no `shapes`, no
`cmap`) -- the A1 hash over every staggered test fixture; C2 each primitive's
area, perimeter and `det J > 0` interior; C3 the merge refuses overlapping
macro-cells and accepts a 2 x 2 array of circles (same answer as the single
circle with the period halved -- a physical identity, two-sided against a
mis-merged grid); C4 `jones` symmetry for the circle; C5 docs (module
docstrings of both files, `docs/PMM_ROADMAP.md`, CHANGELOG, a cookbook
entry) -- documentation is not re-verified (section 6).
**Estimate.**  8-10 agent-hours build, 2 verify.

### 4.4 Phase D -- anisotropic and magnetic materials under a map

`eps' = sqrt(g) J^-1 eps J^-T` and the same for `mu`, applied pointwise at
the quadrature nodes; block-form in, block-form out (the map is
z-independent).  An out-of-plane `eps` or `mu` under a map RAISES (Phase E).
Gates: a uniform rotated-uniaxial film and a gyrotropic film under the
circle map vs `berreman_jones_1d` (exact; spectral decay as B3); a uniform
MAGNETIC film under the map vs the analytic slab -- here the H-partner Gram
trap is REAL in every arm (a material `mu` makes `-R` differ between
regions), so the fail-before is `-R` as the Eq.-25 Gram, which must now be
visible even without mixing helpers (measure it).  **Estimate.**  4-6
agent-hours build, 2 verify.

### 4.5 Phase E -- approved 2026-10-02 (built after D)

2026-10-02: the maintainer approved all four items below -- out-of-plane
tensors under a map, slant x curved, per-layer maps (the non-separable curved
mortar) and the JAX twin -- to be built after Phase D.  Each gets its own
build and one verifier, like the other phases (section 6).

* **Out-of-plane tensors under a map** (a tilted-director LC in a circular
  cell): the first-order generator eliminates `G3` assuming `mu = 1`
  (`_assemble_oop`, lines 2329-2545), and a map makes `mu' != 1`
  EVERYWHERE, so this needs permeability blocks in that generator -- a
  derivation and a prototype-first campaign like the slant one, not a
  switch.
* **Slant x curved**: the shear composes with the map, and the six slant
  blocks were derived for `mu^{lm} = g^{lm}` of a pure shear; the same
  missing permeability blocks.
* **Different maps per layer**: tangential continuity between two maps is
  `E'_a = J_a^T J_b^-T E'_b`, i.e. a mortar whose cross-mass is a 2 x 2
  weighted, NON-separable integral over the overlap of two curved partitions
  (clipping curved cells) -- the expensive route; the cheap route is a common
  map (the Phase C merge), which covers every stack whose curved features do
  not overlap in plan view.
* **The JAX twin**: none exists for the staggered solver.
* **Parity reduction for symmetric maps**: out-of-plane only; the shipped
  structural check on the assembled pencil already falls back safely.

---

## 5. What is NOT in scope, and why

* **Non-tensor (O-grid / unstructured) topologies.**  They remove the four
  singular vertices, but the global basis `V1 = B(u) (x) Btilde(v)` needs a
  tensor grid; an unstructured mimetic spectral-element basis (Kreeft et
  al.) is a different solver.  The vertices are measured harmless (2.6).
* **Resolving the pillar rim.**  The efficiency cap measured in 3.4 is a
  z-direction (top/bottom edge) singularity; attacking it (graded
  z-slicing, rim-fitted singular functions) is a separate question from
  in-plane curvature.
* **Tapered curved walls** (a cone): a z-DEPENDENT map re-introduces the
  dilation generator the slant campaign ruled out ("a shear is not a
  taper"); the z-staircase of mapped layers stays the route.
* **Automatic (variational) map generation** in the Essig-Busch sense: the
  primitives give exact, analytic maps for the shapes that matter; an
  optimiser adds a failure mode (maps that only approximately follow the
  interface) this basis does not need.
* **A faster eigensolver for every in-plane region** (P5's whitening): a
  separate, byte-changing decision (section 7, Q4).

---

## 6. Verification protocol

One independent verifier per BUILD phase (A, B, C, D), launched after the
build agent's report, working from this plan and the build doc only.  The
verifier re-measures every gate on ITS OWN fixtures (a different period,
radius, contrast and wavelength from the ones above, chosen by the verifier),
re-derives each bar's lower and upper gaps from its own measurements, and
reports agreement or a defect with a reproduction.  Documentation-only
rounds (docstrings, CHANGELOG, the build doc's prose) are NOT re-verified;
the verifier reads them once for factual claims that carry numbers and
re-measures only those.  A defect found by a verifier is fixed by the build
agent and re-checked by the SAME verifier on the failing item only (no
full second round).  The 3-D FEM script (3.3.1, NGSolve,
`fem/fem_circle.py`) is the Phase B verifier's oracle for the circle, on the
verifier's own fixture; if it cannot be run, the verifier states which
oracle replaced it and why.

---

## 7. Open questions for the maintainer

**Q1. Who owns the map -- the stack or the layer?**  Recommendation: the
STACK (one map for every layer, built from the union of all layers' shapes),
raising on overlapping curved macro-cells.  Measured basis: with one map
every interface is a square match and the shared geometric eig survives
exactly (2.4); per-layer maps need a non-separable curved mortar (4.5).

**Q2. Accept four singular vertices per closed curve (tensor grid), or build
a non-tensor basis?**  Recommendation: accept.  A smooth field under the
singular circle map converges spectrally to 3.9e-14; the circle's Bloch modes
converge spectrally (3.3e-7 at `M = 11 -> 12` on 3 x 3, 2.5e-9 on 5 x 5); the
default quadrature is 2.8e-8 from a 4x finer one.

**Q3. Ship fillets as geometry fidelity, knowing they do not speed up the
efficiencies?**  Recommendation: yes, and say so in the docstring.  A
fillet of r/side = 0.2 moves T00 by 7.4e-3 (700x the convergence level);
a same-area square is no substitute (it misses the fillet's R00 by up to
2.1e-3 and its T00 by up to 2.6e-3, and moves R00 the wrong way).  The efficiency convergence is rim-capped at ~1e-5 per rung for
square, filleted and circular pillars alike (3.4).

**Q4. A Cholesky-whitened eig for every in-plane region?**  The pencil's
right-hand matrix is Hermitian positive definite in every in-plane region,
mapped or not, and the whitened standard eig ran 8.8-10.7x faster than the
shipped QZ at the same size (P5).  It changes the bytes of every in-plane
solve (round-off), so it is a separate decision with its own fixture sweep.
Recommendation: yes, as its own PATCH before or after this campaign, not
inside it.

**Q5. Maps on the single-polarization entry `pmm_efficiency_2d_staggered`?**
It takes only integer uniform grids today and its tensor refusal exists for
out-of-plane cells.  Recommendation: no -- raise with a pointer to
`pmm_jones_2d_staggered`, which already delegates to the stack.

**Q6. The fillet's smallest radius.**  Recommendation: refuse
`r < 1.414e-3 p` in the primitive (the sliver contract) with a message
offering `radius=0`; no new constant.

**Q7. Release shape.**  Recommendation: Phases A-C as 5.50.0 (A and B are
internal until C exposes them); Phase D as 5.50.1.  One verifier per phase
(section 6).

---

## Appendix -- the probe files

| file | what it is |
|---|---|
| `_curved_scratch.py` | the scratch mapped solver: maps (identity, sine stretch, Gordon-Hall transfinite; circle 3 x 3 / 5 x 5, fillet 5 x 5 builders), the quadrature assembly, mapped region modes with the plain-Gram H partner, the mapped geometric eig, the cofactor far projector, the single-layer solve |
| `p1_identity.py` / `.json` | P1 |
| `p2_stretch.py` / `p2_stretch_{stripe,pillar,film}.json` | P2 and the A3 film gate |
| `p2b_failbefore.py` / `.json` | P2b, the trap arms |
| `p3_circle.py` / `p3_circle_*.json` | P3: curved3, curved5, quadrature, film, staircases (k = 4 stopped after its `M = 4` rung), RCWA |
| `p34_modes.py` / `p34_modes_*.json` | the in-plane Bloch-mode ladders of P3 / P4 |
| `p4_fillet.py` / `p4_fillet_*.json` | P4 and the equal-area control |
| `p5_cost.py` / `p5_cost.json` | P5 |
| `summarize.py` / `summary.json` | the reductions quoted in section 3 |
| `fem/` | the 3-D FEM oracle of 3.3.1 (`fem_circle.py`, `summarize.py`, `results.jsonl`, `summary.json`, the DynaMeta cross-check `dm_check.py` / `.jsonl`, `flux_mask_probe.py`); NGSolve 6.2.2604, not a lumenairy dependency |

# PLAN -- tapered sidewalls, rounded rims and other height-dependent profiles for the pure staggered 2-D PMM and the 1-D PMM

Date: 2026-10-04.  Status: PLAN (pending the maintainer's answers in
section 11).
Author: Claude Opus 5.5 (model ID `claude-opus-5-5`), planning agent.
Branch: `docs/plan-pmm-z-profiles` on `a0aa054d` (the 5.50.0 release),
worktree `C:/tmp/lum_zplan`.  No library code or test is changed by this
plan.
Target release: 5.51.0 (MINOR -- a new solver capability) for Phases Z1-Z4;
Phases Z5 (rounded rims) and Z6 (JAX twin) later.
Evidence: every number quoted as MEASURED is read from a JSON file in
`validation/probe_pmm_z_profiles/` (the probe scripts sit next to their
outputs; `_zcommon.py` is the one-dimensional scratch solver they share).
The probes ran on tesla-ryzen (Windows 11, CPython 3.14.6, numpy 2.4.4,
scipy 1.17.1), BLAS pinned to two threads on the command line,
`lumenairy.__file__` asserted under this worktree in every script.  The box
was shared with other agents, so every WALL TIME is an upper bound; no
accuracy number depends on the load.  Numbers quoted from earlier campaigns
name their build document.

---

## 0. Summary, and the decisions this plan asks for

### 0.1 Words used throughout

* **Height `z`, and the computational height `w`.**  The solvers stack
  layers along `z`.  Inside a layer whose geometry changes with height, the
  plan uses a computational height `w` with `z = w` (the vertical coordinate
  is NOT distorted by anything in the recommended design; section 2.7 is the
  one place where it would be).
* **A profile** is any change of a feature's in-plane outline with height: a
  **linear taper** (straight sloped sidewalls, the outline shrinking at a
  constant rate), a **curved sidewall** (a smooth, non-linear change), and a
  **rounded rim** (the edge where a sidewall meets the top or bottom face,
  rounded with a radius in the x-z section -- a fillet seen from the side).
* **The maintainer's taper rule** (recorded feedback, corrected twice): every
  feature narrows about its OWN centre and the gaps between features widen
  symmetrically.  A taper is not a slant of the whole layer.
* **A slant** (shipped since 5.45) is a constant sideways shear of a whole
  layer, `x = u + t w`: the cross-section is unchanged and translated with
  height.  Phase E1 of 5.50.0 composed it with the curved in-plane map.
* **The pure staggered 2-D PMM** is `PMM2DStackPure` /
  `pmm_jones_2d_staggered` (`lumenairy/elements/pmm/twod_staggered.py`,
  `stack2d_pure.py`; Granet, JOSA A 40, 652 (2023)).  Its in-plane
  discretisation is a grid of rectangles in coordinates `(u, v)` with `M`
  Legendre functions per axis; since 5.50.0 a **coordinate map**
  `(x, y) = Phi(u, v)` bends that grid so that circles, ellipses, fillets and
  sinusoids are grid lines.  **The 1-D PMM** is `PMMStack` /
  `pmm_efficiency_1d` (`lumenairy/elements/pmm/stack.py`, `_core.py`).
* **A z-dependent map** is `(x, y, z) = (Phi(u, v, w), w)`: at every height
  the in-plane map bends the same `(u, v)` grid onto that height's outlines.
  Its **tilt field** `T(u, v, w) = dPhi/dw` is the sideways velocity of the
  grid lines with height (a 2-vector at every point).  A slant is the special
  case `T = t`, constant.
* **The virtual medium.**  Maxwell's equations written in the coordinates
  `(u, v, w)` keep their Cartesian form if every material (vacuum included)
  is replaced by the effective permittivity and permeability tensors of the
  map (transformation optics, Ward & Pendry 1996).  The solver works in that
  virtual space, where every material boundary of the profiled feature is a
  coordinate surface at fixed `(u, v)` -- vertical in `(u, v, w)` -- but the
  virtual medium's tensors vary with `w`.
* **Slab and slice.**  A **slice** is a z-invariant layer of the PHYSICAL
  cross-section at one height (the staircase of today).  A **slab** is a
  z-invariant piece of the VIRTUAL medium: the map's tensors frozen at one
  height, on the shared `(u, v)` grid.  `ns` counts slices, `K` slabs.
* **Square match / mortar.**  Two layers on the same grid meet in a square
  match (the modal fields are equated coefficient by coefficient).  Two
  layers on different grids or maps need a **mortar** (a weak, projected
  continuity; the 2-D one is `BUILD_PMM2D_STAGGERED_MORTAR_2026_09_11.md`,
  its curved version Phase E2).
* **The out-of-plane (OOP) generator** is the first-order `4 q^2` modal
  problem (`_assemble_oop`) that a slanted or tilted-director layer needs; the
  **in-plane pencil** is the second-order `2 q^2` one.
* **Exponential midpoint rule, Magnus, Richardson.**  Freezing a
  height-varying system at each slab's midpoint and propagating each slab
  exactly is the exponential midpoint rule (second order in the slab
  thickness).  The Magnus expansion (Blanes, Casas, Oteo, Ros, Phys. Rep. 470,
  151 (2009)) gives higher-order versions; Richardson extrapolation combines
  two slab counts, `(4 f(2K) - f(K)) / 3`, to cancel the leading error.
* **Oracle, fail-before, rung**: as in the curved-cell plan
  (`PLAN_PMM2D_CURVED_CELLS_2026_09_26.md` 0.1).  Distances below are the
  largest absolute change over every reflected and transmitted efficiency of
  orders -2..2 ("eff"), or over the complex zeroth-order reflection and
  transmission amplitudes ("amp").

### 0.2 The question and the short answer

After 5.50.0 the pure 2-D PMM represents exact curved outlines in the plane,
anisotropic and magnetic materials, a constant slant, per-layer maps and a
JAX twin, but the vertical direction is a stack of z-invariant layers.  A
taper is a staircase of slices, every rim is a sharp horizontal edge, and a
rounded rim cannot be represented.  The question: what would it take to
represent tapers and curved z-profiles in the 2-D and the 1-D PMM?

The short answer: **a z-dependent covariant map is the same mathematics as
the slanted curved cell that Phase E1 already built, with the constant slant
vector promoted to a tilt FIELD.**  Inside a profiled layer the modal problem
stops being z-invariant and becomes an ordinary differential equation in `w`;
solving it as a stack of frozen slabs of the virtual medium, all on ONE
`(u, v)` grid and glued by square matches, converges at second order in the
slab thickness, needs no mortar, removes the staircase's polarization-
dependent edge artefact, and admits Richardson extrapolation to fourth order.
Two earlier verdicts that called this route blocked are overturned by
measurement (0.5).  Linear tapers and smooth curved sidewalls are buildable
now; rounded rims are not, because the tilt field diverges where a rim turns
horizontal (measured, 3.5).

### 0.3 What the probes say, in one paragraph

A scratch one-dimensional solver (`_zcommon.py`) that implements the
recommended route reproduces the shipped `PMMStack` on a vertical ridge to
6.3e-12 (TE) and 9.3e-7 (TM, corner-limited) (P0); its frozen tapered slab
has a real fundamental and the exact `q -> -q`, `q -> conj(q)` pairing to
2.4e-13, where the earlier M5 prototype's form gives a complex fundamental
(`Im q = 0.066 .. 0.112`) on the same lossless slab (P1).  On two linear
tapers (sidewalls 7.6 and 14.9 degrees, TE and TM) the mapped slabs converge
at second order (error ratio 3.96-4.12 per doubling of `K` from `K = 8`), Richardson
pairs at fourth order in TE (ratio 15-30) and at about order 2.6 in TM
(ratio 5.8-6.2),
and dropping the tilt field converges to a wrong answer (1.3e-2 .. 6.5e-2,
flat in `K`).  The shipped physical staircase converges at second order in TE
but only at FIRST order in TM (ratio 1.84-1.94 per doubling, unchanged by
the degree and the grid mode, so it is the staircase's own), so at 32 slices
the TM staircase is 7.5e-4 from the converged answer where 32 mapped slabs
are 7.3e-5 and the Richardson pair (16, 32) is 7.8e-7 (P2, P2b).  In 2-D one
slab costs one out-of-plane region solve, measured equal to the in-plane
region solve on the same grid (4.5 s against 4.2 s at `M = 6`), while one
interface of today's CURVED staircase costs 106 s at `M = 6` in the Phase E2
mortar (P3).  A rounded rim breaks the route: the tilt field grows like
`(distance to the face)^(-1/2)`, the frozen slab's pencil loses its pairing
(7e-3 at a tilt of 3.9, degree 16) and the cascade's energy closure fails
(P4); a linear taper is clean up to a wall tilt of 1 (45 degrees from
vertical), degraded at 2 and broken at 3 at degree 16 (P5).

### 0.4 The recommendation

**Build route B (frozen slabs of the virtual medium on one `(u, v)` grid,
square-matched) with Richardson extrapolation in the slab count as its
higher-order layer, for linear tapers and smooth curved sidewalls whose wall
tilt stays within the measured bound; in the 1-D PMM (one builder round) and
the 2-D PMM (three to four rounds) in parallel; defer rounded rims to a
research phase.**  The decisions are in section 11, each with a
recommendation and its cost.

### 0.5 The earlier verdicts this plan revisits

| verdict (where) | what it said | status after this plan's probes |
|---|---|---|
| M5 spike, T3-6 blocker 1 (`PMM_M5_2D_FEASIBILITY_2026_08_04.md` S3.6) | the frozen covariant taper pencil is non-normal with a `q -> -conj(q)` symmetry, its fundamental is complex on a lossless cell, no forward/backward selector can classify it; research-class | **RESOLVED -- the blocker was the formulation, not the geometry.**  M5 kept, inside a frozen slab, the term `(dQ_ww/dw) d_w E` that belongs to the slab-to-slab variation (section 2.4).  The conservative frozen slab is a lossless virtual medium with an exactly paired spectrum (P1: 2.4e-13) and the shipped flux selector classifies it; the cascade converges at second order (P2) |
| M5 spike, T3-6 blocker 2 (same, S3.6) | the far field lives in a distorted frame; needs a quadrature | **RESOLVED by 5.50.0**: Phase A's cofactor far-field projector (`_far_projector_mapped`) is exactly that quadrature, under any in-plane map |
| M5 S3.4 note (`validation/m5_covariant_taper.py`, `_cov_pencil` docstring) | the antisymmetrised form "measured to degrade the cascade from second order to first order in K" | **NOT REPRODUCED.**  With virtual-frame interface matching (the flux dual of 2.5) the antisymmetrised form IS the conservative one and is second order (P2).  M5's cascade (`_cov_modes` in the same file) matched an `H_x`-like partner built with the mid-depth metric and the plain `du` mass, which omits each slab's own `X_u` from both the lab derivative and the flux pairing; that is the likeliest cause -- an inference from its code, not re-measured (section 10) |
| slant-metric build (`BUILD_PMM2D_SLANT_METRIC_2026_08_16.md` S1.1) and 1-D audit W9 | "a shear is not a taper"; no slant absorbs a dilation (measured 1.00x) | **STANDS**, and is consistent with this plan: a constant slant cannot represent a taper; a POSITION-DEPENDENT tilt field can, and the composite-frame machinery that carries a constant slant under a map carries it (2.1) |
| curved-cell plan section 5 | "a z-DEPENDENT map re-introduces the dilation generator the slant campaign ruled out; the z-staircase of mapped layers stays the route" | **SUPERSEDED**: the dilation generator is harmless in the conservative form (above); a staircase of mapped slices on DIFFERENT maps (route A for curved outlines) costs a curved mortar per interface (P3: 106 s at `M = 6`) and sits in the mortar's algebraic accuracy class (E2 F-E2-3) |
| `docs/PMM_ROADMAP.md` (tapered sidewalls row) | tapers are a z-staircase on either 2-D engine | the staircase stays the route for rims and for profiles beyond the tilt bound; route B is added beside it |

---

## 1. What exists today

All line numbers are `a0aa054d`.

### 1.1 How a user builds a taper today, and what it is known to cost

| recipe | where | what it does | measured behaviour (source) |
|---|---|---|---|
| `PMMStack.add_tapered_grating` | `lumenairy/elements/pmm/stack.py:2012` | 1-D: `n_slices` vertical slices of a linear duty ramp, midpoint rule, optional `shear` | staircase error `O(1/ns^2)`: 3.82e-3 .. 5.70e-6 at `ns` 8 .. 256 against the RCWA twin at `raster='area'` (W9, docstring); cost `O(ns^3.4)` on the shared union grid (0.30 / 2.97 / 34.1 s at `ns` 4 / 8 / 16); Richardson in `ns` REJECTED (1.0x-1.2x gain on steep, narrow, high-contrast designs) |
| `PMMStack.add_sheared_grating` | `stack.py:2168` | 1-D: the pure-shear special case as ONE exact slanted layer | 8.9x more accurate than `ns = 4` at a fifth of the cost; plateaus on the slant path's wall-normal floor (1.8e-3 .. 1.2e-2) (CHANGELOG 5.31.0, audit W9) |
| `PMMStack.add_tapered_ridges` | `stack.py:2341` | 1-D multi-ridge staircase, each ridge about its own centre | as above |
| `PMMStack(layer_grids='per-layer')` | `stack.py:1699` | each slice on its own grid, L2 mortar between slices | removed the union-grid wall-collision pathology M5 S4 found (tapered stacks silently wrong by up to 10x); 17.8x faster; the mortar remainder ~1e-4 at degree 6 to ~1e-6 at degree 10 (CHANGELOG 5.32.0) |
| `RCWAStack.add_tapered_grating` | `lumenairy/elements/rcwa/stack.py:1845` | the Fourier twin; the cross-package staircase arbiter | its realised staircase geometry equals the PMM one pixel for pixel (W9) |
| `PMM2DStackPure.add_tapered_pillar(s)` | `stack2d_pure.py:1834`, `:1905` | 2-D (since 5.45.0): rectangular pillars, each slice on its own 3-segment non-uniform grid (`q = 3 (M - 1)`), slices joined by the mortar; each pillar tapers about its own centre | requires `layer_grids='per-layer'`; a closing tip walks into the sliver contract at about 250 slices; a per-layer solve can be stationary in one `n_modes` and wrong, hence `convergence_floor` (`stack2d_pure.py:1983`) |
| curved staircase (circles) | per-layer `shapes=` (Phase E2) | one shape layer per slice, each on its own map, the curved mortar between | accuracy class of an own-walls-only mortar where an outline cuts the neighbour's cells: ~1e-2 at `M = 6`, ~1e-3 at `M = 8 .. 10` (E2-4); concentric circles a few 1e-4 of the period apart push the cut-cell rule to its 96-node cap and 1e-6 apart is refused (E2-5); cost P3 below |
| exact-disk RCWA z-staircase | `RCWAStack` with disk `shapes=` | the arbiter E1-6(c) used for a slanted circle | cleanly second order in `1 / N_z` after Richardson in the order count; 2.2e-4 .. 2.6e-4 floor (E1-6(c)) |

Two things from this record matter for the design.  The staircase of a
TE-like field converges at second order (W9), but no record measured the
TM-like field separately on a PMM-only ladder; P2 does, and finds first
order.  And nothing ever measured the curved staircase of the curved solver
itself (E1 section 7: "needs per-layer maps"); P3 prices one interface.

### 1.2 The 2-D solver: where a z-dependent map enters

| site | lines | what it does today | what a profile changes |
|---|---|---|---|
| module docstring, SLANT / SLANT x MAP sections | `twod_staggered.py:258-333` | the shear's congruence and the composite frame `x = Phi(u, v) + t w` | gains a PROFILE section (Z2) |
| `_slant_congruence` | `:743` | `A^-1 eps A^-T` for a CONSTANT shear, per cell | applied per quadrature NODE with the local tilt `T(u, v)` (the material tensor is constant per cell, the tilt is not) |
| `_slant_rot_gauge` | `:777` | the `_OOP_ROT_SIGN` flip of the OOP entries and of the slant vector | the same flip on the tilt FIELD (risk R3: a centred symmetric taper makes it invisible) |
| `_stag_map_eff_tensor(..., oop=, tvec=)` | `:2222` | the map's congruence, the OOP entries, the composite slant weights | `tvec` becomes a node array (the tilt field); the float path stays byte-identical |
| `_stag_map_slant_weights` | `:2319` | `tau = adj(J) t / sg`, `kappa = chi_t tau` from `float(tvec[0])`, `float(tvec[1])` | accepts arrays (the only structural change in the weights) |
| `_stag_map_weights(..., oop=, tvec=)` | `:2343` | effective-tensor weights at every node of every cell | evaluates the z-dependent map at the slab's height and its tilt field at the same nodes |
| `Granet2DTransverseE._assemble_oop` / `_assemble_oop_general` | `:3421` / `:3664` | the `4 q^2` generator with the permeability blocks; the slant through node weights `t1 t2 k1 k2` and the `'dL'` derivative-on-test device | **unchanged in structure**: `_assemble_oop_general` already reads node-varying `t1 t2 k1 k2` (lines 3730-3757) because Phase E1's `tau = J^-1 t` varies across a mapped cell |
| `_region_modes_oop` | `:4758` | whitened standard eig, flux split | one call per slab |
| `_stag_parity_gauge` | `:4511` | the parity accelerator | refused under a map (F-E1-5); a centred cone gets no symmetry speed-up |
| `_far_projector_mapped`, `_stag_incident_coeffs_mapped` | `:4008`, `:4267` | the cofactor far field and the incident decomposition under a map | called with the FACE maps of the profiled layer |
| `_validate_stag_cost`, `_STAG_MIN_SEG_FRAC` | `:568`, `:1089` | pencil-size guard; the 1e-3-of-period sliver contract | priced per slab; the contract checked at every slab height |
| `PMM2DStackPure.add_layer(..., slant=, shapes=, cmap=)` | `stack2d_pure.py:1093` | records a layer | gains the profile and slab-count keywords (section 5) |
| `_check_stack_slant` | `stack2d_pure.py:507` | slant validation | the tilt bound (risk R6) |
| `PMM2DStackPure._solve_per_layer` | `stack2d_pure.py:3109` | per-layer grids and maps, mortars, end-grid half-spaces, the generalized cascade | a profiled layer expands into `K` slabs on ONE grid with square matches; its neighbours ride its face maps |
| `_same_map_on_overlap` | `_curvemortar.py:948` | detects two grids carrying the same map on an overlap | the face-map test that keeps a profiled layer's boundary a square match |
| `curved_cross_mass` | `_curvemortar.py:1037` | the curved mortar's cut-cell quadrature | used only where a profiled layer meets a patterned neighbour on a different map |
| `CellMap.geom` (map protocol), `TransfiniteMap` | `_curvemap.py:225`, `:806` | in-plane maps on a frozen `(u, v)` grid | a z-dependent map = a family of these on one grid, plus its w-derivative (section 5) |
| `Shape2D` and the primitives `Rect`, `FilletRect`, `Circle`, `Ellipse`; `compile_shapes`, `_merge` | `shapes2d.py:292`, `:399`, `:485`, `:661`, `:799`, `:1523`, `:1264` | the shape layer and the stack-wide merge | a profile per shape; the merge decided once on the reference outline, checked at every slab height |
| JAX twin refusals | `_jax_twod_staggered.py:397-422`, `:965` | OOP, slant and per-layer stacks have no twin | a profiled layer is OOP: refused until an OOP twin exists (Z6) |

### 1.3 The 1-D solver

| site | lines | what it does |
|---|---|---|
| `_build_sem_slant`, `_sem_modes_slant` | `_core.py:6145`, `:6201` | the scalar slant: periodic C0 spectral elements on `u = x - tan(phi) z`, the quadratic pencil `(A1 - q Ac - q^2 A2)` with `Ac = (2 i t / k0) C` |
| `_pmm_slant_solve` | `:6387` | the scalar slanted single-layer solve |
| `_build_generator_metric`, `_layer_modes_metric` | `:6654`, `:6905` | the `4n` first-order generator for tensors; the slant enters as an exact convection `tan * d/dx` on the slant-0 base (the inline note at `:6676` explains why the static fold was abandoned) |
| `_cov_generator_4n`, `_cov_layer_4n` | `:7143`, `:7246` | the covariant Li-1999 oblique-coordinate path (comment block at `:7046`) |
| `PMMStack.add_layer(..., slant_angle=)` | `stack.py:1895` | mixes vertical and slanted layers in one stack through the general forward/backward S-matrix |
| the M5 prototype | `validation/m5_covariant_taper.py` | the scalar taper pencil whose form P1 re-measures |

Li 1999 here is L. Li, "Oblique-coordinate-system-based Chandezon method for
modeling one-dimensionally periodic, multilayer, inhomogeneous, anisotropic
gratings," JOSA A 16, 2521 (1999) -- the coordinate transformation that makes
a corrugated interface a coordinate line, which the project memory calls the
out-of-plane PMM reference.  Its idea is the one section 2.7 needs for rims.

### 1.4 The JAX twin

`StagJaxTwin` (`_jax_twod_staggered.py:371`) freezes the `(u, v)` grid at a
reference shape and lets traced parameters move the vertex images and edge
curves (`BUILD_PMM2D_CURVED_E3_2026_10_03.md` 2.2); a z-dependent map is
exactly such a family with `w` as the parameter, so the frozen-grid design
carries over.  But the twin refuses every OOP or slanted layer (no twin of
the first-order generator or of the generalized cascade, E3 section 5), and
every profiled slab is OOP.  The degenerate-cluster rule (E3 section 9.1)
wraps each eig problem and its consumer; a centred cone at normal incidence
has degenerate pairs in EVERY slab, so the rule's lifted evaluations are
paid `K` times.

---

## 2. The physics

### 2.1 A z-dependent covariant map, and why it is Phase E1's composite frame

Let `(x, y, z) = (Phi(u, v, w), w)`.  The 3-D Jacobian is

    Lambda = d(x, y, z) / d(u, v, w) = [[J, T], [0, 1]],
    J = d Phi / d(u, v)   (the in-plane Jacobian at height w),
    T = d Phi / d w       (the tilt field),          det Lambda = det J = sg.

It factors pointwise as `Lambda = A(T) Lambda_m` with the shear `A(T) =
[[I, T], [0, 1]]` and `Lambda_m = blockdiag(J, 1)` -- exactly the factorisation
of the slanted curved cell in `BUILD_PMM2D_CURVED_E1_2026_10_03.md` 1.2, with
the constant `t` replaced by the local `T(u, v, w)`.  Every effective tensor is
therefore the map's congruence of the shear's, at each point:

    eps' = sg Lambda_m^-1 [A^-1 eps A^-T] Lambda_m^-T       (and mu' likewise)

and E1's derivation (1.1-1.3) of the first-order generator never used that
`t` is constant: it is pointwise algebra plus one derivative that is moved
onto the continuous test function.  The generator's weights at a node are

    chi'_tt = J^T chi_t J / sg            (unchanged by the tilt)
    1/mu'33 = 1 / (sg mu_33)              (unchanged by the tilt)
    tau     = J^-1 T = adj(J) T / sg      (the tilt seen in (u, v))
    kappa   = chi'_tt tau
    eps'    = the map's congruence of A(T)^-1 eps A(T)^-T, OOP entries included

What is new is only that `J`, `sg` and `T` now depend on `w` as well, and
that the shear congruence `A(T)^-1 eps A(T)^-T` must be taken per node instead
of per cell.  (Sign convention: E1's `tvec` is the INTERNAL frame shear, the
negative of the public `slant`; a profile's `T` is the frame velocity
`dPhi/dw` itself, so it enters where the internal shear does.)  Phase E1 already measured that laying a constant shear on a
mapped cell is wrong by 7.7e-4 .. 4.4e-2 (`tau_unmapped`, E1-9); the tilt
field is the next step of the same rule.

### 2.2 A linear taper about each feature's own centre: what depends on height

The maintainer's rule makes a taper a SCALING of each feature about its own
centre `c`.  Inside the feature's cells, with a reference in-plane map
`Phi_0(u, v)` (the outline at a reference height) and a scale `s(w)`,

    Phi(u, v, w) = c + s(w) (Phi_0(u, v) - c)
    J = s J_0,   sg = s^2 sg_0,   T = s'(w) (Phi_0 - c)

so the tilt field is RADIAL: zero at the centre, largest on the outline,
where it equals the physical wall tilt (`|T| = tan` of the sidewall angle from
the vertical for a circle).  For a LINEAR taper (`s'` constant), working the
congruence through for an isotropic `eps`:

| weight | dependence on height inside the feature |
|---|---|
| `chi'_tt = J_0^T J_0 / sg_0` (vacuum) | none |
| in-plane `eps'_tt`, both its map part `adj(J_0) eps adj(J_0)^T / sg_0` and its tilt part `eps s'^2 adj(J_0)(Phi_0 - c)(Phi_0 - c)^T adj(J_0)^T / sg_0` | none |
| the physical tilt `T` | none |
| `eps'_33 = s^2 sg_0 eps` | grows as `s^2` |
| `1/mu'33 = 1 / (s^2 sg_0)` | falls as `s^-2` |
| the OOP couplings `e'_t3 = e'_3t^T = -eps s s' adj(J_0)(Phi_0 - c)` | proportional to `s` |
| `tau = (s'/s) J_0^-1 (Phi_0 - c)`, `kappa = chi'_tt tau` | proportional to `1/s` |

So inside the feature the height enters through four scalar powers of `s`;
in the HOST cells around it (the transfinite blend between the moving outline
and the fixed macro-cell boundary, where the gaps widen) every weight depends
on height.  No profile makes the whole cell z-invariant -- the feature shrinks
while the period does not -- so the per-layer modal problem does not survive.

### 2.3 What the layer's modal problem becomes

In a z-invariant layer the field is a sum of modes `x(w) = x_0 exp(i q k0 w)`
with `A x = q B x` (`A`, `B` the generator's matrices, E1 1.3).  With
height-dependent weights the same rows read

    B(w) dx/dw = i k0 A(w) x,

a linear first-order ODE in `w` of dimension `4 q^2` -- the "differential
method" setting of Neviere and Popov (Light Propagation in Periodic Media,
Marcel Dekker 2003), here with a spectral-element transverse basis instead of
a Fourier one.  The modal method's job inside a profiled layer is to
integrate this ODE stably through strongly evanescent components, which an
S-matrix cascade of frozen slabs does exactly (each slab propagates its own
modes analytically).

### 2.4 The conservative frozen slab, and why the M5 prototype had complex modes

In the first-order Maxwell form no derivative of a coefficient appears: the
curl acts on the fields and the tensors multiply them.  Freezing `A(w)`,
`B(w)` at a slab's midpoint therefore gives the generator of a genuine
z-invariant, lossless (for lossless materials) virtual medium; its spectrum
pairs `q` with `conj(q)` (flux conservation) and, for reciprocal materials,
with `-q`, and the shipped flux selector classifies it.

The M5 prototype worked with the second-order scalar equation
`d_i (Q^ij d_j E) + k0^2 m E = 0` (`Q = sqrt(g) g^-1`; in 1-D TE `Q_uu =
(1 + S^2)/X_u`, `Q_uw = -S`, `Q_ww = X_u`, `S = X_w` the 1-D tilt).  There
`d_w (Q_ww d_w E) = Q_ww d_w^2 E + (d_w Q_ww) d_w E`, and for a taper
`d_w Q_ww = S'(u)` does not depend on `w`, so M5 kept it inside the frozen
slab while freezing `Q_ww` itself.  That term is the w-variation of the
coefficient, which a slab cascade represents through the steps between slabs;
keeping it inside each slab as well is a non-conservative splitting, not the
Maxwell operator of any lossless medium.  P1 measures the difference on one
slab (section 3.2): the conservative slab is exactly paired, M5's is not.

### 2.5 Interfaces: inside a profiled layer, and at its faces

In the exact problem the map is continuous in `w`, so on every surface
`w = const` the tangent vectors `dPhi/du`, `dPhi/dv` are the same from above
and below, and the VIRTUAL tangential fields (`E'_u`, `E'_v`, `H'_u`, `H'_v`)
are continuous.  The frozen-slab approximation keeps that: consecutive slabs
share the `(u, v)` grid, so they meet in a square match -- no mortar, no
cross-mass, no projection.  (In 1-D TE the matched pair is the nodal `E` and
the flux dual `<v| Q_wu d_u E + Q_ww d_w E>`, which is `X_u` times the lab
`H_x`: the `dx = X_u du` of the flux pairing is inside it.)

The square match is consistent ONLY together with the tilt.  Without it,
consecutive slabs are the physical slices written on the shared grid, and
gluing them coefficient by coefficient equates fields at different physical
points: P2's "NOTILT" arm converges, as `K` grows, to a wrong answer 1.3e-2 ..
6.5e-2 away (3.3) -- the fail-before every build gate needs.

At the layer's top and bottom faces the map is `Phi(., ., 0)` and
`Phi(., ., h)`.  The half-spaces and any homogeneous neighbour take that face
map (Phase E2's riding rule makes it a square match); a patterned neighbour
on a different map meets it through E2's curved mortar, with E2's accuracy
class.  The far field is Phase A's cofactor projector under the face map.

### 2.6 Four routes to the profiled layer

**(A) The physical staircase** (today).  Each slice is the cross-section at
its mid-height on its own grid; slices meet through a mortar.  Popov,
Neviere, Gralak and Tayeb ("Staircase approximation validity for
arbitrary-shaped gratings," JOSA A 19, 33 (2002); the request for this plan
listed Enoch among the authors -- check the citation before it enters a
docstring) showed that
in TM polarization the staircase converges slowly and non-monotonically
because every step adds right-angle edges whose field singularity the
profile does not have.  Measured here on a 1-D dielectric taper: the shipped
staircase is second order in TE and FIRST order in TM (3.3).  In 2-D every
interface is also a non-conforming mortar cut by the neighbour's outline
(E2 F-E2-3: algebraic in `M`), and for curved outlines one curved mortar per
interface (P3).

**(B) Mapped slabs** (recommended).  Section 2.4-2.5: `K` slabs of the
virtual medium, each frozen at its midpoint, one grid, square matches.  The
exponential midpoint rule is symmetric (reversing `w` reverses the method),
so its error expands in even powers of the slab thickness (Hairer, Lubich,
Wanner, Geometric Numerical Integration, Springer 2006, section II.3): second
order, with Richardson extrapolation reaching fourth.  How to MEASURE the
order in 2-D: a uniform film under a z-dependent map against Berreman, a
pillar ladder in `K` read on its ratio per doubling, and the Richardson pair
read against the next pair (the probes' procedure, 3.1-3.3).

**(C) Higher order in the slab thickness.**  (C1) Richardson in `K` on the
complex amplitudes (cost `K + 2K` slabs; measured fourth order, 3.3).  (C2)
The fourth-order Magnus step with two Gauss points per slab, `Omega =
(h/2)(G_1 + G_2) + (sqrt(3) h^2 / 12) [G_2, G_1]` with `G = B^-1 A` at the
two Gauss heights; `Omega / h` is diagonalised like a frozen generator, so
the slab cascade is unchanged, and because commutators of flux-conserving
generators stay in the same Lie algebra (Iserles and Norsett, Phil. Trans. R.
Soc. A 357, 983 (1999)) its modes still pair under the flux form and the
selector applies.  It costs two generator assemblies and two dense products
of `4 q^2` matrices per slab in place of one assembly.  (C3) A direct ODE
integrator (Runge-Kutta between modal checkpoints, the classical differential
method): the high-order evanescent modes make the system stiff (`|q|` up to
~100 `k0` in the probes), so explicit steps would have to be tiny; not
recommended.  Recommendation: ship C1, measure C2 against it in Phase Z4.

**(D) A polynomial-in-`w` block per layer** (a 3-D spectral element inside
the layer, the half-space modes as boundary conditions).  It replaces the
modal propagation by one linear solve of size `2 q^2 (M_z + 1)` per layer
(about 9.7e3 unknowns at `M = 8`, `M_z = 10`: a dense LU near 1e12 flops and
1.5 GB) and needs its own spurious-mode theory (the staggered de Rham
placement extended to `w`) and its own oracles.  It is a different solver
class; out of scope unless the maintainer wants it.

### 2.7 Rounded rims: the tilt singularity and what would handle it

A rim rounded with radius `R` follows a quarter circle in the x-z section.
Written as a lateral map, the outline's half-width at distance `d = h - w`
below the face is `a_0 - R + sqrt(R^2 - (R - d)^2)`, which behaves like
`a_0 - R + sqrt(2 R d)` near the face, so the tilt field grows like
`d^(-1/2)` and is infinite where the rim meets the face.  `det Lambda = det J` stays positive
as long as the flat top keeps a positive width, so the map does not fold; it
degenerates instead: the `w` coordinate lines become nearly horizontal and
the virtual medium becomes extremely anisotropic.  Two consequences:

* The in-plane tensors carry `eps (1 + |T|^2)` (the shear congruence), whose
  integral over `w` diverges logarithmically; the continuum Schur complement
  of the `E_3` elimination cancels that term exactly (as the slant's `|t|^2`
  cancels, E1 1.2), but whether the DISCRETE Galerkin elimination
  (`E3 = A33^-1 (...)`, `twod_staggered.py:3439`) preserves the cancellation
  for an unbounded, varying tilt is not known (section 10).
* MEASURED in 1-D (P4, P5): the frozen slab's pencil loses its exact pairing
  as tilt times degree grows (degree 16: 1.9e-8 at tilt 1.1, 4.2e-6 at 2.1,
  7.1e-3 at 3.9, 2.4e-2 at 7.95; a first-order linearisation is one to two
  decades better but degrades the same way), and the cascade's lossless
  closure fails (0.37 at `K = 16`).  Grading the slabs toward the rim
  (`w = R sigma^2`, which makes the ODE smooth in `sigma`) reaches the
  singular region sooner and fails earlier.

What would handle a rim is the C-method idea (Chandezon, Dupuis, Cornet,
Maystre, JOSA 72, 839 (1982); Li 1999): a VERTICAL map `z = w + a(u, v)` in
which the rounded cap is a coordinate surface, used where the rim is closer
to horizontal than 45 degrees, and the lateral map below that point -- the
x-z analogue of the in-plane circle's four singular vertices (curved-cell plan 2.6),
with the 45-degree point of the rim as a "z-vertex".  It needs three things
the library does not have: the out-of-plane PERMEABILITY blocks (`mu'^{t3}
!= 0`; E1 F-E1-4 wrote the formulas, not the gates), slabs whose
`w = const` boundaries are curved physical surfaces (so the vertical map
must return to `z = w` inside a buffer before the flat face where the
half-space attaches), and a treatment of the z-vertex.  That is research
(Phase Z5).  Until then a rim is the physical staircase's job: P4 measures
the shipped staircase of a rounded rim converging at roughly first order (TE
successive changes 1.9e-4 .. 7.8e-6 over `ns` 4 .. 128; TM 1.5e-3 .. 2.4e-5),
and the rounding itself moves the efficiencies by 9.0e-3 (TE) / 1.0e-2 (TM)
against a sharp rim -- geometry fidelity three decades above the
convergence level, as the in-plane fillet was (curved-cell plan 3.4).

### 2.8 The 1-D PMM

Every statement above holds in the 1-D PMM with `(u, v)` reduced to `u`: the
scalar TE / TM problems are quadratic pencils of dimension `n` (the number of
nodes, as in the probes), the tensor problem the `4 n` first-order
generator, and the tilt field `S(u, w) = X_w` is piecewise linear across the
ridge for a linear taper.  The probes ARE the 1-D
case (scalar TE and TM), so the formulation is measured there, not only
argued.  For the library the 1-D slant exists in three forms (1.3); route B
generalises the scalar pencil's constant `t` to a nodal field (the probes'
operator, P1) and, for tensors, the metric generator's convection
`tan * Dop` to `Dop * S(u)` in its conservative (antisymmetrised) split.  The
1-D build has the richest oracles in the library: Airy and
`berreman_jones_1d` films under the map, the shipped staircase in both grid
modes, the RCWA twin at `raster='area'`, and this plan's scratch solver.

Should 1-D be built first?  The formulation risk the 1-D build would retire
is already retired by P0-P2 (the scratch solver is a working 1-D route B).
What a 1-D LIBRARY build adds is value on its own -- a TM-polarized 1-D
grating with sloped walls converges at first order on today's staircase and
at second (fourth with Richardson) on route B (3.3) -- and a smaller first
contact between the tilt field and the shipped flux selector and generalized
cascade.  It does not share code with the 2-D build, so the two can run in
parallel (decision 2).

---

## 3. The probes

Fixture of P1-P5 unless stated: a 1-D grating, period 0.8, wavelength 1,
height 0.6, a ridge of `eps = 4` in air, air above (the incidence side, `w = 0`
the top face), `n = 1.45` below, normal incidence, orders -2..2 (0 propagates
in air, -1..1 in the substrate), polynomial degree 16 with one element per
region.  The ridge's duty runs linearly from its top value to its bottom value
and every wall moves symmetrically about the ridge centre (the maintainer's
rule).  "Moderate" taper: duty 0.40 -> 0.60, sidewall 7.59 degrees from the
vertical; "strong": 0.30 -> 0.70, 14.93 degrees.

### 3.1 P0 -- the scratch route-B solver is right where it can be checked exactly

`p0_validate.py` -> `p0_validate.json`.

| check | measured | reading |
|---|---|---|
| V1: vertical ridge (identity map, `K = 1`) against the shipped `PMMStack` at degree 20, every R / T | TE 6.1e-10 / 5.5e-11 / 6.3e-12 at degree 12 / 16 / 20; TM 2.2e-5 / 5.3e-6 / 9.3e-7 | the scratch SEM, half-spaces, cascade and far field agree with the library; TM converges more slowly in BOTH codes (the ridge corners) |
| V2: a pure shear (every wall translates, tilt 0.3) at `K = 1` against `K = 16` | 4.7e-10 (TE eff), 2.6e-10 (amp); TM 7.9e-11 / 2.5e-10 | a z-linear map is z-invariant in the virtual frame: the slab count cannot matter; 2e-10 is the scratch cascade's own round-off floor |
| V3: a uniform film (`eps = 4`) under the strong taper's map against the exact Airy slab | TE `K` = 4 / 8 / 16 / 32 / 64: 1.2e-3 / 2.5e-5 / 7.7e-6 / 2.2e-6 / 5.8e-7; TM 2.5e-3 / 6.2e-4 / 1.6e-4 / 4.0e-5 / 1.0e-5 (ratio 3.9-4.0) | a uniform medium under a z-dependent map is NOT exact at finite `K` (the virtual medium varies with `w`); it converges at second order -- so the Berreman gate of a profiled layer is a LADDER, not a round-off identity |
| V3 fail-befores, same ladder | tilt dropped: 3.2e-2 .. 3.7e-2 (TE), 8.5e-2 .. 9.8e-2 (TM), flat in `K`; M5's term kept: 0.76 .. 2.3e3, non-physical | two-sided: the gate sees both defects by 2.5-4 decades at `K = 32` |

### 3.2 P1 -- the frozen slab's spectrum: conservative form against M5's

`p1_spectrum.py` -> `p1_spectrum.json`.  The strong taper frozen at
`w = 0.1 h, 0.5 h, 0.9 h`; `q` the modal index; residuals relative to
`max |q|`.

| polarization, height | conservative: fundamental `q0`; pairing `-q` / `conj q` | M5 form: `q0`; pairing `-q` / `conj q` / `-conj q` |
|---|---|---|
| TE, 0.1 h | 1.70964 + 0i; 6.6e-14 / 6.6e-14 | 1.69981 + **0.11186i**; 1.9e-3 / 1.9e-3 / 1.6e-14 |
| TE, 0.5 h | 1.82954 + 0i; 5.2e-14 / 5.2e-14 | 1.82422 + **0.08613i**; 1.4e-3 / 1.4e-3 / 1.1e-14 |
| TE, 0.9 h | 1.89503 + 0i; 2.4e-13 / 2.4e-13 | 1.89096 + **0.06600i**; 1.4e-3 / 1.4e-3 / 4.0e-14 |
| TM, 0.1 h | 1.41701 + 0i; 2.1e-13 / 2.1e-13 | 1.39906 + **0.06826i**; 1.8e-3 / 1.8e-3 / 1.3e-14 |
| TM, 0.5 h | 1.68350 + 0i; 6.8e-14 / 6.8e-14 | 1.67588 + **0.08549i**; 1.4e-3 / 1.4e-3 / 3.9e-14 |
| TM, 0.9 h | 1.81548 + 0i; 8.9e-14 / 8.9e-14 | 1.81088 + **0.07013i**; 1.4e-3 / 1.4e-3 / 2.8e-14 |

M5's finding is reproduced on a different device (complex fundamental, only
the `q -> -conj(q)` symmetry) and its cause is isolated: removing the one
term restores both pairings to round-off.

### 3.3 P2 and P2b -- a linear taper: mapped slabs against the physical staircase

`p2_taper_ladder.py` -> `p2_taper_ladder.json`.  Reference: the Richardson
pair (128, 256) of route B; its in-plane check (degree 16 against 20 on the
pair (64, 128)) is 4.9e-11 / 1.2e-9 (TE moderate / strong) and 6.7e-6 / 6.5e-6
(TM), so TM distances below ~1e-5 are at the in-plane level.  STAIR is the
shipped `PMMStack.add_tapered_grating` on `layer_grids='per-layer'` at degree
16, scored on efficiencies only (its amplitude phase reference is not the
scratch solver's).

Route B (`eff / amp`), the Richardson pair `(K, 2K)` (`eff / amp`), the
tilt-dropped arm (`eff`), and the staircase at `ns = K` (`eff`):

| taper, pol. | K | B | B-Richardson (K, 2K) | NOTILT | STAIR (ns = K) |
|---|---|---|---|---|---|
| moderate TE | 4 | 2.2e-3 / 6.3e-3 | 1.4e-4 / 7.3e-4 | 1.4e-2 | 2.7e-3 |
| | 8 | 4.5e-4 / 1.6e-3 | 6.1e-6 / 3.3e-5 | 1.4e-2 | 6.3e-4 |
| | 16 | 1.1e-4 / 4.1e-4 | 3.7e-7 / 2.0e-6 | 1.3e-2 | 1.6e-4 |
| | 32 | 2.7e-5 / 1.0e-4 | 2.5e-8 / 1.3e-7 | 1.3e-2 | 3.9e-5 |
| | 64 | 6.9e-6 / 2.6e-5 | 1.7e-9 / 8.6e-9 | 1.3e-2 | -- |
| moderate TM | 4 | 4.6e-3 / 1.1e-2 | 1.2e-4 / 4.5e-4 | 1.6e-2 | 5.2e-3 |
| | 8 | 1.1e-3 / 2.4e-3 | 6.6e-6 / 2.7e-5 | 1.9e-2 | 2.6e-3 |
| | 16 | 2.9e-4 / 5.9e-4 | 7.8e-7 / 2.4e-6 | 2.0e-2 | 1.4e-3 |
| | 32 | 7.3e-5 / 1.5e-4 | 1.6e-7 / 2.8e-7 | 2.0e-2 | 7.5e-4 |
| | 64 | 1.8e-5 / 3.6e-5 | 2.7e-8 / 3.6e-8 | 2.0e-2 | -- |
| strong TE | 4 | 1.9e-2 / 1.8e-2 | 6.2e-4 / 1.7e-3 | 6.6e-2 | 7.7e-3 |
| | 8 | 4.4e-3 / 4.2e-3 | 2.2e-5 / 7.6e-5 | 5.9e-2 | 1.8e-3 |
| | 16 | 1.1e-3 / 1.0e-3 | 7.1e-7 / 4.5e-6 | 5.7e-2 | 4.5e-4 |
| | 32 | 2.7e-4 / 2.6e-4 | 4.8e-8 / 3.9e-7 | 5.6e-2 | 1.1e-4 |
| | 64 | 6.7e-5 / 6.5e-5 | 1.1e-8 / 3.9e-8 | 5.6e-2 | -- |
| strong TM | 4 | 1.1e-2 / 1.9e-2 | 2.2e-4 / 1.1e-3 | 5.6e-2 | 1.9e-2 |
| | 8 | 2.8e-3 / 4.1e-3 | 1.9e-5 / 5.1e-5 | 6.3e-2 | 9.4e-3 |
| | 16 | 7.2e-4 / 9.9e-4 | 3.0e-6 / 4.2e-6 | 6.4e-2 | 4.9e-3 |
| | 32 | 1.8e-4 / 2.4e-4 | 5.2e-7 / 5.0e-7 | 6.5e-2 | 2.5e-3 |
| | 64 | 4.6e-5 / 6.1e-5 | 8.6e-8 / 8.5e-8 | 6.5e-2 | -- |

The M5-form arm (its frozen slab inside the same cascade) sits at 0.22 /
0.16 / 0.43 / 0.16 (moderate TE / TM, strong TE / TM) flat in `K`, and on the
strong TM taper diverges (2e4 .. 3e8) once its misclassified modes are
propagated.  `K = 1` and `2` are pre-asymptotic on these fixtures (errors
4e-2 .. 2.3e-1); every ladder is asymptotic from `K = 4`.

What it says:

* **Route B is second order in both polarizations**: 3.96-4.12 per doubling
  from `K = 8` and 3.97-4.01 from `K = 16`, every column, amplitudes
  included.
* **Richardson works on route B**: 15-30x per doubling in TE (fourth order,
  until the pair meets the reference's own error); 5.8-6.2x in TM once
  asymptotic (order about 2.6), which this plan reads as the TM field
  singularity at the ridge's rim corners entering the slab error with a
  fractional power -- every finite-height ridge has those corners.  On the
  physical staircase the same extrapolation gained nothing on steep designs
  (W9).
* **The physical staircase is second order in TE and FIRST order in TM**
  (1.84-1.94 per doubling, both tapers).  At equal counts route B is 1.4x
  better to 2.4x worse than the staircase in TE (the staircase wins on the
  strong TE taper) and 2-14x better in TM; with Richardson it is 25-1000x
  better at comparable work (3.3.1 shows the TM first order is the
  staircase's own).
* **The tilt field is load-bearing**: without it the cascade converges, to a
  wrong answer (1.3e-2 .. 6.5e-2).
* Wall time (1-D, `n = 48` nodes): 1.2 s at `K = 64`, 4.5 s at `K = 256`;
  the shipped staircase 2.5 s at `ns = 32`.

#### 3.3.1 P2b -- is the first-order TM staircase the staircase?

`p2b_tm_stair.py` -> `p2b_tm_stair.json`.  The moderate taper; the
shipped staircase on both grid modes and at two degrees, scored against
route B's limit; then the top pair extrapolated at first (`p = 1`:
`2 f(2n) - f(n)`) and second (`p = 2`) order, per component.

| arm | ns = 4 | 8 | 16 | 32 | 64 | p = 1 extrapolation vs route B's limit | p = 2 |
|---|---|---|---|---|---|---|---|
| TM, per-layer, degree 16 | 5.16e-3 | 2.55e-3 | 1.38e-3 | 7.51e-4 | 4.07e-4 | 6.3e-5 | 2.9e-4 |
| TM, per-layer, degree 20 | 5.15e-3 | 2.55e-3 | 1.38e-3 | 7.46e-4 | 4.03e-4 | 6.0e-5 | 2.9e-4 |
| TM, shared union grid, degree 16 | 5.16e-3 | 2.55e-3 | 1.37e-3 | 7.38e-4 | -- | 1.1e-4 (pair 16, 32) | 5.3e-4 |
| TE, per-layer, degree 16 | 2.68e-3 | 6.28e-4 | 1.55e-4 | 3.87e-5 | 9.68e-6 | 1.9e-5 | **1.1e-8** |
| TE, shared union grid, degree 16 | 2.68e-3 | 6.28e-4 | 1.55e-4 | 3.87e-5 | -- | 7.8e-5 | 1.2e-7 (pair 16, 32) |

(Wall time at `ns = 32`: 2.6 s per-layer, 1016 s on the shared union grid.)

* **The first order is the staircase's own**: raising the degree from 16 to
  20 and swapping the per-layer mortar for the shared union grid leave the TM
  ladder unchanged to 1.3e-5.
* **Two independent discretisations agree.**  The TE staircase extrapolated
  at second order lands 1.1e-8 from route B's limit.  The TM staircase
  extrapolated at first order lands 6e-5 from it against a last step of
  3.4e-4 (a first-order extrapolation leaves the next term); at second order
  it is worse (2.9e-4), as a first-order sequence should be.
* **Why the PMM staircase and not the RCWA one** (W9 measured second order in
  both polarizations on the RCWA twin): the exact-wall PMM resolves both
  right-angle corners of every step, and those corners carry the TM field
  singularity Popov et al. describe; a Fourier engine at a fixed truncation
  smooths them.  This is an interpretation, consistent with the
  degree-independence above, not a separate measurement.

### 3.4 P3 -- what a slab and a curved-staircase interface cost in the shipped 2-D solver

`p3_cost_2d.py` -> `p3_cost_2d_all.json`.  One layer of an `eps = 4` circular
pillar (radius 0.36, period 1.2, height 0.5, wavelength 1, air above,
`n = 1.45` below) on the Phase B 3 x 3 circle map, solved twice in one
process (cold / warm), BLAS on two threads.  A route-B slab is a curved cell
with a tilt on the OOP generator: its closest shipped equivalent is the
SLANTED circle layer (slant 0.1).  A route-A curved staircase pays, per
interface, the Phase E2 curved mortar between two concentric circles; the
pair below is 0.012 apart in radius (the step of a 20-slice cone whose radius
changes by 0.2 of the period).

| configuration | `M = 5` (cold / warm, s) | `M = 6` | `M = 7` | Python-traced peak (MB) at `M = 5 / 6 / 7` |
|---|---|---|---|---|
| in-plane circle layer (`2 q^2` pencil) | 2.18 / 1.68 | 4.18 / 4.17 | 10.96 / 11.72 | 109 / 143 / 285 |
| slanted circle layer (`4 q^2` OOP generator) | 1.95 / 1.83 | 4.78 / 4.48 | 11.90 / 11.57 | 86 / 194 / 391 |
| two concentric circles on per-layer maps (two layers + one curved mortar) | 19.8 / 28.6 | 107.5 / 105.9 | -- | 494 / 1187 / -- |

* **A route-B slab costs what an in-plane slice costs** on the same grid
  (the whitened standard eig at `4 q^2` against QZ at `2 q^2`, as E1-10
  found: 0.6-1.0x on a loaded box).
* **A curved staircase interface costs 15-25x a slab** (about 25 s at
  `M = 5`, about 100 s at `M = 6` for the mortar alone) and 4.5-8x the memory:
  concentric circles bring the two maps' singular vertices close, which drives
  the cut-cell rule's node count up (E2-5: 41 .. 96 nodes per axis).

### 3.5 P4 -- a rounded rim breaks the lateral map

`p4_rim.py` -> `p4_rim.json`.  Ridge half-width 0.2 (duty 0.5), the top rim
rounded with radius 0.1 (the flat top keeps half-width 0.1), the rest vertical
(one exact slab).  Read: the lossless closure `|1 - R - T|` and the mirror
identity `T(-1) = T(+1)` (both round-off on a sound solve of this symmetric
ridge at normal incidence).

| arm | `K` = 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| uniform slabs on the rim, TE: closure / mirror | 1.5e-7 / 9.5e-11 | 1.2e-2 / 9.4e-5 | 0.37 / 4.2e-4 | 0.13 / 1.3e-3 | 19 / 2.6e-2 |
| uniform slabs, TM | 9.7e-7 / 5.4e-10 | 4.9e2 / 2.2e-5 | 0.86 / 2.8e-3 | 3.0e2 / 7.8e-3 | 1.4e4 / 1.5e-2 |
| graded slabs (`w = R sigma^2`), TE | 24 / 1.7e-4 | 36 / 0.18 | 2.2 / 6.3e-3 | 0.15 / 5.0e-10 | 0.46 / 2.5e-21 |

One frozen rim slab's spectral pairing (relative; companion linearisation /
first-order linearisation), TE:

| tilt at the slab | degree 10 | degree 16 |
|---|---|---|
| 1.13 | 3.3e-12 / 5.2e-13 | 1.9e-8 / 8.1e-11 |
| 2.14 | 3.1e-9 / 2.9e-10 | 4.2e-6 / 7.8e-7 |
| 3.91 | 1.9e-7 / 1.5e-9 | 7.1e-3 / 1.3e-4 |
| 7.95 | 7.5e-7 / 3.6e-7 | 2.4e-2 / 7.0e-3 |

The shipped physical staircase of the same rim (`ns` slices on the rim plus
the vertical part, per-layer grids): successive changes TE 1.9e-4, 7.5e-5,
4.1e-5, 1.8e-5, 7.8e-6 and TM 1.5e-3, 6.1e-4, 2.7e-4, 9.7e-5, 2.4e-5 for
`ns` 4 -> 8 -> .. -> 128 (roughly first order -- the TM ladder's last
step falls 4.1x -- and sound).  Rounded against sharp
(staircase `ns = 128` against the vertical ridge): 9.0e-3 (TE), 1.0e-2 (TM).

### 3.6 P5 -- how steep a linear taper route B carries

`p5_steep.py` -> `p5_steep.json`.  Duty 0.2 -> 0.8 over a height chosen to
give each wall a tilt of 0.5 .. 4 (26.6 .. 76.0 degrees from the vertical);
`K = 16, 32, 64`; worst closure and mirror over the three, and the
amplitude-change ratio per doubling:

| tilt (degrees) | TE degree 12 | TE degree 16 | TM degree 12 | TM degree 16 |
|---|---|---|---|---|
| 0.5 (26.6) | 1.3e-13 / 1.5e-11 / 4.01 | 1.3e-13 / 1.5e-11 / 4.01 | 2.2e-13 / 2.8e-11 / 4.04 | 1.1e-13 / 2.8e-11 / 4.04 |
| 1 (45.0) | 1.1e-13 / 9.1e-12 / 4.01 | 4.1e-13 / 9.3e-12 / 4.01 | 6.3e-14 / 5.2e-12 / 3.98 | 3.4e-13 / 5.2e-12 / 3.99 |
| 2 (63.4) | 3.9e-10 / 1.9e-10 / 4.01 | **1.0e-5** / 4.6e-7 / 3.43 | 5.1e-10 / 2.7e-12 / 3.98 | 7.7e-7 / 1.2e-8 / 4.00 |
| 3 (71.6) | 1.5e-7 / 2.3e-9 / 3.93 | **2e8** (broken) | 2.9e-7 / 3.1e-10 / 4.00 | **1e12** (broken) |
| 4 (76.0) | 5.0e-4 / 2.8e-5 / 0.12 | broken | 7.3e-3 / 1.0e-6 / 1.01 | broken |

The bound is a property of tilt TIMES degree (the slab pencil's
non-normality), not of the geometry's resolution: clean through tilt 1 at
every degree measured, usable at 2 with a closure check, broken from 3 at
degree 16.  Fabricated sidewalls of interest (within 15 degrees of vertical,
tilt <= 0.27) are far inside it; rims and shallow sawtooth profiles are not.

---

## 4. Cost

### 4.1 Per slab

One route-B slab in 2-D = one assembly of the mapped OOP generator at the
slab's height (the quadrature weights plus the six slant blocks; E1-10: the
mapped OOP assembly is 7-16 % of a region solve) + one whitened eig of
dimension `4 q^2` + one Redheffer star of `2 q^2` blocks.  P3: 1.8 s / 4.5 s /
11.6 s at `M = 5 / 6 / 7` on the 3 x 3 circle grid, the same as an in-plane
slice.  The eig de-duplication of the stack (`(cell, slant, map fingerprint)`
key) never hits inside a profiled layer -- every slab is distinct -- so `K`
slabs cost `K` eigs.

### 4.2 A 20-slab tapered pillar against today's staircase

Estimated from P3 (this box, `M = 6`), one pillar layer between half-spaces:

| route | eigs | interfaces | wall time | accuracy class (P2, 1-D analogue) |
|---|---|---|---|---|
| B, `K = 20` | 20 OOP (`4 q^2`) | 19 square matches | 20 x 4.5 = 90 s | second order: ~5e-4 .. 7e-4 at 20 slabs on the 15-degree taper |
| B with Richardson (10, 20) | 30 | square | 135 s | fourth order (TE) / about order 2.6 (TM): ~1e-5 .. 2e-5 |
| A, rectangle (`add_tapered_pillar`, 20 slices) | 20 in-plane (`2 q^2`) on 3-segment grids | 19 separable mortars (cheap) | ~ 20 x 4.2 = 84 s | second order in TE-like fields, first order in TM-like fields; Richardson not usable (W9); mortar class |
| A, circle (20 per-layer circle slices) | 20 in-plane | 19 curved mortars | ~ 20 x 4.2 + 19 x 100 = 33 min | as above, plus the curved mortar's class (E2-4) |

So route B costs about what the rectangular staircase costs at equal count,
reaches a given accuracy with several times fewer slabs (one and a half to
three decades less error at comparable work with Richardson, P2), and is
about 20x cheaper than the curved staircase for circles and ellipses.

### 4.3 Memory

One OOP region at `M = 8` peaks near 0.5 GB (E1-10: 501 MB); `M = 7` 391 MB
traced (P3).  The cascade needs only the running S-matrix (four `2 q^2`
blocks: 12 MB each at `M = 8`), so `K` slabs processed in sequence hold one
region at a time.  `retain_internal` / `layer_absorption` over a profiled
layer need every slab's modes (`K` x ~0.1 GB at `M = 8`): compute them on
demand slab by slab, or refuse above a budget (risk R8).

### 4.4 The JAX twin

Each slab is an OOP region, which the twin refuses (1.4).  Once an OOP twin
exists, a profile parameter (a taper angle) is a smooth parameter of every
slab's map (`T` and the vertex images are linear in `tan` of the angle), so
the frozen-grid design applies unchanged.  Costs that scale with `K`: one
traced eig per slab, the cluster rule's four lifted evaluations per slab
with a degenerate cluster (every slab of a centred cone at normal incidence),
and the compile size (E3 2.3: 3 200 traced operations per region at `M = 4`;
`K` slabs multiply it unless the slab loop is a `lax.scan` over stacked
weights, which the frozen grid makes possible because every slab has the
same shapes).

---

## 5. The design and the API sketch

Names below are PROPOSALS (written without code formatting where they do not
exist yet).

**A z-dependent map** is a frozen `(u, v)` grid plus, for every height `w`, an
in-plane map on it and its `w`-derivative.  For the shape primitives the
in-plane map at height `w` is the transfinite map of the shape's outline at
that height (vertex images and edge curves), and because the Gordon-Hall
blend is LINEAR in its vertex images and edge curves, the tilt field is the
same blend applied to their `w`-derivatives -- exact, at the cost of one more
blend evaluation (a finite difference of the map is the cross-check gate).
Proposed protocol: a method geom_w(sx, sy, U, V, w) returning `X, Y, x_u, x_v,
y_u, y_v` (as `CellMap.geom` does) plus `x_w, y_w`.

**Profiles on shapes, honouring the taper rule.**

    Circle(cx, cy, r, eps=4.0, taper=angle)          # r(z) = r_ref - (z - z_ref) tan(angle)
    Rect(cx, cy, w, h, eps=..., taper=angle)          # every wall moves inward by (z - z_ref) tan(angle)
    FilletRect(..., taper=angle)                      # straight sides and the fillet radius offset together
    Ellipse(..., profile=lambda z: (a(z), b(z)))      # semi-axes as functions of height
    Circle(..., profile=lambda z: r(z))               # any smooth monotone or non-monotone outline

taper is the sidewall angle from the vertical, positive = narrowing upward,
and means an OFFSET of the outline by `(z - z_ref) tan(angle)` along its
normal, about the feature's own centre: exact constant wall angle for circles,
rectangles and filleted rectangles (the offset of a fillet is a fillet of
radius `r - d`).  The offset of an ellipse is not an ellipse; an ellipse takes
profile= (semi-axes as functions of height), and its docstring says the wall
angle then varies around the perimeter.  Gaps widen symmetrically because each
shape's deformation is confined to its own macro-cell (the curved-cell plan's
4.3): the macro-cell boundary is fixed and the host cells inside it absorb the
change.

**Stack level.**

    st = PMM2DStackPure(p, p, n_superstrate=1.0, n_substrate=1.45, n_modes=7)
    st.add_layer(h, shapes=[Circle(cx, cy, r, eps=4.0, taper=np.deg2rad(5))],
                 background_eps=1.0, n_slabs=12)          # or n_slabs='auto'

n_slabs (proposed): the slab count, or 'auto' (decision 9) -- run `K` and
`2K`, return the Richardson value, and report `|f(2K) - f(K)| / 3` as the
profile error estimate, doubling until it is under a tolerance.  A layer with
no profile is today's layer (dispatch, bytes unchanged).  1-D: a mapped
option on `PMMStack.add_tapered_grating` / `add_tapered_ridges` (proposed
method='mapped', n_slabs=), keeping the staircase as the default until the
maintainer flips it.

**Refusals** (raise before any assembly, naming the height and the remedy):

* features whose macro-cells touch or overlap at ANY slab height (the merge
  is decided on the reference outline and re-checked at every height);
* `det J <= 0` at any quadrature node of any slab (`_stag_map_detj_refuse`
  per slab; a tip closing to zero is this case);
* a segment narrower than the sliver contract at any height (a closing
  taper, a fillet radius offset below `sqrt(2) x 1e-3` of the period);
* a change of the outline's TOPOLOGY within the layer (a fillet radius
  reaching zero, a feature vanishing): split the layer at that height;
* a wall tilt above the measured bound (P5: refuse above 2, warn above 1,
  pending the 2-D re-measurement, decision 7) -- which refuses every fully
  rounded rim, pointing at the staircase;
* a profile on `layer_grids='shared'` stacks whose other layers cannot ride
  the face maps (the per-layer machinery is required for two different face
  maps; decision 10).

**JAX.** A profiled layer raises in the twin, naming Phase Z6.

---

## 6. The phased build

Common rules (from `docs/TESTING_STANDARDS.md` and 5.50.0): decisions, not
readings; every bar derived from a measurement with a gap on both sides and
re-dated by the build; every gate two-sided (a fail-before through the real
code path or an engineered mutation); no test over 40 s, no file over 3 min
single-threaded; unit grids <= 3 x 3 at `M <= 5` with `K <= 8` (P3: 1.8 s
per slab at `M = 5`, so such a solve stays near 15 s); BLAS pinned
at file top; every probe asserts `lumenairy.__file__`.  Each phase writes
docs/audits/BUILD_PMM_Z_PROFILES_<PHASE>_<date>.md and gets one independent
verifier (section 9).

**The yardstick for estimates.**  5.50.0 ran the curved-cell plan's seven
build phases (A, B, C, D, E1, E2, E3), each with one verifier, between
2026-10-02 (Phase A commits at 20:12) and 2026-10-04 (release), with E1, E2
and E3 built in parallel on Opus builders.  One **builder round** below means
one such phase: one Opus build session (the curved plan estimated 4-12
agent-hours per phase) plus one verifier session.  The estimates assume the
same mode of work.

### 6.1 Phase Z1 -- route B in the 1-D PMM (one round; can run in parallel with Z2)

**Scope.**  `PMMStack`: a profiled grating as `K` mapped slabs on one element
grid (the mid-height walls), square-matched; scalar (TE / TM through the
scalar pencil with the tilt field) and in-plane tensor cells (the metric
generator with `Dop * S(u)` in the conservative split); normal and oblique
incidence (conical stays refused, as for every slanted 1-D layer); linear
tapers and user profiles; Richardson with the error estimate.

| gate | claim | oracle / arms | bar (from this plan's probes; the build re-measures) |
|---|---|---|---|
| Z1-1 | no profile = 5.50.0 bytes | SHA of R / T / Jones over the 1-D fixture set (vertical, slanted, tapered staircase on both grid modes, oblique) vs `git archive a0aa054d` | equality; fail-before = the identity profile through the new path (round-off, not bytes) |
| Z1-2 | a constant tilt IS the shipped slant | the profiled path with `S = const` against `add_sheared_grating` / `slant_angle=` | operators <= 1e-13 relative; R / T <= 1e-11 (P0 V2: the scratch cascade's floor is 4.7e-10 across 16 slabs, so the library bar is re-derived in the build) |
| Z1-3 | a uniform film under a z-dependent map converges to Airy / `berreman_jones_1d` at second order | film ladder `K` = 8 .. 64, TE / TM / an in-plane tensor | ratio per doubling >= 3.5 on two doublings and error at `K = 32` <= 1e-4 (P0 V3: 3.9-4.0; 2.2e-6 / 4.0e-5); fail-befores tilt dropped (>= 1e-2; measured 3.2e-2 .. 9.8e-2) and M5's term (>= 0.1; measured 0.76 .. 2.3e3) |
| Z1-4 | a tapered ridge: route B and the staircase converge to the same answer | P2's two tapers: route B ladder + Richardson; the shipped staircase ladder | B ratio >= 3.5 per doubling from `K = 8`; the staircase's extrapolated limit (p = 2 in TE, p = 1 in TM, 3.3.1) within 3x its last step of route B's; NOTILT >= 1e-2 at every `K` |
| Z1-5 | every frozen slab is flux-paired and classified | per-slab residuals of `q -> -q`, `q -> conj q`; exactly half forward | <= 1e-10 relative at tilt <= 1 (P1: <= 2.4e-13); fail-before M5's term (>= 1e-4; measured 1.4e-3) |
| Z1-6 | the tilt bound | P5 fixtures at tilt 1, 2, 3 | accepted and clean at 1 (closure <= 1e-11); refused above the bound with the message; fail-before: the bound disabled at tilt 3, degree 16 reads closure >= 1 (measured 2e8) |
| Z1-7 | oblique incidence | TM at 20 degrees on the moderate taper | second order retained (NOT measured here) |
| Z1-8 | Richardson's estimate brackets the error | P2 fixtures | `|f(2K) - f(K)| / 3` within 0.3x .. 3x of the true route-B error from `K = 8` |

**Stop condition.**  If Z1-4's route-B ratio falls below 3 or the two
families disagree beyond the bars on either polarization, stop and report
(the scratch solver measured neither).  **Estimate**: one round.

### 6.2 Phase Z2 -- route B in the 2-D PMM, rectangles (two rounds)

**Scope.**  The tilt field in the 2-D generator; the z-dependent map protocol
and its separable realisation for rectangles (each axis a piecewise-affine
stretch `x = X(u, w)`, `y = Y(v, w)`, the SeparableStretch analogue); a
profiled layer expanding into `K` slabs on one grid in `_solve_per_layer`
with square matches and face maps; Rect(taper=) and the rectangular
`add_tapered_pillar(s)` surface given a mapped option.

Files: `_curvemap.py` (the z-dependent map protocol, the separable z-map),
`twod_staggered.py` (`_stag_map_slant_weights` / `_stag_map_eff_tensor`
taking node arrays; the per-node shear congruence; the rotation gauge on the
field), `stack2d_pure.py` (the profiled layer, its slabs, face maps, the
refusals), tests tests/unit/test_pmm2d_z_profiles_z2.py.

| gate | claim | oracle / arms | bar |
|---|---|---|---|
| Z2-1 | no profile = 5.50.0 bytes | the E1-1 200-key SHA set extended by per-layer, mapped and slanted keys | equality |
| Z2-2 | a constant tilt field through the field path = the shipped slant | operators and full solve, under the identity map and the sheared transfinite map (E1-2's fixtures) | operators <= 1e-11 relative (E1-2 measured 1.3e-14 for the identity map); fail-before a 1e-6 p perturbation of the tilt (moves `Agen` >= 1e-9) |
| Z2-3 | a uniform OOP slab (E1-3's non-reciprocal tensor) under a z-dependent map converges to `berreman_jones_1d` at second order: R, T and BOTH Jones | ladder `K` = 4 .. 32 at normal, oblique (25, 0) and conical (25, 40) incidence | ratio per doubling >= 3.5; fail-befores: tilt dropped (stalls), `tau` without the map (E1-9 `tau_unmapped`), both gauge constants flipped -- on an ASYMMETRIC profile (risk R3) |
| Z2-4 | a rectangular tapered pillar: route B and today's staircase converge to the same answer | `add_tapered_pillar` ladder (per-layer) against the route-B ladder + Richardson, TE- and TM-like incidence | the staircase's limit (its order measured, expected 2 for TE-like and 1 for TM-like components) within 3x its last step of route B's; NOTILT >= 1e-2 |
| Z2-5 | lossless closure and absorption | lossy pillar: `sum layer_absorption == 1 - R - T` over the slabs; lossless closure | closure at the per-layer class of the grid (re-measured); fail-before `-R` as the flux Gram (E2's arm) |
| Z2-6 | reciprocity under a profile | E1-8's channel / reversal at (25, 0) and (25, 40) | spectral decay with `M` (E1-8: 1e-6 .. 1e-8) |
| Z2-7 | the tilt bound in 2-D | the P5 geometry as a rectangular pillar | the 1-D bound re-measured on the whitened generator; bars set from it |
| Z2-8 | cost | route B against the staircase at equal accuracy | build doc only |

**Stop conditions.**  Z2-3's ratio below 3 at any mount, or a gauge constant
that is not map-and-tilt independent (the flipped constants must miss by the
same amount as on E1's slabs).  **Estimate**: two rounds (one for the
generator and the slab cascade with Z2-1 .. Z2-3, one for the stack surface,
the face maps and Z2-4 .. Z2-8).

### 6.3 Phase Z3 -- curved profiles: circles, filleted rectangles, ellipses (one to two rounds)

`shapes2d.py`: taper= / profile= on the primitives, the tilt field by the
Gordon-Hall blend of the derivative data, the merge decided on the reference
outline and re-checked per slab, the refusals of section 5.

| gate | claim | oracle | bar |
|---|---|---|---|
| Z3-1 | geometry exact at every height | slab outline area against the analytic area (`pi r(z)^2`) | 1e-12 relative (Phase B measured 5e-16 for the disk) |
| Z3-2 | the tilt field | Gordon-Hall derivative blend against a centred finite difference of the map | the FD's own error bound |
| Z3-3 | a tapered circular pillar (a truncated cone) against an independent full-wave oracle | the NGSolve quarter-cell script of the curved-cell plan (`validation/probe_pmm2d_curved/fem/fem_circle.py`) with the cylinder replaced by a cone (netgen's OCC kernel builds one); its own three-mesh spread is the bar; run on box B | route B + Richardson inside the FEM's spread at the top rung (the circle landed 1.9e-6 from it) |
| Z3-4 | the same cone against the exact-disk RCWA z-staircase, extrapolated in order count and slice count (E1-6(c)'s procedure) | `RCWAStack` disks | within that oracle's 2e-4-class floor |
| Z3-5 | symmetry | `te(m, n) = tm(n, m)` for a cone; the two circle topologies (3 x 3, 5 x 5) agree | E1-5 / Phase B levels |
| Z3-6 | refusals | touching at the bottom only; a closing tip; a fillet radius crossing zero | each raises naming the height |

**Estimate**: one round, two if the FEM cone needs work beyond swapping the
solid (the FEM runs themselves are hours on box B, not builder time).

### 6.4 Phase Z4 -- the accuracy layer: automatic slab count, Richardson, Magnus (one round)

Richardson on the complex amplitudes and Jones matrices with the error
estimate as n_slabs='auto'; the fourth-order Magnus step (C2) measured
against Richardson at equal cost on the Z2 / Z3 fixtures and kept only if it
wins; documentation (module docstrings, CHANGELOG, `docs/PMM_ROADMAP.md`, a
cookbook entry).  Gates: the Richardson order measured (ratio per doubling
>= 10 on TE-like asymptotic pairs, >= 4 on TM-like ones; P2: 15-30 and
5.8-6.2), the estimate's bracketing (Z1-8), Magnus' flux pairing (its modes
classified like a frozen slab's).

### 6.5 Phase Z5 -- rounded rims (research, two to three rounds, with a stop)

The 45-degree split (lateral map on the steep part of the rim, a vertical
C-method map on the cap), the out-of-plane permeability blocks it needs (E1
F-E1-4), the buffer that returns the vertical map to `z = w` before the
flat face, and the z-vertex.  Prototype first in 1-D on P4's rim (which the
physical staircase converges on at first order, giving the oracle), then 2-D
against the FEM with a filleted cylinder.  **Stop condition**: if the 1-D rim
prototype does not reach closure <= 1e-8 and second-order convergence in `K`
by the end of its first round, stop and record; rims stay on the staircase.

### 6.6 Phase Z6 -- the JAX twin (two rounds, after Z2-Z3)

First the twin of the OOP generator and the generalized cascade (which E3
left out); then profiled layers on it, the taper angle and profile parameters
as traced inputs, the cluster rule per slab, a `lax.scan` over slabs to keep
the compile size independent of `K`.  Gates as E3's: forward parity,
gradients against converged central differences (the frozen-grid offset
recorded), jit compiles once.

### 6.7 Order and total

Z1 and Z2 in parallel; Z3 and Z4 after Z2 (Z4 can start on Z1's 1-D code);
release 5.51.0 after Z4.  Total for the release: five to six builder rounds,
the same order as 5.50.0's seven.  Z5 and Z6 are separate decisions.

---

## 7. Risks and their mitigation

| # | risk | what is known | mitigation |
|---|---|---|---|
| R1 | Wood anomalies and slab modes at cutoff | not probed; inside a profiled layer the medium varies continuously, so SOME slab may have a mode near `q = 0` for some designs | a slab mode at cutoff over a thin slab propagates with `|exp(i q k0 d)| ~ 1`, so its forward/backward label matters little; gate: a wavelength sweep through a slab-mode cutoff must be continuous in R / T (Z2) |
| R2 | the gauge constants with a varying tilt | E1 measured both map-independent (F-E1-3) | gate Z2-3 with BOTH constants flipped, on an asymmetric profile at oblique and conical incidence, reading both Jones matrices |
| R3 | a centred symmetric taper hides the rotation sign | the radial tilt field of a centred taper is odd about the centre, so the 180-degree rotation leaves it unchanged | every gauge gate uses an off-centre or asymmetric profile (a one-sided wedge, a tapered pillar off the cell centre with a slant) |
| R4 | the mortar at a profiled layer's faces | inside the layer there is none (square matches); a patterned neighbour on a different map pays E2's class (1e-2 at `M = 6`, 1e-3 at `M = 8 .. 10` where an outline crosses the neighbour's cells).  Consecutive TAPER slabs are not "non-crossing E2 layers": they are not on different maps at all, so no class applies between them.  (Nested, non-crossing outlines on different maps -- a route-A curved staircase -- are still cut by each other's cells, so they ARE in the algebraic class; non-crossing is not conforming) | stack a cylinder on a cone with matching face outlines on the face map (square match); document the class where a face meets a different outline |
| R5 | the TM-like staircase artefact | measured first order in 1-D TM (P2); route B has no steps | the reason to build; Z2-4 measures the 2-D orders |
| R6 | steep walls and rims | P5: clean through tilt 1, degraded at 2, broken from 3 at degree 16; P4: rims break | the tilt bound as a refusal; rims on the staircase until Z5 |
| R7 | the discrete `E_3` elimination at large tilt | the continuum cancels `eps |T|^2`; the Galerkin elimination with an unbounded tilt is untested | measured inside Z2-7 (closure and pairing against tilt on the 2-D generator) |
| R8 | memory with internal fields | `K` slabs' modes for `layer_absorption` / `retain_internal` | recompute slab by slab, or refuse above a stated budget |
| R9 | cost of many slabs | `K` OOP eigs, no de-duplication, no parity reduction under a map | Richardson keeps `K` small; the parity reduction for symmetric maps is a separate item (F-E1-5) |
| R10 | M5's unexplained "first order" note | not reproduced (0.5) | Z1-4's ladder settles it in library code |

---

## 8. What is NOT in scope

* **Route D** (a polynomial-in-`w` block per layer): a different solver class
  (2.6).
* **Rounded rims in the first release**: Phase Z5, research, with a stop.
* **A full vertical (C-method) map for whole surface-relief profiles** (blazed
  and sinusoidal gratings): Z5's machinery would carry it; not before.
* **Conical incidence in the 1-D route B**: every slanted 1-D layer refuses it
  today; the 1-D route B inherits that scope.
* **The RCWA engines**: unchanged; they remain the staircase arbiters.
* **The parity reduction for symmetric maps** (F-E1-5): a separate plan item.

---

## 9. Verification protocol

One independent verifier per build phase, launched after the builder's
report, working from this plan and the build document only, on its OWN
fixtures (period, wavelength, contrast, taper angle and profile chosen by the
verifier), re-measuring every gate and re-deriving each bar's two gaps.
Documentation is read once for factual claims that carry numbers.  A defect
goes back to the builder and is re-checked by the same verifier on the
failing item only.  The FEM cone (Z3-3) is the Z3 verifier's oracle, run on
box B through the mesh tooling (keep box A for orchestration); if it cannot
be run the verifier states which oracle replaced it.

---

## 10. What this plan could not determine

* **2-D numbers.**  Every order and every bound above is measured in 1-D.
  The 2-D orders (second for route B, first for TM-like staircase fields) are
  expected, not measured; Z2-3 / Z2-4 measure them.
* **The 2-D tilt bound and the discrete `E_3` elimination at large tilt**
  (R7).
* **Oblique incidence** in route B: every probe here is at normal incidence.
* **Why M5's cascade read first order** with the antisymmetrised form (0.5):
  the likeliest cause (an `H_x`-like partner built with the mid-depth metric
  and the plain `du` mass) is an inference from its code, not a measurement.
* **The citation**: Popov, Neviere, Gralak and Tayeb is the author list as
  recalled here; the request for this plan listed Enoch.  Shcherbakov and Tishchenko's
  curvilinear-coordinate grating method (Opt. Express 21, 25236 (2013), as
  recalled) may be the closest published precedent for a z-dependent
  coordinate frame in a modal method; not checked.
* **Whether Magnus-4 beats Richardson** at equal cost (Z4 measures).
* **Why the RCWA staircase reads second order in TM where the PMM one reads
  first** (3.3.1): an interpretation, not measured.

---

## 11. Decisions for the maintainer

1. **Which profiles first?**  Recommendation: linear tapers (taper= as an
   outline offset) on `Rect`, `Circle` and `FilletRect` plus a profile=
   callable for smooth outlines within the tilt bound; ellipses through
   profile= only.  Cost: Z2 + Z3 (three to four rounds).
2. **1-D first, 2-D first, or both?**  Recommendation: both in parallel (they
   share no code); if only one, the 1-D library build first -- one round, and
   it gives TM-polarized 1-D gratings second-order (fourth with Richardson)
   convergence where today's staircase is first order (P2).
3. **Route B then C, or C directly?**  Recommendation: route B with
   Richardson (C1) as the shipped higher-order layer; the Magnus step (C2)
   only if Z4 measures it beating Richardson at equal cost.  Cost: C1 is
   included in Z1/Z4; C2 adds a quarter round.
4. **Rounded rims in the first release?**  Recommendation: no -- refuse them
   (they exceed the tilt bound by construction) and point at the staircase,
   which converges on them at roughly first order (P4); Z5 as a separate research
   decision (two to three rounds, may stop).
5. **Accuracy target.**  Recommendation: 1e-5 per order in R / T (and 1e-4 on
   the Jones matrices), the level at which today's pillars are capped by the
   rim edge in `M` (curved-cell plan 3.4); route B + Richardson reaches it at
   `K` = 8-16 on the probes' tapers.
6. **The FEM oracle.**  Recommendation: extend the NGSolve quarter-cell
   script (NGSolve 6.2.2604 is installed on tesla-ryzen) to a cone for Z3 and
   run it on box B; the verifier owns it.  Cost: hours of box-B time.
7. **The tilt bound.**  Recommendation: refuse wall tilts above 2 (63
   degrees from the vertical), warn above 1 (45 degrees), and re-derive both
   numbers in Z2-7 on the 2-D generator.
8. **Release.**  Recommendation: 5.51.0 (MINOR, a new solver capability) for
   Z1-Z4; Z5 and Z6 in later releases.
9. **Slab count default.**  Recommendation: n_slabs='auto' (Richardson with
   the estimate, doubling to a tolerance) as the default for profiled layers,
   an integer for reproducible studies.
10. **Shared-grid stacks.**  A profiled layer has two face maps, which the
    single stack-wide map of `layer_grids='shared'` cannot carry.
    Recommendation: profiled layers require `layer_grids='per-layer'` in the
    first release (the riding rule makes half-spaces and uniform neighbours
    square matches anyway); a shared-grid route is a later convenience.
11. **The JAX twin.**  Recommendation: after Z3, and only once an OOP twin is
    approved (Z6 is two rounds, the first of which is the OOP twin itself).
12. **The physical staircase's default.**  Recommendation: keep
    `add_tapered_grating` / `add_tapered_pillar(s)` on the staircase by
    default in 5.51.0, with the mapped route as an option, and flip the
    default only after a release of field use.

---

## Appendix -- the probe files

| file | what it is |
|---|---|
| `validation/probe_pmm_z_profiles/_zcommon.py` | the scratch 1-D route-B solver: periodic C0 spectral elements on fixed walls, the linear-taper map, the virtual-medium coefficients (TE and TM), conservative frozen slabs (switches: tilt dropped, M5's term), flux-split modes, square-matched S-matrix cascade, mapped half-spaces, the pulled-back far field |
| `p0_validate.py` / `.json` | P0: vertical ridge against `PMMStack`, the shear null control, the film against Airy with both fail-befores |
| `p1_spectrum.py` / `.json` | P1: the frozen slab's spectrum, conservative against M5's form |
| `p2_taper_ladder.py` / `.json` | P2: route B, Richardson, NOTILT, M5 and the shipped staircase on two linear tapers |
| `p2b_tm_stair.py` / `.json` | P2b: the TM staircase on both grid modes and two degrees, extrapolated |
| `p3_cost_2d.py` / `p3_cost_2d_all.json` | P3: the shipped 2-D solver's in-plane slice, slanted slab and curved-staircase interface costs |
| `p4_rim.py` / `.json` | P4: the rounded rim -- route B's breakdown, the slab pairing against tilt and degree, the staircase's convergence, rounded against sharp |
| `p5_steep.py` / `.json` | P5: the tilt bound on linear tapers |

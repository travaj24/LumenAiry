# BUILD -- curved cells for the pure staggered 2-D PMM, Phase E3 (a JAX twin of the pure staggered path, differentiable in shape parameters)

Date: 2026-10-03.  Status: BUILT + gated, not pushed (section 1 was
recorded before the rest was built, as the brief requires).
Builder: Claude Opus 5.5 (model ID `claude-opus-5-5`).
Mount: worktree `C:/tmp/lum_curved_e3`, branch `feat/pmm2d-curved-e3-jax`,
built on `eae470d9` (Phase D); Windows 11 (tesla-ryzen), CPython 3.14.6,
numpy 2.4.4, scipy 1.17.1, jax / jaxlib 0.11.0, `jax_enable_x64` on in every
probe, `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` on the
command line, `lumenairy.__file__` asserted under the worktree in every
probe (`build_e3/_e3common.py`).  PRE tree for byte identity: `git archive
eae470d9` extracted to `C:/tmp/curved_pre_e3/`.  The box was shared with
four sibling builds (24-30 single-threaded Python processes on 24 logical
cores); every WALL TIME below is an upper bound, no accuracy number depends
on the load.
Plan: `docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md` section 4.5
(Phase E, approved 2026-10-02; E3 = the JAX twin, built in parallel with E1
= out-of-plane and slant under a map, E2 = per-layer maps).
Evidence: every number is read from a JSON file in
`validation/probe_pmm2d_curved/build_e3/` (the probe is named in each table).

---

## 0. Words used here

* **The twin**: the differentiable (JAX) evaluation of the shared-grid
  `PMM2DStackPure` cascade, `lumenairy/elements/pmm/_jax_twod_staggered.py`
  (`StagJaxTwin`).
* **Reference**: the CONCRETE stack the twin is built from.  Every DISCRETE
  decision the NumPy solve takes from values (the wall grid and its
  topology, the adaptive quadrature node count, the corner (Duffy) cells,
  the far-field and incident node counts, the Wood nudge) is taken once, at
  the reference, and frozen.
* **Traced map**: a `TransfiniteMap` on the reference's frozen `(u, v)`
  grid whose vertex images and edge curves are functions of JAX shape
  parameters (`TransfiniteMap._traced`); a shape parameter moves the IMAGES
  of the frozen cells, never the `(u, v)` walls.
* **FD(twin)** / **FD(numpy)**: central finite differences of the twin's own
  forward (the same discrete function the gradient differentiates) and of
  the shipped NumPy solve, whose wall grid moves with the parameter (the
  `Rect` / `Circle` shape route).
* **AD**: the reverse-mode gradient (`jax.jacrev`) through the twin.

---

## 1. Feasibility probe (deliverable 1, measured before the rest was built)

Probe: `f1_feasibility.py rect|circle M` -> `f1_<case>_M<M>.json`.

**Fixtures.**  *rect*: the 3 x 3 rectangular pillar, `w = 0.5`, `h = 0.4`
in a `1.2 x 1.2` cell (units of the wavelength `1.0`), `eps = 4` in vacuum,
depth `0.4`, `n = 1.0 / 1.45`, normal incidence, `n_orders = 2`; the width
enters as a traced transfinite map whose interior vertex images sit at
`cx -+ w / 2` (straight edges: every cell AFFINE).  *circle*: the Phase B
3 x 3 circle, `r = 0.36`, `eps = 4`, depth `0.5`; the four arcs and the
vertex images `c -+ r / sqrt 2` are functions of `r`, the four corner cells
keep their Duffy rule, the Jacobian is analytic.  The derivative is taken of
`R00` and `T00` (incident `E_x`).

**The eig.**  JAX has no generalized eig; the in-plane pencil `L W = g2 G
W` (`G = -R`, Hermitian positive definite in every in-plane region) is
reduced to the standard eig of `G^-1 L` and differentiated through the
library's one custom-VJP eig `rcwa._jax_eig_stable`.  The eigenvector
normalisation differs from QZ's; every output is invariant under a per-mode
rescaling.

### 1.1 Forward parity (max absolute difference over every order of R, T and over the Jones matrix)

| fixture | twin route vs NumPy | M = 4 | M = 5 | M = 6 |
|---|---|---|---|---|
| rect | unmapped twin (kron assembly) vs NumPy | R 4.3e-15, T 3.8e-14, J 1.5e-14 | 3.7e-15, 4.4e-14, 2.2e-14 | 1.2e-14, 8.3e-14, 3.9e-14 |
| rect | mapped twin (identity transfinite map) vs NumPy UNMAPPED | 2.0e-15, 3.9e-14, 1.2e-14 | 3.2e-15, 3.4e-14, 1.6e-14 | 7.0e-15, 4.5e-14, 4.2e-14 |
| rect | mapped twin vs NumPy on the same identity map | 2.0e-15, 3.4e-14, 1.2e-14 | 2.5e-15, 3.9e-14, 1.7e-14 | 8.7e-15, 3.9e-14, 3.0e-14 |
| rect | traced map at the reference width vs the reference map | 2.3e-15, 2.8e-14, 1.7e-14 | 4.3e-15, 3.0e-14, 1.9e-14 | 8.7e-15, 7.4e-14, 1.0e-13 |
| circle | mapped twin vs NumPy (shape route) | -- | 1.5e-14, 1.3e-13, 6.7e-14 | 6.2e-14, 1.7e-13, 2.6e-13 |
| circle | traced map at `r0` vs the reference map | -- | 1.1e-14, 1.1e-13, 6.1e-14 | 7.1e-14, 2.1e-13, 2.9e-13 |

The stop condition (forward parity under 1e-10 on the rectangular cell) is
met by four decades.

### 1.2 Gradients against converged central differences

Step ladder `h / P = 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5`.  Both FD
columns fall like `h^2` down the ladder (circle `M = 5`, `d R00 / d r`:
FD(twin) minus AD = 1.3e-2, 1.2e-3, 1.4e-4, 1.2e-5, 1.4e-6, 1.25e-7,
1.4e-8 -- the truncation law, so the last rung IS converged to its own
`h^2` term); "rung change" below is the change between the last two rungs.

| fixture | M | AD (R00, T00) | AD vs FD(twin), rel. | AD vs FD(numpy), rel. | last rung change (twin / numpy) |
|---|---|---|---|---|---|
| rect, d/dw | 4 | 0.0371416, -0.793364 | 1.5e-9, 6.7e-10 | 2.4e-9, 1.1e-9 | 6.1e-9 / 5.4e-9 |
| rect, d/dw | 5 | 0.0402883, -0.783242 | 6.4e-9, 4.6e-9 | 1.2e-9, 7.6e-10 | 2.4e-9 / 5.8e-9 |
| rect, d/dw | 6 | 0.0406733, -0.775911 | 7.7e-9, 4.2e-9 | 1.3e-9, 1.8e-9 | 1.0e-8 / 5.4e-9 |
| circle, d/dr | 5 | -0.283297, -1.752559 | 5.0e-8, 2.0e-8 | 1.5e-6, 1.8e-6 | 2.5e-7 / 2.5e-7 |
| circle, d/dr | 6 | -0.300874, -1.929498 | 2.3e-8, 4.6e-9 | 7.8e-8, 4.3e-8 | 2.9e-7 / 2.6e-7 |

The stop condition (the circle-radius gradient within 1e-4 relative of a
converged FD at `M = 6`) is met by three and a half decades against EITHER
FD.  The AD-vs-FD(twin) gap sits below the last rung's own change (the FD,
not the AD, is the limiting error).

**FD(numpy) differs from FD(twin) for the circle, and by how much is a
measurement of the frozen grid, not of the gradient.**  The twin evaluated
at `r != r0` is a DIFFERENT parametrisation of the same geometry than the
NumPy solve built at `r` (whose walls sit at `c -+ r / sqrt 2`): the cells
have the same physical images, but the `(u, v)` lengths of the cells
differ, and the L2 projection of the incident field (Phase C; exact only up
to its representation error under a non-polynomial map) weights the cells
by their `(u, v)` area.  Measured at `M = 4`: the twin frozen at `r0 = 0.36`
and evaluated at `r = 0.33` differs from NumPy built at `0.33` by 5.5e-6 on
T (`f8_frozen_vs_moving.json`); the derivative offset falls from 1.5e-6 (`M = 5`)
to 7.8e-8 (`M = 6`), with the incident representation error.  For the
rectangle (affine cells at normal incidence, an exactly representable
incident) the two discretisations coincide to round-off at every width (3.5e-15 at `w = 0.47`
with the twin frozen at `0.5`), and FD(numpy) agrees with AD to 1e-9.

### 1.3 Compile and run times (jit; the box loaded, four probes in parallel)

| fixture | M | template (NumPy reference) | jit forward: first / repeat | jit grad: first / repeat | NumPy solve |
|---|---|---|---|---|---|
| rect | 4 | 0.3 s | 11.4 s / 0.21 s | 28.3 s / 0.32 s | 1.2 s |
| rect | 5 | 1.1 s | 14.1 s / 0.81 s | 35.4 s / 1.35 s | 3.5 s |
| rect | 6 | 9.9 s | 17.9 s / 2.8 s | 47.5 s / 4.2 s | 25.0 s |
| circle | 5 | 3.0 s | 18.8 s / 0.74 s | 56.8 s / 1.7 s | 8.1 s |
| circle | 6 | 8.6 s | 25.5 s / 2.8 s | 62.7 s / 5.4 s | 26.9 s |

(The rect `M = 4` row is the re-measurement after the compile-size work of
section 2.3; before it the same probe compiled the gradient in 213.8 s and
the forward in 29.2 s, `f1_rect_M4.json` holds the first run.)  A compiled
gradient re-runs in 1-5 s, faster than one NumPy QZ solve of the same
cell: the standard eig of `G^-1 L` is cheaper than QZ on the pencil (the
plan's Q4 measured 8.8-10.7x for a Cholesky-whitened eig).

**Verdict.**  Feasible on both counts; the map-parameter chain (shape
parameter -> vertex images and arcs -> analytic Jacobian at the frozen nodes
and Duffy points -> `_stag_map_eff` / `_stag_map_eff_tensor` -> the 18
quadrature blocks -> eig -> cascade) differentiates correctly.  The rest of
the twin was built on it.

---

## 2. What was built

### 2.1 The twin, and what it shares with NumPy

`lumenairy/elements/pmm/_jax_twod_staggered.py` holds only what cannot be
shared: `_stag_geneig_jax` (the pencil eig, section 1), `_shadow` (a
`copy.copy` of a NumPy-built `Granet2DTransverseE` whose materials / map are
replaced by traced values and whose OWN `_assemble` runs with `_xp =
jax.numpy`), `_min_detj` (the in-trace fold guard), `_traced_shape_merge`
(the replay of the shape merge, 2.4), `StagJaxTwin` (the frozen template and
the cascade driver) and the convenience entry `_pmm_jones_2d_staggered_jax`.
Everything numerical is the NumPy modules' code, run with `xp = jax.numpy`
(commit `4d56450c`, byte-identical for NumPy, gate E3-1):

| stage | NumPy object the twin runs |
|---|---|
| the 18 weighted blocks, `S_tt`, the Schur term, `R`, `L` | `Granet2DTransverseE._assemble` / `_eps_weighted` / `_eps_dir` / `_chi_maps` (attribute `_xp`); `_xset` for the block writes |
| one weighted block under a map | `_stag_quad_weighted(xp=)` on the factors of `_stag_quad_axis_factor` |
| effective tensors at the nodes | `_stag_map_weights(xp=)` -> `_stag_map_eff` / `_stag_map_eff_tensor`; Jacobians by `_stag_map_node_jacobian` |
| the map | `TransfiniteMap._blend` -> `_gh` (the ONE Gordon-Hall formula, also batched by `prefetch`); `Line` / `Arc` / `EllipseArc` / `Sinusoid` with traced parameters |
| modes after the eig | `_region_modes_from_eig`, `_homog_geom_from_eig`, `_homog_region_modes`, `_forward_branch_flip(xp=)`, `_inv_lam` |
| far field, incident field | `_far_projector_mapped(xp=, nq_cells=)`, `_stag_incident_coeffs_mapped(xp=, nq_cells=)`, `_pmm2d_project_orders(xp=)`, `_pmm2d_order_kz(xp=)` |
| cascade, efficiencies | `rcwa._core._interface_smatrix`, `_redheffer_star`, `_propagation_smatrix`, `_project_efficiency` (the algebra the NumPy stack uses) |
| the shape layouts | every primitive's own `_layout(ref=)`, `shapes2d._curve_piece` / `_curve_point` / `_fill_mask`, `_merge(parts=)` |

The census test (E3-8) spies on every row and fails if one is not reached
through its NumPy module, and parses the twin module for any function of
the same name (none).

### 2.2 The frozen discrete decisions

| decision | NumPy | twin |
|---|---|---|
| `(u, v)` wall grid, cell topology, curve pieces, painted cells | the shape merge at the current values | the merge at the REFERENCE; a traced parameter moves the vertex images and curves (2.4) |
| adaptive node count of the mapped assembly, corner (Duffy) cells | `_stag_map_nodes`, `singular_vertices` | the reference solver's `_qrule` (frozen); the traced map carries the reference's singular-vertex list |
| far-field / incident node counts per cell | sized from the cell's physical extent | recorded on the reference (`record=`), imposed (`nq_cells=`) |
| Wood nudge, propagating-incidence check, cost guard | at the values | at the reference values (a traced value skips them -- the documented caveat of every JAX twin in the library) |
| wavelength, angles (the Bloch glue `tau` of the basis) | per solve | static; a traced / changed superstrate index at oblique incidence raises |
| incident decomposition, no map | least-squares overlap on `W0` | the same, a NumPy CONSTANT of the template (`W0` is eps-free) |
| forward branch | `_forward_branch_flip` (NumPy) | the same function, `xp = jnp` (`jnp.where`) |
| eig de-duplication | by cell bytes | by the reference key and the identity of the traced leaves |

The node count being frozen means the twin's quadrature does not jump with
the parameter while the NumPy solve's does; the NumPy jump is at the
criterion's tolerance (1e-13 on the weights' moments), far below every
other difference here.

### 2.3 Compile size (a build finding, measured)

The first working twin traced 19 076 operations at `M = 4` (3 x 3 rect) and
compiled its gradient in 213.8 s (measured during the build on the first
`f1_feasibility.py rect 4` run, whose JSON the re-run overwrote; the counts
before and after each change were read with `jax.make_jaxpr`; the final
state is `f10_compile_size_M4.json`).  Three changes, each keeping the NumPy
bytes: (i) the traced quadrature contracts all tensor-rule cells in two
einsums against the per-axis factors EMBEDDED in the global index range
(`_stag_quad_embedded`) instead of one scatter per cell (19 076 -> 10 508
operations); (ii) a traced map memoises its edge curves and its per-cell
evaluations (the assembly, the fold guard, the far field and the incident
load ask for the same nodes) and batches every cell of one node count in
ONE broadcast of the Gordon-Hall formula (`prefetch` -> `_gh`) (10 508 ->
3 194); (iii) the far projector's traced cells are batched per node count.
Gradient compile 213.8 s -> 28.3 s, forward 29.2 s -> 11.4 s
(`f1_rect_M4.json`); `f10_compile_size_M4.json` (box less loaded): 3 200
operations -- per stage: the traced map's Jacobian at every node 634, the
node weights 666, one shadow assembly 852, the cofactor far field 871, the
fold guard 640 -- forward compile 9.2 s, gradient 24.3 s.  The rest of the
compile time is the LAPACK kernels (one `eig` / `solve` / `inv` / custom-VJP
eig gradient compiles in 0.23 / 0.39 / 0.39 / 0.56 s each on a 162 x 162
complex matrix) and grows with the grid (the 5 x 5 fillet: 120-170 s).

### 2.4 Traced shape parameters

A shape object accepts JAX values for its geometry and material (no
`float()`, no concrete validation when traced).  The twin replays the shape
merge (`_traced_shape_merge`): each shape's own `_layout(ref=reference
shape)` computes the traced vertex images and curves while every structural
decision is taken from the concrete reference shape (Rect snapping, the
fillet's `r = 0` branch, the ellipse's axis-aligned branch, the sinusoid's
curved flag); merged walls take the traced value of their first owner,
grid vertices the first claim, edges the first curve piece -- the
reference merge's own order -- and every coincidence the reference relied on
(two walls snapped into one, two claims on one vertex or edge) is checked
in the trace.  The painted cells take the reference's masks and the traced
materials.

### 2.5 The API

```python
import jax
import jax.numpy as jnp
from lumenairy.elements.pmm import Circle, PMM2DStackPure, pmm_jones_2d_staggered
jax.config.update("jax_enable_x64", True)

# (1) forward, JAX arrays out -- the NumPy answer to round-off
orders, R, T, J = pmm_jones_2d_staggered(
    1.2, 1.2, None, 1.45, 1.0, 0.5, 1.0, n_modes=6, n_orders=2,
    shapes=[Circle(0.6, 0.6, 0.36, eps=4.0)], background_eps=1.0,
    backend="jax")

# (2) the gradient of T00 w.r.t. the radius, compiled once
st = PMM2DStackPure(1.2, 1.2, n_superstrate=1.0, n_substrate=1.45,
                    n_modes=6, n_orders=2, backend="jax")
st.add_layer(0.5, shapes=[Circle(0.6, 0.6, 0.36, eps=4.0)], background_eps=1.0)
st.set_source(1.0)
p0 = st.jax_twin().p0

def t00(r):
    p = st.jax_params()
    p["layers"][0]["shapes"] = [Circle(0.6, 0.6, r, eps=4.0)]
    return st.solve(params=p)[2][0, p0]

dT_dr = jax.jit(jax.grad(t00))(0.36)

# (3) material, thickness and index gradients through the same dictionary
def loss(v):
    p = st.jax_params()
    p["layers"][0]["shapes"] = [Circle(0.6, 0.6, 0.36, eps=v[0] + 1j * v[1])]
    p["layers"][0]["thickness"] = v[2]
    p["n_substrate"] = v[3]
    return st.solve(params=p)[2][0, p0]

g = jax.grad(loss)(jnp.array([4.0, 0.0, 0.5, 1.45]))
```

`pmm_jones_2d_staggered(..., backend='jax')` also traces `eps_cell`,
`mu_cell`, `depth` and the indices; traced SHAPES there need
`reference_shapes=` (the same shapes at concrete values).

---

## 3. The gates

### 3.1 E3-1 -- no JAX call = today's NumPy bytes

`e1_bytes.py` (Phase D's `d1_bytes.py` set plus an E3 section: mapped
tensor and magnetic solvers and their modes, the mapped geometric cache,
the cofactor far field, the incident decomposition, every shape primitive
including the rotated ellipse, the 5 x 5 circle, the sinusoid and a lossy
oblique circle, a two-layer merged stack with absorption), run in the PRE
tree (`git archive eae470d9`) and in this tree after each library change:
**156 / 156 SHA-256 equal** (`e1_compare.json`), last run after the final
library edit.  Unit form: every twin-only branch booby-trapped, NumPy
solves of an unmapped tensor cell, a circle and a fillet still run; `_xset`
on NumPy writes in place and returns the same object.  The curved A-D
suites: 71 passed.

### 3.2 E3-2 -- forward parity, the bar derived from the eig stage's round-off

`f2_parity.py M`: 22 fixtures; three solves each -- NumPy, NumPy with ONE
stage changed (QZ replaced by the standard eig of `G^-1 L`, the reduction
the twin uses: the round-off REACH of the eig stage on that fixture), and
the twin.  `M = 4` (max over every order of R / T and over the Jones matrix):

| fixture | twin - NumPy (R / T / J) | eig-stage reach (R / T / J) | operator rel. |
|---|---|---|---|
| scalar_pillar | 3.3e-15 / 2.0e-14 / 1.3e-14 | 2.3e-15 / 1.0e-14 / 9.1e-15 | 2.3e-16 |
| scalar_rect_walls (mapped twin vs unmapped NumPy) | 1.9e-15 / 1.1e-14 / 6.3e-15 | 1.0e-15 / 6.5e-15 / 1.2e-14 | 7.4e-15 |
| scalar_oblique | 2.0e-15 / 9.5e-15 / 6.5e-15 | 2.9e-15 / 1.0e-14 / 5.0e-15 | 2.8e-16 |
| scalar_conical | 8.3e-16 / 3.8e-15 / 2.0e-15 | 2.1e-15 / 9.4e-15 / 5.9e-15 | 5.1e-16 |
| lossy_scalar | 1.2e-15 / 7.3e-15 / 1.2e-14 | 1.0e-15 / 1.7e-14 / 1.0e-14 | 2.0e-16 |
| tensor_lc | 1.4e-15 / 5.9e-15 / 4.0e-15 | 8.3e-16 / 2.1e-15 / 4.9e-15 | 1.5e-16 |
| tensor_lc_conical | 5.7e-16 / 5.7e-15 / 2.5e-15 | 5.1e-16 / 4.3e-15 / 2.7e-15 | 9.4e-16 |
| tensor_gyro | 1.2e-15 / 1.1e-14 / 5.3e-15 | 3.0e-16 / 9.0e-15 / 5.2e-15 | 2.6e-16 |
| magnetic_scalar | 2.1e-15 / 4.8e-15 / 5.7e-15 | 9.1e-16 / 6.4e-15 / 5.1e-15 | 3.4e-15 |
| magnetic_tensor | 9.1e-16 / 6.4e-15 / 4.9e-15 | 2.0e-15 / 4.9e-15 / 4.7e-15 | 2.4e-15 |
| multilayer (uniform / lossy patterned / uniform tensor, conical) | 3.4e-15 / 8.7e-15 / 5.7e-15 | 2.1e-15 / 7.5e-15 / 1.0e-14 | 4.4e-16 |
| mapped_stretch_stripe | 9.4e-16 / 1.0e-14 / 5.1e-15 | 1.5e-15 / 8.7e-15 / 8.0e-15 | 1.6e-14 |
| circle | 2.9e-15 / 3.4e-14 / 1.7e-14 | 6.5e-15 / 2.4e-14 / 2.2e-14 | 5.0e-14 |
| circle_conical | 4.0e-15 / 2.1e-14 / 2.2e-14 | 4.5e-13 / 5.0e-13 / 6.2e-13 | 5.7e-14 |
| circle_lossy_oblique | 2.9e-15 / 3.9e-14 / 3.5e-14 | 1.0e-14 / 2.6e-14 / 8.5e-14 | 3.7e-14 |
| circle5 (5 x 5 layout) | 5.7e-15 / 6.2e-14 / 4.9e-14 | 1.6e-14 / 5.6e-14 / 5.2e-14 | 1.7e-14 |
| fillet | 1.6e-13 / 5.3e-13 / 4.3e-13 | 7.8e-14 / 2.6e-13 / 2.5e-13 | 1.1e-14 |
| ellipse_rot | 5.1e-15 / 1.6e-14 / 2.6e-14 | 4.2e-15 / 2.6e-14 / 3.4e-14 | 6.1e-14 |
| sinusoid | 4.2e-16 / 7.3e-15 / 2.2e-15 | 4.9e-16 / 1.3e-14 / 4.4e-15 | 1.3e-14 |
| two_layer_merged | 6.1e-14 / 1.8e-13 / 1.1e-13 | 4.8e-14 / 8.2e-14 / 9.3e-14 | 1.0e-14 |
| circle_tensor_lc | 3.8e-15 / 5.3e-15 / 1.0e-14 | 6.1e-15 / 1.3e-14 / 1.5e-14 | 3.2e-14 |
| circle_magnetic | 6.0e-15 / 3.8e-14 / 3.3e-14 | 2.6e-15 / 4.4e-14 / 4.8e-14 | 3.9e-14 |

At `M = 3`: max twin - NumPy 3.4e-14, max reach 5.1e-14, max operator
difference 2.1e-14 (`f2_parity_M3.json`).  On these 22 fixtures the
twin's largest difference over R, T and J sits within 2.9x (`M = 3`) / 2.0x
(`M = 4`) of the eig stage's reach on the same fixture (up to 3.8x per
quantity) -- an observation, not a bound (the verifier's fixtures: 4.0x,
6.9x per quantity): its differences from NumPy are that stage's round-off
plus the summation order of the vectorised quadrature (operators <= 6.1e-14
relative).  The E3 unit gate was one global bar **5e-12** (two decades above
the `M = 3` maximum, 8x above the largest `M = 4` reach; the cofactor
mutation, E3-9, moves the mapped T by 6.6e-2); round 2 restates it PER
FIXTURE (section 9.6).  Rectangles-only `shapes=` stacks at OBLIQUE
incidence are not round-off parity (V-E3-2, section 9.4).

### 3.3 E3-3 -- gradients against converged central differences, both builds

`f3_grads.py CASE M [numpy]` through the public API (traced shape objects
in `params`).  FD(twin) on the ladder `h / P = 3e-3 .. 3e-5`; the
rung-to-rung changes fall by 8.8, 11.4, 8.8 for every case except eps at
`M = 4` (8.8, 12.6, 5.5) and the conical circle at `M = 4` (below) (the
`h^2` law at a step ratio of 3 is 9) -- the PREMISE that the FD is in its asymptotic
range -- and the converged value is the Richardson extrapolation of the
last two rungs.  Max relative AD-vs-FD over the case's quantities
(`f3_summary.json`):

| case (quantity) | M | Windows: vs FD(twin) | vs FD(numpy) | WSL: vs FD(twin) |
|---|---|---|---|---|
| rect width (R00, T00) | 4 | 8.6e-9 | 6.2e-9 | 7.1e-9 |
| rect width | 5 | 5.9e-9 | 5.9e-9 | 5.1e-9 |
| circle radius (R00, T00) | 4 | 7.1e-9 | 9.1e-5 | 7.0e-9 |
| circle radius | 5 | 1.2e-7 | 1.9e-6 | 1.2e-7 |
| fillet radius (R00, T00) | 4 | 7.0e-8 | 4.1e-5 | 4.7e-8 |
| fillet radius | 5 | 1.9e-7 | 1.8e-7 | 1.5e-7 |
| sinusoid amplitude (R00, T00) | 4 | 1.5e-8 | 6.1e-9 | 1.2e-8 |
| sinusoid amplitude | 5 | 1.6e-8 | 2.0e-9 | 2.5e-8 |
| LC director angle in a circle (R00, T00, Re / Im J_xy) | 4 | 5.4e-9 | 7.5e-10 | 2.6e-8 |
| LC director angle | 5 | 1.0e-8 | 1.4e-9 | 2.5e-8 |
| eps of the circle (R00, T00) | 4 | 8.7e-9 | 3.0e-9 | 4.9e-9 |
| eps | 5 | 1.2e-9 | 1.5e-9 | 8.3e-9 |
| depth (R00, T00) | 4 | 1.9e-8 | 1.9e-8 | 1.9e-8 |
| depth | 5 | 1.8e-8 | 1.8e-8 | 1.8e-8 |
| circle radius, CONICAL (theta 0.3, phi 0.4) | 5 | 6.1e-8 | 3.5e-5 | -- |
| circle radius, conical | 4 | see below | see below | -- |

The AD VALUES of the two builds agree to every printed digit (7).  Against
FD(twin) the gradient is correct to the FD's own residual (1e-9 .. 2e-7).
Against FD(numpy) the affine cases (rect) and the material / depth / angle
cases agree equally well; the CURVED geometric cases carry the frozen-grid
offset of 1.2 (9.1e-5 circle, 4.1e-5 fillet at `M = 4`, falling to 1.9e-6
/ 1.8e-7 at `M = 5`, 7.8e-8 circle at `M = 6`): the derivative of a
different but converging parametrisation, not a gradient error.  Unit
readings at `M = 3` (`e3_unit_readings.json`): circle radius 6.7e-12, rect
width vs NumPy 1.0e-11, eps (Re, Im) 5.4e-10 / 6.2e-10, depth 4.6e-13.

The conical circle at `M = 4` is the one case whose standard ladder never
reached its asymptotic range (rung changes falling by 0.9, 0.1, 9.0 -- not
`h^2`): T00(r) carries a sharp feature within ~1e-4 P of `r0` there, which
the NumPy solve shows identically (FD(twin) and FD(numpy) agree rung by rung
to 4e-5).  The finer ladder `f3b_fine_ladder.py` (`h / P = 1e-4 .. 1e-7`)
converges onto the AD like `h^2` (relative gap 3.2e-2, 3.5e-3, 3.1e-4,
3.4e-5, 3.6e-6, 2.0e-7): the gradient is right, and the case is a reminder
that the FD premise must be checked, not assumed.

### 3.4 E3-4 -- the non-differentiable events

`f4_events.py 4`.  Smooth side: the twin frozen at `r0 = 0.36`, the radius
moved over `[0.30, 0.42]` (+-17 %): AD vs FD(twin) 1.1e-11 .. 3.4e-8 at every
point; the value differs from the NumPy solve built at that radius by 3.9e-6
.. 7.4e-6 (`r != r0`) and 2.1e-14 (`r = r0`) -- the frozen-grid offset again.

| event | traced (jit): value / gradient | concrete value | control inside the topology |
|---|---|---|---|
| fold: the circle bulging past the cell edge (r 0.62) | NaN / NaN | refused (the merge: the shape must lie inside) | r 0.58: finite, accepted |
| tangency: a rectangle's walls snapped onto the circle's 45-degree walls at r0, r 0.37 separates them | NaN / NaN | refused (TOPOLOGY: wall owners change) | r 0.36: finite, accepted |
| sliver: a rectangle's outer segment below 1e-3 P (w 1.199) | NaN / NaN | refused (the merge's sliver contract) | w 1.19: finite, accepted |
| Duffy count: a fillet radius 1e-5 (traced) / 0 (concrete) -- the four singular vertices vanish | NaN / NaN | refused (TOPOLOGY) | r 0.05: finite, accepted |

Two boundary cases from the Phase C verifier's report (`VERIFY_PMM2D_CURVED_
C_2026_10_03.md` on `verify/pmm2d-curved-c`, its V-D2 and V-D1),
`f4b_verifier_cases.py 3`: a ROTATED ellipse (semi-axes 0.35 / 0.2) frozen
at 0.2 rad -- the merge refuses it as a fold from ~0.34 rad on (scan in the
JSON) -- traced to 0.36 gives NaN / NaN, a concrete 0.36 is refused, 0.28 is
finite; and two rectangles sharing one wall computed two ways (aimed at
V-D1's round-off coincidence; on this layout the sums agree and the merge
reaches the identity): the twin at the reference equals NumPy to 4e-15, one
width traced 1e-6 away separates the shared wall (NaN), a concrete one is
refused.  **V-D1 itself** (a rectangles-only merge missing the identity at
round-off on ~7.5 % of layouts and then running the mapped path) is being
fixed by a 1e-13 vertex snap in `_merge` in the E2 build; every rectangular
fixture here has bit-exact vertices (checked: all take the unmapped path
without the snap), so E3-1 and the parity gates do not depend on that fix
landing first, and the twin's own tolerance for a merged coincidence
(1e-9 of the period) already absorbs a 1-ulp one.

The NaN is multiplicative (`R * where(ok, 1, nan)`): a first version used
`jnp.where(ok, R, nan)`, which hands the cotangent to the finite branch and
returned a SILENT ZERO gradient at the event (measured: 0.0 at the fold and
the tangency) -- caught by this gate, fixed.

### 3.5 E3-5 -- degenerate eigenvalues

`f5_degenerate.py 4`: the centred circle's layer pencil carries 37 and the
homogeneous geometric pencil 41 eigenvalue pairs closer than 1e-10
relative (four-fold symmetry).

| `tau_rel` (eigenvector-VJP broadening) | d T00 / d cx, centred (exact: 0) | d / d r vs FD, rel. (R00 / T00) | off-centre d / d cx vs FD, abs. |
|---|---|---|---|
| 1e-14 | -7.9e-14 | 6.5e-10 / 2.0e-12 | 1.0e-11 / 9.8e-11 |
| 1e-12 (default) | -7.1e-14 | 6.5e-10 / 2.0e-12 | 1.0e-11 / 9.8e-11 |
| 1e-10 | -5.2e-14 | 6.5e-10 / 2.0e-12 | 1.0e-11 / 9.8e-11 |
| 1e-8 | -5.5e-14 | 4.1e-10 / 1.8e-10 | 1.0e-11 / 9.7e-11 |
| 1e-6 | -2.3e-14 | 2.4e-6 / 1.8e-6 | 3.7e-9 / 6.1e-9 |

The symmetry-breaking derivative through the degenerate modes is the
correct zero; the broadening is harmless up to 1e-8 and biases at 1e-6 (an
O(tau) error).  The MUTATION `jnp.linalg.eig` in place of
`_jax_eig_stable`: JAX 0.10 / 0.11 refuse eigenvector derivatives of a
non-symmetric matrix (`NotImplementedError`) -- caught.  JAX's opt-in
UNREGULARISED derivative (`enable_eigvec_derivs=True`) returns the correct
gradients here as well (d T00 / d cx -2.6e-14, d T00 / d r equal to the
regularised to 1.7e-13, `f9_mutations_M4.json`): the outputs are
gauge-invariant functions of the operator and LAPACK splits the degenerate
pairs at round-off, so the regularisation is a safety margin on this
fixture rather than a correction (finding F-E3-4) -- not in general: with
`tau = 0` a NON-degenerate rectangle's width gradient is off by 0.117
(round-off-split pairs of the homogeneous geometric eig), and no
regularisation recovers a symmetry-BREAKING gradient at a symmetric cell
(verifier V-E3-1, 0.3 - 29 %; this section's d / d cx is zero by mirror
parity, so it could not see that).  Fixed in round 2 (section 9.1).

### 3.6 E3-6 -- jit compiles once

`f6_jit.py`: a Python trace counter inside the solved function and jit's
cache size both read 1 after forward calls at four radii and gradient calls
at four radii (`M = 4`, `5`).

| M | template | jit fwd compile + run | jit grad compile + run | jit fwd | jit grad | eager fwd | NumPy solve |
|---|---|---|---|---|---|---|---|
| 4 | 1.3 s | 20.1 s | 37.6 s | 0.26 s | 0.36 s | 70 s | 1.3 s |
| 5 | 1.3 s | 15.9 s | 28.8 s | 0.60 s | 0.85 s | 41 s | 3.0 s |
| 6 (f1) | 8.6 s | 25.5 s | 62.7 s | 2.8 s | 5.4 s | -- | 26.9 s |

Bounds: a compiled forward runs in 0.1-0.2x the time of the NumPy solve of
the same cell, a compiled gradient (forward + reverse) in 0.2-0.3x of ONE
NumPy forward; the EAGER twin (op-by-op dispatch of ~3 200 traced
operations) is 13-54x slower than NumPy -- use `jax.jit`.

### 3.7 E3-7 -- a 1-D sanity: two independent differentiable solvers

`f7_stripe.py`: the stripe `Rect(1.2 - w/2, 0.6, w, P)` (ridge `[0.6,
1.2]`, spanning the period in y, a 2 x 2 grid), `eps 4`, depth `0.4`.
Against the 1-D PMM (degree 40): its JAX twin's AD for eps and depth, the
converged FD of its NumPy solve for the width (neither 1-D twin traces a
wall).  Relative differences:

| M | TM (E_x): R00 value / d w / d eps / d depth | TE (E_y): R00 value / d w / d eps / d depth |
|---|---|---|
| 6 | 2.6e-4 abs / 1.7e-3 / 2.9e-2 / 6.3e-3 | 5.3e-3 abs / 7.4e-2 / 8.7e-2 / 1.2e-1 |
| 7 | 2.6e-4 abs / 1.7e-3 / 2.9e-2 / 6.3e-3 | 4.0e-5 abs / 9.2e-4 / 8.4e-4 / 1.3e-3 |
| 8 | 1.0e-4 abs / 9.6e-4 / 1.0e-2 / 2.3e-3 | 4.0e-5 abs / 9.2e-4 / 8.4e-4 / 1.3e-3 |

The gradients converge onto the 1-D solver's with the 2-D discretisation
(the twin equals the NumPy 2-D solve to round-off, E3-2), in the PAIRED
steps the staggered basis shows on this stripe (`M = 6 -> 7` TM and `7 ->
8` TE leave the value unchanged to 1e-14; F-E3-5).  The TM d / d eps is the
worst-conditioned entry (the gradient itself is small, 4.3e-3).  Unit bars
at `M = 7`: TM 1e-1, TE 1e-2 (readings 2.9e-2 / 1.3e-3).

### 3.8 E3-8 -- the census

Unit test: spies on the 17 `twod_staggered` kernels, two `shapes2d`
kernels, four `rcwa._core` algebra kernels, `rcwa._jax_eig_stable`,
`Granet2DTransverseE._assemble` and `TransfiniteMap._gh`; ONE traced-shape
twin solve (a circle in a two-layer stack with a uniform LC layer, oblique)
reaches every one of them; the twin module defines none of their names.

### 3.9 E3-9 -- the mutation matrix

`f9_mutations.py 4` (the circle, `M = 4`; the clean twin vs NumPy: 2.3e-15 /
1.2e-14 / 3.1e-14):

| mutation | effect | caught by |
|---|---|---|
| the cofactor dropped in the JAX far field (traced path only) | R 5.5e-3, T 6.6e-2, J 1.6e-2 off | E3-2 (bar 5e-12) |
| the frozen far-field node count re-read inside the trace | `TracerArrayConversionError` at trace | the jit gate (E3-6 / E3-9 test) |
| the forward-branch gauge decided on the host from traced data | `TracerArrayConversionError` at trace | the jit gate |
| `_jax_eig_stable` -> plain `jnp.linalg.eig` | `NotImplementedError` (JAX refuses) | E3-5 |
| the eig regularisation over-broadened (`tau_rel = 1e-6`) | d / d r biased by 2.9e-9 (M = 3) .. 2.4e-6 (M = 4) | E3-5 (bar 1e-10) |
| the topology / fold poison removed | the fold returns T00 = 0.503, finite and wrong | E3-4 (NaN expected) |

A test-design note (F-E3-6): `jax.jit` caches the compiled executable per
FUNCTION object, so a mutation patched in after a function was compiled is
invisible to that function; every mutation arm builds a fresh function.

---

## 4. Findings

* **F-E3-1 -- the frozen grid is a re-parametrisation, exact for affine
  cells, converging for curved ones.**  1.2, 3.3, 3.4,
  `f8_frozen_vs_moving.json`: away from the reference a curved shape is
  solved on a grid whose cells have the right physical images but different
  `(u, v)` lengths; the L2 incident projection weights cells by `(u, v)`
  area, so the solution differs from the NumPy solve built at that parameter
  at the incident representation level (5.5e-6 on T at `M = 4` for a 0.03 P
  radius change; the derivative offset 9.1e-5 -> 1.9e-6 -> 7.8e-8 at
  `M = 4, 5, 6`); for the rectangle 1.3e-14 at a width change of 0.03.  For affine cells
  (rectangles) at NORMAL incidence the two coincide to round-off (at
  oblique incidence they differ at the incident representation level,
  verifier V-E3-2, section 9.4).  Not a defect; documented in the
  CHANGELOG's limits.
* **F-E3-2 -- compile size.**  2.3: 19 076 -> 3 194 traced operations,
  gradient compile 213.8 s -> 28.3 s at `M = 4`.
* **F-E3-3 -- a `jnp.where` NaN guard returns a silent zero gradient.**
  3.4: fixed with a multiplicative poison.
* **F-E3-4 -- the eig regularisation is a margin here, not a correction.**
  3.5 -- on THAT fixture; not in general (tau = 0 breaks a non-degenerate
  rectangle by 0.117), and it cannot carry a symmetry-breaking gradient at
  a symmetric cell (V-E3-1; round 2 replaced it there by the
  degenerate-cluster rule, section 9.1).
* **F-E3-5 -- paired convergence steps of the staggered basis on a 1-D
  stripe** (NumPy solver property, measured): `M = 5 / 6` give the same TE
  R00 to 1e-14 and `M = 6 / 7` the same TM R00, at 0.7 % / 0.4 % from the
  1-D solver; the next rung jumps to 4e-5 / 2.5e-3.  Recorded for the
  maintainer -- the plan's convergence ladders step `M` by one.
  EXPLAINED by the verifier (section 9.7): a parity selection rule.
* **F-E3-6 -- jit caches per function object** (3.9).
* **F-E3-7 -- eager is slow.**  3.6: 40-70 s for one eager forward at
  `M = 4 / 5`; every documented use and every test uses `jax.jit`.

---

## 5. Scope decisions

* IN: the shared-grid in-plane path -- scalar, block-form tensor and
  magnetic cells, uniform and patterned layers, multilayers, with and
  without a map (explicit `cmap=` and every shape primitive), normal,
  oblique and conical incidence (forward; gradients measured at normal
  incidence and at conical for the circle radius, section 3.3 note).
* OUT, refused naming the follow-up: OUT-OF-PLANE tensors and `slant`
  (Phase E1: their first-order generator and generalized cascade have no
  twin; the plain `_select_forward_flux` and the parity gauge
  `_stag_parity_gauge` belong to that path and are not traced at all);
  `layer_grids='per-layer'` (Phase E2); `retain_internal` /
  `layer_absorption`; a traced or changed superstrate index at oblique
  incidence; traced period / wavelength / angles.  None of them was cheap:
  each needs the out-of-plane generator, the mortar or a traced Bloch glue.
* The convenience entry rebuilds the template on every call (seconds of
  NumPy at `M = 6`); loops use the stack.

## 6. What moved

Nothing shipped moved: E3-1, 156 / 156 hashes.  New: `backend=` on
`PMM2DStackPure` and `pmm_jones_2d_staggered` (default `'numpy'`),
`PMM2DStackPure.jax_twin()` / `jax_params()` / `solve(params=)`,
`reference_shapes=`; shape primitives, edge curves and transfinite maps
accept JAX values.

## 7. Not measured

* FD(numpy) on the second build (WSL ran AD vs FD(twin) for every case and
  the unit tests).
* GPU / TPU (`jnp.linalg.eig` is CPU-only).
* Gradients at oblique incidence for every case (circle radius only).
* Second derivatives (the custom-VJP eig is first-order only).
* Idle-box wall times.

## 8. Reproduction

```
cd /c/tmp/lum_curved_e3/validation/probe_pmm2d_curved/build_e3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved_e3
# E3-1 (PRE tree = git archive eae470d9 in C:/tmp/curved_pre_e3)
(cd /c/tmp/curved_pre_e3 && PYTHONPATH=C:/tmp/curved_pre_e3 python C:/tmp/lum_curved_e3/validation/probe_pmm2d_curved/build_e3/e1_bytes.py C:/tmp/curved_pre_e3 pre)
python e1_bytes.py C:/tmp/lum_curved_e3 post ; python e1_compare.py
python f1_feasibility.py rect|circle M          # M = 4, 5, 6
./runjobs.sh jobs_win.txt 4                     # f2 .. f7, f9 (jobs_win*.txt)
python f3b_fine_ladder.py 4 ; python f4b_verifier_cases.py 3
python f8_frozen_vs_moving.py ; python f10_compile_size.py 4 jit
./runjobs_wsl.sh jobs_wsl.txt 3                 # second build -> wsl/
python f3_summary.py ; python e3_unit_readings.py
cd /c/tmp/lum_curved_e3 && python -m pytest tests/unit/test_pmm2d_staggered_curved_e3.py --capture=sys -p no:randomly
```

Test tails:

* Windows (CPython 3.14.6, numpy 2.4.4, scipy 1.17.1, jax 0.11.0), serial:
  `tests/unit/test_pmm2d_staggered_curved_e3.py` -> `27 passed in 305.42s`
  (`_tests_e3_final_win.log`; slowest 39.6 s, the eager circle parity;
  durations spliced into `.test_durations`).
* Windows, existing suites, `-n 5` (every `*pmm2d*`, `*stack2d*`,
  `*stagger*`, curved A-D, `verify_pmm2d*`, every `*jax*` file, public API,
  except budget, re-exports, history lint / relocation, doc identifiers,
  walker `__all__`, CI kernel consistency): `2 failed, 1717 passed, 9
  skipped in 1043.47s` -- the two are the history fingerprints of
  `stack2d_pure` (its AST changed; re-recorded with the reason, `--check`:
  "every history document matches its module").
* WSL Ubuntu (CPython 3.12.3, numpy 2.4.6, scipy 1.17.1, jax 0.10.2, BLAS
  pinned, `lumenairy` from `/mnt/c/tmp/lum_curved_e3`): the E3 file plus
  `test_v5_20_2_pmm_jones_2d_jax.py`, `test_v5_14_2_jax_stacks.py`,
  `test_backend_disable_jax.py`, `test_audit_w3_pmm_jax_guards.py` and
  Phase D's file: `93 passed in 242.92s` (`wsl/_tests_final_wsl.log`).
* Curved A-D after the `xp=` parametrisation (Windows, `-n 6`): `71 passed`.
* `python -m mypy` (the configured strict list): `Success: no issues found
  in 33 source files`.  `lumenairy/elements/pmm/` is not on that list;
  `mypy _jax_twod_staggered.py` reports only `no-untyped-def` /
  `no-untyped-call` (the package's convention), the genuine findings
  (annotations, re-exported private names, a variable reused with two
  types) were fixed.
* WSL ruff 0.15.16 on `lumenairy/ tests/ scripts/` and `build_e3/`: `All
  checks passed!`
* The API examples (CHANGELOG and 2.5) run as written at `n_modes = 3`:
  `f11_api_examples.json`.

---

## 9. Round 2 (VERIFY-E3), 2026-10-03

Builder: Claude Opus 5.5 (model ID `claude-opus-5-5`).  Object: the
verifier's report `docs/audits/VERIFY_PMM2D_CURVED_E3_2026_10_03.md`
(branch `verify/pmm2d-curved-e3`, tip `51aa1fdf`), whose recommendation was
"do not ship as is" because of one P1 (V-E3-1).  Worktree
`C:/tmp/lum_curved_e3b`, branch `feat/pmm2d-curved-e3-round2` from
`51aa1fdf`.  PRE trees: `git archive eae470d9` (NumPy bytes) and
`git archive d4e92eb5` (the E3 build, forward bytes of the twin).  Builds:
Windows 11 (CPython 3.14.6, numpy 2.4.4, jax 0.11.0) and WSL Ubuntu
(CPython 3.12.3, numpy 2.4.6, jax 0.10.2); BLAS threads 1 on every command
line, `lumenairy.__file__` asserted under the tree in every probe.
Evidence: `validation/probe_pmm2d_curved/build_e3r2/` (suffix `_win` /
`_wsl` = build).  Words: AD, FD(twin), FD(numpy) and the `h^2` premise as in
section 0 and the verifier's section 0; "rule on / off" = the
degenerate-cluster rule of 9.1 switched on (default) or off
(`_E3_EIG_CLUSTER_GAP_REL = 0`, which is exactly the E3 build's adjoint).

### 9.0 What round 2 changed

| item | verdict of round 2 | where |
|---|---|---|
| V-E3-1 (P1), wrong gradient for a symmetry-breaking parameter at a symmetric cell | FIXED: a reverse-mode rule for degenerate eigenvalue clusters that wraps the eigen-solves AND everything downstream of them; forward values byte-identical | 9.1, 9.2 |
| the same defect class in the library's other JAX twins | MEASURED: the RCWA JAX path and the 1-D PMM twin are affected (pre-existing), the hybrid 2-D PMM twin is not; recorded as maintainer items, pinned by two strict xfails | 9.3 |
| V-E3-3 (P3), the inside-the-cell contract not replayed in the trace | FIXED: the verifier's guard, plus a trace-safe `Ellipse.bbox` it needed | 9.5 |
| V-E3-2 (P2), rectangles at oblique incidence | DOCUMENTED (not changed, as advised); the convergence rate of the route difference is pinned | 9.4 |
| V-E3-4 (P3, documentation) | APPLIED; the convenience entry names itself when JAX is off | 9.8 |
| the global 5e-12 parity bar | RESTATED per fixture | 9.6 |
| F-E3-5 | EXPLAINED (a parity selection rule), campaign note in the plan's section 3 | 9.7 |
| O-1 (pre-existing, the unmapped route's gauge-dependent incident split) | RECORDED for the maintainers, not changed | 9.9 |

### 9.1 V-E3-1: why no rule at the eig boundary can be right, and the rule that is

**The derivation.**  Write the solve as `L(A) = h(eig(A))`: `A = G^-1 L` is a
pencil, `h` is everything downstream of its eigenpairs `(lam, V)`, and `h`
is invariant under a change of basis inside a degenerate cluster (the
S-matrix cascade is).  In eigen-coordinates the first-order change is
`dL = sum_ij (V^-1 dA V)_ij B_ji`, with `B` the consumer's sensitivity.  The
cotangent that `h` hands to `eig` carries `lam_bar_i = B_ii` and, because
`h` is basis-invariant inside the cluster, NOTHING of the off-diagonal
`B_ij` for `i != j` in one cluster: those entries enter `h` only through its
response to a SPLITTING of the cluster.  A parameter that keeps the symmetry
has `(V^-1 dA V)` proportional to the identity on the cluster, so the
missing entries do not matter; a parameter that breaks it does not, and the
gradient is then wrong by a term that depends on the basis LAPACK happened
to pick (hence build-dependent).  Degenerate perturbation theory says the
same: the in-cluster coupling `V_c^-1 dA V_c` must be diagonalised and `h`
evaluated in that basis, which depends on `dA`.  Proof by example
(`test_e3r2_no_eig_level_rule_can_see_the_in_cluster_block`): for
`A = diag(1, 1, 2)` the consumers `Re tr(V e^Lam V^-1 X)` with `X = E_12`
and with `X = 0` hand `eig` the SAME (zero) cotangent, yet their true
gradients (`jax.scipy.linalg.expm`) differ by `e`.  So the rule the round-2
brief asked for -- a cluster-aware VJP written inside `_jax_eig_stable` or
the twin's eig wrapper, from `(lam_bar, V_bar)` alone -- cannot exist; the
rule has to see the consumer.

**The rule** (`rcwa._core._jax_eig_cluster_adjoint`, used by
`StagJaxTwin.solve`, whose solve is now split into its eig PROBLEMS -- the
shared geometric pencil when the map is traced, one per distinct patterned
layer -- and their CONSUMER, a closure over everything else):

* forward: the plain composition `consumer(eig(A_1), eig(A_2), ...)`;
* reverse, when no eigenvalue gap is below `gap_rel = 1e-6` of `max|lam|`:
  the standard VJP of the consumer at the eigenpairs, then
  `_jax_eig_stable`'s (the E3 build's gradient);
* reverse, with a cluster: the standard VJP of eig + consumer at LIFTED
  matrices `A + t d N`, `t = -+1, -+2`, combined by Richardson
  (`(4/3) avg(+-1) - (1/3) avg(+-2)`, error `O(d^4)`), with
  `N = sum_c Q_c Y_c Q_c^H G P_c` -- `P_c` the cluster's spectral projector,
  `Q_c` a `G`-orthonormal basis of the cluster, `Y_c` the traceless,
  unit-RMS compression of a fixed random Hermitian matrix -- and
  `d = 1e-7 max|lam|`, shortened near the consumer's branch points (below).
  The cotangents of the consumer's other inputs come from the exact point.

What each ingredient is for (each was found necessary by a measurement,
9.11): `N` maps into the cluster and annihilates every other eigenvector,
so the eigenpairs outside the clusters are EXACTLY unchanged; it is built
from projectors, so it does not depend on the basis inside a cluster
(gauge invariance, 9.2); its cluster block is similar to the Hermitian
`Y_c`, so the lift moves eigenvalues along the REAL axis and the
forward-branch selector downstream (`sqrt(g2)` onto the forward branch,
decided from the sign of `Im`) picks the same branch at every lifted point;
`G`-orthonormality keeps that true for a cluster of a Hermitian pencil that
merges several distinct eigenvalues; traceless unit RMS splits a pair into
exactly `-+ d` and moves no member of a `k`-cluster more than `sqrt(k) d`
(far below `gap_rel`); clusters are the CONNECTED COMPONENTS of the pairwise
relation (transitive closure in the trace); a lift that comes out
non-finite is dropped (the stencil's weights sum to one, so that eig then
gets the plain VJP).

**The constants.**  `gap_rel = 1e-6`: the W9 envelope of the plain VJP is
2.5e-9 at a relative splitting of 1e-6 (and 4.9e-6 at 1e-9), so a pair
closer than that is treated as a cluster.  `d = split_rel max|lam|` with
`split_rel = 1e-7` and the order-4 stencil (`r2_dev_*_scan*.json`): the
order-2 average showed the expected `d^2` truncation on the square
(2.4e-5 / 2.4e-7 / 2.3e-9 at `d` = 1e-5 / 1e-6 / 1e-7, M = 3); with order 4
the error is flat at the FD's floor from `3e-8` to `3e-7` at M = 3 .. 6 and
rises as `eps_mach / d` below it (5.0e-8 at `1e-8`, M = 5); `1e-7` sits in
the flat range and keeps `2 sqrt(4) d = 4e-7` below `gap_rel` (`3e-7` would
not).  The
anchors: a cluster at distance `a` from the nearest branch point of its
consumer (`g2 = 0` for a patterned layer, `g2_geo = -eps` of every
homogeneous region for the geometric pencil) is lifted by at most
`0.25 a (eps_mach max|lam| / a)^(1/5)`, the order-4 balance of truncation
`(d / a)^4` against round-off; the square fillet's exactly degenerate pair
sits at `2e-5 max|lam|` from the layer cutoff at `M = 3` and `2e-6` at
`M = 5` (`r1c_branch_M*.json`), where the uncapped lift cost 2.8e-7.

### 9.2 Before and after, both builds

AD vs FD at the symmetric reference, max relative over R00, T00 E_x,
T00 E_y (`r2_dev_<case>_M<M>_final_{win,wsl}.json`; FD(numpy) and FD(twin)
are the verifier's premise-checked ladders where they exist, else the twin's
own Richardson FD with its premise recorded; rule off = the E3 build):

| case | M | before (rule off) win / wsl | after (rule on) win / wsl | vs FD(numpy), after, win |
|---|---|---|---|---|
| square pillar d / d w | 3 | 3.1e-3 / 3.8e-2 | 1.9e-10 / 4.4e-10 | 1.9e-10 |
| | 4 | 2.5e-1 / 6.9e-2 | 4.9e-10 / 1.3e-9 | 4.2e-10 |
| | 5 | 1.4e-1 / -- | 1.6e-9 / -- | 1.1e-9 |
| | 6 | -- | 2.1e-9 (own FD) / -- | -- |
| circle -> ellipse d / d a | 3 | 6.5e-3 / 2.8e-2 | 8.6e-11 / 1.8e-11 | 5.1e-4 (frozen-grid offset) |
| | 4 | 3.0e-3 / -- | 5.1e-11 / -- | 7.9e-5 (frozen-grid offset) |
| square fillet d / d w | 3 | 2.9e-1 / 7.5e-1 | 4.1e-9 / 4.0e-9 | 2.0e-5 (frozen-grid offset) |
| | 4 | -- | 6.3e-10 (own FD) / -- | -- |
| | 5 | -- | 1.0e-8 (own FD, premise 10.4) / -- | -- |
| control: square, w and h together | 3 | 4.1e-11 / 4.3e-11 | 4.1e-11 / 4.3e-11 | 1.3e-11 |
| control: non-square pillar d / d w | 3 | 3.4e-11 / 3.4e-11 | 3.4e-11 / 3.4e-11 | 3.0e-11 |

For the curved cells FD(numpy) differs from FD(twin) by the frozen-grid
offset the verifier characterised (its section 2), and the twin's AD sits on
FD(twin) to the FD's own floor.  The gate is therefore AD vs FD(twin) for
curved cells and AD vs FD(numpy) for the rectangle (whose offset is zero at
normal incidence); the verifier's strict xfail on the square pillar
(`< 1e-6` vs FD(numpy)) now passes with five decades to spare.

The near-symmetric sweep (ellipse `b = a (1 + delta)`, d / d a,
`r6_offsym_M3_{win,wsl}.json`), rule off -> rule on:

| delta | 0 | 1e-14 | 1e-13 | 1e-12 | 1e-11 | 1e-10 | 1e-9 .. 1e-4 |
|---|---|---|---|---|---|---|---|
| win | 1.8e-3 -> 4.5e-11 | 1.9e-2 -> 2.4e-11 | 5.3e-2 -> 6.7e-11 | 2.7e-4 -> 6.6e-11 | 3.2e-6 -> 9.2e-11 | 2.5e-9 -> 8.1e-11 | <= 1.2e-10 both |
| wsl | 3.5e-3 -> 7.0e-11 | 2.3e-3 -> 3.6e-11 | 5.6e-3 -> 5.5e-11 | 8.2e-5 -> 6.0e-11 | 2.3e-7 -> 1.1e-10 | 8.2e-10 -> 1.9e-11 | <= 7.4e-11 both |

**Gauge invariance** (`r5_gauge_<case>_M<M>_{win,wsl}.json`): the
eigen-solver is replaced by one whose basis inside every EXACT cluster
(gap <= 1e-12) is a random unitary rotation of LAPACK's, with a random phase
on every mode (two seeds).  Relative change of the gradient:

| case | rule off win / wsl | rule on win / wsl | forward change |
|---|---|---|---|
| square, M = 3 | 0.23, 0.17 / 0.24, 0.11 | 1.6e-10, 3.3e-10 / 4.3e-10, 3.4e-10 | <= 2.6e-15 |
| square, M = 4 | 0.27, 0.35 / -- | 7.8e-10, 2.1e-9 / -- | <= 5.1e-15 |
| ellipse, M = 3 | 2.7e-2, 0.24 / 7.0e-3, 0.28 | 6.6e-11, 7.3e-11 / 5.7e-11, 3.3e-11 | <= 2.7e-15 |
| square fillet, M = 3 | 0.52, 2.8 / 0.23, 0.30 | 9.6e-10, 1.0e-9 / 8.5e-11, 1.2e-10 | <= 2.7e-13 |

That V-E3-1 WAS this dependence is the reading of the left column; the
right column is the rule's own round-off.  On the matrix-function oracle
(`r9_matrix_oracle_{win,wsl}.json`, a random similarity of
`diag(1, 1, 2, 3)`, `Re tr(expm(A + t B) X)`): the rule 2.2e-9 / 1.6e-10,
the plain VJP 0.30 / 1.01 (win / wsl).

**Forward bytes** (`r3_fwd_bytes_*`, `r3_compare.py`): R, T and the Jones
matrix of 14 fixtures (the E3-2 set, a uniform-only stack, the square
pillar, the symmetric ellipse, the oblique rectangle), each eager at the
reference, eager with a JAX shape parameter and under `jax.jit`: 84 / 84
SHA-256 equal to `d4e92eb5` at M = 3 on both builds and at M = 4 on
Windows.  NumPy bytes: the round-2 diff touches the NumPy modules only in
`Ellipse.bbox` (an `xp` switch that is `numpy` for concrete values) and in
the `backend='jax'` branch of `pmm_jones_2d_staggered`.

### 9.3 The library's other JAX twins

One symmetry-breaking gradient at a symmetric configuration per twin, AD vs
a premise-checked NumPy Richardson FD (`r4_other_twins_<case>_{win,wsl}.json`).
The round-2 rule is NOT used by these twins (it needs each twin's
eig consumer as a function), so their gradients are unchanged by round 2 --
these readings are the same before and after.

| twin (entry) | parameter at the symmetric point | win | wsl | symmetry-keeping control |
|---|---|---|---|---|
| RCWA JAX path (`rcwa_efficiency_2d`, JAX `eps_cell`) | 15 x 15 pixels: centre 4, sides 1.5, corners 1; t added to the two x-side blocks | **23 % (TE) / 39 % (TM)** | **28 % / 47 %** | 1.5e-10 .. 6.9e-10 (t on all four side blocks) |
| 1-D PMM twin (`pmm_efficiency_1d`, JAX) | d / d(angle) at EXACTLY 0 (no wall position is traced in this twin), R and T of the +-1 orders, duty 0.5 | **28 % (TE) / 590 % (TM)** | **28 % / 590 %** | -- |
| hybrid 2-D PMM twin, cell path (`pmm_efficiency_2d_cell`, JAX `eps_cell` + `region_layout`) | 3 x 3 regions, same cell, x-side regions | 3.8e-11 / 8.5e-11 | 4.2e-10 / 3.0e-10 | 1.4e-10 .. 7.3e-10 |
| hybrid 2-D PMM twin, pillar entry (`pmm_efficiency_2d`) | d / d theta at 0, (+-1, 0) orders, centred square | 3.3e-10 / 3.5e-10 | 3.4e-11 / 1.0e-10 | -- |

The RCWA JAX path is the V-E3-1 class (pre-existing, P1 for a user who
differentiates a symmetric pixel cell in a symmetry-breaking direction) and
build-dependent; the 1-D twin's is the "exactly 0.0 stays unrecoverable"
case the W9 note of `_jax_eig_stable` documents.  The hybrid 2-D twin is
correct in both directions measured.  Since the fix is local to the
staggered twin, these are MAINTAINER ITEMS (9.9), pinned by
`test_e3r2_rcwa_jax_symmetry_breaking_gradient_at_a_symmetric_cell` and
`test_e3r2_pmm1d_jax_angle_gradient_at_normal_incidence` (`xfail(strict=True,
raises=AssertionError)`: they flip loudly when a maintainer routes those
twins through the rule), and stated in the CHANGELOG's limits.  The
library's other users of `_jax_eig_stable` (BOR, BOR-SEM, EME modes,
Berreman, the PMM stack twins) were not measured.

### 9.4 V-E3-2 -- rectangles at oblique incidence (documented, rate pinned)

Not changed, as the verifier advised (copying the unmapped least-squares
route would import its gauge dependence, O-1).  CHANGELOG: the parity claim
is qualified ("at normal incidence for every fixture, and at oblique /
conical incidence for every fixture except a rectangles-only `shapes=`
stack ... converge with `n_modes`") and the verifier's limits bullet is
added; F-E3-1 and 1.2 carry the at-normal-incidence qualifier; the module
docstring's "to round-off at normal incidence" for curved cells is replaced
by the measured 5.5e-6.  Gate
`test_e3r2_oblique_rectangles_converge_at_the_measured_rate`: the
difference at M = 3 must lie in [1e-5, 1e-3] (2.5e-4) and fall at least 3x
from M = 3 to 4 (measured 6.6x) and 10x from 4 to 5 (measured 58x);
`geometry='static'` reproduces the NumPy route to 1e-12 (8.5e-15).

### 9.5 V-E3-3 -- the traced inside-the-cell guard

The verifier's 15-line guard is applied verbatim in
`_traced_shape_merge`.  It calls `bbox()` on the traced shapes, and
`Ellipse.bbox` was NumPy-only (`np.hypot` on a tracer raised); it is now
array-module generic (NumPy for concrete values, so the NumPy statements are
the shipped ones).  The verifier's event probe re-run on this tree
(`verify_e3/v5_events_M3_r2_{win,wsl}.json`): all eight cases NaN at the
event (value, sum, d / dx, d / d eps) and finite at the control; its strict
xfail flips; `test_e3r2_an_ellipse_inside_the_sliver_margin_is_poisoned_when_traced`
covers the ellipse.

OBSERVED ONCE, not reproduced (WSL, jax 0.10.2): the first WSL run of the
event probe on this tree (before the connected-components fix) completed the
first seven cases and then the interpreter DUMPED CORE (`timeout: the
monitored command dumped core`, no Python traceback) during or right after
the eighth case, `two_rects_reorder` (two rectangles, the second's centre
traced from 0.8; control 0.62, event 0.45 where the walls reorder).  The
Windows suites were running at `-n 8` at the time.  Not reproduced in three
later runs: the eighth case alone, rule on and rule off, twice
(`r12_rects_crash.py`, `r12_rects_crash_wsl.json`: control value 0.895467,
d T00 / dx 0.224934 with both rules, event NaN, the NumPy stack ACCEPTS the
event at T00 0.880131 -- a new topology, as the verifier recorded), and the
full probe once more under `python -X faulthandler` (all eight cases, exit 0,
no fault, `logs/v5_r2_wsl_run1.log`).  No attribution is possible from one
uncaught native crash: it is a crash below Python (XLA, LAPACK or the
allocator), not an exception of the twin; a memory exhaustion would have
been an OOM kill, not a core dump, and the host had 76 GB free when checked
minutes later.  Recorded as observed; a recurrence should be re-run under
`faulthandler` (`run_wsl_v5.sh`).

### 9.6 Per-fixture parity bars

The E3-2 unit gate is restated per fixture (`_PAR_BAR` in the test file):
30x the fixture's eig-stage round-off REACH (the NumPy stack with its QZ
replaced by the standard eig of `G^-1 L`, minus the shipped stack, max over
R, T, J), the larger of the two builds, rounded up
(`r7_parity_reach_{win,wsl}.json`).  The twin's difference sits at 0.25 ..
2.9x the reach on every fixture, both builds, so the factor leaves >= 10x;
bars 4e-14 (sinusoid) .. 3e-12 (fillet).

### 9.7 F-E3-5 explained

The verifier's section 5: on a cell whose sub-cells are centred on mirror
planes, at normal incidence, the excited `E_x` and `E_y` are even about
those planes; raising `M` by one adds one polynomial per cell of
alternating parity, which is orthogonal to the excited sector every other
step -- a selection rule, not convergence (TE pairs 3/4, 5/6, ...; TM pairs
4/5, 6/7, ... to <= 1e-13; oblique incidence in the mirrored direction
breaks the pairing).  Recorded as a campaign-wide note at the head of the
plan's section 3: judge M-ladders on pairs of rungs (`M -> M + 2`).

### 9.8 V-E3-4 -- the documentation corrections

CHANGELOG "Known limits": reverse mode only (`jvp` / `jacfwd` / `hessian`
raise `TypeError`, nested `grad` `NotImplementedError`); the degenerate-mode
bullet (which claimed the correct gradient) is replaced by the measured
rule, its cost and the other twins' readings; the oblique-rectangle bullet;
`LUMENAIRY_DISABLE_JAX` makes both the stack and
`pmm_jones_2d_staggered(backend='jax')` raise naming themselves (the entry
now checks before delegating; gated in
`test_e3_disable_jax_switch_and_x64_are_honoured`).  This record: 1.2 and
F-E3-1 (affine at normal incidence), 3.2 (the "2.9x" is an observation, not
a bound -- the verifier reached 4.0x, 6.9x per quantity -- and the global
bar is restated, 9.6), 3.3 (the `h^2` law held for every case except eps at
M = 4 and the conical circle at M = 4), 3.5 and F-E3-4 (the regularisation
is a margin on that fixture only: tau = 0 breaks a non-degenerate rectangle
by 0.117, and no regularisation carries a symmetry-breaking gradient).  The
twin module's docstring: the frozen-grid paragraph (the verifier's text)
and a paragraph on the reverse pass through degenerate eigenvalues.

### 9.9 Maintainer items

No "mode-pick" item exists in the curved-cell records; this list is new.

* **O-1 (pre-existing on `eae470d9`, not changed).**  The UNMAPPED route
  decomposes the incident wave by a minimum-norm least squares on an
  underdetermined Rayleigh system, which depends on the per-mode
  normalisation of `W0`: rescaling its columns moves R / T by 3.5e-5 /
  4.2e-6 / 9.9e-8 at M = 3 / 4 / 5 (0.3 rad; 1e-15 at normal incidence), so
  a 1e-12 width change moves R / T by up to 7.6e-7 and breaks FD ladders on
  that route (verifier `o1_gauge_*.json`).  The mapped route is gauge-free.
* **The degenerate-cluster rule for the other JAX twins.**  The RCWA JAX
  path (23 - 47 %) and the 1-D PMM twin at exactly normal incidence
  (28 - 590 %) return wrong symmetry-breaking gradients (9.3); routing
  their eig problems and consumers through
  `rcwa._core._jax_eig_cluster_adjoint` is the fix measured here, and the
  two strict xfails flip with it.  The other `_jax_eig_stable` users were
  not measured.
* **F-E3-5**: M-ladders on pairs of rungs (9.7).

### 9.10 What the rule costs

Compiled wall times, best of 5 after the compile, run SERIALLY on Windows
(`r11_timing_M{3,4,5}_win.json`; the box was shared, so these are upper
bounds).  The circle's d / d r keeps the symmetry, but its eigs carry exact
clusters, so the lifted branch runs there too:

| case | M | gradient, rule off | rule on | ratio | compile off / on | forward ratio |
|---|---|---|---|---|---|---|
| square d / d w | 3 | 0.033 s | 0.106 s | 3.2 | 5.9 / 12.6 s | 0.98 |
| | 4 | 0.091 s | 0.359 s | 4.0 | 6.2 / 13.5 s | 1.04 |
| | 5 | 0.302 s | 1.513 s | 5.0 | 6.6 / 15.7 s | 1.05 |
| circle d / d r | 3 | 0.038 s | 0.110 s | 2.9 | 7.5 / 14.2 s | 1.02 |
| | 4 | 0.110 s | 0.363 s | 3.3 | 8.7 / 15.8 s | 0.87 |
| | 5 | 0.374 s | 1.401 s | 3.8 | 8.3 / 17.8 s | 1.01 |

The reverse pass with a cluster evaluates eig + consumer + its VJP at four
lifted points (the order-4 stencil), hence 3 - 5x; the forward pass is
unchanged; a cell without any cluster pays nothing (the plain branch runs
from the forward pass's residuals; under `jax.vmap` both branches of the
`lax.cond` run).  An order-2 stencil would halve the cost at the price of
the fillet's near-cutoff truncation (2.8e-7 instead of 4e-9).

### 9.11 Development record (what each ingredient of the rule answers)

Each of these was a measured failure of an earlier draft, kept so the
design is not re-derived:

1. A complex random lift (`R` Gaussian) moved real eigenvalues off the axis;
   the forward-branch selector flipped modes at the lifted points -> a
   relative error of 6.6e2 on the square (`r2_dev_square_w_M3_win.json`, first run).  Hence
   real shifts.
2. A Hermitian `H` does not give real shifts for the patterned layer, whose
   pencil is NOT Hermitian (anti-Hermitian part 0.33 relative,
   `r1b_herm_M3_win.json`) -> the fillet broke (0.14 .. 71).  Hence the
   compression `Q_c^H H Q_c` in an orthonormal basis of the cluster.
3. An unnormalised compression moved members by up to `2 sqrt(n) d` and
   collided them with eigenvalues outside the cluster at M = 5 (errors 5 -
   20).  Hence traceless unit RMS.
4. With the Euclidean Gram, a cluster that merges two distinct exact pairs
   (the geometric pencil at M = 5) got complex shifts of 1e-9 max|lam|,
   inside the branch selector's band (`r1d_clusters_M5_1e-07_win.json`).
   Hence the `G`-Gram.
5. The fillet's near-cutoff exact pair (2e-5 max|lam| from `g2 = 0` at
   M = 3) set the stencil's truncation (2.8e-7 at the uncapped lift).  Hence
   the order-4 stencil and the anchors.
6. Re-running the verifier's event probe found a NaN gradient at a fillet
   far from its reference: chains of near-degenerate eigenvalues made a
   pairwise mask non-transitive and a masked Gram indefinite.  Hence the
   connected components and the finite guard
   (`r10_fillet_far_from_reference_win.json`, commit `81f3019a`).

### 9.12 Tests, tails, commits

Logs in `validation/probe_pmm2d_curved/build_e3r2/logs/`.

* Windows, SERIAL (durations): `test_pmm2d_staggered_curved_e3.py` +
  `test_verify_pmm2d_curved_e3.py`: `47 passed, 2 xfailed in 777.59s`
  (`serial_e3_win.log`; the two xfails are the maintainer pins of 9.3); the
  durations of both files spliced into `.test_durations` (37 entries
  replaced by 49, order and format kept).
* Windows, `-n 8`, half A (the verifier's list -- E3, curved A-D, every
  `verify_pmm2d*` file, every `*jax*` file -- plus the W9 eig-VJP and W6
  Berreman audits): `802 passed, 8 skipped, 2 xfailed, 41 warnings in
  785.01s` (`suite_A_win.log`).
* Windows, `-n 8`, half B (the verifier's: every other `*pmm2d*` /
  `*stack2d*` / `*stagger*` file, census, public API, walkers,
  doc-consistency, doc identifiers, except budget, re-exports, history
  lint / relocation / fingerprint tool, kernel consistency): `1548 passed,
  8 skipped, 88 warnings in 625.75s` (`suite_B_win.log`).  Its documentation,
  CHANGELOG, history, census, walker and re-export files re-run after the
  final documents, `-n 4`: `924 passed, 7 skipped in 169.16s`.
* WSL, `-n 4` (E3, the verifier's file, curved A-D, `pmm_jones_2d` JAX, the
  JAX stacks, disable-JAX, the PMM-JAX guards, W9, the RCWA 2-D OOP JAX
  file): `2 failed, 199 passed, 2 xfailed in 510.84s` (`suite_wsl.log`):
  (a) `test_e3r2_near_symmetric_cells_are_inside_the_rule`, its FAIL-BEFORE
  arm: the E3 build's adjoint read 1.1e-4 under the old bar `> 1e-3` (the
  probes had read 5.6e-3 on WSL, 5.3e-2 on Windows) -- that size is
  arbitrary by the nature of the defect, so the bar was re-derived to
  `> 1e-5` (the rule's arm passed); (b) the xdist worker running
  `test_v5_20_2_pmm_jones_2d_jax.py::test_pmm_jones_2d_jax_li_formulation_and_three_regions`
  CRASHED inside XLA's `backend_compile_and_load` (compiling a subtraction
  in `_jax_twod_jones._kz_fwd`, a path round 2 does not touch) -- the second
  native crash of jax 0.10.2 on WSL this day (9.5).  Both re-run serially
  on WSL: `15 passed in 51.81s`; the re-derived gate on Windows: `1
  passed`.
* `python -m mypy` (configured strict list): `Success: no issues found in
  33 source files`.  WSL ruff 0.15.16 on `lumenairy/ tests/ scripts/` and
  `build_e3r2/`: `All checks passed!`
* `scripts/record_history_fingerprints.py --check`: `OK: every history
  document matches its module.` -- nothing re-recorded (the only history
  document among the touched modules' neighbours, `stack2d_pure`, is
  untouched by round 2).  No forward version token.

Commits (branch `feat/pmm2d-curved-e3-round2`, not pushed): `9acb0850`
(the degenerate-cluster adjoint, V-E3-1), `f8bbbbc4` (the traced
inside-the-cell guard, V-E3-3), `81f3019a` (clusters are connected
components), `94c5ac9c` (per-fixture parity bars, the oblique-rate pin, the
entry names itself), and the documentation commit carrying this section.

### 9.13 Not measured

* The rule on the second build above M = 4 (WSL ran M = 3 / 4 for the
  square, M = 3 for the ellipse and the fillet).
* FD(numpy) for the fillet's symmetry-breaking gradient above M = 3 and for
  the square above M = 5 (the twin's own FD was used at M = 4 .. 6).
* A degenerate cluster AT (or within round-off of) a consumer branch
  point: its anchor cap shrinks the lift to round-off, so that cluster gets
  in effect the plain VJP; such a point is a cutoff of the discretisation
  (the Wood nudge keeps the half-spaces off it) and was not constructed.
* The other `_jax_eig_stable` users (BOR, BOR-SEM, EME, Berreman, the PMM
  stack twins).
* GPU; idle-box timings.

### 9.14 Reproduction

```
cd /c/tmp/lum_curved_e3b/validation/probe_pmm2d_curved/build_e3r2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=C:/tmp/lum_curved_e3b
python r1_spectrum.py M ; python r1b_herm.py M ; python r1c_branch.py M ; python r1d_clusters.py M SPLIT
R2_NAME=_final ./runjobs.sh jobs_win_final.txt 6        # r2_dev, r5_gauge, r6_offsym, r3, r9
python r2_dev.py M CASE GAP SPLIT [GAP SPLIT ...]        # the scans: R2_ORDER=2|4, R2_NAME=_scan*
python r3_fwd_bytes.py M post ; (LUM_TREE=<git archive d4e92eb5> ...) r3_fwd_bytes.py M preE3 ; python r3_compare.py preE3 post M
python r4_other_twins.py rcwa2d|pmm2d|pmm1d|pmm2d_theta
python r7_parity_reach.py ; python r10_nan_debug.py ; python r11_timing.py M
wsl bash run_wsl.sh jobs_wsl_final.txt 5 [TREE]         # second build
```

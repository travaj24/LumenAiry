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
rectangle (affine cells, an exactly representable incident) the two
discretisations coincide to round-off at every width (3.5e-15 at `w = 0.47`
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
difference 2.1e-14 (`f2_parity_M3.json`).  The twin sits within 2.9x of
the eig stage's own round-off on every fixture: its differences from NumPy
ARE that stage's round-off plus the summation order of the vectorised
quadrature (operators <= 6.1e-14 relative).  Unit bar **5e-12**: two decades
above the `M = 3` maximum, 8x above the largest `M = 4` reach; the
cofactor mutation (E3-9) moves the mapped T by 6.6e-2.

### 3.3 E3-3 -- gradients against converged central differences, both builds

`f3_grads.py CASE M [numpy]` through the public API (traced shape objects
in `params`).  FD(twin) on the ladder `h / P = 3e-3 .. 3e-5`; the
rung-to-rung changes fall by 8.8, 11.4, 8.8 for every case (the `h^2` law at
a step ratio of 3 is 9) -- the PREMISE that the FD is in its asymptotic
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
fixture rather than a correction (finding F-E3-4).

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
  (rectangles) the two coincide to round-off.  Not a defect; documented in
  the CHANGELOG's limits.
* **F-E3-2 -- compile size.**  2.3: 19 076 -> 3 194 traced operations,
  gradient compile 213.8 s -> 28.3 s at `M = 4`.
* **F-E3-3 -- a `jnp.where` NaN guard returns a silent zero gradient.**
  3.4: fixed with a multiplicative poison.
* **F-E3-4 -- the eig regularisation is a margin here, not a correction.**
  3.5.
* **F-E3-5 -- paired convergence steps of the staggered basis on a 1-D
  stripe** (NumPy solver property, measured): `M = 5 / 6` give the same TE
  R00 to 1e-14 and `M = 6 / 7` the same TM R00, at 0.7 % / 0.4 % from the
  1-D solver; the next rung jumps to 4e-5 / 2.5e-3.  Recorded for the
  maintainer -- the plan's convergence ladders step `M` by one.
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

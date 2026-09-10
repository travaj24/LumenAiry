# `probe_verify_mortar` — independent verification probes

Reproducers for `docs/audits/VERIFY_PMM2D_STAGGERED_MORTAR_2026_09_11.md`, the
adversarial verification of the two coupled features built in
`docs/audits/BUILD_PMM2D_STAGGERED_MORTAR_2026_09_11.md`:

1. **non-uniform segment boundaries** in the staggered basis
   (`Basis1D(d, walls, M, tau)`, `Granet2DTransverseE(px, py, wx, wy, ...)`),
2. **per-layer element grids** through an L2 mortar
   (`PMM2DStackPure(layer_grids='per-layer')`).

Nothing here reads a number out of the build doc and checks it: every script
RE-MEASURES, with its own fixtures wherever the claim allows it, and with the
build's own fixture only where the claim is *about that fixture's numbers*
(V5).  Every script asserts which tree it imported (`lumenairy.__file__`).

## Running

```
cd /c/tmp/lum_vmortar
PYTHONPATH=/c/tmp/lum_vmortar OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_verify_mortar/<script>.py [sections]
```

Each script takes its section names as positional arguments (no arguments =
all sections) and appends its results to a JSON beside it.

| script | sections | what it measures |
|---|---|---|
| `v1_bit_identity.py` | `stack`, `pencil` | 21 stack fixtures on the SHARED path + 102 `Basis1D` matrix hashes + 192 assembled-pencil hashes (`Rmat`/`Lmat`/`Stt`/`Schur`/`Agen`/`Bgen`, scalar / in-plane tensor / out-of-plane / gyrotropic / magnetic / slanted).  Run it TWICE — once with `PYTHONPATH` on this tree and `V1_TAG=with`, once with `PYTHONPATH` on the read-only main clone (`a68a0da`) plus `V1_TAG=without V1_EXPECT_ROOT=<that clone>` — then `v1_compare.py with without`. |
| `v1_compare.py` | — | hash-by-hash diff of the two arms. |
| `v2_basis.py` | `scal`, `derham`, `fail`, `invar`, `parity` | the three physical scalings against an independent physical-space quadrature oracle and two analytic identities; `d(Btilde) subset span(B)` on arbitrary walls; the two-sided fail-before for each of the four `J -> J_n` sites; MIRROR and CYCLIC-TRANSLATION device invariances; the FIFTH site (`_stag_parity_1d`), including forcing the reduction past it. |
| `v2_oracles.py` | `oracle1d`, `hybrid`, `farfield` | arbitrary-wall cells against the exact 1-D `PMMStack`, against `PMM2DStackHybrid` (exact walls) and against `pmm_efficiency_2d_cell` (pixel walls), per order; the y-momentum leak and the projector's own quadrature error on non-uniform segments. |
| `v2_quad.py` | `kernel`, `device` | the far-field projector's FIXED per-segment quadrature order `nq = 2M + 8` on non-uniform segments — kernel-level error vs an 8x-refined rule as a function of the longest segment and the highest order, and whether it moves a device answer. |
| `v3_mortar.py` | `g3`, `ident`, `nonconf`, `cap`, `cache`, `absorb`, `oracle1d` | the H-row V1/V2 swap and the BLINDNESS of a conforming gate to it; the conforming identity against a `10 eps cond_2(G)` bar measured here (uniform AND non-uniform grids); transparent splits and non-conforming pairs vs the common refinement; the order cap; eig-cache collision attempts; the absorption budget; the 1-D per-order oracle with the anti-mirror control. |
| `v4_slant_mixed.py` | `bypass`, `oracle`, `split`, `mixed` | THE SLANTED PER-LAYER GATE the build listed as an open item: bit-exactness on coinciding grids, and a y-uniform slanted grating SPLIT across NON-conforming grids scored per order against `pmm_efficiency_1d_slanted` with the wrong-sign arm as the two-sided control.  Plus OOP-over-scalar-over-magnetic on three grids. |
| `v5_equal_dof.py` | `stripe`, `staircase`, `pillar` | the equal-DOF ratios, re-measured with reference ladders built here, INCLUDING the rungs where the mortar loses. |
| `v6_taper_sliver.py` | `taper`, `intra`, `intra2`, `cross` | tapers against the hybrid staircase and against a fine shared lattice; and THE SLIVER QUESTION on the mortar route — `cross` (walls differing BETWEEN layers, the route the sliver FIX doc calls safe) and `intra`/`intra2` (two walls `delta` apart INSIDE ONE layer's own grid, which no prior document considers). |
| `v7_cost.py` | `xmass`, `device` | factored vs dense cross-mass memory and apply speed; per-layer vs shared wall time and peak RSS on the corner-dominated pillar pair. |

## The two reproducers worth keeping

* **`v6_taper_sliver.py intra2`** and the `intra4` variant described in the
  verify doc: a layer whose OWN grid carries two walls `delta` apart, with the
  NEIGHBOURS on different grids.  From `delta ~ 1e-3` down the answer stops
  converging to the `delta`-independent truth and wanders by up to 7.8e-03 in
  absolute `R(0,0)` while the lossless closure stays pinned at 1.3e-05 — and at
  `delta = 1e-7` `_interface_smatrix_mortar_2d` raises a bare
  `numpy.linalg.LinAlgError: Singular matrix` from an UNGUARDED
  `np.linalg.solve`.
* **`v2_quad.py kernel`**: the projector's fixed `nq = 2M + 8` Gauss rule,
  measured wrong by up to 7.5e-04 relative when one segment is 0.96 of the
  period and the retained order reaches 7.

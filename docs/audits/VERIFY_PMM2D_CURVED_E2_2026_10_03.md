# VERIFY -- curved cells for the pure staggered 2-D PMM, Phase E2 (per-layer curved maps joined by the non-separable curved mortar)

Date: 2026-10-03.  Verifier: Opus 5.5 (claude-opus-5-5), independent of the
builder.  Worktree `C:/tmp/lum_vcurved_e2`, branch `verify/pmm2d-curved-e2`,
built on `a785c511` (Phase E2 tip: `cbacae71 .. a785c511` on Phase D
`eae470d9`, Phase C verifier branch merged).  PRE tree: `git archive eae470d9`
extracted to `C:/tmp/vcurved_e2_pre`.  Two builds: Windows 11 (CPython
3.14.6, numpy 2.4.4, scipy 1.17.1) and WSL Ubuntu (CPython 3.12.3, numpy
2.4.6, scipy 1.17.1); BLAS pinned to one thread on every command line;
`lumenairy.__file__` asserted under the tree being measured in every probe
(`verify_e2/_ve.py`).  The box was shared with two sibling verifiers and
other jobs throughout (25-38 Python processes, 99-100 % CPU), so every wall
time is an upper bound; no accuracy number depends on the load.  Nothing
under `lumenairy/` was edited; mutations were applied in-process only.

Evidence: every number below is read from a JSON (or log) file in
`validation/probe_pmm2d_curved/verify_e2/`; the probe that wrote it is
named.  Files are suffixed `_win` / `_wsl` per build.  Prefixes: `v1_`
bytes (item 1), `v2*` the kernel (item 2), `v3m_` the decisive merged-map /
RCWA check and `v3g_` own geometries (item 3), `v4_` the limitation (item
4), `v5_` the defaults (item 5), `v6_` V-D3 (item 6), `v7_` near-singular
vertices (item 7), `v8_` mutations (item 8), `v9_` the Phase C fold-in (item
9), `v10_` docs (item 10).  Decision tests:
`tests/unit/test_verify_pmm2d_curved_e2.py` (6 ids, ~32 s serial on the
loaded box).

---

## 0. Verdict

| item | claim | verdict | where |
|---|---|---|---|
| 1 | E2-1: no map = today's bytes (134/134) | **CONFIRMED** on an own set of 191 keys (65 per-layer mortar keys), both builds, plus 535/535 in-process captures of the 8 shipped mortar test files; trap mutant: unmapped + shared-map suites green, E2 file red | 1 |
| 2 | the cross-mass kernel (formula, H row, exact cuts, spectral) | **CONFIRMED with two defects**: formula and H-row re-derived and confirmed against an own physical brute force to 2e-15 (singularity-free pairs) and to the oracle's own floor (~2e-9) on circle pairs; the H-row identity holds to 1e-16; **V-E2-D1** a grazing cut is silently missed (up to 6.4e-7); **V-E2-D2** crossing closed curves mostly refused | 2 |
| 3a | E2-2 same map through the forced mortar ~1e-15 | **CONFIRMED** (own ellipse / off-centre circle / fillet: 0.8e-15 .. 5.0e-15; H-swap off 0.63-1.05) | 3.1 |
| 3b | E2-3 separable stretches = 1-D factorisation 1.3e-14 | **CONFIRMED** (own derivation + own 1-D oracle: 8.2e-16; stack converges to the physical device) | 3.2 |
| 3c | E2-4 crossing pairs: closure ladders, absorption, vacuum identity | **CONFIRMED as stated**, but closure is NOT an accuracy measure (**V-E2-D3**) | 3.3 |
| 3d | decisive: per-layer vs an independent answer for the crossing device | **CONFIRMED -- converges to the physical answer**: against a hand-built CONFORMING merged map and RCWA the per-layer error is 1.1e-1 / 3.0e-2 / 1.4e-2 / 1.6e-3 / 1.0e-3 / 9.6e-4 / 7.2e-4 at M = 4..10 (closure understates it by 2.2-5 decades) | 3.4 |
| 3e | E2-6 reciprocity (1.0e-6 / 7.3e-6 at M = 7) | **CONFIRMED in class** on an own pair (7.3e-7 / 4.9e-6 at M = 7) | 3.3 |
| 4 | limitation: algebraic where an outline cuts the neighbour's cells, attributed to the rim singularity | **CONFIRMED (rim singularity through a non-conforming cut)**: tail p_M = 1.8-2.2 at eps 4 / 1.1 / 1.02, size proportional to delta-eps; but the build's per-layer-vs-merged column is mostly the merged map's own error -- the per-layer stall shows only beyond M = 8 (~3e-4 vs 1.8e-5) | 4 |
| 5 | q-matching and riding defaults | **CONFIRMED useful, tested ONE-sided** (closed by `test_ve2_6`); the riding direction does not matter beyond the discretisation level; **V-E2-D10** `convergence_floor` is not map-aware (0.14 off, or raises) | 5 |
| 6 | V-D3 (hybrid merge for same-layer supercells) | **DEFERRED, not closed**; "put them in two layers" is NOT an honest workaround (a different device: 0.36-0.51 in R/T from the one-layer macro-cell, never converging to it) | 6 |
| 7 | nearly coincident singular vertices: 96 nodes, RuntimeError at 1.2e-6 | **REPRODUCED, scope WRONG**: the failure starts at 5e-5 .. 7e-5 (not 1.2e-6), on concentric/near-tangent outlines (which a per-layer stack DOES route through the mortar), the error is a bare message, the cap warning understates its error 9x | 7 |
| 8 | E2-8 + mutations | **CONFIRMED** (6.7e-2, 2.6e-15, fast path refused) + **test gaps**: the tangency search off and the square-root substitution off SURVIVE all 12 E2 ids; closed by the decision tests | 8 |
| 9 | V-D1 snap, V-D2 ellipse corners | **CONFIRMED with a residual window** (V-D1 snap 1e-13 < wall merge 1e-12: (1e-13, 1e-12] p still maps; R/T 2.6e-6) | 9 |
| 10 | docs, CHANGELOG, roadmap, gates | **PARTIAL**: 242/243 table numbers match; a handful of number/wording defects; mypy, ruff, fingerprints, gates green | 10 |

**Ship recommendation: SHIP E2 after the documentation edits of section 11
(D3, D2 scope, D7, D9 numbers) and with the two pinned defects (D1, D2)
recorded as known limitations; the kernel fix for D1 is small and should go
in with the integration.**  The per-layer curved mortar is correct (formula,
H row, flux consistency, reciprocity, absorption, byte identity of every
shipped path) and converges to the physical answer of a crossing device
that no single map in the library could build; its per-rung accuracy is the
own-walls-only class (1e-2 at M = 6, 1e-3 at M = 8-10 on the E2-4 device),
which the documentation must state instead of the closure numbers.

---

## 1. Item 1 -- E2-1, byte identity (sub-verifier, `v1_*`)

* Own fixture set `v1_bytes.py`: **191 keys** (operator blocks of every
  dispatch branch, separable cross-mass factors at oblique Bloch phases,
  far projectors, shared stacks with retain_internal / absorption /
  magnetic / out-of-plane / slant, **65 per-layer mortar keys**: lossy
  non-conforming pairs at normal / oblique / conical incidence, non-uniform
  walls with a lossy uniform layer, an out-of-plane tensor under the
  mortar, slant layers on different `n_modes`, a `mu_cell` layer, a uniform
  tensor layer, forced conforming, taper, convergence_floor, the shipped
  test files' own builders loaded by path; Phase C/D mapped shared solves).
  PRE vs POST: **191 / 191 Windows, 191 / 191 WSL** (`v1_compare.json`).
* A second layer, `v1_capture_plugin.py`, hashes in-process every output of
  the solve / mortar functions while the 8 shipped mortar test files run
  unchanged: **535 / 535** identical (Windows).
* Fail-before: one ulp on an eps entry changes the shared and the per-layer
  conical hashes (dR 2.0e-15 / 1.3e-15); the identity map through the
  quadrature path changes the operator hashes.
* Trap mutant (`v1_trap_plugin.py`: `curved_cross_mass`,
  `curved_cross_mass_adaptive`, `StagCrossOpsMapped.__init__` raise):
  mortar group A `52 passed`, group B `35 passed, 1 skipped`, curved a/b/d +
  verify a/b/c `70 passed`, curved_c `21 passed`, trap fired 0 times; the
  E2 file `12 failed` (trap fired 12).
* PRE raises for ANY per-layer shape stack (rectangles too), so those are not
  byte fixtures; in POST a rectangles-only per-layer shape stack is
  byte-identical to `'shared'` (fast path), both builds.

---

## 2. Item 2 -- the cross-mass kernel

### 2.1 Re-derivation

Unknowns are covariant (1-form) components `E' = J^T E`, `H' = J^T H`, `J =
d(x, y)/d(u, v)`.  Between an upper layer on `Phi_a` and a lower one on
`Phi_b`:

* **E row** -- a's tangential E pulled into b's coordinates is
  `E'_b = J_b^T J_a^-T E'_a = T^T E'_a` with `T = J_a^-1 J_b`; it is
  L2-projected onto b's E space in b's PLAIN `du_b dv_b`.  Because the
  projection is tested with b's H functions placed as the Eq.-25 dual (E1
  pairs with H2 in V1, E2 with H1 in V2), this is the metric-free flux
  pairing `INT (E_a x h_b*) . z dx dy` (the cross product of covariant
  components carries `1 / det J`, cancelled by `dx dy = det J du dv`):

      X_{beta alpha} = INT conj(f^b_beta) T_{alpha beta} f^a_alpha du_b dv_b
                     = INT conj(f^b_beta) T_{alpha beta} f^a_alpha / det J_b dx dy.

* **H row** -- b's tangential H pulled into a's coordinates,
  `H'_a = S^T H'_b`, `S = J_b^-1 J_a = T^-1`, L2-projected onto a's H space
  (H1 in V2, H2 in V1) in a's PLAIN `du_a dv_a`:

      CrossH_{kl} = INT conj(g^a_k) S_{lk} g^b_l / det J_a dx dy.

  Written in b's coordinates the weight is `det(T) T^-T` = the COFACTOR of
  `T`; with the H1/H2 placement in V2/V1 and the sign of the flux pairing
  this is exactly `[[X22^H, -X12^H], [-X21^H, X11^H]]` (checked entry by
  entry: e.g. row H1 / column H2 carries `S_21 = -T_21 / det T`, i.e.
  `-X12^H`).  So the E-type and H-type blocks DO carry different Jacobian
  factors (T / det J_b against S^T / det J_a), and the kernel carries them
  through the adjugate identity rather than a second integral -- which is
  legitimate because the identity is algebraic.  The shipped code
  (`_curvemortar._add_pair`: `T` in b's measure, `adj(J_b^-1 J_a)` in a's
  measure; `cross_h_from_x`) implements exactly this.

### 2.2 The independent brute force (`v2_brute.py`)

An iterated integral in PHYSICAL coordinates: outer `y`, inner `x`; the
inner line cut at every crossing with every interior wall of BOTH maps
(own root-finding on the forward maps, per monotone piece of every wall);
outer events at every vertex image, every y-extremum of a wall (`y = y0 + L
xi^2` substitution), every wall-wall intersection of the two maps (exact
polyline intersection + 2-D Newton) and every singular vertex (geometric
grading, ratio 0.15, 18 levels, inner grading toward the vertex); each
sub-interval located in both maps once, every node inverted by an own
damped Newton from an own sample table.  The H row is integrated DIRECTLY
with its own formula (2.1), not derived from X.  Library pieces used: the
forward maps (`geom_points`) and the basis stencils (the definition of the
basis); local Legendre functions are evaluated with numpy's `legval`.

Results (max |kernel - brute| / max |brute|, M = 4 unless stated; the
kernel at its adaptive node count):

| pair | E row | H row (own formula) | H identity applied to the BRUTE X vs brute H | fail-before: H without the off-diagonal cofactor | JSON |
|---|---|---|---|---|---|
| sinusoid x-wall / sinusoid y-wall (crossing, no singular vertex) | **2.0e-15** (n 20), 1.6e-15 (n 30), 3.2e-15 (n 40) | 2.0e-15 | 1.8e-16 | 8.8e-2 | `v2_brute_sinx_siny_M4_*` |
| same, M = 5 | 6.3e-16 | 5.4e-16 | 9.0e-17 | 8.8e-2 | `v2_brute_sinx_siny_M5_n24_*` |
| same map both sides (sinx/sinx, siny/siny) | 6.3e-15 / 5.6e-15 | -- | -- | -- | `v2_brute_sin?_sin?_*` |
| sinusoid / unmapped wall TANGENT to its crest | **3.3e-15** | 3.4e-15 | 2.2e-16 | 1.3e-1 | `v2_brute_sx_tan_*` |
| sinusoid / unmapped wall grazing the crest by 1e-3 | 2.9e-15 | -- | -- | -- | `v2_brute_sx_graze3_*` |
| sinusoid / unmapped wall grazing the crest by **1e-6** | **5.56e-9** (n 16 / 24 / 32 / 44 all 5.5574e-9: the ORACLE is converged) | 5.56e-9 | 1.1e-16 | -- | `v2_brute_sx_graze6_*` |
| circle / crossing sinusoid (the build's pair) | 1.7e-9 (n 20) / 2.3e-9 (n 28) | same | 1.4e-16 / 2.9e-16 | 2.2e-1 | `v2_brute_circ_sin_*` |
| sinusoid / circle (reversed) | 3.7e-9 | same | 2.9e-16 | 2.7e-1 | `v2_brute_sin_circ_*` |
| circle / sinusoid, complex Bloch tau (e^0.7i, e^-0.4i) | 1.7e-9 | same | 1.4e-16 | 2.2e-1 | `v2_brute_circ_sin_tau_*` |
| circle / unmapped wall tangent to the circle bottom | 5.4e-10 | same | 1.5e-16 | 1.1e-1 | `v2_brute_tangent_*` |
| circle / unmapped wall tangent at the 0-deg point | 3.1e-9 | same | 1.6e-16 | 9.9e-2 | `v2_brute_tan_x_*` |
| circle / wall grazing by 1e-3 | 5.4e-10 | same | 1.5e-16 | 1.1e-1 | `v2_brute_graze3_*` |
| circle / wall grazing by **1e-6** | **6.4e-7** (block 22) | same | 1.5e-16 | 1.1e-1 | `v2_brute_graze6_*` |

* On pairs WITHOUT singular vertices the kernel equals the oracle to
  round-off, E row and H row, both builds (WSL `v2_brute_sinx_siny_*_wsl`:
  identical class).
* On CIRCLE pairs the agreement (5e-10 .. 4e-9) is the ORACLE's floor, not
  the kernel's: the oracle moves by the same amount between n = 20 and
  n = 28 (its physical-coordinate grading toward the square-root singular
  vertices is the weak point); the kernel's own n-ladder there is spectral
  to 1e-14 (`v2b_tangent_*`), and the singularity-free pairs pin the cut
  logic itself.  **I could not reach 1e-12 against an independent oracle on
  pairs with singular vertices** -- recorded under "not measured".
* The H-row identity holds on the BRUTE X to 1e-16 in every case: the
  cofactor rule is exact.

### 2.3 Tangency, grazing, the square-root substitution (`v2b_tangent_ladder.py`, `v2d_graze_sweep.py`)

Kernel n-ladder (n = 6, 9, 12, 18, 27, 40, 60) against its own adaptive
value, shipped vs mutants:

| pair | shipped | square-root substitution off | tangency search off |
|---|---|---|---|
| circle / crossing sinusoid | 1.4e-6, 3.2e-10, 5.5e-14, 0, ... (spectral) | identical (no tangency end on this pair) | identical |
| circle / wall tangent to its bottom | 1.8e-5 .. 4.9e-11 .. 8.0e-15 | identical | **3.7e-2 .. 4.3e-2 (does not converge)** |
| circle / wall grazing by 1e-3 | 1.7e-3 .. 3.4e-11 .. 2.0e-14 | **1.2e-5 .. 8.2e-11 at n = 60 (algebraic)** | 1.2e-4 flat |
| sinusoid x / sinusoid y | 5.3e-5 .. 7.6e-14 .. 2.7e-15 | identical | **2.2e-2 .. 2.0e-2 (does not converge)** |

So the tangency breakpoints are load-bearing (2e-2 without them) and the
square-root substitution is load-bearing exactly where a tangency end is
flagged (algebraic 8e-11 at n = 60 without it; on the own stack device of
section 8 it pushes the adaptive rule to its 96-node cap with a warning).

**V-E2-D1 (P2) -- a grazing cut is silently missed.**  `_cell_pieces`
samples each pulled-back wall at `_CURVE_MORTAR_SAMPLES = 65`
Chebyshev-Lobatto points and keeps only the portions that contain a
sample.  A wall that cuts a sliver off a cell between two samples is
dropped entirely (`v2d_graze_sweep_{win,wsl}.json`, the sinusoid crest
grazed by delta):

| delta | sliver length | kernel vs oracle | adaptive change reported | with 8x the samples |
|---|---|---|---|---|
| 0 (tangent) | 0 | 4.7e-15 | 8e-15 | 4.7e-15 |
| 1e-7 | 4.9e-4 | **1.8e-10** | 8e-15 | 1.8e-10 |
| 1e-6 | 1.6e-3 | **5.6e-9** | 8e-15 | 2.4e-15 |
| 1e-5 .. 3e-3 | 4.9e-3 .. 8.5e-2 | 1.8e-15 .. 3.1e-15 | 8e-15 | same |

On the circle bottom grazed by 1e-6 the miss is **6.4e-7** of the
cross-mass.  The adaptive rule cannot see it (the cut geometry is decided
once, before the node ladder).  The worst case scales like the largest
sliver that fits between two samples (~1.5 % of the edge in the middle of a
Chebyshev set), i.e. up to ~1e-6 relative.  The stack consequence is below
the own-walls-only discretisation level at practical M, but it breaks the
"exact cuts / spectral to 1e-12" claim and nothing would flag it.  Pinned:
`test_ve2_3_grazing_cut_is_not_lost_between_samples` (xfail strict).
A prototype fix applied in-process to a copy of `_cell_pieces`
(`v2e_graze_fix.py`: between samples, refine every local minimum of the
curve's outside-distance within 5 % of the cell and every local maximum
of its inside-distance on a 257-point sub-grid, and add a parameter that
crosses) takes the sinusoid grazes at 1e-7 / 1e-6 to 8.7e-15 / 8.3e-15 and
leaves the circle / crossing sinusoid pair unmoved (0.0); the circle
grazed by 1e-6 stays 6.5e-7 and drives the rule to the cap -- the sliver's
near-tangent crossings also need the square-root substitution.  Edit:
section 11.

**V-E2-D2 (P2, scope) -- crossing CLOSED curves are mostly refused.**
`_assign_pairs` refuses any cell pair that touches a singular (45-degree)
vertex of BOTH maps.  For two crossing circles that is the common case, not
the "nearly coincident 45-degree points" corner the build doc describes:
**46 of 56** sampled crossing circle pairs refused, with the nearest two
singular vertices 0.024 .. 0.21 of the period apart (`v2c_circle_pairs_{win,
wsl}.json`); the own-geometry sub-verifier found 25 / 38 (every equal-radius
and every diagonal offset; `v3g_singmap_win.json`).  The message's remedies
("move one outline so the two maps' singular vertices do not share a cell
overlap, or use layer_grids='shared' ... when the outlines do not cross")
do not apply: the outlines DO cross, and for circles the singular vertices
cannot be moved independently of the outline.  Also over-refused: the SAME
physical circle on a `grid_hint` map against its plain 3 x 3 map
(`_same_cell` needs identical cell rectangles).  Pinned:
`test_ve2_4_two_crossing_circles_have_a_cross_mass` (xfail strict,
`NotImplementedError`).  The fix is a design addition (split such a primary
cell into sub-rectangles each touching one map's singular vertex, and
integrate each in the coordinates of that map), not a one-line edit; the
documentation must say it now (section 11).

---

## 3. Item 3 -- gates on own geometries, and the decisive check

### 3.1 E2-2 (sub-verifier, `v3g_e22_*`)

Same map on both sides, forced through the curved mortar, layer 2 a lossy
film (3.1 + 0.12i), P 1.1, lambda 0.95:

| map, M | normal | (0.33, 0) | (0.33, 1.1) conical | per-layer vs shared | H-swap off |
|---|---|---|---|---|---|
| rotated off-centre ellipse, M 4 | 1.2e-15 | 2.9e-15 | 3.2e-15 | 0 | 0.86 / 1.01 / 1.05 |
| off-centre circle, M 4 | 1.3e-15 | 7.8e-16 | 3.0e-15 | 0 | 0.90 / 1.03 / 0.90 |
| off-centre fillet (5 x 5), M 4 | 3.2e-15 | 2.6e-15 | 3.2e-15 | 0 | 0.68 / 0.63 / 0.75 |
| rotated ellipse, M 6 | 1.0e-15 | 1.3e-15 | 5.0e-15 | 0 | 0.91 / 0.87 / 0.88 |

WSL (circle, M 4): 1.2e-15 / 8.9e-16 / 1.3e-15.  CONFIRMED.

### 3.2 E2-3 (sub-verifier, `v3g_e23_*`)

Both layers stretched on BOTH axes (a: 0.05P, -0.07P; b: -0.04P, 0.11P),
different walls, M_a 5 / M_b 6, complex tau.  `T = diag(psi_x', psi_y')`,
`psi = f_a^-1 o f_b`, so `psi_x'` multiplies the x-integral of the B
functions of component 1 (`X11 = kron(Cy[Btilde], Cx'[B])`) and `psi_y'`
the y-integral of the B functions of component 2 (`X22 = kron(Cy'[B],
Cx[Btilde])`).  Own 1-D Gauss oracle (cut at b's walls and the preimages
of a's): X11 8.2e-16, X22 6.8e-16, X12 = X21 = 0 exactly, CrossH 8.2e-16
(WSL 9.6e-16 / 8.2e-16); without psi' 0.38, psi' on the wrong axis 0.65.
Note: strong stretches need n = 80 (fixed n = 6 / 10 / 14 / 20: 2.4e-2 /
2.1e-3 / 1.0e-4 / 2.7e-6) -- close to the 96 cap.  Stack level: mapped vs
unmapped 8.3e-3 / 1.4e-3 / 3.4e-4 / 2.3e-4 at M = 4..7; the gap is carried
by the stretched layers themselves (each alone, mapped vs unmapped, 2.1e-3 /
5.6e-4 at M = 5 / 6), not by the mortar.  CONFIRMED.

### 3.3 E2-4 / E2-6 on own crossing pairs (sub-verifier, `v3g_e24_*`)

Pairs: (i) eps-3.6 circle r 0.33 at (0.52, 0.58) over a sinusoid x-wall (x0
0.50, A 0.10, phase 0.8, eps 2.6); (ii) eps-3.2 circle r 0.28 over an eps-2.4
circle r 0.20 offset along x (accepted -- see D2); (iii) a FilletRect over a
sinusoid y-wall.  The merge refuses all three ("CROSS in plan view").

| pair | M = 4 | 5 | 6 | 7 | 8 | successive dR/dT at the last rung |
|---|---|---|---|---|---|---|
| (i) closure | 1.9e-3 | 2.4e-4 | 7.6e-5 | 3.9e-6 | 1.6e-6 | 6.2e-4 |
| (ii) closure | 1.2e-3 | 5.6e-5 | 4.3e-5 | 9.2e-7 | 5.0e-7 | 1.1e-3 |
| (iii) closure | 8.8e-6 | 1.4e-7 | 1.1e-9 | -- | -- | 1.0e-3 |

Absorption: a lossless layer next to a lossy one absorbs 1.8e-16 .. 3.3e-14
on every pair (bar 1e-10), sum(A) vs 1 - R - T at the discretisation level.
Vacuum-layer identity: bit-identical (0) at every rung (circle M 4-6, fillet
M 4-5); the no-ride arm 1.8e-3 .. 3.3e-3 from the device alone.
Reciprocity, pair (i), order (-1, 0), singular values / full transposed
matrix: (20, 0) 6.8e-4 / 5.0e-5 / 6.8e-5 / **7.3e-7**; (20, 35) 7.5e-4 /
1.1e-4 / 7.6e-5 / **4.9e-6** at M = 4..7; H-swap off 6e-2 at M = 5.  WSL pair
(i) M 5: 7.4e-14 from Windows.  Gates CONFIRMED; the successive differences
are ~1e-3 at M = 8 while closure is 1e-6 -- see 3.4.

### 3.4 The decisive check -- an independent answer for the crossing device (sub-verifier, `v3m_*`)

The build had no independent reference for a crossing device (its section
7).  Two were built here.

**(A) A conforming merged map by hand** (`v3m_geom.py`): a 4 x 4
`TransfiniteMap` in which BOTH outlines are grid lines -- u walls 0,
0.345442, **0.6** (the sinusoid u-line), 0.854558, 1.2; v walls 0, **0.12**
(a straight dummy line for Nx == Ny), 0.345442, 0.854558, 1.2; the sinusoid
crosses the bottom arc at (0.7173328, 0.2596575) and the top arc at
(0.4826672, 0.9403425) (circle residual 0.0), the arcs split there into two
`Arc` pieces each, the u-line four `Sinusoid` pieces.  Validated by the
constructor; det J > 0 on 201^2 points per cell; `singular_vertices` = the
circle's four 45-degree points; disk area - pi r^2 = -6.7e-16, area right of
the sinusoid - 0.72 = -2.2e-16.  **(B) The shipped 2-D RCWA** with the disk
analytic and the wall area-averaged on a 1200 x 300 raster, extrapolated
from n_orders 16 / 18 (closure 1e-13); merged M 8 and RCWA agree to
4.3e-4.  (A staircase of the shipped unmapped PMM converges ~1/n to the same
answer, a weak limit at 1e-2.)

Per-layer (shipped E2) error against the best reference, normal incidence
(`v3m_compare_th0_ph0_win.json`):

| M | per-layer error | per-layer closure | ratio | per-layer d(M, M+1) | merged-map error | per-layer wall | merged wall |
|---|---|---|---|---|---|---|---|
| 4 | 1.10e-1 | 6.5e-4 | 170 | 9.2e-2 | 9.1e-2 | 6 s | 7 s |
| 5 | 2.97e-2 | 1.8e-4 | 165 | 1.6e-2 | 2.2e-2 | 10 s | 46 s |
| 6 | 1.38e-2 | 2.4e-5 | 575 | 1.35e-2 | 5.0e-3 | 51 s | 185 s |
| 7 | 1.63e-3 | 7.6e-7 | 2100 | 2.2e-3 | 2.2e-3 | 131 s | 421 s |
| 8 | 1.00e-3 | 9.7e-7 | 1000 | 1.5e-3 | 4.3e-4 (ref) | 304 s | 1583 s |
| 9 | 9.6e-4 | 1.7e-8 | 5.6e4 | 1.0e-3 | -- | 442 s | -- |
| 10 | 7.2e-4 | 1.2e-8 | 6.0e4 | -- | -- | 884 s | -- |

Conical (25, 40 deg), against RCWA (`v3m_compare_th25_ph40_win.json`):
9.4e-2 / 3.1e-2 / 1.7e-2 / 3.8e-3 / **6.7e-3** / 1.5e-3 / 1.9e-3 at M = 4..10
(not monotone: T(1, 0) moves 0.1594 -> 0.1502 -> 0.1573 at M = 7, 8, 9).
Per-layer M 4..8 is bit-identical to the build's `e2_4_closure_M*.json`;
WSL M 5: 4.4e-14.

**Verdict: the per-layer curved mortar converges to the physical answer of
the crossing device** (the three routes agree to the references' own ~5e-4),
at ~3.8x per rung for M = 4..7 and slower (~M^-2) beyond; on this device a
conforming merged map is NOT more accurate per rung and costs 5.2x more at
M = 8.  **V-E2-D3 (P2): closure is not an accuracy measure** for the
crossing device -- it understates the R/T error by 2.2 to 4.8 decades; the
rung-to-rung change tracks the error within ~2x.  The build doc's E2-4
section and the CHANGELOG present only closure (and the CHANGELOG example
runs this device at `n_modes=4`, where specular T is 0.418 against 0.528).

---

## 4. Item 4 -- the limitation (sub-verifier, `v4_*`)

**(a) The rate.**  Non-crossing pair (x0 0.12, A 0.05), forced per-layer
against the merged map at M = 10 (`v4_fit_win.json`):

| M | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|
| merged vs merged M10 | 1.05e-1 | 1.01e-2 | 9.75e-3 | 4.34e-4 | 4.14e-4 | 1.8e-5 | -- |
| per-layer (q-matched) vs merged M10 | 1.12e-1 | 2.04e-2 | 1.01e-2 | 5.03e-4 | 2.64e-4 | 3.71e-4 | 2.62e-4 |
| per-layer, q-matching off | 1.49e-1 | 3.73e-2 | 1.51e-2 | 4.60e-3 | 7.99e-4 | 1.04e-3 | -- |

Up to M = 8 the two routes converge TOGETHER (same plateau at M 5-6, same
drop at 7): the build doc's "per-layer vs merged at the same M" column (9.9e-2
/ 1.1e-2 / 9.7e-3 / 4.5e-4) is mostly the MERGED map's own error.  The
limitation appears only beyond M = 8: the merged map reaches 1.8e-5 at M = 9
while the per-layer route stalls near 3e-4.  No clean exponent fits this
pair (log-log RMS 0.65).  The clean instrument is the physical no-op -- a
vacuum spacer on non-conforming walls (0.2 / 0.7) over a square pillar, the
SHIPPED separable mortar only: 3.97e-3 .. 1.09e-3 at M = 4..8, **p_M = 1.83
(q^-1.50), RMS 0.035**; conforming walls: 4.7e-6 -> 3.6e-12 (spectral).

**(b) The cause.**  Efficiencies are quadratic in the amplitude, so the
contrasts were compared on normalised amplitudes of the non-specular orders
(`v4_amp_win.json`, spacer walls 0.2 / 0.7):

| eps | M 4 | M 6 | M 8 | M 10 | tail exponent p_M |
|---|---|---|---|---|---|
| 4 | 1.17e-2 | 4.98e-3 | 3.20e-3 | 2.07e-3 | 1.80 |
| 1.1 | 1.97e-3 | 2.17e-4 | 1.07e-4 | 8.6e-5 | 2.20 |
| 1.02 | 2.19e-3 | 4.14e-5 | 2.15e-5 | 1.69e-5 | 1.91 |

The tail exponent does NOT change with contrast, its normalised size falls
in proportion to delta-eps (30x and 5x at M = 8 for delta-eps ratios 30 and
5), and the contrast-independent part (2.0e-3 / 2.2e-3 at M = 4 -- the trace
of the material jump on a non-conforming cut) is resolved fast (53x in two
rungs at eps 1.02).  **Verdict: the algebraic tail is the RIM (edge)
singularity seen through a non-conforming cut, as the build claims** (a
plain material jump would give a contrast-independent normalised error; it
does not).  The brief's either/or test (same exponent => the cut) does not
separate the two here; the amplitude scaling does.  Low-contrast
non-crossing pair (eps 1.1): per-layer 3.0e-6 .. 2.6e-7 at M = 5..9 against
a merged reference converged to 4e-8 -- the merged map 10x better at M = 7
(2x at high contrast).  WSL rungs agree to <= 2.3e-13.

## 5. Item 5 -- the two defaults (sub-verifier `v5_*`, the lead's mutants, `test_ve2_6`)

**q-matching.**  Documented (module docstring, `_perlayer_modal_counts`
comment, CHANGELOG, build 2.4); the `add_layer` `n_modes` docstring
mentions only the uniform-layer rule.  Tested ONE-sided only: removing it
fails E2-4 (crossing), over-applying it to a layer that named `n_modes`
passes all 12 ids.  The all-vacuum claim reproduces exactly: (6,6) 1.005e-6,
(6,8) 1.009e-6, (8,6) 3.857e-9, (8,8) 2.199e-10 -- but that measurement uses
explicit `cmap=` films with named `n_modes`, to which the default never
applies.  On real devices it helps: non-crossing pair ON / OFF vs merged
M10 1.12e-1 / 1.49e-1 .. 3.7e-4 / 1.0e-3 (1.3-9x better), at 2-3.5x the
wall time; on the crossing pair closure ~10x better (7.6e-7 vs 9.5e-6 at M
7), R/T 2.0e-3 vs 4.5e-3 at M 7.  Note: `q_max` counts layers that NAMED
`n_modes` (naming 6 on the circle at stack M 4 silently raises the
sinusoid layer to 9), and the rule counts x-wall segments only (harmless:
every compiled map is square).

**Riding.**  Documented in `_perlayer_geometry`, build 2.5, module docstring
and CHANGELOG -- but the public text does not say which neighbour is ridden
or that a uniform layer naming `n_modes` / walls does not ride.  Tested
one-sided (removing it fails the vacuum identity; over-applying it and
riding BELOW survive all 12 ids).  **The direction does not matter beyond
the discretisation level**: a uniform eps-1.7 layer between the circle and
the sinusoid, error vs the merged map at M 7, above / below / own grid:
7.0e-4 / 5.7e-4 / 3.3e-4 (circle on top), 1.58e-3 / 6.0e-4 / 1.14e-3
(sinusoid on top) -- inside the reference's ~4e-4..1e-3 uncertainty.  The
rider takes the ridden layer's walls, map and M; the other interface is
then a curved mortar (confirmed).  `_perlayer_modal_counts` still reports
the rider's own M (11) although M 4 is used -- cosmetic, but
`convergence_floor` reads it.  A uniform layer NAMING `n_modes` keeps an N =
1 grid (q = M - 1) and is 7x worse at M 7 (4.9e-3 vs 7.0e-4).  The
vacuum / pillar / vacuum sandwich (air both sides) is byte-identical to the
shared stack with the same spacers at M = 4..8 on every route; against the
pillar alone it differs 2.1e-3 .. 3.6e-6 (the mapped half-space effect of
Phase C gate C9, not riding).

**Other silent defaults.**  Naming `n_modes` equal to the stack's own M on
one shape layer moves R/T by **0.117 at M 4 and 0.028 at M 5** (both builds):
it leaves the merged-map fast path AND exempts the layer from q-matching;
the fast-path condition is documented, its size is not.

**V-E2-D10 (P2) -- `convergence_floor` is not map-aware** on per-layer
stacks: it rebuilds each shape layer as its (u, v) cells on STRAIGHT walls
(a different device: 0.144 at M 3, 0.033 at M 5 from the mapped layer;
`v5_cfloor_win.json`), and on a per-layer shape stack that takes the fast
path it raises ("x_walls yields 3 segments but eps_cell has 4 strips";
`v5_cfloor_fast_win.json`).

The two-sided test `test_ve2_6_q_matching_and_riding_are_two_sided` (no
solve, 0.2 s) closes the survivors: it fails under q-matching off, riding
off and riding forced below (`logs/v8_*`), and pins the named-layer
exemptions and the fast-path switch.

---

## 6. Item 6 -- V-D3 (sub-verifier, `v6_*`)

* As ONE layer all three over-refused layouts still refuse (shared and
  per-layer); the FOLD message now names the arc-bulge mechanism (the V-D3
  text fix is applied) (`v6_refusal_win.json`).
* **What the two-layer split computes.**  Pillar A in layer 1 and pillar B
  in layer 2 is a different device (each pillar spans only its own layer).
  Against the one-layer macro-cell composite (`verify_c/v10_macro.py`,
  re-run on this tree: equal to the Phase C JSON to 1e-14), r 0.30 / 0.40,
  t 0.5:

  | device | M 3 | M 4 | M 5 | wall M 5 |
  |---|---|---|---|---|
  | macro-cell (one layer, 6 x 6) | 0 | 0 | 0 | 398 s |
  | A over B (0.25 + 0.25) | 0.485 | 0.435 | 0.504 | 57 s |
  | B over A | 0.430 | 0.498 | 0.512 | 25 s |
  | ABAB (4 x 0.125) | 0.461 | 0.359 | 0.376 | 75 s |

  The splits never approach the supercell (equal circles dy 0.05: 0.413 at
  M 4).  The build's table (0.30 + 0.25 thick) is a third device again.
  Rect at 30 deg: no one-layer reference exists (the macro-cell blend bends
  the rectangle's walls).
* Zero thickness is refused (`thickness must be > 0`); 1e-12 is accepted;
  A over a 1e-6-thick B tends to A alone, not to the supercell, and only at
  the curved mortar's own level (0.112 / 0.094 / 0.053 / 0.013 / 0.015 at
  M = 3..7; a 1e-6 vacuum layer moves A by 1.8e-7).
* **V-D3 is DEFERRED, not closed.**  "Put them in two layers" is not an
  honest user-facing answer for a one-layer supercell; the CROSS message's
  "or split the layer" and the build doc's "E2 now solves all three"
  overclaim (edits in section 11).

---

## 7. Item 7 -- nearly coincident singular vertices (sub-verifier, `v7_*`)

* Builder geometry = CONCENTRIC 3 x 3 circles (r 0.36 against r2), M 4.  n =
  41 / 41 / 62 / 96 reproduced exactly.  Separation 3e-4 .. 7e-5: n 96 (cap),
  correct (ladder at 1e-4: 41 / 62 / 80 / 93 / 96 / 104 -> 2.2e-9 ..
  2.7e-14); but n >= 112 already RAISES there.  **Failure begins between 7e-5
  and 5e-5**, not at 1.2e-6 (every separation 5e-5 .. 1e-8 raises).  WSL
  identical.
* Crossing equal circles shifted in x: refused up front at EVERY shift (1e-8
  .. 0.2) by the both-singular rule (D2), so "the limit is for crossing
  curves with nearly coincident 45-degree points" (build 3.5) is wrong: the
  limit is for NON-crossing nested / near-tangent outlines.
* Near-tangent circles on the diagonal (1e-3 .. 1e-7 apart): the cap warning
  reports "last change" 5.3e-8 .. 7.6e-8, the real error (vs n = 144) is
  4.7e-7 .. 6.5e-7 -- **the warning under-reports ~9x** (it compares n 93
  with n 96).  The own-geometry sub-verifier found the same on crossing
  circles whose 45-degree points are 1e-4 apart (4.4e-8 reported, ~1e-6
  true).
* Stack level: "concentric circles take the fast path" is false below the
  sliver contract: dr 1e-4 is refused by the merge (SLIVER) and solved
  through the curved mortar at n 96 (closure 1.25e-3, 139 s); dr 1e-6 makes
  `solve()` raise the bare `RuntimeError: curved mortar: a quadrature node
  could not be inverted in the neighbouring layer's map.` -- no layers, no
  advice.  The cap warning's advice ("add walls (grid_hint)") cannot be
  followed (`add_layer` takes no `grid_hint`) and is counterproductive
  (with `compile_shapes(grid_hint=5|7)` ordinary crossing pairs that
  converged at n 35 hit the cap, and the near-45-degree pair crashes; 
  `v3g_inv_diag2/3_win.json`).
* No silently wrong cross-mass was found here: every non-warning return
  agrees with a finer rule to 1.6e-14.  Message edits: section 11 (D5, D6).

---

## 8. Item 8 -- E2-8 and the verifier's mutation matrix (`v8_mutations.py`, `v8_mut_plugin.py`)

Stack level, M = 5; the build's E2-4 device and an own crossing device
(disk r 0.33 at (0.55, 0.62), eps 3.2; wall x0 0.66, A 0.1, phase 0.7, eps
1.9; lambda 0.93), both builds for the build device
(`v8_mutations_{builder,own}_M5_{win,wsl}.json`):

| mutation | build device: R/T moved, closure | own device: R/T moved, closure | E2 unit ids that FAIL (`logs/v8_pytest_e2_*.txt`) |
|---|---|---|---|
| shipped | -- , 1.8e-4 / 9.0e-5 | -- , 4.6e-5 / 9.8e-5 | -- |
| H row without the measure change det T | 5.5e-2, **6.9e-2** | 2.3e-2, 8.3e-2 | E2-3, E2-4, E2-6, E2-7 |
| H row without the off-diagonal cofactor | 2.3e-2, 4.0e-2 | 3.0e-2, 4.6e-2 | E2-4, E2-6, E2-7 |
| E row without the pull-back factor T | 5.3e-2, 2.6e-4 (**closure blind**) | 9.2e-2, 1.6e-4 (blind) | E2-3 only |
| maps ignored (the build's arm) | **6.7e-2** (build: 6.7e-2), 2.0e-4 | 1.3e-1, 1.2e-4 | E2-1, E2-2, E2-3, E2-8 |
| Newton step tolerance x 1e3 | **2.6e-15** (build: 2.6e-15) | 7.9e-15 | -- (by design) |
| H-swap off | 0.51, 0.10 / 0.12 | 0.66, 0.37 | (E2-2, E2-4 per the build) |
| fast path forced onto the crossing pair | refused by the vertex-claim check, R/T 0 | refused, 0 | E2-8 |
| square-root substitution off | 0 (no tangency end on this device) | 5.5e-13, adaptive n -> **96 + warning** | **NONE (survives)** |
| tangency search off | 0 | **4.0e-6, closure unchanged** (cross-mass 3.1e-5) | **NONE (survives)** |
| riding forced below instead of above | -- | -- | **NONE (survives)** |
| riding off | -- | -- | E2-4 (vacuum identity) |
| q-matching off | -- | -- | E2-4 (crossing) |

E2-8's three claims are confirmed.  **Test gaps (V-E2-D4, P2):** the
tangency search and the square-root substitution are invisible to all 12 E2
ids (the E2-4 device has no tangency end, and the build's kernel gates
compare the kernel only with itself), although removing the first moves the
cross-mass by 2e-2 (2.3) and R/T by 4e-6 silently.  Closed by the decision
tests `test_ve2_1` (tangency search off: 1.9e-2 >= 1e-4) and `test_ve2_5`
(substitution off: the adaptive rule runs to the cap).  The riding-direction
survivor is discussed in section 5.

---

## 9. Item 9 -- the Phase C fold-in (sub-verifier, `v9_*`)

* `test_vc4`, `test_vc5` pass, no xfail marker left.
* V-D1: random rectangles-only layouts (660 over three periods) that miss
  the identity: PRE 31-40 %, **POST 0 %** (both builds).  But the vertex snap
  (1e-13) is below the wall-merge tolerance (1e-12): walls coinciding to
  (1e-13, 1e-12] p still take the MAPPED path, and R/T jump by **2.55e-6** at
  oblique incidence for a 5e-13 p offset (**V-E2-D8, P3**); with
  `_VERTEX_SNAP = _WALL_SNAP` the window closes and the curved-C files pass
  (`28 passed`).  The snap cannot move geometry above round-off (a 3e-14 ..
  1e-9 p sinusoid still lays out, R/T vs flat 2e-15 .. 6e-15).
* V-D2: rotated Ellipse, aspect 1.01-5 x angle 0 .. +-44.9 deg: **0 / 77
  bad** POST (PRE 50 / 77 fold); 44.99 deg lays out; +-45 / 50 deg raise a
  clear ValueError.  Mirror check +30 / -30 deg: Jones 8.0e-15 (M 4).

---

## 10. Item 10 -- documentation and gates (sub-verifier, `v10_*`, and the lead)

* BUILD_E2 tables against the build JSON (`v10_docnums.json`): **242 / 243
  match**; mismatch "2-11 %" (11.8 % at M = 6); "36 000 random nodes" is
  36 000 / 16 000; "single-map references reach 1e-11 .. 1e-13 by M = 6 / 7"
  -- the sinusoid is 5.3e-9 at M = 6.
* CHANGELOG [Unreleased] E2 entry: accurate on the fast path, q-matching,
  riding, the limitation and the refusal; number defects "< 1e-13"
  (2.7e-13 measured), "2e-6 at M = 7" (2.9e-6), "~1e-15 by 16 nodes"
  (3.5e-11 at 16 on the sinx/siny pair at M = 6); presents closure as the
  accuracy (D3); example at `n_modes=4` (D3).  No forward version token
  introduced.
* Roadmap: "Phase D / E" should read "Phase E1 / E3" (D is built).
* History re-record reason of stack2d_pure still says "exact incident
  decomposition" (V-D4 sweep came after it).
* E2 cost: "the curved cross-mass is 2-11 % of the solve" holds for circle /
  sinusoid only; circle / circle spends 63 / 71 s (M 4) and 133 / 526 s (M 8)
  in it (`v3g_e24_closure_ii_*`).
* E2 unit file serial on the loaded box: `12 passed in 191.40s` (build: 54 s
  idle; ordering consistent, E2-2 slowest).
* mypy (configured list): `Success: no issues found in 33 source files`.
  `record_history_fingerprints.py --check`: `OK: every history document
  matches its module` (stack2d_pure, _core re-recorded with reasons).  WSL
  ruff 0.15.16 on `lumenairy/ tests/ validation/probe_pmm2d_curved/`: `All
  checks passed!` (after this verifier's own files were fixed).

### 10.1 Test tails (this branch, final)

* Decision tests, Windows serial: `3 passed, 2 xfailed in 31.74s` (ids 1-5)
  and `test_ve2_6` 0.2 s; each mutant it targets (q-matching off, riding off,
  riding forced below) fails it.
* Windows sweep (71 files: every pmm2d / stack2d / stagger / curved /
  per-layer / mortar file incl. A-D, E2 and all verifier files, census,
  public API, every walker, doc identifiers, dispatcher-pin doc consistency,
  except budget, history lint / relocation / fingerprint tool, re-exports,
  kernel consistency; `-n 6`): **`1676 passed, 8 skipped, 2 xfailed, 96
  warnings in 677.01s`** (`logs/sweep2_win.txt`; an earlier run before
  `test_ve2_6`: `1671 passed, 8 skipped, 2 xfailed in 1016.68s`).
* WSL: E2 file + decision tests `16 passed, 2 xfailed in 96.25s`
  (`logs/wsl_pytest_e2b.txt`).
* Gate subset (sub-verifier): `917 passed, 7 skipped in 667.10s`.
* `python -m mypy`: `Success: no issues found in 33 source files`;
  `scripts/record_history_fingerprints.py --check`: `OK: every history
  document matches its module.`; WSL ruff 0.15.16 on `lumenairy/ tests/
  validation/probe_pmm2d_curved/`: `All checks passed!`

---

## 11. Defects and exact edits

Severity: P1 = wrong answer in a supported case; P2 = silent error, a scope
claim that misleads a user, or a missing gate; P3 = message / doc / number.
**No P1 was found.**

**V-E2-D1 (P2) -- a grazing cut is lost between wall samples** (2.3): up to
6.4e-7 of the cross-mass, adaptive change 8e-15.
Edit: (a) in `_curvemortar._cell_pieces` set `tk0 = tk` before
`for e, ebox in edges:` and, inside the loop, before
`U, V, _du, _dv, ok = _pullback(Pm, sx, sy, Om, e, tk)` insert
`tk = _grazing_refine(Pm, sx, sy, Om, e, tk0)` and `K = tk.size`; add the
helper `_grazing_refine` exactly as in `verify_e2/v2e_graze_fix.py` (dips
into the cell and excursions out of it between samples, refined on a
257-point sub-grid).  (b) Flag the near-tangent crossings of such a sliver
for `_outer_rule`'s square-root substitution -- (a) alone leaves the circle
case at 6.5e-7.  (c) Flip `test_ve2_3` to a plain gate.

**V-E2-D2 (P2) -- crossing CLOSED curves are mostly refused** (2.3): 46 / 56
(and 25 / 38) crossing circle layouts; the message's remedies do not apply;
build 3.5 describes only "nearly coincident 45-degree points".
Edit, `_curvemortar._assign_pairs`, replace the last two sentences of the
message, `"Move one outline so the two maps' singular vertices do not share
a cell overlap, or use layer_grids='shared' (one merged map) when the
outlines do not cross."`, by `"This refuses most pairs of CROSSING closed
curves (two circles crossing off-axis); outlines offset along x or y may
pass.  Splitting such a cell between the two maps is not implemented
(verifier V-E2-D2)."`.  CHANGELOG, after "... in the same cell overlap
raise, naming the cells." append " -- this refuses most crossing pairs of
closed curves (46 of 56 crossing circle pairs sampled,
verify_e2/v2c_circle_pairs_win.json)".  Build 3.5: see D6.  Design
follow-up: split such a primary cell into sub-rectangles each touching one
map's singular vertex.  Flip `test_ve2_4` when built.

**V-E2-D3 (P2) -- closure is presented as the crossing device's accuracy**
(3.4): it understates the R/T error by 2.2-4.8 decades.
Edit, build doc 3.4, after the closure table: "Against an independent
conforming merged map and RCWA (verifier V3) the R / T error of this device
is 1.1e-1, 3.0e-2, 1.4e-2, 1.6e-3, 1.0e-3 at M = 4..8 (conical 25 / 40 deg:
9.4e-2 .. 6.7e-3, not monotone); closure understates it by 2-5 decades; the
rung-to-rung change tracks it within 2x."  CHANGELOG Gates bullet, after
"1e-6 at `M = 7 .. 8`" insert " (closure is not the accuracy: against an
independent conforming map and RCWA the R / T error is 1.4e-2 at `M = 6`
and 1e-3 at `M = 8`)".  CHANGELOG example: `n_modes=4` -> `n_modes=7` (or
the comment "# n_modes=4 is a smoke rung: 1e-1 error on this device").

**V-E2-D4 (P2) -- test gaps** (8, 5): the tangency search off, the
square-root substitution off, riding forced below, and both defaults
over-applied survive all 12 E2 ids.  Closed by
`tests/unit/test_verify_pmm2d_curved_e2.py` (`test_ve2_1`, `_5`, `_6`).

**V-E2-D5 (P3) -- the 96-cap warning** compares n 93 with n 96 (a 3 % step)
and under-reports the error ~9x; its advice "add walls (grid_hint)" cannot
be followed (`add_layer` has no `grid_hint`) and is counterproductive (7).
Edit, `curved_cross_mass_adaptive`, replace the loop by one that never takes
a step shorter than 1.5x (identical whenever the rule converges):

    n = min(max(ga.M, gb.M) + 4, cap)
    X1 = curved_cross_mass(ga, gb, n)
    n1, chg = n, np.inf
    while True:
        n2 = int(np.ceil(1.5 * n1))
        if n2 > cap:
            break
        X2 = curved_cross_mass(ga, gb, n2)
        sc = float(np.max(np.abs(X2))) or 1.0
        chg = float(np.max(np.abs(X2 - X1))) / sc
        n1, X1 = n2, X2
        if chg <= tol:
            break

(return `X1, n1, chg`); warning text: "The two maps are too different inside
a cell -- add walls (grid_hint) where they are steep." -> "Typical cause:
outlines in adjacent layers that nearly coincide or nearly touch (a closed
curve's 45-degree point close to the other outline); make them identical or
move them >= 1e-3 of the period apart."

**V-E2-D6 (P3) -- near-coincident singular vertices** (7): failure from
5e-5 .. 7e-5 (not 1.2e-6); a finer rung raises where a coarser one
succeeded; `solve()` raises a bare RuntimeError; build 3.5's scope is wrong.
Edit, `_cut_cell_nodes`: the message `"curved mortar: a quadrature node
could not be inverted in the neighbouring layer's map."` -> `f"curved
mortar: a quadrature node of cell ({sx}, {sy}) could not be inverted in
cell ({cx}, {cy}) of the neighbouring layer's map at n = {n}.  Outlines in
adjacent layers that nearly coincide or nearly touch (measured: concentric
circles closer than ~6e-5 of the period) put the two maps' singular points
closer than the inversion resolves: make the outlines identical (the stack
then merges them) or move them >= 1e-3 of the period apart."`; in the
adaptive loop, catch `RuntimeError` at a refinement rung, warn, and keep the
last good rung.  Build 3.5: "at 1.2e-6 apart a node inversion fails ...
nearly coincident 45-degree points." -> "below 5e-5 .. 7e-5 apart (M = 4) a
node inversion fails, and at 7e-5 .. 3e-4 the n >= 112 rules already fail.
Concentric or near-tangent circles closer than the sliver contract are
refused by the merge, so a per-layer stack DOES route them through the
curved mortar (dr 1e-4 solves at n 96; dr 1e-6 raises from solve()).
Crossing circles never reach this limit: they are refused up front
(V-E2-D2)."

**V-E2-D7 (P3) -- V-D3 overclaims** (6).  Build 2.6: "With the shapes in
DIFFERENT layers E2 now solves all three (the per-layer maps are what the
merge could not build)." -> "E2 solves the DIFFERENT two-layer devices (each
shape in a layer of its own); they are not the over-refused one-layer
supercells and not a workaround: against the one-layer macro-cell (t 0.5)
the splits differ by 0.43-0.51 in R / T at M = 3..5 and do not converge to
it (verifier V6)."  `shapes2d` CROSS message: "move the shapes apart, nest
one inside the other, or split the layer." -> "move the shapes apart or
nest one inside the other; if the outlines are in DIFFERENT layers,
PMM2DStackPure(..., layer_grids='per-layer') joins the layers' own maps by
the curved mortar (Phase E2); outlines in ONE layer cannot be split into two
layers without changing the device."  FOLD message, "...; a common map for
them is Phase E of the curved-cell plan." -> "...  If the shapes are in
DIFFERENT layers, layer_grids='per-layer' gives each layer its own map
(Phase E2); shapes in ONE layer have no route yet (the hybrid merge is
deferred)."  `shapes2d` Known limits: "``slant=`` and per-layer grids under
a curved map are Phase E and raise;" -> "``slant=`` and a STACK-wide
``cmap=`` with ``layer_grids='per-layer'`` raise (per-layer maps are Phase
E2);".

**V-E2-D8 (P3) -- V-D1 residual window** (9).  `shapes2d.py`:
`_VERTEX_SNAP = 1e-13` -> `_VERTEX_SNAP = _WALL_SNAP` (walls within
`_WALL_SNAP` are merged, so a claim on the discarded wall sits up to that
far off the merged vertex); pin in `test_vc4`:
`r1 = Rect(0.45, 0.6, 0.3, 0.4, 4.0)`; `x2 = 0.3 + 5e-13 * _P`;
`r2 = Rect(0.5 * (x2 + 0.96), 0.54, 0.96 - x2, 0.6, 3.0)`;
`assert SH._merge(_P, _P, [("a", [r1], 1.0), ("b", [r2], 1.0)])[4]`.
Measured in-process with the edit: curved-C files `28 passed`.

**V-E2-D9 (P3) -- numbers and wording** (10, 4, 5).  Build doc: "2-11 % of
it" -> "2-12 % of it (circle / sinusoid; circle / circle spends most of the
solve in it, verifier V3g)"; "over 36 000 random nodes" -> "over 36 000 /
16 000 random nodes"; 3.4 "The two converge to one answer, slowly --
section 4.3." -> "At M <= 8 the merged map's own error dominates this column
(merged vs merged M10: 1.05e-1, 1.01e-2, 9.75e-3, 4.34e-4, 4.14e-4); the
per-layer limitation appears beyond M = 8 as a stall near 3e-4 while the
merged map reaches 1.8e-5 at M = 9 (verifier V4)."; 4.3, add "The algebraic
tail (p_M 1.8-2.2) keeps its exponent at eps 1.1 and 1.02 while its size
relative to the scattered amplitude falls in proportion to delta-eps -- the
rim singularity through a non-conforming cut (verifier V4)."  CHANGELOG:
"~1e-15 by 16 nodes per piece" -> "~1e-14 by 16 to 20 nodes per piece";
"absorbs < 1e-13" -> "absorbs < 3e-13"; "close to 2e-6 at `M = 7`" ->
"close to 3e-6 at `M = 7`"; after "4.5e-4 at `M = 7`" add " (the merged map
is itself 1e-2 from its converged value at `M = 5 .. 6`; the per-layer route
stalls near 3e-4 from `M = 8` while the merged map reaches 2e-5)".
Roadmap: "Phase D / E, approved 2026-10-02" -> "Phase E1 / E3, approved
2026-10-02".  History record of stack2d_pure (the Phase E2 re-record line):
"exact incident decomposition" -> "unique L2 incident projection".
`stack2d_pure` module docstring: "and a HOMOGENEOUS layer rides its
neighbour's grid (no mortar at that interface)." -> "and a HOMOGENEOUS
layer rides its neighbour's grid -- the nearest non-riding layer above, else
below (no mortar at that interface); a uniform layer that names ``n_modes``
/ ``grid`` / walls keeps its own grid.  Naming ``n_modes`` on any layer,
even the stack's own ``M``, takes a mergeable stack off the merged-map fast
path and exempts that layer from q-matching (measured 0.12 / 0.028 in R / T
at M = 4 / 5)."

**V-E2-D10 (P2) -- `convergence_floor` is not map-aware** on per-layer
stacks (5): 0.144 off at M 3; raises on the fast path.  Edit,
`stack2d_pure.py` (about line 2013), replace

    if L["kind"] == "patterned":
        kw.update(eps_cell=L["eps_cell"], x_walls=_stag_interior(
            L["wx"]), y_walls=_stag_interior(L["wy"]))

by

    own = L.get("own")
    if own is not None and own["cmap"] is not None:
        # PHASE E2: a shape layer on its OWN curved map is that map's
        # (u, v) cell; on straight walls it is another device
        kw.update(eps_cell=own["cell"], cmap=own["cmap"])
        if own["mu"] is not None:
            kw["mu_cell"] = own["mu"]
    elif own is not None:
        kw.update(eps_cell=own["cell"],
                  x_walls=_stag_interior(own["wx"]),
                  y_walls=_stag_interior(own["wy"]))
    elif L["kind"] == "patterned":
        kw.update(eps_cell=L["eps_cell"], x_walls=_stag_interior(
            L["wx"]), y_walls=_stag_interior(L["wy"]))

(not executed here: check the magnetic branch), with a test that the floor's
circle layer equals the shape-alone residual and that a fast-path stack does
not raise.

**V-E2-D11 (P3) -- over-refusal of one circle on two layouts** (3.3): the
same physical circle on a `grid_hint` map against its plain 3 x 3 map raises
the both-singular refusal (`_same_cell` needs identical rectangles).  Edit,
`_assign_pairs`: exempt a both-singular mark when the two singular points
coincide to 1e-12 x scale and the two maps agree on a probe set inside the
overlap.

## 12. What the integration must carry

1. The documentation edits D2, D3, D6, D7, D9 (scope and accuracy claims),
   before the CHANGELOG entry ships.
2. `tests/unit/test_verify_pmm2d_curved_e2.py` (6 ids, ~32 s on a loaded
   box; durations spliced): the only gates that compare the cut-cell kernel
   with an independent oracle and pin both defaults on both sides.
3. D1 (the grazing-cut fix with its near-tangency substitution) and D10
   (`convergence_floor`), each with a test; then flip `test_ve2_3`.
4. D5 / D6 (adaptive loop, messages) and D8 (the snap).
5. Recorded as known limits until built: D2 (crossing closed curves), V-D3
   (same-layer supercells, deferred), the own-walls-only accuracy class (1e-2
   at M = 6, 1e-3 at M = 8-10 on the E2-4 device; a stall near 3e-4 beyond
   M = 8 where a merged map exists), and the 96-node cap on strong
   stretches (n = 80 measured).

## 13. Not measured

* An oracle at 1e-12 on pairs WITH singular vertices: the physical brute
  force reaches 1e-15 on singularity-free pairs but only ~2e-9 on circle
  pairs (its own grading floor); the kernel is confirmed there only to that
  level, plus its own spectral ladder.
* E2-7 (three layers, three maps) was not re-measured independently.
* A 3-D FEM oracle for the crossing device (the references are a hand-built
  conforming map and RCWA, agreeing to 4.3e-4).
* Merged map at M >= 9 (normal) and M = 8 (conical); RCWA's own error at
  conical incidence; staircase beyond n = 8.
* The capture-plugin byte comparison and the trap suites on WSL (Windows
  only; the own 191-key set ran on both builds).
* Reciprocity on the circle / circle and fillet / sinusoid pairs; pair (iii)
  at M = 7, 8.
* The proposed D5 / D6 / D10 edits were not executed; D1's prototype (v2e)
  and D8's edit were.
* Idle-box wall times: every wall time is an upper bound.

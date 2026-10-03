# VERIFY -- curved cells for the pure staggered 2-D PMM, Phase B (transfinite maps, the corner quadrature): an independent adversarial verification

Date: 2026-10-02.  Verifier: Claude (Anthropic), model Opus 5.5
(`claude-opus-5-5`), as an independent agent.
Mount: worktree `C:/tmp/lum_vcurved_b`, branch `verify/pmm2d-curved-b` on the
build tip `91d00288` (`feat/pmm2d-curved-b`, commits `6bc32418..91d00288`
on the Phase A tip `539ce4a3`).  PRE tree for byte identity: this
verifier's own `git archive 539ce4a3` in `C:/tmp/vcurved_b_pre`.
Builds: Windows 11 (CPython 3.14.6, numpy 2.4.4, scipy 1.17.1) and WSL
Ubuntu (CPython 3.12.3, numpy 2.4.6, scipy 1.17.1), BLAS pinned to one
thread on the command line, `lumenairy.__file__` asserted under the tree in
every probe.  The box was shared with two sibling agents and this
verifier's own queues (up to ~50 single-threaded processes on 24 logical
cores for most of the run): every wall time is an upper bound.
Evidence: `validation/probe_pmm2d_curved/verify_b/` (probes + JSON; the
`_win` / `_wsl` suffix names the build).  Decision tests:
`tests/unit/test_verify_pmm2d_curved_b.py`.

Independence.  Every map the verifier solves is built by
`verify_b/_vcommon.py` from the PUBLIC primitives (`TransfiniteMap`,
`Arc.through` on vertex images, `EllipseArc`, `Sinusoid`); the builder's
private `_circle_map_3x3` / `_fillet_map_5x5` / ... are used only where a
probe compares against them on purpose (v1: the verifier's circle, ellipse
and fillet maps reproduce the builder's pointwise to <= 1.05e-14, so the
builder's maps are themselves verified).  The second circle topology uses a
different interior (`inner = 0.6`, the builder's 0.5); the ellipse
(0.42 x 0.24), the sinusoid (A = 0.08 on 0.35 .. 0.85, v walls 0 / 0.36 /
0.9 / 1.2) and two NEW FEM radii are the verifier's own fixtures.  The
quadrature rules (tensor Gauss, tensor Gauss-Jacobi, an own Duffy rule, two
Duffy variants) are coded in `v2_quadrature.py`; reciprocity is computed
from an own power-metric implementation (`v7_summary.py`).

## 1. Verdict table

| # | claim | verdict | the verifier's evidence (own fixtures) |
|---|---|---|---|
| 1 | B1: no map = 109 / 109 bytes; transfinite identity 4.8e-14; areas 5e-16; C0 4.4e-16; mismatched vertices and folding refused | **CONFIRMED** | 37 / 37 SHA-256 on independent fixtures (tensor, magnetic, slanted, absorbing, conical, per-layer, direct functions) on BOTH builds; identity <= 4.0e-14 on a random absorbing 4 x 4 cell at alpha0 != 0; areas <= 1.8e-15 and C0 <= 2.2e-15 on every interior edge of 11 maps; refusals at 1e-11 p, folding refused (s 2) |
| 2 | THE QUADRATURE DECISION: moments `n^-2` under plain Gauss; plain-rule R / T 3e-7 and operators 6-7 % wrong at `2M + 8`; reparametrisation fails; Duffy reaches round-off at 16-20 points | **CONFIRMED and derived** (reparametrisation by derivation only) | det J has a SIMPLE zero (exponent 1.0000) at every singular vertex of every map; the 1-D marginal is log(1/s), on which Gauss-Legendre is `n^-2` (measured on log s directly: ratio 3.9-4.0); the Duffy collapse leaves exponent 0 -- the transformed integrand is analytic.  **Gauss-Jacobi**: in the tensor variables NO exponent fits (the singularity is logarithmic in each), measured a = -0.5 / -0.25 / +0.5 all `~n^-1.8`, worse than plain Gauss (R / T 3.2e-4 at n = 20); in the collapsed variable the derived exponent is 0, i.e. Gauss-Jacobi = Duffy + Gauss-Legendre as built; using the Duffy Jacobian as the Jacobi weight fails (`n^-2`).  **Duffy is the right tool, not merely sufficient** (s 3) |
| 3 | B3: circle vs the saved FEM 7.77e-6 (M = 11), 1.93e-6 (M = 12); c5 1.02e-5; symmetry 8.9e-10 | **CONFIRMED and extended to 3 radii** | FEM provenance: the saved mesh re-run reproduces the record to 1.4e-14; the verifier ran the planner's runner at r = 0.24 and 0.48 (3 meshes each, spreads 1.5e-5 / 6.1e-6): c3 lands at 2.2e-6 / 1.6e-5 (M = 12), c5 at 2.2e-5 / 2.5e-6 (M = 8) (s 4) |
| 4 | B4-B7: staircases 7.07e-2 / 5.36e-2 / 5.65e-3, cosines 0.93 / 0.99; Bloch modes; fillets monotone, same-area square wrong way | **CONFIRMED**, two qualifications | staircases reproduced and converged in M except the 16-step one (pre-asymptotic at M = 4); at r = 0.48 monotone too.  Fillets on the verifier's own map at five radii: monotone over the whole range (the build's small-r values were unconverged).  **The r -> 0 claim as written is not a bound** (a 3-point, 3-parameter fit; its constant spans 1.6e-4 over bases); with five radii and the corner exponent `r^(2 lambda)`, lambda = 0.806, the limit is the sharp square to ~3e-5.  Small fillets converge slowly because the BIG neighbouring cells must resolve the corner field down to the scale r -- not the sliver contract, not Duffy, not conditioning (all three refuted) (s 7) |
| 5 | B8: corner rule = tensor rule on polynomial weights; operators n = 20 vs 40 8.9e-13; plain rule 6.3e-2, warns | **CONFIRMED**; mutation answered | corner rule fed a cell with NO singular vertex: **round-off, not bit-for-bit** (5.0e-14; forced Duffy on smooth weights 1.4e-13), both builds (s 2) |
| 6 | B9: ellipse / sinusoid vs exact-FF RCWA (4.1e-3 / 6.3e-3 at 29 orders; Richardson 3.0e-4 / 2.4e-4); ellipse symmetry broken 0.11 | **CONFIRMED at the 1e-3 level the RCWA can certify** | own ellipse 0.42 x 0.24 and sinusoid A = 0.08: distance x N flat to +-4 % (ellipse) over N = 9 .. 33, but the LOCAL rate wanders 0.2 .. 1.8 -- the 1 / N Richardson is a 1e-3-class reference, not a 1e-5 oracle; the curved answer sits within ~1.4x the extrapolation spread (s 8) |
| 7 | B10: oblique / conical closure, topologies, reciprocity vs wrong pairing; F-B6 | **CONFIRMED to M = 9**; **F-B6 corrected** | reciprocity 3.1e-6 .. 9.7e-9 reproduced by an own SVD; **y-momentum** on a y-uniform stripe under a non-separable curved map: leak 1.8e-7 -> 1.3e-9 (M = 4 .. 8), window-independent, three decades under the allowed-order error.  F-B6: the "4-6 decades" are at equal M; at EQUAL DOF the 3 x 3 and 5 x 5 films are equal (3.5e-6 vs 2.0e-6 at 450 dof); what costs the decades is the map composed with the oblique Bloch phase (s 9) |
| 8 | F-B4: ~1e-9 floor from the min-norm `lstsq` | **CONFIRMED mechanism; scope = Phase A (caught there as F-V2)**; one recommended fix rejected | `Hsup` is well conditioned (cond 5); R / T move with the NULL-SPACE component of `cinc` (6e-4 per unit), on Phase A's stretch exactly as on the circle; the floor is geometry dependent (1.1e-8 at r = 0.48); the overdetermined-window fix removes the draw but NOT the 5e-8 .. 1.3e-7 window dependence and contradicts the per-layer order cap; the exact modal decomposition is the right fix, < 10 % of a solve (s 6) |
| 9 | F-B1: tangential-only periodicity | **CONFIRMED necessary and sufficient** | derived (the interface quantities are built from `Phi` and `Phi_v` only); a normal-KINKED map equals the unmapped solve to 1.6e-14; a kinked re-parametrisation of the circle equals the plain one to the floor (1.7e-9 .. 1.5e-12); a position-shifted seam is refused and, forced, 4e-3 wrong without converging (s 5) |
| 10 | the 18 tests | **ADEQUATE**: 5 / 6 mutants caught, the survivor is EQUIVALENT | m1a / m1b (corner rule off at one corner / one cell), m2 (C0 broken at one seam), m3 (arc -> chord), m5 (symmetry offset) caught by named ids; m4 (radius dropped from `Arc.key`) survives every test and is equivalent (centre + angles + pinned endpoints determine the radius).  Six decision tests + one strict xfail (defect D-1) added in `test_verify_pmm2d_curved_b.py` (s 10) |
| 11 | docs | **MINOR corrections** | numbers match JSON; five wording corrections (s 11); mypy clean; WSL ruff clean; no forward version token |

**New defect D-1 (P2):** a user `EdgeCurve` whose analytic derivative is
not d/ds of its value is accepted and silently changes R / T (3.2e-2 at
M = 5); exact edit in section 12, verified to flip the strict-xfail test.


## 2. B1 -- no map = today's bytes; the transfinite identity; geometry

| claim | verifier's measurement | evidence |
|---|---|---|
| no map = the bytes of `539ce4a3` | **37 / 37** SHA-256 identical on the verifier's OWN fixtures (not Phase A's 109): scalar / TENSOR / MAGNETIC / SLANTED operator assemblies (`Lmat Rmat Stt Schur`) on 4 x 4 non-uniform walls at `alpha0 != 0`; a 3-layer conical stack (absorbing uniform, uniform tensor, patterned absorbing) with retained internals + `layer_absorption`; a per-layer-grids stack with a mortar; direct `pmm_jones_2d_staggered` / `pmm_efficiency_2d_staggered`; `Basis1D` arrays.  **Both builds** (Windows and WSL each against its own PRE run) | `v4_bytes_{pre,post,compare}{,_wsl}.json` |
| `TransfiniteMap(walls)` = the kron assembly | operators <= 9.8e-15 (`M = 5`), 4.0e-14 (`M = 7`) on a random ABSORBING 4 x 4 cell at `alpha0 = (0.8, -0.5)`; `nq = 2 M + 8`, no corner cell; R / T at conical (0.3, 0.5) rad vs the per-layer unmapped path 5.0e-11 / 1.7e-11 (Windows), 5.0e-11 / 5.9e-14 (WSL) -- a different code path (per-layer S-matrix vs the mapped magnetic route); the operator-level identity is the claim and it holds | `v5_identity_{win,wsl}.json` |
| areas exact | 11 maps (circles r = 0.24 / 0.36 / 0.48 on 3 x 3, 5 x 5 with inner 0.6, ellipse 0.42 x 0.24, fillets r = 0.006 / 0.12, graded 7 x 7 / 9 x 9 fillets, sinusoidal ridge, y-uniform curved map): feature area <= 1.8e-15 relative, cell area <= 4.4e-16 | `v1_geometry_{win,wsl}.json` |
| C0 | EVERY interior edge of the 11 maps: positions <= 4.4e-16, tangent <= 2.2e-15; the NORMAL derivative jumps by up to 1.35 (allowed, section 5) | same |
| mismatched vertices refused | a vertex image moved by 1e-9 p or 1e-11 p: refused; 1e-13 p: accepted (below the documented 1e-12 tolerance); an arc with swapped orientation: refused | same |
| folding refused | a circle of r = 0.62 / 0.8 in the 1.2 cell: refused; a sinusoid crossing the cell edge (A > x1): refused.  (An arc bulging INWARD is accepted -- correctly: it is a valid, non-folding D-shaped cell, not a fold.) | same |
| the verifier's maps = the builder's | circle, ellipse, fillets r = 0.12 / 0.03 built here from `Arc.through` reproduce `_circle_map_3x3` / `_ellipse_map_3x3` / `_fillet_map_5x5` pointwise to <= 1.05e-14 (positions and J); the fingerprints differ (the vertex-image bits differ at 1e-16), as they should for a content hash | same |

**Verdict B1: CONFIRMED** on independent fixtures and both builds.  The
corner rule fed a cell with NO singular vertex (claim 5): `corners = []`
lays the tensor nodes through the point path and gives the tensor-rule
operators to **round-off, not bit-for-bit** (Lmat 5.0e-14 relative, never
bitwise -- the point path sums in a different order); a FORCED corner on a
smooth cell (Duffy on polynomial weights) 1.4e-13 (`v5_corner0_{win,wsl}.json`,
identical on both builds).



## 3. THE QUADRATURE DECISION -- re-derived, re-measured, and the Gauss-Jacobi alternative

### 3.1 The singularity, derived

Take a singular vertex at the cell's local corner `(s, t) = (0, 0)` with the
two edges leaving it along `B(s)` (bottom) and `L(t)` (left).  The
Gordon-Hall blend gives `Phi_s = B'(0) + O(s, t)`, `Phi_t = L'(0) + O(s, t)`.
A smooth closed curve made of grid lines turns by 90 degrees in `(u, v)`
where it does not turn in `(x, y)`, so at the vertex `L'(0) = -c B'(0)`
(antiparallel, `c > 0`) and the LINEAR part of `J` is rank one: `det J(0, 0)
= 0`.  The next order does not vanish (the curve has nonzero curvature and
the opposite edges are not parallel to it), so

    det J(s, t) = alpha s + beta t + O(rho^2),   alpha, beta > 0,

a SIMPLE zero.  Measured on every singular vertex of every map in this
verification (circle r = 0.24 / 0.36 / 0.48 on 3 x 3 and 5 x 5, ellipse
0.42 x 0.24, fillets r = 0.006 .. 0.12, a graded 9 x 9 fillet; 21 rays per
vertex, rho = 1e-3 .. 1e-5): the fitted exponent of `det J` is 0.99993 ..
0.99998 (the deficit is the `O(rho^2)` term), and `det J / rho` is the SAME
in every direction of the quadrant (`l_min / l_max = 1.000`; for the fillet
independent of `r` -- the arc cell is a scaled copy of one quarter-disk)
(`v1_geometry_win.json`, `_wsl` identical).  Hence the five geometric
weights `1 / sqrt g`, `g_ij / sqrt g` are homogeneous of degree `-1`:

    w(s, t) = A(theta) / rho + O(1).

**Why the tensor Gauss rule converges as `n^-2`.**  Integrate over `t` first:
for `s > 0` the integrand `1 / (alpha s + beta t)` is analytic on
`t in [0, 1]` (its pole sits at `t = -alpha s / beta`, outside), and the
inner integral is `(1 / beta) log(1 + beta / (alpha s))` -- a LOGARITHMIC
endpoint singularity of the 1-D marginal.  Gauss-Legendre on `int_0^1
log(s) p(s) ds` converges as `n^-2` (the nodes nearest the end sit at
`O(n^-2)`); measured here directly: 5.7e-3, 1.5e-3, 3.9e-4, 9.7e-5, 2.5e-5,
6.2e-6, 1.5e-6 at n = 10 .. 640 (ratio 3.9-4.0 per doubling,
`v2_moments_win.json`, `gl_on_log_s`).  The 2-D moments inherit exactly
that rate: c3 7.5e-3, 1.9e-3, 4.8e-4, 1.2e-4, 3.1e-5, 7.6e-6, 1.9e-6 at
n = 20 .. 1280 (M = 6 and 8 identical from n = 20; c5 / fillet / ellipse the
same rate with smaller constants).  The builder's 5.6e-3 .. 1.4e-6 are the
SELF-GAP `|m(n) - m(2n)|` of the same family (= 3/4 of the error for an
`n^-2` family: 0.75 x 7.53e-3 = 5.6e-3) -- reproduced.

**Why Duffy is EXACT in the sense that matters.**  Collapse a triangle onto
the corner: `(s, t) = xi (a(eta), b(eta))`, `dA = xi |J_0| dxi deta`.  Then
`det J = xi [l(eta) + xi q(eta, xi)]` with `l > 0` on the closed quadrant,
so

    w dA = xi g_ij / det J |J_0| dxi deta = g_ij / (l + xi q) |J_0| dxi deta,

an ANALYTIC function of `(xi, eta)` on the closed unit square (no residual
power of `xi`).  The singularity is removed exactly, not merely weakened;
what remains is polynomial times analytic, so Gauss-Legendre converges
spectrally and -- because the analytic factor is a low-frequency
trigonometric function of the edge parameters -- abruptly: the moment
error drops from 1.5e-10 to 6.7e-15 between n = 12 and 16 at M = 6 (c3),
from 2.2e-5 to 2.7e-13 between 12 and 16 at M = 8; c5 / fillet / ellipse
alike; fillet r / side 0.02 the same counts as 0.2 (floor 4e-14 vs 5e-15).
At the solver's `n = 2 M + 8` every shape is at round-off.  The builder's
`_stag_duffy_points` and the verifier's own Duffy code (`vduffy`) agree to
every printed digit.

### 3.2 The Gauss-Jacobi alternative -- is Duffy the right tool or merely sufficient?

A Gauss-Jacobi rule is exact for (algebraic endpoint weight) x
(polynomial).  The question is which variable carries an algebraic weight.

* **In the tensor variables `(s, t)`: none does.**  The 2-D singularity
  `1 / (alpha s + beta t)` is not a product of 1-D powers; per line it is a
  near-pole, and its 1-D marginal is a LOGARITHM, which no Jacobi exponent
  matches.  Measured (moments, c3 M = 6, weight `(1 - x^2)^a` per axis):
  `a = -0.5`: 8.6e-2, 2.7e-2, 8.2e-3, 2.4e-3, 6.8e-4, 1.9e-4, 5.4e-5
  (n = 10 .. 640); `a = -0.25`: 2.6e-2 .. 2.7e-5; `a = +0.5`: 1.3e-1 ..
  3.6e-4.  All ALGEBRAIC (`~n^-1.8`) and all WORSE than plain Gauss.  In
  full library solves (c3, M = 6, forced node count; `v2_rt_*.json`):
  `a = -0.5` leaves R / T 3.2e-4 / 2.0e-5 and the operators 18 % / 1.7 %
  wrong at n = 20 / 80, against plain Gauss's 3.9e-7 / 2.6e-8 and 6.3 % /
  0.41 %.
* **In the collapsed variable `xi`: the derived exponent is 0.**  After the
  collapse the integrand carries `xi^(-1)` (the weight) times `xi^(+1)` (the
  Jacobian) = `xi^0`: Gauss-Jacobi with the derived exponent IS
  Gauss-Legendre, i.e. the Duffy rule as built.  The tempting variant
  "Gauss-Jacobi with the Duffy Jacobian as its weight" (weight `xi^1`
  applied to the un-multiplied weight `A / xi`) leaves a `1 / xi` pole at the
  end point and fails: 1.1e-1, 3.4e-2, 9.4e-3, 2.5e-3, 6.4e-4, 1.6e-4,
  4.1e-5 at n = 4 .. 256 (`n^-2` again); in the solver R / T 3.5e-7 and
  operators 5.2 % at n = 20 -- the plain rule's numbers.  An OVER-collapse
  (`xi -> xi^2`, Jacobian `xi^3`) still converges spectrally but needs
  n = 24 for what the plain collapse reaches at n = 16.
* **Per-axis re-parametrisation** (the build's option (c)) was not
  re-measured; the derivation agrees with its failure: under `s = q(s')`,
  `t = q(t')` the metric weight `g_11 / sqrt g` picks up `q'(s') / q'(t')`,
  which is singular along the WHOLE edge `t' = 0` wherever `q'(0) = 0` (a
  vertex singularity traded for an edge singularity).

**Verdict: Duffy is the right tool, not merely sufficient** -- it is the
change of variable that makes the derived exponent zero, and the remaining
error is a polynomial-times-analytic quadrature error that vanishes
spectrally; no Jacobi weight in the tensor variables can do that, because
the singularity is not algebraic in either of them.  (Gauss-Jacobi WOULD be
the right tool for an algebraic FIELD singularity -- the `rho^(lambda - 1)`
of a dielectric corner -- which is a resolution question of the basis, not
of the quadrature; section 7.)

### 3.3 What the plain rule costs, re-measured (`v2_rt_*.json`, `v2_rt_summary.json`; full library solves, node count forced, distance to the verifier's own Duffy n = 48)

| rule, n | c3 r = 0.36, M = 6: R / T | operators | c3 r = 0.48, M = 6: R / T | operators |
|---|---|---|---|---|
| plain Gauss 20 | 3.87e-07 | 6.25e-02 | 2.59e-06 | 6.47e-02 |
| plain Gauss 40 / 80 / 160 / 320 | 1.0e-07 / 2.6e-08 / 6.6e-09 / 2.0e-09 | 1.6e-02 / 4.1e-03 / 1.0e-03 / 2.6e-04 | -- / 1.6e-07 / -- / 2.0e-08 | -- / 4.3e-03 / -- / 2.7e-04 |
| Gauss-Jacobi a = -0.5, 20 / 80 | 3.2e-04 / 2.0e-05 | 1.8e-01 / 1.7e-02 | | |
| Duffy + Jacobi-weight xi^1, 20 / 80 | 3.5e-07 / 2.2e-08 | 5.2e-02 / 3.5e-03 | | |
| own Duffy 8 / 12 / 16 / 24 / 32 | 6.8e-08 / 3.8e-10 / 9.0e-10 / 1.7e-09 / 4.7e-10 | 1.5e-03 / 2.8e-09 / 5.1e-13 / 3.7e-13 / 5.8e-13 | (16) 7.0e-09 | 2.1e-09 |
| the SHIPPED rule (builder's Duffy, n = 20) | 1.38e-09 | 6.5e-13 | 1.07e-08 | 3.6e-12 |

Every claim of section 2 of the BUILD doc is reproduced: operators 6.3 %
wrong under the plain rule at `2 M + 8` while R / T move 4e-7; an `n^-2`
family in both; the Duffy operators at round-off from n = 16.  Two
additions: (i) at r = 0.48 the plain rule costs 7x more in R / T (2.6e-6)
-- the larger the disk, the stronger the corner weights; (ii) the R / T
"plateau" of the Duffy ladder (1e-9 at r = 0.36, 1e-8 at r = 0.48 between
rules whose operators agree to 1e-12) is the F-B4 floor (section 6), which
is geometry-dependent: the BUILD doc's "~1e-9" holds for its fixture only.


## 4. B3 / B4 -- the circle against the FEM, at THREE radii

### 4.1 Provenance of the saved oracle

`fem/fem_circle.py` (NGSolve 6.2.2604, the version installed here) is the
runner that produced `fem/results.jsonl`: the verifier re-ran its cheapest
saved mesh (`h1.0`, p = 4, 141 070 dof) unchanged and reproduced the saved
record to **1.4e-14** in every efficiency (`fem_r360_results.jsonl`).
`summary.json` is the mean of the three finest independent meshes, error
bar 8.34e-6 -- re-derived from `results.jsonl` by `v3_summary.py`.

### 4.2 Two NEW radii (this verifier's own FEM runs)

`verify_b/fem_circle_r.py` is the planner's runner with ONE change (`--rad`).
Three independent meshes per radius (the planner's choice: `h1.0_e20 p4`,
`h0.8_e30 p4`, `h1.0 p6`; 391 k - 466 k dof, 35-230 s each):

| radius | FEM spread (max dev from the mean) | R + T - 1 | flux check (independent of the Fourier projection) |
|---|---|---|---|
| 0.48 = 0.4 p | 6.1e-6 | <= 5.3e-7 | R 0.052214 / T 0.947786 (p6) = the order sums |
| 0.24 = 0.2 p | 1.47e-5 | <= 1.7e-7 | R 0.047018 / T 0.952982 (p6) |

The curved PMM, the verifier's maps, largest per-order distance to the FEM
(E along y, 18 entries; `v3_summary.json`):

| M | r = 0.24, c3 | r = 0.36, c3 | r = 0.48, c3 | c5 (inner 0.6): r = 0.24 / 0.36 / 0.48 |
|---|---|---|---|---|
| 4 | | | | 4.4e-4 / 7.0e-4 / 6.6e-3 |
| 5 | | | | 1.7e-4 / 5.9e-5 / 1.7e-3 |
| 6 | 6.0e-4 | 6.21e-3 | 3.7e-2 | 8.9e-5 / 4.0e-5 / 5.6e-5 |
| 7 | 1.35e-4 | 8.18e-4 | 2.4e-2 | 4.5e-5 / 2.0e-5 / 9.9e-6 |
| 8 | 4.9e-5 | 4.58e-4 | 6.2e-3 | 2.2e-5 / **8.9e-6** / **2.5e-6** |
| 9 | 2.8e-5 | 2.83e-5 | 2.8e-3 | |
| 10 | 1.1e-5 | 1.68e-5 | 4.2e-4 | |
| 11 | 5.2e-6 | | 9.4e-5 | |
| 12 | **2.2e-6** | | **1.6e-5** | |

* **r = 0.36 reproduced to every printed digit** (6.21e-3 .. 1.68e-5 at
  M = 6 .. 10 -- the verifier's map is the builder's geometry).
* **The agreement is not a single-fixture fact**: at r = 0.24 the c3 map
  lands at 2.2e-6 (inside the FEM's 1.5e-5), at r = 0.48 at 1.6e-5 (c3) and
  2.5e-6 (c5), against FEM bars 1.5e-5 / 6.1e-6.  The large disk converges
  later on the 3 x 3 map (rung changes 2e-2 .. 1e-4 at M = 6 .. 12): its
  outer cells are compressed to 0.46 of their `u` width where the circle
  comes within 0.12 of the cell edge.
* Two topologies at their top rungs: 2.5e-5 / 1.3e-5 / 1.3e-5 (r = 0.24 /
  0.36 / 0.48) -- each within the slower ladder's own last rung change.
* Four-fold symmetry `te(m, n) - tm(n, m)`: <= 8.4e-9 at M = 6, <= 1.2e-11
  from M = 8, every radius and topology -- the round-off floor (section 6).
* Closure at the top rungs 2e-12 .. 4e-8.

**Verdict B3: CONFIRMED, and strengthened to three radii.**

### 4.3 B4 -- the staircases (shipped solver, `v3_stair*_r*.json`)

r = 0.36, distance to the FEM: k = 1 (4 steps) 7.062e-2 / 7.068e-2 /
7.069e-2 at M = 8 / 10 / 12 (converged); k = 2: 5.33e-2 / 5.35e-2 / 5.36e-2 /
5.37e-2 at M = 5 .. 8; k = 4: 6.20e-3 / 5.65e-3 at M = 3 / 4.  The build's
7.07e-2 / 5.36e-2 / 5.65e-3 reproduced (and, new: the k = 1 and k = 2 values
are converged in M to 1e-4; the k = 4 one is NOT -- it moves 5.5e-4 from
M = 3 to 4, so "5.65e-3" is a pre-asymptotic reading).  Direction cosines
(step vs curved - previous): 0.930 (k1 -> k2), 0.991 (k2 -> k4) -- the build's
0.930 / 0.991 reproduced.  r = 0.48 (new): 0.156 (k = 1, M = 10, NOT
converged: 0.166 at M = 8), 4.73e-2 (k = 2), 2.73e-2 (k = 4); cosines 0.954
/ 0.854.  r = 0.24, k = 1, M = 7: 8.0e-2.  Monotone at every radius.
**Verdict B4: CONFIRMED** (with the k = 4 / M = 4 reading flagged as
unconverged).

## 5. F-B1 -- what the staggered basis needs across a seam (derived, then tested with a deliberately kinked map)

**Derivation.**  In `(u, v)` the curl equations keep their Cartesian form
with `eps' = det J J^-1 eps J^-T` (and `mu'`), and the unknowns are the
covariant `E' = J^T E`, `H' = J^T H`.  Across a grid line `u = const` the
interface conditions of the `(u, v)` problem are continuity of the
TANGENTIAL `E'_v = E . Phi_v`, `E'_z = E_z`, `H'_v`, `H'_z`, and of the
NORMAL fluxes `D'^u`, `B'^u`.  Row `u` of `det J J^-1` is `(y_v, -x_v)`, so

    D'^u = (y_v, -x_v) . D,       B'^u = (y_v, -x_v) . B

-- the physical normal flux times `|Phi_v|`.  Every interface quantity is
built from `Phi` and the TANGENT `Phi_v` alone; the normal derivative
`Phi_u` enters only `E'_u`, `H'_u`, which the staggered basis treats as
BROKEN across `u = const` lines.  Hence the conforming condition is: `Phi`
and `Phi_v` continuous (periodic up to the lattice vector at the seam).
And since `Phi(p_x, v) - Phi(0, v) = (p_x, 0)` FOR ALL `v` implies
`Phi_v(p_x, v) = Phi_v(0, v)`, the tangential check is the derivative of the
position check: **position periodicity along the whole side is the
necessary and sufficient condition**; the tangential Jacobian check is a
consistency check of the analytic derivative on the probe nodes (useful --
see defect D-1 for what happens when a curve's derivative is inconsistent
INSIDE the cell, where nothing checks it).

**Tests** (`v5_kink_win.json`; decision test
`test_verify_b_normal_kink_is_harmless_and_position_shift_is_not`):

* a piecewise-AFFINE map whose `x_u` is 1.67 left of the seam and 1.0 right
  of it (and jumps at both interior lines) vs the UNMAPPED solver on the
  physical walls -- the polynomial spaces coincide, so the answers must be
  equal: 1.6e-14 / 4.9e-14 at M = 5 / 7 (normal); 2.2e-8 / 3.0e-13 at conical
  (0.4, 0.7) rad (the 2.2e-8 at M = 5 is the F-B4 / F-V2 incident-projection
  effect of a non-polynomial `exp(i k . Phi)`, falling spectrally);
* the c3 circle RE-PARAMETRISED with kinked outer cells (`u` walls 0.20 /
  0.709 instead of 0.345 / 0.855, same vertex images and curves; seam `x_u`
  1.38 vs 0.56): 1.7e-9, 1.8e-10, 7.6e-12, 1.5e-12 from the plain c3 circle
  at M = 6 .. 9 -- the per-cell reparametrisation is affine, so this is the
  floor, while the rung change is 5.8e-3 .. 1.3e-5;
* FAIL-BEFORE: the `u = p` side's two interior vertices shifted 0.05 in `y`
  against `u = 0` -- REFUSED by `validate()`; forced past it, the answer is
  4.1e-3, 4.0e-3, 4.0e-3 wrong at M = 5 / 7 / 9 and does not converge away.

**Verdict F-B1: CONFIRMED (necessary and sufficient), with the sharper
statement above.**

## 6. F-B4 -- the round-off floor: the mechanism, its scope, and the fix

Measured (`v9_floor_win.json`, c3 r = 0.36, M = 6), against two controls
(the identity map on the same walls; a Phase-A `SeparableStretch` 0.05 p):

| | c3 | identity | Phase-A stretch |
|---|---|---|---|
| `Hsup`, n_orders = 3 | 98 x 450, FULL ROW RANK, cond 4.9 | 98 x 450, cond 6.8 | 98 x 450, cond 5.0 |
| `Hsup`, n_orders = 10 | 882 x 450, full column rank, cond 119, residual 2.4e-5 | cond 140, residual 2.9e-15 | cond 81, residual 1.9e-5 |
| R / T change per unit (relative) NULL-SPACE component added to `cinc` | 5.9e-4 | 1.6e-3 | 9.6e-4 |
| R / T change under a 1e-15 perturbation of `Hsup` (row-space solve) | 3.9e-16 | 1.3e-15 | 4.4e-15 |
| R / T vs n_orders = 2 / 3 / 5 / 7 / 8 / 10 / 12, against 14 | 1.1e-7 / 2.3e-7 / 1.3e-7 / 2.6e-7 / 1.6e-7 / 5.0e-8 / 9.0e-9 | <= 1.3e-15 | 1.0e-7 / 6.8e-8 / 1.3e-7 / 3.2e-7 / 1.7e-7 / 1.3e-7 / 5.4e-8 |

**Mechanism: the minimum-norm DRAW, not conditioning.**  `Hsup` is well
conditioned (cond 5); a 1e-15 perturbation of its row-space solve moves
nothing.  What moves R / T is the NULL-SPACE component of `cinc`: R / T are
not a function of the window equations alone (sensitivity 6e-4 .. 1.6e-3
per unit, on EVERY map including the identity).  Under the identity map the
incident plane wave is an exact discrete mode, the minimum-norm solution has
exactly zero null-space content, and nothing depends on the window (1e-15).
Under any non-polynomial map -- Phase B's curves AND Phase A's stretch alike
-- the plane wave is not in the span of the discrete half-space modes, the
fit's null-space content is decided by the 2-norm of the coefficient vector
(i.e. by the modes' normalisation inside degenerate multiplets, which a
1e-15 perturbation of the half-space weights rotates), and R / T inherit
both that draw (the ~1e-9 round-off floor) and a systematic WINDOW
dependence of 1e-7 at M = 6 (falling with M).  The floor is geometry
dependent: 1e-9 for r = 0.36, 1.1e-8 for r = 0.48 (section 3.3).

**Scope.**  The mechanism lives in Phase A's `cinc = lstsq(Hsup, rhs)` and
shows on Phase A's own stretch map at the same level; the Phase A verifier
(running in parallel) caught it as its **F-V2 / D4** ("incident overlap
under a non-polynomial map": `n_orders` dependence and vacuum-spacer
dependence, documentation requested).  F-B4 is therefore within the Phase A
verifier's scope and was caught there; nothing is duplicated here beyond
the mechanism split above.

**The builder's two recommended fixes, judged.**

1. "A window sized to be overdetermined whenever a map is present" is NOT
   sufficient: it removes the draw (round-off floor 7.7e-15, the build's
   measurement) but not the window dependence -- the overdetermined fit
   still moves 5.0e-8 between n_orders = 10 and 14 (c3) and 1.3e-7
   (stretch), because a least-squares fit of a non-representable incident
   wave depends on which orders it is fitted on.  It also contradicts the
   library's own order cap: the per-layer path RAISES for `n_orders > (q -
   1) // 2` ("retained order slots alias one another", open item O-9), and
   n_orders = 10 at M = 6 on a 3 x 3 grid is above that cap (7); the shared
   (mapped) path has no cap, so the recommendation would run where the
   per-layer path refuses (verifier note N-3).
2. "An exact modal decomposition of the mapped incident field" IS the right
   fix: project the incident `(E_t, H_t)` -- `J^T E0 exp(i k . Phi)` and its
   `H` partner -- onto the basis with the plain (unweighted, Kronecker)
   Gram, then solve against the half-space's FULL modal matrix (both
   directions; keep the forward amplitudes).  It is window-free by
   construction and reduces to today's answer under the identity map.
   Cost: one quadrature of the same kind as the cofactor far projector
   (`O(dof nq^2)`), one Kronecker Gram solve (cheap), and one dense LU of the
   `2 dof x 2 dof` half-space mode matrix -- `O((2 dof)^3)`, about a tenth of
   the half-space eig the solve already pays (an LU is ~10-20x cheaper than
   a non-symmetric eig of the same size).  Under 10 % of a mapped solve.



## 7. B5-B7 -- Bloch modes, fillets, and the r -> 0 limit (the verifier's own fillet map)

### 7.1 B5 (Bloch modes)

Not re-laddered independently: the B5 unit tests pass on both builds and
the circle's 5 x 5 rate (13.5x per rung, 1.5e-6 at 6 -> 7) and the square's
stall are reproduced by the build tests' own fixtures under the verifier's
run.  The mutation matrix shows the circle-mode test catches the
arc-as-chord mutant (m3).  **Verdict: CONFIRMED (by the tests, not by an
independent ladder).**

### 7.2 B6 / B7 -- the fillet ladder

The verifier's fillet map (`vfillet`, `Arc.through` on vertex images;
reproduces the builder's map to 1.05e-14) at r / side = 0.0125, 0.025,
0.05, 0.1, 0.2 (r = 0.0075 .. 0.12; the build had 0.01 / 0.02 / 0.05 / 0.1 /
0.2), M = 4 .. 8, and the sharp square on the shipped solver to M = 14
(`v6_*.json`, `v6_summary.json`).  Input 'te', shifts against the square at
M = 14:

| r / side | dR00 | dT00 | own rung 6 -> 7 / 7 -> 8, T00 |
|---|---|---|---|
| 0.0125 | -1.92e-5 | +2.48e-5 | 2.2e-4 / 4.2e-5 |
| 0.025 | -8.10e-5 | +1.23e-4 | 1.7e-4 / 2.7e-5 |
| 0.05 | -2.58e-4 | +4.56e-4 | 1.1e-4 / 1.7e-5 |
| 0.1 | -7.46e-4 | +1.72e-3 | 4.6e-5 / 1.3e-5 |
| 0.2 | -1.73e-3 | +7.35e-3 | 2.8e-5 / 1.4e-5 |
| sharp square (3 x 3), rungs M = 6 .. 14, T00 | | | 2.9e-4, 8.8e-5, 1.3e-5, 2.5e-5, 4.8e-6, 1.1e-5, 2.3e-6, 5.3e-6 |

The build's shifts at 0.05 / 0.1 / 0.2 (-2.63e-4 / +4.63e-4, -7.51e-4 /
+1.73e-3, -1.73e-3 / +7.36e-3) are reproduced to 2e-6 against the same
M = 12 square (here shown against M = 14, which moves them by <= 1e-5).  With the small radii carried to M = 8 the
ladder is **MONOTONE in r over the whole range** (the build's r / side
0.01 / 0.02 values at M = 7 were not: +2.4e-6 / -4.7e-5 in R00, inside their
own 2e-4 rung changes).  B7 (same-area square moves R00 the wrong way) is
reproduced by the unit test on both builds.  **Verdict B6 / B7: CONFIRMED.**

### 7.3 Why small fillets converge more slowly -- three hypotheses, measured

1. **The arc cell approaching the sliver contract** -- refuted.  The arc
   cell is `r / sqrt 2` wide (5.3e-3 = 4.4e-3 p at the smallest radius,
   above the 1e-3 p contract), and no solve of the ladder emitted a
   warning (`v6_*.json`, field `warnings`).
2. **The Duffy rule's n at a nearly degenerate vertex** -- refuted.  The
   fillet's arc cell is a SCALED COPY of one quarter-disk for every r: `det
   J / rho` is direction-independent (`l_min / l_max = 1.000`) and identical
   for r = 0.006 and 0.12 (`v1_geometry_win.json`), and the corner-rule
   moments converge identically at r / side 0.02 and 0.2 (2.4e-12 at
   n = 20, round-off at 24; `v2_moments_win.json`).  Nothing at the vertex
   depends on r.
3. **The map's conditioning** -- refuted for the same reason (scale
   invariance); the only r-dependent cells are the thin straight strips of
   width `r / sqrt 2` along each side, which are affine images of
   rectangles.
4. **What it is: the FIELD near the rounded corner, resolved by the BIG
   neighbouring cells.**  Outside a distance ~r from the fillet the field is
   that of a sharp 90-degree eps-4 / eps-1 corner, `|E| ~ rho^(lambda - 1)`
   with the first Meixner exponent `lambda = 0.806` (computed here from
   `eps2 tan(lambda alpha / 2) = -eps1 tan(lambda (pi - alpha / 2))`,
   `alpha = pi / 2`; the odd branch gives 1.194).  The big pillar and air
   cells adjacent to the corner (0.5 .. 0.6 wide) must represent that
   near-singularity down to the scale r; their node spacing near the cell
   end is ~L / M^2 (0.009 at M = 8 for L = 0.6), so for r below it the basis
   does not see the rounding and the fillet converges like the SHARP square
   (algebraically), and only once L / M^2 < r does it enter its own
   spectral regime.  Measured: at the 6 -> 7 rung the two smallest fillets'
   T00 changes (2.2e-4, 1.7e-4) equal the sharp square's (2.9e-4), while the
   r / side 0.2 fillet's is ten times smaller (2.8e-5); by 7 -> 8 the
   smallest has left the square's regime (4.2e-5 vs the square's 8.8e-5).
   A graded variant (two extra straight walls at 2 r inside the pillar,
   7 x 7 cells) does NOT accelerate it at equal dof: r / side 0.0125
   graded reads T00 rungs 2.0e-2, 1.3e-3, 1.4e-4 (M = 5 .. 7; 3528 dof at
   M = 7) against the ungraded 4.2e-5 at M = 8 (2450 dof); its M = 7 value
   sits 9.5e-6 (T00) / 1.3e-5 (R00) from the M = 14 square, consistent with
   the ungraded ladder.  The pillar-side grading leaves the AIR-side corner
   cells and the thin side strips unrefined, so this neither proves nor
   refutes the mechanism; the rung comparison above is the evidence, and a
   grading that pays is a Phase C measurement (section 13).

### 7.4 The r -> 0 limit -- is the stated claim honest?

The build doc (3.6) fits `dT00 = c + a r^2 + b r^3` through r / side 0.05,
0.1, 0.2 and reads the constant (+6.9e-5 T00, -5.4e-5 R00) as a bound "at
the 5e-5 level"; the unit test's docstring says the leading Bloch
`n_eff^2` approaches the square "as r^2".  Two problems:

* **A three-parameter fit through three points has zero residual; its
  constant is whatever the basis makes it.**  Through the same three
  points (against the M = 14 square): `c + r^2 + r^3` gives -5.0e-5 /
  +6.1e-5 (R00 / T00), `c + r^(2 lambda) + r^2` gives +6.2e-5 / +1.3e-4,
  `c + r^(2 lambda) + r^3` +1.4e-5 / -3.7e-5 -- a spread of 1.6e-4, all
  with residual 1e-16.  As
  written the build's number is not a bound on anything.
* **The physical leading power is not r^2.**  Rounding a corner whose
  field is `rho^(lambda - 1)` removes energy ~ `int_0^r rho^(2 lambda - 2)
  rho d rho ~ r^(2 lambda) = r^1.61`.  The measured local exponents of
  dR00 from r / side 0.0125 up are 2.08, 1.67, 1.53, 1.21 (mixed with the
  area term r^2 and higher orders); dT00 2.31, 1.89, 1.92, 2.09.

With FIVE radii (two more equations than unknowns) the bases separate:
`c + a r^(2 lambda) + b r^3` fits to 1.5e-6 (R00) / 5.6e-6 (T00), ten times
better than `c + a r^2 + b r^3` (1.8e-5 / 2.4e-5), and its constant is
+1.1e-5 (R00) / -2.6e-5 (T00) against the M = 14 square (+5.7e-6 / +3.3e-5
against the graded M = 6 square) -- i.e. **the r -> 0 limit IS the sharp
square, to ~3e-5**, which is the smallest fillet's own last rung (4.2e-5)
and the square reference's own uncertainty (rungs 1.1e-5, 2.3e-6, 5.3e-6 at
M = 11 -> 14).  So the build's CONCLUSION survives, but its stated evidence
does not: the honest statement is "five radii, fitted with the corner
exponent, put the limit on the square to 3e-5; the approach is r^1.6 for
R00", and the test docstring's "as r^2" should read "as r^2 or the corner
power r^1.6 -- the [8, 32] window accepts both (9.3 and 16) and rejects an
offset (1) and a linear approach (4)", which is what it actually decides
(defect D-3, documentation).


## 8. B9 -- an ellipse and a sinusoidal wall against exact-form-factor RCWA (the verifier's own shapes)

Ellipse 0.42 x 0.24 (aspect 1.75); sinusoidal ridge A = 0.08 on 0.35 .. 0.85,
`v` walls 0 / 0.36 / 0.9 / 1.2.  The ridge's form factor derived here
(Jacobi-Anger, `F(m, n) = J_-n(m G_x A) (e^{-i m G_x x2} - e^{-i m G_x x1}) /
(-i m G_x p_x)`, `F(0, n) = (x2 - x1) / p_x delta_n0`) checked against a 2048^2
pixel FFT: 7.4e-6 (the pixel error).  `v10_summary.json`:

| | ellipse | sinusoidal ridge |
|---|---|---|
| curved ladder, rung change M = 6 -> 11 | 9.6e-3, 3.6e-4, 7.3e-4, 1.9e-5, 2.9e-5 | 2.3e-4, 4.2e-5, 8.1e-6, 1.3e-5, 3.2e-6 |
| closure at M = 11 | 5.7e-9 | 9.2e-11 |
| four-fold symmetry broken (te(m,n) - tm(n,m)) | 0.150 | 0.420 |
| RCWA (Laurent, exact FF), distance to curved M = 11, N = 9 .. 33 | 2.0e-2, 1.4e-2, 1.1e-2, 9.4e-3, 7.9e-3, 6.7e-3, 5.8e-3 | 5.9e-3, 3.5e-3, 2.5e-3, 2.0e-3, 1.8e-3, 1.6e-3, 1.4e-3 |
| distance x N | 0.181 .. 0.196 (flat to +-4 %) | 0.043 .. 0.053 (not monotone) |
| LOCAL rate p from successive differences | 1.76, 0.49, 0.69, 1.08, 0.51 | 1.47, 1.69, 1.54, 0.96, 0.17 |
| 3-term fit `r_inf + a / N + b / N^2` (N >= 13, 6 points): `r_inf` to curved | 1.2e-3 (leave-one-out spread 8.7e-4) | 7.5e-4 (LOO 4.5e-4) |
| 2-term fit `r_inf + a / N`: `r_inf` to curved | 4.7e-4 | 8.8e-5 |
| the build-style 9 / 13 Richardson pair | 2.4e-3 | 1.8e-3 |

**Is the Richardson-extrapolated RCWA a legitimate reference?**  As a
CONSISTENCY reference at the 1e-3 level, yes: the distance falls like 1 / N
on average (distance x N flat to 4 % for the ellipse), and every
extrapolation lands within ~1.4x its own spread of the curved answer (the
curved answer sits inside the extrapolation error).  As an ACCURACY oracle
at the curved ladder's 1e-5 level, no: the LOCAL rate is not 1 / N -- it
wanders 0.2 .. 1.8 from pair to pair (Laurent-rule RCWA on a curved
dielectric boundary converges non-monotonically), so a two-point 1 / N
Richardson has an error of the size of the distance it extrapolates, and
the fits disagree with each other by 1e-3.  The build doc's "1 / N rate" is
true of the envelope, not of successive pairs; its unit bars (Richardson
within 6e-3) are correspondingly a 1e-3-class check.  The ellipse's broken
symmetry (0.15) and the sinusoid's mirror (B9 test) are the sharp checks.
**Verdict B9: CONFIRMED at the 1e-3 level the RCWA can certify; the curved
ladders themselves converge to 1e-5.**


## 9. B10 -- oblique (25 deg) and conical (25 deg, phi 40 deg); y-momentum; F-B6

### 9.1 Re-measured on the verifier's maps (`v7_summary.json`)

| quantity | theta 25, phi 0 | theta 25, phi 40 |
|---|---|---|
| c3 rung change M = 6 -> 9 | 1.7e-2, 6.2e-3, 8.1e-4 | 3.0e-2, 5.7e-3, 1.7e-3 |
| c3 closure M = 6 .. 9 | 1.9e-3, 6.8e-4, 4.1e-5, 1.3e-5 | 5.4e-3, 4.8e-4, 2.0e-4, 1.3e-5 |
| c5 (inner 0.6) rung M = 4 -> 6; closure at 6 | 1.8e-3, 1.6e-4; 1.0e-6 | 3.1e-3, 1.9e-4; 1.2e-6 |
| topologies (c3 M = 9 vs c5 M = 6, ~1200 dof each) | 2.9e-4 | 2.8e-4 |
| RECIPROCITY, channel -> (-1, 0) vs its reversal, c3 M = 6 .. 9 | 3.1e-6, 8.2e-7, 4.1e-8, 9.7e-9 | 3.9e-5, 9.4e-6, 5.1e-7, 1.3e-7 |
| reciprocity, c5 M = 4 .. 6 | 4.7e-7, 4.5e-8, 6.6e-10 | 1.7e-5, 8.0e-7, 1.4e-8 |
| reciprocity, order (0, -1) | -- | c3: 3.7e-5, 1.7e-5, 5.6e-7, 7.1e-7; c5: 3.0e-5, 1.5e-6, 6.4e-7 |
| WRONG pairing (the reversed run's specular channel) | 0.09 .. 0.11 | 0.09 .. 0.12 |

Reverse angles computed here: (24.2498, 0), (35.2731, -28.0614),
(40.4136, 119.9586) -- the build's.  The build's c3 reciprocity ladder
(3.1e-6, 8.3e-7, 4.1e-8, 9.7e-9) is reproduced to two digits by an
independently coded power-metric SVD; the closure and rung ladders match.
Not re-run: M = 10 (the build's 5.3e-7 / 2.9e-6 closure and the 1.2e-8
mirror reading).  **Verdict: CONFIRMED to M = 9 (c3) / 6 (c5).**

### 9.2 y-MOMENTUM on a y-uniform structure under a genuinely curved map (new)

A lamellar stripe (eps 4 on 0.5 < x < 0.9, straight walls) represented on a
4 x 4 map whose grid line `x = 0.22 + 0.1 sin(2 pi y / p)` runs through the
AIR (a fictitious curved wall: same material on both sides), conical
incidence (25, 40) deg.  The structure has no y-dependence, so every order
with n != 0 must carry zero power; the control is the same 4 x 4 grid with
the curve flattened (A = 0), which is the unmapped solver's discretisation
(2.7e-12 from it at M = 6).  `v7_momentum_*.json`:

| M | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| curved: largest power in ANY n != 0 order | 1.8e-7 | 3.3e-8 | 8.4e-9 | 3.4e-9 | 1.3e-9 |
| flat control | 1.8e-10 | 9e-15 | 2e-19 | 3e-24 | 4e-26 |
| curved vs unmapped, (m, 0) orders | 3.5e-4 | 2.7e-5 | 3.9e-6 | 2.3e-6 | 1.4e-6 |
| curved closure | 2.1e-3 | 2.4e-4 | 2.6e-6 | 2.4e-7 | 5.5e-10 |

* The leak is a DISCRETISATION effect, not the incident projection: it is
  identical (to all printed digits) at `n_orders` = 3, 6 and 10 (M = 6) and
  at 3 / 9 (M = 5).
* It is three decades under the curved-vs-unmapped difference of the
  allowed orders at every M and falls monotonically (5x .. 2.5x per rung).
* The shipped (unmapped) solver's own ladder of this stripe at conical
  incidence moves 2.3e-3, 1.2e-4, 1.9e-4, 2.3e-5 per rung (M = 5 -> 9,
  `v12_stripe_M*.json`) -- so the curved representation's 1e-6 difference
  sits two decades inside the reference's own convergence; the slow
  conical convergence of a lamellar stripe on the 2-D staggered basis is a
  property of the SHIPPED solver (observation N-4, outside this phase).

**Verdict: y-momentum is conserved to the discretisation error, three
decades below the allowed-order error, under a non-separable curved map.**

### 9.3 F-B6 -- why the coarse topology "loses decades only at oblique incidence"

Uniform eps-4 film vs the Airy slab (`v7_film_*.json`), theta = 25 deg:

| map (dof at M) | M = 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| c3 circle r = 0.36 (q = 3 (M - 1)) | 7.2e-4 | 6.8e-5 | 3.5e-6 | 2.1e-7 | 1.0e-8 |
| c3 r = 0.24 / r = 0.48 | 3.1e-5 / 4.8e-3 | 9.8e-6 / 3.2e-4 | 4.1e-7 / 2.2e-5 | 2.0e-8 / 1.5e-6 | 7.7e-10 / 8.4e-8 |
| c5 circle (q = 5 (M - 1)) | 2.0e-6 | 4.0e-8 | 5.8e-10 | 9.2e-12 | 7.7e-14 |
| identity transfinite on c3's walls | 2.9e-6 | 1.7e-8 | 7.3e-11 | 2.4e-13 | 5.8e-14 |
| UNMAPPED, c3's walls (middle cell 0.51) | 2.9e-6 | 1.7e-8 | 7.3e-11 | 2.2e-13 | 1.0e-13 |
| UNMAPPED, middle cell 0.72 = the disk diameter | 1.2e-4 | 4.2e-7 | 3.4e-9 | 2.1e-11 | 3.5e-14 |

**Derivation.**  At normal incidence the film's exact field is the
CONSTANT `E0`; in `(u, v)` it is `J^T E0` -- the map's own derivatives (low-
order trigonometric functions of the edge parameters), which the basis
resolves at 4e-11 by M = 6 on any of these maps.  At oblique incidence the
exact field is `J^T E0 exp(i k_t . Phi(u, v))`: the basis must resolve the
Bloch phase COMPOSED WITH THE MAP across each cell.  Two things then cost
resolution: (i) the physical extent per cell (an unmapped 0.72-wide middle
cell is 40x worse than a 0.51-wide one at M = 4 .. 6), and (ii) the map's
non-affinity -- the disk cell's arcs and four degenerate (det J = 0)
corners, and the outer cells compressed to 0.70 (r = 0.36) / 0.46 (r = 0.48)
of their `u` width -- a further ~3 decades at M = 6 over the unmapped
0.72 cell, growing with the disk (r = 0.24 / 0.36 / 0.48: 4.1e-7 / 3.5e-6 /
2.2e-5 at M = 6).  Nothing of the kind exists at normal incidence because
there is no phase to compose.

**Correction to F-B6 as written.**  "The 3 x 3 map converges 4-6 decades
behind the 5 x 5" compares the two at EQUAL M.  At equal dof (the eig cost)
they are equal: c3 M = 6 (450 dof) 3.5e-6 vs c5 M = 4 (450 dof) 2.0e-6;
c3 M = 8 (882 dof) 1.0e-8 vs c5 M = 5 (800 dof) 4.0e-8.  On the patterned
circle at oblique incidence the 5 x 5 is ~5x better per dof (rung 1.6e-4 at
1250 dof vs c3's 8.1e-4 at 1152), not decades.  What costs the decades is
the curved map itself relative to straight walls (3-4.7 decades at equal
dof on the film), for either topology.  Phase C's rule should therefore
budget M (or cells) for the PHASE across each physical cell at oblique
incidence, not prefer one topology.


## 10. The tests: mutation matrix

Each mutant is a fresh `git archive 91d00288` with ONE edit (asserted to
match exactly once; `v8_mutants.py`), the 18 build tests run against it
with PYTHONPATH pinned to the mutant tree (`mutants_buildtests.log`), then
the verifier's file (`mutants_verifytests.log`):

| mutant | caught by (build tests) | caught by (verifier's tests) |
|---|---|---|
| m1a corner rule disabled at ONE corner of the c3 disk cell | `test_b8_corner_rule_decision_on_the_circle` | `test_verify_b_duffy_is_spectral_on_every_shape_and_plain_gauss_is_n2` |
| m1b corner rule disabled in ONE whole singular cell | `test_b8_corner_rule_decision_on_the_circle` | the same |
| m2 C0 blend broken at one seam (cell (1, 0) blends the chord of its top arc) | 7 ids incl. `test_b1_transfinite_geometry_and_construction`, `test_b2_film...`, `test_b3_circle...`, `test_b10_*` | 3 ids |
| m3 every arc replaced by its chord | 5 ids incl. `test_b1_...geometry`, `test_b3_circle...`, `test_b5_circle_modes...` | 2 ids |
| m4 `Arc.key` drops the radius | **survives all** | **survives all** -- EQUIVALENT: both arc ends are pinned to the vertex images to 1e-12, so centre + angles + vertex images determine the radius; no two valid maps can collide |
| m5 the builder's circle pushed out by 1e-4 r on one axis (valid ellipse arcs) | `test_b1_transfinite_geometry_and_construction` (the AREA check) | (the verifier's tests use their own maps) |

Note on m1a / m1b: they are caught by a STRUCTURAL assertion (the corner
list), not by a physics bar -- with the corner rule off in one c5 / fillet
cell the R / T move only ~3e-7, under every R / T bar.  The verifier's
moment test catches them on the physics (the n^-2 family at n = 2 M + 8).
The build tests never assert the circle's four-fold SYMMETRY of the solve
(only that the ellipse breaks it); the verifier's
`test_verify_b_circle_fourfold_symmetry_is_exact` closes that (bar 1e-7,
measured 2.2e-9; a 1e-5 aspect defect reads 3e-6).

Verifier decision tests (`tests/unit/test_verify_pmm2d_curved_b.py`, spliced
into `.test_durations`):

| id | closes |
|---|---|
| `test_verify_b_fingerprint_sees_every_arc_parameter` | fingerprint content with walls and vertices FIXED (the build's test moves the walls too) |
| `test_verify_b_circle_fourfold_symmetry_is_exact` | the solve's four-fold symmetry (never asserted) |
| `test_verify_b_normal_kink_is_harmless_and_position_shift_is_not` | F-B1, both directions |
| `test_verify_b_corner_rule_on_a_regular_cell_is_the_tensor_rule` | claim 5's mutation (round-off, not bitwise) |
| `test_verify_b_duffy_is_spectral_on_every_shape_and_plain_gauss_is_n2` | the quadrature decision on circle, ellipse AND fillet, and the n^-2 rate |
| `test_verify_b_circle_lands_on_a_second_fem_radius` | B3 at r = 0.24 against the verifier's FEM, fail-before staircase |
| `test_verify_b_inconsistent_curve_derivative_is_refused` | D-1, strict xfail until the edit lands |



## 11. Docs

* **BUILD doc numbers vs JSON.**  Every number of sections 2-5 that this
  verifier re-measured matches its JSON and the re-measurement (b2_compare
  109 / 109; b3_film 8.06e-7 .. 5.42e-14; b3_film_oblique 7.15e-4 .. 9.95e-9;
  b4 ladders; b0 seam mismatches 6.21e-2 / 2.53e-2 / ...; cost 0.104 /
  0.057 s, 3200 corner points = 8 n^2).  Corrections requested:
  (a) section 4.4 / CHANGELOG "a ~1e-9 round-off floor": it is 1e-9 for the
  r = 0.36 fixture and 1.1e-8 at r = 0.48, and it is accompanied by a
  systematic `n_orders` dependence of ~1e-7 at M = 6 (not round-off) --
  state both; (b) section 4.4's alternative fix "a window sized to be
  overdetermined" does not remove the window dependence (section 6) and
  conflicts with the per-layer path's order cap -- drop it or qualify it;
  (c) section 3.6 and the B6 unit-test docstring on the r -> 0 limit
  (section 7 here); (d) section 3.4 "5.65e-03 (k = 4, M = 4)" is an
  unconverged staircase reading (5.5e-4 move from M = 3); (e) section 3.8
  "approaches it like 1 / N" -- the envelope does; successive pairs do not
  (local rate 0.2 .. 1.8), so the Richardson pair is a 1e-3-class reference
  (section 8).
* **CHANGELOG.**  Claims checked: 109 / 109 bytes (re-confirmed 37 / 37 on
  independent fixtures, two builds); 7.8e-6 at M = 11 inside 8.3e-6 (yes);
  staircases 7.1e-2 .. 5.7e-3 converging toward the curved answer (yes; the
  16-step value pre-asymptotic); film 5.4e-14 (yes); corner rule
  "round-off at 16 nodes per direction" (yes at M = 6; at M = 8 2.7e-13 at
  16, round-off at 20); "R / T 3e-07" for the plain rule (3.9e-7 measured).
  Item (a) above applies to its last bullet.  `cmap=` is accepted by both
  `PMM2DStackPure` and `pmm_jones_2d_staggered` (checked);
  `pmm_efficiency_2d_staggered` refuses it with a pointer (checked).
* **`_curvemap.py` docstrings.**  Readable by an optical physicist: every
  term is defined before use (Jacobian, metric, covariant components,
  effective tensors with references; singular vertex; Gordon-Hall formula
  written out; the parametrisation caveat of `EdgeCurve`).  Requested: in
  `CellMap.validate`, say that position periodicity along the whole side
  IMPLIES the tangential-derivative periodicity (the tangential check then
  tests the analytic derivative), and in `EdgeCurve` that the analytic
  derivative must be d/ds of the value (and that `TransfiniteMap` checks it
  once D-1 is applied); in `singular_vertices`, that a NEARLY antiparallel
  corner (beyond 1e-10) is not flagged and falls to the tensor rule, whose
  node criterion then runs to the cap and warns.
* `python -m mypy`: no issues (33 source files).  WSL `ruff check .`: all
  checks passed (the verifier's own files included).  No forward version
  token: the CHANGELOG entry sits under `[Unreleased]`, no `5.50` / `0.x`
  token in the Phase B diff or in this verification's files (grep).


## 12. Defects, with the exact edits requested

**D-1 (P2, silent-wrong on a public extension point).  `TransfiniteMap`
does not check an edge curve's derivative against its value.**  The map's
positions come from `crv(s)[0]`, its Jacobian from `crv(s)[1]`; nothing
ties them.  A `Sinusoid` subclass with a 10 % derivative bug is accepted and
moves R / T by 3.2e-2 (M = 5) / 4.6e-3 (M = 7) (`v5_curveder_win.json`).
`EdgeCurve` is in `__all__` and its docstring invites user curves; the
periodicity check sees only boundary edges.  Requested edit, in
`TransfiniteMap.__init__` (`lumenairy/elements/pmm/_curvemap.py`),
immediately before `self.curved_edges[(kind, i, j)] = crv`:

```python
            # the Jacobian is built from the curve's ANALYTIC derivative and
            # the positions from its value: they must agree, or the map's
            # Jacobian would not be the derivative of its positions (a user
            # curve with a derivative bug gives a silently wrong answer)
            sp = np.linspace(0.1, 0.9, 5)
            hd = 1e-6
            fd = (crv(sp + hd)[0] - crv(sp - hd)[0]) / (2.0 * hd)
            an = crv(sp)[1]
            dmis = float(np.max(np.abs(fd - an)))
            if not dmis <= 1e-6 * max(scale, float(np.max(np.abs(an)))):
                raise ValueError(
                    f"TransfiniteMap: edge {key!r}: the curve's derivative is "
                    f"not d/ds of its value (mismatch {dmis:.3e} against a "
                    f"central difference) -- the map's Jacobian would not "
                    f"match its positions.")
```

Verified in a patched tree (`C:/tmp/vcurved_b_fix`): the strict-xfail test
flips (XPASS(strict)), and the Phase A + B tests and the other verifier
tests pass there (s 14).  When it lands, drop the `xfail` marker.

**D-2 (P3, documentation) -- F-B4 wording.**  BUILD 4.4 and the CHANGELOG
call the effect "a ~1e-9 round-off floor"; it is 1e-9 at r = 0.36 and
1.1e-8 at r = 0.48 (geometry dependent), and it comes with a systematic
`n_orders` dependence of 1e-7 at M = 6 (not round-off).  Replace the
CHANGELOG clause by: "the curved solve's R / T depend on the far-field
window (`n_orders`) at ~1e-7 at M = 6 (falling with M) and carry a
geometry-dependent round-off floor of 1e-9 .. 1e-8, both from the windowed
least-squares incident projection inherited from Phase A (its verifier's
F-V2); measured, not changed in this phase".  In BUILD 4.4 delete or
qualify "or a window sized to be overdetermined whenever a map is present"
(it does not remove the window dependence, s 6, and exceeds the per-layer
order cap).

**D-3 (P3, documentation) -- the r -> 0 limit.**  BUILD 3.6: replace the
"fit constant ... bounds the limit at the 5e-5 level" sentence by the
five-radius statement of section 7.4 (or keep the three-point fit and say
that its constant is basis-dependent over 1.6e-4).  Unit test
`test_b6_fillet_modes_approach_the_square_as_r_squared`: the docstring's
"shrinks like r^2" -> "approaches the square with a power between the
corner exponent 2 lambda = 1.61 and 2; the window [8, 32] rejects an
offset (~1) and a linear approach (4)".

**D-4 (P3, documentation) -- F-B6.**  BUILD 4.6: the "four to six decades
behind" comparison is at equal M; at equal dof the topologies are equal on
the film and within 5x on the oblique pillar.  Replace the recommendation
"Phase C should prefer the 5 x 5 topology when the incidence is oblique" by
"at oblique incidence a curved map needs more M (or more, smaller cells)
than straight walls for the same accuracy, because the basis must resolve
the Bloch phase composed with the map; budget by dof, not by topology".

**D-5 (P3, documentation) -- B4 / B9 wording.**  BUILD 3.4: mark the
16-step staircase reading (5.65e-3 at M = 4) as pre-asymptotic (5.5e-4
from M = 3).  BUILD 3.8: "approaches it algebraically (~1 / N)" -> "its
envelope falls like 1 / N; successive pairs do not (local rate 0.2 .. 1.8),
so the Richardson pair is a 1e-3-class reference".

**Notes (no edit requested).**
* N-1: the corner rule's singular-vertex detection uses a 1e-10 parallelism
  tolerance; a NEARLY antiparallel user corner falls to the tensor rule
  and the node criterion warns at its cap -- acceptable, worth one sentence
  in `singular_vertices`.
* N-2: the identity transfinite map vs the per-layer unmapped path at
  conical incidence on an absorbing random cell differs by 5e-11
  (Windows) / 6e-14 (WSL) -- BLAS / route round-off of a different code
  path; the operator identity (4e-14) is the gate.
* N-3: the shared-grid (mapped) stack has no `n_orders <= (q - 1) // 2`
  cap; the per-layer path raises above it.  Phase C's window choice should
  respect one rule.
* N-4 (outside Phase B): the SHIPPED solver's y-uniform lamellar stripe at
  conical (25, 40) converges only 2.3e-3, 1.2e-4, 1.9e-4, 2.3e-5 per rung
  (M = 5 -> 9) on a 4 x 4 grid -- slower than one would expect of a
  corner-free lamellar structure; not investigated.

## 13. Ship recommendation, and what Phase C must carry

**SHIP Phase B to the feature branch**, with D-1 applied (a three-line
check; verified) and D-2 .. D-5 as documentation edits.  The physics is
right on every gate the verifier could reach independently: the corner
quadrature is the correct and exact tool, the circle lands on an
independent FEM at three radii, the seam condition is necessary and
sufficient, oblique / conical incidence is reciprocal and conserves
y-momentum under a non-separable curved map, and no shipped answer moved.

Phase C must carry:

1. **The incident projection (F-B4 = Phase A F-V2).**  Replace `cinc =
   lstsq(Hsup, rhs)` under a map by the exact modal decomposition (Gram
   projection of `(E_t, H_t)` of the pulled-back plane wave, then the full
   half-space mode matrix); NOT by an overdetermined window.  Gate: R / T
   independent of `n_orders` to round-off on the c3 circle and on a Phase A
   stretch, and identical to today under the identity map.
2. **D-1** if not applied in Phase B, plus a sentence in `EdgeCurve`.
3. **The shape layer's resolution rule at oblique incidence** (D-4): budget
   by dof against the Bloch phase per physical cell; do not hard-code a
   topology.
4. **Fillet primitives** (section 7): the shift approaches the sharp
   corner as `r^(2 lambda)`; small fillets (`r` below ~L / M^2 of the
   neighbouring cells) converge like the sharp square -- the primitive
   should either grade the neighbouring cells toward the fillet on BOTH
   sides (the pillar-side-only grading measured here did not pay at equal
   dof) or document that a small fillet costs the sharp corner's
   convergence.
5. **Symmetry and seam gates** for every primitive: four-fold symmetry of
   the solve where the shape has it (bar ~1e-7), the position-periodicity
   seam check, and the fingerprint over walls + vertices + curve content.
6. **The FEM oracle at oblique incidence** (still not run; s 15).



## 14. Test tails and reproduction

* Windows (CPython 3.14.6, numpy 2.4.4, scipy 1.17.1), the sweep = every
  `test_*pmm2d*`, `*stack2d*`, `*stagger*` file (Phase A's 13 + Phase B's 18
  + the verifier's 7 ids included) + census mechanism, public API,
  doc-consistency, except-budget, history relocation (45 files), `-n 6`, box
  loaded: **`1439 passed, 1 skipped, 1 xfailed in 828.85s`** (the skip is
  the shipped premise gate of `test_fix_pmm2d_mortar_round2.py` -- numpy and
  scipy load different OpenBLAS builds here; the xfail is D-1).
* WSL Ubuntu (CPython 3.12.3, numpy 2.4.6, scipy 1.17.1, BLAS pinned,
  `LUMENAIRY_DISABLE_JAX=1`, `lumenairy` from `/mnt/c/tmp/lum_vcurved_b`),
  Phase A + Phase B + verifier files: **`37 passed, 1 xfailed in 232.90s`**.
* The D-1 fix tree (`C:/tmp/vcurved_b_fix` = `91d00288` + the section 12
  edit), same three files: `1 failed, 37 passed` -- the one failure is the
  strict xfail turning into XPASS(strict), i.e. the edit closes D-1 and
  breaks nothing else.
* The verifier's file alone, Windows, box nearly idle: `6 passed, 1 xfailed
  in 21.03s` (slowest 8.5 s); durations spliced into `.test_durations`.
* `python -m mypy`: `Success: no issues found in 33 source files`.  WSL
  `ruff check .` and `ruff check` on the verifier's files: all checks passed.
  `python scripts/record_history_fingerprints.py --check`: "OK: every
  history document matches its module."
* Mutation matrix: `verify_b/mutants_buildtests.log`,
  `verify_b/mutants_verifytests.log` (section 10).

Reproduction (from `validation/probe_pmm2d_curved/verify_b`, BLAS pinned,
`PYTHONPATH=C:/tmp/lum_vcurved_b`; the queues are `queue.py <probe>
<jobs file> <P> <log>`):

```
python v1_geometry.py                       # maps, areas, C0, refusals, det J zero order
python v2_quadrature.py moments             # rules: gl, gj<a>, duffy, vduffy, vduffy_gj1, vduffy_sq
python v2_quadrature.py rt c3 6 <rule> <n>  # full solves on a rule (jobs_q.txt)
bash run_fem.sh                             # FEM: r = 360 provenance re-run, r = 480 / 240
python v3_circle.py curved c3 0.48 <M> ; python v3_circle.py stair 1 0.36 <M>   (jobs_circle.txt)
python v3_summary.py
python v4_bytes.py <PRE> pre ; python v4_bytes.py <THIS> post ; python v4_bytes.py compare
python v5_identity_kink.py identity|corner0|kink|curveder
python v6_fillet.py fillet <r/side> <M> [grade] ; python v6_fillet.py square <M> [grade] ; python v6_summary.py
python v7_oblique.py rung c3 <M> <th> <ph> ; python v7_oblique.py film <kind> <M> 25 0 ; python v7_oblique.py momentum <M> curved [n_orders]
python v7_summary.py
python v8_mutants.py make 91d00288 ; bash run_mutants.sh tests/unit/<file> <tag>
python v9_floor.py
python v10_shapes.py curved|rcwa ellipse|sine <M|n> ; python v10_shapes.py ffcheck|summary
python v11_bars.py ; python v12_stripe_ref.py <M>
bash run_wsl_probes.sh                      # second build (from WSL)
```


## 15. What could not be measured

* **The FEM at oblique / conical incidence** -- the planner's runner is a
  normal-incidence quarter cell (PMC / PEC mirrors); a Bloch-periodic full
  cell was not attempted.  The new radii were run at normal incidence only.
* **M = 10 and above at oblique incidence** (the build's 5.3e-7 / 2.9e-6
  closure and the 1.2e-8 mirror jump at M = 10) -- not re-run.
* **An independent B5 (in-plane Bloch mode) ladder** -- the build tests'
  fixtures were exercised (both builds, mutation matrix), not a separate
  verifier ladder.
* **A grading that demonstrates the small-fillet mechanism** -- the
  pillar-side grading tried here (to M = 7 for the smallest radius) does
  not accelerate at equal dof; an air-side + strip grading was not run;
  the graded SQUARE at M = 7 (3528 dof) was stopped unfinished after two
  hours (the graded square enters only as a second reference at M = 6).
* **The probe ladders on the second build** -- WSL ran the geometry,
  quadrature-moment, identity, corner-rule and byte probes (identical or to
  5.7e-9 relative on moments above 1e-10) and the unit tests; the heavy
  ladders (circle / fillet / shapes / oblique) are Windows only.
* **Wall times** -- the box carried up to ~50 single-threaded processes
  (this verifier's queues and two sibling agents'); every time is an upper
  bound.


# Verification: JAX gradients at symmetric points (2026-10-04)

Verifier: Claude Opus 5.5 (model ID `claude-opus-5-5`), independent of the
builder.  Object: the fix commit `dabe50f5` ("fix(jax): correct reverse-mode
gradients at symmetric points in rcwa_efficiency_2d and pmm_efficiency_1d"),
as merged in the integration commit `783b369d`, and its build record
`docs/audits/BUILD_JAX_SYMMETRIC_POINT_GRADIENTS_2026_10_03.md`.

Worktree `C:/tmp/lum_symgrad_verify`, branch
`verify/jax-symmetric-point-gradients` from `783b369d`.  PRE tree: a
detached worktree of `5ea82b44` (the fix's parent), removed after use.

Builds: Windows 11 (CPython 3.14.6, numpy 2.4.4, jax 0.11.0) and WSL Ubuntu
(CPython 3.12.3, numpy 2.4.6, jax 0.10.2).  Every probe asserts that
`lumenairy` was imported from the tree it was pointed at; BLAS threads 1 for
the byte comparisons, 2 elsewhere.  The machine was carrying two other full
test suites throughout, so every timing is an upper bound and only ratios
measured in one process are reported.

Evidence: `validation/probe_jax_symgrad_verify/` (probes `v*`, `b*`, `c*`,
`d*`; JSON `<probe>[_<pre|post>]_<win|wsl>.json`; runners `run_b.sh`
(Windows, PRE and POST) and `run_wsl.sh`).  Tests:
`tests/unit/test_verify_jax_symmetric_point_gradients.py`.

## 0. Method and words

* **Own fixtures**, chosen to differ from the builder's:
  * RCWA cell: 24 x 24 pixels, period 0.9, wavelength 1, a square post
    (pixels 7..16, eps 5) in eps 1.8, depth 0.37, n_sup 1, n_sub 1.52,
    `n_orders_x = n_orders_y = 2` (25 orders) and 3 (49 orders).  The post is
    symmetric under both mirrors and x <-> y, so the cell is four-fold (C4v).
    Directions of the parameter t: `x` widens the post along x (keeps both
    mirrors, breaks the 90-degree rotation); `corner` adds one pixel off a
    corner (breaks every symmetry); `diag` adds two opposite corner pixels
    (keeps one diagonal mirror); `keep` scales the whole post (keeps C4v).
  * Weakly modulated cells (eps 2.25 + delta x pattern) to create
    NEAR-degenerate clusters that are not symmetry-forced and clusters of more
    than two members (section 1.3).
  * 1-D grating: period 0.85, ridge index 2.3, groove 1.35, duty 0.4, depth
    0.31, n_sup 1, n_sub 1.6, wavelength 1, degree 10, `stabilize=False`.
* **AD** = `jax.jit(jax.jacrev(f))`.  **FD** = the NumPy solve (RCWA with
  `symmetry=False`, the same full solve the JAX path runs), central
  differences at h, h/2, h/4 (h = 2e-2 for permittivity parameters, 4e-3 rad
  for the angle) and Richardson of the last two.  **Premise**: the ratio of
  successive rung changes must be 4 for a clean h^2 law; it read 3.68 - 4.59
  (norm over outputs) on every row reported below (the extremes, 3.68 and
  4.59, on low-gradient rows at the FD resolution).  **FD
  resolution**: the change of the Richardson value between the two rung
  pairs; reported where it limits the comparison.
* **Error** = max |AD - FD| / max |FD| over all R and T of all orders.
* **Rule off** = `rcwa._core._EIG_CLUSTER_GAP_REL = 0` at trace time, which
  makes `_jax_eig_cluster_adjoint` run the plain composition.  It reproduces
  the PRE gradient: the PRE tree gives the same numbers to all printed digits
  (RCWA x-widening TE / TM 0.266 / 0.0733, 1-D twin at 0 TE / TM 1.6 / 0.893;
  `v1_rcwa_x_te_n2_x_tm_n2_pre_win.json`, `v2_pmm1d_pre_win.json`).

## 1. Claim (1): the two routed entries are now correct at symmetric points

**Verdict: CONFIRMED on both builds**, on every fixture below, including
the stress cases the builder did not run.

### 1.1 RCWA, `rcwa_efficiency_2d` on JAX input (`v1_rcwa`)

| fixture (direction, pol, orders) | rule ON win / wsl | rule OFF (pre-fix) win / wsl |
|---|---|---|
| C4v, x, TE, 5 x 5 | 1.6e-10 / 2.9e-10 | 0.27 / 0.28 |
| C4v, x, TM, 5 x 5 | 7.1e-11 / 6.2e-11 | 0.073 / 0.076 |
| C4v, x, TE, 7 x 7 | 3.2e-10 / 2.3e-10 | 0.30 / 0.11 |
| C4v, x, TM, 7 x 7 | 4.5e-11 / 3.4e-11 | 0.079 / 0.028 |
| C4v, corner, TE / TM | 5.3e-10 / 2.5e-9 ; 4.4e-10 / 1.2e-9 | 0.13 / 0.13 ; 0.35 / 0.35 |
| C4v, diag, TE | 3.6e-10 / 2.1e-10 | 0.13 / 0.35 |
| C4v, **keep** (symmetry kept), TE / TM | 1.3e-11 / 1.0e-11 ; 1.6e-11 / 1.1e-11 | same as ON |
| C4v **lossy** post (eps 5 + 0.8i), x, TE / TM | 1.6e-10 / 3.7e-11 ; 1.1e-10 / 7.3e-11 | 0.31 / 0.070 ; 0.30 / 0.066 |
| C4v, `formulation='li'`, x, TE / TM | 1.1e-10 / 1.2e-10 ; 2.0e-10 / 4.5e-11 | 0.32 / 0.057 ; 0.56 / 0.10 |
| conical theta 0.2 / phi 0.3, corner, TE / TM | 2.0e-9 / 6.1e-10 ; 2.5e-9 / 1.6e-9 | same as ON (no cluster) |
| oblique theta 0.15, x, TM | 9.7e-11 ; 7.3e-11 | same as ON |

Rows with one polarization read `win / wsl`; rows with two read
`win TE / win TM ; wsl TE / wsl TM`.  (The corner and conical rows are at the
FD's own resolution, 0.6 - 3e-9.)
The pre-fix error is build-dependent (0.11 vs 0.30 on the same fixture) --
the LAPACK-basis signature the record describes.  The symmetry-keeping
direction is exact with and without the rule, as the theory says (the
in-cluster block of a symmetry-keeping perturbation is a multiple of the
identity).

### 1.2 1-D PMM twin, `pmm_efficiency_1d` d / d(angle) (`v2_pmm1d`)

| point | TE ON win / wsl | TE OFF | TM ON win / wsl | TM OFF |
|---|---|---|---|---|
| exactly 0 | 4.8e-10 / 4.7e-10 | 1.6 | 6.1e-11 / 5.7e-11 | 0.89 |
| 1e-8 rad | 4.8e-10 / 4.8e-10 | 1.1e-7 | 7.5e-11 / 8.0e-11 | 3.0e-8 |
| 1e-6 .. 1e-3 rad | <= 5.8e-10 | <= 6.1e-10 | <= 8.4e-11 | <= 8.4e-11 |
| lossy ridge (2.3 + 0.05i), 0 | 1.0e-9 / 1.0e-9 | 4.1 | 3.0e-11 / 2.4e-11 | 0.97 |
| ridge index (symmetry kept), 0 | 2.6e-11 (both) | 2.6e-11 | 1.1e-11 (both) | 1.1e-11 |

FD resolution 7.7e-9 (TE) / 8e-10 (TM).  FD-free mirror identity
d X_{+m} / d angle = - d X_{-m} / d angle at 0, relative to the largest
entry: ON <= 2.2e-11 (both builds), OFF 0.30 (TE) / 0.43 (TM), lossy OFF
0.32 / 0.62.

### 1.3 New-breakage probes (`v3_neardeg`, `v4_hess_vmap`)

* **Accidental near-degeneracy that is not a symmetry.**  eps 2.25 + 0.01 x
  a fixed random pattern (no symmetry), conical theta 0.2 / phi 0.3: 23 pairs
  closer than gap_rel = 1e-6 of the spectrum, none exact, 7 of them
  propagating.  Corner direction: TE 3.9e-9 / 1.5e-8, TM 1.1e-8 / 5.9e-9
  (win / wsl; FD resolution 4.5e-9 .. 1.9e-8), identical to the rule-off
  values -- the rule does not harm it.  Same with eps 2.25 + 1e-3 x random at
  normal incidence (clusters of 2, 2, 3 members, none exact): <= 1.4e-8.
* **Clusters with more than two members.**  eps 2.25 + 1e-4 x post (C4v,
  near-uniform): exact E pairs chain with the uniform medium's eight-fold
  multiplets into clusters of up to 8 members (44 members in all).  x
  direction, error on the diffracted orders alone (their O(delta^2)
  efficiencies would hide in a max over the zero order): ON 2.1e-10 /
  1.5e-10 (TE), 7.1e-9 (TM, both); OFF 8.3e-6 / 9.8e-6 (TE), 3.1e-6 /
  3.7e-6 (TM).  eps 2.25 + 1e-2 x post (one 3-member cluster): ON <= 1.5e-8,
  OFF 1.2e-4 .. 3.0e-4.
* **Lossy symmetric cell**: section 1.1 (exact).
* **Uniform cell differentiated toward a pattern** (eps 2.25 + t x post at
  t = 0; the eig the rule sees is the uniform medium's, with exact
  multi-member clusters, while the consumer selects the analytic uniform
  modes; `v7_uniform`, Windows): finite, 1.4e-9 (TE) / 1.5e-9 (TM) / 6.4e-9
  (TE at theta 0.2), equal to the rule-off values.
* **`jax.vmap(jax.grad)`** over [0, 1e-3, -2e-3, 1e-2] (the first point on
  the symmetric configuration, where the batched cluster flag turns the
  rule's `lax.cond` into a select) against per-point `jax.grad`: RCWA 3.2e-11
  / 1.5e-10, 1-D twin TE 2.5e-16 / 0, TM 6.0e-13 / 1.7e-13 (Windows / WSL).
* **`jax.hessian`** through a parameter that enters the eig (eps_cell, the
  angle) raises `NotImplementedError` (JAX's non-symmetric eigenvector JVP)
  on BOTH trees (`v5_hess_tree`, PRE and POST, both builds).  Not a
  regression; see P3-2.  (The FD of the AD gradient is therefore the only
  second-order check: it satisfies the h^2 premise at 3.998 and agrees with
  the NumPy second difference to 1e-6 relative.)

### 1.4 The mechanism the record does not state (`v3_neardeg`, mechanism rows)

The record says the RCWA operator needs no Gram matrix because a split
pair's members "either fall in different symmetry sectors (orthogonal) or
... still lifted cleanly".  Measured on the weak no-symmetry conical
grating: the Euclidean lift of a cluster of DISTINCT eigenvalues gives
complex eigenvalue shifts up to 1.54 d (d = split_rel x max|lam|), and 16
member evaluations (of 46 members x 4 stencil points) put a PROPAGATING
layer root on the other branch of `_sqrt_decay` -- the exact failure that
made the folded 1-D operator wrong by 4.7e3.  The RCWA gradient is
nevertheless right (1.3 above) because a LAYER mode's branch is a
forward / backward relabelling that the S-matrix is invariant under
(lam -> -lam with V -> -V; the `_sqrt_decay` docstring), whereas the 1-D
twin's HALF-SPACE modes define the outgoing directions and are not.  On the
C4v-based fixtures the shifts are real to 1e-8 d and no root changes branch.
This matters for the remaining families (section 5): any twin whose rule
consumer contains half-space or region modes built from a lifted eig must
hand the rule a Gram that keeps those shifts real (the pencil), or the
gradient will fail as the fold did.

## 2. Claim (2): forward values byte-identical

**Verdict: CONFIRMED**, on wider fixtures than the builder's, both builds.

* `b_bytes` (NumPy, plus JAX EAGER forward of the unrouted entries):
  rcwa_efficiency_2d Laurent / Li with `symmetry=False` and `'auto'` (the
  even-sector fold, which now calls the refactored `_scalar_PQ` /
  `_tensor_PQ`), oblique, conical, lossy, uniform, C1, 7 x 7 orders;
  rcwa_jones_2d in-plane uniaxial tensor cell (Laurent / Li, normal /
  conical, symmetry auto / False) and tilted (out-of-plane) director
  (generator path); pmm_efficiency_1d TE / TM at 0 and 0.2 rad with
  `stabilize` True / False, and its JAX eager forward; pmm_jones_1d in-plane
  and tilted tensors, and its JAX eager forward: **40 / 40 SHA-256 equal PRE
  vs POST on Windows, 40 / 40 on WSL.**
* `b2_bytes_jax` (the routed entries on JAX input, eager and `jax.jit`):
  C4v TE / TM / Li-TM, lossy, conical C1, the weak near-degenerate conical
  cell; 1-D twin TE / TM at traced angle 0 and 0.2: **20 / 20 equal on
  Windows, 20 / 20 on WSL.**

No NumPy code path changed: the refactor of `_layer_P_matrix`,
`_tensor_PQ(..., Ez_inv=)`, `_layer_eigenmodes[_tensor](eig_pair=)` and the
`_jpmm_sem_modes` split is arithmetic-identical where it is reachable from
NumPy.

## 3. Claim (3): the refactor and the pencil

**Verdict: CONFIRMED.**  The folded operator handed to the rule
(`v2_pmm1d`, column `fold`; the rule's problems replaced by
`(solve(B, A), None)`):

| angle | TE fold / pencil | TM fold / pencil | lossy TE fold / pencil |
|---|---|---|---|
| 0 | 4.8e-10 / 4.8e-10 | 5.4e-11 / 6.1e-11 | 1.0e-9 / 1.0e-9 |
| 1e-8, 1e-6 | <= 5.8e-10 / <= 5.8e-10 | <= 7.5e-11 / <= 8.4e-11 | -- |
| **1e-5** | **5.6e3** / 4.2e-10 | **2.6e3** / 4.5e-11 | **2.3e4** / 8.1e-10 |
| **1e-4** | **5.1e3** / 5.6e-10 | **3.1e2** / 6.1e-11 | -- |
| 1e-3 | 5.1e-10 / 5.1e-10 | 7.5e-11 / 7.5e-11 | -- |

identical on both builds to two digits.  The builder's 4.7e3 / 7.0e4 on its
grating; same class, same window (where the half-space pairs are split
inside gap_rel but by more than the lift).

Other code paths checked for a changed NumPy route: `_tensor_PQ` now takes
the caller's `Ez_inv` (same `inv(EZZ)`), the even-sector folds call the
refactored blocks, `_layer_eigenmodes_tensor` raises on `eig_pair` with an
out-of-plane or slanted layer (a new guard, unreachable from the shipped
callers), and the rule's short-circuit returns the plain composition when no
eig argument is traced.  All covered by the 40 / 40 + 20 / 20 hashes.

## 4. Claim (4): cost

**Verdict: PARTLY CONFIRMED** (ratios hold; vmap cost real; no public way to
avoid it -- P2-2).  `c_cost`, Windows, one process, rule ON vs OFF
interleaved, best of 9 (jit) / 5 (vmap) / 3 (eager):

Ratios ON / OFF, Windows ; WSL:

| case | jit grad run | jit grad first call (compile) | vmap(grad), 4 points | eager grad (first call) | jit forward |
|---|---|---|---|---|---|
| RCWA C4v (cluster) | 4.0x ; 5.4x | 3.0x ; 3.9x | 3.3x ; 3.9x | 3.0x ; 3.0x (17.9 / 3.8 s ; 23.7 / 4.7 s) | 1.13x ; 0.79x |
| RCWA conical (no cluster) | 1.26x ; 0.93x | 2.8x ; 2.9x | **3.5x ; 5.4x** | 1.37x ; 1.43x | 1.00x ; 1.05x |
| 1-D twin at 0 (cluster) | 4.8x ; 4.5x | 3.9x ; 4.6x | 4.1x ; 5.8x | 4.4x ; 4.7x (16.4 / 3.4 s ; 19.8 / 3.3 s) | 1.02x ; 0.97x |
| 1-D twin at 0.2 (no cluster) | 1.48x ; 1.00x | 3.4x ; 3.4x | **3.7x ; 3.7x** | 1.43x ; 1.43x | 1.00x ; 1.01x |

(absolute, Windows: RCWA C4v jit gradient 0.035 s ON against 0.0087 s OFF;
1-D twin 0.0091 / 0.0019 s.)  Reading: with a cluster 4 - 5.4x (record:
4 - 13x, on a more loaded box); without one 0.9 - 1.5x (record: 1.1 - 1.3x
on the 1-D twin); compile 2.8 - 4.6x (record 2.5 - 5x); forward unchanged
(the 0.79x is load scatter).  **The vmap cost is real**: a vmapped gradient
without any cluster pays 3.5 - 5.4x here (record 7 - 16x), because the
batched flag makes both branches of the `lax.cond` run.  An eager
(un-jitted) gradient at a cluster also pays a first-call cost of 16 - 24 s
against 3.3 - 4.7 s.

**Escape hatch: none documented.**  `rcwa_efficiency_2d` and
`pmm_efficiency_1d` take no argument for the rule; the only switch is the
private module constant `rcwa._core._EIG_CLUSTER_GAP_REL` (the staggered
twin's `_E3_EIG_CLUSTER_GAP_REL` is private too), and neither the
CHANGELOG nor a docstring tells a user who batches gradients over
configurations known to be asymmetric how to avoid the 3.5x.  Finding P2-2.

## 5. Claim (5): the twins not changed by the fix (`d_twins`, `d2`, `d3`)

| entry | fixture and parameter | win | wsl | verdict |
|---|---|---|---|---|
| `pmm_jones_1d` | isotropic tensors on the verifier grating, d / d(angle) at exactly 0 | **0.97** | **0.97** | DEFECT confirmed (builder: 2.7 on its grating) |
| `pmm_jones_1d` | same, 1e-5 / 1e-3 rad | 3.5e-10 / 3.9e-10 | 3.5e-10 / 3.9e-10 | correct off the point |
| BOR-SEM `BORStack(basis='sem')` | Rbig 2.5, n_hs 1.3, eps 4.5, thk 0.4, degree 6, d / d(in-plane anisotropy) at the isotropic layer, m = 0 / 2 | 2.9e-11 / 2.8e-11 | 2.9e-11 / 2.8e-11 | correct, confirmed |
| `rcwa_jones_2d` (UNMEASURED by the builder), Laurent | C4v post as isotropic tensors, x-widening (t I) | **6.1e-2** | **4.4e-2** | DEFECT (cluster class, build-dependent); 1e-4 off: 2.9e-11 / 1.8e-11 |
| `rcwa_jones_2d`, Laurent | d / d(eps_xy) on the post at the isotropic C4v cell; every central difference vanishes (<= 3.6e-13), the truth is 0 | AD **0.088** absolute | **0.083** | DEFECT (the Berreman-like case); 1e-4 off: 2.9e-7 / 4.4e-8 |
| `rcwa_jones_2d`, **Li** | same two parameters, AND a cell with no symmetry (eps 2.25 + 0.6 x random), corner pixel | **0.62 / 0.24** | **0.57 / 0.24** | P1, NOT the cluster class: wrong at every point (0.59 at 1e-4, 0.57 at 0.05 off; 0.21 - 0.24 on the C1 cell) |

The `rcwa_jones_2d` Li defect is pre-existing (identical numbers on
`5ea82b44`, `d2_jones2d_li_pre_*`).  Its cause (`d2`, `d3`): a TRACED
tensor cell cannot be inspected for out-of-plane components, so
`rcwa_jones_2d` routes it to the general (out-of-plane) cascade
(`lumenairy/elements/rcwa/twod.py:1836`), and that branch builds every
block by the direct rule (`twod.py:2022-2037`, comment "Direct-rule
(laurent) / 'li'") -- it has no Li branch.  So under `jax.jit` or `grad`,
`formulation='li'` silently solves the LAURENT formulation: the jitted
forward equals NumPy Laurent to 5e-15 and differs from NumPy / eager Li by
7.6e-2 (C4v) / 3.5e-4 (C1), and the gradient is the Laurent model's.  The
eager forward is right (1e-14), which is why forward-parity tests pass.

Not measured: `RCWAStack` on JAX input and `rcwa_efficiency_2d_shapes`.

## 6. E: the CHANGELOG entry and the W9 note, read as a physicist

* Traceability: every number in the `### Fixed` entry is in the build
  record and in a probe JSON (checked: 23 / 39 / 28 / 47 %, 28 / 590 %,
  0.24 / 0.44 absolute, 1 - 12 % / 0.5 - 53 % gauge moves, 2.8e-10 /
  1.0e-10 / 1.4e-9 / 3.6e-10 / 1.2e-11 / 3.1e-10 after, 33 / 33, the cost
  ranges, the other-twin numbers).
* The four unfixed families ARE named as still wrong, with numbers and the
  workaround (evaluate a little off the symmetric point).
* **The unrouted RCWA entries are NOT mentioned** in the CHANGELOG
  (`rcwa_jones_2d`, `RCWAStack` on JAX input, `rcwa_efficiency_2d_shapes`);
  only the build record's "Not done" says they "very likely carry the
  defect".  `rcwa_jones_2d` does (section 5).  Finding P2-1.
* "A symmetry-keeping parameter was exact in both, and so was every gradient
  taken a little away from the symmetric point" -- true, but "a little"
  means >= ~1e-8 rad / ~1e-10 of the contrast; at 1e-12 offset the PRE error
  was 1e-3 (the record's own table).  Wording, P3.
* The W9 note in `rcwa/_core.py` is accurate: "exactly 0.0 stays
  unrecoverable BY THIS VJP ALONE", the users of the replacement are named,
  and the four still-wrong families are listed.  It does not list the
  unrouted RCWA entries either (same P2-1).
* Build record section 2.1 (P3-1): "ALL 50 eigenvalues [of `P @ Q`] sit in
  pairs" is the spectrum of the NumPy EVEN-SECTOR fold (`symmetry='auto'`,
  N + 1 = 50), which the probe captured because it did not pass
  `symmetry=False`.  The operator the JAX path hands the rule is 98 x 98, of
  which 50 eigenvalues sit in 25 exact pairs and 48 are simple (`v6`).  The
  fixture is also 7 x 7 orders (`n_orders = 3` means -3..3), not "3 x 3".
  The conclusions do not change.
* Build record section 6 (P3-2): "`jax.hessian` (forward-over-reverse)
  keeps working" holds only for parameters downstream of the eig (its probe
  differentiated the depth); through eps_cell or the angle it raises, on
  both trees (an older CHANGELOG entry already says so for eps).
* Build record section 3 (P3-3): the RCWA "no Gram needed" rationale is
  incomplete (section 1.4).

## 7. Both builds, side by side

Every table above carries both builds.  Where they differ by more than the
FD resolution it is the pre-fix (rule-off) error, which depends on the
basis LAPACK returns inside each pair (RCWA 7 x 7 TE 0.30 on Windows, 0.11
on WSL; corner 0.13 against 0.35; `rcwa_jones_2d` Laurent 6.1e-2 against
4.4e-2), and the cost ratios, which depend on the load.  Rule-on errors,
forward hashes (40 / 40, 20 / 20), the fold failure (5.6e3 / 2.6e3 at 1e-5
rad, two digits alike), the `rcwa_jones_2d` Li defect (0.57 - 0.62, 0.24,
jit-vs-eager 7.6e-2, identical) and the mechanism counts (16 branch
changes, 1.54 d) agree.  The builder's spectrum mislabel reproduces on both
(`v6_builder_spectrum_*`: 98 x 98, 50 paired; the even fold 50 x 50).

Test runs of `tests/unit/test_verify_jax_symmetric_point_gradients.py`:
Windows `-n 2`, BLAS threads 2: **25 passed, 5 xfailed** (139 s); WSL serial,
import path asserted, in one run with the builder's gates below: **39
passed, 5 xfailed** (643 s; 25 + 5 from this file).  Longest verifier test:
31 s on WSL (the two pencil tests, serial on the loaded box), 20 s on
Windows.  The builder's gate tests
(`test_jax_symmetric_point_gradients.py`, `test_niche_audit_w9_eig_vjp.py`,
the two flipped `test_e3r2_*` gates) on this tree: Windows **48 passed**;
WSL (the first file and the two `test_e3r2_*` gates, 14 tests): all passed
in the same run.

## 8. Defects, ranked

**P1**

* **P1-1 (pre-existing, outside the fix's scope, found here):**
  `rcwa_jones_2d(..., formulation='li')` on a traced tensor cell solves the
  Laurent formulation; under `jax.jit` its forward is wrong by up to 7.6e-2
  and its gradient by 21 - 62 % at any cell.
  `lumenairy/elements/rcwa/twod.py:1836` (traced tensor -> general path) and
  `twod.py:2022-2037` (the general path has no Li operators).  Pinned:
  `test_rcwa_jones_2d_li_under_jit_solves_li`,
  `test_rcwa_jones_2d_li_gradient_on_a_cell_without_symmetry` (strict
  xfails).

**P2**

* **P2-1:** the CHANGELOG (and the W9 note) do not say that the other RCWA
  JAX entries are unrouted; `rcwa_jones_2d` returns a symmetry-breaking
  gradient 4.4 - 6.1 % wrong at a C4v cell and a d / d(eps_xy) of 0.083 -
  0.088 where the truth is 0.  Pinned:
  `test_rcwa_jones_2d_symmetry_breaking_gradient_at_the_c4v_cell`,
  `test_rcwa_jones_2d_eps_xy_gradient_at_an_isotropic_c4v_cell` (strict
  xfails); control `test_rcwa_jones_2d_laurent_off_symmetry_is_exact`.
* **P2-2:** no public switch for the rule: a vmapped gradient pays
  3.5 - 3.7x (here) even when the user knows no symmetry is present, and the
  only off-switch is the private `rcwa._core._EIG_CLUSTER_GAP_REL`
  (`_core.py:4694`, read at `_core.py:4880`).  Not pinned as a test: the
  remedy is an API choice (a keyword or a documented context manager), and a
  timing assertion on this loaded box would test noise.

**P3**

* P3-1: the spectrum mislabel and "3 x 3 orders" (section 6).
* P3-2: the `jax.hessian` sentence (section 6); pinned as the scope it is:
  `test_hessian_through_an_eig_parameter_is_refused_on_both_paths`.
* P3-3: the RCWA no-Gram rationale; pinned:
  `test_rcwa_lift_moves_propagating_layer_roots_across_the_cut_harmlessly`.
* P3-4: "a little away from the symmetric point" (section 6).

No defect was found in the fix itself: no wrong number, no new breakage, no
changed forward value.

## 9. Tests

`tests/unit/test_verify_jax_symmetric_point_gradients.py`: 25 passing pins
(claims 1 - 3 and 5 on the verifier fixtures, each with its measured values
and a bar with decades on both sides; the rule-off arms make the RCWA and
1-D fixtures fail-before demonstrations) and 5 strict xfails (the
`pmm_jones_1d` and `rcwa_jones_2d` defects).  The xfails were run with
`--runxfail` and fail at their intended assertion (0.061, 0.97, 0.088,
0.237, 0.0755).  `.test_durations` carries the 30 new ids (measured
here with `-n 2` on the loaded box, so they overstate an idle serial run).

## 10. Not done

* `RCWAStack` on JAX input and `rcwa_efficiency_2d_shapes` were not measured
  at a symmetric point.
* Berreman, the 1-D `PMMStack` twins and the hybrid 2-D stack were not
  re-measured (one defective and one correct family were, per the brief).
* Timing on an idle box; GPU.
* The cause of the larger first-call cost of an EAGER gradient at a cluster
  (16 - 18 s against 3.4 - 3.8 s) was not traced; it is a one-time cost per
  process and shape.

# JAX gradients at symmetric points: root cause and fix (2026-10-03)

Builder: Claude Opus 5.5 (model ID `claude-opus-5-5`).  Worktree
`C:/tmp/lum_symgrad`, branch `fix/jax-symmetric-point-gradients` from the
integration commit `5ea82b44`.  Object: the two maintainer items of
`docs/audits/BUILD_PMM2D_CURVED_E3_2026_10_03.md` section 9.9 (pinned there
as strict xfails), and the W9 note of `rcwa/_core.py` ("exactly 0.0 stays
unrecoverable").

Builds: Windows 11 (CPython 3.14.6, numpy 2.4.4, jax 0.11.0) and WSL Ubuntu
(CPython 3.12.3, numpy 2.4.6, jax 0.10.2).  BLAS threads 1 on every probe
command line; every probe asserts `lumenairy.__file__` lies under the tree it
was pointed at.  PRE tree = `git archive 5ea82b44`; POST = this branch.
Evidence: `validation/probe_jax_symgrad/` (`<probe>_<pre|post>_<win|wsl>.json`;
runners `run_win.sh`, `run_wsl.sh`).

Words.  AD = the reverse-mode gradient (`jax.jit(jax.jacrev(f))`).  FD = the
NumPy solve's central difference, Richardson-extrapolated from h = 3e-4 and
1e-4 of a three-rung ladder (1e-3, 3e-4, 1e-4); its h^2 PREMISE is the ratio
of successive rung changes, 11.375 for a clean h^2 law (measured 11.2 - 11.6
on every entry of every probe below, so the FD is used as the oracle
throughout).  Error = max |AD - FD| / max |FD| over the outputs.  A CLUSTER
is a set of eigenvalues closer than a given fraction of the spectrum's
largest magnitude `max|lam|`.

## 1. Fixtures

* RCWA: `rcwa_efficiency_2d` with a JAX `eps_cell`, P = 1.2, wl = 1,
  15 x 15 pixels (centre block eps 4, side blocks 1.5, corners 1: four-fold
  symmetric), depth 0.45, n_sub 1.45, n_sup 1 (the positional order of
  the call; round 1 printed them swapped), `n_orders_x = n_orders_y = 3`,
  i.e. 7 x 7 = 49 orders [corrected in round 2, verifier P3-1]; parameter t
  added to the two x-side blocks (keeps both mirrors, breaks the 90-degree
  rotation); outputs R, T of (0,0), (+-1,0).
* 1-D PMM: `pmm_efficiency_1d` (JAX), P = 1.2, ridge n 2 / groove 1, duty
  0.5, depth 0.45, n_sup 1.45, n_sub 1, wl 1, degree 12, `stabilize=False`;
  parameter the angle; outputs R, T of the +-1 orders.

## 2. Root cause, discriminated

### 2.1 RCWA

**Spectrum** (`a1_rcwa`, section 1).  The only eig of this path is the
layer operator `P @ Q` (the half-spaces are analytic,
`_homogeneous_eigenmodes`).  At the symmetric cell the operator the
NumPy probe captured (its 50 x 50 even-sector fold) had ALL 50 eigenvalues
in pairs closer than 1e-12 max|lam| (min gap 4.2e-17 / 3.8e-17 win / wsl;
max|lam| = 11.16).  [Corrected in round 2, verifier P3-1: the operator the
JAX path hands the rule is the full 98 x 98 `P @ Q`, of which 50
eigenvalues sit in exact pairs and 48 are simple.]  The parameter enters only through `eps_cell`
-> the convolution matrices -> `P @ Q`.  The pairs split linearly in the
offset: min gap 2.1e-13 / 2.1e-11 / 2.1e-9 / 2.1e-7 / 2.1e-5 at offsets
1e-10 / 1e-8 / 1e-6 / 1e-4 / 1e-2.

**Offset sweep** (the cell moved off symmetry by delta in the same
direction, AD vs FD at t = delta), PRE tree:

| delta | 0 | 1e-12 | 1e-10 | 1e-8 | 1e-6 | 1e-4 | 1e-2 |
|---|---|---|---|---|---|---|---|
| TE win / wsl | 0.23 / 0.28 | 5.8e-4 / 5.2e-4 | 1.3e-8 / 2.6e-8 | 3.8e-10 / 3.6e-10 | 1.4e-10 / 7.4e-10 | 2.7e-11 / 4.5e-11 | 4.6e-10 / 2.6e-11 |
| TM win / wsl | 0.39 / 0.47 | 1.0e-3 / 8.8e-4 | 2.2e-8 / 4.3e-8 | 8.1e-10 / 4.9e-10 | 5.3e-10 / 4.5e-10 | 1.2e-10 / 7.9e-11 | 1.2e-10 / 9.9e-10 |

The error vanishes CONTINUOUSLY with the splitting, along the W9 envelope of
`_jax_eig_stable` (exact once the split exceeds ~1e-10 of the spectrum,
degraded below, floored at tau_rel = 1e-12) -- the degenerate-cluster
signature, not a separate defect.  The same holds for a lower-symmetry
offset (one side block, the y-mirror kept: 3.9e-8 / 1.4e-8 at 1e-10, <= 8e-10
above) and for a fixed random pixel pattern with no symmetry at all
(7.9e-2 / 1.2e-4 / 1.2e-8 / 2.7e-10 at 1e-10 / 1e-8 / 1e-6 / 1e-4, both
builds alike).

**Gauge test** (section 3 of `a1_rcwa`).  The eig's basis inside every exact
cluster replaced by a seeded random unitary rotation of LAPACK's (two
seeds): forward change <= 5.1e-15; the gradient moved by 4.8e-2 / 9.1e-3
(TE) and 0.12 / 3.2e-2 (TM) on Windows, 3.8e-2 / 2.8e-2 and 9.4e-2 / 5.8e-2
on WSL.  The gradient was a function of the basis LAPACK picked -- which is
why it was build-dependent.

**Other contributors checked.**  No eigenvalue sorting or order matching in
the path (the eigenpairs feed the S-matrix directly).  `_sqrt_decay`'s
branch point lam^2 = 0 is 2.8e-3 max|lam| from the nearest eigenvalue (not
near the pairs).  `jnp.where` host branches: the uniform-layer select
(`uniform` is False here and depends on `EPS`, not on the eig) and the
grazing guards (concrete geometry, unchanged).  None depends on the
parameter at the symmetric point.

**Root cause, plain language.**  The four-fold cell makes every mode of the
layer come in coincident pairs.  Breaking the symmetry splits each pair, and
how a pair's two modes rotate into each other as it splits is information
the reverse pass of an eigen-solve does not receive -- the S-matrix that
consumes the modes does not care which basis of a pair it gets, so the
cotangent it hands back carries nothing about that rotation.  The gradient
then contained a term set by LAPACK's arbitrary choice of basis inside each
pair (23 - 47 %, different per build).  Discriminating numbers: rotating the
basis moved the gradient 1 - 12 % with the forward fixed to 5e-15; moving
the cell 1e-10 off symmetry brought the error to 1e-8, 1e-8 off to 4e-10.

### 2.2 1-D PMM at normal incidence

**Spectrum** (`a2_pmm1d`, section 1).  Three eigs: the layer, the
superstrate and the substrate fold `eig(B^-1 A)` (24 x 24).  At angle 0
(traced): the layer has NO cluster (min gap 1.6e-4 TE / 8.7e-4 TM of
max|lam| ~ 700); each half-space has 8 eigenvalues inside 1e-6 of the
spectrum, 4 of them exact pairs (gap <= 1e-12; min 9.8e-18 .. 5.5e-17) --
the `+-m` orders of a uniform medium.  Off normal the pairs split as
4.8e-3 x angle (4.8e-10 / 4.8e-8 / 4.8e-6 at 1e-7 / 1e-5 / 1e-3 rad).  The
angle enters through the traced `kx0` (convection `-1j kx0 (C - C^T)` and
`kx0^2` mass in all three operators, the per-order `kx`, `kz_inc`).

**Angle sweep**, PRE tree (identical on both builds to two digits):

| angle | 0 | 1e-9 | 1e-7 | 1e-5 | 1e-3 |
|---|---|---|---|---|---|
| TE rel (abs) | 0.28 (0.24) | 7.3e-7 | 1.3e-10 / 1.5e-10 | 1.0e-10 | 4.6e-11 |
| TM rel (abs) | 5.9 (0.44) | 1.6e-6 | 5.6e-10 / 5.0e-10 | 3.7e-10 / 1.8e-10 | 2.0e-10 / 1.9e-10 |

Again continuous vanishing with the splitting (the W9 table).

**Gauge test, per eig site** (rotation inside exact clusters of ONE site):
layer 1.4e-9 .. 1.0e-8 (no clusters: round-off), superstrate 1.8e-2 /
3.2e-2 (TE) 8.6e-2 / 0.11 (TM), substrate 5.4e-3 / 6.8e-3 (TE) 0.50 / 0.53
(TM); forward <= 6.7e-14.  Both half-space eigs carry the defect, the layer
none.

**Other candidates.**
* `abs` / `sign` of sin(angle): none on the path (`kx0 = Re(n_sup) sin(angle)
  k0`, smooth).
* The kz square root at zero transverse wavenumber: `_kzf(eps, kx)` at
  kx = 0 is `sqrt(eps)`, smooth with zero slope; the modal `sqrt(q2)`'s
  branch point is 4.4e-4 max|lam| from the nearest half-space eigenvalue.
* `jnp.where(theta == 0)` host branches: none for a traced angle -- a traced
  angle (even valued 0) always takes the oblique branch (`kx0` traced); the
  python-literal `kx0 = 0.0` branch is for a concrete angle only.
* The Wood-anomaly guard: the JAX twin has none; the NumPy oracle's
  `_wood_safe_wl_1d` is the identity at every FD abscissa (0, +-1e-4,
  +-3e-4, +-1e-3: shift 0.0, no warning) -- no wavelength jump in the FD.
* `stabilize`: the JAX path refuses `stabilize=True` (host degree scan), so
  the twin always runs `stabilize=False`; the test's NumPy oracle uses the
  same.  (NumPy `stabilize=True` picks another degree for TM here: a
  constant 4.2e-6 offset at 0 and +-1e-4, irrelevant to a derivative.)
* The forward-branch selector `_forward_branch_flip`: a band test on
  Im(q), locally constant away from the band; no half-space q sits in it.
* The TE / TM asymmetry (28 % vs 590 %): the ABSOLUTE errors are 0.24 (TE)
  and 0.44 (TM); the TM gradient is 12x smaller (max |FD| 0.074 vs 0.86), so
  the relative figure is large.  The cluster class predicts an error of the
  size of the missing in-cluster term, unrelated to the size of the true
  gradient; the gauge test shows the TM substrate's in-cluster sensitivity
  is the largest (0.5).
* FD-free confirmation: the grating is mirror-symmetric, so at exactly
  normal incidence d X_{+1} / d angle = - d X_{-1} / d angle.  PRE AD broke
  it by 6.7e-3 (TE) / 3.3e-2 (TM) absolute.

**Root cause, plain language.**  At exactly normal incidence the
superstrate's and the substrate's `+m` and `-m` orders are coincident modes,
and tilting the beam splits them.  As in the RCWA case, the reverse pass of
those two eigen-solves cannot see how each pair rotates as it splits, so the
angle derivative carried a term set by LAPACK's choice of basis inside each
pair (both builds happen to pick the same basis here, hence the same 28 % /
590 %).  It is the whole story: the layer has no cluster and is
basis-insensitive (1e-9); rotating either half-space's pairs moves the
gradient 0.5 - 53 % with the forward fixed to 7e-14; at 1e-7 rad, where the
pairs are 4.8e-10 of the spectrum apart, the error is 1e-10; no other
non-smooth element is on the path.

## 3. The fix (one mechanism)

Both paths now wrap their eig(s) AND everything downstream of them in
`rcwa._core._jax_eig_cluster_adjoint` -- the rule of round 2 (section 9.1 of
the E3 record), unchanged.  No second copy of the rule.

* `rcwa/twod.py`, `rcwa_efficiency_2d` (JAX branch of the full solve): the
  eig problem is the layer operator `P @ Q` built OUTSIDE the eig
  (`_scalar_PQ` for Laurent, `_tensor_PQ` for Li); the consumer is
  `_cascade(eig_pair)` -- `_layer_eigenmodes(..., eig_pair=)` /
  `_layer_eigenmodes_tensor(..., eig_pair=)` and the S-matrix to `(r, t)`;
  anchor lam^2 = 0 (`_sqrt_decay`).  NumPy / CuPy run `_cascade()`
  unchanged.  `fff_nv` has no JAX path (unchanged).
* `rcwa/_core.py`: `_layer_eigenmodes` and `_layer_eigenmodes_tensor` take
  `eig_pair=` (the eigenpairs already taken by the caller; every other line
  the same).  The P block now has ONE definition, `_layer_P_matrix`, used by
  both eigensolvers, `_scalar_PQ` and `_tensor_PQ` (it was written out four
  times); `_tensor_PQ` accepts a precomputed `Ez_inv`.
* `pmm/_core.py`: `_jpmm_sem_modes` is split into `_jpmm_sem_problem`
  (returns the pencil `(A, B)` and TM `invop`) and
  `_jpmm_sem_modes_from_eig` (sqrt, branch selector); `_jpmm_solve` builds
  the three pencils (layer, superstrate, substrate) first and passes
  `_amplitudes` (modes -> S-matrix -> incident projection -> `r_ord,
  t_ord`) as the consumer, `eig_fn = eig(solve(B, A))` (the same fold as
  before), anchor q^2 = 0.

**Why the PENCIL, not the fold** (`c1_gram`).  The first version handed the
rule the folded `B^-1 A` with no Gram.  It was exact at 0 but, at 1e-5 rad
off normal, where the half-space pairs are NEAR-degenerate (inside gap_rel =
1e-6, split comparable to the lift d = 1e-7 max|lam|), it was wrong by
orders of magnitude.  The fold's eigenvectors are B-orthogonal, not
Euclidean-orthogonal, so the Euclidean-Gram lift of a cluster that merges
distinct eigenvalues is complex and the forward-branch selector reads it as
a branch change (round 2's development record, item 4).  With the pencil,
the mass matrix B is the lift's Gram and the shifts stay real:

| angle | 0 | 1e-7 | 1e-6 | 1e-5 | 1e-4 |
|---|---|---|---|---|---|
| fold TE win / wsl | 1.1e-11 / 7.2e-12 | 6.0e-11 / 7.3e-11 | 4.3e-11 / 4.6e-11 | **4.7e3 / 4.7e3** | **1.6e3 / 1.6e3** |
| pencil TE win / wsl | 1.2e-11 / 5.7e-12 | 5.6e-11 / 6.6e-11 | 4.1e-11 / 4.7e-11 | 1.0e-10 / 1.0e-10 | 1.1e-10 / 1.2e-10 |
| fold TM win / wsl | 2.6e-10 / 2.6e-10 | 6.5e-10 / 5.2e-10 | 3.8e-10 / 4.0e-10 | **7.0e4 / 7.0e4** | **1.8e4 / 1.8e4** |
| pencil TM win / wsl | 3.1e-10 / 2.3e-10 | 6.1e-10 / 4.9e-10 | 3.5e-10 / 3.8e-10 | 2.4e-10 / 1.5e-10 | 9.9e-10 / 1.1e-9 |

The RCWA operator has no such Gram; its near-degenerate cases were measured
directly (2.1: one-block and random offsets 1e-10 .. 1e-4, all <= 1.6e-9
after the fix) and need none.  [Corrected in round 2, verifier P3-3: the
reason is not that the lift stays real -- on a weak no-symmetry conical
grating the Euclidean lift gives complex shifts up to 1.54 d and moves 16
propagating layer roots across the `sqrt` cut over the four stencil points
-- but that a LAYER mode's branch is a forward / backward re-labelling the
S-matrix is invariant under (`lam -> -lam`).  A family whose consumer
holds HALF-SPACE modes from an eig needs the mass-matrix Gram; that is why
the PMM Jones and stack twins pass `S0` as the lift Gram in round 2.]

**Forward bytes** (`b1_fwd_bytes`, `b1_compare.py`): SHA-256 of
(orders,) R, T for 11 fixtures -- RCWA symmetric TE / TM / Li, a C2v
rectangle at normal, symmetric TM at theta 0.2 (oblique), the rectangle at
theta 0.2 / phi 0.3 (conical), a uniform cell; 1-D PMM TE / TM at 0 and 0.2
rad (traced angle) -- each NumPy eager, JAX eager and under `jax.jit`:
**33 / 33 equal PRE vs POST on Windows, 33 / 33 on WSL.**

## 4. After the fix

| | PRE win / wsl | POST win / wsl |
|---|---|---|
| RCWA, symmetric cell, TE | 0.23 / 0.28 | 2.8e-10 / 1.0e-10 |
| RCWA, symmetric cell, TM | 0.39 / 0.47 | 1.4e-9 / 3.6e-10 |
| RCWA, Li formulation TE / TM (win) | -- | 6.1e-10 / 2.6e-10 |
| RCWA offset sweep 1e-12 .. 1e-2, both pols | up to 1.0e-3 | <= 1.6e-9 / 9.1e-10 |
| RCWA random-pattern offsets 1e-10 .. 1e-4 | up to 0.13 | <= 1.6e-9 / 9.0e-10 |
| RCWA gauge (gradient change under in-cluster rotation) | 9.1e-3 .. 0.12 / 2.8e-2 .. 9.4e-2 | <= 3.7e-10 / 4.0e-10 |
| 1-D PMM at 0, TE | 0.28 / 0.28 | 1.2e-11 / 5.7e-12 |
| 1-D PMM at 0, TM | 5.9 / 5.9 | 3.1e-10 / 2.3e-10 |
| 1-D PMM angles 1e-9 .. 1e-3 | up to 1.6e-6 | <= 2.6e-9 / 2.6e-9 |
| 1-D PMM mirror defect at 0 (abs) | 6.7e-3 / 3.3e-2 | <= 9.2e-12 |
| 1-D PMM gauge, any site | up to 0.53 | <= 6.2e-11 / 7.6e-11 |

The W9 note's "exactly 0.0 stays unrecoverable" is now scoped to the eig
VJP alone; the note says what replaced it (the cluster rule, its users).

## 5. The other JAX twins (D)

One symmetric-point, symmetry-breaking gradient per family, AD vs a
premise-checked FD (probes `d*_<family>.py`, run by measurement subagents on
both builds; POST tree, but none of these twins is changed by this branch).

Error = AD vs FD at the symmetric point (relative); "off" = the same at a
small offset off it (1e-5 rad / 1e-4 .. 1e-5 relative); premise ratios
11.1 - 11.7 on every row with a nonzero FD.  "Gauge" = relative gradient
change when the eig's basis inside each exact cluster is rotated.

| family (entry, eig site) | symmetric point and parameter | exact cluster? | error win / wsl | off | verdict |
|---|---|---|---|---|---|
| BOR-PMM (`BORStack.solve`, `_jax_bor._jbor_layer_modes` l.82) | radially homogeneous layer, d / d(ring index); m = 0, 1 | none (min gap 1.2e-4 / 5.6e-4; TE / TM decouple at fixed m) | 3.2e-10 / 3.3e-10 (m 0), 1.7e-10 / 3.6e-10 (m 1) | -- | correct |
| BOR-SEM (`BORStack(basis='sem')`, `_jax_sem._jsem_layer_modes` l.223) | isotropic layer, d / d(in-plane anisotropy); m = 0, 1 | none (min gap 4.4e-4 / 1.6e-4) | 3.1e-10 / 3.1e-10 (m 0), 1.2e-8 / 1.2e-8 (m 1) | -- | correct |
| EME modes (`ref_2d_modes[_vector]`, `_jax_modes` l.218 / l.256) | uniform cell, one pixel's eps | yes (34 / 36 scalar, 64 / 64 vector) | cluster-symmetric outputs 1.5e-9 / 9.6e-10 (scalar), 2.1e-10 / 1.7e-10 (vector) | members -> FD floor | NOT this class: the consumer returns sorted eigenvalues only; an individual member of a splitting cluster is not differentiable there (one-sided slopes differ: eig of the in-cluster block), so no gradient is "right" for it; cluster sums are exact |
| Berreman 4x4 (`berreman_jones_1d`, traced tensor -> `_offplane_solve_jax`, `_layer_M_gen_jax` l.399) | isotropic layer at normal incidence, d / d(eps_xy) | yes (two pairs) | **0.99 / 0.99** | 6.7e-11 (1e-5) | DEFECT; d / d(eps_xx - eps_yy) right (1.3e-11) only because LAPACK returns the x / y basis (gauge 0.33 - 0.75: latent); native route (`_solve_jax`) not reachable by a splitting parameter at first order |
| PMM Jones 1-D (`pmm_jones_1d`, `_jpmm_jones_solve`; half-space `_juniform_geo_eig` l.4378) | symmetric grating, d / d(angle) at 0 | yes (half-space Kx^2 pairs) | **2.7 / 2.7** | 3.0e-10 | DEFECT (mirror-identity defect 0.14 relative; gauge 0.24 - 0.59) |
| PMM 1-D stack (`PMMStack.solve`, `_jax_stack` l.338 / l.582, layer `Mbig`) | symmetric grating(s), d / d(angle) at 0 | yes (half-space Kx^2; a uniform spacer's Mbig 48 / 48) | **1.5** (1 layer), **5.2** (2 layers), **1.05** (with spacer), **6.6** (per-layer grids) | 1.2e-10 .. 3.0e-8 | DEFECT |
| hybrid 2-D stack (`PMM2DStackHybrid.solve`, `_jax_stack2d._modes_projected` l.224) | C4v cell (Li / Laurent), d / d(theta) at 0; a 1-D cell (phi 0, conical) | Laurent: yes (50 / 50); Li: none (gap 2.8e-6) | 3e-11 .. 7e-10 | -- | correct for these (the theta coupling vanishes inside the E pairs at first order; the conical clusters are not excited) |
| hybrid 2-D stack, traced region layout | C4v 9-region cell, d / d(corner-region eps), Laurent | yes (50 / 50) | **0.295 / 0.347** | 1.2e-10 (1e-4) | DEFECT (build-dependent; gauge 0.14 - 0.25); the same cell under Li 1.4e-10 (no exact cluster) |

The hybrid 2-D PMM CELL twin that round 2 measured correct (7e-10 at its
symmetric point) is a different entry (`pmm_efficiency_2d_cell`); the stack
twin's traced-layout case above is new.

**Fixes not made here.**  None of the four defective families is a one-line
routing: each needs its eig problems built outside the eig and the rest of
its solve as the consumer, the restructure `_jpmm_solve` received.
Separability was checked for each: Berreman `_layer_M_gen_jax` (problem
`_delta_jax(...)`, standard eig, G = None; consumer the forward/backward
split, `_modes_to_M` and the cascade; the `carries` / `gre` thresholds are
anchor candidates); PMM Jones 1-D and both 1-D stack twins (the geometric
`Kx^2` is a fold `S0^-1 op` with `S0` the SPD mass matrix and `op` Hermitian
for real `kx0` -> hand the PENCIL `(op / k0^2, S0)`; the layer `Mbig` is a
standard eig of products with `inv(S0)` and needs the rule whenever a
layer is uniform; one restructure of `_jpmm_jones_solve` and its stack
analogues serves all three); hybrid 2-D stack (`P @ Q` per patterned layer,
G = None, anchor lam^2 = 0, as the RCWA fix).  BOR / BOR-SEM are cleanly
separable (`eig(solve(Be, Ke))`, the pencil form) should a degenerate
configuration ever be found; none exists at fixed m.  Listed as follow-ups
(section 8).

Method note (from the measurement of these families): rotating `V` AFTER
the plain custom-VJP eig re-expresses the cotangent but leaves the eig's
backward rule in LAPACK's basis, so it can under-report the gauge
dependence (Berreman d / d(eps_xy): a 1e-15 change on a 99 % wrong
gradient).  The faithful rotation is a custom VJP whose backward rule runs
at the rotated basis (`validation/probe_jax_symgrad/_dcommon.py`,
`rotated`).  For the RCWA and 1-D PMM paths the simpler rotation of
section 2 did expose the dependence (1 - 53 %), and the unit gauge test
pins both of its arms.

## 6. Cost (E)

**Caveat first: the box was saturated** for the whole build by another
worktree's full unit suite (`-n 4`, every core at 100 %; a test of
unchanged code took 84x its recorded duration), so every time below is an
upper bound with large scatter, and only the Windows repeats are
comparable.  `e1_timing` (jitted, best of 7 after the compile), Windows,
PRE (two runs) vs POST (three runs), ranges over the runs:

| case | cluster? | gradient PRE | gradient POST | ratio | grad compile PRE / POST | forward PRE / POST |
|---|---|---|---|---|---|---|
| RCWA symmetric cell (`rcwa_sym`) | yes (all 50) | 0.025 - 0.035 s | 0.19 - 0.32 s | 5 - 13x | 5.7 - 8.0 / 9.4 - 16.5 s | 0.017 - 0.019 / 0.016 - 0.018 s |
| RCWA C2v rectangle, theta 0.2 (`rcwa_rect`) | no | 0.024 - 0.026 s | 0.028 - 0.077 s | 1.1 - 3x (scatter) | 2.5 - 2.9 / 7.1 - 11.4 s | 0.019 - 0.020 / 0.018 s |
| 1-D PMM at 0 (`pmm1d_0`) | yes (half-spaces) | 0.0030 - 0.0031 s | 0.012 - 0.023 s | 4 - 7x | 4.2 - 4.3 / 12.7 - 17.8 s | 0.0018 / 0.0017 - 0.0019 s |
| 1-D PMM at 0.2 rad (`pmm1d_02`) | no | 0.0031 - 0.0035 s | 0.0038 - 0.0040 s | 1.1 - 1.3x | 3.0 - 3.5 / 14.5 - 18.2 s | 0.0018 / 0.0018 s |
| `rcwa_rect` under `jax.vmap` (4 points) | no | 0.055 - 0.13 s | 0.88 - 1.10 s | 7 - 16x | 4.0 - 4.5 / 12.9 - 15.3 s | -- |

(WSL, one run each, PRE run under heavier load than POST: `rcwa_sym`
0.22 -> 0.18 s, `pmm1d_0` 0.072 -> 0.012 s, `rcwa_rect_vmap4` 0.30 ->
1.72 s; grad compile 10 - 15 s both -- the WSL PRE run is not a usable
baseline, recorded for completeness in `e1_timing_{pre,post}_wsl.json`.)

Reading:
* **With a cluster** the reverse pass evaluates eig + solve + its VJP at
  four lifted points (the order-4 stencil): 4 - 13x the gradient time --
  the round-2 rule's cost (3 - 5x on the staggered twin), somewhat higher
  here because the whole RCWA / 1-D solve is the consumer.
* **Without a cluster** under `jax.jit` the plain branch runs: the
  runtime ratio is 1.1 - 1.3x on the 1-D twin (the forward pass of the
  gradient now also computes the cluster test and the lift operand, and
  keeps the residuals the other branch would need); the RCWA rectangle's
  1.1 - 3x is within this box's scatter (same code path).
* **Compile time** of a jitted gradient rises 2.5 - 5x (both branches of
  the `lax.cond` are compiled).  The forward pass and its compile are
  unchanged (the rule acts in the reverse pass only).
* **Under `jax.vmap`** the batched cluster flag turns the `lax.cond` into
  a select and BOTH branches run, so a vmapped gradient pays the lifted
  cost even without a cluster (7 - 16x here) -- the same property round 2
  recorded for the staggered twin.
* **Eager** (un-jitted) calls: when no eig argument is traced (an eager
  forward, or a gradient in a variable that enters only downstream of the
  eig, e.g. a thickness) the rule now short-circuits to the plain
  composition (`_jax_eig_cluster_adjoint`, new in this build); warm eager
  times on the 9 x 9-order fixture of `test_v5_10_3_rcwa_2d_autodiff`
  (`e2_eager`): forward 0.08 - 0.14 s both trees, gradient in thickness
  0.23 - 0.62 s POST vs 0.28 - 0.29 s PRE (scatter).  `jax.hessian` and
  `jax.vmap(jax.grad)` of `rcwa_efficiency_2d` still run and return the
  PRE values (40365737700999.75 / -1865837.976..., both builds).
* Forward-mode (`jax.jvp`) through the 2-D RCWA / 1-D PMM JAX paths was
  already refused before (the eig is a custom VJP).  [Corrected in round 2,
  verifier P3-2: `jax.hessian` works only for a parameter DOWNSTREAM of the
  eig (the thickness above); for one that enters the eig (`eps_cell`, the
  angle) it raises `NotImplementedError`, on this tree and on 5ea82b44.]

## 7. Tests

* `tests/unit/test_pmm2d_staggered_curved_e3.py`: the two strict xfails
  flipped into `test_e3r2_rcwa_jax_symmetry_breaking_gradient_at_a_symmetric_cell[te|tm]`
  and `test_e3r2_pmm1d_jax_angle_gradient_at_normal_incidence[te|tm]`; bar
  1e-7 (envelope <= 1.4e-9, defect >= 0.23), FD premise asserted first.
* `tests/unit/test_jax_symmetric_point_gradients.py` (new): the RCWA gauge
  test (rule on <= 1e-8, rule off > 1e-4: the root cause as a property); the
  RCWA no-symmetry near-degenerate offset; the 1-D mirror identity at 0
  (FD-free); the 1-D twin at 1e-9 and 1e-5 rad (the second pins the
  pencil).

* `tests/unit/test_niche_audit_w9_eig_vjp.py`: the theta = 0 pin
  `..._is_an_OPEN_defect[te|tm]` (whose message asked to be re-pinned when
  the defect closed) is now `test_pmm1d_angle_gradient_at_exactly_zero_is_exact`
  (|AD(0) - FD(0)| / resolution = 0.99 / 1.00, bar 1e3, same constant),
  with a fail-before arm `test_the_theta0_pin_fires_with_the_cluster_rule_off`
  (rule off: 3.7e8 / 2.9e9) and the `rcwa1d` control kept as
  `test_the_theta0_pin_passes_on_the_analytic_half_space_solver`; the
  floor-mechanism test runs with the rule off (it is about
  `_jax_eig_stable` alone).  The lstsq-projection test captures eager
  intermediates; it passes because the rule now steps aside when no eig
  argument is traced.
* `.test_durations`: the renamed / parametrized entries (measured where
  run, inherited from the predecessor entry otherwise).

Results (logs in the session scratchpad, summarised here):
* Windows, `-n 2`, 36 related files (the five gate files, both new / changed
  test files, the RCWA / PMM JAX, autodiff, census, re-export and
  public-API files), on the tree before the last two edits (the W9 re-pin
  and the traced-argument short-circuit): `4 failed, 1088 passed` -- the 4
  are the W9 pins above.  After both edits: `test_niche_audit_w9_eig_vjp.py`
  `34 passed`; the gate set (new file, E3 file, `test_v5_20_1_rcwa_2d_oop_jax`,
  `test_v5_20_3_rcwa_1d_oop_jax`, `test_audit_w3_pmm_jax_guards`,
  `test_v5_14_2_jax_stacks`, W9, fingerprint tool), BLAS threads 2:
  `135 passed`; doc-identifier / changelog-walker / history / census /
  re-export / public-API files: `79 passed, 7 skipped` (+ the fingerprint
  test, green after the re-record).
* WSL, serial (import path asserted): the gate set plus
  `test_v5_10_3_rcwa_2d_autodiff`, `test_v5_12_0_pmm_autodiff` and W9:
  `3 failed, 237 passed` (the 3 W9 pins, run before the re-pin), then W9
  after the re-pin: `34 passed`.  No native crash.
* ruff (Windows `python -m ruff`, WSL `~/lumvenv/bin/ruff`) on
  `lumenairy tests scripts`: clean; `python -m mypy`: no issues in 33
  files; `record_history_fingerprints.py --check`: OK (`pmm/_core`
  re-recorded in the same commit).

## 8. Not done (round 1; closed by round 2 except where section 10.8 says)

* The four defective families of section 5 (Berreman with a traced
  tensor, `pmm_jones_1d`, the 1-D `PMMStack` twins, the hybrid 2-D stack
  with a traced region layout) are NOT fixed: none is a one-line routing
  (each needs the `_jpmm_solve`-style split of problems and consumer;
  section 5 names the functions).  Recommended as the next item, with the
  probes `d4` - `d7` as their fail-before.
* The other RCWA JAX entries (`rcwa_jones_2d`, `RCWAStack` on JAX input,
  `rcwa_efficiency_2d_shapes`, the 1-D RCWA) are not routed and were not
  measured at a symmetric point; the 1-D RCWA has analytic half-spaces and
  no symmetry-forced layer degeneracy (its theta = 0 gradient is
  machine-zero, `test_niche_audit_w9_eig_vjp`), the 2-D ones share the
  layer eig of section 2.1 and very likely carry the defect at a
  four-fold cell.
* Timing on an idle box (the box was saturated, section 6); GPU.
* `rcwa_efficiency_2d` with `formulation='li'` was measured on Windows
  only (6.1e-10 / 2.6e-10 after the fix); the 'li' operator of the hybrid
  2-D stack does not keep the C4v degeneracy (gap 2.8e-6, observed by the
  section-5 probe) -- not investigated.

## 9. Reproduction

```
cd /c/tmp/lum_symgrad/validation/probe_jax_symgrad
git -C /c/tmp/lum_symgrad archive 5ea82b44 lumenairy | tar -x -C PRE_DIR
bash run_win.sh PRE_DIR pre a1_rcwa.py a2_pmm1d.py b1_fwd_bytes.py e1_timing.py
bash run_win.sh /c/tmp/lum_symgrad post a1_rcwa.py a2_pmm1d.py b1_fwd_bytes.py c1_gram.py e1_timing.py
wsl -e bash .../run_wsl.sh PRE_DIR_WSL pre ...  ;  ... post ...
python b1_compare.py
```

## 10. Round 2 (2026-10-04): every JAX twin, one switch, and the verifier's findings

Builder: Claude Opus 5.5 (`claude-opus-5-5`).  Brief: the maintainer's
decision of 2026-10-04 (fix every remaining symmetric-point defect of
section 5 with the one mechanism; route the other RCWA JAX entries; a
library-wide switch; re-measure the family table) and the independent
verification of round 1 (`VERIFY_JAX_SYMMETRIC_POINT_GRADIENTS_2026_10_04.md`,
merged from the integration tip `7c0bc8bd` before this round's commits:
five strict xfails to flip, P1-1, P2-1/2, P3-1..4).  Builds as in the
header.  Probes: `validation/probe_jax_symgrad/` (`d*`, `g*`, `h*`, `k1`,
`f1`, `e*`; tag `r2post` = this round, `pre` = `git archive 5ea82b44`; the
`d*_post_*` files of round 1 are the BEFORE readings of the families that
round 1 did not change).  Two measurement subagents produced `g1`/`g2` and
`h*` and the `d*` re-runs; every number below is from those JSON files.

### 10.1 One mechanism, applied mechanically: `_jax_cluster_routed`

Round 1 restructured two solvers by hand into "eig problems" and a
"consumer".  The remaining families (seven entries, several eig sites each,
stacks with shared and per-layer eigs, out-of-plane generator paths) would
have needed the same surgery seven times.  Round 2 adds ONE helper in
`rcwa/_core.py` and routes every solver through it:

* `_jax_twin_eig(L, G=None, K=None)` -- THE eig of every routed twin:
  `(lam, V)` of `G^-1 L` through `_jax_eig_stable` (exactly the expression
  each twin evaluated before, so values do not move), with an optional
  third argument `K`, the cluster lift's inner product (not a value
  input).  `_eig_for(jnp)` returns it, so every RCWA eigensolver takes it
  without change.
* `_jax_cluster_routed(solve)` -- runs a twin's solve once RECORDING its
  `_jax_twin_eig` arguments (the plain forward; returned as is when the
  rule does not apply: an eager forward, the switch off, or no eig argument
  traced); otherwise re-records them inside a trace and hands them to
  `_jax_eig_cluster_adjoint` with the consumer "re-run `solve` REPLAYING
  the eigenpairs in call order".  Record and replay both run inside a
  trace because a solve's control flow may test concreteness: the Berreman
  twin takes an analytic, eig-free branch for a concretely isotropic layer,
  which inside the replay's trace is not concrete -- the first version
  recorded eagerly and ran off the end of its list in
  `test_v5_14_5_emt_and_berreman_jax.py::test_jax_grads_match_fd`.
* `_jax_eig_cluster_adjoint` takes `(L, G, K)` problems (K defaults to G),
  builds the LIFT only on the lifted branch of the reverse pass (the
  forward of the gradient now computes only the O(n^2) cluster test), and
  decides that branch on the OR over a `jax.vmap` batch
  (`_batch_any`, a `jax.custom_batching.custom_vmap` that returns an
  UNBATCHED flag), so a vmapped gradient no longer runs both branches.

Routed (the solve wrapped; the eig sites unchanged except for the `K`):

| family | function wrapped | eig problems | lift Gram K |
|---|---|---|---|
| `rcwa_efficiency_2d` | the `_cascade` closure (round 1's explicit split removed) | layer `P @ Q` | identity |
| `rcwa_jones_2d` | `_layer_rt` (in-plane AND out-of-plane paths) | `P @ Q` / the 4N generator | identity |
| `RCWAStack` (JAX) | `_cascade` / `_rt` of `_solve_once` | every layer's `P @ Q` / generator | identity |
| `pmm_efficiency_1d` | `_amplitudes` of `_jpmm_solve` (round 1's explicit split removed) | the three pencils `(A, B)` | B |
| `pmm_jones_1d` | `_amplitudes` of `_jpmm_jones_solve` | layer `Mbig`, half-space `Kx2` | blockdiag(S0, S0), S0 |
| `PMMStack` (shared, per-layer) | `_solve` of both twins | every `Mbig`, the `Kx2` | as above |
| Berreman, traced tensor | `_cascade` of `_offplane_solve_jax` | every layer's `Delta` | identity |
| hybrid `PMM2DStack` | `_cascade` of the JAX stack | every patterned layer's `P @ Q` | identity |

Why `S0` for the PMM Jones / stack twins (verifier P3-3): `Kx2 = S0^-1 op`
and a uniform layer's `Mbig = eps I - blockdiag(Kx2, Kx2)` are self-adjoint
in `S0`, and their CONSUMER contains half-space modes, whose branch is not a
forward / backward re-labelling (round 1's 4.7e3 with the Euclidean Gram).
The Berreman lift is Euclidean: its forward / backward split of a
propagating mode is by Poynting flux, not by `Re(gam)`, so a real shift of
a propagating cluster keeps the split.

NOT routed: `rcwa_efficiency_2d_shapes` (no JAX path -- it raises
`NotImplementedError`), the 1-D RCWA (analytic half-spaces, no
symmetry-forced layer degeneracy; theta = 0 gradient machine-zero), BOR and
BOR-SEM (no exact degeneracy at a fixed azimuthal order), EME modes (sorted
eigenvalues), the native Berreman cascade (`_solve_jax`: no splitting
parameter reaches its clusters at first order), the hybrid 2-D PMM cell twin
(`pmm_efficiency_2d_cell`: round 2 of E3 measured it exact).  Every RCWA
entry WITH a JAX path is routed.

### 10.2 The switch (item 5, verifier P2-2)

`lumenairy.backend.set_jax_cluster_rule(enabled) -> previous`,
`lumenairy.backend.jax_cluster_rule(enabled)` (context manager),
`lumenairy.backend.jax_cluster_rule_enabled()`; process default from
`LUMENAIRY_JAX_CLUSTER_RULE` (`0` / `off` / `false` / `no` -> off).  It is
read by `_jax_eig_cluster_adjoint` and `_jax_cluster_routed` themselves, so
it covers every twin including the pure staggered one.

**Why a library setting and not a keyword.**  The rule is applied deep
inside private twins, several layers below eleven public entry points
(and below the stacks' methods); a keyword would have to be threaded
through every public signature and every internal call between them, and
would still miss a solver called from user code that does not expose it.
The cost the switch removes is a property of how a caller differentiates
(a vmapped / budgeted gradient far from symmetry), not of one call, so a
process-level setting with a scoped context manager is the honest form.
It is read when a solve is TRACED: a `jax.jit`-compiled function keeps the
setting it was compiled with (documented in the setter and the CHANGELOG).
The round-1 private constant `_EIG_CLUSTER_GAP_REL` stays as the rule's
threshold (tests still use it to engineer the rule-off arm).

**Can the vmapped no-cluster case be cheap without the switch?**  Yes, and
it is now: the branch predicate is the OR over the batch, computed by a
`custom_vmap` rule OUTSIDE the vmapped region, so `lax.cond` stays a branch.
Measured, vmap of 4 at a no-cluster point (RCWA rectangle, theta 0.2;
`f1`, `e1`): Windows ON 0.052 - 0.059 s, OFF 0.047 s, PRE 0.054 s (before
the batch-reduced predicate, ON took 0.88 - 1.10 s); WSL ON 0.088 s, OFF
0.046 s.  The switch remains for compile time (ON compiles both branches:
7.5 - 7.9 s vs 1.9 - 2.1 s for that vmapped gradient) and for a batch that
contains an accidental cluster.

Pins (`test_jax_symmetric_point_gradients.py`): OFF at the four-fold cell
is WRONG (> 1e-3; measured 0.23 / 0.28 TE, 0.39 / 0.47 TM) and ON right
(< 1e-7); away from the symmetry ON and OFF agree (< 1e-12; measured
<= 4e-16); the jaxpr of a vmapped gradient holds exactly one eig with the
switch off (the plain solve) and the lifted eigs with it on; a vmapped
batch mixing the symmetric cell and offset cells equals the per-point
gradients.

### 10.3 Before and after, per family, both builds

AD vs a premise-checked FD (h^2 premise 10.2 - 13.6 on every component above
1e-3 of max |FD|, medians 11.29 - 11.41), at the symmetric point unless
stated.  BEFORE = round-1 `*_post_*` / `pre`; AFTER = `r2post`.

| family | configuration | before win / wsl | after win / wsl | gauge after |
|---|---|---|---|---|
| `rcwa_jones_2d` (g1) | four-fold cell, x-side blocks | 0.34 / 0.38 | 3.7e-10 / 2.4e-10 | <= 2.5e-10 |
| | control: all four sides | 1.8e-10 / 3.6e-10 | same | -- |
| | conical (0.2, 0.3): no cluster | 1.2e-10 / 9.9e-11 | same | -- |
| `rcwa_jones_2d` (verifier) | C4v post, x-widening (Laurent) | 6.1e-2 / 4.4e-2 | < 1e-7 (pin) | -- |
| | isotropic C4v, d / d eps_xy (truth 0) | 0.088 / 0.083 abs | at the FD's resolution (pin) | -- |
| `RCWAStack` (g1) | one layer | 0.23 / 0.28 | 1.8e-10 / 1.9e-10 | <= 6.4e-10 |
| | + uniform spacer | 0.13 / 0.14 | 3.3e-10 / 3.3e-10 | <= 2.8e-10 |
| | + second patterned layer | 0.15 / 0.21 | 4.5e-10 / 2.7e-10 | <= 3.2e-10 |
| Berreman (d4) | isotropic layer, d / d eps_xy | 0.99 / 0.99 | 1.5e-9 / 1.7e-9 | 2.3e-9 / 3.7e-9 |
| | d / d(eps_xx - eps_yy) | 1.3e-11 (gauge 0.75: luck) | 1.1e-9 / 2.1e-9 | 2.9e-9 / 5.6e-9 |
| `pmm_jones_1d` (d5) | d / d angle at 0 | 2.72 / 2.72 | 1.3e-10 / 6.0e-11 | 1.7e-11 |
| | mirror-identity defect of AD | 0.141 | 1.1e-10 (FD 3.3e-11) | -- |
| `PMMStack` (d6) | shared, 1 layer | 1.50 | 1.4e-10 / 1.3e-10 | 9e-12 |
| | shared, 2 layers | 5.17 | 3.3e-10 / 3.0e-10 | 5e-12 |
| | shared, + uniform spacer | 1.05 | 8.3e-11 / 9.1e-11 | 1.6e-11 |
| | per-layer grids, 3 layers | 6.61 | 1.0e-8 / 1.0e-8 | 4e-11 |
| hybrid `PMM2DStack` (d7) | traced layout, corner eps (Laurent) | 0.295 / 0.347 | 2.3e-10 / 2.0e-10 | 3.8e-10 / 3.0e-10 |
| | traced layout, Li (no cluster) | 1.5e-10 | same | -- |

The per-layer 1.0e-8 sits on a fixture whose layer-3 window grid breaks the
mirror at 7.4e-5 (the FD's own mirror defect, reproduced by AD).

Unchanged controls (every JSON leaf identical before and after, both
builds): BOR (d1, 1.7 - 3.6e-10), BOR-SEM (d2, 1.2e-8 .. 3e-10), EME (d3,
cluster sums 1e-9 / 2e-10; individual sorted members 0.76 / 0.11, the
non-differentiable quantity of section 5), the hybrid stack's d / d theta
cases (4e-11 .. 1.2e-9).

Off the symmetric point the routed families are unchanged (1e-10 .. 4e-8
at 1e-5 .. 1e-3 rad / offsets 1e-6 .. 1e-2): the rule changes nothing where
there is no cluster.  "Off" here means a splitting above ~1e-10 of the
spectrum (verifier P3-4): an angle above ~1e-7 rad on the 1-D twins, a
contrast offset above ~1e-8 on the RCWA cell.

**Forward bytes** (SHA-256; NumPy / JAX eager / `jax.jit`; PRE vs r2post):
`b1` 33 / 33, `g2` (rcwa_jones_2d in-plane, conical, out-of-plane uniaxial,
RCWAStack 1 - 2 layers and conical) 24 / 24, `h` (Berreman, pmm_jones_1d 0
and 0.2 rad, PMMStack three fixtures, hybrid stack traced Laurent / Li)
27 / 27 -- on each build.  (The jitted `rcwa_jones_2d(formulation='li')` on
a traced tensor changes, by design: 10.4.)

### 10.4 Verifier P1-1: `rcwa_jones_2d(formulation='li')` on a traced tensor

Root cause as the verifier found it: a traced tensor cannot be inspected for
out-of-plane components, so it goes to the general (out-of-plane-capable)
path, whose 'laurent' / 'li' branch built every block by the direct rule.
NumPy (and an eager JAX call) send an IN-PLANE tensor's 'li' to the in-plane
route (`_inplane_ops`: the Li-1997 inverse rule on the diagonal blocks) and
only an out-of-plane tensor's 'li' to the general path (where it IS the
direct rule).  Built, not refused: the traced general path now computes
both the direct-rule blocks and the in-plane Li diagonal blocks (the SAME
`_inplane_ops` -> `_li_convolutions_2d` the NumPy route uses) and selects
on the tensor's VALUE with `_core._tensor_inplane_mask` -- the one
definition of "in-plane", now also behind the NumPy
`_tensor_offplane_present` (same threshold, same values).  For an in-plane
tensor the off-plane blocks vanish and the folded components are the raw
ones, so only the two diagonal blocks differ.  Measured (k1): jitted
forward vs NumPy 'li' 5.6e-16 / 1.2e-15 at the four-fold cell and 2.2e-15 /
6.9e-16 on a no-symmetry cell (NumPy 'li' vs 'laurent' differ by 2.2e-3 /
7.3e-4 there -- the size of the old error on this fixture); the 'li'
gradient at the four-fold cell 2.6e-10 / 3.2e-10.  The verifier's two P1-1
pins (jitted forward = 'li'; 'li' gradient on a no-symmetry cell) pass.

### 10.5 Found on the way: `RCWAStack` leaked tracers through its mode cache

`g1` found (on PRE and POST, both builds): `RCWAStack` cached its half-space
modes in the module-level `_HOMOG_CACHE` even inside a `jax.jit` trace,
where `Kx` / `Ky` are tracers, so `jit(f)` followed by `jit(jacrev(f))` or
an eager `f` raised `UnexpectedTracerError`.  A solve whose `Kx` is a
tracer now computes the modes uncached.  Pinned by
`test_rcwa_stack_jit_then_grad_then_eager_does_not_leak_tracers`.

### 10.6 Cost

Jitted, test sizes.  Windows measured interleaved PRE / POST on a quiet box
(two rounds, `e1`, after the lazily built lift); WSL one run each, with other
work on the machine.

| case | cluster? | gradient PRE | gradient POST | grad compile PRE / POST |
|---|---|---|---|---|
| RCWA four-fold cell | yes | 0.025 - 0.027 s | 0.10 - 0.11 s (WSL 0.12) | 2.1 - 2.2 / 4.8 - 5.2 s |
| RCWA rectangle, theta 0.2 | no | 0.028 - 0.035 s | 0.028 - 0.030 s (WSL 0.026) | 1.4 - 1.7 / 3.5 - 4.2 s |
| 1-D twin at 0 | yes | 0.0031 - 0.0035 s | 0.014 - 0.015 s (WSL 0.012) | 2.2 - 2.6 / 6.1 - 7.3 s |
| 1-D twin at 0.2 rad | no | 0.0035 s | 0.0035 - 0.0038 s (WSL 0.0042) | 1.6 - 1.7 / 6.5 - 6.6 s |
| RCWA rectangle vmap of 4 | no | 0.054 s | 0.052 - 0.059 s (WSL 0.110) | 1.8 - 2.1 / 5.1 - 5.3 s |

Round-2 families (`h_timing`, Windows, POST with the lazy lift vs PRE):
Berreman 0.2 -> 1.1 ms; `pmm_jones_1d` at 0 13 -> 55 ms and at 0.2 rad
15 -> 58 ms (that point HAS a cluster: the layer `Mbig` holds an accidental
pair 3.1e-7 of the spectrum apart, inside `gap_rel`); `PMMStack` 2 layers
61 -> 255 ms; hybrid stack 45 -> 172 ms, traced layout 68 -> 167 ms; grad
compile 2 - 5.5x.  Before the lift moved into the lifted branch, the
no-cluster forward-of-gradient paid for building it too (round 1's table:
1.1 - 1.3x on the 1-D twin at 0.2 rad); after it, 1.0 - 1.1x.  The verifier's own reading of round 1 (rule on vs off, one
process): jitted with a cluster 4.0 - 5.4x, without 0.93 - 1.48x, compile
2.8 - 4.6x, vmapped without a cluster 3.5 - 5.4x, first eager gradient at a
cluster 16 - 24 s vs 3.3 - 4.7 s -- the vmapped figure is the one this
round's batch-reduced predicate removes.

### 10.7 Tests

* `tests/unit/test_jax_symmetric_point_gradients.py`: one pin per routed
  family (`rcwa_jones_2d`, `RCWAStack`, Berreman, `pmm_jones_1d`,
  `PMMStack` shared and per-layer, hybrid stack traced layout), bar 1e-7
  vs the premise-checked FD; the switch (wrong at the symmetric point /
  inert elsewhere; the vmapped-jaxpr eig count; a mixed vmapped batch);
  P1-1 at the four-fold cell; the cache leak.  The gauge test now uses a
  FAITHFUL rotation (the eig's own backward rule runs on the rotated basis;
  the verifier's method note) and the public switch.  Each test 1 - 22 s
  on Windows except the gauge pair (32 - 39 s; it compiles six gradients).
* `tests/unit/test_verify_jax_symmetric_point_gradients.py` (the
  verifier's): its five strict xfails are flipped into passing tests
  (markers removed; each carries the defect it pinned as a comment); two of
  its probes of the round-1 internals were adapted to the round-2 API
  (problems are `(L, G, K)`; the RCWA entry records through
  `_jax_twin_eig_plain`), what they pin unchanged.

Results (the gate set: the round-1 gate files, the verifier's file, the
twins' own JAX files -- `test_v5_14_5_emt_and_berreman_jax`,
`test_v5_20_10_berreman_internal_jax`, `test_v5_14_2_jax_stacks`,
`test_v5_20_2_pmm_jones_2d_jax`, `test_v5_20_1_rcwa_2d_oop_jax`,
`test_v5_20_3_rcwa_1d_oop_jax`, `test_v5_3_jax_monte_carlo_tolerancing`,
`test_v5_24_3_jax_ci_coverage` -- the W9, autodiff, a12 / a14, w7 and
disable-JAX files):
* Windows, `-n 2`, BLAS threads 2, with the history / doc-identifier /
  changelog-walker / census / re-export / public-API files and
  `test_rcwa.py`: `804 passed, 7 skipped` and one failure, the WP-A17
  history ratchet on a comment in `rcwa/stack.py` naming the date it was
  measured -- moved into `docs/history/lumenairy.elements.rcwa.stack.md`
  and re-recorded; the history files then `768 passed`.  An earlier run
  had failed `test_v5_14_5_emt_and_berreman_jax.py::test_jax_grads_match_fd`
  (the record / replay concreteness mismatch of 10.1, fixed before the
  final runs).
* WSL, serial (import path asserted), the gate set: `641 passed in
  3466 s`; ruff clean; the history / doc-identifier files pass, and
  `test_public_api.py::test_installed_metadata_version_matches_source_version`
  fails on the WSL venv's stale editable-install metadata (5.11.0 vs
  5.49.0 -- an environment property, unrelated to this change).
* `python -m mypy`: no issues in 33 files; `record_history_fingerprints.py
  --check`: OK (`_berreman_jax`, `pmm/_core`, `rcwa/stack` re-recorded in
  the code commit).

### 10.8 Not done

* GPU; an idle-box WSL timing.
* The EME mode solver's individual degenerate eigenvalues stay
  non-differentiable at a splitting cluster (no correct value exists); only
  cluster-symmetric functions of them have a gradient there.
* The hybrid stack's 'li' operator does not keep the C4v degeneracy (gap
  2.8e-6, section 5) -- not investigated (no gradient defect follows).

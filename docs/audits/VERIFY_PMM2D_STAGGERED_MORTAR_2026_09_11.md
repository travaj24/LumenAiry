# VERIFY -- per-layer element grids (L2 mortar) + non-uniform segment boundaries, PURE staggered 2-D PMM

**Date:** 2026-09-11 · **Worktree** `C:/tmp/lum_vmortar`, branch `verify/mortar`
off `wave2/pmm2d` `dd06c09` · **Class:** independent adversarial verification

**Under verification:** `docs/audits/BUILD_PMM2D_STAGGERED_MORTAR_2026_09_11.md`
(the build), `docs/audits/EXPERIMENT_PMM2D_STAGGERED_MORTAR_2026_09_10.md` (the
spec, incl. F1-F5) and, where the two features touch it,
`docs/audits/FIX_PMMSTACK_SLIVER_WALLS_2026_09_11.md` (the 1-D sliver fix and
its S7 caveat about "the in-flight mortar work").

**Binding:** `docs/TESTING_STANDARDS.md`.

**Method.**  Nothing below is read out of the build doc and checked off.  Every
claim is RE-MEASURED by scripts written for this audit
(`validation/probe_verify_mortar/`, with a README), with fixtures of my own
wherever the claim is about the library's behaviour, and with the build's own
fixture only where the claim is *about that fixture's numbers* (the equal-DOF
ratios).  Every script asserts which tree it imported.  The "without" arm is
the READ-ONLY main clone
`D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy`, whose
`HEAD` (`50824e9`) has an EMPTY diff against `a68a0da` over
`lumenairy/elements/pmm/`, `lumenairy/elements/rcwa/` and `lumenairy/cache.py`
-- verified with `git diff --stat`, so it IS the 5.44.0-candidate "without" arm
for everything these two features touch.

| arm | OS / toolchain | python | numpy | scipy |
|---|---|---|---|---|
| **WIN** | Windows 11, MSVC wheels | 3.14.6 | 2.4.4 | 1.17.1 (scipy-openblas) |
| **WSL** | WSL2 Ubuntu, gcc wheels | 3.12.3 | 2.4.6 | 1.17.1 (scipy-openblas) |

`OMP/OPENBLAS/MKL_NUM_THREADS = 1` on both.

**Wall-time caveat, stated once.**  This box ran three other verification /
build agents' worktrees throughout.  Every wall time below therefore carries
heavy contention; ratios are quoted only where both arms were measured
back-to-back inside one script, and absolute seconds are indicative only.

---

## 0. VERDICT TABLE

| # | claim | verdict | key numbers (this audit) |
|---|---|---|---|
| **N1** | the integer wall path is BIT-IDENTICAL to the pre-change library | **CONFIRMED** | **315/315 sha256 equal** against the read-only main clone: 21 stack fixtures, 102 `Basis1D` matrix hashes, 192 assembled-pencil hashes |
| **N2** | an explicit `linspace` array is ULP-close, not bit-identical | **SHAPE CONFIRMED, STATED VALUES REFUTED** | worst 4.626e-16 rel; bit-identical on ONE of six grids (`d = 1.2, N = 2`), not the four the doc and the test docstring claim (S8.2) |
| **N3** | two exact-wall representations agree inside a TRIANGLE bar | **CONFIRMED 5/5 -- and the bar is a TAUTOLOGY** | 5/5 inside, but `gap <= en + eu` is a theorem, so the shipped assertion could not fail; restated (S8.2) |
| **N4** | arbitrary walls converge to the EXACT 1-D answer | **CONFIRMED** | 8.70e-04 -> 2.19e-05 over `M = 4..8` on my own fixture (closure to 5.8e-10); 6.75e-02 -> 8.13e-05 on a second whose longest segment is 0.91 of the period |
| **N6** | the conforming identity survives NON-UNIFORM grids | **CONFIRMED** | 8.61e-15 and 5.33e-14 against derived bars 1.18e-11 / 8.88e-10 |
| **N7** | FAIL-BEFORE for each of the four `J -> J_n` sites | **CONFIRMED 4/4, two-sided** | uniform arm moves **exactly 0.0** (sha equal) at all four; non-uniform arm moves 1.03e+01 / 4.18e-01 / 6.43e-02 / 1.08e-01 |
| **FIFTH SITE** | `_stag_parity_1d` must refuse a non-mirror wall set | **CONFIRMED as behaviour, RATIONALE BOUNDED** | refusal correct; but bypassing the guard changes the stack answer by **exactly 0.0**, because `_stag_block_eig`'s own structural residual reads 6.80e-01 against a 1e-10 bar. The guard is defence-in-depth, not the only thing preventing a wrong answer |
| **SIXTH SITE** | is there one the build missed? | **NO `J_n` site; TWO non-uniformity-sensitive sites found** | (a) `_stag_fourier_projection`'s FIXED `nq = 2M + 8` quadrature ORDER (kernel error to 7.5e-04); (b) the UNGUARDED `np.linalg.solve` in the mortar interface (bare `LinAlgError`) |
| **G1** | conforming per-layer is BIT-EXACT vs `'shared'` | **CONFIRMED** | sha equal, move exactly 0.0, vertical AND slanted |
| **G2** | the conforming identity through the FORCED mortar, derived bar | **CONFIRMED 6/6** | worst 9.09e-14 against a bar derived here from `cond_2(G)` measured here; 201x-16659x inside |
| **G3** | the H-row V1/V2 swap is load-bearing AND invisible to conforming gates | **CONFIRMED, both halves** | swap off costs **1237x / 4837x** on the observable and **1.5e+06x / 3.8e+09x** on closure; the conforming identity reads 1.8e-15 (on) vs 1.9e-14 (off) -- it CANNOT see it |
| **G4** | a transparent split is not measurably worse non-conforming | **CONFIRMED** | 1.4e-15 .. 4.5e-14 on (2,2)/(2,4)/(2,3)/(3,4), normal and oblique; the GENERALIZED twin reproduces `berreman_jones_1d` to 3.1e-11 .. 9.99e-12 at `M = 7` over three mounts (S9.2) |
| **G5** | per order vs the exact 1-D oracle, with the anti-mirror tripwire | **CONFIRMED** | TE 6.21e-02 -> 1.13e-04 over four rungs; mirror/direct 5.4x -> 2529x |
| **G6** | EQUAL-DOF: not worse | **CONFIRMED, and the build's ratios reproduce exactly** | 0.1769 at `q = 12` and 0.0587 at `q = 18` -- the build's 0.176877 / 0.058687 to 4 digits, against a reference built here at degree 16; staircase 0.634 / 0.504 likewise; corner-dominated pillar 0.244 / 0.317 (S7) |
| **G7** | two-sided lossless closure, scalar AND Hermitian tensor | **CONFIRMED** | scalar 2.69e-04 -> 1.47e-06 and gyrotropic 1.18e-03 -> 4.24e-06 over `M = 4/5/6` (S9.3) |
| **G8** | the order cap is DERIVED from the END grids and RAISES | **CONFIRMED** | cap `= (q-1)//2` exactly; raising the END layers' `n_modes` raises it |
| **G9** | the conditioning census refuses nothing in the useful range | **CONFIRMED in the useful range, REFUTED as a general statement** | worst equilibrated `rcond` 1.418e-05 over 9 solves (S9.6); but on a sliver grid the UNGUARDED solves reach `cond_2` 4.9e+10 and then throw (S6.3-S6.4) |
| **O-3** | `retain_internal` / `layer_absorption` budget closes | **CONFIRMED** | 5.35e-05 -> 2.84e-06 -> 3.43e-08 over `M = 5/6/7` |
| **taper** | `add_tapered_pillar` on the pure stack vs the hybrid staircase | **CONFIRMED** | at `M = 7` the pure arm is 5.71e-04 from the hybrid's degree-11 answer -- HALF the hybrid's own 1.12e-03 self-gap -- with a closure 1020x tighter (S6.1) |
| **SLANT** | the per-layer slanted gate the build left OPEN | **BUILT AND PASSING** | split across NON-conforming grids reproduces `pmm_efficiency_1d_slanted` per order to **7.0e-07 .. 5.6e-06**; wrong-sign arm 9.5e+03x-1.3e+05x worse at the SAME closure |
| **equal-DOF regime** | the rung where the mortar LOSES | **NOT REPRODUCED on the library** | the experiment's 1.48x at `q = 30` is its prototype; the shipped library reads 0.317 there, and neither arm is resolvable against the oracle at `q >= 24` (S7.1) |
| **memory** | 900x at `(6,12), M = 6`; the apply is SLOWER below ~100x100 | **CONFIRMED both halves** | 900x and 40.35x; (2,3) reads 0.49x on a 100x225 dense operator (S10.1) |
| **SLIVER on the mortar route** | "the mortar route is the safe one" | **SPLIT VERDICT** | walls differing BETWEEN layers: **CONFIRMED SAFE** (deviation exactly linear in `delta` to 1e-6, closure flat, cross-mass `cond` constant). Walls `delta` apart INSIDE ONE layer with neighbours on other grids: **NEW SILENT-WRONG**, up to 7.8e-03 absolute in `R(0,0)` with the closure pinned at 1.3e-05, and a bare `LinAlgError` below `delta ~ 1e-7` |

---

## 1. WITHOUT-ARM BIT IDENTITY (task 1)

`v1_bit_identity.py` + `v1_compare.py`.  ONE file, run twice with different
`PYTHONPATH`, hashing raw IEEE bytes (dtype + shape + `tobytes`).

| arm | count | mismatches |
|---|---|---|
| **stack fixtures on the SHARED path** -- scalar pillar (normal / conical / `N=3,M=6`), stripe A\|B stack, uniform\|patterned\|uniform sandwich, in-plane tensor, Hermitian gyrotropic, OOP tensor (`symmetry` auto AND False), uniform OOP tensor, magnetic (patterned `mu_cell` AND uniform `mu`), slanted patterned, slanted conical `(t_x, t_y)`, slanted 2-layer stack, `retain_internal` + `layer_absorption` (1-layer lossy and 3-layer), `per_order_amplitudes`, plus TWO 1-D `PMMStack` arms (shared and `layer_grids='per-layer'`, since `stack.py` also changed on this branch) | **21** | **0** |
| **`Basis1D` matrix families** at INTEGER `N` -- 6 grids `(d, N, M, tau)` x 17 families (`mass` tt/bb/tb, eps-weighted mass, `stiff` x2, `mixed` x3, `_global_pair_segmat` x4, `_stag_fourier_projection` at three `alpha0`, geometry) | **102** | **0** |
| **assembled 2-D pencils** `Rmat / Lmat / Stt / Schur / Agen / Bgen` (+ `offplane`, `Ggram_blocks`) over 4 grids x 6 cell kinds (scalar, in-plane tensor, OOP, gyrotropic, MAGNETIC, SLANTED) at `alpha0 = (0.31, -0.17)` | **192** | **0** |

**TOTAL 315/315 identical.**  The arm is NOT vacuous: 8 of the pencil hashes
are real `Agen`/`Bgen` (the OOP and slanted assemblies), 16 are real
`Rmat`/`Lmat`/`Stt`/`Schur`, 4 are the magnetic `Ggram_blocks`, and 24
`offplane` flags confirm the OOP and slanted rows took the first-order
generator.

**N1 CONFIRMED.**  This is a stronger statement than the build's own N1: the
build's durable arm compares against an IN-PROCESS scalar-`J` reimplementation
(12 hashes); this one compares against the SHIPPED PRE-CHANGE LIBRARY on disk,
which is the claim a user cares about.

---

## 2. THE NON-UNIFORM BASIS (task 2)

### 2.1 The three scalings, RE-DERIVED and then measured against an independent oracle

Granet Eq. 31 maps segment `n` by `x = J_n u + c_n` with
`J_n = (x_{n+1} - x_n)/2`.  So, per segment,

| form | derivation | scale |
|---|---|---|
| mass `INT phi psi dx` | `dx = J_n du` | **`J_n`** |
| stiffness `INT phi' psi' dx` | two factors `1/J_n` from `d/dx`, one `J_n` from `dx` | **`1/J_n`** |
| mixed `INT phi psi' dx` | one `1/J_n`, one `J_n` -- they cancel **on every segment whatever its length** | **`1`** |

`v2_basis.py scal` re-measures all three against an INDEPENDENT oracle that
evaluates the global basis functions DIRECTLY in physical space (reference
modified-Legendre values composed with segment `n`'s own affine map) and
integrates with a `4M + 10`-point Gauss rule per segment -- it shares no code
path with `_global_matrix`.  Five wall sets (uniform, two arbitrary, one
mirror-symmetric, one strongly skewed `[0, 0.05, 0.4, 0.42, 1.1, 1.37]`),
`M = 4` and `6`, `tau = 1` and `exp(-0.41i)`, four set pairs, three forms:

* **worst library-vs-oracle relative error 2.375e-14** over all 240 matrices.

Two ANALYTIC identities, which no oracle can be wrong about, both exercised on
the arbitrary wall sets (the `N` hats of `Btilde` at `tau = 1` are a partition
of unity, so the hat block of the mass sums to `d` and of the stiffness to `0`):

* `|sum_ij <T_i|T_j> - d| / d` worst **1.621e-16**;
* `|sum_ij <T_i'|T_j'>|` worst **1.954e-14**.

**The three scalings are CONFIRMED on arbitrary walls**, and `mixed`'s
scale-freeness -- the reason the build edited three sites and not four inside
`Basis1D` -- is confirmed as an identity, not an approximation.

### 2.2 The de Rham property survives non-uniformity

`d(Btilde) subset span(B)` is what makes this basis spurious-free.  Measured
POINTWISE (the Galerkin energy form differences two nearly-equal numbers and
floors at `sqrt(eps)` ~ 1e-8, which the UNIFORM lattice reads too -- estimator
noise, not a defect; the first cut of this probe hit exactly that and was
rewritten): the least-squares residual of each `dBtilde_j` against the `B` set,
evaluated on a dense per-segment Gauss grid, relative to `max|dBtilde_j|`:

**worst 9.047e-15** over the same five wall sets at `M = 4, 6`.

Analytically this is why: `d/dx` of a degree-`<= M-1` local function is degree
`<= M-2`, which is exactly `span(B)` restricted to a segment; the per-segment
factor `1/J_n` is a scalar and cannot leave that span.  The property is
per-segment and scale-free.

### 2.3 FAIL-BEFORE for the four `J -> J_n` sites, re-measured two-sided

`v2_basis.py fail`.  Each site is reverted IN PROCESS to a scalar-`J`
reimplementation written for this audit (`Jn[0]`, i.e. exactly the pre-change
code) and both arms run in the same process on the same build.  Device: one
patterned layer on its own 3-segment grid, at `theta = 0.18, phi = 0.35`,
`M = 5`; the UNIFORM arm puts its walls at `1/3, 2/3` and the NON-UNIFORM arm
at `0.2371, 0.6183`.

| site reverted | UNIFORM arm | NON-UNIFORM arm moved by | its closure |
|---|---|---|---|
| `Basis1D._global_matrix` | **sha256 equal, move exactly 0.0** | **1.028e+01** | 1.49e-05 -> **1.60e+01** |
| `_global_pair_segmat` | **sha256 equal, 0.0** | **4.176e-01** | 1.49e-05 -> **4.00e-01** |
| `Granet2DTransverseE._eps_dir` | **sha256 equal, 0.0** | **6.432e-02** | 1.49e-05 -> **6.67e-02** |
| `_stag_fourier_projection` | **sha256 equal, 0.0** | **1.078e-01** | 1.49e-05 -> **2.06e-01** |

**CONFIRMED 4/4, two-sided.**  A uniform-lattice-only gate cannot see ANY of
the four -- the uniform arm does not move a single bit at any site -- which is
the structural symmetry with G3 the build kept returning to.

### 2.4 Two device invariances a mis-scaled site cannot fake

Neither is in the build's gate list; both are cheap and both are decisive
about the whole assembly rather than about one matrix.

**MIRROR.**  At normal incidence a device and its `x -> P - x` image have
identical efficiencies with the order index negated.  Built on ARBITRARY walls
(`0.2371, 0.6183` against `0.3817, 0.7629`, with the strip tile reversed):

| `M` | `dR` | `dT` |
|---|---|---|
| 4 | 1.096e-15 | 6.217e-15 |
| 5 | 5.163e-15 | 7.661e-15 |

**CYCLIC TRANSLATION** (position invariance).  The same three-strip device
shifted by `1 - w_0` of the period, so the wall SET is a different non-uniform
partition of the same physical structure, at `theta = 0.21`:

| `M` | `dR` | `dT` |
|---|---|---|
| 4 | 1.409e-10 | 2.494e-10 |
| 5 | 2.115e-12 | 7.126e-12 |

The translation residual is NOT round-off and it shrinks ~2 decades per modal
rung -- it is the far-field projector's quadrature, S2.6.

### 2.5 THE SIXTH-SITE CENSUS

`grep --include='*.py' -rn '\.J\b|\.h\b|\.xb' lumenairy/ validation/ tests/`,
then read every hit.

**Library (`lumenairy/`) consumers of `Basis1D.J` and `.h`: NONE outside
`Basis1D.__init__` itself.**  There is no un-migrated scalar-jacobian reader
anywhere in the shipped tree.  (`.J` elsewhere in `lumenairy/` matches only
author initials in bibliography comments -- `Noll, R.J.`, `Kasdin, N.J.` --
and `.h` matches nothing in `pmm/` or `rcwa/` beyond the two lines in
`Basis1D.__init__`.)

**Library consumers of `.xb`**, all five correct:

| site | status |
|---|---|
| `_stag_fourier_projection` (quadrature points) | SITE 4, migrated |
| `_stag_parity_1d` (`np.diff(basis.xb)` mirror test) | the FIFTH site, S2.7 |
| `_stag_cross_mass_1d` (union integration mesh) | correct BY CONSTRUCTION -- it takes the union of the two wall arrays and integrates per sub-interval, and its rule `leggauss(M_a + M_b + 2)` IS exact there (polynomial x polynomial) |
| `_stag_basis_fingerprint` (cache key) | correct -- keys on the wall BYTES, so two grids sharing `N` and `M` cannot collide (measured, S4.6) |
| `stack2d_pure._walls_of` | correct -- hands the wall ARRAY back when `not uniform` |

**Validation tree:** six research probes under
`validation/probe_pmm2d_staggered_oop/` read `.J` / `.h` off a library
`Basis1D`.  All construct their bases with an INTEGER `N`, so they still work;
on a non-uniform basis they would raise `TypeError` immediately, which is the
designed behaviour and not a defect.

**So: no sixth `J -> J_n` site exists.  The migration is complete.**  But the
census turned up TWO OTHER sites whose correctness silently depended on
"segments are `d/N` long", neither of which is a jacobian:

* **S2.6 -- `_stag_fourier_projection`'s quadrature ORDER** (defect D3);
* **S6.4 -- the UNGUARDED `np.linalg.solve` in the mortar interface** (defect
  D2).

### 2.6 The far field on non-uniform segments -- y-momentum, and the quadrature order

`v2_oracles.py farfield` and `v2_quad.py`.

**y-momentum leak.**  A y-UNIFORM device (strips constant along `y`) must put
every photon in an `n = 0` order.  Carried on THREE different y grids, at
`theta = 0.21`:

| y grid | `M = 4` | `M = 5` | `M = 6` |
|---|---|---|---|
| uniform `1/3, 2/3` | 4.71e-29 | 1.83e-28 | 1.50e-27 |
| non-uniform `0.1041, 0.8317` | 2.86e-28 | 8.26e-28 | 1.42e-27 |
| near-coincident `0.4903, 0.5102` | 2.78e-28 | 5.17e-28 | 1.59e-27 |

**No y-momentum leak at any grid: the order orthogonality of the projector is
structural, not incidental.**

**The quadrature ORDER, which IS a defect.**  `_stag_fourier_projection`
integrates `phi_a(x) e^{+i(mG + a0)x}` with a FIXED `nq = 2M + 8` Gauss rule
PER SEGMENT.  On a uniform lattice a segment is `d/N` long, so the phase a
segment carries is at most `2 pi m_max / N` and the rule was sized for it; with
arbitrary walls one segment can be almost the whole period.  The rule was not
re-sized.  Relative error of the shipped projector against an 8x-refined one:

| longest segment | `M=4, m<=3` | `M=4, m<=7` | `M=6, m<=3` | `M=6, m<=7` | `M=8, m<=7` |
|---|---|---|---|---|---|
| 0.33 d (uniform N=3) | 7.7e-15 | 7.7e-15 | 6.3e-15 | 6.3e-15 | 6.7e-15 |
| 0.50 d | 6.4e-15 | 7.4e-11 | 5.4e-15 | 5.4e-15 | -- |
| 0.62 d | 7.1e-15 | 2.2e-08 | 6.0e-15 | 2.0e-12 | -- |
| 0.80 d | 6.3e-15 | 1.3e-05 | 6.1e-15 | 6.3e-09 | -- |
| 0.91 d | 1.0e-13 | 2.4e-04 | 6.1e-15 | 3.0e-07 | -- |
| **0.96 d** | 4.6e-13 | **7.5e-04** | 6.7e-15 | 1.4e-06 | 5.6e-11 |

The scaling is exactly the predicted one -- the error grows with (longest
segment) x (highest order) and falls with `M`, because `nq` grows with `M` and
with nothing else.  **DEVICE-level it is currently masked**, and the reason is
structural: the order cap ties `m_max` to `q = N(M-1)`, so a grid with few
segments cannot ask for a high order.  On a 3-segment grid with the longest
segment at 0.91 d, `theta = 0.21`, `n_orders = 4`:

| `M` | shipped vs 8x-refined | err vs the EXACT 1-D oracle | closure |
|---|---|---|---|
| 4 | 3.17e-11 | 6.75e-02 | 8.08e-02 |
| 5 | 1.33e-15 | 6.89e-03 | 2.71e-03 |
| 6 | 2.78e-15 | 1.20e-03 | 3.12e-04 |
| 7 | 5.00e-15 | 8.13e-05 | 2.14e-05 |

-- i.e. NINE decades below the discretisation error at the worst rung.  On
realistic multi-feature grids (two narrow pillars, `N = 5`, `M = 6`,
`n_orders = 8`) it reads 1.8e-15.  The one construction that reaches the
kernel-level regime (`N = 6` with five 0.02-wide segments and one 0.90 one) is
either REFUSED by `_guarded_lstsq` (rank-deficient far field, at
`n_orders = 7, 8`) or already broken by a much larger error (closure 6.3e+02 at
`n_orders = 7`, `M = 5`).

**BOUNDED, and reported as a latent defect (S10, D3): the kernel reading is
real, the device-level exposure is currently blocked by the order cap and the
least-squares guard, and the one-line fix is to size `nq` from the segment's
own length.**

### 2.7 The FIFTH site (`_stag_parity_1d`), two-sided -- and its rationale bounded

`v2_basis.py parity`.

**The refusal is CORRECT and the acceptance is CORRECT:**

| wall set | `_stag_parity_1d` | expected |
|---|---|---|
| uniform `N = 4` | ACCEPTED | yes |
| **mirror-symmetric non-uniform `0.2, 0.8`** | **ACCEPTED** | yes |
| mirror-symmetric non-uniform `0.15, 0.5, 0.85` | ACCEPTED | yes |
| asymmetric `0.2371, 0.6183` | REFUSED | yes |
| asymmetric `0.1, 0.2, 0.7` | REFUSED | yes |

A MIRROR-symmetric NON-UNIFORM wall set is not merely accepted, it is CORRECT:
on a centred out-of-plane tilted-director cell at normal incidence with walls
`0.2, 0.8`, the reduced solve reproduces the dense `4 q^2` solve to
**1.710e-14** at the stack level, its assembled-pencil structural residuals
read `|RAR+A|/|A| = 1.307e-15` and `|RBR-B|/|B| = 1.322e-16` against the
`1e-10` bar, and forcing the reduction reproduces the dense spectrum to
**1.025e-12** (one-sided Hausdorff over 324 eigenvalues).

**The RATIONALE is BOUNDED.**  The build doc says that without the guard "the
out-of-plane parity block reduction would have been applied to a structure the
pencil does not have".  Measured: bypassing the guard on an asymmetric wall set
(patching `basis.uniform = True` around `_stag_parity_1d`) changes the stack
answer by **exactly 0.0e+00** -- because `_stag_block_eig`'s OWN structural
residual reads `|RAR+A|/|A| = 6.796e-01` and `|RBR-B|/|B| = 6.141e-01` against
its `_STAG_BLOCK_TOL = 1e-10`, i.e. **nine decades over**, and refuses.  The
reduction IS genuinely invalid there -- forced past the residual gate too
(`tol = 1e6`) its spectrum is wrong by **3.921e-01** relative -- but the fifth
site is defence-in-depth and a cost saving, not the barrier.

---

## 3. NON-UNIFORM CELLS AGAINST INDEPENDENT ENGINES (task 2, continued)

### 3.1 Arbitrary walls vs the EXACT 1-D `PMMStack`

`v2_oracles.py oracle1d`.  A y-uniform three-strip device at walls
`0.2371, 0.6183` of a 0.9 um period, `eps = 6.0 / 2.25 / 3.1`, `theta = 0.21`,
`t = 300 nm`.  The exact 1-D pure PMM is the truth for the WHOLE 2-D answer
because the stack is y-uniform; its own degree self-gaps are 4.955e-06
(deg 10-12) and 2.309e-06 (deg 12-14).

| `M` | `q = 3(M-1)` | err vs EXACT 1-D | closure |
|---|---|---|---|
| 4 | 9 | 8.7047e-04 | 1.153e-03 |
| 5 | 12 | 1.3570e-04 | 3.022e-05 |
| 6 | 15 | 5.7503e-05 | 1.564e-06 |
| 7 | 18 | 3.7283e-05 | 8.753e-09 |
| 8 | 21 | **2.1892e-05** | **5.812e-10** |

Converging on the observable and six decades on lossless closure, at walls the
uniform lattice cannot represent at any affordable `q`.  A SECOND fixture
(walls `0.02, 0.93` -- longest segment 0.91 of the period, oracle self-gap
4.239e-07) reads 6.75e-02 -> 6.89e-03 -> 1.20e-03 -> **8.13e-05** over
`M = 4..7` with closure 8.08e-02 -> **2.14e-05**.

**N4 CONFIRMED** on two fixtures of my own.

### 3.2 Arbitrary walls vs the HYBRID, per order

`v2_oracles.py hybrid`, and the oracle trap it exposed, are in S6.1 -- the same
comparison, on a straight pillar and on a staircase.

## 4. THE MORTAR (task 3)

### 4.1 G3 -- the H-row V1/V2 swap, and the BLINDNESS reproduced explicitly

`v3_mortar.py g3`.  MY OWN fixture (deliberately different from the build's):
period 1.05, `wl` 0.72, `eps = 5.0 / 2.10`, `theta = 0.23`, `phi = 0.55`,
pillar 1/2 wide on `N = 2` over pillar 2/3 wide on `N = 3`, scored against the
UNION grid on the common refinement `N = 6` at `M = 4`.

| arms | `PMM2D_MORTAR_H_SWAP` | err vs the union reference | `|R+T-1|` |
|---|---|---|---|
| `M_A = 7, M_B = 5` | **True** (shipped) | **9.2880e-03** | **4.075e-05** |
| `M_A = 7, M_B = 5` | False | **1.1488e+01** | **6.078e+01** |
| `M_A = 10, M_B = 7` | **True** | **3.4906e-03** | **2.351e-08** |
| `M_A = 10, M_B = 7` | False | **1.6884e+01** | **9.033e+01** |

**Ratios 1237x and 4837x on the observable, 1.5e+06x and 3.8e+09x on lossless
closure.**

**THE BLINDNESS, reproduced as its own measurement.**  On a CONFORMING stack
driven through the mortar (`force_mortar=True`) and scored against the bypass,
with the swap DISABLED:

| conforming fixture | identity, swap ON | identity, swap OFF |
|---|---|---|
| stripe \| stripe, `N = 2`, `M = 5` | 1.82e-15 | **1.92e-14** |
| pillar \| pillar, `N = 3`, `M = 5` | 3.35e-14 | **3.65e-13** |

Both arms pass any sane conforming-identity bar with the swap OFF.  **A
conforming-parity gate CANNOT see the swap -- confirmed, not asserted.**  This
is the single most important design claim in the build and it survives an
independent fixture.

### 4.2 G2 / N6 -- the conforming identity, against a bar derived HERE

`v3_mortar.py ident`.  `cond_2(G)` is measured in this audit off the assembled
`-Rmat`, and the bar is `10 eps cond_2(G)`.

| fixture | `M` | worst rel | `cond_2(G)` measured here | bar | inside by |
|---|---|---|---|---|---|
| stripe \| stripe `N=2` normal | 5 | 3.018e-15 | 3.0594e+03 | 6.793e-12 | 2251x |
| pillar \| pillar `N=3` conical | 6 | 1.582e-14 | 7.2271e+03 | 1.605e-11 | 1014x |
| stripe \| uniform \| pillar `N=2` | 5 | 3.377e-14 | 3.0594e+03 | 6.793e-12 | 201x |
| pillar \| pillar `N=2` | 7 | 9.088e-14 | 1.8426e+04 | 4.091e-11 | 450x |
| **NON-UNIFORM walls `0.2371, 0.6183`, 3 layers** | 5 | 8.610e-15 | 5.3189e+03 | 1.181e-11 | 1372x |
| **NON-UNIFORM walls `0.1409, 0.8817`, 2 layers** | 7 | 5.331e-14 | 3.9994e+05 | 8.881e-10 | 16659x |

My `cond_2(G)` readings at `N = 2, M = 5` (3.0594e+03) and `N = 3, M = 6`
(7.2271e+03) reproduce the build's table to five significant digits, so the
DERIVED bar is itself reproducible.  **G2 and N6 CONFIRMED, 6/6.**  Note the
`cond` growth is not only in `M`: the strongly skewed non-uniform wall set
`0.1409, 0.8817` reads 3.9994e+05 at `M = 7` against 1.8426e+04 for the
uniform `N = 2` grid at the same `M` -- a 22x jump from the wall positions
alone, which is the first hint of the mechanism S6.3 measures.

### 4.3 G4 -- the mortar's own error, isolated on a TRANSPARENT split

`v3_mortar.py nonconf`.  One uniform `n = 2` slab, 0.31 thick, split into two
sub-layers on DIFFERENT grids; the interface is physically absent, so the exact
answer is the analytic Fresnel slab and nothing but the mortar can move it.
`|R - R_exact|` at `M = 7`, s-polarisation, sub/superstrate 1.5 / 1.0:

| `theta` | (2,2) conf | (2,4) nested | (2,3) NON-conf | (3,4) NON-conf |
|---|---|---|---|---|
| 0.00 | 2.27e-14 | 1.42e-15 | 6.80e-15 | 8.94e-15 |
| 0.26 | 4.50e-14 | 2.50e-15 | 6.47e-15 | 1.96e-14 |

**A non-conforming mortar interface is not measurably worse than a conforming
one.  CONFIRMED** -- which is what licenses reading every other gap as
resolution.

### 4.4 Nested and non-conforming patterned pairs vs the common refinement

Same script.  The reference is the UNION grid at `M = 4` on the common
refinement, so it is itself unconverged and the residual FLOORS on it -- stated
rather than hidden:

| pair | `M_A, M_B` | err vs union(M=4) | closure |
|---|---|---|---|
| nested (2,4) | 5,5 / 7,5 / 9,7 | 3.25e-02 / 2.01e-02 / 1.88e-02 | 9.09e-05 / 5.91e-07 / **6.42e-10** |
| non-conf (2,3) | 5,5 / 7,5 / 9,7 | 4.79e-02 / 3.63e-03 / 3.75e-03 | 1.38e-04 / 2.08e-05 / **1.03e-08** |

The observable plateaus at the REFERENCE's own error; the closure does not, and
falls five to six decades.  The clean statement of this gate is S4.7's exact
1-D oracle, where the reference has no floor of its own worth mentioning.

### 4.5 G8 -- the far-field order cap

`v3_mortar.py cap`.  End grids `N = 1, M = 5` (`q = 4`, so
`cap = (4-1)//2 = 1`):

* `n_orders = 7` **RAISES**, and the message names the cap AND both end grids
  ("`q = N (M - 1)` columns per axis -- 4 at the top (N=1, M=5) and 4 at the
  bottom ... `n_orders <= (q - 1) // 2 = 1`");
* by `n_orders`: `1 -> ok`, `2 -> raised`, `3 -> raised` -- the cap is exactly
  `(q-1)//2`, not one either side;
* raising the END layers to `n_modes = 10` (`q = 9`, cap 4) makes
  `n_orders = 4` solve.

**CONFIRMED, including that the cap follows the END layers' own `n_modes`.**

### 4.6 The eig-cache key -- collision attempts

`v3_mortar.py cache`.  Four attempts to make two layers share an entry they
must not:

| attempt | result |
|---|---|
| same `N`, same `M`, DIFFERENT wall positions (`1/3, 2/3` vs `0.2371, 0.6183`) | `A\|B` differs from `A\|A` by **1.901e-01** and from `B\|B` by **1.331e-01** -- a collision would have made one of them 0 |
| same grid, two ANGLES (`tau`) in the SAME stack object, then a FRESH stack at the second angle | reused-stack vs fresh **0.000e+00**, while the two angles differ by 2.573e-01 |
| same cell and grid, DIFFERENT slant | `t = 0.17` vs `t = 0.31` **1.323e-01**; `t = 0.17` vs vertical **1.874e-01** |
| direct key inspection: 2 wall sets x 2 `tau` | **4 distinct `StagGridOps.key()`** (expected 4) |

**CONFIRMED: walls, `tau` and slant are all in the key and none of them
collides.**

### 4.7 G5 -- the 1-D per-order oracle, both polarisations, with the anti-mirror control

`v3_mortar.py oracle1d`.  MY fixture: period 0.83, `wl` 0.55, `theta = 0.27`,
duties 1/2 and 1/3 on `N = 2` and `N = 3` (common refinement 6), thicknesses
0.28 / 0.20.  Exact 1-D oracle at degree 14; its own self-gaps 2.467e-05
(10-12) and 1.126e-05 (12-14).

| `M_A, M_B` | TM (row 0) | TE (row 1) | MIRRORED TE | mirror/direct | closure |
|---|---|---|---|---|---|
| 5, 5 | 1.8657e-01 | 6.2090e-02 | 3.3788e-01 | 5.4x | 3.768e-04 |
| 7, 5 | 1.6272e-01 | 5.1244e-02 | 3.2299e-01 | 6.3x | 4.704e-04 |
| 9, 7 | 6.0908e-03 | 8.0620e-04 | 2.8594e-01 | **355x** | 4.015e-07 |
| 11, 8 | **4.5227e-04** | **1.1280e-04** | 2.8525e-01 | **2529x** | **3.127e-09** |

**Both polarisations converge (2.6 decades on TM, 2.7 on TE) and the
anti-mirror tripwire separates by 355x / 2529x at the resolved rungs.  G5
CONFIRMED**, and the finest rung is still 10x above the oracle's own self-gap,
so it is readable.

### 4.8 The GENERALIZED mortar twin vs `berreman_jones_1d`

`v8_gaps.py berreman` -- the table is in S9.2.

### 4.9 G7 -- two-sided lossless closure, scalar AND Hermitian tensor

`v8_gaps.py gyro` -- the table is in S9.3.

### 4.10 G9 -- the conditioning census

`v8_gaps.py census` -- the table is in S9.6.  The
statement that needs qualifying is the build's design argument, not the
readings: *"the two `solve`s stay `solve`s -- LAPACK `gesv` is backward stable,
so a residual screen on them measures nothing -- and the ONE explicit inverse,
`I + BA`, carries the compounded exposure of both and is where the guard
goes."*  Backward stability says nothing about an exactly singular matrix, and
S6.4 measures `cond_2(lhsH) = 4.895e+10` at a wall separation of 1e-3 of the
period on a grid the API accepts, growing until `np.linalg.solve` raises.
**The census is CONFIRMED in the regime it was taken in and REFUTED as a
general statement about where the exposure lands.**

### 4.11 `retain_internal` / `layer_absorption` on non-conforming grids (O-3)

`v3_mortar.py absorb`.  The honest gate is the CROSS-MACHINERY budget
`sum_i A_i == 1 - sum R - sum T` -- an internal Gram-flux quadrature against
the Rayleigh far field.  Lossy pillar pair, `eps = 5.5 + 0.30i` on `N = 2` over
`4.8 + 0.18i` on `N = 3`, `theta = 0.18`, `phi = 0.35`:

| `M` | `sum_i A_i` | `1 - R - T` | gap |
|---|---|---|---|
| 5 | [0.127999, 0.150651] | [0.128027, 0.150704] | 5.353e-05 |
| 6 | [0.147422, 0.157036] | [0.147423, 0.157039] | 2.842e-06 |
| 7 | [0.150453, 0.157098] | [0.150453, 0.157098] | **3.429e-08** |

and on NON-UNIFORM walls (`0.2371, 0.6183` over `0.4009, 0.8817`) 5.472e-05 at
`M = 5` and 2.825e-05 at `M = 6`.  **Discretisation-limited and converging
(19x then 83x per rung on the uniform-wall pair).  O-3 CONFIRMED.**

### 4.12 MIXED KINDS on three different grids

`v4_slant_mixed.py mixed`.  OUT-OF-PLANE tilted-director tensor on `N = 2`,
over a SCALAR pillar on `N = 3`, over a MAGNETIC (`mu_cell`) layer on `N = 2`
-- three grids, two mortars, on the GENERALIZED cascade -- against the
union-grid (`N = 6`) shared cascade, `theta = 0.23`, `phi = 0.55`:

| per-layer `M_i` | err vs union `M = 5` | closure | wall |
|---|---|---|---|
| (6, 5, 6) | 1.7562e-02 | 5.232e-06 | 1.7 s |
| (8, 6, 8) | 2.5172e-03 | 2.341e-08 | 7.7 s |
| (10, 7, 10) | **7.5427e-04** | **3.649e-11** | 28.4 s |

The union reference's own `M = 4` vs `M = 5` self-gap is **4.730e-03**, so the
finest per-layer rung sits INSIDE the reference's own uncertainty.  **Mixed
kinds through the generalized mortar CONFIRMED.**

## 5. THE SLANTED PER-LAYER GATE (the build's open item, built here)

`v4_slant_mixed.py bypass|oracle|split`, and shipped as
`tests/unit/test_verify_pmm2d_perlayer_slant.py`.

The build doc's last open item reads: *"The per-layer path carries `slant`
through unchanged ... It is exercised only by the shared-path suites plus the
per-layer plumbing; a dedicated per-layer slanted gate is NOT in this build and
is the obvious next measurement."*  It matters because a slanted layer takes
the GENERALIZED cascade, i.e. a DIFFERENT mortar function
(`_interface_smatrix_general_mortar_2d`), and additionally carries the
frame-anchor transmission phase, which is a far-field correction the mortar
knows nothing about.

### 5.1 The bypass, on the generalized cascade

Two slanted layers (`t = tan 22 deg`) at the SAME slant on COINCIDING grids,
`layer_grids='per-layer'` against `'shared'`:

| grid | `M` | sha256 | max move | FORCED mortar vs bypass |
|---|---|---|---|---|
| `N = 2` | 6 | **EQUAL** | **0.00e+00** | 7.033e-15 |
| `N = 4` | 5 | **EQUAL** | **0.00e+00** | 7.034e-15 |

### 5.2 THE GATE -- a slanted grating SPLIT across NON-CONFORMING grids, per order

Fixture: a y-uniform binary grating, period 0.80, `wl` 1.0, depth 0.28, duty
1/2, `n_ridge/n_groove = 2.0/1.0`, superstrate 1.0 / substrate 1.5.  The depth
is split into two layers at the SAME slant, the upper on `N = 2` (its only wall
at `P/2`) and the lower on `N = 4` (walls at `P/4, P/2, 3P/4`) -- genuinely
non-conforming.  Scored PER ORDER (`m = -1, 0, +1`) against the shipped 1-D
inclined-coordinate solver `pmm_efficiency_1d_slanted` at degree 22, both
polarisations.  `M_A = 7`, `M_B = 5`.

| slant | mount | TE (+tan) | TM (+tan) | closure | TE (-tan) | ratio | its closure |
|---|---|---|---|---|---|---|---|
| 0 deg | normal | 1.815e-06 | 8.947e-04 | 9.54e-07 | -- | -- | -- |
| 0 deg | 22 deg | 3.343e-06 | 9.708e-05 | 6.09e-09 | -- | -- | -- |
| 15 deg | normal | **7.043e-07** | 6.854e-04 | 9.13e-07 | **9.215e-02** | **130843x** | 9.13e-07 |
| 15 deg | 22 deg | **1.983e-06** | 1.807e-04 | 2.07e-08 | **1.888e-02** | **9523x** | 2.65e-08 |
| 28 deg | normal | **5.611e-06** | 6.815e-04 | 8.97e-07 | **2.106e-01** | **37526x** | 8.97e-07 |
| 28 deg | 22 deg | **2.471e-06** | 2.018e-04 | 3.52e-08 | **3.123e-02** | **12643x** | 3.58e-08 |

Three things at once:

1. **the slanted arm TRACKS the vertical control** (0.39x to 3.1x of it), so
   the mortar adds nothing to the slanted answer;
2. **the SIGN is pinned two-sided, per order**, by four to five decades;
3. **the LOSSLESS TRAP is reproduced through the mortar** -- the wrong-sign arm
   closes energy to the SAME digits (9.13e-07 vs 9.13e-07 at 15 deg normal;
   3.52e-08 vs 3.58e-08 at 28 deg / 22 deg), so an energy check carries ZERO
   information about the sign here.

And it CONVERGES, so the agreement is not a coincidence at one rung:

| `M_A, M_B` | TE vs the 1-D slanted oracle | closure |
|---|---|---|
| 5, 4 | 7.698e-04 | 1.70e-05 |
| 7, 5 | 2.471e-06 | 3.52e-08 |
| 9, 6 | **1.190e-07** | **4.08e-10** |

**The build's last open item is CLOSED with a dedicated gate.**

### 5.3 The layer-split "identity", and why it is NOT one

One slanted layer of depth `d` against two of `d/2` on `N = 2 | N = 4` reads
2.369e-04 (`M = 6`) and 2.258e-04 (`M = 7`) -- flat, which looks like a floor.
It is not a defect and it is not an identity: the two arms are NOT at matched
resolution (`q = 2(M-1)` for the single layer against `q_A = 2(M-1)`,
`q_B = 4M` for the split), and the gap is the COARSER arm's own error.
Attributed by measurement:

| | TE vs the 1-D oracle | TM vs the 1-D oracle |
|---|---|---|
| one layer, `N = 2`, `M = 6` | 1.401e-04 | 2.649e-04 |
| split `N2\|N4`, `M = 6\|7` | 9.665e-05 | 7.944e-05 |
| one layer, `N = 2`, `M = 7` | 8.979e-06 | 2.534e-04 |
| split `N2\|N4`, `M = 7\|8` | 2.676e-06 | 6.121e-05 |

The `max` over all orders and both rows is dominated by the TM row, where the
single-layer arm is the worse of the two; restricted to the TE row and `n = 0`
orders the split difference is 5.588e-05 (`M = 6`) and **6.302e-06** (`M = 7`),
i.e. converging.  **BOUNDED, not a defect** -- and the shipped test therefore
asserts the ORACLE comparison (S5.2), not a same-device "identity" between two
differently-resolved arms.

---

## 6. TAPERS, AND THE SLIVER QUESTION ON THE MORTAR ROUTE (task 5)

### 6.1 `add_tapered_pillar` on the pure engine

`v6_taper_sliver.py taper` and `v2_oracles.py hybrid`.

**Against the hybrid's OWN staircase at the IDENTICAL slices.**  MY fixture:
period 1.2 um, `wl` 0.85, `eps_pillar = 9.0` / `eps_host = 2.25`,
`theta = 0.15`, `phi = 0.35`, 4 midpoint slices interpolating from
`[0.1873 P, 0.7241 P]` to `[0.2917 P, 0.6109 P]`.

| arm | setting | `R(0,0)` | closure | wall |
|---|---|---|---|---|
| hybrid | degree 7, `n_orders` 7 | 0.039154 | 6.991e-03 | 5.0 s |
| hybrid | degree 9 | 0.037952 -- deg-7-vs-9 self-gap **1.202e-03** | 3.999e-03 | 4.5 s |
| hybrid | degree 11 | 0.039071 -- deg-9-vs-11 self-gap **1.119e-03** | 2.608e-03 | 4.7 s |
| **pure NU mortar** | `M = 4` (`q = 9`) | 0.001706 | 3.223e-03 | 0.6 s |
| **pure NU mortar** | `M = 5` (`q = 12`) | 0.031641 | 4.112e-04 | 4.3 s |
| **pure NU mortar** | `M = 6` (`q = 15`) | 0.030042 | 8.651e-06 | 21.6 s |
| **pure NU mortar** | `M = 7` (`q = 18`) | **0.039642** | **2.556e-06** | 63.6 s |

`|pure - hybrid(degree 11)|` runs 3.736e-02 / 7.430e-03 / 9.029e-03 /
**5.708e-04** over `M = 4..7`.  **At `M = 7` the pure arm sits HALF the
hybrid's own degree self-gap away from it (5.71e-04 against 1.12e-03), with a
lossless closure 1020x tighter than the oracle it is scored against
(2.556e-06 vs 2.608e-03).**  So the residual is the oracle's floor, exactly as
the build states, and the arm that is converging is the pure one.

**Against the hybrid on a STRAIGHT arbitrary-wall pillar, per order** -- and
this is where an oracle trap sits that the build doc does not name.  Fixture:
walls `(0.2371, 0.6183)` in x and `(0.3117, 0.7402)` in y, `theta = 0.18`,
`phi = 0.35`.  Scored naively against the hybrid at `n_orders = 5`, the pure
arm PLATEAUS at 2.0e-02 while its own closure falls to 4.05e-09 -- which looks
like a disagreement between engines.  It is not.  The hybrid's `n_orders`
FLOOR, measured by sweeping it at degree 9:

| hybrid `n_orders` | 3 | 5 | 7 | 9 | 11 | 13 |
|---|---|---|---|---|---|---|
| `R(0,0)` | 0.058238 | 0.055145 | 0.053293 | 0.052006 | 0.050777 | 0.050822 |
| closure | 1.26e-02 | 5.05e-03 | 2.74e-03 | 3.36e-04 | 3.35e-04 | 3.34e-05 |

-- a 4.3e-03 move between `n_orders` 5 and 13, which a DEGREE self-gap cannot
see at all.  Against a hybrid converged in BOTH knobs (degree 9 -> 11 at
`n_orders` 13: 1.751e-03; `n_orders` 11 -> 13 at degree 9: 1.791e-03, so the
oracle's own uncertainty is ~1.8e-03):

| pure `M` | per-order max \|pure - hybrid(deg 9, `n_orders` 13)\| | pure closure |
|---|---|---|
| 4 | 4.2884e-03 | 3.064e-04 |
| 5 | 2.8993e-03 | 2.223e-05 |
| 6 | 2.6313e-03 | 3.565e-07 |
| 7 | **2.5514e-03** | **4.051e-09** |

**2.55e-03 against an oracle uncertainty of 1.8e-03 -- agreement at 1.4x the
oracle's own floor.  CONFIRMED**, with the caveat that a hybrid oracle must be
converged in `n_orders` as well as in degree before it can bound a pure-engine
claim at the 1e-3 level.  (`pmm_efficiency_2d_cell` on the pixel-expressible
wall set `3/8, 5/8` shows the same plateau, 1.51e-02 -> 1.57e-02 over
`M = 4..7` at `n_orders = 5`, for the same reason.)

**The cost statement, which is the point of the feature.**  The 4-slice
taper's walls are 0.1873 / 0.7241 / ... / 0.2917 / 0.6109 of the period; the
coarsest common lattice that names them all is `N = 40`, i.e. `q = 40 (M-1)` --
**`q = 80` and a `2 q^2 = 12800`-dimension region eig (a 2.6 GB matrix) at the
minimum `M = 3`**.  On per-layer non-uniform grids each slice is THREE
segments, `q = 3(M-1) = 18` at `M = 7`, eig dimension **648**.  That is a
factor 20 in `q` and 400 in eigenproblem dimension, and it is why the feature
exists.  (A shallower 3-slice taper whose walls DO land on `N = 20` is in the
probe as the shared-lattice control; its union solve is a 3200-dimension eig
per slice and did not finish inside this audit's budget, which is itself the
measurement.)

### 6.2 The sliver route the FIX doc calls SAFE -- CONFIRMED, with numbers

`v6_taper_sliver.py cross`.  `FIX_PMMSTACK_SLIVER_WALLS_2026_09_11.md` S7 says
the mortar route is safe because "the mortar keeps each layer on its OWN grid
and never forms the union".  Re-measured on the fix doc's own fixture class:
two ADJACENT per-layer layers, layer 0's walls at `(0.27865, 0.62505)` of a 1.2
um period and layer 1's opened by `delta`, `wl` 0.85, `theta = 0.15`,
`phi = 0.35`, `eps = 9.0 / 2.25`.  The `delta -> 0` limit is an EXACT reference
because the structure is continuous in `delta`.

| `delta` | `R(0,0)` `M=6` | `\|R00(d) - R00(0)\|` | closure `M=5` / `M=6` | worst mortar `rcond` | `cond(C_bb^x)` |
|---|---|---|---|---|---|
| 1.0e-02 | 0.157322409 | 1.87e-03 | 5.87e-06 / 4.37e-07 | 5.73e-04 | 1.04e+02 |
| 2.6e-03 | 0.155825764 | 3.77e-04 | 6.74e-06 / 3.26e-07 | 1.18e-02 | 9.51e+01 |
| 1.0e-03 | 0.155565387 | 1.17e-04 | 7.54e-06 / 3.55e-07 | 6.53e-02 | 9.44e+01 |
| 3.0e-04 | 0.155479247 | 3.07e-05 | 7.91e-06 / 3.66e-07 | 2.93e-01 | 9.43e+01 |
| 1.0e-04 | 0.155458353 | 9.79e-06 | 8.02e-06 / 3.70e-07 | 6.11e-01 | 9.43e+01 |
| 3.0e-05 | 0.155451454 | 2.89e-06 | 8.05e-06 / 3.71e-07 | 8.52e-01 | 9.43e+01 |
| 1.0e-05 | 0.155449522 | 9.59e-07 | 8.06e-06 / 3.71e-07 | 9.47e-01 | 9.43e+01 |
| 3.0e-06 | 0.155448851 | 2.87e-07 | 8.07e-06 / 3.71e-07 | 9.83e-01 | 9.43e+01 |
| 1.0e-06 | 0.155448659 | 9.57e-08 | 8.07e-06 / 3.71e-07 | 9.94e-01 | 9.43e+01 |
| **0** | 0.155448563 | 0 | 8.07e-06 / 3.72e-07 | 1.00e+00 | 9.43e+01 |

**Every column behaves.**  The deviation from the `delta = 0` limit is EXACTLY
linear in `delta` over five decades (1.87e-03 / 3.77e-04 / 1.17e-04 /
3.07e-05 / 9.79e-06 / 2.89e-06 / 9.59e-07 / 2.87e-07 / 9.57e-08 -- a clean
factor of ~3.3 and ~10 alternating with the `delta` ladder), i.e. the answer
TRACKS THE PHYSICS all the way down.  The closure is FLAT.  The cross-mass
conditioning is CONSTANT at 9.43e+01 -- the 1-D mechanism's `1/w^2` is absent,
because the near-coincident walls appear only in the cross-mass INTEGRATION
mesh, never as spectral elements.  The mortar interface `rcond` IMPROVES toward
1.0 as the two grids converge.

**CONFIRMED: the FIX doc's S7 claim is right, and this is the measurement that
supports it on the SHIPPED library rather than on the F5 prototype.**

### 6.3 THE NEW FINDING -- a sliver INSIDE ONE layer's own grid

Neither the build doc nor the fix doc considers this case: the non-uniform
segment feature lets ONE layer put two of its OWN walls arbitrarily close.  It
is reachable through the public API three ways -- `x_walls=` directly,
`add_tapered_pillars` with two features whose edges nearly meet, and
`add_tapered_pillar` on a taper that closes (the midpoint rule's narrowest
sampled width is `w_bottom / (2 n_slices)`, so `n_slices = 64` on a pillar
closing from half the period already reaches `3.9e-03`, and `n_slices = 1000`
reaches `2.5e-04`).

**Step 1 -- the mechanism IS present in this basis.**  `v6_taper_sliver.py
intra` measures the region spectrum of ONE layer whose middle segment is
`delta` wide (`|lam|` is the modal `gamma/k0`, so a PHYSICAL mode is bounded by
`sqrt(eps_max) = 3.0`):

| `delta` | `k0 J_min` | `\|lam\|max` `M=4` | `M=6` | `M=8` |
|---|---|---|---|---|
| 3e-01 | 1.331e+00 | 5.784e+00 | 1.308e+01 | 2.276e+01 |
| 1e-02 | 4.435e-02 | 1.097e+02 | 2.379e+02 | 4.027e+02 |
| 1e-04 | 4.435e-04 | 1.034e+04 | 2.259e+04 | 3.810e+04 |
| 1e-05 | 4.435e-05 | **1.033e+05** | **2.258e+05** | **3.808e+05** |

`|lam|max` is exactly proportional to `1/(k0 J_min)` -- the SAME `1/w` spurious
spectrum the 1-D nodal SEM has, because the staggered stiffness carries the
same `1/J_n`.

**Step 2 -- WITHOUT a mortar it is harmless.**  A single such layer between the
two half-spaces (which ride its own grid, so both interfaces are the plain
square modal match) is EXACTLY `delta`-independent when the medium is
continuous across the two walls:

| arm | `M = 6` | `M = 8` |
|---|---|---|
| `R(0,0)` at every `delta` from 3e-1 to 1e-6 | 0.060112738 (spread **4.1e-13**) | 0.060112738 (spread **6.7e-12**) |
| closure | 2.5e-12 or better | 4.2e-11 or better |

The spurious modes are enormous and therefore evanescent to machine zero, and
the interface match sees the SAME spurious modes on both sides.

**Step 3 -- WITH a mortar it is a SILENT WRONG ANSWER.**  Put that layer
between two other patterned layers on DIFFERENT grids, so both its interfaces
are mortars.  The middle layer is ALL HOST, i.e. the DEVICE does not depend on
`delta` at all -- the "walls" are element boundaries in a continuous medium:

| `delta` | `R(0,0)` `M = 6` | `\|R00(d) - R00(3e-1)\|` | closure | worst `rcond` |
|---|---|---|---|---|
| 3e-01 | 0.094877628 | 0 | 1.34e-05 | 1.52e-05 |
| 1e-01 | 0.093709718 | 1.168e-03 | 1.43e-05 | 5.89e-05 |
| 1e-02 | 0.097810277 | 2.933e-03 | 1.42e-05 | 2.67e-05 |
| 1e-03 | 0.101122423 | 6.245e-03 | 1.39e-05 | 4.95e-06 |
| 1e-04 | 0.101182207 | 6.305e-03 | 1.33e-05 | 9.55e-06 |
| 1e-05 | 0.101956055 | 7.078e-03 | 1.29e-05 | 2.40e-06 |
| 1e-06 | **0.102676579** | **7.799e-03** | **1.31e-05** | 2.53e-07 |
| 1e-07 | **`LinAlgError: Singular matrix`** | -- | -- | -- |

and at `M = 8`, where the answer is no longer even monotone in `delta`:

| `delta` | 3e-01 | 1e-01 | 1e-02 | 1e-03 | 1e-04 | 1e-05 | 1e-06 | 1e-07 |
|---|---|---|---|---|---|---|---|---|
| `R(0,0)` | 0.096219 | 0.095245 | 0.093922 | 0.094761 | **0.090000** | 0.094118 | 0.091585 | RAISES |
| closure | 1.92e-08 | 1.97e-08 | 2.21e-08 | 2.27e-08 | **2.28e-08** | 2.55e-08 | 1.98e-08 | -- |

**Three things make this a P1.**

1. **It is WRONG.**  The physical answer cannot depend on `delta` at all here
   (the medium is continuous); the legitimate spread from moving element
   boundaries inside a uniform medium is measured at 1.17e-03 (`M = 6`,
   `delta` 3e-1 -> 1e-1) and 9.7e-04 (`M = 8`), while the excursion reaches
   **7.8e-03 in absolute `R(0,0)` on a quantity of ~0.095, i.e. 8 %** -- and
   it does not settle as `delta` falls.
2. **It is ENERGY-INVISIBLE.**  The lossless closure is PINNED -- 1.3e-05 at
   `M = 6` and 2.3e-08 at `M = 8` at every `delta` -- so `_warn_stag_closure`
   never fires and the 1-D sliver fix's `R+T` screen (the whole basis of the
   shipped `PMM_SLIVER_GUARD`) has NOTHING TO SEE.  This is the exact shape
   `TESTING_STANDARDS.md` calls the most dangerous: it reads as authoritative
   and it passes.
3. **The attribution is clean.**  Same layer, same grid, same `delta`, NO
   mortar -> exact to 4e-13.  WITH a mortar -> 7.8e-03.  The corruption is the
   mortar's cross-grid projection of a mode set that contains `1e+05`-scale
   spurious wavenumbers.

The corresponding conditioning, instrumented directly on
`_interface_smatrix_mortar_2d`'s two UNGUARDED `solve`s at `delta = 1e-3`,
`M = 6`:

| operator | `s_min` | `s_max` | `cond_2` |
|---|---|---|---|
| `lhsE = MassE_B W_B` | 4.926e-09 | 2.895e-01 | **5.878e+07** |
| `lhsH = MassH_A V_A` | 2.095e-11 | 1.025e+00 | **4.895e+10** |

against 2.07e+03 / 8.45e+04 on the same pair at `delta = 0`.

### 6.4 ... and the failure mode below `delta ~ 1e-7` is a BARE `LinAlgError`

Two distinct unguarded `np.linalg.solve` calls throw:

| path | file:line | call |
|---|---|---|
| MORTAR interface (NEW on this branch) | `pmm/_core.py:5270` | `B = np.linalg.solve(lhsH, rhsH)` in `_interface_smatrix_mortar_2d` |
| plain square interface (pre-existing, newly REACHABLE) | `pmm/_core.py:1850` | `a = np.linalg.solve(Wb, Wa)` in `_interface_smatrix` |

Both surface as `numpy.linalg.LinAlgError: Singular matrix` with no message,
no hint and no naming of the offending grid -- in a library whose whole
conditioning story (`_guarded_solve`, `_guarded_lstsq`, `_guarded_inverse`,
the M1 census) is built on refusing loudly and explaining.  The build doc's
design argument for guarding only `I + BA` -- *"LAPACK `gesv` is backward
stable, so a residual screen on them measures nothing"* -- is true and
irrelevant: backward stability says nothing about an exactly singular matrix,
and S6.3 shows the two unguarded operators are the ones that blow up FIRST
(`cond` 4.9e+10 while `I + BA`'s guarded `rcond` still reads 2.5e-07).

---

## 7. THE EQUAL-DOF CLAIMS (task 4)

`v5_equal_dof.py`.  The fixtures ARE the build's (this is a verification of its
numbers, so it has to be); the REFERENCES are rebuilt here, one rung finer than
the build's, and the LOSING rungs are run rather than avoided.

### 7.1 The stripe pair (G6)

Exact 1-D `PMMStack` reference at **degree 16** (the build used 14); its own
self-gaps 8.598e-06 (12-14) and **4.526e-06** (14-16).  Per axis
`q_union(M) = 6(M-1) = q_A(3M-2) = q_B(2M-1)`, so every region eigenproblem in
both arms is `2 q^2`.

| `q` | eig dim | union `M` | union err | mortar `M_A, M_B` | mortar err | **RATIO** | closure union -> mortar | wall union / mortar |
|---|---|---|---|---|---|---|---|---|
| 12 | 288 | 3 | 1.5102e-01 | 7, 5 | 2.6713e-02 | **0.1769** | 8.51e-04 -> 7.27e-05 | 2.0 / 2.5 s |
| 18 | 648 | 4 | 5.0739e-03 | 10, 7 | 2.9781e-04 | **0.0587** | 2.27e-05 -> 6.57e-10 | 24.4 / 27.5 s |
| 24 | 1152 | 5 | 7.8455e-04 | 13, 9 | 6.1296e-06 | 0.0078 | 1.58e-07 -> 2.86e-12 | 141.8 / 161.9 s |
| 30 | 1800 | 6 | 9.5536e-06 | 16, 11 | 3.0258e-06 | 0.3167 | 2.73e-10 -> 2.26e-12 | 494.5 / 649.7 s |

**The build's two headline ratios reproduce EXACTLY: 0.1769 at `q = 12` and
0.0587 at `q = 18`, against the build's 0.176877 and 0.058687 -- with a
DIFFERENT reference (degree 16 rather than 14).  CONFIRMED.**

**The regime where the mortar LOSES is documented in the EXPERIMENT and is NOT
reproducible on the shipped library.**  The experiment (its own prototype
implementation, S7.3) reports 0.13x at `q = 18`, 0.14x at `q = 24` and
**1.48x at `q = 30`**; the shipped library reads 0.0587 / 0.0078 / 0.3167.  The
honest reading is the one the experiment itself gives in words: **at `q >= 24`
neither arm is resolvable against this oracle** -- the mortar's 6.13e-06 at
`q = 24` and 3.03e-06 at `q = 30` are AT or BELOW the reference's own 4.53e-06
self-gap, so those two ratios are upper bounds, not measurements.  What
survives is:

* at the two READABLE rungs the mortar is 5.7x and 17x more accurate at
  identical eigenproblem sizes;
* its lossless closure is 11.7x / 34,500x / 55,300x / 121x tighter and NEVER
  worse;
* the wall-time cost is 1.25x / 1.13x / 1.14x / 1.31x.

**"NOT WORSE" is CONFIRMED and is a CONSERVATIVE statement of what the library
does.  The specific losing rung is a prototype number and should not be quoted
as a library measurement.**

### 7.2 The 3-slice staircase

Slice widths 1/2, 1/3, 1/6 (so per-slice `N = 2, 3, 6`, union `N = 6`),
y-uniform, exact 1-D reference at degree 16 (self-gaps 4.901e-06 and
2.616e-06):

| `q` | union `M` | union err | per-layer `M_i` | per-layer err | **RATIO** | closure union -> per-layer | wall |
|---|---|---|---|---|---|---|---|
| 12 | 3 | 1.2943e-01 | 7, 5, 3 | 8.2060e-02 | **0.6340** | 4.76e-03 -> 4.46e-03 | 2.9 / 2.8 s |
| 18 | 4 | 4.7914e-03 | 10, 7, 4 | 2.4122e-03 | **0.5035** | 5.15e-05 -> 5.13e-05 | 32.9 / 37.7 s |

**Both of the build's staircase ratios (0.634 and 0.504) reproduce to three
digits.  CONFIRMED.**

**The DOF FLOOR, checked directly**: `n_modes = 2` RAISES, so the smallest `q`
a lattice can carry is `2 N` -- union `N = 6` cannot be run below `q = 12`, the
sibling union `N = 12` (widths 1/2, 1/3, 1/4) cannot be run below `q = 24`, and
a per-layer `N = 2` slice reaches `q = 4`.  **CONFIRMED as a floor, not a
tuning choice.**

### 7.3 The corner-dominated 2-D pillar pair (F2)

No exact oracle exists, so the reference is the union grid at the top of its
OWN ladder and the ladder is reported:

| union `M` | `q` | `R(0,0)` | rel gap vs previous | closure | wall |
|---|---|---|---|---|---|
| 3 | 12 | 0.109330399 | -- | 9.25e-04 | 2.4 s |
| 4 | 18 | 0.141001073 | 2.246e-01 | 5.59e-06 | 30.7 s |
| 5 | 24 | 0.143453913 | 1.710e-02 | 2.69e-08 | 198.6 s |
| **6** | **30** | **0.143418877** | **2.443e-04** | 1.00e-10 | 594.0 s |

Reference = union `M = 6`; its own uncertainty **2.443e-04** relative.

| `q` | eig dim | union err (rel) | mortar `M_A, M_B` | mortar err (rel) | **RATIO** | closure union -> mortar | READABLE? |
|---|---|---|---|---|---|---|---|
| 12 | 288 | 2.377e-01 | 7, 5 | **5.791e-02** | **0.244** | 9.25e-04 -> 2.06e-05 (45x) | yes |
| 18 | 648 | 1.686e-02 | 10, 7 | **5.337e-03** | **0.317** | 5.59e-06 -> 1.09e-08 (513x) | yes |
| 24 | 1152 | 2.443e-04 | 13, 9 | 3.692e-03 | 15.11 | 2.69e-08 -> 2.11e-12 (12750x) | **NO** -- the union's error at `q = 24` IS the reference's own uncertainty, by construction |

**The experiment's F2 sign and its regime qualification both CONFIRMED**: the
mortar is 4.1x and 3.2x more accurate at `q = 12` and `q = 18` (F2 read 1.9x
and 5.5x on its own fixture), the advantage is gone somewhere between `q = 18`
and `q = 24`, and the `q = 24` point is NOT resolvable against this reference
in either study.  **The closure advantage is 45x / 513x / 12,750x and does not
decay** -- F2's "the closure half is the durable one" reproduces.

## 8. DURABILITY AUDIT OF THE TWO NEW TEST FILES (task 7)

Both files were read line by line, every numeric constant traced to a stated
origin, and every stated MEASURED value re-measured on this build.  47 tests
pass on Windows and on WSL.

### 8.1 What is sound

* Every bar in both files is either DERIVED at runtime (`G2`/`N6`'s
  `10 eps cond_2(G)`, `N3`'s triangle sum, the taper's oracle self-gap), a
  DECISION about a ladder's shape, or a magnitude bar with its gap stated.
  No test pins a converged reading.
* Both files cap `OMP/OPENBLAS/MKL_NUM_THREADS` before importing numpy.
* The two fail-before arms (`G3`'s `PMM2D_MORTAR_H_SWAP`, `N7`'s four sites)
  restore state in `finally` and assert RATIOS, not readings.
* `N7`'s uniform arm is a genuine byte claim and it holds at all four sites
  (independently re-measured, S2.3).
* No `pytest.skip` on a resource check anywhere in either file.

### 8.2 What this audit changed (test-only; committed)

| file | test | finding | change |
|---|---|---|---|
| nonuniform | `test_n2_...ulp_close...` | **the stated MEASURED values do not reproduce.**  Docstring: "exactly 0.0 at `d = 1.2` for `N = 2, 3, 4, 6`; 1.96e-16 at `d = 0.9, N = 4`".  RE-MEASURED here: 0.0 only at `(1.2, 2)`; `(1.2, 4)` 3.084e-16, `(1.2, 3)` 1.156e-16, `(1.2, 6)` 4.626e-16, `(0.9, 4)` 2.056e-16, `(0.9, 3)` 3.084e-16 -- and `np.all(np.diff(linspace) == d/N)` is TRUE only for `N = 2`, so "exactly 0 at `d = 1.2`" is false by construction for `N = 3, 4, 6`.  Identical for `tau = 1` and Bloch `tau`, and across `mass<B\|B>` / `stiff`.  The CONCLUSION (keep the integer path distinct) is right; the numbers were not.  `TESTING_STANDARDS`: *"right-conclusion-wrong-numbers is the most dangerous shape"* | docstring table replaced with the re-measurement, dated |
| nonuniform | `test_n3_...triangle_bar` | **the assertion is a TAUTOLOGY.**  `gap <= en + eu` is a theorem for three max-norms over one order set, so the test cannot fail whatever the library does.  Read as a margin it is also misleading: on this fixture the `M = 5` rung sits inside the bar by a factor **1.0002** | triangle line kept as a REPORTED quantity with the theorem stated; three real endpoint decisions added (`gap` 51x, `err NU` 35x, `err uniform` 1183x over `M = 4 -> 7`, barred at 10x) plus a readability check against the oracle |
| mortar | `test_g4_transparent_...` | the docstring's MEASURED table is the build doc's `M = 7` fixture while the test runs `M = 6`, and it quotes a `(2,4)` arm the test does not run.  RE-MEASURED at the test's own setting: 2.278e-13 / 1.641e-13 / 1.474e-13 / 3.197e-14, closures to 3.12e-13 -- so the docstring's "closure at or below 8.2e-14" is also false for this fixture.  The 1e-10 bar is unaffected (440x) | docstring table replaced with the test's own re-measurement |
| mortar | `test_g4_generalized_...` | quotes a `(3,4)` reading for a pair the parametrization does not run | annotated; the verification's own three-mount, three-pair re-run added |
| mortar | `test_g9_conditioning_census` | "MEASURED over 9 solves on three non-conforming configurations at `M = 4, 5, 6`" -- the test runs **4 solves on two configurations at `M = 4, 5`** | corrected, and the design argument's limits added (S6.3) |
| mortar | `test_cross_mass_kron_...` | "across four grid pairs" -- the test runs three | corrected; the verification's five-pair table added |
| mortar | `test_taper_agrees_with_the_hybrid_staircase...` | **assertion 1 is near-vacuous**: `abs(r6-r5) < 0.5*max(abs(r5-r9),1e-12) OR abs(r6-r5) < 5e-2`.  MEASURED on its own fixture `\|r6-r5\| = 2.797e-02` against `0.5\|r5-r9\| = 1.442e-02`, so the first clause is FALSE and the test passes only through the escape hatch -- the convergence it claims to assert is not asserted | third rung `M = 6` added; the claim is now `\|r(M6)-r(M5)\| < 0.5 \|r(M5)-r(M4)\|`, MEASURED 9.356e-04 against 1.399e-02, i.e. 15x inside.  Cost +16 s |
| mortar | `test_g2_forced_mortar...` | `_gram_cond` is evaluated at NORMAL incidence for oblique/conical rows, unstated | measured the angle dependence (`cond_2` moves at most 8.7 % over `alpha0` in `(0,0)..(1.7,-0.9)`, and normal is the LARGER reading on two of three grids) and stated it as a deliberate conservative proxy |

The rest of both docstrings' MEASURED values that this audit re-measured
DO reproduce: the taper test's `0.026323 / 0.026838 / 5.15e-04 / 0.055159 /
0.027188 / 8.65e-04 / 13x` are correct to every digit printed, the `G2`
`cond_2(G)` table reproduces to five significant digits, and `G3`'s ratio
claim, `G5`'s ladder shape, `G6`'s ratios and the `O-3` budget all reproduce.

### 8.3 Bars whose margin is thin, reported and NOT changed

These are sub-decade but the quantity is discretisation-limited (so the
cross-build spread is small) and I do not have a second build's READINGS, only
its pass/fail.  Changing a bar I cannot bracket on two builds would itself
violate the standard, so they are reported:

| test | assertion | measured margin |
|---|---|---|
| `test_g5_...` | `errs[2] < 0.5 * errs[1]` | measured ratio 0.39 -> **1.3x** inside |
| `test_g5_...` | `mirs[1] > 4.0 * errs[1]` | measured 8.3x -> **2.1x** inside |
| `test_g7_...` | `clos[1] < 0.3 * clos[0]` | measured 0.19 on the shipped fixture -> 1.6x inside; on a NEARBY fixture of my own the same ladder reads **0.293**, i.e. it would clear the bar by 1.02x.  The bar is measuring a fixture, not a property; the durable restatement is "the last rung is a decade inside the first" (measured 184x on my fixture, 58x on the shipped one) |
| `test_staircase_...` | `ep < 1.2 * eu` | measured 0.634 -> 1.9x inside |

### 8.4 Re-timings, both builds

Windows 11 / CPython 3.14.6 / numpy 2.4.4 and WSL2 Ubuntu / CPython 3.12.3 /
numpy 2.4.6, `OMP/OPENBLAS/MKL_NUM_THREADS = 1`, **on a box shared with three
other agents' worktrees throughout** -- so these are upper bounds:

| run | WIN | WSL |
|---|---|---|
| the two files AS SHIPPED | **47 passed, 392.4 s** | (run inside the combined row below) |
| the two files + `test_verify_pmm2d_perlayer_slant.py`, AS SHIPPED | -- | **51 passed, 551.1 s** |
| the same three AFTER this audit's edits | **51 passed, 395.5 s** | -- |
| slowest, WIN as shipped | 39.8 s `test_g3_h_row_v1_v2_swap_is_load_bearing` | 43.3 s (same test) |
| slowest, WIN after the edits | 31.9 s `test_slanted_per_layer_...bit_exact_vs_shared`, then 28.9 s `test_taper_agrees_...` (was 17.5 s; the third rung) | 46.8 s `test_slanted_per_layer_...bit_exact_vs_shared` |

The build doc reports 286.9 s for the pair on WIN and 273 s on WSL; the 1.4x /
2.0x excess here is contention, not a change (`test_g3` reads 39.8 s here
against the build's 32.9 s on the same box under lighter load, and every other test scales similarly).
`.test_durations` is re-spliced with the post-edit timings.

---

## 9. THE REMAINING GATES, RE-MEASURED (`v8_gaps.py`)

### 9.1 G1 -- the identical-grid bypass, VERTICAL

| stack | sha256 R / T / Jones | max move |
|---|---|---|
| pillar \| uniform \| stripe, `N = 2`, `M = 5` | **EQUAL** | **0.00e+00** |
| pillar \| uniform \| stripe, `N = 3`, `M = 6` | **EQUAL** | **0.00e+00** |

(and the slanted twin in S5.1, also exactly 0.)  **G1 CONFIRMED.**

### 9.2 G4 (out-of-plane) -- the GENERALIZED mortar twin vs `berreman_jones_1d`

One uniform tilted-director LC slab (`n_o = 1.48`, `n_e = 1.72`, tilt 41 deg,
azimuth 17 deg, 0.33 thick) SPLIT across grids; the interface is physically
absent, so `berreman_jones_1d` is exact.  `max|J - J_berreman|`:

| mount | `M` | (2,2) conf | (2,3) NON-conf | (3,4) NON-conf |
|---|---|---|---|---|
| normal | 5 | 8.34e-15 | 5.65e-15 | 1.98e-14 |
| normal | 7 | 5.30e-14 | 4.88e-14 | 6.28e-14 |
| 25 deg, phi 40 | 5 | 3.47e-08 | 4.51e-07 | 1.83e-08 |
| 25 deg, phi 40 | 7 | **7.33e-13** | **9.99e-12** | **8.45e-14** |
| 35 deg, phi 70 | 5 | 3.50e-06 | 2.52e-05 | 1.03e-06 |
| 35 deg, phi 70 | 7 | **6.04e-10** | **4.21e-09** | **3.06e-11** |

**CONFIRMED, and two-sided in the sense the shipped bar needs**: the `M = 7`
readings are 4-6 decades inside the 1e-8 bar and the `M = 5` rung of the same
fixture is 1-3 decades OUTSIDE it, so the bar has a real signal above it.

### 9.3 G7 -- two-sided lossless closure, scalar AND Hermitian gyrotropic

`(2,3)` non-conforming pillar pair, `theta = 0.18`, `phi = 0.35`:

| cell | `M = 4` | `M = 5` | `M = 6` | per-rung |
|---|---|---|---|---|
| scalar | 2.692e-04 | 7.891e-05 | **1.465e-06** | 3.4x then 53.8x |
| **gyrotropic** (`e12 = -e21 = 0.9i`, Hermitian, absorbs nothing) | 1.180e-03 | 2.169e-04 | **4.244e-06** | 5.4x then 51.1x |

**CONFIRMED, and the material-independence claim with it** -- a Hermitian
tensor closes exactly as a scalar does, so the mortar is a geometric
projection.  NOTE for durability: the shipped test's `clos[1] < 0.3*clos[0]`
would clear by only 1.02x on this fixture (0.293), see S8.3.

### 9.4 N2 -- the ULP case

See S8.2: the SHAPE reproduces (ULP-level, period-dependent), the stated
values do not.  Worst re-measured 4.626e-16 relative against the 1e-13 bar.

### 9.5 N3 -- two exact-wall representations

See S8.2: 5/5 inside a bar that is a theorem.  The re-measured numbers, and the
real decisions now asserted, are in the test's docstring.

### 9.6 G9 -- the conditioning census, re-run

Worst equilibrated `rcond` over all guarded sites (4 per solve: the mortar
interface, the two per-layer star denominators, the plain mode match), on MY
fixture (period 1.05, `wl` 0.72, `eps = 5.0/2.10`):

| config | `M = 4` | `M = 5` | `M = 6` |
|---|---|---|---|
| pillar (2,3) | 9.973e-04 | 1.819e-04 | 1.505e-05 |
| pillar (2,6) | 2.465e-05 | 1.860e-05 | **1.418e-05** |
| stripe (2,3) | 1.817e-03 | 2.662e-03 | 9.161e-04 |

Worst reading anywhere **1.418e-05**, i.e. **three decades above M1's 1e-8
screen**, and `_guarded_lstsq` / `_guarded_solve` refused nothing.
**CONFIRMED in the useful range** -- and see S6.3 for the regime where the
argument behind it does not hold.

## 10. COST AND MEMORY (task 6)

`v7_cost.py`.

### 10.1 The cross-mass: factored vs dense

| grids | `M` | dense operator | dense | factors | **memory ratio** | apply, factored/dense | identity |
|---|---|---|---|---|---|---|---|
| (2,3) | 6 | 100 x 225 | 0.343 MB | 0.0046 MB | **75x** | **0.49x** | 5.08e-16 |
| (3,6) | 6 | 225 x 900 | 3.090 MB | 0.0137 MB | **225x** | 3.04x | 4.38e-16 |
| (4,6) | 8 | 784 x 1764 | 21.103 MB | 0.0359 MB | **588x** | 15.57x | 4.52e-16 |
| **(6,12)** | 6 | 900 x 3600 | 49.438 MB | 0.0549 MB | **900x** | **40.35x** | 2.73e-16 |
| (8,16) | 6 | 1600 x 6400 | 156.250 MB | 0.0977 MB | 1600x | 66.59x | 3.42e-16 |

**The build's headline -- 900x memory at `(6,12), M = 6` and ~39x apply --
reproduces exactly (900x, 40.35x).  So does the honest half: the SMALL pairs
are slower factored (0.49x at (2,3), whose dense operator is 100x225 -- exactly
the ~100x100 crossover the build states).  The factorisation is an identity to
2.7-5.1e-16 on every pair.  CONFIRMED.**

### 10.2 Per-layer vs shared wall time and RSS, corner-dominated pillar pair

Matched DOF, both arms measured back-to-back in one warm process:

| `q` | eig dim | union wall | per-layer wall | **ratio** | union RSS delta | per-layer RSS delta |
|---|---|---|---|---|---|---|
| 12 | 288 | 2.46 s | 3.05 s | **1.24x** | (cold start, 212 MB) | 1 MB |
| 18 | 648 | 31.94 s | 39.26 s | **1.23x** | 1 MB | 2 MB |
| 24 | 1152 | 183.78 s | 229.71 s | **1.25x** | 0 MB | 4 MB |

**The build's 1.14-1.25x wall-time cost CONFIRMED (1.23-1.25x here).**  Peak
RSS does NOT discriminate at these sizes -- both arms allocate a few MB
incrementally once the process is warm, because the union arm's matrices are
freed and reused.  The memory claim that matters is the structural one:

| staircase | union | per-layer | eig-work ratio |
|---|---|---|---|
| widths 1/2, 1/3, 1/6 at `M = 8` | `N = 6`, dim 3528, **0.19 GB** | dims [392, 882, **3528**], 0.185 GB | **1x** -- the 1/6 slice IS the union grid |
| **widths 1/2, 1/3, 1/4 at `M = 8`** | `N = 12`, dim 14112, **2.97 GB** | dims [392, 882, 1568], **0.037 GB** | **610.8x per eig, 1832.3x for the three-slice stack** |

**The build's "2.97 GB union matrix against 0.037 GB per-layer" reproduces
EXACTLY, and its 1832x is the THREE-SLICE total (`3 x 14112^3 / (392^3 + 882^3
+ 1568^3) = 1832.3`), not the per-eig ratio (610.8x).  CONFIRMED, with the
arithmetic made explicit.**  The first row is a useful counter-example the
build doc does not state: when one slice's own `N` already equals the common
refinement, per-layer grids buy nothing in eigenwork on that slice.

---

## 11. DEFECTS

No library edit was made.  Each item below has a reproducer in
`validation/probe_verify_mortar/`.

### D1 (P1, SILENT WRONG) -- an intra-layer sliver corrupts the mortar, energy-invisibly

**Where:** `PMM2DStackPure(layer_grids='per-layer')` whenever ONE layer's own
non-uniform grid carries two walls closer than ~1e-3 of the period AND its
neighbours sit on different grids (so its interfaces are mortars).

**Reachable through the public API three ways:** `x_walls=` / `y_walls=`
directly; `add_tapered_pillars` with two features whose edges nearly meet; and
`add_tapered_pillar` on a taper that closes -- the midpoint rule's narrowest
sampled width is `w_bottom / (2 n_slices)`, so `n_slices = 64` on a pillar
closing from half the period already reaches 3.9e-03 and `n_slices = 1000`
reaches 2.5e-04.

**Symptom:** on a middle layer that is physically UNIFORM (all-host, so the
answer cannot depend on the wall separation at all), `R(0,0)` moves by up to
**7.799e-03** in absolute efficiency on a quantity of ~0.095 -- 8 % -- as
`delta` falls from 3e-1 to 1e-6, and it does NOT settle; at `M = 8` it is not
even monotone.  The legitimate variation from moving element boundaries inside
a uniform medium, measured on the same fixture between `delta` 3e-1 and 1e-1,
is 1.17e-03.  **The lossless closure is PINNED at 1.31e-05 (`M = 6`) /
2.3e-08 (`M = 8`) across the whole ladder, so `_warn_stag_closure` never fires
and the 1-D sliver fix's `R+T` screen has nothing to see.**

**Attribution, measured:** the SAME layer, same grid, same `delta`, with NO
mortar (both neighbours on its own grid) is exactly `delta`-independent --
spread **4.1e-13** at `M = 6` and **6.7e-12** at `M = 8` down to
`delta = 1e-6`.  The spurious `1/w` spectrum the sliver creates
(`|lam|max = 1.03e+05` at `delta = 1e-5` against a physical ceiling of 3.0) is
harmless inside one grid and corrupts the CROSS-GRID projection.

**Reproducer:** `v6_taper_sliver.py intra intra2`, and the decisive all-host
variant in S6.3 of this document.

**Suggested fix, in the shape the 1-D fix already ships:** a screen at the
grid's OWN entry point.  `Basis1D.__init__` currently accepts any strictly
increasing wall array with no minimum-feature test; the natural guard is the
`_SLIVER_OWN_SCALE_RATIO`-style scale-free one -- refuse (or warn and name the
`min_feature` that removes it) when `min(diff(xb)) / max(diff(xb))` falls below
~1e-2 on a grid that will be MORTARED, which is knowable in
`_solve_per_layer`.  A `R+T` screen will NOT work here: this defect conserves
energy.

### D2 (P2, unhelpful failure) -- two UNGUARDED `np.linalg.solve` calls

Below `delta ~ 1e-7` the same fixtures raise
`numpy.linalg.LinAlgError: Singular matrix` with no message, no hint and no
naming of the offending grid, from

| path | site | call |
|---|---|---|
| MORTAR interface, NEW on this branch | `pmm/_core.py:5270` | `B = np.linalg.solve(lhsH, rhsH)` in `_interface_smatrix_mortar_2d` |
| plain square interface, pre-existing but newly REACHABLE | `pmm/_core.py:1850` | `b = np.linalg.solve(Vb, Va)` in `_interface_smatrix` (its sibling at 1849 is the same shape) |

in a library whose entire conditioning story (`_guarded_solve`,
`_guarded_lstsq`, `_guarded_inverse`, the M1 census) is built on refusing
loudly and explaining.  The build's argument for guarding only `I + BA` --
*"LAPACK `gesv` is backward stable, so a residual screen on them measures
nothing"* -- is true and beside the point: backward stability says nothing
about a singular matrix, and MEASURED these two are the operators that blow up
FIRST (`cond_2` 5.878e+07 and **4.895e+10** at `delta = 1e-3`, while
`I + BA`'s guarded `rcond` still reads 2.5e-07).

The same shape appears three more times on the paths this branch adds or
re-reaches: `_core.py:5264` (`A = np.linalg.solve(lhsE, rhsE)`, the mortar's
E row) and `_core.py:5322` (`X = np.linalg.solve(A, B)` in
`_interface_smatrix_general_mortar_2d`, which an OUT-OF-PLANE or SLANTED
per-layer stack takes), plus the 1-D mortar's own pair at 5075-5076 /
5101-5102, which predates this branch.  Only the two in the table are
demonstrated by this reproducer; the others are the same construction.

### D3 (P3, latent accuracy) -- the far-field projector's FIXED quadrature order

`_stag_fourier_projection` uses `nq = 2 M + 8` Gauss points PER SEGMENT.  That
was sized for a segment of length `d/N`; with arbitrary walls a segment can be
almost the whole period and the rule under-resolves the `e^{i m G x}` kernel.
MEASURED kernel error against an 8x-refined rule: **7.5e-04** at longest
segment 0.96 d, `M = 4`, orders to 7 (against 6-8e-15 on a uniform `N = 3`
lattice) -- see the full scaling table in S2.6.

Device-level it is currently masked, and the reason is structural rather than
lucky: the order cap ties `m_max` to `q = N(M-1)`, so a grid with few segments
cannot request a high order, and the one construction that reaches the regime
(`N = 6` with five 0.02-wide segments and one 0.90 one) is either refused by
`_guarded_lstsq` or already broken by a larger error.  On the fixtures a user
would actually build it reads 1.3e-15 .. 3.2e-11.

**One-line fix:** size the rule from the segment's own length, e.g.
`nq = 2 M + 8 + ceil(4 * m_max * (xb[n+1]-xb[n]) / d)`.

### D4 (P3, doc accuracy) -- three claims in the build doc that do not reproduce

1. **the N2 table** (`0.0` at `d = 1.2` for `N = 2, 3, 4, 6`): re-measured
   3.084e-16 / 1.156e-16 / 4.626e-16 at `N = 4 / 3 / 6`, and `0.0` only at
   `N = 2`.  S8.2.
2. **"the shipped surface RAISES ... `_stag_parity_1d` must refuse a
   non-mirror-symmetric wall set [because otherwise] the parity block
   reduction would have been applied to a structure the pencil does not
   have"**: bypassing the guard changes the answer by exactly 0.0, because
   `_stag_block_eig`'s assembled-pencil residual refuses nine decades over its
   own tolerance.  The guard is right; the stated reason for it is not the
   operative one.  S2.7.
3. **the equal-DOF losing rung** ("1.48x at `q = 30`"): that is the
   EXPERIMENT's prototype.  The shipped library reads 0.317 at `q = 30` on the
   same fixture with a degree-16 reference -- and neither arm is resolvable
   there.  S7.1.

Item 1 is corrected in the shipped test's docstring by this audit; items 2 and
3 are annotations that belong to the build doc, which this audit does not
edit.

### D5 (P3, test durability) -- two assertions that could not fail, corrected

See S8.2: `test_n3_...`'s triangle bar is a theorem, and the taper test's
convergence line passed only through an `or ... < 5e-2` escape hatch.  Both
are restated in this branch's commit `6d8ce03`.

---

## 12. WHAT I COULD NOT VERIFY

Stated plainly, because a verification that reports only what it managed is
worth less than one that says where it stopped.

1. **A THIRD BLAS family.**  Both arms here link scipy-openblas, exactly as the
   build's do.  The OS, compiler, python and numpy differ; the BLAS family does
   not.  Every cross-build spread quoted anywhere in this campaign is therefore
   a LOWER bound, and the build doc's own O-1 caveat stands unchanged.
2. **Cross-build READINGS for the thin bars of S8.3.**  I have WSL pass/fail
   for `test_g5`, `test_g7` and `test_staircase`, not the WSL values of the
   quantities their bars compare.  I therefore reported those margins and did
   NOT change the bars -- restating a bar I cannot bracket on two builds would
   itself be the mistake `TESTING_STANDARDS.md` describes.
3. **The shared-lattice taper control.**  The 4-slice taper's walls need
   `N = 40`, i.e. a 12800-dimension region eig per slice; the shallower
   3-slice control that DOES land on `N = 20` is a 3200-dimension eig per
   slice and did not finish inside this audit's budget on a contended box.
   The comparison against the hybrid staircase (S6.1) covers the same claim
   with a converged oracle; the shared-lattice arm would only have restated
   the cost.
4. **The intra-layer sliver at `M = 10`** (S6.3): the ladder completed at
   `M = 6` and `M = 8` and the `M = 10` run was cut short.  The defect is
   established on two modal counts plus an exact no-mortar control, and the
   `M = 8` arm is the worse of the two, so I do not expect `M = 10` to change
   the verdict -- but it is not measured.
5. **`add_tapered_pillars` under the sliver.**  I demonstrated the intra-layer
   sliver through `x_walls=` and argued the taper builders reach it
   arithmetically (the midpoint rule's `w_bottom / (2 n_slices)`); I did not
   run a `n_slices >= 64` taper to the sliver band, because each such stack is
   64 mortared solves.
6. **Absolute wall times.**  Three other agents' worktrees ran on this box
   throughout.  Ratios measured back-to-back inside one script are sound;
   absolute seconds are upper bounds and the test-file totals (392 s / 395 s
   WIN, 551 s WSL) sit 1.4-2.0x above the build's, which is contention, not a
   change (`test_g3` reads 28.7-39.8 s here against the build's 32.9 s
   depending on load).
7. **`PMM2DStackPure` under JAX** (the build's open item O-4) and the
   `prepare()` / wavelength-sweep path (O-5): both are still absent, and I did
   not attempt to verify claims about surfaces that do not exist.
8. **The `tau`-keyed cache cost** (O-8).  I verified that `tau` is IN the key
   and that no collision occurs (S4.6); I did not measure the cost of the
   rebuild it forces on an angle sweep.

---

## 13. COMMANDS AND COMMITS

```
# the WITHOUT arm (read-only main clone) and the WITH arm, then the diff
PYTHONPATH=/c/tmp/lum_vmortar V1_TAG=with \
  python validation/probe_verify_mortar/v1_bit_identity.py
PYTHONPATH='D:\...\Lumenairy' V1_TAG=without \
  V1_EXPECT_ROOT='D:\...\Lumenairy' \
  python /c/tmp/lum_vmortar/validation/probe_verify_mortar/v1_bit_identity.py
python validation/probe_verify_mortar/v1_compare.py with without

# every other probe (sections in the README)
python validation/probe_verify_mortar/v2_basis.py
python validation/probe_verify_mortar/v2_oracles.py
python validation/probe_verify_mortar/v2_quad.py
python validation/probe_verify_mortar/v3_mortar.py
python validation/probe_verify_mortar/v4_slant_mixed.py
python validation/probe_verify_mortar/v5_equal_dof.py
python validation/probe_verify_mortar/v6_taper_sliver.py
python validation/probe_verify_mortar/v7_cost.py
python validation/probe_verify_mortar/v8_gaps.py

# the suites
python -m pytest -q tests/unit/test_pmm2d_staggered_nonuniform.py \
  tests/unit/test_pmm2d_staggered_mortar.py \
  tests/unit/test_verify_pmm2d_perlayer_slant.py
wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_vmortar && OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_vmortar \
  ~/lumvenv/bin/python -m pytest -q <the same three files>"
```

### 13.1 Test runs

| suite | result |
|---|---|
| `test_pmm2d_staggered_slant.py`, `..._oop.py`, `..._oop_block_eig.py`, `..._anisotropic.py`, `..._magnetic.py`, `..._wood_list.py`, `test_v5_12_0_pmm2d_staggered.py`, `test_v5_21_pmm2d_staggered_oblique.py`, `test_v5_14_0_pmm2d_stack.py`, `test_p2c_pmm2d_stack_cascade.py` | **299 passed** in 1002.9 s (the build doc's 299 in 960 s) |
| `test_pmm2d_staggered_nonuniform.py` + `test_pmm2d_staggered_mortar.py`, AS SHIPPED, WIN | **47 passed** in 392.4 s |
| the same two + `test_verify_pmm2d_perlayer_slant.py`, WSL | **51 passed** in 551.1 s |
| the same three AFTER this audit's test edits, WIN | **51 passed** in 395.5 s |
| `ruff check lumenairy/ tests/` | **clean** (also clean over `validation/probe_verify_mortar/`) |

### 13.2 Commits on `verify/mortar`

| commit | what |
|---|---|
| `d2224ea` | `validation/probe_verify_mortar/` -- the re-measurement probes and their README |
| `d9cff18` | `tests/unit/test_verify_pmm2d_perlayer_slant.py` -- the PER-LAYER SLANTED gate the build listed as an open item (4 tests), `.test_durations` spliced |
| `6d8ce03` | durability fixes to the two shipped test files: three stated measurements that did not reproduce, two assertions that could not fail, and the `_gram_cond` angle proxy stated |

No library file was modified on this branch; no merge, push, tag or version
bump was made.

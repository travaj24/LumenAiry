# BUILD -- the normal-incidence PARITY block reduction for the PURE staggered OUT-OF-PLANE eig

Date: 2026-09-09/10.  Branch `feat/pmm2d-oop-block-eig`, worktree
`C:/tmp/lum_oopfast`, on top of `fb3fd93` (5.43.0 plus one backend commit).
This closes **open item 3** of
`docs/audits/BUILD_PMM2D_STAGGERED_OOP_2026_09_09.md` ("the normal-incidence
involution accelerator was NOT integrated") and the corresponding line of
`docs/audits/VERIFY_PMM2D_STAGGERED_OOP_2026_09_09.md` section 10.  The
structure was measured on the prototype generator in
`docs/audits/EXPERIMENT_PMM2D_STAGGERED_OOP_2026_09_09.md` S11 T2
(`||R A R + A|| / ||A|| = 2.6e-15`, sectors 288/288, and correctly FAILING at
0.36 - 0.73 off centre); the Fourier twin of the same reduction, and the
verify-then-use discipline copied here, are
`docs/audits/EXPERIMENT_PMM2D_OOP_BLOCK_EIG_2026_08_17.md` /
`rcwa._core._generator_block_eig`.

**Mount.**  Windows 11, python 3.14.6, numpy 2.4.4, scipy 1.17.1
(scipy-openblas), tesla-ryzen, 24 cores / 128 GB.  Every probe and test
exported `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1`
**before** python started and asserted `lumenairy.__file__` under
`C:\tmp\lum_oopfast` (version 5.43.0).  Every number below is printed by a
script in `validation/probe_pmm2d_staggered_oop_blockeig/` (JSON in its
`results/`).

---

## 0.  Summary, and the default

**Shipped, default ON** (`symmetry='auto'`, the hybrid's spelling and the
hybrid's precedent).  The two-sided gate the brief made the condition for that
default holds with decades to spare on every arm:

| gate | verdict | numbers |
|---|---|---|
| (1) ON vs OFF on parity-symmetric out-of-plane cells | **PASS** | dR <= **4.9e-15**, dT <= **1.6e-13**, dJones <= **2.6e-14**; eigenvalue SET <= **6.9e-12**; forward/backward split exactly `2q^2 / 2q^2` on both arms, 24 combinations |
| (2) fallback bit-identity | **PASS** | sha256 of (R, T, Jones) IDENTICAL on 6 refusing cases; their structure residual **6.2e-02 .. 6.9e-01** (or the gauge refuses outright at oblique/conical) against an accept bar of **1e-10** -- 8.8 decades of gap above, 3.8 below |
| (3) the parity operator itself | **PASS** | `J^2 - I` = **0.0 EXACTLY** (12 (Nx, M) combinations); eps-free masses parity-invariant to **8.0e-17 .. 1.8e-16**; the derivative bracket parity-ODD to **6.6e-17 .. 1.4e-16** |
| (4) speed, interleaved | **PASS** | region solve **3.3 - 4.2x**; whole solve **1.49 - 1.55x** single-layer, **1.80 - 1.87x** on a three-layer stack |
| (5) fail-before | **PASS** | with the runtime verification disarmed the shipped entry point is wrong by **1.5e-03 .. 2.2e-01**, i.e. 7 to 12 decades above the accepted agreement |
| (6) existing suites | **PASS** | 20 files, **1181 passed** (76 + 148 + 957) -- including every other consumer of `PMM2DStackPure` / `pmm_jones_2d_staggered`, which is the arm that proves the DEFAULT change moves nothing |

One honest difference from the Fourier twin: that one produced eigenpairs
*better* than the dense `zgeev` it replaced.  This one does not -- the
staggered reduction squares the spectrum through `Xh Yh`, and the measured
backward-error ratio is **0.28 .. 7.20** (table B6).  It stays within one
decade of the dense solve's own backward error, at 2.1e-14 absolute, which is
11 decades below anything the observables can see; that is what the test
asserts, and it is stated rather than glossed.

---

## 1.  The structure, derived on THIS state ordering

The out-of-plane pencil is `A x = q B x` on `x = [E1; E2; G1; G2]`
(`_assemble_oop`), with `B = blkdiag(Ggram1, Ggram2, Ggram2, Ggram1)`.

**The parity map.**  `x -> d - x` on one axis sends segment `n -> N-1-n` and
the reference coordinate `u -> -u`, so the local modified-Legendre functions
permute:

```
Ltilde_1(-u) = (1+u)/2 = Ltilde_2(u)          the two HALF-HATS swap
(L_a - L_{a-2})(-u) = (-1)^a (...)            the BUBBLES carry a sign
```

Lifted through `Basis1D._build_sets`, `Btilde`'s hat at node `j` goes to the
hat at node `(N - j) mod N` with sign `+1` (the seam hat is fixed only when
`tau = 1`, which is why the gauge is normal-incidence-only), its bubbles
`(seg, a) -> (N-1-seg, a)` with sign `(-1)^a`; `B`'s two independent per-segment
half-hats swap, `(seg, 0) <-> (N-1-seg, 1)`, sign `+1`, and its bubbles behave
as `Btilde`'s.  Both are SIGNED PERMUTATIONS with the sign constant on every
orbit, so `J^2 = I` exactly.  `_stag_parity_1d` builds them in closed form; the
prototype's least-squares construction (`m7_cost.py`) agrees with it to
**2.2e-16 .. 3.3e-16**, which is the least-squares residual itself -- the
closed form is the exact one.

**The sign.**  With every component map `e11 .. e33` parity-EVEN on the grid,
every eps-weighted mass is parity-EVEN (`P_i A_ij P_j = A_ij`) and each of the
four SINGLE-derivative blocks `P13, P23, CwE1, CwE2` is parity-ODD.  Feeding

```
e1 -> P1 e1 ,  e2 -> P2 e2 ,  g1 -> -P2 g1 ,  g2 -> -P1 g2
```

through the two eliminations gives `e3 -> +P3 e3` (its eps terms and its
derivative terms each pick up two flips) and `g3 -> -Pw g3`, after which each of
the four pencil rows has its `B` side EVEN and its `A` side ODD.  With
`S = diag(I, I, -I, -I)` and `Pst = blkdiag(P1, P2, P2, P1)`,

```
R = S . Pst ,   R^2 = I ,   R A R = -A ,   R B R = +B
```

Neither factor works alone: the parity alone is broken by the derivative
blocks, the sign alone by the eps blocks -- the same statement the Fourier case
makes about the order flip and the E/H sign flip, and the sign pattern is
DERIVED here on `[E1; E2; G1; G2]` rather than transplanted from
`[Ex; Ey; Hx; Hy]`.

**The reduction.**  `R` is real orthogonal, so its `+1`/`-1` eigenspaces `U+`,
`U-` are spanned by `(e_i +/- e_{J(i)})/sqrt(2)` and each has dimension exactly
`2 q^2`: `tr R = 0` because on a square grid the two `E` blocks and the two `G`
blocks contribute equal and opposite parity traces (asserted from the signed
permutation in the tests, not from an eig).  In that basis

```
U^T A U = [[0, X], [Y, 0]]        U^T B U = blkdiag(Bp, Bm)
```

and whitening each sector by its own Cholesky (`Bp = Lp Lp^H`, `Bm = Lm Lm^H`,
`Xh = Lp^-1 X Lm^-H`, `Yh = Lm^-1 Y Lp^-H`) leaves ONE standard `2 q^2` eig:

```
Xh Yh up = q^2 up ,   um = Yh up / q ,
x = U+ Lp^-H up  +/-  U- Lm^-H um       for the +/- q pair
```

`U` is a signed pairing, so `X, Y, Bp, Bm` and the `4 q^2` eigenvector
expansion are `O(n^2)`; the only cubic work is at `2 q^2`.  The whitened vector
`[up; +/- um]` is normalised to unit 2-norm, because it is the whitened vector
in the `(U, Cholesky)` factorization of `B` -- unitarily equivalent to the dense
path's -- and `_select_forward_flux` reads RELATIVE noise ceilings, so the
scale is load-bearing (the same trap the Fourier twin documents).

---

## 2.  Measurements

### B0  the parity operator's own properties

`b1_gates.py` section B0, x-axis basis, 12 combinations (`Nx` 2/3/4,
`M` 5/6/7/8).

| quantity | envelope over the 12 |
|---|---|
| `J^2 - I` (both sets) | **0.0 exactly** |
| `P^T Mtt P - Mtt` / `max|Mtt|` (and the `B` twin) | 8.02e-17 .. 1.80e-16 |
| `P_b^T <B\|d\|Btilde> P_t + <B\|d\|Btilde>` / `max` (parity-ODD) | 6.59e-17 .. 1.39e-16 |

The masses and the derivative bracket are Gauss-Legendre quadratures of
polynomials, so their parity image differs from themselves only by the
summation order -- which is what the envelope says.  The test also asserts the
WRONG sign FAILS (`P^T C P - C` above `1e-3 * max|C|`), so "the derivative is
odd" is a two-sided claim rather than a small number.

### B2  the structure residual on the ASSEMBLED pencil (the `_STAG_BLOCK_TOL` table)

`b1_gates.py` section B2.  `dA = max|R A R + A| / max|A|`,
`dB = max|R B R - B| / max|B|` -- exactly the two quantities the shipped gate
computes.

| cell | grids x M | `dA` | `dB` | engages |
|---|---|---|---|---|
| uniform tilted uniaxial | (2,2) and (3,3), M 5..8 | 1.33e-15 .. 4.79e-15 | 5.80e-17 .. 1.96e-16 | YES |
| centro-symmetric pillar | same | 5.84e-16 .. 1.15e-14 | same | YES |
| centro-symmetric NON-RECIPROCAL | same | 5.94e-16 .. 1.15e-14 | same | YES |
| centro-symmetric LOSSY | same | 5.82e-16 .. **1.48e-14** | same | YES |
| **OFF-CENTRE pillar** | (2,2) M6 / (3,3) M6 | **6.95e-01** / **5.77e-01** | 1.74e-16 / 1.96e-16 | no |
| **parity-BREAKING tensor grid** | (2,2) M6 / (3,3) M6 | **6.17e-02** / **2.89e-01** | same | no |
| uniform, OBLIQUE 25 | (2,2), (3,3) M6 | *gauge refuses* | -- | no |
| uniform, CONICAL 25/40 | (2,2), (3,3) M6 | *gauge refuses* | -- | no |

**Carrying envelope 5.8e-16 .. 1.5e-14; smallest real violation 6.2e-02.**
`_STAG_BLOCK_TOL = 1e-10` sits **3.8 decades above** the first and **8.8
decades below** the second -- the same bar, and the same shape of gap, as the
hybrid's `_OOP_BLOCK_TOL`.

Note the `B` residual never moves: `B` is the eps-free block Gram, so it is
parity-invariant on any uniform wall grid.  The discriminating quantity is
`dA`, and the shipped gate tests both anyway (`B` is what the sector Choleskys
rely on).

The **parity-BREAKING tensor grid** is the row that matters most: its PATTERN
is its own parity image (the same two mirror pixels are filled) and only the
TENSORS differ, so any gate that tested the shape of `eps` rather than the
assembled operators would accept it.  The shipped gate refuses it at 6.2e-02.

### B1  reduction ON vs OFF -- observables and the eigenvalue set

`b1_gates.py` section B1, through the SHIPPED entry point
`pmm_jones_2d_staggered`, both incident polarizations, per order; 24
combinations (`Nx` 2/3, `M` 5/6/7, four cells).

| cell | grid | M | dR | dT | dJones | d(eig set) | `min\|q\|/max\|q\|` |
|---|---|---|---|---|---|---|---|
| uniform tilted uniaxial | (2,2) | 6 | 6.87e-16 | 1.89e-14 | 2.05e-15 | 7.36e-13 | 5.71e-02 |
| centro pillar | (2,2) | 6 | 5.83e-16 | 1.57e-14 | 3.88e-15 | 7.42e-13 | 1.94e-02 |
| centro NON-RECIPROCAL | (2,2) | 7 | 2.23e-15 | 1.87e-14 | 1.19e-14 | 6.95e-12 | 4.10e-03 |
| centro LOSSY | (3,3) | 7 | 4.70e-15 | 6.53e-14 | 2.61e-14 | 4.18e-12 | 1.23e-02 |
| **worst over all 24** | | | **4.87e-15** | **1.65e-13** | **2.61e-14** | **6.95e-12** | 4.1e-03 .. 9.5e-02 |

The eigenvalue set is compared by symmetric Hausdorff distance, not by sorting:
a near-degenerate doublet would otherwise be paired wrongly and manufacture a
mismatch.  `min|q|/max|q|` is the quantity `_STAG_GAM_FLOOR = 1e-13` guards
(the reconstruction divides by `q`); it never approaches the floor away from a
Rayleigh cutoff, which the entry points already warn about.

### B3  a refusal is BIT-IDENTICAL to `symmetry=False`

`b1_gates.py` section B3, sha256 over `(orders, R, T, Jones)`.

| case | grid | M | mount | sha(ON) | sha(OFF) | identical | `dA` |
|---|---|---|---|---|---|---|---|
| OFF-CENTRE pillar | (2,2) | 6 | normal | `5ea862fb24f5796a` | `5ea862fb24f5796a` | **YES** | 6.95e-01 |
| OFF-CENTRE pillar | (3,3) | 6 | normal | `3fbe1cd4cb0c27bd` | `3fbe1cd4cb0c27bd` | **YES** | 5.77e-01 |
| parity-BREAKING tensor | (2,2) | 6 | normal | `56c5bae0e71fc0b6` | `56c5bae0e71fc0b6` | **YES** | 6.17e-02 |
| parity-BREAKING tensor | (3,3) | 6 | normal | `507c26aab6be8014` | `507c26aab6be8014` | **YES** | 2.89e-01 |
| uniform, OBLIQUE 25 | (2,2) | 6 | oblique | `03441a6110e44860` | `03441a6110e44860` | **YES** | gauge refuses |
| centro pillar, CONICAL 25/40 | (3,3) | 6 | conical | `d2795d6395a64383` | `d2795d6395a64383` | **YES** | gauge refuses |

Bit-identity here is not a tolerance claim: a refusal literally executes the
same dense branch.

### B5 / B5b  FAIL-BEFORE -- the runtime verification is load-bearing

`b1_gates.py` section B5 (spectrum) and `b3_residual.py` section B5b
(observables, through the SHIPPED entry point with `_STAG_BLOCK_TOL`
monkeypatched to 1.0 so the structure test cannot refuse).

| case | grid | M | `dA` | d(eig set) forced vs dense | dR | dT | dJones |
|---|---|---|---|---|---|---|---|
| OFF-CENTRE pillar | (2,2) | 6 | 6.95e-01 | **4.32e-01** | **9.44e-03** | **2.22e-01** | **6.32e-02** |
| OFF-CENTRE pillar | (3,3) | 6 | 5.77e-01 | **2.53e-01** | **2.89e-03** | **1.06e-01** | **4.42e-02** |
| parity-BREAKING tensor | (2,2) | 6 | 6.17e-02 | **5.07e-02** | **2.11e-03** | **7.39e-03** | **7.43e-03** |
| parity-BREAKING tensor | (3,3) | 6 | 2.89e-01 | **1.24e-01** | **1.46e-03** | **1.71e-02** | **9.72e-03** |
| *accepted agreement, for contrast* | | | *<= 1.5e-14* | *<= 6.9e-12* | *<= 4.9e-15* | *<= 1.6e-13* | *<= 2.6e-14* |

Twelve decades on the eigenvalue set, eleven on `dR`.  The oblique and conical
arms are refused one step earlier still -- by the gauge, before the pencil is
looked at -- which the same table records.

### B6  eigenpair backward error, factored vs the dense `zgeev` on the SAME pencil

`b3_residual.py`, normwise `|A x - q B x| / ((|A| + |q||B|) |x|)`, 18
combinations.

| cell | grid | M | `4q^2` | dense | factored | ratio | forward count dense / factored |
|---|---|---|---|---|---|---|---|
| uniform tilted uniaxial | (2,2) | 5 | 256 | 1.58e-15 | **4.49e-16** | 0.28 | 128 / 128 |
| centro NON-RECIPROCAL | (2,2) | 7 | 576 | 2.70e-15 | 1.94e-14 | **7.20** | 288 / 288 |
| centro pillar | (3,3) | 7 | 1296 | 5.07e-15 | 2.09e-14 | 4.11 | 648 / 648 |
| **envelope over 18** | | | | **1.13e-15 .. 5.27e-15** | **4.49e-16 .. 2.11e-14** | **0.28 .. 7.20** | equal in every row |

The factored path is NOT uniformly better than dense here (unlike the Fourier
twin, which never squares its spectrum).  The mechanism is explicit: the
reduction solves `Xh Yh up = q^2 up`, so the conditioning of the eigenvalue
squares.  The absolute number stays at 2.1e-14, and the flux selector
classifies identically -- exactly `2 q^2` forward on both arms in every row.

### B4a  interleaved whole-solve A/B (separate subprocesses, min over 4 rounds)

`b2_speed.py`.  Both arms are SEPARATE subprocesses, alternating round-robin,
same worktree, same process shape, differing only in the `symmetry` flag --
the instrument the Fourier experiment's S5 uses, so these ratios are directly
comparable with its 1.61 - 2.35x.

| grid | M | layers | `4q^2` | dense [s] | factored [s] | speedup |
|---|---|---|---|---|---|---|
| (2,2) | 6 | 1 | 400 | 0.361 | 0.233 | **1.55x** |
| (2,2) | 7 | 1 | 576 | 0.985 | 0.646 | **1.53x** |
| (2,2) | 8 | 1 | 784 | 2.151 | 1.401 | **1.54x** |
| (3,3) | 6 | 1 | 900 | 3.092 | 2.052 | **1.51x** |
| (3,3) | 7 | 1 | 1296 | 8.424 | 5.666 | **1.49x** |
| (3,3) | 8 | 1 | 1764 | 21.381 | 14.213 | **1.50x** |
| (2,2) | 7 | **3 distinct layers** | 576 | 2.425 | 1.298 | **1.87x** |
| (3,3) | 7 | **3 distinct layers** | 1296 | 18.860 | 10.500 | **1.80x** |

Per-round spreads are in `results/b2_speed.json`; the dense arm's rounds vary
by 1-10% and the factored arm's by 1-11%, and the table takes the min on both
sides (the conservative reading).

### B4b  where the time goes

Same script, in-process, after a warm-up.

| grid | M | `4q^2` | region OFF [s] | region ON [s] | region ratio | whole OFF [s] | region share |
|---|---|---|---|---|---|---|---|
| (2,2) | 6 | 400 | 0.182 | 0.055 | **3.31x** | 0.366 | 49.6% |
| (2,2) | 7 | 576 | 0.511 | 0.138 | **3.72x** | 0.982 | 52.1% |
| (2,2) | 8 | 784 | 1.033 | 0.273 | **3.78x** | 2.172 | 47.6% |
| (3,3) | 6 | 900 | 1.499 | 0.400 | **3.74x** | 3.146 | 47.6% |
| (3,3) | 7 | 1296 | 3.795 | 1.022 | **3.71x** | 8.328 | 45.6% |
| (3,3) | 8 | 1764 | 9.427 | 2.271 | **4.15x** | 22.640 | 41.6% |

The region solve is **42 - 52%** of a single-layer solve, so Amdahl caps the
single-layer speedup at `1/(1 - 0.42..0.52 * 0.73) ~= 1.44 - 1.62x`, which is
what B4a measures.  A multi-layer stack pays the region solve once per DISTINCT
layer against one cascade, which is why the three-layer rows reach 1.80 -
1.87x.  The flop-count ideal for the eig alone is 8x; the realised 3.3 - 4.2x
is the `O(n^2)` overheads, the two sector Choleskys, the four `2 q^2`
triangular solves and the `2 q^2` eig's own constant.

---

## 3.  What shipped

| file | change |
|---|---|
| `lumenairy/elements/pmm/twod_staggered.py` | NEW `_stag_parity_1d`, `_stag_parity_gauge`, `_stag_block_eig`, `_STAG_BLOCK_TOL`, `_STAG_GAM_FLOOR`; `_region_modes_oop(..., symmetry=False)`; `pmm_jones_2d_staggered(..., symmetry='auto')`; module docstring cost line |
| `lumenairy/elements/pmm/stack2d_pure.py` | `PMM2DStackPure(..., symmetry='auto')` via the shared `_symmetry_on`, threaded to `_region_modes_oop` |
| `tests/unit/test_pmm2d_staggered_oop_block_eig.py` | NEW -- 36 tests |
| `validation/probe_pmm2d_staggered_oop_blockeig/` | NEW -- `b1_gates.py`, `b2_speed.py`, `b3_residual.py` + `results/` |
| `CHANGELOG.md` | `## [Unreleased]` created (this work + the three previously unlogged BOR SEM / backend commits) |
| `.test_durations` | 36 entries spliced (measured, 100.8 s total) |

### Gating and fallback

The accelerator is offered only when `symmetry` is on and the layer is
OUT-OF-PLANE (an in-plane or scalar layer never reaches `_region_modes_oop`).
`_stag_parity_gauge` supplies `(perm, r)` and returns `None` at oblique or
conical incidence -- a free necessary-condition gate, because `tau != 1` breaks
the hat permutation.  Everything else is decided by `_stag_block_eig` on the
assembled pencil:

* `max|R A R + A| <= _STAG_BLOCK_TOL * max|A|` and
  `max|R B R - B| <= _STAG_BLOCK_TOL * max|B|` -- **1e-10**, derived in B2,
  row-blocked at 256 rows so no second `4 q^2 x 4 q^2` transient is allocated,
  and read at CALL time so a test can walk the ladder through the shipped code;
* the two sector counts must each be exactly `2 q^2`;
* `Bp`, `Bm` must be positive definite (a `LinAlgError` falls back);
* `min|q| > _STAG_GAM_FLOOR * max|q|` -- **1e-13**, because the reconstruction
  divides by `q` (measured `min|q|/max|q|` = 4.1e-03 .. 9.5e-02);
* a non-finite reconstruction.

Any of them returns `None` and the dense Cholesky-whitened `4 q^2` eig runs on
the untouched pencil, in the same code as before this build -- which is why the
refusal is bit-identical rather than merely close.

---

## 4.  Tests, durations, suites re-run

Command (from the worktree, every run):

```
PYTHONPATH=/c/tmp/lum_oopfast OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python -m pytest -q -p no:randomly <files>
```

| file | tests | duration |
|---|---|---|
| **`tests/unit/test_pmm2d_staggered_oop_block_eig.py`** (new) | **36 passed** | **101 - 117 s** (three runs) |
| `tests/unit/test_pmm2d_staggered_oop.py` + `test_pmm2d_staggered_anisotropic.py` | 76 passed | 161 s |
| `test_v5_12_0_pmm2d_staggered.py`, `test_v5_21_pmm2d_staggered_oblique.py`, `test_staggered.py`, `test_audit_p1_staggered_guard.py`, `test_p2c_pmm2d_stack_cascade.py`, `test_p2t_pmm2d_tree_cascade.py`, `test_pmm2d_lossless_closure_two_sided.py`, `test_audit_s1_3_pmm2d_lossless_tripwire.py`, `test_v5_14_0_pmm2d_stack.py` | 148 passed | 468 s |
| every OTHER consumer of `PMM2DStackPure` / `pmm_jones_2d_staggered` (`test_audit_dynameta_consumer_api_2.py`, `test_audit_w3_entry_validation.py`, `test_audit_w6_pmm_rcwa.py`, `test_niche_audit_w7_pmm.py`, `test_v5_11_0_pmm2d.py`, `test_v5_12_0_pmm2d_loss.py`, `test_v5_14_0_pmm2d_cell.py`, `test_v5_14_0_pmm_audit_fixes.py`, `test_public_api.py`) -- the arm that proves the DEFAULT change moves nothing | **957 passed** | 1743 s |
| `tests/unit/test_pmm2d_staggered_oop_corner_convergence.py` (from the reference study, same branch) | 1 passed | 20 s |

**1218 tests green across 22 files** with the reduction on by default (1181 pre-existing plus the 37 new).

Slowest tests in the new file: `test_forcing_..._wrong_by_decades[(3,3)]`
14.7 s, `test_reduction_matches_the_dense_path_on_R_T_and_jones[(3,3)]`
12.6 s, `test_a_refusal_is_bit_identical[conical (3,3)]` 11.6 s.  All well
under the 40 s per-test rule; the file is under the 3 min rule; grids are
(2,2)/(3,3)/(4,4)-basis-only and `M <= 8` throughout.

`ruff check lumenairy/ tests/` clean.

---

## 5.  Open items

1. **Oblique and conical incidence stay at 1.00x**, and provably so: at
   `tau != 1` the seam hat's image carries the Bloch factor on the other leg,
   so the map is not a signed permutation of the set at all.  This is the exact
   staggered analogue of the Fourier statement (the order set is not closed
   under the flip off normal), and nothing built on this structure can help
   there.
2. **Off-centre cells are refused rather than recentred.**  The parity
   implemented is `x -> -x mod d`; a cell that is centro-symmetric about
   `c = k h / 2` for integer `k` would carry the structure under `x -> 2c - x`,
   which in this basis is a segment ROLL rather than a phase.  Rolling the
   pixel grid to the inferred centre before meshing would recover those cells;
   unmeasured, and it changes which cells the discretisation sees, so it is a
   separate experiment.  (The hybrid leaves the same door open for the same
   reason.)
3. **The reduction squares the spectrum** (B6): backward error up to 7.2x the
   dense solve's on the same pencil.  Harmless at 2.1e-14, but it is the one
   respect in which this is worse than its Fourier twin, and a cell driven to
   a near-null mode would hit `_STAG_GAM_FLOOR` and fall back rather than
   degrade.
4. **`symmetry` is honoured only by the OUT-OF-PLANE region solve.**  The
   in-plane `2 q^2` pencil already carries the `[W; -V] <-> -lam` symmetry and
   is not touched; whether the parity ALSO folds that pencil into two `q^2`
   sectors is a separate (and probably worthwhile) question this build did not
   measure.
5. **NumPy only**, as the whole staggered path is.

## Corrections (verification 2026-09-10, VERIFY_PMM2D_STAGGERED_OOP_BLOCK_EIG)

* Gate (2)'s "8.8 decades of gap above" describes the four violating FIXTURES, not the
  gate: an engineered 1e-7-relative parity break reads dA = 2.535e-09, only 1.4 decades
  above the 1e-10 bar.  The bar is nevertheless safe for a reason this doc did not
  measure: the observable error is QUADRATIC in the residual (dR ~ 0.56 dA^2 over
  five decades), so the worst error the gate can accept is ~6e-21.  The crossing sits
  exactly at 1e-10 (last accepted 5.070e-11, first refused 1.521e-10).
* The fail-before's `_STAG_BLOCK_TOL = 1.0` was not a full disarm (dA <= 2; an off-centre
  metal pillar at dA = 1.033 was still refused); the test now disarms with `np.inf`.
* Backward-error ratio re-measured 0.435..5.390 (this doc: 0.28..7.20); the 1e2 bar is
  two-sided (a forced reduction reads 5.85e12..2.03e13).
* Cold-subprocess single-layer speed is 1.12..1.51x (~0.7 s per-process warm-up dilutes
  it); the 1.49-1.55x figure is the warm number.

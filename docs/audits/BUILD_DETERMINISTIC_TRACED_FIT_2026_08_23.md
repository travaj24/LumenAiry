# BUILD -- the TRACED EXIT FIELD, deterministic by construction at any thread count

Branch `feat/deterministic-traced-fit`, cut from `origin/main` @ `a33d69a`
(the 5.41.0 release) -- worktree `C:/tmp/lum_dt`.  Closes the item
`docs/audits/BUILD_DETERMINISTIC_CARRIER_FIT_2026_08_22.md` S7.3 declared
open ("This is a live open item, not a closed one").

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X   137.4 GB
python 3.14.6   numpy 2.4.4      scipy 1.17.1
scipy-openblas 0.3.31.188.0      USE64BITINT DYNAMIC_ARCH NO_AFFINITY
lumenairy 5.41.0+ (worktree C:/tmp/lum_dt)
```

---

## 0. VERDICT

`apply_real_lens_traced(carrier='auto')`'s **exit FIELD** is now
bit-reproducible on one build at any BLAS thread count and over any number of
repeats -- the claim D14 wrote, measured failing, and withdrew.  Default
**ON** (`DETERMINISTIC_TRACED_FIT = True`), because the whole-call overhead at
the worst-shaped traced call measured is **+1.6%**, and the accuracy goes UP
by three decades on the widest fits, not down.

**FOUR REFUTATIONS, and three of them are of the D14 audit that scoped this
work out.**  They are stated first because each one changed the design:

1. **THE TWO 120-TERM FITS ARE NOT RESIDUAL-EIKONAL FITS.**  D14 S7.3 and its
   pinning test both name them that.  Censused with the caller's own source
   line (S1), they are `_lens_imap.build_inverse_map`'s total-degree-14 exit
   solves -- a different module, a different basis, and a 3-column right-hand
   side.  The residual-eikonal fit is a 27-term gradient fit that
   `test_niche_c13` separately shows screens CLEAR by six orders.
2. **D14's COST ARGUMENT PRICED A KERNEL THAT WOULD NOT HAVE FIXED THEM.**
   The 34x it quoted was the cost of putting those fits on the deterministic
   GRAM.  Measured (S4), every non-carrier fit on the traced chain screens
   numerically singular under C13 -- Gram rcond 1.6e-9 at 28 terms and 9.6e-11
   at 120, against the 1e-8 screen -- and takes `_solve_lstsq_qr`, a threaded
   `dgeqrf` over the full design matrix.  **A deterministic Gram alone changes
   nothing on that path.**  D14's "one hole, declared" is not a corner here;
   it is the default.
3. **THE OBVIOUS KERNEL IS LESS ACCURATE THAN THE ROUTE IT REPLACES.**  A
   per-block `np.einsum` over D14's 4096-row block reads, at the traced fits'
   own shapes, 3-10x worse than D14's ufunc partial and **2-6x worse than the
   threaded family's WORST legal partition** -- a determinism fix that pays
   for itself in error.  Cause: einsum accumulates sequentially where
   `np.sum` is pairwise.  Cure: a 64-row einsum block (S5).
4. **REFINEMENT MUST BE ABLE TO REFUSE, and a test suite is what proved it.**
   The first cut refined wherever the screen fired.  On the niche-D7
   hard-mask design matrix -- whose equilibrated Gram has a non-positive
   eigenvalue -- refinement does not converge and the route returned a fit
   missing the least-squares residual by **1.4e5x**.  Caught by
   `test_niche_d7::test_c13_cures_the_hard_mask_fold_at_the_d7_order`, not by
   any test written for this layer (S6.3).

**AND THE NON-CLAIM, STATED FIRST BECAUSE IT IS THE ONE THAT GETS
OVERREACHED.**  Nothing here claims cross-build or cross-platform bit
identity, and no test asserts it.  The property delivered is exactly:

> ONE build, ANY thread count, ANY number of repeats -> the same bytes.

---

## 1. THE CENSUS, RE-TAKEN AND RE-ATTRIBUTED

`validation/probe_traced_det/p01_census.py`, N = 512 fixture, one arm per
process, width pinned before NumPy loads.  Each row is a
`_solve_lstsq_thread_safe` call with its CALLER's file and line:

| shape | caller | OMP=1 | OMP=4 | OMP=8 |
|---|---|---|---|---|
| `(119936, 5)` | `_lens_traced:4338 _compute_carrier` | `0deff607e8` | `0deff607e8` | `0deff607e8` |
| `(1457, 28)` x3 | `_lens_traced:2721 _Cheb2DEvaluator.__init__` | 3 values | same 3 | same 3 |
| `(1337, 120)` | **`_lens_imap:1550 build_inverse_map._solve`** | `692283176c` | `a5685172e8` | `a5685172e8` |
| `(1337, 120)` | **`_lens_imap:1727 build_inverse_map`** | `f3437277bd` | `932500c51a` | `932500c51a` |

The carrier fit is invariant (D14 works).  The two 120-term fits move, in the
`{1,2} / {4,8}` pattern of a BLAS split -- and they are in `_lens_imap`, not
in the residual-eikonal fit D14 named.  Refutation 1.

### 1.1 Whether that reaches the FIELD is fixture-dependent, and that is a trap

Eight traced fixtures, shipped route, widths {1, 2, 4, 8}
(`p05_field_scan.py`).  The two 120-term coefficient sets move on **all
eight**.  The returned exit field moves on **one**:

| fixture | shipped field hashes over 4 widths |
|---|---|
| N=384/512/640/768, `ray_subsample` default (7 cases) | **1 distinct** |
| N=512, `ray_subsample=2` | **4 distinct** (`3ec709b4` / `bf3fbf06` / `5a3b5a05` / `9f50d29a`) |

So a fail-before taken on the natural fixture reads as green.  That is the
same trap D14 S1.2 recorded from the other side (a carrier so strong the
design matrix collapsed below the splitting threshold), and it is why the
D15 suite's fixture carries `ray_subsample=2` with the reason in its comment.

---

## 2. THE ROUTES, ADJUDICATED BY MEASUREMENT

`p02_routes.py`, all routes sharing the same block partition and the same
carry-stack tree so only the per-block partial differs.  Best of three,
OMP = 1 and 8, hashes over both widths:

| shape | `A.T@A` (BLAS) | D14 ufunc | **einsum** | einsum on a transposed tile |
|---|---|---|---|---|
| 1337 x 120, 3 rhs | 0.0006 s | 0.0260 s | **0.0037 s** | 0.0066 s |
| 141471 x 66 | 0.0241 s | 0.3687 s | **0.1097 s** | 0.2231 s |
| 1e6 x 120, 3 rhs | 0.4914 s | 8.5347 s | **2.7985 s** | 4.9857 s |
| 119936 x 5 | 0.0009 s | **0.0019 s** | 0.0025 s | 0.0016 s |

einsum's hashes are identical at both widths at every shape; D14's ufunc
route is 43x BLAS at the traced 120-term shape and einsum is 6x.  The
transposed tile that HELPS the ufunc partial (contiguous pair reads) HURTS
einsum, because einsum already reads its operands in the order it wants --
recorded so the D14 constant is not transplanted by analogy.

Option (b) of the brief -- "blocked syrk-shaped manual tiling with ufuncs at
a better memory-access pattern" -- IS that transposed-tile column, and it is
2x slower than the plain row-block einsum at every wide shape.  Option (c),
exploiting the Gram's symmetry, is not taken: einsum computes both triangles,
and recovering the 2x would need `M` separate einsums, i.e. exactly the
Python-call overhead the route exists to remove.  At the traced shapes the
whole solve is 3.7 ms against a 2.4 s call, so the 2x is unbuyable anyway.

### 2.1 The crossover is the same constant as the byte-compatibility boundary

`p06`, n = 200 000, einsum time / ufunc time:

| M | 5 | 6 | **8** | 10 | 12 | 16 | 28 | 66 | 120 |
|---|---|---|---|---|---|---|---|---|---|
| einsum / ufunc | 1.36x | 1.03x | **0.81x** | 0.63x | 0.53x | 0.40x | 0.35x | 0.30x | 0.29x |

The two routes cross between 6 and 8 terms.  `_DET_EINSUM_MIN_TERMS = 8`
therefore does two jobs with one number: it puts each fit on its faster
kernel, AND it keeps the 5-term carrier fit on D14's exact arithmetic, so
nothing 5.41.0 released moves (S7.1).

---

## 3. IS `einsum` REALLY BLAS-FREE ON THIS BUILD?

Verified the way D14 verified the ufunc reductions -- hashes across pinned
widths in fresh interpreters -- plus the two things a hash cannot show
(`p03_einsum_blas.py`, best of three):

| shape | call | w=1 | w=2 | w=4 | w=8 | hash |
|---|---|---|---|---|---|---|
| 500 000 x 120 | `einsum(optimize=False)` | 1.1806 | 1.1825 | 1.1765 | 1.1771 | **1 value** |
| 500 000 x 120 | `einsum(optimize=True)` | 0.1744 | 0.1120 | 0.0754 | 0.0813 | = BLAS |
| 500 000 x 120 | `A.T @ A` | 0.1739 | 0.1157 | 0.0754 | 0.0836 | -- |
| 2e6 x 66 | `einsum(optimize=False)` | 1.5157 | 1.5215 | 1.5157 | 1.5210 | **1 value** |

`optimize=False` is flat to **0.5% across a 8x width sweep** -- it gains
nothing from threads because it uses none -- and reads one hash at every
width and every shape.

**`optimize=True` IS NOT A HARMLESS DEFAULT AND THAT IS THE POINT.**  It
returns bytes **identical to `A.T @ A`** at every shape and width tried, and
speeds up 15x from one thread to four: einsum's optimizer routes the
contraction through `tensordot` -> BLAS `dgemm`.  So the kernel passes
`optimize=False` EXPLICITLY (it is also NumPy's current default -- passing it
is how the guarantee survives a default flip), and
`test_einsum_is_blas_free_and_optimize_true_is_not` pins both halves.

---

## 4. THE C13 STEP-DOWN IS THE DEFAULT HERE, NOT A CORNER

Instrumented on the real traced call (`ray_subsample=2` fixture):

```
  gram rconds:  (5,5)     1.0
                (28,28)   1.611784801670436e-09   x3
                (120,120) 9.64413121384644e-11    x2
  screen:       _LSTSQ_GRAM_RCOND_MIN = 1e-8
  QR calls:     (24065,28) x3, (23573,120) x2      <- ALL FIVE
```

Every non-carrier fit is below the screen, so all five re-solve through
`_solve_lstsq_qr` -- `scipy.linalg.qr(mode='r')` on `[A | b]`, LAPACK
`dgeqrf`, a threaded blocked BLAS-3 factorisation.  With a deterministic Gram
and nothing else, the returned coefficients were measured **byte-identical to
the shipped route at each width, and still four-valued across widths**: the
kernel was doing exactly nothing.  Refutation 2.

The rconds themselves read identically at OMP 1/4/8, so the BRANCH is
width-invariant even though the branch's answer is not -- which is what makes
a deterministic replacement possible at all.

---

## 5. THE SCHEME

Three parts.  The first is D14's, kept; the second and third are new.

### 5.1 A per-block partial that is affordable AND accurate

`_det_partial_einsum(Ab, Bb)` -- two `np.einsum(..., optimize=False)`
contractions per block.  `_det_normal_equations` dispatches to it at
`n_terms >= _DET_EINSUM_MIN_TERMS`, and to D14's `_det_partial_ufunc` below
that.  Both feed the SAME carry-stack tree.

**THE BLOCK LENGTH IS NOT D14's, AND THE REASON IS A FAILED TEST.**  Oracle
comparison (D14's Dekker two-product + `math.fsum` reference, imported not
re-derived) against the legal-partition family, at the traced fits' shapes:

**A = 1337 x 120**

| block | G | rhs | coefficients |
|---|---|---|---|
| family best | 8.50e-17 | 2.18e-16 | 4.91e-16 |
| family worst | 1.70e-16 | 7.64e-16 | 9.82e-16 |
| einsum @ 4096 rows | 7.65e-16 | 1.64e-15 | **3.14e-15** |
| **einsum @ 64 rows** | **8.50e-17** | **3.27e-16** | **4.91e-16** |

**A = 1457 x 28**

| block | G | rhs | coefficients |
|---|---|---|---|
| family best / worst | 7.80e-17 / 1.17e-16 | 1.94e-16 / 7.77e-16 | 3.50e-16 / 1.23e-15 |
| einsum @ 4096 rows | 7.41e-16 | 1.55e-15 | 2.28e-15 |
| **einsum @ 64 rows** | **7.80e-17** | **3.89e-16** | **3.50e-16** |

**A = 200 000 x 120** (the scaling check)

| block | G | rhs | coefficients |
|---|---|---|---|
| family best / worst | 1.46e-16 / 1.09e-15 | 1.96e-15 / 1.36e-14 | 2.04e-15 / 1.38e-14 |
| einsum @ 4096 rows | 2.18e-16 | 3.44e-15 | 3.28e-15 |
| **einsum @ 64 rows** | **7.28e-17** | **3.13e-16** | **3.20e-16** |

At 4096 rows the transplant is worse than the family's WORST draw on two of
the three shapes -- refutation 3.  At 64 rows every quantity lands at or
below the family's BEST draw, and the time cost of the shorter block is
1.15x against 128 and 1.33x against 4096 (0.6375 s vs 0.5405 / 0.4790 at
200 000 x 120) -- against a solve that is ~1% of the call.  Accuracy wins that
trade without argument.  The sweep is flat and monotone in accuracy from 32
to 4096 rows, so 64 is a choice on a slope, not a cliff; it is pinned, and
`test_the_block_length_is_what_makes_that_true` asserts the fail-before by
re-running the oracle at D14's block length.

### 5.2 A deterministic replacement for the QR step-down

`_det_refine(A, b, x, small)` -- `_DET_REFINE_STEPS = 1` step of iterative
refinement on the deterministic normal equations:

```
x  <-  x + G^-1 A^T (b - A x)
```

with `A x` through `_det_matvec` (an `optimize=False` einsum, so the
contraction over the TERM axis is as scheduling-free as the one over rows) and
`A^T r` through `_det_at_b` (the same block partition and the same pairwise
tree as the Gram; asserted bit-identical to `_det_normal_equations(A, b)[1]`).
`small` is the caller's already-built `M x M` Cholesky, reused rather than
re-factorised.

It is the textbook cure for the `cond(A)^2` loss and it is valid here by a
wide margin -- `1/rcond` = 1e10, `cond(A)^2 eps` ~ 2e-6, far inside the
convergence condition.

**IT CANNOT BE SCORED AGAINST THE QR, and that is a design constraint, not a
shortcut.**  C13 picks between two candidates on `||b - A x||`.  `r_qr` moves
with the thread count, so a deterministic route that BRANCHES on it is not
deterministic -- the nondeterminism only moves from the value into the
choice.  So the refinement is taken unconditionally where it converges, and
what it owes instead is S6.2's measurement.

### 5.3 ...which must be able to REFUSE

`_DET_REFINE_MAX_CORRECTION = 1e-3`.  If the correction is not small relative
to the answer, `_det_refine` returns `None`, the solve falls back to the
threaded QR, and the caller is warned that the guarantee lapsed -- D14's hole,
kept for exactly the systems that need it.  There is a cheap second half:
`_gram_rcond(G) == 0.0` means the EQUILIBRATED Gram has a non-positive
eigenvalue, so the Cholesky the correction would be applied through has no
meaning even where LAPACK accepted the raw matrix, and the route does not even
try.  Both conditions are computed from deterministic quantities.

---

## 6. WHAT THE REPLACEMENT MEASURES

### 6.1 Without refinement, the deterministic answer is three decades worse

`p12_refine_guard.py`, `resid(x) / resid(x_qr)` on the traced fits' OWN
matrices:

| fit | unrefined normal equations | **refined** |
|---|---|---|
| `(119936, 5)` carrier | 0.9703x | 0.9701x |
| `(1457, 28)` x3 | 1x | 1x |
| `(1337, 120)` | **1843x** | **0.9154x** |
| `(1337, 120)` | **2138x** | **0.7835x** |
| `(23573, 120)` | **1802x** | **0.8433x** |
| `(23573, 120)` | **1681x** | **0.8757x** |

Refinement is not polish on this path.  Without it, D15 would have shipped an
answer three decades worse than the route it replaces.

### 6.2 With refinement, it is at least as good as the QR everywhere

`p09_stepdown.py`, `||b - A x||`, lower is better:

| fit | QR | refined | verdict |
|---|---|---|---|
| `(24065, 28)` | 1.625695382470e-08 | 1.625695382488e-08 | 1.1e-11 rel worse |
| `(24065, 28)` | 1.625695382650e-08 | 1.625695382371e-08 | better |
| `(24065, 28)` | 9.219562469059e-10 | 9.219562469490e-10 | tie (and the shipped screen left this one on the RAW normal equations at 9.2195670e-10) |
| `(23573, 120)` | 9.941051543201e-16 | **8.445324889519e-16** | **15% better** |
| `(23573, 120)` | 1.024192624649e-13 | **1.023040848567e-13** | better |
| `(1337, 120)` | 1.620049952648e-16 | **1.433667961365e-16** | **11% better** |
| `(1337, 120)` | 2.777621808774e-14 | **2.360288195237e-14** | **15% better** |

Worst case, the refined answer is **1.0e-9 relative** worse than the QR --
three decades inside C13's own `_LSTSQ_RESID_MARGIN` of 1e-6, which is the
margin a candidate must BEAT to displace an incumbent.  That is the bar
`test_the_refined_step_down_fits_at_least_as_well_as_the_qr_it_replaces`
asserts, and it is C13's own, not one invented here.

A second refinement step moves the coefficients by ~2e-13 and the residual not
at all -- refinement has already converged, so `_DET_REFINE_STEPS = 1`.

### 6.3 The refusal corridor, measured on both populations

`max|d| / max|x|` after one step:

| population | value |
|---|---|
| MUST REFINE -- `(119936, 5)` carrier | 1.6e-16 |
| MUST REFINE -- `(1457, 28)` / `(24065, 28)` | 1.3e-10 .. 3.2e-08 |
| MUST REFINE -- `(1337, 120)` / `(23573, 120)` | 4.1e-08 .. **9.8e-08** |
| (harmless either way) column perturbed 1e-6 | 3.0e-04 |
| MUST REFUSE -- column perturbed 1e-9 | **1.007** |
| MUST REFUSE -- column perturbed 1e-12 | **0.999** |
| MUST REFUSE -- niche-D7 hard mask (`rcond` exactly 0.0) | diverges; fit misses by 1.4e5x |

Four decades of corridor below the bar and three above it.  1e-3 sits at the
top of it, which also keeps the 1e-6-perturbed C13 fixture on the refined
route -- and that one is harmless either way (its refined residual is 1.000x
the QR's).

**THIS IS THE ONE A TEST SUITE FOUND, NOT THIS LAYER'S OWN TESTS.**  The
first cut refined wherever the screen fired and passed all 19 D15 tests;
`test_niche_d7::test_c13_cures_the_hard_mask_fold_at_the_d7_order` failed with
`137106.440151x` where it wanted `<= 1.001`.  Recorded because reading the
code would not have caught it: the hard-mask Gram FACTORISES (LAPACK accepts
it) and only the equilibrated spectrum shows it is gone.

---

## 7. WHAT MUST NOT MOVE, AND WHAT DID

### 7.1 The carrier fit and the analytic path: TWO-TREE BYTE-IDENTICAL

`p08_twotree.py`, run once in the 5.41.0 tree
(`D:/.../Lumenairy` @ `a33d69a`) and once in this worktree, both with
`PYTHONPATH` pinned and `lumenairy.__file__` asserted, output diffed:

```
n=119936    M=5   K=1  G=a1d43591d51056b2 rhs=ca18e55fa50ae1b1
n=67348     M=5   K=1  G=ac9c24c52e010ea3 rhs=3580af39066f3654
n=30000     M=6   K=1  G=756e74b7c37bad87 rhs=1ccd48d75c746eb2
n=5000      M=7   K=2  G=4f9e6a8d725a2b20 rhs=b536c6847b2a6692
n=1800000   M=5   K=1  G=e19b4b9c921d8621 rhs=43de83e1d9115726
n=4000      M=4   K=3  G=3f3eb0ce7069cf39 rhs=b496c34317ce1b63
n=999       M=2   K=1  G=6c24d2ecbb0afa26 rhs=954335613aecfded
TWO-TREE IDENTICAL
```

Every term count below `_DET_EINSUM_MIN_TERMS` is bit-for-bit what 5.41.0
shipped.  `test_the_analytic_entry_is_inert_across_this_flag` asserts the
same thing at the FIELD, both arms in one process.

### 7.2 The C1/C6/C8/C9 contracts: two-tree in shape, so they survive

**Adjudicated by reading what they actually assert, not by assuming.**  Every
byte-identity assertion in `test_niche_c1_consolidation.py`,
`test_niche_c6_fit_guard.py`, `test_niche_c6_stationary_phase_launch.py`,
`test_niche_c8_inverse_support_bound.py` and
`test_niche_c9_sphere_parab_exact_conversion.py` is
`np.array_equal(a, b)` between **two arms computed in the same process on the
same build** -- a null-decentre against the concentric path, a flag on against
the flag off, a core region against its own reference.  There is not one
stored golden array, hash constant or `.npy` in the five files.  Such a
contract is invariant under any numerics change that moves BOTH arms
identically, which is what a summation-order change does.

Measured rather than argued: all five suites pass unchanged (S8).  **No
C1/C6/C8/C9 assertion was weakened, restated or deleted.**

### 7.3 What DID move, declared

Three test files needed work, and none of them is a byte-identity contract:

* **D14's scope pin**, `test_the_traced_chain_has_a_second_reduction_this_
  niche_does_not_own`.  It asserted "exactly ONE solve on this path is
  deterministic".  RESTATED, not deleted: it now pins the BOUNDARY BETWEEN
  THE TWO FLAGS -- with `DETERMINISTIC_TRACED_FIT` off, D14's flag still gates
  the carrier fit and only the carrier fit (so `DETERMINISTIC_NORMAL_EQUATIONS
  = False` stays a clean fail-before for 5.41.0's bits); with it on, NONE of
  the six is left on the BLAS route.  Its docstring carries both of D14's
  wrong sentences (refutations 1 and 2) rather than quietly dropping them.
* **D14's step-down pin**, `test_the_step_down_hole_is_declared_rather_than_
  hidden`.  Its C13 arm is INVERTED (the guarantee no longer lapses there, so
  it must not be declared to); its rank-deficient arm is unchanged.  Same
  engineered fixtures, same `rcond` precondition assertions.
* **Two niche-D1 era-pinned witnesses.**  These reconstruct the pre-D1 library
  to reproduce a historical ghost, and already pin `DECENTRED_FIT_PREDICTOR`,
  `DECENTRED_FIT_ARBITER` and `LSTSQ_CONDITIONING_STEPDOWN` to their old
  values with a comment saying each one ALONE suppresses the witness.
  `DETERMINISTIC_TRACED_FIT` is a THIRD independent cure -- it acts a step
  earlier than C13's, forming the hard-mask Gram through a pairwise tree
  instead of a threaded reduction, and the null-space draw does not survive it
  (the ghost lobe collapses from 6.737 mm to 1.200 mm against a 1.400 mm
  bar).  Added to the era pin with that measurement in the comment.  No
  assertion changed.

Four test-double signatures also grew `**kw`, because the real solver takes
`deterministic=` and a stub that refuses the caller's keyword tests the stub:
`test_niche_c13` (1), `test_fix_newton_pool_memory` (3),
`test_niche_d7_decentred_fit` (1).  Same fix 5.41.0 already made to the R3
stub.

### 7.4 How far the traced field moved

Cold calls, one per process, `ray_subsample=2` fixture, saved and differenced
(`p11_cold.py`); field peak 1.025:

| quantity | value | relative |
|---|---|---|
| shipped cross-width spread | 1.341e-12 | 1.308e-12 |
| **deterministic cross-width spread** | **0** | **0** |
| \|deterministic - shipped\| | 8.50e-13 .. 1.294e-12 | 8.29e-13 .. 1.262e-12 |

**The deterministic answer lands INSIDE the shipped route's own cross-width
spread** -- it is an answer the shipped route could have returned on some
thread count.  The NaN support pattern is identical on all eight arms.

**On fixtures where the shipped route does NOT split, the difference is
larger and it is not a reordering.**  At N = 1024 the shipped field is
width-invariant and the deterministic one differs by 1.9e-10 relative,
because the C13 refinement genuinely moves the answer: the shipped screen
left one 28-term fit on the raw normal equations at 9.2195670e-10 residual
and the refinement takes it to the QR's own 9.2195625e-10 (its coefficients
move from 3.2e-08 to 5.3e-14 of the QR's).  That is an accuracy IMPROVEMENT,
stated plainly rather than folded into an "inside the noise" claim it does
not satisfy.  It sits ~6 decades below anything a physical readout resolves,
and it is the same order as the step-down C13 already performs.

---

## 8. COST, AND THE DEFAULT DECISION

### 8.1 The solves are ~1% of an `apply_real_lens_traced` call

Measured with the solver instrumented in-call (`p04`, `p11_cold`):

| fixture | whole call | all solves | fraction |
|---|---|---|---|
| N=512, default subsample | 1.96 s | 0.016 s | **0.83%** |
| N=1024, default subsample | 2.54 s | 0.023 s | **0.91%** |
| N=512, `ray_subsample=2` (six solves) | 2.34 s | 0.156 s | **6.7%** |

That is the number the brief asked for and it settles the default before any
kernel is chosen: even a 5x on the solve is 5% of the call at worst, and 4%
at the default shape.

### 8.2 Interleaved cold A/B at the worst measured shape

One cold call per process (the inverse map is CACHED after the first call, so
an in-process repeat measures the second call, not the build), alternated
OFF/ON/OFF/ON/OFF/ON at OMP = 8:

| | OFF | ON |
|---|---|---|
| whole call, best of 3 | 2.339 s | 2.377 s |
| solves only | 0.156 - 0.170 s | 0.207 - 0.219 s |

**Whole-call overhead +1.6%**, solves +28%.  At the default-subsample shape
the whole-call overhead is under the run-to-run noise (measured -0.9% to
+0.5% over repeats, i.e. not resolvable).

**Default ON**, comfortably under the brief's 10% bar and under D14's own 5%
one.

### 8.3 Footprint

Same instrument as D14 / BUILD_LENS_32K_MEMORY (`tracemalloc` peak, quoted in
float64 grids), whole traced call, one thing changed:

| fixture | det OFF | det ON | delta |
|---|---|---|---|
| N=512, default | 30.283 grids (63.5 MB) | 30.282 grids (63.5 MB) | -0.001 |
| N=512, `ray_subsample=2` | 66.655 grids (139.8 MB) | 66.655 grids (139.8 MB) | 0.000 |
| N=1024, default | 25.266 grids (211.9 MB) | 25.140 grids (210.9 MB) | **-0.126** |

Zero or better, for a structural reason: the einsum partial reads its block as
a VIEW of the design matrix -- no transposed tile, no row buffer, where D14's
partial allocates both -- and the carry-stack holds `O(log2(n / blk))` partial
Grams rather than `O(n / blk)`.  `test_the_traced_fit_footprint_is_not_
regressed` asserts it against an allowance DERIVED from the constants in
force, not a pinned number.

### 8.4 The one scaling caveat, declared

The einsum partial is ~5.7x the BLAS route's time at 200 000 x 120 and the
ratio grows with the term count, so a fit far larger than the traced chain's
would cost proportionally more.  `_lens_imap`'s own `GRAM` guard already
refuses the largest such build (the `test_niche_c1` exit-NA case,
`n_good` = 5 764 801 at 120 terms) before any solve happens, and
`DETERMINISTIC_TRACED_FIT = False` is the escape hatch if one ever gets
through.

---

## 9. THREAD-INVARIANCE EVIDENCE

Eight fixtures x four widths, one fresh interpreter per arm, the width pinned
in `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` /
`NUMEXPR_NUM_THREADS` before NumPy loads, each child asserting its own
`lumenairy.__file__` against `LUMENAIRY_ROOT`:

| fixture | shipped, distinct field hashes | deterministic |
|---|---|---|
| N=384 w60 R=1 | 1 | **1** |
| N=512 w80 R=1 | 1 | **1** |
| N=512 w120 R=1 | 1 | **1** |
| N=512 w80 R=0.5 | 1 | **1** |
| N=512 w80 ap=10mm | 1 | **1** |
| N=768 w120 | 1 | **1** |
| N=640 w100 R=2 | 1 | **1** |
| **N=512 `ray_subsample=2`** | **4** | **1** |

Plus `test_the_traced_exit_field_hashes_the_same_at_every_thread_count`: four
widths x two repeats = eight calls, one hash, asserted UNCONDITIONALLY.

---

## 10. THE TESTS

`tests/unit/test_niche_d15_deterministic_traced_fit.py` -- **20 tests**.

| test | claim |
|---|---|
| `test_the_traced_exit_field_hashes_the_same_at_every_thread_count` | THE NEW CLAIM: 1 hash over 4 widths x 2 repeats, unconditional |
| `test_the_shipped_route_is_the_fail_before_on_this_fixture` | matched pair; `xfail`s with the evidence if this build never splits |
| `test_einsum_is_blas_free_and_optimize_true_is_not` | the determinism argument for the partial, and the `optimize=True` refutation |
| `test_the_partial_route_reads_the_term_count_and_nothing_else` | dispatch and block length are shape functions only |
| `test_the_gram_is_symmetric_in_both_partials` | the Cholesky never sees disagreeing triangles |
| `test_the_rhs_only_reduction_is_the_same_reduction` | `_det_at_b` == `_det_normal_equations[1]`, bit for bit |
| `test_the_einsum_partial_is_at_least_as_close_to_the_truth[3 shapes]` | oracle non-regression vs the legal-partition family |
| `test_the_block_length_is_what_makes_that_true` | the 64-row constant is load-bearing, as a fail-before |
| `test_the_screened_singular_exit_no_longer_leaves_the_deterministic_route` | refutation 2, as an assertion |
| `test_the_rank_deficient_hole_is_still_declared` | what is LEFT of D14's hole |
| `test_the_refinement_refuses_where_it_does_not_converge` | refutation 4, both sides of the corridor |
| `test_the_refined_step_down_fits_at_least_as_well_as_the_qr_it_replaces` | the claim the replacement earns, on the fits' own matrices |
| `test_every_least_squares_solve_on_the_traced_path_is_deterministic` | the scope, from the other side |
| `test_the_flag_off_returns_the_shipped_bits_exactly` | the fail-before is a value of the flag |
| `test_the_carrier_fit_keeps_d14s_arithmetic_exactly` | acceptance (6), at the kernel |
| `test_the_analytic_entry_is_inert_across_this_flag` | acceptance (6), at the field |
| `test_multi_rhs_matches_the_single_rhs_column_by_column` | the 3-column exit fit is the same reduction |
| `test_the_traced_fit_footprint_is_not_regressed` | derived allowance |

The module docstring states the cross-build non-claim in block capitals, as a
COMMENT and not an assertion, per `docs/TESTING_STANDARDS.md`: the property is
false and a test encoding it would be the S4 floor-bar shape.

---

## 11. FILES

```
lumenairy/elements/_lens_traced.py
    + DETERMINISTIC_TRACED_FIT, _DET_EINSUM_MIN_TERMS,
      _DET_EINSUM_BLOCK_ROWS, _DET_REFINE_STEPS,
      _DET_REFINE_MAX_CORRECTION
    + _det_partial_ufunc, _det_partial_einsum, _det_at_b, _det_matvec,
      _det_refine
    ~ _det_normal_equations: dispatches on the term count; B normalised
      contiguous at entry
    ~ _solve_lstsq_thread_safe: the C13 exit refines instead of rerouting
    ~ _warn_det_stepdown docstring: the hole shrank to one exit
    ~ 4 call sites pass deterministic=bool(DETERMINISTIC_TRACED_FIT)
lumenairy/elements/_lens_imap.py
    ~ build_inverse_map's two solves pass deterministic=_det_traced()
lumenairy/elements/_traced_flags.py     + 5 registry rows (layer D15)
docs/audits/TRACED_LAYER_MAP.md         + rows 37-41
tests/unit/test_niche_d15_deterministic_traced_fit.py   NEW, 20 tests
tests/unit/test_niche_d14_deterministic_carrier_fit.py  2 tests RESTATED
tests/unit/test_niche_d1_tilted_carrier.py              2 era pins extended
tests/unit/test_niche_c13_lstsq_conditioning.py         1 stub signature
tests/unit/test_fix_newton_pool_memory.py               3 stub signatures
tests/unit/test_niche_d7_decentred_fit.py               1 stub signature
validation/probe_traced_det/                            NEW, 12 probes
CHANGELOG.md                            [Unreleased]
```

---

## 12. SUITES

The regression set is not "everything"; it is every test file in the tree that
touches the traced chain's least-squares fits, found by
`grep --include='*.py' -rl "apply_real_lens_traced\|_solve_lstsq_thread_safe\|build_inverse_map\|_Cheb2DEvaluator\|_fit_residual_eikonal" tests/`
(94 files), run in batches, plus the modules the change edits.

| gate | result |
|---|---|
| `test_niche_d15_deterministic_traced_fit.py` (NEW) | **20 passed** |
| C1/C6/C8/C9/C11/C12/C13/C14/C15 + D1 + D7 + D14 + R6/R7 + newton-pool | **420 passed, 1 skipped** |
| `test_audit_lens` + `lens_memory_levers` + `hammer_h6` + `hammer_h3` + `obl_banded_halo` + `lens_chunked_sag` + `tangent_facet` | **378 passed** |
| `niche_audit_w3_*` + `d9_grid_origin` + `p11` + `p1_traced_tiltaware` + `p4_gbd_reexpand` + `r3_gbd_mem_lstsq` + `s10_sibling_patterns` | **187 passed** |
| `ruff check lumenairy/ tests/` | **All checks passed** |

**1005 tests across the traced-fit surface, 0 failures.**

Not run: the whole `tests/unit` tree, and `test_audit_lens_models_2026_07.py`,
which did not finish inside a 10-minute budget on this box.  Said plainly
rather than implied by omission.

### 12.1 Reproducing the evidence

```bash
export LUMENAIRY_ROOT=C:/tmp/lum_dt PYTHONPATH=C:/tmp/lum_dt
# thread-invariance, one arm per process, width pinned BEFORE numpy loads
for T in 1 2 4 8; do
  OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T \
  NUMEXPR_NUM_THREADS=$T python validation/probe_traced_det/p05_field_scan.py 1
done
# the whole-call A/B, one COLD call per process (the inverse map is cached)
for D in 0 1; do
  OMP_NUM_THREADS=8 python validation/probe_traced_det/p11_cold.py 512 2 $D
done
```

`PYTHONPATH` alone would not have pinned the library (BUILD_LENS_32K_MEMORY
S9): every child asserts `lumenairy.__file__` against `LUMENAIRY_ROOT` and
refuses rather than reporting numbers from the wrong tree.

---

## 13. OPEN

* **A latent `n == 0` guard in `_det_normal_equations` is dead code, and it
  predates this work.**  `B.reshape(B.shape[0], -1)` raises on an empty `b`
  before the `if n == 0` branch below it can return the zero Gram it was
  written to return.  Not fixed here: it is D14's line, no caller reaches it,
  and touching it would move a released function for no measured reason.
  Recorded so the next person does not rediscover it as a new bug.
* **The 28-term traced Chebyshev fits screen singular at rcond 1.6e-9** on
  every fixture measured.  That is two orders under the C13 screen and it is
  worth asking whether the basis or its weighting should be conditioned
  rather than re-solved -- a question for the C13 layer, not this one.
* **`_gram_rcond` runs `eigvalsh` on the `M x M` Gram**, which is in
  principle a threaded LAPACK call, and its output selects a branch.  Measured
  identical to 16 digits at OMP 1/4/8 on every traced fit, and the branch
  boundary sits two decades away from any measured value, so nothing here
  turns on it -- but it is the last unowned reduction on the path and it is
  named rather than left implicit.

---

## 14. FOLLOW-UP -- the 5.42.0 gate, and the sweep that missed it

Branch `fix/d15-followup` off `origin/main` @ `b2e6344` (D15, merged).

The 5.42.0 `Unit tests + lint` gate failed on all four CI pythons with three
deterministic failures that S12's 1005-test sweep did not reach.  Both causes
are tests calibrated to the unrefined solve; neither is a defect in the
shipped library.

### 14.1 THE PROCESS FAILURE, FIRST, BECAUSE IT IS THE REUSABLE PART

S12 says the regression set was "every test file in the tree that touches the
traced chain's least-squares fits, found by `grep ... (94 files)`".  **The
grep was right and the run was not.**  The 94-file list was built, written to
`validation/probe_traced_det/scope.txt`, and then RUN THROUGH A NAME FILTER
(`grep -E "niche_c1_|niche_c6_|..."`) that selected roughly a third of it.
`test_fix_runner_oom_2026_08_13.py` was in the list from the start, matched no
batch pattern, and was never executed; `test_niche_d3_guards.py` likewise.

A scope that is derived correctly and then sampled is not the scope -- it is a
sample wearing the scope's name, and S12 reported it as the former.  The
follow-up ran all 94 files, split four ways, with no name filter (S14.5).

### 14.2 FAILURES 1 AND 2 -- the two arms were solved differently

`test_fix_runner_oom_2026_08_13.py::test_the_forward_fit_coefficients_are_bit_identical[False]`
and `[True]`: "the forward fit moved bits: max |diff| = 5.954e-19 / 2.840e-16".

That file's claim is about the DESIGN MATRIX LAYOUT -- `_Cheb2DEvaluator`
builds its design C-contiguous where it used to build the transpose of an
`(n_terms, n)` buffer -- and it is a same-process two-arm comparison against a
retired reference construction whose own docstring says it uses "the same keep
mask, **the same solver**".  It hard-coded `_solve_lstsq_thread_safe(A, rhs)`
back when the solver had one route.  D15 gave it two, the live evaluator took
the new one, the reference stayed on the old default -- so the comparison
began reading a SOLVER difference as a LAYOUT one.

ADJUDICATION: fix the arms, not the claim.  The reference now takes
`deterministic=None`, meaning "whatever the live evaluator would do", so the
two cannot drift apart by inheriting a default again.  Measured at BOTH flag
states and on both branches, the layout claim holds exactly:

| weighted | flag | vs reference at the SAME state | vs reference at the OLD default |
|---|---|---|---|
| False | off | identical, 0.0 | identical, 0.0 |
| False | **on** | **identical, 0.0** | differs, 6.624e-19 |
| True | off | identical, 0.0 | identical, 0.0 |
| True | **on** | **identical, 0.0** | differs, 1.130e-16 |

STRENGTHENED WHILE FIXING: the test now sweeps `DETERMINISTIC_TRACED_FIT` as a
second parameter, so its four cases pin C-contiguous == transposed-buffer bits
on the BLAS route (the original claim, unchanged) AND on the deterministic one.

Worth stating because a reader chasing this will look for it: the fit does
**not** reach the C13 screen (Gram rcond 6.8e-02 unweighted, 3.4e-05 weighted,
against the 1e-8 screen), so D15's refinement never runs here.  The only thing
that moved is the order the Gram is summed in.

### 14.3 FAILURE 3 -- a bar whose boundary was already inside its own spread

`test_niche_d7_decentred_fit.py::test_a_shrunken_basis_domain_is_a_liability_outside_itself`,
on py3.12 and py3.13 only: `corner` 2.823e-13 against a `1e3 x inside` bar of
1.110e-12.

The test builds two mathematically identical order-12 fits -- one on the
launch square, one re-mapped onto a small off-centre disc -- and requires them
to agree inside the tight domain and diverge outside it.  Both screen singular
(Gram rcond 6.33e-10, two decades under the C13 screen), so D15's refinement
is live on both.

**The claim still holds and is still unconditional.**  Refinement does not fix
a shrunken DOMAIN, only the residual.  What was wrong is the BAR.  In ULP of
the function's own magnitude at the evaluation points (0.5387, so 1 ULP =
1.196e-16):

| build / arm | inside | corner | ratio |
|---|---|---|---|
| Windows py3.14 / np2.4.4, D15 off | 1.86 | 36 486 | 19 650 |
| Windows py3.14 / np2.4.4, D15 on | 1.86 | 11 633 | 6 267 |
| WSL py3.12 / np2.4.6, D15 on | 1.86 | 11 633 | 6 267 |
| Linux CI py3.12 + py3.13, D15 on | 9.28 | 2 360 | **254** |

Both quantities sit at the float64 noise floor and both are build-dependent,
so their RATIO compounds the jitter: it spans **77x** across three readings of
the same claim, and the 1e3 bar sits inside that span.  That is the
`TESTING_STANDARDS` S4 shape -- a pass/fail boundary living inside the
measurement's own spread -- and it was true BEFORE D15.  D15 only shrank
`corner` by 3x (the refined solve is closer to the true least-squares answer,
so two parameterisations of it agree better) and pushed over a test that was
already standing on the edge.

Note the WSL row.  py3.12 on this box reproduces the WINDOWS numbers exactly,
not the CI ones, so the spread is a **BLAS / CPU-dispatch** fact and not a
python-version one.  A restatement tuned to "py3.12" would have been tuned to
the wrong variable.

RESTATEMENT, in two parts so nothing is lost:

* the SHIPPED-configuration claim keeps its place and stays unconditional,
  with both bars taken against a DERIVED scale -- one ULP of the function
  value -- so neither divides by the other's noise: `inside <= 50 ULP` and
  `corner >= 300 ULP`.  Margins on the WORST of the four readings: **5.4x**
  and **7.9x**; on the best, 27x and 122x.  The fail-before is MEASURED, not
  estimated: run the same fixture with the second basis not shrunken at all
  (`a = 1, b = 0`) and the two fits come out bit-identical everywhere --
  `inside` 0.00 ULP, `corner` **0.00 ULP** -- so the corner assertion fails
  outright.  The bar detects the shrunken domain, not merely the presence of
  arithmetic.
* the original 1e3-contrast figure is kept at FULL STRENGTH on a new
  era-pinned witness, `test_the_shrunken_basis_liability_at_its_calibrated_magnitude`,
  which pins `DETERMINISTIC_TRACED_FIT = False` -- the solver it was
  calibrated on, where it reads 19 650 and passed on every CI python before
  D15 landed.  Same pattern as this file's two ghost witnesses and
  `test_c13_cures_the_hard_mask_fold_at_the_d7_order`.

Nothing is marked `xfail` and nothing is deleted.

### 14.4 CLOSING THE CLASS

`grep -rniE "bit_identical.*fit|fit.*coefficients|coeffs.*identical"` over
`tests/`, plus every `_solve_lstsq_thread_safe(` call in the tree that does
not pass `deterministic`:

| site | adjudication |
|---|---|
| `test_fix_runner_oom::_retired_cheb_coeffs` | THE defect -- a reference arm on the old default.  Fixed (S14.2) |
| `test_niche_c13_lstsq_conditioning` (11 calls) | direct tests OF the solver at its default; both sides of every comparison are the same call.  Correct as written, all pass |
| `test_niche_r3_gbd_mem_lstsq` (3 calls) | same -- GBD's own lstsq, not a traced fit.  Pass |
| `test_niche_newton_pool_both_fits` (2 bit-identity tests) | pool vs serial, BOTH live library paths, so both carry the flag.  Pass |
| `test_niche_audit_w3_oracles::test_w4_t3_jax_fit_matches_numpy_in_coefficients_and_evaluation` | cross-backend, tolerance-based, outside the traced-fit scope.  Pass |
| `test_niche_d3_guards` (fit-coefficient perturbation) | in the 94-file scope, missed by S12's name filter.  Run, passes |

`_retired_cheb_coeffs` is the only reference-arm stub in the tree that
hard-codes a solver default and is compared against a live library call.  The
class is one member wide.

### 14.5 SUITES -- the full scope this time, and both mounts

All 94 files from the S12 grep, split four ways with NO name filter, plus the
D14/D15 suites on both mounts.  The WSL mount runs py3.12 / numpy 2.4.6, the
same python major.minor the 5.42.0 gate failed on.

| gate | result |
|---|---|
| the three failing tests + `test_niche_d3_guards` (Windows py3.14) | **95 passed** |
| full scope batch 1/4 (25 files) | **1070 passed, 1 skipped** |
| full scope batch 2/4 (22 files) | **601 passed** |
| full scope batch 3/4 (24 files) | **434 passed, 1 skipped**, 1 failed -- S14.6 |
| full scope batch 4/4 (23 files) | **421 passed, 2 deselected** |
| `test_fix_runner_oom` + `test_niche_d7_decentred_fit` (WSL py3.12 / np2.4.6) | **54 passed** |
| D14 + D15 suites (Windows py3.14 / np2.4.4) | **37 passed** |
| D14 + D15 suites (WSL py3.12 / np2.4.6) | **37 passed** |
| `ruff check lumenairy/ tests/` | **All checks passed** |

**2526 tests over the full 94-file scope**, plus 37 x 2 on the deterministic
suites across two mounts.  The one failure is characterised in S14.6 and is
not reachable from this change.

### 14.6 ONE FAILURE THAT IS NOT THIS CHANGE'S, AND IS NOT CALLED A FLAKE

`test_niche_p4_gbd_reexpand.py::test_frame_completeness_metric_published`
failed once, in full-scope batch 3, while FIVE pytest processes were running
concurrently on this box.  It is recorded here rather than dismissed, and it
is NOT closed.

WHY IT CANNOT BE D15 OR THIS FOLLOW-UP, by construction rather than by
retrying until green:

* the failing test exercises `apply_real_lens_gbd`'s re-expansion, and
  **`_solve_lstsq_thread_safe` does not exist anywhere in the GBD module** --
  `grep` over `lumenairy/` puts every call site in `_lens_traced.py` and
  `_lens_imap.py` only.  D15 changed that function and its callers; the GBD
  re-expansion reaches neither.  The file is in the 94-file scope only because
  it mentions `apply_real_lens_traced` once, on line 16;
* neither file this follow-up edits is in batch 3 at all;
* it passed in the 5.42.0 CI gate on all four pythons -- it is not one of the
  three failures this branch is answering.

WHAT WAS TRIED, AND WHAT IS STILL UNKNOWN:

| arm | result |
|---|---|
| the single test alone | passed (7.8 s) |
| the whole `test_niche_p4_gbd_reexpand.py` file alone | 14 passed |
| batch 3's ordered 11-file prefix up to and including p4 | 253 passed, 1 skipped |
| batch 3 entire, rerun under a deliberate 4-way concurrent load -- the SAME condition it failed in | **435 passed, 1 skipped** |

Five arms, including a deliberate reconstruction of the five-process load it
failed under, and it did not recur.  The counts reconcile exactly (436 items
either way: 434 passed + 1 failed + 1 skipped, against 435 passed + 1
skipped), so nothing was silently deselected between the two runs.

**THE INSTRUMENTATION LESSON, WHICH IS THE SECOND ONE THIS FOLLOW-UP OWES.**
The batch commands piped pytest through
`grep -E "^(FAILED|ERROR)|passed|failed"`, which keeps the `FAILED` summary
line and DISCARDS the assertion above it.  So the one failure in 2526 arrived
without the one piece of information needed to adjudicate it, and three
re-runs were spent trying to recover what a wider `grep` would have kept.  A
filter tight enough to be readable is tight enough to throw away the evidence;
`-rA` or an unfiltered log file belongs on any batch whose failures matter.

**IT STAYS OPEN, AND IT IS NOT BEING CALLED A FLAKE.**  One unreproduced
failure is not evidence of correctness; it is evidence that the instrument
was not watching.  Re-running until green is exactly the move
``feedback_flaky_is_bad_math`` forbids, and five green arms after one red one
do not retire the red one -- they only say the trigger is not file ordering
and not this box's load alone.  The candidate mechanism is
resource pressure (the box was running five pytest processes at
`OMP_NUM_THREADS=8` each), and the two assertions that could plausibly move
under it are `diag['frame_completeness'] > 0.99` and the
`np.array_equal(E_a, E_b)` byte-identity between a diagnosed and an
undiagnosed call.  The second would be the interesting one -- a returned field
that depends on whether a diagnostics dict was passed -- and it is the reason
this is not being written off.

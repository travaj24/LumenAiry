# BUILD -- the `carrier='auto'` fit, deterministic by construction at any thread count

Branch `feat/deterministic-carrier-fit`, rebased onto `origin/main` @
`9a24d76` (the 5.40.0 release; the work was cut from its parent `7bd61372`
while the release gate was still running) -- worktree `C:/tmp/lum_dg`.
Follows
`docs/audits/BUILD_LENS_32K_MEMORY_2026_08_22.md` S4.2, which caught the
defect and left it open.

```
Windows 11 Pro 10.0.26200        AMD Ryzen 9 5950X   137.4 GB (104 GB free)
python 3.14.6   numpy 2.4.4      scipy 1.17.1
scipy-openblas 0.3.31.188.0      USE64BITINT DYNAMIC_ARCH NO_AFFINITY
                                 Haswell  MAX_THREADS=24
lumenairy 5.40.0+ (worktree C:/tmp/lum_dg)
```

---

## 0. VERDICT

`carrier='auto'` is now **bit-reproducible on one build at any BLAS thread
count and over any number of repeats**, and it is so BY CONSTRUCTION rather
than by staying below somebody's threading threshold.  Default **ON**
(`DETERMINISTIC_NORMAL_EQUATIONS = True`), because the whole-fit overhead at
the production shape is **+3.3%** and the accuracy goes UP, not down.

Four things came out of the work that were not in the brief, and three of them
are refutations:

1. **THE ATTRIBUTION IN S4.2 IS WRONG, AND MEASURABLY SO.**  It named
   `G = A.T @ A`.  At the carrier fit's shape (`M` = 5 terms) the GEMM is
   width-invariant at every row count up to 1.6e7 on this build; it is
   `rhs = A.T @ b` -- the GEMV -- that splits the reduction, from 100 000 rows
   up, giving three distinct values over widths {1,2}/{4}/{8}.  Both products
   go through the deterministic kernel anyway, because *which* of the two
   splits is itself a build fact.
2. **`threadpoolctl` CANNOT BE THE MECHANISM** in this library.  It is an
   OPTIONAL dependency (`rcwa/_core.py::_threadpoolctl_available` exists
   precisely because it is often absent), so a per-block `gemm` under a cap of
   1 would be a guarantee that silently evaporates on a box without it.  The
   shipped kernel uses no BLAS at all in the reduction.
3. **THE TRACED CHAIN HAS A SECOND, INDEPENDENT NONDETERMINISTIC REDUCTION**
   that this niche does not own.  `apply_real_lens_traced(carrier='auto')`
   runs SIX least-squares solves; two 120-term residual-eikonal fits stay on
   the BLAS route by design and were measured moving with the width.  The
   end-to-end traced FIELD is therefore NOT claimed thread-invariant, and the
   test that would have claimed it was written, measured failing, and
   rewritten.  See S7.3.
4. The deterministic route is **more accurate**, not less: on the fit's own
   design matrix it beats the BEST legal partitioned draw by 84-162x on the
   recovered coefficients.

**AND THE NON-CLAIM, STATED FIRST BECAUSE IT IS THE ONE THAT GETS
OVERREACHED.**  Nothing here claims cross-build or cross-platform bit
identity, and no test asserts it.  `np.sum`'s pairwise block size, SIMD width
and unroll are properties of the NumPy build and the CPU it dispatched for;
`**`, `sqrt` and `angle` upstream of the fit go through the platform libm.  A
different libm or codegen may legally differ in the last ULP.  The property
delivered is exactly:

> ONE build, ANY thread count, ANY number of repeats -> the same bytes.

---

## 1. THE DEFECT, RE-MEASURED AND RE-ATTRIBUTED

### 1.1 What S4.2 established, and what it got wrong

S4.2's evidence stands: `carrier='auto'` read 2 distinct field hashes over 4
identical N = 4096 calls, and 2 over 8 on shipped 5.39.1; `OMP_NUM_THREADS=1`
restored reproducibility and returned the threaded majority value; a
closed-form carrier showed nothing.  Diagnosis: a threaded reduction, not the
FFT, not the levers, not the field.  All correct.

Its one wrong sentence is the attribution.  Raw products at `M` = 5, one arm
per process, the width pinned in `OMP` / `OPENBLAS` / `MKL` before NumPy
loads (SHA-256 of the float64 bytes, first 8 hex):

| rows | `A.T @ A` w=1 / 2 / 4 / 8 | `A.T @ b` w=1 | w=2 | w=4 | w=8 |
|---|---|---|---|---|---|
| 20 000 | `2cf72c32` x4 | `4d074a6a` | `4d074a6a` | `4d074a6a` | `4d074a6a` |
| 100 000 | `168221d5` x4 | `a0da1988` | `a0da1988` | `5f89650a` | `c78b738c` |
| 200 000 | `2673ccdd` x4 | `d674da5a` | `d674da5a` | `2d857480` | `e2887356` |
| 1 000 000 | `0e5326cd` x4 | `88f8e5e8` | `88f8e5e8` | `def6f4d0` | `533432bb` |
| 1 800 000 | `87f827ce` x4 | `e79b513e` | `e79b513e` | `80faab2c` | `e86efe85` |
| 4 000 000 | `ef22fb5c` x4 | `84fb6cec` | `84fb6cec` | `eba3dfc0` | `b9388d04` |
| 16 000 000 | `cbd9db94` x4 | `cbdf8c50` | `cbdf8c50` | `8eb1a319` | `80434f83` |

**The GEMM never moves; the GEMV moves from 100 000 rows up**, and it does so
in the pattern of an OpenBLAS K-split: widths 1 and 2 agree, 4 and 8 each
differ.  The threshold sits between 20 000 and 100 000 rows.  (At 28e6 rows a
sixteenth arm was added -- `w = 16` gives yet another value, `bdae07f6`.)

That is why a low-order fit could look reproducible in the 8960-arm two-tree
sweep and not in production, and the arithmetic is checkable rather than a
story: that sweep's largest grid is N = 129, whose design matrix can hold at
most `2 * 129 * 128` = **33 024 rows** even at 100% bright -- under the
measured 100 000-row splitting threshold and in the band where the table above
reads one value at every width.

### 1.2 The same thing through the library

`_compute_carrier('auto', ...)` on the S7 fixture, design matrix
**119 936 x 5**, hashes of the fit's own `A.T @ A` / `A.T @ b`:

```
  OMP=1   G 8952d21aa3   rhs 3154f478da
  OMP=2   G 8952d21aa3   rhs 3154f478da
  OMP=4   G 8952d21aa3   rhs a964bd5c69
  OMP=8   G 8952d21aa3   rhs 9dfe6878a9
```

**A FIXTURE NOTE THAT COST AN HOUR AND IS NOW IN THE TEST'S COMMENT.**  The
first version of this fixture used a strong carrier (R = 0.045 m).  Its local
tilt exceeds the grid Nyquist tilt past r = 26 px, so R6's connected-core
restriction correctly threw the support away and the design matrix collapsed
to **1 616 rows** -- below the splitting threshold.  The fail-before then did
not fire, and the arms all agreed.  A fail-before that silently does not fire
reads exactly like a fix.  R = 1.0 m keeps the whole bright support and the
matrix is 119 936 rows.

---

## 2. THE SCHEME

`_det_normal_equations(A, b)` in `lumenairy/elements/_lens_traced.py`.

### 2.1 Three parts, each carrying one requirement

* **Fixed-size row blocks.**  `_det_block_rows(n_terms)` =
  `max(_DET_GRAM_MIN_BLOCK_ROWS, _DET_GRAM_TILE_BYTES // (8 * n_terms))`.
  A function of the TERM COUNT ONLY -- never of the row count, the free
  memory, the thread count or a cache probe.  Any of those would be a
  scheduling input and would put the defect back one level down.
* **A per-block partial that cannot be threaded.**  Each block's `(M, M)`
  partial is built from `np.multiply` and `np.sum` alone.  NumPy's ufunc
  reductions are single-threaded and take no BLAS path at any size, so the
  partial is scheduling-independent BY CONSTRUCTION.
* **A fixed pairwise tree over the blocks.**  A carry-stack merges two nodes
  as soon as they reach equal depth, then folds the leftovers right to left.
  That reproduces the level-by-level pairwise tree of the same block list
  exactly (checked for counts 3, 5, 6, 7 by hand and by test) at `O(log n)`
  live partials instead of `O(n / blk)`.

### 2.2 WHY NOT THE `threadpoolctl` OPTION (a) THE BRIEF OFFERED

The brief's option (a) -- per-block `gemm` under `threadpool_limits(1)` while
parallelising across blocks -- is rejected on a fact about this library, not
on taste: **`threadpoolctl` is optional here.**  `rcwa/_core.py` carries a
whole once-per-process warning (`_warn_blas_cap_uncontrollable`) for the case
where a requested cap is INERT because the package is absent.  A determinism
guarantee whose mechanism is inert on some installs is not a guarantee.  The
NumPy-ufunc kernel needs nothing outside NumPy.

Option (b) -- blocks small enough that the partial falls below the BLAS
threading threshold -- is refused by the brief itself and by S1.1: that
threshold is a build fact (it sits between 20 000 and 100 000 rows *here*), so
sizing to it is sizing to one build.

### 2.3 The transposed tile is a SPEED device, and that is checked

Each block is copied into a contiguous `(n_terms, rows)` tile so the pair
products read contiguous memory.  The products, their order and the `np.sum`
lengths are identical to the same scheme run on `A`'s strided columns, so the
bits are the same -- **measured, and pinned by
`test_the_transposed_tile_is_a_speed_device_and_not_an_arithmetic_one`**:

At the SHIPPED block sizes, best of four, idle box:

| shape | block rows | strided columns | transposed tile | tile buys | same bits |
|---|---|---|---|---|---|
| 28e6 x 5 | 13 107 | 0.626 s | **0.564 s** | 1.11x | **yes** |
| 141 471 x 66 | 4 096 | 0.880 s | **0.465 s** | 1.89x | **yes** |

Bit-identity is the load-bearing half: if the tile changed a bit, the tile
constant would be load-bearing in a second, undocumented way, and the block
size could not be tuned without moving every carrier fit in the library.

### 2.4 The pinned constants

`_DET_GRAM_TILE_BYTES = 1 << 19` (512 KB, an L2-resident tile).  Swept at the
production shape (28e6 x 5) against the BLAS route's 0.230 s:

| tile | block rows | blocks | time | vs BLAS |
|---|---|---|---|---|
| 64 KB | 1 638 | 17 095 | 1.409 s | 6.12x |
| 256 KB | 6 553 | 4 273 | 0.587 s | 2.55x |
| **512 KB** | **13 107** | **2 137** | **0.504 s** | **2.19x** |
| 1 MB | 26 214 | 1 069 | 0.598 s | 2.59x |
| 2 MB | 52 428 | 535 | 0.529 s | 2.29x |
| 4 MB | 104 857 | 268 | 0.545 s | 2.37x |
| 16 MB | 419 430 | 67 | 0.583 s | 2.53x |

Flat to +-10% over three decades above 256 KB, so the pin is not on a cliff.
The 64 KB row is Python loop overhead, and it is why
`_DET_GRAM_MIN_BLOCK_ROWS = 4096` exists: without the floor, a 66-term fit
reads 1.14 s instead of 0.42 s.

### 2.5 Where it is wired, and where it deliberately is not

`_solve_lstsq_thread_safe(A, b, deterministic=False)` gains the keyword.
`_compute_carrier`'s `'auto'` branch passes
`deterministic=bool(DETERMINISTIC_NORMAL_EQUATIONS)`, read at CALL time.
**Every other caller keeps the BLAS route and its historical bits.**  That is
a cost decision with a number on it (S5): the deterministic kernel is 2.2x on
the 5-term carrier fit and **34x** on the 66-term traced fits, whose
byte-identity contracts (niches C1/C6/C8/C9) predate this work.

---

## 3. THE ONE HOLE, DECLARED

Determinism covers the normal-equations route.  If the Gram is rank-deficient
outright, or screens numerically singular under niche C13, the solve reroutes
to `_solve_lstsq_qr`, whose `geqrf` runs over the FULL `A` and is a threaded
BLAS-3 factorisation.  On that route the answer is NOT
scheduling-independent.

A deterministic caller is therefore **warned** (`RuntimeWarning`, at both
exits), rather than handed a weaker guarantee silently.  The shipped route
stays silent -- the warning is about the promise this caller was given, not
about the data.  Pinned by
`test_the_step_down_hole_is_declared_rather_than_hidden`, whose two states are
ENGINEERED (a duplicated column for the rank-deficient exit; a column
perturbed by 1e-6 relative -- rcond 2.5e-13, five decades under the 1e-8
screen -- for the C13 exit).  A merely rescaled column does not work:
`_gram_rcond` equilibrates the diagonal first, so scale is exactly what it
ignores.  Measured on the production carrier fit, the screen does not fire.

---

## 4. THREAD-INVARIANCE EVIDENCE

Four widths x two repeats, one fresh interpreter per arm, the width pinned in
`OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` / `MKL_NUM_THREADS` /
`NUMEXPR_NUM_THREADS` before NumPy loads.  Each child asserts its own
`lumenairy.__file__` against `LUMENAIRY_ROOT` (S9 of BUILD_LENS_32K_MEMORY:
`PYTHONPATH` alone is not enough).  Fixture: N = 512, dx = 30 um, w = 80 px,
R = 1.0 m -> 60 245 bright pixels -> **119 936 fit rows**.

### 4.1 `_compute_carrier('auto')` -- the shared function

| arm | w=1 | w=2 | w=4 | w=8 | distinct |
|---|---|---|---|---|---|
| shipped | `d8e0395989a29ba7` | `d8e0395989a29ba7` | `9fe3279fade1ed48` | `174d4f66f231baba` | **3** |
| deterministic | `4542f1adeb54d33e` | `4542f1adeb54d33e` | `4542f1adeb54d33e` | `4542f1adeb54d33e` | **1** |

### 4.2 CONSUMER 1 -- `apply_real_lens(carrier='auto', surface_model='tangent_facet')`, the returned FIELD

| arm | w=1 | w=2 | w=4 | w=8 | distinct |
|---|---|---|---|---|---|
| shipped | `2d614d041d7b47a9` | `2d614d041d7b47a9` | `962ec721aa752e4b` | `d087825cf1faa064` | **3** |
| deterministic | `5348e5f3101235ca` | `5348e5f3101235ca` | `5348e5f3101235ca` | `5348e5f3101235ca` | **1** |

### 4.3 CONSUMER 2 -- the carrier fit as `apply_real_lens_traced(carrier='auto')` reaches it

| arm | w=1 | w=2 | w=4 | w=8 | distinct |
|---|---|---|---|---|---|
| shipped | `a3196a106fa22fa2` | `a3196a106fa22fa2` | `644d3a6ae2908745` | `889f36eee8fea3b8` | **3** |
| deterministic | `40b71a91ce8dbbd2` | `40b71a91ce8dbbd2` | `40b71a91ce8dbbd2` | `40b71a91ce8dbbd2` | **1** |

In every arm both repeats inside a process agreed, so the eight calls per row
collapse to the four printed.  The `{1,2} / {4} / {8}` grouping of the shipped
rows is the GEMV K-split of S1.1 seen end to end.

### 4.4 How far the field moved when determinism was turned on

Relative to the field peak, over the same 4 x 2 arms:

| consumer | shipped cross-width spread | deterministic spread | \|det - shipped\| |
|---|---|---|---|
| `_compute_carrier` | 3.162e-15 | **0** | 3.162e-15 |
| analytic field | 2.837e-14 | **0** | 2.836e-14 |
| traced carrier fit | 3.162e-15 | **0** | 3.162e-15 |

**The deterministic answer lands INSIDE the shipped route's own spread in all
three** -- i.e. it is an answer the shipped route could have returned on some
thread count.  That is the envelope statement, and it is what
`test_the_field_moves_only_at_the_summation_noise` asserts (bar =
`max(10 x measured spread, 1e-9)`; the floor covers a build whose BLAS keeps
the reduction serial, and sits ~3 decades above the largest measurement and
~6 below any field difference that would move a physical readout).

---

## 5. ORACLE ACCURACY

### 5.1 The oracle, proved before it is used

Dekker `TwoProduct` (exact, vectorised, no FMA needed) followed by
`math.fsum`: the returned float64 is the CORRECT ROUNDING of the exact
mathematical value of the sum, over the exact float64 inputs.  Not another
float64 summation order, and not a compensated accumulator whose own error
would have to be bounded.

Validated against exact rational arithmetic (`Fraction` products summed
without rounding, then rounded once), n = 20 000, M = 5:

```
  G   bit-identical: True   maxabs 0.0
  rhs bit-identical: True   maxabs 0.0
```

**A first attempt at the oracle was a Neumaier accumulator over per-block
`einsum` partials, and it is recorded here as a rejected instrument**: it
compensates the block COMBINATION but not the accumulation WITHIN a block, so
at n = 20 000 it read 2.05e-15 on the right-hand side where the thing it was
supposed to referee read 1.14e-16.  A reference that is worse than the
candidate cannot bound it.

### 5.2 What the two routes measure against it

The bar is not one BLAS call -- one call is one draw, and it can be the lucky
one.  It is the **legal-partition family**: a threaded BLAS reducing
`A^T A` / `A^T b` splits K into contiguous chunks and adds them, and which `k`
it picks is the scheduling fact at issue.  Reproducing the family in one
process (`k` in {1,2,3,4,5,8,12,16,24,32}) is what makes the envelope
build-free -- the state is ENGINEERED through legal reorderings rather than
waited for.

Two real carrier design matrices, oracle-relative:

**A = 67 348 x 5** (N = 384 fixture)

| quantity | deterministic | family best | family worst |
|---|---|---|---|
| `G` | 1.613e-16 | 1.613e-16 | 4.838e-16 |
| rhs | 2.573e-14 | 1.804e-14 | 6.429e-13 |
| **coefficients** | **3.954e-18** | 3.331e-16 | 9.881e-15 |

**A = 119 936 x 5** (N = 512 fixture)

| quantity | deterministic | family best | family worst |
|---|---|---|---|
| `G` | 8.858e-20 | 1.495e-19 | 1.270e-15 |
| rhs | 1.506e-15 | 3.539e-15 | 1.344e-12 |
| **coefficients** | **2.740e-18** | 4.441e-16 | 6.439e-15 |

Read three things off that:

* **The coefficients -- the only thing the fit returns -- beat even the BEST
  legal draw, by 84x and 162x**, against a family spanning 3.3e-16 to
  9.9e-15.  Two decades of gap on each side, which is what makes it a bar
  rather than noise.
* **`G` and rhs are only claimed against the WORST draw, and that is
  deliberate.**  On the 67 348-row fixture the deterministic right-hand side
  is 1.4x *worse* than the luckiest draw.  A `<= best` assertion on those two
  would have been testing noise -- and it was written that way first, failed
  on exactly that fixture, and was restated.  Margins against the worst draw:
  3x / 14 000x (`G`), 25x / 890x (rhs).
* The deterministic number does not depend on the width; every family number
  does.

Same comparison against a live threaded `A.T @ A` / `A.T @ b` at four widths
on the 119 936-row matrix, for the record:

```
  OMP=1   naive G 3.628e-16  rhs 3.539e-15  coef 2.887e-15
  OMP=2   naive G 3.628e-16  rhs 3.539e-15  coef 2.887e-15
  OMP=4   naive G 3.628e-16  rhs 1.344e-12  coef 1.776e-15
  OMP=8   naive G 3.628e-16  rhs 5.385e-14  coef 3.331e-16
  any     det   G 8.858e-20  rhs 1.506e-15  coef 2.740e-18
```

---

## 6. SPEED AND FOOTPRINT

### 6.1 Interleaved A/B, production shape

Interleaved (BLAS, det, BLAS, det, ...), best of four, one process, OMP = 8:

| what | BLAS | deterministic | overhead |
|---|---|---|---|
| solve alone, 28e6 x 5 | 0.232 s | 0.460 s | +98% |
| solve alone, 1.8e6 x 5 (N = 4096 scale) | 0.015 s | 0.031 s | +107% |
| solve alone, 141 471 x 66 (traced Chebyshev shape) | 0.0106 s | 0.3623 s | **+3320%** |
| **whole `_compute_carrier('auto')`, N = 8192, 26.1% bright, 35 070 898 rows** | **8.584 s** | **8.864 s** | **+3.3%** |

**The number that decides the default is the last one: +3.3%**, under the
brief's 5% bar, so it ships ON.  The solve is ~2.7% of the fit, so doubling it
is invisible next to the tilt-field and masking work around it.  The row count
in that arm (35.07e6) is ABOVE the audit's production figure for N = 16384
(28e6), so the production shape is covered rather than extrapolated to.

The 66-term row is why the scope is the carrier fit alone.  34x on a fit the
Newton loop runs repeatedly is not a trade worth making for a reduction that
is 141 471 rows long instead of 28 million.

### 6.2 Footprint -- the 32k wave's profile is not regressed

Same instrument as BUILD_LENS_32K_MEMORY S3.2 (`tracemalloc` peak, quoted in
float64 GRIDS), the fit alone with `need_W=False`, N = 4096:

| bright | fit peak, det OFF | det ON | delta |
|---|---|---|---|
| 5.0% | 10.125 | 10.125 | -0.0000 |
| 21.0% | 10.125 | 10.125 | -0.0000 |
| 59.0% | 17.530 | 17.530 | -0.0000 |
| 89.3% | 24.800 | 24.800 | -0.0000 |

**These are NOT the same numbers as S3.2's `2.75 / 3.73 / 11.25 / 18.65` and
do not replace them.**  That table measured the fit's peak INSIDE a full
`apply_real_lens` call with the accumulators spilled, so its baseline is the
rest of the call; this one measures the fit standalone, so it carries the
tilt-field and mask grids the fit builds before the solve.  What transfers is
the A/B -- same instrument, same fixture, one thing changed -- and the A/B is
the claim.

Zero to four decimals in every row, and it is zero for a structural reason:
the kernel's live set is one 512 KB tile, one row buffer, and -- because the
partials are combined by a carry-stack rather than collected in a list --
`O(log2(n / blk))` partial Grams instead of `O(n / blk)` of them.  At the
production shape (2 137 blocks) the stack holds at most 12 live `(5, 5)`
partials, not 2 137 of them.  The
list version would have been 513 KB at `M` = 5 (2 137 blocks x 240 B) and
**249 MB** at `M` = 66 (6 836 blocks x 36 432 B) with 28e6 rows; the carry-stack removes that scaling entirely.

`test_the_fit_footprint_is_not_regressed` asserts it at N = 1024 against a
DERIVED allowance computed from the two constants, not a pinned number.

---

## 7. THE TESTS, AND WHAT EACH ONE REFUSES TO CLAIM

`tests/unit/test_niche_d14_deterministic_carrier_fit.py` -- **17 passed**.

| test | claim |
|---|---|
| `test_the_auto_carrier_hashes_the_same_at_every_thread_count[fit/analytic/traced]` | THE NEW CLAIM: 1 distinct hash over 4 widths x 2 repeats, all three consumers |
| `test_the_traced_chain_has_a_second_reduction_this_niche_does_not_own` | scope, pinned structurally (S7.3) |
| `test_the_shipped_route_is_the_fail_before_wherever_this_build_splits` | matched pair on a row-count ladder |
| `test_the_oracle_is_an_oracle` | two-product + `fsum` == exact rational, bit for bit |
| `test_the_deterministic_route_is_at_least_as_close_to_the_truth` | oracle non-regression vs the legal-partition family |
| `test_the_two_routes_agree_inside_the_derived_summation_envelope` | agreement inside a bar derived in-run |
| `test_the_field_moves_only_at_the_summation_noise[fit/analytic/traced]` | field-level envelope, both bars measured in-run |
| `test_the_partition_reads_the_term_count_and_nothing_else` | the determinism argument, as an assertion |
| `test_the_transposed_tile_is_a_speed_device_and_not_an_arithmetic_one` | the tile moves no bit |
| `test_the_flag_off_returns_the_shipped_bits_exactly` | the fail-before is a value of the flag |
| `test_multi_rhs_matches_the_single_rhs_column_by_column` | 2-D `b` is the same reduction |
| `test_the_step_down_hole_is_declared_rather_than_hidden` | S3, both exits, engineered states |
| `test_the_fit_footprint_is_not_regressed` | S6.2, derived allowance |

### 7.1 The cross-build non-claim is a COMMENT

Per the brief and per `TESTING_STANDARDS`, the module docstring states in
block capitals that no test here may assert cross-build bit identity, and
why.  It is not an assertion because there is nothing to assert: the property
is false, and a test encoding it would be the S4 floor-bar shape.

### 7.2 The fail-before is a LADDER, not a hope

A fail-before that depends on a library's threading threshold cannot be
asserted unconditionally -- a build whose BLAS never splits at these sizes is
legitimate.  So the ladder (20 000 / 100 000 / 1e6 / 4e6 rows) asserts the
DETERMINISTIC invariance on every rung unconditionally, and where the shipped
route is not invariant on a rung, that rung is the matched pair.  If no rung
splits, the test `xfail`s with the ladder printed rather than hard-failing on
somebody's BLAS.

### 7.3 THE CLAIM THAT WAS WRITTEN, MEASURED, AND WITHDRAWN

The obvious consumer-2 test is "`apply_real_lens_traced(carrier='auto')`
returns the same field at every width".  It was written that way.  It passed
the 8-call hash arm and then FAILED the 4-call array arm at 1.98e-14 -- i.e.
it is intermittent, which is the worst way for a byte-identity test to be
green.

Censusing the solves settled it.  The traced chain runs six:

```
  (119936,   5)  deterministic   <- this niche
  (  1457,  28)  BLAS   x3       invariant across widths at this size
  (  1337, 120)  BLAS   x2       MOVES: 692283176c / f3437277bd at OMP=1
                                        a5685172e8 / 932500c51a at OMP=4 and 8
```

So the carrier fit was never the traced path's only nondeterministic
reduction, only its longest.  Consumer 2 now hashes **the carrier fit as the
traced entry reaches it**, which is what the fix owns and what the brief asked
for ("a traced-path carrier fit call site"), and
`test_the_traced_chain_has_a_second_reduction_this_niche_does_not_own` pins
the residual structurally so a future change has to come back and restate it.

**This is a live open item, not a closed one.**  Making the traced field
thread-invariant means putting the two 120-term fits on the deterministic
kernel, which costs 34x at their shape and moves the C1/C6/C8/C9 byte-identity
contracts.  That is a separate decision with its own campaign.

### 7.4 Two bars that were wrong first

Recorded because reading them would not have caught either:

* `err_det <= err_naive` for `G` and rhs against a single live BLAS call.
  Failed on the first fixture it met (rhs 2.573e-14 against 1.804e-14) --
  a `<=` between two quantities of the same order is noise, not a bar.
  Restated against the legal-partition family's worst draw (25x / 890x of
  margin) and, for the coefficients, its best (84x / 162x).
* `d_spread == 0` on the traced FIELD.  True in one 8-call sample and false in
  the next 4-call one.  See S7.3.

---

## 8. FILES

```
lumenairy/elements/_lens_traced.py
    + DETERMINISTIC_NORMAL_EQUATIONS, _DET_GRAM_TILE_BYTES,
      _DET_GRAM_MIN_BLOCK_ROWS
    + _det_block_rows, _det_normal_equations, _warn_det_stepdown
    ~ _solve_lstsq_thread_safe(A, b, deterministic=False)
    ~ _compute_carrier: the 'auto' solve passes deterministic=...
lumenairy/elements/_traced_flags.py     + 3 registry rows (layer D14)
docs/audits/TRACED_LAYER_MAP.md         + rows 34-36
tests/unit/test_niche_d14_deterministic_carrier_fit.py   NEW, 17 tests
CHANGELOG.md                            [Unreleased]
```

---

## 9. SUITES

The regression set is not "everything"; it is **every test file in the tree
that touches the `'auto'` carrier**, found by
`grep --include='*.py' -rln "carrier='auto'\|_compute_carrier\|ANALYTIC_CARRIER" tests/`,
plus the solver and registry files the change edits.  That grep is the honest
scope: the carrier fit's bits moved, and nothing else's did.

| gate | result |
|---|---|
| `test_niche_d14_deterministic_carrier_fit.py` (NEW) | **17 passed** |
| `test_niche_r6_auto_carrier_fit` + `d1` + `p1` + `c13` + `c14` + `test_lens_memory_levers` + `test_audit_lens` | **294 passed** |
| `test_hammer_h6_traced_carrier_eikonal` + `c6_stationary_phase_launch` + `d9_grid_origin` + `s10_sibling_patterns` | **78 passed** |
| `test_audit_lens_models_2026_07 -k "carrier or auto or fit"` | **13 passed** (56 deselected) |
| `test_niche_d7_decentred_fit` + `r7_intragroup_curvature` + `p4_gbd_reexpand` + `test_obl_banded_halo` | **143 passed** |
| the new suite at parent BLAS widths 1 / 4 / 16 (the in-process oracle and envelope bars must not be per-width) | **10 passed x3** |
| `ruff check lumenairy/ tests/` | **All checks passed** |

**532 tests across the `'auto'`-carrier surface, 0 failures.**  Not run: the
whole `tests/unit` tree -- it was started twice and abandoned twice, at 64 and
69 tests in ~30 min each, which is not a proportionate gate for a change whose
blast radius is one function's summation order.  Said plainly rather than
implied by omission.

### 9.1 Reproducing the evidence

```bash
export LUMENAIRY_ROOT=C:/tmp/lum_dg PYTHONPATH=C:/tmp/lum_dg
# thread-invariance, one arm per process, width pinned BEFORE numpy loads
for T in 1 2 4 8; do
  OMP_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T MKL_NUM_THREADS=$T \
  NUMEXPR_NUM_THREADS=$T python -m pytest \
    tests/unit/test_niche_d14_deterministic_carrier_fit.py -q
done
```

`PYTHONPATH` alone would not have pinned the library (S9 of
BUILD_LENS_32K_MEMORY): every child asserts `lumenairy.__file__` against
`LUMENAIRY_ROOT` and refuses rather than reporting numbers from the wrong
tree.

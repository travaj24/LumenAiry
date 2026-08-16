# The 2-D PMM cascade: where the 23 ms/layer actually goes, and two ways to spend less of it -- 2026-08-17

Branch `feat/pmm2d-tree-cascade` off `origin/main` (fca83e2, the
`chore/test-hygiene` merge).  `git log origin/main..HEAD` was empty before
branching.

**Roadmap item #1 off `EXPERIMENT_PMM2D_EIG_RECYCLE_2026_08_16.md` S7:
"effort on the all-distinct in-plane cost is better spent on the cascade,
which is 55-61% of the solve and rises with n".  This branch profiles that
cascade to the operation, states the ceiling the measurement allows, and then
ships two things against it: a BYTE-IDENTICAL restructure that is the new
default everywhere, and two opt-in reassociated folds -- `cascade='fused'`
and `cascade='tree'` -- each with a derived bar.  S2 is the profile, S3 the
ceiling arithmetic, S4 the identity evidence, S5 the speed, S6 the tree's
conditioning (the finding that most constrains it), S7 the memory, S8 the
fail-before evidence, S9 the refutations.**

**Mount.**  Windows py3.14.6, numpy 2.4.4, scipy 1.17.1, scipy-openblas
0.3.31, 24 cores / 128 GB (tesla-ryzen).  Every timing ran with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` exported **in the
environment before python started**, best-of-N after a warm-up solve, on an
otherwise quiet box.  Every probe asserted `"lum_tc" in lumenairy.__file__`
(and `"lum_tcb" not in`) and printed it:
`C:\tmp\lum_tc\lumenairy\__init__.py`, version 5.36.1.  The baseline oracle is
a **separate pristine worktree** at `C:/tmp/lum_tcb`, detached at fca83e2,
driven by the same probe scripts through `PYTHONPATH` -- so "byte-identical to
the pre-change path" is measured against code that predates this branch, not
against this branch's own escape hatch.

---

## S1.  What was already there

`PMM2DStackHybrid` at fca83e2 already had the 5.36.0 fast cascade
(`BUILD_PMM2D_CASCADE_2026_08_16.md`): interface dedup, adjacent identical-run
merge, priced `_geom_cache` / `_eig_cache`, and a threaded per-layer
eigensolve.  Its own S5.2 records the one workload left at exactly **1.00x**:
the ALL-DISTINCT stack, where there is nothing to dedup and nothing to merge.
Its S5.2 also records why threading does not rescue it -- **"the Redheffer
cascade after it is strictly sequential, so Amdahl caps this near 1.6x
whatever the worker count"**.

Both statements point at the same object.  So the first question was not what
to build but where, inside one Redheffer star, the 23.0 ms/layer goes.

## S2.  The profile: the cascade's cost is FLOPS, and a third of them do nothing

Per-operation wall clock, single thread, best-of-7, one patterned 6x6 cell,
period 0.9 um, wl 1.55 um.  Everything is also priced in **2N zgemm
equivalents**, because that is the unit the restructure trades in:

| n_orders | 2N | zgemm | zsolve | zinv | `_interface_smatrix` | star @ prop | star @ ifc |
|---|---|---|---|---|---|---|---|
| 4 | 162 | 0.516 ms | 1.175 (2.28) | 1.169 (2.27) | 6.441 (**12.49**) | 5.331 (**10.34**) | 9.110 (**17.66**) |
| 5 | 242 | 1.726 ms | 3.600 (2.09) | 3.573 (2.07) | 19.916 (**11.54**) | 17.912 (**10.38**) | 29.936 (**17.35**) |
| 6 | 338 | 4.804 ms | 9.681 (2.01) | 10.419 (2.17) | 52.322 (**10.89**) | 49.801 (**10.37**) | 81.751 (**17.02**) |
| 8 | 578 | 25.963 ms | 46.122 (1.78) | 44.109 (1.70) | 249.655 (**9.62**) | 241.040 (**9.28**) | 382.747 (**14.74**) |

Marginal per-layer cascade (`ifc` + both stars): **20.9 / 67.8 / 183.9 /
873.4 ms** -- i.e. **39.3 zgemm-equivalents per layer at `n_orders`=5**, and
the 5.36.0 doc's 23.0 ms/layer at `n_orders`=4 reproduces (20.9 ms here for
the same three operations).

Hand-counting the three operations against those readings closes the budget
and identifies the waste:

```
star(A, B) general :  D  = inv(I - B11 @ A22)          1 gemm + 1 inv
                      F  = inv(I - A22 @ B11)          1 gemm + 1 inv
                      C11 = A11 + A12@D@B11@A21        3 gemm
                      C12 = A12@D@B12                  2 gemm
                      C21 = B21@F@A21                  2 gemm
                      C22 = B22 + B21@F@A22@B12        3 gemm
                                      = 12 gemm + 2 inv  (predicted 16.1, read 17.4)

star(A, prop)      :  B11 == 0 -> D = F = I (LITERAL), yet the code still ran
                      A12@I, @B11(=0), @A21, A12@I@B12, B21@I@A21,
                      B21@I@A22@B12       = 10 gemm     (predicted 10.0, read 10.4)
                      ... of which SIX are against a literal identity or a
                      literal zero block.

_interface_smatrix :  a=solve(Wb,Wa), b=solve(Vb,Va), iapb=inv(a+b),
                      S11=iapb@amb, S21=0.5*(apb - amb@iapb@amb), S22=amb@iapb
                                      = 4 gemm + 2 solve + 1 inv
                      ... of which ONE recomputes S22 inside S21.
```

**Allocation and memory traffic are not the story.**  A 2N complex block is
0.42 / 0.94 / 1.83 MB at `n_orders` 4 / 5 / 6 and a general star allocates ten
of them, so this was worth testing rather than assuming.  Measured, `A @ B`
against `np.matmul(A, B, out=preallocated)`: **+2.2% / -5.2% / +3.0%** -- sign
included, i.e. inside the noise.  A full preallocated-workspace variant of the
star (`out=` on every product, one cached `eye`) measured **2.160 vs 2.157 ms**
(`n_orders`=4), **7.001 vs 7.013** (5) and **19.840 vs 20.048** (6) against the
plain restructured star: no gain.  **The 23 ms/layer is flops, so only removing
flops can move it.**  The workspace variant was written, measured and dropped
(S8).

## S3.  The ceiling, computed from the measurement before anything was built

Two floors, because there are two identity contracts.

**Byte-identical floor.**  Only products whose value is FIXED by a zero or
identity block may go, and only associations that `@` already produces may be
hoisted:

| operation | now | byte-identical floor | why it is byte-identical |
|---|---|---|---|
| `_interface_smatrix` | 4 gemm | 3 gemm | `amb @ iapb` IS `S22`, and `@` left-associates, so `amb @ iapb @ amb` is `(amb@iapb) @ amb` |
| star @ ifc | 12 gemm | 10 gemm | hoist `AD = A12@D`, `BF = B21@F`; same three products in the same order |
| star @ prop | 10 gemm | 4 gemm | `A12 @ I` is `A12` exactly (one term `A12[i,j]*(1+0j)`, the rest exact zeros); `(...) @ 0` is the exact zero matrix; `A11 + 0` is `A11` |

At `n_orders`=5 that is **39.27 -> 29.9 zgemm-equivalents per layer, 1.31x on
the marginal cascade.**  Against a distinct-layer solve whose eig
(`_layer_modes_projected`) reads 47.88 ms = 27.7 gemm-equivalents, the
whole-solve ceiling is **(47.9 + 67.8) / (47.9 + 51.6) = 1.16x**.

**Reassociated floor.**  `_propagation_star` -- already in
`rcwa/_core.py` since RCWA-LEV-2, already used by `RCWAStack`, the 1-D
`PMMStack`, `berreman`, `pmm/conical`, `pmm/twod` and **this class's own JAX
twin** -- collapses the star against a propagation S-matrix to three O(n^2)
row/column scalings.  Measured **0.34 / 0.21 / 0.36 / 0.21 gemm-equivalents**
at 2N = 162 / 242 / 338 / 578, i.e. 29-52x on that one star.  The 2-D NumPy
stack was the ONE stack path in the library still doing it the slow way.

It is **not** bit-for-bit: measured `max|d|` = 2.483e-16 / 1.241e-16 /
4.441e-16 at `n_orders` 4 / 5 / 6, with **zero sign-of-zero differences**.  The
mechanism is that OpenBLAS's zgemm rounds its complex products with FMA and an
elementwise scaling does not; the values are the same real numbers to the last
bit only when the FMA and non-FMA roundings agree, which they do not
everywhere.

Floor: **26.0 zgemm-equivalents/layer, 1.51x on the marginal cascade**,
whole-solve ceiling **1.25x**.

**So the ceiling arithmetic says: ~1.16x byte-identical, ~1.25x with the
reassociated propagation star, on an all-distinct stack -- and MORE on any
stack whose eigensolves are deduped or cached, because there the cascade's
share is larger.**  S5 measures both.

## S4.  What was built, and the three identity contracts

### S4.1  Byte-identical (the new default, whole library)

`lumenairy/elements/rcwa/_core.py`:

* `_redheffer_star` gains two explicit zero-block branches (`B11 == 0`, the
  layer-propagation shape that feeds half the stars in every stack solve, and
  `A22 == 0`) that drop the six products fixed by the zero/identity blocks,
  plus CSE on the general path (12 -> 10 gemm) that leaves the association
  order untouched.  The JAX guards are unchanged: a tracer can express neither
  `.any()` nor the branch, so it always takes the general path.
* `_interface_smatrix` and `_interface_smatrix_general` hoist the `S22`
  product they already computed twice (4 -> 3 gemm).

These are shared by RCWA, the 1-D PMM stack, EME, Berreman and the 2-D stack,
so every stack path in the library gets them.

### S4.2  `cascade='fused'` (opt-in, derived bar)

`PMM2DStackHybrid(cascade='fused')` additionally routes every star against a
layer-propagation S-matrix through `_propagation_star` /
`_propagation_star_general`.

### S4.3  `cascade='tree'` (opt-in, derived bar)

`'fused'` plus a balanced-tree reduction.  The chain is
`ifc_0, prop_0, ifc_1, prop_1, ..., prop_{N-1}, ifc_N` (2N+1 leaves); the star
is associative, so the tree reduces it pairwise left-to-right:

* **level 1** pairs each `ifc_i` with its own `prop_i` -- exactly the fused
  cheap star -- leaving `N+1` nodes;
* **levels 2..** join those, `N` general stars in `ceil(log2(N+1))` levels.

**Work is identical to the sequential fold** (`N` general stars either way --
asserted by test, both count 8 for 8 layers); the tree buys `O(log N)` DEPTH.
`solve(max_workers=W)` then fans out three phases: the distinct eigensolves,
the distinct interface builds (new: `_interface_list`), and each tree level.

Leaves stay LAZY through the first pass, so the tree never materialises the
`2N+1` leaf S-matrices at once (S7).

### The identity claims

| arm | claim | evidence |
|---|---|---|
| `cascade='monolithic'` | **BIT-FOR-BIT `origin/main`** | 432-case matrix: **432/432 byte-identical, worst disagreement 0.000e+00** |
| `cascade='fast'` (default) | **BIT-FOR-BIT `origin/main`** | same matrix: **432/432 byte-identical, 0.000e+00** |
| `_redheffer_star` / `_interface_smatrix*` | byte-identical to the pre-P2T algebra | asserted directly in `tests/unit/test_p2t_pmm2d_tree_cascade.py` against an inline reference implementation, on all three branch shapes and at normal + conical incidence |
| `cascade='fused'` | bounded, derived bar | 432-case matrix: worst **4.057e-13** |
| `cascade='tree'` | bounded, derived bar | 432-case matrix: worst **2.380e-11**; deep-stack envelope in S6 |
| `max_workers` = 1, 2, 4, 8 under `'tree'` | **byte-identical to each other** | 72-case tree matrix, all True |

The 432-case matrix is the shipped 5.36.0 shape: `{2, 3, 5}` layers x
`{normal, oblique 0.30, conical 0.30/0.70}` x `{degree 7 / n_orders 4,
degree 9 / n_orders 5, degree 9 / n_orders 8}` x `{repeated, distinct}` x
`{scalar cell, mixed cell+lossy film, in-plane tensor, out-of-plane tensor}` x
`{symmetry off, on}`, both incident polarizations on every case, zero raises
on every arm.  (36 of the 432 are byte-identical on every arm: those are the
even-symmetry folds, which bypass the cascade entirely.)

For scale, the SAME matrix run `monolithic`-vs-`fast` on the baseline
worktree reads 279/432 byte-identical, worst 6.189e-14 -- that is the shipped
5.36.0 merge, which this branch does not touch.

## S5.  Speed -- interleaved, and reported per lever

Interleaved per the 5.36.0 doc's S5.1 correction (a sequential before/after on
this box is not a valid instrument at the 1.3x level): baseline and branch run
as SEPARATE subprocesses, alternating, round-robin, min over 4 rounds, 3
timed reps each.  Per-round spreads were 1.7-6%.

### S5.1  Single thread, byte-identical vs opt-in fused

`n_orders`=5, degree 9, 6x6 cell, `symmetry=False`.

| case | PRE (fca83e2) | branch `'fast'` (byte-identical) | | branch `'fused'` (opt-in) | |
|---|---|---|---|---|---|
| 16-slice graded taper, ALL distinct | 2.0125 s | 1.8019 s | **1.117x** | 1.6700 s | **1.205x** |
| 8-slice graded taper, ALL distinct | 1.0234 s | 0.9150 s | **1.118x** | 0.8592 s | **1.191x** |
| 5 distinct layers, conical | 0.6572 s | 0.5958 s | **1.103x** | 0.5401 s | **1.217x** |
| ABAB, 8 periods (16 layers, 2 distinct) | 0.9915 s | 0.7745 s | **1.280x** | 0.6426 s | **1.543x** |
| re-solve, 5 distinct (eig cache warm) | 0.3798 s | 0.2967 s | **1.280x** | 0.2633 s | **1.443x** |
| 16 identical layers (merged to one) | 0.1542 s | 0.1374 s | **1.122x** | 0.1315 s | **1.173x** |
| OOP tensor, 4 distinct layers, `n_orders`=3 | 0.1937 s | 0.1852 s | **1.046x** | 0.1822 s | **1.063x** |

Read the split honestly:

* **On an ALL-DISTINCT stack the byte-identical restructure gives 1.10-1.12x,
  which is BELOW the 1.15x ship threshold this campaign set for a single
  lever.**  It is kept anyway, for a reason the threshold does not cover: it
  changes no contract, costs no code path, and applies to every stack solver in
  the library -- there is nothing to trade off against it.  But it is reported
  as what it measures, not as what the ceiling arithmetic hoped (1.16x).
* **On a CASCADE-DOMINATED stack it gives 1.28x** -- ABAB and the warm
  re-solve, where the eigensolves are deduped or cached and the cascade is
  most of what is left.  That is the workload class the roadmap item named.
* **`cascade='fused'` clears 1.15x on every in-plane case** (1.19-1.22x
  all-distinct, 1.44-1.54x cascade-dominated) and is opt-in.
* **The OOP path barely moves (1.05-1.06x).**  Expected: the eig-recycle doc's
  S1 measured the 4Nf generator eig at **67.9%** of an OOP solve against a
  17.1% cascade, so there is almost nothing here for a cascade lever to take.

### S5.2  Worker scaling: the tree breaks the Amdahl cap whose serial term IS the cascade

Branch-vs-branch, interleaved, min over 3 rounds.  `fused-wN` is the SHIPPED
5.36.0 parallel path (`max_workers` fans out the eigensolves only, the cascade
stays a sequential fold); `tree-wN` additionally fans out the distinct
interface builds and every level of the tree.  24-core box.

**40 all-distinct graded slices, `n_orders`=5, degree 9, conical:**

| arm | time | vs its own 1-worker | vs the default `fast` serial |
|---|---|---|---|
| `cascade='fast'`, serial (the default) | 4.3206 s | -- | 1.00x |
| `'fused'`, serial | 4.0843 s | -- | 1.06x |
| `'fused'`, `max_workers=1` | 4.1127 s | 1.00x | 1.05x |
| `'fused'`, `max_workers=4` | 2.6361 s | 1.56x | 1.64x |
| **`'fused'`, `max_workers=8`** | **2.3944 s** | **1.72x** | 1.80x |
| `'tree'`, `max_workers=1` | 4.1259 s | 1.00x | 1.05x |
| `'tree'`, `max_workers=2` | 2.2525 s | 1.83x | 1.92x |
| `'tree'`, `max_workers=4` | 1.3526 s | 3.05x | 3.19x |
| `'tree'`, `max_workers=8` | 0.8903 s | 4.63x | 4.85x |
| **`'tree'`, `max_workers=16`** | **0.7381 s** | **5.59x** | **5.85x** |

**20 slices, `n_orders`=5:**

| arm | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| `'fused'` (eig fan-out only) | 2.1273 s | -- | 1.3579 (1.57x) | 1.2520 (**1.70x**) | -- |
| `'tree'` | 2.1613 s | 1.2111 (1.78x) | 0.7609 (2.84x) | 0.5444 (3.97x) | 0.4730 (**4.57x**) |

**20 out-of-plane tensor slices, `n_orders`=3 (the generalized cascade):**

| arm | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| `'fused'` (eig fan-out only) | 0.8800 s | -- | 0.3587 (2.45x) | 0.3088 (**2.85x**) | -- |
| `'tree'` | 0.8925 s | 0.4924 (1.81x) | 0.3044 (2.93x) | 0.2331 (3.83x) | 0.2089 (**4.27x**) |

Read it:

* **The 5.36.0 doc's "Amdahl caps this near 1.6x whatever the worker count"
  reproduces exactly**: the eig-only fan-out saturates at **1.70-1.72x** on
  the in-plane path (1.72x at 8 workers, and 8 workers bought 1.10x over 4).
* **The tree removes that ceiling**: 4.63x at 8 workers and 5.59x at 16 on the
  same 40-layer stack, i.e. **3.2x more than the shipped parallel path at the
  same worker count.**
* **Scaling is not linear and is not claimed to be.**  The tree's top levels
  have 3, 2 and 1 independent stars, so the last few levels are serial
  whatever `W` is; 16 workers buy only 1.21x over 8.  The measured 8-worker
  efficiency is 58% (4.63x / 8), which is what a `ceil(log2(N+1))`-deep
  reduction with a shrinking level width predicts.
* **The OOP path parallelises well even without the tree** (2.85x at 8
  workers) because its `4Nf` eig is 68% of the solve; the tree still adds 1.5x
  on top at 16 workers.

## S6.  The finding that most constrains the tree: Redheffer stability is a PREFIX property

The sequential fold's partial products are all **anchored to the
superstrate**: every one of them is a physical two-port from the top of the
stack to somewhere inside it, and its blocks stay bounded.  A tree forms
**interior sub-chain** products, which are anchored to nothing.  Measured on
the running build, over the two folds' OWN intermediates
(`kappa` = max `||inv(I - B11 A22)||_inf`, `G` = max `||.||_inf` over every
intermediate block):

| stack (scalar, `n_orders`=4, degree 7) | sequential `kappa` / `G` | tree `kappa` / `G` | `max\|seq - tree\|` on R/T/J |
|---|---|---|---|
| 5 layers, all-distinct, conical | 4.6 / 25 | 10.7 / 2.0e+02 | 8.9e-13 |
| 10 layers, graded taper, conical | 10.1 / 38 | 10.2 / 5.2e+02 | 1.4e-11 |
| 20 layers, graded taper, conical | 2.6 / 25 | 22.3 / 8.1e+02 | 1.3e-09 |
| **40 layers, graded taper, conical** | **7.1 / 4.3e+01** | **4.4e+02 / 1.4e+04** | **5.2e-07** |
| 40 layers, all-distinct, conical | 120 / 2.9e+02 | 34.9 / 2.8e+02 | 6.5e-11 |
| 40 layers, repeated, any incidence | -- | -- | **0.0 .. 2.0e-15** |
| 40 layers, graded taper, OOP tensor | -- | -- | 2.8e-14 |

(The `kappa` / `G` columns are read off a chain reconstructed outside the
solver so the two associations run on IDENTICAL leaves; the last column is
the SOLVE-level difference on `R`/`T`/Jones.  The S-matrix-level difference
is the larger of the two -- 4.4e-06 against 5.2e-07 on the 40-layer conical
taper -- so the bar scored below is scored against the conservative one.)

**The tree's intermediates are up to 63x worse-conditioned and 326x larger in
norm than the sequential fold's on a deep, strongly-graded, high-contrast
scalar taper, and the answer moves 5.2e-07 there.**  That is still four orders
below the hybrid's own `n_orders` Fourier floor and it is an ABSOLUTE
difference on an O(1) efficiency, so it is not a wrong answer -- but it is a
real, measurable, structural cost of reassociating, and it is why the tree is
opt-in and not wired to `max_workers` behind the user's back.

`cascade='fused'` does **not** pay it: it reassociates one star at a time, not
the fold, and its worst reading on the same 72-case matrix is 2.5e-11 (against
the tree's 5.2e-07 on the same case).

### The derived bar

A reassociation perturbs each intermediate at `eps_mach * G`; the answer
passes through at most `2 n + 2` stars, each amplified by its own star
denominator, whose worst measured norm on THAT fold is `kappa`:

```
bar = 1e4 * (2 * n_layers + 2) * eps_mach * G * max(1, kappa)
```

with `G` and `kappa` measured **on the fold under test** (a bar built on the
sequential fold's conditioning is ~300x too tight for the tree -- the table
above is exactly that measurement).  Scored over layers {5, 10, 20, 40} x
{normal, oblique, conical} x {all-distinct, graded taper}:

| candidate | worst observed / bar | margin |
|---|---|---|
| `1e3 (2n+2) eps r` (the shipped P2C merge-bar shape) | 1.9e+03 | **fails** |
| `1e3 (2n+2) eps G` | 1.7e+01 | **fails** |
| `1e3 (2n+2) eps G kappa` | 3.9e-02 | 26x |
| **`1e4 (2n+2) eps G kappa`** | **3.9e-03** | **259x** |

Gap on the other side: the smallest real signal the cascade must resolve --
reversing an asymmetric stack -- moves the answer by O(0.1), about **1e6x
above** the bar.  Both sides are asserted in
`test_p2t_reassoc_bar_has_a_gap_on_both_sides`.

Note what the first row means: **the shipped 5.36.0 merge bar's shape does not
extend to the tree**, and using it would have been the "one build's residual
pinned universally" error the standards name.  It was scored and rejected on
measurement.

## S7.  Memory accounting

One S-matrix is four `2Nf x 2Nf` complex128 blocks = `4 (2Nf)^2 * 16` bytes:
**0.42 / 0.94 / 1.83 / 5.35 / 21.4 MB** at `n_orders` = 4 / 5 / 6 / 7 / 8.

* The **sequential** fold holds the interface list (`N+1`, which the dedup memo
  owns either way) plus ONE accumulator.
* The **tree** holds the same interface list plus, at its widest level,
  `ceil((N+1)/2)` level-1 results.  Leaves stay lazy through the first pass, so
  the `2N+1` leaf S-matrices are never all live.

```
extra_peak = ceil((N+1)/2) * 4 * (2 Nf)^2 * 16
```

which is `ceil((N+1)/2)` S-matrices -- 78.7 MB for 40 layers at `n_orders`=5,
**449 MB** for 40 layers at `n_orders`=8.  That is enough to matter, so it is
**priced**: `PMM2DStackHybrid.tree_budget()` reads
`lumenairy.memory.get_ram_budget()` **at solve time** (so `set_max_ram` applies
immediately) scaled by 25%, overridable with `tree_max_bytes=`.  When the
projection exceeds the budget the tree **refuses** and the sequential fused
fold runs instead -- a different association order, not a worse answer, and the
one that costs no extra memory at all.  `cascade_stats()` reports
`requested / engaged / peak_bytes / budget / leaves / depth`.

Both sides are FORCED by test (`tree_max_bytes=4 GB` -> engaged;
`tree_max_bytes=1` -> refused, and the refused answer asserted BYTE-IDENTICAL
to `cascade='fused'`, because that is literally what it ran).  No
`pytest.skip`, no dependence on how big the box is.

### S7.1  Measured, against the projection

Peak working set (`psutil` `peak_wset`), one fresh subprocess per arm, two
solves each so the caches are warm and the delta is the CASCADE's:

| layers | `n_orders` | 2Nf | one S-matrix | fused peak | tree peak | measured extra | projected extra | ratio |
|---|---|---|---|---|---|---|---|---|
| 10 | 5 | 242 | 3.6 MB | 412.7 MB | 431.8 MB | 19.1 MB | 21.4 MB | 0.89 |
| 20 | 5 | 242 | 3.6 MB | 515.6 MB | 561.4 MB | 45.8 MB | 39.3 MB | 1.16 |
| 40 | 5 | 242 | 3.6 MB | 725.0 MB | 822.0 MB | 96.9 MB | 75.1 MB | 1.29 |
| 40 | 6 | 338 | 7.0 MB | 1119.1 MB | 1312.8 MB | 193.6 MB | 146.4 MB | 1.32 |
| 20 | 8 | 578 | 20.4 MB | 1528.4 MB | 1794.4 MB | 266.1 MB | 224.3 MB | 1.19 |

The projection is right to **0.89-1.32x** across a 10x range of extra
footprint; the excess over 1.0 at the larger sizes is the allocator's high
water mark, not a different count.  **The tree costs 13-24% more peak RSS than
the sequential fold on these stacks** -- material but not structural, and the
gate above bounds it.  The relation "double the layers doubles the extra set,
double the basis quadruples it" is asserted directly, rather than any MB
reading here being pinned.

## S8.  Fail-before evidence

`tests/unit/test_p2t_pmm2d_tree_cascade.py`, **57 tests**, green at
`OMP/OPENBLAS/MKL_NUM_THREADS` = **1, 2 and 4** (91 tests including the
shipped `test_p2c_pmm2d_stack_cascade.py`, all green on all three).

| claim | how it is made build-free |
|---|---|
| the star restructure moves ZERO bits | compared against a **reference implementation written out inline in the test file** -- the pre-P2T algebra -- on all three branch shapes (`star(ifc, prop)`, `star(prop, ifc)`, `star(ifc, ifc)`) at normal and conical incidence.  Needs no second worktree to be true |
| the interface CSE moves ZERO bits | same, for `_interface_smatrix` and `_interface_smatrix_general` |
| `'fused'` is a REAL change of arithmetic | asserted non-zero AND below `1e3 eps ||S||` -- if it were byte-identical there would be nothing to gate and the opt-in would be pointless |
| `'tree'` really reassociates | asserted byte-DIFFERENT from the sequential fused fold on a 12-layer taper.  A `'tree'` that quietly folded left to right would pass every bar test above; this is the one that fails |
| the tree is WORK-optimal | star calls counted by monkeypatch: `fused` = `tree` = `n_layers` general stars (8 for 8 layers); `monolithic` = `2 n` |
| depth is logarithmic | `cascade_stats()['tree']['depth'] == ceil(log2(nlay + 1))` at nlay 4/8/16, and `< nlay` |
| worker counts agree | byte-identity asserted WITHIN the threaded contract (1/2/4/8), bar-agreement only ACROSS it -- asserting bit-equality against the uncapped `max_workers=None` default would be the S3 environment-dependent shape |
| the bar has a gap on BOTH sides | measured envelope below (S6), and the smallest real signal -- reversing an asymmetric stack -- asserted `> 1e2 x bar` above |
| the bar is not an accident of shallow stacks | an explicit 40-layer, eps-DOUBLING, conical arm (the worst case in the envelope), asserted at the bar AND asserted to show `fused` < `tree`, the S6 ordering |
| overflow regime | thickness derived from the layer's OWN measured eigenvalues so `exp(+lam k0 t)` provably overflows float64 on this build (asserted non-finite), then the reassociated cascade shown finite, per-pol `R+T <= 1`, `T < 1e-12`, and agreeing with the sequential fold at the bar |
| energy closure | per-polarization (sum orders WITHIN a pol, max over pols), bar = the SEQUENTIAL path's own closure on the same stack times decades -- so it tracks the hybrid's `n_orders` Fourier floor rather than pinning it |
| reciprocity | a real-symmetric tensor gives `J01 == J10`; measured 1.82e-15 against `max\|J\|` 0.213, and the reassociated folds must not be worse than 10x the sequential reading |
| the memory gate | **two-sided and FORCED**: `tree_max_bytes=4 GB` -> engaged; `tree_max_bytes=1` -> refused, and the refused answer asserted **BYTE-IDENTICAL to `cascade='fused'`**, because that is literally the fold it ran.  No `pytest.skip`, no dependence on box size |
| the projected peak | asserted as a RELATION (double the layers -> double the extra set; double the basis -> quadruple it), not as an MB reading |
| the default is unchanged | a no-`cascade=` stack asserted byte-identical to explicit `'fast'` AND byte-DIFFERENT from `'fused'` / `'tree'` |

**Injector run.**  Each byte-identity claim was checked against a deliberately
broken build, through monkeypatching only (no source edit):

| injected defect | caught by |
|---|---|
| `_redheffer_star` reassociated to `A12 @ (D @ (B11 @ A21))` -- mathematically identical | star byte-identity test, `max\|d\|` 4.441e-16 |
| the FUSED arithmetic promoted to the default star | star byte-identity test (8.882e-16) AND the default-is-unchanged test |
| `_interface_smatrix` `S21` reassociated to `amb @ (iapb @ amb)` | interface byte-identity test |
| `_star_tree` silently replaced by a left-to-right fold | `test_p2t_tree_actually_reassociates` (added BECAUSE the injector showed the first draft missed it) |
| the memory gate forced to always engage | the two-sided gate test |

The fourth row is the one that mattered: the first draft of this suite had no
test that failed when the tree was quietly sequential, and every bar test
stayed green.  The injector found it; the test was added.

### Regression run

The star and interface changes are shared by every stack solver in the
library, so the gate is the FULL unit suite, not a `-k` slice:

```
pytest tests/unit -q -p no:randomly -n 10 --dist loadfile
    1 failed, 12129 passed, 74 skipped in 51m48s   (BLAS pinned to 1)
```

The one failure is
`test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_fff_nv_stripe_reduces_to_rigorous_1d`,
which **fails identically on the pristine `origin/main` worktree**
(re-measured side by side on `C:/tmp/lum_tcb`, `1 failed in 2.03s`).  It is in
`rcwa_jones_2d`'s FFF-NV surface -- a module this branch does not touch -- and
is the same pre-existing failure the 5.36.0 doc's S6 already recorded.  It is
noted here so the next reader does not attribute it to this change.

The shipped `test_p2c_pmm2d_stack_cascade.py` (34 tests, the regression floor
for the 5.36.0 cascade) is green unchanged at BLAS 1, 2 and 4.

## S9.  Refutations -- levers measured and NOT shipped

* **Preallocated workspace / `out=` on every product.**  Written and measured
  (S2): 2.160 vs 2.157 / 7.001 vs 7.013 / 19.840 vs 20.048 ms against the
  plain restructured star.  Allocation is 2-5% of a star and inside the noise;
  the cost is flops.  Not shipped.
* **Dropping one of the two star inverses** via the push-through identity
  `F = I + A22 D B11`.  Trades 1 gemm + 1 inv (3.07 gemm-equivalents) for
  2 gemm -- a 1.07 gemm saving of 16 -- and is NOT bit-for-bit, so it would
  have needed its own opt-in for 3% of one operation.  Not built.
* **`A12 @ inv(M)` as `solve(M.T, A12.T).T`.**  Saves ~1 gemm but detaches the
  inverse from `_guarded_inverse`, whose screen-and-refuse on the star
  denominators is the M1/X-1 conditioning guard.  Not built: the guard is
  load-bearing and the saving is not.
* **Using the sequential fold's conditioning for the tree's bar.**  Scored and
  rejected (S6): 300x too tight, and the shipped merge-bar shape misses by
  1.9e+03.
* **Wiring the tree to `max_workers > 1` automatically.**  Refused on the S6
  measurement: the association is measurably worse-conditioned on deep graded
  stacks, so a caller who asks only for threads must not silently get a
  different fold.  `cascade='tree'` is a separate, explicit request.

## S10.  Files

| file | change |
|---|---|
| `lumenairy/elements/rcwa/_core.py` | `_redheffer_star` zero-block branches + CSE; `_interface_smatrix` / `_interface_smatrix_general` `S22` hoist.  All byte-identical, all library-wide |
| `lumenairy/elements/pmm/stack2d.py` | `cascade=` accepts `'fused'` / `'tree'`; `tree_max_bytes=`; `_star_tree`, `_tree_peak_extra_bytes`, `_interface_list` (replaces `_interface_memo`, adds the threaded distinct-interface fan-out), `_tree_gate`, `tree_budget`, `cascade_stats` |
| `tests/unit/test_p2t_pmm2d_tree_cascade.py` | NEW |
| `docs/audits/BUILD_PMM2D_TREE_CASCADE_2026_08_17.md` | NEW -- this document |
| `CHANGELOG.md` | `[Unreleased]` |

No public API is removed or repointed.  `cascade='fast'` remains the default
and remains bit-for-bit `origin/main`.

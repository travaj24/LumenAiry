# The 2-D PMM stack cascade: what was already there, what actually cost, and what was built -- 2026-08-16

Branch `feat/pmm2d-stack-cascade` off `origin/main` (40f28ae, the 5.35.5
release commit).  `git log origin/main..HEAD` was empty before branching.

**The mandate was to bring the 1-D `PMMStack` architecture -- per-layer
eigensolve, reuse, stable S-matrix cascade -- to the 2-D stack path.  S1 records
the measurement that refutes the premise: all three were already in `main`.
S2 records what a 2-D multilayer solve actually costs, which is not where the
plan assumed.  S3-S6 record what was built against the measurement instead.**

**Mount.**  Windows py3.14.6, numpy 2.4.4, scipy-openblas 0.3.31, 24 cores /
128 GB (tesla-ryzen).  Every timing below ran with
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` exported **in the
environment before python started** (a `setdefault` after import does nothing),
best-of-3 after a warm-up call, on an otherwise quiet box.

**Import provenance.**  Every probe asserted `"lum_p2c" in lumenairy.__file__`
and printed it.  Confirmed throughout:
`lumenairy.__file__ = C:\tmp\lum_p2c\lumenairy\__init__.py`, version 5.35.5.
The baseline oracle is a **separate pristine worktree** at
`C:/tmp/lum_base` (detached at 40f28ae), driven by the same probe script
through `PYTHONPATH`, so "agreement with the pre-change path" is measured
against code that predates this branch -- not against this branch's own escape
hatch.

---

## S1.  The premise was already shipped

`PMM2DStackHybrid` (`lumenairy/elements/pmm/stack2d.py`, in `main` since v5.13)
already had, before this branch:

| plan item | status in `main` @40f28ae |
|---|---|
| per-layer eigensolve in a shared basis | yes -- `_layer_modes_projected` / `_tensor_layer_modes` per layer |
| identical-layer reuse | yes -- audit F4's per-call `_mode_cache`, keyed on `_geom_key` |
| stable Redheffer S-matrix cascade | yes -- `_interface_smatrix` / `_propagation_smatrix` / `_redheffer_star`, plus a generalized promotion for out-of-plane tensor layers |
| wavelength-independent build reuse | yes -- audit F4 part 2's `_geom_cache` |
| even-symmetry fold | yes -- audit F2, `_symmetric_cascade_rt` |
| JAX twin | yes -- `_jax_stack2d.py` |
| threaded sweep with a process-global BLAS cap | yes -- `solve_vs_wavelength`, the M4 contract |

The "pure PMM2D cascade prototype" the brief asked to hunt for in the history
has also already graduated: it is `stack2d_pure.PMM2DStackPure`, a full
no-floor staggered stack, and its module docstring records that the historical
"A|B blows up energy" defect traced to the far-field projection-kernel order
mirror, since fixed.  Per-layer grids are likewise not an issue here -- the
hybrid projects every layer into the shared Rayleigh basis, so it has **no
union-grid constraint at all** (that constraint, and the degree>=8 oblique
pathology behind `fix/pmm-union-grid-conditioning`, belong to the 1-D
`PMMStack` and to `PMM2DStackPure`).

Building any of the above again would have been rework.  So the question
became: measured, where does a 2-D multilayer solve actually spend its time?

## S2.  Measured: the cascade costs more than the eigensolves

Per-operation cost at the two working sizes (`degree` 7/9, one patterned
6x6 cell, period 0.9 um, wl 1.55 um):

| operation | `n_orders`=4 (n=162) | `n_orders`=5 (n=242) |
|---|---|---|
| `_layer_modes_projected` (the eig) | 18.48 ms | 47.88 ms |
| `_interface_smatrix(A, B)` | 6.37 ms | 20.00 ms |
| `_redheffer_star(S, S_prop)` | 5.77 ms | 18.40 ms |
| `_redheffer_star(S, S_ifc)` | 9.15 ms | 29.79 ms |
| `_propagation_smatrix` | 0.02 ms | 0.04 ms |

A `cProfile` of one 5-distinct-layer solve at `n_orders`=5 (0.602 s total)
splits as **eig 0.236 s cumulative (39%)** against **cascade 0.353 s
(`_redheffer_star` 0.237 + `_interface_smatrix` 0.116, 59%)**.  The cascade is
the larger half, not the eigensolve.

The gap is starker once F4's dedup does its job.  Adding an **identical** layer
to a stack costs no eig at all, yet still costs:

| N identical layers, `n_orders`=4 | total | per layer |
|---|---|---|
| 2 | 0.0699 s | 34.9 ms |
| 4 | 0.1094 s | 27.3 ms |
| 8 | 0.1962 s | 24.5 ms |
| 16 | 0.3675 s | 23.0 ms |

23.0 ms/layer against a predicted `ifc 6.37 + star_prop 5.77 + star_ifc 9.15 =
21.3 ms` -- i.e. **on a repeated stack, essentially 100% of the marginal cost
is cascade, 0% is eigensolve.**  And that cascade work is provably redundant:

```
max|S_ifc(A, A) - swap|   =  4.934e-15   (n_orders=4, n=162)
                          =  8.910e-15   (n_orders=5, n=242)
```

where `swap = ((0, I), (I, 0))`.  The interface between a layer and *itself* is
the no-op two-port to 5e-15, so both the 6.37 ms build and the 9.15 ms star
that consumes it are, analytically, work that does nothing.

Three further measurements:

* **No cross-call reuse.**  Four successive `solve()` calls on one object at
  one source: `0.6252 / 0.6007 / 0.6086 / 0.6090 s`.  Flat.  F4's `_mode_cache`
  was a per-CALL local dict, so every call re-eigged every layer.
* **`_geom_cache` grew without bound.**  It was a bare `dict`, cleared only by
  `add_layer`.  A dispersive sweep mints one entry per wavelength by design
  (the tile bytes are in the key), so the footprint is linear in sweep length:
  a 12-point sweep retained 8.41 MB, a 24-point sweep 16.82 MB -- exactly 2x
  for exactly 2x the points.
* **...and it was shared, unlocked, across sweep workers.**
  `solve_vs_wavelength` hands each worker a `copy.copy(self)`; measured,
  `copy.copy(st)._geom_cache is st._geom_cache` -> `True`.

The unbounded-retention shape is the same one audit P3-32 already fixed on the
1-D `_PreparedPMMStack`; it had simply never been applied to the 2-D twin.

## S3.  What was built

`cascade='fast'` (the new default) on `PMM2DStackHybrid`, with
`cascade='monolithic'` as the escape hatch.

**1. Interface dedup.**  The per-interface mode-match S-matrix is memoized on
the `(above, below)` modal-content key pair.  A hit is **byte-identical** to a
rebuild -- the same `W`/`V` bytes through the same LAPACK -- so this is free.
It also applies to the generalized (out-of-plane) cascade, where `_modes_to_M`
and `_interface_smatrix_general` are memoized the same way.

**2. Identical-run merge.**  A maximal run of *adjacent* layers sharing a modal
key collapses into ONE propagation of the summed thickness.  Adjacent layers
with the same modal basis **are** one thicker layer, so this is exact physics,
not an approximation; numerically it replaces
`prop(t1) * ifc(A,A) * prop(t2)` with `prop(t1+t2)`.  Non-adjacent repeats are
NOT merged (an A-B-A stack keeps three cascade entries; pinned).

**3. Priced, bounded, thread-safe caches** (`_stack2d_cache.LayerCache`)
replacing the two bare dicts:

* `_geom_cache` -- the wavelength-independent nodal build.
* `_eig_cache` -- NEW: the modal eigensolve, keyed on geometry AND source
  (`wl, k0, kx0, ky0, formulation`), retained **across `solve()` calls**.  This
  is the 1-D `_PreparedPMMStack._eig_cache` analogue the 2-D path lacked.

Pricing follows the refuse-never-degrade precedent.  The budget is read from
`lumenairy.memory.get_ram_budget()` **at query time** (so `set_max_ram` applies
immediately rather than being frozen at construction), scaled by
`budget_fraction` (5%) and floored at `min_budget` (256 MB); `cache_max_bytes=`
overrides both.  Eviction is LRU but **never touches an entry minted by the
current solve**: entries carry a generation stamp bumped at the top of every
`solve()`.  If the live working set alone exceeds the budget the cache
**refuses** to store rather than evict-and-immediately-remiss.  A refusal costs
recomputation and nothing else -- the recompute is byte-identical.

Byte accounting charges each array against the ROOT buffer it looks through,
counted once: the projected-operator dicts share buffers, and charging views
separately would price the cache out early.  (This was caught by its own test
-- the first implementation double-charged views.)

**4. Threaded per-layer eigensolve**, `solve(max_workers=N)`.  Only the
DISTINCT modal sets are fanned out.  See S6 for why the default is `None` and
not `1`.

### The identity claim, and where it is bounded

Two regimes, pinned separately:

* **`cascade='monolithic'` is BIT-FOR-BIT `origin/main`.**  Measured over a
  360-case matrix (S4): worst disagreement **0.000e+00**, zero non-identical
  arrays.
* **`cascade='fast'` on a stack with no two adjacent layers alike is also
  bit-for-bit** -- there is nothing to merge, and dedup cannot move a bit.
  Asserted as exact equality, so there is no tolerance to be per-build about.
* **`cascade='fast'` on a mergeable stack is BOUNDED-identical.**  The bound is
  derived from conditioning, not from a residual: the merge perturbs by the
  interface's departure from the exact swap, `r = max|ifc(A,A) - swap|`, which
  is itself a `~cond(W) cond(V) eps_mach` readout, amplified by at most the
  `2n+2` stars and matches the cascade applies to it.  The test computes `r` on
  the running build and uses `bar = 1e3 * (2n+2) * r`; nothing pins a number
  this campaign read.

## S4.  Oracle matrix

360 cases: `{2, 3, 5}` layers x `{normal, oblique (theta=0.30), conical
(theta=0.30, phi=0.70)}` x `{degree 7/n_orders 4, degree 9/n_orders 5,
degree 9/n_orders 8}` x `{repeated, distinct}` x `{scalar cell, mixed
cell+lossy film, in-plane tensor (full Jones), out-of-plane tensor}` x
`{symmetry off, symmetry on}`.  Both incident polarizations are driven on
every case (the solver always returns the 2-row `R`/`T` and the 2x2 Jones).

| comparison | cases | worst abs disagreement | bit-identical |
|---|---|---|---|
| `origin/main` vs branch `cascade='monolithic'` | 360 | **0.000e+00** | 360/360 |
| `origin/main` vs branch `cascade='fast'` | 360 | **2.073e-14** | 243/360 |

The 117 non-identical cases are exactly the repeated-layer ones (the merge);
the worst reading, 2.073e-14, is case 96 = **2 layers / conical / repeated /
degree 9 / `n_orders`=8 / scalar / symmetry off** -- the largest basis and the
worst-conditioned incidence in the matrix, as expected.

The bar for that exact arm was then MEASURED rather than extrapolated (the
interface-identity residual `r` on that layer, at that incidence, at that
basis size):

| `n_orders` | incidence | `n` | `r = max\|ifc(A,A) - swap\|` | bar `= 1e3 (2n_lay+2) r` |
|---|---|---|---|---|
| 4 | normal | 162 | 4.934e-15 | 2.960e-11 |
| 5 | normal | 242 | 8.910e-15 | 5.346e-11 |
| 8 | conical | 578 | **2.141e-14** | **1.285e-10** |

So the worst observed disagreement sits **6200x below** the bar derived for its
own arm, and `r` scales with the basis exactly as a conditioning readout should
(4.9e-15 at n=162 -> 2.1e-14 at n=578).  Zero failures, zero raises.

Per-polarization energy closure was checked with the house convention (sum
orders WITHIN a polarization, max over polarizations -- never sum the two).
`fast` and `monolithic` closures agree to the same 1e-14 throughout; the
absolute closure values track the hybrid's `n_orders` Fourier floor, which this
change does not touch.

## S5.  Speed and memory

### S5.1  A benchmarking correction, recorded because it changed the answer

The first pass ran the whole baseline suite, then the whole branch suite, back
to back in one process each.  It credited the branch **1.39x on an all-DISTINCT
16-slice tapered stack** and **1.36x on a 5-distinct-layer conical stack** --
cases where the fast path has, by construction, nothing to dedup and nothing to
merge.

Re-measured **interleaved** (baseline and branch alternating as separate
subprocesses, round-robin, min over 4 rounds), both claims evaporate:

| case | PRE-P2C | P2C fast | speedup |
|---|---|---|---|
| 16 identical layers | 0.3591 s | **0.0514 s** | **6.98x** |
| ABAB, 8 periods (16 layers, 2 distinct) | 0.3801 s | 0.3006 s | **1.26x** |
| 16-slice tapered, ALL distinct | 0.6739 s | 0.6781 s | **0.99x** |
| 5 distinct layers, conical | 0.6375 s | 0.6369 s | **1.00x** |

Per-round spreads were tight (e.g. tapered baseline
`[0.6829, 0.6739, 0.6883, 0.6764]`), so the sequential pass's 1.39x was box
drift between the two runs, not a gain.  **The honest claim is: large where
there is structure to exploit, exactly neutral where there is not.**  A
sequential before/after on this box is not a valid instrument at the 1.3x
level; only the interleaved numbers are reported below.

### S5.2  Where the speedup comes from, and where it does not

* **Adjacent identical layers -> up to 7x, and the cost stops scaling.**  A
  repeated stack now costs what ONE layer costs: measured across the sequential
  sweep, `nlay` = 2 / 4 / 8 / 16 all landed at 0.054 s (against 0.081 / 0.118 /
  0.219 / 0.438 s before).  That is the merge turning an O(N) cascade into
  O(1), which is the structural result, not a constant-factor one.
* **Non-adjacent repeats (ABAB) -> 1.26x**, from interface dedup alone: a
  16-layer ABAB stack has only 2 distinct layer-layer interface pairs plus the
  2 half-space ones, where before it built 17.
* **All-distinct stacks -> 1.00x.**  Nothing to dedup, nothing to merge, and --
  importantly -- **no regression** from the added keying and cache lookups.
* **Cross-call re-solve -> 1.65x (distinct) / 4.44x (repeated)**, interleaved,
  5 layers at `n_orders=5`.  This is the new `_eig_cache`: the second and later
  `solve()` on one object at one source do no eigensolves at all, where before
  they redid every one.

  | case | PRE-P2C | P2C fast | speedup |
  |---|---|---|---|
  | re-solve, 5 distinct layers | 0.5889 s | 0.3561 s | 1.65x |
  | re-solve, 5 identical layers | 0.4049 s | 0.0912 s | 4.44x |

* **Threaded layer eig -> 1.52x at 8 workers** on the one workload that has
  anything to fan out (8 DISTINCT tapered slices, `n_orders=5`), interleaved,
  branch-vs-branch:

  | `max_workers` | time | speedup |
  |---|---|---|
  | 1 | 0.9824 s | 1.00x |
  | 4 | 0.6863 s | 1.43x |
  | 8 | 0.6465 s | 1.52x |

  Well short of linear, and expectedly so: S2 showed the eig is only ~40% of a
  distinct-layer solve and the Redheffer cascade after it is strictly
  sequential, so Amdahl caps this near 1.6x whatever the worker count.  8
  workers on a 24-core box buys 0.09x over 4.  **This is the weakest of the
  four levers and is opt-in for that reason** -- the process-global BLAS cap it
  must enter makes it actively harmful if nested inside another threaded
  driver.

### S5.3  Cache footprint against its priced bound

Measured per layer-entry (root-buffer accounting, one patterned 6x6 cell,
degree 9):

| `n_orders` | `Nf` | geom | eig | total/entry | 500-pt dispersive sweep wants |
|---|---|---|---|---|---|
| 4 | 81 | 0.747 MB | 0.842 MB | 1.589 MB | 0.79 GB |
| 5 | 121 | 1.523 MB | 1.878 MB | 3.401 MB | 1.70 GB |
| 6 | 169 | 2.859 MB | 3.661 MB | 6.520 MB | 3.26 GB |
| 8 | 289 | 8.135 MB | 10.700 MB | 18.835 MB | **9.42 GB** |

The measured budget on this box was **5.60 GB** (5% of the RAM budget), which
bounds retention at ~7492 / 3682 / 1960 / 687 geom entries respectively.  The
`n_orders=8` row is the case that matters: a 500-point dispersive sweep wants
9.42 GB and the pre-P2C bare dict would simply have taken it.  Now the cache
refuses past the budget and the sweep completes with byte-identical results at
the cost of recomputation.

Note what is and is not fixed: the footprint is still **linear in sweep length
up to the budget**, and then it stops.  The bound is the fix; the linearity
below it is the cache doing its job.  Verified directly at 12 / 24 / 48 points
(8.41 / 16.82 / 33.64 MB geom, 0 refused, 0 evicted -- correctly below budget),
and the bound's engagement is pinned by tests that FORCE a small budget rather
than wait for a big box.

## S6.  Fail-before evidence, and a real bug this branch's own test caught

`tests/unit/test_p2c_pmm2d_stack_cascade.py`, 27 tests, green at
`OPENBLAS_NUM_THREADS` = 1, 2 and 4.

| claim | how it is made build-free |
|---|---|
| merged run == one explicitly thick layer | independent oracle (a 1-layer stack has no run to merge); bar derived from the build's own `r` |
| one eps entry apart must not share a solve | **engineered injector**: the DECISION is asserted on `_mode_key` and on the eig cache holding 2 entries; the CONSEQUENCE is scanned up a ladder derived from the running build's bar, hard-failing only when exhausted (a fixed nudge is the parametrized-injection sub-shape the standards name) |
| the key carries the SOURCE | asserted on the key for wl / k0 / kx0 / ky0, then confirmed by a re-solve at a second wavelength moving |
| cascade order | reversing an asymmetric stack must move the answer >100x the bar, and each order is checked against its own monolithic path |
| merge is ADJACENT-only | asserted on the cascade sequence length (3 for A-B-A, 2 for A-A-B) |
| overflow regime | thickness derived from the layer's OWN measured eigenvalues so that `exp(+lam k0 t)` provably overflows float64 on this build (asserted non-finite), then the S-cascade is shown finite, per-pol `R+T <= 1`, `T < 1e-12`, and agreeing with monolithic |
| cache pricing | **two-sided and forced**: priced IN (`cache_max_bytes=4 GB` -> 4 entries, 0 refused, 0 evicted) and priced OUT (`cache_max_bytes=1` -> 0 entries, refusals > 0, and the answer BIT-IDENTICAL to the rich run).  No `pytest.skip` anywhere |
| no self-eviction | budget forced to ~2.5 entries for a 5-layer solve: `evicted == 0`, `refused > 0`, answer unchanged |
| the unbounded-growth defect | doubling the sweep must NOT double the footprint, and the footprint must stay under a FORCED budget -- stated as a relation, not as the pre-fix MB readings (which are recorded here in S2 instead) |

**The bug the byte-identity test caught in this branch's own code.**  The first
version of `_layer_mode_sets` entered the BLAS cap only on the pooled branch
and ran the serial branch uncapped.  With `OPENBLAS_NUM_THREADS=2` in the
environment, `max_workers=1` and `max_workers=2` then disagreed on `R` -- the
M4 defect, reintroduced.  The fix makes `max_workers` select between two
explicit contracts: `None` (the default) is serial with NO cap entered and is
bit-for-bit the pre-P2C path; any INT enters one process-wide cap around
**both** branches, so `max_workers` = 1, 2, 4, 8 are byte-identical to each
other.  The test asserts identity *within* the threaded contract and only
bar-agreement *across* the two -- asserting bit-equality across them would be
the S3 environment-dependent shape.

## S7.  A pre-existing environment-dependent test on `origin/main`

`tests/unit/test_v5_14_1_device_geometry.py::test_pmm2d_stack_dispersive_sweep`
asserts `np.max(np.abs(R[i] - R1)) == 0.0` between a `solve_vs_wavelength`
point and a standalone control `solve()`.  Measured on **`origin/main`
@40f28ae, unmodified**:

| `OPENBLAS_NUM_THREADS` | result |
|---|---|
| 1 | passed |
| 2 | **failed**, `max\|R - R1\|` = 8.327e-16 |
| 4 | **failed** |
| 8 | **failed** |

The mechanism is the M4 contract seen from the other side: the sweep runs every
point under one process-wide cap of `blas_per_worker` (=1) threads, while the
control runs at the AMBIENT count.  When those differ, the two land on
different BLAS reduction orders and a `== 0.0` bar is a statement about the
box, not about the library -- the S3 shape with an S4 floor bar at zero.  It
survived the 5.35 whole-suite sweep because that sweep's runs all had the caps
exported to 1.

Fixed here by **forcing the precondition** (rule 4) rather than weakening the
claim: the control loop runs inside `_blas_threads_quiet(1), _blas_limit()`, so
bit-equality is true by construction.  Green at 1, 2, 4 and 8 on the branch.
This is a test-side fix; no library behaviour changes.

### Regression run

`pytest tests/unit -k "pmm or stack or rcwa"` on the branch, BLAS pinned to 1:
**1849 passed, 1 failed, 2 skipped** (51m23s).

The one failure is
`test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_fff_nv_stripe_reduces_to_rigorous_1d`,
which fails **identically on the pristine `origin/main` worktree** (checked
side by side, both `1 failed in ~1.4s`).  It is in `rcwa_jones_2d`'s FFF-NV
surface -- a module this branch does not touch -- and is left alone as out of
scope.  It is recorded here so the next reader does not attribute it to this
change.

## S8.  What is NOT claimed

* The even-symmetry (`symmetry='auto'`) fold path is untouched: when it
  engages it bypasses the cascade entirely, so `fast` and `monolithic` are
  identical there by construction (and the matrix confirms it).
* `retain_internal=True` forces the per-layer cascade -- the partial cascades
  are indexed per layer, so the merge is off there.  Dedup still applies and
  is byte-identical, so `retain_internal` results are bit-equal to
  `monolithic`.
* The merge is a *cascade* optimization; it does not change the hybrid's
  `n_orders` Fourier floor, its conditioning at degree >= 8, or any
  convergence property.
* `max_workers > 1` must not be passed from inside another threaded driver.
  `solve_vs_wavelength` does not pass it.
* No speedup is claimed for all-distinct stacks.  S5.1 records that a
  sequential benchmark appeared to show one and that it was an artefact.

## S9.  Files

| file | change |
|---|---|
| `lumenairy/elements/pmm/_stack2d_cache.py` | NEW -- `LayerCache` (LRU + lock + RAM-priced budget + generation guard) and `cached_nbytes` |
| `lumenairy/elements/pmm/stack2d.py` | `cascade=` / `cache_max_bytes=` ctor args; `_mode_key` / `_build_layer_modes` / `_layer_mode_sets` / `_cascade_sequence` / `_cascade_sequence_general` / `_interface_memo` / `clear_cache` / `cache_stats`; `solve(max_workers=, blas_per_worker=)`; both caches repriced |
| `tests/unit/test_p2c_pmm2d_stack_cascade.py` | NEW -- 27 tests |
| `tests/unit/test_v5_14_1_device_geometry.py` | S7 -- the pre-existing BLAS-thread-count dependence |
| `CHANGELOG.md` | `[Unreleased]` |

No public API is removed or repointed; `PMM2DStack`'s transitional
`DeprecationWarning` and the `PMM2DStackPure` cutover plan are untouched.

# Thread-count-dependent bars in the fff_nv and PMM/RCWA-upgrade suites

2026-08-08.  Branch `feat/pmm-per-layer-roadmap` (tree at `2275e2e`).

`FIX_UNION_GRID_2THREAD_2026_08_06.md` S9.6 referred two failures that its
blast-radius sweep turned up and its switch evidence exonerated:

```text
test                                                 1 thr  2 thr  default
v5_20_13_pmm_jones_2d_fff_nv::                       pass   FAIL   FAIL
  test_pmm_fff_nv_matches_rcwa_fff_nv
v5_20_0_pmm_rcwa_upgrades::                          pass   FAIL   FAIL
  test_stack_sweep_geom_cache_is_transparent_and_reused
```

Both fail with byte-identical numbers whether `PMM_FORWARD_GROWTH_REPAIR` is
on or off, so neither is caused by the union-grid work.  Both are the standing
disease of this campaign, named in `PMM_FOURNAME_ADJUDICATION_2026_08_05` and
again in `PMM_M1_CONDITIONING_2026_08_04`: **a per-configuration magnitude
asserted as a universal constant.**  This document fixes both, sweeps the two
files for siblings, and reports the two siblings that sweep found.

Nothing outside `tests/unit/test_v5_20_13_pmm_jones_2d_fff_nv.py`,
`tests/unit/test_v5_20_0_pmm_rcwa_upgrades.py` and this file changed.  No
library change was needed: **neither failure is a solver defect**, and S2/S3
say why with the measurement rather than with an opinion.

---

## S1.  Fail-before, reproduced first

Windows = python 3.14.6 / numpy 2.4.4 / scipy 1.17.1, scipy-openblas 0.3.31
Haswell, MAX_THREADS 24, `threadpool_info()` non-empty (so the library's own
BLAS cap is LIVE on this interpreter -- it is inert on some, which matters in
S3.4).  WSL = python 3.12.3 / numpy 2.4.6 / scipy 1.17.1, OpenBLAS Haswell,
MAX_THREADS 24, venv `~/lumen_venv` (threadpoolctl 3.6.0 present).

```text
mount    OPENBLAS_NUM_THREADS   selection                      result
Windows  2                      the two names                  2 failed (24 s)
WSL      2                      the two names + the whole      2 failed,
                                second file                    19 passed,
                                                               2 skipped (36 s)
Windows  1                      both files                     27 passed (208 s)
```

The Windows 2-thread failure messages, verbatim:

* `AssertionError: PMM reference unstable` -- `|sum R + sum T - 2|` = 2.820e-03
  against the 1e-3 guard (the referring document recorded the *RCWA* guard
  firing at 2.9865e-03; that is the same test failing at the DEFAULT thread
  count instead.  Both sides of the same comparison break, at different thread
  counts -- S2.1);
* `assert np.array_equal(R[i], R1) and np.array_equal(T[i], T1)` -> `False`.

---

## S2.  Name 1 -- `test_pmm_fff_nv_matches_rcwa_fff_nv`

### S2.1  What the guard was actually reading

The test compares `pmm_jones_2d(fff_nv)` with `rcwa_jones_2d(fff_nv)` on a
rotated-director stripe, and since 2026-08-04 it first asserts that each engine
conserves (`|sum R + sum T - 2| < 1e-3`) so the comparison cannot be read
against an unstable answer.  That guard was right in kind and wrong in form:
the truncation each engine was asked to conserve AT was hard-coded.

Holding code, build and geometry fixed and varying only
`OPENBLAS_NUM_THREADS`, the closure `sum R + sum T - 2` of each rung
[Windows]:

```text
rcwa n_orders_x      1 thread     2 threads    24 threads
              7      -2.40e-05    -1.81e-02    +2.26e-04
              9      +3.10e-07    +3.86e-03    -6.07e-06
             11      -2.26e-05    -9.87e-05    +2.99e-03   <- was pinned here
             13      +2.64e-02    -5.54e-06    +5.55e-04
             15      +1.59e-04    +1.59e-04    +1.24e-03

pmm n_orders (degree 11)
              7      +1.72e-03    +2.86e-03    +1.72e-03
              9      -1.75e-04    -1.75e-04    -1.75e-04
             11      +2.86e-04    -5.35e-04    -2.81e-04
             13      -2.93e-04    +2.82e-03    +1.60e-04   <- was pinned here
             15      +3.38e-04    +3.38e-04    +3.38e-04
```

Every thread count has clean rungs and unstable rungs.  They are simply not the
SAME rungs.  The pinned pair happened to be clean at one thread -- which is
where the 2026-08-04 fix measured them -- so the test passed there, failed at 2
on the PMM side (2.82e-03) and failed at the default on the RCWA side
(2.99e-03).  One disease, reported from two sides, which is why the referring
document saw a different message than the one that reproduces at 2 threads.

Two further readings matter:

1. **`sum(R)` is essentially thread-invariant** for both engines -- twelve
   identical digits at 1 / 2 / 24 threads for every rcwa rung except 15.  The
   defect lives entirely in the transmitted orders, i.e. in exactly the
   quantity the engines' own tripwire calls suspect ("the PER-ORDER
   efficiencies are suspect").
2. **Closure is necessary and NOT sufficient.**  At one thread, rcwa
   `n_orders_x = 15` closes to 1.59e-04 -- comfortably inside the 1e-3 guard --
   while its `sum(R)` reads 0.064400 against the 0.06181 the 7/9/11 cluster
   agrees on: 2.6e-03 away, i.e. R and T redistribute between thread counts at
   FIXED closure.  A guard that only asked "does it conserve" would have
   admitted that rung as a reference.

### S2.2  The restructure

Nothing is pinned any more.

* **The reference is chosen on the run that is executing.**  `_scan` solves the
  rcwa ladder `(7, 9, 11, 13, 15)` -- 0.1-1.0 s a rung, so the whole ladder is
  cheap -- and scores each rung by its OWN closure.
  `_corroborated_reference` then takes the CLEANEST rung whose sums another
  clean rung reproduces to `_AGREE_TOL = 2e-3` (half the comparison bar).  The
  corroboration step is what rejects the `n_orders_x = 15` case in S2.1.2.
* **If no rung qualifies, the test fails naming the reference** and printing
  the whole ladder -- "This is a statement about the reference, NOT about
  pmm_jones_2d" -- instead of comparing against a non-closing reference and
  blaming the PMM.
* **The subject is every pmm rung that conserves**, not one.  The pmm ladder is
  `(9, 11, 13)`, cheapest first, and the scan stops once two rungs have closed.
  `n_orders = 7` is excluded from the ladder by measurement, not by taste: it
  misses closure by 1.7e-03 .. 2.9e-03 at every thread count, i.e. it is
  under-resolved rather than unstable.
* **The 4e-3 cross-solver bar is UNCHANGED**, and so is the 1e-3 closure
  tolerance -- it is now a per-rung classifier instead of a pass/fail assertion
  on a pinned rung.

What the test asserts is therefore strictly MORE than before (2-3 pmm
truncations against a corroborated reference, instead of one against one), and
the selection it makes is visible in the run log:

```text
1 thread    reference rcwa n_orders_x=9  (closure 3.097e-07); pmm rungs [9, 11]
2 threads   reference rcwa n_orders_x=13 (closure 5.542e-06); pmm rungs [9, 11]
default     reference rcwa n_orders_x=9  (closure 6.068e-06); pmm rungs [9, 11]
```

### S2.3  Headroom

Worst cross-solver residual over ALL clean pmm rungs against the selected
reference, over the three thread counts [Windows]: **4.7e-04 in `sum(R)` and
7.6e-04 in `sum(T)`** against the 4e-3 bar -- 5x.  For comparison the pinned
pair was reading 2.7e-02 at the truncation the 2026-08-04 fix moved off.

### S2.4  Teeth

The bar is not vacuous and the refusal is not decorative:

```text
probe                                              result
pmm solved on a 2 % thicker layer (same cell,      clean rungs [9, 11],
  reference untouched)                             |dR| 8.1e-03 / 8.3e-03
                                                   -> FIRES (bar 4e-3)
_CLOSE_TOL squeezed to 0 (no rung can qualify)     _corroborated_reference
                                                   -> None -> the test refuses
                                                   with the REFERENCE named
```

Note the first row: the 2 %-wrong solves still CONSERVE (closure 1.6e-04 /
5.5e-04).  Energy closure does not catch them -- the lossless trap, again --
and the cross-solver bar does.  That is the division of labour the restructure
preserves.

### S2.5  Cost

The test got FASTER, because the scan stops at two clean rungs and those are
the cheap ones.  Windows, one thread: **34.8 s -> 22.7 s** (`--durations`,
before vs after) -- the single pinned `n_orders = 13` solve cost 36 s on its
own, against 4.7 s + 14.1 s for the two rungs the scan actually uses.  After,
at two threads: 14.8 s (the before-time at 2 threads is not comparable -- the
test failed there).  The rcwa ladder adds 0.4 s at one thread and 2.6 s at 24.

---

## S3.  Name 2 -- `test_stack_sweep_geom_cache_is_transparent_and_reused`

### S3.1  The bit-identity crossed a boundary the sweep itself introduces

The test ran a 3-wavelength `solve_vs_wavelength`, then a bare `solve()` per
wavelength on a fresh stack, and asserted `np.array_equal`.  But
`PMM2DStackHybrid.solve_vs_wavelength` wraps the WHOLE dispatch in

```python
with _blas_threads_quiet(blas_per_worker), _blas_limit():
```

with `blas_per_worker = 1` by design -- byte-identity across worker counts
needs every solve, serial and threaded, at the SAME BLAS thread count, and
applying a cap is process-global on OpenBLAS (the M4 referral,
`PMM_M4_HYGIENE_2026_08_04.md` S2.6).  So the sweep solved at ONE BLAS thread
and the reference at N, and `np.array_equal` was reading the OpenBLAS reduction
order.

Measured, Windows, over the three sweep wavelengths [absolute max over R and
T]:

```text
reference for the sweep's result          2 threads              24 threads
bare solve(), AMBIENT threads             False  5.2e-14/1.2e-14  False  2.6e-13
bare solve(), under the sweep's own cap   True   0 / 0            True   0 / 0
cache-COLD one-wavelength sweep           True   0 / 0            True   0 / 0
```

That is the whole failure.  It passed at `OPENBLAS_NUM_THREADS=1` because
there is no boundary there (1 == 1), and the cache -- the thing the test is
named for -- was never implicated: its own control (row 3, a cache-cold sweep
of the same wavelength through the same code path) is bit-exact at every
thread count.

### S3.2  The restructure

The bit claims are made where they are true by construction, and the
cross-boundary claim is kept as what it actually is:

* **(a) the cache control** -- the 3-wavelength sweep (entry built once, reused
  twice) against a cache-COLD one-wavelength sweep.  Same code path, same cap;
  the cache is the only difference, so `np.array_equal` here is a statement
  about the cache and nothing else.
* **(b) the sweep/solve control** -- a bare `solve()` run under the cap the
  sweep applies, mirroring `_solve_one` exactly
  (`_blas_threads_quiet(1), _blas_limit(), _blas_threads_quiet(None)`).
  Bit-identical, which is what makes (c) a thread-count statement rather than a
  physics one.
* **(c) the cross-boundary comparison** -- a bare `solve()` at the AMBIENT
  thread count, asserted at 1e-9: the same physics, the sweep's internal cap
  does not move it.

`1e-9` is the file's own dedup bar and sits nearly four decades over the
measured 2.6e-13 residual.  `rtol = atol = 1e-12` is what this identical
disease got in `test_v5_14_0_pmm2d_stack::test_sweep_matches_per_wavelength`
under audit S5-12 -- on THIS cell (the stack warns `max R+T = 1.03` at its own
truncation, so the conditioning is poor) 1e-12 would sit only 4x over the
24-thread residual, which is not headroom.

### S3.3  Teeth

A cache that is NOT transparent is still caught, by both surviving controls.
Poisoning the geometry-cache entry with the build of a slightly different cell
(eps 6.0 -> 5.5) under the same key -- what a too-loose `_geom_key` would do:

```text
control                                           reading
(a) cached sweep vs cache-cold sweep              array_equal False, |dR| 7.4e-03
(c) cached sweep vs ambient bare solve()          |dR| 7.4e-03 vs the 1e-9 bar
```

### S3.4  A note for whoever runs this next

`_blas_limit()` needs `threadpoolctl`; without it the cap is inert and the
sweep runs at the ambient thread count like everything else.  The WSL venv
`~/lumvenv` (which `FIX_UNION_GRID_2THREAD_2026_08_06.md` used) has no
threadpoolctl, so this failure CANNOT reproduce there -- `~/lumen_venv` (which
has it, 3.6.0) is the venv that shows it, and is what S1 and S5 use.  The
rewritten test is correct in both environments: with the cap inert, (a), (b)
and (c) all run at the ambient count and all three are bit-exact.

---

## S4.  Sibling sweep of the two files

Every numeric assertion in both files was instrumented and re-measured at 1 / 2
/ max BLAS threads on Windows and at 1 / 2 on WSL (`threadpoolctl` sets the
effective count in-process, cross-checked against the env-var runs -- the
1-thread column reproduces the env-var numbers to all printed digits; the WSL
max-threads column of the instrumented probe was abandoned, see S4.2, and that
CELL is covered as pass/fail by S5).  "Worst observed" is over every cell
measured.  Ratio = observed / bar; smaller is more headroom.

```text
assertion (file / test)                          worst observed   bar     worst ratio
fff_nv: 1-D reference closure  [NEW GUARD]        3.8e-13         1e-3    4e-10
fff_nv: ef, fff_nv vs rigorous 1-D                9.1e-05         1e-3    0.09
fff_nv: ef < 0.2*el (vs laurent)                  0.062           1.0     0.06
fff_nv: lossy absorptance split  [WAS 0.998]      see S4.1
fff_nv: uniform cell fff_nv == laurent            0.0             1e-12   0
upgrades: symmetry fold te/tm (R, T)              6.7e-14         1e-11   0.007
upgrades: stack dedup (J, R, T)                   7.8e-14         1e-9    8e-05
upgrades: circular truncation closure             2.7e-03         1e-2    0.27
upgrades: circular vs rectangular R0              1.1e-04         5e-3    0.02
upgrades: solve_vs_wavelength angle trio          0.0             1e-12   0
upgrades: stack symmetry fold (R, T, J)           5.8e-14         1e-11   0.006
upgrades: tensor jones fold, li (dJ)              1.6e-12         1e-11   0.16  <- S4.2
upgrades: tensor jones fold, laurent (dJ)         5.8e-12         1e-11   0.58  <- S4.2
upgrades: slant closure after shared eig          2.9e-06         1e-5    0.29
```

Everything except the two rows called out below has at least 3.4x, and mostly
2-13 decades.  Two rows are not that.

### S4.1  Lossy absorptance split: the reference double-counted the polarizations (FIXED)

At 0.998 of its own bar in every cell measured -- the tightest reading in
either file, and thread-INVARIANT (identical to all printed digits at 1, 2 and
24 threads on both mounts), which is what said it was not this campaign's
disease.  It is a different one.

`rcwa_jones_1d_segments` returns `R`/`T` shaped `(2, 2*n_orders+1)`: one row
per incident polarization, each row a complete energy budget.  The test's
reference absorptance summed BOTH rows:

```text
A1 = 1 - sum(R1) - sum(T1)              = -0.140868      <- not an absorptance
A1 per polarization                     = [0.436884, 0.422248]
pmm fff_nv  per polarization            = [0.436935, 0.422256]
pmm laurent per polarization            = [0.438720, 0.422247]
```

Both engines were then compared against a target 0.57 away from either of them,
and the assertion survived only because that constant offset is common to both
sides and cancels in the difference:

```text
quantity                          fff_nv       laurent     fff_nv is
vs the (2-row) reference          0.570464     0.571352    0.16 % closer
vs the per-pol reference          2.967e-05    9.177e-04     31x closer
per-pol, worst polarization       5.082e-05    1.836e-03     36x closer
```

The claim ("fff_nv's absorptance tracks the rigorous 1-D split at least as
closely as laurent") is unchanged and is now made per polarization, row for
row, against a reference that is an absorptance.  It carries a factor of 36
instead of 0.16 %.  Identical at 1, 2 and 24 threads on both mounts before and
after.

### S4.2  Tensor jones symmetry fold: 1.7x headroom (REFERRED, not touched)

The even-parity fold vs the full tensor solve, `max |dJ|`, against a 1e-11 bar
-- both parametrizations of the same test:

```text
formulation  mount    1 thread   2 threads   max threads
laurent      Windows  5.815e-12  9.662e-13   2.154e-14
laurent      WSL      5.071e-14  1.231e-13   (not measured)
li           Windows  8.097e-14  3.964e-14   1.508e-13
li           WSL      7.829e-14  1.573e-12   (not measured)
```

(The WSL max-threads column of the instrumented sweep was abandoned: the WSL
VM suspends whenever no `wsl` command is in flight, so a background probe there
makes no progress.  The WSL max-threads CELL of the suite itself is covered in
S5 as pass/fail.)

It passes in every cell measured, but the quantity ranges over 270x with the
thread count and the mount and its worst reading (laurent, Windows, one thread)
is within 1.7x of the bar -- the same shape as the two names fixed here (an
absolute bar on a magnitude that moves with the reduction order), one build away
from being the next referral.  Note it is not a property of one formulation:
`li` reaches 1.6e-12 on WSL at two threads while `laurent` is at 1.2e-13 in the
same cell, so whichever is worst depends on the cell, not on the projection.

It is NOT a reference-closure bar, no cell fails, and repairing it means
deciding what an even-basis fold residual should be measured in units OF --
so it is recorded here with its numbers rather than adjusted inside a fix for
something else.

---

## S5.  Result

`-q -p no:randomly`, both files together (27 tests).

```text
mount    OPENBLAS_NUM_THREADS   after
Windows  1                      27 passed                   216 s
Windows  2                      27 passed                   185 s
Windows  default (24)           27 passed                   181 s
WSL      1                      24 passed, 3 skipped        221 s
WSL      2                      24 passed, 3 skipped        115 s
WSL      default (24)           24 passed, 3 skipped        513 s
```

The three WSL skips are `pytest.importorskip("jax")` --
`test_cell_jax_path_factorized_matches_numpy`,
`test_cell_jax_path_honours_max_nodal_dof` and
`test_pmm_fff_nv_crossed_and_offplane_and_jax_raise`; that venv has no jax,
unrelated and unchanged by this work.  (The WSL default-threads leg is slow
because an unrelated full-suite run was holding the box; it is a wall-clock
artefact, not a numeric one.)

The before column, measured rather than quoted, and per cell because that is
the point of the exercise:

```text
mount    threads   before
Windows  1         27 passed (208 s) -- both names PASS here, which is the
                   whole disease
Windows  2         both names FAIL (S1 has the verbatim messages); the second
                   file alone: 1 failed, 21 passed
Windows  default   the pinned rcwa rung's closure reads +2.98653e-03 at 24
                   threads (S2.1), i.e. the old `RCWA reference unstable`
                   guard fires -- and that is the referring document's
                   2.9865286872832186e-03 to six digits; the geom-cache
                   identity likewise fails (array_equal False, |dR| 2.6e-13)
WSL      2         both names FAIL: 2 failed, 19 passed, 2 skipped over
                   `::test_pmm_fff_nv_matches_rcwa_fff_nv` + the whole second
                   file
```

Third environment, for the S3.4 point: the geom-cache test alone on WSL
`~/lumvenv`, which has NO threadpoolctl (so the sweep's BLAS cap is inert and
the boundary does not exist) -- 1 passed at 1, 2 and default threads.

`ruff check` clean on both files.  Every file touched is pure ASCII
(cp1252-safe).

---

## S6.  Open

1. **S4.2 is open** -- `test_jones_2d_tensor_symmetry_fold_matches_full`,
   both parametrizations: 1.7x headroom on a quantity that moves 270x with the
   thread count and the mount.  Numbers above.
2. **The geom-cache cell is near-singular.**  `PMM2DStack(degree=9,
   n_orders=3)` on that 6x6 cell warns `energy not conserved (max R+T = 1.03)`
   on EVERY solve, sweep and single -- the test is a transparency test so this
   does not invalidate it, but it does mean the cell amplifies reduction-order
   noise (2.6e-13 at 24 threads, against 1e-14-ish on the well-conditioned
   siblings in the same file).  Raising the truncation until it conserves is a
   separate change with a separate cost.
3. **The `blas_per_worker=1` boundary is a library-wide property, not a
   property of this test.**  Any test that compares a `solve_vs_wavelength`
   result with a bare `solve()` bit-for-bit is the same latent failure;
   `test_v5_14_0_pmm2d_stack::test_sweep_matches_per_wavelength` already
   learned it (S5-12, `rtol=atol=1e-12`), and this file is the second.  A sweep
   over the other 18 files that call `solve_vs_wavelength` was NOT done here.
4. **Neither name needed a library change.**  S2 and S3 are both test-side
   restructures; `lumenairy/elements/rcwa/**` is untouched, and no
   reference-side numerics defect was proven (the rcwa reference's per-rung
   instability is a property of this cell's truncation ladder, and the fix is
   to select a rung that is clean on the run rather than to change the solver).

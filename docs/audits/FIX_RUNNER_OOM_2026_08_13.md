# FIX -- the v5.35.0 release runner was killed by an unpriced fit

`fix/runner-oom`.  Files: `lumenairy/elements/_lens_imap.py`,
`lumenairy/elements/_lens_traced.py`,
`tests/unit/test_fix_runner_oom_2026_08_13.py`,
`validation/integration/test_subsample.py`.

---

## 0. VERDICT

**Two blockers, two different causes, one shared origin: v5.35.0 turned the
C15 inverse characteristic on by default, and nothing on that path was priced
against the box it runs on.**

1. **The publish `verify` shard died of RAM, not of time.**  The killed test is
   `test_niche_c1_consolidation.py::test_the_element_reports_the_measured_exit_na_and_the_aliased_power`,
   whose second `apply_real_lens_traced` call reaches **5,764,801 retained fit
   samples at degree 14 (P = 120 terms)** -- a 5.53 GB float64 design matrix.
   Measured peak transient with the RAM budget pinned to a runner-sized
   6.5 GB: **32.9 GB**.  The 128 GB dev box simply took it, which is why every
   pre-merge gate was green.  Now **4.3 GB** (7.7x), and the call is 3.3x
   faster as a side effect (64.5 s -> 19.4 s in-suite).
2. **`Validate` was a STALE EXPECTATION, not a bug and not an OOM.**
   `validation/integration/test_subsample.py` asserted that the exit-phase
   error must GROW with `ray_subsample` (10-50 / 50-200 / 200-600 nm bands).
   That is a property of the coarse-Newton + `map_coordinates` upsample path.
   With the inverse map engaged the measured error collapses to **0.006 nm at
   every `sub`** -- the suite was failing a strictly better answer.

No test was skipped, xfailed, or shrunk.  The physics claim of every failing
check still runs and still passes; §4 states where each one now lives.

---

## 1. THE EVIDENCE, AND WHY IT READ AS DETERMINISTIC

Both `verify` legs (`3.12 / shard 1` and `3.10 / shard 1`, both on the tag
commit `667e0d1`) end identically:

```text
tests/unit/test_niche_c15_inverse_map.py .......                        [ 46%]
##[error]The runner has received a shutdown signal.  This can happen when the
         runner service is stopped, or a manually started runner is canceled.
tests/unit/test_niche_c1_consolidation.py ....
##[error]The operation was canceled.
```

Three things in that shape:

* **`shutdown signal`, not a timeout and not an assertion.**  The step cap is
  30 minutes and the job died at ~5 and ~6 minutes.  A hosted runner reports a
  *kernel* kill of the runner service this way.
* **`....` then nothing.**  Four tests of the file completed; the fifth was
  in flight.  The gap between the `c15` line and the kill is 42 s (3.12) and
  44 s (3.10) -- consistent with a test that `.test_durations` records at
  **54.09 s**, dying partway.
* **Same file on two different Pythons whose shard membership differs** (2225
  vs 2204 selected tests; `test_audit_misc.py` is in 3.12's shard-1 and in
  neither log).  A shared member that costs 30 GB is the only thing both
  selections have in common at that point.

`test_the_element_reports_the_measured_exit_na_and_the_aliased_power` is the
only test in the file whose recorded duration is above 11 s, and
`pytest-split`'s `least_duration` packs the longest items first -- it is in
shard 1 on every leg.  Confirmed by construction: splitting the file alone
puts that single test in group 1/5 with an "estimated duration: 54.09s".

`c15`'s own tests did NOT inflate: 5 dots (3.12) and 7 dots (3.10) is shard
membership, not growth, and its heaviest recorded test is 2.18 s.

---

## 2. WHERE THE 33 GB WAS

Watchdog stack dumps at 8 / 12 / 18 / 25 / 31 GB crossings, RAM budget pinned
to 6.5 GB, on the killed test's own second call (2048^2 grid,
`ray_subsample=2`):

| crossing | frame |
|---|---|
| 8, 12, 18 GB | `_lens_imap._td_design` <- `build_inverse_map` |
| 25, 31 GB | `_lens_imap._td_design_grad` <- `build_inverse_map` |

Three unbounded constructions, all the same shape -- a fancy-index GATHER per
factor plus their product, each a full `(n_samples, P)` float64 array:

```python
# _lens_imap._td_design            3 x n*P*8 live at once
Vx[:, terms[:, 0]] * Vy[:, terms[:, 1]]
# _lens_imap._td_design_grad       4 gathers + 2 products, WITH A alive
(Dx[:, t0] * Vy[:, t1], Vx[:, t0] * Dy[:, t1])
# _lens_traced._Cheb2DEvaluator    3 x n*n_terms*8, then a contiguous copy
(Tu_np[K1_np] * Tv_np[K2_np]).reshape(n_terms, -1).T
```

At `n = 5,764,801`, `P = 120` one copy is 5.53 GB.  The measured peak was
**5.9 x** that.  Nothing upstream bounds either factor: `n` is the retained
launch lattice (grid size / `ray_subsample`) and `P` is the exit degree.

The RAM clamps that DO exist on this path were all calibrated for a big box
and none of them cover this: `parallel_amp_min_free_gb = 48.0` (correctly
disengages on a runner, and is about a different array), and the Newton-pool
clamp (`_NEWTON_POOL_RAM_FRAC`, `_NEWTON_POOL_MIN_FREE_GB`) prices WORKER
COUNT, not the serial fit.  The inverse-map build -- new-by-default in
v5.35.0 -- had no pricing at all.

---

## 3. THE FIX, AT THREE LAYERS

### 3.1 Row-blocking (memory only, bit-identical)

`_td_design`, `_td_design_grad` and `_Cheb2DEvaluator.__init__` now build into
a preallocated output in row blocks bounded by `_IMAP_FIT_CHUNK_ENTRIES` /
`_CHEB_FIT_CHUNK_ENTRIES` = 8e6 float64 entries (64 MB) of scratch,
independent of `n` and `P`.  These are **elementwise products with no
reduction**, so there is no summation order to change and the bits are
identical -- asserted against the retired expressions, recomputed verbatim in
the test file, at several degrees and across forced chunk boundaries.

One layout note that is load-bearing: `_Cheb2DEvaluator` now builds its design
C-contiguous rather than as the transpose of a C-contiguous `(n_terms, n)`
buffer.  That is safe *only* because `_solve_lstsq_thread_safe` opens with
`np.ascontiguousarray(A)` -- it always squared a C-contiguous copy, so the
Gram `A.T @ A` sees the identical array and the same BLAS reduction order.
It also retires that copy, which is the second 1.29 GB saved.

### 3.2 GRAM -- the inverse-map build is priced before it runs

`build_inverse_map` gained a guard alongside G1-G8:

```text
need   = _IMAP_BUILD_PEAK_COPIES * n_good * P * 8  +  3 * _IMAP_FIT_CHUNK_ENTRIES * 8
budget = _IMAP_BUILD_RAM_FRAC * lumenairy.get_ram_budget()
need > budget  ->  refuse ('GRAM'), keep the incumbent path
```

`_IMAP_BUILD_PEAK_COPIES = 4.0` is measured, not assumed (the design matrix
stays alive across the weighted solve, the LAPACK working copy, and the two
gradient designs), and the test file re-measures it against the shipped code.
`_IMAP_BUILD_RAM_FRAC = 0.5` because the map is never the only thing in the
process.

**Refuse, never degrade.**  The refusal returns the caller to the shipped
coarse-Newton + `map_coordinates` path -- the same outcome as
`TRACED_INVERSE_MAP = False`, the module's documented fail-before switch.  That
is asserted byte-for-byte, not asserted about
(`test_a_gram_refusal_is_byte_identical_to_the_fail_before_switch`).

The guard is deliberately toothless on ordinary calls: a 512^2 grid at degree
14 projects to ~0.1 GB.  It bites only where the fit is a multi-GB object, so
**dev-box behaviour is unchanged** and the C15 accuracy win is kept everywhere
it can be paid for.

### 3.3 What was NOT done

* No `xfail`, no `skip`, no shrunk fixture.  The killed test runs the same
  2048^2 geometry it always ran and asserts the same things.
* The `_ram_guard()` / `_MIN_FREE_GIB = 3.0` skip idiom already in the niche
  test files was NOT retuned.  It was never the mechanism here -- it passed on
  the runner (the dots are not `s`), because the runner *did* have >= 3 GiB
  free when the test started.  A guard that measures free RAM at entry cannot
  see a call that will demand 33 GB.
* Normal-equations blocking of the SOLVE was considered and rejected: this
  module has an explicit conditioning doctrine (niche C13,
  `LSTSQ_CONDITIONING_STEPDOWN`) and changing how the Gram is accumulated is a
  numerics change, not a memory one.

---

## 4. MEASURED

Peak process RSS, `psutil`'s AVAILABLE reading pinned to 6.5 GB (the emulation
the runner is), `test_niche_c15_inverse_map.py` + `test_niche_c1_consolidation.py`
run in one process as the shard runs them:

| test | before | after |
|---|---|---|
| `c1::..._measured_exit_na_and_the_aliased_power` | **34.73 GB** (64.5 s) | **6.15 GB** (19.4 s) |
| `c1::..._chain_reports_the_measured_na...` | 3.88 GB | 3.14 GB |
| `c1::..._guard_stays_silent_where_the_grid...` | 3.48 GB | 2.75 GB |
| `c1::..._breaking_the_tilted_path...[no chief-ray shift]` | 2.96 GB | 2.96 GB |
| every remaining test in both files | < 2.8 GB | < 2.8 GB |
| all 78 tests | 78 passed, 166 s | 78 passed, 115 s |
| `c15` alone -- the map-default file | max 1.23 GB | max 1.00 GB |

The `after` column is whole-process RSS, so it carries the ~2.0-2.6 GB the
pytest session already held when the test started; the test's own transient is
the `tracemalloc` figure below.  **`c15`'s own tests did not inflate** -- the
5-vs-7 dots between the two runner legs is shard membership, and its heaviest
test peaks at 1.0 GB.

Isolated transient for the killed call (`tracemalloc`, so allocator-exact,
excluding the ~2 GB session baseline):

| budget | before | after |
|---|---|---|
| 6.5 GB pinned (runner) | 32.83 GB | **4.26 GB** -- GRAM refuses, incumbent path |
| dev box (~80 GB free) | 32.83 GB | **20.17 GB** -- map still built, blocking only |

The dev-box row is the point of the two-layer fix: the map is still built where
it can be afforded, at 1.6x less peak than before.

---

## 5. `Validate` -- the stale expectation

Reproduced locally at `667e0d1`: `validation/run_all.py` is 36/37 PASS with
`test_subsample.py` failing three checks, all reading `measured 0.0 nm`.

Measured cause, both arms on the same beam (512^2, dx = 19.5 um):

| `ray_subsample` | `TRACED_INVERSE_MAP = False` | shipped default (map on) |
|---|---|---|
| 4 | 21.2 nm | 0.006 nm |
| 8 | 86.5 nm | 0.006 nm |
| 16 | 348.5 nm | 0.006 nm |

Not a cache bug (`_imap_key` hashes `xs_in`'s bytes, and the three runs are
three cache MISSES producing three non-identical fields).  The map is a global
polynomial model of the exit characteristic; the launch-lattice spacing barely
moves it.  The `0.0` in the log was `%.1f` rounding of 0.006.

**Fix (envelope rule):** the scaling law is a claim about the upsample path, so
it is now pinned to that path (`TRACED_INVERSE_MAP = False`) with the SAME
three bands and the same measured numbers, and the collapse is added as its own
claim -- three new checks at a `< 1 nm` bar, which the upsample path misses by
21x / 87x / 349x at the same `sub`.  `test_subsample.py` is 14/14.

Keeping the old bands as-is would have required asserting that the library must
stay wrong by 10-600 nm.

---

## 6. WHAT IS PINNED

`tests/unit/test_fix_runner_oom_2026_08_13.py`, 14 tests:

* bit-identity of `_td_design` (degrees 3/6/14) and `_td_design_grad`
  (degrees 4/9) against the retired expressions, with the chunk budget forced
  small enough to guarantee >= 6 blocks and the block cover asserted contiguous;
* bit-identity of `_Cheb2DEvaluator`'s fitted coefficients, weighted and
  unweighted, against the retired construction plus the same solver;
* the transient bounds, each with its own **fail-before measured in the same
  test** (the retired arm must still over-allocate, or the pin has stopped
  being able to see the fix) -- plus a size-independence check: the excess over
  the output must not grow when `n` doubles;
* GRAM fires on a small box, does not fire on a large one, and its numbers are
  in the guard record;
* a GRAM refusal is byte-identical to `TRACED_INVERSE_MAP = False`;
* the guard's projection is not smaller than the build's measured coexisting
  high-water (design alive while the gradient pair is built);
* **the release blocker itself**: the 2048^2 / `ray_subsample=2` call on a
  pinned 6.5 GB box allocates < 5.0 GB and returns a finite 2048^2 field;
* the map still engages on a box with room, so the fix cannot be read as
  "turn C15 off in CI".

---

## 7. WHAT WAS RUN

| gate | result |
|---|---|
| `ruff check lumenairy/ tests/unit/` | clean |
| `mypy` (pyproject whitelist, the blocking CI job) | 0 issues, 10 files |
| `test_fix_runner_oom_2026_08_13.py` (the new pins) | 14 passed |
| `c15` + `c1` under the 6.5 GB emulation | 78 passed |
| the 47 lens / traced / imap / newton / carrier / fit / `niche_c*` / `niche_d*` files | 1056 passed, 55 skipped |
| `validation/integration/test_subsample.py` | 14/14 (was 8/11) |
| `validation/run_all.py` | 37/37 files |
| collected, fast gate | 11,718 selected of 11,954 (worktree) vs 11,704 of 11,940 (`origin/main` mount) -- the 14 are this file's pins |

Two failures appeared in the 47-file run and are NOT this change:
`test_niche_d6_exact_tilted_leg.py::test_untilted_tiltedcarrier_takes_the_scalar_path_byte_identically`
and `::test_a_harmless_readout_window_clamp_stays_silent`.  That run used
`pytest -n 10` (which the fast gate deliberately does not: see unit-tests.yml's
"each shard runs SERIALLY" note) while an orphaned 66 GB python from an earlier
profiling run was squeezing the box -- and both tests are exactly the class
that reads: one is a two-arm byte-identity comparison, the other asserts a
RAM-driven readout clamp stays silent.  Re-run serially the file is **38/38**,
and **38/38 again under the pinned 6.5 GB emulation**, i.e. at both extremes of
the budget the guard could see.

## 8. FOLLOW-UPS, NOT DONE HERE

* **`.test_durations` is now wrong for this test** (54.09 s recorded, ~19 s
  actual), which makes `pytest-split` over-weight shard 1.  Harmless, but the
  file is due a regeneration anyway -- see the standing note in
  `unit-tests.yml` about capturing it serially with BLAS pinned.
* **No CHANGELOG entry and no version bump.**  Whoever cuts the release owns
  those; the publish gate's CHANGELOG walkers read `file:line` citations, which
  drift if written before the final diff lands.
* **The next headroom item is the forward fit, not the map.**  With GRAM
  refusing on a runner, what remains of the killed call's ~2.3 GB transient is
  `_Cheb2DEvaluator` at 2401^2 samples: A (1.29 GB) plus the Chebyshev planes.
  Bringing that down means touching the solve, which is a numerics change (see
  §3.3) and wants its own round.
* **GRAM reads a volatile quantity** (psutil AVAILABLE, via
  `get_ram_budget()`), so a call sitting exactly at the threshold can build on
  one run and refuse on the next.  That is the same contract every other
  memory-aware path in this library already has (`parallel_amp`, the Newton
  pool, `n_fine_cap`), and the refusal is REPORTED (`INVERSE_MAP_GUARD='warn'`
  by default, with the projected and available GB in the message), but it is
  worth knowing before a byte-identity pin is written against a near-threshold
  build.

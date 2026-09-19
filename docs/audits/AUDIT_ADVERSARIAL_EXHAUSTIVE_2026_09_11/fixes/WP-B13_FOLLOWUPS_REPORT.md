# WP-B13 FOLLOW-UPS -- closing D1, D2, D3, D4 and D7, and measuring D5 and D6

Branch `fix/wp-b13-followups`, worktree `C:\tmp\lum_pool2`, base `b631ce79` (`wave5/audit-leftovers`,
which already carries WP-B13 and its verification).  The input is
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B13.md` section 7 ("Defects")
and its "Requested changes outside my ownership".

Two builds throughout: **Windows python 3.14.6 / numpy 2.4.4 / scipy 1.17.1** and
**WSL python 3.12.3 / numpy 2.4.6**, both resolving `lumenairy` to this worktree (every probe prints
`lumenairy.__file__` as its first line, with the tree pinned through `PYTHONPATH`).  Every command
carried `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` on the command line.  Probes and
their per-build JSON are in `validation/probe_wp_b13_followups/`.

**Scope.**  D1, D2, D3, D4 and D7 are CLOSED here.  **D5 and D6 are maintainer decisions and are
deliberately still open**: nothing in this branch changes a default, adds a timeout, or edits
`lumenairy/propagators/carrier.py`.  Sections 6 and 7 are the two recommendations, with the numbers
a maintainer would otherwise have to go and measure.

---

## 0. Verdict table

| id | severity | what it was | status | the proof |
|---|---|---|---|---|
| D1 | P2 (test quality) | `_thread_dump()` wrote to an `io.StringIO`, so every wedge detection raised `io.UnsupportedOperation: fileno` and lost its message and its thread dump | **CLOSED** | wedge injected under the harness; the report now names the wedged frame.  Fail-before: the shipped helper re-injected -> both new pins fail with `io.UnsupportedOperation: fileno` |
| D2 | P3 (resource) | `_shutdown_pool_bounded`'s expiry appended to `_ABANDONED_POOLS` and nothing removed it | **CLOSED** | driven expiries: the census returns to its prior length once the teardown finishes, three times over, and survives the adverse lock order.  Fail-before: the shipped helper re-injected -> 3 failed |
| D3 | P3 (derivation) | the idle-worker footprint was measured on `ex.map(abs, ...)` workers, not on the state the ceiling rule leaves behind | **CLOSED** | re-measured on both builds with both states present in ONE pool: served **98.7-100.8 MB** (Win) / **73.2-76.9 MB** (WSL), never-served **52.2 / 39.2 MB**.  Docstring, report sec. 5.2 and the CHANGELOG restated with the state named |
| D4 | P3 (comment drift) | `_POOL_INFLIGHT`'s comment said "chunks"; the counter is moved once per DISPATCH | **CLOSED** (comment, not counter -- consumers read) | every consumer is a zero-vs-non-zero read; pinned by `test_the_in_flight_counter_is_one_claim_per_dispatch_not_per_chunk` |
| D7 | P3 (pin blind spot) | the AST pin saw only `x.shutdown(...)`, so a `with ProcessPoolExecutor(...)` teardown passed it | **CLOSED** | the extended detector reports `carrier._multi_parallel_results:11553:with ProcessPoolExecutor`; the shipped one reports `[]`.  Identical on both builds |
| D5 | P2 (pre-existing) | `as_completed` with no timeout | **OPEN -- maintainer decision**, measured in sec. 6 | -- |
| D6 | P2 (pre-existing) | a wedged manager thread still hangs interpreter EXIT | **OPEN -- maintainer decision**, measured in sec. 7 | -- |

No field moved.  Every pooled field produced by every probe in this package equals its
`n_workers=1` reference under `np.array_equal`, `max|delta| = 0.0`.

---

## 1. D1 -- the wedge report lost its message and its thread dump

**What it was.**  `tests/unit/test_fix_newton_pool_broken_fallback.py::_thread_dump` passed an
`io.StringIO` to `faulthandler.dump_traceback`, which writes through the file's `fileno()`.  The
call raised `io.UnsupportedOperation: fileno`.  The helper is reached from exactly one place --
`_with_deadline`'s failure path -- so *every* wedge detection in the file reported that unrelated
exception instead of its own message, with no thread dump.  Since the library call is still parked
on a daemon thread when the deadline fires, that dump is the only artifact a maintainer gets.

**What changed.**  The helper writes to a `tempfile.TemporaryFile`, which has a real descriptor
(the exact shape the verification asked for), with the reason recorded in its docstring.

**How it is proved.**  Two behavioural pins, not a source pin -- a source pin would have passed the
day the defect shipped:

* `test_the_wedge_report_carries_the_stack_of_the_wedged_frame` **engineers** a wedge (a daemon
  thread parks in a uniquely named module-level frame, `_b13_wedged_frame_for_the_dump`), drives it
  through `_with_deadline` with a 1 s deadline, and requires the resulting failure to carry both
  `_with_deadline`'s own message and the name of the frame that is stuck.  The premise -- what an
  `io.StringIO` does on the running build -- is **measured inside the test** and reported in the
  assertion message rather than asserted, so a future CPython that grew a StringIO path would make
  the fix redundant rather than the pin red.
* `test_the_dump_helper_writes_through_a_real_descriptor` calls the helper directly and requires
  its own frame to appear.

**Fail-before**, `validation/probe_wp_b13_followups/fu_prefix_plugin.py` (`FU_INJECT=dump` restores
the shipped helper in memory at collection time):

| arm | result |
|---|---|
| shipped helper re-injected, the two new pins | **2 failed**, both ending `io.UnsupportedOperation: fileno` |
| this branch, the two new pins | **2 passed** in 1.26 s |

**And at the real call site.**  With the pre-fix teardown helpers injected so the file's three
genuine wedge tests actually fire (`FU_INJECT=joins`, which reuses
`validation/probe_verify_b13/vp7_wedge_plugin.py::_install_joins` so the two demonstrations cannot
drift apart), Windows 3.14.6, 138.51 s: each of the three failures now reads

```
Failed: close_worker_pool on a broken pool did not return within 30 s -- this is the wedge
P1 4.1b is about, not slowness (...).  Thread dump:
Thread 0x00010d80 [deadline-close_worker_pool on a broken pool] (most recent call first):
  File ".../threading.py", line 369 in wait
  File ".../test_fix_newton_pool_broken_fallback.py", line 217 in shutdown
  File ".../vp7_wedge_plugin.py", line 30 in _abandon_pool
  File ".../lumenairy/elements/_lens_traced.py", line 1638 in close_worker_pool
  ...
```

-- its own message, then the wedged frames.  That is what the file promised and did not deliver.

---

## 2. D2 -- an expired bounded teardown pinned its executor forever

**What it was.**  `_shutdown_pool_bounded`'s expiry path appended the executor to
`_ABANDONED_POOLS` and nothing ever took it back out, unlike `_abandon_pool`, whose daemon reaper
removes in a `finally`.  The list grew by one per expiry and held each dead executor, its
`_processes` and its queues, for the life of the process.

**What changed.**  `_ABANDONED_POOLS` is now documented and implemented as a **census of pending
teardowns**, not a ledger: both mechanisms take their entry back out when their `shutdown` returns,
so a quiescent process reads an empty list however many pools it has retired.  The monotone count
of teardowns that overran is `_POOL_SHUTDOWN_TIMEOUTS`, and the module comment now says to read
that, not `len()`.

**Not the one-liner, and the difference is measured.**  The verification asked for a bare
`finally: remove ...; done.set()` in the helper thread.  The helper can reach the census in the
same instant the caller's bounded wait expires, and if the helper's REMOVE runs before the caller's
APPEND, the remove finds nothing and the append re-pins exactly the entry the repair was meant to
drop.  So the shipped helper publishes `done` **before** it takes `_ABANDONED_POOLS_LOCK`, and the
caller re-reads `done` under that same lock before appending -- and, finding it set, reports the
teardown as COMPLETED rather than as an expiry, which it is.  Cost: one extra `is_set` on the
expiry path.

**How it is proved.**  Four pins, all two-sided, all engineered rather than sampled:

| pin | what it decides |
|---|---|
| `test_an_expired_bounded_teardown_releases_its_executor_when_it_finishes` | the entry must be THERE while the teardown is outstanding (otherwise the census is useless) and GONE once it completes; `_POOL_SHUTDOWN_TIMEOUTS` unchanged across the release |
| `test_repeated_expiries_do_not_grow_the_census` | three driven expiries: the count rises by three, the census ends where it started |
| `test_a_teardown_landing_exactly_on_the_expiry_leaves_nothing_behind` | the boundary, engineered by holding the census lock across the expiry, eight times |
| `test_the_census_survives_the_helper_first_lock_order` | the adverse arrival order, produced **by construction** -- a census lock that admits the `lumenairy-newton-pool-close` thread first -- because `threading.Lock` makes no fairness promise and an ordering must not be waited for |

**Fail-before**, Windows 3.14.6:

| arm | result |
|---|---|
| `FU_INJECT=d2` (the shipped helper: append, never remove) | **3 failed** in 47.11 s |
| `FU_INJECT=d2one` (the one-line repair the verification asked for) | **3 passed**, and `test_the_census_survives_the_helper_first_lock_order` **FAILS** in 16.27 s |
| this branch | **4 passed** in 3.61 s |

So the one-liner is not merely different from what shipped: it is measurably insufficient in the
one ordering that cannot be produced by sleeping.

**And on real pools.**  Every rung of `fu1_worker_footprint.py` (real 12-worker spawn pools, both
builds, three field sizes) ends with `abandoned_pools_len: 0` and `shutdown_timeouts: 0`.

---

## 3. D3 -- the idle-worker footprint, re-measured in the state that actually occurs

**What it was.**  `_get_persistent_worker_pool`'s docstring justified never shrinking the pool with
"33.0 MB mean / 48.8 MB peak per idle worker ... 2 % of one working worker".  That figure was
measured on workers warmed with `ex.map(abs, ...)`, which is not a state this rule produces.

**How it was re-measured** (`fu1_worker_footprint.py`).  A **12-wide pool serving a 4-worker
dispatch**, so both reachable states are present in the SAME pool: four workers serve a Newton
chunk, eight serve none.  Resident sets are read PARENT-side with `psutil` against the executor's
own `_processes`, so the reading does not perturb the worker; each worker is then labelled by its
own `_WORKER_PAYLOADS`, with every worker forced to take exactly one labelling task by a Manager
`Barrier` of the pool's width.  `label_coverage_complete: true` on every rung, so "which workers
exist" and "which were labelled" are decisions rather than samples.

| build | N | served (mean / max) | never-served (mean / max) | served pids |
|---|---|---|---|---|
| Windows 3.14.6 | 256 | **98.67 / 99.01 MB** | 52.19 / 52.38 MB | 4 of 12 |
| | 512 | **98.76 / 100.12 MB** | 52.22 / 52.33 MB | 4 of 12 |
| | 1024 | **100.76 / 101.56 MB** | 52.31 / 52.47 MB | 4 of 12 |
| WSL 3.12.3 | 256 | **73.21 / 73.21 MB** | 39.26 / 39.26 MB | 4 of 12 |
| | 512 | **74.08 / 75.02 MB** | 39.24 / 39.25 MB | 4 of 12 |
| | 1024 | **76.93 / 76.94 MB** | 39.24 / 39.25 MB | 4 of 12 |

Every rung: `identical_to_serial: true`, `max_delta 0.0`.

This confirms the verification's reading and sharpens it.  The worst kept worker is **~102 MB**, not
33.0 MB; a 16-wide kept pool of served workers holds about **1.6 GB**, not 0.5 GB; and against the
~1.7 GB per ACTIVE worker the clamp itself models the fraction is **6 %**, not 2 %.  Note also that
the never-served figure on this branch (52.2 MB Windows) is above the verification's 28-39 MB: a
spawn worker of *this* pool necessarily imports `lumenairy.elements._lens_traced`, because that is
where its `initializer=_newton_pool_init` is resolved from, so 52.2 MB is the true floor for a
worker of this pool on this build and 39.2 MB is the WSL floor.

**What changed.**  The docstring now names both states and carries the ladder above with its date
and its probe.  `WP-B13_NEWTON_POOL_REPORT.md` section 5.2 and the `[Unreleased]` CHANGELOG entry
carry the correction **with the old figure quoted**, so the trail is explicit rather than silently
overwritten.  The conclusion -- never shrink implicitly -- is unchanged, and it was always the
trade and not the number that it rested on.

---

## 4. D4 -- `_POOL_INFLIGHT` counts dispatches

**The decision, taken by reading the consumers**: fix the COMMENT, not the counter.

| consumer | what it reads |
|---|---|
| `_get_persistent_worker_pool`, `elif _POOL_INFLIGHT > 0:` | zero vs non-zero.  The only library reader |
| `test_the_in_flight_counter_cannot_go_negative_or_leak` | `== 0` |
| `validation/probe_verify_b13/vp2_broken_drivers.py`, `vp5_concurrency.py` (6 sites) | reported, asserted `0` |

Nothing anywhere reads the magnitude, so a per-chunk counter would buy nothing and would add two
lock acquisitions per chunk.  The comment was the defect.

`test_the_in_flight_counter_is_one_claim_per_dispatch_not_per_chunk` pins the three facts that
decision rests on: the dispatcher claims exactly once; it claims BEFORE its first `submit`, so the
claim covers the dispatch rather than tracking chunks; and every `ast.Compare` against
`_POOL_INFLIGHT` in the module is against zero.  The moment one is not, the magnitude has acquired
a meaning and the counter -- not the comment -- is what has to change, which is what the failure
message says.

---

## 5. D7 -- the join pin was blind to `with ProcessPoolExecutor(...)`

**What it was.**  `test_only_the_bounded_helper_ever_joins_an_executor` walked for `ast.Call` nodes
whose `func.attr == 'shutdown'`.  A joining teardown written as `with ProcessPoolExecutor(...) as
ex:` contains no such call -- `Executor.__exit__` IS `shutdown(wait=True)` -- so the pin stayed
green on exactly the shape that carries the same exposure in `carrier._multi_parallel_results`.

**What changed.**  The sweep is now a named helper, `_unbounded_executor_joins`, which sees two
shapes: a `.shutdown(...)` call whose `wait` is absent or not the constant `False`, reading `wait`
from the **positional** slot as well as the keywords (so `ex.shutdown(False)` is correctly not an
offender, and `Executor.shutdown(ex, True)`, whose first positional is `self`, correctly is); and a
`with <X>Executor(...)` item.  Scope is PROCESS pools -- the exposure is CPython's
`_terminate_broken`, which `ThreadPoolExecutor` does not have and whose `__exit__` joins work the
enclosing `result()` calls have already collected -- and thread-pool blocks are returned as
`informational` and printed in the failure message rather than silently exempted.

**Three tests, because a detector that is never shown finding anything can go blind without going
red**: the invariant on `_lens_traced` (no offenders); `test_the_join_detector_sees_the_with_form`,
a synthetic positive control the test owns; and
`test_the_sibling_process_pool_still_carries_an_unbounded_join`, the **premise gate**.

**The premise gate, and why it is an assertion that the defect is STILL THERE.**  D5 is a maintainer
decision scoped out of this package, so `carrier.py` is not edited here.  The extended detector must
therefore report `_multi_parallel_results`'s `with ProcessPoolExecutor(...)` as an unbounded join
TODAY.  That single assertion does two jobs: it is the fail-before for the `with` extension (without
it the detector reports nothing on that module, which is precisely how the shipped pin stayed
green), and it is the record -- the pin must not be made green by quietly editing the sibling
module.  Its failure message says what to do when the sibling pool is eventually repaired: retire
the test, and take the open-D5 note out of this report.

**Measured both builds** (`fu2_d7_pin_detectors.py`, which runs the shipped detector and the
extended one side by side; `fu2_win.json`, `fu2_wsl.json`, byte-identical verdicts):

| source | shipped detector | extended detector |
|---|---|---|
| synthetic (all six shapes) | misses `teardown_with`; **false-positives** on `ex.shutdown(False)` | `teardown_call`, `teardown_unbound`, `teardown_with`; no false positive; `threads` informational |
| `lumenairy.elements._lens_traced` | `[]` | `[]` (+1 informational `ThreadPoolExecutor` at `apply_real_lens_traced:10731`) |
| `lumenairy.propagators.carrier` | **`[]`** | **`_multi_parallel_results:11553:with ProcessPoolExecutor`** |

`{"shipped_pin_is_blind_to_the_sibling_pool": true, "extended_pin_detects_the_sibling_pool": true,
"lens_traced_clean_under_extended_pin": true}` on 3.14.6 and on 3.12.3.

---

## 6. RECOMMENDATION 1 (D5) -- what a timeout on `as_completed` would cost

**The exposure, restated.**  `as_completed(future_to_idx)` in `_invert_newton_parallel` has no
timeout, so a worker that never comes up hangs the call forever, on both trees.  It is reproducible
on demand: `validation/probe_verify_b13/vp2_broken_drivers.py --mode slowboot`.

**The first thing a maintainer needs to know is a CPython fact, not a measurement.**
`concurrent.futures.as_completed(fs, timeout=T)` does **not** bound a chunk.  `T` is an absolute
deadline taken when the generator is created, so a single `timeout=` bounds the WHOLE iteration --
the total dispatch.  That is the wrong quantity, because the total scales with the field while the
pathology does not.

**Measured healthy side** (`fu3_chunk_timing.py`, 8 workers, the traced-lens fixture ladder, the
real dispatcher instrumented through `_get_persistent_worker_pool`'s substitution point; every rung
`pool_engaged: true`, 8 chunks, `identical: true`, `max|delta| 0.0`):

| build | N | cold first result | cold slowest chunk | cold total | warm first | warm slowest | warm total |
|---|---|---|---|---|---|---|---|
| Windows 3.14.6 | 256 | 1.005 | 1.019 | 1.069 | 0.012 | 0.029 | 0.030 |
| | 512 | 1.065 | 1.139 | 1.194 | 0.065 | 0.154 | 0.154 |
| | 1024 | **1.411** | **1.935** | **2.006** | 0.484 | 1.502 | 1.502 |
| WSL 3.12.3 | 256 | 0.748 | 0.773 | 0.773 | 0.012 | 0.025 | 0.025 |
| | 512 | 0.774 | 0.917 | 0.936 | 0.055 | 0.137 | 0.138 |
| | 1024 | **1.339** | **2.400** | **2.423** | 0.432 | 1.401 | 1.402 |

(seconds; "cold" = `close_worker_pool()` first, so the first chunk pays the spawn bootstrap)

**Spawn bootstrap on its own** -- pool construction to every worker having imported
`scipy.interpolate` and `lumenairy.elements._lens_traced`, all 8 forced to participate by a Manager
Barrier:

| build | construct | all 8 ready | ready spread | distinct pids |
|---|---|---|---|---|
| Windows 3.14.6 | 0.013 s | **1.179 s** | 0.054 s | 8 |
| WSL 3.12.3 | 0.029 s | **0.841 s** | 0.024 s | 8 |

The spread confirms the verification's refutation of the report's "one at a time, 3-6 minutes each":
the workers bootstrap concurrently, in under 1.2 s, on both builds.

**Cost of a sentinel round-trip** (the cheapest possible probe of "is a worker alive at all"):

| build | cold (pool just built) | warm, worst of 5 |
|---|---|---|
| Windows 3.14.6 | 0.3708 s | **0.0007 s** |
| WSL 3.12.3 | 0.3194 s | **0.0004 s** |

### 6.1 The two candidate bars, and which one the data supports

**Candidate A -- `as_completed(future_to_idx, timeout=T)` with one fixed `T`.**  `T` must dominate
the largest legitimate TOTAL dispatch.  My ladder stops at N = 1024 (worst cold total 2.006 s
Windows / 2.423 s WSL), but this library's own comments size the Newton path at N = 32768 ("free
~17 GB at N=32768 before Newton starts"), i.e. **1024x** the pixels of my top rung; the warm total at
N = 1024 is already 1.5 s, and scaling with pixel count puts an N = 32768 dispatch in the
**tens of minutes to hours**.  The WP's own report separately records 45-56 s for a first cold
`apply_real_lens_traced` on a loaded box.

> **The data does not support a fixed `T`, and that is the finding.**  Any `T` large enough to be
> safe at N = 32768 (hours) is useless at N = 256, where the pathology would be caught in seconds;
> any `T` derived from my ladder (`10x` the worst cold chunk plus the bootstrap = **20.5 s**
> Windows / **24.8 s** WSL) would abort legitimate large dispatches.  Picking one would be picking a
> number, which is exactly what `docs/TESTING_STANDARDS.md` forbids.

**Candidate B -- bound the BOOTSTRAP, not the dispatch.  RECOMMENDED.**  Submit one trivial
sentinel ahead of the chunks and give *that* a timeout; iterate `as_completed` unbounded as today.
The quantity bounded is then "did any worker answer at all", which is N-independent, and it is the
quantity that actually separates a slow box from a pool that never came up -- which is the
`slowboot` pathology, verbatim.

Derivation of the bar, from measurements rather than taste:

* floor, this box, quiet: sentinel cold round-trip **0.371 s** (Windows) / **0.319 s** (WSL);
  full 8-worker bootstrap **1.179 s** / **0.841 s**;
* envelope: the loudest healthy reading anyone has recorded for a first cold traced call on a
  LOADED box is the WP's own **45-56 s**;
* bar = **10x that envelope, i.e. `_POOL_BOOTSTRAP_TIMEOUT = 600.0` s**.  That is **1618x** the
  measured cold sentinel on Windows (**1879x** on WSL), **509x** the full 8-worker bootstrap, and
  **10.7x** the worst loaded-box reading in the record -- with the state on the other side being
  UNBOUNDED, so the gap above is infinite.  Same shape as `_POOL_SHUTDOWN_TIMEOUT`, which is the
  precedent in this module.
* cost on the healthy path: **one sentinel round-trip per dispatch**, measured at **0.0007 s**
  (Windows) / **0.0004 s** (WSL) on a warm pool, i.e. **0.05 %** of the fastest warm dispatch
  measured here (1.5 s at N = 1024) and below the noise of every rung in the table.  On a cold pool
  it is not an extra cost at all: the sentinel simply pays the spawn the first chunk would have
  paid.
* the fallback is already right: `TimeoutError` is an `OSError` subclass, so it lands in the
  dispatcher's existing infrastructure clause and takes the bit-identical serial rung -- the cost
  of a false positive is wall time only, which is what makes a generous bar the correct shape.

**Candidate C -- a per-`__next__` bar** (re-create `as_completed` with a fresh deadline per result)
would bound stalls mid-dispatch as well, but its bar has to clear the slowest legitimate CHUNK,
which scales with N exactly as candidate A's does.  Not recommended for the same reason.

`carrier.py::_multi_parallel_results` carries the identical unbounded exposure through its `with
ProcessPoolExecutor(...)` block (section 5) and should be taken in the same work package; its pool
has a different failure policy (it degrades to serial on an unpicklable registry) so the sentinel
would have to be priced against its own initializer, which ships `uniq`, `groups_k`, `common` and a
glass snapshot -- a genuinely expensive bootstrap that I did not measure.

---

## 7. RECOMMENDATION 2 (D6) -- the interpreter-exit hang

**The exposure, restated.**  WP-B13 moved the wedge from the middle of a computation, where the
caller's result is lost, to after the call has returned its bit-identical serial answer.  The
PROCESS still cannot exit.  Interpreter shutdown joins the wedged `_ExecutorManagerThread` twice
over: `concurrent.futures.process._python_exit`, registered through `threading._register_atexit`,
joins every thread in `_threads_wakeups`; and `threading._shutdown` then joins every non-daemon
thread's `_tstate_lock`.

**Measured** (`fu4_exit_hang.py`, WSL 3.12.3, the verification's NATURAL `sigign` wedge -- the chunk
installs `SIG_IGN` for SIGTERM and holds, then the probe SIGKILLs exactly one worker, so the pool
breaks and CPython's `p.terminate()` / `p.join()` on the survivors is genuinely unbounded; no stub
executor anywhere).  The harness separates RETURNED (the library call came back) from EXITED (the
process terminated):

| arm | library call | exit line at | process gone at | hang after the exit line | exited on its own |
|---|---|---|---|---|---|
| `control` (nothing broken) | 0.949 s, `identical: true` | t = 2.569 s | t = 3.043 s | **0.474 s** | yes, rc 0 |
| `sigign`, prototype OFF | 1.282 s, `identical: true`, `max|delta| 0.0` | t = 2.825 s | t = 60.012 s | **57.187 s (SIGKILLed at the deadline)** | **no** |
| `sigign`, prototype ON (all three steps confirmed applied) | 1.282 s, `identical: true` | t = 3.032 s | t = 60.012 s | **56.980 s (SIGKILLed)** | **no** |
| `sigign`, surviving workers SIGKILLed after the answer is in | 1.302 s, `identical: true` | t = 3.063 s | t = 3.443 s | **0.380 s** | **yes, rc 0** |
| `sigign`, both | 1.269 s, `identical: true` | t = 2.995 s | t = 3.439 s | **0.444 s** | yes, rc 0 |

So the cost of the hang is: the right answer, on time, and then a process that has to be killed --
**57.2 s against a 0.47 s control on the same script**, and 57.2 s is only my deadline; nothing
suggests it would ever end.  Every arm returned `identical: true` with `max|delta| = 0.0`, so what
is at stake is the process, never the number.

**The prototype, and why it is not in `lumenairy/`.**  The candidate the verification pointed at --
`atexit` + `_abandon_pool` daemonising the reaper -- needs three steps, all on private CPython
attributes, because step 2 alone is not enough:

```python
import concurrent.futures.process as cfp
mgr = ex._executor_manager_thread
cfp._threads_wakeups.pop(mgr, None)        # 1. _python_exit must not join it
mgr._daemonic = True                       # 2. the public setter refuses a started thread
threading._shutdown_locks.discard(mgr._tstate_lock)   # 3. added at START, while it was
                                           #    still non-daemon; step 2 does NOT remove it
```

That is three private attributes across two stdlib modules, and step 3 is the one a reader would
miss: `Thread._set_tstate_lock` adds the lock to `threading._shutdown_locks` at start time based on
the daemon flag as it was THEN, so daemonising afterwards leaves the join in place.

### 7.1 The prototype does NOT close it -- and the stacks say why

This is the measurement that matters, and it contradicts the shape the verification suggested.
`faulthandler` was armed in the child AFTER its exit line (`exit=False`, so the dump does not
destroy the measurement), and the two wedged arms park in DIFFERENT places:

**prototype OFF** -- the main thread is exactly where the verification said:

```
Thread <main>:
  /usr/lib/python3.12/threading.py:1167  _wait_for_tstate_lock
  /usr/lib/python3.12/threading.py:1147  join
  /usr/lib/python3.12/concurrent/futures/process.py:102  _python_exit
  /usr/lib/python3.12/threading.py:1592  _shutdown
```

**prototype ON** -- that join is GONE.  The interpreter gets past `_python_exit` and hits the
NEXT unbounded join, one layer down:

```
Thread <main>:
  /usr/lib/python3.12/multiprocessing/popen_fork.py:27   poll
  /usr/lib/python3.12/multiprocessing/popen_fork.py:43   wait
  /usr/lib/python3.12/multiprocessing/process.py:149     join
  /usr/lib/python3.12/multiprocessing/util.py:360        _exit_function
```

`multiprocessing.util._exit_function` is an ordinary `atexit` handler and it ends with
`for p in active_children(): p.join()`.  The surviving workers are NOT daemonic and they are
ignoring SIGTERM, so that join is unbounded too.  In both arms the manager thread is still parked in
`_terminate_broken -> _join_executor_internals -> Process.join -> popen_fork.poll`, and the
`lumenairy-newton-pool-reaper` daemon thread is still inside `shutdown` at `_lens_traced.py:1299` --
costing nothing, exactly as its comment claims.

**So the three-step daemonising prototype does what it says and is not sufficient.**  It removes one
of two unbounded joins at exit; the second is the workers themselves.  The arm that DOES close the
hang is the one that removes the workers -- `sigign` + SIGKILL, with no prototype at all, exits in
0.380 s -- because killing them also releases the manager thread's `p.join()` and therefore
`_python_exit`'s join as well.

### 7.2 Recommendation

**The repair is to make the surviving workers DIE, not to daemonise a thread.**  Concretely, in
`_abandon_pool`'s reaper, after the non-joining `shutdown(wait=False, cancel_futures=True)` has been
issued: wait a grace period, then `Process.kill()` (SIGKILL / TerminateProcess) any of the
executor's `_processes` still alive, then return.  The daemon reaper is already the right place --
it is not joined at exit and it already owns the retired pool.

* **The grace bar is already derived in this module.**  A healthy `close_worker_pool` was measured
  at worst 0.379 s (Windows, 16 workers) and 2.240 s (WSL, 16 workers, loaded box) by the
  verification.  `_POOL_SHUTDOWN_TIMEOUT = 120 s` is the existing bar over exactly that quantity,
  at 53-317x the healthy worst, and the escalation should reuse it rather than introduce a second
  number.
* **What it costs when it fires**: workers that were going to exit anyway are killed a little
  sooner.  The pool is already retired and broken at that point -- it can serve nobody -- and the
  computation has already returned its bit-identical serial answer, so nothing downstream reads
  anything those processes hold.
* **What it costs when it should not fire**: nothing, because it only fires 120 s after a pool has
  been abandoned.
* **The risk that makes it a maintainer decision, not a defect fix**: SIGKILL is not a request.  A
  worker mid-write to a file or a shared resource loses it.  Today's code never kills a worker, and
  changing that is a policy change about what this library is allowed to do to processes it
  started.
* **The daemonising steps are then unnecessary** -- killing the workers releases both joins -- which
  is worth knowing, because those three private CPython attributes are version-sensitive and step 3
  (`threading._shutdown_locks.discard(mgr._tstate_lock)`) is the kind of thing that silently becomes
  a no-op on an upgrade.

Until that is taken deliberately, **the release note should say out loud what ships**: on a wedged
pool the computation is saved but the process is not.  A batch script that hits this gets the right
numbers and then hangs at exit.

---

## 8. Probes and evidence

| file | what it measures |
|---|---|
| `fu_prefix_plugin.py` | the pytest plugin that puts each pre-fix defect back in memory: `dump` (D1), `d2` / `d2one` (D2 and the one-line alternative), `joins` (reused verbatim from the verification's `vp7`) |
| `fu1_worker_footprint.py` | the D3 ladder: served / never-served resident sets in one pool, both builds (`fu1_win.json`, `fu1_wsl.json`) |
| `fu2_d7_pin_detectors.py` | the D7 fail-before: the shipped detector and the extended one, side by side, over `_lens_traced`, `carrier` and synthetic source (`fu2_win.json`, `fu2_wsl.json`) |
| `fu3_chunk_timing.py` | the D5 healthy side: per-chunk / first-result / total, cold and warm, plus the spawn bootstrap and the sentinel round-trip (`fu3_win.json`, `fu3_wsl.json`) |
| `fu4_exit_hang.py` | the D6 cost and the daemonising prototype behind `--daemonise on`, default OFF (`fu4_wsl.json`) |
| `logs/` | the raw stdout of every run, including the D1 fail-before pytest arms |

---

## 9. Runs

All commands carried `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` on the command line
and ran from the worktree root.  The box was shared throughout with two other agents' lanes
(a GBD pytest lane and a `probe_gbd_projection` sweep were resident for most of it), which is why
the wall times below are not comparable with the verification's; the COUNTS are the claim.

### 9.1 The thirteen traced-lens / pool files, four arms

`pytest tests/unit/test_fix_newton_pool_broken_fallback.py tests/unit/test_fix_newton_pool_memory.py
tests/unit/test_niche_newton_pool_both_fits.py tests/unit/test_carrier_field.py
tests/unit/test_carrier_referenced.py tests/unit/test_audit2609_a3_traced_lens.py
tests/unit/test_audit2609_a3_verify_traced.py tests/unit/test_niche_audit_w9_traced_determinism.py
tests/unit/test_niche_d15_deterministic_traced_fit.py tests/unit/test_hammer_h3_traced_nyquist_guard.py
tests/unit/test_niche_audit_w3_elements.py tests/unit/test_niche_perf_round2_2026_08_10.py
tests/unit/test_niche_audit_e_prepared_and_enums.py -q [--capture=sys]`

The file count is unchanged; the TEST count is **373 collected against the verification's 364**,
because this package adds nine tests to `test_fix_newton_pool_broken_fallback.py` (two for D1, four
for D2, one for D4, and D7's single pin becomes three).  The expected reading is therefore
**372 passed, 1 skipped** -- the same single skip, `test_fix_newton_pool_memory.py` on this BLAS.

| build | capture | result | wall |
|---|---|---|---|
| Windows 3.14.6 | default (`fd`) | **372 passed, 1 skipped** | 442.40 s |
| Windows 3.14.6 | `--capture=sys` | **372 passed, 1 skipped** | 537.57 s |
| WSL 3.12.3 | default (`fd`) | **372 passed, 1 skipped** | 556.98 s |
| WSL 3.12.3 | `--capture=sys` | **372 passed, 1 skipped** | 469.41 s |

Four arms, four identical readings, and the same single skip on every one
(`test_fix_newton_pool_memory.py`, with the WP's own reason: this BLAS reduces identically at every
width tried, so the box cannot witness the defect that test is about).  The capture axis moves
nothing but wall time.

### 9.2 The rest of the gate

| command | result | wall |
|---|---|---|
| `pytest test_audit_except_budget.py test_public_api.py test_v4_16_2_dispatcher_pin_doc_consistency.py test_audit2609_a17_history_lint.py test_audit2609_a23_census_mechanism.py test_v4_14_2_dispatcher_pin_cache_locks.py test_eme_census_determinacy.py test_audit2609_a17_history_relocation.py -q` (Windows) | **892 passed, 6 skipped** | 246.46 s |
| `pytest tests/unit -q -k walker` (Windows) | **118 passed, 6 skipped, 16194 deselected** | 78.75 s |
| `python scripts/record_history_fingerprints.py --check` | green (re-recorded for `lumenairy.elements._lens_traced` in the D4 commit, reason recorded in the document header) | -- |
| `wsl ruff check lumenairy/ tests/ validation/probe_wp_b13_followups/` | **All checks passed** | -- |

| command | result | wall |
|---|---|---|
| `pytest tests/unit/test_verify_b13_newton_pool.py -q` (Windows) | 3 passed | 22.04 s |
| `pytest tests/unit/test_verify_b13_newton_pool.py -q` (WSL) | 3 passed | 16.28 s |
| the section 9.2 sweep, re-run on **WSL** | **1 failed, 891 passed, 6 skipped** | 243.71 s |

**The one WSL red is an environment fact, not a change of mine, and it is premise-gated.**
`test_public_api.py::test_installed_metadata_version_matches_source_version` compares
`importlib.metadata.version('lumenairy')` against `lumenairy.__version__`.  The WSL venv's editable
install is stale (`5.11.0`), while the source on this branch and on its base both read `5.47.0`, so
the test reports the venv.  Run on the **untouched base worktree** in the same venv it fails
identically (`assert '5.11.0' == '5.47.0'`), and on Windows it passes.  Nothing in this package
touches versioning.  I deliberately did NOT repair it by reinstalling: that venv is shared with
other agents' lanes that were resident throughout, and silently re-installing under them is not a
change this package is entitled to make.  The right fix is `pip install -e .` in `~/lumvenv`, by
whoever owns it.

### 9.3 The fail-before arms

| arm | command | result |
|---|---|---|
| D1, the new pins | `FU_INJECT=dump ... -k "dump or wedge_report"` | **2 failed** (`io.UnsupportedOperation: fileno`) |
| D1, the real wedge tests | `FU_INJECT=joins ... <whole file>` | **5 failed, 12 passed** in 138.51 s, each wedge failure carrying its own message and a dump naming `shutdown <- _abandon_pool <- close_worker_pool`.  (Two of the five are an artifact of editing `_lens_traced.py` while that run was in flight -- `inspect.getsource` read a shifted file -- and are not reproducible on the final tree; the three wedge failures are the point of the arm) |
| D2, shipped helper | `FU_INJECT=d2 ... -k "census or expir"` | **3 failed** in 47.11 s |
| D2, the one-line alternative | `FU_INJECT=d2one ... -k helper_first` | **1 failed** in 16.27 s (and the other three pass) |
| D7 | `fu2_d7_pin_detectors.py`, both builds | shipped detector `[]` on `carrier.py`; extended detector reports the `with` site |

### 9.4 Orphan sweep

Every probe that breaks a pool leaves worker processes behind by construction.  Swept on both
builds at the end of the package:

| build | command | my processes found |
|---|---|---|
| Windows 3.14.6 | `Get-CimInstance Win32_Process -Filter "Name='python.exe'"`, matched on command line | **0** |
| WSL 3.12.3 | `ps -eo pid,ppid,etimes,cmd \| grep -i python` | **0** |

Nothing with `lum_pool2` in its command line survived, on either build, after every probe and every
gate arm.  The processes that ARE resident are other agents' -- a GBD pytest lane in WSL, and in
Windows a `lum_mb3` probe pair, an `r3run.py`, a `probe_v_e5_freeze.py`, the VS Code isort server
and the blender-mcp extension -- identified by their command lines and left alone.

This is a better outcome than the verification's (12 + 4 swept), and the reason is worth recording:
the two `sigign` arms that hold SIGTERM-ignoring workers were the only runs that could leave any,
and on this box their children did not survive the parent's SIGKILL.  That is a property of the
run, not a guarantee -- a `sigign` arm that is interrupted differently can leave workers sleeping
for the remaining 900 s, so the sweep is still the right thing to do after any run of
`fu4_exit_hang.py`.

---

## 10. What I could not measure

1. **Whether a fixed `as_completed` timeout is safe at production field sizes.**  My ladder stops at
   N = 1024 because a single cold N = 32768 dispatch would not fit in this package's budget, and the
   module's own comments put the Newton path there ("free ~17 GB at N=32768 before Newton starts").
   Everything I say about N = 32768 in section 6 is an EXTRAPOLATION from the N <= 1024 ladder and is
   labelled as one; it is the reason I recommend a bootstrap bar, whose derivation does not depend
   on it.

2. **The sibling pool's bootstrap cost.**  `carrier._multi_parallel_results`'s initializer ships
   `uniq`, `groups_k`, `common` and a glass snapshot, so a sentinel bar there has to be priced
   against a genuinely expensive bootstrap that I did not measure.  I confirmed only that the
   exposure exists (statically, and now under the extended pin), which is what D7 required; I did
   not build a wedge driver for it.

3. **The D6 result on Windows.**  The `sigign` wedge is POSIX-only by construction -- it depends on
   a worker ignoring SIGTERM, and `TerminateProcess` cannot be ignored -- so the exit-hang numbers
   and the two stacks in section 7 are WSL 3.12.3 only.  The verification reached the same wall for
   the same reason (four independent natural break drivers failed to wedge Windows 3.14.6 on either
   tree), so this is a property of the platform, not a gap in the measurement.  Whether Windows can
   reach the same exit hang by some other route is unknown.

4. **Whether the D2 boundary race was ever hit in the wild.**  It is reachable by construction and
   the shipped code now survives it, but I have no evidence it has occurred; the leak the
   verification actually measured is the ordinary expiry, which all four pins cover.

5. **Why the 2026-09-14 breakage happened.**  Unchanged from the WP report and the verification: the
   trigger state is gone.  Nothing in this package went looking for it.

6. **Why the combined `FU_INJECT=joins,dump` arm did not finish.**  I ran the whole file with BOTH
   the pre-fix teardown helpers and the pre-fix `_thread_dump` injected, intending a side-by-side
   against the `joins`-only arm.  It printed all seventeen outcomes (`FFF............FF`) and then
   never reached pytest's summary line; I killed it and did not chase it, because it is a property
   of the injection harness and not of the library, and because the two arms that matter are
   unambiguous on their own -- `FU_INJECT=dump` is the fail-before (both new pins red with
   `io.UnsupportedOperation: fileno`) and `FU_INJECT=joins` on this branch is the fixed-path
   demonstration (three wedge failures, each with its own message and a real dump).  The
   `joins`-only arm completed normally in 138.51 s, so whatever stalled the combined run is not
   reproducible from the `joins` injection alone.

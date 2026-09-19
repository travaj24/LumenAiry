# WP-B13 -- the traced lens's Newton worker pool hangs instead of falling back

Handoff item **4.1b (P1)**, Wave 5 item **B**.  Branch
`fix/newton-pool-broken-fallback`, cut from `96cb2096`.

---

## 0. Terms

The defect lives in three pieces of machinery whose names are easy to confuse, so
they are defined once here and used with these meanings throughout.

| term | what it means in this report |
|---|---|
| **the Newton pool** | the module-level, persistent `ProcessPoolExecutor` (spawn context) that `apply_real_lens_traced` uses to split its Newton ray-map inversion into chunks.  Built lazily by `lumenairy/elements/_lens_traced.py::_get_persistent_worker_pool`, torn down by `close_worker_pool`. |
| **the dispatcher** | `_invert_newton_parallel`, the closure inside `apply_real_lens_traced` that decides whether to use the pool, splits the points into chunks, submits them and collects the results. |
| **the clamp** | `_newton_resolve_workers`, which lowers the requested worker count to what the box can hold.  It reads LIVE free memory, so its answer is not a constant. |
| **the size gate / the bars** | `_POOL_MIN_PIXELS` (200 000 points, cold) and `_POOL_MIN_PIXELS_WARM` (8 000, once a pool is up): below them the dispatcher runs in process because a pool cannot amortise. |
| **a broken pool** | an executor that has lost a worker.  CPython fails every pending future with `BrokenProcessPool`; the executor can serve nobody afterwards. |
| **the fallback** | the dispatcher's answer to a broken pool: run the identical inversion IN PROCESS.  It is bit-identical to the pooled answer by construction (the payload pins the Chebyshev backend and ships the parent's fit -- `test_fix_newton_pool_memory.py`), so it costs wall time and moves no number. |
| **the wedge** | the defect: the fallback never ran, because its first action blocked forever. |
| **the manager thread** | CPython's `concurrent.futures.process._ExecutorManagerThread`, one per executor.  On a dead worker it runs `_terminate_broken`. |
| **the feeder thread** | `multiprocessing.queues.Queue`'s `QueueFeederThread`, which writes pickled tasks from the parent into the call-queue pipe.  The pipe buffer is 8 KiB and the Newton payload is ~1.9 MB, so this thread is normally BLOCKED inside `connection._send_bytes` while a dispatch is in flight. |

Everything below was measured on the maintainer's box (Ryzen 9 5950X, 24 threads,
137.4 GB) on 2026-09-14, python 3.14.6 / numpy 2.4.4 on Windows and python 3.12 in
WSL, with `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` on every command
line.  **The box was carrying other agents' full pytest runs and the maintainer's own
long jobs throughout**, which is realistic for this defect (it is a load-sensitive
failure) but inflates every wall time quoted here by roughly 3-5x against an idle box.

---

## 1. The two captured dumps

The handoff records two `faulthandler` captures from the 5.47.0 release gate
(`handoff/orchestrator_log_2026_09_13-14.md`, the 13:13 and 13:39 entries).  The raw
dumps lived in the orchestrator's scratch logs, which are gone; what the log preserves
is the reading, and it is unambiguous:

> Slow lane #2 HUNG at `test_audit2609_b4_collins_transport.py::TestGateCTwoGroupChain`
> (fixture `p5_arms`) -- faulthandler dump: `close_worker_pool -> ProcessPoolExecutor.shutdown(wait=True)`
> on a BROKEN pool joins the `QueueFeederThread` stuck in `connection._send_bytes`
> (CPython `_terminate_broken` deadlock).

> POOL HANG LOCALISED (library side): `_invert_newton_parallel`'s broken-pool fallback
> calls `close_worker_pool()` -> `shutdown(wait=True)` -> CPython `_terminate_broken`
> joins the feeder thread stuck in `_send_bytes` -> deadlock; the serial fallback is
> never reached.

Four threads, one process: the main thread in `close_worker_pool -> shutdown`, a second
main-ish thread in `as_completed` waiting on futures a broken pool will never complete,
the manager thread inside `_terminate_broken -> join`, the feeder inside `_send_bytes`.

**A dump of the same wedge was re-captured here**, deterministically, on the pre-fix
tree -- section 4.

---

## 2. What was NOT reproducible, and what that rules out

The handoff's reproduction recipe was two pytest ids under pytest's DEFAULT
file-descriptor capture, "5 of 5 attempts".  Run on this branch's tree today:

| run | capture | result |
|---|---|---|
| `test_audit2609_b4_collins_transport.py -k test_both_transports_reproduce_the_audit_s_own_readings` | default (`fd`) | **1 passed in 203.70 s** (faulthandler fired at its 180 s mark and showed the QueueFeederThread idle in `notempty.wait()` and the main thread doing real work at `_lens_traced.py:3376`, i.e. slow, not wedged) |
| the same id, with the pool instrumented | default (`fd`) | **1 passed in 403.95 s** |
| the same id, with the pool instrumented | `--capture=sys` | **1 passed in 396.59 s** |

So the hang did not reproduce, on either capture, on the tree it was reported against.
That is consistent with the handoff's own note that the interaction "appeared on this
box at about 11:00" and is "a box-state / harness interaction"; the box has since been
through other work, and whatever state produced it is gone.

Two things follow, and they are the useful part:

1. **The fd-capture axis shows nothing today.**  Both captures pass, both take the same
   time to within 2 %, and -- section 3 -- both drive the pool through the *identical*
   pathological pattern.  The "spawned workers inheriting or duplicating captured file
   descriptors" hypothesis is not supported by anything measured here; it is also not
   refuted, because the trigger state is gone.  This is recorded as NOT ESTABLISHED
   (section 8).
2. **The library defect is independent of it.**  Whatever breaks the pool, the library's
   response to a broken pool is a deadlock, and that is reproducible on demand -- which
   is what sections 3 to 5 do.
3. **A hang of a DIFFERENT shape does reproduce, on WSL, on both trees.**  There the
   parent sits in as_completed (not in close_worker_pool) because the spawned
   workers bootstrap one at a time at three to six minutes each.  That is a second,
   independent exposure and it is written up in section 7.4; it is what makes the b4
   reproducer unusable as a gate on that build.

---

## 3. The measured cause of the pool breakage

### 3.1 The suspected race is refuted for this reproducer

The handoff's leading hypothesis was "a race between the two arm threads on the shared
pool's teardown / re-creation": `propagate_traced_carrier_chain`'s two arms run in a
`ThreadPoolExecutor(max_workers=2)` and share the persistent pool.

Every `ProcessPoolExecutor` construction, `submit`, `shutdown` and every
`_get_persistent_worker_pool` / `close_worker_pool` / `_newton_resolve_workers` call was
logged with a thread id through a pytest plugin
(`validation/probe_newton_pool/pool_probe_plugin.py`), for the b4 reproducer, under both
captures.  **Every pool event in both runs came from the MAIN thread** (tid 62256 under
fd capture).  The stack recorded at each construction is the same one every time:

```
test_audit2609_b4_collins_transport.py:638:p5_arms
carrier.py:10427:propagate_traced_carrier_chain
_lens_traced.py:13147:apply_real_lens_traced
_lens_traced.py:12539:_invert_newton_parallel
_lens_traced.py:1215:_get_persistent_worker_pool
```

`apply_real_lens_traced`'s two-thread arm (`parallel_amp`) runs `apply_real_lens`, which
does NOT invert a ray map, so it never reaches the Newton pool.  The two-arm sharing is
real in principle -- and section 5.2 closes it -- but it is not what happened here.

### 3.2 What actually happened: the pool was rebuilt on almost every dispatch

The same instrumentation shows the real pattern.  Four dispatches in one test:

| dispatch | clamp input (`n_total`, `fit_points`) | clamp answer | getter's action |
|---|---|---|---|
| 1 | 1 048 576, 6 027 025 | **6** | build (no pool yet) |
| 2 | 1 048 576, 2 805 625 | **8** | shutdown(wait=False) + build |
| 3 | 1 048 576, 6 027 025 | **4** | shutdown(wait=False) + build |
| 4 | 1 048 576, 2 805 625 | **8** | shutdown(wait=False) + build |

Four constructions, three teardowns, **26 spawned worker interpreters for four
dispatches**, with the outgoing and incoming pools alive at the same time because
`shutdown(wait=False)` does not wait.  The same test under `--capture=sys` read
**5, 8, 5, 8** -- four constructions again.

The clamp's answer moves because `_newton_resolve_workers` divides live available memory
by a measured per-worker commit model; on a box whose free memory is swinging (three
other pytest runs and two long-running jobs were resident), the same call site answers
6 at one moment and 4 a minute later.  The shipped getter turned *any* change of count
into a full teardown-and-respawn:

```python
if _PERSISTENT_POOL_NWORKERS == n_workers:
    return _PERSISTENT_POOL
_PERSISTENT_POOL.shutdown(wait=False)   # ...else rebuild
```

That is the pool-breaking pressure: a spawn storm of short-lived worker interpreters,
each importing numpy/scipy/numba, overlapping with the pool that is still dying, on a
box already at its memory and scheduler limits.  It is exactly the condition under which
a worker fails to come up or is killed -- and the executor reports that as
`BrokenProcessPool`.

### 3.3 A rebuild alone does not break the pool

Stated so the fix is not credited with more than it earns.  A probe
(`probe_p4_rebuild_race.py`) ran a real traced-carrier chain on one thread while a
second thread forced a rebuild every 200 ms -- **88 rebuilds during one 19 s dispatch**,
on the PRE-FIX tree.  The dispatch completed with the correct field
(`sha 3b968cab...`, the same hash the undisturbed run produces).  `shutdown(wait=False)`
lets in-flight work finish, so the churn is a cost and a hazard, not a guaranteed break.
The hazard is real all the same: the teardown can land between the getter returning an
executor and the dispatcher submitting to it, and then `submit` raises
`RuntimeError('cannot schedule new futures after shutdown')`, which the dispatcher's
own except clause treats as pool infrastructure failure -- i.e. it reaches the wedge.

---

## 4. The wedge, read from CPython and reproduced

### 4.1 Why `wait=False` alone is not the fix

The handoff proposed a one-line minimum fix: `shutdown(wait=False, cancel_futures=True)`
in the broken-pool branch.  Reading CPython 3.14.6's
`Lib/concurrent/futures/process.py` shows why that is necessary but not sufficient:

* line 290: `self.shutdown_lock = executor._shutdown_lock` -- the manager thread and the
  executor share ONE lock object.
* line 505-507: `def terminate_broken(self, cause): with self.shutdown_lock: self._terminate_broken(cause)`
  -- the manager holds that lock for the whole of `_terminate_broken`.
* inside `_terminate_broken`: `self.call_queue._terminate_broken()` (which ends in
  `join_thread()`, an unbounded join of the feeder thread) and
  `self._join_executor_internals(broken=True)` (which ends in `p.join()` per worker,
  also unbounded).
* line 859-868: `def shutdown(self, wait=True, *, cancel_futures=False): with self._shutdown_lock: ...`
  -- **`shutdown` acquires that same lock FIRST, before it ever looks at `wait`.**

So a parent calling `shutdown` on an executor whose manager thread is inside
`_terminate_broken` blocks on the lock regardless of what it passes for `wait`.  The
real requirement is stronger than a flag, and it is what this fix implements: *do not
call `shutdown` on the calling thread at all.*

### 4.2 The fail-before, archive-to-archive

`validation/probe_newton_pool/probe_p7_broken_fallback.py --mode stub` installs, as the
module's persistent pool, an executor that (a) fails every future with
`BrokenProcessPool` and (b) never returns from a `shutdown` that is asked to wait.  No
child processes are involved, so what it measures is the library's code path and nothing
else.  Run against a `git archive` tree of the parent commit `96cb2096`, in a child
process with `PYTHONPATH` pointed at that tree and `lumenairy.__file__` printed and
asserted under it:

```
{"lumenairy": "...\\scratchpad\\base96cb\\lumenairy\\__init__.py", "version": "5.47.0",
 "python": "3.14.6", "event": "import"}
{"event": "serial_reference_enter"}
{"sha": "3b968cab...", "event": "serial_reference_exit"}
{"workers": 4, "event": "stub_installed"}
Timeout (0:03:00)!
...
  File "...\threading.py", line 369 in wait
  File "...\threading.py", line 670 in wait
  File "...\probe_p7_broken_fallback.py", line 132 in shutdown
  File "...\base96cb\lumenairy\elements\_lens_traced.py", line 1409 in close_worker_pool
  File "...\base96cb\lumenairy\elements\_lens_traced.py", line 12624 in _invert_newton_parallel
  File "...\probe_p7_broken_fallback.py", line 92 in _run
```

`_lens_traced.py` line 12624 (as it then was) is the `close_worker_pool()` call in the
dispatcher's broken-pool branch, and line 1409 (as it then was) is
`_PERSISTENT_POOL.shutdown(wait=True)`.  The serial line one below never ran.  Exit code
1, killed by the probe's own 180 s deadline -- the process would otherwise have hung
forever.

### 4.3 The fix-after

The identical probe on this branch:

```
{"fallback_seconds": 14.584, "sha": "3b968cab...", "ref_sha": "3b968cab...",
 "identical": true, "shutdown_calls": [{"wait": false, "cancel_futures": true}],
 "event": "fallback_returned"}
```

and with a REAL spawn pool whose every worker kills itself with `os._exit` the moment it
is handed a chunk (`--mode kill`):

```
{"fallback_seconds": 37.479, "sha": "3b968cab...", "ref_sha": "3b968cab...",
 "identical": true, "shutdown_timeouts": 0, "event": "fallback_returned"}
```

Both were re-run on the final tree once the box had quietened, with the same verdicts and
smaller numbers (4.498 s and 4.836 s) -- the wall time is the SERIAL Newton inversion the
fallback runs, so it tracks box load and is not a claim about anything.

---

## 5. The fix

Both layers are in `lumenairy/elements/_lens_traced.py`.

### 5.1 Layer (a) -- robustness: the invariant, and the fallback ladder

> **INVARIANT.  No Newton-pool code path joins an executor for an unbounded time.**

Two mechanisms carry it, and no other function in the module is allowed to call
`shutdown` directly (an AST pin in the new test file enforces that, because a text
search cannot: the comments that *explain* the rule contain the string it forbids).

| helper | what it does | used for |
|---|---|---|
| `_abandon_pool(ex)` | appends `ex` to `_ABANDONED_POOLS` (its own lock, so it is safe to call from inside `_PERSISTENT_POOL_LOCK`) and starts a **daemon** thread that calls `shutdown(wait=False, cancel_futures=True)` and then drops the reference.  A daemon thread that blocks on a wedged `_shutdown_lock` costs nothing: nothing joins it. | a pool that has marked itself broken; the stale pool a rebuild replaces; the fall-through when `threading.Thread` itself refuses (gh-109047, interpreter finalising) |
| `_shutdown_pool_bounded(ex, timeout=None)` | runs the joining `shutdown(wait=True)` that `close_worker_pool` promises on a helper thread and joins THAT for at most `_POOL_SHUTDOWN_TIMEOUT`; on expiry it records `_POOL_SHUTDOWN_TIMEOUTS += 1`, abandons the pool and returns `False` | every healthy teardown |

**The fallback ladder**, in the order the dispatcher takes it:

1. The pool answers -> use the pooled result.  (Unchanged.)
2. A chunk raises `NewtonPayloadNotResident` -> re-submit that chunk WITH the payload.
   (Unchanged; still ordered ahead of the infrastructure clause, still pinned.)
3. A worker refuses the pinned Chebyshev backend (`NewtonWorkerBackendUnavailable`) ->
   release the in-flight claim, warn once per process, run serial.  (Unchanged except
   for the claim release, which stops a long serial re-run from blocking another
   thread's legitimate rebuild.)
4. `BrokenProcessPool` / `RuntimeError` / `OSError` / `EOFError` -> release the claim,
   call `close_worker_pool()` (which is now bounded, and on a broken pool does not join
   at all), **then run serial**.  This is the rung that never used to be reached.
5. `close_worker_pool()` itself: swap the pool reference out under `_PERSISTENT_POOL_LOCK`
   and clear the promotion state, then tear down OUTSIDE the lock -- `_abandon_pool` if
   the executor has already marked itself broken, `_shutdown_pool_bounded` otherwise.
   Tearing down outside the lock is the second half of the fix: the shipped version held
   the module lock across the join, so a wedged close froze every other thread's
   `_get_persistent_worker_pool`, `_note_pool_deferral` and `_pool_reuse_is_likely` too.

**The bound.**  `_POOL_SHUTDOWN_TIMEOUT = 120.0`.  It bounds a WAIT; nothing derives a
number from it.  Measured healthy `close_worker_pool` over a worker ladder, warm pools,
three reps each (`probe_p6_teardown_ladder.py`, 2026-09-14):

| workers | close_worker_pool seconds |
|---|---|
| 1 | 0.188  0.199  0.201 |
| 2 | 0.483  0.302  0.269 |
| 4 | 0.470  0.299  0.226 |
| 8 | 0.770  1.334  1.022 |
| 16 | 3.723  4.062  4.592 |

Worst 4.592 s; the state on the other side is UNBOUNDED.  120 s is 26x the worst rung,
leaving room for a wider pool on a loaded box, and the cost of it being too LOW is only
that a slow-but-healthy teardown finishes on the reaper thread instead of inline.

### 5.2 Layer (b) -- the root cause: the rebuild rule

> **INVARIANT.  The live pool's worker count is a CEILING on concurrency, not a promise
> about one call.  It is raised only when a call needs more AND nothing is in flight; it
> is never lowered except by `close_worker_pool()`.  A pool that has marked itself broken
> is retired and replaced unconditionally.**

Why the ceiling is sound: the clamp exists to bound the memory a dispatch commits, and
that is set by the CHUNK COUNT, which the dispatcher derives from its own `n_cpu`
(`np.array_split(np.arange(n_total), n_cpu)`, unchanged).  Concurrent chunk memory is
`min(pool_workers, n_chunks) x bytes_per_chunk`, which for `pool_workers >= n_chunks` is
exactly what the clamp allowed for `n_chunks` workers and for `pool_workers < n_chunks`
is less.  The surplus workers stay idle; their cost is their resident set.

> **CORRECTED 2026-09-15 (VERIFY-WP-B13 defect D3).**  This paragraph read
> "**33.0 MB mean / 48.8 MB peak** each (15 warm pools, `probe_p6_teardown_ladder.py`)
> ... 2 % of one working worker".  That figure was measured on workers warmed with
> `ex.map(abs, ...)`, which is not a state this rule produces.  Re-measured on both
> builds with the two reachable states separated inside ONE pool
> (`validation/probe_wp_b13_followups/fu1_worker_footprint.py`, a 12-wide pool serving a
> 4-worker dispatch, each worker labelled by its own `_WORKER_PAYLOADS`): a worker that
> has **served a Newton chunk** reads **98.7 / 98.8 / 100.8 MB mean** at N = 256 / 512 /
> 1024 on Windows 3.14.6 (worst 101.6 MB) and **73.2 / 74.1 / 76.9 MB** on WSL 3.12.3
> (worst 76.9 MB); a worker in the same pool that **served none** reads **52.2 MB**
> (Windows) / **39.2 MB** (WSL).  Against the **~1.7 GB per ACTIVE worker** the clamp
> models for a 262 144-point / 279^2-fit dispatch that is **6 %**, not 2 %, and a
> 16-wide kept pool holds about **1.6 GB**, not 0.5 GB.  The trade the rule rests on is
> unchanged; the number is not the number.

Effect on the measured sequence: 6, 8, 4, 8 built four pools before and builds **two**
now (6, then 8; the 4 is served by the live 8).

Concurrency: `_POOL_INFLIGHT` counts DISPATCHES in flight on the cached pool (corrected
2026-09-15, VERIFY-WP-B13 defect D4: this said "chunks", and the counter is moved once
per dispatch, not once per chunk), moved by
`_note_pool_inflight` under the pool lock, taken by the dispatcher immediately after it
is handed an executor and released in a `finally` on every exit path (and early, by
hand, before either fallback's long serial re-run).  While it is non-zero a wider
request is served by the LIVE pool rather than by tearing it down.

One window is left open on purpose.  Between `_get_persistent_worker_pool` returning an
executor and the dispatcher claiming it, a rebuild by another thread can still land, and
`ex.submit` then raises `RuntimeError` -- which takes rung 4 of the ladder, i.e. a
bit-identical serial answer at a wall-time cost.  Closing it would mean re-reading
`_PERSISTENT_POOL` after the claim and OVERRIDING whatever the getter returned; that was
implemented, and it broke `test_niche_audit_e_prepared_and_enums.py`'s two H2 pins
(`test_h2_pool_engagement_follows_the_ram_pricing_both_ways` and
`test_h2_resolved_cap_travels_in_the_pickled_payload`), which wrap the returned executor in
a spy to observe what the parent submits.  The getter is a substitution point the library's
own tests -- and any caller with a wrapped executor -- depend on, and a bounded
bit-identical fallback is the better trade than breaking that contract, so the re-read was
removed and the window is documented in the source.

One consequential detail: the warm-pool size bar now reads
`_PERSISTENT_POOL_NWORKERS >= n_cpu` rather than `== n_cpu`.  Under the ceiling rule a
pool wide enough to serve the call has no spawn left to amortise, which is the only
thing that bar is about; leaving `==` would have sent calls in the 8 000-200 000 point
band back to the serial path whenever the clamp's answer dipped below the live pool's
width.  The pool path is bit-identical to serial, so this moves wall time only.

### 5.3 What the fix does NOT claim

Once CPython's manager thread is wedged inside `_terminate_broken`, the process still
cannot exit cleanly: `concurrent.futures.process._python_exit` is registered through
`threading._register_atexit` and joins every live manager thread.  MEASURED here
(`probe_p3_atexit_state.py`): at the time this module's own `atexit` handler runs, the
executor's children are ALREADY gone and the handler has nothing to reap -- `_python_exit`
ran first.  So the library's atexit registration is a no-op on a healthy process, and on
a wedged one the hang moves to interpreter exit.

That is the fix's actual gain, stated honestly: the wedge moves from the MIDDLE of a
computation, where the caller's result is lost and a release gate dies, to AFTER the call
has returned its bit-identical answer.  Fixing the exit-time join would require removing
the manager thread from `threading._shutdown`'s private bookkeeping; that was judged not
worth doing on the default path of every traced-lens call.

### 5.4 A sibling with the same exposure, deliberately NOT changed here

`lumenairy/propagators/carrier.py::_multi_parallel_results` -- the congruence-level
process pool behind `congruence_workers` (niche D8) -- runs its executor as
`with ProcessPoolExecutor(...) as ex:`.  `Executor.__exit__` is `shutdown(wait=True)`, so
that pool has exactly the same exposure to a wedged `_terminate_broken`: if a congruence
worker dies while the feeder is mid-write, the `with` block's exit can block forever.
The release gate's other hang was in that path
(`test_niche_d8_congruence_workers::test_parallel_recombined_field_is_fp_identical_to_serial`,
orchestrator log 15:51).  It is a different pool with a different lifetime and a different
failure policy (it RAISES rather than falling back), so it is out of this item's scope and
is recorded here as a follow-up rather than changed without its own measurement and
verifier.

---

## 6. Bit-identity of the fallback

The fallback's whole claim is that it moves no number.  Checked on twelve fields that
vary N, aperture, radius, `ray_subsample`, worker count and both fit backends
(`probe_p8_identity_fields.py`), each computed twice -- once with `n_workers=1` and once
with the pool engaged -- and compared byte for byte.

| # | N | aperture (mm) | radius (mm) | subsample | fit | workers | Newton points | serial s | pooled s | win identical | WSL identical | max abs delta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 256 | 3.0 | 9.0 | 2 | spline | 2 | 16384 | 3.252 | 1.405 | yes | yes | 0.0 / 0.0 |
| 2 | 256 | 3.0 | 9.0 | 2 | spline | 4 | 16384 | 0.284 | 1.71 | yes | yes | 0.0 / 0.0 |
| 3 | 256 | 3.0 | 9.0 | 2 | polynomial | 4 | 16384 | 1.541 | 0.25 | yes | yes | 0.0 / 0.0 |
| 4 | 256 | 3.0 | 12.0 | 2 | spline | 3 | 16384 | 0.312 | 1.539 | yes | yes | 0.0 / 0.0 |
| 5 | 256 | 4.0 | 9.0 | 2 | polynomial | 2 | 16384 | 0.314 | 0.115 | yes | yes | 0.0 / 0.0 |
| 6 | 320 | 3.0 | 9.0 | 2 | spline | 4 | 25600 | 0.539 | 1.678 | yes | yes | 0.0 / 0.0 |
| 7 | 320 | 3.0 | 15.0 | 2 | polynomial | 4 | 25600 | 0.931 | 0.526 | yes | yes | 0.0 / 0.0 |
| 8 | 384 | 3.0 | 9.0 | 3 | spline | 5 | 16384 | 0.418 | 2.136 | yes | yes | 0.0 / 0.0 |
| 9 | 384 | 5.0 | 12.0 | 3 | polynomial | 3 | 16384 | 1.12 | 0.242 | yes | yes | 0.0 / 0.0 |
| 10 | 512 | 3.0 | 9.0 | 4 | spline | 4 | 16384 | 0.504 | 2.273 | yes | yes | 0.0 / 0.0 |
| 11 | 512 | 3.0 | 9.0 | 4 | polynomial | 6 | 16384 | 0.712 | 0.725 | yes | yes | 0.0 / 0.0 |
| 12 | 512 | 6.0 | 18.0 | 4 | spline | 2 | 16384 | 0.5 | 2.348 | yes | yes | 0.0 / 0.0 |

12 of 12 identical on both builds, worst `max|delta|` **0.0** on both -- the serial
fallback reproduces the pooled field byte for byte at every row.  The `serial s` /
`pooled s` columns are there to show the rows are real work, not to make a performance
claim: the pool loses at these deliberately small sizes, which is exactly what the size
bars this probe suppresses exist to prevent.

The same identity is asserted inside the two fallback tests (the field returned by a
broken-pool call is compared with `np.array_equal` against an `n_workers=1` reference)
and by the shipped `test_fix_newton_pool_memory.py::test_a_capped_pool_is_bit_identical
_to_serial`, which this change leaves green.

---

## 7. Verification

### 7.1 The traced / carrier / pool families, per build and per capture

Files: `test_fix_newton_pool_broken_fallback.py`, `test_fix_newton_pool_memory.py`,
`test_niche_newton_pool_both_fits.py`, `test_carrier_field.py`,
`test_carrier_referenced.py`, `test_audit2609_a3_traced_lens.py`,
`test_audit2609_a3_verify_traced.py`, `test_niche_audit_w9_traced_determinism.py`,
`test_niche_d15_deterministic_traced_fit.py`, `test_hammer_h3_traced_nyquist_guard.py`,
`test_niche_audit_w3_elements.py`, `test_niche_perf_round2_2026_08_10.py`,
`test_niche_audit_e_prepared_and_enums.py`.  (There is no `tests/unit/test_lens_traced*.py`
in this tree; these are the traced-lens and Newton-pool families that exist.)

| build | capture | result | wall |
|---|---|---|---|
| Windows, python 3.14.6 | default (`fd`) | **363 passed, 1 skipped** | 989.84 s |
| Windows, python 3.14.6 | `--capture=sys` | **363 passed, 1 skipped** | 395.02 s |
| WSL, python 3.12.3 | default (`fd`) | **363 passed, 1 skipped** | 906.14 s |
| WSL, python 3.12.3 | `--capture=sys` | **363 passed, 1 skipped** | 430.64 s |

The one skip is the same on all four: `test_fix_newton_pool_memory.py:1217`, whose BLAS
reduces identically at every width tried, so the box cannot witness the defect that test is
about.

### 7.2 The new file on its own

| build | capture | result | wall | slowest test |
|---|---|---|---|---|
| Windows, python 3.14.6 | default (`fd`), loaded box | **15 passed** | 189.02 s | 92.70 s (`..._under_either_pytest_capture[sys]`) |
| Windows, python 3.14.6 | default (`fd`), quiet box | **15 passed** | 26.21 s | -- |
| WSL, python 3.12.3 | default (`fd`) | **15 passed** | 52.23 s | 17.65 s (`..._under_either_pytest_capture[sys]`) |

Three tests spawn a CHILD pytest or a child interpreter and are marked `slow`; their wall
time is dominated by that child's own import and first-traced-call warm-up (measured at
45-56 s for the first `apply_real_lens_traced` in a cold process while the box was loaded,
2-4 s once warm).  On a quiet box the whole file runs in 26 s.  No test in the file can hang
the suite that runs it: each either joins a daemon thread with a deadline or runs a child
under `subprocess.run(timeout=...)`.

### 7.3 The two handoff reproducers

`test_audit2609_b4_collins_transport.py` + `test_niche_audit_w9_dispatch2.py`, whole files.

| build | capture | tree | result |
|---|---|---|---|
| Windows 3.14.6 | default (`fd`) | this branch | **206 passed** in 310.16 s |
| Windows 3.14.6 | `--capture=sys` | this branch | 205 passed, **1 failed** in 318.29 s -- see below |
| Windows 3.14.6 | `--capture=sys` | base `96cb2096` | **206 passed** in 363.61 s |
| WSL 3.12.3 | default (`fd`) | this branch | **did not complete** (section 7.4) |
| WSL 3.12.3 | default (`fd`) | base `96cb2096` | **did not complete**, same dump |
| WSL 3.12.3 | `--capture=sys` | this branch | **did not complete**, same shape |

The single Windows `--capture=sys` failure is
`test_niche_audit_w9_dispatch2.py::test_the_routed_member_beats_the_old_one_against_an_exact_oracle[512-0.4111]`,
and it is not a pool test: it asserts that an `angular_spectrum_propagate_mft` oracle is
pad-converged, and on that run the 4096-square pad returned values of order 1e71
(`fid = 0.103`).  There is no traced lens, no Newton inversion and no process pool anywhere
in it.  It passes alone on this tree (3 passed in 9.25 s), the whole `w9_dispatch2` file
passes alone on the base tree (80 passed in 11.47 s), and the same two files pass under
`--capture=sys` on the base tree (206 passed) -- so it is an order-and-box-state MFT blow-up
of the same family as the `c7` / `c8` halo reds the handoff records for this box, and it is
recorded rather than explained.

### 7.4 The WSL reproducer does not complete, on EITHER tree

Run on WSL, the b4 fixture wedges, and the dump is the same on the pre-fix tree:

```text
Thread ... [QueueFeederThread]      connection._send_bytes     <- call-queue pipe full
Thread ... [ExecutorManagerThread]  wait_result_broken_or_wakeup
Thread ... [MainThread]             concurrent.futures._base.as_completed
                                    _lens_traced.py line 12799 in _invert_newton_parallel
```

Note what is NOT in it: `close_worker_pool`.  The pool has not broken; the parent is simply
waiting for chunks that never come.  MEASURED with `ps` and `/proc` while it sat there
(three separate runs -- this branch under `fd`, base `96cb2096` under `fd`, this branch under
`--capture=sys`): of the four or five spawned workers, **exactly one at a time makes
progress**, each taking three to six minutes of CPU to finish its bootstrap, while its
siblings sit at 37 MB resident, ~200 memory mappings (a bare interpreter), 0:00 CPU and
identical context-switch counts.  The WSL box itself was idle (load average 0.52, 70 GB
free), so this is not starvation: worker startup over the `/mnt/c` mount is serialised and
costs minutes per worker, and `as_completed` in the dispatcher has no timeout.

Three consequences, kept separate:

1. It is NOT this change: the identical dump and the identical `ps` picture appear on the
   base tree.
2. It is NOT the capture: `--capture=sys` shows the same picture on WSL.
3. It IS a second, independent exposure, and a P1-shaped one -- **`as_completed` in
   `_invert_newton_parallel` has no timeout, so a worker that never comes up hangs the call
   forever.**  It is plausibly the other half of the release-gate story (a worker that takes
   minutes to start is eventually seen as dead, and the pre-fix fallback then wedged), and it
   is recorded as the top follow-up rather than fixed here: putting a timeout on
   `as_completed` moves a default on every traced-lens call, the bar would have to be derived
   from a chunk-cost model rather than picked, and `TimeoutError` is an `OSError` subclass, so
   it would land in the existing fallback clause -- a design that deserves its own
   measurement and its own verifier.  The user-side workaround is unchanged and already
   documented: `n_workers=1`.

The fix does reduce the exposure, by construction rather than by luck: the rebuild rule
removes three of the four pool constructions in that very reproducer, and each construction
is a worker-startup storm.

### 7.5 Walkers, pins and lint

| check | result |
|---|---|
| `scripts/check_doc_identifiers.py` | OK -- 621 API-claiming identifiers, 0 unresolved |
| `scripts/check_source_line_citations.py` (V18) | ok=107, drift=0 |
| `scripts/record_history_fingerprints.py --check` | OK, after re-recording `lumenairy.elements._lens_traced` with its reason |
| walker suite (13 `test_v4_16_0_walker_*` / `test_v5_*_walker_*` files) + `test_eme_census_determinacy.py` | 101 passed, 6 skipped |
| `test_audit_except_budget.py`, `test_public_api.py`, `test_v4_16_2_dispatcher_pin_doc_consistency.py`, `test_audit2609_a17_history_lint.py`, `test_audit2609_a23_census_mechanism.py` | passed.  The broad-`except` budget is untouched: every clause added here names its types |
| `test_v4_14_2_dispatcher_pin_cache_locks.py` | 102 passed, 6 skipped -- the new `_ABANDONED_POOLS_LOCK` needed an entry in that file's `_LOCK_WITHOUT_CACHE_EXEMPTIONS` list, with its reason, exactly as `_PERSISTENT_POOL_LOCK` has one |
| `wsl ruff check lumenairy/ tests/ validation/probe_newton_pool/` | All checks passed |
| `.test_durations` | valid JSON, 16 201 entries, all 15 new ids present, every value a float.  `durations_gap.py` reads 7 tracked ids without an entry: 2 in `test_v4_14_2_dispatcher_pin_cache_locks.py` and 1 in `test_niche_k3_perf.py` were already missing at the handoff, and the other 4 are in `test_v5_14_5_viewer_polarization.py`, untouched here |

### 7.6 Fail-before / fix-after

| probe | pre-fix tree (`96cb2096`, `git archive`, child process, `lumenairy.__file__` asserted under it) | this tree |
|---|---|---|
| `probe_p7 --mode stub` (broken AND wedged pool) | never returned; killed by its own 180 s deadline with the main thread in `close_worker_pool -> shutdown` | returns in 4.5 s (14.6 s on the loaded box), field byte-identical to serial, exactly one `shutdown(wait=False, cancel_futures=True)` asked of the broken executor |
| `probe_p7 --mode kill` (real spawn pool, every worker `os._exit`s on its first chunk) | reaches the same branch | returns in 4.8 s (37.5 s loaded), byte-identical, 0 bounded-wait expiries |
| `probe_p8` identity, 12 fields | -- | 12 of 12 byte-identical on both builds, worst delta 0.0 |

---

## 8. What could not be established

1. **Why pytest's default fd capture triggered the pool breakage on 2026-09-14.**  It is
   not reproducible on this box today: the two handoff reproducers pass under default
   capture, and the pool behaves identically under `--capture=sys` (same clamp sequence,
   same number of constructions, wall times within 2 %).  Without the trigger state there
   is nothing to measure, and the report declines to guess.  What IS established is that
   the pool's response to being broken no longer depends on knowing.
2. **Whether the 2026-09-14 breakage was the spawn storm of section 3.2.**  The storm is
   measured and is a sufficient pressure for a worker to fail to come up on a loaded box,
   and it is removed.  That it was THE cause on the day is an inference, not a
   measurement: no worker-side record of the failure survived (the Windows Application
   log recorded no python.exe faults, per the handoff).
3. **A natural (non-engineered) wedge of CPython's `_terminate_broken` on this build.**
   Every attempt to produce one here -- an early worker death, a worker dying while the
   feeder was blocked mid-`_send_bytes`, both with 8 MB payloads --
   (`probe_p1_synthetic_break.py`) ended with the feeder correctly unblocked by
   gh-107219's writer close and `shutdown(wait=True)` returning in 4-12 ms.  The wedge in
   sections 4.2/4.3 is therefore ENGINEERED (a stub executor that blocks in a joining
   `shutdown`), which is the right shape for a regression test anyway: it tests the
   library's decision, not CPython's timing.
4. **Why the WSL workers bootstrap one at a time.**  The serialisation is measured
   (section 7.4) and so is its cost, but the blocking point inside
   multiprocessing.spawn is not: the stalled children sit at a bare interpreter with
   ~200 mappings, and neither /proc/<pid>/syscall nor /proc/<pid>/wchan is
   populated on this WSL2 kernel, with no strace, gdb or py-spy available to
   ask directly.  The reading that fits every observation is that worker startup over the
   /mnt/c 9p mount is serialised, but that is an inference.
5. **One unexplained observation.**  In the instrumented b4 run, the module's `atexit`
   handler found `_PERSISTENT_POOL` already `None` although the last dispatch had built a
   pool; the direct probe of the same question (`probe_p3_atexit_state.py`) found it NOT
   `None`.  Nothing in the fix depends on which it is -- the pool is reaped by
   `_python_exit` either way (section 5.3) -- and it was not chased further.

---

## 9. Files

| file | what |
|---|---|
| `lumenairy/elements/_lens_traced.py` | both layers of the fix |
| `tests/unit/test_fix_newton_pool_broken_fallback.py` | 15 tests, none of which can hang the suite |
| `validation/probe_newton_pool/probe_p1_synthetic_break.py` | CPython-only break/shutdown scenarios |
| `validation/probe_newton_pool/probe_p2_chain_instrumented.py` | the b4 chain without pytest, pool events logged per thread |
| `validation/probe_newton_pool/pool_probe_plugin.py` | pytest plugin: every executor life-cycle event as JSONL |
| `validation/probe_newton_pool/probe_p3_atexit_state.py` | what atexit sees (section 5.3) |
| `validation/probe_newton_pool/probe_p4_rebuild_race.py` | forced-rebuild race + identity |
| `validation/probe_newton_pool/probe_p5_gate.py` | which gate keeps a call off the pool |
| `validation/probe_newton_pool/probe_p6_teardown_ladder.py` | the healthy-teardown ladder and idle-worker footprint |
| `validation/probe_newton_pool/probe_p7_broken_fallback.py` | the fail-before / fix-after wedge, stub and real-kill modes |
| `validation/probe_newton_pool/probe_p8_identity_fields.py` | serial == pooled over twelve fields |
| `validation/probe_newton_pool/*.json`, `*.jsonl` | the measurements quoted above, committed as evidence |
| `validation/probe_newton_pool/*.log` | the raw run logs (thread dumps, pytest tails).  `validation/**/*.log` is gitignored in this repository -- these are regenerable by re-running the probe or the pytest command beside each table, and the numbers that matter are quoted inline above and in the committed JSON |

# VERIFY-WP-B13 -- independent adversarial re-verification of the Newton worker-pool repair

Branch under test `fix/newton-pool-broken-fallback` (`70931f51`, `cd03e9f5`), base `96cb2096`.
Verification worktree `C:\tmp\lum_vpool` (branch `verify/wp-b13`); a read-only PRE tree at `96cb2096` in
`C:\tmp\lum_vpool_pre`.  Two builds: **Windows python 3.14.6 / numpy 2.4.4 / scipy 1.17.1** and
**WSL python 3.12.3 / numpy 2.4.6**.  Every command carried
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`.  Every probe that drives the pool ran in a child
process under `subprocess.run(..., timeout=...)` with `faulthandler.dump_traceback_later(..., exit=True)`
inside it, and every probe printed `lumenairy.__file__` as its first line with the tree pinned through
`PYTHONPATH`.  Probes and their JSON/JSONL are in `validation/probe_verify_b13/`.

Nothing under `lumenairy/` was edited.

---

## 0. Verdict table

| # | claim (WP-B13) | verdict | my number |
|---|---|---|---|
| 1a | `terminate_broken` holds `_shutdown_lock` across unbounded joins; `shutdown` takes it before reading `wait`, so `shutdown(wait=False, cancel_futures=True)` -- the handoff's one-liner -- blocks too | **CONFIRMED** | measured, not read: with the manager thread made to hold the lock 20.000 s, a non-waiting cancelling `shutdown` returned in **19.998 s** (3.14.6) and **20.004 s** (3.12.3); **0.000 s** on a healthy pool on both.  Source facts identical on both interpreters (3.14: 505/553/859; 3.12: 530/578/864; neither `join_thread()` nor `p.join()` takes a timeout) |
| 1b | fail-before: on `96cb2096` a broken pool wedges in `close_worker_pool -> shutdown`, and the serial fallback is never reached | **CONFIRMED, and stronger than the report claims** | the report calls its wedge ENGINEERED (sec. 8.3).  I produced a **natural** one: one worker SIGKILLed while the survivors ignore SIGTERM, WSL/3.12.3 -- PRE tree never returned (dump: manager in `terminate_broken -> _terminate_broken -> _join_executor_internals -> Process.join`, main thread in `process.py:865 shutdown <- _lens_traced.py:1409 close_worker_pool <- :12624 _invert_newton_parallel`), killed at 200 s.  Same driver on the fixed tree returned in **2.458 s**, byte-identical |
| 1c | on the fixed tree a broken pool reaches the serial fallback byte-identically | **CONFIRMED** | four independent drivers I wrote (worker `os._exit`, parent-side kill mid-flight, worker raising during **unpickling**, all workers killed with the feeder CONFIRMED blocked in `_send_bytes`) plus `sigign`: **eight broken-pool runs on the fixed tree across the two builds, all eight returned**, `identical: true`, `max|delta| 0.0`, `_POOL_SHUTDOWN_TIMEOUTS 0`, `_POOL_INFLIGHT 0`; two unbroken controls likewise identical |
| 2a | the pool-breaking cause is not the two-arm thread race | **CONFIRMED** | two threads dispatching at once, 6 dispatches each: fixed tree 3.06 s, PRE 8.11 s, no hang on either, every field byte-identical to its own `n_workers=1` reference |
| 2b | the clamp reads LIVE memory and its answer moves for the same call site | **CONFIRMED** | with 24 GB allocated and touched, the same call's clamp answer walked 6 -> 5 -> 4 -> 4 and back to 5 on free (`vp4 --mode memalloc`) |
| 2c | the shipped getter turned every change into a teardown-and-respawn: 4 constructions / 3 teardowns / 26 spawned interpreters for 4 dispatches; post-fix 2 constructions | **CONFIRMED** | clamp driven to 7, 8, 4, 8 through `set_max_ram` (the same `_free_b` the clamp reads): **PRE 4 constructions, 3 rebuild teardowns, 36 distinct worker pids; FIXED 2 constructions, 1 rebuild teardown, 23 distinct pids** |
| 3a | no Newton-pool path joins an executor for an unbounded time | **CONFIRMED** | `close_worker_pool()` on a pool that is broken and whose `shutdown` never returns came back in **0.000 s**, the wedge parked on `lumenairy-newton-pool-reaper`, and a fresh dispatch then completed in **0.96 s** byte-identical |
| 3b | the pool width is a CEILING: never shrunk, raised only when idle, broken pools replaced unconditionally | **CONFIRMED** | getter driven directly: 4 -> ask 2 = same pool; ask 4 = same; ask 8 idle = rebuild; **ask 16 while in flight = same 8-wide pool**; release then ask 16 = rebuild; mark broken then ask 1 = replaced.  Every replaced pool asked for `shutdown(wait=False, cancel_futures=True)` and nothing else |
| 3c | the clamp is honoured by the CHUNK COUNT, so surplus workers are idle | **CONFIRMED** (measured; the report only argues it) | on a **16-wide** pool, dispatcher clamp 3 / 5 / 10 gave chunks run 3 / 5 / 10 and **peak concurrent chunks 3 / 5 / 10**, fields byte-identical |
| 3d | `_POOL_SHUTDOWN_TIMEOUT = 120 s` is 26x the worst healthy teardown (4.592 s at 16 workers) | **CONFIRMED, wider than claimed** | quiet box, teardown through the public `close_worker_pool` after a real dispatch: 2w 0.109-0.169, 4w 0.146-0.180, 8w 0.168-0.297, 16w 0.321-0.379 -- worst **0.379 s**, i.e. **317x**.  The WP's own `probe_p6` re-run today reads worst 0.411 s / 291.9x |
| 3e | an idle kept-alive worker costs 33.0 MB mean / 48.8 MB peak, "2 % of one working worker" | **RESTATED -- the number is wrong by 2-3x** (defect D3) | that figure comes from workers warmed with `ex.map(abs, ...)`.  Workers in the state the ceiling rule actually leaves behind -- they have served a real Newton chunk -- measure **97.0 MB mean / 98.8 MB max** (N=512) and **99.4 / 99.9 MB** (N=1024).  The WP's own probe re-run today reads 45.7 MiB mean (its report says 33.0).  The conclusion (the trade is cheap against a ~1.7 GB active worker) survives; the quoted number does not |
| 4a | 13 files, 363 passed / 1 skipped under both capture modes on both builds | **CONFIRMED** | all four arms re-run here: Windows `fd` 300.88 s, Windows `sys` 981.38 s, WSL `fd` 1788.21 s, WSL `sys` 1743.32 s -- **363 passed, 1 skipped** every time, same single skip |
| 4b | serial == pooled byte-for-byte | **CONFIRMED** | every pooled field I produced -- 9 broken-driver runs, 4 clamp-ladder steps, 12 two-thread dispatches, 6 close-race dispatches, 3 wider-than-clamp dispatches -- equals its `n_workers=1` reference under `np.array_equal`, `max|delta| = 0.0`.  The WP's own 12-field `probe_p8` re-run here reads `all_identical: true`, `worst_delta: 0.0` in 210.3 s |
| 4c | the 15 new tests cannot hang the suite | **CONFIRMED, with a defect in the failure path** (D1) | pre-fix behaviour injected in memory: unbounded joins -> **3 failed, 12 passed in 118.80 s**; pre-fix rebuild rule -> **3 failed, 12 passed in 13.69 s**; both -> **6 failed, 9 passed in 118.69 s**.  No run hung.  But the three wedge tests report `io.UnsupportedOperation: fileno` instead of their own message, and the thread dump they promise is lost |
| 5a | the 2026-09-14 fd-capture trigger is not reproducible | **CONFIRMED** | nothing I ran showed a capture-dependent difference |
| 5b | a SECOND P1 exposure: `as_completed` in `_invert_newton_parallel` has no timeout | **CONFIRMED, and made deterministic** | with a pool initializer that blocks, BOTH trees hang on Windows with exactly the report's WSL stack: MainThread in `as_completed` at `_lens_traced.py:12799`, QueueFeederThread in `_send_bytes`, manager in `wait_result_broken_or_wakeup` (`vp2_slowboot_win_dump.txt`) |
| 5c | on WSL the spawned workers bootstrap one at a time at 3-6 minutes each | **REFUTED as a current reading** | same mount, same venv, today: 8 workers all import `lumenairy.elements._lens_traced` within **0.969 s**, 8 distinct pids, first-answer spread 24 ms.  Full ladder in sec. 5.  The EXPOSURE (5b) is real; the WSL bootstrap cost that exhibited it is not reproducible |
| 5d | a wedged manager thread still hangs INTERPRETER EXIT | **CONFIRMED** | fixed tree, WSL `sigign`: the call returned the right answer in 2.458 s and printed its exit line, then the process never terminated -- SIGKILL at 200 s, the runner reaping 5 processes and a later sweep finding 4 more (3 spawn workers + a resource tracker) from the repeat run |

---

## 1. What I did not take on trust

Three of the report's load-bearing statements were readings of somebody else's source or of a box state that
has since changed.  Each was re-derived here from an independent measurement:

* **The lock.**  The report quotes CPython line numbers.  I ran the executor instead: `vp1_cpython_lock.py`
  wraps `_ExecutorManagerThread._join_executor_internals` with a fixed sleep, kills a worker so the manager
  really enters `terminate_broken`, and times a `shutdown(wait=False, cancel_futures=True)` issued from the
  main thread.  The block is the hold, to 2 ms, on both interpreters; the same call on a healthy pool
  returns in 0.000 s.  That is the whole justification for "do not call `shutdown` on the calling thread",
  and it now rests on a measurement rather than on a reading.  It is also pinned
  (`test_a_non_waiting_shutdown_still_blocks_on_a_terminating_executor`), so a future CPython that fixes
  gh-\* and releases the lock will turn the pin red and invite the design to be simplified rather than
  leaving it over-built forever.
* **The clamp's movement.**  The report's 6, 8, 4, 8 came from a box whose free memory happened to swing.
  I drove it: `_newton_resolve_workers` reads
  `min(psutil.virtual_memory().available, lumenairy.memory.get_ram_budget())`, and `budget = 0.5*free - 2 GB`,
  `allowed = budget // (1.75 GB + 268 B/chunk-point + 850 B/fit-point)`, so the target free-byte figure for a
  wanted answer is arithmetic, not luck.  The probe asserts the answer it engineered before it counts
  anything.  I also moved the **live** arm by allocating and touching 24 GB, which walked the same call's
  answer 6 -> 5 -> 4.
* **The idle-worker cost.**  Re-measured, and it does not hold -- see D3.

---

## 2. Claim 1 -- the wedge (task A)

### 2.1 Four drivers, both trees, Windows 3.14.6

`vp2_broken_drivers.py`, each mode an independent way of breaking the pool, none of them sharing code with
the WP's `probe_p7`.  `pool_engaged` is asserted in every row (the probe records the gate decision and the
getter calls, because a probe whose pool silently never engaged measures nothing -- the first version of this
probe hit exactly that: `_script_has_main_guard` refuses the pool for any driver script whose top level does
real work, and `T0 = time.monotonic()` at module scope is enough to trigger it).

| driver | what breaks | PRE `96cb2096` | FIXED |
|---|---|---|---|
| `exit` | worker calls `os._exit(7)` on its first chunk | returned 1.031 s, identical | returned 1.869 s, identical |
| `kill` | parent TerminateProcess/SIGKILLs the live workers mid-flight | returned 1.814 s, identical | returned 1.658 s, identical |
| `unpickle` | work item raises inside the worker's `call_queue.get()` (a `__reduce__` naming a raising callable) -- the worker dies in the queue, not in the task | returned 1.266 s, identical | returned 1.681 s, identical |
| `feederkill` | all workers killed **after** the parent's `QueueFeederThread` is confirmed inside `connection._send_bytes` (the release gate's captured state) | returned 6.036 s, identical | returned 7.067 s, identical |

So on Windows / 3.14.6 **no natural break wedges either tree** -- which is exactly what the report's section
8.3 says, and it is worth being explicit that this makes the Windows fail-before an *engineered* one.

### 2.2 The natural wedge the report could not produce

On POSIX, `_terminate_broken` does `p.terminate()` (SIGTERM) and then an untimed `p.join()`.  A worker that
ignores SIGTERM makes that join genuinely unbounded, with no stub executor anywhere.  Driver `sigign`: the
chunk function installs `SIG_IGN` for SIGTERM and holds; the probe then SIGKILLs exactly **one** worker, so
the pool breaks and CPython tries to terminate the survivors.

* **PRE tree, WSL 3.12.3**: no `dispatch_returned` line ever appeared.  `faulthandler` at 90 s
  (`vp2_sigign_wsl_PRE_dump.txt`):

  ```
  [manager]   process.py:593 _join_executor_internals <- 528 _terminate_broken <- 532 terminate_broken
              -> multiprocessing/process.py:149 join -> popen_fork.py:27 poll
  [main]      process.py:865 shutdown
              <- lumenairy/elements/_lens_traced.py:1409 close_worker_pool
              <- _lens_traced.py:12624 _invert_newton_parallel
              <- _lens_traced.py:13147 apply_real_lens_traced
  ```

  This is the release gate's dump, reproduced without a stub, on the base commit.  Killed at 200 s.
* **FIXED tree, same driver**: `dispatch_returned` with `call_seconds 2.458`, `identical true`,
  `abandoned 1`, `shutdown_timeouts 0`, then `exiting`.  The computation is correct and bounded.
  The **process** then never exits (sec. 7, D6).

### 2.3 Both interpreters read the same way

| fact | 3.14.6 | 3.12.3 |
|---|---|---|
| `terminate_broken` body is `with self.shutdown_lock:` | yes (line 505) | yes (line 530) |
| `_terminate_broken` ends in `_join_executor_internals(broken=True)` | yes | yes |
| that helper calls `call_queue.join_thread()` and `p.join()` | yes (line 553) | yes (line 578) |
| either join takes a timeout | **no** | **no** |
| `shutdown`'s first statement is `with self._shutdown_lock:` | yes (line 859) | yes (line 864) |
| measured block of `shutdown(wait=False, cancel_futures=True)` against a 20.000 s hold | 19.998 s | 20.004 s |
| same call, healthy pool | 0.000 s | 0.000 s |

---

## 3. Claim 2 -- the respawn cause (task B)

`vp4_respawn.py --mode sequence`, clamp answers engineered to the report's own 6, 8, 4, 8 ladder (the
running clamp answered 7, 8, 4, 8 -- the ladder is priced from the call's real fit-grid size, learned from a
warm-up dispatch through a spy, so the first rung lands one worker high).  Counters wrap
`concurrent.futures.ProcessPoolExecutor` (construction and every `shutdown` with its arguments) and
`_newton_resolve_workers`; a 30 ms sampler thread records every child pid the process ever had.

| tree | constructions | rebuild teardowns | `shutdown` arguments | distinct worker pids |
|---|---|---|---|---|
| PRE `96cb2096` | **4** | 3 | `wait=False` x3 | **36** |
| FIXED | **2** | 1 | `wait=False, cancel_futures=True` x1 | **23** |

Both counts include the 8-worker warm-up pool that was alive when sampling started, so the measured window
is 28 spawned interpreters before and 15 after.  Step 2 of the fixed run is the ceiling doing its job: the
clamp answered 4, the live 8-wide pool served it, no rebuild, and the field is byte-identical to the serial
reference.

The live-memory arm, moved by allocation rather than by the budget knob (`--mode memalloc`, fit grid
6 027 025 points so the boundary is reachable without exhausting the box):

| allocated | psutil available | clamp answer |
|---|---|---|
| 0 | 87.03 GB | 6 |
| 8 GB | 78.81 GB | 5 |
| 16 GB | 69.58 GB | 4 |
| 24 GB | 61.87 GB | 4 |
| freed | 86.09 GB | 5 |

The same call site, the same arguments, three different answers -- and note the last row: back at 86 GB it
answers 5, not the 6 it started with.  The report's "6 at one moment and 4 a minute later" is confirmed.

---

## 4. Claim 3 -- the invariants under concurrency (task C) and the bar (task D)

### 4.1 Concurrency

| scenario | fixed tree | PRE tree |
|---|---|---|
| two threads dispatching at once, 6 dispatches each (the two-arm chain shape) | 3.060 s, no hang, 12/12 byte-identical, `_POOL_INFLIGHT` 0 | 8.107 s, no hang, 12/12 byte-identical |
| `close_worker_pool()` every 50 ms against a dispatching thread | 8 closes / 6 dispatches, no hang, 6/6 byte-identical, `_POOL_INFLIGHT` 0 | 11 closes / 6 dispatches, no hang, 6/6 byte-identical |
| a dispatch while a retired pool's reaper is wedged inside `shutdown` | close returned **0.000 s**; reaper thread alive holding `shutdown(wait=False, cancel_futures=True)`; dispatch 0.960 s byte-identical | n/a (the machinery does not exist) |
| 100 consecutive FAILING dispatches | `_POOL_INFLIGHT` ends at **0** (Windows 3.14.6 and WSL 3.12.3) | n/a |

The same three scenarios on WSL 3.12.3 read the same way: two threads 7.957 s with 12/12 byte-identical and
`_POOL_INFLIGHT` 0; 4 closes against 6 dispatches all byte-identical; the wedged-reaper close returned in
**0.015 s** with the reaper thread left holding `shutdown(wait=False, cancel_futures=True)` and the next
dispatch completing in 2.279 s byte-identical.

Two residual windows, both checked and both benign because the fallback is bit-identical:

* the one the source already documents -- a rebuild landing between the getter returning an executor and the
  dispatcher claiming it, so `submit` raises `RuntimeError` and the call takes the serial rung;
* one the source does not: the broken-pool rung calls `close_worker_pool()` unconditionally, so if another
  thread has ALREADY retired the broken pool and built a fresh one, this close tears down that healthy
  replacement and the other thread's dispatch takes its own serial rung.  The close-race scenario above is
  that shape at 50 ms intervals: 6 of 6 fields still byte-identical, no hang.  Worth a sentence in the
  source next to the window that is documented.

### 4.2 The ceiling, from both sides

Driven directly through the getter with a counting fake executor:

| step | asked | pool after | constructed | same object as before? |
|---|---|---|---|---|
| build | 4 | 4 | 1 | -- |
| narrower | 2 | **4** | 1 | yes (never shrinks) |
| same | 4 | 4 | 1 | yes |
| wider, idle | 8 | 8 | 2 | no (grows) |
| wider, **in flight** | 16 | **8** | 2 | **yes** (refused) |
| wider, released | 16 | 16 | 3 | no |
| marked broken | 1 | 1 | 4 | no (replaced unconditionally) |

The same seven steps read identically on WSL 3.12.3 (4 constructions, `_POOL_INFLIGHT` 0).

and from the outside, with a real 16-worker pool and the next dispatch priced down:

| dispatcher's clamp | chunks run | **peak concurrent chunks** | pool width | field |
|---|---|---|---|---|
| 3 | 3 | **3** | 16 | identical |
| 5 | 5 | **5** | 16 | identical |
| 10 | 10 | **10** | 16 | identical |

The clamp is carried by the chunk count, exactly as the design argues.  This is the memory-safety argument
for keeping a wide pool alive and nothing in the WP's own tests measured it; it is now pinned
(`test_a_pool_wider_than_the_clamp_runs_only_clamp_chunks_at_once`).

### 4.3 The bounded-wait bar, and what an expiry costs

Healthy teardown through the public `close_worker_pool` after a real dispatch, quiet box, 3 reps:

| workers | close seconds |
|---|---|
| 1 | *(no pool: the size gate answers `n_cpu <= 1` serially, so there is nothing to close)* |
| 2 | 0.113  0.109  0.169 |
| 4 | 0.146  0.173  0.180 |
| 8 | 0.297  0.169  0.168 |
| 16 | 0.321  0.327  0.379 |

Worst **0.379 s**; the bar is 120 s, i.e. **317x**.  The same ladder on WSL 3.12.3, taken while three other
agents' pytest lanes were resident, reads 2w 0.246-0.638, 4w 0.380-1.098, 8w 0.939-2.240, 16w 1.008-1.834 --
worst **2.240 s**, **53.6x**, i.e. a loaded box moves the ladder by about 6x and still leaves the bar a
decade and a half clear.  The WP's own `probe_p6` re-run today gives worst
0.411 s / 291.9x against its reported 4.592 s / 26x -- the difference is box load, and both readings leave
the bar decades above the healthy side and infinitely below the state it guards.  (The report's 1-worker
rung comes from `_get_persistent_worker_pool` directly; the public path never builds a 1-worker pool.)

**Cost of an expiry**, forced by driving the bar to 0.05 s (`--mode expiry`):

| what | measurement |
|---|---|
| `close_worker_pool` on a healthy-looking pool whose join never returns | returned in 0.054 s, `_POOL_SHUTDOWN_TIMEOUTS` +1, pool handed to `_ABANDONED_POOLS`, `lumenairy-newton-pool-close` thread left holding the join |
| `close_worker_pool` on a real 8-worker pool with the same bar | returned in 0.053 s, `_POOL_SHUTDOWN_TIMEOUTS` +1 |
| worker processes still alive at t+0 / t+1 s / t+5 s / t+30 s | 6 / **0** / 0 / 0 |
| parent RSS before -> after | 366.0 MB -> 365.3 MB |
| `_ABANDONED_POOLS` length afterwards | **2, and it never returns to 0** (defect D2) |

So an expiry costs about a second of overlapping worker lifetime and no parent memory -- except that the
executor object is retained forever, which is D2.

---

## 5. Claim 5 -- the second exposure (task E)

### 5.1 The WSL bootstrap cost does not reproduce

`vp3_worker_bootstrap.py`, same `/mnt/c` mount, same venv, three stages per rung (bare interpreter; import
`lumenairy`; import `lumenairy.elements._lens_traced` + scipy, i.e. what a Newton chunk pays first):

| stage | workers | Windows 3.14.6 total s | WSL 3.12.3 total s | distinct pids (WSL) |
|---|---|---|---|---|
| spawn | 1 / 2 / 4 / 8 | 0.136 / 0.142 / 0.152 / 0.168 | 0.117 / 0.104 / 0.154 / 0.112 | 1 / 1 / 1 / 1 * |
| import lumenairy | 1 / 2 / 4 / 8 | 0.397 / 0.418 / 0.489 / 0.538 | 1.633 / 0.398 / 0.431 / 0.471 | 1 / 2 / 4 / 8 |
| chunk imports | 1 / 2 / 4 / 8 | 0.966 / 1.105 / 1.283 / 1.400 | 0.877 / 0.855 / 0.952 / **0.969** | 1 / 2 / 4 / 8 |

\* the trivial task is fast enough that one worker serves every item before the others are needed.

At 8 workers the WSL first-answer times are 0.945 ... 0.969 s -- a 24 ms spread, i.e. concurrent, not
serialised.  The report's "exactly one at a time, three to six minutes each" is **not reproducible on this
box today**, on the same mount.  Like the fd-capture trigger, the state that produced it is gone.

### 5.2 The exposure itself is real, and can be made deterministic

The gap is in the code, not in the mount: `as_completed` in `_invert_newton_parallel` has no timeout.
Driver `slowboot` installs a pool initializer that blocks, so the workers never come up.  On **both** trees,
Windows 3.14.6, the call never returns, and the 60 s `faulthandler` dump is the report's WSL picture exactly
(`vp2_slowboot_win_dump.txt`):

```
MainThread             concurrent/futures/_base.py:237 as_completed
                       lumenairy/elements/_lens_traced.py:12799 _invert_newton_parallel
QueueFeederThread      multiprocessing/connection.py:303 _send_bytes
Thread-1 (manager)     concurrent/futures/process.py:416 wait_result_broken_or_wakeup
```

**Does it need a timeout?**  Yes, and it is a maintainer-facing default change, so it belongs in its own
work package.  What a bar would have to be derived from, stated so the next owner does not have to
rediscover it:

* the quantity to bound is *time to FIRST result*, not total dispatch time -- a slow box makes every chunk
  slow, but a worker that never comes up produces nothing at all;
* the floor is worker startup, which is measured above: 1.4 s (Windows) / 0.97 s (WSL) for 8 workers to
  reach the chunk's imports on a quiet box, and the WP's own report records 45-56 s for a first
  `apply_real_lens_traced` in a cold process on a loaded one -- so a first-result bar below ~10 minutes
  would be picking a number, not deriving one;
* `TimeoutError` is an `OSError` subclass, so it lands in the existing infrastructure clause and the
  fallback is already bit-identical -- the cost of a false positive is wall time only, which is what makes
  a generous bar the right shape here;
* the alternative with no new default is to bound only the *bootstrap*: fail if no worker has answered
  within a multiple of the measured spawn ladder.

---

## 6. Runs (task G)

All commands were run from `C:\tmp\lum_vpool` with the three thread caps on the command line.

The box was shared throughout (another agent's pytest lanes and the maintainer's long job were resident),
which is why the `--capture=sys` lane took three times the default lane's wall time for the same 364
outcomes -- the counts, not the durations, are the claim.

### 6.1 The thirteen traced-lens / pool files

`pytest tests/unit/test_fix_newton_pool_broken_fallback.py tests/unit/test_fix_newton_pool_memory.py
tests/unit/test_niche_newton_pool_both_fits.py tests/unit/test_carrier_field.py
tests/unit/test_carrier_referenced.py tests/unit/test_audit2609_a3_traced_lens.py
tests/unit/test_audit2609_a3_verify_traced.py tests/unit/test_niche_audit_w9_traced_determinism.py
tests/unit/test_niche_d15_deterministic_traced_fit.py tests/unit/test_hammer_h3_traced_nyquist_guard.py
tests/unit/test_niche_audit_w3_elements.py tests/unit/test_niche_perf_round2_2026_08_10.py
tests/unit/test_niche_audit_e_prepared_and_enums.py -q [--capture=sys]`

| build | capture | result | wall |
|---|---|---|---|
| Windows 3.14.6 | default (`fd`) | **363 passed, 1 skipped** | 300.88 s |
| Windows 3.14.6 | `--capture=sys` | **363 passed, 1 skipped** | 981.38 s |
| WSL 3.12.3 | default (`fd`) | **363 passed, 1 skipped** | 1788.21 s |
| WSL 3.12.3 | `--capture=sys` | **363 passed, 1 skipped** | 1743.32 s |

The one skip is `test_fix_newton_pool_memory.py:1217` on every arm, with the report's own reason (this BLAS
reduces identically at every width tried, so the box cannot witness the defect that test is about).  Four
arms, four identical 363/1 readings: claim 4a **CONFIRMED**, and the capture axis moves nothing but wall
time (WSL 1788.21 s under `fd` against 1743.32 s under `sys`, a 2.5 % difference on a shared box).

### 6.2 The rest of the gate

| command | result | wall |
|---|---|---|
| `pytest tests/unit/test_niche_d8_congruence_workers.py -q` (Windows) | 36 passed | 169.77 s |
| the same with `--capture=sys` (Windows) | 36 passed | 203.14 s |
| `pytest tests/unit/test_niche_d8_congruence_workers.py -q` (WSL) | 36 passed | 186.35 s |
| `pytest test_audit_except_budget.py test_public_api.py test_v4_16_2_dispatcher_pin_doc_consistency.py test_audit2609_a17_history_lint.py test_audit2609_a23_census_mechanism.py test_v4_14_2_dispatcher_pin_cache_locks.py test_eme_census_determinacy.py -q` | 140 passed, 6 skipped | 291.11 s |
| `pytest tests/unit -q -k walker` | 118 passed, 6 skipped, 16082 deselected | 218.50 s |
| `pytest tests/unit/test_audit2609_a15a_durations_staleness.py -q` | 4 passed | 228.34 s |
| `pytest tests/unit/test_verify_b13_newton_pool.py -q` (this file, Windows) | 3 passed | 55.02 s (repeat, `-p no:randomly`, 71.99 s) |
| `pytest tests/unit/test_verify_b13_newton_pool.py -q` (WSL) | 3 passed | 34.96 s (repeat 36.26 s) |
| `wsl ruff check lumenairy/ tests/ validation/probe_verify_b13/` | All checks passed | -- |

The cache-lock walker reports `_ABANDONED_POOLS_LOCK` on its exemption list with its reason, which is the
entry WP-B13 added; the broad-`except` budget is unchanged.

### 6.3 Test durability under an injected regression (task F)

`validation/probe_verify_b13/vp7_wedge_plugin.py` puts the pre-fix behaviour back in memory and the
15-test file is run under it.  The question is not coverage but whether a regression is REPORTED or merely
hangs:

| injection | result | wall | hung? |
|---|---|---|---|
| `VP7_INJECT=joins` (both teardown helpers become an unbounded in-line `shutdown(wait=True)`) | **3 failed, 12 passed** | 118.80 s | no |
| `VP7_INJECT=rebuild` (pre-fix getter: any change of width respawns) | **3 failed, 12 passed** | 13.69 s | no |
| `VP7_INJECT=both` | **6 failed, 9 passed** | 118.69 s | no |

The three wedge tests fail through `_with_deadline`'s daemon-thread deadline and the three rebuild tests
through ordinary assertions; `test_only_the_bounded_helper_ever_joins_an_executor` correctly stays green
under an in-memory injection because it is an AST pin on the source.  Claim 4c is confirmed -- with D1: the
three wedge failures report `io.UnsupportedOperation: fileno` from `_thread_dump`, not their own message,
and the thread dump is lost.

The same demonstration for this verifier's own file, through `VERIFY_B13_PREFIX` (documented in the file):
`VERIFY_B13_PREFIX=joins pytest tests/unit/test_verify_b13_newton_pool.py -k reaper` -> **1 failed** in
124.82 s, the child's own `faulthandler` dump attached, no hang.

---

### 6.4 Probes and evidence

| file | what it measures |
|---|---|
| `vp_run.py` | the harness: launches a probe against a chosen tree under `subprocess.run(timeout=)`, kills the whole process tree on expiry, and separates RETURNED (the library call came back) from EXITED (the process terminated) |
| `vp1_cpython_lock.py` | the `_shutdown_lock` measurement, both arms, both interpreters (`vp1_{win,wsl}_{healthy,broken}.json`) |
| `vp2_broken_drivers.py` | the five break drivers plus a control (`vp2_win.jsonl`, `vp2_wsl.jsonl`); thread dumps in `vp2_sigign_wsl_PRE_dump.txt`, `vp2_sigign_wsl_FIXED_dump.txt`, `vp2_slowboot_win_dump.txt` |
| `vp3_worker_bootstrap.py` | the worker-startup ladder (`vp3_win.json`, `vp3_wsl_mntc.json`) |
| `vp4_respawn.py` | the construction/teardown counts with the clamp engineered (`vp4_{fixed,pre}_win.json`) and the live-memory arm (`vp4_memalloc_win.json`) |
| `vp5_concurrency.py` | the six concurrency scenarios (`vp5_{fixed,pre}_{win,wsl}_*.json`) |
| `vp6_teardown_bar.py` | the healthy ladder and the expiry cost (`vp6_ladder_{win,wsl}.json`, `vp6_expiry_win.json`, `vp6_bigchunk_win.json`) plus a re-run of the WP's own probe (`vp6_wp_p6_rerun.json`) |
| `vp7_wedge_plugin.py` | the pytest plugin that puts the pre-fix behaviour back for the durability runs |
| `vp8_wp_p8_rerun.json` | the WP's own 12-field identity probe, re-run here |

---

## 7. Defects

| id | severity | where | what | reproducer |
|---|---|---|---|---|
| D1 | P2 (test quality) | `tests/unit/test_fix_newton_pool_broken_fallback.py:137` | `_thread_dump()` calls `faulthandler.dump_traceback(file=io.StringIO())`, which raises `io.UnsupportedOperation: fileno`.  The helper is only ever called from `_with_deadline`'s failure path, so **every** wedge detection in the file errors with an unrelated exception instead of its own message, and the thread dump -- the single most useful artifact for this defect -- is never produced.  The tests still go red, so the gate works; the diagnostic does not | `VERIFY_B13_PREFIX=joins` (or `validation/probe_verify_b13/vp7_wedge_plugin.py` with `VP7_INJECT=joins`) then run the file: 3 failures, each ending in `io.UnsupportedOperation: fileno` |
| D2 | P3 (resource) | `lumenairy/elements/_lens_traced.py::_shutdown_pool_bounded` | the expiry path appends the executor to `_ABANDONED_POOLS` and **nothing ever removes it** -- unlike `_abandon_pool`, whose reaper removes in a `finally`.  The list grows monotonically with expiries and pins each dead executor (its `_processes`, its queues) for the life of the process | `vp6_teardown_bar.py --mode expiry --bar 0.05` leaves `_ABANDONED_POOLS` at length 2 after both shutdowns have completed.  The one-screen contrast: a stub whose join is released AFTER a 0.2 s expiry stays in the list (`still_in_list True`), while a pool sent through `_abandon_pool` is removed (`s2_in_list False`) |
| D3 | P3 (derivation) | `_lens_traced.py` (`_get_persistent_worker_pool` docstring, "33.0 MB mean / 48.8 MB peak ... 2 % of one working worker") and report sec. 5.2 | the figure is measured on workers warmed with `ex.map(abs, ...)`.  The state the ceiling rule actually leaves behind is a worker that has served a **Newton chunk**: 97.0 MB mean / 98.8 MB max (N=512, 15 workers) and 99.4 / 99.9 MB (N=1024).  The WP's own probe re-run today reads 45.7 MiB mean, not 33.0.  A 16-wide kept pool holds ~1.6 GB, not ~0.5 GB.  The cleanest evidence is a single WSL run in which both states are present at once: workers that served a chunk read 73-76 MB and workers in the same pool that served none read 28-39 MB -- the quoted figure is the never-used state.  The trade is still cheap; the number is not the number | `vp6_teardown_bar.py --mode ladder` vs `validation/probe_newton_pool/probe_p6_teardown_ladder.py` |
| D4 | P3 (comment drift) | `_lens_traced.py` `_POOL_INFLIGHT` comment ("Chunks currently dispatched") and report sec. 5.2 ("counts chunks dispatched on the cached pool") | the counter is moved **once per dispatch**, not once per chunk.  Nothing depends on the magnitude -- only zero vs non-zero is read -- so this is a comment defect, not a behaviour one | read `_note_pool_inflight` call sites |
| D7 | P3 (pin blind spot) | `tests/unit/test_fix_newton_pool_broken_fallback.py::test_only_the_bounded_helper_ever_joins_an_executor` | the pin walks for `ast.Call` nodes whose `func.attr == 'shutdown'`.  A joining teardown written as `with ProcessPoolExecutor(...) as ex:` has no such call -- `Executor.__exit__` IS `shutdown(wait=True)` -- so the pin would stay green on exactly the shape that carries the same exposure in `carrier.py::_multi_parallel_results`.  `Executor.shutdown(ex)` (an unbound call) is the same blind spot | read the pin; the sibling module is the live example |
| D5 | P2 (pre-existing, open) | `_lens_traced.py:12799` | `as_completed` with no timeout: a worker that never comes up hangs the call forever, on both trees | `vp2_broken_drivers.py --mode slowboot` |
| D6 | P2 (pre-existing, acknowledged) | CPython / `_lens_traced.py` | once the manager thread is wedged the process cannot exit even on the fixed tree.  Measured: the call returned its correct answer in 2.458 s and printed its exit line; the process then had to be SIGKILLed at 200 s, leaving 5 processes for the sweep | `vp2_broken_drivers.py --mode sigign` on WSL, fixed tree |

None of D1-D4 or D7 is a correctness defect in the shipped field: every pooled and every fallback field I produced
is byte-identical to its serial reference.

### Requested changes outside my ownership

1. `tests/unit/test_fix_newton_pool_broken_fallback.py`, replace `_thread_dump`:

   ```python
   def _thread_dump() -> str:
       with tempfile.TemporaryFile('w+') as fh:      # faulthandler needs a real fd
           faulthandler.dump_traceback(file=fh, all_threads=True)
           fh.seek(0)
           return fh.read()
   ```

2. `lumenairy/elements/_lens_traced.py::_shutdown_pool_bounded`, in the helper thread's `finally`, drop the
   executor from `_ABANDONED_POOLS` the way `_abandon_pool._reap` does:

   ```python
       finally:
           with _ABANDONED_POOLS_LOCK:
               try:
                   _ABANDONED_POOLS.remove(ex)
               except ValueError:
                   pass
           done.set()
   ```

3. `tests/unit/test_fix_newton_pool_broken_fallback.py`, extend the AST pin to the `with` form (D7): after
   the `ast.Call` sweep, walk `ast.With` / `ast.AsyncWith` and flag any item whose `context_expr` is a call
   to a name ending in `Executor`, unless it is inside `_shutdown_pool_bounded`.  Without it the pin cannot
   see the shape the sibling module uses.

4. `_get_persistent_worker_pool`'s docstring: re-state the idle-worker footprint from workers that have
   served a Newton chunk (97.0 MB mean / 99.9 MB max, 2026-09-15, this box), and say which state it is
   measured in.  Fix `_POOL_INFLIGHT`'s comment to say "dispatches", not "chunks".

---

## 8. What I could not verify

1. **Why the 2026-09-14 breakage happened.**  Same answer as the report's: the trigger state is gone.  I add
   only that four independent natural break drivers and a confirmed feeder-blocked kill all fail to wedge
   Windows 3.14.6 on either tree, so whatever broke the pool that day is still unidentified.
2. **That the spawn storm of section 3.2 was the cause.**  Confirmed as a mechanism and as a pressure
   (28 -> 15 spawned interpreters), not as the cause on the day.
3. **Why the WSL workers bootstrapped one at a time on 2026-09-14.**  Not reproducible (sec. 5.1), so the
   report's inference about the 9p mount can be neither confirmed nor refuted here.
4. **The Windows `--capture=sys` `w9_dispatch2` red.**  Out of scope for the pool; I did not re-run the MFT
   oracle ladder.
5. **`carrier.py::_multi_parallel_results`.**  I confirmed by reading that it still runs its executor as
   `with ProcessPoolExecutor(...)`, i.e. `shutdown(wait=True)` on exit, and therefore carries the identical
   exposure; I did not build a driver for it (it is a different pool with a different failure policy, and the
   report scopes it out).  `test_niche_d8_congruence_workers.py` was run as a gate, not as a wedge test.

---

## 9. Ship recommendation

**SHIP.**  The change does what it says, it does it on both interpreters, and the thing it is supposed to
prevent is now demonstrable on the base commit and absent on the branch:

* the mechanism is measured, not inferred -- a non-waiting cancelling `shutdown` blocks for exactly as long
  as the manager thread holds the lock, on 3.14.6 and on 3.12.3, so the handoff's one-line fix would not
  have worked and the two-mechanism teardown is the right shape;
* the fail-before is now a NATURAL one (`sigign` on WSL): base commit wedges forever in
  `close_worker_pool -> shutdown` with the dump the release gate captured; the branch returns the serial
  answer byte for byte in 2.458 s;
* nine independent broken-pool drivers across two builds, plus a close racing a dispatch, plus two threads
  dispatching at once, plus a dispatch issued while a retired pool's reaper is permanently wedged: every one
  returns, and every field equals its `n_workers=1` reference with `max|delta| = 0.0`;
* the ceiling rule's safety argument -- the clamp is carried by the chunk count -- is measured for the first
  time here and holds exactly (peak concurrent chunks 3 / 5 / 10 against clamp 3 / 5 / 10 on a 16-wide pool);
* the gate is green on both builds under both captures, and the new test file fails rather than hangs when
  the pre-fix behaviour is injected under it.

**Ship with these four follow-ups**, none of which blocks:

1. **D1 before the next release** (one-line, test-only): `_thread_dump` must write to a real file
   descriptor.  Today, the moment one of the three wedge tests actually fires it loses the thread dump and
   reports an unrelated `io.UnsupportedOperation`.  That is the one place in this work package where the
   diagnostic quality of the gate is worse than it reads.
2. **D2** (one-line, library): `_shutdown_pool_bounded` must drop the executor from `_ABANDONED_POOLS` when
   its helper thread finishes, the way `_abandon_pool` already does.
3. **D3/D4** (comments): re-derive the idle-worker footprint in the state the ceiling rule actually leaves
   behind (97.0 MB mean / 99.9 MB max here, not 33.0 MB), and say `dispatches` where `_POOL_INFLIGHT`'s
   comment says `chunks`.
4. **D5 as its own work package**: `as_completed` with no timeout is a live P1-shaped exposure that this
   change reduces but does not remove, and it is now reproducible on demand on both builds
   (`vp2_broken_drivers.py --mode slowboot`).  Section 5.2 states what a bar would have to be derived from.
   `carrier.py::_multi_parallel_results` carries the identical unbounded-join exposure through its
   `with ProcessPoolExecutor(...)` block and should be taken in the same package.

**What ships that is still broken, and should be said out loud in the release note**: on a wedged pool the
computation is saved but the PROCESS is not.  Measured here: the traced call returned its correct answer in
2.458 s, printed its exit line, and then never terminated -- SIGKILL at 200 s, leaving four worker processes
that had to be swept by hand.  A batch script that hits this gets the right numbers and then hangs at exit.
The report says so; it is worth the release note saying so too, because "the fix" reads like the hang is
gone.

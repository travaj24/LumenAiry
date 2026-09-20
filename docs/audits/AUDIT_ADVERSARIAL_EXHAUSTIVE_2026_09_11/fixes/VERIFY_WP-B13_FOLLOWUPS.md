# VERIFY WP-B13 FOLLOW-UPS -- independent adversarial verification

Subject: branch `fix/wp-b13-followups` (`bbeea7a2` D1, `806e92fb` D2, `a8fb2b4c` D3, `5730517e` D4,
`766103d1` D7, `5db3f09d` report + probes + CHANGELOG), base `b631ce79`; report
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B13_FOLLOWUPS_REPORT.md`.

Verifier's worktree `C:\tmp\lum_vpool2`, branch `verify/wp-b13-followups`.  **Nothing under
`lumenairy/` was edited.**  Every number below is my own re-measurement with my own drivers
(`validation/probe_verify_b13_followups/`), on two builds -- **Windows python 3.14.6 / numpy 2.4.4 /
scipy 1.17.1** and **WSL python 3.12.3 / numpy 2.4.6** -- each probe printing `lumenairy.__file__`
as its first line with the tree pinned through `PYTHONPATH`, and every command carrying
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` on the command line.

**Box load, stated first because it bounds every wall time in section 6.**  The box was shared
throughout with three other agents' lanes (a `lum_mb3` pytest lane, an `r3fine.py` sweep with six
`r3scan.py` children, and two `probe_verify_wave5_e` probes).  `psutil.cpu_percent(interval=1)` read
**98.4-100 %** on Windows and **94.5-100 %** inside WSL at the start of every timing probe, with
58-72 GB of 128 GB free.  I could not obtain a quiet box: those processes are other agents' and I
left them alone.  **Every wall time I report is therefore an upper bound on a quiet box, and I say
where that changes a conclusion.**  The COUNTS, the byte-identity results and the hang/no-hang
decisions are unaffected.

---

## 0. Verdict table

| id | the branch's claim | my verdict | my numbers (Windows / WSL) |
|---|---|---|---|
| D1 | `_thread_dump` writes through a real descriptor | **CONFIRMED** | my own wedge (a different frame, parked on `queue.Queue.get` not `Event.wait`): the failure carries `_with_deadline`'s own message AND `_vf1_parked_on_a_queue_get`, both builds |
| D1 | an engineered wedge's message carries the wedged frame's name | **CONFIRMED** | `carries_with_deadline_message: true`, `names_the_parked_frame: true`, both builds |
| D1 | the `io.StringIO` premise is measured in-test | **CONFIRMED** | the pin measures it and REPORTS it; my own measurement: `io.UnsupportedOperation: fileno` on 3.14.6 and on 3.12.3 |
| D1 | fail-before with the shipped helper = `io.UnsupportedOperation: fileno` | **CONFIRMED** | shipped helper re-injected into my own wedge: message lost, `UnsupportedOperation` raised, both builds.  Mutation `m1` on the real suite: **5 failed**, four of them the D1 pins |
| D2 | `_ABANDONED_POOLS` is a census the helper drops its own entry from | **CONFIRMED** | plain expiry, 3 driven: `pre` ends 3 pinned, shipped ends 0, both builds.  Real library function under an adverse lock: 0 leaks in **8/8** reps, both orders, both builds |
| D2 | the requested one-liner has a race | **CONFIRMED, at leak rate 1.0** | one-liner under a helper-first census lock: **8/8 leaked** on both builds; under caller-first **0/8**.  The shipped ordering: **0/8 in both orders** |
| D2 | the shipped ordering publishes `done` before the lock; the caller re-reads it | **CONFIRMED** | helper-first: the caller returns **True** and never appends, `_POOL_SHUTDOWN_TIMEOUTS` unchanged.  caller-first: returns **False**, counter +1, helper removes |
| D2 | a helper-first census lock produces the adverse order by construction | **CONFIRMED** | and so does its mirror, which the branch does not pin -- I added that test |
| D2 | on real 12-worker pools every rung ends with the census empty | **CONFIRMED** | `abandoned_pools_len: 0`, `shutdown_timeouts: 0` on every rung of my footprint ladder, both builds |
| D2 | the census cannot leak or double-remove | **CONFIRMED for leaks; REFUTED for one removal case** | 0 leaks under: both lock orders, a helper raising three classes it does not catch (incl. `BaseException`), two abandons racing, and an 8-point scan across the expiry boundary.  BUT the helper's `remove` is **unconditional** and drops a foreign entry **6/6** by construction -- defect **VD2** |
| D3 | served 98.7-101.6 MB (Win) / 73.2-76.9 MB (WSL) | **CONFIRMED to within 5 %** | my labelling (a socket rendezvous, no Manager): served max **93.6 / 96.45 MB** (Win, N=256/1024), **69.46 / 72.99 MB** (WSL) |
| D3 | never-served 52.2 / 39.2 MB | **CONFIRMED to within 5 %** | **49.6 / 49.5 MB** (Win), **37.27 / 37.29 MB** (WSL) |
| D3 | a 16-wide kept pool is ~1.6 GB, 6 % of an active worker | **CONFIRMED (1.5 GB, 5.5 %)** | 16 x 96.45 MB = **1.51 GB** = **5.5 %** of 1.7 GB (Win); 1.14 GB / 4.2 % (WSL) |
| D3 | the never-served state is "what the SURPLUS of a pool wider than the clamp is" | **REFUTED** -- defect **VD5** | `ProcessPoolExecutor` spawns LAZILY: a 12-wide pool holds **0** processes until submitted to, and **4** after a 4-chunk dispatch, on every rung of both builds.  The surplus worker does not exist |
| D4 | the counter is zero-vs-non-zero to its only library reader | **CONFIRMED** | grep + AST over all of `lumenairy/`: 3 Name loads -- `:1486` compare-against-zero (the only consumer), `:1399` the counter's own update, `:1400` `return _POOL_INFLIGHT`.  1 claim, 3 releases, every call site an expression statement |
| D4 | every comparison is against zero (AST) | **CONFIRMED, and the pin is two-sided-wrong at the edges** -- defect **VD7/VD8** | the branch's own check, driven on 5 shapes: it MISSES the magnitude leaving via `_note_pool_inflight`'s return value, and FALSE-FAILS on the legal `0 < _POOL_INFLIGHT` |
| D7 | the extended pin sees `.shutdown(...)` with positional `wait` and `with <X>Executor(...)`, scoped to process pools | **CONFIRMED** | 20-shape corpus, identical on both builds: extended **12 TP / 0 FP / 2 FN / 6 TN**; shipped **6 / 2 / 8 / 4** |
| D7 | the premise gate asserts `carrier._multi_parallel_results:11553` is detected today | **CONFIRMED** | extended detector reports `_multi_parallel_results:11553:with ProcessPoolExecutor`; shipped reports `[]`; `_lens_traced` clean (+1 informational `apply_real_lens_traced:10731:with ThreadPoolExecutor`) |
| D7 | the shipped detector false-positives on `ex.shutdown(False)` | **CONFIRMED** | and on `ProcessPoolExecutor.shutdown(ex, False)`: 2 FP out of 2 such shapes |
| D7 | (implied) the extended detector is complete | **REFUTED** -- defect **VD6** | 2 FN: `with PPE(...)` where the class was import-aliased, and `with ex:` on a pre-built executor |
| D5 | `as_completed(fs, timeout=T)` bounds the WHOLE iteration | **CONFIRMED** | 4 futures 0.4 s apart, `timeout=1.0`: **2** yielded, then `TimeoutError` at 1.013 s (Win) / 1.008 s (WSL) |
| D5 | "`TimeoutError` is an `OSError` subclass, so the existing clause catches it" | **CONFIRMED on 3.11+, FALSE inside `requires-python`** -- defect **VD3** | driven through the real dispatcher: builtin MRO -> falls back to serial, byte-identical; the **pre-3.11 `concurrent.futures.TimeoutError` MRO -> ESCAPES** to the caller.  `pyproject.toml` says `requires-python = ">=3.10"` |
| D5 | candidate B ("bound the BOOTSTRAP") | **its bar is sound; its coverage claim needs a caveat** -- defect **VD4** | a pool whose first submit ANSWERS and whose later chunks never complete leaves the dispatch running past a 20 s deadline on both builds.  The sentinel narrows the exposure; it does not close it |
| D5 | the healthy-side ladder | **CONFIRMED in shape, 3-17x slower on my loaded box** | see section 6 |
| D6 | the five-arm exit-hang table | **REPRODUCED with my own wedge** | control exits rc 0; wedge alone and wedge+prototype both **SIGKILLed at the deadline**; workers killed -> **rc 0** |
| D6 | the stacks (OFF in `threading._shutdown -> _python_exit -> join`; ON one layer down in `multiprocessing.util._exit_function -> Process.join`) | **CONFIRMED, frame for frame** | see section 7 |
| D6 | daemonising the reaper is not sufficient; only killing the workers closes the hang | **CONFIRMED** | all three prototype steps confirmed applied and the process still had to be killed |
| -- | attribution | **RECORDED, not amended** | all six commits carry `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`, not the campaign's line |

**No field moved anywhere in this verification.**  Every pooled field produced by every probe and
every child here equals its `n_workers=1` reference under `np.array_equal`, `max|delta| = 0.0`:
the D3 ladder (4 rungs), the D5 ladder (12 rungs), the D5 clause-reach arms (3 that fall back), and
all five D6 arms.

---

## 1. D1 -- the wedge report, with a wedge the branch never produces

`vf1_d1_dump.py`.  The branch's pin parks a daemon thread in `_b13_wedged_frame_for_the_dump` on a
`threading.Event`.  A dump helper that happened to work only for that shape would pass it, so my
wedge uses a different frame name AND a different park mechanism -- a blocking `queue.Queue.get()`.

| arm | Windows 3.14.6 | WSL 3.12.3 |
|---|---|---|
| premise: `faulthandler.dump_traceback(file=io.StringIO())` | raises `UnsupportedOperation: fileno` | raises `UnsupportedOperation: fileno` |
| helper called directly, with a second thread parked | 1041 chars, **2** thread headers, names the OTHER thread's frame | 867 chars, **2** headers, names it |
| branch helper, through `_with_deadline` | carries its own message, `Thread dump:`, and `_vf1_parked_on_a_queue_get` | same |
| shipped `io.StringIO` helper re-injected | message LOST, `io.UnsupportedOperation: fileno` | same |

The branch's decision to MEASURE the `io.StringIO` premise in the test and report it in the
assertion message rather than assert it is the right shape, and it is what
`docs/TESTING_STANDARDS.md` rule 5 asks for: a CPython that grew a StringIO path would make the fix
redundant, not the pin red.

Mutation `m1_dump_stringio` against the real suite caught it with **4 D1 pins red** (the branch's
two and my two).

## 2. D2 -- the census, attacked

`vf2_d2_census.py` drives THREE orderings against one census and one interleaver: `pre` (base
`b631ce79`), `oneline` (the repair VERIFY-WP-B13 asked for), `shipped` (this branch) -- as copies,
so all three sit under the same interleaver, AND the real
`lumenairy.elements._lens_traced._shutdown_pool_bounded`, so a copy is never the only evidence.
The adverse order is produced by construction: a census lock that admits one named thread first and
parks the other until it has left its critical section, in BOTH directions.

| arm (8 reps) | `pre` | `oneline` | `shipped` | real library |
|---|---|---|---|---|
| helper-first lock order | 8 leaked | **8 leaked** | 0 | 0, returns `True` |
| caller-first lock order | 8 leaked | 0 | 0 | 0, returns `False`, counter +6 |

Identical on Windows 3.14.6 and WSL 3.12.3.  **So the one-liner is not merely different from what
shipped; it leaks in 100 % of the adverse order.**  The branch's claim is confirmed at a leak rate
of 1.0, not at a sampled rate -- which is the point of building the order rather than sleeping for
it.

Attacks on the shipped ordering, all clean on both builds:

| attack | result |
|---|---|
| helper raises a class it does NOT catch (`MemoryError`, `KeyboardInterrupt`, a `BaseException` subclass) | census returns to empty, 3 reps each.  The `finally` is load-bearing and it holds |
| `_abandon_pool` and a bounded teardown racing on the SAME executor | 0 leaks, 0 premature removals, 6 reps |
| the caller's wait expiring exactly at the append (8-point scan across the bound) | 0 leaks; returns `{True: 7, False: 1}` (Win), `{True: 6, False: 2}` (WSL) -- both orders occur, both end clean |

**One case is NOT clean: see defect VD2.**  The helper's `_ABANDONED_POOLS.remove(ex)` runs whether
or not the caller ever appended.  If the executor is in the census because `_abandon_pool` put it
there, a bounded teardown of the same object that COMPLETES INSIDE ITS BOUND removes the reaper's
entry: **6/6 by construction on both builds**.  No library path hands one executor to both
mechanisms today, so it is latent; I pinned that precondition rather than leave it as a reading.

## 3. D3 -- the footprint, and what the surplus really is

`vf3_d3_footprint.py`.  Same experiment shape as the branch's (a 12-wide pool serving a 4-worker
dispatch, resident sets read parent-side with `psutil` against `_processes`), **different labelling
mechanism**: a listening socket the parent owns, so no `Manager` process sits inside the
measurement and "every worker was labelled" is the parent's own accept count.
`label_coverage_complete: true` on every rung of both builds.

| build | N | served (mean / max) | never-served (mean / max) | processes after construct | after dispatch |
|---|---|---|---|---|---|
| Windows 3.14.6 | 256 | 93.53 / **93.60** MB | 49.60 / 49.78 MB | **0** | **4** |
| | 1024 | 95.91 / **96.45** MB | 49.52 / 49.74 MB | **0** | **4** |
| WSL 3.12.3 | 256 | 69.45 / **69.46** MB | 37.27 / 37.27 MB | **0** | **4** |
| | 1024 | 72.99 / **72.99** MB | 37.29 / 37.31 MB | **0** | **4** |

Every rung `identical_to_serial: true`, `max|delta| 0.0`, `abandoned_pools_len: 0`,
`shutdown_timeouts: 0`.

The branch's numbers (98.7-101.6 / 52.2 Win, 73.2-76.9 / 39.2 WSL) are **4-7 % above mine**, in the
same direction on both builds and in both states; the difference is the reading moment (the branch
reads after labelling every worker, I report both before and after, and after-labelling is what the
table above shows) and the box.  The CONCLUSIONS are unchanged: the worst kept worker is ~96 MB not
33.0 MB, a 16-wide served pool is ~1.5 GB not 0.5 GB, and the fraction of a 1.7 GB active worker is
**5.5 %** not 2 %.  I would restate the docstring's "~102 MB" as "~95-102 MB, measured 2026-09-15 and
re-measured 2026-09-19" rather than leave a single box's peak standing as the number.

**The mechanism in the docstring's second bullet is wrong (VD5).**  It says the never-served state
"is what the SURPLUS of a pool wider than the clamp is".  CPython's `ProcessPoolExecutor` spawns
lazily -- `_adjust_process_count` runs inside `submit` -- so a pool constructed at width 12 holds
**zero** worker processes, and a 4-chunk dispatch on it leaves **four**.  The eight "never-served"
workers in the branch's own table exist only because its labelling step submitted to them.  The
surplus of a wider pool costs nothing until it is used, and the state that actually persists under
the ceiling rule is a worker that SERVED a chunk -- which is the figure the rule should be priced
on, and is.  The trade is if anything cheaper than the docstring says.

## 4. D4 -- the counter's readers, enumerated two ways

`vf6_d4_readers.py`, grep + AST over all of `lumenairy/`, `tests/` and `validation/`; byte-identical
output on both builds.

| where | shape | is it a magnitude read? |
|---|---|---|
| `_lens_traced.py:1486` `elif _POOL_INFLIGHT > 0:` | compare-against-zero | no -- the only library CONSUMER, exactly as claimed |
| `_lens_traced.py:1399` `_POOL_INFLIGHT = max(0, _POOL_INFLIGHT + int(delta))` | arithmetic | no -- the counter's own update |
| `_lens_traced.py:1400` `return _POOL_INFLIGHT` | returned | **the magnitude leaves the module here** |
| `test_fix_newton_pool_broken_fallback.py:650` | compare-against-zero | no |
| `validation/probe_verify_b13/vp5_concurrency.py:294` | assigned, then asserted `== 0` | no |
| `validation/probe_wp_b13_followups/fu4_exit_hang.py:221` | reported into JSON | no |

`_note_pool_inflight` is called 4 times (1 claim, 3 releases) and **every call site is an expression
statement**, so nothing consumes the returned magnitude today.  The D4 decision -- fix the comment,
not the counter -- is therefore correct.  But the pin the branch built to keep it correct does not
cover the route the magnitude actually takes out of the module.  Driven on five shapes, the branch's
own check:

| shape | should | branch pin says |
|---|---|---|
| `_POOL_INFLIGHT > 0` | PASS | PASS |
| `_POOL_INFLIGHT == 0` | PASS | PASS |
| `0 < _POOL_INFLIGHT` (the same decision, operands swapped) | PASS | **FAIL** (VD8) |
| `_POOL_INFLIGHT > 1` | FAIL | FAIL |
| `n = _note_pool_inflight(0); if n > 1:` | FAIL | **PASS** (VD7) |

## 5. D7 -- the detector, graded on a corpus I own

`vf4_d7_detector_corpus.py` runs the base-commit detector (transcribed verbatim from
`b631ce79`) and the branch's `_unbounded_executor_joins` side by side over 20 shapes, each carrying
its own expected verdict derived from the semantics of the construct.  Identical on both builds.

| detector | TP | FP | FN | TN |
|---|---|---|---|---|
| shipped (`b631ce79`) | 6 | **2** | **8** | 4 |
| extended (this branch) | **12** | **0** | 2 | 6 |

The shipped detector's 2 false positives are `ex.shutdown(False)` and
`ProcessPoolExecutor.shutdown(ex, False)` -- confirmed, exactly as the report says.  Its 8 misses are
every `with` shape.  The extended detector fixes all of that and adds no false positive, including
on the thread-pool shapes, which it correctly routes to `informational`.

Its two remaining misses (**VD6**) are `with PPE(n)` where the class was imported under an alias
that does not end in `Executor`, and `with ex:` on a pre-built executor name (the with-item is an
`ast.Name`, so no class name is visible).  Both are `shutdown(wait=True)` on exit, i.e. the exposure
D7 exists to catch.  I did not ask for the detector to be widened -- I swept the package for those
two shapes instead and pinned that it contains none, with a positive control.

On the real modules, both builds:

| module | shipped | extended |
|---|---|---|
| `lumenairy.elements._lens_traced` | `[]` | `[]` (+1 informational `apply_real_lens_traced:10731:with ThreadPoolExecutor`) |
| `lumenairy.propagators.carrier` | **`[]`** | **`_multi_parallel_results:11553:with ProcessPoolExecutor`** |

## 6. D5 -- the healthy side, re-measured, and the recommendation challenged

`vf5_d5_timing.py`.  Independent of the branch's `fu3` in its primary instrument: chunk ARRIVALS
come from the dispatcher's own `sub_progress` callback (`newton chunk k/n done`, emitted inside the
`as_completed` loop), a consumer-side channel that needs nothing wrapped.  The executor proxy
supplies submit timestamps only and is reported as a cross-check.  `progress_channel_live: true`
(8 arrivals for 8 chunks) on every rung; `pool_engaged: true`, `identical_to_serial: true`,
`max|delta| 0.0` on every rung.

**Load first (see the header): 98.4-100 % CPU on Windows, 94.5-100 % inside WSL.**

| build | N | cold first arrival | cold total | warm first | warm total |
|---|---|---|---|---|---|
| Windows 3.14.6 | 256 | 9.898 | **13.715** | 0.031 | 0.070 |
| | 512 | 5.475 | 6.344 | 0.118 | 0.337 |
| | 1024 | 7.139 | 11.526 | 0.988 | **2.664** |
| WSL 3.12.3 | 256 | 2.881 | 3.422 | 0.064 | 0.124 |
| | 512 | 3.553 | 5.011 | 0.269 | 0.632 |
| | 1024 | 4.935 | **11.696** | 1.082 | **3.060** |

(seconds; "cold" = `close_worker_pool()` first.  Worker-count preconditions forced, size bars
lowered to 1 -- `docs/TESTING_STANDARDS.md` rule 4 -- and the chunk count asserted.)

Spawn bootstrap (construct -> every worker has imported `scipy.interpolate` and
`lumenairy.elements._lens_traced`, rendezvoused on a **socket**, no Manager):

| build | construct | all 8 ready | ready spread | distinct pids |
|---|---|---|---|---|
| Windows 3.14.6 | 0.005 s | **20.357 s** | 10.073 s | 8 |
| WSL 3.12.3 | 0.118 s | **6.996 s** | 1.826 s | 8 |

Sentinel round trip: Windows **2.375 s** cold / **0.0017 s** warm-worst; WSL **1.280 s** cold /
**0.0642 s** warm-worst.

**How this compares with the branch's ladder, and why it matters.**  Every one of my readings is
**3-17x** the branch's on the same box: cold total at N=256 13.715 s against 1.069 s, bootstrap
20.357 s against 1.179 s, warm sentinel 0.0642 s (WSL) against 0.0004 s.  I also measured the WSL
bootstrap at **3.223 s** and then at **6.996 s** ten minutes apart on the same box with the same
probe.  Nothing here refutes the branch's numbers -- it re-measures the same quantities under a
heavier load and gets the same ORDERING -- but it does refute any bar derived from them.

### 6.1 The recommendation, challenged

**The CPython fact is confirmed and it is the load-bearing part.**  `timeout=` on `as_completed` is
an absolute deadline taken when the generator is created: 4 futures resolving 0.4 s apart under
`timeout=1.0` yielded **2** and then raised at 1.013 s (Win) / 1.008 s (WSL).  So a single
`timeout=` bounds the total dispatch, which scales with the field while the pathology does not.
Candidate A is correctly rejected, and my numbers strengthen the rejection rather than weaken it:
the report's own derived candidate-A bar (`10x` the worst cold chunk plus the bootstrap = 20.5 s
Windows) is **below my measured bootstrap alone** (20.357 s) and below my worst cold total at
N = 256 (13.715 s plus a 20.4 s bootstrap).  A bar derived from one box's ladder would abort
legitimate *small* dispatches on a loaded box, which is the opposite of the failure it was sized
against.  That is exactly the S1/S4 shape `docs/TESTING_STANDARDS.md` forbids, and the report's
refusal to pick a number is right.

**Candidate B's bar survives my numbers.**  600 s is 29x my worst measured bootstrap (20.357 s),
253x my worst cold sentinel (2.375 s), and 9 300x my worst warm sentinel -- with the state on the
other side unbounded, so the gap above is infinite.  I would keep it.

**Does the sentinel add a dispatch on the healthy path?  Yes, and the cost is real but small.**  One
extra round trip per dispatch: 0.0017 s (Win) / 0.0642 s (WSL) warm, against warm totals of 2.664 s
and 3.060 s at N = 1024, i.e. **0.06 %** and **2.1 %**.  The WSL figure is 91x the branch's 0.0004 s
and is a loaded-box reading; it is still small, but "0.05 % and below the noise of every rung" is a
quiet-box statement and should be written as a range.  On a cold pool it is not an extra cost: the
sentinel pays the spawn the first chunk would have paid.

**What happens when the sentinel passes and a LATER chunk wedges: the exposure is NOT closed, and
the report should say so (VD4).**  Driven, both builds: a pool whose first submit answers
immediately and whose later submits never complete leaves the dispatch **still running at a 20 s
deadline** (`still_running_at_deadline: true`, `sentinel_answered: true`, 4 submits).  The
unbounded `as_completed` is still behind the sentinel.  Candidate B closes the `slowboot` shape
exactly -- a pool that never comes up -- and narrows nothing else.  Section 6.1 of the follow-ups
report implies this by rejecting candidate C "for the same reason", but the recommendation itself
reads as if the exposure is bounded, and a release note written from it would be wrong.  **One
sentence is needed: "this bounds a pool that never comes up; a worker that answers once and then
stops answering is still unbounded."**

**Is `TimeoutError` really caught by the existing infrastructure clause?  On 3.11+ yes; inside the
declared `requires-python` no (VD3).**  I read the tuple -- `except (BrokenProcessPool, RuntimeError,
OSError, EOFError)` at `_lens_traced.py:12943` -- and then drove it, through the real dispatcher,
with a stub pool raising each class:

| raised | MRO | outcome |
|---|---|---|
| builtin `TimeoutError` | `TimeoutError -> OSError -> ...` | falls back to serial, `identical: true` |
| `concurrent.futures.TimeoutError` on this build | identical (it IS the builtin on 3.11+) | falls back, `identical: true` |
| **pre-3.11 `concurrent.futures.TimeoutError` MRO** (`Error(Exception)`) | `-> Exception` | **ESCAPES to the caller** |
| `OSError` (control) | | falls back, `identical: true` |
| `ValueError` (control) | | escapes, correctly |

`concurrent.futures.TimeoutError` became an alias of the builtin only in Python 3.11 (gh-90315);
`pyproject.toml` declares `requires-python = ">=3.10"`.  I could not run a 3.10 interpreter -- none
is installed here -- so the 3.10 arm is a reconstruction of the class hierarchy driven through the
real dispatcher, not a run on 3.10 itself.  The fix is one word in the recommendation: whoever adds
the bar must name the timeout class in the except tuple rather than rely on `OSError`.  My
`test_a_timeout_on_the_dispatch_must_name_its_own_exception_class` enforces that the day the bar
lands, and asserts the MRO fact on whatever interpreter runs it.

## 7. D6 -- the exit hang, reproduced with my own wedge

`vf7_d6_exit_hang.py`, WSL 3.12.3.  The branch's wedge installs `SIG_IGN` for SIGTERM.  Mine
installs a no-op SIGTERM **handler** -- the signal is delivered, caught and discarded, a different
kernel path and the shape a worker with a cleanup handler would have -- and then one worker is
SIGKILLed so the pool marks itself broken and CPython's `_terminate_broken` runs its unbounded
`p.terminate()` / `p.join()` over the survivors.  `pool_is_broken: true` on every wedged arm.

| arm | library call | process gone after | exited on its own | identical |
|---|---|---|---|---|
| control | 4.237 s | 10.721 s | yes, rc 0 | true, max delta 0.0 |
| wedge, prototype OFF | 3.783 s | **45.078 s (SIGKILLed at the deadline)** | **no** | true, 0.0 |
| wedge, prototype ON (all 3 steps confirmed applied) | 4.742 s | **45.082 s (SIGKILLed)** | **no** | true, 0.0 |
| wedge, surviving workers SIGKILLed after the answer | 5.547 s | 24.748 s | **yes, rc 0** | true, 0.0 |
| wedge, both | 4.662 s | 20.358 s | yes, rc 0 | true, 0.0 |

(My wall times are larger than the branch's throughout -- my deadline is 45 s not 60 s, my figures
are measured from process START not from the exit line, and the box was carrying four pytest lanes.
The DECISION -- exited on its own, or had to be killed -- is identical in all five arms.)

**The stacks confirm the report frame for frame.**  I select the main thread out of the
`faulthandler` dump by its identifier, which the child reports, because `faulthandler` prints
`Thread 0x<hex>` with no name:

* prototype **OFF**, main thread:
  `threading.py:1167 _wait_for_tstate_lock` <- `threading.py:1147 join` <-
  `concurrent/futures/process.py:102 _python_exit` <- `threading.py:1592 _shutdown`
* prototype **ON**, main thread: that join is GONE and the interpreter is one layer down --
  `multiprocessing/popen_fork.py:27 poll` <- `popen_fork.py:43 wait` <-
  `multiprocessing/process.py:149 join` <- `multiprocessing/util.py:360 _exit_function`
* the manager thread, in BOTH arms:
  `popen_fork.py:27 poll` <- `popen_fork.py:43 wait` <- `process.py:149 join` <-
  `process.py:593 _join_executor_internals` <- `process.py:528 _terminate_broken`

So the three-step daemonising prototype does exactly what the report says it does -- it removes one
of two unbounded joins at exit -- and the second join is the workers themselves.  **Only the arm
that removes the workers closes the hang.**  Confirmed.

### 7.1 On the maintainer recommendation

I agree with it, and I would take it further in one respect.  The recommendation -- in
`_abandon_pool`'s reaper, after the non-joining `shutdown(wait=False, cancel_futures=True)`, wait a
grace period and then `Process.kill()` any of the executor's `_processes` still alive -- is the only
thing my five arms show closing the hang, and the reaper is the right place because it is a daemon
thread that nothing joins at exit and it already owns the retired pool.  Reusing
`_POOL_SHUTDOWN_TIMEOUT = 120 s` as the grace bar rather than inventing a second number is right;
against my own measured healthy teardowns it is 50-300x.  The risk the report names is the real one
and it is a policy question, not a defect: SIGKILL is not a request, and a worker mid-write to a
shared resource loses it -- although in this specific state (the pool is broken, retired, and the
computation has already returned its bit-identical serial answer) nothing downstream reads anything
those processes hold.  The one thing I would add: **the release note should say what ships today,
and it should say it in the terms my table measures** -- on a wedged pool the computation is saved
and the PROCESS is not; a batch script gets the right numbers and then has to be killed.  That
sentence is worth more to a user than the repair is, because the repair is a decision someone has to
take and the note costs nothing.

---

## 8. Defects

| id | severity | where | what | reproducer |
|---|---|---|---|---|
| VD1 | P3 | `.test_durations` | the nine tests this branch adds to `test_fix_newton_pool_broken_fallback.py` have **no entries**, so a sharded CI run schedules them as unknown-duration.  (I spliced them, and my own, with measured values -- see section 9) | `python -c "import json;d=json.load(open('.test_durations'));print('test_the_join_detector_sees_the_with_form' in str(d))"` on `5db3f09d` -> `False` |
| VD2 | P3 (latent) | `_lens_traced.py::_shutdown_pool_bounded`, the helper's `finally` | `_ABANDONED_POOLS.remove(ex)` is UNCONDITIONAL: it runs even when the caller never appended, so it drops an entry `_abandon_pool` put there.  Measured **6/6** by construction on both builds.  Unreachable today only because no library path gives one executor to both mechanisms | `vf2_d2_census.py::attack_unconditional_remove`; pinned by `test_one_executor_never_reaches_both_teardown_mechanisms` |
| VD3 | P2 | `WP-B13_FOLLOWUPS_REPORT.md` sec. 6.1, candidate B, last bullet | "`TimeoutError` is an `OSError` subclass, so it lands in the dispatcher's existing infrastructure clause" is true on 3.11+ and FALSE on 3.10, which `pyproject.toml` declares as supported.  Driven: the pre-3.11 MRO escapes the clause and reaches the caller | `vf5_d5_timing.py::measure_clause_reach`; pinned by `test_a_timeout_on_the_dispatch_must_name_its_own_exception_class` |
| VD4 | P3 | same section | the recommendation does not say that a sentinel bounds only the BOOTSTRAP: a worker that answers once and then stops answering is still unbounded.  Driven on both builds | `vf5_d5_timing.py::measure_residual_exposure`; pinned by `test_a_bootstrap_bar_would_not_bound_a_chunk_that_wedges_later` |
| VD5 | P3 | `_get_persistent_worker_pool` docstring, second cost bullet, and `WP-B13_FOLLOWUPS_REPORT.md` sec. 3 | "a worker that has only run `_newton_pool_init` and never been given a chunk, **which is what the SURPLUS of a pool wider than the clamp is**" -- that worker does not exist.  `ProcessPoolExecutor` spawns lazily: 0 processes after construction, one per submitted chunk thereafter, on both builds | `vf3_d3_footprint.py` (`processes_after_construct: 0` on every rung); pinned by `test_a_pool_wider_than_the_dispatch_holds_no_surplus_processes` |
| VD6 | P3 | `test_fix_newton_pool_broken_fallback.py::_unbounded_executor_joins` | 2 residual false negatives: `with <aliased class>(...)` and `with <pre-built name>:`.  Both are `shutdown(wait=True)` on exit | `vf4_d7_detector_corpus.py` shapes `c04`, `c17`; made harmless by `test_no_module_uses_an_executor_teardown_the_join_detector_cannot_see` |
| VD7 | P3 | `test_the_in_flight_counter_is_one_claim_per_dispatch_not_per_chunk` | the pin walks `ast.Compare` on `_POOL_INFLIGHT`, but the magnitude leaves the module as `_note_pool_inflight`'s RETURN VALUE, which no such Compare can see | `vf6_d4_readers.py::branch_d4_check`, shape 5; closed by `test_nothing_consumes_the_in_flight_counters_magnitude` |
| VD8 | P4 | same pin | it false-fails on `0 < _POOL_INFLIGHT`, a legal spelling of the same zero-vs-non-zero decision: `others` is read from `node.comparators` only, so a reversed comparison puts the Name in the "something other than zero" bucket | `vf6_d4_readers.py::branch_d4_check`, shape 2 |
| VD9 | P4 | `WP-B13_FOLLOWUPS_REPORT.md` sec. 0, D3 row | the verdict row quotes "served **98.7-100.8 MB** (Win)" (the MEANS) while sec. 3 and the docstring quote a 101.56 MB max and "~102 MB".  Cosmetic, but the row reads as a range of maxima | read both |

### Requested changes (outside my ownership)

1. **`lumenairy/elements/_lens_traced.py::_shutdown_pool_bounded`** (VD2) -- make the removal
   conditional on the caller having added it:

   ```python
       added = []                       # guarded by _ABANDONED_POOLS_LOCK

       def _run():
           try:
               ex.shutdown(wait=True)
           except (RuntimeError, OSError, ValueError):
               pass
           finally:
               done.set()
               with _ABANDONED_POOLS_LOCK:
                   if added:            # ... only OUR entry, never a reaper's
                       try:
                           _ABANDONED_POOLS.remove(ex)
                       except ValueError:
                           pass
       ...
       with _ABANDONED_POOLS_LOCK:
           if done.is_set():
               return True
           _POOL_SHUTDOWN_TIMEOUTS += 1
           _ABANDONED_POOLS.append(ex)
           added.append(True)
   ```

   The ordering is unchanged (`done` is still published before the lock, the caller still re-reads
   it under the lock), so every D2 pin stays green; the only difference is that the helper stops
   removing entries it did not put there.

2. **`_get_persistent_worker_pool` docstring** (VD5) -- replace the second bullet.  The two states
   are "a worker that has SERVED a chunk" and "a worker that does not exist yet", because
   `ProcessPoolExecutor` spawns lazily; the surplus of a wider pool costs nothing until it is used,
   and the ladder's never-served column is the cost of a worker created by a labelling submit, not
   of anything the ceiling rule leaves behind.  Say so, and state the served figure as the measured
   RANGE (93.6-101.6 MB Windows over two independent measurements, 69.5-76.9 MB WSL) rather than one
   box's peak.

3. **`WP-B13_FOLLOWUPS_REPORT.md` sec. 6.1** (VD3, VD4) -- two sentences:
   *"`concurrent.futures.TimeoutError` is an alias of the builtin (and so an `OSError`) only from
   Python 3.11; this package supports 3.10, so whoever adds the bar must name the timeout class in
   the except tuple rather than rely on `OSError`."* and *"this bounds a pool that never comes up; a
   worker that answers once and then stops answering remains unbounded, so the exposure is narrowed,
   not closed."*

4. **`test_the_in_flight_counter_is_one_claim_per_dispatch_not_per_chunk`** (VD7, VD8) -- read the
   non-zero operands from `[node.left, *node.comparators]` minus the `_POOL_INFLIGHT` node itself,
   so a reversed comparison is not a false failure; and add the call-site check my
   `test_nothing_consumes_the_in_flight_counters_magnitude` carries, or delete that arm of the pin
   in favour of it.

5. **`.test_durations`** (VD1) -- I spliced the nine ids with measured values in this branch's
   commit.  If the maintainer would rather the WP branch carried them, take them from there.

---

## 9. Runs

### 9.1 The thirteen traced-lens / pool files, four arms

`pytest tests/unit/test_fix_newton_pool_broken_fallback.py tests/unit/test_fix_newton_pool_memory.py
tests/unit/test_niche_newton_pool_both_fits.py tests/unit/test_carrier_field.py
tests/unit/test_carrier_referenced.py tests/unit/test_audit2609_a3_traced_lens.py
tests/unit/test_audit2609_a3_verify_traced.py tests/unit/test_niche_audit_w9_traced_determinism.py
tests/unit/test_niche_d15_deterministic_traced_fit.py tests/unit/test_hammer_h3_traced_nyquist_guard.py
tests/unit/test_niche_audit_w3_elements.py tests/unit/test_niche_perf_round2_2026_08_10.py
tests/unit/test_niche_audit_e_prepared_and_enums.py -q [--capture=sys]`

The file count and the test count are unchanged from the branch's own run: **373 collected**
(364 before this package, +9 from it), so the expected reading is 372 passed / 1 skipped.  My tests
live in their own file and are NOT in this gate.

| build | capture | result | wall |
|---|---|---|---|
| Windows 3.14.6 | default (`fd`) | **372 passed, 1 skipped** | 1614.04 s |
| Windows 3.14.6 | `--capture=sys` | **372 passed, 1 skipped** | 1464.41 s |
| WSL 3.12.3 | default (`fd`) | **372 passed, 1 skipped** | 1543.57 s |
| WSL 3.12.3 | `--capture=sys` | **372 passed, 1 skipped** | 1375.76 s |

Four arms, four identical readings, the same single skip every time
(`test_fix_newton_pool_memory.py`, whose own reason is that this BLAS reduces identically at every
width it tries).  The capture axis moves nothing but wall time.  The wall times are 2.6-3.7x the
branch's for the reason in the header: two of these arms and a mutation matrix were running against
three other agents' lanes on the same box.

### 9.2 The rest of the gate

| command | build | result | wall |
|---|---|---|---|
| `pytest test_audit_except_budget.py test_public_api.py test_v4_16_2_dispatcher_pin_doc_consistency.py test_audit2609_a17_history_lint.py test_audit2609_a23_census_mechanism.py test_v4_14_2_dispatcher_pin_cache_locks.py test_eme_census_determinacy.py test_audit2609_a17_history_relocation.py -q` | Windows | **892 passed, 6 skipped** | 531.51 s |
| the same sweep | WSL | **1 failed, 891 passed, 6 skipped** -- the one red is environmental, see below | 242.29 s |
| `pytest tests/unit -q -k walker` | Windows | **118 passed, 6 skipped, 16205 deselected** | 249.87 s |
| `pytest tests/unit/test_verify_b13_newton_pool.py -q` | Windows | **3 passed** | 57.27 s |
| `pytest tests/unit/test_verify_b13_newton_pool.py tests/unit/test_verify_b13_followups.py -q` | WSL | **14 passed** (3 + my 11) | 41.22 s |
| `pytest tests/unit/test_verify_b13_followups.py tests/unit/test_fix_newton_pool_broken_fallback.py -q --durations=0` | Windows | **35 passed** | 170.00 s |
| `ruff check lumenairy/ tests/ validation/probe_verify_b13_followups/` | WSL | **All checks passed** | -- |

**The one WSL red is an environment fact and I confirmed it independently**, rather than taking the
branch's word for it.  `test_public_api.py::test_installed_metadata_version_matches_source_version`
compares `importlib.metadata.version('lumenairy')` against `lumenairy.__version__`.  In `~/lumvenv`
the editable install's metadata reads **5.11.0**; the source reads **5.47.0** on this branch AND at
base `b631ce79` (`git show b631ce79:lumenairy/__init__.py` -> `__version__ = "5.47.0"`), so the base
fails it identically and nothing in either package touches versioning.  The same test passes on
Windows.  The repair is `pip install -e .` in that venv, by whoever owns it; like the branch, I did
not silently reinstall under other agents' lanes.

The walker sweep deselects 16205 where the branch's run deselected 16194: the difference is the
eleven tests this verification adds, which is the only thing that changed in the collection.

Durations for all twenty new test ids (the branch's nine and my eleven) are spliced into
`.test_durations`, JSON-validated on reload (16 268 -> 16 288 entries).  **They were missing for the
branch's nine -- defect VD1.**  The slowest new test is
`test_a_pool_wider_than_the_dispatch_holds_no_surplus_processes` at **29.2 s**, comfortably inside
the 60 s bar; every other one of mine is under 20 s and most are under 3 s.

### 9.3 My probes

| file | what it measures | JSON |
|---|---|---|
| `vf1_d1_dump.py` | D1, with a different frame and a different park mechanism | `vf1_win.json`, `vf1_wsl.json` |
| `vf2_d2_census.py` | D2: three orderings x two built arrival orders, plus four attacks | `vf2_win.json`, `vf2_wsl.json` |
| `vf3_d3_footprint.py` | D3, labelled through a socket rendezvous; the lazy-spawn finding | `vf3_win.json`, `vf3_wsl.json` |
| `vf4_d7_detector_corpus.py` | D7: 20 shapes, both detectors, graded TP/FP/FN/TN | `vf4_win.json`, `vf4_wsl.json` |
| `vf5_d5_timing.py` | D5: the ladder off the `sub_progress` channel, the `as_completed` semantics, the clause reach, the residual exposure | `vf5_win.json`, `vf5_wsl.json` |
| `vf6_d4_readers.py` | D4: grep + AST reader census, and the branch pin graded | `vf6_win.json`, `vf6_wsl.json` |
| `vf7_d6_exit_hang.py` | D6: five arms, per-thread stacks selected by identifier | `vf7_wsl.json` |
| `vf8_mutations.py` | the mutation matrix (plugin + driver) | `vf8_win.json`, `vf8_wsl.json` |

### 9.4 The mutation matrix

`vf8_mutations.py` is a pytest plugin that puts ONE regression back in memory plus a driver that
runs every one of them and tabulates which tests went red.  Each arm runs the 24 relevant tests of
`test_fix_newton_pool_broken_fallback.py` and `test_verify_b13_followups.py` under a subprocess
deadline (1200 s) and, where the venv has the plugin, `--timeout=300`, so a mutation that made a
test HANG instead of fail would be recorded as `driver_timed_out` rather than as a pass.  The
child-process tests re-apply the injected mutation from `VF8_MUTATE`, because a child is a fresh
interpreter and an in-memory injection in the parent is invisible to it -- an earlier version of
this matrix reported `m10` as uncaught for exactly that reason, which is the kind of harness
artifact that reads as a gap.

| mutation | what it breaks | Windows: caught by | WSL |
|---|---|---|---|
| `m1_dump_stringio` | `_thread_dump` back to `io.StringIO` (D1) | **4** -- both branch D1 pins and both of mine | 4 |
| `m2_census_neverdrop` | the pre-fix census: append, never remove (D2) | **7** -- five branch census pins, my caller-first order, my two helper-crash arms | 7 |
| `m3_census_oneline` | the one-line repair the verification asked for | **1** -- `test_the_census_survives_the_helper_first_lock_order`, and only that one | 1 |
| `m4_census_noappend` | never append: the census stops being a census | **6** | 6 |
| `m5_detector_calls` | the join detector back to the `ast.Call`-only sweep (D7) | **2** -- the positive control and the sibling-pool premise gate | 2 |
| `m6_detector_nowait` | the detector stops exempting `wait=False` | **2** -- the `_lens_traced` invariant and the positive control | 2 |
| `m7_inflight_perchunk` | the counter claims once per CHUNK (D4) | **1** -- `test_the_in_flight_counter_cannot_go_negative_or_leak` | 1 |
| `m8_inflight_never` | the dispatcher never claims the pool | **1** | 1 |
| `m9_helper_no_finally` | the helper publishes `done` OUTSIDE the `finally` | **3** -- both my helper-crash arms and the branch's helper-first order | 3 |
| `m10_lazy_eager` | the pool pre-spawns its workers (D3's lazy-spawn fact) | **1** -- `test_a_pool_wider_than_the_dispatch_holds_no_surplus_processes` | 1 |

**10 of 10 caught on both builds, 0 uncaught, 0 errored, 0 hung, baseline clean (24 passed) on
both.**  Every mutation is caught by the test that is supposed to catch it, and `m3` -- the one-line
census repair -- is caught by exactly one test, which is the whole of the branch's D2 argument
turned into a decision.

One methodological note worth recording, because it nearly produced a false reading: the WSL venv
has no `pytest-timeout`, so passing `--timeout` there makes pytest exit on the command line.  The
first WSL run of this matrix reported **all ten mutations uncaught** for that reason alone.  The
driver now (a) omits `--timeout` where the plugin is absent and (b) reports an arm whose pytest
produced no summary line as `ERROR`, never as "not caught".

### 9.5 Orphan sweep

Every probe that breaks a pool leaves worker processes behind by construction, and
`vf7_d6_exit_hang.py` leaves SIGTERM-swallowing ones.  Swept on both builds after every probe and
again at the end of the package:

| build | command | processes of MINE found |
|---|---|---|
| Windows 3.14.6 | `Get-CimInstance Win32_Process -Filter "Name='python.exe'"`, matched on command line against `lum_vpool2`, `vf*_`, `spawn_main` | **0** |
| WSL 3.12.3 | `pgrep -af spawn_main`, `pgrep -af "vf[0-9]_"`, `ps -eo pid,ppid,etimes,cmd \| grep -i lum_vpool2` | **0** |

`vf7_d6_exit_hang.py` sweeps after every arm and once more at the end; its final sweep reported
`{'found': 1, 'killed': []}`, the one being the driver process itself, which it correctly skips.
The processes resident on the box at the end are other agents' -- a `lum_mb3` pytest lane, an
`r3fine.py` sweep with six `r3scan.py` children, two `probe_verify_wave5_e` probes, the VS Code
isort server and the blender-mcp extension -- identified by their command lines and left alone.

---

## 10. What I could not measure

1. **Python 3.10.**  The VD3 arm is a reconstruction of the pre-3.11 `concurrent.futures.TimeoutError`
   class hierarchy driven through the real dispatcher, not a run on a 3.10 interpreter; none is
   installed on this box (3.13 and 3.14 on Windows, 3.12 in WSL).  The CPython change is gh-90315.
2. **A quiet box.**  Every wall time in section 6 was taken at 94-100 % CPU with three other agents'
   lanes resident.  They bound the true figures from above; the branch's own ladder is the lighter
   reading of the same quantities and I did not reproduce a lighter one.
3. **D6 on Windows.**  The wedge needs a worker that survives a terminate request;
   `TerminateProcess` cannot be ignored, so the arms are WSL-only.  The branch reached the same wall
   for the same reason.
4. **The sibling pool's bootstrap cost.**  Unchanged from the branch's own limitation: I confirmed
   `carrier._multi_parallel_results`'s exposure statically and under the extended detector, and did
   not price a sentinel against its initializer.
5. **N = 32768.**  My ladder stops at 1024 for the same budget reason the branch's does, so
   everything either of us says about production field sizes is extrapolation.
6. **Whether VD2 has ever been hit.**  It is reachable only by a caller that hands one executor to
   both teardown mechanisms, and no library path does.

---

## 11. Ship recommendation

**SHIP**, with the four documentation/test edits in section 8 taken either here or as a follow-up;
none of them is a correctness defect in the shipped field, and none blocks.

What the branch claims, it does, and it does it on both interpreters:

* **D1, D2, D3, D4 and D7 are genuinely closed.**  I reproduced every fail-before with my own
  drivers -- a different wedge frame and park mechanism for D1, a census lock built in both
  directions for D2, a socket rendezvous instead of a Manager Barrier for D3, a 20-shape corpus for
  D7 -- and the branch's code survives all of them on Windows 3.14.6 and WSL 3.12.3.
* **The D2 argument against the requested one-liner is correct and now has a rate attached to it**:
  the one-liner leaks in 8 of 8 adverse-order reps and the shipped ordering in 0 of 8, on both
  builds.  This is the strongest part of the package: the race was produced by construction, not
  waited for.
* **The mutation matrix says the new tests are load-bearing**, not decoration: every mutation is
  caught, and caught by the test that is supposed to catch it.
* **The D5 and D6 recommendations are sound.**  D6's conclusion -- daemonising is not sufficient,
  only killing the workers closes the hang -- I reproduced independently, stacks and all.  D5's
  refusal to pick a fixed `T` is right and my loaded-box numbers make it more right.  The two edits
  I ask for there are one sentence each and neither changes the recommendation.
* **Nothing moved a number.**  Every field produced anywhere in this verification is byte-identical
  to its `n_workers=1` reference.

The one thing I would not let ship silently is the release note: on a wedged pool the computation is
saved and the process is not.  The branch's own report says so in ITS section 7.2; it belongs in the
CHANGELOG entry, not only in the audit trail.

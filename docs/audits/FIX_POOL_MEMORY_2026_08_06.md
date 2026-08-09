# FIX -- the Newton process pool had no resource accounting

**2026-08-06.  Branch `feat/pmm-per-layer-roadmap`.  Closes open item 1 of
`CAPSTONE_D121_2026_08_06.md` ("the Newton pool memory clamp -- the one blocker
to running design 121 unattended at the shipped defaults on a 128 GB box").
No git command was run.**

---

## 0. VERDICT

> **Design 121's Stage-B acceptance now COMPLETES at the shipped
> `n_workers=8`.  346.6 s wall, peak RSS 21.37 GB, peak system commit
> 46.0 / 244.3 GB, minimum free physical RAM 93.6 GB, and the banner reads
> `AT-PLANE dz=0: 3.350um / 90.3 / 99.7 / 99.8`, on-axis.  Before the fix the
> same command drove commit to 205.7 / 227.5 GB and free RAM to 0.0 GB and had
> to be killed at 9.7 min.**

And the finding that the assignment asked for and that the capstone could not
see from outside the library:

> **The 22.1 GB/worker was NOT the chunk, the pickled payload, or the fine
> grid.  It was the caller's own program.  `focus_scan_121.py` has no
> `if __name__ == '__main__':` guard, so every `spawn` worker RE-EXECUTED the
> entire design-121 acceptance chain before serving its Newton chunk.  The
> intrinsic cost of the chunk those workers were given is 1.9 GB.  22.1 / 1.9 =
> 11.5x, and 22.1 GB is within 6 % of the 20.86 GB the SAME chain peaks at
> serially -- because that is exactly what each worker was running.**

Two rules ship, because the measurement found two terms and only one of them is
chunk-shaped.  Section 4.

---

## 1. WHAT WAS WRONG

`apply_real_lens_traced(..., n_workers=K)` dispatches the Newton inversion
through `_invert_newton_parallel` (`lumenairy/elements/_lens_traced.py`), which
`np.array_split`s the wave grid into `K` chunks and submits all of them to a
persistent `ProcessPoolExecutor`.  There was **no memory accounting anywhere on
that path.**

The fine grid those chunks come from IS clamped -- `_memory_bounded_n_fine`
(`carrier.py:3729`) caps it against `_FINE_GRID_RAM_FRAC (0.5) x` the RAM budget
using a cost model of `_FINE_GRID_WORK_ARRAYS (4) x 16 B x n^2`.  **That model
is single-process.**  It approves a grid, hands it to the element, and the
element forks the Newton over it `n_workers` ways.

The same library already fixed the identical bug for the OTHER pool:
`carrier._multi_resolve_workers` clamps `congruence_workers` against free RAM,
and its comment records the failure being fixed there ("3 workers each correctly
decided they could afford a 16384^2 fine grid (17.2 GB) and then collectively
asked for 123 GB of a 127 GB box -- MEASURED on design 121's fan").  The Newton
pool never received that treatment.

---

## 2. GROUND TRUTH -- WHAT THE STAGE-B DISPATCHES ACTUALLY LOOK LIKE

Measured by spying on `_Cheb2DEvaluator.__init__` (gives the ray-fit grid
`n_launch`) and `.ev_value_and_grad` (gives the Newton point count) through a
real, serial, complete Stage-B run of `focus_scan_121.py`
(`N=2048, rs=4, NFC=8192, WF=4.0`).  Not read off a docstring:

```
  fit grid [387, 387]  evaluators 3   max newton pts 262144    coarse group 1
  fit grid [407, 407]  evaluators 3   max newton pts 262144    coarse group 2
  fit grid [439, 439]  evaluators 3   max newton pts 262144    coarse group 3
  fit grid [449, 449]  evaluators 3   max newton pts 262144    coarse group 4
  fit grid [461, 461]  evaluators 3   max newton pts 262144    coarse group 5
  fit grid [531, 531]  evaluators 6   max newton pts 262144    coarse group 6
  fit grid [465, 465]  evaluators 3   max newton pts  36481    EXACT FINAL LEG
```

Two things follow immediately, and both contradict the pre-fix reading of the
failure:

1. **The Newton grid is 512^2 = 262 144 points, not 8192^2 = 6.7e7.**  The
   capstone's note that each worker got "1/8 of the 6.7e7-point Newton grid" was
   the FINE GRID's size, not the Newton lattice's: `ray_subsample` sub-samples
   it.  At 8 workers a chunk is **32 768 points = 262 kB per coordinate array.**
2. **The exact final leg's Newton is the SMALLEST dispatch on the chain**
   (36 481 points, from `n_fine=8192` at the pitch-preserving `rs_fine`), and at
   36 481 points it sits below `_POOL_MIN_PIXELS` (200 000) entirely.  The pool
   engages on the six COARSE groups, not on the leg the capstone attributed it
   to.

So no chunk-sized model can produce 22.1 GB.  The gap had to be per-PROCESS,
and the capstone said so ("this is per-PROCESS overhead, not the chunk data")
without being able to say what it was.

---

## 3. THE SCALING LAW (3-point measurement, not a guess)

Method: run `_newton_invert_chunk` in a **fresh interpreter** -- which is
exactly what a `spawn` worker is -- with a payload of the real shapes, and read
`psutil` `peak_pagefile` (Windows peak commit charge) and `peak_wset`.  Peak
rather than sampled, so nothing is missed between samples.

Box: Windows 11 Pro 10.0.26200, Ryzen 9 5950X / 24 logical CPUs, 137.4 GB
physical + 107.0 GB pagefile = 244.3 GB commit limit, python 3.14.6, numpy
2.4.4, psutil 7.2.2, numba present.  Quiet box (96-120 GB free throughout).

### 3.1 Chunk axis, fit grid fixed at design 121's own 531^2

| chunk points | peak commit | peak wset | marginal |
|---|---|---|---|
| 32 768 | 1.916 GB | 0.319 GB | -- |
| 2 097 152 | 2.288 GB | 0.714 GB | -- |
| 4 194 304 | 2.849 GB | 1.274 GB | **267.42 B/pt** |
| 8 388 608 | 3.970 GB | 2.392 GB | **267.13 B/pt** |
| 16 777 216 | 6.211 GB | 4.629 GB | **267.16 B/pt** |

**Dead linear at 267.2 B per Newton point -- 0.1 % spread across an 8x range**
-- on a **1.728 GB intercept**.

267 B/point is ~33 float64 temporaries per point, which is what the Newton loop
actually holds live: `xa/ya/xw/yw`, the six Jacobian entries, `rx/ry/det/
inv_det/dxe/dye/xa_new/ya_new/res`, the numba Chebyshev kernel's
`u_flat/v_flat/f/fx/fy` for each of two evaluators, plus the chunk itself.  The
model is not fitted to a black box; it lands where the code says it should.

### 3.2 The intercept is import, not physics

| process state | peak commit |
|---|---|
| bare python | 0.012 GB |
| `import numpy` | 0.831 GB |
| `import lumenairy.elements._lens_traced` | 1.649 GB |
| ...after the first numba Chebyshev JIT | ~1.72 GB |

**Eight workers commit ~14 GB before touching a single Newton point.**  Note
the commit/working-set split: after import, commit is 1.65 GB while the working
set is 0.098 GB.  Commit is the quantity that killed the capstone run (Windows
fails allocations on commit exhaustion), so commit is what the model prices.

### 3.3 Fit-grid axis, chunk fixed at 2 097 152

| fit edge | fit points | peak commit | marginal |
|---|---|---|---|
| 531 | 281 961 | 2.288 GB | -- |
| 1024 | 1 048 576 | 2.626 GB | 440.3 B/pt |
| 2048 | 4 194 304 | 5.274 GB | 841.8 B/pt |

Superlinear (the Chebyshev fit builds an `(n_fit, 28)` design matrix and
LAPACK copies it), so the **larger** measured slope is shipped.

### 3.4 The shipped model, checked against every point

```
per_worker_bytes = 1.75e9                       # _NEWTON_WORKER_BASE_BYTES
                 + 268.0 * chunk_points         # _NEWTON_WORKER_BYTES_PER_POINT
                 + 850.0 * fit_points           # _NEWTON_WORKER_FIT_BYTES_PER_POINT
```

| chunk | fit | measured | model | model/measured |
|---|---|---|---|---|
| 32 768 | 531^2 | 1.916 GB | 1.998 GB | 1.04 |
| 2 097 152 | 531^2 | 2.288 GB | 2.552 GB | 1.12 |
| 4 194 304 | 531^2 | 2.849 GB | 3.114 GB | 1.09 |
| 8 388 608 | 531^2 | 3.970 GB | 4.238 GB | 1.07 |
| 16 777 216 | 531^2 | 6.211 GB | 6.486 GB | 1.04 |
| 2 097 152 | 1024^2 | 2.626 GB | 3.204 GB | 1.22 |
| 2 097 152 | 2048^2 | 5.274 GB | 5.877 GB | 1.11 |

Over-predicts everywhere by 4-22 %, which is the direction a resource clamp has
to err in.  `test_the_per_worker_model_reproduces_the_measurements` pins both
sides of that band.

### 3.5 ...and it does not explain 22.1 GB

At Stage B's actual shape (32 768 points/chunk, 531^2 fit) the model says
**2.00 GB/worker**, measured **1.92 GB**.  Eight of those is 16 GB against
~96 GB free -- **the memory clamp alone would NOT have bound, and would NOT
have prevented the OOM.**  That is stated here rather than hidden, because it is
what made the second rule necessary.

---

## 4. ROOT CAUSE OF THE 22.1 GB/WORKER

`multiprocessing.spawn._fixup_main_from_path` does

```python
main_content = runpy.run_path(main_path, run_name="__mp_main__")
```

in **every spawned child**.  A `__main__` with no `if __name__ == '__main__':`
guard therefore executes its **entire module body once per worker**, before the
worker ever touches the task it was spawned for.

`validation/repro_traced_carrier_121/focus_scan_121.py` has no such guard, and
its module body IS the design-121 acceptance chain.

### 4.1 Proved by running it

A minimal probe (0.5 GB ballast, 2 workers, a real one-group lumenairy chain,
no main guard) printed, from the WORKER processes:

```
[module scope] pid=43808 __name__='__main__'
[module scope] pid=33084 __name__='__mp_main__'      <- worker re-running the body
[module scope] pid=44648 __name__='__mp_main__'      <- worker re-running the body
[done]         pid=33084 field finite=True           <- worker ran the WHOLE chain
[done]         pid=44648 field finite=True
  [worker pid=33084] peak_pagefile=3.232 GB  peak_wset=1.126 GB
  [worker pid=44648] peak_pagefile=3.234 GB  peak_wset=1.126 GB
```

The workers ran the caller's whole program and then served their chunk.

### 4.2 The arithmetic closes

| quantity | value | source |
|---|---|---|
| serial peak of the Stage-B chain | 20.86 GB | capstone sec 4.2 |
| MEASURED per-worker commit, 8-worker run | 19.0-22.1 GB | capstone sec 4.1 |
| intrinsic cost of the chunk a worker got | 1.92 GB | sec 3.4 above |
| ratio | 11.5x | |

Each worker's footprint is the SERIAL CHAIN's footprint to within 6 %.  It was
never the Newton chunk.

### 4.3 Why the nested pool did not blow up further

The re-executed body reaches its own pool dispatch, whose grandchild spawn
raises `RuntimeError('...bootstrapping phase...')`.  `_invert_newton_parallel`
catches `RuntimeError` in its pool-infrastructure `except` and falls back to
serial -- **silently**.  So the library already absorbed the symptom of this
exact defect and reported nothing: an 8x re-execution of the caller's program
presented as a slow run.  `carrier._multi_looks_like_spawn_bootstrap` shows the
library knows this failure mode and that its remedy is the caller's main guard;
the Newton pool just never said so.

### 4.4 Why the remedy is SERIAL, not a smaller pool

There is no worker count that makes an unguarded re-execution acceptable:

* the memory is unbounded from the library's side (it is whatever the caller's
  program costs -- here 20.9 GB, but it could be anything);
* the TIME is unbounded the same way (each worker re-runs a 5-minute chain
  before starting a 0.05-second chunk);
* and the caller's **side effects run K extra times** -- prints, cache writes,
  registry mutation.  That is a correctness hazard, not a resource one, and no
  clamp addresses it.

So this rule returns 1.

---

## 5. WHAT SHIPPED

All in `lumenairy/elements/_lens_traced.py`.  New module-level, testable
helpers (mirroring `_multi_resolve_workers`, which is also module-level and
testable rather than a closure):

```
_newton_worker_bytes(chunk_points, fit_points)      the measured law of sec 3.4
_newton_resolve_workers(requested, n_total, fit_points, ...)
_spawn_reexecuted_main_script()                     mirrors multiprocessing.spawn
_script_has_main_guard(path)                        top-level AST check, cached
_is_main_guard_test(node)
_reset_newton_pool_resource_state()                 called by close_worker_pool
```

### 5.1 The cap formula

```
free_b   = min(psutil.virtual_memory().available, get_ram_budget())
budget_b = _NEWTON_POOL_RAM_FRAC * free_b - _NEWTON_POOL_MIN_FREE_GB * 1e9
allowed  = max(1, budget_b // _newton_worker_bytes(n_total / requested, fit))
# then RE-PRICE: fewer workers means bigger chunks, so shrink until
# allowed * _newton_worker_bytes(n_total / allowed, fit) <= budget_b
```

with `_NEWTON_POOL_RAM_FRAC = 0.5` -- **the same 0.5 the fine grid's own
ceiling uses** (`carrier._FINE_GRID_RAM_FRAC`), so the two clamps that meet on
the exact final leg speak the same language -- and
`_NEWTON_POOL_MIN_FREE_GB = 2.0`.

The reserve is the `_multi_resolve_workers` idiom
(`congruence_worker_min_free_gb`, 8 GB) **scaled to this pool**: that one guards
~24 GB chain workers, these are ~2 GB Newton workers, and an 8 GB reserve here
would refuse a 2-worker Newton dispatch on any box under ~20 GB.  A clamp that
binds where nothing is wrong is how a resource guard gets turned off.

The re-pricing loop is not decoration: without it the cap is computed at the
REQUESTED chunk size, and the smaller pool it approves then gets proportionally
bigger chunks and over-spends the very budget that produced it.

### 5.2 The two rules, in order

1. `_spawn_reexecuted_main_script()` is not None -> **return 1**, with a
   once-per-process `RuntimeWarning` naming the file, the mechanism, the
   measured cost, and the one-line remedy.
2. otherwise, the memory cap above; when it binds, one `RuntimeWarning` per
   binding dispatch naming what was asked for, what one worker costs, what the
   box has, what the budget rule allowed, and what will run.

Both warnings state that the result is bit-identical to serial, so a reader
cannot mistake a resource-limited run for a different run.

### 5.3 The import-safety predicate mirrors CPython, it does not guess

Verified on this box for all four rows, because getting this wrong would
serialise every pytest run in the repo:

| invocation | `__main__.__spec__.name` | `__file__` | child re-runs body? |
|---|---|---|---|
| `python -m pytest` | `pytest.__main__` | `.../pytest/__main__.py` | **no** (ends in `.__main__`) |
| `pytest` (console script) | `__main__` | `...pytest.exe/__main__.py` | **no** |
| `python script.py` | `None` | `script.py` | **yes** |
| `runpy.run_path(x, run_name='__main__')` | `None` | `x` | **yes** |
| `python -m yourscript` | `yourscript` | `yourscript.py` | **yes** (`run_module`) |

The last row is the one that is easy to get wrong in the other direction:
`__spec__.name` IS set there, but it does not end in `__main__`, so
`_fixup_main_from_name` falls through to `runpy.run_module(...,
run_name='__mp_main__')` and the body re-runs exactly as it does for a path.

plus multiprocessing's own `ipython` carve-out and the frozen/`__file__`-less
cases.  A file that cannot be read or parsed returns "guarded" -- "cannot prove
it is unguarded" must not mean "assume the worst", or a zipapp silently loses
its pool.  The guard must be TOP-LEVEL: one nested inside a function does not
protect the module body, which is the whole failure.

### 5.4 Composition with the V5 cost-class gate

The clamp runs **upstream** of the cost gate, and deliberately so.  The cost
gate's promotion evidence is keyed by `(worker count, fit backend, point-count
band)`; if the clamp ran afterwards, that evidence would be keyed by a worker
count the dispatch will not use.  So:

```
resource clamp  ->  bounds the CEILING (how many workers may ever run)
cost gate       ->  decides whether to dispatch AT ALL at that ceiling
```

`test_the_dispatch_path_consults_the_resolver` pins both the call and the
order.

Running first has one cost, and it was found by running the suite rather than
by reasoning: the clamp then also sees calls the SIZE gate is about to answer
serially at any worker count, and announcing a cap on a dispatch that will
never happen is noise that trains a reader to ignore the warning.  Observed on
`test_fga.py::test_universal_dispatcher_multivalued_avoids_traced`, where a
24-CPU default against a 1125^2 ray-fit grid produced a perfectly CORRECT
`24 -> 20` clamp (24 x 2.83 GB = 67.8 GB against a 56.8 GB budget) for a
**16-points-per-chunk** call that then ran in-process anyway.  So the resolver
takes `min_pool_points` and, below it, still CLAMPS (the count feeds the gate)
but stays SILENT.  `_invert_newton_parallel` passes `_POOL_MIN_PIXELS_WARM`.
Pinned by `test_a_call_that_cannot_reach_the_pool_is_clamped_silently` and by
the wiring test.

That clamp is worth reading twice, because it is the clamp doing real work on a
HEALTHY box: the fit-grid term is 1.08 GB/worker at 1125^2, every worker re-fits
it, and 24 of them is 68 GB of commit for a Newton grid of 384 points.  The
memory arm is not dead code that only fires in the pathological case.

---

## 6. STAGE B -- BEFORE AND AFTER

Same command, same shipped `n_workers=8`, same box.

| | BEFORE (capstone sec 4.1) | AFTER (this fix, final code) |
|---|---|---|
| outcome | **KILLED at 9.7 min**, OOM trajectory | **COMPLETED, 346.6 s (5.78 min)** |
| system commit | **205.7 / 227.5 GB** | **46.0 / 244.3 GB** |
| min free physical RAM | **0.0 GB** | **93.6 GB** |
| peak RSS (run's own tree) | 103.2 GB | **21.37 GB** |
| per-worker commit | 22.08-22.11 GB x 8 | n/a -- **no workers spawned** |
| worker count chosen | 8 (unclamped) | **1 (serial)**, rule 1 |
| paging | working sets trimmed to 0 from ~8 min | none |
| banner | never reached | `3.350um / 90.3 / 99.7 / 99.8` |

Run twice, once mid-fix and once on the final code after the sec-7.2 lock /
registry changes touched the very warn-once path this exercises:

| | wall | chain | peak RSS | commit | min free | guard warns | cap warns |
|---|---|---|---|---|---|---|---|
| mid-fix | 345.9 s | 309 s | 20.87 GB | 44.0 GB | 96.1 GB | 1 | 0 |
| **final code** | **346.6 s** | **310 s** | **21.37 GB** | **46.0 GB** | **93.6 GB** | **1** | **0** |

Identical metrics, 0.2 % apart in wall time.

Run detail (`CAPSTONE_NW=8 CAPSTONE_WARN=1 python capstone_stageB.py`;
`CAPSTONE_WARN=1` only neutralises the harness's blanket
`filterwarnings('ignore')` so the guard surface is readable -- the capstone
established at sec 4.4 that it does not move any metric):

```
config: N=2048 rs=4 nfc=8192 wf=4.0 nout=2048 chain_kwargs={}  (pure library defaults)
chain done 310s
AT-PLANE: FWHM=3.350um EE3=90.3% EE6=99.7% EE12=99.8% off=(+0.00,+0.00)um
BEST-FOCUS[peak] dz=+0um: FWHM=3.350um EE3=90.3% EE6=99.7% EE12=99.8% pk=5.529e+03
BEST-FOCUS      dz=+5um: FWHM=3.450um EE3=89.6% EE6=99.7% EE12=99.8%
CAPSTONE STAGE B WALL 346.6 s (5.78 min)  peak RSS 21.37 GB at t=291 s
```

and the fix's own line, emitted **exactly once** in the whole run (counted:
1 occurrence of the guard warning, 0 of the memory-cap warning -- the memory arm
correctly did not bind at 94 GB free):

```
_lens_traced.py:9114: RuntimeWarning: apply_real_lens_traced: running the Newton
inversion SERIAL instead of on 8 workers, because this process's __main__
(...\focus_scan_121.py) has no top-level `if __name__ == '__main__':` guard.
multiprocessing's spawn workers RE-EXECUTE the whole __main__ module body before
serving their chunk, so each worker would re-run your entire program: MEASURED
on design 121, 22.1 GB committed per worker (~177 GB across 8) against 1.9 GB of
actual chunk, which took a 227.5 GB-commit box to 0.0 GB free.  Wrap the
top-level code of that file in the guard to get the pool back -- the serial
result is bit-identical either way, so nothing but wall time changes here.
```

### 6.1 Banner confirmation

| | FWHM (um) | EE3 % | EE6 % | EE12 % | offset |
|---|---|---|---|---|---|
| recorded acceptance | 3.350 | 90.3 | 99.7 | 99.8 | on-axis |
| capstone, serial (`n_workers=1`) | 3.350 | 90.3 | 99.7 | 99.8 | (+0.00, +0.00) um |
| **this fix, shipped `n_workers=8`** | **3.350** | **90.3** | **99.7** | **99.8** | **(+0.00, +0.00) um** |

Identical to every printed digit.  Wall time 346.6 s against the capstone's
346.1 s serial (0.14 %), `chain done` 310 s against 308 s, peak RSS 21.37 GB
against 20.86 GB -- i.e. the run IS the serial run, which is what "the clamp
chose 1 worker" means and what "bit-identical" promises.

**Note on the commit limit.** The capstone read 227.5 GB; this box now reads
244.3 GB (137.4 physical + 107.0 pagefile).  Windows grows the pagefile, so the
denominators differ by 7 %.  It does not affect the comparison: the numerator
went from 205.7 to 46.0 GB and free RAM from 0.0 to 93.6 GB.

### 6.2 Liveness

The measurement is not vacuous in either direction:

* the external monitor's python-process histogram over the run's 189 samples
  is `{4: 11, 5: 8, 6: 170}` -- i.e. the box never held more than 6 python
  processes, which is the 5 pre-existing ones (VS Code's isort LSP, two MCP
  servers, the monitor itself) plus the run.  **Zero Newton workers were
  spawned**; an 8-worker dispatch would have shown 14.  That is the observable
  form of "the clamp chose serial", independent of the warning text.
* the banner scan is live: `pk` runs 2.400e+02 at dz=-80 um, 5.529e+03 at 0,
  3.226e+02 at +80 um -- single-peaked and maximised at the plane, so the
  metric is not reading a constant.
* the other Stage-B guard warnings the capstone recorded still fire unchanged
  (40.0 % / 29.2 % / 26.5 % Newton non-convergence, the 2.75x exit-wavefront
  under-sampling), so the run is the same run and this fix silenced nothing.

---

## 7. TESTS

`tests/unit/test_fix_newton_pool_memory.py`, 31 tests, all green.

**(a) fail-before, scarce memory.**
`test_the_pre_fix_path_would_have_submitted_every_chunk` states the pre-fix gate
verbatim (`n_cpu = n_workers`, no memory term), proves at the mocked scarcity
that that dispatch does NOT fit the budget, then asserts the shipped resolver
returns a strictly smaller count that DOES -- priced at the chunk size that
count implies, which is the re-pricing arm of sec 5.1.
`test_the_cap_degrades_all_the_way_to_serial` walks 200/60/30/12/6/1/0 GB and
requires the sequence to be monotone and to bottom out at 1, never 0 and never
an exception.

**(b) abundant memory.**
`test_abundant_memory_leaves_the_requested_worker_count_alone` -- 8 stays 8 at
400 GB free, **and warns nothing**; a clamp that trimmed a worker on a healthy
box would be a silent permanent slowdown on the default path.
`test_the_clamp_can_only_ever_lower_the_count` pins monotonicity from above.
`test_a_missing_memory_oracle_keeps_the_historical_behaviour` -- no psutil, no
clamp, exactly as `_multi_resolve_workers` does.

**(c) bit-identity, serial vs a CAPPED pool.**
`test_a_capped_pool_is_bit_identical_to_serial` runs the real 262 144-point
chain three ways -- clamped to 2 workers, uncapped at 4, and serial.  Both
inputs of the clamp are pinned (frozen psutil snapshot at 20 GB available, the
per-worker cost forced flat at 3 GB) so the cap lands on exactly 2 by
arithmetic, not by this box's mood, and a `_WidthSpy` asserts the dispatch
widths were really 2 and really 4 -- without which the identity assertions would
pass for free if the clamp stopped working.

It asserts **two different contracts, kept apart deliberately**, and the reason
is in section 8.1: conflating them made a pre-existing library defect look like
a clamp bug on this fix's own first full-tree sweep.

1. **The clamp's contract, unconditional:** the 2-worker pool the clamp sized
   must equal the 4-worker pool the caller asked for.  That is a different
   `np.array_split` of the same lattice, which is exactly what the clamp could
   break and nothing else is.  Both arms fit in fresh workers, so 8.1 cannot
   reach this comparison.
2. **The library's contract (serial == pool),** checked against the UNCAPPED
   pool first.  If serial already disagrees with a pool the clamp never
   touched, the process cannot honour that contract at all, and the test skips
   with the measured delta and the parent's resolved backend rather than
   blaming the clamp.

**(d) the warning.**
`test_the_cap_warning_fires_once_and_names_the_numbers` -- exactly one warning
per binding dispatch, and it must contain the requested count, the per-worker
GB, the points/chunk, the fit-grid size, the available GB, the computed budget,
the count that will run, and the words `bit-identical to serial`.

**The root-cause rule.**
`test_a_spawn_worker_really_does_rerun_an_unguarded_main` is the PREMISE, proved
by subprocess: an unguarded script's top-level body runs **twice** (parent +
one spawn worker, the second under `__mp_main__`), the guarded control runs it
**once**.  Same file, same imports, one line moved -- so the remedy the warning
recommends is verified to be the remedy.
`test_an_unguarded_main_forces_serial_and_says_why` is the fail-before (pre-fix
this returned 8 even with infinite RAM).
`test_the_unguarded_warning_fires_once_per_process` pins the once-per-process
dedup and that `close_worker_pool` re-arms it.
`test_a_guarded_main_is_left_alone` is the control.
`test_the_guard_detector_reads_the_ast_not_the_text` parametrises 7 spellings
including two that must NOT count (a guard nested in a function; a guard in a
comment).
`test_the_predicate_mirrors_multiprocessing` parametrises the four invocation
rows of sec 5.3.
`test_an_unreadable_main_keeps_the_pool` pins the conservative direction.

**The model and the wiring.**
`test_the_per_worker_model_reproduces_the_measurements` replays all seven
measured points and requires the model to bound each from above and stay within
25 %.  `test_the_chunk_slope_is_the_measured_one` pins 267.2 B/pt and the import
intercept range.  `test_the_budget_rule_matches_the_sibling_pool` pins
`_NEWTON_POOL_RAM_FRAC == carrier._FINE_GRID_RAM_FRAC`.
`test_the_dispatch_path_consults_the_resolver` pins the call, and that it comes
BEFORE the cost gate, and that the split and the pool both use the clamped
count -- bit-identity cannot notice a missing clamp (an unclamped pool returns
the same numbers, right up until it does not return at all), so the wiring pin
is the only durable guard.
`test_the_fine_grid_ceiling_is_still_single_process` records the other half of
the capstone's finding as a live fact, so whoever eventually teaches
`_memory_bounded_n_fine` about `n_workers` finds this clamp and decides
deliberately instead of double-counting.

### 7.0 Suite coverage actually run

The blast radius was taken as every suite that references
`apply_real_lens_traced` / `propagate_traced_carrier_chain` / `_lens_traced` /
`n_workers`: **97 files, 2527 tests.**

All of the following are on the FINAL code.

| run | result |
|---|---|
| `test_niche_newton_pool_both_fits.py` (the contract this fix must not move) | **23 passed** |
| `test_fix_newton_pool_memory.py` (new) | **31 passed** |
| both pool suites + the two structural cache pins of sec 7.2 | **152 passed, 4 skipped** |
| blast-radius **HEAD**, 59 files, ordered (58.6 min) | **1888 passed, 1 skipped, 2 failed** -- the 2 are sec 8.2 |
| blast-radius **TAIL**, 38 files, ordered (14.6 min) | **633 passed, 4 skipped, 0 failed** |
| full `tests/unit` tree, ordered, `-x` | **1912 passed** before stopping on sec 8.1 (now attributed, not failing) |

Head + tail is the whole 97-file / 2527-test blast radius, and **the only
failures anywhere in it are the two pre-existing ones of sec 8.2**, reproduced
from a single import and proved identical with this fix neutralised.

Two liveness notes, because a green sweep is only worth what it exercised:

* `test_a_capped_pool_is_bit_identical_to_serial` **PASSED** (not skipped) in
  the ordered 97-file run -- verified in the per-test log.  The sec-8.1
  parent/worker split does not occur in that context; it needs the full tree,
  where a suite outside this blast radius triggers it.  So the serial arm was
  really asserted here, not skipped away.
* the 2 failures were located by ORDER, not guessed: the `-q` progress
  percentage was mapped back onto `--collect-only` output, and when that
  mis-pointed twice the run was redone with `-v` so the failures named
  themselves.  Two earlier estimates from percentages alone pointed at the
  wrong files.

The full 11 534-test tree was killed by the harness's background time limit
three times before completing, so **"the entire tree is green" is NOT
claimed** -- only the blast radius, which is the set of suites that can reach
this code.

### 7.1 Change to an existing suite

`tests/unit/test_niche_newton_pool_both_fits.py::_skip_if_low_ram` gains a
premise guard: every dispatch-count assertion in that file assumes the new
resource clamp does not bind, because on a small box the clamp legitimately
answers a pool-sized call with fewer workers.  The guard **asks the shipped
resolver** rather than inventing a GB threshold, so it tracks the model instead
of drifting from it.  On this box the clamp does not bind and all 23 tests run
for real (they pass).

---

### 7.2 Two structural pins fired on the first sweep, and they were right

`_MAIN_GUARD_CACHE` is a module-level `_*_CACHE` dict, which this repo has two
meta-walkers for:

* `test_v4_14_2_dispatcher_pin_cache_locks` -- every `_*_CACHE` needs a
  companion `_*_LOCK`;
* `test_v4_16_1_dispatcher_pin_cache_registry_enrollment` -- every cache-owning
  module must call `register_cache_clearer`.

Both were violated by the first cut and both are correct: `apply_real_lens_
traced` runs on a `ThreadPoolExecutor` whenever `parallel_amp` is on, so two
threads can reach the resolver at once, and the warn-once ledger's
`if x not in s: s.add(x)` is a torn read-modify-write whose entire contract is
"exactly once".  Fixed by complying rather than by renaming around the pins:
`_MAIN_GUARD_LOCK` guards both the verdict cache and the ledger (the AST parse
stays OUTSIDE the lock -- it is pure, so two threads racing it recompute the
same verdict instead of serialising on file I/O), and the module now enrols
`lens_traced_main_guard`, so `clear_all_registered_caches()` and
`lumenairy_context(clear_caches_on_exit=True)` reach the guard verdict like any
other cache.

---

## 8. WHAT THIS DOES NOT CLOSE

1. **`focus_scan_121.py` still has no main guard.**  It is out of scope for a
   library fix and the library now degrades correctly and says so -- but the
   pool speed-up on design 121 stays unavailable until someone adds the guard
   to the harness.  With it added, rule 1 stops firing and rule 2 (the memory
   cap) becomes the operative one; at Stage B's shape the model then approves
   all 8 workers on this box (16 GB projected against ~96 GB free).
2. **`fan_multi_121.py` was not touched** (out of scope by instruction).  It
   has the same shape, so Stage C inherits the same serial degradation.
3. **No measured pool speed-up for design 121 is claimed.**  The 8-worker run
   still never completes *as a pool* on this harness, so there is no pooled
   timing for this workload and none is quoted.  The library's own cold/warm
   tables put the pool at 1.10-1.53x on chain-sized problems.
4. **`_memory_bounded_n_fine` is still single-process.**  The clamp is on the
   pool side, which is the correct side (the fine grid is sized before the
   caller's `n_workers` is even in scope at that call site), but the asymmetry
   is now pinned by a test rather than left as a comment.
5. **The commit-vs-working-set asymmetry** (1.65 GB commit / 0.098 GB wset
   after import) means the model is conservative on Linux, where `available`
   and overcommit behave differently.  Measured on Windows only.

### 8.1 NEW FINDING (out of scope here, but real): serial == pool is CONDITIONAL

Found by this fix's own full-tree sweep, which failed
`test_a_capped_pool_is_bit_identical_to_serial` at `max|delta| = 1.830e-12`
after 1912 passing tests -- and would have failed
`test_niche_newton_pool_both_fits.py::test_pool_result_is_bit_identical_to_
serial` the same way had it been reached, because the exposure is identical.

Isolated and measured directly, one process, N=1024 / rs=2, the same chain both
ways:

```
parent _NUMBA_AVAILABLE=True   ->  serial == pool,  max|delta| 0.000e+00
parent _NUMBA_AVAILABLE=False  ->  DIFFER,          max|delta| 5.167e-14
```

**Mechanism.** `_newton_invert_chunk` rebuilds `_Cheb2DEvaluator` in a FRESH
interpreter, and `ev_value_and_grad` takes the `@njit(parallel=True,
fastmath=True)` Chebyshev kernel if numba resolves THERE, else a pure-xp
Vandermonde contraction.  The pickled payload carries `newton_fit`,
`fit_poly_order`, `fit_weights` and `newton_max_iters` -- but **no backend
flag**.  So a parent that has resolved a different evaluator than its workers
runs a different floating-point ORDER for the same mathematics, and the two
paths part company in the last bits.  This is the same class of gap as audit
E-H2's `newton_max_iters` (a resolved decision that did not travel), and the
parent's resolution is process state: `_NUMBA_KERNELS['cheb2d']` caches a
`None` permanently once `_load_numba()` has failed even once.

**Not fixed here, deliberately.** The remedy is one field in `_spline_data` plus
one gate in the worker, but it lives on the NUMERICAL path, and this change is a
resource-safety fix whose whole licence is that it cannot move a number.  It is
recorded here, reproduced in two lines (set
`_lens_traced._NUMBA_AVAILABLE = False` in the parent, run the same chain at
`n_workers=1` and `n_workers=4`, compare), and pinned in the test: arm 2 of
`test_a_capped_pool_is_bit_identical_to_serial` checks serial against the
UNCAPPED pool first and, on mismatch, skips with the measured delta and the
parent's resolved backend rather than blaming the clamp.  Verified to behave
that way by running the test with the parent gate forced off: it SKIPS with
`...already differs from serial by 5.167e-14 ... (parent _NUMBA_AVAILABLE=False,
cheb kernel=None)`, while the clamp's own contract still passes.

**The polluting suite was not identified.**  No test in `tests/` leaves
`_NUMBA_AVAILABLE` flipped (the two that touch it restore it, and both run
later), so the full-tree trigger is something that makes `_load_numba()` fail
once, or an equivalent parent/worker split in the lstsq backend -- the same BLAS
-threading sensitivity this repo already tracks.  Narrowing that is the follow-
up; the mechanism above is proved regardless of which door it is reached
through.

### 8.2 Pre-existing test pollution, proved and left alone

The sweep also failed
`test_niche_d8_congruence_workers.py::test_snapshot_is_picklable` and
`::test_a_clean_glass_snapshot_reports_nothing_unpicklable`.  Both reproduce
with a single import and have nothing to do with this fix:

```
pytest tests/unit/test_g1_cache_memory.py <the two d8 tests>
  -> AssertionError: assert {'_G1CACHE'} == set()
```

`test_g1_cache_memory.py:24` writes a module-scope `lambda` into the global
`GLASS_REGISTRY` at import time and never removes it; a lambda is unpicklable,
so `_multi_capture_worker_state()`'s snapshot stops round-tripping for the rest
of the session.  At least nine test modules do this (`_G1CACHE`, `_G2_A`,
`_G2_B`, `_H1_FIX_GLASS`, `_H2_DISP_GLASS`, `_H3_FIX_GLASS`, `_H6_FIX_GLASS`,
`_H7_FIX_GLASS`, `_K1CAU`, ...).  Confirmed independent of this change by
re-running the same pair with the fix NEUTRALISED to the pre-fix gate
(`_newton_resolve_workers = lambda requested, *a, **k: requested`): identical
2 failures.  Not touched -- it is a test-hygiene issue in suites this change
does not own.

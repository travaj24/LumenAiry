# PMM per-layer roadmap -- final gate

Date: 2026-08-05
Tree: `feat/pmm-per-layer-roadmap` @ `822e214` (working tree, uncommitted)
Baseline for the delta: `013c388`

Environments:

* Windows 11, Python 3.14.6, numpy 2.4.4, scipy 1.17.1, 24 logical CPUs
* WSL2 / Ubuntu, Python 3.12.3, numpy 2.4.6, scipy 1.17.1 (`~/lumvenv`)

Two items, both closed.  Only `lumenairy/propagators/carrier.py` changed
(107 insertions, 26 deletions); `CHANGELOG.md` untouched, nothing committed.

---

## 1. The seven broad `except Exception` sites in `carrier.py`

All seven arrived with niche D8's process-pool code and the D8/D9 follow-ups,
none of which existed at `013c388` (which had **zero** broad excepts in this
file).  Attribution by commit:

| Commit | Broad excepts in `carrier.py` |
| --- | --- |
| `013c388` (baseline) | 0 |
| `95a9849` D8 process parallelism | 1 (the pool block) |
| `e83854b` workers inherit runtime state | +4 |
| `87c6914` divide the RAM budget | +1 (6 total) |
| `0e3f66e` spawn pin + callable model glass | +1 (7 total) |

**The budget was NOT raised.**  `_NON_UI_EXCEPT_BUDGET` stays at 48
(`b821254` precedent).  The non-`ui/` count was 55 -- seven over -- and is now
exactly 48.  `carrier.py` is back to zero broad excepts.

### Per-site derivation

| # | Site | Guarded call | Narrowed to |
| --- | --- | --- | --- |
| 1 | `_multi_capture_worker_state` L6992 | `importlib.import_module(mod_name)` over `_WORKER_STATE_MODULES` | `ImportError` |
| 2 | `_multi_capture_worker_state` L7023 | `from ..memory import get_ram_budget`; `get_ram_budget() // max(1, int(n_workers))` | `(ImportError, OSError, RuntimeError)` |
| 3 | `_multi_unpicklable_glass` L7051 | `pickle.dumps(val)` | `(pickle.PickleError, AttributeError, TypeError, ValueError, RecursionError)` |
| 4 | `_multi_apply_worker_state` L7074 | `setattr(importlib.import_module(mod_name), n, val)` | `(ImportError, ValueError, AttributeError)` |
| 5 | `_multi_apply_worker_state` L7091 | `from .. import glass`; `tgt.update(table)` | `(ImportError, AttributeError, TypeError, ValueError)` |
| 6 | `_multi_apply_worker_state` L7103 | `from ..memory import set_max_ram`; `set_max_ram(int(...))` | `(ImportError, TypeError, ValueError)` |
| 7 | `_multi_parallel_results` L7338 | the whole process-pool block | **restructured** (below) + `(RuntimeError, OSError, MemoryError, ValueError, TypeError, AttributeError, pickle.PickleError)` |

Notes on the non-obvious ones:

* **#1** deliberately catches *only* `ImportError`.  The names in
  `_WORKER_STATE_MODULES` are all first-party, so the only legitimate reason
  one is absent is a trimmed install.  A module body that raises anything else
  is a real defect, and silently skipping it is precisely the
  silent-different-physics failure the capture exists to prevent.
* **#2** deliberately does **not** catch `TypeError`/`ValueError`.  Those can
  only come from `int(n_workers)` -- a caller bug, which must surface rather
  than degrade the budget to `None`.  `OSError`/`RuntimeError` are psutil's
  platform-layer memory probe under `get_ram_budget`.
* **#3** is the full documented refusal surface of `pickle.dumps`, verified on
  this build: `PicklingError` for a module-level lambda *and* for a closure
  (CPython 3.14 reports both that way; older CPythons raise `AttributeError`
  for the local-object case, hence its inclusion), `TypeError` for a C object
  that cannot be reduced (locks, generators, modules), `ValueError` for
  ctypes-with-pointers, `RecursionError` for a self-referential table.
* **#4** includes `ValueError` because a malformed flag key with no `':'`
  separator leaves an empty module name, which `import_module` rejects with
  `ValueError`, not `ImportError`.
* **#6** includes `ValueError` because the `if state.get('ram_budget'):`
  truthiness test lets a *negative* budget through, and `set_max_ram` rejects
  `<= 0` with `ValueError`.

### Site 7: why it needed a restructure, not just a tuple

The pool block could not be honestly narrowed as written.  A worker runs the
entire traced chain, so it can raise anything the library raises -- a set that
cannot be enumerated -- and `Executor.map` **re-raises** it into the parent
frame.  Any tuple written there would either have been `Exception` with extra
steps or silently incomplete (e.g. `np.linalg.LinAlgError` is not a
`ValueError` subclass).

The fix removes the need to catch it at all:

* `ex.map(...)` -> `futs = [ex.submit(...)]` plus `failure = fut.exception()`.
  `Future.exception()` **returns** the worker's exception as a value instead of
  raising it, so the untypeable half of the failure surface is handled without
  any `except` clause.
* The remaining `except` therefore only has to name the pool's own,
  parent-side modes: `RuntimeError` (`BrokenProcessPool`/`BrokenExecutor`, and
  submission after shutdown), `OSError` (process launch failure),
  `ValueError` (the executor rejecting `max_workers` -- this includes
  Windows' hard 61-worker cap), `PickleError`/`TypeError`/`AttributeError`
  (pickling the initializer payload, which `Process.start()` performs in the
  *parent* under spawn), and `MemoryError` (that payload not fitting).
* Both branches of the message site (spawn-bootstrap vs. generic) are
  unchanged, moved below the `try` and fed from a `failure` value so the two
  sources converge without one wrapping the other.
* `import multiprocessing`/`import pickle` were hoisted **above** the `try`:
  the except clause names `_pickle`, and clauses are evaluated at raise time,
  so binding it inside the guarded block would have turned any early failure
  into a `NameError`.
* An explicit `cancel()` loop replaces what `map`'s iterator teardown did for
  the queued futures.

Verified by two probes (scratchpad, not committed):

1. **End-to-end.**  A bogus `traced_kwargs` entry is forwarded verbatim into
   the per-congruence chain call, so it can only fail inside a worker.  Serial
   control raises `TypeError: apply_real_lens_traced() got an unexpected
   keyword argument`.  Under `congruence_workers=2` the pool raises
   `RuntimeError: ... a congruence worker failed under congruence_workers=2`
   with `__cause__` preserved as the original `TypeError`.  That path is
   `# pragma: no cover - env`, so nothing in the suite exercises it.
2. **Mechanism, submit vs. map head-to-head.**  A worker raising `KeyError` --
   a type deliberately *outside* the narrowed tuple -- comes back from
   `fut.exception()` as a value and is never raised.  Both arms consume the
   same results before the failure, return the same failure object, and pay
   the same teardown (43.0 s submit vs. 45.0 s map; neither can cancel an
   already-running future, so both wait out the in-flight stragglers).  No
   regression.

---

## 2. WSL confirmation of the already-fixed names

One targeted run of the three files on WSL.  See the green list below.

---

## Green list

| Suite | Windows (3.14.6) | WSL (3.12.3) |
| --- | --- | --- |
| `tests/unit/test_audit_except_budget.py` | 2 passed | 2 passed |
| `tests/unit/test_niche_d8_congruence_workers.py` (FULL file) | 27 passed, 350.6 s | 27 passed, 548.2 s |
| `tests/unit/test_audit_optimize.py` | -- | see run 2 |
| `tests/unit/test_m1_conditioning_guard.py` | -- | see run 2 |
| `tests/unit/test_niche_audit_w8_raster.py` | -- | see run 2 |
| `ruff check --no-cache .` | All checks passed | All checks passed |

The WSL D8 run is the second confirmation asked for: the three D8 tests that
failed on WSL before `0e3f66e` -- the fork/libgomp trap
(`test_pool_uses_spawn_never_the_platform_default_fork`) and the callable-glass
pickling pair (`test_a_callable_model_glass_is_detected_as_unpicklable`,
`test_unpicklable_glass_degrades_to_serial_and_says_so`) -- now pass on WSL,
so those fixes are confirmed on the build that exposed them.

## Tree state

```text
 M lumenairy/propagators/carrier.py
?? docs/audits/PMM_FINAL_GATE_2026_08_05.md
```

The one modified source file is this gate's narrowings; the untracked file is
this document.  Nothing else.  No commits, no tag, no `CHANGELOG.md` edit.

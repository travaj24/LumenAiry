# FIX -- the five E-H2 setup ERRORs after round 2, and the DLASCL line next to them

**2026-08-10.  Branch `perf/traced-hotpath`, on top of `0b4a385`
(`FIX_PERF_ROUND2_2026_08_10`).  Closes the `ERROR at setup of test_h2_*`
family that round 2 introduced across every unit-test shard and every Python
from 3.10 to 3.13.  Files edited:
`tests/unit/test_niche_audit_e_prepared_and_enums.py` (the E-H2 spy and one new
guard test) and one dangling cross-reference in
`lumenairy/elements/_lens_traced.py`.  NO library behaviour changes -- sec 5
gives the reason and the evidence.  `CHANGELOG.md` was not touched.**

---

## 0. HEADLINE

> **A TEST read the wire, round 2 moved the wire, and the read was never
> updated.**  Item 3 of round 2 stopped embedding the Newton payload in every
> chunk's arg tuple; the parent now pickles it ONCE and submits
> `(payload_key, blob_or_None, x_chunk, y_chunk)`.  The E-H2 executor spy still
> did `args[0].get('newton_max_iters', ...)`, and `args[0]` is now the 128-bit
> digest STRING.  `AttributeError: 'str' object has no attribute 'get'`, raised
> inside a module-scoped fixture, which pytest reports as an ERROR at setup of
> every test that fixture feeds -- FIVE tests, not the three first noticed.
>
> **It is not CI-only.**  It reproduces on this Windows box in 4.2 s and in WSL
> in 5.6 s, at the CI runner's own 2-core geometry AND at 24 cores.  Nothing
> about the runner is required: the wire shape changed unconditionally.
>
> **The library is correct and is not changed.**  The resolved cap still
> travels in the pickled payload -- that is precisely what the content digest
> is taken over -- so E-H2's mechanism holds; only the reader was stale.  Sec 5
> records what was checked before concluding that.
>
> **The DLASCL line is unrelated and pre-dates round 2.**  It appears in CI
> logs from 2026-08-06 and 2026-08-08, in jobs with zero E-H2 errors, and is
> ABSENT from the py3.10/shard-3 job of the round-2 run that DID have them.
> Sec 6.

---

## 1. THE TRACEBACK

From `ci_r2.log`, py3.11 shard 2/3 (identical in all nine affected jobs, modulo
the two dictcomp frames that 3.12+ elides):

```
___________ ERROR at setup of test_h2_pool_path_is_actually_engaged ____________
tests/unit/test_niche_audit_e_prepared_and_enums.py:156: in _pool_runs
    _pool_run(12)          # warm-up: the FIRST N=512 call in a process differs
tests/unit/test_niche_audit_e_prepared_and_enums.py:137: in _pool_run
    out = LT.apply_real_lens_traced(
lumenairy/elements/_lens_traced.py:10128: in apply_real_lens_traced
    opl_map = _invert_newton_parallel(
lumenairy/elements/_lens_traced.py:9650: in _invert_newton_parallel
    future_to_idx = {
lumenairy/elements/_lens_traced.py:9651: in <dictcomp>
    ex.submit(_newton_invert_chunk, (_pkey, _send, _xc, _yc)): i
tests/unit/test_niche_audit_e_prepared_and_enums.py:123: in submit
    seen['caps'].append(args[0].get('newton_max_iters', '<missing>'))
E   AttributeError: 'str' object has no attribute 'get'
```

The last library frame is the fix's own line.  `_pkey` is the payload's
`blake2b` digest hexdigest -- a `str`.

**Blast radius: five tests, one fixture.**  `_pool_runs` is
`scope='module'`, so every consumer errors:

| test | shard(s) it landed in |
| --- | --- |
| `test_h2_pool_path_is_actually_engaged` | 1 |
| `test_h2_resolved_cap_travels_in_the_pickled_payload` | 1 |
| `test_h2_pool_and_serial_share_one_newton_solution` | 1, 3 |
| `test_h2_newton_max_iters_is_live_on_the_pool_path` | 2, 3 |
| `test_h2_pool_emits_the_same_unconverged_warning_as_serial` | 2, 3 |

The three named in the bug report are the ones that happened to be sampled;
the split just distributes the same fixture's consumers by recorded duration.
The other 30 tests in the file passed on CI and pass here, which is the first
sign the fault is confined to the spy.

---

## 2. MECHANISM

Round 2 item 3 (`FIX_PERF_ROUND2_2026_08_10` sec 6.2) removed the
once-per-chunk pickling of the ~11 MB payload.  The submit call went from

```python
ex.submit(_newton_invert_chunk, (_spline_data, _xc, _yc))       # before
ex.submit(_newton_invert_chunk, (_pkey, _send, _xc, _yc))       # after
```

where `_send` is `pickle.dumps(_spline_data)` or `None` when the parent
believes the live workers already hold that digest.  `_newton_invert_chunk`
takes BOTH shapes (`if len(args) == 4:` at `_lens_traced.py:742`), which is why
the direct-call test `test_h2_worker_reads_the_cap_and_stays_backward_
compatible` -- and both pool suites, which spy only on the WORKER COUNT --
stayed green.  The one place in the tree that reads the arg tuple's head as a
payload dict is the E-H2 spy, and it was not re-anchored.

**Why "the cap travels in the pickled payload" is still the right pin.**  The
cap is inside `_spline_data`, the digest is taken over the wire bytes of that
dict, and any change to any field changes the key.  So the assertion in
`test_h2_resolved_cap_travels_in_the_pickled_payload` is unchanged and still
means what it says; the spy simply has to UNPICKLE the blob instead of reading
a dict it is no longer handed.

**Why the memo has to be module-global.**  A dispatch whose payload digests the
same as the resident one ships the key ALONE.  The pool, the parent's residency
belief (`_POOL_RESIDENT_PAYLOAD_KEY`) and therefore the key->bytes association
all outlive any single `_pool_run` call, so a per-run cache would go blind the
first time a payload repeats.  Measured on this branch at the CI runner's
2 cores, the fixture's own sequence never actually takes that path -- the cap
alternates 12 -> 1 -> 12 so consecutive digests differ and all six submits carry
the blob:

```
available_cpus 2
    ('warmup12', 'str', '7c6353de', blob_is_None=False)
    ('warmup12', 'str', '7c6353de', blob_is_None=False)
    ('pool1',    'str', 'f61ab761', blob_is_None=False)
    ('pool1',    'str', 'f61ab761', blob_is_None=False)
    ('pool12',   'str', '7c6353de', blob_is_None=False)
    ('pool12',   'str', '7c6353de', blob_is_None=False)
KEY-ONLY (blob is None) submits: 0
```

Note `7c6353de` recurring: the cap-12 payload IS re-dispatched, and only the
intervening cap-1 run kept it from being a key-only send.  Reorder the fixture
and it becomes one.  The memo is therefore not speculative, and the new guard
test exercises it directly.

---

## 3. THE FIX

`tests/unit/test_niche_audit_e_prepared_and_enums.py`, three changes, no
assertion in any of the five tests touched:

1. `_submitted_cap(args)` -- a reader that accepts both wire shapes: the
   historical `(knot_data, x, y)` and round 2's `(key, blob_or_None, x, y)`.
   On a blob it unpickles and memoises `key -> cap` in the module-global
   `_CAP_BY_PAYLOAD_KEY`; on a key-only send it answers from the memo.  Only
   the CAP is memoised, never the payload, so this holds bytes and not the
   ~2 MB of fit grids.
2. `_ExecutorSpy.submit` calls it instead of `args[0].get(...)`.
3. A new `test_h2_cap_reader_tracks_every_wire_shape_the_worker_accepts`, which
   needs no pool (so it runs on a one-core box) and pins the reader against the
   library's OWN producer, `LT._newton_payload_blob`: both shapes, the key-only
   re-send, the keyless pre-5.29.1 payload, and the unseen-key case.

A key-only send for a digest whose bytes were never seen returns a
self-describing sentinel rather than raising.  That is deliberate: the failure
this document exists to fix was an EXCEPTION inside a module-scoped fixture,
which converts one broken read into five setup ERRORs and hides which pin
actually regressed.  A residency regression now fails
`test_h2_resolved_cap_travels_in_the_pickled_payload` by name, with the
offending key in the message, and leaves the other four tests reporting.

---

## 4. FAIL-BEFORE AND GREEN

**Fail-before** -- the pristine `0b4a385` file, run from git, at the CI
runner's 2-core geometry on both mounts.  Windows uses `PYTHON_CPU_COUNT=2`
(honoured by `os.process_cpu_count`, which is `available_cpus`' first
preference); WSL uses `taskset -c 0,1` (honoured by `sched_getaffinity`, its
second):

```
Windows py3.14, PYTHON_CPU_COUNT=2 : 30 passed, 5 errors in 6.72s
WSL     py3.12, taskset -c 0,1     : 30 passed, 5 errors in 5.64s
```

All five are `AttributeError: 'str' object has no attribute 'get'` at line 123
-- the CI failure exactly, and the same 30 passes beside it.

**Green after** (`-p no:randomly`; every run below is the fixed tree):

| what | mount | cores | result |
| --- | --- | --- | --- |
| `-k h2` (5 fixture tests + 2 unit) | Windows py3.14 | 24 | 7 passed |
| `-k h2` | WSL py3.12 | 2 (`taskset`) | 7 passed |
| whole E-H2 file | Windows py3.14 | 2 (`PYTHON_CPU_COUNT`) | 36 passed |
| whole E-H2 file | Windows py3.14 | 24 | 36 passed |
| E-H2 file + `test_fix_newton_pool_memory` + `test_niche_newton_pool_both_fits` + `test_niche_perf_round2_2026_08_10` | WSL py3.12 | 2 (`taskset`) | 130 passed, 3 skipped |
| the three pool suites | Windows py3.14 | 2 (`PYTHON_CPU_COUNT`) | 97 passed |
| the same four files | Windows py3.14 | 24 | 133 passed |

The 3 WSL skips are `threadpoolctl absent` in that venv, not new.

**Spawn-verified.**  The pool these tests exercise is a real spawn pool, not an
in-process stand-in:

```
executor:    ProcessPoolExecutor
mp_context:  SpawnContext spawn
initializer: <function _newton_pool_init>
worker pids: [64]   parent: 11
```

That matters for the two pins that would otherwise be vacuous:
`test_h2_pool_and_serial_share_one_newton_solution` compares a POOL result to a
SERIAL one at 1e-11, and `test_h2_pool_emits_the_same_unconverged_warning_as_
serial` requires the warning to have crossed back from a worker.

---

## 5. WHY THE LIBRARY IS NOT CHANGED

The residency path shipped in round 2 was read for a genuine degenerate-input
or 2-core-geometry defect before concluding the fault was the spy's.  What was
checked, and why each is sound:

* **The cap still reaches the worker.**  It is a field of `_spline_data`, the
  blob is `pickle.dumps` of that dict, and `_newton_invert_chunk` reads
  `knot_data.get('newton_max_iters', _NEWTON_MAX_ITERS)` after resolving the
  payload.  `test_h2_newton_max_iters_is_live_on_the_pool_path` (cap 1 vs 12
  must differ) and `test_h2_pool_and_serial_share_one_newton_solution`
  (pool must equal serial at both caps) both pass, which is the end-to-end
  proof that the shipped path honours it.
* **Residency cannot return a stale answer.**  The key is a digest of the wire
  bytes, so a changed backend pin, a changed built fit or a changed cap changes
  the key; a worker asked for a key it lacks raises `NewtonPayloadNotResident`
  and the parent re-submits that chunk WITH the blob.  Worst case is a round
  trip.
* **The 2-core geometry is not special.**  `n_cpu=2` gives two chunks and two
  workers; the probe in sec 2 was run at exactly that width and shows a
  well-formed dispatch.  The failure reproduces identically at 24 cores.
* **No degenerate matrix reaches the worker.**  With `cheb_fit` shipped
  (v5.33.0) the worker EVALUATES the parent's coefficients and fits nothing, so
  there is no least-squares call on the pool path to feed a NaN-sized or
  zero-dim matrix to LAPACK.  See sec 6 for what the DLASCL line actually is.
* **The other two spies were audited.**  `_WidthSpy` in
  `test_fix_newton_pool_memory.py` and the pool spy in
  `test_niche_newton_pool_both_fits.py` wrap `_get_persistent_worker_pool`
  only, record `n_workers`, and never touch the arg tuple -- which is why they
  were green on CI throughout and are green here.

One edit was made in `_lens_traced.py`, and it is documentation only: the ARG
SHAPE note in `_newton_invert_chunk`'s docstring said "See
`_newton_payload_for_worker`", a symbol that exists nowhere in the tree (`grep`
over the whole repo returns that one line).  The function is
`_newton_worker_payload`.  Corrected, so the next reader of this protocol lands
on it.  No test pins that string and no behaviour depends on it.

---

## 6. THE DLASCL LINE IS A DECOUPLED, PRE-EXISTING CONDITION

```
 ** On entry to DLASCL parameter number  4 had an illegal value
```

Parameter 4 of `DLASCL` is `CFROM`; LAPACK sets `INFO = -4` when it is zero or
NaN, i.e. a scale factor derived from a matrix norm came back degenerate.  It
is a `xerbla` diagnostic printed to stderr, not an exception -- nothing fails
because of it, and it is not one of the ERRORs.

**It is not caused by round 2, and not by E-H2.**  Three independent
observations, all from the logs already on this box:

1. It appears in `ci_failed_full.log` (2026-08-06) and `ci2_failed.log`
   (2026-08-08), both BEFORE `0b4a385`.  In those runs it lands in jobs --
   "JAX unit tests (Python 3.12)", `shard 3/3`, `shard 1/3` -- that have zero
   E-H2 errors.
2. In the round-2 run it appears ONLY in the shard-2 jobs (py3.11 / 3.12 /
   3.13).  The py3.10 shard-3 job has the E-H2 setup ERRORs and NO DLASCL line.
   The two do not co-occur.
3. It is always exactly TWO lines per job, in runs whose E-H2 error count is 0,
   2 or 3.  The count does not track the errors.

**Why it prints after the summary.**  `xerbla` writes to C-level stderr.  Under
pytest's fd-level capture that fd is a regular file, so libc buffers it FULLY
instead of line-by-line; the text sits in the buffer until the process exits,
by which time capture has been torn down and fd 2 points at the raw job stream
again.  So the position in the log carries no information about WHEN the call
happened -- it is not, as the ordering suggests, a teardown-time event.

**Not chased further, deliberately.**  It is stderr noise from some
least-squares or eigen call elsewhere in the shard (the tree has ~20
`lstsq` sites), it predates this branch, it fails nothing, and identifying it
means running a full 25-minute shard to bisect a message that carries no
traceback.  It is recorded here so the next reader does not re-attach it to the
pool path: the E-H2 ERRORs are fully explained by sec 2, and they are fully
fixed without touching it.

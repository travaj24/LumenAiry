"""VERIFY WP-B13 FOLLOW-UPS -- the decisions the follow-ups package argues
for in prose but does not pin, and the gaps its own pins leave open.

Companion to `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/
VERIFY_WP-B13_FOLLOWUPS.md`.  Every claim below was MEASURED first, on
Windows 3.14.6 and WSL 3.12.3, by the probes in
`validation/probe_verify_b13_followups/`; the numbers and dates in the
docstrings are those measurements.

What each test closes, and why the branch's own pin does not:

D1  `test_the_wedge_report_names_a_frame_the_branch_pin_never_parks_in`
    the branch pins ONE wedge (a daemon thread on `threading.Event.wait`, in
    one named frame).  A dump helper that worked only for that shape would
    pass it.  This parks in a different frame through a different mechanism
    (`queue.Queue.get`).

D2  `test_the_census_survives_the_caller_first_lock_order`
    the branch pins the HELPER-first arrival order.  The other order is a
    different code path (the caller appends, the helper removes) and is not
    pinned anywhere.
    `test_a_helper_that_raises_an_uncaught_class_still_publishes_and_clears`
    `_shutdown_pool_bounded`'s helper catches only
    `(RuntimeError, OSError, ValueError)`.  Everything else rides the
    `finally`, and nothing pins that the `finally` is load-bearing.
    `test_one_executor_never_reaches_both_teardown_mechanisms`
    the helper's `remove` is UNCONDITIONAL, so it would drop an entry that
    `_abandon_pool` put there.  That is harmless only because no library path
    gives one executor to both; this pins that precondition.

D3  `test_a_pool_wider_than_the_dispatch_holds_no_surplus_processes`
    the docstring's second cost bullet says the surplus of a wider pool is a
    worker that ran the initializer and never got a chunk.  CPython spawns
    LAZILY, so that worker does not exist.  MEASURED 2026-09-19: a 12-wide
    pool holds 0 processes until submitted to, and 4 after a 4-chunk
    dispatch, on both builds.

D4  `test_nothing_consumes_the_in_flight_counters_magnitude`
    the branch pins every `ast.Compare` against `_POOL_INFLIGHT`.  The
    counter also leaves the module as `_note_pool_inflight`'s RETURN VALUE,
    which no Compare can see.

D5  `test_a_timeout_on_the_dispatch_must_name_its_own_exception_class`
    `concurrent.futures.TimeoutError` became an alias of the builtin (and so
    an `OSError`) only in Python 3.11; `pyproject.toml` declares
    `requires-python = ">=3.10"`.  A `timeout=` added to the dispatcher
    without naming the class in the except tuple would escape the
    infrastructure clause on a supported interpreter.
    `test_a_bootstrap_bar_would_not_bound_a_chunk_that_wedges_later`
    PREMISE GATE for the open D5: a sentinel ahead of the chunks leaves the
    exposure open, which the recommendation should say out loud.

D7  `test_no_module_uses_an_executor_teardown_the_join_detector_cannot_see`
    the extended detector still misses two `with` shapes (an aliased class,
    and `with ex:` on a pre-built executor).  That is harmless only while no
    module uses them; this pins that, with a positive control.

Anything that drives a real process pool runs in a CHILD under
`subprocess.run(timeout=...)`, so no test here can hang the suite.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import queue
import subprocess
import sys
import threading
import time

import pytest

from lumenairy.elements import _lens_traced as LT

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          '..', '..'))
_FIXFILE = os.path.join(os.path.dirname(__file__),
                        'test_fix_newton_pool_broken_fallback.py')


def _fix_module():
    """The branch's own test module.

    The module pytest ALREADY imported is preferred, and that matters: a
    fresh private copy would be immune to anything the session did to the
    real one, so a mutation-matrix arm that puts the pre-fix ``_thread_dump``
    back would leave these tests green and read as a gap that is not there.
    Only if pytest has not imported it (someone importing this file directly)
    is a private copy loaded.
    """
    for mod in list(sys.modules.values()):
        f = getattr(mod, '__file__', None)
        if f and os.path.abspath(f) == os.path.abspath(_FIXFILE)                 and hasattr(mod, '_thread_dump'):
            return mod
    import importlib.util
    spec = importlib.util.spec_from_file_location('_b13_fixmod', _FIXFILE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('_b13_fixmod', mod)
    spec.loader.exec_module(mod)
    return mod


def _run_child(tmp_path, name, body, timeout):
    """Run ``body`` as a child script; FAIL (never hang) if it overruns."""
    script = tmp_path / name
    script.write_text(body, encoding='utf-8')
    env = dict(os.environ)
    env['PYTHONPATH'] = _REPO_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[var] = '1'
    try:
        proc = subprocess.run([sys.executable, str(script)],
                              capture_output=True, text=True,
                              timeout=timeout, env=env)
    except subprocess.TimeoutExpired as exc:
        got = exc.stdout
        if isinstance(got, bytes):
            got = got.decode('utf-8', 'replace')
        pytest.fail(
            f'{name} did not finish within {timeout} s -- a hang, which is '
            f'the pathology this file is about, not slowness.\n'
            f'stdout so far:\n{(got or "")[-2000:]}')
    tail = [ln for ln in proc.stdout.splitlines() if ln.startswith('JSON ')]
    assert tail, (f'{name} produced no JSON line.\nrc={proc.returncode}\n'
                  f'stdout:\n{proc.stdout[-3000:]}\n'
                  f'stderr:\n{proc.stderr[-3000:]}')
    return json.loads(tail[-1][len('JSON '):]), proc


# ===========================================================================
# D1 -- the wedge report, with a wedge the branch's pin never produces
# ===========================================================================
def _vb13_parked_on_a_queue_get(q):
    """A uniquely named frame that parks on ``Queue.get()``.

    Deliberately NOT the branch's ``_b13_wedged_frame_for_the_dump`` and
    deliberately not an ``Event.wait``: the claim is that the report names
    WHATEVER frame is stuck.
    """
    return q.get()


def test_the_wedge_report_names_a_frame_the_branch_pin_never_parks_in():
    """A second, independent wedge shape must also reach the dump.

    MEASURED 2026-09-19 (``vf1_d1_dump.py``, both builds): with the branch's
    ``tempfile`` helper the failure carries ``_with_deadline``'s own message
    AND ``_vf1_parked_on_a_queue_get``; with the shipped ``io.StringIO``
    helper rebound in memory it carries neither and raises
    ``io.UnsupportedOperation: fileno`` instead.

    The premise -- what ``io.StringIO`` does on the running build -- is
    measured here and REPORTED, never asserted: a CPython that grew a
    StringIO path would make the fix redundant, not wrong.
    """
    import faulthandler
    import io
    fm = _fix_module()

    q = queue.Queue()
    try:
        with pytest.raises(pytest.fail.Exception) as err:
            fm._with_deadline(lambda: _vb13_parked_on_a_queue_get(q), 1.0,
                              'a VERIFY-B13 queue wedge')
    finally:
        q.put(None)
    msg = str(err.value)

    try:
        faulthandler.dump_traceback(file=io.StringIO(), all_threads=True)
        premise = 'io.StringIO accepted the dump on this build'
    except Exception as exc:                       # noqa: BLE001 -- reported
        premise = f'io.StringIO raised {type(exc).__name__}: {exc}'

    assert 'a VERIFY-B13 queue wedge did not return within' in msg, (
        f'the wedge report lost _with_deadline\'s own message on a wedge '
        f'shape the branch\'s own pin does not produce.  Premise on this '
        f'build: {premise}.  Got:\n{msg}')
    assert '_vb13_parked_on_a_queue_get' in msg, (
        f'the report does not name the frame that is stuck when the park is '
        f'a Queue.get rather than an Event.wait, so the dump helper is '
        f'shape-dependent.  Premise: {premise}.  Got:\n{msg}')


def test_the_dump_helper_reaches_a_thread_that_is_not_the_caller():
    """``all_threads=True`` is the whole point, so decide it with two threads.

    A dump taken with only the calling thread alive cannot distinguish a
    working ``all_threads`` from a broken one.  MEASURED 2026-09-19: 2 thread
    headers on both builds, the parked frame named.
    """
    fm = _fix_module()
    q = queue.Queue()
    side = threading.Thread(target=_vb13_parked_on_a_queue_get, args=(q,),
                            name='verify-b13-side', daemon=True)
    side.start()
    try:
        time.sleep(0.2)
        dump = fm._thread_dump()
    finally:
        q.put(None)
    headers = dump.count('Thread 0x') + dump.count('Current thread 0x')
    assert headers >= 2, (
        f'the dump names {headers} thread(s); with a second thread parked it '
        f'must name at least two, or all_threads=True is not working and a '
        f'wedge on a daemon thread would produce nothing useful.\n{dump}')
    assert '_vb13_parked_on_a_queue_get' in dump, (
        f'the dump does not reach the OTHER thread\'s frame, which is the '
        f'only frame a wedge report ever needs.\n{dump}')


# ===========================================================================
# D2 -- the census, in the orders and on the paths the branch does not pin
# ===========================================================================
class _JoiningStub:
    """Joins forever until released.  No child processes involved."""

    def __init__(self, raise_cls=None):
        self._gate = threading.Event()
        self._raise = raise_cls
        self.shutdown_calls = []

    def shutdown(self, wait=True, **kw):
        self.shutdown_calls.append(bool(wait))
        if wait:
            self._gate.wait(60.0)
        if self._raise is not None:
            raise self._raise('engineered: the helper does not catch this')

    def release(self):
        self._gate.set()


class _CallerFirstLock:
    """The census lock, made to admit the CALLER before the helper.

    The mirror image of the branch's ``_HelperFirstLock``.  ``threading.Lock``
    makes no fairness promise, so neither order can be obtained by sleeping;
    both have to be built.
    """

    def __init__(self, real, patience=20.0):
        self._real = real
        self._patience = patience
        self._caller_done = threading.Event()

    @staticmethod
    def _is_helper():
        return threading.current_thread().name == 'lumenairy-newton-pool-close'

    def acquire(self, *a, **kw):
        if self._is_helper():
            self._caller_done.wait(self._patience)
        return self._real.acquire(*a, **kw)

    def release(self):
        caller = not self._is_helper()
        out = self._real.release()
        if caller:
            self._caller_done.set()
        return out

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def _settle(pred, seconds=15.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def test_the_census_survives_the_caller_first_lock_order(monkeypatch):
    """The OTHER arrival order, which the branch's pin does not cover.

    Caller first is the ordinary expiry path taken to its extreme: the
    caller appends while the helper is held out of the census entirely, and
    the helper must then find and remove its own entry.  MEASURED 2026-09-19
    (``vf2_d2_census.py``, 8 reps, both builds): 0 leaks under this order
    for the shipped ordering and for the one-line alternative, 8/8 leaks for
    the pre-fix code.  The two orders are different code paths -- under
    helper-first the caller returns True and never appends -- so pinning one
    says nothing about the other.
    """
    before_len = len(LT._ABANDONED_POOLS)
    before_timeouts = LT._POOL_SHUTDOWN_TIMEOUTS
    monkeypatch.setattr(LT, '_ABANDONED_POOLS_LOCK',
                        _CallerFirstLock(threading.Lock()))
    pool = _JoiningStub()
    box = {}

    def _call():
        box['ret'] = LT._shutdown_pool_bounded(pool, 0.05)

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    time.sleep(0.30)                    # the 0.05 s bound has expired
    assert _settle(lambda: pool in LT._ABANDONED_POOLS, 10.0), (
        'the caller did not put the outstanding teardown in the census, so '
        'nothing can see that this process is holding a dead executor')
    pool.release()
    t.join(30.0)
    assert not t.is_alive(), (
        'the bounded teardown never returned under the caller-first order')
    assert box['ret'] is False, (
        'a teardown that really did overrun its bound was not reported as '
        'an expiry')
    assert _settle(lambda: pool not in LT._ABANDONED_POOLS), (
        'the helper did not remove its own entry under the caller-first '
        'arrival order: the census pins the dead executor for the life of '
        'the process, which is defect D2 in its original form')
    assert len(LT._ABANDONED_POOLS) == before_len, (
        f'census length {len(LT._ABANDONED_POOLS)}, was {before_len}')
    assert LT._POOL_SHUTDOWN_TIMEOUTS == before_timeouts + 1, (
        'a real expiry must still be counted -- the monotone counter is the '
        'diagnostic the census deliberately is not')


@pytest.mark.parametrize('raiser', ['MemoryError', 'BaseException'])
def test_a_helper_that_raises_an_uncaught_class_still_publishes_and_clears(
        raiser):
    """The helper's ``finally`` is load-bearing, and nothing pinned it.

    ``_shutdown_pool_bounded``'s helper catches only
    ``(RuntimeError, OSError, ValueError)``.  A ``MemoryError`` from a
    teardown on an exhausted box -- or anything deriving from
    ``BaseException`` -- rides the ``finally``, which is the only thing that
    publishes ``done`` and removes the entry.  If the publish were moved out
    of the ``finally``, the caller would wait its whole bound, count an
    expiry AND leak the executor, and every existing pin would stay green.

    MEASURED 2026-09-19 (``vf2_d2_census.py``, three uncaught classes, three
    reps each, both builds): the census returns to empty every time.
    """
    cls = MemoryError if raiser == 'MemoryError' else type(
        '_VB13Uncaught', (BaseException,), {})
    before_len = len(LT._ABANDONED_POOLS)
    pool = _JoiningStub(raise_cls=cls)
    try:
        ok = LT._shutdown_pool_bounded(pool, 0.2)
        assert ok is False, 'the expiry was not reported'
        assert pool in LT._ABANDONED_POOLS, (
            'the outstanding teardown is not in the census')
    finally:
        pool.release()
    assert _settle(lambda: pool not in LT._ABANDONED_POOLS), (
        f'the helper raised {cls.__name__}, which it does not catch, and the '
        f'executor stayed in the census -- the removal is not on the '
        f'``finally`` path')
    assert len(LT._ABANDONED_POOLS) == before_len


def test_one_executor_never_reaches_both_teardown_mechanisms():
    """Why the helper's UNCONDITIONAL ``remove`` is safe today.

    ``_shutdown_pool_bounded``'s helper removes ``ex`` from the census
    whether or not its caller ever appended it.  If the same executor were
    also in the census because ``_abandon_pool`` put it there, a bounded
    teardown that COMPLETED INSIDE ITS BOUND would silently drop the
    reaper's entry and the census would under-report an outstanding
    teardown.  DRIVEN 2026-09-19 (``vf2_d2_census.py``
    ``attack_unconditional_remove``): it does exactly that.

    It is harmless only because no library path hands one executor to both
    mechanisms, and that is a property of the source, not of the helper.
    This pins it: in ``close_worker_pool`` the two calls are the two arms of
    one ``if``/``else``, and ``_get_persistent_worker_pool`` retires its
    stale pool through ``_abandon_pool`` alone.

    If this ever goes red, the repair is three lines in
    ``_shutdown_pool_bounded``: have the caller record under
    ``_ABANDONED_POOLS_LOCK`` that it appended, and have the helper remove
    only then.
    """
    src = inspect.getsource(LT)
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree)
           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name in ('close_worker_pool', '_get_persistent_worker_pool'):
        assert name in fns, f'{name} is gone; this pin needs restating'

    def _calls(node):
        return [c for c in ast.walk(node)
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]

    # close_worker_pool: the two teardowns must be mutually exclusive arms.
    cw = fns['close_worker_pool']
    exclusive = False
    for node in ast.walk(cw):
        if not isinstance(node, ast.If):
            continue
        body = {c.func.id for c in _calls(ast.Module(body=node.body,
                                                     type_ignores=[]))}
        orelse = {c.func.id for c in _calls(ast.Module(body=node.orelse,
                                                       type_ignores=[]))}
        if ('_abandon_pool' in body and '_shutdown_pool_bounded' in orelse) \
                or ('_shutdown_pool_bounded' in body
                    and '_abandon_pool' in orelse):
            exclusive = True
    assert exclusive, (
        'close_worker_pool no longer routes a pool to EXACTLY ONE teardown '
        'mechanism.  _shutdown_pool_bounded\'s helper removes from '
        '_ABANDONED_POOLS unconditionally, so a pool that reaches both '
        'mechanisms can have an outstanding _abandon_pool entry dropped by '
        'the bounded helper.')

    gp = fns['_get_persistent_worker_pool']
    gp_names = {c.func.id for c in _calls(gp)}
    assert '_shutdown_pool_bounded' not in gp_names, (
        'the pool constructor now also runs a bounded teardown; check that '
        'it cannot be the same executor _abandon_pool already retired')


# ===========================================================================
# D3 -- what a pool wider than the dispatch actually holds
# ===========================================================================
_LAZY_SPAWN_CHILD = r'''
import json, os, sys, time, warnings
import numpy as np
import psutil
import lumenairy as la
from lumenairy.elements import _lens_traced as LT

# The `if __name__ == '__main__'` guard below is LOAD-BEARING, not decoration:
# `_newton_resolve_workers` refuses workers outright when this process's
# __main__ is unguarded, because a spawn worker would re-execute it.  Without
# it the dispatch silently goes serial and the measurement is vacuous (which
# is why `chunks_submitted` is asserted in the test).
LT._POOL_MIN_PIXELS = 1
LT._POOL_MIN_PIXELS_WARM = 1

# MUTATION-MATRIX HOOK.  Tests that drive a real pool run in a CHILD, and a
# child is a fresh interpreter: an in-memory regression injected into the
# PARENT is invisible to it.  `validation/probe_verify_b13_followups/
# vf8_mutations.py` therefore passes its choice down in VF8_MUTATE, and the
# child re-applies it here.  Outside that probe the variable is unset and
# this block does nothing.
_MUT = os.environ.get('VF8_MUTATE', '')
if _MUT:
    try:
        import vf8_mutations as _VF8
        _VF8.pytest_configure(None)
    except Exception as _exc:          # noqa: BLE001 -- reported, not fatal
        print('child could not apply VF8_MUTATE=%s: %s' % (_MUT, _exc))

# FORCE THE PRECONDITION (docs/TESTING_STANDARDS.md rule 4): the resource
# clamp legitimately refuses workers on a loaded box, which would send the
# dispatch serial and make the measurement vacuous.  The clamp is not what
# is under test here, so it is forced; the tests assert the number of chunks
# actually submitted, which is the gate.
_REAL_RESOLVE = LT._newton_resolve_workers
LT._newton_resolve_workers = lambda n_req, *a, **kw: max(1, int(n_req))


def presc(ap=3e-3, r=9e-3):
    return {'name': 'fast_singlet', 'aperture_diameter': ap,
            'thicknesses': [3e-3],
            'surfaces': [
                {'radius': r, 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'conic': 0.0, 'aspheric_coeffs': None},
                {'radius': -r, 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'conic': 0.0, 'aspheric_coeffs': None}]}

def traced(nw, N):
    ap = 3e-3; dx = 2.2 * ap / N
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X**2 + Y**2) / 1.2e-3**2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            E0, prescription=presc(ap), wavelength=1.31e-6, dx=dx,
            ray_subsample=2, newton_fit='spline', n_workers=nw,
            parallel_amp=False, newton_amp_mask_rel=0.0,
            on_undersample='silent', on_aperture_beam='silent'))

class Counting:
    def __init__(self, inner):
        self._inner = inner
        self.submits = 0
    def submit(self, fn, *a, **kw):
        self.submits += 1
        return self._inner.submit(fn, *a, **kw)
    def __getattr__(self, k):
        return getattr(self._inner, k)

def main():
    WIDE, NARROW, N = 8, 3, 256
    LT.close_worker_pool()
    ref = traced(1, N)
    ex = LT._get_persistent_worker_pool(WIDE)
    time.sleep(0.4)
    after_construct = len(getattr(ex, '_processes', {}) or {})
    seen = {}
    _real_get = LT._get_persistent_worker_pool
    def _wrapped(nw):
        c = Counting(_real_get(nw))
        seen['proxy'] = c
        return c
    LT._get_persistent_worker_pool = _wrapped
    try:
        got = traced(NARROW, N)
    finally:
        LT._get_persistent_worker_pool = _real_get
    time.sleep(0.4)
    ex2 = LT._PERSISTENT_POOL
    after_dispatch = len(getattr(ex2, '_processes', {}) or {})
    rss = []
    for pid in list(getattr(ex2, '_processes', {}) or {}):
        try:
            rss.append(psutil.Process(int(pid)).memory_info().rss / 2**20)
        except Exception:
            pass
    out = {'pool_width': WIDE, 'dispatch_workers': NARROW,
           'chunks_submitted': seen['proxy'].submits if 'proxy' in seen else 0,
           'pool_reused': ex2 is ex,
           'pool_nworkers': LT._PERSISTENT_POOL_NWORKERS,
           'processes_after_construct': after_construct,
           'processes_after_dispatch': after_dispatch,
           'served_rss_MB_max': round(max(rss), 2) if rss else None,
           'identical': bool(np.array_equal(got, ref)),
           'max_delta': float(np.nanmax(np.abs(got - ref))),
           'abandoned_len': len(LT._ABANDONED_POOLS)}
    LT.close_worker_pool()
    print('JSON ' + json.dumps(out))

if __name__ == '__main__':
    main()
'''


def test_a_pool_wider_than_the_dispatch_holds_no_surplus_processes(tmp_path):
    """The docstring's "surplus worker" does not exist.

    ``_get_persistent_worker_pool``'s cost paragraph names two states and
    calls the second -- "a worker that has only run ``_newton_pool_init``
    and never been given a chunk" -- "what the SURPLUS of a pool wider than
    the clamp is".  CPython's ``ProcessPoolExecutor`` spawns LAZILY
    (``_adjust_process_count`` runs inside ``submit``), so a pool built at
    width W holds ZERO processes until W tasks have been submitted, and a
    narrower dispatch on a wider pool leaves the surplus as nothing at all.

    MEASURED 2026-09-19 (``vf3_d3_footprint.py``, both builds, 12-wide pool,
    4-chunk dispatch): 0 processes after construction, 4 after the dispatch;
    the 8 "never-served" workers of the branch's own table exist only
    because its labelling step submitted to them.

    Both sides are asserted, because only the pair is the claim: the pool's
    recorded width really is the wider one (so this is not a pool that was
    never built), and the process count really is the narrower one.
    """
    out, _proc = _run_child(tmp_path, 'vb13_lazy_spawn.py',
                            _LAZY_SPAWN_CHILD, timeout=180)
    assert out['chunks_submitted'] >= 1, (
        f'the dispatch never reached the pool ({out}); the size bars or the '
        f'worker clamp sent it serial, so nothing about surplus workers is '
        f'being measured here')
    assert out['pool_reused'] is True, (
        'the narrow dispatch rebuilt the pool instead of reusing it, so this '
        'is not the state the ceiling rule leaves behind')
    assert out['pool_nworkers'] == out['pool_width'], (
        f"the pool's recorded width is {out['pool_nworkers']}, not "
        f"{out['pool_width']}: the rule under test (never shrink) did not "
        f'apply, so the surplus claim is not being measured')
    assert out['processes_after_construct'] == 0, (
        f"a pool constructed at width {out['pool_width']} already holds "
        f"{out['processes_after_construct']} worker processes.  CPython used "
        f'to spawn them lazily, which is why the surplus of a wider pool '
        f'costs nothing until it is used; if that changed, the cost '
        f'paragraph in _get_persistent_worker_pool has to be re-derived')
    assert out['processes_after_dispatch'] == out['chunks_submitted'], (
        f"after a {out['chunks_submitted']}-chunk dispatch on a "
        f"{out['pool_width']}-wide pool the pool holds "
        f"{out['processes_after_dispatch']} processes.  One process per "
        f'submitted chunk is what the lazy spawn produces; the '
        f'surplus workers the docstring prices are not there, and the state '
        f'that actually persists is a worker that SERVED a chunk')
    assert out['identical'] is True and out['max_delta'] == 0.0, (
        f'the pooled field is not the serial field: {out}')
    assert out['abandoned_len'] == 0, (
        f"the census did not come back empty: {out['abandoned_len']}")


# ===========================================================================
# D4 -- the counter's magnitude leaves the module by a route no Compare sees
# ===========================================================================
def test_nothing_consumes_the_in_flight_counters_magnitude():
    """The D4 decision rests on the CONSUMERS, so enumerate them properly.

    The branch's pin walks every ``ast.Compare`` whose operands include the
    Name ``_POOL_INFLIGHT`` and requires it to be against zero.  That is one
    of three reads in the module: the other two are the counter's own update
    (``_POOL_INFLIGHT = max(0, _POOL_INFLIGHT + int(delta))``) and
    ``_note_pool_inflight``'s ``return _POOL_INFLIGHT``, which hands the
    MAGNITUDE to every caller.  A consumer that did
    ``if _note_pool_inflight(0) > 1:`` would be a magnitude read that no
    Compare on ``_POOL_INFLIGHT`` can see, and the branch's pin would stay
    green.

    So the durable statement is about the call sites: every
    ``_note_pool_inflight(...)`` in the library is an expression STATEMENT,
    i.e. its return value is discarded.  MEASURED 2026-09-19
    (``vf6_d4_readers.py``, grep + AST over all of ``lumenairy/``, identical
    on both builds): 4 call sites, 0 of them consuming the value; 1 claim,
    3 releases.
    """
    src = inspect.getsource(LT)
    tree = ast.parse(src)
    consumed, statements = [], 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Name) \
                and node.value.func.id == '_note_pool_inflight':
            statements += 1
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == '_note_pool_inflight'):
            continue
        consumed.append(node.lineno)
    assert consumed, (
        'nothing calls _note_pool_inflight any more; the rebuild rule has '
        'lost the claim that keeps a rebuild off a live pool')
    assert statements == len(consumed), (
        f'{len(consumed) - statements} of {len(consumed)} '
        f'_note_pool_inflight call sites use the RETURN VALUE (lines '
        f'{consumed}).  That value is the magnitude of a DISPATCH count, and '
        f'the D4 decision -- fix the comment, not the counter -- is only '
        f'sound while nothing reads it.  A consumer that needs a chunk count '
        f'has to change the counter, not read this one.')

    # ... and the getter's guard really is the zero-vs-non-zero read the
    # decision names, stated here as a decision rather than as a count of
    # Compare nodes.
    names = {n.id for n in ast.walk(tree)
             if isinstance(n, ast.Name) and n.id == '_POOL_INFLIGHT'}
    assert names, 'the counter is gone'
    guards = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)
              and any(isinstance(x, ast.Name) and x.id == '_POOL_INFLIGHT'
                      for x in [n.left, *n.comparators])]
    assert guards, (
        'nothing compares _POOL_INFLIGHT; the REBUILD RULE guard is gone')
    for g in guards:
        others = [x for x in [g.left, *g.comparators]
                  if not (isinstance(x, ast.Name)
                          and x.id == '_POOL_INFLIGHT')]
        assert all(isinstance(x, ast.Constant) and x.value == 0
                   for x in others), (
            f'line {g.lineno} compares _POOL_INFLIGHT against something '
            f'other than zero, so its magnitude has acquired a meaning')


# ===========================================================================
# D5 -- what a timeout would have to name, and what a bootstrap bar leaves
# ===========================================================================
def test_a_timeout_on_the_dispatch_must_name_its_own_exception_class():
    """`TimeoutError` is only an `OSError` from Python 3.11.

    The follow-ups report recommends bounding the bootstrap and argues the
    fallback is already right because "``TimeoutError`` is an ``OSError``
    subclass, so it lands in the dispatcher's existing infrastructure
    clause".  That is true on 3.11+ and FALSE on 3.10:
    ``concurrent.futures.TimeoutError`` became an alias of the builtin only
    in 3.11 (gh-90315); before that it derived from
    ``concurrent.futures._base.Error(Exception)``.  ``pyproject.toml``
    declares ``requires-python = ">=3.10"``.

    MEASURED 2026-09-19 (``vf5_d5_timing.py::measure_clause_reach``, driven
    through the real dispatcher on both builds): an exception with the
    builtin MRO falls back to serial byte-identically; an exception with the
    PRE-3.11 MRO escapes the clause and reaches the caller.

    So this is a two-sided, build-free statement: the MRO on the RUNNING
    build is measured, not assumed, and the structural requirement -- if the
    dispatcher ever passes ``timeout=`` to ``as_completed``, the enclosing
    ``except`` must name a timeout class explicitly -- holds on every
    interpreter.
    """
    import concurrent.futures as cf

    measured = {
        'cf_TimeoutError_is_builtin': cf.TimeoutError is TimeoutError,
        'is_OSError_subclass': issubclass(cf.TimeoutError, OSError),
        'mro': [c.__name__ for c in cf.TimeoutError.__mro__],
        'python': sys.version_info[:2],
    }

    src = inspect.getsource(LT)
    tree = ast.parse(src)
    bounded_calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'as_completed'
                and any(kw.arg == 'timeout' for kw in node.keywords)):
            bounded_calls.append(node.lineno)

    if not bounded_calls:
        # The exposure is OPEN (VERIFY-WP-B13 D5).  Assert the premise, so
        # this test cannot quietly become vacuous, and say what to do when
        # the bar is added.
        assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == 'as_completed' for n in ast.walk(tree)), (
            'the dispatcher no longer iterates as_completed at all; this pin '
            'and the whole of D5 need restating against whatever replaced it')
        pytest.skip.Exception  # noqa: B018 -- never skip; see below
        assert measured['is_OSError_subclass'] or not measured[
            'cf_TimeoutError_is_builtin'], measured
        return

    # A bar was added: the clause that catches it must name the class.
    named = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(ln in range(node.lineno, getattr(node, 'end_lineno',
                                                    node.lineno) + 1)
                   for ln in bounded_calls):
            continue
        for handler in node.handlers:
            txt = ast.dump(handler.type) if handler.type else ''
            if 'TimeoutError' in txt:
                named = True
    assert named, (
        f'as_completed is now called with timeout= at lines {bounded_calls}, '
        f'but no enclosing except clause names TimeoutError.  On this build '
        f'{measured}; on Python 3.10 -- inside this package\'s declared '
        f'requires-python -- concurrent.futures.TimeoutError does NOT derive '
        f'from OSError, so the dispatcher\'s (BrokenProcessPool, '
        f'RuntimeError, OSError, EOFError) clause would not catch it and the '
        f'timeout would reach the caller instead of taking the '
        f'bit-identical serial rung.')


_SENTINEL_CHILD = r'''
import json, os, sys, threading, time, warnings
import numpy as np
from concurrent.futures import Future
import lumenairy as la
from lumenairy.elements import _lens_traced as LT

# The __main__ guard below is load-bearing: see the note in the lazy-spawn
# child.  Without it the dispatcher refuses workers and never reaches the
# stub pool, and the arm measures nothing.
LT._POOL_MIN_PIXELS = 1
LT._POOL_MIN_PIXELS_WARM = 1

# MUTATION-MATRIX HOOK.  Tests that drive a real pool run in a CHILD, and a
# child is a fresh interpreter: an in-memory regression injected into the
# PARENT is invisible to it.  `validation/probe_verify_b13_followups/
# vf8_mutations.py` therefore passes its choice down in VF8_MUTATE, and the
# child re-applies it here.  Outside that probe the variable is unset and
# this block does nothing.
_MUT = os.environ.get('VF8_MUTATE', '')
if _MUT:
    try:
        import vf8_mutations as _VF8
        _VF8.pytest_configure(None)
    except Exception as _exc:          # noqa: BLE001 -- reported, not fatal
        print('child could not apply VF8_MUTATE=%s: %s' % (_MUT, _exc))
# FORCE THE PRECONDITION: the resource clamp refuses workers on a loaded box,
# which would send the dispatch serial and never reach the stub pool.  The
# test asserts `submits >= 2`, which is the gate.
_REAL_RESOLVE = LT._newton_resolve_workers
LT._newton_resolve_workers = lambda n_req, *a, **kw: max(1, int(n_req))

class SentinelThenWedge:
    """First submit answers at once; every later submit never completes."""
    _broken = None
    def __init__(self):
        self.n = 0
        self.held = []
    def submit(self, fn, *a, **kw):
        self.n += 1
        f = Future()
        if self.n == 1:
            f.set_result((np.zeros(0), 0))
        else:
            self.held.append(f)
        return f
    def shutdown(self, wait=True, **kw):
        pass

def presc(ap=3e-3, r=9e-3):
    return {'name': 'fast_singlet', 'aperture_diameter': ap,
            'thicknesses': [3e-3],
            'surfaces': [
                {'radius': r, 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'conic': 0.0, 'aspheric_coeffs': None},
                {'radius': -r, 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'conic': 0.0, 'aspheric_coeffs': None}]}

def traced(nw, N):
    ap = 3e-3; dx = 2.2 * ap / N
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X**2 + Y**2) / 1.2e-3**2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            E0, prescription=presc(ap), wavelength=1.31e-6, dx=dx,
            ray_subsample=2, newton_fit='spline', n_workers=nw,
            parallel_amp=False, newton_amp_mask_rel=0.0,
            on_undersample='silent', on_aperture_beam='silent'))

def main():
    # PRECONDITION_BOUND bounds the ray trace / spline fit that happens
    # BEFORE the first submit -- wall time that scales with the box's load
    # and has nothing to do with the claim.  DEADLINE is the only bar that
    # decides anything, and it starts once the sentinel has answered and a
    # later chunk is outstanding.
    N, DEADLINE, PRECONDITION_BOUND = 256, 8.0, 180.0
    LT.close_worker_pool()
    pool = SentinelThenWedge()
    LT._get_persistent_worker_pool = lambda nw, _p=pool: _p
    box = {}
    def run():
        try:
            box['v'] = traced(4, N)
        except BaseException as exc:
            box['exc'] = '%s: %s' % (type(exc).__name__, exc)
    t = threading.Thread(target=run, daemon=True, name='vb13-sentinel')
    t0 = time.monotonic()
    t.start()
    end = t0 + PRECONDITION_BOUND
    while time.monotonic() < end and t.is_alive():
        if pool.n >= 2 and pool.held:
            break
        time.sleep(0.05)
    reached_at = round(time.monotonic() - t0, 2)
    precondition = pool.n >= 2 and bool(pool.held)
    t1 = time.monotonic()
    t.join(DEADLINE)
    alive = t.is_alive()
    out = {'deadline': DEADLINE,
           'precondition_bound': PRECONDITION_BOUND,
           'seconds_to_first_submit': reached_at,
           'submits': pool.n,
           'held_futures': len(pool.held),
           'sentinel_answered': pool.n >= 1,
           'precondition_reached': bool(precondition),
           'still_running_at_deadline': bool(alive),
           'elapsed_after_precondition': round(time.monotonic() - t1, 2),
           'outcome': box.get('exc', 'returned')}
    print('JSON ' + json.dumps(out))
    sys.stdout.flush()
    os._exit(0)

if __name__ == '__main__':
    main()
'''


def test_a_bootstrap_bar_would_not_bound_a_chunk_that_wedges_later(tmp_path):
    """PREMISE GATE for the OPEN D5, and a caveat on the recommendation.

    The follow-ups report recommends candidate B -- submit one trivial
    sentinel ahead of the chunks and bound THAT, leaving ``as_completed``
    unbounded.  That bounds "did any worker answer at all", which is the
    ``slowboot`` pathology exactly.  It does NOT bound a worker that answers
    once and then stops answering, because the unbounded ``as_completed``
    loop is still there behind it.

    DRIVEN 2026-09-19 (``vf5_d5_timing.py::measure_residual_exposure``, and
    here): a pool whose FIRST submit answers and whose later submits never
    complete leaves the dispatch running past the deadline on both builds.
    The exposure is therefore narrowed, not closed, and a release note that
    says "bounded" would be wrong.

    WHEN THIS GOES RED because the dispatch now returns, that is the right
    outcome and not a broken test: D5 has been closed by something that
    bounds the iteration as well as the bootstrap.  Retire this test and
    take the open-D5 paragraph out of the verification.
    """
    out, _proc = _run_child(tmp_path, 'vb13_sentinel_then_wedge.py',
                            _SENTINEL_CHILD, timeout=300)
    assert out['precondition_reached'] is True, (
        f'the arm never reached the state it is about -- a sentinel that has '
        f'ANSWERED and at least one later chunk outstanding -- within '
        f'{out["precondition_bound"]} s, so it models nothing.  That bound '
        f'covers the ray trace and spline fit that precede the first submit, '
        f'which is wall time this claim does not depend on: {out}')
    assert out['sentinel_answered'] is True and out['held_futures'] >= 1, (
        f'the arm does not model candidate B: {out}')
    assert out['still_running_at_deadline'] is True, (
        f'the dispatch returned within {out["deadline"]} s although a chunk '
        f'after the first never completed.  If that is because a bound was '
        f'added to the as_completed iteration, D5 is closed -- retire this '
        f'test and the open-D5 note.  Outcome: {out}')


# ===========================================================================
# D7 -- the shapes the extended detector still cannot see
# ===========================================================================
_BLIND_SPOT_SOURCES = (
    # (label, source, why the extended detector misses it)
    ('aliased_class',
     'from concurrent.futures import ProcessPoolExecutor as PPE\n'
     'def teardown(n):\n'
     '    with PPE(n) as ex:\n'
     '        ex.submit(abs, -1)\n',
     'the detector matches on a class NAME ending in "Executor"'),
    ('prebuilt_name',
     'def teardown(n):\n'
     '    ex = ProcessPoolExecutor(n)\n'
     '    with ex:\n'
     '        ex.submit(abs, -1)\n',
     'the with-item is a Name, not a Call, so no class name is visible'),
)


def _executor_aliases(src):
    """Local names bound to a process-pool executor class under a name the
    branch detector cannot recognise.

    Only bindings whose LOCAL name does not itself end in ``Executor`` count:
    ``from concurrent.futures import ProcessPoolExecutor`` is a shape
    ``_unbounded_executor_joins`` already sees, and counting it here would
    double-report every clean module.
    """
    out = set()
    for node in ast.walk(ast.parse(src)):
        names = []
        if isinstance(node, ast.ImportFrom):
            names = [(a.name, a.asname or a.name) for a in node.names]
        elif isinstance(node, ast.Import):
            names = [(a.name, a.asname or a.name.split('.')[-1])
                     for a in node.names]
        for orig, local in names:
            if orig.endswith('Executor') and 'Process' in orig                     and not local.endswith('Executor'):
                out.add(local)
    return out


def _blind_spot_teardowns(src):
    """``with`` teardowns of a process pool the branch detector cannot see.

    Two shapes: a ``with`` on a call to a locally ALIASED executor class, and
    a ``with`` on a bare Name that was assigned such a call in the same
    function.
    """
    tree = ast.parse(src)
    aliases = _executor_aliases(src)
    # names assigned a ProcessPool*Executor(...) call anywhere in the file
    pool_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            f = node.value.func
            nm = (f.id if isinstance(f, ast.Name)
                  else f.attr if isinstance(f, ast.Attribute) else None)
            if nm and (nm in aliases
                       or (nm.endswith('Executor') and 'Process' in nm)):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        pool_names.add(t.id)
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            ce = item.context_expr
            if isinstance(ce, ast.Call):
                f = ce.func
                nm = (f.id if isinstance(f, ast.Name)
                      else f.attr if isinstance(f, ast.Attribute) else None)
                if nm in aliases:
                    hits.append(f'{node.lineno}:aliased {nm}')
            elif isinstance(ce, ast.Name) and ce.id in pool_names:
                hits.append(f'{node.lineno}:prebuilt {ce.id}')
    return sorted(hits)


def test_no_module_uses_an_executor_teardown_the_join_detector_cannot_see():
    """The residual blind spots, made harmless rather than merely noted.

    MEASURED 2026-09-19 (``vf4_d7_detector_corpus.py``, 20 shapes, both
    builds): the branch's ``_unbounded_executor_joins`` grades 12 TP / 0 FP /
    2 FN / 6 TN against the shipped detector's 6 / 2 / 8 / 4.  The two
    misses are a ``with`` on an ALIASED executor class (``import ... as
    PPE``) and a ``with`` on a pre-built executor NAME -- both of which are
    ``shutdown(wait=True)`` on exit exactly like the shape D7 added.

    A blind spot only matters if something walks into it, so this pin sweeps
    the whole package for those two shapes.  The positive control is the
    synthetic source below: without it, a sweep that had itself gone blind
    would report a clean package and read as a pass.
    """
    # positive control first -- the sweep must find what it is looking for
    for label, src, why in _BLIND_SPOT_SOURCES:
        assert _blind_spot_teardowns(src), (
            f'the blind-spot sweep found nothing in the {label} control '
            f'({why}); it has gone blind and the package sweep below means '
            f'nothing')
    # ... and must not fire on the shapes the branch detector already covers
    assert _blind_spot_teardowns(
        'from concurrent.futures import ProcessPoolExecutor\n'
        'def t(n):\n'
        '    with ProcessPoolExecutor(n) as ex:\n'
        '        ex.submit(abs, -1)\n') == [], (
        'the sweep fires on the plain shape, which the branch detector '
        'already reports; it would double-count')

    import lumenairy
    pkg = os.path.dirname(os.path.abspath(lumenairy.__file__))
    offenders = {}
    for dirpath, dirnames, filenames in os.walk(pkg):
        dirnames[:] = [d for d in dirnames if d != '__pycache__']
        for fn in filenames:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, encoding='utf-8', errors='replace') as fh:
                    src = fh.read()
            except OSError:
                continue
            if 'Executor' not in src:
                continue
            try:
                hits = _blind_spot_teardowns(src)
            except SyntaxError:
                continue
            if hits:
                offenders[os.path.relpath(path, pkg)] = hits
    assert offenders == {}, (
        f'{offenders} tear an executor down through a `with` shape that '
        f'test_only_the_bounded_helper_ever_joins_an_executor cannot see -- '
        f'an aliased class name, or a `with` on a pre-built executor.  Both '
        f'are shutdown(wait=True) on exit, i.e. the exposure WP-B13 removed '
        f'from the Newton pool.  Either route the teardown through '
        f'_shutdown_pool_bounded, or teach _unbounded_executor_joins these '
        f'two shapes (it grades 12/0/2/6 on the VERIFY corpus today).')

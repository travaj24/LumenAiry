"""The Newton worker pool must FALL BACK, never wedge (handoff P1 4.1b).

WHAT THIS FILE IS ABOUT
-----------------------
``apply_real_lens_traced`` inverts its ray map with Newton's method and may
dispatch the work in chunks to a module-level, persistent
``ProcessPoolExecutor`` (spawn).  Three terms, used throughout:

``broken pool``
    An executor that has lost a worker.  CPython fails every pending future
    with ``BrokenProcessPool`` and the pool can serve nobody afterwards.
``the fallback``
    The dispatcher's own answer to that: run the identical Newton inversion
    IN PROCESS.  The two paths are bit-identical by contract (the payload
    pins the Chebyshev backend and ships the parent's fit -- see
    ``test_fix_newton_pool_memory.py``), so falling back costs wall time and
    moves no number.
``the wedge``
    What was measured instead at the 5.47.0 release gate (2026-09-14, python
    3.14.6): the fallback first called ``close_worker_pool()``, whose
    ``shutdown(wait=True)`` entered CPython's
    ``_ExecutorManagerThread._terminate_broken``.  That function joins the
    queue-feeder thread and every worker process WITHOUT a timeout, holding
    the executor's ``_shutdown_lock`` -- the same lock ``shutdown`` acquires
    before it even looks at its ``wait`` argument.  The serial line below was
    never reached and the process hung forever.

THE INVARIANT under test: **no Newton-pool path joins an executor for an
unbounded time, and a broken pool always reaches the bit-identical serial
answer.**

HOW A REGRESSION FAILS RATHER THAN HANGS
----------------------------------------
Every test here either runs the library call on a daemon thread joined with a
deadline (``_with_deadline``, which dumps every thread's stack into the
failure message) or runs it in a CHILD PROCESS with a hard
``subprocess.run(timeout=...)``.  Nothing in this file can wedge the pytest
process.

The fail-before for the two in-process wedge tests was taken
ARCHIVE-TO-ARCHIVE on the parent commit ``96cb2096`` (``git archive`` tree,
child process, ``lumenairy.__file__`` asserted under it,
``validation/probe_newton_pool/probe_p7_broken_fallback.py --mode stub``):
the call never returned and the 180 s ``faulthandler`` deadline caught the
main thread in ``_lens_traced.py:12624 _invert_newton_parallel ->
close_worker_pool -> shutdown``.  On this tree the same probe returns in
14.6 s with a byte-identical field.
"""
from __future__ import annotations

import ast
import faulthandler
import inspect
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          '..', '..'))
_WL = 1.31e-6

# Field small enough that a whole test stays well inside 60 s, with the pool's
# SIZE bars forced down so the dispatch happens at this size -- the precondition
# is engineered, never hoped for (docs/TESTING_STANDARDS.md, rule 4).  The bars
# themselves are a separate, two-sided claim in test_niche_newton_pool_both_fits.
_N, _RS = 256, 2


def _fast_singlet(ap=3e-3, r=9e-3):
    return {'name': 'fast_singlet', 'aperture_diameter': ap,
            'thicknesses': [3e-3],
            'surfaces': [
                {'radius': r, 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'conic': 0.0, 'aspheric_coeffs': None},
                {'radius': -r, 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'conic': 0.0, 'aspheric_coeffs': None}]}


def _gauss(N, dx, w0):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _traced(n_workers):
    """One traced-lens call, returned as a plain array."""
    ap = 3e-3
    dx = 2.2 * ap / _N
    E0 = _gauss(_N, dx, 1.2e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = la.apply_real_lens_traced(
            E0, prescription=_fast_singlet(ap), wavelength=_WL, dx=dx,
            ray_subsample=_RS, newton_fit='spline', n_workers=n_workers,
            parallel_amp=False, newton_amp_mask_rel=0.0,
            on_undersample='silent', on_aperture_beam='silent')
    return np.asarray(out)


@pytest.fixture(autouse=True)
def _cold_pool():
    """Every test starts and ends with a cold process."""
    LT.close_worker_pool()
    yield
    LT.close_worker_pool()


@pytest.fixture
def _forced_pool_bars(monkeypatch):
    """Make the dispatch happen at this file's (small) field size.

    The pool's size bars exist to keep a one-shot call off a pool it cannot
    amortise; they are not what is under test here, so they are lowered
    rather than worked around with a field big enough to clear them (which
    would put every test in this file past 60 s).
    """
    monkeypatch.setattr(LT, '_POOL_MIN_PIXELS', 1)
    monkeypatch.setattr(LT, '_POOL_MIN_PIXELS_WARM', 1)


@pytest.fixture(scope='module')
def serial_reference():
    """The answer ``n_workers=1`` gives: what every fallback must reproduce."""
    LT.close_worker_pool()
    return _traced(1)


def _thread_dump() -> str:
    """Every thread's stack, as text.

    ``faulthandler.dump_traceback`` writes through the file's ``fileno()``,
    so an ``io.StringIO`` raises ``io.UnsupportedOperation: fileno`` instead
    of producing a dump -- and because this helper is only ever called from
    ``_with_deadline``'s failure path, that turned every wedge detection in
    this file into an unrelated exception with no thread dump attached
    (VERIFY-WP-B13 defect D1, 2026-09-15).  A real temporary file has a real
    descriptor.  ``test_the_wedge_report_carries_the_stack_of_the_wedged_frame``
    is the behavioural pin.
    """
    with tempfile.TemporaryFile('w+') as fh:
        faulthandler.dump_traceback(file=fh, all_threads=True)
        fh.seek(0)
        return fh.read()


def _with_deadline(fn, seconds, what):
    """Run ``fn()`` on a daemon thread; FAIL (never hang) if it overruns.

    Returns ``fn``'s value.  A daemon thread is used on purpose: if the call
    really is wedged, the test fails immediately and the stuck thread cannot
    hold the interpreter open at exit.
    """
    box = {}

    def _target():
        try:
            box['value'] = fn()
        except BaseException as exc:            # noqa: BLE001 -- reported
            box['exc'] = exc

    t = threading.Thread(target=_target, name='deadline-' + what, daemon=True)
    t0 = time.monotonic()
    t.start()
    t.join(seconds)
    if t.is_alive():
        pytest.fail(
            f'{what} did not return within {seconds:.0f} s -- this is the '
            f'wedge P1 4.1b is about, not slowness (the same call returns in '
            f'seconds on a healthy tree).  Thread dump:\n{_thread_dump()}')
    if 'exc' in box:
        raise box['exc']
    box['seconds'] = time.monotonic() - t0
    return box['value']


class _WedgedBrokenPool:
    """A pool that is broken on submit and WEDGED on a joining shutdown.

    This is CPython's ``_terminate_broken`` state made deterministic: every
    future fails with ``BrokenProcessPool`` (what the parent sees when a
    worker exits mid-flight), and any ``shutdown`` that is asked to WAIT never
    returns (what ``_terminate_broken``'s unbounded ``join_thread`` /
    ``p.join()`` do when the feeder thread is stuck in
    ``connection._send_bytes``).  No child processes are involved, so what the
    test measures is the library's code path and nothing else.
    """

    def __init__(self, broken=True):
        if broken:
            self._broken = ('A child process terminated abruptly, the '
                            'process pool is not usable anymore')
        self.shutdown_calls = []
        self._never = threading.Event()

    def submit(self, fn, *a, **kw):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool
        f = Future()
        f.set_exception(BrokenProcessPool(
            'A process in the process pool was terminated abruptly while '
            'the future was running or pending.'))
        return f

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.shutdown_calls.append({'wait': bool(wait),
                                    'cancel_futures': bool(cancel_futures)})
        if wait:
            self._never.wait()

    def release(self):
        self._never.set()


def _install_pool(monkeypatch, pool, n_workers=4):
    monkeypatch.setattr(LT, '_PERSISTENT_POOL', pool, raising=False)
    monkeypatch.setattr(LT, '_PERSISTENT_POOL_NWORKERS', n_workers,
                        raising=False)
    monkeypatch.setattr(LT, '_get_persistent_worker_pool', lambda nw: pool)


# ===========================================================================
# 1.  The broken-pool fallback reaches serial
# ===========================================================================

def test_a_broken_pool_reaches_the_serial_fallback_without_joining_it(
        monkeypatch, _forced_pool_bars, serial_reference):
    """THE DEFECT, stated as a decision rather than a reading.

    With a pool that is broken AND whose joining shutdown never returns, the
    traced call must still produce the serial answer.  Pre-fix the call never
    returned at all (fail-before on ``96cb2096``, in this module's docstring);
    the only thing that can make it return is the library declining to join a
    broken executor.

    The second assertion is what makes the first durable: the ONLY shutdown
    the library is allowed to ask of a broken pool is a non-joining one.
    """
    pool = _WedgedBrokenPool(broken=True)
    _install_pool(monkeypatch, pool)
    try:
        got = _with_deadline(lambda: _traced(4), 45.0,
                             'the traced call on a broken, wedged pool')
    finally:
        pool.release()

    assert np.array_equal(got, serial_reference), (
        'the broken-pool fallback did not reproduce the serial answer; the '
        'fallback exists precisely because the two paths are bit-identical, '
        f'max|delta| = {np.abs(got - serial_reference).max():.3e}')
    assert pool.shutdown_calls == [{'wait': False, 'cancel_futures': True}], (
        f'the library asked a BROKEN pool for {pool.shutdown_calls!r}; a '
        f'joining shutdown on a broken pool is the wedge (CPython holds the '
        f'executor shutdown lock across two unbounded joins inside '
        f'_terminate_broken), so the only permitted request is '
        f"wait=False with cancel_futures=True")


def test_close_worker_pool_never_joins_a_broken_pool(monkeypatch):
    """The same rule at the public entry point.

    ``close_worker_pool`` is documented as the return-to-cold call and is the
    process's ``atexit`` handler; on a broken pool it must retire the
    executor rather than join it, and must still leave the module cold.
    """
    pool = _WedgedBrokenPool(broken=True)
    _install_pool(monkeypatch, pool, n_workers=3)
    try:
        _with_deadline(LT.close_worker_pool, 30.0,
                       'close_worker_pool on a broken pool')
    finally:
        pool.release()
    assert pool.shutdown_calls == [{'wait': False, 'cancel_futures': True}]
    assert LT._PERSISTENT_POOL is None
    assert LT._PERSISTENT_POOL_NWORKERS is None
    assert LT._POOL_DEFERRED_NWORKERS is None, (
        'close_worker_pool must also clear the second-call promotion, '
        'otherwise the very next 65k call rebuilds the pool that just failed')


def test_close_worker_pool_is_bounded_even_on_a_pool_it_may_join(monkeypatch):
    """The other side of the same bar.

    A pool that has NOT marked itself broken is joined -- that is what
    ``close_worker_pool`` promises -- but only for ``_POOL_SHUTDOWN_TIMEOUT``
    seconds, after which it is abandoned to the reaper and the caller
    returns.  The bound is shortened here so the test is a second long
    instead of two minutes; the shipped value is derived from a measured
    ladder in the module comment (worst healthy close 4.592 s at 16 workers,
    2026-09-14).
    """
    monkeypatch.setattr(LT, '_POOL_SHUTDOWN_TIMEOUT', 1.0)
    before = LT._POOL_SHUTDOWN_TIMEOUTS
    pool = _WedgedBrokenPool(broken=False)
    _install_pool(monkeypatch, pool, n_workers=2)
    try:
        _with_deadline(LT.close_worker_pool, 30.0,
                       'close_worker_pool on a wedged (unbroken) pool')
    finally:
        pool.release()
    assert pool.shutdown_calls == [{'wait': True, 'cancel_futures': False}], (
        'a pool that has not declared itself broken must still get the '
        'joining shutdown; only the WAIT is bounded')
    assert LT._POOL_SHUTDOWN_TIMEOUTS == before + 1, (
        'the bounded wait expired but the module did not record it, so a '
        'wedge would be invisible to anyone reading the process state')
    assert LT._PERSISTENT_POOL is None


# ===========================================================================
# 2.  A REAL worker that dies mid-flight -- in a child process
# ===========================================================================

_KILL_SCRIPT = '''\
"""Child: break a REAL spawn pool and check the fallback.

Kept free of top-level work on purpose: a spawn worker re-imports this module,
and lumenairy refuses the pool for a ``__main__`` whose body would re-run.
"""
import json
import os
import sys
import warnings

import numpy as np


def killer_chunk(args):
    """Stands in for ``_newton_invert_chunk`` and kills its worker.

    ``os._exit`` leaves no result item and no traceback, so the executor sees
    only a closed sentinel -- the exact shape of the release-gate failure.
    """
    os._exit(7)


def fast_singlet(ap):
    return {'name': 'fast_singlet', 'aperture_diameter': ap,
            'thicknesses': [3e-3],
            'surfaces': [
                {'radius': 9e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'conic': 0.0,
                 'aspheric_coeffs': None},
                {'radius': -9e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'conic': 0.0,
                 'aspheric_coeffs': None}]}


def run(la, n_workers, N, wl):
    ap = 3e-3
    dx = 2.2 * ap / N
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / 1.2e-3 ** 2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            E0, prescription=fast_singlet(ap), wavelength=wl, dx=dx,
            ray_subsample=2, newton_fit='spline', n_workers=n_workers,
            parallel_amp=False, newton_amp_mask_rel=0.0,
            on_undersample='silent', on_aperture_beam='silent'))


def main():
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(120.0, exit=True)
    import lumenairy as la
    from lumenairy.elements import _lens_traced as LT
    assert os.path.abspath(la.__file__).startswith(sys.argv[1]), (
        la.__file__, sys.argv[1])
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1
    N, wl = 256, 1.31e-6

    ref = run(la, 1, N, wl)
    LT.close_worker_pool()
    LT._newton_invert_chunk = killer_chunk
    got = run(la, 4, N, wl)
    LT._newton_invert_chunk = None
    LT.close_worker_pool()
    faulthandler.cancel_dump_traceback_later()
    print('RESULT ' + json.dumps({
        'identical': bool(np.array_equal(got, ref)),
        'max_delta': float(np.abs(got - ref).max()),
        'lumenairy': la.__file__,
        'shutdown_timeouts': int(LT._POOL_SHUTDOWN_TIMEOUTS),
    }))


if __name__ == '__main__':
    main()
'''


def _run_child(tmp_path, name, body, timeout, extra_args=()):
    script = tmp_path / name
    script.write_text(body, encoding='utf-8')
    env = dict(os.environ)
    env['PYTHONPATH'] = _REPO_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[var] = '1'
    try:
        proc = subprocess.run(
            [sys.executable, str(script), _REPO_ROOT, *extra_args],
            cwd=str(tmp_path), env=env, timeout=timeout,
            stdin=subprocess.DEVNULL, capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f'{name} did not finish within {timeout:.0f} s -- the child is '
            f'wedged, which is the defect this file is about.\n'
            f'stdout so far:\n{(exc.stdout or b"")!r}\n'
            f'stderr so far:\n{(exc.stderr or b"")!r}')
    return proc


@pytest.mark.slow
def test_a_real_worker_dying_mid_flight_still_returns_the_serial_answer(
        tmp_path):
    """The realistic arm: a genuine ``spawn`` pool, genuinely broken.

    Every worker kills itself with ``os._exit`` the moment it is handed a
    chunk, so the executor really does enter ``_terminate_broken``.  The call
    must still return, and must return the serial answer byte for byte.  Run
    in a child process with a hard timeout, so a regression is a FAILED test
    and not a hung suite.
    """
    proc = _run_child(tmp_path, 'kill_a_worker.py', _KILL_SCRIPT, 240.0)
    assert proc.returncode == 0, (
        f'child exited {proc.returncode}\nstdout:\n{proc.stdout[-4000:]}\n'
        f'stderr:\n{proc.stderr[-4000:]}')
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith('RESULT ')]
    assert line, f'child printed no RESULT line:\n{proc.stdout[-4000:]}'
    got = json.loads(line[-1][len('RESULT '):])
    assert os.path.abspath(got['lumenairy']).startswith(_REPO_ROOT), got
    assert got['identical'], (
        f'a pool whose workers all died returned a DIFFERENT field from the '
        f"serial path (max|delta| = {got['max_delta']:.3e}); the fallback's "
        f'whole claim is that it moves no number')
    assert got['shutdown_timeouts'] == 0, (
        'the child had to abandon a pool on the bounded wait, so the broken '
        'pool was still being joined somewhere')


# ===========================================================================
# 3.  The two-arm traced chain, under pytest's DEFAULT capture
# ===========================================================================

_CHAIN_TEST = '''\
"""Child pytest module: the traced carrier chain with the pool engaged."""
import warnings

import numpy as np

import lumenairy as la
from lumenairy.elements import _lens_traced as LT


def _doublet(ap):
    surfs, before = [], 'air'
    for R, g in ((61.5e-3, 'N-BK7'), (-45.0e-3, 'N-SF5'), (-128.0e-3, 'air')):
        surfs.append({'radius': R, 'glass_before': before, 'glass_after': g,
                      'conic': 0.0, 'radius_y': None, 'conic_y': None,
                      'aspheric_coeffs': None, 'aspheric_coeffs_y': None})
        before = g
    return {'name': 'doublet', 'aperture_diameter': ap,
            'thicknesses': [4.0e-3, 2.5e-3], 'surfaces': surfs}


def _run(n_workers, parallel_amp):
    N, W0, WL = 192, 1.0e-3, 1.31e-6
    ap = 1.2 * 2.0 * W0
    dx = float(2.2 * max(ap, 3.0 * W0) / N)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env0 = np.exp(-(X ** 2 + Y ** 2) / W0 ** 2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = la.propagate_traced_carrier_chain(
            env0, [{'prescription': _doublet(ap), 'gap_before': 0.0}],
            WL, dx, r_in=np.inf, ray_subsample=2, n_workers=n_workers,
            final_distance=0.0,
            traced_kwargs=dict(parallel_amp=parallel_amp,
                               on_undersample='silent',
                               newton_fit='spline'))
    return np.asarray(res.field)


def test_the_chain_completes_with_the_pool_engaged():
    """Drives BOTH amplitude arms (``parallel_amp=True`` runs them on a
    ``ThreadPoolExecutor(2)``) while the Newton inversion dispatches to the
    persistent process pool -- the shape the release gate hung on.  The
    pooled answer must equal the serial one byte for byte."""
    LT.close_worker_pool()
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1
    serial = _run(1, False)
    pooled = _run(4, True)
    LT.close_worker_pool()
    assert np.array_equal(serial, pooled), (
        'the two-arm chain on a process pool moved the field by '
        f'{np.abs(serial - pooled).max():.3e}')
'''


@pytest.mark.slow
@pytest.mark.parametrize('capture', ['fd', 'sys'])
def test_the_traced_chain_completes_under_either_pytest_capture(tmp_path,
                                                                capture):
    """pytest's DEFAULT capture is ``fd``; the release gate could only be
    finished by switching every lane to ``--capture=sys``, so ``fd`` is the
    arm that matters and ``sys`` is its control.

    The child is a pytest process of its own with a hard timeout, so a
    regression fails this test rather than hanging the suite that runs it.
    """
    mod = tmp_path / 'test_chain_child.py'
    mod.write_text(_CHAIN_TEST, encoding='utf-8')
    env = dict(os.environ)
    env['PYTHONPATH'] = _REPO_ROOT + os.pathsep + env.get('PYTHONPATH', '')
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[var] = '1'
    cmd = [sys.executable, '-m', 'pytest', str(mod), '-q', '-p', 'no:cacheprovider',
           '-p', 'no:randomly', f'--capture={capture}']
    try:
        proc = subprocess.run(cmd, cwd=str(tmp_path), env=env, timeout=300.0,
                              stdin=subprocess.DEVNULL, capture_output=True, text=True)
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f'the traced chain hung under --capture={capture}; that is the '
            f'release-gate failure.\nstdout:\n{(exc.stdout or b"")!r}')
    assert proc.returncode == 0, (
        f'--capture={capture}: child pytest exited {proc.returncode}\n'
        f'{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}')
    assert '1 passed' in proc.stdout, proc.stdout[-4000:]


# ===========================================================================
# 4.  The rebuild rule -- the ROOT CAUSE
# ===========================================================================

class _FakeExecutor:
    """Counts constructions without spawning anything."""

    made: list = []

    def __init__(self, max_workers=None, mp_context=None, initializer=None,
                 initargs=(), **kw):
        self.max_workers = max_workers
        type(self).made.append(int(max_workers))

    def submit(self, fn, *a, **kw):
        raise AssertionError('this executor is a counter, not a pool')

    def shutdown(self, wait=True, *, cancel_futures=False):
        pass


@pytest.fixture
def count_pool_constructions(monkeypatch):
    import concurrent.futures as cf
    _FakeExecutor.made = []
    monkeypatch.setattr(cf, 'ProcessPoolExecutor', _FakeExecutor)
    LT.close_worker_pool()
    yield _FakeExecutor.made
    LT.close_worker_pool()


def test_a_moving_worker_clamp_no_longer_rebuilds_the_pool_every_call(
        count_pool_constructions):
    """THE MEASURED ROOT CAUSE, replayed.

    ``_newton_resolve_workers`` clamps the worker count against LIVE free
    memory, so its answer moves between calls on a busy box.  MEASURED here
    2026-09-14 (python 3.14.6) on one run of
    ``test_audit2609_b4_collins_transport.py -k
    test_both_transports_reproduce_the_audit_s_own_readings``: 6, 8, 4, 8 for
    the SAME two lens groups -- and the shipped getter answered that with four
    pool constructions and three teardowns, i.e. 26 spawned worker
    interpreters for four dispatches, old and new pools alive together
    (the teardown does not wait).  The same run under ``--capture=sys`` read
    5, 8, 5, 8 and also built four, so this is the dispatch pattern and not a
    capture interaction.

    Under the rebuild rule the pool's width is a CEILING: it is raised when a
    call needs more and never lowered implicitly, so this sequence builds
    exactly two pools -- one per distinct increase -- and the 4 is served by
    the 8 that is already up.
    """
    seen = []
    for n in (6, 8, 4, 8):
        seen.append(LT._get_persistent_worker_pool(n).max_workers)
    assert count_pool_constructions == [6, 8], (
        f'the clamp sequence 6, 8, 4, 8 built pools {count_pool_constructions};'
        f' expected one construction per INCREASE (6 then 8), with the 4 '
        f'served by the live 8')
    assert seen == [6, 8, 8, 8], seen


def test_a_wider_request_still_grows_the_pool_when_nothing_is_in_flight(
        count_pool_constructions):
    """The control for the test above: the ceiling is a floor on nothing.

    A call that genuinely needs more workers than the live pool has must get
    them, or the rule would silently cap every later dispatch at whatever the
    first call happened to measure.
    """
    LT._get_persistent_worker_pool(2)
    LT._get_persistent_worker_pool(5)
    assert count_pool_constructions == [2, 5]


def test_a_rebuild_is_refused_while_a_dispatch_holds_the_pool(
        count_pool_constructions):
    """The concurrency half of the invariant, without running a lens.

    While a dispatch holds the in-flight claim, a wider request must be served
    by the LIVE pool rather than by tearing it down: a teardown under chunks
    in flight is exactly what looks like a broken pool to the thread waiting
    on them.
    """
    first = LT._get_persistent_worker_pool(2)
    assert count_pool_constructions == [2]
    LT._note_pool_inflight(1)
    try:
        during = LT._get_persistent_worker_pool(8)
        assert during is first, (
            'a wider request tore down a pool that had chunks in flight')
        assert count_pool_constructions == [2], count_pool_constructions
    finally:
        LT._note_pool_inflight(-1)
    after = LT._get_persistent_worker_pool(8)
    assert after is not first
    assert count_pool_constructions == [2, 8], (
        'the claim was released but the pool still refuses to grow')


def test_the_in_flight_counter_cannot_go_negative_or_leak():
    """Bookkeeping: an unbalanced release must not wedge the rule shut."""
    LT._note_pool_inflight(-5)
    assert LT._POOL_INFLIGHT == 0
    assert LT._note_pool_inflight(1) == 1
    assert LT._note_pool_inflight(-1) == 0


@pytest.mark.slow
def test_a_forced_rebuild_during_a_live_dispatch_does_not_break_it(
        monkeypatch, _forced_pool_bars, serial_reference):
    """The race, engineered rather than sampled.

    A second thread asks for a WIDER pool at the exact moment the dispatching
    thread has taken its in-flight claim, and is held there until it has
    tried.  The dispatch must complete and return the serial answer, and the
    second thread must have been handed the same live pool.

    Timing is not left to luck (docs/TESTING_STANDARDS.md rule 3): the two
    threads hand off through events, so the interleaving under test happens on
    every run and on every build.
    """
    claimed = threading.Event()
    rebuild_tried = threading.Event()
    box = {}

    real_note = LT._note_pool_inflight
    real_get = LT._get_persistent_worker_pool

    def note(delta):
        n = real_note(delta)
        if delta > 0 and not claimed.is_set():
            claimed.set()
            rebuild_tried.wait(30.0)
        return n

    def get(nw):
        ex = real_get(nw)
        box.setdefault('dispatch_pool', ex)
        return ex

    monkeypatch.setattr(LT, '_note_pool_inflight', note)
    monkeypatch.setattr(LT, '_get_persistent_worker_pool', get)

    def rival():
        try:
            if claimed.wait(40.0):
                box['rival_pool'] = real_get(
                    (LT._PERSISTENT_POOL_NWORKERS or 4) + 4)
        finally:
            rebuild_tried.set()

    t = threading.Thread(target=rival, name='rival-rebuild', daemon=True)
    t.start()
    try:
        got = _with_deadline(lambda: _traced(4), 55.0,
                             'the dispatch racing a forced rebuild')
    finally:
        rebuild_tried.set()
        t.join(10.0)

    assert claimed.is_set(), (
        'the dispatch never took an in-flight claim, so this test did not '
        'exercise the race it is named for')
    assert 'rival_pool' in box, 'the rival thread never asked for a pool'
    assert box['rival_pool'] is box['dispatch_pool'], (
        'a concurrent wider request replaced the pool a live dispatch was '
        'using; that teardown is what the dispatching thread sees as a '
        'BrokenProcessPool')
    assert np.array_equal(got, serial_reference), (
        f'the raced dispatch moved the field by '
        f'{np.abs(got - serial_reference).max():.3e}')


# ===========================================================================
# 5.  Unchanged behaviour
# ===========================================================================

def test_n_workers_1_never_touches_the_pool(monkeypatch, _forced_pool_bars,
                                            serial_reference):
    """The documented escape hatch stays exactly what it was: no pool, no
    spawn, and the same numbers."""
    calls = []
    real_get = LT._get_persistent_worker_pool
    monkeypatch.setattr(LT, '_get_persistent_worker_pool',
                        lambda nw: (calls.append(nw), real_get(nw))[1])
    got = _with_deadline(lambda: _traced(1), 45.0, 'the n_workers=1 call')
    assert calls == [], f'n_workers=1 asked for a pool {calls}'
    assert LT._PERSISTENT_POOL is None
    assert np.array_equal(got, serial_reference)


# ===========================================================================
# 6.  Wiring pins -- a refactor must not reintroduce an unbounded join
# ===========================================================================

def _module_source():
    import inspect
    return inspect.getsource(LT)


def _last_name(node):
    """The trailing identifier of a ``Name`` / ``Attribute``, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _executor_class_name(node):
    """The executor class a CALL constructs, or None.

    ``ProcessPoolExecutor(...)``, ``futures.ProcessPoolExecutor(...)`` and
    ``cf.ThreadPoolExecutor(...)`` all answer; anything else does not.
    """
    if not isinstance(node, ast.Call):
        return None
    name = _last_name(node.func)
    return name if name and name.endswith('Executor') else None


def _unbounded_executor_joins(src, exempt=('_shutdown_pool_bounded',)):
    """Every site in ``src`` that joins an executor without a bound.

    TWO SHAPES, because one of them has no ``shutdown`` call in it at all:

    ``x.shutdown(...)``
        with ``wait`` absent, or present and not the constant ``False``.
        ``wait`` is read from the keywords AND from the positional slot, so
        ``ex.shutdown(False)`` is correctly NOT an offender and
        ``Executor.shutdown(ex, True)`` -- the unbound form, whose first
        positional is ``self`` -- correctly is.
    ``with ProcessPoolExecutor(...) as ex:``
        ``Executor.__exit__`` IS ``shutdown(wait=True)``, so the block's exit
        joins the feeder thread and every worker process with no timeout.
        The shipped pin saw only the first shape and was therefore blind to
        exactly the form the sibling pool uses (VERIFY-WP-B13 defect D7).

    Scope is PROCESS pools.  The exposure is CPython's
    ``_ExecutorManagerThread._terminate_broken``, which holds the executor's
    ``_shutdown_lock`` across an untimed ``call_queue.join_thread()`` and an
    untimed ``p.join()`` per worker; ``ThreadPoolExecutor`` has no such
    machinery and its ``__exit__`` joins work that its own ``result()`` calls
    have already collected.  Thread-pool ``with`` blocks are returned
    separately as ``informational`` rather than silently dropped, so the
    reading is visible in the failure message instead of being an
    undocumented exemption.

    Returns ``(offenders, informational)``, each a sorted list of
    ``'owner:lineno:shape'``.
    """
    tree = ast.parse(src)
    # Innermost enclosing function of every node, so a match is attributed to
    # the closure that makes it rather than to the whole 8000-line public
    # function it happens to sit in.  A text search cannot do this: the
    # comments that EXPLAIN the rule contain the string it forbids.
    owner = {}
    enclosing = {}
    for parent in ast.walk(tree):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(parent):
                if child is not parent:
                    owner[child] = parent.name
                    enclosing.setdefault(child, set()).add(parent.name)

    def _exempt(node):
        return bool(set(exempt) & enclosing.get(node, set()))

    offenders, info = [], []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'shutdown'):
            # ``Executor.shutdown(ex, wait)`` puts ``self`` first.
            unbound = _executor_class_name(
                ast.Call(func=node.func.value, args=[], keywords=[])) \
                if isinstance(node.func.value,
                              (ast.Name, ast.Attribute)) else None
            slot = 1 if unbound else 0
            waits = [kw.value for kw in node.keywords if kw.arg == 'wait']
            if len(node.args) > slot:
                waits.append(node.args[slot])
            joining = (not waits) or any(
                not (isinstance(v, ast.Constant) and v.value is False)
                for v in waits)
            if joining and not _exempt(node):
                offenders.append(
                    f'{owner.get(node)}:{node.lineno}:shutdown')
            continue
        if isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                cls = _executor_class_name(item.context_expr)
                if cls is None:
                    continue
                where = f'{owner.get(node)}:{node.lineno}:with {cls}'
                if 'Process' not in cls:
                    info.append(where)
                elif not _exempt(node):
                    offenders.append(where)
    return sorted(offenders), sorted(info)


def test_only_the_bounded_helper_ever_joins_an_executor():
    """The durable form of the invariant.

    Bit-identity cannot see a join, and a wedge only shows up on a box under
    load, so the guard has to be structural: an unbounded join of a process
    pool may appear in exactly one place -- the helper that runs it on
    another thread and joins THAT with a timeout.

    The detector is ``_unbounded_executor_joins``, which sees the ``with``
    form as well as the ``shutdown`` call.  Its positive control is
    ``test_the_join_detector_sees_the_with_form`` and its live example is
    ``test_the_sibling_process_pool_still_carries_an_unbounded_join``; a pin
    whose detector is never shown finding anything is a pin that can go blind
    without going red.
    """
    offenders, info = _unbounded_executor_joins(_module_source())
    assert offenders == [], (
        f'{offenders} join a process pool with no bound.  CPython joins the '
        f'queue-feeder thread and every worker process inside '
        f'_terminate_broken, under the same lock shutdown takes first, so a '
        f'direct joining shutdown -- or a `with ProcessPoolExecutor(...)` '
        f'block, whose __exit__ IS shutdown(wait=True) -- can wedge forever; '
        f'route it through _shutdown_pool_bounded.  Thread-pool blocks seen '
        f'and deliberately out of scope: {info}')


def test_the_join_detector_sees_the_with_form():
    """The positive control, on synthetic source.

    The shipped pin walked only for ``ast.Call`` nodes whose ``func.attr``
    was ``shutdown``, so a teardown written as a ``with`` block -- which has
    no such call anywhere -- passed it (VERIFY-WP-B13 D7).  This is that
    blind spot stated as a decision, on source this test owns, so it stays
    meaningful however the two real modules evolve.
    """
    src = (
        'def teardown_with(n):\n'
        '    with ProcessPoolExecutor(max_workers=n) as ex:\n'
        '        ex.submit(abs, -1)\n'
        'def teardown_call(ex):\n'
        '    ex.shutdown(wait=True)\n'
        'def teardown_unbound(ex):\n'
        '    Executor.shutdown(ex, True)\n'
        'def retire(ex):\n'
        '    ex.shutdown(wait=False, cancel_futures=True)\n'
        'def retire_positional(ex):\n'
        '    ex.shutdown(False)\n'
        'def threads(n):\n'
        '    with ThreadPoolExecutor(max_workers=n) as tp:\n'
        '        tp.submit(abs, -1)\n')
    offenders, info = _unbounded_executor_joins(src, exempt=())
    shapes = sorted(o.split(':', 1)[0] for o in offenders)
    assert shapes == ['teardown_call', 'teardown_unbound', 'teardown_with'], (
        f'the detector reported {offenders}.  It must see all three joining '
        f'shapes -- the `with` block, the bound call and the unbound call -- '
        f'and must NOT report a non-joining shutdown written either with a '
        f'keyword or positionally')
    assert [i.split(':', 1)[0] for i in info] == ['threads'], (
        f'thread-pool blocks must be reported as informational, not dropped '
        f'and not counted: {info}')


def test_the_sibling_process_pool_still_carries_an_unbounded_join():
    """PREMISE GATE, and the record of an OPEN defect.

    `lumenairy.propagators.carrier._multi_parallel_results` runs its own
    spawn pool as ``with ProcessPoolExecutor(...) as ex:``, i.e.
    ``shutdown(wait=True)`` on exit, and therefore carries the identical
    unbounded-join exposure that WP-B13 removed from the Newton pool.  That
    is VERIFY-WP-B13 D5, a MAINTAINER DECISION deliberately left open here
    (see `WP-B13_FOLLOWUPS_REPORT.md`): it is a different pool with a
    different failure policy and it belongs in its own work package.

    This test therefore asserts that the extended detector DETECTS it.  Two
    things ride on that.  It is the fail-before for the `with` extension --
    without it the detector reports nothing on this module, which is exactly
    how the shipped pin stayed green on the shape.  And it is the record:
    the pin must not be made green by quietly editing `carrier.py`.

    WHEN THIS GOES RED because the sibling pool was repaired, that is the
    right outcome and not a broken test: delete this test, move the open-defect
    note out of the report, and rely on
    ``test_the_join_detector_sees_the_with_form`` for the positive control.
    """
    import inspect

    from lumenairy.propagators import carrier as CA
    offenders, info = _unbounded_executor_joins(inspect.getsource(CA),
                                                exempt=())
    owners = {o.split(':', 1)[0] for o in offenders}
    assert '_multi_parallel_results' in owners, (
        f'the detector no longer reports an unbounded executor join in '
        f'carrier._multi_parallel_results.  Either the sibling pool was '
        f'repaired -- in which case say so and retire this test, and take '
        f'the open D5 note out of the WP-B13 follow-ups report -- or the '
        f'detector has gone blind to the `with ProcessPoolExecutor(...)` '
        f'shape, which is the whole of defect D7.  Offenders seen: '
        f'{offenders}; informational: {info}')
    assert any(o.startswith('_multi_parallel_results:')
               and ':with ProcessPoolExecutor' in o for o in offenders), (
        f'the site is reported, but not as the `with` shape the sibling '
        f'module actually uses: {offenders}')


def test_the_dispatcher_releases_its_claim_on_every_exit_path():
    """The claim is what keeps a rebuild off a live pool; a path that forgets
    to release it would wedge the rule shut for the rest of the process."""
    import inspect
    src = inspect.getsource(la.apply_real_lens_traced)
    i = src.index('def _invert_newton_parallel')
    j = src.index('\n    def ', i + 1)
    body = src[i:j]
    assert '_note_pool_inflight(1)' in body, (
        'the dispatch no longer claims the pool, so a concurrent rebuild can '
        'tear it down under chunks in flight')
    assert body.count('_note_pool_inflight(-1)') >= 2, (
        'the claim is taken but not released on every exit path')
    assert 'finally:' in body, (
        'the claim must be released in a finally, not only on the happy path')


def test_the_broken_pool_branch_still_backs_off_the_promotion():
    """A pool that just failed must not be rebuilt by the very next call: the
    dispatcher's back-off is ``close_worker_pool``, which clears the
    second-call promotion as well as the executor."""
    import inspect
    src = inspect.getsource(la.apply_real_lens_traced)
    i = src.index('def _invert_newton_parallel')
    j = src.index('\n    def ', i + 1)
    body = src[i:j]
    k = body.index('except (BrokenProcessPool')
    assert 'close_worker_pool()' in body[k:], (
        'the pool-infrastructure fallback no longer drops the cached pool')


# ===========================================================================
# 7.  The wedge REPORT itself (VERIFY-WP-B13 defect D1)
# ===========================================================================

def _b13_wedged_frame_for_the_dump(gate):
    """A uniquely named frame that parks forever.

    Module level on purpose: the dump has to name a frame that exists nowhere
    else in the tree, so matching it is a decision ("the report reached the
    wedged call") and not a substring coincidence.
    """
    gate.wait()


def test_the_wedge_report_carries_the_stack_of_the_wedged_frame():
    """A wedge detection must report ITS OWN message and a real thread dump.

    Every test in this file that can catch a wedge catches it through
    ``_with_deadline``, whose failure message is the only artifact a
    maintainer gets: the library call is on a daemon thread that is still
    stuck, so the stack is not in the traceback.  Shipped, ``_thread_dump``
    wrote to an ``io.StringIO``; ``faulthandler.dump_traceback`` writes
    through ``fileno()``, so the dump raised instead of being produced and
    all three wedge tests reported ``io.UnsupportedOperation: fileno`` with
    no stack at all (VERIFY-WP-B13 D1, reproduced with the pre-fix behaviour
    injected: 3 failed, each ending in that exception).

    The wedge here is ENGINEERED rather than hoped for: a daemon thread parks
    in a uniquely named module-level frame and the deadline is 1 s, so this
    test costs a second and asserts a decision -- the report names the frame
    that is stuck.
    """
    import io

    gate = threading.Event()
    try:
        with pytest.raises(pytest.fail.Exception) as err:
            _with_deadline(lambda: _b13_wedged_frame_for_the_dump(gate),
                           1.0, 'an ENGINEERED wedge')
    finally:
        gate.set()
    msg = str(err.value)

    # The premise, MEASURED on this build rather than quoted from the defect
    # report: what the shipped helper's sink does when faulthandler asks it
    # for a descriptor.  Reported, not asserted -- a CPython that grew a
    # StringIO path would not make the fix wrong, only redundant.
    try:
        faulthandler.dump_traceback(file=io.StringIO(), all_threads=True)
        premise = 'io.StringIO accepted the dump on this build'
    except Exception as exc:                      # noqa: BLE001 -- reported
        premise = f'io.StringIO raised {type(exc).__name__}: {exc}'

    assert 'an ENGINEERED wedge did not return within' in msg, (
        f'the wedge report lost _with_deadline\'s own message; that is D1 '
        f'(the dump helper raised before the message was built).  '
        f'Premise on this build: {premise}.  Got:\n{msg}')
    assert 'Thread dump:' in msg and '_b13_wedged_frame_for_the_dump' in msg, (
        f'the wedge report does not name the frame that is actually stuck, '
        f'so the one artifact a maintainer gets from a wedge is useless.  '
        f'Premise on this build: {premise}.  Got:\n{msg}')


def test_the_dump_helper_writes_through_a_real_descriptor():
    """The durable form of D1, independent of the deadline machinery.

    ``_thread_dump`` is called from exactly one place and only when something
    has already gone wrong, so a defect in it is invisible until the day it
    matters.  Call it directly and require that it produced this very frame.
    """
    dump = _thread_dump()
    assert 'test_the_dump_helper_writes_through_a_real_descriptor' in dump, (
        'the thread dump does not contain the calling frame, so it is not a '
        'thread dump; faulthandler needs a file with a real fileno()')


# ===========================================================================
# 8.  The abandoned-pool census (VERIFY-WP-B13 defect D2)
# ===========================================================================

def _settle(pred, seconds=15.0):
    """Wait (bounded) for a background thread to make ``pred`` true."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.005)
    return pred()


def test_an_expired_bounded_teardown_releases_its_executor_when_it_finishes():
    """``_ABANDONED_POOLS`` is a census of PENDING teardowns, not a ledger.

    The expiry path hands the executor to the list so a diagnostic can see
    that a teardown is still outstanding.  As shipped nothing ever took it
    back out -- unlike ``_abandon_pool``, whose reaper removes in a
    ``finally`` -- so the list grew by one per expiry and pinned each dead
    executor's ``_processes`` and queues for the life of the process
    (VERIFY-WP-B13 D2, measured: length 2 after both shutdowns had in fact
    completed).

    Both sides are asserted, because only the pair is a fix: while the
    teardown is outstanding the entry must be THERE (otherwise the census is
    useless), and once it completes the entry must be GONE.  The number of
    expiries that happened is carried by ``_POOL_SHUTDOWN_TIMEOUTS``, which
    is monotone and is asserted to stay so.
    """
    before_len = len(LT._ABANDONED_POOLS)
    before_timeouts = LT._POOL_SHUTDOWN_TIMEOUTS
    pool = _WedgedBrokenPool(broken=False)
    try:
        ok = LT._shutdown_pool_bounded(pool, 0.25)
        assert ok is False, (
            'a shutdown that never returns must report the expiry')
        assert LT._POOL_SHUTDOWN_TIMEOUTS == before_timeouts + 1
        assert len(LT._ABANDONED_POOLS) == before_len + 1, (
            'an outstanding teardown is not in the census, so nothing can '
            'see that this process is holding a dead executor')
        assert pool in LT._ABANDONED_POOLS
    finally:
        pool.release()
    assert _settle(lambda: pool not in LT._ABANDONED_POOLS), (
        'the teardown completed and the executor is STILL in '
        '_ABANDONED_POOLS; the list grows by one per expiry and pins every '
        'dead executor for the life of the process (D2)')
    assert len(LT._ABANDONED_POOLS) == before_len, (
        f'the census did not return to its prior length: '
        f'{len(LT._ABANDONED_POOLS)} vs {before_len}')
    assert LT._POOL_SHUTDOWN_TIMEOUTS == before_timeouts + 1, (
        'the count of expiries must stay monotone -- it is the diagnostic '
        'the census is NOT')


def test_repeated_expiries_do_not_grow_the_census():
    """The defect's own shape: 'the list grows monotonically with expiries'.

    Three driven expiries, each released in turn.  The count of expiries
    grows by three; the census ends where it started.  Under the shipped
    behaviour the census ends three longer, which is the fail-before.
    """
    before_len = len(LT._ABANDONED_POOLS)
    before_timeouts = LT._POOL_SHUTDOWN_TIMEOUTS
    pools = []
    try:
        for _ in range(3):
            p = _WedgedBrokenPool(broken=False)
            pools.append(p)
            assert LT._shutdown_pool_bounded(p, 0.1) is False
        assert len(LT._ABANDONED_POOLS) == before_len + 3
    finally:
        for p in pools:
            p.release()
    assert _settle(lambda: len(LT._ABANDONED_POOLS) == before_len), (
        f'three expiries left {len(LT._ABANDONED_POOLS) - before_len} '
        f'executors pinned after every one of them had completed')
    assert LT._POOL_SHUTDOWN_TIMEOUTS == before_timeouts + 3


def test_a_teardown_landing_exactly_on_the_expiry_leaves_nothing_behind():
    """The ordering the one-line fix does not cover.

    The helper can return in the same instant the caller's wait expires.  A
    bare ``finally: remove`` would then run BEFORE the caller's ``append``
    and leave exactly the entry it was meant to drop.  The interleaving is
    ENGINEERED here rather than waited for: the test holds
    ``_ABANDONED_POOLS_LOCK``, lets the bounded wait expire, releases the
    executor's join, and only then drops the lock -- so both the caller and
    the helper are queued on it and either may win.  Both orders are legal;
    the decision asserted is that both end with the census where it started.
    """
    before_len = len(LT._ABANDONED_POOLS)
    orders = []
    for _ in range(8):
        pool = _WedgedBrokenPool(broken=False)
        box = {}

        def _call(_p=pool, _b=box):
            _b['ret'] = LT._shutdown_pool_bounded(_p, 0.05)

        t = threading.Thread(target=_call, daemon=True)
        with LT._ABANDONED_POOLS_LOCK:
            t.start()
            time.sleep(0.25)            # the 0.05 s bound has expired
            pool.release()              # the join returns; both queue on us
            time.sleep(0.05)
        t.join(15.0)
        assert not t.is_alive(), 'the bounded teardown did not return'
        orders.append(box.get('ret'))
        assert _settle(lambda _p=pool: _p not in LT._ABANDONED_POOLS), (
            'a teardown that completed on the expiry boundary stayed in the '
            'census -- the race the ordering note in _shutdown_pool_bounded '
            'is about')
    assert len(LT._ABANDONED_POOLS) == before_len, (
        f'eight boundary races left {len(LT._ABANDONED_POOLS) - before_len} '
        f'executors pinned')
    assert set(orders) <= {True, False}, orders


class _HelperFirstLock:
    """The census lock, made to admit the teardown HELPER before the caller.

    ``threading.Lock`` makes no fairness promise, so the adverse arrival order
    cannot be produced by sleeping and hoping.  This wrapper produces it by
    construction: the ``lumenairy-newton-pool-close`` thread is let through
    immediately, and any other thread waits until that helper has finished its
    critical section.  It is installed through the module attribute, which is
    the same substitution point the library's own tests use.
    """

    def __init__(self, real):
        self._real = real
        self._helper_done = threading.Event()

    @staticmethod
    def _is_helper():
        return threading.current_thread().name == 'lumenairy-newton-pool-close'

    def acquire(self, *a, **kw):
        if not self._is_helper():
            self._helper_done.wait(20.0)
        return self._real.acquire(*a, **kw)

    def release(self):
        helper = self._is_helper()
        out = self._real.release()
        if helper:
            self._helper_done.set()
        return out

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def test_the_census_survives_the_helper_first_lock_order(monkeypatch):
    """Why the repair is not the one-line ``finally: remove``.

    Two threads reach the census at the boundary: the caller, whose bounded
    wait has expired and which wants to ADD the executor, and the helper,
    whose ``shutdown`` has just returned and which wants to REMOVE it.  If the
    helper's removal is allowed to run before the caller's append -- which is
    what a bare ``finally: remove`` ahead of ``done.set()`` permits -- the
    remove finds nothing, the append then runs, and the census keeps exactly
    the entry the repair was meant to drop.

    The shipped helper therefore publishes ``done`` BEFORE it takes the lock
    and the caller re-reads ``done`` under it, so this order ends with no
    append at all.  MEASURED both ways on 2026-09-15 (Windows 3.14.6 and WSL
    3.12.3): this test passes as shipped and fails with the one-line form
    injected (``FU_INJECT=d2one``, ``validation/probe_wp_b13_followups/``).
    """
    before_len = len(LT._ABANDONED_POOLS)
    before_timeouts = LT._POOL_SHUTDOWN_TIMEOUTS
    monkeypatch.setattr(LT, '_ABANDONED_POOLS_LOCK',
                        _HelperFirstLock(threading.Lock()))
    pool = _WedgedBrokenPool(broken=False)
    box = {}

    def _call():
        box['ret'] = LT._shutdown_pool_bounded(pool, 0.05)

    t = threading.Thread(target=_call, daemon=True)
    t.start()
    time.sleep(0.30)                    # the 0.05 s bound has expired
    pool.release()                      # now the helper reaches the census
    t.join(20.0)
    assert not t.is_alive(), (
        'the bounded teardown never returned under the helper-first order')
    assert _settle(lambda: pool not in LT._ABANDONED_POOLS), (
        'the executor is pinned in the census under the helper-first arrival '
        'order: the helper removed nothing and the caller then added it')
    assert len(LT._ABANDONED_POOLS) == before_len, (
        f'census length {len(LT._ABANDONED_POOLS)}, was {before_len}')
    # The teardown DID complete, so this is not an expiry and must not be
    # counted as one -- the counter is the durable diagnostic.
    assert box['ret'] is True, (
        'a teardown that completed before the caller reached the census was '
        'reported as an expiry')
    assert LT._POOL_SHUTDOWN_TIMEOUTS == before_timeouts, (
        'a completed teardown was counted as a bounded-wait expiry')


# ===========================================================================
# 9.  What _POOL_INFLIGHT counts (VERIFY-WP-B13 defect D4)
# ===========================================================================

def test_the_in_flight_counter_is_one_claim_per_dispatch_not_per_chunk():
    """The counter's comment said "chunks"; the counter counts DISPATCHES.

    VERIFY-WP-B13 D4.  The decision taken here was to fix the COMMENT rather
    than the counter, and the reason is the set of consumers: every read in
    the library and in its tests and probes is a zero-vs-non-zero read, so
    the magnitude carries no meaning and a per-chunk counter would only add
    two lock acquisitions per chunk.  This test pins the three facts that
    decision rests on, so the corrected comment cannot drift back:

    1. the dispatcher takes its claim exactly ONCE;
    2. it takes it BEFORE it submits anything, so the claim covers the whole
       dispatch rather than tracking the chunks;
    3. every comparison against ``_POOL_INFLIGHT`` in the module is against
       zero -- the moment one is not, the magnitude has acquired a meaning
       and the counter, not the comment, is what has to change;
    4. and no call site CONSUMES ``_note_pool_inflight``'s return value,
       which is the other route the magnitude has out of the module.

    RESTATED 2026-09-19 (VERIFY-WP-B13-FOLLOWUPS defects VD7 and VD8), in
    both directions it was wrong in:

    * VD7, a MISS.  Fact 3 walked ``ast.Compare`` nodes only, and the
      magnitude does not have to travel through one:
      ``_note_pool_inflight`` ends ``return _POOL_INFLIGHT``, so
      ``if _note_pool_inflight(0) > 1:`` is a magnitude read that no Compare
      on the Name can see, and the pin stayed green on it (driven,
      ``vf6_d4_readers.py::branch_d4_check`` shape 5).  Fact 4 closes that
      route by requiring every call site to be an expression STATEMENT, i.e.
      the returned count is discarded.  MEASURED 2026-09-19, grep + AST over
      all of ``lumenairy/`` on both builds: 4 call sites, 1 claim and 3
      releases, 0 of them consuming the value.
    * VD8, a FALSE FAILURE.  The non-zero operands were read from
      ``node.comparators`` alone, so ``0 < _POOL_INFLIGHT`` -- the same
      zero-vs-non-zero decision with the operands swapped -- put the Name
      itself in the "something other than zero" bucket and failed the pin
      (same driver, shape 2).  The operands are now read from
      ``[node.left, *node.comparators]`` MINUS the ``_POOL_INFLIGHT`` node,
      so the pin is about the decision rather than about which side of the
      operator the counter was written on.
    """
    import ast
    import inspect

    src = inspect.getsource(la.apply_real_lens_traced)
    i = src.index('def _invert_newton_parallel')
    j = src.index('\n    def ', i + 1)
    body = src[i:j]
    assert body.count('_note_pool_inflight(1)') == 1, (
        f'the dispatcher claims the pool '
        f'{body.count("_note_pool_inflight(1)")} times; the claim is one per '
        f'DISPATCH, which is what _POOL_INFLIGHT counts')
    assert body.index('_note_pool_inflight(1)') < body.index('ex.submit('), (
        'the claim is taken after the first submit, so it no longer covers '
        'the whole dispatch')

    tree = ast.parse(_module_source())
    comparisons = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        operands = [node.left, *node.comparators]
        if not any(isinstance(n, ast.Name) and n.id == '_POOL_INFLIGHT'
                   for n in operands):
            continue
        # BOTH compare orders are the same decision: read the OTHER operands
        # from the whole list minus the counter's own node, never from
        # ``comparators`` alone (VD8 -- ``0 < _POOL_INFLIGHT`` false-failed).
        others = [x for x in operands
                  if not (isinstance(x, ast.Name)
                          and x.id == '_POOL_INFLIGHT')]
        non_zero = not all(isinstance(x, ast.Constant) and x.value == 0
                           for x in others)
        comparisons.append((node.lineno, non_zero))
    assert comparisons, (
        'nothing compares _POOL_INFLIGHT any more -- the rebuild rule has '
        'lost the guard that keeps a rebuild off a live pool')
    against_non_zero = [ln for ln, non_zero in comparisons if non_zero]
    assert against_non_zero == [], (
        f'lines {against_non_zero} compare _POOL_INFLIGHT against something '
        f'other than zero.  Its magnitude is a count of DISPATCHES, not of '
        f'chunks; a consumer that needs chunks has to change the counter '
        f'(and its comment), not read this one')

    # 4. the OTHER route out of the module: ``_note_pool_inflight`` returns
    #    the count, so a call site that uses the returned value is a
    #    magnitude read no Compare above can see (VD7).
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == '_note_pool_inflight']
    discarded = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and isinstance(n.value.func, ast.Name)
                 and n.value.func.id == '_note_pool_inflight']
    assert calls, (
        'nothing calls _note_pool_inflight any more; the rebuild rule has '
        'lost the claim that keeps a rebuild off a live pool')
    assert len(discarded) == len(calls), (
        f'{len(calls) - len(discarded)} of {len(calls)} _note_pool_inflight '
        f'call sites (lines {[n.lineno for n in calls]}) CONSUME the return '
        f'value.  That value is the magnitude of a DISPATCH count, and the '
        f'D4 decision -- fix the comment, not the counter -- is only sound '
        f'while nothing reads it; a consumer that needs a chunk count has to '
        f'change the counter, not read this one')


# ===========================================================================
# D5 / VD3 -- what the dispatcher's infrastructure clause actually reaches
# ===========================================================================
class _PreThreeElevenError(Exception):
    """CPython < 3.11's ``concurrent.futures._base.Error``."""


class _PreThreeElevenTimeoutError(_PreThreeElevenError):
    """CPython < 3.11's ``concurrent.futures.TimeoutError``.

    Before gh-90315 ("concurrent.futures.TimeoutError and
    asyncio.TimeoutError are now aliases of TimeoutError", Python 3.11) this
    class derived from ``concurrent.futures._base.Error(Exception)`` and had
    NOTHING to do with ``OSError``.  Reconstructed here rather than imported,
    because no 3.10 interpreter is installed on this box -- the shape that
    matters is the MRO, and the MRO is what an ``except`` tuple matches on.
    """


class _RaisingPool:
    """A stub executor whose ``submit`` raises a chosen exception.

    Installed through the same substitution point every other test in this
    file uses, so what is exercised is the shipped dispatcher and the shipped
    ``except (BrokenProcessPool, RuntimeError, OSError, EOFError)`` tuple.
    """

    _broken = None

    def __init__(self, exc):
        self._exc = exc
        self.shutdown_calls = []

    def submit(self, fn, *a, **kw):
        raise self._exc

    def shutdown(self, wait=True, **kw):
        self.shutdown_calls.append(bool(wait))


def test_the_infrastructure_clause_reaches_an_oserror_timeout_and_no_other(
        monkeypatch, _forced_pool_bars, serial_reference):
    """What the shipped dispatcher does with each timeout MRO, driven.

    VERIFY-WP-B13-FOLLOWUPS defect VD3.  The follow-ups report recommended
    bounding the pool BOOTSTRAP and argued the fallback was already right
    because "``TimeoutError`` is an ``OSError`` subclass, so it lands in the
    dispatcher's existing infrastructure clause".  That is true of the
    BUILTIN on every supported interpreter (PEP 3151), and true of
    ``concurrent.futures.TimeoutError`` only from Python 3.11, where
    gh-90315 made it an alias of the builtin.  ``pyproject.toml`` declares
    ``requires-python = ">=3.10"`` and CI runs 3.10, so on a supported
    interpreter the pre-3.11 class -- whose MRO is
    ``TimeoutError -> Error -> Exception`` -- walks straight past
    ``(BrokenProcessPool, RuntimeError, OSError, EOFError)``.

    THIS TEST ASSERTS THE OUTCOME THE CODE SHIPS, not the outcome a future
    bar would need.  Nothing in ``_invert_newton_parallel`` asks for a
    timeout today -- neither ``as_completed`` nor ``Future.result`` is given
    one, which the last block below re-reads from the source -- so no path
    inside the ``try`` can raise either timeout class, the escape is
    unreachable, and adding ``concurrent.futures.TimeoutError`` to the tuple
    would be a no-op on 3.11+ (it IS ``OSError`` there) while on 3.10 it
    would newly swallow a WORKER-raised timeout into a silent serial re-run
    -- exactly what the tuple's own comment refuses to do for ``ValueError``,
    ``ImportError`` and ``MemoryError``.  The tuple is therefore left as it
    is, and the requirement recorded for whoever adds the bar: naming the
    timeout class is the FIRST line of that change, and
    ``test_verify_b13_followups.py::``
    ``test_a_timeout_on_the_dispatch_must_name_its_own_exception_class``
    turns red the day a ``timeout=`` lands without it.

    MEASURED 2026-09-19 through this same reproducer on Windows 3.14.6 and
    WSL 3.12.3 (``validation/probe_verify_b13_followups/vf5_d5_timing.py::``
    ``measure_clause_reach``, re-driven here): builtin MRO -> serial,
    byte-identical; pre-3.11 MRO -> escapes to the caller.
    """
    import concurrent.futures as cf

    # The premise, MEASURED on the running build rather than assumed: the
    # builtin is an OSError everywhere, and whether cf.TimeoutError IS the
    # builtin is exactly the 3.11 boundary this defect is about.
    measured = {
        'python': tuple(sys.version_info[:2]),
        'builtin_TimeoutError_is_OSError': issubclass(TimeoutError, OSError),
        'cf_TimeoutError_is_builtin': cf.TimeoutError is TimeoutError,
        'cf_TimeoutError_is_OSError': issubclass(cf.TimeoutError, OSError),
        'pre_3_11_mro': [c.__name__
                         for c in _PreThreeElevenTimeoutError.__mro__],
    }
    assert measured['builtin_TimeoutError_is_OSError'], measured
    assert not issubclass(_PreThreeElevenTimeoutError, OSError), (
        f'the reconstructed pre-3.11 class is an OSError, so it no longer '
        f'models the MRO the defect is about: {measured}')

    cases = [
        ('builtin TimeoutError', TimeoutError('engineered'), True),
        ('concurrent.futures.TimeoutError on this build',
         cf.TimeoutError('engineered'),
         measured['cf_TimeoutError_is_OSError']),
        ('pre-3.11 concurrent.futures.TimeoutError MRO',
         _PreThreeElevenTimeoutError('engineered'), False),
        ('OSError (control)', OSError('engineered'), True),
        ('ValueError (control)', ValueError('engineered'), False),
    ]
    rows = []
    for label, exc, want_caught in cases:
        pool = _RaisingPool(exc)
        with monkeypatch.context() as mp:
            _install_pool(mp, pool)
            try:
                got = _traced(4)
                caught, escaped = True, None
            except BaseException as raised:            # noqa: BLE001
                caught, escaped = False, type(raised).__name__
                got = None
        identical = (None if got is None
                     else bool(np.array_equal(got, serial_reference)))
        rows.append({'case': label, 'want_caught': want_caught,
                     'caught': caught, 'escaped_as': escaped,
                     'identical_to_serial': identical,
                     'mro': [c.__name__ for c in type(exc).__mro__]})
        verb = 'took the serial fallback' if caught else 'escaped'
        assert caught == want_caught, (
            f'{label}: the dispatcher {verb}, which is not what the shipped '
            f'except (BrokenProcessPool, RuntimeError, OSError, EOFError) '
            f'tuple says it should do.  This build measures {measured}; '
            f'rows so far: {rows}')
        if caught:
            assert identical, (
                f'{label} took the serial fallback but the answer moved -- '
                f'the fallback exists because the two paths are '
                f'bit-identical, max|delta| = '
                f'{np.abs(got - serial_reference).max():.3e}')

    # And the reason the escape above is unreachable today: the dispatcher
    # never asks for a timeout, so nothing inside its try can raise one.
    src = inspect.getsource(la.apply_real_lens_traced)
    i = src.index('def _invert_newton_parallel')
    j = src.index('\n    def ', i + 1)
    tree = ast.parse(textwrap.dedent(src[i:j]))
    timed = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, 'id', None) or getattr(
            node.func, 'attr', None)
        if name in ('as_completed', 'result', 'wait') and any(
                kw.arg == 'timeout' for kw in node.keywords):
            timed.append((name, node.lineno))
    assert timed == [], (
        f'the dispatcher now asks for a timeout at {timed}, so a timeout '
        f'class CAN be raised inside the try block.  The infrastructure '
        f'clause has to name that class explicitly before this ships: on '
        f'Python 3.10 -- inside this package requires-python -- '
        f'concurrent.futures.TimeoutError is not an OSError and would reach '
        f'the caller instead of the bit-identical serial rung.  Measured '
        f'MROs on this build: {measured}')

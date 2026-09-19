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

import faulthandler
import json
import os
import subprocess
import sys
import tempfile
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
            capture_output=True, text=True)
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
                              capture_output=True, text=True)
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


def test_only_the_bounded_helper_ever_joins_an_executor():
    """The durable form of the invariant.

    Bit-identity cannot see a join, and a wedge only shows up on a box under
    load, so the guard has to be structural: ``shutdown(wait=True)`` may
    appear in exactly one place -- the helper that runs it on another thread
    and joins THAT with a timeout.
    """
    import ast
    src = _module_source()
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
    offenders = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'shutdown'):
            continue
        waits = [kw.value for kw in node.keywords if kw.arg == 'wait']
        joining = (not waits) or any(
            not (isinstance(v, ast.Constant) and v.value is False)
            for v in waits)
        if joining and '_shutdown_pool_bounded' not in enclosing.get(
                node, set()):
            offenders.append(f'{owner.get(node)}:{node.lineno}')
    assert offenders == [], (
        f'{offenders} call shutdown(wait=True) directly.  CPython joins the '
        f'queue-feeder thread and every worker process inside '
        f'_terminate_broken, under the same lock shutdown takes first, so a '
        f'direct joining shutdown can wedge forever; route it through '
        f'_shutdown_pool_bounded')


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


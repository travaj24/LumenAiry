"""VERIFY-B13 -- the three Newton-pool decisions WP-B13's own file leaves open.

WP-B13 (`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/
WP-B13_NEWTON_POOL_REPORT.md`) makes three load-bearing claims that
`test_fix_newton_pool_broken_fallback.py` argues for in prose but does not
pin as a decision.  This file pins them, each as a property that a regression
makes FAIL rather than hang:

1. **The handoff's one-line fix is not sufficient.**  The whole two-mechanism
   design rests on the reading that CPython's
   ``_ExecutorManagerThread.terminate_broken`` holds the executor's
   ``_shutdown_lock`` across unbounded joins, and that
   ``ProcessPoolExecutor.shutdown`` takes that same lock BEFORE it looks at
   ``wait`` -- so even ``shutdown(wait=False, cancel_futures=True)`` blocks.
   Nothing measured that.  VERIFY-B13 measured it on python 3.14.6 (Windows)
   and 3.12.3 (WSL): the non-waiting shutdown blocked 19.998 s and 20.004 s
   against a 20.000 s hold, and 0.000 s on a healthy pool.  The test below is
   that measurement, shrunk to a few seconds, on whatever interpreter runs it.

2. **The clamp is honoured by the CHUNK COUNT, not by the pool's width.**
   That is the entire safety argument for the ceiling rule (a pool wider than
   the clamp is allowed to survive because the surplus workers stay idle).
   The shipped tests check the getter's bookkeeping; none of them checks that a
   dispatch on a WIDER pool still runs only ``n_cpu`` chunks at once.
   MEASURED by VERIFY-B13 on a 16-wide pool: peak concurrent chunks 3, 5 and
   10 against clamp answers 3, 5 and 10.

3. **A retired-but-wedged pool does not poison the next dispatch.**  The
   shipped file checks that ``close_worker_pool`` does not JOIN a broken pool;
   it does not check the thing a caller cares about -- that a fresh traced call
   completes, byte-identically, while that pool's reaper thread is still stuck
   inside ``shutdown``.

Everything that drives a process pool runs in a CHILD under
``subprocess.run(timeout=...)``, so no test here can hang the suite that runs
it; the CPython measurement runs in-process but joins only bounded waits.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                          '..', '..'))


def _run_child(tmp_path, name, body, timeout, extra_args=()):
    """Run ``body`` as a child script; FAIL (never hang) if it overruns."""
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
            f'wedged, which is the defect VERIFY-B13 is about.\n'
            f'stdout so far:\n{(exc.stdout or b"")!r}\n'
            f'stderr so far:\n{(exc.stderr or b"")!r}')
    return proc


def _result(proc, name):
    for line in proc.stdout.splitlines():
        if line.startswith('RESULT '):
            return json.loads(line[len('RESULT '):])
    pytest.fail(f'{name} printed no RESULT line.\nstdout:\n{proc.stdout}\n'
                f'stderr:\n{proc.stderr}')


# ===========================================================================
# 1.  shutdown(wait=False, cancel_futures=True) is NOT an escape
# ===========================================================================

def test_a_non_waiting_shutdown_still_blocks_on_a_terminating_executor():
    """Why this module may not call ``shutdown`` on the calling thread.

    The bar is DERIVED, not a per-build number: the manager thread is made to
    hold the lock for ``hold`` seconds, and the assertion is that a
    ``shutdown`` asking for NO wait is nonetheless delayed by most of that
    hold.  Half the hold is the decision boundary, and the two states it
    separates are 0.000 s (healthy; measured on both builds) and the full hold
    -- decades apart, so no build's scheduler noise can cross it.

    The same measurement at hold=20 s read 19.998 s (python 3.14.6, Windows)
    and 20.004 s (python 3.12.3, WSL) on 2026-09-15.

    PREMISE-GATED 2026-09-20 (5.48.0, CI run 35501791535): the mechanism is
    an interpreter fact.  From 3.12 the manager thread performs its broken
    teardown's join while HOLDING the executor's ``_shutdown_lock``, which is
    what a non-waiting ``shutdown`` then blocks on; on 3.10 and 3.11 the join
    runs outside that lock and ``shutdown`` returned in 0.000 s on both CI
    shards.  The wrapper now READS whether the lock is held at the join and
    the decision follows the reading on both arms: held -> blocked by most
    of the hold; not held -> returned promptly, and the interpreter must be
    older than 3.12 (a 3.12+ interpreter reading not-held means the layout
    this module's teardown is built for has changed, and that is a failure).
    """
    import concurrent.futures.process as cfp
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    hold = 6.0
    marks = {}
    # CPython renamed the method in 3.12 (3.10/3.11: join_executor_internals);
    # resolve the live name rather than pin one interpreter's spelling.
    _join_name = next((n for n in ('_join_executor_internals',
                                   'join_executor_internals')
                       if hasattr(cfp._ExecutorManagerThread, n)), None)
    assert _join_name is not None, (
        'concurrent.futures.process._ExecutorManagerThread has neither join '
        'method name this test knows; the interpreter changed the internals')
    orig = getattr(cfp._ExecutorManagerThread, _join_name)

    def slow(self, broken=False):
        marks.setdefault('entered', time.monotonic())
        # The premise, measured where it matters: is the executor's shutdown
        # lock held by this (manager) thread at the join?
        marks.setdefault('lock_held', bool(self.shutdown_lock.locked()))
        time.sleep(hold)
        marks['left'] = time.monotonic()
        return orig(self, broken=broken)

    setattr(cfp._ExecutorManagerThread, _join_name, slow)
    try:
        ex = ProcessPoolExecutor(max_workers=2,
                                 mp_context=mp.get_context('spawn'))
        try:
            list(ex.map(abs, range(2)))          # workers up and answering
            ex.submit(_suicide, 0)               # ...and now one dies
            deadline = time.monotonic() + 60.0
            while 'entered' not in marks and time.monotonic() < deadline:
                time.sleep(0.01)
            assert 'entered' in marks, (
                'the executor never entered its broken teardown, so this test '
                'did not exercise the state it is named for')
            t0 = time.monotonic()
            ex.shutdown(wait=False, cancel_futures=True)
            blocked = time.monotonic() - t0
        finally:
            marks.setdefault('entered', 0.0)
    finally:
        setattr(cfp._ExecutorManagerThread, _join_name, orig)

    held = marks.get('lock_held')
    assert held is not None, 'the join wrapper never ran, so the premise was not read'
    if held:
        assert blocked > 0.5 * hold, (
            f'shutdown(wait=False, cancel_futures=True) returned in {blocked:.3f} '
            f's while the manager thread held the executor\'s _shutdown_lock for '
            f'{hold:.1f} s.  If that is real, the handoff\'s one-line fix WOULD '
            f'have been enough and this module\'s two-mechanism teardown is '
            f'over-built -- re-derive it before relaxing anything.')
    else:
        assert blocked < 0.1 * hold, (
            f"the manager thread did NOT hold _shutdown_lock at its join, yet "
            f"shutdown(wait=False, cancel_futures=True) blocked for {blocked:.3f} "
            f"s of a {hold:.1f} s hold -- something other than the lock is "
            f"blocking it; re-derive the mechanism")
        assert sys.version_info < (3, 12), (
            f"python {sys.version.split()[0]} performed the broken teardown's "
            f"join WITHOUT holding _shutdown_lock; from 3.12 the join is under "
            f"the lock, which is the layout this module's two-mechanism "
            f"teardown is built for -- the interpreter changed the internals")


def _suicide(_):
    """Worker that leaves no result item: the executor sees a dead child."""
    os._exit(7)


# ===========================================================================
# 2.  The ceiling is priced by the chunk count
# ===========================================================================

_CEILING_SCRIPT = '''\
"""Child: a pool WIDER than the clamp must still run only clamp chunks.

Top level is literal constants and definitions only: a spawn worker re-imports
this module, and lumenairy refuses the pool for a __main__ whose body would
re-run (``_script_has_main_guard``).
"""
import json
import os
import sys
import tempfile
import time
import warnings

import numpy as np


def recording_chunk(args):
    """The real chunk worker, plus one (start, end) line per chunk served."""
    from lumenairy.elements import _lens_traced as LT
    t0 = time.time()
    out = LT._newton_invert_chunk(args)
    t1 = time.time()
    d = os.environ.get('VERIFY_B13_SPANDIR')
    if d:
        with open(os.path.join(d, str(os.getpid())), 'a') as fh:
            fh.write(repr(t0) + ' ' + repr(t1) + os.linesep)
    return out


def singlet(ap):
    return {'name': 'v', 'aperture_diameter': ap, 'thicknesses': [3e-3],
            'surfaces': [
                {'radius': 9e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'conic': 0.0,
                 'aspheric_coeffs': None},
                {'radius': -9e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'conic': 0.0,
                 'aspheric_coeffs': None}]}


def run(la, n_workers, N):
    ap = 3e-3
    dx = 2.2 * ap / N
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / 1.2e-3 ** 2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            E0, prescription=singlet(ap), wavelength=1.31e-6, dx=dx,
            ray_subsample=2, newton_fit='spline', n_workers=n_workers,
            parallel_amp=False, newton_amp_mask_rel=0.0,
            on_undersample='silent', on_aperture_beam='silent',
            on_pool_memory='silent'))


def peak_overlap(spans):
    edges = sorted([(a, 1) for a, _ in spans] + [(b, -1) for _, b in spans])
    cur = peak = 0
    for _, d in edges:
        cur += d
        peak = max(peak, cur)
    return peak


def maybe_inject_prefix(LT):
    """Put the PRE-FIX behaviour back, for the fail-before demonstration.

    ``VERIFY_B13_PREFIX=joins`` restores the unbounded, in-line
    ``shutdown(wait=True)`` both teardown routes used before WP-B13;
    ``rebuild`` restores the pre-fix getter (any change of worker count tears
    the pool down).  Unset in every normal run -- this exists so the fail-
    before is a command anyone can repeat, not a claim in a report.
    """
    mode = os.environ.get('VERIFY_B13_PREFIX', '')
    if mode in ('joins', 'both'):
        def abandon(ex):
            if ex is not None:
                ex.shutdown(wait=True)

        def bounded(ex, timeout=None):
            if ex is not None:
                ex.shutdown(wait=True)
            return True

        LT._abandon_pool = abandon
        LT._shutdown_pool_bounded = bounded
    if mode in ('rebuild', 'both'):
        def getter(n_workers):
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor
            with LT._PERSISTENT_POOL_LOCK:
                if LT._PERSISTENT_POOL is not None:
                    if LT._PERSISTENT_POOL_NWORKERS == n_workers:
                        return LT._PERSISTENT_POOL
                    try:
                        LT._PERSISTENT_POOL.shutdown(wait=False)
                    except (RuntimeError, OSError, BrokenPipeError):
                        pass
                    LT._PERSISTENT_POOL = None
                LT._PERSISTENT_POOL = ProcessPoolExecutor(
                    max_workers=int(n_workers),
                    mp_context=_mp.get_context('spawn'),
                    initializer=LT._newton_pool_init)
                LT._PERSISTENT_POOL_NWORKERS = int(n_workers)
                LT._POOL_RESIDENT_PAYLOAD_KEY = None
                return LT._PERSISTENT_POOL

        LT._get_persistent_worker_pool = getter


def main():
    import faulthandler
    faulthandler.enable()
    faulthandler.dump_traceback_later(120.0, exit=True)
    import lumenairy as la
    from lumenairy.elements import _lens_traced as LT
    assert os.path.abspath(la.__file__).startswith(sys.argv[1]), (
        la.__file__, sys.argv[1])
    maybe_inject_prefix(LT)
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1
    N, wide = 256, 8

    d = tempfile.mkdtemp(prefix='verifyb13')
    os.environ['VERIFY_B13_SPANDIR'] = d       # set BEFORE the workers spawn
    LT.close_worker_pool()
    ref = run(la, 1, N)
    LT.close_worker_pool()
    LT._newton_invert_chunk = recording_chunk
    la.set_max_ram(None)
    # FORCED precondition (5.48.0, CI run 35499375120): on a 2-CPU / 7 GB
    # runner the free-RAM pricing rule took the requested 8 down to 2, so the
    # pool was never wider than the clamp and this id could not exercise the
    # ceiling -- a resource-conditioned premise, which TESTING_STANDARDS says
    # to FORCE rather than read off the box.  The WIDE build alone bypasses
    # the pricing rule (the getter builds the requested width); the priced
    # call below still goes through the real resolver, which is what the
    # decision measures.
    _real_resolve_wide = LT._newton_resolve_workers
    LT._newton_resolve_workers = (lambda requested, n_total, fit_points, **kw:
                                  max(1, int(requested)))
    try:
        run(la, wide, N)                        # build the WIDE pool
    finally:
        LT._newton_resolve_workers = _real_resolve_wide
    pool_workers = LT._PERSISTENT_POOL_NWORKERS

    seen = []
    real_resolve = LT._newton_resolve_workers

    priced = []

    def spy(requested, n_total, fit_points, **kw):
        # The real pricing still runs and is RECORDED, but the clamp this id
        # dispatches with is FORCED to 2: the priced number is a reading of
        # the runner's free memory (the first 5.48.0 publish verification
        # read 1 on its runner, which is a SERIAL dispatch with no pool
        # chunks at all, while the same commit's main matrix read 2), and
        # what this id measures is the CEILING rule -- that an 8-wide pool
        # runs only `clamp` chunks at once -- not the pricing arithmetic,
        # which has its own pins.
        priced.append(int(real_resolve(requested, n_total, fit_points, **kw)))
        ans = 2
        seen.append(int(ans))
        return ans

    LT._newton_resolve_workers = spy
    for f in os.listdir(d):
        os.remove(os.path.join(d, f))
    # Price the next call down to a handful of workers: the clamp reads
    # min(psutil available, get_ram_budget()), so the budget knob moves it.
    n_total = (N // 2) * (N // 2)
    per_worker = LT._newton_worker_bytes(n_total / 2.0, 30625)
    la.set_max_ram(((2 + 0.5) * per_worker + 2.0e9) / 0.5 / 1e9)
    field = run(la, wide, N)
    clamp = max(seen)

    spans = []
    for f in os.listdir(d):
        with open(os.path.join(d, f)) as fh:
            for line in fh:
                if line.strip():
                    a, b = line.split()
                    spans.append((float(a), float(b)))
    LT._newton_invert_chunk = None
    la.set_max_ram(None)
    LT.close_worker_pool()
    faulthandler.cancel_dump_traceback_later()
    print('RESULT ' + json.dumps({
        'pool_workers': int(pool_workers or 0),
        'priced_by_ram': (max(priced) if priced else None),
        'clamp': int(clamp),
        'chunks_run': len(spans),
        'peak_concurrent': peak_overlap(spans),
        'identical': bool(np.array_equal(field, ref)),
        'lumenairy': la.__file__}))


if __name__ == '__main__':
    main()
'''


@pytest.mark.slow
def test_a_pool_wider_than_the_clamp_runs_only_clamp_chunks_at_once(tmp_path):
    """The ceiling rule's safety argument, measured rather than asserted.

    The rebuild rule lets a pool stay WIDER than the current call's clamped
    worker count.  That is only safe because the memory the clamp bounds is
    committed by the CHUNKS, and the dispatcher derives the chunk count from
    the clamp -- so the surplus workers stay idle.  This child builds a wide
    pool, prices the next dispatch down, and counts the chunk executions that
    actually OVERLAP in wall-clock time.

    The bar is an identity, not a tolerance: peak overlap must not exceed the
    clamp the dispatcher itself resolved (read through a spy, not recomputed,
    because the fit-grid size the clamp is priced on belongs to the call).
    """
    proc = _run_child(tmp_path, 'ceiling_child.py', _CEILING_SCRIPT, 240.0)
    res = _result(proc, 'ceiling_child.py')
    assert os.path.abspath(res['lumenairy']).startswith(_REPO_ROOT)
    assert res['pool_workers'] >= res['clamp'] + 1, (
        f'the pool was not wider than the clamp ({res["pool_workers"]} vs '
        f'{res["clamp"]}), so this run did not exercise the ceiling; the '
        f'budget arithmetic that prices the dispatch down has drifted')
    assert res['chunks_run'] >= res['clamp'], res
    assert res['peak_concurrent'] <= res['clamp'], (
        f'{res["peak_concurrent"]} chunks ran CONCURRENTLY on a '
        f'{res["pool_workers"]}-wide pool whose clamp answered '
        f'{res["clamp"]}.  The ceiling rule is only memory-safe while the '
        f'chunk count carries the clamp, so this is the rule failing, not a '
        f'timing artefact.')
    assert res['identical'], (
        'the pooled field on a wider-than-clamp pool is not byte-identical '
        'to the serial answer')


# ===========================================================================
# 3.  A wedged reaper does not poison the next dispatch
# ===========================================================================

_REAPER_SCRIPT = '''\
"""Child: retire a broken AND wedged pool, then dispatch again."""
import json
import os
import sys
import threading
import warnings

import numpy as np


class WedgedBrokenPool:
    """Broken on submit; its shutdown NEVER returns, waiting or not.

    That second half is what CPython's ``_shutdown_lock`` does to a parent
    while ``_terminate_broken`` is joining: the flags on ``shutdown`` do not
    matter, the lock does.
    """

    def __init__(self):
        self._broken = ('A child process terminated abruptly, the process '
                        'pool is not usable anymore')
        self.calls = []
        self.entered = threading.Event()
        self._never = threading.Event()

    def submit(self, fn, *a, **kw):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool
        f = Future()
        f.set_exception(BrokenProcessPool('verify-b13'))
        return f

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.calls.append([bool(wait), bool(cancel_futures)])
        self.entered.set()
        self._never.wait()


def singlet(ap):
    return {'name': 'v', 'aperture_diameter': ap, 'thicknesses': [3e-3],
            'surfaces': [
                {'radius': 9e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'conic': 0.0,
                 'aspheric_coeffs': None},
                {'radius': -9e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'conic': 0.0,
                 'aspheric_coeffs': None}]}


def run(la, n_workers, N):
    ap = 3e-3
    dx = 2.2 * ap / N
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / 1.2e-3 ** 2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            E0, prescription=singlet(ap), wavelength=1.31e-6, dx=dx,
            ray_subsample=2, newton_fit='spline', n_workers=n_workers,
            parallel_amp=False, newton_amp_mask_rel=0.0,
            on_undersample='silent', on_aperture_beam='silent',
            on_pool_memory='silent'))


def maybe_inject_prefix(LT):
    """Put the PRE-FIX behaviour back, for the fail-before demonstration.

    ``VERIFY_B13_PREFIX=joins`` restores the unbounded, in-line
    ``shutdown(wait=True)`` both teardown routes used before WP-B13;
    ``rebuild`` restores the pre-fix getter (any change of worker count tears
    the pool down).  Unset in every normal run -- this exists so the fail-
    before is a command anyone can repeat, not a claim in a report.
    """
    mode = os.environ.get('VERIFY_B13_PREFIX', '')
    if mode in ('joins', 'both'):
        def abandon(ex):
            if ex is not None:
                ex.shutdown(wait=True)

        def bounded(ex, timeout=None):
            if ex is not None:
                ex.shutdown(wait=True)
            return True

        LT._abandon_pool = abandon
        LT._shutdown_pool_bounded = bounded
    if mode in ('rebuild', 'both'):
        def getter(n_workers):
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor
            with LT._PERSISTENT_POOL_LOCK:
                if LT._PERSISTENT_POOL is not None:
                    if LT._PERSISTENT_POOL_NWORKERS == n_workers:
                        return LT._PERSISTENT_POOL
                    try:
                        LT._PERSISTENT_POOL.shutdown(wait=False)
                    except (RuntimeError, OSError, BrokenPipeError):
                        pass
                    LT._PERSISTENT_POOL = None
                LT._PERSISTENT_POOL = ProcessPoolExecutor(
                    max_workers=int(n_workers),
                    mp_context=_mp.get_context('spawn'),
                    initializer=LT._newton_pool_init)
                LT._PERSISTENT_POOL_NWORKERS = int(n_workers)
                LT._POOL_RESIDENT_PAYLOAD_KEY = None
                return LT._PERSISTENT_POOL

        LT._get_persistent_worker_pool = getter


def main():
    import faulthandler
    import time
    faulthandler.enable()
    faulthandler.dump_traceback_later(120.0, exit=True)
    import lumenairy as la
    from lumenairy.elements import _lens_traced as LT
    assert os.path.abspath(la.__file__).startswith(sys.argv[1]), (
        la.__file__, sys.argv[1])
    maybe_inject_prefix(LT)
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1
    N = 256
    LT.close_worker_pool()
    ref = run(la, 1, N)
    LT.close_worker_pool()

    wedged = WedgedBrokenPool()
    LT._PERSISTENT_POOL = wedged
    LT._PERSISTENT_POOL_NWORKERS = 4
    t0 = time.monotonic()
    LT.close_worker_pool()
    close_s = time.monotonic() - t0
    entered = wedged.entered.wait(60.0)
    t1 = time.monotonic()
    field = run(la, 4, N)
    dispatch_s = time.monotonic() - t1
    LT.close_worker_pool()
    faulthandler.cancel_dump_traceback_later()
    print('RESULT ' + json.dumps({
        'close_seconds': round(close_s, 3),
        'reaper_entered_shutdown': bool(entered),
        'reaper_shutdown_args': wedged.calls,
        'dispatch_seconds': round(dispatch_s, 3),
        'identical': bool(np.array_equal(field, ref)),
        'reaper_threads': [t.name for t in threading.enumerate()
                           if 'newton-pool' in t.name],
        'lumenairy': la.__file__}))


if __name__ == '__main__':
    main()
'''


@pytest.mark.slow
def test_a_dispatch_succeeds_while_a_retired_pool_s_reaper_is_still_wedged(
        tmp_path):
    """The caller-facing half of "close does not join a broken pool".

    Retiring a wedged pool is only useful if the NEXT call works.  The child
    hands the module a pool that is broken and whose ``shutdown`` never returns
    at all, closes it, and then runs a fresh traced call: the close must return
    promptly, the reaper must be the one left holding the wedge, and the new
    call must produce the ``n_workers=1`` answer byte for byte.

    The close bar (5 s) sits three decades below the unbounded state it guards
    and two above the measured healthy close (0.000 s here; 0.042-0.411 s over
    a 1-16 worker ladder, VERIFY-B13, 2026-09-15).
    """
    proc = _run_child(tmp_path, 'reaper_child.py', _REAPER_SCRIPT, 240.0)
    res = _result(proc, 'reaper_child.py')
    assert os.path.abspath(res['lumenairy']).startswith(_REPO_ROOT)
    assert res['reaper_entered_shutdown'], (
        'nothing ever called shutdown on the retired pool, so it was leaked '
        'rather than reaped')
    assert res['reaper_shutdown_args'] == [[False, True]], (
        f'the retired pool was asked for {res["reaper_shutdown_args"]}; a '
        f'broken pool must be asked for a NON-waiting, cancelling shutdown '
        f'and nothing else')
    assert res['close_seconds'] < 5.0, (
        f'close_worker_pool took {res["close_seconds"]:.3f} s on a pool whose '
        f'shutdown never returns -- it joined the wedge instead of handing it '
        f'to the reaper')
    assert res['reaper_threads'], (
        'no reaper thread is alive holding the wedged shutdown; the wedge '
        'must be parked on a daemon thread, not resolved')
    assert res['identical'], (
        'the dispatch that ran while the reaper was wedged did not reproduce '
        'the serial answer byte for byte')

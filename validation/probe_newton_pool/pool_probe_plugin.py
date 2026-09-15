"""pytest plugin: log every ProcessPoolExecutor life-cycle event.

Loaded with ``-p pool_probe_plugin`` (put this directory on ``PYTHONPATH``).
It answers the question the release-gate dumps could not: under pytest's
DEFAULT file-descriptor capture, WHO creates a process pool, with how many
workers, and what happens to it.

Every event is one JSON object on the file named by ``LUM_POOL_PROBE_LOG``
(default: ``pool_probe_events.jsonl`` in the current directory), so pytest's
capture cannot swallow it and a wedged phase shows up as a missing exit line.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback

_LOG = os.environ.get('LUM_POOL_PROBE_LOG', 'pool_probe_events.jsonl')
_LOCK = threading.Lock()
T0 = time.monotonic()


def emit(event, **kw):
    kw['event'] = event
    kw['t'] = round(time.monotonic() - T0, 4)
    kw['tid'] = threading.get_ident()
    kw['tname'] = threading.current_thread().name
    kw['pid'] = os.getpid()
    with _LOCK:
        with open(_LOG, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(kw) + '\n')


def _stack(depth=14):
    out = []
    for fr in traceback.extract_stack()[:-2][-depth:]:
        out.append(f'{os.path.basename(fr.filename)}:{fr.lineno}:{fr.name}')
    return out


def _install():
    import concurrent.futures.process as cfp

    _init = cfp.ProcessPoolExecutor.__init__
    _submit = cfp.ProcessPoolExecutor.submit
    _shutdown = cfp.ProcessPoolExecutor.shutdown
    _tb = cfp._ExecutorManagerThread._terminate_broken

    def init(self, max_workers=None, mp_context=None, initializer=None,
             initargs=(), *, max_tasks_per_child=None):
        emit('ppe_init', max_workers=max_workers,
             ctx=type(mp_context).__name__ if mp_context else None,
             stack=_stack())
        return _init(self, max_workers=max_workers, mp_context=mp_context,
                     initializer=initializer, initargs=initargs,
                     max_tasks_per_child=max_tasks_per_child)

    def submit(self, fn, /, *a, **kw):
        fut = _submit(self, fn, *a, **kw)
        emit('ppe_submit', pool=id(self), fn=getattr(fn, '__name__', str(fn)))
        return fut

    def shutdown(self, wait=True, *, cancel_futures=False):
        emit('ppe_shutdown_enter', pool=id(self), wait=bool(wait),
             cancel_futures=bool(cancel_futures),
             broken=bool(getattr(self, '_broken', None)), stack=_stack())
        t = time.monotonic()
        try:
            return _shutdown(self, wait=wait, cancel_futures=cancel_futures)
        finally:
            emit('ppe_shutdown_exit', pool=id(self),
                 seconds=round(time.monotonic() - t, 4))

    def terminate_broken(self, cause):
        emit('mgr_terminate_broken_enter',
             cause=(''.join(cause)[-800:] if cause else None))
        t = time.monotonic()
        try:
            return _tb(self, cause)
        finally:
            emit('mgr_terminate_broken_exit',
                 seconds=round(time.monotonic() - t, 4))

    cfp.ProcessPoolExecutor.__init__ = init
    cfp.ProcessPoolExecutor.submit = submit
    cfp.ProcessPoolExecutor.shutdown = shutdown
    cfp._ExecutorManagerThread._terminate_broken = terminate_broken


def _install_lens_traced():
    try:
        from lumenairy.elements import _lens_traced as lt
    except Exception as exc:  # noqa: BLE001
        emit('lens_traced_import_failed', text=str(exc)[:300])
        return

    _get = lt._get_persistent_worker_pool
    _close = lt.close_worker_pool
    _resolve = lt._newton_resolve_workers

    def get_persistent_worker_pool(n_workers):
        cached = lt._PERSISTENT_POOL_NWORKERS
        emit('getpool_enter', n_workers=int(n_workers), cached=cached,
             rebuild=bool(lt._PERSISTENT_POOL is not None
                          and cached != n_workers))
        ex = _get(n_workers)
        emit('getpool_exit', n_workers=int(n_workers), pool=id(ex))
        return ex

    def close_worker_pool():
        emit('close_enter', pool=id(lt._PERSISTENT_POOL), stack=_stack())
        t = time.monotonic()
        try:
            _close()
        finally:
            emit('close_exit', seconds=round(time.monotonic() - t, 4))

    def newton_resolve_workers(requested, n_total, fit_points, **kw):
        n = _resolve(requested, n_total, fit_points, **kw)
        emit('resolve', requested=int(requested), n_total=int(n_total),
             fit_points=int(fit_points), resolved=int(n))
        return n

    lt._get_persistent_worker_pool = get_persistent_worker_pool
    lt.close_worker_pool = close_worker_pool
    lt._newton_resolve_workers = newton_resolve_workers


_install()
emit('plugin_loaded', log=_LOG)


def pytest_configure(config):
    _install_lens_traced()
    emit('pytest_configure', capture=config.getoption('capture', None))

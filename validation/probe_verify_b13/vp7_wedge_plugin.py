"""VP7 -- a pytest plugin that puts the PRE-FIX defect back, in memory.

Loaded with ``-p validation.probe_verify_b13.vp7_wedge_plugin`` (or by path),
and switched by the environment variable ``VP7_INJECT``:

  joins    the two teardown helpers are replaced by the shipped-before
           behaviour: an UNBOUNDED, in-line ``shutdown(wait=True)``.  This is
           the wedge WP-B13 exists to remove.
  rebuild  ``_get_persistent_worker_pool`` is replaced by the pre-fix rule --
           any change of worker count tears the pool down and respawns, with
           no in-flight accounting and no ceiling.
  both     both of the above.

The point is DURABILITY, not coverage: a test that would HANG the suite under
a regression is a test that cannot be relied on to report it.  Each test must
go RED (or error) inside its own deadline instead.
"""
from __future__ import annotations

import os
import threading


def _install_joins(LT):
    def _abandon_pool(ex):
        if ex is None:
            return
        # PRE-FIX: join it here, on the caller's thread, with no bound.
        try:
            ex.shutdown(wait=True)
        except (RuntimeError, OSError, ValueError):
            pass

    def _shutdown_pool_bounded(ex, timeout=None):
        if ex is None:
            return True
        try:
            ex.shutdown(wait=True)
        except (RuntimeError, OSError, ValueError):
            pass
        return True

    LT._abandon_pool = _abandon_pool
    LT._shutdown_pool_bounded = _shutdown_pool_bounded


def _install_rebuild(LT):
    def _get_persistent_worker_pool(n_workers):
        # The shipped-before rule, verbatim in behaviour: an exact-match reuse
        # and a teardown-and-respawn on ANY change.
        with LT._PERSISTENT_POOL_LOCK:
            if LT._PERSISTENT_POOL is not None:
                if LT._PERSISTENT_POOL_NWORKERS == n_workers:
                    return LT._PERSISTENT_POOL
                try:
                    LT._PERSISTENT_POOL.shutdown(wait=False)
                except (RuntimeError, OSError, BrokenPipeError):
                    pass
                LT._PERSISTENT_POOL = None
            import multiprocessing as _mp
            from concurrent.futures import ProcessPoolExecutor
            LT._PERSISTENT_POOL = ProcessPoolExecutor(
                max_workers=int(n_workers),
                mp_context=_mp.get_context('spawn'),
                initializer=LT._newton_pool_init)
            LT._PERSISTENT_POOL_NWORKERS = int(n_workers)
            LT._POOL_RESIDENT_PAYLOAD_KEY = None
            return LT._PERSISTENT_POOL

    LT._get_persistent_worker_pool = _get_persistent_worker_pool


def pytest_configure(config):
    mode = os.environ.get('VP7_INJECT', '')
    if not mode:
        return
    from lumenairy.elements import _lens_traced as LT
    if mode in ('joins', 'both'):
        _install_joins(LT)
    if mode in ('rebuild', 'both'):
        _install_rebuild(LT)
    config._vp7_mode = mode
    print(f'\nVP7: injected pre-fix behaviour [{mode}] into '
          f'{LT.__file__}; threads={threading.active_count()}')

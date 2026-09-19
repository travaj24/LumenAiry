"""FU -- put a WP-B13 FOLLOW-UP defect back in memory, under pytest.

Loaded with ``-p validation.probe_wp_b13_followups.fu_prefix_plugin`` and
switched by ``FU_INJECT`` (comma-separated):

  dump    restore the pre-fix ``_thread_dump`` in
          ``tests/unit/test_fix_newton_pool_broken_fallback.py`` -- the
          ``io.StringIO`` sink of VERIFY-WP-B13 defect D1.  Every wedge
          detection then reports ``io.UnsupportedOperation: fileno`` and no
          thread dump.
  joins   the pre-fix teardown helpers (an unbounded, in-line
          ``shutdown(wait=True)``), reused verbatim from
          ``validation/probe_verify_b13/vp7_wedge_plugin.py`` so the two
          demonstrations cannot drift apart.  This is what makes the file's
          three wedge tests actually FIRE, which is the only state in which
          D1 is observable.
  d2      restore the pre-fix ``_shutdown_pool_bounded``: append the expired
          executor to ``_ABANDONED_POOLS`` and never remove it (D2).
  d2one   install the ONE-LINE repair the verification report asked for (a
          bare ``finally: remove`` ahead of ``done.set()``), to measure which
          of the three D2 pins it does and does not close.

The point is FAIL-BEFORE: each follow-up pin must go red with its defect put
back, and the reason it goes red must be the defect and not an accident.
"""
from __future__ import annotations

import os
import sys
import threading

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))


def _prefix_thread_dump():
    """``_thread_dump`` exactly as WP-B13 shipped it."""
    import faulthandler
    import io

    def _thread_dump() -> str:
        buf = io.StringIO()
        faulthandler.dump_traceback(file=buf, all_threads=True)
        return buf.getvalue()

    return _thread_dump


def _install_d2(LT):
    """``_shutdown_pool_bounded`` exactly as WP-B13 shipped it."""
    def _shutdown_pool_bounded(ex, timeout=None):
        if ex is None:
            return True
        if timeout is None:
            timeout = LT._POOL_SHUTDOWN_TIMEOUT
        done = threading.Event()

        def _run():
            try:
                ex.shutdown(wait=True)
            except (RuntimeError, OSError, ValueError):
                pass
            finally:
                done.set()

        try:
            threading.Thread(target=_run,
                             name='lumenairy-newton-pool-close',
                             daemon=True).start()
        except RuntimeError:
            LT._abandon_pool(ex)
            return False
        if done.wait(timeout):
            return True
        with LT._ABANDONED_POOLS_LOCK:
            LT._POOL_SHUTDOWN_TIMEOUTS += 1
            LT._ABANDONED_POOLS.append(ex)
        return False

    LT._shutdown_pool_bounded = _shutdown_pool_bounded


def _install_d2_oneliner(LT):
    """The one-line repair VERIFY-WP-B13 section 7 asked for, verbatim.

    ``finally: remove ...; done.set()`` in the helper thread, with the caller
    unchanged.  It closes the ordinary expiry but NOT the boundary case, in
    which the helper's ``remove`` runs before the caller's ``append``.
    """
    def _shutdown_pool_bounded(ex, timeout=None):
        if ex is None:
            return True
        if timeout is None:
            timeout = LT._POOL_SHUTDOWN_TIMEOUT
        done = threading.Event()

        def _run():
            try:
                ex.shutdown(wait=True)
            except (RuntimeError, OSError, ValueError):
                pass
            finally:
                with LT._ABANDONED_POOLS_LOCK:
                    try:
                        LT._ABANDONED_POOLS.remove(ex)
                    except ValueError:
                        pass
                done.set()

        try:
            threading.Thread(target=_run,
                             name='lumenairy-newton-pool-close',
                             daemon=True).start()
        except RuntimeError:
            LT._abandon_pool(ex)
            return False
        if done.wait(timeout):
            return True
        with LT._ABANDONED_POOLS_LOCK:
            LT._POOL_SHUTDOWN_TIMEOUTS += 1
            LT._ABANDONED_POOLS.append(ex)
        return False

    LT._shutdown_pool_bounded = _shutdown_pool_bounded


def _modes():
    return [m.strip() for m in os.environ.get('FU_INJECT', '').split(',')
            if m.strip()]


def pytest_configure(config):
    modes = _modes()
    if not modes:
        return
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from lumenairy.elements import _lens_traced as LT
    if 'joins' in modes:
        from validation.probe_verify_b13 import vp7_wedge_plugin as vp7
        vp7._install_joins(LT)
    if 'd2' in modes:
        _install_d2(LT)
    if 'd2one' in modes:
        _install_d2_oneliner(LT)
    print(f'\nFU: injected pre-fix behaviour {modes} into {LT.__file__}; '
          f'threads={threading.active_count()}')


def pytest_collection_modifyitems(session, config, items):
    """The ``dump`` arm patches the TEST module, so it runs after import."""
    if 'dump' not in _modes():
        return
    patched = set()
    for item in items:
        mod = getattr(item, 'module', None)
        if mod is None or mod.__name__ in patched:
            continue
        if hasattr(mod, '_thread_dump'):
            mod._thread_dump = _prefix_thread_dump()
            patched.add(mod.__name__)
    print(f'\nFU: restored the pre-fix _thread_dump in {sorted(patched)}')

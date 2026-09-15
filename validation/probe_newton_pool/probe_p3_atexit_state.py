"""P3 -- what is the state of ``_PERSISTENT_POOL`` when the atexit handler
runs, and does the atexit handler actually reap the workers?

Creates a persistent Newton pool through the library's own entry point, then
registers an atexit handler of its own AHEAD of the library's (atexit is LIFO,
so a handler registered LATER runs FIRST) and one AFTER, and reports the pool
global from each.  Also reports the live child PIDs at each point, which is
what says whether workers are orphaned.
"""
from __future__ import annotations

import atexit
import faulthandler
import json
import os
import sys
import time

T0 = time.monotonic()


def emit(event, **kw):
    kw['event'] = event
    kw['t'] = round(time.monotonic() - T0, 4)
    sys.stderr.write(json.dumps(kw) + '\n')
    sys.stderr.flush()


def children():
    try:
        import psutil
        return sorted(p.pid for p in psutil.Process().children(recursive=True))
    except Exception:  # noqa: BLE001
        return None


def main():
    faulthandler.enable()
    faulthandler.dump_traceback_later(120.0, exit=True)
    import lumenairy  # noqa: F401
    from lumenairy.elements import _lens_traced as lt
    emit('import', file=lumenairy.__file__, pid=os.getpid())

    ex = lt._get_persistent_worker_pool(3)
    # Force the workers to actually start.
    list(ex.map(abs, [-1, -2, -3, -4, -5, -6]))
    emit('pool_up', pool=id(ex), nworkers=lt._PERSISTENT_POOL_NWORKERS,
         children=children())

    # Registered LAST -> runs FIRST (atexit is LIFO), i.e. BEFORE the
    # library's own close_worker_pool handler.
    def before_lib():
        emit('atexit_before_library',
             pool_is_none=lt._PERSISTENT_POOL is None,
             nworkers=lt._PERSISTENT_POOL_NWORKERS, children=children())

    atexit.register(before_lib)
    emit('registered')
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(120.0, exit=True)


if __name__ == '__main__':
    main()

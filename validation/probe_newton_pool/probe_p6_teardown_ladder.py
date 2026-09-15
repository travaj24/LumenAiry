"""P6 -- the numbers the bounded teardown's bar and the ceiling rule rest on.

1. HEALTHY TEARDOWN LADDER.  ``close_worker_pool``'s bound
   (``_POOL_SHUTDOWN_TIMEOUT``) has to sit decades above what a healthy pool
   costs to shut down and infinitely below the state it guards (an unbounded
   join inside CPython's ``_terminate_broken``).  This measures the healthy
   side over a worker ladder, with the pool WARM (its workers have each run a
   task, which is the state ``close_worker_pool`` is normally called in).

2. IDLE-WORKER FOOTPRINT.  The rebuild rule keeps a pool that is wider than
   the current call's clamped worker count rather than respawning a narrower
   one.  The cost of that choice is the resident set of the workers that stay
   idle; this measures it per worker so the trade is a number rather than an
   assertion.

Output: one JSON object per row on stdout, plus a final summary object.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import time

T0 = 0.0


def emit(event, **kw):
    kw['event'] = event
    kw['t'] = round(time.monotonic() - T0, 4)
    sys.stdout.write(json.dumps(kw) + '\n')
    sys.stdout.flush()


def _rss_of(pids):
    try:
        import psutil
    except ImportError:
        return None
    out = []
    for pid in pids:
        try:
            out.append(psutil.Process(pid).memory_info().rss)
        except Exception:  # noqa: BLE001 -- the worker may already be gone
            pass
    return out


def main():
    global T0
    T0 = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument('--ladder', default='1,2,4,8,16')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    faulthandler.enable()
    faulthandler.dump_traceback_later(900.0, exit=True)

    import lumenairy
    from lumenairy.elements import _lens_traced as LT
    emit('import', lumenairy=lumenairy.__file__, python=sys.version.split()[0],
         pid=os.getpid(), bound=LT._POOL_SHUTDOWN_TIMEOUT)

    ladder = [int(x) for x in args.ladder.split(',') if x]
    rows = []
    idle = []
    for n in ladder:
        for rep in range(args.reps):
            LT.close_worker_pool()
            ex = LT._get_persistent_worker_pool(n)
            # Warm every worker: a pool whose children have run is the state
            # close_worker_pool is normally called in.
            list(ex.map(abs, list(range(-4 * n, 0))))
            pids = [p.pid for p in getattr(ex, '_processes', {}).values()]
            rss = _rss_of(pids)
            if rss:
                idle.extend(rss)
            t = time.monotonic()
            LT.close_worker_pool()
            dt = time.monotonic() - t
            rows.append({'n_workers': n, 'rep': rep,
                         'close_seconds': round(dt, 4),
                         'worker_rss_mb': ([round(r / 2 ** 20, 1)
                                            for r in rss] if rss else None)}
                        )
            emit('close_row', **rows[-1])

    worst = max(r['close_seconds'] for r in rows)
    summary = {
        'n_rows': len(rows),
        'close_seconds_max': worst,
        'close_seconds_min': min(r['close_seconds'] for r in rows),
        'bound': LT._POOL_SHUTDOWN_TIMEOUT,
        'bound_over_worst': round(LT._POOL_SHUTDOWN_TIMEOUT / worst, 1)
        if worst > 0 else None,
        'idle_worker_rss_mb_mean': (round(sum(idle) / len(idle) / 2 ** 20, 1)
                                    if idle else None),
        'idle_worker_rss_mb_max': (round(max(idle) / 2 ** 20, 1)
                                   if idle else None),
        'shutdown_timeouts': LT._POOL_SHUTDOWN_TIMEOUTS,
        'abandoned_left': len(LT._ABANDONED_POOLS),
        'python': sys.version.split()[0],
    }
    emit('summary', **summary)
    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump({'rows': rows, 'summary': summary}, fh, indent=1)
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()

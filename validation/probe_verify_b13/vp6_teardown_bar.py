"""VP6 -- the bounded-wait bar, re-measured, and what an EXPIRY costs.

Claim (3) of the WP report puts ``_POOL_SHUTDOWN_TIMEOUT`` at 120 s and
justifies it with a healthy ladder whose worst rung was 4.592 s at 16 workers.
This probe re-measures that ladder through the PUBLIC teardown entry point
(``close_worker_pool``) after a real dispatch has warmed the pool, and then
asks the question the report does not: when the bounded wait EXPIRES, what is
left behind?

MODES

ladder   close_worker_pool seconds over a worker ladder, reps each, with the
         per-worker resident set measured both when the workers are IDLE and
         right after they have served a real Newton chunk (the report quotes
         only the idle figure).
expiry   ``_POOL_SHUTDOWN_TIMEOUT`` is driven to a value the healthy teardown
         cannot meet, so every close expires: count the expiries, then follow
         the worker pids and the parent's resident set to see whether an
         expiry orphans processes or memory.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
import sys
import threading
import time
import warnings

import numpy as np

T0 = 0.0
_WL = 1.31e-6


def emit(event, **kw):
    kw['event'] = event
    kw['t'] = round(time.monotonic() - T0, 4)
    sys.stdout.write(json.dumps(kw, sort_keys=True) + "\n")
    sys.stdout.flush()


def _singlet(ap, r):
    return {'name': 'vp6', 'aperture_diameter': ap, 'thicknesses': [3e-3],
            'surfaces': [
                {'radius': r, 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'conic': 0.0, 'aspheric_coeffs': None},
                {'radius': -r, 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'conic': 0.0, 'aspheric_coeffs': None}]}


def _traced(la, n_workers, N, rs):
    ap = 3e-3
    dx = 2.2 * ap / N
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = la.apply_real_lens_traced(
            E0, prescription=_singlet(ap, 9e-3), wavelength=_WL, dx=dx,
            ray_subsample=rs, newton_fit='spline', n_workers=n_workers,
            parallel_amp=False, newton_amp_mask_rel=0.0,
            on_undersample='silent', on_aperture_beam='silent',
            on_pool_memory='silent')
    return np.asarray(out)


def _worker_rss(LT):
    import psutil
    ex = LT._PERSISTENT_POOL
    procs = getattr(ex, '_processes', None) if ex is not None else None
    rss = []
    for p in (procs or {}).values():
        try:
            rss.append(psutil.Process(p.pid).memory_info().rss / 1e6)
        except Exception:
            pass
    return rss


class _StuckHealthyPool:
    """Looks healthy; its joining shutdown never returns."""

    def __init__(self, n):
        self.n = n
        self._broken = None
        self.calls = []
        self._never = threading.Event()

    def submit(self, fn, *a, **kw):
        from concurrent.futures import Future
        f = Future()
        f.set_result(None)
        return f

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.calls.append({'wait': bool(wait),
                           'cancel_futures': bool(cancel_futures)})
        if wait:
            self._never.wait()


def mode_ladder(la, LT, a):
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1
    rows = []
    for w in [int(x) for x in a.ladder.split(',')]:
        for rep in range(a.reps):
            LT.close_worker_pool()
            la.set_max_ram(None)
            _traced(la, w, a.N, a.rs)                 # build + warm + dispatch
            if LT._PERSISTENT_POOL is None:
                rows.append({'workers': w, 'rep': rep, 'skipped': True})
                continue
            got = LT._PERSISTENT_POOL_NWORKERS
            rss_after_work = _worker_rss(LT)
            time.sleep(a.settle)
            rss_idle = _worker_rss(LT)
            t = time.monotonic()
            LT.close_worker_pool()
            dt = time.monotonic() - t
            rows.append({'workers': w, 'pool_workers': got, 'rep': rep,
                         'close_seconds': round(dt, 3),
                         'rss_after_chunk_mb': [round(x, 1)
                                                for x in rss_after_work],
                         'rss_idle_mb': [round(x, 1) for x in rss_idle],
                         'timeouts': LT._POOL_SHUTDOWN_TIMEOUTS})
            emit('rung', **rows[-1])
    closes = [r['close_seconds'] for r in rows if 'close_seconds' in r]
    idle = [x for r in rows for x in r.get('rss_idle_mb', [])]
    work = [x for r in rows for x in r.get('rss_after_chunk_mb', [])]
    return {'mode': 'ladder', 'rows': rows,
            'worst_close_seconds': max(closes) if closes else None,
            'bar': LT._POOL_SHUTDOWN_TIMEOUT,
            'margin_x': (round(LT._POOL_SHUTDOWN_TIMEOUT / max(closes), 1)
                         if closes else None),
            'idle_worker_mb_mean': round(float(np.mean(idle)), 1) if idle else None,
            'idle_worker_mb_max': round(max(idle), 1) if idle else None,
            'post_chunk_worker_mb_mean': (round(float(np.mean(work)), 1)
                                          if work else None),
            'post_chunk_worker_mb_max': round(max(work), 1) if work else None}


def mode_expiry(la, LT, a):
    import psutil
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1
    out = {'mode': 'expiry', 'bar_used': a.bar}
    # (1) a STUB healthy-looking pool whose join never returns: the bounded
    #     wait must expire, count it, and return.
    LT.close_worker_pool()
    LT._POOL_SHUTDOWN_TIMEOUT = a.bar
    stub = _StuckHealthyPool(4)
    LT._PERSISTENT_POOL = stub
    LT._PERSISTENT_POOL_NWORKERS = 4
    before = LT._POOL_SHUTDOWN_TIMEOUTS
    t = time.monotonic()
    LT.close_worker_pool()
    out['stub_close_seconds'] = round(time.monotonic() - t, 3)
    out['stub_timeouts_delta'] = LT._POOL_SHUTDOWN_TIMEOUTS - before
    out['stub_shutdown_calls'] = stub.calls
    out['stub_in_abandoned'] = any(x is stub for x in LT._ABANDONED_POOLS)
    out['stub_close_threads_alive'] = [
        t.name for t in threading.enumerate()
        if 'lumenairy-newton-pool' in t.name]
    emit('stub_expiry', **{k: out[k] for k in list(out) if k != 'mode'})

    # (2) a REAL pool torn down with a bar it cannot meet: what is orphaned?
    LT.close_worker_pool()
    LT._POOL_SHUTDOWN_TIMEOUT = a.bar
    la.set_max_ram(None)
    _traced(la, a.workers, a.N, a.rs)
    ex = LT._PERSISTENT_POOL
    pids = [p.pid for p in (getattr(ex, '_processes', {}) or {}).values()]
    before = LT._POOL_SHUTDOWN_TIMEOUTS
    parent_rss0 = psutil.Process().memory_info().rss / 1e6
    t = time.monotonic()
    LT.close_worker_pool()
    out['real_close_seconds'] = round(time.monotonic() - t, 3)
    out['real_timeouts_delta'] = LT._POOL_SHUTDOWN_TIMEOUTS - before
    out['real_worker_pids'] = pids
    alive = []
    for wait_s in (0.0, 1.0, 5.0, 30.0):
        if wait_s:
            time.sleep(wait_s - (alive[-1]['at'] if alive else 0.0))
        n = 0
        for pid in pids:
            try:
                p = psutil.Process(pid)
                if p.is_running() and p.status() != psutil.STATUS_ZOMBIE:
                    n += 1
            except Exception:
                pass
        alive.append({'at': wait_s, 'workers_alive': n})
        emit('orphan_check', **alive[-1])
    out['orphan_ladder'] = alive
    out['parent_rss_mb_before'] = round(parent_rss0, 1)
    out['parent_rss_mb_after'] = round(
        psutil.Process().memory_info().rss / 1e6, 1)
    out['abandoned_len'] = len(LT._ABANDONED_POOLS)
    return out


def main():
    global T0
    T0 = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('ladder', 'expiry'), default='ladder')
    ap.add_argument('--ladder', default='1,2,4,8,16')
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--bar', type=float, default=0.05)
    ap.add_argument('--settle', type=float, default=1.0)
    ap.add_argument('--N', type=int, default=512)
    ap.add_argument('--rs', type=int, default=2)
    ap.add_argument('--dump-after', type=float, default=1200.0)
    ap.add_argument('--json-out', default=None)
    a = ap.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(a.dump_after, exit=True)
    import lumenairy as la
    from lumenairy.elements import _lens_traced as LT
    emit('import', lumenairy=la.__file__, python=sys.version.split()[0],
         pid=os.getpid(), mode=a.mode)
    out = (mode_ladder(la, LT, a) if a.mode == 'ladder'
           else mode_expiry(la, LT, a))
    out['python'] = sys.version.split()[0]
    out['lumenairy'] = la.__file__
    emit('dispatch_returned', **out)
    if a.json_out:
        with open(a.json_out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    faulthandler.cancel_dump_traceback_later()
    emit('exiting')
    return 0


if __name__ == '__main__':
    sys.exit(main())

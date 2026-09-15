"""P4 -- the rebuild race, and the bit-identity of the fallback.

Two measurements on the SHIPPED traced-lens path, both driven without pytest:

RACE
    Thread A runs a traced-carrier chain whose Newton inversion dispatches on
    the persistent pool.  Thread B, concurrently, asks
    ``_get_persistent_worker_pool`` for a DIFFERENT worker count, which is
    what the shipped getter answers by tearing the live pool down and
    respawning.  The question the probe answers is what thread A then sees:
    a completed dispatch, a ``BrokenProcessPool`` that reaches the serial
    fallback, or a wedge.

    The worker count really does move between calls on this box: a single
    ``test_audit2609_b4_collins_transport`` run measured
    ``_newton_resolve_workers`` answering 6, 8, 4, 8 for the same two lens
    groups, because the clamp reads LIVE free memory.  Thread B only makes
    that deterministic.

IDENTITY
    ``--identity K`` runs the same chain at ``n_workers=1`` (serial Newton)
    and at ``n_workers=K`` (pooled) and compares the two fields BYTE for
    BYTE, which is the contract the broken-pool fallback relies on: falling
    back to serial may not move a number.

Every phase is a JSON line with a monotonic timestamp and a thread id; a
``faulthandler`` deadline is armed throughout so the probe aborts with a full
thread dump rather than joining the hang it is studying.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import sys
import threading
import time
import warnings

import numpy as np

# Module body kept free of top-level WORK: a spawn worker re-imports
# __main__, and ``_script_has_main_guard`` (rightly) refuses the pool for
# a script whose body does anything but imports, defs and literals.
T0 = 0.0
_EMIT_LOCK = None


def emit(event, **kw):
    kw['event'] = event
    kw['t'] = round(time.monotonic() - T0, 4)
    kw['tname'] = threading.current_thread().name
    with _EMIT_LOCK:
        sys.stdout.write(json.dumps(kw) + '\n')
        sys.stdout.flush()


_WL = 1.31e-6
_W0 = 1.0e-3
_N, _RS = 1024, 2          # 262144 Newton points: past the cold pool bar


def _doublet(ap):
    surfs, before = [], 'air'
    for R, g in ((61.5e-3, 'N-BK7'), (-45.0e-3, 'N-SF5'), (-128.0e-3, 'air')):
        surfs.append({'radius': R, 'glass_before': before, 'glass_after': g,
                      'conic': 0.0, 'radius_y': None, 'conic_y': None,
                      'aspheric_coeffs': None, 'aspheric_coeffs_y': None})
        before = g
    return {'name': 'doublet', 'aperture_diameter': ap,
            'thicknesses': [4.0e-3, 2.5e-3], 'surfaces': surfs}


def _run(n_workers):
    import lumenairy as la
    ap = 1.2 * 2.0 * _W0
    dx = float(2.2 * max(ap, 3.0 * _W0) / _N)
    x = (np.arange(_N) - _N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env0 = np.exp(-(X ** 2 + Y ** 2) / _W0 ** 2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = la.propagate_traced_carrier_chain(
            env0, [{'prescription': _doublet(ap), 'gap_before': 0.0}],
            _WL, dx, r_in=np.inf, ray_subsample=_RS, n_workers=n_workers,
            final_distance=0.0,
            traced_kwargs=dict(parallel_amp=False, on_undersample='silent',
                               newton_fit='polynomial'))
    return np.asarray(res.field)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def instrument(lt, counters):
    _get = lt._get_persistent_worker_pool
    _close = lt.close_worker_pool
    _resolve = lt._newton_resolve_workers

    def resolve(requested, n_total, fit_points, **kw):
        n = _resolve(requested, n_total, fit_points, **kw)
        emit('resolve', requested=int(requested), n_total=int(n_total),
             fit_points=int(fit_points), resolved=int(n))
        return n

    lt._newton_resolve_workers = resolve

    def get_pool(n_workers):
        cached = lt._PERSISTENT_POOL_NWORKERS
        rebuild = (lt._PERSISTENT_POOL is not None and cached != n_workers)
        if rebuild:
            counters['rebuild'] += 1
        emit('getpool_enter', n=int(n_workers), cached=cached,
             rebuild=bool(rebuild))
        ex = _get(n_workers)
        emit('getpool_exit', n=int(n_workers),
             pool=id(ex) if ex is not None else None)
        return ex

    def close_pool():
        counters['close'] += 1
        emit('close_enter', pool=id(lt._PERSISTENT_POOL))
        t = time.monotonic()
        try:
            _close()
        finally:
            emit('close_exit', seconds=round(time.monotonic() - t, 4))

    lt._get_persistent_worker_pool = get_pool
    lt.close_worker_pool = close_pool


def scenario_race(args, lt, counters):
    out = {}
    stop = threading.Event()

    def arm_a():
        try:
            t = time.monotonic()
            f = _run(args.workers)
            out['a'] = {'sha': _sha(f), 'seconds': round(
                time.monotonic() - t, 3)}
            emit('arm_a_done', **out['a'])
        except BaseException as exc:  # noqa: BLE001
            import traceback
            out['a'] = {'exc': type(exc).__name__, 'text': str(exc)[:400]}
            emit('arm_a_raise', tb=traceback.format_exc()[-1500:], **out['a'])
        finally:
            stop.set()

    def arm_b():
        # Force a rebuild every `--churn` seconds by asking for a worker
        # count the live pool does not have -- exactly what the memory clamp
        # does on its own when free RAM moves between calls.
        alt = [args.workers_alt, args.workers]
        i = 0
        while not stop.wait(args.churn):
            try:
                lt._get_persistent_worker_pool(alt[i % 2])
            except BaseException as exc:  # noqa: BLE001
                emit('arm_b_raise', exc=type(exc).__name__,
                     text=str(exc)[:300])
            i += 1
        emit('arm_b_stop', iterations=i)

    ta = threading.Thread(target=arm_a, name='arm-A-chain')
    tb = threading.Thread(target=arm_b, name='arm-B-churn', daemon=True)
    ta.start()
    time.sleep(args.churn_delay)
    tb.start()
    ta.join()
    stop.set()
    tb.join(timeout=10.0)
    return out


def scenario_identity(args, lt, counters):
    out = {}
    for nw in (1, args.workers):
        lt.close_worker_pool()
        t = time.monotonic()
        f = _run(nw)
        out[str(nw)] = {'sha': _sha(f), 'seconds': round(
            time.monotonic() - t, 3), 'shape': list(f.shape),
            'dtype': str(f.dtype)}
        emit('identity_arm', n_workers=nw, **out[str(nw)])
    out['identical'] = out['1']['sha'] == out[str(args.workers)]['sha']
    emit('identity_verdict', identical=out['identical'])
    return out


def main():
    global T0, _EMIT_LOCK
    T0 = time.monotonic()
    _EMIT_LOCK = threading.Lock()
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=('race', 'identity'),
                    default='race')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--workers-alt', type=int, default=2)
    ap.add_argument('--churn', type=float, default=0.25)
    ap.add_argument('--churn-delay', type=float, default=0.0)
    ap.add_argument('--dump-after', type=float, default=600.0)
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    faulthandler.enable()
    faulthandler.dump_traceback_later(args.dump_after, exit=True)

    import lumenairy
    from lumenairy.elements import _lens_traced as lt
    emit('import', lumenairy=lumenairy.__file__,
         version=lumenairy.__version__, python=sys.version.split()[0],
         pid=os.getpid())
    counters = {'rebuild': 0, 'close': 0}
    instrument(lt, counters)

    if args.scenario == 'race':
        out = scenario_race(args, lt, counters)
    else:
        out = scenario_identity(args, lt, counters)
    out['counters'] = counters
    emit('summary', **out)
    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    lt.close_worker_pool()
    faulthandler.cancel_dump_traceback_later()
    return 0


if __name__ == '__main__':
    sys.exit(main())

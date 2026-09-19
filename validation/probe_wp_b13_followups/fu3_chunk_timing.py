"""FU3 -- what a timeout on `as_completed` would have to clear (D5).

VERIFY-WP-B13 D5, a MAINTAINER DECISION left open here: `as_completed` in
``_invert_newton_parallel`` has no timeout, so a worker that never comes up
hangs the call forever (reproducible on demand,
``validation/probe_verify_b13/vp2_broken_drivers.py --mode slowboot``).
Nothing is changed by this probe; it measures the HEALTHY side, which is the
side a bar has to clear.

Three quantities, because `concurrent.futures.as_completed(fs, timeout=T)`
does NOT bound a chunk: ``T`` is an absolute deadline taken when the
generator is created, so it bounds the WHOLE iteration.

  ``first_result_seconds``   first submit -> first future done.  This is the
                             quantity that actually separates "a slow box"
                             from "a worker that never came up", and it is
                             nearly independent of N.
  ``slowest_chunk_seconds``  submit -> done, per chunk, worst of the run.
  ``total_seconds``          first submit -> last future done.  This is what
                             a single ``timeout=`` on ``as_completed`` bounds,
                             and it grows with N.

Each N is run COLD (``close_worker_pool()`` first, so the first chunk pays
the spawn bootstrap) and WARM (pool already built and already payload-
resident).  The spawn bootstrap is also measured on its own: every worker is
forced to take exactly one task by a Manager Barrier of the pool's width, so
"all W workers bootstrapped" is a decision rather than a sample.

    PYTHONPATH=<tree> python validation/probe_wp_b13_followups/
    fu3_chunk_timing.py --workers 8 --n 256 512 1024 --out fu3_win.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings


def bootstrap_task(barrier):
    """What a Newton chunk pays before it computes anything."""
    import scipy.interpolate  # noqa: F401

    from lumenairy.elements import _lens_traced  # noqa: F401
    t_ready = time.monotonic()
    try:
        barrier.wait(300.0)
    except Exception:                             # noqa: BLE001
        pass
    return {'pid': os.getpid(), 'ready': t_ready}


class _TimingExecutor:
    """Wraps the real executor and timestamps every chunk.

    The dispatcher takes whatever ``_get_persistent_worker_pool`` returns --
    a substitution point the library's own tests rely on -- so this measures
    the shipped code path rather than a re-implementation of it.
    """

    def __init__(self, ex):
        self._ex = ex
        self.events = []

    def submit(self, fn, *a, **kw):
        rec = {'submit': time.monotonic()}
        self.events.append(rec)
        fut = self._ex.submit(fn, *a, **kw)
        fut.add_done_callback(
            lambda f, r=rec: r.setdefault('done', time.monotonic()))
        return fut

    def __getattr__(self, key):
        return getattr(self._ex, key)


def _fast_singlet(ap=3e-3, r=9e-3):
    return {'name': 'fast_singlet', 'aperture_diameter': ap,
            'thicknesses': [3e-3],
            'surfaces': [
                {'radius': r, 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'conic': 0.0, 'aspheric_coeffs': None},
                {'radius': -r, 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'conic': 0.0, 'aspheric_coeffs': None}]}


def _traced(la, n_workers, N):
    import numpy as np
    ap = 3e-3
    dx = 2.2 * ap / N
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / 1.2e-3 ** 2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            E0, prescription=_fast_singlet(ap), wavelength=1.31e-6, dx=dx,
            ray_subsample=2, newton_fit='spline', n_workers=n_workers,
            parallel_amp=False, newton_amp_mask_rel=0.0,
            on_undersample='silent', on_aperture_beam='silent'))


def _summarise(events):
    done = [e for e in events if 'done' in e]
    if not done:
        return None
    t_first_submit = min(e['submit'] for e in done)
    per_chunk = sorted(round(e['done'] - e['submit'], 3) for e in done)
    return {'chunks': len(done),
            'per_chunk_seconds': per_chunk,
            'slowest_chunk_seconds': per_chunk[-1],
            'median_chunk_seconds': per_chunk[len(per_chunk) // 2],
            'first_result_seconds': round(
                min(e['done'] for e in done) - t_first_submit, 3),
            'total_seconds': round(
                max(e['done'] for e in done) - t_first_submit, 3)}


def measure_bootstrap(LT, workers):
    """Pool construction -> every worker has imported what a chunk needs."""
    import multiprocessing as mp
    LT.close_worker_pool()
    t0 = time.monotonic()
    ex = LT._get_persistent_worker_pool(workers)
    t_built = time.monotonic()
    mgr = mp.Manager()
    bar = mgr.Barrier(workers)
    rows = list(ex.map(bootstrap_task, [bar] * workers))
    t_all = time.monotonic()
    ready = sorted(r['ready'] for r in rows)
    out = {'workers': workers,
           'construct_seconds': round(t_built - t0, 3),
           'all_ready_seconds': round(t_all - t0, 3),
           'first_ready_seconds': round(ready[0] - t0, 3),
           'last_ready_seconds': round(ready[-1] - t0, 3),
           'ready_spread_seconds': round(ready[-1] - ready[0], 3),
           'distinct_pids': len({r['pid'] for r in rows})}
    mgr.shutdown()
    LT.close_worker_pool()
    return out


def ping(_):
    """The cheapest possible round-trip: what a BOOTSTRAP BAR would cost.

    The recommendation in the report bounds the time a worker takes to
    answer AT ALL, by submitting one of these ahead of the chunks and
    giving it a timeout.  Its cost on the healthy path is one round-trip of
    this function, which is what ``sentinel`` measures -- cold (the bar's
    whole point) and warm (what every subsequent dispatch pays).
    """
    return os.getpid()


def measure_sentinel(LT, workers, reps=5):
    """One trivial round-trip, cold and warm, through the real pool."""
    LT.close_worker_pool()
    ex = LT._get_persistent_worker_pool(workers)
    t0 = time.monotonic()
    ex.submit(ping, 0).result()
    cold = round(time.monotonic() - t0, 4)
    warm = []
    for _ in range(reps):
        t0 = time.monotonic()
        ex.submit(ping, 0).result()
        warm.append(round(time.monotonic() - t0, 4))
    LT.close_worker_pool()
    return {'workers': workers, 'cold_seconds': cold,
            'warm_seconds': sorted(warm),
            'warm_worst_seconds': max(warm)}


def one_rung(la, LT, workers, N):
    import numpy as np

    real_get = LT._get_persistent_worker_pool
    holder = {}

    def _get(nw):
        prox = _TimingExecutor(real_get(nw))
        holder['prox'] = prox
        return prox

    row = {'N': N, 'workers': workers}
    LT._get_persistent_worker_pool = _get
    try:
        LT.close_worker_pool()
        t0 = time.monotonic()
        got_cold = _traced(la, workers, N)
        row['cold_call_seconds'] = round(time.monotonic() - t0, 3)
        row['cold'] = _summarise(holder['prox'].events)

        t0 = time.monotonic()
        got_warm = _traced(la, workers, N)
        row['warm_call_seconds'] = round(time.monotonic() - t0, 3)
        row['warm'] = _summarise(holder['prox'].events)
    finally:
        LT._get_persistent_worker_pool = real_get

    ref = _traced(la, 1, N)
    row['cold_identical'] = bool(np.array_equal(got_cold, ref))
    row['warm_identical'] = bool(np.array_equal(got_warm, ref))
    row['max_delta'] = float(max(np.abs(got_cold - ref).max(),
                                 np.abs(got_warm - ref).max()))
    row['pool_engaged'] = row['cold'] is not None and row['cold']['chunks'] > 1
    LT.close_worker_pool()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--n', type=int, nargs='+', default=[256, 512, 1024])
    ap.add_argument('--out', default='fu3.json')
    args = ap.parse_args()

    import numpy as np

    import lumenairy as la
    from lumenairy.elements import _lens_traced as LT
    print('lumenairy.__file__ =', la.__file__, flush=True)
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1

    out = {'lumenairy': la.__file__, 'python': sys.version,
           'numpy': np.__version__, 'platform': sys.platform,
           'workers': args.workers, 'rungs': []}
    out['bootstrap'] = measure_bootstrap(LT, args.workers)
    print('BOOTSTRAP', json.dumps(out['bootstrap']), flush=True)
    out['sentinel'] = measure_sentinel(LT, args.workers)
    print('SENTINEL', json.dumps(out['sentinel']), flush=True)
    for N in args.n:
        row = one_rung(la, LT, args.workers, N)
        print(json.dumps(row), flush=True)
        out['rungs'].append(row)

    # The derived bar, computed here so the report cannot quote a number the
    # data does not support.  The rule: a first-result bar has to clear the
    # COLD first result (which includes the spawn bootstrap) with decades to
    # spare, because the cost of a false positive is only wall time.
    worst_cold_first = max(r['cold']['first_result_seconds']
                           for r in out['rungs'] if r['cold'])
    worst_total = max(r['cold']['total_seconds']
                      for r in out['rungs'] if r['cold'])
    worst_chunk = max(r['cold']['slowest_chunk_seconds']
                      for r in out['rungs'] if r['cold'])
    out['derived'] = {
        'worst_cold_first_result_seconds': worst_cold_first,
        'worst_cold_slowest_chunk_seconds': worst_chunk,
        'worst_cold_total_seconds': worst_total,
        'bootstrap_all_ready_seconds': out['bootstrap']['all_ready_seconds'],
        'rule_10x_chunk_plus_bootstrap': round(
            10 * worst_chunk + out['bootstrap']['all_ready_seconds'], 2),
        'rule_100x_first_result_plus_bootstrap': round(
            100 * worst_cold_first + out['bootstrap']['all_ready_seconds'],
            2),
        'sentinel_cold_seconds': out['sentinel']['cold_seconds'],
        'sentinel_warm_worst_seconds': out['sentinel']['warm_worst_seconds']}
    print('DERIVED', json.dumps(out['derived']), flush=True)
    here = os.path.dirname(os.path.abspath(__file__))
    path = args.out if os.path.isabs(args.out) else os.path.join(here,
                                                                 args.out)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print('WROTE', path, flush=True)


if __name__ == '__main__':
    main()

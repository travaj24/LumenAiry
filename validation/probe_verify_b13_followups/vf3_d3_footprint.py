"""VERIFY WP-B13 follow-ups, D3 -- the idle-worker footprint, re-measured.

The claim: a worker that has SERVED a Newton chunk holds 98.7-101.6 MB
(Windows) / 73.2-76.9 MB (WSL); a worker of the same pool that served NONE
holds 52.2 MB / 39.2 MB; a 16-wide kept pool of served workers is therefore
~1.6 GB, i.e. ~6 % of the ~1.7 GB per ACTIVE worker the clamp models.

This probe re-measures it with a DIFFERENT labelling mechanism.  The branch's
probe forces every worker to take exactly one labelling task with a
``multiprocessing.Manager`` Barrier.  Here the rendezvous is a listening
SOCKET the parent owns: each labelling task connects, reports its pid and its
``_WORKER_PAYLOADS`` keys, and blocks on ``recv`` until the parent has
accepted one connection per worker.  No Manager process sits inside the
measurement, and "every worker was labelled" is the parent's own accept
count rather than a barrier's internal state.

Resident sets are read PARENT-side against the executor's own ``_processes``
(so the reading does not perturb the worker), at three moments: BEFORE the
dispatch, AFTER the dispatch, and AFTER labelling.  The figure reported is
the AFTER-DISPATCH one, because labelling itself touches every worker.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=<tree> python validation/probe_verify_b13_followups/\
vf3_d3_footprint.py --pool 12 --dispatch 4 --n 256 1024 --out vf3_win.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings


def _label_task(addr):
    """Report who I am and whether I have ever served a Newton chunk.

    Blocks on the socket until the parent has one connection per worker, so
    no worker can take two labelling tasks while another takes none.
    """
    import socket

    from lumenairy.elements import _lens_traced as LT
    keys = sorted(getattr(LT, '_WORKER_PAYLOADS', {}).keys())
    sk = socket.create_connection(addr, timeout=300.0)
    try:
        sk.sendall(b'r')
        sk.recv(1)
    finally:
        sk.close()
    return {'pid': os.getpid(), 'payload_keys': [str(k) for k in keys],
            'served': bool(keys)}


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


def _rss(ex):
    """Resident set of every live worker, read from the PARENT."""
    import psutil
    out = {}
    for pid in list(getattr(ex, '_processes', {}) or {}):
        try:
            out[int(pid)] = psutil.Process(int(pid)).memory_info().rss
        except Exception:                          # noqa: BLE001
            pass
    return out


def label_workers(ex, width):
    """One labelling task per worker, rendezvoused on a socket."""
    import socket
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', 0))
    srv.listen(width + 4)
    srv.settimeout(300.0)
    addr = srv.getsockname()
    futs = [ex.submit(_label_task, addr) for _ in range(width)]
    conns = []
    try:
        for _ in range(width):
            c, _a = srv.accept()
            c.recv(1)
            conns.append(c)
        for c in conns:
            c.sendall(b'g')
    finally:
        for c in conns:
            try:
                c.close()
            except OSError:
                pass
        srv.close()
    rows = [f.result(300.0) for f in futs]
    return rows, len(conns)


def one_rung(la, LT, pool_width, dispatch_workers, N):
    import numpy as np
    LT.close_worker_pool()
    serial = _traced(la, 1, N)

    # Build the WIDE pool first, then dispatch NARROW: the rebuild rule
    # never shrinks, so the surplus workers are exactly the never-served
    # state the docstring is about.
    ex = LT._get_persistent_worker_pool(pool_width)
    time.sleep(0.5)
    rss_before = _rss(ex)
    got = _traced(la, dispatch_workers, N)
    time.sleep(0.5)
    ex2 = LT._PERSISTENT_POOL
    same_pool = ex2 is ex
    rss_after = _rss(ex2)
    rows, accepted = label_workers(ex2, pool_width)
    rss_labelled = _rss(ex2)

    served = {r['pid'] for r in rows if r['served']}
    never = {r['pid'] for r in rows if not r['served']}
    live_after_dispatch = set(rss_after)
    live_after_label = set(rss_labelled)
    # Coverage is decided against the set of workers that EXIST once every
    # worker has been given a task -- which is not the same set as the one
    # that existed after the dispatch, because CPython's
    # ProcessPoolExecutor spawns LAZILY (``_adjust_process_count`` runs in
    # ``submit``).  That difference is itself a finding; see
    # ``surplus_workers_before_any_submit``.
    coverage = ((served | never) == live_after_label
                and len(rows) == pool_width
                and accepted == pool_width)

    def _mb(pids, table):
        vals = [table[p] / 2 ** 20 for p in pids if p in table]
        return ({'n': len(vals), 'mean_MB': round(sum(vals) / len(vals), 2),
                 'max_MB': round(max(vals), 2), 'min_MB': round(min(vals), 2)}
                if vals else {'n': 0})

    out = {'N': N, 'pool_width': pool_width,
           'dispatch_workers': dispatch_workers,
           'pool_reused_for_the_dispatch': bool(same_pool),
           # THE LAZY-SPAWN FINDING: how many worker PROCESSES exist at each
           # moment.  A pool constructed at width 12 holds zero processes
           # until something is submitted to it.
           'processes_after_construct': len(rss_before),
           'processes_after_dispatch': len(live_after_dispatch),
           'processes_after_labelling': len(live_after_label),
           'surplus_workers_before_any_submit': len(rss_before),
           'live_workers': len(live_after_dispatch),
           'labelled_workers': len(rows),
           'accepted_connections': accepted,
           'label_coverage_complete': bool(coverage),
           'served_pids': len(served), 'never_served_pids': len(never),
           'served_after_dispatch': _mb(served, rss_after),
           'never_served_after_dispatch': _mb(never, rss_after),
           'served_after_labelling': _mb(served, rss_labelled),
           'never_served_after_labelling': _mb(never, rss_labelled),
           'all_before_dispatch': _mb(set(rss_before), rss_before),
           'identical_to_serial': bool(np.array_equal(got, serial)),
           'max_abs_delta': float(np.nanmax(np.abs(
               np.asarray(got) - np.asarray(serial)))),
           'abandoned_pools_len': len(LT._ABANDONED_POOLS),
           'shutdown_timeouts': LT._POOL_SHUTDOWN_TIMEOUTS}
    LT.close_worker_pool()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=12)
    ap.add_argument('--dispatch', type=int, default=4)
    ap.add_argument('--n', type=int, nargs='+', default=[256, 1024])
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    from lumenairy.elements import _lens_traced as LT
    la = lumenairy
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1

    out = {'python': sys.version, 'platform': sys.platform,
           'lumenairy_file': lumenairy.__file__,
           'rendezvous': 'localhost socket (no Manager)',
           'pool_width': args.pool, 'dispatch_workers': args.dispatch}
    try:
        import psutil
        out['load_at_start'] = {
            'cpu_percent_1s': psutil.cpu_percent(interval=1.0),
            'mem_available_GB': round(
                psutil.virtual_memory().available / 2 ** 30, 2)}
    except Exception as exc:                       # noqa: BLE001
        out['load_at_start'] = f'unavailable: {exc}'

    rungs = []
    for N in args.n:
        r = one_rung(la, LT, args.pool, args.dispatch, N)
        rungs.append(r)
        print('rung', json.dumps(r), flush=True)
    out['rungs'] = rungs

    served_max = [r['served_after_labelling'].get('max_MB')
                  for r in rungs if r['served_after_labelling'].get('n')]
    never_mean = [r['never_served_after_labelling'].get('mean_MB')
                  for r in rungs if r['never_served_after_labelling'].get('n')]
    out['summary'] = {
        'worst_served_MB': max(served_max) if served_max else None,
        'never_served_mean_MB_range': (
            [min(never_mean), max(never_mean)] if never_mean else None),
        'sixteen_wide_served_pool_GB': (
            round(16 * max(served_max) / 1024, 2) if served_max else None),
        'fraction_of_a_1p7GB_active_worker_percent': (
            round(100 * max(served_max) / 1740.8, 1) if served_max else None),
        'every_rung_label_coverage_complete': all(
            r['label_coverage_complete'] for r in rungs),
        'every_rung_identical_to_serial': all(
            r['identical_to_serial'] for r in rungs),
        'census_empty_on_every_rung': all(
            r['abandoned_pools_len'] == 0 for r in rungs),
        'a_wider_pool_holds_no_processes_until_submitted_to': all(
            r['processes_after_construct'] == 0 for r in rungs),
        'processes_after_dispatch_equals_dispatch_width': all(
            r['processes_after_dispatch'] == r['dispatch_workers']
            for r in rungs),
    }
    with open(args.out, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(out, fh, indent=1)
    print('SUMMARY', json.dumps(out['summary'], indent=1))
    print('wrote', args.out)


if __name__ == '__main__':
    main()

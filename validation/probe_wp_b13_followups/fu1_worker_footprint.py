"""FU1 -- what an IDLE kept-alive Newton worker actually costs (D3).

VERIFY-WP-B13 defect D3: the footprint in ``_get_persistent_worker_pool``'s
docstring ("33.0 MB mean / 48.8 MB peak") was measured on workers warmed with
``ex.map(abs, ...)``.  The state the CEILING rule actually leaves behind is a
worker that has served a real Newton chunk and is now idle, which is a very
different resident set.

Three states, all of them reachable and all of them measured PARENT-SIDE with
``psutil`` against the executor's own ``_processes``, so the reading does not
itself perturb the worker:

  ``spawned_trivial``   every worker has run the pool initializer and exactly
                        one trivial task.  This is the WP's own state.
  ``newton_served``     the worker has served at least one Newton chunk, i.e.
                        it holds an entry in ``_WORKER_PAYLOADS``.
  ``newton_unserved``   a worker in the SAME pool that served no chunk.

Every worker is forced to take exactly one task by a Manager ``Barrier`` of
the pool's own width, so "which workers exist" and "which workers were
labelled" are decisions, not samples.

Run (both builds), with the three thread caps on the command line:

    PYTHONPATH=<tree> python validation/probe_wp_b13_followups/
    fu1_worker_footprint.py --workers 8 --n 512 1024 --out fu1_win.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings


def _self_rss():
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:                             # noqa: BLE001
        return None


def barrier_touch(barrier):
    """Wait at the barrier, then report this worker's identity and state.

    Waiting at a barrier of the pool's own width guarantees that every worker
    takes exactly one of these -- a plain ``map`` can be served entirely by
    one fast worker.
    """
    from lumenairy.elements import _lens_traced as _LT
    try:
        barrier.wait(120.0)
    except Exception:                             # noqa: BLE001 -- reported
        pass
    return {'pid': os.getpid(),
            'payloads': sorted(_LT._WORKER_PAYLOADS),
            'self_rss': _self_rss()}


def _child_rss(ex):
    """Resident set of every live worker of ``ex``, read from the PARENT."""
    import psutil
    out = {}
    for pid in list(getattr(ex, '_processes', {}) or {}):
        try:
            out[int(pid)] = int(psutil.Process(int(pid)).memory_info().rss)
        except Exception:                         # noqa: BLE001
            out[int(pid)] = None
    return out


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


def _stats(values):
    vals = [v / 1e6 for v in values if v]
    if not vals:
        return None
    return {'n': len(vals),
            'mean_mb': round(sum(vals) / len(vals), 2),
            'min_mb': round(min(vals), 2),
            'max_mb': round(max(vals), 2),
            'values_mb': [round(v, 2) for v in sorted(vals)]}


def one_rung(la, LT, workers, N, dispatch_workers=None):
    import multiprocessing as mp

    import numpy as np
    import psutil

    LT.close_worker_pool()
    dispatch_workers = int(dispatch_workers or workers)
    row = {'pool_width_requested': workers, 'N': N,
           'dispatch_workers': dispatch_workers}

    # --- state 1: spawned, one trivial task each -------------------------
    ex = LT._get_persistent_worker_pool(workers)
    mgr = mp.Manager()
    bar = mgr.Barrier(workers)
    t0 = time.monotonic()
    touched = list(ex.map(barrier_touch, [bar] * workers))
    row['spawn_and_touch_seconds'] = round(time.monotonic() - t0, 3)
    row['spawned_pids'] = sorted(t['pid'] for t in touched)
    time.sleep(0.5)
    rss_trivial = _child_rss(ex)
    row['spawned_trivial'] = _stats(rss_trivial.values())
    row['parent_rss_mb_after_spawn'] = round(
        psutil.Process().memory_info().rss / 1e6, 1)

    # --- the dispatch ----------------------------------------------------
    t0 = time.monotonic()
    got = _traced(la, dispatch_workers, N)
    row['traced_seconds'] = round(time.monotonic() - t0, 3)
    row['pool_is_the_same_object'] = (LT._PERSISTENT_POOL is ex)
    row['pool_width'] = LT._PERSISTENT_POOL_NWORKERS
    ex = LT._PERSISTENT_POOL
    time.sleep(0.5)
    rss_after = _child_rss(ex)
    row['parent_rss_mb_after_traced'] = round(
        psutil.Process().memory_info().rss / 1e6, 1)

    # --- label served / unserved, AFTER the resident sets were read ------
    width = int(LT._PERSISTENT_POOL_NWORKERS or workers)
    bar2 = mgr.Barrier(width)
    labels = list(ex.map(barrier_touch, [bar2] * width))
    served = {r['pid'] for r in labels if r['payloads']}
    row['labelled_pids'] = sorted(r['pid'] for r in labels)
    row['served_pids'] = sorted(served)
    row['label_coverage_complete'] = bool(
        set(row['labelled_pids']) >= set(rss_after))
    row['newton_served'] = _stats(
        [v for p, v in rss_after.items() if p in served])
    row['newton_unserved'] = _stats(
        [v for p, v in rss_after.items() if p not in served])
    row['newton_all'] = _stats(rss_after.values())

    # the field itself, so the rung is not measuring a call that failed
    ref = _traced(la, 1, N)
    row['identical_to_serial'] = bool(np.array_equal(got, ref))
    row['max_delta'] = float(np.abs(got - ref).max())
    row['shutdown_timeouts'] = int(LT._POOL_SHUTDOWN_TIMEOUTS)
    row['abandoned_pools_len'] = len(LT._ABANDONED_POOLS)
    mgr.shutdown()
    LT.close_worker_pool()
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8,
                    help='POOL width -- what the ceiling rule keeps alive')
    ap.add_argument('--dispatch-workers', type=int, default=None,
                    help='what the traced call asks for; smaller than '
                         '--workers leaves idle workers that served nothing')
    ap.add_argument('--n', type=int, nargs='+', default=[512, 1024])
    ap.add_argument('--out', default='fu1.json')
    args = ap.parse_args()

    import numpy as np

    import lumenairy as la
    from lumenairy.elements import _lens_traced as LT
    print('lumenairy.__file__ =', la.__file__, flush=True)
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1

    out = {'lumenairy': la.__file__,
           'python': sys.version,
           'numpy': np.__version__,
           'platform': sys.platform,
           'pool_width_requested': args.workers,
           'dispatch_workers': args.dispatch_workers or args.workers,
           'rungs': []}
    for N in args.n:
        row = one_rung(la, LT, args.workers, N,
                       args.dispatch_workers)
        print(json.dumps(row), flush=True)
        out['rungs'].append(row)
    here = os.path.dirname(os.path.abspath(__file__))
    path = args.out if os.path.isabs(args.out) else os.path.join(here,
                                                                 args.out)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print('WROTE', path, flush=True)


if __name__ == '__main__':
    main()

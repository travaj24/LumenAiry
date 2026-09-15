"""VP4 -- does a MOVING memory clamp still rebuild the pool on every dispatch?

Claim (2) of the WP report, re-measured with the clamp's answer ENGINEERED
instead of hoped for.  ``_newton_resolve_workers`` reads
``min(psutil.virtual_memory().available, lumenairy.memory.get_ram_budget())``,
so ``lumenairy.set_max_ram`` moves exactly the quantity the report says a busy
box moves, and it moves it to a value this probe chooses.  The arithmetic the
clamp does is

    budget = 0.5 * free - 2 GB ;  allowed = budget // per_worker_bytes

with ``per_worker_bytes = 1.75 GB + 268 B/chunk-point + 850 B/fit-point``, so
the ladder below is derived, not guessed, and the probe ASSERTS the answer it
engineered before it counts anything.

MODES

sequence   four dispatches whose clamp answers are driven to 6, 8, 4, 8 -- the
           report's own measured sequence -- counting ProcessPoolExecutor
           CONSTRUCTIONS, shutdown calls and distinct spawned worker pids.
           Pre-fix: one construction per change.  Post-fix: one per INCREASE.
memalloc   the same clamp read, with the memory moved by actually allocating
           and touching a large array rather than by the budget knob, to show
           the live ``psutil`` arm moves the answer the same way.
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

T0 = 0.0
_WL = 1.31e-6


def emit(event, **kw):
    kw['event'] = event
    kw['t'] = round(time.monotonic() - T0, 4)
    sys.stdout.write(json.dumps(kw, sort_keys=True) + "\n")
    sys.stdout.flush()


def _singlet(ap, r):
    return {'name': 'vp4', 'aperture_diameter': ap, 'thicknesses': [3e-3],
            'surfaces': [
                {'radius': r, 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'conic': 0.0, 'aspheric_coeffs': None},
                {'radius': -r, 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'conic': 0.0, 'aspheric_coeffs': None}]}


def _traced(la, n_workers, N, rs, r):
    ap = 3e-3
    dx = 2.2 * ap / N
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = la.apply_real_lens_traced(
            E0, prescription=_singlet(ap, r), wavelength=_WL, dx=dx,
            ray_subsample=rs, newton_fit='spline', n_workers=n_workers,
            parallel_amp=False, newton_amp_mask_rel=0.0,
            on_undersample='silent', on_aperture_beam='silent',
            on_pool_memory='silent')
    return np.asarray(out)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class _PidSampler(threading.Thread):
    """Every distinct child pid this process ever had."""

    def __init__(self):
        super().__init__(name='vp4-pids', daemon=True)
        self.pids = set()
        self.stop = threading.Event()

    def run(self):
        try:
            import psutil
        except ImportError:
            return
        me = psutil.Process()
        while not self.stop.is_set():
            try:
                for c in me.children(recursive=True):
                    self.pids.add(c.pid)
            except Exception:
                pass
            self.stop.wait(0.03)


def _install_counters(LT):
    """Count executor constructions and shutdown calls, keeping real pools."""
    import concurrent.futures as cf
    real = cf.ProcessPoolExecutor
    counts = {'constructed': 0, 'shutdown_calls': 0, 'shutdown_waits': []}

    class Counting(real):
        def __init__(self, *a, **kw):
            counts['constructed'] += 1
            super().__init__(*a, **kw)

        def shutdown(self, wait=True, *, cancel_futures=False):
            counts['shutdown_calls'] += 1
            counts['shutdown_waits'].append(
                {'wait': bool(wait), 'cancel_futures': bool(cancel_futures)})
            return super().shutdown(wait=wait, cancel_futures=cancel_futures)

    cf.ProcessPoolExecutor = Counting
    resolved = []
    real_resolve = LT._newton_resolve_workers

    def spy(requested, n_total, fit_points, **kw):
        ans = real_resolve(requested, n_total, fit_points, **kw)
        resolved.append({'requested': int(requested),
                         'n_total': int(n_total),
                         'fit_points': int(fit_points), 'answer': int(ans)})
        return ans

    LT._newton_resolve_workers = spy
    return counts, resolved


def _budget_for(target_workers, per_worker_b):
    """Invert the clamp: the free-byte figure that yields ``target_workers``."""
    budget_b = (target_workers + 0.5) * per_worker_b
    return (budget_b + 2.0e9) / 0.5


def mode_sequence(la, LT, a):
    from lumenairy.memory import get_ram_budget
    N, rs = a.N, a.rs
    n_total = (N // rs) * (N // rs)
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1
    emit('gate', unguarded_main=LT._spawn_reexecuted_main_script())

    LT.close_worker_pool()
    ref = _traced(la, 1, N, rs, 9e-3)
    emit('reference', sha=_sha(ref))

    # Learn the REAL fit-grid size this call uses, so the ladder is priced
    # with the number the clamp will actually see rather than a guess.
    counts, resolved = _install_counters(LT)
    la.set_max_ram(None)
    _traced(la, a.request, N, rs, 9e-3)
    fit_points = max(r['fit_points'] for r in resolved)
    n_total = max(r['n_total'] for r in resolved)
    emit('learned', fit_points=fit_points, n_total=n_total,
         warmup_resolves=list(resolved))
    counts['constructed'] = 0
    counts['shutdown_calls'] = 0
    counts['shutdown_waits'] = []
    del resolved[:]
    sampler = _PidSampler()
    sampler.start()
    LT.close_worker_pool()

    ladder = [int(x) for x in a.ladder.split(',')]
    rows = []
    for k, want in enumerate(ladder):
        pw = LT._newton_worker_bytes(float(n_total) / float(want), fit_points)
        free_needed = _budget_for(want, pw)
        la.set_max_ram(free_needed / 1e9)
        got_clamp = LT._newton_resolve_workers(
            a.request, n_total, fit_points, min_pool_points=1,
            on_pool_memory='silent')
        before = counts['constructed']
        t = time.monotonic()
        field = _traced(la, a.request, N, rs, 9e-3 if k % 2 == 0 else 12e-3)
        rows.append({'step': k, 'engineered_target': want,
                     'clamp_answer': got_clamp,
                     'ram_budget_gb': round(free_needed / 1e9, 2),
                     'live_budget_gb': round(get_ram_budget() / 1e9, 2),
                     'constructed_before': before,
                     'constructed_after': counts['constructed'],
                     'built_here': counts['constructed'] - before,
                     'pool_workers': LT._PERSISTENT_POOL_NWORKERS,
                     'seconds': round(time.monotonic() - t, 3),
                     'sha': _sha(field),
                     'identical_to_serial_ref': bool(
                         np.array_equal(field, ref)) if k % 2 == 0 else None})
        emit('step', **rows[-1])
    sampler.stop.set()
    sampler.join(timeout=5)
    out = {'mode': 'sequence', 'ladder': ladder, 'rows': rows,
           'constructions': counts['constructed'],
           'shutdown_calls': counts['shutdown_calls'],
           'shutdown_waits': counts['shutdown_waits'],
           'distinct_worker_pids': len(sampler.pids),
           'resolve_calls': resolved,
           'ref_sha': _sha(ref)}
    return out


def mode_memalloc(la, LT, a):
    n_total = (a.N // a.rs) * (a.N // a.rs)
    reads = []

    def read(tag):
        import psutil
        avail = psutil.virtual_memory().available
        ans = LT._newton_resolve_workers(a.request, n_total, a.fit_points,
                                         min_pool_points=1,
                                         on_pool_memory='silent')
        reads.append({'tag': tag, 'available_gb': round(avail / 1e9, 2),
                      'clamp_answer': ans})
        emit('read', **reads[-1])

    read('before')
    gb = a.gb
    blocks = []
    try:
        for i in range(int(gb)):
            b = np.empty(1 << 30, dtype=np.uint8)
            b[::4096] = 1                      # touch, so it is committed
            blocks.append(b)
            if (i + 1) % 8 == 0:
                read(f'alloc_{i + 1}gb')
        read(f'alloc_{int(gb)}gb')
    finally:
        del blocks
        import gc
        gc.collect()
    read('after_free')
    return {'mode': 'memalloc', 'reads': reads, 'requested_gb': gb}


def main():
    global T0
    T0 = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('sequence', 'memalloc'),
                    default='sequence')
    ap.add_argument('--ladder', default='6,8,4,8')
    ap.add_argument('--request', type=int, default=8)
    ap.add_argument('--N', type=int, default=256)
    ap.add_argument('--rs', type=int, default=2)
    ap.add_argument('--fit-points', type=int, default=250000)
    ap.add_argument('--gb', type=float, default=40.0)
    ap.add_argument('--dump-after', type=float, default=600.0)
    ap.add_argument('--json-out', default=None)
    a = ap.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(a.dump_after, exit=True)

    import lumenairy as la
    from lumenairy.elements import _lens_traced as LT
    emit('import', lumenairy=la.__file__, python=sys.version.split()[0],
         pid=os.getpid(), mode=a.mode)
    out = (mode_sequence(la, LT, a) if a.mode == 'sequence'
           else mode_memalloc(la, LT, a))
    out['python'] = sys.version.split()[0]
    out['lumenairy'] = la.__file__
    emit('summary', **out)
    if a.json_out:
        with open(a.json_out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    try:
        LT.close_worker_pool()
    except Exception:
        pass
    faulthandler.cancel_dump_traceback_later()
    return 0


if __name__ == '__main__':
    sys.exit(main())

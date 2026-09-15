"""P7 -- does a BROKEN Newton pool reach the serial fallback?

Two ways of breaking it, both driving the shipped ``apply_real_lens_traced``
path and both printing a JSON phase line per step, so a wedge is a MISSING
line rather than a silent hang.  ``--dump-after`` arms
``faulthandler.dump_traceback_later(..., exit=True)``, so the probe aborts
with every thread's stack instead of joining the hang it is studying.

MODES
-----
stub    The deterministic one.  The module's persistent pool is replaced by a
        stub executor that (a) fails every future with
        ``BrokenProcessPool`` -- what a worker dying mid-flight looks like to
        the parent -- and (b) BLOCKS FOREVER in ``shutdown``, which is what
        CPython's ``_ExecutorManagerThread._terminate_broken`` does when it
        joins a queue-feeder thread stuck in ``connection._send_bytes`` or a
        worker process that will not die.  No child processes are involved,
        so the wedge is a property of the library's code path alone.  A
        library that joins the pool on the broken-pool path hangs here; a
        library that retires it reaches the serial line and returns.

kill    The realistic one.  A REAL spawn pool is built through the library's
        own getter and every worker kills itself with ``os._exit`` the moment
        it is handed a chunk, so the executor really does break.  The call
        must still return, and must return the SERIAL answer byte for byte.

Both modes compare the returned field against a serial reference computed
with ``n_workers=1``, because the fallback's whole claim is that it moves no
number.
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
_REAL_CHUNK = None

_WL = 1.31e-6
_W0 = 1.0e-3
_N, _RS = 1024, 2          # 262 144 Newton points: past the cold pool bar


def emit(event, **kw):
    kw['event'] = event
    kw['t'] = round(time.monotonic() - T0, 4)
    sys.stdout.write(json.dumps(kw) + '\n')
    sys.stdout.flush()


def killer_chunk(args):
    """Stand-in for ``_newton_invert_chunk`` that kills its worker.

    Defined at module scope so the spawn worker can resolve it: the parent
    pickles it by name, and the worker re-imports this guarded module as
    ``__mp_main__``.  ``os._exit`` leaves no result item and no traceback --
    the executor sees only a closed sentinel, which is exactly the
    ``BrokenProcessPool`` shape the release-gate dumps caught.
    """
    os._exit(7)


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


class _WedgedBrokenPool:
    """Broken on submit, wedged on shutdown -- the captured failure state."""

    def __init__(self):
        self._broken = ('A child process terminated abruptly, the process '
                        'pool is not usable anymore')
        self.shutdown_calls = []
        self.entered = threading.Event()
        self._never = threading.Event()

    def submit(self, fn, *a, **kw):
        from concurrent.futures import Future
        from concurrent.futures.process import BrokenProcessPool
        f = Future()
        f.set_exception(BrokenProcessPool(
            'A process in the process pool was terminated abruptly while '
            'the future was running or pending.'))
        return f

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.shutdown_calls.append({'wait': bool(wait),
                                    'cancel_futures': bool(cancel_futures)})
        self.entered.set()
        if wait:
            # CPython's _terminate_broken joins the feeder thread and the
            # worker processes with no timeout, under the same lock shutdown
            # acquires first.  This is that state, made deterministic.
            self._never.wait()


def scenario_stub(args, LT):
    emit('serial_reference_enter')
    LT.close_worker_pool()
    ref = _run(1)
    emit('serial_reference_exit', sha=_sha(ref))

    stub = _WedgedBrokenPool()
    LT.close_worker_pool()
    LT._PERSISTENT_POOL = stub
    LT._PERSISTENT_POOL_NWORKERS = args.workers
    orig_get = LT._get_persistent_worker_pool
    LT._get_persistent_worker_pool = lambda nw: stub
    emit('stub_installed', workers=args.workers)
    t = time.monotonic()
    try:
        got = _run(args.workers)
    finally:
        LT._get_persistent_worker_pool = orig_get
        LT._PERSISTENT_POOL = None
        LT._PERSISTENT_POOL_NWORKERS = None
    out = {'fallback_seconds': round(time.monotonic() - t, 3),
           'sha': _sha(got), 'ref_sha': _sha(ref),
           'identical': bool(np.array_equal(got, ref)),
           'shutdown_calls': stub.shutdown_calls}
    emit('fallback_returned', **out)
    return out


def scenario_kill(args, LT):
    global _REAL_CHUNK
    emit('serial_reference_enter')
    LT.close_worker_pool()
    ref = _run(1)
    emit('serial_reference_exit', sha=_sha(ref))

    LT.close_worker_pool()
    _REAL_CHUNK = LT._newton_invert_chunk
    LT._newton_invert_chunk = killer_chunk
    emit('killer_installed', workers=args.workers)
    t = time.monotonic()
    try:
        got = _run(args.workers)
    finally:
        LT._newton_invert_chunk = _REAL_CHUNK
    out = {'fallback_seconds': round(time.monotonic() - t, 3),
           'sha': _sha(got), 'ref_sha': _sha(ref),
           'identical': bool(np.array_equal(got, ref)),
           'shutdown_timeouts': LT._POOL_SHUTDOWN_TIMEOUTS
           if hasattr(LT, '_POOL_SHUTDOWN_TIMEOUTS') else None}
    emit('fallback_returned', **out)
    return out


def main():
    global T0
    T0 = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('stub', 'kill'), default='stub')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--dump-after', type=float, default=300.0)
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    faulthandler.enable()
    faulthandler.dump_traceback_later(args.dump_after, exit=True)

    import lumenairy
    from lumenairy.elements import _lens_traced as LT
    emit('import', lumenairy=lumenairy.__file__,
         version=lumenairy.__version__, python=sys.version.split()[0],
         pid=os.getpid())

    out = (scenario_stub(args, LT) if args.mode == 'stub'
           else scenario_kill(args, LT))
    out['mode'] = args.mode
    emit('summary', **out)
    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    LT.close_worker_pool()
    faulthandler.cancel_dump_traceback_later()
    return 0 if out.get('identical') else 4


if __name__ == '__main__':
    sys.exit(main())

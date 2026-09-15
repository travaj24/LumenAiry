"""VP2 -- five independent ways of breaking the Newton pool.

Each mode drives the SHIPPED ``apply_real_lens_traced`` path and asks one
question: does the call RETURN, and does it return the serial answer byte for
byte?  Every mode emits a JSON line per phase, so a wedge is a MISSING line
rather than a silent hang, and arms ``faulthandler.dump_traceback_later`` so
the probe aborts with every thread's stack instead of joining the hang.

MODES (none of them shares code with the WP's own probe_p7)

exit        the worker calls ``os._exit`` on its first chunk.
kill        the PARENT kills the live worker processes (TerminateProcess /
            SIGKILL) while their chunks are in flight.
unpickle    the work item raises WHILE THE WORKER UNPICKLES it (an object
            whose ``__reduce__`` names a callable that raises), so the worker
            dies inside ``call_queue.get()`` rather than inside the task.
feederkill  the workers are made slow so the executor's QueueFeederThread is
            confirmed BLOCKED inside ``connection._send_bytes`` (the release
            gate's captured state), and only then are all workers killed.
sigign      POSIX only: the workers ignore SIGTERM, so CPython's
            ``_terminate_broken`` -> ``p.terminate()`` -> ``p.join()`` is a
            genuinely UNBOUNDED join with no stub executor anywhere.
control     nothing is broken; the pooled answer must equal the serial one.

The reference is always a fresh ``n_workers=1`` call in the same process.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import os
import signal
import sys
import threading
import time
import warnings

import numpy as np

# Literal constants only at top level: a spawn worker re-executes this
# module body, and the library REFUSES the pool for a script whose top
# level does real work (_script_has_main_guard).  T0 is set in main().
T0 = 0.0
_WL = 1.31e-6


def emit(event, **kw):
    kw['event'] = event
    kw['t'] = round(time.monotonic() - T0, 4)
    sys.stdout.write(json.dumps(kw, sort_keys=True) + "\n")
    sys.stdout.flush()


# ---- worker-side payloads (module scope so spawn can resolve them) --------

def killer_chunk(args):
    """os._exit: no result item, no traceback -- a dead worker."""
    os._exit(7)


def slow_chunk(args):
    """Hold the worker so the parent's feeder thread backs up."""
    time.sleep(600.0)
    return args


def boom_on_unpickle():
    raise RuntimeError('VP2: raised while unpickling the work item')


class _UnpicklableChunk:
    """Pickles fine in the parent; RAISES in the worker's call_queue.get."""

    def __reduce__(self):
        return (boom_on_unpickle, ())


def slow_init():
    """Pool initializer that never finishes: the workers never come up.

    This is the SECOND exposure the WP report files as open -- ``as_completed``
    in ``_invert_newton_parallel`` has no timeout -- engineered so it does not
    depend on a WSL mount being slow.
    """
    time.sleep(3600.0)


def sigign_chunk(args):
    """Ignore SIGTERM, then hold the worker: p.terminate() cannot end it."""
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except (ValueError, OSError, AttributeError):
        pass
    time.sleep(600.0)
    return args


# ---- the field ------------------------------------------------------------

def _singlet(ap=3e-3, r=9e-3):
    return {'name': 'vp2_singlet', 'aperture_diameter': ap,
            'thicknesses': [3e-3],
            'surfaces': [
                {'radius': r, 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'conic': 0.0, 'aspheric_coeffs': None},
                {'radius': -r, 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'conic': 0.0, 'aspheric_coeffs': None}]}


def _traced(la, n_workers, N, rs, fit):
    ap = 3e-3
    dx = 2.2 * ap / N
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E0 = np.exp(-(X ** 2 + Y ** 2) / (1.2e-3) ** 2).astype(np.complex128)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = la.apply_real_lens_traced(
            E0, prescription=_singlet(ap), wavelength=_WL, dx=dx,
            ray_subsample=rs, newton_fit=fit, n_workers=n_workers,
            parallel_amp=False, newton_amp_mask_rel=0.0,
            on_undersample='silent', on_aperture_beam='silent')
    return np.asarray(out)


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---- parent-side killers --------------------------------------------------

def _pool_procs(LT):
    ex = LT._PERSISTENT_POOL
    procs = getattr(ex, '_processes', None) if ex is not None else None
    return list(procs.values()) if procs else []


def _feeder_blocked_in_send():
    """Is any QueueFeederThread inside connection._send_bytes right now?"""
    frames = sys._current_frames()
    names = {t.ident: t.name for t in threading.enumerate()}
    for ident, frame in frames.items():
        if 'QueueFeederThread' not in str(names.get(ident, '')):
            continue
        f = frame
        while f is not None:
            if f.f_code.co_name in ('_send_bytes', '_send'):
                return True
            f = f.f_back
    return False


def killer_thread(LT, mode, out, stop):
    """Wait for the pool to be busy, then kill it the way ``mode`` asks."""
    t_end = time.monotonic() + 240.0
    waited_feeder = False
    while time.monotonic() < t_end and not stop.is_set():
        procs = _pool_procs(LT)
        if len(procs) >= 1 and all(p.pid for p in procs):
            if mode == 'feederkill':
                if _feeder_blocked_in_send():
                    waited_feeder = True
                else:
                    time.sleep(0.05)
                    continue
            else:
                # give the dispatch a moment to actually hand chunks over
                time.sleep(1.5)
            alive = [p for p in procs if p.is_alive()]
            if mode == 'sigign':
                # Kill exactly ONE worker: the pool breaks, and CPython's
                # _terminate_broken then p.terminate()s (SIGTERM) the SURVIVORS
                # -- which ignore it -- and p.join()s them without a bound.
                alive = alive[:1]
            out['killed_pids'] = [p.pid for p in alive]
            out['feeder_was_blocked_in_send_bytes'] = waited_feeder
            for p in alive:
                try:
                    p.kill()
                except (OSError, AttributeError, ValueError):
                    pass
            emit('workers_killed', n=len(alive), feeder_blocked=waited_feeder)
            return
        time.sleep(0.05)
    out['killed_pids'] = []
    emit('killer_gave_up')


def main():
    global T0
    T0 = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True,
                    choices=('exit', 'kill', 'unpickle', 'feederkill',
                             'sigign', 'slowboot', 'control'))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--N', type=int, default=256)
    ap.add_argument('--rs', type=int, default=2)
    ap.add_argument('--fit', default='spline')
    ap.add_argument('--dump-after', type=float, default=240.0)
    ap.add_argument('--json-out', default=None)
    a = ap.parse_args()

    faulthandler.enable()
    faulthandler.dump_traceback_later(a.dump_after, exit=True)

    import lumenairy as la
    from lumenairy.elements import _lens_traced as LT
    emit('import', lumenairy=la.__file__, version=la.__version__,
         python=sys.version.split()[0], pid=os.getpid(), mode=a.mode)

    # Engineer the precondition: this field must go to the pool, and SAY so
    # -- a probe whose pool silently never engaged measures nothing
    # (docs/TESTING_STANDARDS.md rule 4, shape S3).
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1
    emit('gate', unguarded_main=LT._spawn_reexecuted_main_script(),
         resolved_workers=LT._newton_resolve_workers(
             a.workers, a.N * a.N // (a.rs * a.rs), 1000,
             min_pool_points=1, on_pool_memory='silent'))
    _getter = LT._get_persistent_worker_pool
    _got = []

    def _spy(nw):
        _got.append(int(nw))
        return _getter(nw)

    LT._get_persistent_worker_pool = _spy

    LT.close_worker_pool()
    emit('reference_enter')
    ref = _traced(la, 1, a.N, a.rs, a.fit)
    emit('reference_exit', sha=_sha(ref))

    out = {'mode': a.mode, 'workers': a.workers, 'N': a.N, 'rs': a.rs,
           'fit': a.fit, 'python': sys.version.split()[0],
           'lumenairy': la.__file__, 'ref_sha': _sha(ref)}

    LT.close_worker_pool()
    stop = threading.Event()
    if a.mode == 'exit':
        LT._newton_invert_chunk = killer_chunk
    elif a.mode == 'unpickle':
        LT._newton_invert_chunk = _UnpicklableChunk()
    elif a.mode == 'slowboot':
        LT._newton_pool_init = slow_init
    elif a.mode in ('kill', 'feederkill', 'sigign'):
        if a.mode == 'feederkill':
            LT._newton_invert_chunk = slow_chunk
        elif a.mode == 'sigign':
            LT._newton_invert_chunk = sigign_chunk
        kmode = a.mode if a.mode in ('feederkill', 'sigign') else 'kill'
        threading.Thread(target=killer_thread, args=(LT, kmode, out, stop),
                         name='vp2-killer', daemon=True).start()

    emit('dispatch_enter')
    t = time.monotonic()
    got = _traced(la, a.workers, a.N, a.rs, a.fit)
    out['call_seconds'] = round(time.monotonic() - t, 3)
    stop.set()
    out['sha'] = _sha(got)
    out['identical'] = bool(np.array_equal(got, ref))
    out['max_abs_delta'] = (float(np.max(np.abs(got - ref)))
                            if got.shape == ref.shape else None)
    out['shutdown_timeouts'] = getattr(LT, '_POOL_SHUTDOWN_TIMEOUTS', None)
    out['inflight'] = getattr(LT, '_POOL_INFLIGHT', None)
    out['abandoned'] = len(getattr(LT, '_ABANDONED_POOLS', []) or [])
    out['pool_is_none_after'] = LT._PERSISTENT_POOL is None
    out['pool_getter_calls'] = list(_got)
    out['pool_engaged'] = bool(_got)
    emit('dispatch_returned', **out)

    if a.json_out:
        with open(a.json_out, 'w', encoding='utf-8') as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    faulthandler.cancel_dump_traceback_later()
    emit('exiting')
    return 0 if out['identical'] else 4


if __name__ == '__main__':
    sys.exit(main())

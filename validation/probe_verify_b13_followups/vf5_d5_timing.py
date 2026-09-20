"""VERIFY WP-B13 follow-ups, D5 -- re-measuring the healthy side, and
challenging the "bound the BOOTSTRAP with a 600 s sentinel" recommendation.

Independent of ``fu3_chunk_timing.py`` in its primary instrument: chunk
ARRIVALS are read from the dispatcher's own ``sub_progress`` callback
(``newton chunk k/n done``, emitted inside ``_invert_newton_parallel``'s
``as_completed`` loop), which is a consumer-side channel the probe does not
have to wrap anything to obtain.  The executor proxy is used only for SUBMIT
timestamps and its per-chunk figures are reported as a CROSS-CHECK of the
progress channel, never in place of it.

Sections
  timing      cold / warm first-arrival, slowest gap, total, at each N
  bootstrap   construct -> every worker has imported what a chunk needs
              (a raw ``multiprocessing.Barrier`` through an explicit spawn
              context, NOT a Manager Barrier)
  sentinel    cold / warm round-trip of one trivial task
  semantics   what ``as_completed(fs, timeout=T)`` actually bounds, measured
  clause      does the dispatcher's ``except (BrokenProcessPool, RuntimeError,
              OSError, EOFError)`` clause CATCH the exception a timeout would
              raise?  Driven, on three exception classes, including a
              reconstruction of the pre-3.11 ``concurrent.futures.TimeoutError``
  residual    with a sentinel ahead of the chunks, does a LATER wedged chunk
              still hang the dispatch?  Driven under a hard deadline.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=<tree> python validation/probe_verify_b13_followups/\
vf5_d5_timing.py --workers 8 --n 256 512 1024 --out vf5_win.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import warnings


# --------------------------------------------------------------------------
# worker-side task for the bootstrap measurement
# --------------------------------------------------------------------------
def _bootstrap_task(addr):
    """Import what a Newton chunk needs, then rendezvous on a SOCKET.

    A socket rendezvous rather than a ``Manager`` Barrier: the parent does
    not release anyone until it has accepted one connection per worker, so
    "every worker took exactly one task" is a decision the parent makes from
    its own accept count, with no Manager process in the measurement.
    """
    import socket

    import scipy.interpolate  # noqa: F401

    from lumenairy.elements import _lens_traced  # noqa: F401
    t_ready = time.monotonic()
    sk = socket.create_connection(addr, timeout=300.0)
    try:
        sk.sendall(b'r')
        sk.recv(1)                    # blocks until the parent has them all
    finally:
        sk.close()
    return (os.getpid(), t_ready)


def _trivial(x):
    return x


# --------------------------------------------------------------------------
class _SubmitStamps:
    """Records submit times only.  Arrivals come from ``sub_progress``."""

    def __init__(self, ex):
        self._ex = ex
        self.submits = []
        self.dones = []

    def submit(self, fn, *a, **kw):
        self.submits.append(time.monotonic())
        fut = self._ex.submit(fn, *a, **kw)
        fut.add_done_callback(lambda f: self.dones.append(time.monotonic()))
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


def _field(N, ap=3e-3):
    import numpy as np
    dx = 2.2 * ap / N
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / 1.2e-3 ** 2).astype(np.complex128), dx


def _traced(la, n_workers, N, progress=None):
    import numpy as np
    E0, dx = _field(N)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(
            E0, prescription=_fast_singlet(), wavelength=1.31e-6, dx=dx,
            ray_subsample=2, newton_fit='spline', n_workers=n_workers,
            parallel_amp=False, newton_amp_mask_rel=0.0,
            on_undersample='silent', on_aperture_beam='silent',
            progress=progress))


def _timed_run(la, LT, workers, N, cold):
    """One dispatch, instrumented on both channels."""
    if cold:
        LT.close_worker_pool()
    arrivals = []
    seen = {'engaged': False}

    def _progress(*args):
        # ``call_progress`` invokes a top-level progress callback as
        # ``cb(stage, fraction, message)``; a two-argument callback raises
        # TypeError, which ``call_progress`` SWALLOWS -- so the arity has to
        # be right or this channel silently reports nothing (measured: it
        # did, on the first run of this probe).  ``arrivals_seen`` in the
        # output is the guard against that happening unnoticed again.
        msg = args[-1] if args and isinstance(args[-1], str) else ''
        if 'newton chunk' in msg:
            arrivals.append(time.monotonic())

    stamps = {}
    real_get = LT._get_persistent_worker_pool

    def _wrapped(nw):
        ex = real_get(nw)
        proxy = _SubmitStamps(ex)
        stamps['proxy'] = proxy
        seen['engaged'] = True
        return proxy

    LT._get_persistent_worker_pool = _wrapped
    try:
        t0 = time.monotonic()
        got = _traced(la, workers, N, progress=_progress)
        wall = time.monotonic() - t0
    finally:
        LT._get_persistent_worker_pool = real_get

    proxy = stamps.get('proxy')
    subs = sorted(proxy.submits) if proxy else []
    out = {'N': N, 'cold': bool(cold), 'wall_seconds': round(wall, 3),
           'pool_engaged': bool(seen['engaged']),
           'chunks_submitted': len(subs),
           'arrivals_seen': len(arrivals),
           'progress_channel_live': len(arrivals) == len(subs) and bool(subs)}
    if subs and arrivals:
        t_first_submit = subs[0]
        a = sorted(arrivals)
        gaps = [round(a[0] - t_first_submit, 3)] + [
            round(a[i] - a[i - 1], 3) for i in range(1, len(a))]
        out.update({
            'first_arrival_seconds': round(a[0] - t_first_submit, 3),
            'total_seconds': round(a[-1] - t_first_submit, 3),
            'largest_inter_arrival_gap_seconds': max(gaps),
            'arrival_gaps_seconds': gaps,
        })
    if proxy and proxy.dones:
        d = sorted(proxy.dones)
        out['xcheck_slowest_chunk_seconds'] = round(
            max(dd - ss for dd, ss in zip(d, subs)), 3) if subs else None
        out['xcheck_total_seconds'] = round(d[-1] - subs[0], 3) if subs else None
    return out, got


def measure_bootstrap(LT, workers):
    """Pool construction -> every worker has imported what a chunk needs.

    The rendezvous is a listening socket the parent owns (see
    ``_bootstrap_task``), so no Manager process sits inside the measurement
    and "all W workers participated" is the parent's own accept count.
    """
    import socket
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', 0))
    srv.listen(workers + 4)
    srv.settimeout(300.0)
    addr = srv.getsockname()

    LT.close_worker_pool()
    t0 = time.monotonic()
    ex = LT._get_persistent_worker_pool(workers)
    t_built = time.monotonic()
    futs = [ex.submit(_bootstrap_task, addr) for _ in range(workers)]
    conns = []
    try:
        for _ in range(workers):
            c, _a = srv.accept()
            c.recv(1)
            conns.append(c)
        t_all_connected = time.monotonic()
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
    pids = {p for p, _ in rows}
    readies = [r for _, r in rows]
    return {'workers': workers,
            'rendezvous': 'localhost socket (no Manager)',
            'construct_seconds': round(t_built - t0, 4),
            'all_ready_seconds': round(t_all_connected - t0, 4),
            'ready_spread_seconds': round(max(readies) - min(readies), 4),
            'accepted_connections': len(conns),
            'distinct_pids': len(pids),
            'all_workers_participated': (len(pids) == workers
                                         and len(conns) == workers)}


def measure_sentinel(LT, workers):
    LT.close_worker_pool()
    ex = LT._get_persistent_worker_pool(workers)
    t0 = time.monotonic()
    ex.submit(_trivial, 1).result()
    cold = time.monotonic() - t0
    warm = []
    for _ in range(5):
        t = time.monotonic()
        ex.submit(_trivial, 1).result()
        warm.append(time.monotonic() - t)
    return {'cold_seconds': round(cold, 4),
            'warm_worst_seconds': round(max(warm), 6),
            'warm_all_seconds': [round(w, 6) for w in warm]}


# --------------------------------------------------------------------------
# as_completed semantics, measured rather than read
# --------------------------------------------------------------------------
def measure_as_completed_semantics():
    """Is ``timeout=`` per-result or an absolute deadline for the whole
    iteration?  Four futures, each resolved 0.4 s apart, timeout=1.0.

    If T were per-``__next__`` all four would arrive.  If T is an absolute
    deadline taken at generator creation, the iteration raises after ~1.0 s
    having yielded only the first two.
    """
    from concurrent.futures import Future, TimeoutError as CFTimeout, as_completed

    futs = [Future() for _ in range(4)]
    for f in futs:
        f.set_running_or_notify_cancel()

    def _resolve():
        for i, f in enumerate(futs):
            time.sleep(0.4)
            if not f.done():
                try:
                    f.set_result(i)
                except Exception:                  # noqa: BLE001
                    pass

    threading.Thread(target=_resolve, daemon=True).start()
    t0 = time.monotonic()
    got, raised = 0, None
    try:
        for _ in as_completed(futs, timeout=1.0):
            got += 1
    except CFTimeout as exc:                       # noqa: PERF203
        raised = type(exc).__name__
    el = time.monotonic() - t0
    time.sleep(1.8)                                # let _resolve finish
    for f in futs:
        if not f.done():
            try:
                f.set_result(None)
            except Exception:                      # noqa: BLE001
                pass
    return {'resolved_every_seconds': 0.4, 'timeout': 1.0,
            'results_yielded_before_raise': got,
            'exception': raised,
            'elapsed_seconds': round(el, 3),
            'timeout_is_absolute_deadline_for_the_whole_iteration': (
                raised is not None and got < 4),
            'cf_TimeoutError_is_builtin_TimeoutError':
                CFTimeout is TimeoutError,
            'cf_TimeoutError_mro': [c.__name__ for c in CFTimeout.__mro__],
            'cf_TimeoutError_is_OSError_subclass':
                issubclass(CFTimeout, OSError)}


# --------------------------------------------------------------------------
# Does the dispatcher's infrastructure clause CATCH it?  Driven.
# --------------------------------------------------------------------------
class _PreThreeEleven_TimeoutError(Exception):
    """CPython < 3.11's ``concurrent.futures._base.TimeoutError``.

    Before gh-90315 (Python 3.11: "concurrent.futures.TimeoutError and
    asyncio.TimeoutError are now aliases of TimeoutError") this class
    subclassed ``concurrent.futures._base.Error(Exception)`` -- NOT
    ``OSError``.  ``pyproject.toml`` declares ``requires-python = ">=3.10"``,
    so this MRO is inside the supported range.
    """


class _RaisingPool:
    """A stub executor whose ``submit`` raises a chosen exception.

    Installed through ``_get_persistent_worker_pool``, the same substitution
    point the library's own tests use, so what is exercised is the shipped
    dispatcher and its shipped ``except`` tuple.
    """

    def __init__(self, exc):
        self._exc = exc
        self.shutdown_calls = []

    def submit(self, fn, *a, **kw):
        raise self._exc

    def shutdown(self, wait=True, **kw):
        self.shutdown_calls.append(wait)

    _broken = None


def measure_clause_reach(la, LT, serial_ref, N=192):
    import numpy as np
    rows = []
    cases = [
        ('builtin TimeoutError', TimeoutError('t')),
        ('concurrent.futures.TimeoutError (this build)', None),
        ('pre-3.11 cf.TimeoutError MRO', _PreThreeEleven_TimeoutError('t')),
        ('OSError (control, must be caught)', OSError('o')),
        ('ValueError (control, must NOT be caught)', ValueError('v')),
    ]
    from concurrent.futures import TimeoutError as CFTimeout
    cases[1] = ('concurrent.futures.TimeoutError (this build)',
                CFTimeout('t'))
    real_get = LT._get_persistent_worker_pool
    for label, exc in cases:
        LT.close_worker_pool()
        pool = _RaisingPool(exc)
        LT._get_persistent_worker_pool = lambda nw, _p=pool: _p
        caught, err, identical = None, None, None
        try:
            got = _traced(la, 4, N)
            caught = True
            identical = bool(np.array_equal(got, serial_ref))
        except BaseException as e:                 # noqa: BLE001
            caught = False
            err = f'{type(e).__name__}: {e}'
        finally:
            LT._get_persistent_worker_pool = real_get
            LT.close_worker_pool()
        rows.append({'exception': label,
                     'mro': [c.__name__ for c in type(exc).__mro__],
                     'fell_back_to_serial': caught,
                     'identical_to_serial': identical,
                     'escaped_as': err})
    return rows


# --------------------------------------------------------------------------
# Residual exposure: a sentinel ahead of the chunks does NOT bound a chunk
# that wedges LATER.  Driven under a hard deadline.
# --------------------------------------------------------------------------
class _SentinelThenWedgePool:
    """First submit answers immediately; every later submit never completes.

    This is exactly the state candidate B leaves behind: the bootstrap probe
    succeeds, so the bar is cleared, and the dispatch then waits forever on
    an unbounded ``as_completed``.
    """

    def __init__(self):
        from concurrent.futures import Future
        self._Future = Future
        self.n = 0
        self.held = []

    def submit(self, fn, *a, **kw):
        self.n += 1
        f = self._Future()
        if self.n == 1:
            f.set_result((None, 0))
        else:
            self.held.append(f)
        return f

    def shutdown(self, wait=True, **kw):
        pass

    _broken = None


def measure_residual_exposure(la, LT, N=192, deadline=20.0):
    """Sentinel OK, later chunk wedged: does the dispatch return?"""
    real_get = LT._get_persistent_worker_pool
    LT.close_worker_pool()
    pool = _SentinelThenWedgePool()
    LT._get_persistent_worker_pool = lambda nw, _p=pool: _p
    box = {}

    def _run():
        try:
            box['value'] = _traced(la, 4, N)
        except BaseException as exc:               # noqa: BLE001
            box['exc'] = f'{type(exc).__name__}: {exc}'

    t = threading.Thread(target=_run, daemon=True, name='vf5-residual')
    t0 = time.monotonic()
    t.start()
    t.join(deadline)
    alive = t.is_alive()
    el = time.monotonic() - t0
    for f in pool.held:
        try:
            f.set_result((None, 0))
        except Exception:                          # noqa: BLE001
            pass
    LT._get_persistent_worker_pool = real_get
    time.sleep(0.5)
    LT.close_worker_pool()
    return {'deadline_seconds': deadline,
            'submits_seen': pool.n,
            'sentinel_answered': pool.n >= 1,
            'still_running_at_deadline': bool(alive),
            'elapsed_seconds': round(el, 2),
            'outcome': ('HUNG -- a sentinel bar does NOT close this'
                        if alive else box.get('exc', 'returned'))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--n', type=int, nargs='+', default=[256, 512, 1024])
    ap.add_argument('--out', required=True)
    ap.add_argument('--skip-timing', action='store_true')
    args = ap.parse_args()

    import numpy as np

    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    from lumenairy.elements import _lens_traced as LT
    la = lumenairy                                 # apply_real_lens_traced
    # FORCE THE PRECONDITION (docs/TESTING_STANDARDS.md rule 4): the pool's
    # size bars keep a one-shot call off a pool it cannot amortise.  They are
    # not what is measured here, so they are lowered rather than cleared with
    # a field big enough to make every rung cost minutes.
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1

    out = {'python': sys.version, 'platform': sys.platform,
           'lumenairy_file': lumenairy.__file__,
           'workers': args.workers,
           'load_at_start': None}
    try:
        import psutil
        out['load_at_start'] = {
            'cpu_percent_1s': psutil.cpu_percent(interval=1.0),
            'n_cpu': os.cpu_count(),
            'mem_available_GB': round(
                psutil.virtual_memory().available / 2 ** 30, 2)}
        if hasattr(os, 'getloadavg'):
            out['load_at_start']['loadavg'] = os.getloadavg()
    except Exception as exc:                       # noqa: BLE001
        out['load_at_start'] = f'unavailable: {exc}'

    out['as_completed_semantics'] = measure_as_completed_semantics()
    print('semantics', json.dumps(out['as_completed_semantics']), flush=True)

    # serial reference at the clause-reach size
    LT.close_worker_pool()
    ref = _traced(la, 1, 192)
    out['clause_reach'] = measure_clause_reach(la, LT, ref, N=192)
    print('clause_reach', json.dumps(out['clause_reach'], indent=1),
          flush=True)

    out['residual_exposure'] = measure_residual_exposure(la, LT, N=192)
    print('residual', json.dumps(out['residual_exposure']), flush=True)

    if not args.skip_timing:
        out['bootstrap'] = measure_bootstrap(LT, args.workers)
        print('bootstrap', json.dumps(out['bootstrap']), flush=True)
        out['sentinel'] = measure_sentinel(LT, args.workers)
        print('sentinel', json.dumps(out['sentinel']), flush=True)

        rows = []
        for N in args.n:
            LT.close_worker_pool()
            serial = _traced(la, 1, N)
            for cold in (True, False):
                rec, got = _timed_run(la, LT, args.workers, N, cold)
                rec['identical_to_serial'] = bool(np.array_equal(got, serial))
                rec['max_abs_delta'] = float(
                    np.nanmax(np.abs(np.asarray(got) - np.asarray(serial))))
                rows.append(rec)
                print('timing', json.dumps(rec), flush=True)
        out['timing'] = rows
        LT.close_worker_pool()

    try:
        import psutil
        out['load_at_end'] = {
            'cpu_percent_1s': psutil.cpu_percent(interval=1.0)}
        if hasattr(os, 'getloadavg'):
            out['load_at_end']['loadavg'] = os.getloadavg()
    except Exception:                              # noqa: BLE001
        pass

    with open(args.out, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', args.out)


if __name__ == '__main__':
    main()

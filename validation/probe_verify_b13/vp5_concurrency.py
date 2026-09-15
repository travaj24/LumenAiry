"""VP5 -- the pool invariants under concurrency.

Claim (3) of the WP report says the live pool's width is a CEILING, raised
only when a call needs more AND nothing is in flight, lowered only by
``close_worker_pool``.  This probe attacks that from the outside (two real
dispatching threads, a close racing a dispatch, a dispatch while a reaper is
wedged) and from the inside (the getter driven directly with a fake executor).

Every scenario is a decision, not a reading: a hang, or a field that is not
byte-identical to the ``n_workers=1`` answer, is a defect.

SCENARIOS

twothread    two threads dispatch at once -- the shape of
             ``propagate_traced_carrier_chain``'s two arms.
closerace    one thread dispatches in a loop while another calls
             ``close_worker_pool()`` every few milliseconds.
abandonrace  a pool that is broken AND wedged is retired (so the daemon reaper
             is blocked forever inside ``shutdown``), and a fresh dispatch must
             still complete.
ceiling      the getter is driven directly with a counting fake executor: does
             a narrower request shrink the pool?  does a wider request rebuild
             while a dispatch is in flight?
leak         many dispatches, half of them failing, then read
             ``_POOL_INFLIGHT``: it must be exactly 0.
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
    return {'name': 'vp5', 'aperture_diameter': ap, 'thicknesses': [3e-3],
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


def recording_chunk(args):
    """The real chunk worker, plus a file per worker pid that served one.

    Module scope so a spawn worker can resolve it.  This is how the CEILING
    argument is checked from the outside: the claim is that the clamp is
    honoured by the CHUNK COUNT, so a pool wider than the clamp must leave the
    surplus workers idle.
    """
    import os as _os
    import time as _t

    from lumenairy.elements import _lens_traced as _LT
    t0 = _t.time()
    out = _LT._newton_invert_chunk(args)
    t1 = _t.time()
    d = _os.environ.get('VP5_PIDDIR')
    if d:
        try:
            with open(_os.path.join(d, str(_os.getpid())), 'a') as fh:
                fh.write('%r %r' % (t0, t1) + chr(10))
        except OSError:
            pass
    return out


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class _WedgedBrokenPool:
    """Broken on submit, never returns from a joining shutdown."""

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
        f.set_exception(BrokenProcessPool('vp5: broken'))
        return f

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.shutdown_calls.append({'wait': bool(wait),
                                    'cancel_futures': bool(cancel_futures)})
        self.entered.set()
        # Even a NON-waiting shutdown blocks here: that is CPython's
        # _shutdown_lock held by a wedged _terminate_broken (VP1).
        self._never.wait()


class _FakeExecutor:
    """Enough of an executor for the getter; counts its own construction."""

    made = []

    def __init__(self, max_workers=None, mp_context=None, initializer=None,
                 **kw):
        self.max_workers = max_workers
        self._broken = None
        self.shutdowns = []
        _FakeExecutor.made.append(self)

    def submit(self, fn, *a, **kw):
        from concurrent.futures import Future
        f = Future()
        f.set_result(None)
        return f

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.shutdowns.append({'wait': bool(wait),
                               'cancel_futures': bool(cancel_futures)})


def sc_twothread(la, LT, a):
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1
    LT.close_worker_pool()
    ref_a = _traced(la, 1, a.N, a.rs, 9e-3)
    ref_b = _traced(la, 1, a.N, a.rs, 12e-3)
    emit('references', a=_sha(ref_a)[:12], b=_sha(ref_b)[:12])
    LT.close_worker_pool()
    res = {}

    def arm(tag, r, nw):
        try:
            for i in range(a.reps):
                f = _traced(la, nw, a.N, a.rs, r)
                res.setdefault(tag, []).append(_sha(f))
        except BaseException as exc:                      # noqa: BLE001
            res[tag + '_exc'] = repr(exc)

    t1 = threading.Thread(target=arm, args=('a', 9e-3, 4), name='arm-a')
    t2 = threading.Thread(target=arm, args=('b', 12e-3, 8), name='arm-b')
    t0 = time.monotonic()
    t1.start()
    t2.start()
    t1.join(a.arm_timeout)
    t2.join(a.arm_timeout)
    hung = [t.name for t in (t1, t2) if t.is_alive()]
    return {'scenario': 'twothread', 'hung_threads': hung,
            'seconds': round(time.monotonic() - t0, 3),
            'a_all_match_serial': all(s == _sha(ref_a)
                                      for s in res.get('a', [])),
            'b_all_match_serial': all(s == _sha(ref_b)
                                      for s in res.get('b', [])),
            'a_count': len(res.get('a', [])), 'b_count': len(res.get('b', [])),
            'exceptions': {k: v for k, v in res.items() if k.endswith('_exc')},
            'final_inflight': getattr(LT, '_POOL_INFLIGHT', None),
            'pool_workers': LT._PERSISTENT_POOL_NWORKERS}


def sc_closerace(la, LT, a):
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1
    LT.close_worker_pool()
    ref = _traced(la, 1, a.N, a.rs, 9e-3)
    emit('reference', sha=_sha(ref)[:12])
    LT.close_worker_pool()
    stop = threading.Event()
    closes = {'n': 0}

    def closer():
        while not stop.is_set():
            LT.close_worker_pool()
            closes['n'] += 1
            stop.wait(0.05)

    res = {'shas': [], 'exc': None}

    def worker():
        try:
            for i in range(a.reps):
                res['shas'].append(_sha(_traced(la, 4, a.N, a.rs, 9e-3)))
        except BaseException as exc:                      # noqa: BLE001
            res['exc'] = repr(exc)

    ct = threading.Thread(target=closer, name='closer', daemon=True)
    wt = threading.Thread(target=worker, name='dispatcher')
    t0 = time.monotonic()
    ct.start()
    wt.start()
    wt.join(a.arm_timeout)
    hung = wt.is_alive()
    stop.set()
    ct.join(timeout=10)
    return {'scenario': 'closerace', 'hung': hung,
            'seconds': round(time.monotonic() - t0, 3),
            'closes': closes['n'], 'dispatches': len(res['shas']),
            'all_match_serial': all(s == _sha(ref) for s in res['shas']),
            'exception': res['exc'],
            'final_inflight': getattr(LT, '_POOL_INFLIGHT', None)}


def sc_abandonrace(la, LT, a):
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1
    LT.close_worker_pool()
    ref = _traced(la, 1, a.N, a.rs, 9e-3)
    emit('reference', sha=_sha(ref)[:12])
    LT.close_worker_pool()
    wedged = _WedgedBrokenPool()
    LT._PERSISTENT_POOL = wedged
    LT._PERSISTENT_POOL_NWORKERS = 4
    t0 = time.monotonic()
    # close_worker_pool must hand this to the reaper and RETURN.
    LT.close_worker_pool()
    close_s = round(time.monotonic() - t0, 3)
    entered = wedged.entered.wait(30.0)
    # ...and a fresh dispatch must work while that reaper is stuck.
    t1 = time.monotonic()
    field = _traced(la, 4, a.N, a.rs, 9e-3)
    return {'scenario': 'abandonrace',
            'close_worker_pool_seconds': close_s,
            'reaper_entered_shutdown': bool(entered),
            'reaper_shutdown_args': wedged.shutdown_calls,
            'dispatch_seconds': round(time.monotonic() - t1, 3),
            'identical': bool(np.array_equal(field, ref)),
            'live_threads_named_reaper': [
                t.name for t in threading.enumerate()
                if 'reaper' in t.name or 'close' in t.name],
            'final_inflight': getattr(LT, '_POOL_INFLIGHT', None),
            'abandoned_len': len(getattr(LT, '_ABANDONED_POOLS', []) or [])}


def sc_ceiling(la, LT, a):
    import concurrent.futures as cf
    cf.ProcessPoolExecutor = _FakeExecutor
    LT.close_worker_pool()
    _FakeExecutor.made.clear()
    rows = []

    def step(tag, n, expect_same_as=None):
        ex = LT._get_persistent_worker_pool(n)
        rows.append({'step': tag, 'asked': n,
                     'pool_workers': LT._PERSISTENT_POOL_NWORKERS,
                     'constructions': len(_FakeExecutor.made),
                     'is_previous': (ex is expect_same_as)
                     if expect_same_as is not None else None})
        emit('ceiling_step', **rows[-1])
        return ex

    p4 = step('build_4', 4)
    step('narrower_2_must_not_shrink', 2, p4)
    step('same_4', 4, p4)
    p8 = step('wider_8_idle_must_grow', 8)
    # now claim the pool and ask wider again
    if hasattr(LT, '_note_pool_inflight'):
        LT._note_pool_inflight(1)
        step('wider_16_inflight_must_not_rebuild', 16, p8)
        LT._note_pool_inflight(-1)
        step('wider_16_idle_must_rebuild', 16)
        inflight_after = LT._POOL_INFLIGHT
    else:
        step('wider_16_no_inflight_machinery', 16)
        inflight_after = None
    # a broken pool must be replaced unconditionally
    LT._PERSISTENT_POOL._broken = 'vp5 marked broken'
    step('broken_pool_is_replaced', 1)
    return {'scenario': 'ceiling', 'rows': rows,
            'constructions': len(_FakeExecutor.made),
            'inflight_after': inflight_after,
            'shutdowns': [e.shutdowns for e in _FakeExecutor.made]}


def sc_leak(la, LT, a):
    import concurrent.futures as cf
    cf.ProcessPoolExecutor = _FakeExecutor
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1
    LT.close_worker_pool()
    _FakeExecutor.made.clear()
    n_exc = 0
    for i in range(a.reps):
        try:
            _traced(la, 4, a.N, a.rs, 9e-3)
        except BaseException:                             # noqa: BLE001
            n_exc += 1
    return {'scenario': 'leak', 'reps': a.reps, 'exceptions': n_exc,
            'final_inflight': getattr(LT, '_POOL_INFLIGHT', None),
            'constructions': len(_FakeExecutor.made)}


def sc_chunkcap(la, LT, a):
    """A pool wider than the clamp must run only ``clamp`` workers."""
    import tempfile
    LT._POOL_MIN_PIXELS = 1
    LT._POOL_MIN_PIXELS_WARM = 1
    LT.close_worker_pool()
    ref = _traced(la, 1, a.N, a.rs, 9e-3)
    LT.close_worker_pool()
    # The pid directory must be in the environment BEFORE the workers spawn:
    # a spawn child inherits the environment as it was at fork/spawn time.
    d = tempfile.mkdtemp(prefix='vp5cap')
    os.environ['VP5_PIDDIR'] = d
    LT._newton_invert_chunk = recording_chunk
    # Build a WIDE pool first, then price the next call down to a few workers.
    la.set_max_ram(None)
    _traced(la, a.wide, a.N, a.rs, 9e-3)
    wide = LT._PERSISTENT_POOL_NWORKERS
    rows = []
    seen = []
    _real_resolve = LT._newton_resolve_workers

    def _resolve_spy(requested, n_total_, fit_points_, **kw):
        ans = _real_resolve(requested, n_total_, fit_points_, **kw)
        seen.append({'fit_points': int(fit_points_), 'answer': int(ans)})
        return ans

    LT._newton_resolve_workers = _resolve_spy
    for want in [int(x) for x in a.caps.split(',')]:
        del seen[:]
        for f in os.listdir(d):
            try:
                os.remove(os.path.join(d, f))
            except OSError:
                pass
        n_total = (a.N // a.rs) * (a.N // a.rs)
        pw = LT._newton_worker_bytes(float(n_total) / float(want), a.fit_points)
        la.set_max_ram(((want + 0.5) * pw + 2.0e9) / 0.5 / 1e9)
        field = _traced(la, a.wide, a.N, a.rs, 9e-3)
        clamp = max((r['answer'] for r in seen), default=None)
        spans = []
        for f in os.listdir(d):
            with open(os.path.join(d, f)) as fh:
                for line in fh:
                    p0, p1 = line.split()
                    spans.append((float(p0), float(p1)))
        edges = sorted([(s0, 1) for s0, _ in spans]
                       + [(s1, -1) for _, s1 in spans])
        cur = peak = 0
        for _, delta in edges:
            cur += delta
            peak = max(peak, cur)
        rows.append({'engineered_clamp': want, 'clamp_answer': clamp,
                     'pool_workers': LT._PERSISTENT_POOL_NWORKERS,
                     'workers_that_served_a_chunk': len(os.listdir(d)),
                     'chunks_run': len(spans),
                     'dispatcher_resolve_calls': list(seen),
                     'peak_concurrent_chunks': peak,
                     'identical': bool(np.array_equal(field, ref))})
        emit('cap_row', **rows[-1])
    return {'scenario': 'chunkcap', 'wide_pool': wide, 'rows': rows,
            'final_inflight': getattr(LT, '_POOL_INFLIGHT', None)}


_SC = {'chunkcap': sc_chunkcap, 'twothread': sc_twothread, 'closerace': sc_closerace,
       'abandonrace': sc_abandonrace, 'ceiling': sc_ceiling, 'leak': sc_leak}


def main():
    global T0
    T0 = time.monotonic()
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', required=True, choices=sorted(_SC))
    ap.add_argument('--N', type=int, default=256)
    ap.add_argument('--rs', type=int, default=2)
    ap.add_argument('--reps', type=int, default=6)
    ap.add_argument('--arm-timeout', type=float, default=180.0)
    ap.add_argument('--wide', type=int, default=16)
    ap.add_argument('--caps', default='2,4,8')
    ap.add_argument('--fit-points', type=int, default=250000)
    ap.add_argument('--dump-after', type=float, default=600.0)
    ap.add_argument('--json-out', default=None)
    a = ap.parse_args()
    faulthandler.enable()
    faulthandler.dump_traceback_later(a.dump_after, exit=True)
    import lumenairy as la
    from lumenairy.elements import _lens_traced as LT
    emit('import', lumenairy=la.__file__, python=sys.version.split()[0],
         pid=os.getpid(), scenario=a.scenario)
    out = _SC[a.scenario](la, LT, a)
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

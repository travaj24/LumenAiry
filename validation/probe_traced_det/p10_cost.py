"""Whole-call overhead, INTERLEAVED, and how far the exit field moves.

Interleaved (OFF, ON, OFF, ON, ...) in ONE process at one pinned width, so a
drifting box cannot be read as a cost.  argv: N [ray_subsample] [reps]
Also saves both fields so the driver can quote the field delta against the
shipped route's own cross-width spread.
"""
import hashlib
import os
import sys
import time
import tracemalloc
import warnings

import numpy as np

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_want = os.path.realpath(os.path.join(os.environ['LUMENAIRY_ROOT'],
                                      'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

LAM = 1.31e-6


def _h(a):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


def build(N, DX=30e-6, wpx=None, R=1.0):
    wpx = wpx or int(0.15625 * N)
    x = (np.arange(N) - N / 2) * DX
    X, Y = np.meshgrid(x, x)
    w = wpx * DX
    E = (np.exp(-(X * X + Y * Y) / (w * w))
         * np.exp(1j * (2 * np.pi / LAM) * (X * X + Y * Y) / (2 * R))
         ).astype(np.complex128)
    presc = {'wavelength': LAM, 'aperture_diameter': 14e-3, 'surfaces': [
        {'radius': 51.68e-3, 'thickness': 4e-3, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 7e-3},
        {'radius': -51.68e-3, 'thickness': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 7e-3}],
        'thicknesses': [4e-3], 'stop_index': 0}
    return E, presc, DX


def one(E, presc, DX, det, sub, solve_log):
    LT.DETERMINISTIC_TRACED_FIT = bool(det)
    del solve_log[:]
    orig = LT._solve_lstsq_thread_safe

    def spy(A, b, deterministic=False):
        t0 = time.perf_counter()
        out = orig(A, b, deterministic=deterministic)
        solve_log.append((np.shape(A), time.perf_counter() - t0))
        return out

    LT._solve_lstsq_thread_safe = spy
    kw = {} if sub is None else {'ray_subsample': sub}
    try:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            F = la.apply_real_lens_traced(
                E, prescription=presc, wavelength=LAM, dx=DX,
                carrier='auto', on_undersample='silent', **kw)
        return time.perf_counter() - t0, np.asarray(F)
    finally:
        LT._solve_lstsq_thread_safe = orig


def main():
    N = int(sys.argv[1])
    sub = None if len(sys.argv) < 3 or sys.argv[2] == 'none' else int(sys.argv[2])
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    E, presc, DX = build(N)
    log = []
    tof, ton, sof, son = [], [], [], []
    Foff = Fon = None
    for _ in range(reps):
        t, Foff = one(E, presc, DX, False, sub, log)
        tof.append(t)
        sof.append(sum(e[1] for e in log))
        shapes = [e[0] for e in log]
        t, Fon = one(E, presc, DX, True, sub, log)
        ton.append(t)
        son.append(sum(e[1] for e in log))
    w = os.environ.get('OMP_NUM_THREADS')
    print('N=%d sub=%s OMP=%s reps=%d  solves %s' % (N, sub, w, reps, shapes))
    print('  whole call   OFF %.3f s   ON %.3f s   overhead %+.2f%%'
          % (min(tof), min(ton), 100.0 * (min(ton) / min(tof) - 1.0)))
    print('  solves only  OFF %.4f s   ON %.4f s   (%.2f%% / %.2f%% of call)'
          % (min(sof), min(son), 100.0 * min(sof) / min(tof),
             100.0 * min(son) / min(ton)))
    d = np.abs(np.nan_to_num(Fon) - np.nan_to_num(Foff))
    pk = float(np.nanmax(np.abs(Foff)))
    print('  field  OFF %s  ON %s   |ON-OFF|/peak %.3e'
          % (_h(Foff), _h(Fon), float(d.max()) / pk))
    out = os.environ.get('D15_SAVE')
    if out:
        np.save(out + '_off.npy', Foff)
        np.save(out + '_on.npy', Fon)

    # ---- footprint, same instrument on both arms ----
    for det in (False, True):
        LT.DETERMINISTIC_TRACED_FIT = det
        tracemalloc.start()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            la.apply_real_lens_traced(
                E, prescription=presc, wavelength=LAM, dx=DX, carrier='auto',
                on_undersample='silent',
                **({} if sub is None else {'ray_subsample': sub}))
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        print('  footprint det=%d peak %.3f float64 grids (%.1f MB)'
              % (det, peak / (8.0 * N * N), peak / 1e6))
    sys.stdout.flush()


if __name__ == '__main__':
    main()

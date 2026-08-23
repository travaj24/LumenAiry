"""Whole-call instrument: the traced exit field's hash, the total call time,
and the time each least-squares solve on the path costs.

Run one arm per process with the BLAS width pinned before NumPy loads.
argv: N  [det_traced 0/1]
"""
import hashlib
import os
import sys
import time
import warnings

import numpy as np

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_want = os.path.realpath(os.path.join(os.environ['LUMENAIRY_ROOT'],
                                      'lumenairy'))
_got = os.path.realpath(os.path.dirname(la.__file__))
assert _got == _want, 'imported %r, expected %r' % (_got, _want)

LAM = 1.31e-6


def _h(a):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]


def fixture(N, DX=30e-6, R=1.0, wfrac=0.15625):
    x = (np.arange(N) - N / 2) * DX
    X, Y = np.meshgrid(x, x)
    w = wfrac * N * DX
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


def main():
    N = int(sys.argv[1])
    det = bool(int(sys.argv[2])) if len(sys.argv) > 2 else False
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if hasattr(LT, 'DETERMINISTIC_TRACED_FIT'):
        LT.DETERMINISTIC_TRACED_FIT = det
    E, presc, DX = fixture(N)
    orig = LT._solve_lstsq_thread_safe
    log = []

    def spy(A, b, **kw):
        t0 = time.perf_counter()
        out = orig(A, b, **kw)
        log.append((np.shape(A), kw.get('deterministic', False),
                    time.perf_counter() - t0, _h(out)))
        return out

    LT._solve_lstsq_thread_safe = spy
    try:
        for r in range(reps):
            del log[:]
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                F = la.apply_real_lens_traced(
                    E, prescription=presc, wavelength=LAM, dx=DX,
                    carrier='auto', on_undersample='silent')
            dt = time.perf_counter() - t0
            solve_t = sum(e[2] for e in log)
            print('CALL N=%d det=%d OMP=%s rep=%d  %.3f s  solves %.4f s '
                  '(%.2f%%)  FIELD %s'
                  % (N, det, os.environ.get('OMP_NUM_THREADS'), r, dt,
                     solve_t, 100.0 * solve_t / dt, _h(np.asarray(F))))
            for shp, d, t, hx in log:
                print('    %-16s det=%-5s %8.4f s  x=%s' % (shp, d, t, hx))
            sys.stdout.flush()
    finally:
        LT._solve_lstsq_thread_safe = orig


if __name__ == '__main__':
    main()

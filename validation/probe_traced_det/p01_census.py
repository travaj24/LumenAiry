"""Census every _solve_lstsq_thread_safe call on the traced chain, with
the CALLER's source line, so the '120-term residual-eikonal fit' claim in
BUILD_DETERMINISTIC_CARRIER_FIT S7.3 is attributed rather than assumed."""
import hashlib
import os
import sys
import traceback
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
        np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:10]


def fixture(N=512, DX=30e-6, R=1.0, wpx=80):
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


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    E, presc, DX = fixture(N=N)
    seen = []
    orig = LT._solve_lstsq_thread_safe

    def spy(A, b, deterministic=False):
        st = traceback.extract_stack()
        caller = st[-2]
        out = orig(A, b, deterministic=deterministic)
        seen.append((np.shape(A), bool(deterministic),
                     os.path.basename(caller.filename) + ':'
                     + str(caller.lineno),
                     caller.name, _h(A), _h(np.asarray(b)), _h(out)))
        return out

    LT._solve_lstsq_thread_safe = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            F = la.apply_real_lens_traced(
                E, prescription=presc, wavelength=LAM, dx=DX,
                carrier='auto', on_undersample='silent')
    finally:
        LT._solve_lstsq_thread_safe = orig
    print('N=%d  OMP=%s  FIELD %s' % (N, os.environ.get('OMP_NUM_THREADS'),
                                      _h(np.asarray(F))))
    for shp, det, where, name, ha, hb, ho in seen:
        print('  SOLVE %-16s det=%-5s %-22s %-24s A=%s b=%s x=%s'
              % (shp, det, where, name, ha, hb, ho))


if __name__ == '__main__':
    main()

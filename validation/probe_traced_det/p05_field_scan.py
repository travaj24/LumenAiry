"""Scan fixtures for one where the traced EXIT FIELD itself moves with the
BLAS width -- the field-level fail-before the acceptance criterion wants.

The coefficient-level fail-before is already measured (p04): the two
(1337, 120) inverse-map solves read different bytes at widths {1,2} and
{4,8}.  Whether that reaches the FIELD is a separate, fixture-dependent
question, and this answers it by search rather than by assumption.
"""
import hashlib
import os
import sys
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


CASES = [
    # (N, dx, beam px, carrier R, aperture mm, ray_subsample)
    (384, 30e-6, 60, 1.0, 14.0, None),
    (512, 30e-6, 80, 1.0, 14.0, None),
    (512, 30e-6, 120, 1.0, 14.0, None),
    (512, 30e-6, 80, 0.5, 14.0, None),
    (512, 20e-6, 80, 1.0, 10.0, None),
    (768, 30e-6, 120, 1.0, 14.0, None),
    (640, 30e-6, 100, 2.0, 14.0, None),
    (512, 30e-6, 80, 1.0, 14.0, 2),
]


def main():
    det = bool(int(sys.argv[1])) if len(sys.argv) > 1 else False
    if hasattr(LT, 'DETERMINISTIC_TRACED_FIT'):
        LT.DETERMINISTIC_TRACED_FIT = det
    for i, (N, DX, wpx, R, apmm, sub) in enumerate(CASES):
        x = (np.arange(N) - N / 2) * DX
        X, Y = np.meshgrid(x, x)
        w = wpx * DX
        E = (np.exp(-(X * X + Y * Y) / (w * w))
             * np.exp(1j * (2 * np.pi / LAM) * (X * X + Y * Y) / (2 * R))
             ).astype(np.complex128)
        presc = {'wavelength': LAM, 'aperture_diameter': apmm * 1e-3,
                 'surfaces': [
                     {'radius': 51.68e-3, 'thickness': 4e-3,
                      'glass_before': 'air', 'glass_after': 'N-BK7',
                      'semi_diameter': apmm * 0.5e-3},
                     {'radius': -51.68e-3, 'thickness': 0.0,
                      'glass_before': 'N-BK7', 'glass_after': 'air',
                      'semi_diameter': apmm * 0.5e-3}],
                 'thicknesses': [4e-3], 'stop_index': 0}
        kw = {} if sub is None else {'ray_subsample': sub}
        seen = []
        orig = LT._solve_lstsq_thread_safe

        def spy(A, b, _seen=seen, **kwx):
            out = orig(A, b, **kwx)
            _seen.append((np.shape(A), _h(out), bool(kwx.get('deterministic', False))))
            return out

        LT._solve_lstsq_thread_safe = spy
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                F = la.apply_real_lens_traced(
                    E, prescription=presc, wavelength=LAM, dx=DX,
                    carrier='auto', on_undersample='silent', **kw)
        except Exception as exc:                          # noqa: BLE001
            print('CASE %d N=%d SKIP %s' % (i, N, type(exc).__name__))
            continue
        finally:
            LT._solve_lstsq_thread_safe = orig
        wide = ['%s%s' % (h[:8], 'D' if d else 'b') for s, h, d in seen if s[1] >= 100]
        narrow = ['%s%s' % (h[:6], 'D' if d else 'b') for s, h, d in seen if s[1] < 100]
        print('CASE %d N=%-4d wpx=%-4d R=%-4g ap=%-5g sub=%s OMP=%s det=%d '
              'FIELD %s  wide=%s  narrow=%s'
              % (i, N, wpx, R, apmm, sub, os.environ.get('OMP_NUM_THREADS'),
                 det, _h(np.asarray(F)), ','.join(wide), ','.join(narrow)))
        sys.stdout.flush()


if __name__ == '__main__':
    main()

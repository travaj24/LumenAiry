"""Capture every least-squares fit a representative traced call makes.

Wraps ``_solve_lstsq_thread_safe`` script-side (no library edit) and writes each
``(A, b)`` to an ``.npz`` with its call-site tag, so the adjudication can be run
offline, repeatedly, without re-driving the propagator.

Usage:  LUMENAIRY_ROOT=... python capture.py <fixture> <outdir>
Fixtures: singlet | singlet_nosub | singlet_big | biconcave | fast_decentred
"""
import os
import sys
import traceback
import warnings

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la  # noqa: E402
from lumenairy.elements import _lens_traced as LT  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
_got = os.path.realpath(os.path.dirname(la.__file__))
assert _got == _want, f'imported {_got!r}, expected {_want!r}'

LAM = 1.31e-6


def _singlet(semi=7e-3, R=51.68e-3, th=4e-3):
    return {'wavelength': LAM, 'aperture_diameter': 2 * semi, 'surfaces': [
        {'radius': R, 'thickness': th, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': semi},
        {'radius': -R, 'thickness': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': semi}],
        'thicknesses': [th], 'stop_index': 0}


def _biconcave():
    return {'wavelength': LAM, 'aperture_diameter': 24e-3, 'surfaces': [
        {'radius': -51.68e-3, 'thickness': 3e-3, 'glass_before': 'air',
         'glass_after': 'N-BK7', 'semi_diameter': 12e-3},
        {'radius': 51.68e-3, 'thickness': 0.0, 'glass_before': 'N-BK7',
         'glass_after': 'air', 'semi_diameter': 12e-3}],
        'thicknesses': [3e-3], 'stop_index': 0}


def _gauss(N, dx, w_frac=0.15625, R_in=None, tilt=None, ctr=(0.0, 0.0)):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    w = w_frac * N * dx
    Xc, Yc = X - ctr[0], Y - ctr[1]
    E = np.exp(-(Xc * Xc + Yc * Yc) / (w * w)).astype(np.complex128)
    k = 2 * np.pi / LAM
    if R_in is not None:
        E = E * np.exp(1j * k * (Xc * Xc + Yc * Yc) / (2.0 * R_in))
    if tilt is not None:
        E = E * np.exp(1j * k * (tilt[0] * X + tilt[1] * Y))
    return E


FIXTURES = {
    # p09's fixture: the one the 5.42.0 audit measured 1.6e-9 on.
    'singlet': dict(N=512, dx=30e-6, presc=_singlet(), R_in=1.0,
                    kw=dict(carrier='auto', ray_subsample=2)),
    'singlet_nosub': dict(N=512, dx=30e-6, presc=_singlet(), R_in=1.0,
                          kw=dict(carrier='auto')),
    'singlet_big': dict(N=768, dx=20e-6, presc=_singlet(), R_in=1.0,
                        kw=dict(carrier='auto', ray_subsample=2)),
    # the P4 module's own traced oracle call (converging input, negative lens)
    'biconcave': dict(N=384, dx=10e-6, presc=_biconcave(), R_in=-35e-3,
                      w_frac=0.26, kw=dict(carrier=-35e-3,
                                           on_noncollimated='ignore')),
    # design-121-LIKE: a fast lens, a DECENTRED illuminated patch and a tilted
    # carrier -- the shape that puts the fit disc off the basis centre and so
    # engages the weighted (two-scale) rows C13 was written for.
    'fast_decentred': dict(N=512, dx=20e-6, presc=_singlet(semi=5e-3,
                                                          R=25.0e-3, th=6e-3),
                           R_in=1.0, w_frac=0.06, ctr=(1.6e-3, -1.1e-3),
                           tilt=(0.02, -0.013),
                           kw=dict(carrier='auto', ray_subsample=2)),
}


def main():
    name = sys.argv[1]
    outdir = sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    f = FIXTURES[name]
    E = _gauss(f['N'], f['dx'], w_frac=f.get('w_frac', 0.15625),
               R_in=f.get('R_in'), tilt=f.get('tilt'), ctr=f.get('ctr', (0, 0)))
    cap = []
    orig = LT._solve_lstsq_thread_safe

    def spy(A, b, deterministic=False):
        cap.append((np.ascontiguousarray(A, np.float64),
                    np.array(b, dtype=np.float64), bool(deterministic)))
        return orig(A, b, deterministic=deterministic)

    LT._solve_lstsq_thread_safe = spy
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            la.apply_real_lens_traced(E, prescription=f['presc'],
                                      wavelength=LAM, dx=f['dx'],
                                      on_undersample='silent', **f['kw'])
    except Exception:
        traceback.print_exc()
    finally:
        LT._solve_lstsq_thread_safe = orig
    for i, (A, b, det) in enumerate(cap):
        np.savez_compressed(os.path.join(outdir, f'{name}_{i:02d}.npz'),
                            A=A, b=b, det=np.array([det]))
        print(f"  {name}[{i}] A={A.shape} b={b.shape} deterministic={det}")
    print(f"{name}: {len(cap)} fits -> {outdir}")


if __name__ == '__main__':
    main()

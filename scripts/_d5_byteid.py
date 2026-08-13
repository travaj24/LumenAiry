"""BYTE-IDENTITY of the POLYNOMIAL default path, measured rather than argued.

Emits an .npz of returned fields for a matrix of fit-domain configurations.
Run once in each tree (origin/main and the fix) and compare with --compare.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

import warnings  # noqa: E402

import numpy as np  # noqa: E402

import lumenairy as la  # noqa: E402

_WL = 1.31e-6
_N, _DX, _W = 256, 12e-6, 0.60e-3


def _singlet(ap, R=18e-3):
    return {'name': 's', 'aperture_diameter': ap, 'thicknesses': [3.0e-3],
            'surfaces': [
                {'radius': R, 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': -R, 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def _field(cx=0.0, n=_N, dx=_DX, w=_W):
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    return np.exp(-((X - cx) ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


#: (tag, kwargs) -- every branch of the fit-domain resolver on the DEFAULT
#: basis: no restriction, concentric disc (hard mask), decentred disc (D1
#: weights + C11 arbiter + D7 order raise), the too-few-samples abandonment,
#: and the ray-density amplitude branch that reads det J.
CASES = [
    ('bare', dict()),
    ('frbf2', dict(fit_radius_beam_factor=2.0)),
    ('frbf1p2', dict(fit_radius_beam_factor=1.2)),
    ('frbf2_poly', dict(fit_radius_beam_factor=2.0, newton_fit='polynomial')),
    ('frbf2_auto', dict(fit_radius_beam_factor=2.0, newton_fit='auto')),
    ('frbf2_dec', dict(fit_radius_beam_factor=2.0, cx=0.35e-3,
                       beam_centre=(0.35e-3, 0.0))),
    ('frbf2_dec_o10', dict(fit_radius_beam_factor=2.0, cx=0.35e-3,
                           beam_centre=(0.35e-3, 0.0),
                           decentred_fit_poly_order=10)),
    ('frbf2_raydens', dict(fit_radius_beam_factor=2.0,
                           amplitude_model='ray_density')),
    ('frbf2_remap', dict(fit_radius_beam_factor=2.0,
                         amplitude_model='ray_density',
                         preserve_input_phase='remap', carrier=-0.06)),
    ('frbf_tiny', dict(fit_radius_beam_factor=0.05)),
    ('frbf2_sub2', dict(fit_radius_beam_factor=2.0, ray_subsample=2)),
    ('frbf2_fit', dict(fit_radius_beam_factor=2.0, inversion_method='fit')),
]


def run_all():
    out = {}
    for tag, kw in CASES:
        kw = dict(kw)
        cx = kw.pop('cx', 0.0)
        kw.setdefault('on_undersample', 'silent')
        kw.setdefault('on_noncollimated', 'silent')
        kw.setdefault('on_aperture_beam', 'silent')
        kw.setdefault('ray_subsample', 4)
        kw.setdefault('n_workers', 1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                f = np.asarray(la.apply_real_lens_traced(
                    _field(cx), prescription=_singlet(3.0e-3), wavelength=_WL,
                    dx=_DX, **kw))
            out[tag] = f
        except Exception as exc:                          # noqa: BLE001
            out[tag] = np.array([repr(type(exc).__name__) + ':' + str(exc)[:80]])
    return out


if __name__ == '__main__':
    if '--compare' in sys.argv:
        a = np.load(sys.argv[-2], allow_pickle=True)
        b = np.load(sys.argv[-1], allow_pickle=True)
        bad = 0
        for tag, _kw in CASES:
            u, v = a[tag], b[tag]
            if u.dtype.kind in 'US' or v.dtype.kind in 'US':
                ok = bool(np.array_equal(u, v))
            else:
                ok = (u.shape == v.shape
                      and u.tobytes() == v.tobytes())
            bad += (not ok)
            extra = ''
            if not ok and u.dtype.kind not in 'US' and u.shape == v.shape:
                extra = '   max|d| = %.3e' % float(np.abs(u - v).max())
            print('%-18s %s%s' % (tag, 'IDENTICAL' if ok else 'DIFFERS',
                                  extra))
        print('\n%d of %d cases differ' % (bad, len(CASES)))
        sys.exit(1 if bad else 0)
    np.savez(sys.argv[1], **run_all())
    print('wrote', sys.argv[1])

"""VERIFY-B14 arm 1 -- the c7/c8 manufactured halo, re-measured independently.

Rebuilds the C8 ``_GHOST`` stimulus from the prescription and the field
construction in the test file (not by importing the test module), then reads
``max|E| beyond 3 w / peak`` with the C6 launch on, the C6 fit guard on/off and
the C8 support bound on/off, at whatever ``decentred_fit_poly_order`` is asked
for.  The library tree is whatever ``PYTHONPATH`` points at and is asserted in
the output, so the same script measures any archived commit.

Usage: probe_v1_halo_order.py --orders 6,8,10,16 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL

_GHOST = dict(n=768, dx=25e-6, w=1.5e-3, rc=-0.15, alpha=5.0,
              r1=150e-3, r2=-150e-3, th=4e-3, z=6e-3, ap=18e-3)
_CLEAN = dict(n=384, dx=30e-6, w=0.9e-3, rc=-0.20, alpha=3.0,
              r1=200e-3, r2=-200e-3, th=4e-3, z=5e-3, ap=11e-3)


def _flat():
    return {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _surf(r, gb, ga):
    d = _flat()
    d['radius'], d['glass_before'], d['glass_after'] = r, gb, ga
    return d


def _singlet(s, glass='N-BK7'):
    return {'name': 'v1_singlet', 'aperture_diameter': s['ap'],
            'surfaces': [_surf(s['r1'], 'air', glass),
                         _surf(s['r2'], glass, 'air'), _flat()],
            'thicknesses': [s['th'], s['z']]}


def _field(s, cx=0.0, cy=0.0):
    n, dx, w, rc, alpha = s['n'], s['dx'], s['w'], s['rc'], s['alpha']
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    sg = 1.0 if rc > 0 else -1.0
    rho = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + rc * rc)
    Wc = sg * (rho - abs(rc))
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    a = (alpha / _K0) * (r2 / (w * w)) ** 2
    return (np.exp(-r2 / (w * w))
            * np.exp(1j * _K0 * (Wc + a))).astype(np.complex128)


def _call(s, *, bound, guard, order, launch=True, cx=0.0, cy=0.0, **over):
    import warnings
    E = _field(s, cx=cx, cy=cy)
    presc = _singlet(s)
    old = (LT.REMAP_STATIONARY_PHASE_LAUNCH,
           LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
           LT.REMAP_INVERSE_SUPPORT_BOUND)
    LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(launch)
    LT.REMAP_STATIONARY_PHASE_FIT_GUARD = bool(guard)
    LT.REMAP_INVERSE_SUPPORT_BOUND = bool(bound)
    kw = dict(wavelength=_WL, amplitude_model='ray_density',
              preserve_input_phase='remap', remap_sampling='full',
              parallel_amp=False, on_undersample='silent',
              on_noncollimated='silent', on_aperture_beam='silent',
              ray_subsample=4, fit_radius_beam_factor=2.0, dx=s['dx'])
    if order is not None:
        kw['decentred_fit_poly_order'] = int(order)
    kw.update(over)
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            F = np.asarray(la.apply_real_lens_traced(
                E, prescription=presc, carrier=s['rc'], **kw))
    finally:
        (LT.REMAP_STATIONARY_PHASE_LAUNCH,
         LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
         LT.REMAP_INVERSE_SUPPORT_BOUND) = old
    return F, [str(w.message) for w in wl]


def _halo(F, s, factor=3.0, cx=0.0, cy=0.0):
    ax = (np.arange(s['n']) - s['n'] // 2) * s['dx']
    X, Y = np.meshgrid(ax, ax)
    R = np.hypot(X - cx, Y - cy)
    a = np.abs(F)
    pk = float(a.max())
    m = R > factor * s['w']
    return (float(a[m].max()) / pk) if (m.any() and pk > 0.0) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orders', default='6,8,10,12,14,16,18')
    ap.add_argument('--fixture', default='GHOST', choices=('GHOST', 'CLEAN'))
    ap.add_argument('--cx', type=float, default=0.0)
    ap.add_argument('--alpha', type=float, default=None)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    spec = dict(_GHOST if a.fixture == 'GHOST' else _CLEAN)
    if a.alpha is not None:
        spec['alpha'] = a.alpha
    orders = [None if o == 'default' else int(o)
              for o in a.orders.split(',') if o]
    res = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'fixture': a.fixture, 'spec': spec, 'cx': a.cx, 'rows': []}
    try:
        res['default_order'] = int(LT._DECENTRED_FIT_POLY_ORDER)
    except Exception as exc:                       # pragma: no cover
        res['default_order'] = f'ERR {exc}'
    for o in orders:
        row = {'order': o}
        for name, kw in (('off', dict(bound=False, guard=True)),
                         ('on', dict(bound=True, guard=True)),
                         ('guard_off', dict(bound=False, guard=False))):
            try:
                F, msgs = _call(spec, order=o, cx=a.cx, cy=0.0, **kw)
                row[name] = _halo(F, spec, cx=a.cx)
                row[name + '_repr'] = repr(row[name])
                row[name + '_msgs'] = len(msgs)
                row[name + '_peak'] = float(np.abs(F).max())
                row[name + '_pow'] = float((np.abs(F) ** 2).sum())
            except Exception as exc:
                row[name] = f'ERR {type(exc).__name__}: {exc}'
        if isinstance(row.get('off'), float) and isinstance(row.get('on'), float):
            row['ratio'] = row['off'] / row['on'] if row['on'] else None
        res['rows'].append(row)
        print(f"order={o} off={row.get('off')} on={row.get('on')} "
              f"guard_off={row.get('guard_off')} ratio={row.get('ratio')}",
              flush=True)
    if a.out:
        with open(a.out, 'w', encoding='cp1252') as fh:
            json.dump(res, fh, indent=1)
    print('lumenairy:', la.__file__, 'default_order:', res['default_order'])


if __name__ == '__main__':
    main()

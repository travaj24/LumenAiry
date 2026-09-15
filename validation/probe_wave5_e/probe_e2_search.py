"""WAVE5-E item E2 -- find a CHEAP, DURABLE default-order stimulus for the
C8 inverse-support bound.

VERIFY-WP-B14 F1 measured that the C8 defect class is still reachable at the
SHIPPED default ``decentred_fit_poly_order`` (16) -- ratio 301x at
``alpha=3.0, cx=1.5 mm, z=12 mm`` and 3164x at ``alpha=3.5, cx=1.35 mm,
z=13 mm`` on the 768^2 ``_GHOST`` geometry -- but declined to turn it into a
test, because a 27-cell neighbourhood reads ratio 1.00 in 23 of 27 cells: a
knife edge in ``(alpha, cx, z)``, which ``docs/TESTING_STANDARDS.md`` forbids
pinning as a single cell.  The sweep also costs ~14 min.

This probe searches for the two things a durable fail-before needs:

  * a **cheaper geometry** -- the smallest ``n`` at which the ratio is still
    >= 10x and STABLE under an ``n`` doubling, and
  * a **free parameter the trip is monotone in**, with >= 3 cells that trip,
    so the pin is a ladder and not a cell.

The candidate free parameter is ``fit_radius_beam_factor``: it sets the radius
(in input beam widths) of the traced-sample disc the exit map is FITTED on,
while the C8 support comes from the traced samples BEFORE any fit.  Shrinking
it therefore leaves the bounded halo alone and makes the unbounded one worse,
monotonically -- the shape a ladder needs.  ``alpha`` and ``cx`` are swept too,
as controls.

Subcommands:
  time   -- wall clock of one bound-off/bound-on pair per n
  sweep  -- the (n x free-parameter) ladder, default order
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

import numpy as np

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL

#: The C8 ``_GHOST`` geometry, verbatim from
#: ``tests/unit/test_niche_c8_inverse_support_bound.py``.
_GHOST = dict(n=768, dx=25e-6, w=1.5e-3, rc=-0.15, alpha=5.0,
              r1=150e-3, r2=-150e-3, th=4e-3, z=6e-3, ap=18e-3)


def _flat():
    return {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _surf(r, gb, ga):
    d = _flat()
    d['radius'], d['glass_before'], d['glass_after'] = r, gb, ga
    return d


def _singlet(s, glass='N-BK7'):
    return {'name': 'e2_singlet', 'aperture_diameter': s['ap'],
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


def _call(s, *, bound, order=None, cx=0.0, cy=0.0, frbf=2.0, **over):
    """One element call at the DEFAULT fit order unless ``order`` is given."""
    import warnings
    E = _field(s, cx=cx, cy=cy)
    presc = _singlet(s)
    old = (LT.REMAP_STATIONARY_PHASE_LAUNCH,
           LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
           LT.REMAP_INVERSE_SUPPORT_BOUND)
    LT.REMAP_STATIONARY_PHASE_LAUNCH = True
    LT.REMAP_STATIONARY_PHASE_FIT_GUARD = True
    LT.REMAP_INVERSE_SUPPORT_BOUND = bool(bound)
    kw = dict(wavelength=_WL, amplitude_model='ray_density',
              preserve_input_phase='remap', remap_sampling='full',
              parallel_amp=False, on_undersample='silent',
              on_noncollimated='silent', on_aperture_beam='silent',
              ray_subsample=4, fit_radius_beam_factor=float(frbf), dx=s['dx'])
    if order is not None:
        kw['decentred_fit_poly_order'] = int(order)
    kw.update(over)
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            F = np.asarray(la.apply_real_lens_traced(
                E, prescription=presc, carrier=s['rc'], **kw))
    finally:
        (LT.REMAP_STATIONARY_PHASE_LAUNCH,
         LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
         LT.REMAP_INVERSE_SUPPORT_BOUND) = old
    return F


def _halo(F, s, factor=3.0, cx=0.0, cy=0.0):
    ax = (np.arange(s['n']) - s['n'] // 2) * s['dx']
    X, Y = np.meshgrid(ax, ax)
    R = np.hypot(X - cx, Y - cy)
    a = np.abs(F)
    pk = float(a.max())
    m = R > factor * s['w']
    return (float(a[m].max()) / pk) if (m.any() and pk > 0.0) else 0.0


def _scaled(n, base=None):
    """``_GHOST`` re-sampled onto ``n`` cells over the SAME physical extent.

    ``n * dx`` is held at the 768-cell fixture's 19.2 mm, so every cell of the
    ladder looks at the same optic and the same halo annulus; only the sampling
    changes.  That is what makes "stable under an n doubling" a statement about
    the defect and not about a different geometry.
    """
    s = dict(base or _GHOST)
    s['n'] = int(n)
    s['dx'] = (_GHOST['n'] * _GHOST['dx']) / float(n)
    return s


_LADDER_FACTORS = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)


def _cell(n, alpha, cx, z, frbf, order, w=None, ap=None,
          factors=_LADDER_FACTORS):
    s = _scaled(n)
    s['alpha'] = float(alpha)
    s['z'] = float(z)
    if w is not None:
        s['w'] = float(w)
    if ap is not None:
        s['ap'] = float(ap)
    row = {'n': int(n), 'dx': s['dx'], 'alpha': float(alpha), 'cx': float(cx),
           'z': float(z), 'frbf': float(frbf), 'order': order, 'w': s['w'],
           'ap': s['ap']}
    t0 = time.perf_counter()
    try:
        Foff = _call(s, bound=False, order=order, cx=cx, frbf=frbf)
        Fon = _call(s, bound=True, order=order, cx=cx, frbf=frbf)
        h_off, h_on = _halo(Foff, s, cx=cx), _halo(Fon, s, cx=cx)
        row['off'], row['on'] = h_off, h_on
        row['ratio'] = (h_off / h_on) if h_on else None
        row['pw_off'] = float((np.abs(Foff) ** 2).sum())
        row['pw_on'] = float((np.abs(Fon) ** 2).sum())
        row['subtractive'] = bool(row['pw_on'] <= row['pw_off'] * (1 + 1e-12))
        loff = _halo_ladder(Foff, s, factors, cx=cx)
        lon = _halo_ladder(Fon, s, factors, cx=cx)
        lad = []
        for lo, ln in zip(loff, lon):
            supp = (ln['halo'] / lo['halo']) if lo['halo'] > 0 else None
            lad.append({'factor': lo['factor'], 'ncells': lo['ncells'],
                        'off': lo['halo'], 'on': ln['halo'],
                        'suppression': supp,
                        'ratio': (1.0 / supp) if supp else None})
        row['ladder'] = lad
        trips = [c for c in lad
                 if isinstance(c['ratio'], float) and c['ratio'] >= 10.0]
        row['n_trip_10x'] = len(trips)
        row['min_trip_ratio'] = min((c['ratio'] for c in trips), default=None)
        sup = [c['suppression'] for c in lad
               if c['suppression'] is not None]
        row['monotone'] = bool(
            len(sup) >= 3
            and all(sup[i + 1] <= sup[i] * (1 + 1e-9)
                    for i in range(len(sup) - 1)))
    except Exception as exc:                                # pragma: no cover
        row['err'] = '%s: %s' % (type(exc).__name__, exc)
    row['seconds'] = time.perf_counter() - t0
    return row


def _halo_ladder(F, s, factors, cx=0.0, cy=0.0):
    """``max|E| / peak`` in each annulus ``r > factor * w``, and the count of
    cells in it -- the ladder axis for the durable C8 pin.

    The C8 bound zeroes exit pixels with NO traced ray behind them, so the
    further out the annulus sits the larger the fraction of it that is outside
    the traced footprint and the more completely the bound must empty it.  That
    makes the annulus radius a MONOTONE driver of the suppression, where
    ``(alpha, cx, z)`` and ``fit_radius_beam_factor`` are not (measured: the
    frbf ladder in ``e2_sweep1_n256_*.json`` reads 1.0 / 1793.9 / 1.0 / 1.3).
    """
    ax = (np.arange(s['n']) - s['n'] // 2) * s['dx']
    X, Y = np.meshgrid(ax, ax)
    R = np.hypot(X - cx, Y - cy)
    a = np.abs(F)
    pk = float(a.max())
    out = []
    for f in factors:
        m = R > float(f) * s['w']
        out.append({'factor': float(f), 'ncells': int(m.sum()),
                    'halo': (float(a[m].max()) / pk)
                            if (m.any() and pk > 0.0) else 0.0,
                    'pow_frac': (float((a[m] ** 2).sum())
                                 / float((a ** 2).sum()))
                                if (m.any() and pk > 0.0) else 0.0})
    return out


def cmd_radius(a):
    """The radius ladder on one geometry, bound off vs on, at the DEFAULT order."""
    factors = [float(x) for x in a.factors.split(',')]
    rows = []
    for n in [int(x) for x in a.ns.split(',')]:
        s = _scaled(n)
        s['alpha'], s['z'] = a.alpha, a.z
        t0 = time.perf_counter()
        Foff = _call(s, bound=False, order=a.order, cx=a.cx, frbf=a.frbf)
        Fon = _call(s, bound=True, order=a.order, cx=a.cx, frbf=a.frbf)
        loff = _halo_ladder(Foff, s, factors, cx=a.cx)
        lon = _halo_ladder(Fon, s, factors, cx=a.cx)
        row = {'n': n, 'dx': s['dx'], 'alpha': a.alpha, 'cx': a.cx, 'z': a.z,
               'frbf': a.frbf, 'order': a.order,
               'seconds': time.perf_counter() - t0, 'ladder': []}
        for lo, ln in zip(loff, lon):
            supp = (ln['halo'] / lo['halo']) if lo['halo'] > 0 else None
            row['ladder'].append(
                {'factor': lo['factor'], 'ncells': lo['ncells'],
                 'off': lo['halo'], 'on': ln['halo'],
                 'suppression': supp,
                 'ratio': (1.0 / supp) if supp else None,
                 'off_pow_frac': lo['pow_frac'],
                 'on_pow_frac': ln['pow_frac']})
        rows.append(row)
        print('n=%-5d a=%.2f cx=%.3f mm z=%.1f mm frbf=%.2f (%.1f s)'
              % (n, a.alpha, a.cx * 1e3, a.z * 1e3, a.frbf, row['seconds']),
              flush=True)
        for c in row['ladder']:
            print('    r>%.2f w  ncells=%-8d off=%.4e on=%.4e  supp=%s'
                  % (c['factor'], c['ncells'], c['off'], c['on'],
                     ('%.3e' % c['suppression'])
                     if c['suppression'] is not None else 'None'), flush=True)
    return {'rows': rows}


def cmd_time(a):
    rows = []
    for n in [int(x) for x in a.ns.split(',')]:
        r = _cell(n, 5.0, 0.0, _GHOST['z'], 2.0, None)
        rows.append(r)
        print('n=%-5d dx=%.3e  %.2f s  off=%.4e on=%.4e ratio=%s'
              % (r['n'], r['dx'], r['seconds'], r.get('off', -1),
                 r.get('on', -1), r.get('ratio')), flush=True)
    return {'rows': rows}


def _fmt_ratio(v):
    return ('%.1f' % v) if isinstance(v, float) else str(v)


def cmd_sweep(a):
    ns = [int(x) for x in a.ns.split(',')]
    alphas = [float(x) for x in a.alphas.split(',')]
    cxs = [float(x) for x in a.cxs.split(',')]
    zs = [float(x) for x in a.zs.split(',')]
    frbfs = [float(x) for x in a.frbfs.split(',')]
    aps = [None if x in ('', 'default') else float(x)
           for x in a.aps.split(',')]
    ws = [None if x in ('', 'default') else float(x) for x in a.ws.split(',')]
    rows = []
    t0 = time.perf_counter()
    for n, alpha, cx, z, frbf, ap, w in itertools.product(
            ns, alphas, cxs, zs, frbfs, aps, ws):
        r = _cell(n, alpha, cx, z, frbf, a.order, w=w, ap=ap)
        rows.append(r)
        print('n=%-5d a=%-5.2f cx=%.3f z=%.1f frbf=%.2f ap=%.1fmm w=%.2fmm -> '
              'off=%.3e on=%.3e ratio=%-10s trip10x=%s mono=%s (%.1f s)'
              % (n, alpha, cx * 1e3, z * 1e3, frbf, r['ap'] * 1e3,
                 r['w'] * 1e3, r.get('off', -1), r.get('on', -1),
                 _fmt_ratio(r.get('ratio')), r.get('n_trip_10x'),
                 r.get('monotone'), r['seconds']), flush=True)
    print('TOTAL %.1f s over %d cells' % (time.perf_counter() - t0, len(rows)),
          flush=True)
    return {'rows': rows}


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('time')
    p.add_argument('--ns', default='192,256,384,512,768')
    p.add_argument('--out', default=None)
    p.set_defaults(fn=cmd_time)
    p = sub.add_parser('radius')
    p.add_argument('--ns', default='256,512')
    p.add_argument('--alpha', type=float, default=5.0)
    p.add_argument('--cx', type=float, default=0.0)
    p.add_argument('--z', type=float, default=0.006)
    p.add_argument('--frbf', type=float, default=2.0)
    p.add_argument('--factors', default='2.0,2.5,3.0,3.5,4.0,4.5,5.0,5.5')
    p.add_argument('--order', default=None,
                   type=lambda v: None if v in ('', 'default') else int(v))
    p.add_argument('--out', default=None)
    p.set_defaults(fn=cmd_radius)
    p = sub.add_parser('sweep')
    p.add_argument('--ns', default='256')
    p.add_argument('--alphas', default='3.0,5.0')
    p.add_argument('--cxs', default='0.0')
    p.add_argument('--zs', default='0.006')
    p.add_argument('--frbfs', default='2.0')
    p.add_argument('--aps', default='default')
    p.add_argument('--ws', default='default')
    p.add_argument('--order', default=None,
                   type=lambda v: None if v in ('', 'default') else int(v))
    p.add_argument('--out', default=None)
    p.set_defaults(fn=cmd_sweep)
    a = ap.parse_args()
    res = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'platform': sys.platform,
           'default_order': int(LT._DECENTRED_FIT_POLY_ORDER),
           'argv': sys.argv[1:]}
    print('lumenairy.__file__ =', la.__file__,
          ' default order =', res['default_order'], flush=True)
    res.update(a.fn(a))
    if a.out:
        with open(a.out, 'w', encoding='cp1252') as fh:
            json.dump(res, fh, indent=1)
        print('wrote', a.out, flush=True)


if __name__ == '__main__':
    main()

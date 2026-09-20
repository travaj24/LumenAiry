"""VERIFY-WAVE5-E / E2: the C8 default-order ladder, re-measured independently.

Re-runs the selected cell at n = 256 and 512 and the order-10 ``_GHOST``
control, and then asks the two questions the item report's claim rests on:

  * is the ANNULUS RADIUS axis monotone as a property of the mechanism, or
    only over the seven rungs the pin happens to choose?  A 25-rung radial
    sweep (2.0 .. 8.0 beam widths) answers it.
  * does the trip counter behave at a rung the bound empties COMPLETELY?
    (a rung whose 'on' amplitude is exactly 0 gives suppression 0.0)

Nothing here states a fit order except the control, which states 10.
"""
import json
import sys
import warnings

import numpy as np

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL

_GHOST = dict(n=768, dx=25e-6, w=1.5e-3, rc=-0.15, alpha=5.0,
              r1=150e-3, r2=-150e-3, th=4e-3, z=6e-3, ap=18e-3)
_CELL = dict(alpha=3.0, cx=1.25e-3, z=12e-3, frbf=2.0)
_FINE = tuple(np.round(np.arange(2.0, 8.01, 0.25), 3))


def _flat():
    return {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _surf(r, gb, ga):
    d = _flat()
    d['radius'], d['glass_before'], d['glass_after'] = r, gb, ga
    return d


def _singlet(s, glass='N-BK7'):
    return {'name': 'v_wave5e_c8', 'aperture_diameter': s['ap'],
            'surfaces': [_surf(s['r1'], 'air', glass),
                         _surf(s['r2'], glass, 'air'), _flat()],
            'thicknesses': [s['th'], s['z']]}


def _field(s, cx):
    n, dx, w, rc, alpha = s['n'], s['dx'], s['w'], s['rc'], s['alpha']
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    sg = 1.0 if rc > 0 else -1.0
    rho = np.sqrt((X - cx) ** 2 + Y ** 2 + rc * rc)
    Wc = sg * (rho - abs(rc))
    r2 = (X - cx) ** 2 + Y ** 2
    a = (alpha / _K0) * (r2 / (w * w)) ** 2
    return (np.exp(-r2 / (w * w))
            * np.exp(1j * _K0 * (Wc + a))).astype(np.complex128)


def _call(s, *, bound, cx, frbf, order=None):
    E = _field(s, cx)
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
              ray_subsample=4, fit_radius_beam_factor=float(frbf),
              dx=s['dx'])
    if order is not None:
        kw['decentred_fit_poly_order'] = int(order)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return np.asarray(la.apply_real_lens_traced(
                E, prescription=presc, carrier=s['rc'], **kw))
    finally:
        (LT.REMAP_STATIONARY_PHASE_LAUNCH,
         LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
         LT.REMAP_INVERSE_SUPPORT_BOUND) = old


def _ladder(Foff, Fon, s, cx, factors):
    ax = (np.arange(s['n']) - s['n'] // 2) * s['dx']
    X, Y = np.meshgrid(ax, ax)
    R = np.hypot(X - cx, Y)
    aoff, aon = np.abs(Foff), np.abs(Fon)
    poff = float(aoff.max()) or 1.0
    pon = float(aon.max()) or 1.0
    rows = []
    for f in factors:
        m = R > f * s['w']
        if not m.any():
            continue
        o = float(aoff[m].max()) / poff
        b = float(aon[m].max()) / pon
        rows.append(dict(factor=float(f), off=o, on=b,
                         suppression=(b / o) if o > 0 else 1.0,
                         ratio=(o / b) if b > 0 else float('inf'),
                         on_is_exactly_zero=bool(b == 0.0),
                         n_pixels=int(m.sum())))
    return rows


def _cell(alpha, cx, z, frbf, n, order=None):
    s = dict(_GHOST)
    s['n'] = int(n)
    s['dx'] = (_GHOST['n'] * _GHOST['dx']) / float(n)
    s['alpha'], s['z'] = alpha, z
    Foff = _call(s, bound=False, cx=cx, frbf=frbf, order=order)
    Fon = _call(s, bound=True, cx=cx, frbf=frbf, order=order)
    bitident = bool(np.array_equal(Foff.view(np.float64),
                                   Fon.view(np.float64)))
    p_off = float((np.abs(Foff) ** 2).sum())
    p_on = float((np.abs(Fon) ** 2).sum())
    return s, Foff, Fon, dict(bit_identical=bitident,
                              power_off=p_off, power_on=p_on,
                              power_ratio=(p_on / p_off) if p_off else 1.0)


def main():
    out = dict(lumenairy_file=la.__file__, version=la.__version__,
               python=sys.version.split()[0], platform=sys.platform,
               numpy=np.__version__,
               shipped_order=int(LT._DECENTRED_FIT_POLY_ORDER),
               cell=_CELL, rungs=[])
    print('lumenairy.__file__ =', la.__file__,
          'order =', LT._DECENTRED_FIT_POLY_ORDER, flush=True)
    for n in (256, 512):
        s, Foff, Fon, meta = _cell(_CELL['alpha'], _CELL['cx'], _CELL['z'],
                                   _CELL['frbf'], n)
        rows = _ladder(Foff, Fon, s, _CELL['cx'], _FINE)
        sup = [r['suppression'] for r in rows]
        rises = [i for i in range(len(sup) - 1)
                 if sup[i + 1] > sup[i] * (1 + 1e-9)]
        seven = _ladder(Foff, Fon, s, _CELL['cx'],
                        (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0))
        out['rungs'].append(dict(
            n=n, meta=meta, fine_rows=rows, seven_rows=seven,
            fine_monotone=not rises, fine_rises_at=rises,
            n_fine_rungs=len(rows),
            n_seven_trip=sum(1 for r in seven if r['ratio'] >= 10.0),
            ratio_at_3w=[r['ratio'] for r in seven if r['factor'] == 3.0],
            any_on_exactly_zero=bool(any(r['on_is_exactly_zero']
                                         for r in rows)),
        ))
        print('n=%d: %d fine rungs, monotone=%s, 7-rung trips=%d, '
              'ratio@3w=%s, power %.6g -> %.6g'
              % (n, len(rows), not rises,
                 out['rungs'][-1]['n_seven_trip'],
                 out['rungs'][-1]['ratio_at_3w'],
                 meta['power_off'], meta['power_on']), flush=True)

    # the order-10 _GHOST control
    s, Coff, Con, cmeta = _cell(_GHOST['alpha'], 0.0, _GHOST['z'], 2.0,
                                _GHOST['n'], order=10)
    crows = _ladder(Coff, Con, s, 0.0, (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0))
    cbest = max(r['ratio'] for r in crows)
    out['order10_control'] = dict(rows=crows, best_ratio=cbest, meta=cmeta)
    print('order-10 _GHOST control best ratio = %.4g (%s -> %s)'
          % (cbest, crows[0]['off'], crows[0]['on']), flush=True)

    # an INERT cell, to check the "exactly 1.0" claim below the bar
    s2, Ioff, Ion, imeta = _cell(3.0, 1.40e-3, 12e-3, 2.0, 256)
    irows = _ladder(Ioff, Ion, s2, 1.40e-3, (2.0, 3.0, 4.0, 5.0))
    out['inert_cell'] = dict(cell=dict(alpha=3.0, cx=1.40e-3, z=12e-3,
                                       frbf=2.0, n=256),
                             meta=imeta, rows=irows,
                             all_ratios_exactly_one=bool(
                                 all(r['ratio'] == 1.0 for r in irows)))
    print('inert cell: bit_identical=%s ratios=%s'
          % (imeta['bit_identical'], [r['ratio'] for r in irows]), flush=True)
    print(json.dumps(out, indent=1))


main()

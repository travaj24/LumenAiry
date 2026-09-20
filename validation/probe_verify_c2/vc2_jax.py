"""VERIFY-WP-C2 item 6 -- CPU/JAX parity, and the question the C2 report
does not ask: does the JAX tracer apply the SAME domain clamp?

Two parts:

1. **Parity** under all four CPU ``(renormalize, sphere_normal)`` settings
   plus the library default, on this verifier's own prescriptions.
2. **The clamp.**  ``_refract_jax`` builds the pure-spherical normal as
   ``(x, y, z - R) / R`` from the intersection's own ``z`` and applies NO
   ``h**2/R**2 < 0.9999`` gate -- the NumPy side applies that gate on BOTH
   routes.  If that reading is right the two backends' vignetting bands
   differ by the whole outer ``h > 0.99995 |R|`` annulus, not by one ULP,
   and that difference is PRE-EXISTING rather than something WP-C2 moved.
   Measured here on a ball lens and a hemisphere, whose clear aperture
   actually reaches there, with the CPU arms taken BOTH ways.

Usage: ``LUMENAIRY_ROOT=<root> python vc2_jax.py OUT.json``
"""
import json
import math
import os
import sys

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import numpy as np                                            # noqa: E402
import lumenairy as la                                        # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

from lumenairy.raytrace.surface import RayBundle, Surface     # noqa: E402
from lumenairy.raytrace.trace import trace                    # noqa: E402

WL = 587.5618e-9
CLAMP = math.sqrt(0.9999)


def _pres(name, rows, thick, aperture):
    """A prescription dict, the form ``trace_jax`` consumes."""
    return dict(name=name, aperture_diameter=aperture,
                surfaces=[dict(radius=r, conic=0.0, aspheric_coeffs=None,
                               radius_y=None, conic_y=None,
                               aspheric_coeffs_y=None,
                               glass_before=gb, glass_after=ga)
                          for (r, gb, ga) in rows],
                thicknesses=list(thick))


PRES = {
    'doublet3': _pres('doublet3',
                      [(0.0517, 'air', 'N-BK7'),
                       (-0.0345, 'N-BK7', 'N-SF5'),
                       (-0.1200, 'N-SF5', 'air')],
                      [0.0090, 0.0025], 0.025),
    'singlet2': _pres('singlet2',
                      [(0.0200, 'air', 'N-BK7'), (-0.0200, 'N-BK7', 'air')],
                      [0.0060], 0.020),
    'planoconvex': _pres('planoconvex',
                         [(0.0515, 'air', 'N-BK7'),
                          (float('inf'), 'N-BK7', 'air')],
                         [0.0041], 0.024),
    'ball': _pres('ball',
                  [(0.0125, 'air', 'N-BK7'), (-0.0125, 'N-BK7', 'air')],
                  [0.0250], 0.025),
    'hemisphere': _pres('hemisphere',
                        [(0.0125, 'air', 'N-BK7'),
                         (float('inf'), 'N-BK7', 'air')],
                        [0.0125], 0.025),
}


def _rays(n, hmax, tilt=0.0, lo=0.0):
    rng = np.random.default_rng(20260920)
    u = rng.uniform(0, 1, n)
    r = hmax * ((lo + (1.0 - lo) * u) if lo > 0 else np.sqrt(u))
    th = rng.uniform(0, 2 * np.pi, n)
    x = r * np.cos(th)
    y = r * np.sin(th)
    L = np.full(n, math.sin(math.radians(tilt)))
    M = np.zeros(n)
    Nd = np.sqrt(np.maximum(1.0 - L ** 2, 0.0))
    return x, y, L, M, Nd


def main(out_path):
    res = dict(meta=dict(python=sys.version.split()[0],
                         numpy=np.__version__, lumenairy=la.__version__,
                         file=la.__file__))
    try:
        import jax
        from lumenairy.raytrace.jax_trace import trace_jax
        jax.config.update('jax_enable_x64', True)
        res['meta']['jax'] = jax.__version__
    except Exception as e:                                  # pragma: no cover
        res['jax_unavailable'] = str(e)
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(res, fh, indent=1, sort_keys=True)
        print('JAX unavailable:', e)
        return

    import inspect
    sig_cpu = set(inspect.signature(trace).parameters)
    sig_jax = set(inspect.signature(trace_jax).parameters)
    res['jax_has_sphere_normal'] = 'sphere_normal' in sig_jax
    res['jax_has_renormalize'] = 'renormalize' in sig_jax
    res['cpu_has_both'] = {'sphere_normal', 'renormalize'} <= sig_cpu

    from lumenairy.raytrace.jax_trace import make_jax_ray_state
    from lumenairy.raytrace import surfaces_from_prescription as _sfp

    combos = [('surface', 'generic'), ('surface', 'analytic'),
              ('exit', 'generic'), ('exit', 'analytic'), ('DEFAULT', None)]
    parity = {}
    for name in ('doublet3', 'singlet2', 'planoconvex'):
        pres = PRES[name]
        surfs = _sfp(pres)
        for tilt in (0.0, 4.0):
            x, y, L, M, Nd = _rays(3000, 0.0090, tilt=tilt)
            z = np.zeros(len(x))
            st = make_jax_ray_state(x=x, y=y, z=z, L=L, M=M, N=Nd)
            out = trace_jax(st, pres, WL)
            jx, jy = np.asarray(out.x), np.asarray(out.y)
            jo, ja = np.asarray(out.opd), np.asarray(out.alive, dtype=bool)
            for rn, sn in combos:
                kw = {} if sn is None else dict(renormalize=rn,
                                                sphere_normal=sn)
                rb = RayBundle(x=x.copy(), y=y.copy(), z=z.copy(),
                               L=L.copy(), M=M.copy(), N=Nd.copy(),
                               wavelength=WL,
                               alive=np.ones(len(x), dtype=bool),
                               opd=np.zeros(len(x)))
                ir = trace(rb, surfs, WL, output_filter='last',
                           **kw).image_rays
                m = ja & np.asarray(ir.alive, dtype=bool)
                key = f'{name}@{tilt:g}deg/{rn}/{sn}'
                parity[key] = dict(
                    dx=float(np.max(np.abs(np.asarray(ir.x)[m] - jx[m])))
                    if m.any() else 0.0,
                    dy=float(np.max(np.abs(np.asarray(ir.y)[m] - jy[m])))
                    if m.any() else 0.0,
                    dopd=float(np.max(np.abs(np.asarray(ir.opd)[m] - jo[m])))
                    if m.any() else 0.0,
                    alive_equal=bool(np.array_equal(
                        np.asarray(ir.alive, dtype=bool), ja)),
                    n_compared=int(m.sum()))
    res['parity'] = parity
    res['parity_worst_dx'] = max(v['dx'] for v in parity.values())
    res['parity_worst_dopd'] = max(v['dopd'] for v in parity.values())
    res['parity_alive_equal_everywhere'] = all(v['alive_equal']
                                               for v in parity.values())
    by_combo = {}
    for k, v in parity.items():
        c = k.split('/', 1)[1]
        by_combo.setdefault(c, []).append(v['dx'])
    res['parity_worst_dx_by_cpu_setting'] = {k: max(v)
                                             for k, v in by_combo.items()}

    # ---- the clamp ------------------------------------------------
    clamp = {}
    for name in ('ball', 'hemisphere'):
        pres = PRES[name]
        surfs = _sfp(pres)
        x, y, L, M, Nd = _rays(40000, 0.0125 * 0.999999, lo=0.9990)
        z = np.zeros(len(x))
        h = np.sqrt(x ** 2 + y ** 2) / 0.0125
        past = h > CLAMP
        st = make_jax_ray_state(x=x, y=y, z=z, L=L, M=M, N=Nd)
        out = trace_jax(st, pres, WL)
        ja = np.asarray(out.alive, dtype=bool)
        row = dict(n_rays=int(len(h)),
                   n_past_clamp=int(past.sum()),
                   jax_alive_past_clamp=int((ja & past).sum()),
                   jax_alive_total=int(ja.sum()))
        for rn, sn in (('surface', 'generic'), ('exit', 'analytic')):
            rb = RayBundle(x=x.copy(), y=y.copy(), z=z.copy(), L=L.copy(),
                           M=M.copy(), N=Nd.copy(), wavelength=WL,
                           alive=np.ones(len(x), dtype=bool),
                           opd=np.zeros(len(x)))
            ir = trace(rb, surfs, WL, output_filter='last',
                       renormalize=rn, sphere_normal=sn).image_rays
            a = np.asarray(ir.alive, dtype=bool)
            ec = np.asarray(ir.error_code)
            row[f'cpu_{rn}_{sn}'] = dict(
                alive_past_clamp=int((a & past).sum()),
                alive_total=int(a.sum()),
                ray_nan_past_clamp=int(((ec == 4) & past).sum()),
                codes={int(k): int(c) for k, c in
                       zip(*np.unique(ec, return_counts=True))})
            row[f'alive_disagreements_with_jax_{rn}_{sn}'] = int(
                np.sum(a != ja))
        clamp[name] = row
    res['clamp'] = clamp
    res['jax_keeps_rays_the_cpu_clamp_kills'] = {
        k: v['jax_alive_past_clamp'] > v['cpu_exit_analytic'][
            'alive_past_clamp'] for k, v in clamp.items()}

    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print('jax has sphere_normal:', res['jax_has_sphere_normal'],
          '| renormalize:', res['jax_has_renormalize'])
    print('parity worst dx / dopd:', res['parity_worst_dx'],
          res['parity_worst_dopd'],
          '| alive equal everywhere:',
          res['parity_alive_equal_everywhere'])
    print('per CPU setting:',
          json.dumps(res['parity_worst_dx_by_cpu_setting'], sort_keys=True))
    for k, v in clamp.items():
        print(k, 'rays past the clamp:', v['n_past_clamp'],
              '| JAX keeps', v['jax_alive_past_clamp'],
              '| CPU(exit/analytic) keeps',
              v['cpu_exit_analytic']['alive_past_clamp'],
              '| CPU(surface/generic) keeps',
              v['cpu_surface_generic']['alive_past_clamp'],
              '| alive disagreements vs JAX:',
              v['alive_disagreements_with_jax_exit_analytic'])


if __name__ == '__main__':
    main(sys.argv[1])

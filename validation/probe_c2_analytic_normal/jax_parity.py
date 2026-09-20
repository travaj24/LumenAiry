"""WP-C2 item 4 -- CPU / JAX trace parity, re-derived under both defaults.

The brief asks whether the JAX tracer "has the same switches -- it must
take the same defaults or the CPU/JAX parity pins move".  It does NOT
have them, and the reason matters:

* ``jax_trace._refract_jax`` has ALWAYS used a closed-form sphere normal
  -- ``(x, y, z - R) / R``, taken at the intersection point the JAX
  Newton returned -- for any surface with a finite radius and no conic
  or aspheric term.  There is no ``sphere_normal`` switch to flip.
* it has no per-surface renormalisation to hoist either: the shared
  ``refract_snell`` core returns a unit vector from a unit normal, and
  the JAX body never rescales.  There is no ``renormalize`` switch.

So the CPU default flip moves the CPU tracer TOWARD the JAX one rather
than away from it, and the question the pins ask is quantitative: does
CPU / JAX parity get better, worse, or stay put?  This probe measures
it, per field, under all four CPU settings, on several prescriptions.

Note the two closed forms are NOT the same arithmetic: NumPy computes
``nz = sqrt(1 - h^2/R^2)`` by substituting the near-branch sag, JAX
computes ``(z - R)/R`` from the intersection's own ``z``.  They agree to
rounding, and that is what is measured here.

Usage:  LUMENAIRY_ROOT=<root> python jax_parity.py <out.json>
"""
import json
import math
import os
import sys

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)

import lumenairy as la  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__

import jax  # noqa: E402

jax.config.update('jax_enable_x64', True)

from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax  # noqa: E402
from lumenairy.raytrace.trace import _make_bundle, trace  # noqa: E402

WL = 587.6e-9

PRESCRIPTIONS = [
    ('singlet 51.5/-80 N-BK7',
     lambda: la.make_singlet(51.5e-3, -80e-3, 4.1e-3, 'N-BK7',
                             aperture=12e-3)),
    ('singlet 20/-20 N-BK7',
     lambda: la.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7',
                             aperture=10e-3)),
    ('doublet 51.7/-34.5/-120',
     lambda: la.make_doublet(R1=51.7e-3, R2=-34.5e-3, R3=-120.0e-3,
                             d1=8e-3, d2=3e-3, glass1='N-BK7',
                             glass2='N-SF5', aperture=25.4e-3)),
    ('plano-convex 51.5/inf',
     lambda: la.make_singlet(51.5e-3, np.inf, 4.1e-3, 'N-BK7',
                             aperture=12.0e-3)),
]

ARMS = {
    'generic_surface': dict(sphere_normal='generic', renormalize='surface'),
    'analytic_surface': dict(sphere_normal='analytic',
                             renormalize='surface'),
    'generic_exit': dict(sphere_normal='generic', renormalize='exit'),
    'analytic_exit': dict(sphere_normal='analytic', renormalize='exit'),
    'library_default': {},
}


def main():
    out_path = sys.argv[1]
    results = {}
    n = 33
    for label, build in PRESCRIPTIONS:
        pres = build()
        S = la.surfaces_from_prescription(pres)
        semi = min(float(getattr(s, 'semi_diameter', np.inf) or np.inf)
                   for s in S)
        rad = 0.85 * (semi if np.isfinite(semi) else 6e-3)
        for tilt in (0.0, 4.0):
            x = np.linspace(-rad, rad, n)
            z = np.zeros(n)
            L = np.full(n, math.sin(math.radians(tilt)))
            st = make_jax_ray_state(
                x=x, y=z, z=z, L=L, M=z,
                N=np.sqrt(np.maximum(1.0 - L ** 2, 0.0)))
            out = trace_jax(st, pres, WL)
            jx = np.asarray(out.x)
            jy = np.asarray(out.y)
            jopd = np.asarray(out.opd)
            jalive = np.asarray(out.alive)
            key = f'{label} @ {tilt:.0f} deg'
            entry = {}
            for arm, kw in ARMS.items():
                b = _make_bundle(x, z, L, z.copy(), WL)
                img = trace(b, S, WL, output_filter='last', **kw).image_rays
                m = np.asarray(img.alive) & jalive
                entry[arm] = dict(
                    alive_equal=bool(np.array_equal(
                        np.asarray(img.alive), jalive)),
                    n_alive=int(m.sum()),
                    dx=float(np.max(np.abs(jx[m] - img.x[m])))
                    if m.any() else 0.0,
                    dy=float(np.max(np.abs(jy[m] - img.y[m])))
                    if m.any() else 0.0,
                    dopd=float(np.max(np.abs(jopd[m] - img.opd[m])))
                    if m.any() else 0.0,
                )
            results[key] = entry
            g = entry['generic_surface']
            a = entry['analytic_surface']
            d = entry['library_default']
            print('%-32s  generic dx %.3e opd %.3e | analytic dx %.3e '
                  'opd %.3e | default dx %.3e opd %.3e | alive eq %s'
                  % (key, g['dx'], g['dopd'], a['dx'], a['dopd'],
                     d['dx'], d['dopd'],
                     all(entry[k]['alive_equal'] for k in ARMS)),
                  flush=True)
    summary = {}
    for arm in ARMS:
        summary[arm] = dict(
            worst_dx=max(v[arm]['dx'] for v in results.values()),
            worst_dy=max(v[arm]['dy'] for v in results.values()),
            worst_dopd=max(v[arm]['dopd'] for v in results.values()),
            alive_equal_everywhere=all(v[arm]['alive_equal']
                                       for v in results.values()),
        )
        print(arm, summary[arm])
    summary['default_matches_analytic'] = all(
        v['library_default'] == v['analytic_surface']
        for v in results.values())
    print('library default == analytic/surface arm:',
          summary['default_matches_analytic'])
    meta = dict(python=sys.version, numpy=np.__version__,
                jax=jax.__version__, lumenairy=la.__version__,
                platform=sys.platform)
    with open(out_path, 'w') as fh:
        json.dump(dict(meta=meta, summary=summary, results=results),
                  fh, indent=1)
    print('wrote', out_path)


if __name__ == '__main__':
    main()

"""Task B -- re-measure the blow-up's MECHANISM on the VERIFY fixtures.

Four arms, three of them re-measurements of WP-B7c's own controls and one a
control WP-B7c did not run:

1. **the sub-pixel area statistic**, derived INDEPENDENTLY.  For a
   rotationally-symmetric collimated launch the mapped-area ratio of a launch
   triangle at radius ``r`` is ``|y dy/dr / r|`` where ``y(r)`` is the
   geometric landing height -- computed here from the oracle's own exact
   meridional trace, not from the library -- so the mapped area in pixels is
   ``ratio * 0.5 h^2 / dx^2`` with ``h = ray_subsample * dx``.  Reported as a
   distribution over the launch lattice, beside the library's own
   ``n_branch_max`` and ``n_triangles_degenerate``.
2. **the ``ray_subsample`` inversion** -- 4 / 2 / 1 at a blow-up plane and at a
   healthy one, with the fitted power law in the launch pitch.
3. **the three rejected hypotheses**, each on its own axis: the fold-member
   ordering (``caustic_band`` 'plain' vs 'ludwig'), the Jacobian clip
   (``min_area_ratio`` 1e-8 .. 1e-3) and the branch count on one pixel.
4. **the pixel-refinement control WP-B7c did not run**: hold the LAUNCH
   lattice and the physical window fixed and refine the OUTPUT pixel
   (``N -> fN``, ``dx -> dx/f``, ``ray_subsample -> f * ray_subsample``).  A
   point-sampled area quadrature whose ring has collapsed onto a handful of
   pixels must IMPROVE when the same ring is spread over f^2 times as many
   pixels; a reconstruction wrong for a physical reason must not.

Usage: python probe2_mech.py <FIXTURE> <z_blowup_um> <z_healthy_um> <out.json>
"""
# ruff: noqa: E402, I001  (sys.path is set up between the imports)
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'oracles'))
import caustic_fold_truth as cft                          # noqa: E402
import fixtures as FX                                     # noqa: E402
import oracle as OR                                       # noqa: E402

import lumenairy                                          # noqa: E402
from lumenairy.elements._lens_traced_multibranch import (  # noqa: E402
    apply_real_lens_traced_multibranch)


def mb(fx, E, z, dx=None, **kw):
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            Eo, d = apply_real_lens_traced_multibranch(
                E, prescription=fx['prescription'],
                wavelength=fx['wavelength'],
                dx=(dx if dx is not None else fx['dx']),
                output_plane_distance=z, return_diagnostics=True, **kw)
    except RuntimeError as e:
        return None, {'bracket': None, 'error': str(e)[:160],
                      'n_branch_max': None, 'n_triangles': None,
                      'n_triangles_degenerate': None,
                      'power_ratio': None, 'power_ratio_triangles': None,
                      'launched_power': None, 'n_warn': None}
    vals = [float(v) for v in (d.get('power_ratio'),
                               d.get('power_ratio_triangles'))
            if v is not None and np.isfinite(v)]
    return Eo, {'bracket': (min(vals) if vals else None),
                'power_ratio': d.get('power_ratio'),
                'power_ratio_triangles': d.get('power_ratio_triangles'),
                'n_branch_max': d.get('n_branch_max'),
                'n_triangles': d.get('n_triangles'),
                'n_triangles_degenerate': d.get('n_triangles_degenerate'),
                'launched_power': d.get('launched_power'),
                'n_warn': len(rec)}


def area_stats(fx, z, ray_subsample=2, n=4000, dx=None):
    """Mapped-area statistics of the launch lattice, from the ORACLE's trace.

    ``ratio(r) = |y dy/dr / r|`` is the area Jacobian of the axially symmetric
    map ``r -> y(r)``; the launch triangle area is ``0.5 h^2``, so its mapped
    area in pixels is ``ratio * 0.5 h^2 / dx^2``.  Lattice nodes are weighted
    by their multiplicity in a square lattice, i.e. by ``r``.
    """
    dx = fx['dx'] if dx is None else dx
    job = OR.job_for(fx, z, n_fan=n)
    sf = cft._prep_surfaces(job['surfaces'])
    aper = 0.5 * float(fx['prescription']['aperture_diameter']) * 0.98
    zv = sum(s['t'] for s in sf[:-1])
    r = np.linspace(aper / n, aper, n)
    y = np.full(n, np.nan)
    for i, h in enumerate(r):
        res = cft.trace_ray(float(h), sf, None)
        if res is None:
            continue
        p, d, _ = res
        t = (zv + z - p[0]) / d[0]
        y[i] = p[1] + t * d[1]
    ok = np.isfinite(y)
    r, y = r[ok], y[ok]
    yp = np.gradient(y, r)
    ratio = np.abs(y * yp / r)
    h = ray_subsample * dx
    apx = ratio * 0.5 * h * h / (dx ** 2)
    w = r / r.sum()
    order = np.argsort(apx)
    cw = np.cumsum(w[order])

    def q(f):
        return float(apx[order][min(int(np.searchsorted(cw, f)),
                                    apx.size - 1)])

    return {'Apx_p05': q(0.05), 'Apx_p50': q(0.50), 'Apx_p95': q(0.95),
            'frac_subpixel_area': float(w[apx < 1.0].sum()),
            'ratio_p50': float(np.median(ratio)),
            'ratio_min': float(ratio.min()),
            'amp_J_max': float(1.0 / np.sqrt(max(ratio.min(), 1e-300))),
            'y_caustic_um': float(np.abs(y).min() * 1e6),
            'h_um': h * 1e6}


def main(argv):
    name = argv[1]
    z_bad = float(argv[2]) * 1e-6
    z_ok = float(argv[3]) * 1e-6
    out = argv[4]
    fx = FX.FIXTURES[name]
    E = FX.input_field(fx)
    res = {'lumenairy_file': lumenairy.__file__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'fixture': name, 'note': fx['note'], 'N': fx['N'], 'dx': fx['dx'],
           'z_blowup_um': z_bad * 1e6, 'z_healthy_um': z_ok * 1e6}

    res['area_stats'] = {'blowup': area_stats(fx, z_bad),
                         'healthy': area_stats(fx, z_ok)}

    sub = {}
    for tag, z in (('blowup', z_bad), ('healthy', z_ok)):
        rows = []
        for rs in (4, 2, 1):
            _, d = mb(fx, E, z, ray_subsample=rs)
            rows.append({'ray_subsample': rs, **d})
        good = [(g['ray_subsample'], g['bracket']) for g in rows
                if g.get('bracket')]
        slope = None
        if len(good) >= 2:
            lp = np.polyfit(np.log([g[0] for g in good]),
                            np.log([g[1] for g in good]), 1)
            slope = float(lp[0])
        sub[tag] = {'rows': rows, 'log_slope_vs_pitch': slope}
    res['ray_subsample'] = sub

    band = {}
    for tag, z in (('blowup', z_bad), ('healthy', z_ok)):
        row = {}
        for cb in ('ludwig', 'plain'):
            _, d = mb(fx, E, z, caustic_band=cb)
            row[cb] = d['bracket']
        row['rel_delta'] = ((abs(row['plain'] - row['ludwig'])
                             / max(abs(row['ludwig']), 1e-300))
                            if row['plain'] and row['ludwig'] else None)
        band[tag] = row
    res['caustic_band'] = band

    clip = {}
    for tag, z in (('blowup', z_bad), ('healthy', z_ok)):
        row = {}
        for mar in (1e-8, 1e-6, 1e-4, 1e-3):
            _, d = mb(fx, E, z, min_area_ratio=mar)
            row[f'{mar:g}'] = {'bracket': d['bracket'],
                               'n_degenerate': d['n_triangles_degenerate'],
                               'n_triangles': d['n_triangles']}
        row['identical_1e-08_vs_1e-06'] = (row['1e-08']['bracket']
                                           == row['1e-06']['bracket'])
        clip[tag] = row
    res['min_area_ratio'] = clip

    ref = {}
    for tag, z in (('blowup', z_bad), ('healthy', z_ok)):
        rows = []
        for f in (1, 2, 4):
            N2, dx2, rs2 = fx['N'] * f, fx['dx'] / f, 2 * f
            E2 = FX.gauss(N2, dx2, fx['w0'])
            _, d = mb(fx, E2, z, dx=dx2, ray_subsample=rs2)
            rows.append({'pixel_refine': f, 'N': N2, 'dx_um': dx2 * 1e6,
                         'ray_subsample': rs2,
                         'Apx_p50': area_stats(fx, z, ray_subsample=rs2,
                                               dx=dx2)['Apx_p50'], **d})
        ref[tag] = rows
    res['pixel_refinement'] = ref

    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == '__main__':
    main(sys.argv)

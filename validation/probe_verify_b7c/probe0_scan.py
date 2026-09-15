"""Cheap ladder scan -- NO oracle.  Locates, per fixture, the fold planes the
uniform completion accepts and the planes where the multibranch's own
bracketed gain leaves the band, so the expensive oracle scoring of
``probe1_band.py`` can be aimed.

Usage: python probe0_scan.py <FIXTURE> <z_lo_um> <z_hi_um> <n> [out.json]
"""
# ruff: noqa: E402, I001  (sys.path is set up between the imports)
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures as FX                                    # noqa: E402

import lumenairy                                         # noqa: E402
from lumenairy.elements._lens_traced_multibranch import (  # noqa: E402
    apply_real_lens_traced_multibranch)
from lumenairy.elements._lens_traced_uniform import (      # noqa: E402
    apply_real_lens_traced_uniform)


def row(fx, E, z, **kw):
    out = {'z_um': z * 1e6}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        _, md = apply_real_lens_traced_multibranch(
            E, prescription=fx['prescription'], wavelength=fx['wavelength'],
            dx=fx['dx'], output_plane_distance=z, return_diagnostics=True,
            **kw)
    vals = [float(v) for v in (md.get('power_ratio'),
                               md.get('power_ratio_triangles'))
            if v is not None and np.isfinite(v)]
    out['mb_power_ratio'] = md.get('power_ratio')
    out['mb_power_ratio_triangles'] = md.get('power_ratio_triangles')
    out['mb_bracket'] = (min(vals) if vals else None)
    out['n_branch_max'] = md.get('n_branch_max')
    out['mb_warnings'] = [str(w.message)[:60] for w in rec]
    try:
        with warnings.catch_warnings(record=True) as rec2:
            warnings.simplefilter('always')
            _, ud = apply_real_lens_traced_uniform(
                E, prescription=fx['prescription'],
                wavelength=fx['wavelength'], dx=fx['dx'],
                output_plane_distance=z, return_diagnostics=True, **kw)
        out['uni_error'] = None
        for k in ('fell_back', 'reason', 'zeta_extrapolation', 'fit_residual',
                  'zeta_band', 'r_c', 'kappa', 'power_ratio_decision',
                  'zeta_linear_range', 'l_airy', 'dark_fill_depth',
                  'zeta_curvature'):
            v = ud.get(k)
            out['uni_' + k] = (float(v) if isinstance(v, (int, float, np.floating))
                               and not isinstance(v, bool) else v)
        out['uni_warnings'] = [str(w.message)[:60] for w in rec2]
    except RuntimeError as e:
        out['uni_error'] = str(e)[:160]
    return out


def main(argv):
    name = argv[1]
    zs = np.linspace(float(argv[2]), float(argv[3]), int(argv[4])) * 1e-6
    fx = FX.FIXTURES[name]
    E = FX.input_field(fx)
    res = {'lumenairy_file': lumenairy.__file__,
           'lumenairy_version': lumenairy.__version__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'fixture': name, 'note': fx['note'], 'N': fx['N'],
           'dx': fx['dx'], 'w0': fx['w0'], 'rows': []}
    for z in zs:
        r = row(fx, E, float(z))
        res['rows'].append(r)
        print(f"{name} z={r['z_um']:9.2f} br={r['mb_bracket']} "
              f"nbr={r['n_branch_max']} "
              f"uni={r.get('uni_reason') or 'RAISE'} "
              f"zx={r.get('uni_zeta_extrapolation')}", flush=True)
    if len(argv) > 5:
        with open(argv[5], 'w') as fh:
            json.dump(res, fh, indent=1, default=str)
    return res


if __name__ == '__main__':
    main(sys.argv)

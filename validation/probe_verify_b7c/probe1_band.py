"""Task A -- re-derive the refusal band on the VERIFY fixtures.

For every plane requested: the multibranch's own bracketed gain reading, the
uniform completion (or the ``RuntimeError`` it now raises), the ray-to-wave
hand-off, and all three scored against the direct Rayleigh-Sommerfeld oracle
``validation/oracles/caustic_fold_truth.py``.  The classification the band is
derived from -- "the oracle accepts this completed field" vs "it is broken" --
can only be taken on a build that still RETURNS the field, so this probe is
run on the base tree (96cb2096) as well as on HEAD and the two are joined by
``z``.

Usage::

    python probe1_band.py <FIXTURE> <out.json> <z_um> [<z_um> ...]
    python probe1_band.py <FIXTURE> <out.json> --range lo hi n
"""
# ruff: noqa: E402, I001  (sys.path is set up between the imports)
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures as FX                                     # noqa: E402
import oracle as OR                                       # noqa: E402

import lumenairy                                          # noqa: E402
from lumenairy.elements._lens_traced_multibranch import (  # noqa: E402
    apply_real_lens_traced_multibranch)
from lumenairy.elements._lens_traced_uniform import (      # noqa: E402
    apply_real_lens_traced_uniform)


def _f(v):
    if v is None or isinstance(v, bool) or isinstance(v, str):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def one(fx, E, z, n_fan=6000, n_rho=2400):
    r = {'z_um': z * 1e6}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        E_mb, md = apply_real_lens_traced_multibranch(
            E, prescription=fx['prescription'], wavelength=fx['wavelength'],
            dx=fx['dx'], output_plane_distance=z, return_diagnostics=True)
    vals = [float(v) for v in (md.get('power_ratio'),
                               md.get('power_ratio_triangles'))
            if v is not None and np.isfinite(v)]
    r['mb_power_ratio'] = _f(md.get('power_ratio'))
    r['mb_power_ratio_triangles'] = _f(md.get('power_ratio_triangles'))
    r['mb_bracket'] = (min(vals) if vals else None)
    r['n_branch_max'] = md.get('n_branch_max')
    r['n_triangles_degenerate'] = md.get('n_triangles_degenerate')
    r['n_triangles'] = md.get('n_triangles')
    r['mb_warnings'] = len(rec)

    E_uni = None
    try:
        with warnings.catch_warnings(record=True) as rec2:
            warnings.simplefilter('always')
            E_uni, ud = apply_real_lens_traced_uniform(
                E, prescription=fx['prescription'],
                wavelength=fx['wavelength'], dx=fx['dx'],
                output_plane_distance=z, return_diagnostics=True)
        r['uni_error'] = None
        r['uni_warnings'] = len(rec2)
        for k in ('fell_back', 'reason', 'zeta_extrapolation', 'fit_residual',
                  'zeta_band', 'r_c', 'kappa', 'power_ratio_decision',
                  'zeta_linear_range', 'l_airy', 'dark_fill_depth',
                  'zeta_curvature', 'zeta_linear_resid', 'fit_halfwidth',
                  'multibranch_power_ratio_bracketed'):
            r['uni_' + k] = _f(ud.get(k))
    except RuntimeError as e:
        r['uni_error'] = str(e)[:400]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_wv = np.asarray(lumenairy.apply_real_lens_traced(
            E, prescription=fx['prescription'], wavelength=fx['wavelength'],
            dx=fx['dx'], output_plane_distance=z,
            amplitude_model='ray_density', caustic='wave', n_workers=1,
            on_undersample='silent'))
        E_rd = np.asarray(lumenairy.apply_real_lens_traced(
            E, prescription=fx['prescription'], wavelength=fx['wavelength'],
            dx=fx['dx'], output_plane_distance=z,
            amplitude_model='screen', caustic='wave', n_workers=1,
            on_undersample='silent'))

    E_or, rho, E_rho, ex = OR.oracle_field(fx, z, n_fan=n_fan, n_rho=n_rho)
    dx = fx['dx']
    p_or = OR.power(E_or, dx)
    r['oracle_power'] = p_or
    r['oracle_launched_power'] = float(ex['P_in'])
    r['oracle_energy_closure'] = float(
        2.0 * np.pi * np.trapezoid(np.abs(E_rho) ** 2 * rho, rho) / ex['P_in'])
    r['fid_mb'] = OR.fidelity(E_mb, E_or)
    r['pow_mb_over_oracle'] = OR.power(E_mb, dx) / p_or
    r['fid_wave'] = OR.fidelity(E_wv, E_or)
    r['pow_wave_over_oracle'] = OR.power(E_wv, dx) / p_or
    r['fid_screen'] = OR.fidelity(E_rd, E_or)
    r['pow_screen_over_oracle'] = OR.power(E_rd, dx) / p_or
    if E_uni is not None:
        r['fid_uni'] = OR.fidelity(E_uni, E_or)
        r['pow_uni_over_oracle'] = OR.power(E_uni, dx) / p_or
    else:
        r['fid_uni'] = None
        r['pow_uni_over_oracle'] = None
    return r


def main(argv):
    name, out = argv[1], argv[2]
    if argv[3] == '--range':
        zs = np.linspace(float(argv[4]), float(argv[5]), int(argv[6])) * 1e-6
    else:
        zs = np.array([float(v) for v in argv[3:]]) * 1e-6
    fx = FX.FIXTURES[name]
    E = FX.input_field(fx)
    res = {'lumenairy_file': lumenairy.__file__,
           'lumenairy_version': lumenairy.__version__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'fixture': name, 'note': fx['note'],
           'N': fx['N'], 'dx': fx['dx'], 'w0': fx['w0'],
           'index_control': {g: OR.index_control(g, fx['wavelength'])
                             for g in sorted({s['glass_after'] for s
                                              in fx['prescription']['surfaces']
                                              if s['glass_after'] != 'air'})},
           'rows': []}
    for z in zs:
        r = one(fx, E, float(z))
        res['rows'].append(r)
        print(f"{name} z={r['z_um']:9.2f} br={r['mb_bracket']!s:>22} "
              f"nbr={r['n_branch_max']!s:>5} fid_uni={r['fid_uni']!s:>8.8} "
              f"pow_uni={r['pow_uni_over_oracle']!s:>10.10} "
              f"fid_wv={r['fid_wave']:.4f} "
              f"zx={r.get('uni_zeta_extrapolation')!s:>10.10} "
              f"{r.get('uni_reason') or 'RAISE'}", flush=True)
        with open(out, 'w') as fh:
            json.dump(res, fh, indent=1, default=str)
    return res


if __name__ == '__main__':
    main(sys.argv)

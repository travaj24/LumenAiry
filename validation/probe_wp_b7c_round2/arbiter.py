"""WP-B7c round 2 -- the PIXEL-HALVING ARBITER's population.

For every plane requested this records, on whichever tree it is run:

* the branch sum's own bracketed gain reading (the round-1 tripwire), its
  ``n_branch_max`` and degeneracy census;
* the CONTINUITY ratio -- the deposited power of the render at ``dx`` over the
  same mapped triangles' deposited power at ``dx/2`` on the same window --
  together with the decision the shipped bar takes on it.  Measured through
  the library's own ``_multibranch_render(..., pixel_halving_arbiter=True)``
  where that exists, and by an EXTERNAL double call (two public branch-sum
  calls, ``dx`` and ``dx/2`` with ``ray_subsample`` doubled) where it does
  not -- so a tree without the arbiter still yields the reading and the two
  paths can be cross-checked against each other;
* the uniform completion (or the ``RuntimeError`` it raises), the ray-to-wave
  hand-off, and all of them scored against the direct Rayleigh-Sommerfeld
  oracle.

The classification the bar is derived from -- "does the oracle accept this
completed field" -- can only be read on a tree that still RETURNS the field,
so this probe is run on the audit base (96cb2096, which refuses nothing) as
well as on HEAD, and the two are joined on ``z`` by ``summarise.py``.

Usage::

    python arbiter.py <FIXTURE> <out.json> [--phi debye|exact] [--nofan N]
                      <z_um> [<z_um> ...]
    python arbiter.py <FIXTURE> <out.json> [...] --range <lo> <hi> <n>
"""
# ruff: noqa: E402, I001
from __future__ import annotations

import json
import os
import sys
import time
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

try:                                    # the arbiter, where the tree has it
    from lumenairy.elements._lens_traced_multibranch import (
        _multibranch_render)
except ImportError:                     # the audit base / round-1 tree
    _multibranch_render = None


def _f(v):
    if v is None or isinstance(v, (bool, str)):
        return v
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _bracket(d):
    vals = [float(v) for v in (d.get('power_ratio'),
                               d.get('power_ratio_triangles'))
            if v is not None and np.isfinite(v)]
    return min(vals) if vals else None


def continuity_external(fx, z, E=None):
    """The same ratio without any library support: two PUBLIC branch-sum
    calls, ``dx`` and ``dx/2`` with ``ray_subsample`` doubled, which holds the
    launch pitch ``ray_subsample * dx`` and the physical window fixed."""
    N, dx = fx['N'], fx['dx']
    out = []
    for n_r, dx_r, sub in ((N, dx, 2), (2 * N, 0.5 * dx, 4)):
        Ei = FX.gauss(n_r, dx_r, fx['w0'])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            Eo, _d = apply_real_lens_traced_multibranch(
                Ei, prescription=fx['prescription'],
                wavelength=fx['wavelength'], dx=dx_r,
                output_plane_distance=z, ray_subsample=sub,
                return_diagnostics=True)
        out.append(float(np.sum(np.abs(np.asarray(Eo)) ** 2)) * dx_r * dx_r)
    return (out[0] / out[1]) if out[1] > 0 else None


def one(fx, E, z, n_fan=6000, n_rho=2400, phi='debye', n_phi=128,
        external=True):
    r = {'z_um': z * 1e6}
    t0 = time.perf_counter()
    kw = dict(prescription=fx['prescription'], wavelength=fx['wavelength'],
              dx=fx['dx'], output_plane_distance=z, return_diagnostics=True)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        if _multibranch_render is not None:
            E_mb, md = _multibranch_render(E, pixel_halving_arbiter=True, **kw)
        else:
            E_mb, md = apply_real_lens_traced_multibranch(E, **kw)
    r['t_mb_with_arbiter'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        apply_real_lens_traced_multibranch(E, **kw)
    r['t_mb_plain'] = time.perf_counter() - t0

    r['mb_power_ratio'] = _f(md.get('power_ratio'))
    r['mb_power_ratio_triangles'] = _f(md.get('power_ratio_triangles'))
    r['mb_bracket'] = _bracket(md)
    r['n_branch_max'] = md.get('n_branch_max')
    r['n_triangles'] = md.get('n_triangles')
    r['n_triangles_degenerate'] = md.get('n_triangles_degenerate')
    r['mb_warnings'] = len(rec)
    r['continuity'] = _f(md.get('pixel_continuity'))
    r['continuity_decision'] = md.get('pixel_continuity_decision')
    r['grid_power'] = _f(md.get('grid_power'))
    r['launched_power'] = _f(md.get('launched_power'))
    if external:
        t0 = time.perf_counter()
        r['continuity_external'] = _f(continuity_external(fx, z))
        r['t_continuity_external'] = time.perf_counter() - t0

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
                  'multibranch_power_ratio_bracketed',
                  'multibranch_pixel_continuity', 'pixel_continuity_decision'):
            r['uni_' + k] = _f(ud.get(k))
    except RuntimeError as e:
        r['uni_error'] = str(e)[:500]
        r['uni_refused_on'] = ('pixel_continuity' if 'NOT CONVERGED' in str(e)
                               else 'power_ratio')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_wv = np.asarray(lumenairy.apply_real_lens_traced(
            E, prescription=fx['prescription'], wavelength=fx['wavelength'],
            dx=fx['dx'], output_plane_distance=z,
            amplitude_model='ray_density', caustic='wave', n_workers=1,
            on_undersample='silent'))

    t0 = time.perf_counter()
    E_or, rho, E_rho, ex, odet = OR.oracle_field(
        fx, z, n_fan=n_fan, n_rho=n_rho, phi=phi, n_phi=n_phi,
        return_detail=True)
    r['oracle_detail'] = odet
    r['t_oracle'] = time.perf_counter() - t0
    dx = fx['dx']
    p_or = OR.power(E_or, dx)
    r['oracle_phi'] = phi
    r['oracle_phi_safety'] = (n_phi if phi == 'exact' else None)
    r['oracle_n_fan'] = n_fan
    r['oracle_power'] = p_or
    r['oracle_launched_power'] = float(ex['P_in'])
    r['oracle_energy_closure'] = float(
        2.0 * np.pi * np.trapezoid(np.abs(E_rho) ** 2 * rho, rho) / ex['P_in'])
    r['oracle_rms_radius_um'] = OR.rms_radius(rho, E_rho) * 1e6
    r['fid_mb'] = OR.fidelity(E_mb, E_or)
    r['pow_mb_over_oracle'] = OR.power(E_mb, dx) / p_or
    r['fid_wave'] = OR.fidelity(E_wv, E_or)
    r['pow_wave_over_oracle'] = OR.power(E_wv, dx) / p_or
    if E_uni is not None:
        r['fid_uni'] = OR.fidelity(E_uni, E_or)
        r['pow_uni_over_oracle'] = OR.power(E_uni, dx) / p_or
    else:
        r['fid_uni'] = None
        r['pow_uni_over_oracle'] = None
    return r


def main(argv):
    name, out = argv[1], argv[2]
    rest = list(argv[3:])
    phi, n_phi, n_fan, n_rho, external = 'debye', 1.5, 6000, 2400, True
    while rest and rest[0].startswith('--') and rest[0] != '--range':
        flag = rest.pop(0)
        if flag == '--phi':
            phi = rest.pop(0)
        elif flag == '--nphi':
            n_phi = float(rest.pop(0))
        elif flag == '--nfan':
            n_fan = int(rest.pop(0))
        elif flag == '--nrho':
            n_rho = int(rest.pop(0))
        elif flag == '--noexternal':
            external = False
        else:
            raise SystemExit(f'unknown flag {flag}')
    if rest and rest[0] == '--range':
        zs = np.linspace(float(rest[1]), float(rest[2]), int(rest[3])) * 1e-6
    else:
        zs = np.array([float(v) for v in rest]) * 1e-6

    fx = FX.FIXTURES[name]
    E = FX.input_field(fx)
    res = {'lumenairy_file': lumenairy.__file__,
           'lumenairy_version': lumenairy.__version__,
           'has_arbiter': _multibranch_render is not None,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'fixture': name, 'note': fx['note'],
           'N': fx['N'], 'dx': fx['dx'], 'w0': fx['w0'],
           'index_control': {g: OR.index_control(g, fx['wavelength'])
                             for g in sorted({s['glass_after'] for s
                                              in fx['prescription']['surfaces']
                                              if s['glass_after'] != 'air'})},
           'rows': []}
    print(f"lumenairy {lumenairy.__file__}  arbiter="
          f"{res['has_arbiter']}  phi={phi}", flush=True)
    for z in zs:
        r = one(fx, E, float(z), n_fan=n_fan, n_rho=n_rho, phi=phi,
                n_phi=n_phi, external=external)
        res['rows'].append(r)
        print(f"{name:6s} z={r['z_um']:9.2f} br={r['mb_bracket']!s:>10.10} "
              f"C={r['continuity']!s:>8.8}/{r.get('continuity_external')!s:>8.8} "
              f"nbr={r['n_branch_max']!s:>4} "
              f"fid_uni={r['fid_uni']!s:>7.7} "
              f"pow_uni={r['pow_uni_over_oracle']!s:>8.8} "
              f"fid_wv={r['fid_wave']:.4f} "
              f"{r.get('uni_reason') or r.get('uni_refused_on') or ''}",
              flush=True)
        with open(out, 'w') as fh:
            json.dump(res, fh, indent=1, default=str)
    return res


if __name__ == '__main__':
    main(sys.argv)

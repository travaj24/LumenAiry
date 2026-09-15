"""Probe 1 -- plane scan of the uniform completion's diagnostics and energy.

Runs ``apply_real_lens_traced_uniform`` on a ladder of output planes and
records, per plane: the fold geometry the module resolved, the diagnostics it
reports, the multibranch power ratio it inherits, and TWO energy readings taken
outside the module (grid power of the completed field and of the plain
multibranch, both against the launched aperture power).

Usage:  python probe1_scan.py <fixture> <z0_um> <z1_um> <nz> [out.json]
"""
# ruff: noqa: E402, I001  (sys.path is set up between the imports)
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixtures import FIXTURES, input_field          # noqa: E402


def run_plane(la, fx, E_in, z):
    """One plane: uniform + multibranch, diagnostics and energies."""
    dx = fx['dx']
    out = {'z_um': z * 1e6}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        try:
            E_u, d = la.elements._lens_traced_uniform.\
                apply_real_lens_traced_uniform(
                    E_in, prescription=fx['prescription'],
                    wavelength=fx['wavelength'], dx=dx,
                    output_plane_distance=z, return_diagnostics=True)
        except Exception as exc:                      # noqa: BLE001
            out['error'] = f'{type(exc).__name__}: {exc}'
            out['warnings'] = [str(w.message)[:160] for w in rec]
            return out
    out['warnings'] = [str(w.message)[:200] for w in rec]
    out['n_warnings'] = len(rec)
    for key in ('fell_back', 'reason', 'r_c', 'kappa', 'fit_residual',
                'fit_halfwidth', 'zeta_band', 'zeta_extrapolation',
                'power_ratio', 'power_ratio_triangles', 'n_triangles',
                'n_triangles_finite', 'n_triangles_degenerate'):
        v = d.get(key)
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        if isinstance(v, float) and not np.isfinite(v):
            v = repr(v)
        out[key] = v
    nb = d.get('n_branch')
    out['n_branch_max'] = int(np.max(nb)) if nb is not None else None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        E_mb = np.asarray(
            la.elements._lens_traced_multibranch.
            apply_real_lens_traced_multibranch(
                E_in, prescription=fx['prescription'],
                wavelength=fx['wavelength'], dx=dx,
                output_plane_distance=z))
    p_ap = float(np.sum(np.abs(E_in) ** 2)) * dx * dx
    out['P_uniform_over_Pin'] = float(np.sum(np.abs(E_u) ** 2)) * dx * dx / p_ap
    out['P_mb_over_Pin'] = float(np.sum(np.abs(E_mb) ** 2)) * dx * dx / p_ap
    out['max_abs_uniform'] = float(np.max(np.abs(E_u)))
    out['max_abs_mb'] = float(np.max(np.abs(E_mb)))
    out['max_abs_in'] = float(np.max(np.abs(E_in)))
    return out


def main():
    import lumenairy as la
    fxname = sys.argv[1]
    z0, z1, nz = float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    dest = sys.argv[5] if len(sys.argv) > 5 else None
    fx = FIXTURES[fxname]
    E_in = input_field(fx)
    rows = []
    hdr = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'numpy': np.__version__, 'python': sys.version.split()[0],
           'fixture': fxname, 'note': fx['note'], 'N': fx['N'],
           'dx': fx['dx'], 'w0': fx['w0'], 'wavelength': fx['wavelength']}
    print(json.dumps(hdr))
    for z in np.linspace(z0, z1, nz) * 1e-6:
        r = run_plane(la, fx, E_in, float(z))
        rows.append(r)
        print(f"z={r['z_um']:9.2f}  fb={r.get('fell_back')!s:5s} "
              f"reason={str(r.get('reason'))[:28]:28s} "
              f"zx={r.get('zeta_extrapolation')!s:>10.10s} "
              f"fitres={r.get('fit_residual')!s:>8.8s} "
              f"Pmb={r.get('power_ratio')!s:>10.10s} "
              f"Puni/Pin={r.get('P_uniform_over_Pin', float('nan')):10.4g} "
              f"Pmbg/Pin={r.get('P_mb_over_Pin', float('nan')):10.4g} "
              f"nw={r.get('n_warnings')}", flush=True)
    if dest:
        with open(dest, 'w', encoding='cp1252') as fh:
            json.dump({'header': hdr, 'rows': rows}, fh, indent=1)
        print('wrote', dest)


if __name__ == '__main__':
    main()

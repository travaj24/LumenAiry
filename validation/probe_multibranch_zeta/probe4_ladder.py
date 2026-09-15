"""Probe 4 -- the scoring ladder: four members against the exact reference.

For each output plane of a fixture this records, against the lumenairy-free
direct Rayleigh-Sommerfeld oracle (``validation/oracles/caustic_fold_truth.py``,
the same oracle VERIFY-B7b used):

* ``fidelity`` -- |<a,b>| / (|a||b|), phase- and scale-invariant;
* ``P/P_oracle`` -- total grid power against the oracle's;
* ``P/P_in`` -- total grid power against the launched aperture power (the
  model-free energy-conservation reading: a lossless element cannot deliver
  more power to a plane than it launched);

for the four members ``uniform`` (the CFU completion), ``multibranch``,
``traced(amplitude_model='ray_density')`` and ``traced(caustic='wave')``, plus
every diagnostic the uniform layer reports and the multibranch power ratios it
inherits.

Usage:
  python probe4_ladder.py <fixture> <z_um,z_um,...> <out.json> [n_fan] [n_rho]
"""
# ruff: noqa: E402, I001  (sys.path is set up between the imports)
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fixtures import FIXTURES, input_field          # noqa: E402
import oracle as orc                                 # noqa: E402


def _clean(v):
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return repr(v)
    if isinstance(v, complex):
        return [v.real, v.imag]
    return v


def members(la, fx, E_in, z):
    """Every member at one plane, with the warnings each emitted."""
    presc, wl, dx = fx['prescription'], fx['wavelength'], fx['dx']
    out = {}
    common = dict(prescription=presc, wavelength=wl, dx=dx,
                  output_plane_distance=z)

    def _run(name, fn, **kw):
        t0 = time.time()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            try:
                r = fn(E_in, **kw)
            except Exception as exc:                  # noqa: BLE001
                out[name] = {'error': f'{type(exc).__name__}: {exc}'[:400],
                             'warnings': [str(w.message)[:150] for w in rec]}
                return None, None
        d = None
        if isinstance(r, tuple):
            r, d = r
        out[name] = {'seconds': round(time.time() - t0, 2),
                     'warnings': [str(w.message)[:150] for w in rec]}
        return np.asarray(r), d

    E_u, d_u = _run('uniform',
                    la.elements._lens_traced_uniform.
                    apply_real_lens_traced_uniform,
                    return_diagnostics=True, **common)
    E_mb, d_mb = _run('multibranch',
                      la.elements._lens_traced_multibranch.
                      apply_real_lens_traced_multibranch,
                      return_diagnostics=True, **common)
    # ``output_plane_distance`` is honoured only by multibranch / uniform /
    # wave, so the single-valued members are read through the ray-to-wave
    # hand-off, which is ONE recursion at the exit vertex plus one
    # band-limited ASM leg (``_lens_traced.py``, the ``_wave_caustic``
    # dispatch).  ``ray_density`` = the ray-density exit amplitude,
    # ``wave`` = the default 'screen' exit amplitude; both then ASM.
    E_rd, _ = _run('ray_density', la.apply_real_lens_traced,
                   amplitude_model='ray_density', caustic='wave', n_workers=1,
                   on_undersample='silent', **common)
    E_wv, _ = _run('wave', la.apply_real_lens_traced,
                   amplitude_model='screen', caustic='wave', n_workers=1,
                   on_undersample='silent', **common)
    return out, {'uniform': E_u, 'multibranch': E_mb, 'ray_density': E_rd,
                 'wave': E_wv}, d_u, d_mb


def main():
    import lumenairy as la
    fxname = sys.argv[1]
    zs = [float(v) * 1e-6 for v in sys.argv[2].split(',')]
    dest = sys.argv[3]
    n_fan = int(sys.argv[4]) if len(sys.argv) > 4 else 9000
    n_rho = int(sys.argv[5]) if len(sys.argv) > 5 else 3600
    fx = FIXTURES[fxname]
    E_in = input_field(fx)
    dx = fx['dx']
    hdr = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'numpy': np.__version__, 'python': sys.version.split()[0],
           'fixture': fxname, 'note': fx['note'], 'N': fx['N'], 'dx': dx,
           'w0': fx['w0'], 'wavelength': fx['wavelength'],
           'oracle_n_fan': n_fan, 'oracle_n_rho': n_rho,
           'index_control_delta': {
               g: orc.index_control(g, fx['wavelength'])
               for g in {s['glass_after'] for s in fx['prescription']['surfaces']}
               if g != 'air'}}
    print(json.dumps(hdr), flush=True)
    # launched power inside the traced launch radius (the model-free reference)
    ap = fx['prescription'].get('aperture_diameter')
    lr = 0.5 * float(ap) * 0.98 if ap else 0.5 * fx['N'] * dx
    xg = (np.arange(fx['N']) - fx['N'] / 2.0) * dx
    XG, YG = np.meshgrid(xg, xg)
    inside = np.sqrt(XG ** 2 + YG ** 2) <= lr
    P_in = float(np.sum(np.abs(E_in[inside]) ** 2)) * dx * dx
    hdr['P_in_launch_radius'] = P_in
    rows = []
    for z in zs:
        t0 = time.time()
        E_or, _, _, ex = orc.oracle_field(fx, z, n_fan=n_fan, n_rho=n_rho)
        P_or = orc.power(E_or, dx)
        info, fields, d_u, d_mb = members(la, fx, E_in, z)
        row = {'z_um': z * 1e6, 'P_oracle': P_or,
               'P_oracle_over_launched': P_or / ex['P_in'],
               'oracle_P_in': ex['P_in'], 'P_in_grid': P_in}
        for name, E in fields.items():
            if E is None:
                continue
            info[name]['fidelity'] = orc.fidelity(E, E_or)
            info[name]['P_over_oracle'] = orc.power(E, dx) / P_or
            info[name]['P_over_Pin'] = orc.power(E, dx) / P_in
        row['members'] = info
        if d_u is not None:
            row['uniform_diag'] = {
                k: _clean(d_u.get(k)) for k in
                ('fell_back', 'reason', 'r_c', 'kappa', 'fit_residual',
                 'fit_halfwidth', 'zeta_band', 'zeta_extrapolation',
                 'power_ratio', 'power_ratio_triangles',
                 'n_triangles_finite', 'n_triangles_degenerate')}
            nb = d_u.get('n_branch')
            row['uniform_diag']['n_branch_max'] = (
                int(np.max(nb)) if nb is not None else None)
        if d_mb is not None:
            row['mb_diag'] = {k: _clean(d_mb.get(k)) for k in
                              ('power_ratio', 'power_ratio_triangles',
                               'n_triangles_finite',
                               'n_triangles_degenerate')}
            nb = d_mb.get('n_branch')
            row['mb_diag']['n_branch_max'] = (
                int(np.max(nb)) if nb is not None else None)
        rows.append(row)
        ud = row.get('uniform_diag', {})
        print(f"z={row['z_um']:9.2f} ({time.time() - t0:5.1f}s)  "
              f"zx={ud.get('zeta_extrapolation')!s:>9.9s} "
              f"fb={ud.get('fell_back')!s:5s} "
              f"mbP={ud.get('power_ratio')!s:>10.10s}", flush=True)
        for name in ('uniform', 'multibranch', 'ray_density', 'wave'):
            m = info.get(name, {})
            if 'fidelity' in m:
                print(f"    {name:12s} fid={m['fidelity']:.4f} "
                      f"P/Por={m['P_over_oracle']:10.5g} "
                      f"P/Pin={m['P_over_Pin']:10.5g} "
                      f"nw={len(m['warnings'])}", flush=True)
            else:
                print(f"    {name:12s} {m.get('error', '?')[:110]}", flush=True)
        with open(dest, 'w', encoding='cp1252') as fh:
            json.dump({'header': hdr, 'rows': rows}, fh, indent=1)
    print('wrote', dest)


if __name__ == '__main__':
    main()

"""Task E -- the ``zeta_linear_range`` diagnostic and the dark-fill depth.

Three arms:

1. **reported**: the key is present, finite and positive on every fold plane,
   beside ``zeta_curvature``, ``zeta_linear_resid``, ``dark_fill_depth`` and
   ``l_airy``, and its value is the band's own ``0.1 kappa / |q|``.
2. **never applied**: ``_trace_meridional_fold`` is monkey-patched to return
   an ABSURD ``zeta_linear_range`` (1e-12 m, i.e. far inside one Airy length,
   and 1e+6 m) with every other key untouched.  If the fill were clipped by it
   the returned field would move; the digest is asserted byte-identical.  The
   same is done for ``zeta_curvature``.
3. **the 3-Airy-length concentration**: cumulative dark-side energy of the
   completed field against the oracle's, in the annulus ``r_c + n * l_airy``
   for n = 1, 2, 3, 5, 10, 20, so the "more than 94 % of the excess is inside
   3 Airy lengths" premise is re-measured on the VERIFY fixtures.

Usage: python probe3_tail.py <FIXTURE> <out.json> <z_um> [<z_um> ...]
"""
# ruff: noqa: E402, I001  (sys.path is set up between the imports)
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures as FX                                     # noqa: E402
import oracle as OR                                       # noqa: E402

import lumenairy                                          # noqa: E402
from lumenairy.elements import _lens_traced_uniform as U   # noqa: E402


def uni(fx, E, z):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        Eo, d = U.apply_real_lens_traced_uniform(
            E, prescription=fx['prescription'], wavelength=fx['wavelength'],
            dx=fx['dx'], output_plane_distance=z, return_diagnostics=True)
    return np.asarray(Eo), d, len(rec)


def digest(E):
    return hashlib.sha256(np.asarray(E).tobytes()).hexdigest()


def annulus_profile(fx, E_uni, E_or, r_c, l_airy):
    """Cumulative dark-side energy, completed vs oracle, by fill depth."""
    N, dx = fx['N'], fx['dx']
    x = (np.arange(N) - N / 2.0) * dx
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X ** 2 + Y ** 2)
    out = {}
    for n in (1, 2, 3, 5, 10, 20):
        m = (r > r_c) & (r <= r_c + n * l_airy)
        po = float(np.sum(np.abs(E_or[m]) ** 2))
        pu = float(np.sum(np.abs(E_uni[m]) ** 2))
        out[str(n)] = (pu / po) if po > 0 else None
    return out


def main(argv):
    name, out = argv[1], argv[2]
    zs = [float(v) * 1e-6 for v in argv[3:]]
    fx = FX.FIXTURES[name]
    E = FX.input_field(fx)
    res = {'lumenairy_file': lumenairy.__file__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'fixture': name, 'note': fx['note'], 'rows': []}
    real_trace = U._trace_meridional_fold
    for z in zs:
        r = {'z_um': z * 1e-6 * 1e12}
        r['z_um'] = z * 1e6
        try:
            E_u, d, nw = uni(fx, E, z)
        except RuntimeError as e:
            res['rows'].append({'z_um': z * 1e6, 'error': str(e)[:160]})
            continue
        r['reason'] = d.get('reason')
        r['fell_back'] = d.get('fell_back')
        for k in ('zeta_linear_range', 'zeta_curvature', 'zeta_linear_resid',
                  'dark_fill_depth', 'l_airy', 'r_c', 'kappa',
                  'zeta_extrapolation', 'fit_halfwidth'):
            v = d.get(k)
            r[k] = (float(v) if v is not None else None)
        if r['zeta_linear_range'] and r['l_airy']:
            r['u_star_over_l_airy'] = r['zeta_linear_range'] / r['l_airy']
            r['fill_cells'] = r['dark_fill_depth'] / r['l_airy']
        # --- arm 2: the value cannot reach the field
        base = digest(E_u)
        r['digest'] = base
        moved = {}
        for tag, val in (('tiny', 1e-12), ('huge', 1e6)):
            def patched(*a, _v=val, **kw):
                f = real_trace(*a, **kw)
                if isinstance(f, dict) and f.get('ok'):
                    f = dict(f)
                    f['zeta_linear_range'] = _v
                    f['zeta_curvature'] = _v
                return f
            U._trace_meridional_fold = patched
            try:
                E_p, dp, _ = uni(fx, E, z)
                moved[tag] = {'digest_equal': digest(E_p) == base,
                              'reported': dp.get('zeta_linear_range')}
            finally:
                U._trace_meridional_fold = real_trace
        r['monkeypatch'] = moved
        # --- arm 3: the depth profile against the oracle
        if r.get('r_c') and r.get('l_airy'):
            E_or, _rho, _E_rho, _ex = OR.oracle_field(fx, z)
            r['annulus_ratio'] = annulus_profile(fx, E_u, E_or, r['r_c'],
                                                 r['l_airy'])
            a = r['annulus_ratio']
            if a.get('3') and a.get('20'):
                r['excess_inside_3_frac'] = (
                    (a['3'] - 1.0) / (a['20'] - 1.0)
                    if abs(a['20'] - 1.0) > 1e-12 else None)
        res['rows'].append(r)
        print(f"{name} z={r['z_um']:9.2f} {r.get('reason')} "
              f"u*/l_airy={r.get('u_star_over_l_airy')} "
              f"patch={ {k: v['digest_equal'] for k, v in moved.items()} } "
              f"annulus={r.get('annulus_ratio')}", flush=True)
        with open(out, 'w') as fh:
            json.dump(res, fh, indent=1, default=str)
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1, default=str)


if __name__ == '__main__':
    main(sys.argv)

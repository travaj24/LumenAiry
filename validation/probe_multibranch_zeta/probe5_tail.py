"""Probe 5 -- zeta(r) BEYOND the two-branch band, and the dark-tail depth.

Two independent readings of the same question (WP-B7b section 7):

GEOMETRY SIDE (no oracle).  The module fits ``zeta(r) = kappa (r_c - r)`` on the
two-branch band.  Here the same band is fitted with a QUADRATIC,
``zeta = kappa u + q u^2`` (``u = r_c - r``), and the radius at which the
quadratic term reaches 10 % of the linear one, ``u* = 0.1 kappa / |q|``, is
reported in Airy lengths.  ``u*`` is how far the linear normal form can be
carried before its own measured curvature says 10 %.

ORACLE SIDE.  From the exact Rayleigh-Sommerfeld field's DARK-side decay,
``|E| ~ exp(-2/3 x^{3/2}) / x^{1/4}`` with ``x = k^{2/3} kappa (r - r_c)``, the
EFFECTIVE ``kappa_eff(r)`` is extracted in windows across the fill depth and
compared with the fitted ``kappa``; and the CUMULATIVE dark-side energy of the
completed field is compared with the oracle's, annulus by annulus, out to
``_AIRY_TAIL_CELLS``.  The depth at which the cumulative ratio leaves a stated
band is the derived bound for clipping the fill.

Usage:  python probe5_tail.py <fixture> <z_um,...> <out.json> [n_fan] [n_rho]
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
import oracle as orc                                 # noqa: E402


def quad_band_fit(la, fx, z):
    """Re-fit the module's own two-branch band with a quadratic zeta."""
    import lumenairy.raytrace as rt
    presc, wl, dx, N = (fx['prescription'], fx['wavelength'], fx['dx'],
                        fx['N'])
    ap = presc.get('aperture_diameter')
    lr = 0.5 * float(ap) * 0.98 if ap else 0.5 * N * dx
    n_fan = 4000
    surfaces = rt.surfaces_from_prescription(presc)
    xs = np.linspace(lr / n_fan, lr, n_fan)
    rays = rt.RayBundle(x=xs.copy(), y=np.zeros(n_fan), z=np.zeros(n_fan),
                        L=np.zeros(n_fan), M=np.zeros(n_fan),
                        N=np.ones(n_fan), wavelength=wl,
                        alive=np.ones(n_fan, dtype=bool), opd=np.zeros(n_fan))
    ex = rt.trace(rays, surfaces, wl).at_exit_vertex()
    Nz = np.where(np.abs(ex.N) > 1e-30, ex.N, 1e-30)
    t = z / Nz
    xo_all = ex.x + t * ex.L
    opl_all = ex.opd + t
    alive = ex.alive & np.isfinite(xo_all) & np.isfinite(opl_all)
    xo, S = xo_all[alive], opl_all[alive]
    dxo = np.diff(xo)
    turns = np.where(np.diff(np.sign(dxo)) != 0)[0] + 1
    if turns.size != 1:
        return None
    i_f = int(turns[0])
    r_c = float(abs(xo[i_f]))
    from scipy.interpolate import CubicSpline

    def _sp(yv, sv):
        yu, idx = np.unique(yv, return_index=True)
        return CubicSpline(yu, sv[idx]) if yu.size >= 4 else None

    spa = _sp(np.abs(xo[:i_f + 1]), S[:i_f + 1])
    spb = _sp(np.abs(xo[i_f:]), S[i_f:])
    if spa is None or spb is None:
        return None
    r_lo = float(max(np.abs(xo[:i_f + 1]).min(), np.abs(xo[i_f:]).min()))
    band = r_c - r_lo
    if band <= 0:
        return None
    rb = np.linspace(r_lo + 0.02 * band, r_c - 0.02 * band, 256)
    zeta = (0.75 * np.abs(spa(rb) - spb(rb))) ** (2.0 / 3.0)
    u = r_c - rb
    lin = np.polyfit(u, zeta, 1)
    quad = np.polyfit(u, zeta, 2)
    kappa = float(lin[0])
    q = float(quad[0])
    lin_resid = float(np.max(np.abs(np.polyval(lin, u) - zeta))
                      / (np.max(zeta) + 1e-300))
    quad_resid = float(np.max(np.abs(np.polyval(quad, u) - zeta))
                       / (np.max(zeta) + 1e-300))
    k0 = 2.0 * np.pi / wl
    l_airy = 1.0 / (k0 ** (2.0 / 3.0) * kappa)
    u_star = (0.1 * kappa / abs(q)) if q != 0.0 else float('inf')
    return dict(r_c=r_c, band=band, kappa=kappa, kappa_quad=float(quad[1]),
                q=q, lin_resid=lin_resid, quad_resid=quad_resid,
                l_airy=l_airy, u_star=u_star,
                u_star_in_l_airy=u_star / l_airy,
                band_in_l_airy=band / l_airy)


def kappa_eff_from_oracle(rho, E_rho, r_c, kappa, k0, l_airy, n_win=8,
                          depth_cells=20.0):
    """Effective kappa from the oracle's dark-side decay, in windows."""
    a = k0 ** (2.0 / 3.0) * kappa
    out = []
    edges = np.linspace(0.5, depth_cells, n_win + 1)
    mag = np.abs(E_rho)
    for i in range(n_win):
        lo, hi = edges[i] * l_airy, edges[i + 1] * l_airy
        m = (rho > r_c + lo) & (rho < r_c + hi) & (mag > 0)
        if int(m.sum()) < 8:
            out.append(None)
            continue
        d = rho[m] - r_c
        y = np.log(mag[m]) + 0.25 * np.log(np.maximum(a * d, 1e-300))
        s = d ** 1.5
        sl = float(np.polyfit(s, y, 1)[0])
        if sl >= 0:
            out.append(None)
            continue
        a_eff = (-1.5 * sl) ** (2.0 / 3.0)
        out.append({'from_cells': float(edges[i]), 'to_cells': float(edges[i + 1]),
                    'kappa_eff': float(a_eff / k0 ** (2.0 / 3.0)),
                    'kappa_eff_over_kappa': float(a_eff / a),
                    'n_pts': int(m.sum())})
    return out


def main():
    import lumenairy as la
    fxname = sys.argv[1]
    zs = [float(v) * 1e-6 for v in sys.argv[2].split(',')]
    dest = sys.argv[3]
    n_fan = int(sys.argv[4]) if len(sys.argv) > 4 else 9000
    n_rho = int(sys.argv[5]) if len(sys.argv) > 5 else 6000
    fx = FIXTURES[fxname]
    E_in = input_field(fx)
    dx, N, wl = fx['dx'], fx['N'], fx['wavelength']
    k0 = 2.0 * np.pi / wl
    hdr = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'numpy': np.__version__, 'python': sys.version.split()[0],
           'fixture': fxname, 'oracle_n_fan': n_fan, 'oracle_n_rho': n_rho}
    print(json.dumps(hdr), flush=True)
    xg = (np.arange(N) - N / 2.0) * dx
    XG, YG = np.meshgrid(xg, xg)
    RG = np.sqrt(XG ** 2 + YG ** 2)
    rows = []
    for z in zs:
        row = {'z_um': z * 1e6}
        qb = quad_band_fit(la, fx, z)
        row['band_fit'] = qb
        if qb is None:
            rows.append(row)
            continue
        E_or2, rho, E_rho, _ = orc.oracle_field(fx, z, n_fan=n_fan,
                                                n_rho=n_rho)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            E_u, d = la.elements._lens_traced_uniform.\
                apply_real_lens_traced_uniform(
                    E_in, prescription=fx['prescription'], wavelength=wl,
                    dx=dx, output_plane_distance=float(z),
                    return_diagnostics=True)
        row['fell_back'] = bool(d.get('fell_back'))
        row['reason'] = d.get('reason')
        row['zeta_extrapolation'] = (None if d.get('zeta_extrapolation') is None
                                     else float(d['zeta_extrapolation']))
        row['warnings'] = [str(w.message)[:120] for w in rec]
        r_c, l_airy = qb['r_c'], qb['l_airy']
        row['kappa_eff'] = kappa_eff_from_oracle(rho, E_rho, r_c, qb['kappa'],
                                                 k0, l_airy)
        # cumulative dark-side energy on the library grid, model vs oracle
        cum = []
        Iu = np.abs(np.asarray(E_u)) ** 2
        Io = np.abs(E_or2) ** 2
        for n in (1, 2, 3, 5, 8, 10, 12, 15, 20):
            m = (RG > r_c) & (RG <= r_c + n * l_airy)
            eu = float(Iu[m].sum()) * dx * dx
            eo = float(Io[m].sum()) * dx * dx
            cum.append({'cells': n, 'E_model': eu, 'E_oracle': eo,
                        'ratio': (eu / eo) if eo > 0 else None,
                        'n_px': int(m.sum())})
        row['dark_cumulative'] = cum
        rows.append(row)
        print(f"z={row['z_um']:9.2f} zx={row['zeta_extrapolation']!s:>9.9s} "
              f"fb={row['fell_back']!s:5s} band/l_airy={qb['band_in_l_airy']:8.4g} "
              f"lin_resid={qb['lin_resid']:.4f} quad_resid={qb['quad_resid']:.4f} "
              f"u*/l_airy={qb['u_star_in_l_airy']:9.4g}", flush=True)
        ke = ' '.join(f"{w['kappa_eff_over_kappa']:.3f}" if w else '  -  '
                      for w in row['kappa_eff'])
        print(f"    kappa_eff/kappa by depth window: {ke}", flush=True)
        cr = ' '.join(f"{c['cells']}:{c['ratio']:.3f}" if c['ratio'] else
                      f"{c['cells']}:-" for c in cum)
        print(f"    cumulative dark E model/oracle: {cr}", flush=True)
        with open(dest, 'w', encoding='cp1252') as fh:
            json.dump({'header': hdr, 'rows': rows}, fh, indent=1)
    print('wrote', dest)


if __name__ == '__main__':
    main()

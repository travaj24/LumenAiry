"""Probe 2 -- the mechanism of the multibranch 1/sqrt|J| blow-up.

Three decisive, independent measurements at the blow-up planes of fixture V:

A. KNOB SWEEP.  ``min_area_ratio`` (the only clip on ``1/sqrt|J|``) and
   ``caustic_band`` ('ludwig' pair swap vs 'plain') and ``ray_subsample`` (the
   launch-lattice pitch).  Which knob moves the blow-up names the mechanism.

B. GEOMETRY CENSUS.  The module's own launch grid (``_trace_launch_grid``) is
   re-triangulated here with the module's own rule, and every mapped triangle is
   measured in PIXEL units: its area ``A/dx^2`` and its minimum altitude
   ``(2A/longest side)/dx``.  The reconstruction writes one amplitude per pixel
   CENTRE it covers, so its quadrature is exact only while a mapped triangle
   spans at least one pixel in BOTH directions.

C. PREDICTED OVER-COUNT.  Sum over triangles of the energy the point-sampled
   write actually deposits, ``max(1, A/dx^2) * dx^2 * |E|^2 / ratio``, against
   the energy the triangle launches, ``A_tri |E|^2``.  If that predicted ratio
   tracks the measured ``power_ratio``, the blow-up is a QUADRATURE defect of
   the rasteriser, not a wrong branch or an unclipped Jacobian.

Usage:  python probe2_mechanism.py <fixture> <z_um>[,<z_um>...] [out.json]
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


def mb(la, fx, E_in, z, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        E, d = la.elements._lens_traced_multibranch.\
            apply_real_lens_traced_multibranch(
                E_in, prescription=fx['prescription'],
                wavelength=fx['wavelength'], dx=fx['dx'],
                output_plane_distance=z, return_diagnostics=True, **kw)
    return np.asarray(E), d, [str(w.message)[:90] for w in rec]


def geometry_census(la, fx, E_in, z):
    """Re-triangulate the module's own launch grid and measure every mapped
    triangle in pixel units.  Returns the census plus the predicted over-count."""
    m = la.elements._lens_traced_multibranch
    dx = fx['dx']
    N = fx['N']
    presc = fx['prescription']
    aperture = presc.get('aperture_diameter')
    launch_radius = (0.5 * float(aperture) * 0.98 if aperture is not None
                     else 0.5 * N * dx)
    sub = 2
    n_launch = max(9, int(2 * launch_radius / (dx * sub)))
    if n_launch % 2 == 0:
        n_launch += 1
    g = m._trace_launch_grid(presc, fx['wavelength'], launch_radius, n_launch,
                             z, 1.0, L0=0.0, M0=0.0)
    XO, YO = g['x_out'], g['y_out']
    ok = g['alive'] & np.isfinite(XO) & np.isfinite(g['opl'])
    xs_in = g['xs_in']
    h = xs_in[1] - xs_in[0]
    tri_launch_area = 0.5 * h * h
    # the module's own input sampling, for the per-triangle launch power
    fi = np.clip((xs_in / dx) + N / 2.0, 0.0, N - 1.0 - 1e-9)
    i0 = np.floor(fi).astype(int)
    w = fi - i0
    Exx = ((1 - w)[None, :] * ((1 - w)[:, None] * E_in[np.ix_(i0, i0)]
                               + w[:, None] * E_in[np.ix_(i0 + 1, i0)])
           + w[None, :] * ((1 - w)[:, None] * E_in[np.ix_(i0, i0 + 1)]
                           + w[:, None] * E_in[np.ix_(i0 + 1, i0 + 1)]))
    E_launch = Exx.T
    ok4 = ok[:-1, :-1] & ok[1:, :-1] & ok[:-1, 1:] & ok[1:, 1:]
    ci, cj = np.nonzero(ok4)
    V0i = np.concatenate([ci, ci + 1])
    V0j = np.concatenate([cj, cj])
    V1i = np.concatenate([ci + 1, ci + 1])
    V1j = np.concatenate([cj, cj + 1])
    V2i = np.concatenate([ci, ci])
    V2j = np.concatenate([cj + 1, cj + 1])
    x0, y0 = XO[V0i, V0j], YO[V0i, V0j]
    x1, y1 = XO[V1i, V1j], YO[V1i, V1j]
    x2, y2 = XO[V2i, V2j], YO[V2i, V2j]
    area2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    A = np.abs(0.5 * area2)
    ratio = A / tri_launch_area
    finite = np.isfinite(area2)
    good = finite & (ratio >= 1e-6)
    # inside the output grid at all?
    ing = ((np.abs(x0) <= 0.5 * N * dx) & (np.abs(y0) <= 0.5 * N * dx))
    sel = good & ing
    Etri = ((np.abs(E_launch[V0i, V0j]) ** 2 + np.abs(E_launch[V1i, V1j]) ** 2
             + np.abs(E_launch[V2i, V2j]) ** 2) / 3.0)
    Apx = A / (dx * dx)
    # minimum altitude of the mapped triangle = 2A / longest side
    s0 = np.hypot(x2 - x1, y2 - y1)
    s1 = np.hypot(x0 - x2, y0 - y2)
    s2 = np.hypot(x1 - x0, y1 - y0)
    smax = np.maximum(np.maximum(s0, s1), s2)
    with np.errstate(divide='ignore', invalid='ignore'):
        alt = np.where(smax > 0, 2.0 * A / np.maximum(smax, 1e-300), 0.0)
    alt_px = alt / dx
    # predicted over-count: a triangle writes max(1, A/dx^2) pixel centres, each
    # carrying intensity |E|^2 / ratio over an area dx^2.
    npix = np.maximum(1.0, Apx)
    with np.errstate(divide='ignore', invalid='ignore'):
        e_written = np.where(sel, npix * dx * dx * Etri / np.maximum(ratio, 1e-300), 0.0)
    e_launched = np.where(sel, Etri * tri_launch_area, 0.0)
    pred = float(np.sum(e_written) / max(float(np.sum(e_launched)), 1e-300))
    q = lambda a, p: float(np.percentile(a[sel], p)) if sel.any() else float('nan')  # noqa: E731
    return {
        'n_tri': int(area2.size), 'n_good_in_grid': int(sel.sum()),
        'tri_launch_area_px': float(tri_launch_area / (dx * dx)),
        'Apx_min': q(Apx, 0), 'Apx_p01': q(Apx, 1), 'Apx_p50': q(Apx, 50),
        'Apx_max': q(Apx, 100),
        'alt_px_min': q(alt_px, 0), 'alt_px_p01': q(alt_px, 1),
        'alt_px_p50': q(alt_px, 50),
        'frac_A_sub_pixel': float(np.mean(Apx[sel] < 1.0)) if sel.any() else None,
        'frac_alt_sub_pixel': float(np.mean(alt_px[sel] < 1.0)) if sel.any() else None,
        'ratio_min': float(np.min(ratio[sel])) if sel.any() else None,
        'amp_J_max': float(1.0 / np.sqrt(max(np.min(ratio[sel]), 1e-300)))
                      if sel.any() else None,
        'predicted_power_ratio': pred,
    }


def main():
    import lumenairy as la
    fxname = sys.argv[1]
    zs = [float(v) * 1e-6 for v in sys.argv[2].split(',')]
    dest = sys.argv[3] if len(sys.argv) > 3 else None
    fx = FIXTURES[fxname]
    E_in = input_field(fx)
    hdr = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'numpy': np.__version__, 'python': sys.version.split()[0],
           'fixture': fxname}
    print(json.dumps(hdr))
    rows = []
    for z in zs:
        row = {'z_um': z * 1e6}
        # A. knob sweep
        knobs = {}

        def _try(label, **kw):
            try:
                _, d, _ = mb(la, fx, E_in, z, **kw)
            except Exception as exc:                  # noqa: BLE001
                knobs[label] = {'refused': f'{type(exc).__name__}',
                                'power_ratio': float('nan'),
                                'n_branch_max': -1}
                return
            knobs[label] = {
                'power_ratio': d['power_ratio'],
                'n_degenerate': d['n_triangles_degenerate'],
                'n_branch_max': int(np.max(d['n_branch']))}

        for mar in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
            _try(f'min_area_ratio={mar:g}', min_area_ratio=mar)
        for cb in ('ludwig', 'plain'):
            _try(f'caustic_band={cb}', caustic_band=cb)
        for sub in (1, 2, 4):
            _try(f'ray_subsample={sub}', ray_subsample=sub)
        row['knobs'] = knobs
        # B + C
        row['census'] = geometry_census(la, fx, E_in, z)
        rows.append(row)
        print(f"--- z = {row['z_um']:.1f} um ---")
        for kk, vv in knobs.items():
            print(f"    {kk:24s} P/Pin={vv['power_ratio'] if vv['power_ratio']==vv['power_ratio'] else float('nan'):12.5g} "
                  f"nbr={vv['n_branch_max']:5d} "
                  f"ndeg={vv.get('n_degenerate', '-')}")
        c = row['census']
        print(f"    census: n_good={c['n_good_in_grid']} "
              f"A_tri={c['tri_launch_area_px']:.3f} px^2 | "
              f"Apx min/p01/p50={c['Apx_min']:.3g}/{c['Apx_p01']:.3g}/"
              f"{c['Apx_p50']:.3g} | alt_px min/p01={c['alt_px_min']:.3g}/"
              f"{c['alt_px_p01']:.3g}")
        print(f"    frac(A<1px)={c['frac_A_sub_pixel']:.4f} "
              f"frac(alt<1px)={c['frac_alt_sub_pixel']:.4f} "
              f"amp_J_max={c['amp_J_max']:.4g} "
              f"PREDICTED P/Pin={c['predicted_power_ratio']:.5g}", flush=True)
    if dest:
        with open(dest, 'w', encoding='cp1252') as fh:
            json.dump({'header': hdr, 'rows': rows}, fh, indent=1)
        print('wrote', dest)


if __name__ == '__main__':
    main()

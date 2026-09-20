"""VERIFY-WP-B7c round 3 -- geometry reconnaissance for MY population.

Everything the plane ladders are laid out from is measured here on the
running build, so no plane list in this verification is a number copied from a
report: the index control against the library's own Sellmeier, the NA, the
paraxial and marginal foci, the number of interior STATIONARY POINTS in the
focal locus ``f(h)`` (which is what makes ``VA``'s locus non-monotone and
every spherical optic's monotone), and the ``z`` window in which the
meridional LANDING map has an interior turning point -- the fold ring.

Usage:  python v3geom.py out.json [optic ...]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import v3fixtures as FX
import v3oracle as OR

_WAVELENGTHS = (532e-9, 633e-9, 780e-9, 850e-9, 1.064e-6, 1.31e-6, 2.0e-6)


def na_and_foci(presc, wl, n=801):
    """NA, paraxial focus, marginal focus, and the focal LOCUS ``f(h)``."""
    surfs = presc['surfaces']
    r_edge = 0.5 * float(presc['aperture_diameter']) * 0.98
    h = np.linspace(r_edge / n, r_edge, n)
    _ye, _o0, y0, ok0 = OR.trace_fan(surfs, wl, h, 0.0)
    _ye1, _o1, y1, ok1 = OR.trace_fan(surfs, wl, h, 1e-3)
    slope = (y1 - y0) / 1e-3
    ok = ok0 & ok1 & np.isfinite(slope) & (np.abs(slope) > 0.0)
    f = np.where(ok, -y0 / np.where(ok, slope, 1.0), np.nan)
    na = float(abs(slope[ok][-1]) / np.sqrt(1.0 + slope[ok][-1] ** 2))
    return na, float(f[ok][0]), float(f[ok][-1]), h[ok], f[ok]


def turns(v):
    """Interior sign changes of the first difference of ``v``."""
    d = np.diff(np.asarray(v, dtype=float))
    m = np.isfinite(d) & (d != 0.0)
    s = np.sign(d[m])
    return int(np.count_nonzero(np.diff(s) != 0))


def fold_window(presc, wl, zs, n=501):
    """Interior turning radii of the LANDING map at each ``z``."""
    surfs = presc['surfaces']
    r_edge = 0.5 * float(presc['aperture_diameter']) * 0.98
    h = np.linspace(r_edge / n, r_edge, n)
    out = []
    for z in zs:
        _y, _o, yl, ok = OR.trace_fan(surfs, wl, h, float(z))
        yv = yl[ok]
        if yv.size < 4:
            out.append((float(z), [], float('nan')))
            continue
        d = np.diff(yv)
        s = np.sign(d)
        idx = np.nonzero(np.diff(s) != 0)[0]
        out.append((float(z), [float(abs(yv[i + 1])) for i in idx],
                    float(np.abs(yv).max())))
    return out


def main():
    out_path = sys.argv[1]
    names = sys.argv[2:] or list(FX.FIXTURES)
    rep = {}
    ctrl = {g: {str(w): OR.index_control(g, w) for w in _WAVELENGTHS}
            for g in OR.SELLMEIER}
    rep['index_control'] = ctrl
    rep['index_control_max'] = max(max(v.values()) for v in ctrl.values())
    vf = FX.FIXTURES['V']
    rep['aspheric_inertness_on_V'] = OR.aspheric_inertness_control(
        vf['prescription'], vf['wavelength'], 1.761e-3)
    cf = FX.FIXTURES['VC']
    rep['aspheric_inertness_on_VC_conic'] = OR.aspheric_inertness_control(
        cf['prescription'], cf['wavelength'], 2.0e-3)
    for nm in names:
        fx = FX.FIXTURES[nm]
        presc, wl = fx['prescription'], fx['wavelength']
        na, fp, fm, hs, fl = na_and_foci(presc, wl)
        lo, hi = min(fm, fp) * 0.55, max(fm, fp) * 1.45
        zs = np.linspace(lo, hi, 90)
        fw = fold_window(presc, wl, zs)
        fold_zs = [z for z, rc, _ in fw if rc]
        halfw = 0.5 * fx['N'] * fx['dx']
        rep[nm] = dict(
            note=fx['note'], provenance=FX.PROVENANCE.get(nm), na=na,
            f_paraxial=fp, f_marginal=fm, grid_halfwidth=halfw,
            N=fx['N'], dx=fx['dx'], wavelength=wl,
            turns_in_focal_locus=turns(fl),
            focal_locus_min=float(np.nanmin(fl)),
            focal_locus_max=float(np.nanmax(fl)),
            fold_z_range=([min(fold_zs), max(fold_zs)] if fold_zs else None),
            ymax_over_z_at_fmar=float(
                0.5 * fx['N'] * fx['dx'] / fm) if fm else None,
            table=[(z, rc, ym) for z, rc, ym in fw])
        print(f"{nm:8s} NA {na:.4f}  f_par {fp * 1e6:9.1f}  "
              f"f_mar {fm * 1e6:9.1f}  turns {turns(fl)}  "
              f"halfwidth {halfw * 1e6:7.1f}  fold "
              + ('none' if not fold_zs else
                 f'{min(fold_zs) * 1e6:.1f}..{max(fold_zs) * 1e6:.1f} um'),
              flush=True)
    with open(out_path, 'w', encoding='cp1252') as f:
        json.dump(rep, f, indent=1, default=float)
    print('index control max      ', rep['index_control_max'])
    print('aspheric inertness V   ', rep['aspheric_inertness_on_V'])
    print('aspheric inertness VC  ', rep['aspheric_inertness_on_VC_conic'])
    print('wrote', out_path)


if __name__ == '__main__':
    main()

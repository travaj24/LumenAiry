"""Geometry reconnaissance for the round-3 population.

Index control, NA, paraxial / marginal foci, the ``z`` window in which the
meridional landing map has an INTERIOR turning point (the fold ring), and the
grid's own half-width -- everything the scan ladders are laid out from, so no
plane list in this round is a number copied from an earlier report.

Also runs the aspheric extension's CONTROL: on a prescription with no
aspheric term this module's trace must reproduce the round-2 verifier's to
machine precision.

Usage:  python r3geom.py out.json [optic ...]
"""
from __future__ import annotations

import json
import sys

import numpy as np
import r3fixtures as FX
import r3oracle as OR


def na_and_foci(presc, wl):
    surfs = presc['surfaces']
    aper = presc['aperture_diameter']
    r_edge = 0.5 * float(aper) * 0.98
    h = np.array([r_edge * 1e-4, r_edge])
    y0, o0, yl0, ok0 = OR.trace_fan(surfs, wl, h, 0.0)
    y1, o1, yl1, ok1 = OR.trace_fan(surfs, wl, h, 1e-3)
    slope = (yl1 - yl0) / 1e-3
    na = float(abs(slope[1]) / np.sqrt(1.0 + slope[1] ** 2))
    f_par = float(-yl0[0] / slope[0])
    f_mar = float(-yl0[1] / slope[1])
    return na, f_par, f_mar


def fold_window(presc, wl, zs, n=401):
    """Interior turning radii of the landing map at each ``z``."""
    surfs = presc['surfaces']
    aper = presc['aperture_diameter']
    r_edge = 0.5 * float(aper) * 0.98
    h = np.linspace(r_edge / n, r_edge, n)
    out = []
    for z in zs:
        _, _, yl, ok = OR.trace_fan(surfs, wl, h, float(z))
        yv = yl[ok]
        if yv.size < 4:
            out.append((float(z), [], float('nan')))
            continue
        d = np.diff(yv)
        s = np.sign(d)
        turns = np.nonzero(np.diff(s) != 0)[0]
        rc = [float(abs(yv[i + 1])) for i in turns]
        out.append((float(z), rc, float(np.abs(yv).max())))
    return out


def main():
    out_path = sys.argv[1]
    names = sys.argv[2:] or list(FX.OPTICS)
    rep = {}
    ctrl = {g: {str(w): OR.index_control(g, w)
                for w in (532e-9, 633e-9, 780e-9, 850e-9, 1.064e-6, 1.31e-6,
                          2.0e-6)}
            for g in OR.SELLMEIER}
    rep['index_control_max'] = max(max(v.values()) for v in ctrl.values())
    rep['index_control'] = ctrl
    # the aspheric extension is inert on a spherical prescription
    vf = FX.FIXTURES['V']
    rep['aspheric_extension_control_on_V'] = OR.spherical_path_control(
        vf['prescription'], vf['wavelength'], vf['w0'], 1.761e-3)
    kf = FX.FIXTURES['K']
    rep['aspheric_extension_control_on_K_conic'] = OR.spherical_path_control(
        kf['prescription'], kf['wavelength'], kf['w0'], 2.0e-3)
    for nm in names:
        fx = FX.FIXTURES[nm]
        presc, wl = fx['prescription'], fx['wavelength']
        try:
            na, fp, fm = na_and_foci(presc, wl)
        except Exception as exc:                            # noqa: BLE001
            rep[nm] = dict(error=f'{type(exc).__name__}: {exc}')
            continue
        lo = min(fm, fp) * 0.55
        hi = max(fm, fp) * 1.40
        zs = np.linspace(lo, hi, 70)
        fw = fold_window(presc, wl, zs)
        halfw = 0.5 * fx['N'] * fx['dx']
        fold_zs = [z for z, rc, ym in fw if rc]
        rep[nm] = dict(
            note=fx['note'], provenance=FX.PROVENANCE.get(nm), na=na,
            f_paraxial=fp, f_marginal=fm, grid_halfwidth=halfw,
            N=fx['N'], dx=fx['dx'], wavelength=wl,
            fold_z_range=[min(fold_zs), max(fold_zs)] if fold_zs else None,
            table=[(z, rc, ym, ym / z if z else None) for z, rc, ym in fw])
        print(nm, 'NA %.4f' % na, 'f_par %.1f um' % (fp * 1e6),
              'f_mar %.1f um' % (fm * 1e6), 'halfwidth %.1f um' % (halfw * 1e6),
              'fold z',
              None if not fold_zs else '%.1f..%.1f um' % (min(fold_zs) * 1e6,
                                                          max(fold_zs) * 1e6),
              flush=True)
    with open(out_path, 'w', encoding='cp1252') as f:
        json.dump(rep, f, indent=1, default=float)
    print('index control max', rep['index_control_max'])
    print('aspheric control on V', rep['aspheric_extension_control_on_V'])
    print('aspheric control on K', rep['aspheric_extension_control_on_K_conic'])
    print('wrote', out_path)


if __name__ == '__main__':
    main()

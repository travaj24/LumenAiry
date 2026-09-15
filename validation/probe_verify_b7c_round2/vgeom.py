"""Geometry reconnaissance for the verifier's optics: index control, marginal
and paraxial foci, the NA, and the z window where the meridional map has an
interior fold."""
from __future__ import annotations

import json
import sys

import numpy as np
import vfixtures as FX
import vroracle as OR


def na_and_foci(presc, wl):
    surfs = presc['surfaces']
    aper = presc['aperture_diameter']
    r_edge = 0.5 * float(aper) * 0.98
    # marginal ray direction after the last surface
    h = np.array([r_edge * 1e-4, r_edge])
    y0, o0, yl0, ok0 = OR.trace_fan(surfs, wl, h, 0.0)
    y1, o1, yl1, ok1 = OR.trace_fan(surfs, wl, h, 1e-3)
    # slope dy/dz of each ray between the two planes
    slope = (yl1 - yl0) / 1e-3
    na = float(abs(slope[1]) / np.sqrt(1.0 + slope[1] ** 2))
    f_par = float(-yl0[0] / slope[0])
    f_mar = float(-yl0[1] / slope[1])
    return na, f_par, f_mar


def fold_window(presc, wl, zs, n=801):
    """For each z, the interior turning point of the landing map (the fold
    ring radius), or NaN."""
    surfs = presc['surfaces']
    aper = presc['aperture_diameter']
    r_edge = 0.5 * float(aper) * 0.98
    h = np.linspace(r_edge / n, r_edge, n)
    out = []
    for z in zs:
        _, _, yl, ok = OR.trace_fan(surfs, wl, h, float(z))
        yv = yl[ok]
        d = np.diff(yv)
        s = np.sign(d)
        turns = np.nonzero(np.diff(s) != 0)[0]
        # interior turning points (not the axis crossing)
        rc = [float(abs(yv[i + 1])) for i in turns]
        out.append((float(z), rc, float(abs(yv).max())))
    return out


def main():
    names = sys.argv[1:] or ['W', 'X', 'Y', 'Z', 'C']
    rep = {}
    # index control
    ctrl = {}
    for g in OR.SELLMEIER:
        ctrl[g] = {str(w): OR.index_control(g, w)
                   for w in (532e-9, 633e-9, 780e-9, 850e-9, 1.064e-6,
                             1.31e-6)}
    rep['index_control_max'] = max(max(v.values()) for v in ctrl.values())
    rep['index_control'] = ctrl
    for nm in names:
        fx = FX.FIXTURES[nm]
        presc, wl = fx['prescription'], fx['wavelength']
        na, fp, fm = na_and_foci(presc, wl)
        zs = np.linspace(0.55 * fm, 1.35 * fp, 60)
        fw = fold_window(presc, wl, zs)
        rep[nm] = dict(note=fx['note'], na=na, f_paraxial=fp, f_marginal=fm,
                       grid_halfwidth=0.5 * fx['N'] * fx['dx'],
                       ymax_over_z=[(z, rc, ym) for z, rc, ym in fw])
    print(json.dumps(rep, indent=1, default=float))


if __name__ == '__main__':
    main()

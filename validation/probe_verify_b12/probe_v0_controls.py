"""VERIFY-WP-B12 probe V0 -- the oracle's own controls.

Before any claim is re-measured, the independent tracer has to be shown to
describe the SAME optic the library does.  Four controls, each stated with
the number it produced:

1. my typed-in Sellmeier vs ``lumenairy.glass.get_glass_index`` on the two
   catalogue glasses the fixtures use;
2. my LAST-SURFACE state vs ``lumenairy.raytrace.trace(...).image_rays``
   (position, direction cosines, OPL) on every fixture;
3. my EXIT-VERTEX state vs ``TraceResult.at_exit_vertex()`` on every
   fixture -- the plane the repair claims to move to;
4. the slope-vs-direction-cosine trap: ``max |u - L|``, the scale the two
   agreements above have to be compared against.

Also prints, per fixture, the sag of the last surface at the rim and the
fixture's own geometric best focus (both derived here, never pinned).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vb12_common as C  # noqa: E402


def main():
    import lumenairy as la
    from lumenairy import glass as G
    from lumenairy.raytrace import surfaces_from_prescription, trace
    print('lumenairy.__file__ =', os.path.abspath(la.__file__))
    print('version', la.__version__)
    out = {'env': C.env_block(), 'glass': {}, 'fixtures': {}}

    # -- 1. glass ---------------------------------------------------------
    for nm in C.SELLMEIER:
        for lam in (633e-9, 780e-9, 1.03e-6, 1.31e-6):
            mine = C.n_sellmeier(nm, lam)
            lib = float(G.get_glass_index(nm, lam))
            out['glass'][f'{nm}@{lam:.3e}'] = dict(
                mine=mine, lib=lib, diff=abs(mine - lib))
    worst = max(v['diff'] for v in out['glass'].values())
    print(f'CONTROL 1  glass index, max |mine - lumenairy| = {worst:.3e}')

    # -- 2/3/4 per fixture ------------------------------------------------
    for fx in C.fixtures():
        pres = fx.prescription()
        surfs = surfaces_from_prescription(pres)
        lam = fx.lam
        # A fan of rays at the fixture's own input directions.
        nh = 61
        h = np.linspace(-fx.semi * 0.98, fx.semi * 0.98, nh)
        zeros = np.zeros_like(h)
        L0, M0, N0 = fx.input_dirs(h.shape)
        rb = la.raytrace.RayBundle(
            x=h.copy(), y=zeros.copy(), z=zeros.copy(),
            L=L0.copy(), M=M0.copy(), N=N0.copy(), wavelength=lam,
            alive=np.ones(nh, bool), opd=zeros.copy())
        res = trace(rb, surfs, lam)
        img = res.image_rays
        ev = res.at_exit_vertex()
        n_ex = fx.n_exit()

        st = C.trace3d(h.copy(), zeros.copy(), zeros.copy(),
                       L0.copy(), M0.copy(), N0.copy(), zeros.copy(),
                       fx.oracle_surfaces())
        vt = C.to_vertex(st, n_ex, fx.oracle_surfaces()[-1]['zv'])
        alive = np.asarray(img.alive, bool)

        def mx(a, b):
            return float(np.nanmax(np.abs(np.asarray(a)[alive]
                                          - np.asarray(b)[alive])))

        rec = dict(
            note=fx.note, n_exit=n_ex, lam=lam, semi=fx.semi,
            n_alive=int(alive.sum()),
            surface_dx=mx(st['x'], img.x), surface_dz=mx(st['z'] - fx.oracle_surfaces()[-1]['zv'], img.z),
            surface_dL=mx(st['L'], img.L), surface_dN=mx(st['N'], img.N),
            surface_dopl=mx(st['opl'], img.opd),
            vertex_dx=mx(vt['x'], ev.x), vertex_dopl=mx(vt['opl'], ev.opd),
            vertex_dz=float(np.nanmax(np.abs(np.asarray(ev.z)[alive]))),
            sag_rim=float(C.sag(np.array([fx.semi * 0.98]),
                                np.array([0.0]),
                                fx.oracle_surfaces()[-1])[0]),
            sag_rim_waves=float(abs(C.sag(np.array([fx.semi * 0.98]),
                                          np.array([0.0]),
                                          fx.oracle_surfaces()[-1])[0])
                                * n_ex / lam),
            slope_vs_cosine=float(np.nanmax(np.abs(
                np.asarray(img.L)[alive] / np.asarray(img.N)[alive]
                - np.asarray(img.L)[alive]))),
        )
        if not fx.mirror_last:
            rec['best_focus_m'] = fx.best_focus()
        out['fixtures'][fx.key] = rec
        print(f"[{fx.key:17s}] surf dx={rec['surface_dx']:.2e} "
              f"dopl={rec['surface_dopl']:.2e} | vert dx={rec['vertex_dx']:.2e} "
              f"dopl={rec['vertex_dopl']:.2e} | sag_rim="
              f"{rec['sag_rim_waves']:.3f} waves | u-L="
              f"{rec['slope_vs_cosine']:.2e}")

    worst_s = max(v['surface_dx'] for v in out['fixtures'].values())
    worst_so = max(v['surface_dopl'] for v in out['fixtures'].values())
    worst_v = max(v['vertex_dx'] for v in out['fixtures'].values())
    worst_vo = max(v['vertex_dopl'] for v in out['fixtures'].values())
    out['summary'] = dict(max_surface_dx=worst_s, max_surface_dopl=worst_so,
                          max_vertex_dx=worst_v, max_vertex_dopl=worst_vo,
                          max_glass_diff=worst)
    print(f'CONTROL 2  last-surface state:  max dx={worst_s:.3e} m  '
          f'max dOPL={worst_so:.3e} m')
    print(f'CONTROL 3  exit-vertex state:   max dx={worst_v:.3e} m  '
          f'max dOPL={worst_vo:.3e} m')
    C.dump(out, 'probe_v0_controls')


if __name__ == '__main__':
    main()

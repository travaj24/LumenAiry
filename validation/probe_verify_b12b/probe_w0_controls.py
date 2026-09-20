"""VERIFY-WP-B12b probe W0 -- controls on MY oracle before anything is
scored with it.

Four controls, each an INDEPENDENT reading rather than a restatement:

1. my 3-D tracer's LAST-SURFACE state against ``lumenairy.raytrace.trace``'s
   ``image_rays`` (height, optical path), on every fixture -- including the
   biconic, the XY-polynomial freeform, the field-frame decentred one and the
   mirror, which is what makes this oracle able to score them;
2. my EXIT-VERTEX state against ``TraceResult.at_exit_vertex()`` -- the
   library's OTHER vertex operator, a different implementation;
3. the slope-vs-direction-cosine trap: ``max|u - L|`` on the same rays, so
   the agreements above are shown to be real comparisons and not two names
   for the same array;
4. the diffraction oracle's own convergence floor: halve the ray quadrature,
   halve the propagation refinement.

Usage::

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    LUMENAIRY_MEM_BUDGET_MB=4096 PYTHONPATH=<tree> \\
    python validation/probe_verify_b12b/probe_w0_controls.py

Author: VERIFY-WP-B12b
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vb12b_common import (  # noqa: E402
    assert_tree,
    dump,
    env_block,
    fixtures,
    gbd_surfaces,
)


def main():
    assert_tree()
    import lumenairy as la
    from lumenairy.raytrace import RayBundle, trace

    out = dict(env=env_block(), rows=[], converge=[])
    for key, fx in fixtures().items():
        surfs = gbd_surfaces(fx.prescription())
        n_r, n_az = 21, 8
        r = np.linspace(fx.semi / (2 * n_r), fx.semi * 0.98, n_r)
        az = (np.arange(n_az) + 0.5) * (np.pi / n_az)
        R, A = np.meshgrid(r, az, indexing='ij')
        h = (R * np.cos(A)).ravel()
        y = (R * np.sin(A)).ravel()
        z = np.zeros_like(h)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tr = trace(RayBundle(x=h.copy(), y=y.copy(), z=z.copy(),
                                 L=z.copy(), M=z.copy(),
                                 N=np.ones_like(h), wavelength=fx.lam,
                                 alive=np.ones(h.size, bool), opd=z.copy()),
                       surfs, fx.lam)
            img = tr.image_rays
            ev = tr.at_exit_vertex()
        alive = np.asarray(img.alive, bool)

        # -- my own 3-D trace, same rays -----------------------------------
        osurfs = fx.oracle_surfaces()
        from vb12b_common import to_vertex, trace3d
        st = trace3d(h, y, z, z, z, np.ones_like(h), z, osurfs)
        mv = to_vertex(st, fx.n_exit(), osurfs[-1]['zv'])
        zloc = st['z'] - osurfs[-1]['zv']

        def mx(a, b):
            d = np.abs(np.asarray(a) - np.asarray(b))[alive]
            return float(np.nanmax(d)) if d.size else float('nan')

        row = dict(
            key=key, note=fx.note, n_rays=int(h.size),
            n_alive=int(alive.sum()),
            surface_dx=mx(st['x'], img.x), surface_dy=mx(st['y'], img.y),
            surface_dz=mx(zloc, img.z), surface_dopl=mx(st['opl'], img.opd),
            surface_dL=mx(st['L'], img.L), surface_dN=mx(st['N'], img.N),
            vertex_dx=mx(mv['x'], ev.x), vertex_dy=mx(mv['y'], ev.y),
            vertex_dopl=mx(mv['opl'], ev.opd),
            sag_max=float(np.nanmax(np.abs(zloc[alive]))),
            sag_waves=float(np.nanmax(np.abs(zloc[alive])) / fx.lam),
            n_exit=fx.n_exit(),
            exit_N_sign=float(np.sign(np.nanmedian(st['N'][alive]))),
            # the trap: the unreduced slope is NOT the direction cosine
            slope_vs_cosine=float(np.nanmax(np.abs(
                (st['L'] / st['N'] - st['L'])[alive]))),
        )
        out['rows'].append(row)
        print(f"{key:15s} surf dx {row['surface_dx']:.3e} dopl "
              f"{row['surface_dopl']:.3e} | vertex dx {row['vertex_dx']:.3e} "
              f"dopl {row['vertex_dopl']:.3e} | sag {row['sag_waves']:.3f} wv "
              f"| N sign {row['exit_N_sign']:+.0f} "
              f"| u-L {row['slope_vs_cosine']:.3e}")

    # -- the diffraction oracle's own floor -------------------------------
    for key in ('conic', 'asph', 'flatbase_asph', 'bicon', 'freeform',
                'fieldframe', 'flat_last'):
        fx = fixtures()[key]
        zf = fx.best_focus()
        cv = fx.oracle_converge(zf)
        cv.update(key=key, z_focus=zf, airy=fx.airy_radius(),
                  na=fx.numerical_aperture(), dx_over_airy=fx.dx / fx.airy_radius())
        out['converge'].append(cv)
        print(f"{key:15s} z_focus {zf * 1e3:.4f} mm  NA {cv['na']:.4f}  "
              f"airy {cv['airy'] * 1e6:.2f} um  dx/airy "
              f"{cv['dx_over_airy']:.3f}  src-halved infid "
              f"{cv['src_halved_infidelity']:.3e}  grid-halved infid "
              f"{cv['grid_halved_infidelity']:.3e}")

    dump(out, 'probe_w0_controls')
    print('lumenairy version', la.__version__)


if __name__ == '__main__':
    main()

"""WP-B12b probe A -- the mechanism: the deleted in-line conic-sag copy against
the shared exit-vertex projection, per beamlet, on every surface class the GBD
per-surface path can reach.

For each fixture this runs the SAME differential primitive twice -- once on the
last surface (``reference='surface'``, the plane the pre-WP-B12b GBD consumed)
and once on the exit-vertex plane (``reference='exit_vertex'``, what it consumes
now) -- and reads three things off the pair:

* ``sag_shared``   -- the sag the shared projection actually applied, recovered
  from the height move as ``(x_s - x_v) / ux`` (and cross-checked against the
  package's own ``_surface_sag_xy``);
* ``sag_inline``   -- the deleted copy, reproduced in ``gbdproj_common``;
* the resulting per-beamlet HEIGHT and OPTICAL-PATH error of the old code.

The optical-path error is the one that matters to a coherent beamlet sum: it is
``(sag_inline - n_exit*sign(N)*sag_shared) * sec`` in metres, reported in waves.
A conic last surface must read ZERO to the floor (the copy was exact there); an
aspheric / biconic / freeform / field-frame one must not; a flat one must read
exactly 0.0; and a mirror must show the SIGN defect.

Everything here is measured against the library's own
``TraceResult.at_exit_vertex`` as well, so the reading needs no diffraction
model at all.

Run: OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1, PYTHONPATH pinned.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbdproj_common import (  # noqa: E402
    FIXTURES,
    MECHANISM_ONLY,
    NONSYM,
    assert_tree,
    build_tag,
    dump,
    gbd_surfaces,
    inline_conic_sag,
)

ROOT = os.environ.get('B12B_TREE', os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def _fan(semi, n=41, n_az=8):
    """A collimated 2-D fan over the clear aperture.

    Radii x azimuths, NOT a meridional line: a BICONIC or an XY-polynomial
    freeform departs from the rotationally-symmetric conic only off the x
    axis, so a y = 0 fan would read the biconic defect as exactly zero and
    the probe would report a false negative.  ``n_az=1`` gives the meridional
    fan back (used for the WP-B12 probe-D cross-check).
    """
    r = np.linspace(semi / (2 * n), semi * 0.98, n)
    az = (np.arange(n_az) + 0.5) * (np.pi / n_az) if n_az > 1         else np.zeros(1)
    R, A = np.meshgrid(r, az, indexing='ij')
    return (R * np.cos(A)).ravel(), (R * np.sin(A)).ravel()


def _measure(presc, lam, semi, tag, note, fan=None):
    """One fixture: the two reference planes, the two sags, the two errors."""
    from lumenairy.raytrace import differential as D
    from lumenairy.raytrace.exit_vertex import resolve_exit_index
    from lumenairy.raytrace.surface import _surface_sag_xy

    surfs = gbd_surfaces(presc)
    h, yy = _fan(semi, **(fan or {}))
    z = np.zeros_like(h)
    # Which backend does GBD's jacobian='auto' actually use here?
    backend = 'analytic'
    fns = [D.ray_transfer_jacobian_analytic, D.ray_transfer_jacobian]
    dt = dv = None
    for i, fn in enumerate(fns):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                dt = fn(h.copy(), yy.copy(), z.copy(), z.copy(), surfs, lam,
                        per_surface=True)
                dv = fn(h.copy(), yy.copy(), z.copy(), z.copy(), surfs, lam,
                        per_surface=True, reference='exit_vertex')
            backend = ('analytic', 'fd')[i]
            break
        except NotImplementedError:
            continue
    assert dt is not None, f'{tag}: no differential backend succeeded'

    ok = np.asarray(dt.alive, dtype=bool)
    xs, ys = np.asarray(dt.x), np.asarray(dt.y)
    ux, uy = np.asarray(dt.ux), np.asarray(dt.uy)
    sec = np.sqrt(1.0 + ux * ux + uy * uy)
    nz = D._exit_direction_sign(surfs)
    n_exit = float(resolve_exit_index(surfs, lam, fn_name='probe_a'))

    # the sag the shared projection applied, three ways
    dxv = xs - np.asarray(dv.x)
    dyv = ys - np.asarray(dv.y)
    u2 = ux * ux + uy * uy
    with np.errstate(divide='ignore', invalid='ignore'):
        sag_from_height = np.where(u2 > 1e-24, (dxv * ux + dyv * uy) / u2,
                                   np.nan)
    sag_from_opd = ((np.asarray(dt.opd) - np.asarray(dv.opd))
                    / (n_exit * nz * sec))
    sag_kernel = np.asarray(_surface_sag_xy(xs, ys, surfs[-1]),
                            dtype=np.float64)
    sag_inline = np.asarray(inline_conic_sag(surfs[-1], xs, ys))

    # the OPL the old code applied vs the OPL the projection applies.  The old
    # code folded -sag_inline into the leg t = (z_image - sag)*sec, i.e. it
    # subtracted sag_inline*sec of path with n = 1 and sign = +1; the shared
    # projection subtracts n_exit*sign(N)*sag*sec.
    d_opl = (sag_inline - n_exit * nz * sag_kernel) * sec
    d_h = np.abs(sag_inline - sag_kernel) * np.hypot(ux, uy)

    def _mx(a):
        a = np.asarray(a)[ok]
        a = a[np.isfinite(a)]
        return float(np.abs(a).max()) if a.size else float('nan')

    return {
        'note': note,
        'backend': backend,
        'n_rays_alive': int(ok.sum()),
        'n_exit': n_exit,
        'exit_direction_sign': float(nz),
        'sag_vanishes_structurally': bool(
            D._last_surface_sag_vanishes(surfs[-1])),
        'max_true_sag_m': _mx(sag_kernel),
        'max_true_sag_waves': _mx(sag_kernel) / lam,
        # controls: the projection's own three readings of the same sag
        'control_sag_height_vs_kernel_m': _mx(sag_from_height - sag_kernel),
        'control_sag_opd_vs_kernel_m': _mx(sag_from_opd - sag_kernel),
        # the defect
        'inline_sag_error_m': _mx(sag_inline - sag_kernel),
        'inline_sag_error_waves': _mx(sag_inline - sag_kernel) / lam,
        'inline_sag_error_frac_of_sag': (
            _mx(sag_inline - sag_kernel) / _mx(sag_kernel)
            if _mx(sag_kernel) > 0 else 0.0),
        'inline_opl_error_m': _mx(d_opl),
        'inline_opl_error_waves': _mx(d_opl) / lam,
        'inline_height_error_m': _mx(d_h),
    }


def main():
    assert_tree(ROOT)
    import lumenairy as la
    from lumenairy.raytrace import trace
    from lumenairy.raytrace.trace import _make_bundle

    out = {'build': build_tag(), 'version': la.__version__, 'fixtures': {}}

    for key, fx in list(FIXTURES.items()) + list(MECHANISM_ONLY.items()):
        fx.register()
        out['fixtures'][key] = _measure(
            fx.prescription(), fx.lam, fx.semi, key, fx.note)

    # The WP-B12 probe-D cross-check: the SAME optic, the SAME meridional fan
    # (401 heights, y = 0), so section 5.1's reading is reproduced like for
    # like on a tree where the in-line copy no longer exists.
    for key in ('b12d_conic', 'b12d_asphere'):
        fx = MECHANISM_ONLY[key]
        fx.register()
        out['fixtures'][key + '_meridional_401'] = _measure(
            fx.prescription(), fx.lam, fx.semi, key, fx.note,
            fan=dict(n=401, n_az=1))

    for key, spec in NONSYM.items():
        out['fixtures'][key] = _measure(
            spec['presc'](), spec['lam'], spec['semi'], key, spec['note'])

    # ---- an independent control on the shared projection itself: the
    # library's own ray-bundle operator, TraceResult.at_exit_vertex ----
    fx = FIXTURES['asphere_field']
    fx.register()
    surfs = gbd_surfaces(fx.prescription())
    h, yy = _fan(fx.semi, n=25, n_az=8)
    z = np.zeros_like(h)
    rb = _make_bundle(h.copy(), yy.copy(), z.copy(), z.copy(), fx.lam)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = trace(rb, surfs, fx.lam)
        ev = res.at_exit_vertex()
        from lumenairy.raytrace import differential as D
        dv = D.ray_transfer_jacobian(
            h.copy(), yy.copy(), z.copy(), z.copy(), surfs, fx.lam,
            per_surface=True, reference='exit_vertex')
    alive = np.asarray(ev.alive, bool) & np.asarray(dv.alive, bool)
    out['control_at_exit_vertex_asphere'] = {
        'max_height_gap_m': float(np.abs(
            np.asarray(ev.x)[alive] - np.asarray(dv.x)[alive]).max()),
        'max_opl_gap_m': float(np.abs(
            np.asarray(ev.opd)[alive] - np.asarray(dv.opd)[alive]).max()),
        'n_rays': int(alive.sum()),
    }

    hdr = (f"{'fixture':16s} {'backend':9s} {'sag[wv]':>9s} "
           f"{'inline err[wv]':>15s} {'frac of sag':>12s} "
           f"{'OPL err[wv]':>12s} {'dh[m]':>11s}")
    print(hdr)
    print('-' * len(hdr))
    for k, v in out['fixtures'].items():
        print(f"{k:16s} {v['backend']:9s} {v['max_true_sag_waves']:9.3f} "
              f"{v['inline_sag_error_waves']:15.4f} "
              f"{v['inline_sag_error_frac_of_sag']:12.4f} "
              f"{v['inline_opl_error_waves']:12.4f} "
              f"{v['inline_height_error_m']:11.3e}")
    c = out['control_at_exit_vertex_asphere']
    print(f"\ncontrol -- the projection vs TraceResult.at_exit_vertex on the "
          f"asphere: height {c['max_height_gap_m']:.3e} m, "
          f"OPL {c['max_opl_gap_m']:.3e} m ({c['n_rays']} rays)")

    dump(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      f'probe_a_sag_{sys.platform}_'
                      f'{sys.version_info.major}{sys.version_info.minor}.json'),
         out)


if __name__ == '__main__':
    main()

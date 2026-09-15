"""WP-B12 probe A -- the mechanism, with numbers, on six fixtures.

For each fixture:

* controls -- the oracle's Sellmeier index against ``lumenairy.glass``, the
  oracle's exit-VERTEX state against ``TraceResult.at_exit_vertex()``, the
  oracle's LAST-SURFACE state against ``ray_transfer_jacobian``, and the
  slope-vs-direction-cosine trap (a check that reads zero if the two states
  being compared were secretly the same quantity);
* the defect -- ``ray_transfer_jacobian`` / ``..._analytic`` state against
  ``at_exit_vertex()``: |dx| in metres, |d(opd)| in metres AND in waves, and
  the last surface's sag at the marginal ray.

Run with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b12_common import (  # noqa: E402
    FIXTURES,
    assert_tree,
    build_tag,
    dump,
)

ROOT = os.environ.get('B12_TREE', os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def main():
    assert_tree(ROOT)
    import lumenairy as la
    from lumenairy import raytrace as rt
    from lumenairy.propagators.fga import _system_na
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.differential import (
        ray_transfer_jacobian,
        ray_transfer_jacobian_analytic,
    )

    out = {'build': build_tag(), 'fixtures': {}}
    for key, fx in FIXTURES.items():
        rec = {'note': fx.note, 'glass': fx.glass, 'lam_m': fx.lam,
               'R1': fx.R1, 'R2': fx.R2, 't': fx.t, 'semi': fx.semi,
               'N': fx.N, 'dx': fx.dx, 'w0': fx.w0}
        p = fx.prescription()
        surfs = [s for s in surfaces_from_prescription(p)]
        # FGA zeroes the last surface's transfer; mirror that here.
        import copy as _copy
        surfs = [_copy.copy(s) for s in surfs]
        surfs[-1].thickness = 0.0

        # --- control 1: dispersion --------------------------------------
        # the oracle's own dispersion (typed-in Sellmeier, or the fixture's
        # model index) against the library's registry
        rec['dn_sellmeier_vs_library'] = abs(
            fx.index() - float(la.get_glass_index(fx.glass, fx.lam)))

        # --- the ray fan -------------------------------------------------
        n = 801
        h = np.linspace(fx.semi / (2 * n), fx.semi * (1.0 - 1.0 / (2 * n)), n)
        zeros = np.zeros(n)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dt = ray_transfer_jacobian(h.copy(), zeros.copy(), zeros.copy(),
                                       zeros.copy(), surfs, fx.lam)
            try:
                dta = ray_transfer_jacobian_analytic(
                    h.copy(), zeros.copy(), zeros.copy(), zeros.copy(),
                    surfs, fx.lam)
            except NotImplementedError as exc:
                dta = None
                rec['analytic'] = f'NotImplementedError: {exc}'
            bundle = rt.RayBundle(
                x=h.copy(), y=zeros.copy(), z=zeros.copy(), L=zeros.copy(),
                M=zeros.copy(), N=np.ones(n), wavelength=fx.lam,
                alive=np.ones(n, bool), opd=zeros.copy())
            res = rt.trace(bundle, surfs, fx.lam)
            img = res.image_rays
            ex = res.at_exit_vertex()
        ok = np.asarray(ex.alive, bool) & np.asarray(dt.alive, bool)
        rec['n_rays_alive'] = int(ok.sum())
        assert ok.sum() > n // 2, (key, ok.sum())

        # --- control 2: the oracle vs the library, both planes ------------
        _h, xs, us, ols, xv, uv, olv, n_out = fx.oracle_exit(n_h=n)
        # the oracle launches on the SAME heights
        assert np.allclose(_h, h, rtol=0, atol=0), 'launch heights differ'
        rec['n_exit_oracle'] = float(n_out)
        rec['oracle_vs_at_exit_vertex'] = {
            'dx_m': float(np.abs(np.asarray(ex.x)[ok] - xv[ok]).max()),
            'du': float(np.abs((np.asarray(ex.L) / np.asarray(ex.N))[ok]
                               - uv[ok]).max()),
            'dopl_m': float(np.abs(np.asarray(ex.opd)[ok] - olv[ok]).max()),
        }
        rec['oracle_vs_image_rays'] = {
            'dx_m': float(np.abs(np.asarray(img.x)[ok] - xs[ok]).max()),
            'dopl_m': float(np.abs(np.asarray(img.opd)[ok] - ols[ok]).max()),
        }
        rec['oracle_vs_ray_transfer_jacobian'] = {
            'dx_m': float(np.abs(np.asarray(dt.x)[ok] - xs[ok]).max()),
            'du': float(np.abs(np.asarray(dt.ux)[ok] - us[ok]).max()),
            'dopl_m': float(np.abs(np.asarray(dt.opd)[ok] - ols[ok]).max()),
        }
        # --- control 3: the slope-vs-direction-cosine trap ---------------
        L = np.asarray(ex.L)[ok]
        rec['slope_vs_cosine_max'] = float(
            np.abs((np.asarray(ex.L) / np.asarray(ex.N))[ok] - L).max())

        # --- the defect ---------------------------------------------------
        lam = fx.lam
        for name, d in (('fd', dt), ('analytic', dta)):
            if d is None:
                continue
            dx = np.abs(np.asarray(d.x)[ok] - np.asarray(ex.x)[ok])
            dop = np.abs(np.asarray(d.opd)[ok] - np.asarray(ex.opd)[ok])
            rec[f'defect_{name}'] = {
                'max_dx_m': float(dx.max()),
                'max_dopd_m': float(dop.max()),
                'max_dopd_waves': float(dop.max() / lam),
                'rim_dopd_waves': float(dop[-1] / lam),
            }
        # --- the repair: reference='exit_vertex' reproduces at_exit_vertex ---
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                dv = ray_transfer_jacobian(
                    h.copy(), zeros.copy(), zeros.copy(), zeros.copy(), surfs,
                    fx.lam, reference='exit_vertex')
                dva = (None if dta is None else ray_transfer_jacobian_analytic(
                    h.copy(), zeros.copy(), zeros.copy(), zeros.copy(), surfs,
                    fx.lam, reference='exit_vertex'))
            rec['repair_fd_vs_at_exit_vertex'] = {
                'dx_m': float(np.abs(np.asarray(dv.x)[ok]
                                     - np.asarray(ex.x)[ok]).max()),
                'dopd_m': float(np.abs(np.asarray(dv.opd)[ok]
                                       - np.asarray(ex.opd)[ok]).max()),
                'bit_identical_to_surface_arm': bool(
                    np.array_equal(np.asarray(dv.x), np.asarray(dt.x))
                    and np.array_equal(np.asarray(dv.opd),
                                       np.asarray(dt.opd))
                    and np.array_equal(np.asarray(dv.jacobian),
                                       np.asarray(dt.jacobian))),
            }
            if dva is not None:
                rec['repair_analytic_vs_at_exit_vertex'] = {
                    'dx_m': float(np.abs(np.asarray(dva.x)[ok]
                                         - np.asarray(ex.x)[ok]).max()),
                    'dopd_m': float(np.abs(np.asarray(dva.opd)[ok]
                                           - np.asarray(ex.opd)[ok]).max()),
                    'bit_identical_to_surface_arm': bool(
                        np.array_equal(np.asarray(dva.x), np.asarray(dta.x))
                        and np.array_equal(np.asarray(dva.opd),
                                           np.asarray(dta.opd))
                        and np.array_equal(np.asarray(dva.jacobian),
                                           np.asarray(dta.jacobian))),
                }
                # masked to ALIVE base rays: a vignetted FD companion makes
                # the FD Jacobian row 0 (nan_to_num) where the analytic path,
                # which carries no aperture, still returns a number.
                jv, jva = (np.asarray(dv.jacobian)[ok],
                           np.asarray(dva.jacobian)[ok])
                jt, jta = (np.asarray(dt.jacobian)[ok],
                           np.asarray(dta.jacobian)[ok])
                rec['repair_fd_vs_analytic_jacobian_rel'] = float(
                    np.abs(jv - jva).max() / np.abs(jva).max())
                rec['surface_fd_vs_analytic_jacobian_rel'] = float(
                    np.abs(jt - jta).max() / np.abs(jta).max())
        except TypeError as exc:
            rec['repair'] = f'reference= not present in this tree: {exc}'
        rec['last_surface_sag_at_rim_m'] = float(np.asarray(img.z)[ok][-1])
        rec['last_surface_sag_at_rim_waves'] = float(
            abs(np.asarray(img.z)[ok][-1]) / lam)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            rec['system_na'] = float(_system_na(p, lam))
        rec['best_focus_m'] = fx.best_focus()
        out['fixtures'][key] = rec
        print(f'== {key}: NA {rec["system_na"]:.4f}  focus '
              f'{rec["best_focus_m"] * 1e6:.2f} um  sag@rim '
              f'{rec["last_surface_sag_at_rim_waves"]:.2f} waves  '
              f'defect_fd {rec["defect_fd"]["max_dopd_waves"]:.3f} waves / '
              f'{rec["defect_fd"]["max_dx_m"] * 1e6:.4f} um')
    dump(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      f'probe_a_mechanism_{sys.platform}_'
                      f'{sys.version_info.major}{sys.version_info.minor}.json'),
         out)


if __name__ == '__main__':
    main()

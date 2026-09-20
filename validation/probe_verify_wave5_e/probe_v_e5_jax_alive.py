"""VERIFY-WAVE5-E / E5-new: does the JAX analytic path carry vignetting?

The item-E report records, out of scope, that ``ray_transfer_jacobian_analytic``
reports every ray alive on its JAX path.  This is an APERTURE LADDER over the
same fan on three paths -- the FD bundle tracer, the NumPy analytic backend and
the JAX analytic backend -- with the clear aperture shrunk rung by rung, so the
claim is measured as a function of the aperture rather than at one cell.

It also records, for the characterisation the defect needs, whether the
``opd``/state the JAX path returns for a ray the other two paths kill is
finite and how far it is from the NumPy answer.
"""
import copy
import json
import sys
import warnings

import numpy as np

import lumenairy
from lumenairy import raytrace as rt
from lumenairy.raytrace import surfaces_from_prescription
from lumenairy.raytrace.differential import ray_transfer_jacobian, ray_transfer_jacobian_analytic

_LAM = 1.03e-6
_T = 0.55e-3
_GLASS = 'N-SSK8'
_N_RAYS = 201
_HALF = 0.45e-3                     # the fan half-width, FIXED
_SEMIS = (0.45e-3, 0.30e-3, 0.22e-3, 0.15e-3, 0.10e-3, 0.06e-3)


def _presc(semi):
    s1 = {'radius': np.inf, 'conic': 0.0, 'thickness': _T,
          'glass_before': 'air', 'glass_after': _GLASS, 'semi_diameter': semi}
    s2 = {'radius': -1.05e-3, 'conic': -0.35, 'thickness': 0.0,
          'glass_before': _GLASS, 'glass_after': 'air',
          'semi_diameter': semi}
    return {'name': 'v_jax_alive', 'aperture_diameter': 2 * semi,
            'surfaces': [s1, s2], 'thicknesses': [_T], 'stop_index': 0}


def _surfs(semi):
    s = [copy.copy(x) for x in surfaces_from_prescription(_presc(semi))]
    s[-1].thickness = 0.0
    return s


def _fan():
    h = np.linspace(-_HALF, _HALF, _N_RAYS)
    z = np.zeros(_N_RAYS)
    return h, z.copy(), z.copy(), z.copy()


def main():
    out = dict(lumenairy_file=lumenairy.__file__, python=sys.version.split()[0],
               platform=sys.platform, numpy=np.__version__,
               n_rays=_N_RAYS, fan_half_width_m=_HALF, rungs=[])
    try:
        import jax
        jax.config.update('jax_enable_x64', True)
        import jax.numpy as jnp
        out['jax'] = jax.__version__
    except ImportError:                                  # pragma: no cover
        out['jax'] = None
        jnp = None

    for semi in _SEMIS:
        surfs = _surfs(semi)
        h, y, ux, uy = _fan()
        rung = dict(semi_diameter_m=semi,
                    fan_half_over_semi=_HALF / semi)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # 1 -- the production bundle tracer
            nz = np.ones(_N_RAYS)
            b = rt.RayBundle(x=h.copy(), y=y.copy(), z=np.zeros(_N_RAYS),
                             L=ux * nz, M=uy * nz, N=nz, wavelength=_LAM,
                             alive=np.ones(_N_RAYS, bool),
                             opd=np.zeros(_N_RAYS))
            img = rt.trace(b, surfs, _LAM).image_rays
            rung['bundle_dead'] = int((~np.asarray(img.alive, bool)).sum())
            # 2 -- FD differential
            fd = ray_transfer_jacobian(h.copy(), y.copy(), ux.copy(),
                                       uy.copy(), surfs, _LAM,
                                       reference='surface')
            rung['fd_dead'] = int((~np.asarray(fd.alive, bool)).sum())
            # 3 -- NumPy analytic
            npa = ray_transfer_jacobian_analytic(
                h.copy(), y.copy(), ux.copy(), uy.copy(), surfs, _LAM,
                reference='surface')
            rung['numpy_analytic_dead'] = int(
                (~np.asarray(npa.alive, bool)).sum())
            # 4 -- JAX analytic
            if jnp is not None:
                jx = ray_transfer_jacobian_analytic(
                    jnp.asarray(h), jnp.asarray(y), jnp.asarray(ux),
                    jnp.asarray(uy), surfs, _LAM, reference='surface')
                ja = np.asarray(jx.alive, dtype=bool)
                rung['jax_analytic_dead'] = int((~ja).sum())
                rung['jax_alive_dtype'] = str(np.asarray(jx.alive).dtype)
                rung['jax_alive_all_true'] = bool(ja.all())
                dead_np = ~np.asarray(npa.alive, bool)
                if dead_np.any():
                    jo = np.asarray(jx.opd, dtype=float)[dead_np]
                    no = np.asarray(npa.opd, dtype=float)[dead_np]
                    rung['jax_opd_on_numpy_dead_finite'] = int(
                        np.count_nonzero(np.isfinite(jo)))
                    rung['jax_opd_on_numpy_dead_n'] = int(dead_np.sum())
                    rung['max_abs_jax_opd_on_dead'] = float(
                        np.nanmax(np.abs(jo)))
                    rung['max_abs_diff_jax_vs_numpy_on_dead'] = float(
                        np.nanmax(np.abs(jo - no)))
                live = np.asarray(npa.alive, bool)
                if live.any():
                    rung['max_abs_diff_jax_vs_numpy_on_live_opd'] = float(
                        np.nanmax(np.abs(
                            np.asarray(jx.opd, float)[live] -
                            np.asarray(npa.opd, float)[live])))
        out['rungs'].append(rung)

    out['verdict'] = dict(
        jax_dead_total=sum(r.get('jax_analytic_dead', 0)
                           for r in out['rungs']),
        numpy_analytic_dead_total=sum(r['numpy_analytic_dead']
                                      for r in out['rungs']),
        fd_dead_total=sum(r['fd_dead'] for r in out['rungs']),
        bundle_dead_total=sum(r['bundle_dead'] for r in out['rungs']),
        jax_monotone_in_aperture=len({r.get('jax_analytic_dead')
                                      for r in out['rungs']}) > 1,
    )
    print(json.dumps(out, indent=1))


main()

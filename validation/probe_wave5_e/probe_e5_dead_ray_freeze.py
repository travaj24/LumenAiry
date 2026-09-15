"""WAVE5-E item E5 (VERIFY-WP-B12 open item O-1) -- the dead-ray freeze in
``differential._project_to_exit_vertex_plane``, measured on every fixture class
and on every backend.

WHAT O-1 SAYS.  ``lumenairy.raytrace.exit_vertex.exit_vertex_transfer`` freezes
a dead ray -- position, direction and OPL -- "exactly", because a vignetted ray
never reached the vertex plane.  ``_project_to_exit_vertex_plane`` applied its
arithmetic to every row, moving dead rays' ``opd`` by up to 1.96e-05 m on a fan
clipped at the last surface.  Unobservable today (all four ``fga.py`` consumers
zero the dead beamlets first) but a divergence between the module's two
vertex-plane operators.

WHAT THIS PROBE READS, per fixture and per backend:

  D1. ``max|projected - surface|`` over DEAD rows, on x / y / opd and on the
      Jacobian.  Before the fix this is the 1e-05-scale drift; after it is
      exactly 0.0, i.e. the projection is the identity on dead rows.
  D2. Whether the dead rows equal ``TraceResult.at_exit_vertex()``'s own frozen
      state -- the operator the fix is being brought into line with.
  D3. md5 of the ALIVE rows' x / y / ux / uy / opd / Jacobian.  Run in the PRE
      tree and in this one, these must be IDENTICAL: the mask must not move a
      single live bit.
  D4. The non-finite census on dead rows, before and after -- the projection
      used to ``nan_to_num`` the whole Jacobian including dead rows, and the
      frozen rows now carry the surface-reference Jacobian instead.

Backends: the FD bundle tracer (``ray_transfer_jacobian``), the analytic
backend (``ray_transfer_jacobian_analytic``, which routes to the numba kernel
or the NumPy dual), and the JAX path of the same analytic entry point.  All
three reach ONE projection implementation, which is the point of the test.

Usage:  python probe_e5_dead_ray_freeze.py <out.json>
        (PYTHONPATH selects the tree; ``lumenairy.__file__`` is recorded)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'probe_verify_b12'))

import vb12_common as C  # noqa: E402

#: The fan runs this many times past the clear aperture, so most rays vignette.
_OVER = 3.0
_NRAYS = 201


def _md5(*arrays):
    h = hashlib.md5()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a), dtype=np.float64)
                 .tobytes())
    return h.hexdigest()


def _states(dt):
    return (np.asarray(dt.x, dtype=np.float64),
            np.asarray(dt.y, dtype=np.float64),
            np.asarray(dt.ux, dtype=np.float64),
            np.asarray(dt.uy, dtype=np.float64),
            np.asarray(dt.opd, dtype=np.float64),
            np.asarray(dt.jacobian, dtype=np.float64))


def _one(fx, backend):
    from lumenairy import raytrace as rt
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.differential import (
        ray_transfer_jacobian,
        ray_transfer_jacobian_analytic,
    )
    surfs = surfaces_from_prescription(fx.prescription())
    n = _NRAYS
    h = np.linspace(-_OVER * fx.semi, _OVER * fx.semi, n)
    z = np.zeros(n)

    def _call(reference):
        if backend == 'fd':
            return ray_transfer_jacobian(
                h.copy(), z.copy(), z.copy(), z.copy(), surfs, fx.lam,
                reference=reference)
        if backend == 'analytic':
            return ray_transfer_jacobian_analytic(
                h.copy(), z.copy(), z.copy(), z.copy(), surfs, fx.lam,
                reference=reference)
        if backend == 'jax':
            import jax.numpy as jnp
            return ray_transfer_jacobian_analytic(
                jnp.asarray(h), jnp.asarray(z), jnp.asarray(z),
                jnp.asarray(z), surfs, fx.lam, reference=reference)
        raise AssertionError(backend)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        srf = _call('surface')
        vtx = _call('exit_vertex')
    sx, sy, sux, suy, sopd, sj = _states(srf)
    vx, vy, vux, vuy, vopd, vj = _states(vtx)
    alive = np.asarray(srf.alive, dtype=bool)
    # The freeze set is REACHED THE LAST SURFACE, taken from the production
    # ray-bundle trace -- NOT the differential's ``alive``, which on the FD
    # backend is ``base_alive & companion_alive`` and so also drops rays whose
    # 9-ray companion bundle vignettes while the base ray landed
    # (VERIFY-WP-B12 O-4).  Those rays DID reach the vertex plane and
    # ``at_exit_vertex`` projects them.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        n_ = n
        nz_ = 1.0 / np.sqrt(1.0 + z ** 2 + z ** 2)
        bnd = rt.RayBundle(x=h.copy(), y=z.copy(), z=z.copy(),
                           L=z.copy(), M=z.copy(), N=nz_,
                           wavelength=fx.lam, alive=np.ones(n_, bool),
                           opd=z.copy())
        res0 = rt.trace(bnd, surfs, fx.lam)
    reached = np.asarray(res0.image_rays.alive, dtype=bool)
    # The rows the freeze claim is ABOUT: the bundle trace says they never
    # reached the last surface AND this backend marks them dead.  The JAX path
    # of ``ray_transfer_jacobian_analytic`` reports every ray alive whatever
    # the aperture, so its set is EMPTY and the claim is vacuous there -- which
    # the probe records rather than scoring as a failure.
    dead = (~reached) & (~np.asarray(srf.alive, dtype=bool))

    def _dmax(a, b, m):
        if not m.any():
            return 0.0
        d = np.abs(np.asarray(a)[m] - np.asarray(b)[m])
        return float(np.nanmax(d)) if d.size else 0.0

    row = {
        'fixture': fx.key, 'backend': backend, 'n_rays': n,
        'n_dead': int(dead.sum()), 'n_alive': int(alive.sum()),
        'n_reached': int(reached.sum()),
        'n_missed': int((~reached).sum()),
        'n_backend_dead': int((~alive).sum()),
        'n_companion_only_dead': int((reached & ~alive).sum()),
        'freeze_set_empty': bool(not dead.any()),
        'flat_last_surface': bool(fx.flat_last),
        # D1 -- how far the projection moves a DEAD row
        'dead_dx': _dmax(vx, sx, dead),
        'dead_dy': _dmax(vy, sy, dead),
        'dead_dopd': _dmax(vopd, sopd, dead),
        'dead_djac': (float(np.nanmax(np.abs(
            vj[..., dead, :, :] - sj[..., dead, :, :])))
            if dead.any() and sj.shape[-3] == n else
            (float(np.nanmax(np.abs(vj[dead] - sj[dead])))
             if dead.any() else 0.0)),
        # D3 -- the ALIVE rows' bits
        'alive_md5_state': _md5(vx[reached], vy[reached], vux[reached],
                                vuy[reached], vopd[reached]),
        'alive_md5_jac': _md5(vj[..., reached, :, :] if vj.ndim == 4
                              else vj[reached]),
        'alive_moves_from_surface': _dmax(vx, sx, reached),
        'alive_moves_opd': _dmax(vopd, sopd, reached),
        # D4 -- the non-finite census on the frozen rows
        'dead_nonfinite_state_surface': int(np.count_nonzero(
            ~np.isfinite(sx[dead]) | ~np.isfinite(sopd[dead]))),
        'dead_nonfinite_state_vertex': int(np.count_nonzero(
            ~np.isfinite(vx[dead]) | ~np.isfinite(vopd[dead]))),
        'dead_nonfinite_jac_surface': int(np.count_nonzero(
            ~np.isfinite(sj[dead] if sj.ndim == 3 else sj[:, dead]))),
        'dead_nonfinite_jac_vertex': int(np.count_nonzero(
            ~np.isfinite(vj[dead] if vj.ndim == 3 else vj[:, dead]))),
        'alive_flags_unchanged': bool(np.array_equal(
            alive, np.asarray(vtx.alive, dtype=bool))),
    }
    row['dead_frozen'] = bool(row['dead_dx'] == 0.0
                              and row['dead_dy'] == 0.0
                              and row['dead_dopd'] == 0.0
                              and row['dead_djac'] == 0.0)

    # D2 -- against the ray-bundle operator this is being aligned with.
    if backend == 'fd':
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            b = rt.RayBundle(x=h.copy(), y=z.copy(), z=z.copy(), L=z.copy(),
                             M=z.copy(), N=np.ones(n), wavelength=fx.lam,
                             alive=np.ones(n, bool), opd=z.copy())
            res = rt.trace(b, surfs, fx.lam)
            ev = res.at_exit_vertex()
        img = res.image_rays
        bdead = ~np.asarray(img.alive, dtype=bool)
        row['bundle_n_dead'] = int(bdead.sum())
        row['bundle_freezes_dead'] = bool(
            np.array_equal(np.asarray(ev.x)[bdead], np.asarray(img.x)[bdead])
            and np.array_equal(np.asarray(ev.opd)[bdead],
                               np.asarray(img.opd)[bdead]))
    return row


def _jax_twin(fx):
    """The JAX arm with REAL dead rows, on the projection primitive itself.

    ``ray_transfer_jacobian_analytic``'s JAX path returns an all-alive mask
    (measured: 0 dead of 201 on every fixture, where the FD bundle tracer
    vignettes 134), so routing a fan through it does NOT exercise the freeze.
    The twin claim is therefore made where it is meaningful: ONE
    ``DifferentialTransfer`` carrying real vignetting, projected twice --
    once with NumPy arrays and once with the identical values as
    ``jax.numpy`` arrays -- and the two results compared field by field.
    """
    import jax.numpy as jnp

    from lumenairy import raytrace as rt
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.differential import (
        DifferentialTransfer,
        _project_to_exit_vertex_plane,
        ray_transfer_jacobian,
    )
    surfs = surfaces_from_prescription(fx.prescription())
    n = _NRAYS
    h = np.linspace(-_OVER * fx.semi, _OVER * fx.semi, n)
    z = np.zeros(n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        srf = ray_transfer_jacobian(h.copy(), z.copy(), z.copy(), z.copy(),
                                    surfs, fx.lam, reference='surface')
    alive = np.asarray(srf.alive, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nz_ = 1.0 / np.sqrt(1.0 + z ** 2 + z ** 2)
        bnd = rt.RayBundle(x=h.copy(), y=z.copy(), z=z.copy(), L=z.copy(),
                           M=z.copy(), N=nz_, wavelength=fx.lam,
                           alive=np.ones(n, bool), opd=z.copy())
        res0 = rt.trace(bnd, surfs, fx.lam)
    reached = np.asarray(res0.image_rays.alive, dtype=bool)
    fn = "probe_e5_jax_twin"
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        np_out = _project_to_exit_vertex_plane(srf, surfs, fx.lam, None, fn)
        jx_in = DifferentialTransfer(
            jacobian=jnp.asarray(srf.jacobian), x=jnp.asarray(srf.x),
            y=jnp.asarray(srf.y), ux=jnp.asarray(srf.ux),
            uy=jnp.asarray(srf.uy), opd=jnp.asarray(srf.opd),
            alive=jnp.asarray(alive))
        jx_out = _project_to_exit_vertex_plane(jx_in, surfs, fx.lam, None, fn)
    dead = (~reached) & (~alive)
    row = {'fixture': fx.key, 'backend': 'jax_twin', 'n_rays': n,
           'n_dead': int(dead.sum()), 'n_alive': int(alive.sum()),
           'n_reached': int(reached.sum()),
           'n_missed': int((~reached).sum()),
           'n_backend_dead': int((~alive).sum()),
           'n_companion_only_dead': int((reached & ~alive).sum()),
           'freeze_set_empty': bool(not dead.any()),
           'flat_last_surface': bool(fx.flat_last)}
    for name in ('x', 'y', 'opd'):
        a = np.asarray(getattr(np_out, name), dtype=np.float64)
        b = np.asarray(getattr(jx_out, name), dtype=np.float64)
        c = np.asarray(getattr(srf, name), dtype=np.float64)
        row['twin_d' + name] = float(np.nanmax(np.abs(a - b)))
        row['jax_dead_d' + name] = (float(np.nanmax(np.abs(b[dead] - c[dead])))
                                    if dead.any() else 0.0)
    ja = np.asarray(np_out.jacobian, dtype=np.float64)
    jb = np.asarray(jx_out.jacobian, dtype=np.float64)
    jc = np.asarray(srf.jacobian, dtype=np.float64)
    row['twin_djac'] = float(np.nanmax(np.abs(ja - jb)))
    # The XLA-vs-NumPy arithmetic floor, in ULP of the entries themselves --
    # NOT a property of the mask (the STATE is exactly identical on every
    # fixture; only the doublet's Jacobian moves, by one ULP).
    _sc = float(np.nanmax(np.abs(ja))) or 1.0
    row['twin_djac_ulp'] = row['twin_djac'] / float(np.spacing(_sc))
    row['jax_dead_djac'] = (float(np.nanmax(np.abs(jb[dead] - jc[dead])))
                            if dead.any() else 0.0)
    row['dead_frozen'] = bool(row['jax_dead_dx'] == 0.0
                              and row['jax_dead_dy'] == 0.0
                              and row['jax_dead_dopd'] == 0.0
                              and row['jax_dead_djac'] == 0.0)
    row['twin_state_identical'] = bool(
        row['twin_dx'] == 0.0 and row['twin_dy'] == 0.0
        and row['twin_dopd'] == 0.0)
    row['twin_identical'] = bool(row['twin_state_identical']
                                 and row['twin_djac'] == 0.0)
    row['alive_md5_state'] = _md5(
        np.asarray(jx_out.x, dtype=np.float64)[reached],
        np.asarray(jx_out.y, dtype=np.float64)[reached],
        np.asarray(jx_out.ux, dtype=np.float64)[reached],
        np.asarray(jx_out.uy, dtype=np.float64)[reached],
        np.asarray(jx_out.opd, dtype=np.float64)[reached])
    row['alive_md5_jac'] = _md5(jb[reached])
    return row


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else 'e5_dead_ray.json'
    import lumenairy as la
    res = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'platform': sys.platform, 'rows': []}
    print('lumenairy.__file__ =', la.__file__, flush=True)
    backends = ['fd', 'analytic']
    try:
        import jax
        # x64 BEFORE the first jax call: JAX defaults to float32, which alone
        # moves the projected state by ~1e-12 (3e-5 on the mirror) and makes
        # the twin comparison a dtype comparison.  Setting it here rather than
        # inside the twin arm also keeps the JIT cache from being built at one
        # precision and read at the other, which is ORDER-DEPENDENT: measured,
        # a late flip left the doublet's Jacobian 1 ULP apart on one run and
        # bit-identical on the next.
        jax.config.update('jax_enable_x64', True)
        backends.append('jax')
        backends.append('jax_twin')
        res['jax'] = jax.__version__
        res['jax_enable_x64'] = True
    except ImportError:
        res['jax'] = None
    for fx in C.fixtures():
        for backend in backends:
            try:
                row = (_jax_twin(fx) if backend == 'jax_twin'
                       else _one(fx, backend))
            except Exception as exc:                        # noqa: BLE001
                row = {'fixture': fx.key, 'backend': backend,
                       'err': '%s: %s' % (type(exc).__name__, exc)}
            res['rows'].append(row)
            if 'err' in row:
                print('%-18s %-9s ERROR %s' % (fx.key, backend, row['err']),
                      flush=True)
                continue
            print('%-18s %-9s dead=%-4d frozen=%-5s dopd=%.3e djac=%.3e '
                  'twin=%-5s alive_state_md5=%s'
                  % (fx.key, backend, row['n_dead'], row['dead_frozen'],
                     row.get('dead_dopd', row.get('jax_dead_dopd', 0.0)),
                     row.get('dead_djac', row.get('jax_dead_djac', 0.0)),
                     row.get('twin_state_identical', '-'),
                     row['alive_md5_state'][:12]), flush=True)
    ok = [r for r in res['rows'] if 'err' not in r]
    scored = [r for r in ok if not r.get('freeze_set_empty')]
    res['n_cells_vacuous'] = len(ok) - len(scored)
    res['all_dead_frozen'] = bool(scored) and all(
        r['dead_frozen'] for r in scored)
    tw = [r for r in ok if r['backend'] == 'jax_twin']
    res['all_jax_twins_state_identical'] = bool(tw) and all(
        r['twin_state_identical'] for r in tw)
    res['all_jax_twins_bit_identical'] = bool(tw) and all(
        r['twin_identical'] for r in tw)
    res['max_twin_djac_ulp'] = max((r['twin_djac_ulp'] for r in tw),
                                   default=None)
    res['n_cells'] = len(ok)
    print('ALL DEAD ROWS FROZEN:', res['all_dead_frozen'],
          '(%d scored cells, %d vacuous -- the backend marks nothing dead)'
          % (len(scored), res['n_cells_vacuous']), flush=True)
    print('ALL JAX TWIN STATES BIT-IDENTICAL TO NUMPY:',
          res['all_jax_twins_state_identical'],
          ' (Jacobians within %.2f ULP)' % (res['max_twin_djac_ulp'] or 0.0),
          flush=True)
    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1)
    print('wrote', out_path, flush=True)


if __name__ == '__main__':
    main()

"""VERIFY-WP-B12 probe V8 -- the edges the repair could have broken.

E1. **Dead rays.**  ``exit_vertex.exit_vertex_transfer`` FREEZES a dead ray --
position, direction, OPL and ``z`` -- because a vignetted ray never reached the
vertex plane.  ``_project_to_exit_vertex_plane`` applies its arithmetic to
every row.  Measured here: whether the projection turns a finite un-projected
state into a non-finite one on a heavily vignetted fan, and whether the four
``fga.py`` sites mask on ``alive`` before consuming it.

E2. **Grazing rays.**  A ray with ``|N| -> 0`` cannot reach the vertex plane;
``exit_vertex_transfer`` kills it with ``RAY_MISSED_SURFACE``.  The projection
has no such guard -- its ``sec`` simply grows.  Measured: what the primitive
returns for a very large input slope, and whether it stays finite.

E3. **Chunking.**  ``apply_real_lens_fga``'s memory budget splits the swarm and
the launch lattice; the projection now runs once per chunked trace.  Measured:
a chunked run against an unchunked one, bit for bit.

E4. **The config-object entry.**  ``LensConfig`` / ``LensNumerics`` reach the
same code, so the repair must reach them too: measured as byte identity against
the keyword form.

E5. **float32 / complex64 input.**  The primitives cast to float64; the FGA
entry takes a complex64 field.  Measured: it still runs and the projection is
still applied (the field differs from the forced-'surface' arm).
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vb12_common as C  # noqa: E402
from probe_v3_field import (  # noqa: E402
    _install_forced_surface,
    _remove_forced_surface,
)


def e1_dead_rays():
    from lumenairy import raytrace as rt
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.differential import ray_transfer_jacobian
    fx = C.fixture('asph')
    surfs = surfaces_from_prescription(fx.prescription())
    # a fan that runs WELL past the clear aperture, so most rays vignette
    n = 201
    h = np.linspace(-3.0 * fx.semi, 3.0 * fx.semi, n)
    z = np.zeros(n)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        s = ray_transfer_jacobian(h.copy(), z.copy(), z.copy(), z.copy(),
                                  surfs, fx.lam)
        v = ray_transfer_jacobian(h.copy(), z.copy(), z.copy(), z.copy(),
                                  surfs, fx.lam, reference='exit_vertex')
        b = rt.RayBundle(x=h.copy(), y=z.copy(), z=z.copy(), L=z.copy(),
                         M=z.copy(), N=np.ones(n), wavelength=fx.lam,
                         alive=np.ones(n, bool), opd=z.copy())
        res = rt.trace(b, surfs, fx.lam)
        ev = res.at_exit_vertex()
    dead = ~np.asarray(s.alive, bool)
    fin_s = np.isfinite(np.asarray(s.x)) & np.isfinite(np.asarray(s.opd))
    fin_v = np.isfinite(np.asarray(v.x)) & np.isfinite(np.asarray(v.opd))
    rec = dict(
        n_rays=n, n_dead=int(dead.sum()), n_alive=int((~dead).sum()),
        finite_before_dead=int(fin_s[dead].sum()),
        finite_after_dead=int(fin_v[dead].sum()),
        newly_nonfinite=int((fin_s & ~fin_v).sum()),
        dead_rays_moved=float(np.nanmax(np.abs(
            np.asarray(v.x)[dead] - np.asarray(s.x)[dead]))
            if dead.any() else 0.0),
        at_exit_vertex_freezes_dead=bool(
            np.array_equal(np.asarray(ev.x)[dead],
                           np.asarray(res.image_rays.x)[dead])),
        alive_flags_unchanged=bool(np.array_equal(np.asarray(s.alive),
                                                  np.asarray(v.alive))),
    )
    print(f"  E1 {rec['n_dead']}/{n} dead; newly non-finite after the "
          f"projection: {rec['newly_nonfinite']}; the projection MOVES dead "
          f"rays by up to {rec['dead_rays_moved']:.3e} m where "
          f"at_exit_vertex freezes them ({rec['at_exit_vertex_freezes_dead']})",
          flush=True)
    return rec


def e2_grazing():
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.differential import ray_transfer_jacobian
    fx = C.fixture('asph')
    surfs = surfaces_from_prescription(fx.prescription())
    out = {}
    for u in (1.0, 10.0, 1e3, 1e8):
        h = np.array([0.0, fx.semi * 0.5])
        z = np.zeros(2)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                v = ray_transfer_jacobian(h.copy(), z.copy(),
                                          np.full(2, u), z.copy(), surfs,
                                          fx.lam, reference='exit_vertex')
            out[f'u={u:g}'] = dict(
                alive=[bool(a) for a in np.asarray(v.alive)],
                finite_x=bool(np.all(np.isfinite(np.asarray(v.x)))),
                finite_opd=bool(np.all(np.isfinite(np.asarray(v.opd)))),
                x=[repr(float(q)) for q in np.asarray(v.x)])
        except Exception as exc:                            # noqa: BLE001
            out[f'u={u:g}'] = dict(error=f'{type(exc).__name__}: {exc}')
        print(f'  E2 u={u:g}: {out[f"u={u:g}"]}', flush=True)
    return out


def e3_chunking():
    from lumenairy.propagators.fga import apply_real_lens_fga
    fx = C.fixture('asph').coarse()
    kw = dict(prescription=fx.prescription(), wavelength=fx.lam, dx=fx.dx,
              output_plane_distance=fx.best_focus())
    a = apply_real_lens_fga(fx.E_in(), mem_budget_mb=4000.0, **kw)
    b = apply_real_lens_fga(fx.E_in(), mem_budget_mb=2.0, **kw)
    c = apply_real_lens_fga(fx.E_in(), chunk=2, **kw)
    rec = dict(sha_big=C.sha(a), sha_tiny=C.sha(b), sha_chunk2=C.sha(c),
               tiny_vs_big=float(np.max(np.abs(a - b))),
               chunk2_vs_big=float(np.max(np.abs(a - c))),
               rel_tiny=float(np.max(np.abs(a - b)) / np.max(np.abs(a))),
               rel_chunk2=float(np.max(np.abs(a - c)) / np.max(np.abs(a))))
    print(f"  E3 chunked vs unchunked: |d| {rec['tiny_vs_big']:.3e} "
          f"(rel {rec['rel_tiny']:.3e}) / {rec['chunk2_vs_big']:.3e} "
          f"(rel {rec['rel_chunk2']:.3e})", flush=True)
    return rec


def e4_config_object():
    import lumenairy as la
    from lumenairy.propagators.fga import apply_real_lens_fga
    fx = C.fixture('asph').coarse()
    z = fx.best_focus()
    a = apply_real_lens_fga(fx.E_in(), prescription=fx.prescription(),
                            wavelength=fx.lam, dx=fx.dx,
                            output_plane_distance=z)
    rec = {}
    try:
        geom = la.LensGeometry(output_plane_distance=z)
        b = apply_real_lens_fga(fx.E_in(), prescription=fx.prescription(),
                                wavelength=fx.lam, dx=fx.dx, geometry=geom)
        rec = dict(sha_kw=C.sha(a), sha_geometry=C.sha(b),
                   bit_identical=bool(np.array_equal(a, b)))
    except Exception as exc:                                # noqa: BLE001
        rec = dict(error=f'{type(exc).__name__}: {exc}', sha_kw=C.sha(a))
    print(f'  E4 config-object entry: {rec}', flush=True)
    return rec


def e5_complex64():
    from lumenairy.propagators.fga import apply_real_lens_fga
    fx = C.fixture('asph').coarse()
    z = fx.best_focus()
    E = fx.E_in().astype(np.complex64)
    kw = dict(prescription=fx.prescription(), wavelength=fx.lam, dx=fx.dx,
              output_plane_distance=z)
    _remove_forced_surface()
    a = apply_real_lens_fga(E, **kw)
    _install_forced_surface()
    b = apply_real_lens_fga(E, **kw)
    _remove_forced_surface()
    rec = dict(dtype_in=str(E.dtype), dtype_out=str(np.asarray(a).dtype),
               projection_changed_the_field=not bool(np.array_equal(a, b)),
               rel_change=float(np.max(np.abs(a - b))
                                / max(float(np.max(np.abs(a))), 1e-300)))
    print(f'  E5 complex64: {rec}', flush=True)
    return rec


def main():
    import lumenairy as la
    print('lumenairy.__file__ =', os.path.abspath(la.__file__), flush=True)
    out = {'env': C.env_block()}
    out['e1_dead_rays'] = e1_dead_rays()
    out['e2_grazing'] = e2_grazing()
    out['e3_chunking'] = e3_chunking()
    out['e4_config_object'] = e4_config_object()
    out['e5_complex64'] = e5_complex64()
    C.dump(out, 'probe_v8_edges')


if __name__ == '__main__':
    main()

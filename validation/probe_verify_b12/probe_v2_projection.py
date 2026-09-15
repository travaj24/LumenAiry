"""VERIFY-WP-B12 probe V2 -- the projection helper, derived and re-measured.

Three independent checks of ``differential._project_to_exit_vertex_plane``:

**B1. The state map.**  ``x_v = x - s u_x``, ``y_v = y - s u_y``,
``opd_v = opd - n_exit sign(N) s sec`` is re-derived here from the straight-line
transfer ``t = -z/N`` and evaluated with MY OWN sag kernels, then differenced
against what the library returns.  (Probe V1 already does the end-to-end form;
this one isolates the three formulae.)

**B2. The Jacobian.**  The vertex-plane state is a function of the INPUT state
through the composed map, so

    J_v = P J_s,   P = [[1 - s_x u_x,   - s_y u_x,  -s,   0 ],
                        [  - s_x u_y, 1 - s_y u_y,   0,  -s ],
                        [      0     ,      0     ,   1,   0 ],
                        [      0     ,      0     ,   0,   1 ]]

(rows 0/1: ``d(x - s(x,y) u_x)/d(.)`` with ``s`` evaluated at the LANDING
point, so its transverse gradient enters; rows 2/3: the slopes are untouched
by a straight-line transfer).  Two measurements:

* ``P`` built from MY sag gradients against the library's own
  ``J_v (J_s)^-1``;
* the library's ``J_v`` against a CENTRAL FINITE DIFFERENCE of my own
  vertex-plane map, over a step ladder, with the unprojected ``J_s`` scored
  against the same reference for contrast.

The asphere and the biconic are the point: WP-B12's oracle is a
rotationally-symmetric CONIC trace, so the aspheric departure and the biconic
y-branch are exactly what it could not measure.

**B3. The JAX path.**  ``jax.grad`` of a scalar read of the projected state
against a central finite difference of the same scalar.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vb12_common as C  # noqa: E402

NR = 25


def _my_vertex_state(fx, x, y, ux, uy):
    """MY vertex-plane map: (x, y, ux, uy) -> (xv, yv, uxv, uyv, opl)."""
    N = 1.0 / np.sqrt(1.0 + ux ** 2 + uy ** 2)
    L, M = ux * N, uy * N
    st = C.trace3d(x, y, np.zeros_like(x), L, M, N, np.zeros_like(x),
                   fx.oracle_surfaces())
    vt = C.to_vertex(st, fx.n_exit(), fx.oracle_surfaces()[-1]['zv'])
    return (vt['x'], vt['y'], vt['L'] / vt['N'], vt['M'] / vt['N'], vt['opl'])


def _fd_jacobian(fx, x, y, ux, uy, h_pos, h_slope):
    """Central-difference 4x4 Jacobian of MY vertex-plane map."""
    base = [x, y, ux, uy]
    steps = [h_pos, h_pos, h_slope, h_slope]
    J = np.zeros((len(x), 4, 4))
    for j in range(4):
        pp = [a.copy() for a in base]
        mm = [a.copy() for a in base]
        pp[j] = pp[j] + steps[j]
        mm[j] = mm[j] - steps[j]
        sp = _my_vertex_state(fx, *pp)
        sm = _my_vertex_state(fx, *mm)
        for i in range(4):
            J[:, i, j] = (sp[i] - sm[i]) / (2.0 * steps[j])
    return J


def _row_rel(A, B):
    """Per-row relative max error, normalised by that row of the reference."""
    num = np.abs(A - B).max(axis=2)
    den = np.maximum(np.abs(B).max(axis=2), 1e-300)
    return float(np.nanmax(num / den))


def b1_b2(fx, analytic):
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.differential import (
        ray_transfer_jacobian,
        ray_transfer_jacobian_analytic,
    )
    fn = ray_transfer_jacobian_analytic if analytic else ray_transfer_jacobian
    surfs = surfaces_from_prescription(fx.prescription())
    lam = fx.lam
    x = np.linspace(-fx.semi * 0.9, fx.semi * 0.9, NR)
    y = np.linspace(-fx.semi * 0.55, fx.semi * 0.55, NR)      # off-axis in y
    L0, M0, N0 = fx.input_dirs(x.shape)
    ux = L0 / N0
    uy = M0 / N0
    dt_s = fn(x.copy(), y.copy(), ux.copy(), uy.copy(), surfs, lam)
    dt_v = fn(x.copy(), y.copy(), ux.copy(), uy.copy(), surfs, lam,
              reference='exit_vertex')
    alive = np.asarray(dt_s.alive, bool) & np.asarray(dt_v.alive, bool)

    # -- B1: the three state formulae, from MY sag ------------------------
    last = fx.oracle_surfaces()[-1]
    s_own = C.sag(np.asarray(dt_s.x), np.asarray(dt_s.y), last)
    sx_own, sy_own = C.sag_grad(np.asarray(dt_s.x), np.asarray(dt_s.y), last)
    sec = np.sqrt(1.0 + dt_s.ux ** 2 + dt_s.uy ** 2)
    sgn = -1.0 if fx.mirror_last else 1.0
    pred_x = dt_s.x - s_own * dt_s.ux
    pred_y = dt_s.y - s_own * dt_s.uy
    pred_o = dt_s.opd - fx.n_exit() * sgn * s_own * sec

    def mx(a, b):
        return float(np.nanmax(np.abs(np.asarray(a)[alive]
                                      - np.asarray(b)[alive])))

    rec = dict(backend='analytic' if analytic else 'fd',
               b1_dx=mx(dt_v.x, pred_x), b1_dy=mx(dt_v.y, pred_y),
               b1_dopl=mx(dt_v.opd, pred_o),
               sag_max=float(np.nanmax(np.abs(s_own[alive]))),
               sag_grad_max=float(np.nanmax(np.abs(sx_own[alive]))))

    # -- B2a: P from MY sag gradients vs the library's J_v J_s^-1 ---------
    n = int(alive.sum())
    P_own = np.zeros((n, 4, 4))
    P_own[:, 0, 0] = 1.0 - sx_own[alive] * dt_s.ux[alive]
    P_own[:, 0, 1] = -sy_own[alive] * dt_s.ux[alive]
    P_own[:, 0, 2] = -s_own[alive]
    P_own[:, 1, 0] = -sx_own[alive] * dt_s.uy[alive]
    P_own[:, 1, 1] = 1.0 - sy_own[alive] * dt_s.uy[alive]
    P_own[:, 1, 3] = -s_own[alive]
    P_own[:, 2, 2] = 1.0
    P_own[:, 3, 3] = 1.0
    Js = np.asarray(dt_s.jacobian)[alive]
    Jv = np.asarray(dt_v.jacobian)[alive]
    rec['b2a_PJ_vs_Jv'] = _row_rel(P_own @ Js, Jv)

    # -- B2b: the library's J_v against a FD of MY map, on a step ladder --
    ladder = []
    for hp, hs in ((1e-6, 5e-5), (3e-7, 2e-5), (1e-7, 1e-5), (3e-8, 3e-6)):
        Jfd = _fd_jacobian(fx, x.copy(), y.copy(), ux.copy(), uy.copy(),
                           hp, hs)[alive]
        ladder.append(dict(h_pos=hp, h_slope=hs,
                           projected=_row_rel(Jv, Jfd),
                           unprojected=_row_rel(Js, Jfd)))
    rec['b2b_ladder'] = ladder
    best = min(ladder, key=lambda r: r['projected'])
    rec['b2b_projected_best'] = best['projected']
    rec['b2b_unprojected_at_same_step'] = best['unprojected']
    rec['b2b_ratio'] = (best['unprojected'] / best['projected']
                        if best['projected'] > 0 else float('inf'))
    return rec


def b3_jax(fx):
    """jax.grad through the projection, against a central FD of the same
    scalar read."""
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.differential import ray_transfer_jacobian_analytic
    surfs = surfaces_from_prescription(fx.prescription())
    lam = fx.lam
    x0 = np.linspace(-fx.semi * 0.8, fx.semi * 0.8, 9)
    y0 = np.linspace(-fx.semi * 0.4, fx.semi * 0.4, 9)
    L0, M0, N0 = fx.input_dirs(x0.shape)
    ux0, uy0 = L0 / N0, M0 / N0
    w_o = np.cos(np.linspace(0.3, 2.1, 9))
    w_x = np.sin(np.linspace(0.2, 1.7, 9))

    def scalar(shift, reference):
        xs = jnp.asarray(x0) + shift
        out = ray_transfer_jacobian_analytic(
            xs, jnp.asarray(y0), jnp.asarray(ux0), jnp.asarray(uy0),
            surfs, lam, reference=reference)
        return (jnp.sum(jnp.asarray(w_o) * out.opd)
                + jnp.sum(jnp.asarray(w_x) * out.x)
                + jnp.sum(out.jacobian[:, 0, 0]))

    rec = {}
    for reference in ('surface', 'exit_vertex'):
        g = float(jax.grad(lambda s: scalar(s, reference))(0.0))
        lad = []
        for h in (1e-7, 3e-8, 1e-8, 3e-9):
            fp = float(scalar(h, reference))
            fm = float(scalar(-h, reference))
            fd = (fp - fm) / (2.0 * h)
            lad.append(dict(h=h, fd=fd, rel=abs(g - fd) / max(abs(fd), 1e-300)))
        best = min(lad, key=lambda r: r['rel'])
        rec[reference] = dict(grad=g, fd_best=best['fd'], rel=best['rel'],
                              h=best['h'], ladder=lad)
    rec['grads_differ'] = abs(rec['surface']['grad']
                              - rec['exit_vertex']['grad'])
    # the JAX branch's own state must also agree with the NumPy branch
    from lumenairy.raytrace.differential import ray_transfer_jacobian_analytic as A
    npv = A(x0.copy(), y0.copy(), ux0.copy(), uy0.copy(), surfs, lam,
            reference='exit_vertex')
    jxv = A(jnp.asarray(x0), jnp.asarray(y0), jnp.asarray(ux0),
            jnp.asarray(uy0), surfs, lam, reference='exit_vertex')
    rec['jax_vs_numpy_dx'] = float(np.max(np.abs(np.asarray(jxv.x) - npv.x)))
    rec['jax_vs_numpy_dopl'] = float(
        np.max(np.abs(np.asarray(jxv.opd) - npv.opd)))
    rec['jax_vs_numpy_djac'] = float(
        np.max(np.abs(np.asarray(jxv.jacobian) - npv.jacobian)))
    return rec


def main():
    import lumenairy as la
    print('lumenairy.__file__ =', os.path.abspath(la.__file__))
    out = {'env': C.env_block(), 'fixtures': {}, 'jax': {}}
    for fx in C.fixtures():
        rows = {}
        for analytic in (False, True):
            try:
                rows['analytic' if analytic else 'fd'] = b1_b2(fx, analytic)
            except NotImplementedError as exc:
                rows['analytic' if analytic else 'fd'] = dict(
                    refused=f'{type(exc).__name__}')
            except Exception as exc:                       # noqa: BLE001
                rows['analytic' if analytic else 'fd'] = dict(
                    error=f'{type(exc).__name__}: {exc}')
        out['fixtures'][fx.key] = dict(note=fx.note, rows=rows)
        for nm, r in rows.items():
            if 'b1_dx' not in r:
                print(f'[{fx.key:17s}/{nm:8s}] {r}')
                continue
            print(f'[{fx.key:17s}/{nm:8s}] B1 dx={r["b1_dx"]:.2e} '
                  f'dopl={r["b1_dopl"]:.2e} | B2a P.J vs J_v='
                  f'{r["b2a_PJ_vs_Jv"]:.2e} | B2b proj='
                  f'{r["b2b_projected_best"]:.2e} unproj='
                  f'{r["b2b_unprojected_at_same_step"]:.2e} '
                  f'(x{r["b2b_ratio"]:.3g})')
    for key in ('asph', 'menisc'):
        try:
            out['jax'][key] = b3_jax(C.fixture(key))
            r = out['jax'][key]
            print(f'[JAX {key}] surface: grad vs FD rel='
                  f'{r["surface"]["rel"]:.2e} | exit_vertex: rel='
                  f'{r["exit_vertex"]["rel"]:.2e} | grads differ by '
                  f'{r["grads_differ"]:.4e} | jax-vs-numpy dopl='
                  f'{r["jax_vs_numpy_dopl"]:.2e} djac='
                  f'{r["jax_vs_numpy_djac"]:.2e}')
        except Exception as exc:                           # noqa: BLE001
            out['jax'][key] = dict(error=f'{type(exc).__name__}: {exc}')
            print(f'[JAX {key}] ERROR {exc}')
    C.dump(out, 'probe_v2_projection')


if __name__ == '__main__':
    main()

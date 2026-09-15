"""VERIFY-WP-B12 probe V1 -- the defect and the repair, at the PRIMITIVE.

For each fixture and each backend (finite-difference ``ray_transfer_jacobian``
and analytic ``ray_transfer_jacobian_analytic``), the base-ray state is read
twice -- ``reference='surface'`` (the pre-WP-B12 behaviour, still the default)
and ``reference='exit_vertex'`` (what the four ``fga.py`` sites now ask for) --
and both are compared against MY OWN tracer's two planes.

Claim (1) of the report is the ``surface`` column measured against MY vertex
plane: the spurious transverse offset ``sag * u`` and the spurious optical
path ``n_exit * sag * sec(theta)`` an image leg started on the wrong plane
carries.  Claim (2) is the ``exit_vertex`` column measured against the same
reference: it has to collapse to the tracer's own floor.

Flat controls: the projection must be the IDENTITY, bit for bit
(``np.array_equal`` on every field, not a tolerance).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vb12_common as C  # noqa: E402

NRAY = 81


def measure(fx, analytic):
    from lumenairy.raytrace import surfaces_from_prescription
    from lumenairy.raytrace.differential import (
        ray_transfer_jacobian,
        ray_transfer_jacobian_analytic,
    )
    fn = ray_transfer_jacobian_analytic if analytic else ray_transfer_jacobian
    surfs = surfaces_from_prescription(fx.prescription())
    lam = fx.lam
    h = np.linspace(-fx.semi * 0.97, fx.semi * 0.97, NRAY)
    z = np.zeros_like(h)
    L0, M0, N0 = fx.input_dirs(h.shape)
    ux0 = L0 / N0
    uy0 = M0 / N0

    dt_s = fn(h.copy(), z.copy(), ux0.copy(), uy0.copy(), surfs, lam)
    dt_v = fn(h.copy(), z.copy(), ux0.copy(), uy0.copy(), surfs, lam,
              reference='exit_vertex')

    st = C.trace3d(h.copy(), z.copy(), z.copy(), L0.copy(), M0.copy(),
                   N0.copy(), z.copy(), fx.oracle_surfaces())
    zvl = fx.oracle_surfaces()[-1]['zv']
    vt = C.to_vertex(st, fx.n_exit(), zvl)
    alive = np.asarray(dt_s.alive, bool) & np.asarray(dt_v.alive, bool)

    def mx(a, b):
        a = np.asarray(a)[alive]
        b = np.asarray(b)[alive]
        return float(np.nanmax(np.abs(a - b)))

    lamw = lam
    rec = dict(
        backend='analytic' if analytic else 'fd',
        n_rays=int(alive.sum()),
        # the defect: 'surface' state read AS IF it were the vertex plane
        defect_dx=mx(dt_s.x, vt['x']),
        defect_dopl=mx(dt_s.opd, vt['opl']),
        defect_waves=mx(dt_s.opd, vt['opl']) / lamw,
        # the repair
        fixed_dx=mx(dt_v.x, vt['x']),
        fixed_dy=mx(dt_v.y, vt['y']),
        fixed_dopl=mx(dt_v.opd, vt['opl']),
        fixed_waves=mx(dt_v.opd, vt['opl']) / lamw,
        # the default is unchanged: 'surface' still reads the last surface
        default_dx=mx(dt_s.x, st['x']),
        default_dopl=mx(dt_s.opd, st['opl']),
        # the predicted defect, from MY sag: n_exit * sag * sec
        predicted_waves=float(np.nanmax(np.abs(
            fx.n_exit()
            * C.sag(np.asarray(st['x'])[alive], np.asarray(st['y'])[alive],
                    fx.oracle_surfaces()[-1])
            / np.abs(np.asarray(st['N'])[alive]))) / lamw),
        slopes_unchanged=bool(np.array_equal(dt_s.ux, dt_v.ux)
                              and np.array_equal(dt_s.uy, dt_v.uy)),
        bit_identical=bool(
            np.array_equal(dt_s.x, dt_v.x) and np.array_equal(dt_s.y, dt_v.y)
            and np.array_equal(dt_s.opd, dt_v.opd)
            and np.array_equal(dt_s.jacobian, dt_v.jacobian)),
        jac_moved=float(np.nanmax(np.abs(
            np.asarray(dt_s.jacobian)[alive]
            - np.asarray(dt_v.jacobian)[alive]))),
    )
    return rec


def main():
    import lumenairy as la
    print('lumenairy.__file__ =', os.path.abspath(la.__file__))
    out = {'env': C.env_block(), 'fixtures': {}}
    for fx in C.fixtures():
        rows = {}
        for analytic in (False, True):
            try:
                rows['analytic' if analytic else 'fd'] = measure(fx, analytic)
            except Exception as exc:                      # noqa: BLE001
                rows['analytic' if analytic else 'fd'] = dict(
                    error=f'{type(exc).__name__}: {exc}')
        out['fixtures'][fx.key] = dict(note=fx.note, flat_last=fx.flat_last,
                                       lam=fx.lam, rows=rows)
        for nm, r in rows.items():
            if 'error' in r:
                print(f'[{fx.key:17s}/{nm:8s}] {r["error"]}')
                continue
            print(f'[{fx.key:17s}/{nm:8s}] defect dx={r["defect_dx"]:.4e} m '
                  f'opl={r["defect_waves"]:9.4f} waves '
                  f'(predicted {r["predicted_waves"]:9.4f}) | '
                  f'fixed dx={r["fixed_dx"]:.2e} opl={r["fixed_waves"]:.2e} w '
                  f'| bit-identical={r["bit_identical"]}')
    C.dump(out, 'probe_v1_mechanism')


if __name__ == '__main__':
    main()

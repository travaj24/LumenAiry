"""VERIFY-WP-B12 probe V6 -- the new pins' actual margins.

A bar is durable only if the quantity it reads sits decades from it on BOTH
sides (``docs/TESTING_STANDARDS.md`` rule 5).  This probe re-runs the readings
that the fourteen new tests assert and prints the number against the bar, so
each pin can be scored ``margin = bar / reading`` rather than "it was green".

It imports the test module itself, so the readings are the tests' own.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..')))
import vb12_common as C  # noqa: E402


def main():
    import lumenairy as la
    print('lumenairy.__file__ =', os.path.abspath(la.__file__), flush=True)
    import tests.unit.test_audit2609_b12_fga_reference_plane as T
    from lumenairy.propagators import fga
    from lumenairy.raytrace.differential import ray_transfer_jacobian
    out = {'env': C.env_block(), 'margins': {}}

    # -- registering the module glasses the test's conftest fixture adds ---
    from lumenairy import glass as G
    for k, v in T.MODULE_GLASSES.items():
        G.GLASS_REGISTRY[k] = v

    # 1/2/3: the plane agreements ----------------------------------------
    presc = T._biconvex()
    surfs, h, z, img, ex = T._fan(presc)
    ok = np.asarray(ex.alive, bool)
    d = T._both(presc, h, z, surfs)
    rows = {}
    for nm in ('fd', 'analytic'):
        s, v = d[(nm, 'surface')], d[(nm, 'exit_vertex')]
        rows[f'{nm}_surface'] = max(
            float(np.abs(np.asarray(s.x)[ok] - np.asarray(img.x)[ok]).max()),
            float(np.abs(np.asarray(s.opd)[ok]
                         - np.asarray(img.opd)[ok]).max()))
        rows[f'{nm}_exit_vertex'] = max(
            float(np.abs(np.asarray(v.x)[ok] - np.asarray(ex.x)[ok]).max()),
            float(np.abs(np.asarray(v.opd)[ok]
                         - np.asarray(ex.opd)[ok]).max()))
    out['margins']['t1_plane_agreement'] = dict(
        bar=1e-14, readings=rows, worst=max(rows.values()),
        margin=1e-14 / max(max(rows.values()), 1e-300))

    # 6: the projected Jacobian vs its own FD -----------------------------
    n = 65
    hh = np.linspace(T._SEMI / (2 * n), T._SEMI * 0.98, n)
    zz = np.zeros(n)

    def _state(xx, yy, uxx, uyy):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            q = ray_transfer_jacobian(xx, yy, uxx, uyy, surfs, T._LAM,
                                      reference='exit_vertex')
        return np.stack([q.x, q.y, q.ux, q.uy], axis=-1)

    def _fd_jac(sp, ss):
        cols = []
        for dim, st in enumerate((sp, sp, ss, ss)):
            ap = [hh.copy(), zz.copy(), zz.copy(), zz.copy()]
            am = [hh.copy(), zz.copy(), zz.copy(), zz.copy()]
            ap[dim] = ap[dim] + st
            am[dim] = am[dim] - st
            cols.append((_state(*ap) - _state(*am)) / (2.0 * st))
        return np.stack(cols, axis=-1)

    j_ref = _fd_jac(1e-6, 5e-5)
    j_half = _fd_jac(5e-7, 2.5e-5)
    scale = np.abs(j_ref).max(axis=(0, 2))[None, :, None]

    def _rel(a, b):
        return float((np.abs(a - b) / scale).max())

    ladder = _rel(j_ref, j_half)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        j_v = ray_transfer_jacobian(hh.copy(), zz.copy(), zz.copy(),
                                    zz.copy(), surfs, T._LAM,
                                    reference='exit_vertex').jacobian
        j_s = ray_transfer_jacobian(hh.copy(), zz.copy(), zz.copy(),
                                    zz.copy(), surfs, T._LAM).jacobian
    bar = max(10.0 * ladder, 1e-7)
    out['margins']['t6_jacobian'] = dict(
        ladder=ladder, bar=bar, projected=_rel(j_v, j_ref),
        unprojected=_rel(j_s, j_ref),
        margin_projected=bar / max(_rel(j_v, j_ref), 1e-300),
        margin_unprojected=_rel(j_s, j_ref) / (10.0 * bar))

    # 7: the two-backend ratio --------------------------------------------
    r = {}
    for ref in ('surface', 'exit_vertex'):
        a = np.asarray(d[('fd', ref)].jacobian)[ok]
        b = np.asarray(d[('analytic', ref)].jacobian)[ok]
        r[ref] = float(np.abs(a - b).max()) / float(np.abs(b).max())
    out['margins']['t7_backend_ratio'] = dict(
        readings=r, ratio=r['exit_vertex'] / r['surface'], bar='(0.5, 2.0)')

    # 8: per-surface product vs composite ---------------------------------
    surfs2, h2, z2, _i, _e = T._fan(presc, n=101)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pv = ray_transfer_jacobian(h2.copy(), z2.copy(), z2.copy(), z2.copy(),
                                   surfs2, T._LAM, per_surface=True,
                                   reference='exit_vertex')
        cv = ray_transfer_jacobian(h2.copy(), z2.copy(), z2.copy(), z2.copy(),
                                   surfs2, T._LAM, reference='exit_vertex')
    prod = np.asarray(pv.jacobian)[0]
    for k in range(1, np.asarray(pv.jacobian).shape[0]):
        prod = np.asarray(pv.jacobian)[k] @ prod
    rel = (float(np.abs(prod - np.asarray(cv.jacobian)).max())
           / float(np.abs(np.asarray(cv.jacobian)).max()))
    out['margins']['t8_per_surface_product'] = dict(
        bar=1e-6, reading=rel, margin=1e-6 / max(rel, 1e-300))

    # 9: the aspheric gap --------------------------------------------------
    pa = T._biconvex(asph={4: 4.0e8, 6: -8.0e17})
    sa, ha, za, _ia, exa = T._fan(pa)
    oka = np.asarray(exa.alive, bool)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        va = ray_transfer_jacobian(ha.copy(), za.copy(), za.copy(), za.copy(),
                                   sa, T._LAM, reference='exit_vertex')
        ssa = ray_transfer_jacobian(ha.copy(), za.copy(), za.copy(),
                                    za.copy(), sa, T._LAM)
    ra = np.hypot(np.asarray(ssa.x), np.asarray(ssa.y))
    seca = np.sqrt(1.0 + np.asarray(ssa.ux) ** 2 + np.asarray(ssa.uy) ** 2)
    conic_only = np.asarray(ssa.opd) - T._sag_of(ra, -T._R) * seca
    out['margins']['t9_aspheric'] = dict(
        agreement=float(np.abs(np.asarray(va.opd)[oka]
                               - np.asarray(exa.opd)[oka]).max()),
        agreement_bar=1e-12,
        gap=float(np.abs(conic_only[oka] - np.asarray(va.opd)[oka]).max()),
        gap_bar=1e-9)
    out['margins']['t9_aspheric']['gap_margin'] = (
        out['margins']['t9_aspheric']['gap'] / 1e-9)

    # 12: the caustic zone -------------------------------------------------
    N, dx, w0 = 192, 2.55e-6, 105e-6
    xs = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    zone_rows = {}
    for label, (pr, R2, glass) in {
            'curved': (T._biconvex(), -T._R, T._GLASS),
            'flat': (T._flat_last(), np.inf, 'N-LASF9')}.items():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            zone = fga._caustic_zone(E, dx, pr, T._LAM)
        row = np.abs(E[N // 2, :])
        xs_h, amp_h = xs[N // 2:], row[N // 2:]
        good = amp_h > 0.05 * amp_h.max()
        rr = np.linspace(xs_h[good][0], xs_h[good][-1], 25)
        rr = rr[rr > 0]
        ng = float(la.get_glass_index(glass, T._LAM))
        R1 = pr['surfaces'][0]['radius']
        t = pr['thicknesses'][0]
        o = T._oracle_trace(rr, [(0.0, R1, 0.0, None, ng),
                                 (t, R2, 0.0, None, 1.0)])
        zz2 = -o[3] / o[4]
        zz2 = zz2[zz2 > 0]
        want = (float(np.percentile(zz2, 5)), float(np.percentile(zz2, 95)))
        rels = [abs(g - w) / w for g, w in zip(zone, want)]
        zone_rows[label] = dict(zone=list(map(float, zone)), want=list(want),
                                rel=rels, worst=max(rels), bar=1e-4,
                                margin=1e-4 / max(max(rels), 1e-300))
    out['margins']['t12_caustic_zone'] = zone_rows

    for k, v in out['margins'].items():
        print(f'{k}: {v}', flush=True)
    C.dump(out, 'probe_v6_margins')


if __name__ == '__main__':
    main()

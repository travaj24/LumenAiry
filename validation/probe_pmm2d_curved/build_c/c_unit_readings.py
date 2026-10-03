"""The readings behind the unit-test bars of
``tests/unit/test_pmm2d_staggered_curved_c.py`` -- each arm computes, at the
test's own size, exactly what the test asserts, plus its fail-before.

  c_unit_readings.py <arm>
arms: c3fid (C3 fillet fidelity), c6 (two-layer stack), c13 (2 x 2 circle
array vs the halved period), c14 (four-fold symmetry), c16 (tensor routing),
c11 (viewer), c12 (geometry), c9 (incident decomposition at the test size)
Output: c_unit_<arm>.json
"""
import sys
import time
import warnings

import _common as C
import numpy as np

from lumenairy.elements.pmm import (  # noqa: E402
    Circle,
    Ellipse,
    FilletRect,
    PMM2DStackPure,
    Rect,
    SinusoidalWall,
    compile_shapes,
    pmm_jones_2d_staggered,
)

SP, TS = C.SP, C.TS


def stack(layers, M, theta=0.0, phi=0.0, period=C.P, n_orders=3,
          retain=False):
    st = PMM2DStackPure(period, period, n_superstrate=C.N_SUP,
                        n_substrate=C.N_SUB, n_modes=M, n_orders=n_orders)
    for t, spec in layers:
        if isinstance(spec, tuple):
            st.add_layer(t, shapes=spec[0], background_eps=spec[1])
        else:
            st.add_layer(t, eps=spec)
    st.set_source(C.WL, theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve(retain_internal=retain)
    return st, np.asarray(o), np.asarray(R), np.asarray(T), J


def c3fid():
    """C3: the fillet moves the zeroth orders far above convergence; the
    sharp square is the Rect route (no map) on the same walls."""
    res = {"env": C.env_record()}
    side = 0.6
    arms = [("fillet0.2_M4", FilletRect(0.6, 0.6, side, side, 0.2 * side,
                                        4.0), 4),
            ("fillet0.2_M5", FilletRect(0.6, 0.6, side, side, 0.2 * side,
                                        4.0), 5),
            ("eqarea_M5", Rect(0.6, 0.6, *(2 * [float(np.sqrt(
                side ** 2 - (4 - np.pi) * (0.2 * side) ** 2))]), 4.0), 5)]
    arms += [(f"sharp_M{M}", Rect(0.6, 0.6, side, side, 4.0), M)
             for M in (4, 5, 6)]
    for name, shp, M in arms:
        t0 = time.perf_counter()
        _st, o, R, T, _J = stack([(C.DEPTH, ([shp], 1.0))], M)
        i0 = C.idx(o, [(0, 0)])[0]
        res[name] = {"R00": float(R[1, i0]), "T00": float(T[1, i0]),
                     "t": time.perf_counter() - t0}
    print(res, flush=True)
    C.dump("c_unit_c3fid.json", res)


def c6():
    """C6 at the test size (M = 3, the merged 7 x 7 grid): a circle (layer
    1) nested inside a filleted square's footprint (layer 2) on ONE map.
    lossless closure; lossy absorption balance (and the -R-as-flux-Gram
    fail-before); the vacuum identity ON THE SAME MAP -- layer 2 painted
    vacuum-on-vacuum against a uniform vacuum layer 2 riding the merged map
    explicitly (the shared geometric eig) -- and its fail-before (eps 1.1)."""
    res = {"env": C.env_record()}
    circ = Circle(0.6, 0.6, 0.2, 4.0)

    def fr(e):
        return FilletRect(0.6, 0.6, 0.9, 0.9, 0.09, e)
    for M in (3, 4):
        t0 = time.perf_counter()
        st, o, R, T, _J = stack([(0.3, ([circ], 1.0)),
                                 (0.2, ([fr(2.25)], 1.0))], M)
        res[f"lossless_M{M}"] = {
            "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1))),
            "grid": list(st.cmap.shape), "t": time.perf_counter() - t0}
        st, o, R, T, _J = stack([(0.3, ([circ], 1.0)),
                                 (0.2, ([fr(2.25 + 0.4j)], 1.0))], M,
                                retain=True)
        A = np.asarray(st.layer_absorption())
        bal = 1 - R.sum(1) - T.sum(1)
        G = st._internal["G"]
        st._internal["G"] = -C.TS.Granet2DTransverseE(
            C.P, C.P, st.cmap.u_walls, st.cmap.v_walls, M,
            np.ones(st.cmap.shape, complex), cmap=st.cmap).Rmat
        Abad = np.asarray(st.layer_absorption())
        st._internal["G"] = G
        res[f"lossy_M{M}"] = {"absorption": A.tolist(),
                              "balance": bal.tolist(),
                              "abs_vs_balance": float(np.max(np.abs(
                                  A.sum(0) - bal))),
                              "minusR_flux_failbefore": float(np.max(np.abs(
                                  Abad.sum(0) - bal)))}
        st2, o2, R2, T2, _ = stack([(0.3, ([circ], 1.0)),
                                    (0.2, ([fr(1.0)], 1.0))], M)
        cm = st2.cmap
        cell1 = st2._layers[0]["eps_cell"]
        st3 = PMM2DStackPure(C.P, C.P, n_superstrate=C.N_SUP,
                             n_substrate=C.N_SUB, n_modes=M, n_orders=3,
                             cmap=cm)
        st3.add_layer(0.3, eps_cell=cell1)
        st3.add_layer(0.2, eps=1.0)
        st3.set_source(C.WL)
        o3, R3, T3, _J3 = st3.solve()
        st4, o4, R4, T4, _ = stack([(0.3, ([circ], 1.0)),
                                    (0.2, ([fr(1.1)], 1.0))], M)
        res[f"vacuum_M{M}"] = {
            "d_vacuum_same_map": float(np.max(np.abs(
                C.vec(o2, R2, T2) - C.vec(np.asarray(o3), np.asarray(R3),
                                          np.asarray(T3))))),
            "d_eps1.1_failbefore": float(np.max(np.abs(
                C.vec(o2, R2, T2) - C.vec(o4, R4, T4)))),
            "t_total": time.perf_counter() - t0}
        print(res, flush=True)
    C.dump("c_unit_c6.json", res)


def c13():
    """C13: a 2 x 2 array of circles in a doubled period is the single
    circle: R(2m, 2n; 2p) = R(m, n; p), every odd order zero."""
    res = {"env": C.env_record()}
    r = C.R_CIRC
    P2 = 2 * C.P
    four = [Circle(cx, cy, r, 4.0) for cx in (0.6, 1.8) for cy in (0.6, 1.8)]
    three = four[:3]
    for M in (3, 4):
        out = {}
        st, o, R, T, _ = stack([(C.DEPTH, ([Circle(0.6, 0.6, r, 4.0)],
                                           1.0))], M + 1)
        single = {(int(m), int(n)): (R[:, k], T[:, k]) for k, (m, n) in
                  enumerate(o)}
        for name, shp in (("four", four), ("three_failbefore", three)):
            t0 = time.perf_counter()
            st, o2, R2, T2, _ = stack([(C.DEPTH, (shp, 1.0))], M,
                                      period=P2, n_orders=3)
            even, odd = 0.0, 0.0
            for k, (m, n) in enumerate(o2):
                m, n = int(m), int(n)
                if m % 2 == 0 and n % 2 == 0:
                    s = single.get((m // 2, n // 2))
                    if s is not None:
                        even = max(even, float(np.max(np.abs(R2[:, k] - s[0]))),
                                   float(np.max(np.abs(T2[:, k] - s[1]))))
                else:
                    odd = max(odd, float(np.max(R2[:, k])),
                              float(np.max(T2[:, k])))
            out[name] = {"even_vs_single": even, "odd_max": odd,
                         "grid": list(st.cmap.shape),
                         "closure": float(np.max(np.abs(R2.sum(1)
                                                        + T2.sum(1) - 1))),
                         "t": time.perf_counter() - t0}
        res[f"M{M}"] = out
    print(res, flush=True)
    C.dump("c_unit_c13.json", res)


def c14():
    res = {"env": C.env_record()}
    for name, shp in (("circle", Circle(0.6, 0.6, C.R_CIRC, 4.0)),
                      ("ellipse_1.02", Ellipse(0.6, 0.6, C.R_CIRC,
                                               1.02 * C.R_CIRC, 4.0))):
        for M in (5, 6):
            _st, o, R, T, _ = stack([(C.DEPTH, ([shp], 1.0))], M)
            sym = 0.0
            for m in (-1, 0, 1):
                for n in (-1, 0, 1):
                    i = C.idx(o, [(m, n)])[0]
                    j = C.idx(o, [(n, m)])[0]
                    sym = max(sym, abs(R[1, i] - R[0, j]),
                              abs(T[1, i] - T[0, j]))
            res[f"{name}_M{M}"] = float(sym)
    print(res, flush=True)
    C.dump("c_unit_c14.json", res)


def c16():
    """Tensor routing: a tensor Rect rides the unmapped solver (identity
    map) and equals the shipped integer-grid tensor solve."""
    eps_t = np.array([[4.0, 0.3, 0], [0.3, 3.0, 0], [0, 0, 3.5]], complex)
    res = {"env": C.env_record()}
    M = 5
    st, o, R, T, J = stack([(C.DEPTH, ([Rect(0.6, 0.6, 0.4, 0.4, eps_t)],
                                       1.0))], M)
    cell = np.broadcast_to(np.eye(3, dtype=complex), (3, 3, 3, 3)).copy()
    cell[1, 1] = eps_t
    o2, R2, T2, J2 = pmm_jones_2d_staggered(C.P, C.P, cell, C.N_SUB, C.N_SUP,
                                            C.DEPTH, C.WL, n_modes=M,
                                            n_orders=3)
    res["tensor_rect_vs_integer_grid"] = float(max(
        np.max(np.abs(R - R2)), np.max(np.abs(T - T2)),
        np.max(np.abs(J - J2))))
    res["walls"] = [st._shape_walls[0].tolist(), st._shape_walls[1].tolist()]
    print(res, flush=True)
    C.dump("c_unit_c16.json", res)


def c12():
    """Mapped area and outline length of every primitive vs the analytic
    values; min det J on the interior Gauss nodes."""
    from numpy.polynomial.legendre import leggauss
    xg, wg = leggauss(48)
    res = {"env": C.env_record()}
    shapes = {
        "rect": Rect(0.55, 0.62, 0.5, 0.4, 4.0),
        "circle3": Circle(0.58, 0.63, 0.33, 4.0),
        "circle5": Circle(0.6, 0.6, 0.36, 4.0, core=0.5),
        "fillet": FilletRect(0.6, 0.58, 0.7, 0.5, 0.08, 4.0),
        "ellipse": Ellipse(0.6, 0.6, 0.40, 0.28, 4.0),
        "ellipse_rot": Ellipse(0.6, 0.6, 0.40, 0.28, 4.0,
                               angle=np.deg2rad(20.0)),
        "sine_ridge": SinusoidalWall("x", 0.3, 0.12, eps=4.0, width=0.5),
        "sine_y_half": SinusoidalWall("y", 0.5, 0.08, 2, 0.3, eps=4.0),
    }
    for name, sh in shapes.items():
        eps, xw, yw, cm = compile_shapes(C.P, C.P, [sh], 1.0)
        A, mind = 0.0, np.inf
        for sx in range(cm.shape[0]):
            Ju = 0.5 * (cm.u_bounds[sx + 1] - cm.u_bounds[sx])
            U = 0.5 * (cm.u_bounds[sx] + cm.u_bounds[sx + 1]) + Ju * xg
            for sy in range(cm.shape[1]):
                Jv = 0.5 * (cm.v_bounds[sy + 1] - cm.v_bounds[sy])
                V = 0.5 * (cm.v_bounds[sy] + cm.v_bounds[sy + 1]) + Jv * xg
                _X, _Y, xu, xv, yu, yv = cm.geom(sx, sy, U, V)
                det = xu * yv - xv * yu
                mind = min(mind, float(det.min()))
                if eps[sx, sy] != 1.0:
                    A += float(np.sum(np.outer(wg, wg) * det)) * Ju * Jv
        # outline length: every cell edge between two different materials
        L = 0.0
        nx, ny = cm.shape
        for sx in range(nx):
            for sy in range(ny):
                for di, dj in ((1, 0), (0, 1)):
                    k2 = ((sx + di) % nx, (sy + dj) % ny)
                    if eps[sx, sy] == eps[k2]:
                        continue
                    if di:          # right edge of (sx, sy): u = u1, v runs
                        J1 = 0.5 * (cm.v_bounds[sy + 1] - cm.v_bounds[sy])
                        V = 0.5 * (cm.v_bounds[sy] + cm.v_bounds[sy + 1]) \
                            + J1 * xg
                        U = np.full_like(V, cm.u_bounds[sx + 1])
                        g = cm.geom_points(sx, sy, U, V)
                        L += float(wg @ np.hypot(g[3], g[5])) * J1
                    else:
                        J1 = 0.5 * (cm.u_bounds[sx + 1] - cm.u_bounds[sx])
                        U = 0.5 * (cm.u_bounds[sx] + cm.u_bounds[sx + 1]) \
                            + J1 * xg
                        V = np.full_like(U, cm.v_bounds[sy + 1])
                        g = cm.geom_points(sx, sy, U, V)
                        L += float(wg @ np.hypot(g[2], g[4])) * J1
        per = sh.perimeter()
        if isinstance(sh, SinusoidalWall) and sh.width is None:
            per += C.P           # the half-plane also meets the seam x/y = p
        res[name] = {"area_rel": A / sh.area() - 1.0,
                     "perimeter_rel": L / per - 1.0, "min_detJ": mind,
                     "grid": list(cm.shape), "map": type(cm).__name__}
    print(res, flush=True)
    C.dump("c_unit_c12.json", res)


if __name__ == "__main__":
    globals()[sys.argv[1]]()

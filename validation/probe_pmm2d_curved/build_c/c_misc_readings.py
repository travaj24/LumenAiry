"""Small readings behind unit-test docstrings of
``tests/unit/test_pmm2d_staggered_curved_c.py``.

  c_misc_readings.py c4      the C4 pillar arms at M = 5 (primitive, the
                             hand-built lattice-wall map, the snapped circle)
  c_misc_readings.py d6      the SeparableStretch wall trap (verifier D6)
  c_misc_readings.py c11     the viewer outline deviation and its
                             fail-before; the section boundaries
  c_misc_readings.py ell     the ROTATED ellipse (moved corner vertices)
                             against its y-mirror: R(m, n; +a) = R(m, -n; -a)
  c_misc_readings.py c16     how far a SCALAR-route surrogate of a tensor
                             under the circle map is from the true
                             effective tensor sqrt(g) J^-1 eps J^-T
Output: c_misc_<arm>.json
"""
import sys

import _common as C
import numpy as np

from lumenairy.elements.pmm import Circle, compile_shapes  # noqa: E402

CM = C.CM


def c4():
    sys.path.insert(0, C.HERE)
    import c4_walls as W
    eps = np.ones((3, 3), complex)
    eps[1, 1] = C.EPS_P
    M = 5
    o, R, T, _J = C.solve(W.maps()["prim"], eps, M)
    v0 = C.vec(o, R, T)
    o, R, T, _J = C.solve(W.hand_map((0.4, 0.8)), eps, M)
    v1 = C.vec(o, R, T)
    _e, _x, _y, cms = compile_shapes(C.P, C.P, [Circle(0.6, 0.6, 0.2 *
                                                       np.sqrt(2.0), 4.0)],
                                     1.0)
    o, R, T, _J = C.solve(cms, eps, M)
    v2 = C.vec(o, R, T)
    res = {"env": C.env_record(), "M": M,
           "d_same_device_lattice_walls": float(np.max(np.abs(v0 - v1))),
           "d_snapped_failbefore": float(np.max(np.abs(v0 - v2)))}
    print(res, flush=True)
    C.dump("c_misc_c4.json", res)


def d6():
    trap = CM.SeparableStretch(np.array([0.0, 0.20, 0.65, C.P]), 3,
                               fx=CM.SineStretch(0.08), period_y=C.P)
    xb, _yb = trap.physical_walls()
    res = {"env": C.env_record(), "walls_asked": [0.20, 0.65],
           "walls_built": xb[1:3].tolist(),
           "max_off": float(np.max(np.abs(xb[1:3] - [0.20, 0.65])))}
    print(res, flush=True)
    C.dump("c_misc_d6.json", res)


def c11():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = C.PMM2DStackPure(C.P, C.P, n_modes=4)
    st.add_layer(0.3, shapes=[Circle(0.6, 0.6, C.R_CIRC, 4.0)],
                 background_eps=1.0)
    st.add_layer(0.2, eps=2.25)
    axes = st.plot_geometry()
    pts = np.concatenate([np.column_stack(ln.get_data())
                          for ln in axes[0].lines])
    dev = float(np.max(np.abs(np.hypot(pts[:, 0] - 0.6, pts[:, 1] - 0.6)
                              - C.R_CIRC)))
    c = st.cmap
    s = np.linspace(0.0, 1.0, 64)
    U = c.u_bounds[1] + s * (c.u_bounds[2] - c.u_bounds[1])
    raw = np.column_stack([U, np.full_like(U, c.v_bounds[1])])
    dev_fb = float(np.max(np.abs(np.hypot(raw[:, 0] - 0.6, raw[:, 1] - 0.6)
                                 - C.R_CIRC)))
    plt.close(axes[0].figure)
    runs = st._mapped_section(st._layers[0], 0, 0.6)
    edges = [r[1] for r in runs[:-1]]
    res = {"env": C.env_record(), "n_vertices": int(pts.shape[0]),
           "outline_dev": dev, "outline_dev_over_r": dev / C.R_CIRC,
           "uv_cell_dev_failbefore": dev_fb,
           "uv_cell_dev_over_r": dev_fb / C.R_CIRC,
           "section_edges": edges,
           "section_edge_err": float(np.max(np.abs(
               np.asarray(edges) - [0.6 - C.R_CIRC, 0.6 + C.R_CIRC])))}
    print(res, flush=True)
    C.dump("c_misc_c11.json", res)


def c16():
    """At every Gauss node of the 3 x 3 circle map: the TRUE effective
    tensor (in-plane block) T = sqrt(g) J^-1 eps_t J^-T of a block-form eps,
    against the best a SCALAR route can do, s sqrt(g) g^-1 with s the mean
    of eps_xx, eps_yy (the scalar route carries only the metric).  Relative
    max-norm difference per node, max over nodes (identity cells: 0 by
    construction for an isotropic eps, |aniso| for a tensor)."""
    from numpy.polynomial.legendre import leggauss
    _e, _x, _y, cm = compile_shapes(C.P, C.P, [Circle(0.6, 0.6, C.R_CIRC,
                                                      4.0)], 1.0)
    eps_t = np.array([[4.0, 0.3], [0.3, 3.0]])
    s = 0.5 * (eps_t[0, 0] + eps_t[1, 1])
    xg, _ = leggauss(16)
    worst, worst_map = 0.0, 0.0
    for sx in range(3):
        U = 0.5 * (cm.u_bounds[sx] + cm.u_bounds[sx + 1]) + 0.5 * (
            cm.u_bounds[sx + 1] - cm.u_bounds[sx]) * xg
        for sy in range(3):
            V = 0.5 * (cm.v_bounds[sy] + cm.v_bounds[sy + 1]) + 0.5 * (
                cm.v_bounds[sy + 1] - cm.v_bounds[sy]) * xg
            _X, _Y, xu, xv, yu, yv = cm.geom(sx, sy, U, V)
            for a in range(xg.size):
                for b in range(xg.size):
                    J = np.array([[xu[a, b], xv[a, b]], [yu[a, b], yv[a, b]]])
                    dj = np.linalg.det(J)
                    Ji = np.linalg.inv(J)
                    T = dj * Ji @ eps_t @ Ji.T
                    g = J.T @ J
                    Ts = s * dj * np.linalg.inv(g)
                    rel = np.max(np.abs(T - Ts)) / np.max(np.abs(T))
                    worst = max(worst, rel)
                    # the map's own share: T(J) vs T(I) = eps_t
                    worst_map = max(worst_map, np.max(np.abs(
                        T - eps_t)) / np.max(np.abs(eps_t)))
    res = {"env": C.env_record(), "eps_t": eps_t.tolist(),
           "scalar_surrogate_rel": float(worst),
           "identity_cell_surrogate_rel": float(
               np.max(np.abs(eps_t - s * np.eye(2))) / np.max(np.abs(eps_t))),
           "map_changes_tensor_rel": float(worst_map)}
    print(res, flush=True)
    C.dump("c_misc_c16.json", res)


def ell():
    from lumenairy.elements.pmm import Ellipse
    res = {"env": C.env_record()}
    for M in (4, 5):
        out = []
        for al in (np.deg2rad(20.0), -np.deg2rad(20.0)):
            st = C.PMM2DStackPure(C.P, C.P, n_superstrate=C.N_SUP,
                                  n_substrate=C.N_SUB, n_modes=M, n_orders=3)
            st.add_layer(C.DEPTH, shapes=[Ellipse(0.6, 0.6, 0.40, 0.28, 4.0,
                                                  angle=al)],
                         background_eps=1.0)
            st.set_source(C.WL)
            o, R, T, _J = st.solve()
            out.append((np.asarray(o), np.asarray(R), np.asarray(T)))
        (o1, R1, T1), (o2, R2, T2) = out
        idx = {(int(m), int(n)): k for k, (m, n) in enumerate(o1)}
        mir = same = 0.0
        for (m, n), k in idx.items():
            j = idx[(m, -n)]
            mir = max(mir, float(np.max(np.abs(R1[:, k] - R2[:, j]))),
                      float(np.max(np.abs(T1[:, k] - T2[:, j]))))
            same = max(same, float(np.max(np.abs(R1[:, k] - R1[:, j]))),
                       float(np.max(np.abs(T1[:, k] - T1[:, j]))))
        res[f"M{M}"] = {"mirror": mir, "unmirrored_failbefore": same,
                        "closure": float(np.max(np.abs(R1.sum(1) + T1.sum(1)
                                                       - 1)))}
    print(res, flush=True)
    C.dump("c_misc_ell.json", res)


if __name__ == "__main__":
    globals()[sys.argv[1]]()

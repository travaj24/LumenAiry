"""V5 -- identity, the corner rule on a REGULAR cell, F-B1 (what the seam
needs), and the curve-derivative trust gap.

  identity   TransfiniteMap(walls) with no curve vs the shipped kron
             assembly: operators and (oblique) R / T, on THIS verifier's
             fixture (4 x 4 non-uniform walls, absorbing cell, alpha0 != 0)
  corner0    the corner rule fed a cell with NO singular vertex: corners = []
             (the tensor nodes through the point path) and a forced corner on
             a smooth cell (Duffy on polynomial weights) -- bit-for-bit or
             round-off?
  kink       F-B1: a piecewise-AFFINE map whose normal derivative jumps at
             the periodic seam and at every interior line (passes the
             tangential check) vs the unmapped solve on the physical walls
             (same polynomial spaces: must agree to round-off); the 3 x 3
             circle re-parametrised with KINKED outer cells vs the plain c3
             circle (same physical circle: the difference must fall with M);
             and the NEGATIVE control: a map whose u = p side is shifted
             against u = 0 (fails the POSITION periodicity), forced past
             validate(): the answer must be wrong and must not converge away
  curveder   an EdgeCurve whose analytic derivative is NOT d/ds of its value
             (a user-supplied curve with a bug): accepted by TransfiniteMap?
             what does it do to the answer?
Output: v5_<mode>_<build>.json
"""
import sys

import _vcommon as C
import numpy as np

CM, TS = C.CM, C.TS


def rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def identity():
    out = {}
    wx = np.array([0.0, 0.2, 0.55, 0.8, C.P])
    wy = np.array([0.0, 0.3, 0.6, 0.85, C.P])
    rng = np.random.default_rng(3)
    eps = 1.0 + 3.0 * rng.random((4, 4)) + 0.2j * rng.random((4, 4))
    for M in (5, 7):
        s0 = TS.Granet2DTransverseE(C.P, C.P, wx, wy, M, eps, alpha0x=0.8,
                                    alpha0y=-0.5, k0=C.K0)
        tf = CM.TransfiniteMap(wx, wy)
        s1 = TS.Granet2DTransverseE(C.P, C.P, wx, wy, M, eps, alpha0x=0.8,
                                    alpha0y=-0.5, k0=C.K0, cmap=tf)
        out[f"ops_M{M}"] = {a: rel(getattr(s1, a), getattr(s0, a))
                            for a in ("Lmat", "Rmat", "Stt", "Schur")}
        out[f"nq_M{M}"] = s1._qrule.n
        out[f"corner_cells_M{M}"] = len(s1._qrule.points)
        # stack, conical
        o0, R0, T0 = C.solve_walls(wx[1:-1], wy[1:-1], eps, M, 0.3, 0.5)
        o1, R1, T1, _st, w = C.solve_map(tf, eps, M, 0.3, 0.5)
        out[f"RT_M{M}"] = float(max(np.max(np.abs(R1 - R0)),
                                    np.max(np.abs(T1 - T0))))
        print(M, out[f"ops_M{M}"], out[f"RT_M{M}"], flush=True)
    C.dump(f"v5_identity_{C.build_tag()}.json", out)


def corner0():
    out = {}
    cm, eps = C.vcircle3(0.36)
    M = 6
    tf = CM.TransfiniteMap(cm.u_walls, cm.v_walls)       # identity, no corner
    base = TS.Granet2DTransverseE(C.P, C.P, tf.u_walls, tf.v_walls, M, eps,
                                  k0=C.K0, cmap=tf)
    orig = TS._stag_map_singular_corners
    for tag, fake in (("empty_corner_list", {(1, 1): []}),
                      ("empty_list_two_cells", {(1, 1): [], (0, 2): []}),
                      ("forced_corner_smooth", {(1, 1): [(-1, -1)]}),
                      ("forced_4corners_smooth",
                       {(1, 1): [(-1, -1), (-1, 1), (1, -1), (1, 1)]})):
        TS._stag_map_singular_corners = lambda c, f=fake: f
        try:
            s = TS.Granet2DTransverseE(C.P, C.P, tf.u_walls, tf.v_walls, M,
                                       eps, k0=C.K0, cmap=tf)
        finally:
            TS._stag_map_singular_corners = orig
        out[tag] = {a: {"rel": rel(getattr(s, a), getattr(base, a)),
                        "bitwise": bool(np.array_equal(getattr(s, a),
                                                       getattr(base, a)))}
                    for a in ("Lmat", "Rmat", "Stt", "Schur")}
        print(tag, out[tag], flush=True)
    C.dump(f"v5_corner0_{C.build_tag()}.json", out)


def kink():
    out = {}
    # (1) piecewise-affine kinked map vs the unmapped physical walls
    uw = np.array([0.0, 0.3, 0.9, C.P])
    xi = np.array([0.0, 0.5, 0.9, C.P])            # physical x of the u walls
    yw = np.array([0.0, 0.3, 0.9, C.P])
    V = C.grid_vertices(uw, yw)
    V[:, :, 0] = xi[:, None]
    km = CM.TransfiniteMap(uw, yw, V)
    g0 = km.geom(0, 1, np.array([1e-9]), np.array([0.6]))
    g2 = km.geom(2, 1, np.array([C.P - 1e-9]), np.array([0.6]))
    out["affine_seam_xu_left_right"] = [float(g0[2][0, 0]), float(g2[2][0, 0])]
    eps = np.ones((3, 3), complex)
    eps[1, 1] = C.EPS_P
    out["affine"] = {}
    for M in (5, 7):
        for th, ph in ((0.0, 0.0), (0.4, 0.7)):
            o1, R1, T1, _s, _w = C.solve_map(km, eps, M, th, ph)
            o0, R0, T0 = C.solve_walls(xi[1:-1], yw[1:-1], eps, M, th, ph)
            d = float(max(np.max(np.abs(R1 - R0)), np.max(np.abs(T1 - T0))))
            out["affine"][f"M{M}_th{th}_ph{ph}"] = d
            print("affine kink", M, th, ph, d, flush=True)
    # (2) the c3 circle with kinked outer cells (u walls moved, vertex images
    #     = the physical 45-degree points) vs the plain c3 circle
    r = 0.36
    c3, e3 = C.vcircle3(r)
    h = r / np.sqrt(2)
    uk = np.array([0.0, 0.20, 0.20 + 2 * h, C.P])    # same middle width
    Vk = c3.vertex_images.copy()
    ck = CM.TransfiniteMap(uk, uk, Vk, dict(c3.curved_edges))
    g0 = ck.geom(0, 1, np.array([1e-9]), np.array([0.6]))
    g2 = ck.geom(2, 1, np.array([C.P - 1e-9]), np.array([0.6]))
    out["circle_kink_seam_xu_left_right"] = [float(g0[2][0, 0]),
                                             float(g2[2][0, 0])]
    out["circle_kink"] = {}
    for M in (6, 7, 8, 9):
        a = C.vec(*C.solve_map(c3, e3, M)[:3])
        b = C.vec(*C.solve_map(ck, e3, M)[:3])
        out["circle_kink"][M] = {"plain_vs_kinked": float(np.max(np.abs(a - b))),
                                 "kinked_vec": b, "plain_vec": a}
        print("circle kink", M, out["circle_kink"][M]["plain_vs_kinked"],
              flush=True)
    # (3) NEGATIVE control: the u = p side shifted in y against u = 0
    tf = CM.TransfiniteMap(np.array([0.0, 0.3, 0.9, C.P]),
                           np.array([0.0, 0.3, 0.9, C.P]))
    Vb = tf.vertex_images.copy()
    Vb[3, 1, 1] += 0.05
    Vb[3, 2, 1] += 0.05
    try:
        CM.TransfiniteMap(tf.u_walls, tf.v_walls, Vb)
        out["shifted_side_accepted"] = True
    except ValueError as e:
        out["shifted_side_accepted"] = False
        out["shifted_side_refusal"] = str(e)[:200]
    ov = CM.CellMap.validate
    CM.CellMap.validate = lambda self, n=7: self
    try:
        bad = CM.TransfiniteMap(tf.u_walls, tf.v_walls, Vb)
    finally:
        CM.CellMap.validate = ov
    out["shifted"] = {}
    for M in (5, 7, 9):
        a = C.solve_walls([0.3, 0.9], [0.3, 0.9], eps, M)
        b = C.solve_map(bad, eps, M)
        out["shifted"][M] = float(np.max(np.abs(C.vec(*a) - C.vec(*b[:3]))))
        print("shifted side (forced)", M, out["shifted"][M], flush=True)
    C.dump(f"v5_kink_{C.build_tag()}.json", out)


class BadSinusoid(CM.Sinusoid):
    """A user curve with a bug: the derivative is 10 % too large."""

    def __call__(self, s):
        v, d = super().__call__(s)
        d = d.copy()
        d[:, 0] *= 1.1
        return v, d


def curveder():
    out = {}
    good, eps = C.vsine_ridge(0.3, 0.9, 0.1)
    ed = {}
    for k, cv in good.curved_edges.items():
        ed[k] = BadSinusoid(cv.base, cv.amplitude, cv.period, cv.t0, cv.t1)
    try:
        bad = CM.TransfiniteMap(good.u_bounds, good.v_bounds,
                                good.vertex_images, ed)
        out["accepted"] = True
    except ValueError as e:
        out["accepted"] = False
        out["refusal"] = str(e)[:200]
        C.dump(f"v5_curveder_{C.build_tag()}.json", out)
        return
    out["RT_diff"] = {}
    for M in (5, 7):
        a = C.vec(*C.solve_map(good, eps, M)[:3])
        b = C.vec(*C.solve_map(bad, eps, M)[:3])
        out["RT_diff"][M] = float(np.max(np.abs(a - b)))
        print("bad derivative", M, out["RT_diff"][M], flush=True)
    C.dump(f"v5_curveder_{C.build_tag()}.json", out)


if __name__ == "__main__":
    {"identity": identity, "corner0": corner0, "kink": kink,
     "curveder": curveder}[sys.argv[1]]()

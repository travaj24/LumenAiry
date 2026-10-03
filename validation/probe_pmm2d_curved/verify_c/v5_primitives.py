"""V5 -- THE PRIMITIVES at their limits, against analytic geometry.

  geom   : every primitive (own parameters, off-centre, rotated, aspect 5,
           strong sinusoid), _geom.check exactness + mapped area / outline
           length vs analytic; the refusal limits (fillet radius at / around
           sqrt(2) 1e-3 p, a circle tangent to / overlapping the cell edge,
           a sinusoid touching the cell edge) -> RAISE or SOLVE.
  ladder <case> <M> : closure + R00 / T00 of a few limit devices (aspect-5
           ellipse, a sinusoid 1.5e-3 p from the cell edge, a fillet at
           1.5e-3 p vs the sharp Rect) for the convergence picture.
"""
import sys
import time
import warnings

import numpy as np
from _geom import check, verdict
from _vc import BUILD, dump
from numpy.polynomial.legendre import leggauss

from lumenairy.elements.pmm import (
    Circle,
    Ellipse,
    FilletRect,
    PMM2DStackPure,
    Rect,
    SinusoidalWall,
    compile_shapes,
)

warnings.simplefilter("ignore")
P = 1.2
OUT = {}


def area_len(cmap, cell, eps):
    """mapped area of cells with eps, and (separately) total det J"""
    xg, wg = leggauss(16)
    U, V = cmap.u_bounds, cmap.v_bounds
    A = 0.0
    for i in range(cmap.shape[0]):
        for j in range(cmap.shape[1]):
            if abs(cell[i, j] - eps) > 1e-12:
                continue
            uq = 0.5 * (U[i] + U[i + 1]) + 0.5 * (U[i + 1] - U[i]) * xg
            vq = 0.5 * (V[j] + V[j + 1]) + 0.5 * (V[j + 1] - V[j]) * xg
            _X, _Y, xu, xv, yu, yv = cmap.geom(i, j, uq, vq)
            det = xu * yv - xv * yu
            A += float(np.einsum("i,j,ij->", wg, wg, det)) * 0.25 * (
                U[i + 1] - U[i]) * (V[j + 1] - V[j])
    return A


def one(name, shapes, bg=1.0, px=P, py=P, eps_check=None):
    rec = {}
    try:
        cell, xw, yw, cm = compile_shapes(px, py, shapes, bg)
    except Exception as e:                         # noqa: BLE001
        rec = dict(outcome="RAISE", type=type(e).__name__,
                   message=str(e)[:400])
        OUT[name] = rec
        print(f"{name:40s} RAISE {type(e).__name__}: {str(e)[:150]}")
        return
    g = check(cm, [cell], [shapes], [bg])
    rec = dict(outcome="SOLVE", verdict=verdict(g), geometry=g,
               grid=list(cm.shape))
    if eps_check is not None:
        sh = shapes[-1]
        A = area_len(cm, cell, eps_check)
        rec["area_rel_err"] = abs(A - sh.area()) / sh.area()
    OUT[name] = rec
    L = g["layers"][0]
    print(f"{name:40s} SOLVE {rec['verdict']:9s} grid {cm.shape} paint "
          f"{L['paint_wrong']} side {L['boundary_side_mismatch']} cover "
          f"{L['outline_covered_max']:.1e} area "
          f"{rec.get('area_rel_err', float('nan')):.1e} detJmin "
          f"{g['detJ_min']:.2e} spread {g['sigma_spread_max']:.2f}")


def geom():
    rmin = np.sqrt(2.0) * 1e-3 * P
    one("fillet_offc_w_ne_h", [FilletRect(0.55, 0.62, 0.7, 0.4, 0.09, 4.0)],
        eps_check=4.0)
    one("fillet_r_1.414e-3p", [FilletRect(0.6, 0.6, 0.6, 0.6, 1.414e-3 * P,
                                          4.0)])
    one("fillet_r_sqrt2e-3p_exact", [FilletRect(0.6, 0.6, 0.6, 0.6, rmin,
                                                4.0)], eps_check=4.0)
    one("fillet_r_1.5e-3p", [FilletRect(0.6, 0.6, 0.6, 0.6, 1.5e-3 * P,
                                        4.0)], eps_check=4.0)
    one("fillet_r_1.3e-3p", [FilletRect(0.6, 0.6, 0.6, 0.6, 1.3e-3 * P,
                                        4.0)])
    one("fillet_r_0", [FilletRect(0.6, 0.6, 0.6, 0.6, 0.0, 4.0)])
    one("fillet_nonsquare_period", [FilletRect(0.6, 0.4, 0.6, 0.4,
                                               1.42e-3 * 0.8, 4.0)],
        px=1.2, py=0.8)
    one("circle_offc", [Circle(0.47, 0.71, 0.33, 4.0)], eps_check=4.0)
    one("circle_core_offc", [Circle(0.47, 0.71, 0.33, 4.0, core=0.4)],
        eps_check=4.0)
    one("circle_tangent_r0.5p", [Circle(0.6, 0.6, 0.6, 4.0)])
    one("circle_r0.5p_minus_1e-3p", [Circle(0.6, 0.6, 0.6 - 1.2e-3, 4.0)],
        eps_check=4.0)
    one("circle_overlap_r0.6p", [Circle(0.6, 0.6, 0.72, 4.0)])
    one("ellipse_aspect5", [Ellipse(0.6, 0.6, 0.5, 0.1, 4.0)],
        eps_check=4.0)
    one("ellipse_aspect5_rot30_offc", [Ellipse(0.62, 0.57, 0.4, 0.08, 4.0,
                                               angle=np.deg2rad(30))],
        eps_check=4.0)
    one("ellipse_rot44.9", [Ellipse(0.6, 0.6, 0.35, 0.2, 4.0,
                                    angle=np.deg2rad(44.9))], eps_check=4.0)
    try:
        Ellipse(0.6, 0.6, 0.35, 0.2, 4.0, angle=np.deg2rad(45.0))
        OUT["ellipse_rot45"] = "constructed"
    except ValueError as e:
        OUT["ellipse_rot45"] = "RAISE at construction: " + str(e)[:200]
    # the rotated-ellipse FOLD domain of a LONE ellipse (aspect x angle)
    dom = {}
    for asp in (1.05, 1.2, 1.5, 2.0, 3.0, 5.0):
        row = {}
        for ang in (5, 10, 15, 20, 25, 30, 35, 40, 44):
            b = 0.08 * 5 / asp if asp >= 3 else 0.2
            a = b * asp
            try:
                compile_shapes(P, P, [Ellipse(0.6, 0.6, a, b, 4.0,
                                              angle=np.deg2rad(ang))], 1.0)
                row[ang] = "ok"
            except ValueError as e:
                row[ang] = ("FOLD" if "FOLDS" in str(e) else
                            "RAISE:" + str(e)[:60])
        dom[f"aspect {asp}"] = row
        print("aspect", asp, row)
    OUT["rotated_ellipse_domain"] = dom
    one("sine_halfplane_x", [SinusoidalWall("x", 0.5, 0.2, eps=4.0)])
    one("sine_near_edge_1.5e-3p", [SinusoidalWall("x", 0.3, 0.3 - 1.5e-3 * P,
                                                  eps=4.0)])
    one("sine_touch_edge", [SinusoidalWall("x", 0.3, 0.3, eps=4.0)])
    one("sine_ridge_y_n3_phase", [SinusoidalWall("y", 0.3, 0.1,
                                                 period_count=3, phase=0.7,
                                                 eps=4.0, width=0.5)])
    one("sine_ridge_steep", [SinusoidalWall("x", 0.25, 0.18, eps=4.0,
                                            width=0.2)])
    one("rect_spanning_x", [Rect(0.6, 0.5, 1.2, 0.3, 4.0)])
    # SinusoidalWall area/perimeter vs brute force
    sw = SinusoidalWall("x", 0.3, 0.1, period_count=2, phase=0.3, eps=4.0,
                        width=0.4)
    compile_shapes(P, P, [sw], 1.0)
    t = np.linspace(0, P, 400001)
    w = sw._wall(t, 0.3)
    dl = np.sum(np.hypot(np.diff(w), np.diff(t)))
    OUT["sine_perimeter_rel_err"] = abs(2 * dl - sw.perimeter()) / (2 * dl)
    print("sine perimeter rel err vs polyline", OUT["sine_perimeter_rel_err"])
    from scipy.integrate import quad
    e = Ellipse(0.6, 0.6, 0.5, 0.1, 4.0)
    L = quad(lambda s: np.hypot(0.5 * np.sin(s), 0.1 * np.cos(s)), 0,
             2 * np.pi, limit=200)[0]
    OUT["ellipse_perimeter_rel_err"] = abs(L - e.perimeter()) / L
    print("ellipse perimeter rel err", OUT["ellipse_perimeter_rel_err"])
    dump(f"v5_primitives_geom_{BUILD}.json", OUT)


CASES = {
    "ellipse5": lambda: [Ellipse(0.6, 0.6, 0.5, 0.1, 4.0)],
    "ellipse5rot": lambda: [Ellipse(0.62, 0.57, 0.4, 0.08, 4.0,
                                    angle=np.deg2rad(30))],
    "sine_edge": lambda: [SinusoidalWall("x", 0.3, 0.3 - 1.5e-3 * P,
                                         eps=4.0)],
    "sine_mid": lambda: [SinusoidalWall("x", 0.6, 0.2, eps=4.0)],
    "fillet1.5e-3": lambda: [FilletRect(0.6, 0.6, 0.6, 0.6, 1.5e-3 * P,
                                        4.0)],
    "rect": lambda: [Rect(0.6, 0.6, 0.6, 0.6, 4.0)],
}


def ladder(case, M):
    t0 = time.time()
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=3)
    st.add_layer(0.5, shapes=CASES[case](), background_eps=1.0)
    st.set_source(1.0)
    o, R, T, J = st.solve()
    o = np.asarray(o)
    i0 = int(np.nonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    res = dict(case=case, M=M, R00=np.asarray(R)[:, i0].tolist(),
               T00=np.asarray(T)[:, i0].tolist(),
               closure=float(np.max(np.abs(np.asarray(R).sum(1)
                                           + np.asarray(T).sum(1) - 1))),
               grid=list(st._grid), wall_s=time.time() - t0)
    print(res, flush=True)
    dump(f"v5_ladder_{case}_M{M}_{BUILD}.json", res)


if __name__ == "__main__":
    if sys.argv[1] == "geom":
        geom()
    else:
        ladder(sys.argv[2], int(sys.argv[3]))

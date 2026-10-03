"""V8 -- THE VIEWERS against the analytic outlines (C11, every primitive).

For each primitive (one shape layer + a uniform layer) and for two merged
two-layer stacks: every vertex of every line ``plot_geometry`` draws on a
layer's panel must lie on that layer's analytic outline (or on the cell edge
for a half-plane sinusoid), and every visible outline point must be covered
by a drawn line (segment projection).  ``plot_section`` at a cut: the
material runs' boundaries vs the analytic intersections (root-found).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from _geom import _seg_dist, min_outline_dist  # noqa: E402
from _vc import BUILD, dump  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from lumenairy.elements.pmm import (  # noqa: E402
    Circle,
    Ellipse,
    FilletRect,
    PMM2DStackPure,
    SinusoidalWall,
)

P = 1.2
OUT = {}


def panel_check(ax, shapes):
    lines = [np.column_stack(ln.get_data()) for ln in ax.lines]
    if not lines:
        return dict(n_lines=0)
    pts = np.concatenate(lines)
    d = min_outline_dist(shapes, pts[:, 0], pts[:, 1])
    on_edge = (np.abs(pts[:, 0]) < 1e-12) | (np.abs(pts[:, 0] - P) < 1e-12) \
        | (np.abs(pts[:, 1]) < 1e-12) | (np.abs(pts[:, 1] - P) < 1e-12)
    off = np.where(on_edge, 0.0, d)
    cov = 0.0
    for sh in shapes:
        b = sh.boundary_points(400)
        cov = max(cov, float(np.max(_seg_dist(b, lines))))
    return dict(n_lines=len(lines), n_pts=int(pts.shape[0]),
                drawn_off_outline_max=float(off.max()),
                outline_not_drawn_max=cov)


def section_check(st, k, shapes, bg, y):
    runs = st._mapped_section(st._layers[k], 0, y)
    got = [r[1] for r in runs[:-1]]

    def f(x):
        e = bg
        for sh in shapes:
            if sh.contains(np.array([x]), np.array([y]))[0]:
                e = sh.eps
        return e
    xs = np.linspace(0, P, 4001)
    es = [f(x) for x in xs]
    want = []
    for i in range(len(xs) - 1):
        if es[i] != es[i + 1]:
            def g(x, i=i):
                return min_outline_dist(shapes, np.array([x]),
                                        np.array([y]))[0] * (
                    1 if f(x) == es[i] else -1)
            want.append(brentq(lambda x: min(
                [float(np.asarray(sh.signed_distance(x, y))) for sh in
                 shapes], key=abs), xs[i], xs[i + 1], xtol=1e-15))
    if len(got) != len(want):
        return dict(got=got, want=want, err=float("inf"))
    return dict(n=len(got), err=float(np.max(np.abs(np.subtract(got, want))))
                if got else 0.0)


cases = {
    "circle": [Circle(0.55, 0.63, 0.33, 4.0)],
    "circle_core": [Circle(0.6, 0.6, 0.36, 4.0, core=0.5)],
    "fillet": [FilletRect(0.6, 0.58, 0.7, 0.5, 0.08, 4.0)],
    "ellipse": [Ellipse(0.6, 0.6, 0.45, 0.2, 4.0)],
    "ellipse_rot": [Ellipse(0.6, 0.6, 0.40, 0.28, 4.0,
                            angle=np.deg2rad(20))],
    "ellipse_rot_neg": [Ellipse(0.58, 0.62, 0.40, 0.28, 4.0,
                                angle=np.deg2rad(-15))],
    "sine_ridge": [SinusoidalWall("x", 0.3, 0.12, eps=4.0, width=0.5)],
    "sine_halfplane_y": [SinusoidalWall("y", 0.5, 0.08, 2, 0.3, eps=4.0)],
}
for name, shapes in cases.items():
    st = PMM2DStackPure(P, P, n_modes=3)
    st.add_layer(0.3, shapes=shapes, background_eps=1.0)
    st.add_layer(0.2, eps=2.25)
    axes = st.plot_geometry()
    r = panel_check(axes[0], shapes)
    r["uniform_panel_lines"] = len(axes[1].lines)
    r["section_y0.6"] = section_check(st, 0, shapes, 1.0, 0.6)
    plt.close(axes[0].figure)
    ax = st.plot_section(y=0.6)
    plt.close(ax.figure)
    OUT[name] = r
    print(name, r, flush=True)

# merged two-layer drawings: each panel shows ITS layer's outline only
two = {
    "annulus": ([Circle(0.6, 0.6, 0.25, 4.0)], [Circle(0.6, 0.6, 0.45,
                                                       2.25)]),
    "circle_fillet": ([Circle(0.6, 0.6, 0.2, 4.0)],
                      [FilletRect(0.6, 0.6, 0.9, 0.9, 0.09, 2.25)]),
    "sine_circle": ([SinusoidalWall("x", 0.15, 0.05, eps=2.25)],
                    [Circle(0.6, 0.6, 0.3, 4.0)]),
}
for name, (l1, l2) in two.items():
    st = PMM2DStackPure(P, P, n_modes=3)
    st.add_layer(0.3, shapes=l1, background_eps=1.0)
    st.add_layer(0.2, shapes=l2, background_eps=1.0)
    axes = st.plot_geometry()
    r = {"layer1": panel_check(axes[0], l1), "layer2": panel_check(axes[1],
                                                                   l2)}
    # the OTHER layer's outline must NOT be drawn on a panel
    r["layer1_vs_layer2_outline"] = panel_check(axes[0], l2)
    r["section_l1"] = section_check(st, 0, l1, 1.0, 0.55)
    r["section_l2"] = section_check(st, 1, l2, 1.0, 0.55)
    plt.close(axes[0].figure)
    OUT[name] = r
    print(name, r, flush=True)
dump(f"v8_viewer_{BUILD}.json", OUT)

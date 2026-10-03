"""V2 -- THE MERGE RULE, adversarially (verifier, Phase C).

For each multi-shape / multi-layer scenario: does the merge SOLVE (and then
is the map EXACT against the analytic outlines -- _geom.check), RAISE (and
does the message name the shapes), or build a WRONG map silently?

Also: (a) re-measures the builder's 0.105 for the plan's macro-cell blend,
(b) the macro-cell blend vs the per-edge claims on the scenarios where the
per-edge rule refuses.

  python v2_merge.py            -> v2_merge_<build>.json
"""

import numpy as np
from _geom import check, verdict
from _vc import BUILD, dump

from lumenairy.elements.pmm import (
    Circle,
    Ellipse,
    FilletRect,
    Rect,
    SinusoidalWall,
    _curvemap as CM,
    shapes2d as SH,
)

OUT = {}


def run(name, px, py, layers, note=""):
    """layers: list of (shapes, bg)"""
    lab = [(f"layer {k + 1}", sh, bg) for k, (sh, bg) in enumerate(layers)]
    rec = {"note": note}
    try:
        U, V, cmap, cells, ident = SH._merge(px, py, lab)
    except Exception as e:                         # noqa: BLE001
        msg = str(e)
        rec["outcome"] = "RAISE"
        rec["type"] = type(e).__name__
        rec["message"] = msg[:600]
        names = [s.name for sh, _bg in layers for s in sh]
        rec["names_in_message"] = [n for n in names if n in msg]
        OUT[name] = rec
        print(f"{name:34s} RAISE {type(e).__name__}: {msg[:140]}")
        return None
    g = check(cmap, cells, [sh for sh, _bg in layers],
              [bg for _sh, bg in layers])
    rec.update(outcome="SOLVE", identity=bool(ident), geometry=g,
               verdict=verdict(g), grid=[len(U) - 1, len(V) - 1],
               fingerprint=cmap.fingerprint[:16])
    OUT[name] = rec
    L = g["layers"]
    print(f"{name:34s} SOLVE {verdict(g):9s} grid {len(U) - 1}x{len(V) - 1} "
          f"paint_wrong {[x['paint_wrong'] for x in L]} on "
          f"{sum(x['boundary_side_mismatch'] for x in L)}/{sum(x['boundary_points_checked'] for x in L)} cover "
          f"{max(x['outline_covered_max'] for x in L):.1e} detJmin "
          f"{g['detJ_min']:.2e} spread {g['sigma_spread_max']:.2f}")
    return cmap, cells


P = 1.2
c = 0.6
# ---- the builder's own nested case + controls ------------------------------
run("lone_circle", P, P, [([Circle(c, c, 0.36, 4.0)], 1.0)])
run("lone_circle_offcentre", P, P, [([Circle(0.5, 0.71, 0.3, 4.0)], 1.0)])
# ---- (2a) annulus stack: circle r1 in layer 1, concentric r2 in layer 2 ----
run("annulus_2layer", P, P, [([Circle(c, c, 0.25, 4.0)], 1.0),
                             ([Circle(c, c, 0.45, 2.25)], 1.0)])
run("annulus_2layer_offc", P, P, [([Circle(0.55, 0.62, 0.2, 4.0)], 1.0),
                                  ([Circle(0.55, 0.62, 0.42, 2.25)], 1.0)])
run("annulus_1layer", P, P, [([Circle(c, c, 0.45, 4.0),
                               Circle(c, c, 0.25, 1.0)], 1.0)])
run("annulus_ratio_1.2", P, P, [([Circle(c, c, 0.30, 4.0)], 1.0),
                                ([Circle(c, c, 0.36, 2.25)], 1.0)],
    "ratio < sqrt2: inner 45-deg wall sits inside the outer bulge zone?")
# ---- (2b) curves sharing an EDGE with different parameters -----------------
run("same_circle_two_layers", P, P, [([Circle(c, c, 0.36, 4.0)], 1.0),
                                     ([Circle(c, c, 0.36, 2.0)], 1.0)])
run("circle3x3_vs_circle5x5", P, P,
    [([Circle(c, c, 0.36, 4.0)], 1.0),
     ([Circle(c, c, 0.36, 2.0, core=0.5)], 1.0)],
    "SAME physical outline, two layouts")
run("circle_vs_ellipse_a_eq_b", P, P,
    [([Circle(c, c, 0.36, 4.0)], 1.0), ([Ellipse(c, c, 0.36, 0.36, 2.0)],
                                        1.0)], "SAME outline, EllipseArc")
run("circle_vs_r_1e-13", P, P,
    [([Circle(c, c, 0.36, 4.0)], 1.0),
     ([Circle(c, c, 0.36 * (1 + 1e-13), 2.0)], 1.0)])
run("circle_vs_r_1e-10", P, P,
    [([Circle(c, c, 0.36, 4.0)], 1.0),
     ([Circle(c, c, 0.36 * (1 + 1e-10), 2.0)], 1.0)])
run("circle_vs_r_1e-6", P, P,
    [([Circle(c, c, 0.36, 4.0)], 1.0),
     ([Circle(c, c, 0.36 * (1 + 1e-6), 2.0)], 1.0)])
run("fillet_vs_fillet_same_box_diff_r", P, P,
    [([FilletRect(c, c, 0.6, 0.6, 0.06, 4.0)], 1.0),
     ([FilletRect(c, c, 0.6, 0.6, 0.12, 2.0)], 1.0)],
    "same box, different radius: different curves on one corner")
run("fillet_vs_fillet_nested_diff_r", P, P,
    [([FilletRect(c, c, 0.5, 0.5, 0.05, 4.0)], 1.0),
     ([FilletRect(c, c, 0.9, 0.9, 0.12, 2.0)], 1.0)])
# ---- (2c) circle transition cell abutting a fillet's arc cell --------------
run("circle_beside_fillet", 2.0, 2.0,
    [([Circle(0.6, 1.0, 0.3, 4.0)], 1.0),
     ([FilletRect(1.4, 1.0, 0.6, 0.8, 0.1, 2.25)], 1.0)])
run("circle_wall_in_fillet_arc_row", 2.0, 2.0,
    [([Circle(0.6, 0.6, 0.0849, 4.0)], 1.0),
     ([FilletRect(1.4, 1.0, 0.6, 0.8, 0.1, 2.25)], 1.0)],
    "circle's top 45-deg wall v=0.66 cuts the fillet's arc row [0.629, 0.7]")
# ---- (2d) three circles in a row -------------------------------------------
run("three_circles_equal", 3.6, 1.2,
    [([Circle(0.6, c, 0.36, 4.0), Circle(1.8, c, 0.36, 4.0),
       Circle(3.0, c, 0.36, 4.0)], 1.0)])
run("three_circles_r_0.30_0.36_0.40", 3.6, 1.2,
    [([Circle(0.6, c, 0.36, 4.0), Circle(1.8, c, 0.30, 4.0),
       Circle(3.0, c, 0.40, 4.0)], 1.0)],
    "a supercell of different radii (the metasurface case)")
run("two_circles_r_0.30_0.40", 2.4, 1.2,
    [([Circle(0.6, c, 0.30, 4.0), Circle(1.8, c, 0.40, 4.0)], 1.0)])
run("two_circles_r_0.25_0.36", 2.4, 1.2,
    [([Circle(0.6, c, 0.25, 4.0), Circle(1.8, c, 0.36, 4.0)], 1.0)],
    "ratio 1.44 > sqrt 2")
run("two_circles_r_0.3_0.3_dy", 2.4, 1.2,
    [([Circle(0.6, 0.55, 0.3, 4.0), Circle(1.8, 0.65, 0.3, 4.0)], 1.0)],
    "equal radii, centres offset 0.1 in y")
run("two_circles_two_layers_r_0.3_0.4", 2.4, 1.2,
    [([Circle(0.6, c, 0.30, 4.0)], 1.0), ([Circle(1.8, c, 0.40, 4.0)], 1.0)])
# ---- (2e) a sinusoidal wall crossing a circle's transition cell -----------
run("sine_x_left_of_circle_A0.05", P, P,
    [([SinusoidalWall("x", 0.15, 0.05, eps=2.25)], 1.0),
     ([Circle(c, c, 0.3, 4.0)], 1.0)])
run("sine_x_left_of_circle_A0.14", P, P,
    [([SinusoidalWall("x", 0.15, 0.14, eps=2.25)], 1.0),
     ([Circle(c, c, 0.3, 4.0)], 1.0)], "max x of the wall 0.29 vs circle 0.30")
run("sine_x_crossing_circle", P, P,
    [([SinusoidalWall("x", 0.25, 0.1, eps=2.25)], 1.0),
     ([Circle(c, c, 0.3, 4.0)], 1.0)])
run("sine_y_under_circle_A0.05", P, P,
    [([SinusoidalWall("y", 0.15, 0.05, period_count=2, eps=2.25)], 1.0),
     ([Circle(c, c, 0.3, 4.0)], 1.0)], "the wall's h edge cut at the "
    "circle's u walls")
run("ridge_under_circle", P, P,
    [([SinusoidalWall("x", 0.2, 0.08, eps=2.25, width=0.8)], 1.0),
     ([Circle(c, c, 0.3, 4.0)], 1.0)],
    "the ridge (0.12..1.08 in x) and the circle overlap in plan view, "
    "outlines do not cross")
# ---- other common two-layer devices ----------------------------------------
run("stripe_under_circle_crossing", P, P,
    [([Rect(c, c, P, 0.2, 2.25)], 1.0), ([Circle(c, c, 0.36, 4.0)], 1.0)],
    "an electrode stripe under a pillar")
run("stripe_beside_circle", P, P,
    [([Rect(c, 0.1, P, 0.2, 2.25)], 1.0), ([Circle(c, c, 0.36, 4.0)], 1.0)])
run("rect_far_right_wall_in_bulge", 2.0, 1.2,
    [([Circle(0.6, 0.6, 0.36, 4.0)], 1.0),
     ([Rect(1.6, 0.85, 0.4, 0.3, 2.25)], 1.0)],
    "rect far to the right; its wall y=0.7 lies in the circle's bulge zone "
    "(0.855..0.96)? (y0 = 0.7, y1 = 1.0)")
run("rect_far_right_wall_above_bulge", 2.0, 1.2,
    [([Circle(0.6, 0.6, 0.36, 4.0)], 1.0),
     ([Rect(1.6, 1.08, 0.4, 0.2, 2.25)], 1.0)],
    "rect far right, walls 0.98 / 1.18: above the circle's top 0.96")
run("rect_far_right_wall_just_above", 2.0, 1.2,
    [([Circle(0.6, 0.6, 0.36, 4.0)], 1.0),
     ([Rect(1.6, 1.06 + 0.0015, 0.4, 0.2, 2.25)], 1.0)],
    "rect bottom wall at 0.9615: 1.5e-3 above the circle's top")
run("small_square_over_disk", P, P,
    [([Circle(c, c, 0.4, 4.0)], 1.0), ([Rect(c, c, 0.2, 0.2, 2.25)], 1.0)])
run("square_off_centre_over_disk", P, P,
    [([Circle(c, c, 0.4, 4.0)], 1.0),
     ([Rect(0.66, 0.53, 0.2, 0.14, 2.25)], 1.0)])
# ---- (3) a boundary passing exactly through a vertex another shape owns -----
h = 0.36 / np.sqrt(2)
run("rect_corner_at_circle_45deg", P, P,
    [([Circle(c, c, 0.36, 4.0)], 1.0),
     ([Rect(c + h + 0.1, c + h + 0.1, 0.2, 0.2, 2.25)], 1.0)],
    "rect corner exactly the circle's 45-deg point (touching)")
run("rect_corner_on_circle_30deg", P, P,
    [([Circle(c, c, 0.36, 4.0)], 1.0),
     ([Rect(c + 0.36 * np.cos(np.pi / 6) + 0.05,
            c + 0.36 * np.sin(np.pi / 6) + 0.05, 0.1, 0.1, 2.25)], 1.0)],
    "rect corner exactly ON the circle at 30 deg (touching)")
run("circle_through_rect_corner_same_layer", P, P,
    [([Rect(c + h + 0.1, c + h + 0.1, 0.2, 0.2, 2.25),
       Circle(c, c, 0.36, 4.0)], 1.0)])

# ---- (a) the builder's 0.105: the plan's macro-cell blend bends a wall -----
cm3, _w = CM._circle_map_3x3(P, 0.36)
U3, V3 = cm3.u_bounds, cm3.v_bounds
# a horizontal (u) wall at v0 inside the circle's TOP cell, as the plan's
# rule would place it: RefinedMap subdivides, the line takes the macro blend
bulge = {}
for v0 in (0.86, 0.88, 0.9, 0.96, 1.0, 1.1):
    Vf = np.sort(np.r_[V3, v0])
    rm = CM.RefinedMap(cm3, U3, Vf)
    j = int(np.searchsorted(Vf, v0))           # fine cell above v0 -> row j
    uu = np.linspace(U3[1], U3[2], 401)
    Y = rm.geom_points(1, j, uu, np.full_like(uu, v0))[1]
    bulge[f"v0={v0}"] = float(np.max(np.abs(Y - v0)))
# the extreme: v0 -> top edge of the disk cell (t -> 0)
t = 1e-9
v0 = V3[2] + t * (V3[3] - V3[2])
uu = np.linspace(U3[1], U3[2], 401)
Y = cm3.geom_points(1, 2, uu, np.full_like(uu, v0))[1]
bulge["v0->arc (limit)"] = float(np.max(np.abs(Y - v0)))
bulge["r - r/sqrt2"] = 0.36 * (1 - 1 / np.sqrt(2))
OUT["macro_cell_blend_bulge"] = bulge
print("macro-cell blend: max |y - v0| of a straight u-wall inside the "
      "circle's top cell:", {k: round(v, 4) for k, v in bulge.items()})

dump(f"v2_merge_{BUILD}.json", OUT)

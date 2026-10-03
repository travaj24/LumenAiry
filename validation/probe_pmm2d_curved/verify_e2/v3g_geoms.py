"""The verifier's E2-4 pairs and awkward geometries (P = 1.1)."""
import numpy as np
from v3g_fix import P

from lumenairy.elements.pmm.shapes2d import Circle, FilletRect, Rect, SinusoidalWall

H1 = 0.33 / np.sqrt(2.0)
# a sinusoid x = x0 + A sin(2 pi y / P + ph) through the circle's
# upper-right 45-degree point (0.52 + H1, 0.58 + H1) with slope
_xs, _ys = 0.52 + H1, 0.58 + H1
_ph45 = 0.4
_x0_45 = _xs - 0.08 * np.sin(2 * np.pi * _ys / P + _ph45)


def pairs(lossy=None):
    """name -> (layer1 shapes, layer2 shapes, t1, t2).  lossy: None,
    'l1', 'l2', 'both' -> add Im 0.35 / 0.25 to layer 1 / layer 2."""
    e1 = {"circ": 3.6, "c2": 3.2, "fil": 3.0}
    e2 = {"circ": 2.6, "c2": 2.4, "fil": 2.1}
    i1 = 0.35j if lossy in ("l1", "both") else 0.0
    i2 = 0.25j if lossy in ("l2", "both") else 0.0
    return {
        "i_circ_sin": ([Circle(0.52, 0.58, 0.33, e1["circ"] + i1)],
                       [SinusoidalWall("x", 0.5, 0.1, phase=0.8,
                                       eps=e2["circ"] + i2)], 0.26, 0.22),
        # the brief's suggestion (r 0.28 at (0.48, 0.5) over r 0.26 at
        # (0.68, 0.62)) is REFUSED by the curved mortar (a piece touching
        # singular vertices of both maps); this pair is an ACCEPTED crossing
        # one (v3g_singmap.py)
        "ii_circ_circ_refused": ([Circle(0.48, 0.5, 0.28, e1["c2"] + i1)],
                                 [Circle(0.68, 0.62, 0.26, e2["c2"] + i2)],
                                 0.24, 0.2),
        "ii_circ_circ": ([Circle(0.55, 0.55, 0.28, e1["c2"] + i1)],
                         [Circle(0.87, 0.57, 0.2, e2["c2"] + i2)],
                         0.24, 0.2),
        "iii_fil_sin": ([FilletRect(0.55, 0.55, 0.5, 0.5, 0.15,
                                    e1["fil"] + i1)],
                        [SinusoidalWall("y", 0.36, 0.1, phase=0.3,
                                        eps=e2["fil"] + i2)], 0.25, 0.2),
    }


def awkward():
    return {
        # sinusoid nearly tangent to the circle from OUTSIDE (gap 1e-4)
        "tangent_out": ([Circle(0.55, 0.55, 0.3, 3.6)],
                        [SinusoidalWall("x", 0.15, 0.0999, phase=-np.pi / 2,
                                        eps=2.6)]),
        # grazing crossing (overlap 5e-4)
        "graze_in": ([Circle(0.55, 0.55, 0.3, 3.6)],
                     [SinusoidalWall("x", 0.1505, 0.1, phase=-np.pi / 2,
                                     eps=2.6)]),
        # sinusoid through the circle's 45-degree (singular) vertex
        "sin_thru_45": ([Circle(0.52, 0.58, 0.33, 3.6)],
                        [SinusoidalWall("x", _x0_45, 0.08, phase=_ph45,
                                        eps=2.6)]),
        # a Rect whose straight wall line x = cx + r/sqrt2 passes exactly
        # through the circle's 45-degree points (and whose y-walls cut the
        # disk elsewhere)
        "rect_wall_at_45": ([Circle(0.52, 0.58, 0.33, 3.6)],
                            [Rect(0.52 + H1 + 0.15, 0.5, 0.3, 0.5, 2.6)]),
        # two crossing circles whose 45-degree points nearly coincide
        "circ_45_near": ([Circle(0.5, 0.5, 0.3, 3.2)],
                         [Circle(0.5 + 0.3 / np.sqrt(2) + 0.2 / np.sqrt(2)
                                 + 1e-4, 0.5 + 0.3 / np.sqrt(2)
                                 - 0.2 / np.sqrt(2), 0.2, 2.4)]),
        # a circle whose 45-degree point lies ON the other circle's outline
        "circ_45_on_outline": ([Circle(0.5, 0.5, 0.3, 3.2)],
                               [Circle(0.5 + 0.3 / np.sqrt(2) + 0.18,
                                       0.5 + 0.3 / np.sqrt(2), 0.18, 2.4)]),
    }

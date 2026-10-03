"""E2-6: oblique and conical incidence through the curved mortar on the E2-4
device (circle over a sinusoidal wall that crosses it): lossless closure at
(25 deg, 0) and (25 deg, 40 deg), and reflection RECIPROCITY -- the singular
values of the power-normalised 2 x 2 Jones block of reflection order (-1, 0)
against those of the reversed channel (the Phase C instrument).  The
fail-before pairs the forward block with the WRONG reverse order (0, 0).

Usage: python e2_6_angles.py <M>"""
import sys
import time

import numpy as np
from _common import (
    D1,
    D2,
    EPS_P,
    EPS_W,
    R_CIRC,
    dump,
    jones_block,
    order_index,
    reverse_angles,
    shapes_stack,
    solve,
)

from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall

M = int(sys.argv[1])
lay = [(D1, [Circle(0.6, 0.6, R_CIRC, EPS_P)], 1.0),
       (D2, [SinusoidalWall("x", 0.6, 0.12, eps=EPS_W)], 1.0)]
out = {"M": M}
for th_d, ph_d in ((25.0, 0.0), (25.0, 40.0)):
    th, ph = np.deg2rad(th_d), np.deg2rad(ph_d)
    t0 = time.perf_counter()
    st = shapes_stack(lay, M)
    o, R, T, J = solve(st, th, ph)
    k = order_index(o, (-1, 0))
    Nf = jones_block(st, k)
    tr, pr = reverse_angles(th, ph, -1, 0)
    st2 = shapes_stack(lay, M)
    o2, R2, T2, J2 = solve(st2, tr, pr)
    Nr = jones_block(st2, order_index(o2, (-1, 0)))
    Nw = jones_block(st2, order_index(o2, (0, 0)))
    sf = np.linalg.svd(Nf, compute_uv=False)
    sr = np.linalg.svd(Nr, compute_uv=False)
    sw = np.linalg.svd(Nw, compute_uv=False)
    key = f"th{th_d:g}_ph{ph_d:g}"
    out[key] = dict(closure=np.abs(R.sum(1) + T.sum(1) - 1.0),
                    closure_reverse=np.abs(R2.sum(1) + T2.sum(1) - 1.0),
                    recip=float(np.max(np.abs(sf - sr))),
                    wrong_pair=float(np.max(np.abs(sf - sw))),
                    reverse_angles_deg=[float(np.rad2deg(tr)),
                                        float(np.rad2deg(pr))],
                    wall=time.perf_counter() - t0)
    print(key, out[key])
dump(f"e2_6_angles_M{M}.json", out)

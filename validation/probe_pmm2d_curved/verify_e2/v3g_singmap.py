"""How often does the curved mortar REFUSE a CROSSING circle pair (a piece
touching singular vertices of both maps)?  Circle A r_a at (0.55, 0.55);
circle B of radius r_b at offset (d cos t, d sin t), crossing (|r_a - r_b|
< d < r_a + r_b) and inside the cell.  Only _assign_pairs is evaluated
(no solve)."""
import numpy as np
from _ve import dump
from v3g_fix import P

from lumenairy.elements.pmm import _curvemortar as CMM
from lumenairy.elements.pmm.shapes2d import Circle, compile_shapes


def view(shp):
    cell, xw, yw, cm = compile_shapes(P, P, [shp], 1.0)
    return CMM._MapView(cm, cm.u_walls, cm.v_walls, P, P)


ra = 0.28
A = view(Circle(0.55, 0.55, ra, 3.0))
rows = []
for rb in (0.12, 0.2, 0.28):
    for frac in (0.3, 0.6, 0.9):
        d = abs(ra - rb) + frac * (2 * min(ra, rb))
        for tdeg in (0, 15, 30, 45, 60, 90):
            t = np.deg2rad(tdeg)
            cx, cy = 0.55 + d * np.cos(t), 0.55 + d * np.sin(t)
            if not (rb < cx < P - rb and rb < cy < P - rb):
                continue
            try:
                B = view(Circle(cx, cy, rb, 2.0))
                CMM._assign_pairs(A, B)
                ok = True
            except NotImplementedError:
                ok = False
            except Exception as ex:  # noqa: BLE001
                ok = f"{type(ex).__name__}: {ex}"[:120]
            rows.append(dict(rb=rb, d=round(d, 4), t_deg=tdeg, cx=cx, cy=cy,
                             accepted=ok))
            print(rows[-1], flush=True)
acc = [r for r in rows if r["accepted"] is True]
print(f"accepted {len(acc)} / {len(rows)} crossing configurations")
dump("v3g_singmap", {"rows": rows, "n_accepted": len(acc),
                     "n_total": len(rows)})

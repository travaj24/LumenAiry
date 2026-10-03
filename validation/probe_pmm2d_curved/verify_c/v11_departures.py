"""V11 -- the builder's four departures from the plan, as a user meets them.

(c) raw eps_cell layers cannot mix with shape layers: what a user who
    z-staircases a TAPERED circular pillar gets (add_tapered_pillar after a
    shape layer; the same taper as concentric Circle layers -- merge, grid
    size, cost growth);
(d) a straight edge on a fillet's flat side raises: a rounded pillar on a
    pedestal of the same footprint (common), and on a wider pedestal;
(a) per-edge claims: how often a supercell of two circles of radii r1 < r2
    (side by side, same row) is REFUSED, over r2 / r1.
(b) the steep-cell split: see v5_primitives ladder (sine 1.5e-3 p from the
    cell edge vs mid-cell).
"""
import numpy as np
from _vc import BUILD, dump

from lumenairy.elements.pmm import Circle, FilletRect, PMM2DStackPure, Rect, shapes2d as SH

P = 1.2
OUT = {}

# (c) add_tapered_pillar after a shape layer
st = PMM2DStackPure(P, P, n_modes=3)
st.add_layer(0.2, shapes=[Circle(0.6, 0.6, 0.3, 4.0)], background_eps=1.0)
try:
    st.add_tapered_pillar(0.3, eps_pillar=4.0, eps_host=1.0,
                          x_bounds_bottom=(0.3, 0.9),
                          y_bounds_bottom=(0.3, 0.9),
                          x_bounds_top=(0.4, 0.8), y_bounds_top=(0.4, 0.8),
                          n_slices=3)
    OUT["tapered_after_shapes"] = "accepted"
except Exception as e:                             # noqa: BLE001
    OUT["tapered_after_shapes"] = f"{type(e).__name__}: {str(e)[:300]}"
print("tapered after shapes:", OUT["tapered_after_shapes"])
# the taper as concentric circle layers
for n in (2, 3, 4, 6):
    radii = np.linspace(0.3, 0.42, n)
    lay = [(f"layer {k + 1}", [Circle(0.6, 0.6, r, 4.0)], 1.0)
           for k, r in enumerate(radii)]
    try:
        U, V, cm, cells, ident = SH._merge(P, P, lay)
        OUT[f"taper_{n}_circles"] = dict(grid=list(cm.shape),
                                         radii=radii.tolist())
    except ValueError as e:
        OUT[f"taper_{n}_circles"] = "RAISE: " + str(e)[:160]
    print(n, "circle steps:", OUT[f"taper_{n}_circles"])

# (d) a rounded pillar on a pedestal
for w in (0.6, 0.6 + 1.2e-3, 0.62, 0.7):
    try:
        SH._merge(P, P, [("layer 1", [FilletRect(0.6, 0.6, 0.6, 0.6, 0.06,
                                                 4.0)], 1.0),
                         ("layer 2", [Rect(0.6, 0.6, w, w, 2.25)], 1.0)])
        OUT[f"pedestal_w{w:.4f}"] = "ok"
    except ValueError as e:
        OUT[f"pedestal_w{w:.4f}"] = "RAISE: " + str(e)[:160]
    print("pedestal", w, OUT[f"pedestal_w{w:.4f}"])

# (a) two circles side by side, r2 / r1
rows = {}
for ratio in (1.0, 1.05, 1.1, 1.2, 1.3, 1.4, 1.414, 1.42, 1.5, 1.7):
    r1 = 0.25
    r2 = r1 * ratio
    if r2 > 0.55:
        continue
    try:
        SH._merge(2 * P, P, [(None, [Circle(0.6, 0.6, r1, 4.0),
                                     Circle(1.8, 0.6, r2, 4.0)], 1.0)])
        rows[ratio] = "ok"
    except ValueError as e:
        rows[ratio] = "FOLD" if "FOLDS" in str(e) else str(e)[:60]
OUT["two_circles_ratio"] = rows
print("two circles r2/r1:", rows)
# offset in y: equal radii, centres dy apart
rows = {}
for dy in (0.0, 0.02, 0.05, 0.1, 0.2):
    try:
        SH._merge(2 * P, P, [(None, [Circle(0.6, 0.6, 0.3, 4.0),
                                     Circle(1.8, 0.6 + dy, 0.3, 4.0)], 1.0)])
        rows[dy] = "ok"
    except ValueError as e:
        rows[dy] = "FOLD" if "FOLDS" in str(e) else str(e)[:60]
OUT["two_circles_dy"] = rows
print("two equal circles, dy:", rows)
dump(f"v11_departures_{BUILD}.json", OUT)

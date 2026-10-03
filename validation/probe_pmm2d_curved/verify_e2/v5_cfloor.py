"""V5 (c): convergence_floor on a per-layer MAPPED (Phase E2) stack.  It
rebuilds each layer as eps_cell + x_walls / y_walls -- for a shape layer on
its own curved map that is the (u, v) cell on STRAIGHT walls, i.e. a
different device.  Measured here: the single-layer solve exactly as
convergence_floor builds it vs the same cell WITH the layer's map
(add_layer(eps_cell=, cmap=)) vs the shape layer alone (shared path), at
M = 3 and 5.  usage: v5_cfloor.py"""
import warnings

import numpy as np
from _ve import dump, solve

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall
from lumenairy.elements.pmm.stack2d_pure import _stag_interior

P = 1.2
out = {}
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=3,
                    n_orders=2, layer_grids="per-layer")
st.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.36, 4.0)], background_eps=1.0)
st.add_layer(0.25, shapes=[SinusoidalWall("x", 0.6, 0.12, eps=2.25)],
             background_eps=1.0)
L = st._layers[0]


def d(a, b):
    return float(max(np.abs(a[1] - b[1]).max(), np.abs(a[2] - b[2]).max()))


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for M in (3, 5):
        kw = dict(n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                  n_orders=2)
        a = PMM2DStackPure(P, P, layer_grids="per-layer", **kw)
        a.add_layer(0.3, eps_cell=L["eps_cell"], n_modes=M,
                    x_walls=_stag_interior(L["wx"]),
                    y_walls=_stag_interior(L["wy"]))
        ra = solve(a)
        b = PMM2DStackPure(P, P, layer_grids="per-layer", **kw)
        b.add_layer(0.3, eps_cell=L["eps_cell"], n_modes=M, cmap=L["cmap"])
        rb = solve(b)
        c = PMM2DStackPure(P, P, **kw)
        c.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.36, 4.0)],
                    background_eps=1.0)
        rc = solve(c)
        out[f"M{M}"] = dict(floor_device_vs_mapped=d(ra, rb),
                            mapped_cell_vs_shape_alone=d(rb, rc),
                            floor_device_vs_shape_alone=d(ra, rc),
                            R_floor=ra[1], R_mapped=rb[1])
    st.set_source(1.0)
    out["convergence_floor"] = st.convergence_floor()
print({k: ({kk: vv for kk, vv in v.items() if not kk.startswith("R_")}
           if isinstance(v, dict) else v) for k, v in out.items()})
dump("v5_cfloor", out)

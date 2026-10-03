"""V5 (c): convergence_floor on a per-layer shape stack that takes the
merged-map FAST PATH (circle over the non-crossing wall): does it run, and
on which device?  usage: v5_cfloor_fast.py"""
import warnings

from _ve import dump

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall

P = 1.2
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=3,
                    n_orders=2, layer_grids="per-layer")
st.add_layer(0.3, shapes=[Circle(0.6, 0.6, 0.36, 4.0)], background_eps=1.0)
st.add_layer(0.25, shapes=[SinusoidalWall("x", 0.12, 0.05, eps=2.25)],
             background_eps=1.0)
st.set_source(1.0)
out = dict(fast=bool(st._perlayer_fast_ok()),
           cell_shapes=[list(L["eps_cell"].shape) for L in st._layers],
           wall_counts=[len(L["wx"]) - 1 for L in st._layers])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        out["floor"] = st.convergence_floor()
    except Exception as ex:          # noqa: BLE001 -- recorded
        out["raised"] = f"{type(ex).__name__}: {str(ex)[:400]}"
print(out)
dump("v5_cfloor_fast", out)

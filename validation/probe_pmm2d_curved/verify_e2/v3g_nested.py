"""Non-crossing (nested / concentric) circle pairs: does the curved mortar
accept them when the fast path is off (a per-layer n_modes keyword)?
Kernel-level _assign_pairs only, plus the stack's fast-path flag."""
from _ve import dump
from v3g_fix import P, PMM2DStackPure

from lumenairy.elements.pmm import _curvemortar as CMM
from lumenairy.elements.pmm.shapes2d import Circle, compile_shapes


def view(shp):
    cell, xw, yw, cm = compile_shapes(P, P, [shp], 1.0)
    return CMM._MapView(cm, cm.u_walls, cm.v_walls, P, P)


out = {}
for name, (a, b) in {
        "concentric_0.36_0.30": (Circle(0.55, 0.55, 0.36, 3.0),
                                 Circle(0.55, 0.55, 0.30, 2.0)),
        "concentric_0.36_0.20": (Circle(0.55, 0.55, 0.36, 3.0),
                                 Circle(0.55, 0.55, 0.20, 2.0)),
        "nested_offc": (Circle(0.55, 0.55, 0.36, 3.0),
                        Circle(0.62, 0.5, 0.18, 2.0)),
        "disjoint": (Circle(0.3, 0.3, 0.2, 3.0),
                     Circle(0.8, 0.8, 0.2, 2.0))}.items():
    try:
        d, exc = CMM._assign_pairs(view(a), view(b))
        r = dict(accepted=True, default=d, n_exc=len(exc))
    except NotImplementedError as ex:
        r = dict(accepted=False, msg=str(ex)[:160])
    st = PMM2DStackPure(P, P, n_modes=4, layer_grids="per-layer",
                        n_substrate=1.5)
    st.add_layer(0.2, shapes=[a], background_eps=1.0)
    st.add_layer(0.2, shapes=[b], background_eps=1.0, n_modes=5)
    r.update(merge_ok=st._shapes_fast, fast_path=st._perlayer_fast_ok())
    out[name] = r
    print(name, r)
dump("v3g_nested", out)

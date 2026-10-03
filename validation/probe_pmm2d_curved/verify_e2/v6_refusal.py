"""V6a -- V-D3 refusals: each over-refused layout as ONE layer (shared and
per-layer stacks) and as TWO layers (shared: refusal text; per-layer: the
kept _merge_refusal and the fast-path verdict); the per-layer stack with
BOTH shapes in EACH of two layers; zero / negative thickness; the geometry
oracle of the macro-cell composite for every layout (EXACT for two-circle
rows; for the rect-at-30-deg layout the circle's blend bends the rect's
material walls).  No solves."""
import sys
import warnings

import numpy as np
from _ve import dump
from v6_layouts import LAYOUTS, macro

from lumenairy.elements.pmm import PMM2DStackPure, _curvemap as CM

sys.path.insert(0, __file__.rsplit("verify_e2", 1)[0] + "verify_c")
from _geom import check, verdict  # noqa: E402

warnings.simplefilter("ignore")
out = {}


def attempt(f):
    try:
        r = f()
        return {"ok": True, **(r or {})}
    except Exception as ex:   # noqa: BLE001 -- the refusal IS the datum
        return {"ok": False, "type": type(ex).__name__, "msg": str(ex)}


for name, (px, py, a, b) in LAYOUTS.items():
    res = {}
    for lg in ("shared", "per-layer"):
        def one(lg=lg):
            st = PMM2DStackPure(px, py, n_modes=4, n_orders=2, layer_grids=lg)
            st.add_layer(0.5, shapes=[a, b], background_eps=1.0)
        res[f"one_layer_{lg}"] = attempt(one)

    def two_shared():
        st = PMM2DStackPure(px, py, n_modes=4, n_orders=2)
        st.add_layer(0.25, shapes=[a], background_eps=1.0)
        st.add_layer(0.25, shapes=[b], background_eps=1.0)
    res["two_layers_shared"] = attempt(two_shared)

    def two_pl():
        st = PMM2DStackPure(px, py, n_modes=4, n_orders=2,
                            layer_grids="per-layer")
        st.add_layer(0.25, shapes=[a], background_eps=1.0)
        st.add_layer(0.25, shapes=[b], background_eps=1.0)
        return {"merge_refusal": st._merge_refusal,
                "fast_ok": bool(st._perlayer_fast_ok()),
                "own_grids": [list(L["own"]["cell"].shape[:2])
                              for L in st._layers]}
    res["two_layers_perlayer"] = attempt(two_pl)

    def both_each():
        st = PMM2DStackPure(px, py, n_modes=4, n_orders=2,
                            layer_grids="per-layer")
        st.add_layer(0.25, shapes=[a, b], background_eps=1.0)
        st.add_layer(0.25, shapes=[a, b], background_eps=1.0)
    res["both_shapes_in_each_of_two_perlayer"] = attempt(both_each)

    for t in (0.0, -0.1, 1e-12):
        def thin(t=t):
            st = PMM2DStackPure(px, py, n_modes=4, n_orders=2,
                                layer_grids="per-layer")
            st.add_layer(0.5, shapes=[a], background_eps=1.0)
            st.add_layer(t, shapes=[b], background_eps=1.0)
            return {"fast_ok": bool(st._perlayer_fast_ok())}
        res[f"second_layer_t={t!r}"] = attempt(thin)

    # macro-cell geometry oracle
    if name.startswith("rect"):
        base, _ = CM._circle_map_3x3(1.2, a.r, center=(a.cx, a.cy))
        x0, x1 = b.cx - b.w / 2, b.cx + b.w / 2
        y0, y1 = b.cy - b.h / 2, b.cy + b.h / 2
        ub = np.unique(np.r_[base.u_bounds, x0, x1])
        vb = np.unique(np.r_[base.v_bounds, y0, y1])
        while len(vb) < len(ub):
            k = int(np.argmax(np.diff(vb)))
            vb = np.insert(vb, k + 1, 0.5 * (vb[k] + vb[k + 1]))
        while len(ub) < len(vb):
            k = int(np.argmax(np.diff(ub)))
            ub = np.insert(ub, k + 1, 0.5 * (ub[k] + ub[k + 1]))
        cm = CM.RefinedMap(base, ub, vb)
        nx, ny = cm.shape
        eps = np.ones((nx, ny), complex)
        U, V = cm.u_bounds, cm.v_bounds
        for i in range(nx):
            for j in range(ny):
                X, Y = cm.geom(i, j, np.array([0.5 * (U[i] + U[i + 1])]),
                               np.array([0.5 * (V[j] + V[j + 1])]))[:2]
                if b.contains(X, Y)[0]:
                    eps[i, j] = 2.25
                elif a.contains(X, Y)[0]:
                    eps[i, j] = 4.0
    else:
        cm, eps = macro(name)
    g = check(cm, [eps], [[a, b]], [1.0])
    res["macro_geometry_verdict"] = verdict(g)
    res["macro_geometry"] = g
    res["macro_grid"] = list(cm.shape)
    out[name] = res
    print(name, {k: (v if not isinstance(v, dict) else
                     {kk: (str(vv)[:90]) for kk, vv in v.items()})
                 for k, v in res.items() if k != "macro_geometry"},
          flush=True)
dump("v6_refusal", out)

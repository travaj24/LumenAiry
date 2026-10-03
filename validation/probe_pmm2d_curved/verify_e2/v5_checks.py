"""V5 static checks (no large solves): the proposed two-sided default tests,
per-axis segment counts of compiled maps (does q-matching's x-only count
matter?), and convergence_floor on a per-layer MAPPED stack (is it
map-aware?).  usage: v5_checks.py"""
import warnings

import numpy as np
from _ve import dump, solve

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.pmm.shapes2d import Circle, Ellipse, Rect, SinusoidalWall, compile_shapes
from lumenairy.elements.pmm.stack2d_pure import _stag_walls_n

P = 1.2
out = {}


def circ(eps=4.0):
    return [Circle(0.6, 0.6, 0.36, eps)]


def sinw(x0=0.6, A=0.12, eps=2.25):
    return [SinusoidalWall("x", x0, A, eps=eps)]


def stack(layers, M, **kw):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=3, layer_grids="per-layer", **kw)
    for t, shp, bg, extra in layers:
        if shp is None:
            st.add_layer(t, eps=bg, **extra)
        else:
            st.add_layer(t, shapes=shp, background_eps=bg, **extra)
    return st


# ---- (1) the proposed two-sided default tests --------------------------
chk = {}
st = stack([(0.3, circ(), 1.0, {}), (0.25, sinw(), 1.0, {})], 4)
chk["qmatch_default"] = st._perlayer_modal_counts()             # [4, 6]
st = stack([(0.3, circ(), 1.0, {}), (0.25, sinw(), 1.0, {"n_modes": 4})], 4)
chk["qmatch_named_kept"] = st._perlayer_modal_counts()          # [4, 4]
st = stack([(0.3, circ(), 1.0, {"n_modes": 6}), (0.25, sinw(), 1.0, {})], 4)
chk["qmatch_named_raises_other"] = st._perlayer_modal_counts()  # [6, 9]
st = stack([(0.25, sinw(eps=1.0), 1.0, {}), (0.3, circ(), 1.0, {})], 4)
Ms = st._perlayer_modal_counts()
geo = st._perlayer_geometry(Ms)
chk["homog_not_matched"] = Ms
chk["homog_rides_circle"] = bool(geo[0][3] is st._layers[1]["cmap"]
                                 and geo[0][2] == 4)
st = stack([(0.3, circ(), 1.0, {}), (0.1, None, 1.7, {}),
            (0.25, sinw(), 1.0, {})], 4)
Ms = st._perlayer_modal_counts()
geo = st._perlayer_geometry(Ms)
chk["uniform_Ms_reported"] = Ms
chk["uniform_rides_above"] = bool(geo[1][3] is st._layers[0]["cmap"]
                                  and geo[1][2] == 4)
chk["uniform_geo_M_used"] = geo[1][2]
st = stack([(0.3, circ(), 1.0, {}), (0.1, None, 1.7, {"n_modes": 4}),
            (0.25, sinw(), 1.0, {})], 4)
geo = st._perlayer_geometry(st._perlayer_modal_counts())
chk["uniform_named_no_ride"] = bool(geo[1][3] is None)
chk["uniform_named_grid_N"] = _stag_walls_n(geo[1][0])
half = np.array([[4.0, 1.0], [1.0, 1.0]], complex)
st = PMM2DStackPure(P, P, n_modes=4, n_orders=1, layer_grids="per-layer")
st.add_layer(0.2, eps_cell=half, x_walls=[0.45], y_walls=[0.55])
st.add_layer(0.1, eps=2.0)
geo = st._perlayer_geometry(st._perlayer_modal_counts())
chk["unmapped_no_ride"] = bool(geo[1][3] is None
                               and np.ndim(geo[1][0]) == 0)
st = stack([(0.3, circ(), 1.0, {}), (0.25, sinw(0.12, 0.05), 1.0, {})], 4)
chk["fast_plain"] = bool(st._perlayer_fast_ok())
st = stack([(0.3, circ(), 1.0, {}),
            (0.25, sinw(0.12, 0.05), 1.0, {"n_modes": 4})], 4)
chk["fast_named_same_M"] = bool(st._perlayer_fast_ok())
out["default_checks"] = chk
print(chk)

# ---- (2) per-axis segment counts of compiled maps -------------------------
nxy = {}
for name, P_x, shp in (
        ("circle", P, circ()),
        ("sinus_x", P, sinw()),
        ("ellipse", P, [Ellipse(0.6, 0.6, 0.4, 0.25, 4.0)]),
        ("two_circles_2.4", 2.4, [Circle(0.6, 0.6, 0.3, 4.0),
                                  Circle(1.8, 0.6, 0.3, 4.0)]),
        ("rect_stripe", P, [Rect(0.6, 0.6, 0.4, P, 2.25)])):
    try:
        _e, U, V, cm = compile_shapes(P_x, P, shp, 1.0)
        if cm is None or not hasattr(cm, "u_walls"):
            nxy[name] = ("identity", len(np.atleast_1d(U)) - 1,
                         len(np.atleast_1d(V)) - 1)
        else:
            nxy[name] = (len(cm.u_walls) - 1, len(cm.v_walls) - 1)
    except Exception as ex:          # noqa: BLE001 -- recorded
        nxy[name] = f"{type(ex).__name__}: {str(ex)[:120]}"
out["map_Nxy"] = nxy
print(nxy)

# ---- (3) convergence_floor on a MAPPED per-layer stack ---------------------
cf = {}
st = stack([(0.3, circ(), 1.0, {}), (0.25, sinw(), 1.0, {})], 3)
st.set_source(1.0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        floor, per = st.convergence_floor()
        cf["floor"], cf["per_layer"] = floor, per
    except Exception as ex:          # noqa: BLE001 -- recorded
        cf["raised"] = f"{type(ex).__name__}: {str(ex)[:300]}"
    # what the circle layer's own residual SHOULD be (map-aware): the
    # circle alone on its map at M and M + 2, between the same half-spaces
    Ms = st._perlayer_modal_counts()
    vals = []
    for M in (Ms[0], Ms[0] + 2):
        s1 = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                            n_modes=M, n_orders=1)
        s1.add_layer(0.3, shapes=circ(), background_eps=1.0)
        o, R, T, J = solve(s1)
        vals.append(np.concatenate([R.ravel(), T.ravel()]))
    cf["circle_own_residual_mapaware"] = float(np.abs(vals[0]
                                                      - vals[1]).max())
    # and the device convergence_floor builds for that layer: the (u, v)
    # cell on straight walls (no map)
    L = st._layers[0]
    cf["layer0_has_cmap"] = L.get("cmap") is not None
    cf["layer0_kind"] = L["kind"]
out["convergence_floor"] = cf
print(cf)
dump("v5_checks", out)

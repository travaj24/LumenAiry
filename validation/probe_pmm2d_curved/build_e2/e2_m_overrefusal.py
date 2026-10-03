"""E2-M: the Phase C verifier's V-D3 OVER-REFUSALS of the per-edge merge
(two circles of radii 0.30 / 0.40 side by side, period 2.4 x 1.2; two equal
circles r = 0.3 offset 0.05 in y; a rectangle touching a circle at 30 deg)
-- what Phase E2's per-layer maps do with each when the two shapes sit in
DIFFERENT layers, and that the same pair in ONE layer still refuses.
Per scenario: the stack-wide merge verdict, the per-layer verdict, lossless
closure at M (arg 1), n_orders = 4 (every propagating order of the doubled
period)."""
import sys
import time
import warnings

import numpy as np
from _common import N_SUB, N_SUP, WL, PMM2DStackPure, dump

from lumenairy.elements.pmm.shapes2d import Circle, Rect

warnings.simplefilter("ignore")
M = int(sys.argv[1])
c = 0.6
SC = {
    "two_circles_r_0.30_0.40": (2.4, 1.2, [Circle(0.6, c, 0.30, 4.0)],
                                [Circle(1.8, c, 0.40, 4.0)]),
    "equal_circles_dy_0.05": (2.4, 1.2, [Circle(0.6, 0.575, 0.3, 4.0)],
                              [Circle(1.8, 0.625, 0.3, 4.0)]),
    "rect_touching_circle_30deg": (1.2, 1.2, [Circle(c, c, 0.36, 4.0)],
                                   [Rect(c + 0.36 * np.cos(np.pi / 6) + 0.05,
                                         c + 0.36 * np.sin(np.pi / 6) + 0.05,
                                         0.1, 0.1, 2.25)]),
}
out = {"M": M}
for name, (px, py, s1, s2) in SC.items():
    res = {}
    # one layer, both shapes: the per-edge merge
    try:
        st = PMM2DStackPure(px, py, n_modes=M, n_orders=4,
                            layer_grids="per-layer")
        st.add_layer(0.3, shapes=s1 + s2, background_eps=1.0)
        res["one_layer"] = "accepted"
    except ValueError as ex:
        res["one_layer"] = "refused: " + str(ex)[:120]
    # two layers: shared refuses, per-layer solves
    try:
        st = PMM2DStackPure(px, py, n_modes=M, n_orders=4)
        st.add_layer(0.3, shapes=s1, background_eps=1.0)
        st.add_layer(0.25, shapes=s2, background_eps=1.0)
        res["two_layers_shared"] = "accepted"
    except ValueError as ex:
        res["two_layers_shared"] = "refused: " + str(ex)[:120]
    t0 = time.perf_counter()
    st = PMM2DStackPure(px, py, n_superstrate=N_SUP, n_substrate=N_SUB,
                        n_modes=M, n_orders=4, layer_grids="per-layer")
    st.add_layer(0.3, shapes=s1, background_eps=1.0)
    st.add_layer(0.25, shapes=s2, background_eps=1.0)
    res["perlayer_fast_path"] = st._perlayer_fast_ok()
    st.set_source(WL)
    try:
        o, R, T, J = st.solve()
        res["perlayer_closure"] = np.abs(np.asarray(R).sum(1)
                                         + np.asarray(T).sum(1) - 1.0)
    except Exception as ex:
        res["perlayer_error"] = f"{type(ex).__name__}: {str(ex)[:160]}"
    res["wall"] = time.perf_counter() - t0
    out[name] = res
    print(name, res)
dump(f"e2_m_overrefusal_M{M}.json", out)

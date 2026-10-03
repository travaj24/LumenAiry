"""F-E3-1: the twin's FROZEN grid against the NumPy solve built at the moved
geometry (whose grid moves with the parameter).  M = 4.

    python f8_frozen_vs_moving.py

* the circle frozen at r0 = 0.36, evaluated at r0 (through traced shapes and
  through an explicit traced map) and at r = 0.33, against NumPy at 0.33;
* the rectangle frozen at w = 0.5, evaluated at w = 0.47, against NumPy at
  0.47 (affine cells: the two discretisations coincide);
* a concrete width that changes the topology (w = 1.2) is refused.
Output f8_frozen_vs_moving.json.
"""
import jax.numpy as jnp
from _e3common import (
    CIRC,
    WL,
    P,
    PMM2DStackPure,
    StagJaxTwin,
    absd,
    circle_stack,
    circle_traced_map,
    dump,
)

from lumenairy.elements.pmm import Circle, Rect

M = 4
out = {}
o, R, T, J = circle_stack(M).solve()
tw = StagJaxTwin(circle_stack(M))


def fr(r):
    p = tw.params()
    p["layers"][0]["shapes"] = [Circle(CIRC["c"], CIRC["c"], r, CIRC["eps"])]
    return tw.solve(p)


_o, R1, T1, J1 = fr(jnp.asarray(CIRC["r"]))
out["circle_traced_shapes_at_r0_vs_numpy"] = [absd(R1, R), absd(T1, T),
                                              absd(J1, J)]
_o, R2, T2, J2 = tw.solve(cmap=circle_traced_map(tw.cmap_ref,
                                                 jnp.asarray(CIRC["r"])))
out["circle_traced_map_at_r0_vs_numpy"] = [absd(R2, R), absd(T2, T),
                                           absd(J2, J)]
_o, R3, T3, J3 = fr(0.33)
o, R4, T4, J4 = circle_stack(M, r=0.33).solve()
out["circle_frozen_at_0.36_eval_0.33_vs_numpy_0.33"] = [absd(R3, R4),
                                                        absd(T3, T4),
                                                        absd(J3, J4)]


def rs(w):
    s = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                       n_orders=2)
    s.add_layer(0.4, shapes=[Rect(0.6, 0.6, w, 0.4, 4.0)], background_eps=1.0)
    s.set_source(WL)
    return s


twr = StagJaxTwin(rs(0.5))
pr = twr.params()
pr["layers"][0]["shapes"] = [Rect(0.6, 0.6, 0.47, 0.4, 4.0)]
_o, R5, T5, J5 = twr.solve(pr)
o, R6, T6, J6 = rs(0.47).solve()
out["rect_frozen_at_0.5_eval_0.47_vs_numpy_0.47"] = [absd(R5, R6),
                                                     absd(T5, T6),
                                                     absd(J5, J6)]
try:
    pr["layers"][0]["shapes"] = [Rect(0.6, 0.6, 1.2, 0.4, 4.0)]
    twr.solve(pr)
    out["rect_w_1.2"] = "ACCEPTED"
except ValueError as exc:
    out["rect_w_1.2"] = "refused: " + str(exc)[:120]
dump("f8_frozen_vs_moving.json", out)
print(out)

"""R12: the verifier's 'two_rects_reorder' event case alone (the WSL run of
v5_events.py dumped core after the seventh case): two rectangles, the
second's centre traced from 0.8; control 0.62, event 0.45 (the walls
reorder).  Value and d T00 / dx with the degenerate-cluster rule on and
off, and the NumPy stack at the event.  Run with ``python -X faulthandler``.

    python -X faulthandler r12_rects_crash.py [on|off|both]
"""
import sys

from _r2 import P, PMM2DStackPure, dump, jax, np

import lumenairy.elements.pmm._jax_twod_staggered as JT
from lumenairy.elements.pmm import Rect

WHICH = sys.argv[1] if len(sys.argv) > 1 else "both"


def shp(x, e=3.5):
    return [Rect(0.3, 0.6, 0.2, 0.3, e), Rect(x, 0.6, 0.2, 0.3, 2.2)]


st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=3,
                    n_orders=2, backend="jax")
st.add_layer(0.4, shapes=shp(0.8), background_eps=1.0)
st.set_source(1.0)
tw = st.jax_twin()


def f(x):
    p = tw.params()
    p["layers"][0]["shapes"] = shp(x)
    return st.solve(params=p)[2][0, tw.p0]


out = {}
for rule, gap in (("on", None), ("off", 0.0)):
    if WHICH not in (rule, "both"):
        continue
    JT._E3_EIG_CLUSTER_GAP_REL = gap
    try:
        for name, x in (("control", 0.62), ("event", 0.45)):
            v = float(jax.jit(lambda y: f(y))(x))
            print(rule, name, "value", v, flush=True)
            g = float(jax.jit(jax.grad(lambda y: f(y)))(x))
            print(rule, name, "grad", g, flush=True)
            out[f"{rule}_{name}"] = {"value": v, "grad": g}
    finally:
        JT._E3_EIG_CLUSTER_GAP_REL = None
sn = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=3,
                    n_orders=2)
sn.add_layer(0.4, shapes=shp(0.45), background_eps=1.0)
sn.set_source(1.0)
out["numpy_event_T00"] = float(sn.solve()[2][0, 12])
print("numpy event", out["numpy_event_T00"], flush=True)
print(dump("r12_rects_crash.json", out))
_ = np

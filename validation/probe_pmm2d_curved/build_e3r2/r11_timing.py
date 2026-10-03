"""R11: what the degenerate-cluster rule costs -- compiled forward and
gradient wall times (best of 5 after the compile), rule on vs off, for the
square pillar d/dw and the circle d/dr (a symmetric direction, but the
circle's eigs carry exact clusters, so the rule's lifted branch runs), and
for a cell WITHOUT clusters (the 0.5 x 0.4 rectangle's layer -- its
half-space eig still has round-off pairs, see r1).  Run SERIALLY.

    python r11_timing.py M
"""
import sys
import time

from _r2 import P, PMM2DStackPure, dump, jax, jnp, np

import lumenairy.elements.pmm._jax_twod_staggered as JT
from lumenairy.elements.pmm import Circle, Rect

M = int(sys.argv[1])
CASES = {
    "square_w": (0.5, lambda x: [Rect(0.6, 0.6, x, 0.5, 3.5)]),
    "circle_r": (0.36, lambda x: [Circle(0.6, 0.6, x, 4.0)]),
}
out = {"M": M}


def best(fn, x, k=5):
    jax.block_until_ready(fn(x))
    ts = []
    for _ in range(k):
        t0 = time.perf_counter()
        r = fn(x)
        jax.block_until_ready(r)
        ts.append(time.perf_counter() - t0)
    return min(ts)


for case, (x0, shp) in CASES.items():
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2, backend="jax")
    st.add_layer(0.45, shapes=shp(x0), background_eps=1.0)
    st.set_source(1.0)
    tw = st.jax_twin()

    def f(x, tw=tw, st=st, shp=shp):
        p = tw.params()
        p["layers"][0]["shapes"] = shp(x)
        _o, R, T, J = st.solve(params=p)
        return T[0, tw.p0]
    rec = {}
    for rule, gap in (("on", None), ("off", 0.0)):
        JT._E3_EIG_CLUSTER_GAP_REL = gap
        try:
            fw = jax.jit(lambda x: f(x))
            gr = jax.jit(jax.grad(lambda x: f(x)))
            t0 = time.perf_counter()
            jax.block_until_ready(gr(x0))
            tc = time.perf_counter() - t0
            rec[rule] = {"forward_s": best(fw, x0), "grad_s": best(gr, x0),
                         "grad_compile_s": tc}
        finally:
            JT._E3_EIG_CLUSTER_GAP_REL = None
    rec["grad_ratio"] = rec["on"]["grad_s"] / rec["off"]["grad_s"]
    rec["forward_ratio"] = rec["on"]["forward_s"] / rec["off"]["forward_s"]
    out[case] = rec
    print(case, M, "grad on %.3fs off %.3fs ratio %.2f | fwd ratio %.2f | "
          "compile on %.1fs off %.1fs" % (
              rec["on"]["grad_s"], rec["off"]["grad_s"], rec["grad_ratio"],
              rec["forward_ratio"], rec["on"]["grad_compile_s"],
              rec["off"]["grad_compile_s"]), flush=True)
print(dump(f"r11_timing_M{M}.json", out))
_ = np
_ = jnp

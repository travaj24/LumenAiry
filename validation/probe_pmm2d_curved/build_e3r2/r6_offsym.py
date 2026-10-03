"""R6: the NEAR-symmetric sweep -- an ellipse a = 0.33, b = a (1 + delta),
d / d a of (R00, T00 E_x, T00 E_y) by AD (degenerate-cluster rule on / off)
vs the twin's own Richardson FD at every delta (one twin, frozen at the
circle; a and b both traced, one compile each).  The verifier measured,
rule off (= the E3 build): 1.8e-3 (delta 0), 1.1e-1 (1e-14), 1.1e-3 (1e-13),
8.0e-6 (1e-12), 3.2e-6 (1e-11), 1.2e-8 (1e-10), <= 7e-11 from 1e-9.

    python r6_offsym.py M
"""
import sys

from _r2 import P, PMM2DStackPure, dump, jax, jnp, ladder, np

import lumenairy.elements.pmm._jax_twod_staggered as JT
from lumenairy.elements.pmm import Ellipse

M = int(sys.argv[1])
A0 = 0.33
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                    n_orders=2, backend="jax")
st.add_layer(0.45, shapes=[Ellipse(0.6, 0.6, A0, A0, 3.5)],
             background_eps=1.0)
st.set_source(1.0)
tw = st.jax_twin()


def f(a, b):
    p = tw.params()
    p["layers"][0]["shapes"] = [Ellipse(0.6, 0.6, a, b, 3.5)]
    _o, R, T, J = st.solve(params=p)
    return jnp.stack([R[0, tw.p0], T[0, tw.p0], T[1, tw.p0]])


fj = jax.jit(f)
grads = {}
for rule, gap in (("on", None), ("off", 0.0)):
    JT._E3_EIG_CLUSTER_GAP_REL = gap
    try:
        grads[rule] = jax.jit(jax.jacrev(lambda a, b: f(a, b)))
        grads[rule](A0, A0)            # compile under this setting
    finally:
        JT._E3_EIG_CLUSTER_GAP_REL = None
out = {"M": M, "rows": []}
for d in (0.0, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4):
    b = A0 * (1.0 + d)
    _r, fd, _c, rat = ladder(lambda a: fj(a, b), A0, [1e-3, 3e-4, 1e-4], P)
    sc = float(np.max(np.abs(fd)))
    row = {"delta": d, "FD_twin": fd.tolist(),
           "premise": np.asarray(rat).ravel().round(3).tolist()}
    for rule, g in grads.items():
        ad = np.asarray(g(A0, b))
        row[f"rule_{rule}"] = float(np.max(np.abs(ad - fd))) / sc
    out["rows"].append(row)
    print("delta %g  on %.2e  off %.2e" % (d, row["rule_on"], row["rule_off"]),
          flush=True)
print(dump(f"r6_offsym_M{M}.json", out))

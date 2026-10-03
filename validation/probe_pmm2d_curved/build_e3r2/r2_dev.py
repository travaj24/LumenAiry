"""R2 (development scan): the symmetry-breaking gradients of V-E3-1 under the
degenerate-cluster rule, against the VERIFIER's FD ladders (FD(twin) and
FD(numpy), read from ``verify_e3/v6b_symbreak_<CASE>_M<M>_<build>.json``).

    python r2_dev.py M CASE GAP SPLIT [GAP SPLIT ...]

GAP <= 0 switches the rule off.  Prints AD, its max relative difference
(over R00, T00 E_x, T00 E_y, relative to max |FD|) to both FDs, and the wall
time of the compiled gradient.
"""
import json
import os
import sys
import time

from _r2 import BUILD, HERE, P, PMM2DStackPure, dump, jax, jnp, np

import lumenairy.elements.pmm._jax_twod_staggered as JT
import lumenairy.elements.rcwa._core as RCC
from lumenairy.elements.pmm import Ellipse, FilletRect, Rect

RCC._EIG_CLUSTER_ORDER = int(os.environ.get("R2_ORDER", "4"))

M, CASE = int(sys.argv[1]), sys.argv[2]
pairs = [(float(a), float(b)) for a, b in zip(sys.argv[3::2], sys.argv[4::2])]
CASES = {
    "square_w": (0.5, lambda x: [Rect(0.6, 0.6, x, 0.5, 3.5)]),
    "square_wh": (0.5, lambda x: [Rect(0.6, 0.6, x, x, 3.5)]),
    "rect_nonsq_w": (0.5, lambda x: [Rect(0.6, 0.6, x, 0.4, 3.5)]),
    "ellipse_a": (0.33, lambda x: [Ellipse(0.6, 0.6, x, 0.33, 3.5)]),
    "fillet_sq_w": (0.6, lambda x: [FilletRect(0.6, 0.6, x, 0.6, 0.1,
                                               3.5)]),
}
x0, shp = CASES[CASE]
vf = os.path.join(HERE, "..", "verify_e3",
                  f"v6b_symbreak_{CASE}_M{M}_{BUILD}.json")
if not os.path.exists(vf):
    vf = os.path.join(HERE, "..", "verify_e3",
                      f"v6b_symbreak_{CASE}_M{M}_win.json")
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45, n_modes=M,
                    n_orders=2, backend="jax")
st.add_layer(0.45, shapes=shp(x0), background_eps=1.0)
st.set_source(1.0)
tw = st.jax_twin()


def f(x):
    p = tw.params()
    p["layers"][0]["shapes"] = shp(x)
    _o, R, T, J = st.solve(params=p)
    return jnp.stack([R[0, tw.p0], T[0, tw.p0], T[1, tw.p0]])


if os.path.exists(vf):
    V = json.load(open(vf))
    fdt, fdn = np.asarray(V["FD_twin"]), np.asarray(V["FD_numpy"])
    premise = None
else:                       # the twin's own Richardson FD (h^2 premise kept)
    from _r2 import ladder
    vf = "own FD(twin)"
    _rows, fdt, _ch, rat = ladder(jax.jit(f), x0, [1e-3, 3e-4, 1e-4], P)
    fdn = fdt * np.nan
    premise = np.asarray(rat).ravel().round(3).tolist()
sc = float(np.max(np.abs(fdt)))
out = {"premise": premise, "order": RCC._EIG_CLUSTER_ORDER, "M": M,
       "case": CASE, "fd_source": os.path.basename(vf),
       "FD_twin": fdt.tolist(), "FD_numpy": fdn.tolist(), "runs": []}
for gap, split in pairs:
    JT._E3_EIG_CLUSTER_GAP_REL = gap
    JT._E3_EIG_CLUSTER_SPLIT_REL = split
    try:
        gfun = jax.jit(jax.jacrev(f))
        t0 = time.perf_counter()
        g = np.asarray(gfun(x0))
        t1 = time.perf_counter()
        g = np.asarray(gfun(x0))
        t2 = time.perf_counter()
    finally:
        JT._E3_EIG_CLUSTER_GAP_REL = None
        JT._E3_EIG_CLUSTER_SPLIT_REL = None
    row = {"gap_rel": gap, "split_rel": split, "AD": g.tolist(),
           "vs_FDtwin": float(np.max(np.abs(g - fdt))) / sc,
           "vs_FDnumpy": float(np.max(np.abs(g - fdn))) / sc,
           "compile_plus_run_s": t1 - t0, "run_s": t2 - t1}
    out["runs"].append(row)
    print(CASE, M, "gap", gap, "split", split, "vsFDtwin %.2e vsFDnumpy %.2e"
          % (row["vs_FDtwin"], row["vs_FDnumpy"]),
          "run %.2fs" % row["run_s"], flush=True)
tag = os.environ.get("R2_NAME", "")
print(dump(f"r2_dev_{CASE}_M{M}{tag}.json", out))

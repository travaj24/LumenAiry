"""M7c -- disambiguate the TypeError M7b hit on the hybrid with a y-uniform
IN-PLANE tensor stripe + slant.  Is the SLANT implicated, or is it the y-uniform
tensor cell on its own?  Four arms: {stripe, pillar} x {slant, no slant}.
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slant_lib import assert_worktree, tensor_uniaxial            # noqa: E402
from lumenairy.elements.pmm import PMM2DStackHybrid               # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PX = PY = 0.75
WL, DEPTH = 1.0, 0.30
INPL = tensor_uniaxial(1.5, 1.7, np.pi / 2, np.deg2rad(25))
res = {"lumenairy": assert_worktree()}


def stripe(N):
    c = np.zeros((N, N, 3, 3), dtype=complex)
    c[:, :] = np.eye(3)
    c[:N // 2, :] = INPL
    return c


def pillar(N):
    c = np.zeros((N, N, 3, 3), dtype=complex)
    c[:, :] = np.eye(3)
    c[N // 4:N // 2, N // 4:N // 2] = INPL
    return c


for geom, fn in (("y_uniform_stripe", stripe), ("pillar", pillar)):
    for sname, sl in (("no_slant", None), ("slant", (0.5, 0.0))):
        key = f"{geom}/{sname}"
        try:
            st = PMM2DStackHybrid(PX, PY, n_superstrate=1.0, n_substrate=1.5,
                                  degree=11, n_orders=9, symmetry=False)
            if sl is None:
                st.add_layer(DEPTH, eps_tensor_cell=fn(24))
            else:
                st.add_layer(DEPTH, eps_tensor_cell=fn(24), slant=sl)
            st.set_source(WL, theta=np.deg2rad(25), phi=0.0)
            st.solve()
            res[key] = "OK"
        except Exception as exc:                              # noqa: BLE001
            res[key] = f"{type(exc).__name__}: {exc}"
        print(f"{key:32s} -> {str(res[key])[:120]}", flush=True)

with open(os.path.join(OUT, "m7c_hybrid_stripe_anomaly.json"), "w") as f:
    json.dump(res, f, indent=1, default=str)
print("WROTE results/m7c_hybrid_stripe_anomaly.json")

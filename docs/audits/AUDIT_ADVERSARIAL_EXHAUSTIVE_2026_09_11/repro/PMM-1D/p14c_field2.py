"""PROBE 14c: WHERE does the tangential-field jump across a z-interface live,
and does it converge with degree?  (Ex, Ey, Hx, Hy are TANGENTIAL to a z=const
interface and must be continuous; Ez may jump, Hz may not.)
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import PMMStack

per, wl = 0.8e-6, 1.0e-6
WALLS = [0.45 * per]          # layer-1 wall; layer 3 has one at 0.3*per


def run(degree):
    st = PMMStack(per, n_substrate=1.5, n_superstrate=1.0, degree=degree,
                  far_field_orders=15)
    st.add_layer(0.12e-6, segments=[(0.45, (2.0 + 0.05j) ** 2), (0.55, 1.0)])
    st.add_layer(0.08e-6, eps=(1.7 + 0.02j) ** 2)
    st.add_layer(0.15e-6, segments=[(0.3, (3.0 + 0.10j) ** 2),
                                    (0.7, (1.45 + 0.0j) ** 2)])
    st.set_source(wl, angle=np.deg2rad(22.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.solve(retain_internal=True)
        F = st.internal_field(np.array([0.12e-6 - 1e-11, 0.12e-6 + 1e-11]),
                              component="all", pol=0)
    return F


print(f"{'deg':>4} | " + " ".join(f"{c:>12}" for c in
                                  ("Ex all", "Ex off-wall", "Hy all",
                                   "Hy off-wall", "Hz all")))
for degree in (10, 14, 18, 22, 28, 34):
    F = run(degree)
    x = np.asarray(F["x"]) if "x" in F else None
    out = []
    for c in ("Ex", "Hy", "Hz"):
        v = np.asarray(F[c])
        j = np.abs(v[0] - v[1])
        s = max(float(np.max(np.abs(v))), 1e-300)
        out.append(float(np.max(j)) / s)
        if c != "Hz":
            if x is not None:
                mask = np.min(np.abs(x[None, :] - np.array(
                    WALLS + [0.3 * per, 0.0, per])[:, None]), axis=0) > 0.02 * per
            else:
                mask = np.ones(j.shape, bool)
            out.append(float(np.max(j[mask])) / s if mask.any() else np.nan)
    print(f"{degree:4d} | " + " ".join(f"{o:12.3e}" for o in
                                       [out[0], out[1], out[2], out[3],
                                        out[4]]))
print()
F = run(22)
print("keys:", sorted(F))
if "x" in F:
    x = np.asarray(F["x"])
    v = np.asarray(F["Ex"])
    j = np.abs(v[0] - v[1])
    k = int(np.argmax(j))
    print(f"largest Ex jump at x/P = {x[k]/per:.6f} "
          f"(grating walls at 0.45 and 0.30); jump {j[k]:.3e}")
    order = np.argsort(j)[::-1][:8]
    print("top-8 jump locations x/P:",
          np.array2string(x[order] / per, precision=4))

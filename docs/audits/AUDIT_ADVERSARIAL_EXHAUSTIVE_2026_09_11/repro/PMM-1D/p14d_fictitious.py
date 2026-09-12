"""PROBE 14d (decisive): the same z-interface continuity test across a
FICTITIOUS interface -- two adjacent layers with IDENTICAL segments.  There the
field is analytically smooth, the interface S-matrix is the identity, and ANY
jump is a reconstruction error in internal_field, not physics.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import PMMStack

per, wl = 0.8e-6, 1.0e-6
SEG = [(0.45, (2.0 + 0.05j) ** 2), (0.55, 1.0)]


def run(degree, split_z=0.06e-6, uniform=False):
    st = PMMStack(per, n_substrate=1.5, n_superstrate=1.0, degree=degree,
                  far_field_orders=13)
    segs = [(1.0, (2.0 + 0.05j) ** 2)] if uniform else SEG
    st.add_layer(split_z, segments=segs)              # layer 1
    st.add_layer(0.12e-6 - split_z, segments=segs)    # layer 2 == layer 1
    st.add_layer(0.10e-6, eps=(1.7 + 0.02j) ** 2)
    st.set_source(wl, angle=np.deg2rad(22.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.solve(retain_internal=True)
        F = st.internal_field(np.array([split_z - 1e-11, split_z + 1e-11]),
                              component="all", pol=0)
    out = {}
    for c in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"):
        v = np.asarray(F[c])
        out[c] = float(np.max(np.abs(v[0] - v[1]))) / max(
            float(np.max(np.abs(v))), 1e-300)
    return out, F


print("=== FICTITIOUS interface inside a PATTERNED region ===")
print(f"{'deg':>4} | " + " ".join(f"{c:>11}" for c in
                                  ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")))
for d in (10, 14, 18, 24):
    o, F = run(d)
    print(f"{d:4d} | " + " ".join(f"{o[c]:11.3e}" for c in
                                  ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")))

print()
print("=== FICTITIOUS interface inside a UNIFORM region (no walls at all) ===")
print(f"{'deg':>4} | " + " ".join(f"{c:>11}" for c in
                                  ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")))
for d in (10, 14, 18, 24):
    o, F = run(d, uniform=True)
    print(f"{d:4d} | " + " ".join(f"{o[c]:11.3e}" for c in
                                  ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")))

print()
o, F = run(18)
print("internal_field keys:", sorted(F))
if "x" in F:
    x = np.asarray(F["x"])
    v = np.asarray(F["Ex"])
    j = np.abs(v[0] - v[1])
    k = int(np.argmax(j))
    print(f"largest Ex jump at x/P = {x[k]/per:.6f} (segment wall at 0.45); "
          f"jump {j[k]:.3e}")
    idx = np.argsort(j)[::-1][:10]
    print("top-10 jump x/P:", np.array2string(x[idx] / per, precision=4))
    print("median |jump| over x:", float(np.median(j)))

"""PROBE 14: PMMStack.layer_absorption must sum to 1 - R - T, and
internal_field must be continuous across interfaces.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import PMMStack

per, wl = 0.8e-6, 1.0e-6


def mk(degree=18, ffo=15):
    st = PMMStack(per, n_substrate=1.5, n_superstrate=1.0, degree=degree,
                  far_field_orders=ffo)
    st.add_layer(0.12e-6, segments=[(0.45, (2.0 + 0.05j) ** 2),
                                    (0.55, 1.0)])
    st.add_layer(0.08e-6, eps=(1.7 + 0.02j) ** 2)
    st.add_layer(0.15e-6, segments=[(0.3, (3.0 + 0.10j) ** 2),
                                    (0.7, (1.45 + 0.0j) ** 2)])
    st.set_source(wl, angle=np.deg2rad(22.0))
    return st


print("=== 14a: sum(layer_absorption) vs 1 - R - T ===")
for degree in (12, 16, 20, 26):
    st = mk(degree)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve(retain_internal=True)
        A = st.layer_absorption()
    A = np.asarray(A)
    tot = R.sum(axis=1) + T.sum(axis=1)
    print(f"  deg={degree:3d}: 1-R-T = {1-tot}  sum(A) = {A.sum(axis=0)}  "
          f"resid = {np.abs((1-tot) - A.sum(axis=0))}")
    print(f"          per-layer A = \n{np.array2string(A, precision=8)}")

print()
print("=== 14b: by_material ===")
st = mk(20)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    st.solve(retain_internal=True)
    try:
        Am = st.layer_absorption(by_material=True)
        print("  ", Am)
    except Exception as e:
        print("   by_material:", type(e).__name__, str(e)[:120])

print()
print("=== 14c: LOSSLESS stack -> absorption must be ~0 ===")
st = PMMStack(per, n_substrate=1.5, n_superstrate=1.0, degree=20,
              far_field_orders=15)
st.add_layer(0.12e-6, segments=[(0.45, 4.0), (0.55, 1.0)])
st.add_layer(0.15e-6, segments=[(0.3, 9.0), (0.7, 2.1)])
st.set_source(wl, angle=np.deg2rad(22.0))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    o, R, T, J = st.solve(retain_internal=True)
    A = np.asarray(st.layer_absorption())
print(f"  1-R-T = {1-(R.sum(1)+T.sum(1))}   sum(A) = {A.sum(axis=0)}")

print()
print("=== 14d: internal_field continuity across an interface ===")
st = mk(22)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    st.solve(retain_internal=True)
zs = np.array([0.12e-6 - 1e-12, 0.12e-6 + 1e-12])
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F = st.internal_field(zs, component="all", pol="x")
    for k, v in (F.items() if isinstance(F, dict) else []):
        v = np.asarray(v)
        if v.ndim >= 1 and v.shape[0] == 2:
            d = np.max(np.abs(v[0] - v[1]))
            s = max(np.max(np.abs(v)), 1e-300)
            print(f"   {k:6s}: max jump across z=0.12um = {d:.3e} "
                  f"(rel {d/s:.3e})")
except Exception as e:
    import traceback
    traceback.print_exc()

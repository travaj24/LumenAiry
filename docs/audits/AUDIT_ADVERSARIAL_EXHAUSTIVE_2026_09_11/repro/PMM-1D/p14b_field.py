"""PROBE 14b: internal_field continuity across an interface + Poynting-flux
conservation with depth on a LOSSLESS stack.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import PMMStack

per, wl = 0.8e-6, 1.0e-6
st = PMMStack(per, n_substrate=1.5, n_superstrate=1.0, degree=22,
              far_field_orders=15)
st.add_layer(0.12e-6, segments=[(0.45, (2.0 + 0.05j) ** 2), (0.55, 1.0)])
st.add_layer(0.08e-6, eps=(1.7 + 0.02j) ** 2)
st.add_layer(0.15e-6, segments=[(0.3, (3.0 + 0.10j) ** 2),
                                (0.7, (1.45 + 0.0j) ** 2)])
st.set_source(wl, angle=np.deg2rad(22.0))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    st.solve(retain_internal=True)

for zi, lbl in ((0.12e-6, "L1|L2"), (0.20e-6, "L2|L3")):
    zs = np.array([zi - 1e-11, zi + 1e-11])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F = st.internal_field(zs, component="all", pol=0)
    print(f"--- interface {lbl} at z = {zi*1e6:.3f} um ---")
    if isinstance(F, dict):
        for k in sorted(F):
            v = np.asarray(F[k])
            if v.ndim >= 1 and v.shape[0] == 2:
                d = np.max(np.abs(v[0] - v[1]))
                s = max(float(np.max(np.abs(v))), 1e-300)
                print(f"   {k:>4}: max jump = {d:.3e}   rel = {d/s:.3e}")
    else:
        v = np.asarray(F)
        print("   shape", v.shape, " jump",
              np.max(np.abs(v[0] - v[1])))

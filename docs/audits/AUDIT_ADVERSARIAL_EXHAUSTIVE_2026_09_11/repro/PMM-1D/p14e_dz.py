"""PROBE 14e: is the apparent tangential-field 'jump' across a z-interface a
DISCONTINUITY, or just the physical z-variation over the finite sampling gap?
Scale the gap dz and watch.  A discontinuity is dz-INDEPENDENT; a sampling
artefact is proportional to dz.
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

print("interface z = 0.12 um, degree 22; rel jump of the TANGENTIAL fields")
print(f"{'dz (m)':>10} " + " ".join(f"{c:>11}" for c in
                                    ("Ex", "Ey", "Hx", "Hy", "Hz", "Ez")))
for dz in (1e-10, 1e-11, 1e-12, 1e-13, 1e-14):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F = st.internal_field(np.array([0.12e-6 - dz, 0.12e-6 + dz]),
                              component="all", pol=0)
    row = []
    for c in ("Ex", "Ey", "Hx", "Hy", "Hz", "Ez"):
        v = np.asarray(F[c])
        row.append(float(np.max(np.abs(v[0] - v[1])))
                   / max(float(np.max(np.abs(v))), 1e-300))
    print(f"{dz:10.0e} " + " ".join(f"{r:11.3e}" for r in row))

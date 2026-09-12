"""PROBE 15b: single-layer inclined-coordinate slanted PMM vs a z-STAIRCASE
of vertical layers solved by the (different) symmetric vertical cascade, and
vs the library's RCWA staircase.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import pmm_efficiency_1d_slanted, PMMStack

per, wl, depth, duty = 1.0e-6, 0.8e-6, 0.5e-6, 0.5
nr, ng, nsub, nsup = 2.0, 1.0, 1.444, 1.0
slant = np.deg2rad(20.0)


def staircase(ns, degree, pol_col):
    st = PMMStack(per, n_substrate=nsub, n_superstrate=nsup, degree=degree,
                  far_field_orders=15, min_feature=per * 1e-3)
    dz = depth / ns
    for i in range(ns):
        c = (((i + 0.5) * dz * np.tan(slant)) / per) % 1.0
        a = (c - duty / 2) % 1.0
        b = (a + duty) % 1.0
        if a < b:
            segs = [(a, ng ** 2), (duty, nr ** 2), (1 - b, ng ** 2)]
        else:
            segs = [(b, nr ** 2), (a - b, ng ** 2), (1 - a, nr ** 2)]
        segs = [(w, e) for w, e in segs if w > 1e-9]
        s = sum(w for w, _ in segs)
        st.add_layer(dz, segments=[(w / s, e) for w, e in segs])
    st.set_source(wl, angle=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve()
    return o, R[pol_col], T[pol_col]


for pol, col in (("te", 1), ("tm", 0)):
    print(f"--- {pol.upper()} (slant 20 deg, n=2.0/1.0, P=1um, wl=0.8um, "
          f"d=0.5um) ---")
    for deg in (16, 22, 28):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T = pmm_efficiency_1d_slanted(
                per, nr, ng, nsub, nsup, depth, duty, wl, slant_angle=slant,
                polarization=pol, degree=deg, far_field_orders=11,
                stabilize=False)
        i = {int(m): k for k, m in enumerate(o)}
        print(f"  slanted PMM deg={deg:3d} (1 layer): "
              f"R-1={R[i[-1]]:.8f} R0={R[i[0]]:.8f} R+1={R[i[1]]:.8f} "
              f"T0={T[i[0]]:.8f} tot={R.sum()+T.sum():.10f}")
    for ns in (8, 24, 64, 128):
        try:
            o2, R2, T2 = staircase(ns, 12, col)
        except Exception as e:
            print(f"  staircase ns={ns:4d}: {type(e).__name__}: {str(e)[:70]}")
            continue
        j = {int(m): k for k, m in enumerate(o2)}
        print(f"  PMM staircase ns={ns:4d} deg=12: "
              f"R-1={R2[j[-1]]:.8f} R0={R2[j[0]]:.8f} R+1={R2[j[1]]:.8f} "
              f"T0={T2[j[0]]:.8f} tot={R2.sum()+T2.sum():.10f}")
    print()

"""PROBE 15: the SLANTED 1-D grating (single-layer inclined-coordinate PMM)
vs an independent RCWA z-staircase of the same slant, and vs the vertical
limit (slant -> 0).
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import pmm_efficiency_1d_slanted, pmm_efficiency_1d
from lumenairy.elements.rcwa import RCWAStack

per, wl, depth, duty = 1.0e-6, 1.55e-6, 0.5e-6, 0.5
nr, ng, nsub, nsup = 3.48, 1.0, 1.444, 1.0

print("=== 15a: slant -> 0 must reproduce the VERTICAL solver ===")
for pol in ("te", "tm"):
    o0, R0, T0 = pmm_efficiency_1d(per, nr, ng, nsub, nsup, depth, duty, wl,
                                   polarization=pol, degree=20,
                                   far_field_orders=11, stabilize=False)
    for sl in (1e-9, 1e-6, 1e-3):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T = pmm_efficiency_1d_slanted(
                per, nr, ng, nsub, nsup, depth, duty, wl, slant_angle=sl,
                polarization=pol, degree=20, far_field_orders=11,
                stabilize=False)
        k = min(len(o), len(o0))
        ia, ib = (len(o) - k) // 2, (len(o0) - k) // 2
        print(f"  {pol} slant={sl:.0e}: max|dR|="
              f"{np.max(np.abs(R[ia:ia+k]-R0[ib:ib+k])):.3e} "
              f"max|dT|={np.max(np.abs(T[ia:ia+k]-T0[ib:ib+k])):.3e} "
              f"tot-1={R.sum()+T.sum()-1:+.2e}")

print()
print("=== 15b: 20 deg slant vs an RCWA z-staircase ===")
slant = np.deg2rad(20.0)
for pol in ("te", "tm"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T = pmm_efficiency_1d_slanted(
            per, nr, ng, nsub, nsup, depth, duty, wl, slant_angle=slant,
            polarization=pol, degree=24, far_field_orders=11,
            stabilize=False)
    i0 = int(np.where(o == 0)[0][0])
    im = int(np.where(o == -1)[0][0])
    print(f"  {pol} PMM(single slanted layer, deg 24): R0={R[i0]:.9f} "
          f"R-1={R[im]:.9f} T0={T[i0]:.9f} tot={R.sum()+T.sum():.10f}")
    for ns in (8, 24, 64, 128):
        st = RCWAStack(per, n_substrate=nsub, n_superstrate=nsup,
                       n_orders=121)
        dz = depth / ns
        for i in range(ns):
            # wall centre of slice i, shifted by the slant
            shift = (i + 0.5) * dz * np.tan(slant) / per
            c = (shift % 1.0)
            # ridge occupying [c-duty/2, c+duty/2) modulo 1
            a = (c - duty / 2) % 1.0
            b = (a + duty) % 1.0
            if a < b:
                segs = [(a, ng ** 2), (duty, nr ** 2), (1 - b, ng ** 2)]
            else:
                segs = [(b, nr ** 2), (a - b, ng ** 2), (1 - a, nr ** 2)]
            segs = [(w, e) for w, e in segs if w > 1e-12]
            s = sum(w for w, _ in segs)
            segs = [(w / s, e) for w, e in segs]
            st.add_layer(dz, segments=segs)
        st.set_source(wl, angle=0.0, polarization=pol)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = st.solve()
        oo = np.asarray(res.orders)
        RR = np.asarray(res.R)
        TT = np.asarray(res.T)
        j0 = int(np.where(oo == 0)[0][0])
        jm = int(np.where(oo == -1)[0][0])
        print(f"      rcwa staircase ns={ns:4d}: R0={RR[j0]:.9f} "
              f"R-1={RR[jm]:.9f} T0={TT[j0]:.9f} tot={RR.sum()+TT.sum():.10f}")

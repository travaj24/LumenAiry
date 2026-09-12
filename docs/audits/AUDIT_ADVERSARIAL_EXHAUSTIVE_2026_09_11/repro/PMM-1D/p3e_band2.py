"""PROBE 3e: is the [min_feature, ~8*min_feature] hazard band a property of the
ONE fixture, or general?  Second, independent geometry + a second min_feature.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as pst


def solve(per, wl, ang, e1, e2, s, degree, mf):
    pst.PMM_SLIVER_GUARD = False
    try:
        st = PMMStack(per, n_substrate=1.52, n_superstrate=1.0, degree=degree,
                      far_field_orders=13, min_feature=mf)
        st.add_layer(0.18e-6, segments=[(0.4, e1), (0.6, e2)])
        st.add_layer(0.22e-6, segments=[(0.4 + s, e2), (0.6 - s, e1)])
        st.set_source(wl, angle=ang)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = st.solve()
        m0 = int(np.where(o == 0)[0][0])
        return T[1, m0], float(np.max(R.sum(1) + T.sum(1)))
    except Exception as e:
        return None, None
    finally:
        pst.PMM_SLIVER_GUARD = True


per, wl, ang = 0.55e-6, 0.7e-6, np.deg2rad(31.0)
e1, e2 = 2.35 ** 2, 1.46 ** 2
degs = (10, 14, 18, 22, 26)
for mf in (per * 1e-5, per * 1e-4, per * 1e-3):
    print(f"### min_feature = {mf/per:.0e} of a period "
          f"(library default = 1e-05) ###")
    ref, _ = solve(per, wl, ang, e1, e2, 0.0, 30, mf)
    print(f"  reference (s=0, deg 30): T0 = {ref:.9f}")
    print(f"{'s/P':>9} | " + " ".join(f"{'d'+str(d):>13}" for d in degs))
    for s in (0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 15.0, 30.0, 100.0):
        sv = s * (mf / per)
        row = []
        for d in degs:
            v, tot = solve(per, wl, ang, e1, e2, sv, d, mf)
            if v is None:
                row.append("        ERR  ")
            else:
                bad = "*" if abs(v - ref) > 5e-3 else " "
                row.append(f"{v:12.8f}{bad}")
        print(f"{sv:9.1e} | " + " ".join(row) + f"   ({s:.1f}x mf)")
    print()

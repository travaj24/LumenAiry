"""PROBE 3b: fine scan of the sliver DANGER BAND (s between min_feature and
the width at which the sliver is harmless), asking:
  * is the pathology (degree-dependent, >1 efficiency) present?
  * does the guard REFUSE exactly there?
  * what does the REFUSED solve actually return, vs the continuous trend?
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as pst

per = 1.0e-6
wl = 1.55e-6
ang = np.deg2rad(12.0)
eps_hi, eps_lo = 3.48 ** 2, 1.444 ** 2


def solve(s, degree=16, guard=True, mf=None):
    old = pst.PMM_SLIVER_GUARD
    pst.PMM_SLIVER_GUARD = guard
    try:
        st = PMMStack(per, n_substrate=1.444, n_superstrate=1.0,
                      degree=degree, far_field_orders=11, min_feature=mf)
        st.add_layer(0.25e-6, segments=[(0.5, eps_hi), (0.5, eps_lo)])
        st.add_layer(0.25e-6, segments=[(0.5 + s, eps_hi), (0.5 - s, eps_lo)])
        st.set_source(wl, angle=ang)
        with warnings.catch_warnings(record=True) as W:
            warnings.simplefilter("always")
            try:
                o, R, T, J = st.solve()
            except Exception as e:
                kinds = sorted({type(w.message).__name__ for w in W})
                return None, None, None, kinds, type(e).__name__
            kinds = sorted({str(w.message).split(":")[0][:28] for w in W})
        return o, R, T, kinds, None
    finally:
        pst.PMM_SLIVER_GUARD = old


# reference: exact coincident-wall geometry, i.e. s = 0 (ridge 0.5 both layers)
o0, R0, T0, _k, _e = solve(0.0, degree=20, guard=False)
m0 = int(np.where(o0 == 0)[0][0])
T0ref = T0[1, m0]
# and the exact linear trend: dT/ds measured at s = 1e-3 (clean rung)
o1, R1, T1, _k, _e = solve(1e-3, degree=20, guard=False)
slope = (T1[1, m0] - T0ref) / 1e-3
print(f"reference T0(s=0) = {T0ref:.12f}, dT/ds = {slope:.6g}")
print()
print("s scan -- 'EXPECT' = T0ref + slope*s (the continuous physical trend)")
hdr = f"{'s':>10} {'EXPECT':>14} " + " ".join(f"{'d'+str(d):>13}" for d in
                                              (10, 12, 14, 16, 18, 20, 24))
print(hdr + "   guard@deg16")
for s in (3e-6, 5e-6, 8e-6, 1.0e-5, 1.2e-5, 1.5e-5, 2e-5, 3e-5, 5e-5, 8e-5,
          1e-4, 3e-4, 1e-3):
    exp = T0ref + slope * s
    vals = []
    for deg in (10, 12, 14, 16, 18, 20, 24):
        o, R, T, k, e = solve(s, degree=deg, guard=False)
        if e:
            vals.append("  RAISE      ")
        else:
            mm = int(np.where(o == 0)[0][0])
            v = T[1, mm]
            bad = "*" if abs(v - exp) > 1e-4 else " "
            vals.append(f"{v:13.9f}{bad}")
    o, R, T, k, e = solve(s, degree=16, guard=True)
    g = e if e else ("warn:" + ",".join(k) if k else "clean")
    print(f"{s:10.1e} {exp:14.9f} " + " ".join(vals) + f"   {g}")

"""PROBE 10e (v2): count the EXTRA full PMMStack.solve() calls the round-4
sliver arbiter runs.  Deterministic -> immune to CPU contention.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as pst

CNT = [0]
_orig_solve = PMMStack.solve


def counting_solve(self, **kw):
    CNT[0] += 1
    return _orig_solve(self, **kw)


PMMStack.solve = counting_solve


def build(nslice, s, degree=14):
    st = PMMStack(1.0e-6, n_substrate=1.444, n_superstrate=1.0, degree=degree,
                  far_field_orders=11)
    for i in range(nslice):
        st.add_layer(0.4e-6 / nslice,
                     segments=[(0.5 + i * s, 3.48 ** 2),
                               (0.5 - i * s, 1.444 ** 2)])
    st.set_source(1.55e-6, angle=np.deg2rad(12.0))
    return st


print(f"{'nslice':>7} {'s':>9} {'guard ON solves':>16} "
      f"{'guard OFF solves':>17} {'x':>6}  note")
for nslice, s in ((4, 3e-4), (8, 3e-4), (8, 1e-4), (16, 5e-5), (4, 0.0),
                  (1, 0.0)):
    res = {}
    for g in (True, False):
        pst.PMM_SLIVER_GUARD = g
        CNT[0] = 0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    build(nslice, s).solve()
                except Exception as e:
                    pass
        finally:
            pst.PMM_SLIVER_GUARD = True
        res[g] = CNT[0]
    note = "no sliver" if s == 0.0 else ""
    print(f"{nslice:7d} {s:9.1e} {res[True]:16d} {res[False]:17d} "
          f"{res[True]/max(res[False],1):6.2f}  {note}")

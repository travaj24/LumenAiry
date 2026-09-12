"""PROBE 10b: what does the round-4 sliver ARBITER cost on a stack that
carries a manufactured sliver?  (Round 4 removed the super-unity precondition,
so the arbiter's THREE extra solves are paid on EVERY solve of such a stack.)
"""
import sys, time, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as pst

per, wl = 1.0e-6, 1.55e-6
eps_hi, eps_lo = 3.48 ** 2, 1.444 ** 2


def build(nslice, s, degree):
    st = PMMStack(per, n_substrate=1.444, n_superstrate=1.0, degree=degree,
                  far_field_orders=11)
    for i in range(nslice):
        st.add_layer(0.4e-6 / nslice,
                     segments=[(0.5 + i * s, eps_hi), (0.5 - i * s, eps_lo)])
    st.set_source(wl, angle=np.deg2rad(12.0))
    return st


def timeit(fn, n=3):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


print(f"{'nslice':>7} {'deg':>4} {'guard ON (s)':>14} {'guard OFF (s)':>14} "
      f"{'ratio':>7}  note")
for nslice, degree, s in ((4, 14, 3e-4), (8, 14, 3e-4), (8, 20, 3e-4),
                          (16, 14, 1e-4), (4, 14, 0.0)):
    def go(g):
        def f():
            old = pst.PMM_SLIVER_GUARD
            pst.PMM_SLIVER_GUARD = g
            try:
                st = build(nslice, s, degree)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        st.solve()
                    except Exception:
                        pass
            finally:
                pst.PMM_SLIVER_GUARD = old
        return f
    t_on = timeit(go(True))
    t_off = timeit(go(False))
    note = "no sliver" if s == 0.0 else ""
    print(f"{nslice:7d} {degree:4d} {t_on:14.3f} {t_off:14.3f} "
          f"{t_on/max(t_off,1e-9):7.2f}  {note}")

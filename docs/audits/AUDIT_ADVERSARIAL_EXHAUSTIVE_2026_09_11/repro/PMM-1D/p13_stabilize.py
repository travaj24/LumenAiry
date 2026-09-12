"""PROBE 13: does the DEFAULT stabilize=True ever return a WORSE answer than
stabilize=False at the SAME requested degree?

_energy_clean_pick picks, on a structure some scanned degree calls lossless
(|tot-1| < 1e-6), the cluster member whose TOTAL is closest to 1.  On a
LOSSLESS grating the S-matrix conserves energy to ~1e-14 at EVERY degree, so
that tie-break is decided by round-off, and it can select a LOWER (less
converged) degree than the one the user asked for -- up to _PER_ORDER_TOL =
3e-3 away.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import pmm_efficiency_1d

CASES = [
    ("Si/SiO2 P=1.0um wl=1.55 17deg", 1.0e-6, 3.48, 1.444, 1.444, 1.0,
     0.45e-6, 0.5, 1.55e-6, 17.0),
    ("Si/air  P=0.9um wl=0.94  0deg", 0.9e-6, 3.48, 1.0, 1.5, 1.0,
     0.30e-6, 0.4, 0.94e-6, 0.0),
    ("TiO2/air P=0.4um wl=0.55 33deg", 0.4e-6, 2.4, 1.0, 1.46, 1.0,
     0.35e-6, 0.55, 0.55e-6, 33.0),
]
for pol in ("te", "tm"):
    for (lbl, per, nr, ng, nsub, nsup, dep, duty, wl, angd) in CASES:
        args = (per, nr, ng, nsub, nsup, dep, duty, wl)
        kw = dict(angle=np.deg2rad(angd), polarization=pol,
                  far_field_orders=15)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T = pmm_efficiency_1d(*args, degree=64, stabilize=False, **kw)
        ref = dict(zip([int(m) for m in o], R))
        print(f"--- {lbl}  {pol.upper()} ---   (reference: degree 64)")
        print(f"{'deg':>4} {'stab=False err':>16} {'stab=True err':>16} "
              f"{'worse by':>12}")
        for deg in (8, 10, 12, 14, 16, 20, 24, 28):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                oF, RF, TF = pmm_efficiency_1d(*args, degree=deg,
                                               stabilize=False, **kw)
                try:
                    oT, RT, TT = pmm_efficiency_1d(*args, degree=deg,
                                                   stabilize=True, **kw)
                except Exception as e:
                    print(f"{deg:4d}  stabilize=True RAISED "
                          f"{type(e).__name__}: {str(e)[:60]}")
                    continue
            eF = max(abs(RF[i] - ref[int(m)]) for i, m in enumerate(oF)
                     if int(m) in ref)
            eT = max(abs(RT[i] - ref[int(m)]) for i, m in enumerate(oT)
                     if int(m) in ref)
            flag = "  <== WORSE" if eT > 3 * eF + 1e-12 else ""
            print(f"{deg:4d} {eF:16.3e} {eT:16.3e} {eT/max(eF,1e-300):12.2f}x"
                  f"{flag}")
        print()

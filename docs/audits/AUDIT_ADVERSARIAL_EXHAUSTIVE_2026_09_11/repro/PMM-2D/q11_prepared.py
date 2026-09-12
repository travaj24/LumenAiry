from common import *
from lumenairy.elements.pmm import (pmm_efficiency_2d, prepare_pmm_2d,
                                    pmm_efficiency_2d_cell, prepare_pmm_2d_cell,
                                    pmm_efficiency_2d_vs_wavelength)
wl, Px, Py, dep = 1.0e-6, 0.9e-6, 0.9e-6, 0.3e-6
nsup, nsub = 1.0, 1.45
print("== PreparedPMM2D.solve(wl) vs pmm_efficiency_2d(wl): docstring claims ~1e-13 ==")
for sym in ("auto", False):
    o1,R1,T1 = pmm_efficiency_2d(Px,Py,12.25,1.0,(0.25*Px,0.75*Px),(0.25*Py,0.75*Py),
                                 nsub,nsup,dep,wl,degree=9,n_orders=4,symmetry=sym)[:3]
    p = prepare_pmm_2d(Px,Py,12.25,1.0,(0.25*Px,0.75*Px),(0.25*Py,0.75*Py),
                       nsub,nsup,dep,degree=9,n_orders=4)
    o2,R2,T2 = p.solve(wl)[:3]
    print(f"  pmm_efficiency_2d(symmetry={sym!r:6s}) vs prepared: max|dR|={np.max(np.abs(R1-R2)):.3e} "
          f"max|dT|={np.max(np.abs(T1-T2)):.3e}", flush=True)
print("  -> prepare_pmm_2d has NO `symmetry` parameter and PreparedPMM2D.solve never folds;")
print("     it also never calls _warn_lossless_energy_2d and has no `truncation` support.")
print()
print("== truncation='circular' on the direct entry vs the sweep helper ==")
o1,R1,T1 = pmm_efficiency_2d(Px,Py,12.25,1.0,(0.25*Px,0.75*Px),(0.25*Py,0.75*Py),
                             nsub,nsup,dep,wl,degree=9,n_orders=4,truncation="circular")[:3]
print(f"  direct circular: {len(o1)} orders retained")
o2,R2,T2 = pmm_efficiency_2d_vs_wavelength(Px,Py,12.25,1.0,(0.25*Px,0.75*Px),(0.25*Py,0.75*Py),
                                           nsub,nsup,dep,[wl],degree=9,n_orders=4)
print(f"  vs_wavelength  : {len(o2)} orders retained (no `truncation` kwarg exists)", flush=True)
try:
    pmm_efficiency_2d_vs_wavelength(Px,Py,12.25,1.0,(0.25*Px,0.75*Px),(0.25*Py,0.75*Py),
                                    nsub,nsup,dep,[wl],degree=9,n_orders=4,truncation="circular")
    print("  !! accepted truncation kwarg")
except TypeError as e:
    print("  TypeError as expected:", str(e)[:110], flush=True)
print()
print("== energy warning suppressed on the PREPARED path? ==")
import warnings as W
with W.catch_warnings(record=True) as rec:
    W.simplefilter("always")
    pmm_efficiency_2d(Px,Py,12.25,1.0,(0.25*Px,0.75*Px),(0.25*Py,0.75*Py),nsub,nsup,dep,wl,
                      degree=11,n_orders=15)
    n_direct = len(rec)
with W.catch_warnings(record=True) as rec:
    W.simplefilter("always")
    p = prepare_pmm_2d(Px,Py,12.25,1.0,(0.25*Px,0.75*Px),(0.25*Py,0.75*Py),nsub,nsup,dep,
                       degree=11,n_orders=15)
    r = p.solve(wl)
    n_prep = len(rec)
print(f"  warnings: direct entry = {n_direct}, prepared = {n_prep}")
print(f"  prepared energy sum R+T = {np.sum(r[1])+np.sum(r[2]):.8f}", flush=True)

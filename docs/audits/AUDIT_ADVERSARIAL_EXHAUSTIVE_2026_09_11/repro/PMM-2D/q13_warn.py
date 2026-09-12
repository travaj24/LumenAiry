from common import *
import warnings as W
from lumenairy.elements.pmm import pmm_efficiency_2d, prepare_pmm_2d
Px=Py=0.9e-6; wl=1.0e-6; dep=0.3e-6
args = (Px,Py,12.25,1.0,(0.25*Px,0.75*Px),(0.25*Py,0.75*Py),1.45,1.0,dep)
print("hunting a config where the DIRECT entry warns (lossless closure) ...")
found=None
for deg in (7,9,11):
    for nn in range(2, (3*deg-1)//2 + 1):
        try:
            with W.catch_warnings(record=True) as rec:
                W.simplefilter("always")
                r = pmm_efficiency_2d(*args, wl, degree=deg, n_orders=nn)
                e = float(np.sum(r[1])+np.sum(r[2]))
            if rec:
                print(f"  DIRECT warns at degree={deg} n_orders={nn}: E={e:.6f}  [{rec[0].category.__name__}]")
                found=(deg,nn,e); break
        except Exception:
            pass
    if found: break
if found:
    deg,nn,e = found
    with W.catch_warnings(record=True) as rec:
        W.simplefilter("always")
        p = prepare_pmm_2d(*args, degree=deg, n_orders=nn)
        rp = p.solve(wl)
        ep = float(np.sum(rp[1])+np.sum(rp[2]))
    print(f"  PREPARED at the same config: E={ep:.6f}  warnings={len(rec)}")
    print("  -> PreparedPMM2D.solve reproduces the SAME broken energy and emits NO warning"
          if len(rec)==0 else "  -> prepared warns too")
else:
    print("  none found in the scanned range")

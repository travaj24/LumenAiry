from common import *
import time
from lumenairy.elements.pmm import pmm_jones_2d, PMM2DStackHybrid
from lumenairy.elements.rcwa import rcwa_jones_2d
wl, Px, Py, dep = 1.0e-6, 0.47e-6, 0.47e-6, 0.3e-6
nsup, nsub = 1.0, 1.5
def iso(e): return e*np.eye(3, dtype=complex)
S=24;  c4 = np.zeros((S,S,3,3),dtype=complex); c4[...] = iso(1.0); c4[6:18,6:18] = iso(12.25)
S2=96; c4b= np.zeros((S2,S2,3,3),dtype=complex); c4b[...]= iso(1.0); c4b[24:72,24:72]= iso(12.25)
print("== C4 Si pillar (eps 12.25, 50% duty), normal incidence: Jxx + energy vs truncation ==")
for nn in (3,5,7,9,11):
    out=[f"n={nn:2d}"]
    for form in ("laurent","li"):
        t=time.time()
        try:
            o,R,T,J = pmm_jones_2d(Px,Py,c4,nsub,nsup,dep,wl,theta=0.0,phi=0.0,degree=11,n_orders=nn,formulation=form)
            out.append(f"| PMM-{form[:3]} Jxx={J[0,0]: .7f} E={R.sum(1)[0]+T.sum(1)[0]:.7f} {time.time()-t:5.1f}s")
        except Exception as e: out.append(f"| PMM-{form[:3]} ERR {e}")
    print("".join(out), flush=True)
for nn in (3,5,7,9,11,15):
    out=[f"n={nn:2d}"]
    for form in ("laurent","li"):
        t=time.time()
        try:
            o,R,T,J = rcwa_jones_2d(Px,Py,c4b,nsub,nsup,dep,wl,theta=0.0,phi=0.0,n_orders_x=nn,n_orders_y=nn,formulation=form)
            out.append(f"| RCWA-{form[:3]} Jxx={J[0,0]: .7f} E={R.sum(1)[0]+T.sum(1)[0]:.7f} {time.time()-t:5.1f}s")
        except Exception as e: out.append(f"| RCWA-{form[:3]} ERR {e}")
    print("".join(out), flush=True)

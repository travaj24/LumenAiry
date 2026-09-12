from common import *
from lumenairy.elements.pmm import (pmm_jones_2d_staggered,
                                    pmm_efficiency_2d_staggered,
                                    pmm_efficiency_2d_cell)
wl, Px, Py, dep = 1.0e-6, 0.9e-6, 0.9e-6, 0.3e-6
nsup, nsub = 1.0, 1.45
S = 12
cell = np.full((S,S), 1.0+0j); cell[3:9,3:9] = 12.25
c3 = np.zeros((S,S,3,3), dtype=complex); c3[...] = np.eye(3); c3[3:9,3:9] = 12.25*np.eye(3)

print("== twod_staggered IS LIVE: both public entries run ==")
o,R,T = pmm_efficiency_2d_staggered(Px,Py,cell,nsub,nsup,dep,wl,degree=6,n_orders=3)[:3]
print(f"  pmm_efficiency_2d_staggered OK, sum(R+T)={R.sum()+T.sum():.14f}", flush=True)
try:
    res = pmm_jones_2d_staggered(Px,Py,c3,nsub,nsup,dep,wl,degree=6,n_orders=3)
    o2,R2,T2,J2 = res
    print(f"  pmm_jones_2d_staggered OK, sum(R+T) row0={R2.sum(1)[0]+T2.sum(1)[0]:.14f}  Jxx={J2[0,0]:.8f}", flush=True)
except Exception as e:
    print("  pmm_jones_2d_staggered:", type(e).__name__, str(e)[:200], flush=True)

print()
print("== no-floor claim: energy must be n_orders-INVARIANT and track only degree M ==")
for nn in (2,3,4):
    o,R,T = pmm_efficiency_2d_staggered(Px,Py,cell,nsub,nsup,dep,wl,degree=6,n_orders=nn)[:3]
    print(f"  n_orders={nn}: sum(R+T)={R.sum()+T.sum():.14f}", flush=True)
for M in (5,6,8):
    o,R,T = pmm_efficiency_2d_staggered(Px,Py,cell,nsub,nsup,dep,wl,degree=M,n_orders=3)[:3]
    print(f"  degree(M)={M}: sum(R+T)={R.sum()+T.sum():.14f}  T00={T[np.all(o==0,1)][0]:.10f}", flush=True)
print("  HYBRID on the same grating for contrast:")
for nn in (5,9,11):
    o,R,T = pmm_efficiency_2d_cell(Px,Py,cell,nsub,nsup,dep,wl,degree=11,n_orders=nn)[:3]
    print(f"  hybrid n_orders={nn}: sum(R+T)={R.sum()+T.sum():.10f}  T00={T[np.all(o==0,1)][0]:.10f}", flush=True)

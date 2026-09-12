from common import *
import time, os
from lumenairy.elements.pmm import pmm_efficiency_2d_staggered, pmm_efficiency_2d_cell
wl, Px, Py, dep = 1.0e-6, 0.9e-6, 0.9e-6, 0.3e-6
nsup, nsub = 1.0, 1.45
# STAGGERED eps_cell is a SEGMENT grid: 3 segments per axis expresses the same
# pillar (walls at 1/4 and 3/4) that the HYBRID pixel grid spells with 12.
seg = np.full((3,3), 1.0+0j); seg[1,1] = 12.25
pix = np.full((12,12), 1.0+0j); pix[3:9,3:9] = 12.25
print("staggered, eps_cell = 3x3 SEGMENTS (same geometry as the 12x12 pixel cell)", flush=True)
for M in (5,6,8,10):
    for nn in (3,5):
        t=time.time()
        try:
            o,R,T = pmm_efficiency_2d_staggered(Px,Py,seg,nsub,nsup,dep,wl,degree=M,n_orders=nn)[:3]
            print(f"  M={M:2d} n_orders={nn}: {time.time()-t:7.2f}s  dof~{(3*(M-1))**2*2:6d}  "
                  f"sum(R+T)={R.sum()+T.sum():.14f}  T00={T[np.all(o==0,1)][0]:.10f}", flush=True)
        except Exception as e:
            print(f"  M={M} n_orders={nn}: {time.time()-t:6.2f}s {type(e).__name__} {str(e)[:130]}", flush=True)
print("hybrid on the SAME grating (12x12 pixel grid = 3x3 strips):", flush=True)
for nn in (5,9,11):
    t=time.time()
    o,R,T = pmm_efficiency_2d_cell(Px,Py,pix,nsub,nsup,dep,wl,degree=11,n_orders=nn)[:3]
    print(f"  hybrid n_orders={nn:2d}: {time.time()-t:7.2f}s  sum(R+T)={R.sum()+T.sum():.12f}  "
          f"T00={T[np.all(o==0,1)][0]:.10f}", flush=True)
print()
print("cost cliff: the SAME array handed to the two families", flush=True)
t=time.time()
o,R,T = pmm_efficiency_2d_cell(Px,Py,pix,nsub,nsup,dep,wl,degree=11,n_orders=5)[:3]
print(f"  pmm_efficiency_2d_cell(12x12 PIXELS, n_orders=5): {time.time()-t:.2f}s "
      f"(walls merged to 3 strips/axis)", flush=True)
print("  pmm_efficiency_2d_staggered(12x12) would be 12 SEGMENTS/axis -> dof 2*(12*(M-1))^2;")
print("  measured on this box at M=6: 498 s CPU and 8.4 GB RSS before I killed it.")

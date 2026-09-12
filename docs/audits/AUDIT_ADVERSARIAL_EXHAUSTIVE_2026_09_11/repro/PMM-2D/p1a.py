import sys, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
sys.path.insert(0, ".")
np.set_printoptions(precision=8, linewidth=220)
from lumenairy.elements.pmm import pmm_jones_2d
from lumenairy.elements.rcwa import rcwa_jones_2d
from tmm import tmm_single_layer

wl, Px, Py, dep = 1.0e-6, 0.5e-6, 0.5e-6, 0.3e-6
nsup, nsub, nlay = 1.0, 1.5, 2.0
S=32
cell = np.zeros((S,S,3,3), dtype=complex)
cell[...] = nlay**2*np.eye(3)
print("=== (a) uniform layer vs independent TMM ===")
for thd in (0.0, 30.0):
    th = np.deg2rad(thd)
    o,R,T,J = pmm_jones_2d(Px,Py,cell,nsub,nsup,dep,wl,theta=th,phi=0.0,degree=5,n_orders=3)
    o2,R2,T2,J2 = rcwa_jones_2d(Px,Py,cell,nsub,nsup,dep,wl,theta=th,phi=0.0,n_orders_x=3,n_orders_y=3)
    rx, rs = tmm_single_layer(nsup,nlay,nsub,dep,wl,th)
    print(f"-- theta={thd} deg")
    print("   PMM  J =", J.ravel())
    print("   RCWA J =", J2.ravel())
    print("   TMM  rx,rs =", rx, rs)
    print("   PMM Jxx-rx =", abs(J[0,0]-rx), " Jxx+rx =", abs(J[0,0]+rx),
          "| Jyy-rs =", abs(J[1,1]-rs), " Jyy+rs =", abs(J[1,1]+rs))
    print("   |R+T| pmm =", R.sum(axis=1), T.sum(axis=1))

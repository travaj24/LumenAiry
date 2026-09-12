import sys, numpy as np, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"); sys.path.insert(0,".")
np.set_printoptions(precision=10, linewidth=240)
from lumenairy.elements.pmm import pmm_jones_2d
from lumenairy.elements.rcwa import rcwa_jones_2d
from tmm import tmm_single_layer
wl, Px, Py, dep = 1.0e-6, 0.47e-6, 0.47e-6, 0.3e-6
nsup, nsub, nlay = 1.0, 1.5, 2.0
S=32
cell = np.zeros((S,S,3,3), dtype=complex); cell[...] = nlay**2*np.eye(3)
print("=== (a') uniform, non-degenerate period ===")
for thd in (0.0, 30.0):
    th=np.deg2rad(thd)
    o,R,T,J = pmm_jones_2d(Px,Py,cell,nsub,nsup,dep,wl,theta=th,phi=0.0,degree=5,n_orders=3)
    rx,rs = tmm_single_layer(nsup,nlay,nsub,dep,wl,th)
    print(f" th={thd}: |Jxx-rx|={abs(J[0,0]-rx):.3e} |Jyy-rs|={abs(J[1,1]-rs):.3e} |Jxy|={abs(J[0,1]):.3e} |Jyx|={abs(J[1,0]):.3e}")
    # phi = 45 deg: te/tm no longer x/y
    o,R,T,J45 = pmm_jones_2d(Px,Py,cell,nsub,nsup,dep,wl,theta=th,phi=np.pi/4,degree=5,n_orders=3)
    c,s = np.cos(np.pi/4), np.sin(np.pi/4)
    # rotate: E_x,E_y -> (p,s) basis. p along (c,s), s along (-s,c)
    Rm = np.array([[c,s],[-s,c]])
    Jps = Rm@J45@Rm.T
    print(f"   phi=45: J in (p,s) basis diag = {np.diag(Jps)}  offdiag={Jps[0,1]:.3e},{Jps[1,0]:.3e}")
    print(f"           expect rx={rx:.10f} rs={rs:.10f}")

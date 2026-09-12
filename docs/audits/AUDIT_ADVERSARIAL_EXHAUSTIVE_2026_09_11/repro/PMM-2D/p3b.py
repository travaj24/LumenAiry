import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1"); os.environ.setdefault("OMP_NUM_THREADS","1")
import sys, numpy as np, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"); sys.path.insert(0,".")
warnings.simplefilter("ignore")
np.set_printoptions(precision=10, linewidth=240)
from lumenairy.elements.pmm import PMM2DStackHybrid, pmm_efficiency_2d_cell
wl, Px, Py, dep = 1.0e-6, 0.47e-6, 0.47e-6, 0.3e-6
nsup, nsub = 1.0, 1.5
S=24
cx = np.full((S,S), 1.0+0j); cx[6:18,:] = 12.25      # x-patterned
cy = cx.T.copy()                                      # y-patterned

def j00(o,R,T,J): return J
print("== 90-deg rotation covariance: PMM2DStackHybrid, single 1-D-grating layer ==")
print("   (x-patterned, incident E_x) vs (y-patterned, incident E_y): must be EQUAL")
for form in ("li","laurent"):
  for sym in ("auto", False):
    sx = PMM2DStackHybrid(Px,Py,n_superstrate=nsup,n_substrate=nsub,degree=7,n_orders=5,formulation=form,symmetry=sym)
    sx.add_layer(dep, eps_cell=cx); sx.set_source(wl, theta=0.0, phi=0.0)
    ox_,Rx,Tx,Jx = sx.solve()
    sy = PMM2DStackHybrid(Px,Py,n_superstrate=nsup,n_substrate=nsub,degree=7,n_orders=5,formulation=form,symmetry=sym)
    sy.add_layer(dep, eps_cell=cy); sy.set_source(wl, theta=0.0, phi=0.0)
    oy_,Ry,Ty,Jy = sy.solve()
    idx = {tuple(v):i for i,v in enumerate(map(tuple,oy_))}
    perm=[idx[(n,m)] for m,n in map(tuple,ox_)]
    # row0 = E_x incidence on x-patterned <-> row1 = E_y on y-patterned
    dT = np.max(np.abs(Tx[0]-Ty[1][perm]))
    print(f"  form={form:8s} sym={str(sym):5s}: max|T(Ex,x-pat) - T(Ey,y-pat)| = {dT:.3e}"
          f"   Jxx(x-pat)={Jx[0,0]:.8f} Jyy(y-pat)={Jy[1,1]:.8f} |d|={abs(Jx[0,0]-Jy[1,1]):.3e}")
print()
print("== same test on the SINGLE-LAYER entry pmm_efficiency_2d_cell (has the per-slot fix) ==")
for form in ("li","laurent"):
    o1,R1,T1 = pmm_efficiency_2d_cell(Px,Py,cx,nsub,nsup,dep,wl,polarization="tm",degree=7,n_orders=5,formulation=form)
    o2,R2,T2 = pmm_efficiency_2d_cell(Px,Py,cy,nsub,nsup,dep,wl,polarization="te",degree=7,n_orders=5,formulation=form)
    idx={tuple(v):i for i,v in enumerate(map(tuple,o2))}; perm=[idx[(n,m)] for m,n in map(tuple,o1)]
    print(f"  form={form:8s}: max|T_tm(x-pat) - T_te(y-pat)| = {np.max(np.abs(T1-T2[perm])):.3e}")
print()
print("== reference: 1-D RCWA / PMM oracle for T00 (TM on x-patterned) ==")
from lumenairy.elements.rcwa import rcwa_efficiency_1d
from lumenairy.elements.pmm import pmm_efficiency_1d
o,R1d,T1d = rcwa_efficiency_1d(Px, np.sqrt(12.25), 1.0, nsub, nsup, dep, 0.5, wl, polarization="tm", n_orders=40)[:3]
print("  rcwa_1d TM T00 =", T1d[np.where(o==0)[0][0]])
o,R1d,T1d = rcwa_efficiency_1d(Px, np.sqrt(12.25), 1.0, nsub, nsup, dep, 0.5, wl, polarization="te", n_orders=40)[:3]
print("  rcwa_1d TE T00 =", T1d[np.where(o==0)[0][0]])

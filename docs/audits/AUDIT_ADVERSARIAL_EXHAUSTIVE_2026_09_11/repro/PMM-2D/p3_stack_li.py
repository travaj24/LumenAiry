import sys, numpy as np, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"); sys.path.insert(0,".")
warnings.simplefilter("ignore")
np.set_printoptions(precision=10, linewidth=240)
from lumenairy.elements.pmm import PMM2DStackHybrid, pmm_efficiency_2d_cell
wl, Px, Py, dep = 1.0e-6, 0.47e-6, 0.47e-6, 0.3e-6
nsup, nsub = 1.0, 1.5
S=24
cx = np.full((S,S), 1.0+0j); cx[6:18,:] = 12.25      # x-patterned (walls normal to x)
cy = cx.T.copy()                                      # y-patterned

print("== single-layer entry pmm_efficiency_2d_cell (has the P3-33 per-slot fix) ==")
for form in ("li","laurent"):
    o1,R1,T1 = pmm_efficiency_2d_cell(Px,Py,cx,nsub,nsup,dep,wl,polarization="te",degree=7,n_orders=5,formulation=form)
    # rotate: y-patterned cell, te<->tm swap under 90 deg rotation
    o2,R2,T2 = pmm_efficiency_2d_cell(Px,Py,cy,nsub,nsup,dep,wl,polarization="tm",degree=7,n_orders=5,formulation=form)
    # match order (m,n) -> (n,m)
    idx = {tuple(v):i for i,v in enumerate(map(tuple,o2))}
    perm = [idx[(n,m)] for m,n in map(tuple,o1)]
    print(f"  form={form}: max|T_x(te) - T_y(tm)| = {np.max(np.abs(T1-T2[perm])):.3e}  T00x={T1[np.all(o1==0,1)][0]:.10f} T00y={T2[np.all(o2==0,1)][0]:.10f}")

print("== PMM2DStackHybrid (single layer, same grating) ==")
for form in ("li","laurent"):
    sx = PMM2DStackHybrid(Px,Py,n_superstrate=nsup,n_substrate=nsub,degree=7,n_orders=5,formulation=form)
    sx.add_layer(dep, eps_cell=cx)
    rx_ = sx.solve(wl, polarization="te")
    sy = PMM2DStackHybrid(Px,Py,n_superstrate=nsup,n_substrate=nsub,degree=7,n_orders=5,formulation=form)
    sy.add_layer(dep, eps_cell=cy)
    ry_ = sy.solve(wl, polarization="tm")
    o1,R1,T1 = rx_[0], rx_[1], rx_[2]
    o2,R2,T2 = ry_[0], ry_[1], ry_[2]
    idx = {tuple(v):i for i,v in enumerate(map(tuple,o2))}
    perm = [idx[(n,m)] for m,n in map(tuple,o1)]
    print(f"  form={form}: max|T_x(te) - T_y(tm)| = {np.max(np.abs(T1-T2[perm])):.3e}  T00x={T1[np.all(o1==0,1)][0]:.10f} T00y={T2[np.all(o2==0,1)][0]:.10f}")
    print(f"      energy x: {R1.sum()+T1.sum():.8f}   energy y: {R2.sum()+T2.sum():.8f}")

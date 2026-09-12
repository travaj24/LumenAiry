from common import *
from lumenairy.elements.pmm import PMM2DStackHybrid, pmm_efficiency_2d_cell
from lumenairy.elements.rcwa import rcwa_efficiency_1d
wl, Px, dep = 1.0e-6, 0.47e-6, 0.3e-6
nsup, nsub = 1.0, 1.5
S=24; cx=np.full((S,S),1.0+0j); cx[6:18,:]=12.25; cy=cx.T.copy()
o,R1,T1 = rcwa_efficiency_1d(Px, np.sqrt(12.25), 1.0, nsub, nsup, dep, 0.5, wl, polarization="tm", n_orders=60)[:3]
ref = T1[np.where(o==0)[0][0]]
print(f"oracle T00_TM = {ref:.10f}")
print(" n |  cell-entry x(TM)  err   |  cell-entry y(TM)  err   |  stack x(TM)     err   |  stack y(TM)     err")
for nn in (5,9):
    row=[f"{nn:2d} "]
    # single-layer entry: x-patterned with TM (E along x) ; y-patterned with TE (E along y = TM rel. grating)
    a = pmm_efficiency_2d_cell(Px,Px,cx,nsub,nsup,dep,wl,polarization="tm",degree=11,n_orders=nn,formulation="li")
    i0=int(np.where((a[0][:,0]==0)&(a[0][:,1]==0))[0][0]); ta=a[2][i0]
    b = pmm_efficiency_2d_cell(Px,Px,cy,nsub,nsup,dep,wl,polarization="te",degree=11,n_orders=nn,formulation="li")
    i0=int(np.where((b[0][:,0]==0)&(b[0][:,1]==0))[0][0]); tb=b[2][i0]
    row.append(f"| {ta:.8f} {ta-ref:+.2e} | {tb:.8f} {tb-ref:+.2e} ")
    outs=[]
    for cell,col in ((cx,0),(cy,1)):
        st=PMM2DStackHybrid(Px,Px,n_superstrate=nsup,n_substrate=nsub,degree=11,n_orders=nn,formulation="li")
        st.add_layer(dep, eps_cell=cell); st.set_source(wl, theta=0.0, phi=0.0)
        o_,R_,T_,J_=st.solve(); i0=int(np.where((o_[:,0]==0)&(o_[:,1]==0))[0][0]); outs.append(T_[col][i0])
    row.append(f"| {outs[0]:.8f} {outs[0]-ref:+.2e} | {outs[1]:.8f} {outs[1]-ref:+.2e}")
    print("".join(row), flush=True)

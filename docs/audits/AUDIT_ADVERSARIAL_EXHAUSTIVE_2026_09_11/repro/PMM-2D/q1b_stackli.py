from common import *
from lumenairy.elements.pmm import PMM2DStackHybrid, pmm_efficiency_2d_cell
from lumenairy.elements.rcwa import rcwa_efficiency_1d
wl, Px, dep = 1.0e-6, 0.47e-6, 0.3e-6
nsup, nsub = 1.0, 1.5
S=24; cx=np.full((S,S),1.0+0j); cx[6:18,:]=12.25; cy=cx.T.copy()
o,R1,T1 = rcwa_efficiency_1d(Px, np.sqrt(12.25), 1.0, nsub, nsup, dep, 0.5, wl, polarization="tm", n_orders=60)[:3]
ref_tm = T1[np.where(o==0)[0][0]]
o,R1,T1 = rcwa_efficiency_1d(Px, np.sqrt(12.25), 1.0, nsub, nsup, dep, 0.5, wl, polarization="te", n_orders=60)[:3]
ref_te = T1[np.where(o==0)[0][0]]
print(f"1-D RCWA oracle (n_orders=60): T00_TM={ref_tm:.10f}  T00_TE={ref_te:.10f}")
print("PMM2DStackHybrid, single 1-D-grating layer. TM = E along the grating vector.")
print(" n_or | x-pat E_x (TM)          | y-pat E_y (TM)          | err_x      err_y")
for nn in (3,5,7,9):
  for form in ("li","laurent"):
    res=[]
    for cell in (cx,cy):
        st = PMM2DStackHybrid(Px,Px,n_superstrate=nsup,n_substrate=nsub,degree=11,n_orders=nn,formulation=form)
        st.add_layer(dep, eps_cell=cell); st.set_source(wl, theta=0.0, phi=0.0)
        o_,R_,T_,J_ = st.solve()
        i0 = int(np.where((o_[:,0]==0)&(o_[:,1]==0))[0][0])
        res.append((T_[0][i0], T_[1][i0]))
    tx = res[0][0]   # x-patterned, E_x incidence = TM
    ty = res[1][1]   # y-patterned, E_y incidence = TM
    print(f" {nn:4d} {form:8s}| {tx:.10f} | {ty:.10f} | {tx-ref_tm:+.3e} {ty-ref_tm:+.3e}", flush=True)

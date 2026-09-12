from common import *
from lumenairy.elements.pmm import (PMM2DStackHybrid, PMM2DStackPure,
                                    pmm_jones_2d, pmm_jones_2d_staggered,
                                    pmm_efficiency_2d_staggered)
wl, Px, Py, dep = 1.0e-6, 0.9e-6, 0.9e-6, 0.3e-6
nsup, nsub = 1.0, 1.45
S = 12
cell = np.full((S,S), 1.0+0j); cell[3:9,3:9] = 12.25

print("== A) period_x mutated after add_layer: geometry semantics or stale cache? ==")
def mk(P, c):
    st = PMM2DStackHybrid(P, Py, n_superstrate=nsup, n_substrate=nsub, degree=7, n_orders=3)
    st.add_layer(dep, eps_cell=c); st.set_source(wl, theta=0.0, phi=0.0); return st
a = mk(Px, cell); _=a.solve(); a.period_x = Px*1.05; reused = a.solve()
fresh_same_duty = mk(Px*1.05, cell).solve()
# a cell whose walls sit at the SAME METRES inside the longer period:
S2 = 210   # 0.25*Px = 0.225um ; 0.75*Px = 0.675um ; new period 0.945um
c2 = np.full((S2,S2), 1.0+0j)
i0 = int(round(0.25*Px/(Px*1.05)*S2)); i1 = int(round(0.75*Px/(Px*1.05)*S2))
c2[i0:i1,:] = 12.25
fresh_same_metres = mk(Px*1.05, c2).solve()
print(f"  |reused - fresh(same duty)|   = {np.max(np.abs(np.asarray(reused[3])-np.asarray(fresh_same_duty[3]))):.3e}")
print(f"  |reused - fresh(same METRES)| = {np.max(np.abs(np.asarray(reused[3])-np.asarray(fresh_same_metres[3]))):.3e}")
print("  (small 2nd number => the stored walls are absolute metres; mutating period_x")
print("   silently changes the duty cycle, it is NOT a stale cache hit)", flush=True)

print()
print("== B) twod_staggered.py is LIVE: which public entries reach it? ==")
import sys, lumenairy.elements.pmm.twod_staggered as TS
hit = {"twod_staggered": 0, "twod": 0, "twod_jones": 0, "stack2d": 0, "stack2d_pure": 0}
def tr(frame, ev, arg):
    if ev == "call":
        fn = frame.f_code.co_filename
        for k in hit:
            if fn.endswith("pmm\\" + k + ".py") or fn.endswith("pmm/" + k + ".py"):
                hit[k] += 1
    return None
def probe(label, fn):
    for k in hit: hit[k] = 0
    sys.settrace(tr)
    try: fn()
    finally: sys.settrace(None)
    print(f"  {label:34s}: {dict(hit)}", flush=True)
c3 = np.zeros((S,S,3,3), dtype=complex); c3[...] = np.eye(3); c3[3:9,3:9] = 12.25*np.eye(3)
probe("pmm_jones_2d", lambda: pmm_jones_2d(Px,Py,c3,nsub,nsup,dep,wl,degree=5,n_orders=2))
probe("pmm_efficiency_2d_staggered", lambda: pmm_efficiency_2d_staggered(Px,Py,cell,nsub,nsup,dep,wl,degree=5,n_orders=2))
try:
    probe("pmm_jones_2d_staggered", lambda: pmm_jones_2d_staggered(Px,Py,c3,nsub,nsup,dep,wl,degree=5,n_orders=2))
except Exception as e: print("  pmm_jones_2d_staggered:", type(e).__name__, str(e)[:160])
def _pure():
    sp = PMM2DStackPure(Px,Py,n_superstrate=nsup,n_substrate=nsub,n_modes=5,n_orders=2)
    sp.add_layer(dep, eps_cell=cell); sp.set_source(wl, theta=0.0, phi=0.0); sp.solve()
probe("PMM2DStackPure.solve", _pure)

print()
print("== C) no-floor claim: staggered energy vs n_orders (must be n_orders-INVARIANT) ==")
for nn in (2,3,4):
    o,R,T = pmm_efficiency_2d_staggered(Px,Py,cell,nsub,nsup,dep,wl,degree=6,n_orders=nn)[:3]
    print(f"  n_orders={nn}: sum(R+T)={R.sum()+T.sum():.14f}", flush=True)
for M in (5,6,8):
    o,R,T = pmm_efficiency_2d_staggered(Px,Py,cell,nsub,nsup,dep,wl,degree=M,n_orders=3)[:3]
    print(f"  degree(M)={M}: sum(R+T)={R.sum()+T.sum():.14f}  T00={T[np.all(o==0,1)][0]:.10f}", flush=True)
print("  hybrid on the same grating for contrast:")
from lumenairy.elements.pmm import pmm_efficiency_2d_cell
for nn in (5,9,11):
    o,R,T = pmm_efficiency_2d_cell(Px,Py,cell,nsub,nsup,dep,wl,degree=11,n_orders=nn)[:3]
    print(f"  hybrid n_orders={nn}: sum(R+T)={R.sum()+T.sum():.10f}  T00={T[np.all(o==0,1)][0]:.10f}", flush=True)

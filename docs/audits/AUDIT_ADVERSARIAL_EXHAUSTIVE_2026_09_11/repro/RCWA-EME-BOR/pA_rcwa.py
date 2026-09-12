import sys, os, numpy as np, warnings, time
sys.path.insert(0, r'docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/RCWA-EME-BOR')
t0=time.perf_counter()
from tmm import tmm_jones
from oracle1d import oracle_1d
from lumenairy.elements.rcwa import (rcwa_jones_1d, RCWAStack, rcwa_efficiency_1d,
                                     rcwa_efficiency_2d, rcwa_jones_2d)
print("import %.1f s"%(time.perf_counter()-t0), flush=True)
def fmt(z): return f"{abs(z):.10f}<{np.angle(z):+.9f}"
E3=np.eye(3)

print("\n===== A) rcwa_jones_1d UNPATTERNED film vs TMM (amplitude+phase) =====", flush=True)
wl=0.633e-6; P=0.4e-6
for nf,nsub,d in ((2.0,1.5,0.25e-6),(2.0+0.05j,1.5,0.25e-6)):
  for ang in (0.0,45.0):
    t=time.perf_counter()
    o,Rq,Tq,Jr,Jt = rcwa_jones_1d(P,(nf**2)*E3,(nf**2)*E3,nsub,1.0,d,0.5,wl,
                      angle=np.deg2rad(ang),n_orders=3,return_jones_transmission=True)
    tm=tmm_jones([1.0,nf,nsub],[d],wl,np.deg2rad(ang))
    print(f" nf={nf} ang={ang} ({time.perf_counter()-t:.2f}s)", flush=True)
    print(f"   Jr   xx={fmt(Jr[0,0])} yy={fmt(Jr[1,1])} xy={abs(Jr[0,1]):.2e} yx={abs(Jr[1,0]):.2e}")
    print(f"   TMM  rxx={fmt(tm['rxx'])} ryy={fmt(tm['ryy'])}")
    print(f"   Jt   xx={fmt(Jt[0,0])} yy={fmt(Jt[1,1])}")
    print(f"   TMM  txx={fmt(tm['txx'])} tyy={fmt(tm['tyy'])}")
    print(f"   d(Jrxx,rxx)={abs(Jr[0,0]-tm['rxx']):.3e}  d(Jrxx,-rxx)={abs(Jr[0,0]+tm['rxx']):.3e}"
          f"  d(Jryy,ryy)={abs(Jr[1,1]-tm['ryy']):.3e}  d(Jryy,-ryy)={abs(Jr[1,1]+tm['ryy']):.3e}")
    print(f"   d(Jtxx,txx)={abs(Jt[0,0]-tm['txx']):.3e}  d(Jtxx,-txx)={abs(Jt[0,0]+tm['txx']):.3e}"
          f"  d(Jtyy,tyy)={abs(Jt[1,1]-tm['tyy']):.3e}  d(Jtyy,-tyy)={abs(Jt[1,1]+tm['tyy']):.3e}", flush=True)

print("\n===== B) RCWAStack 3-layer UNPATTERNED vs TMM =====", flush=True)
try:
    st=RCWAStack(0.4e-6, n_superstrate=1.0, n_substrate=1.5, n_orders=3)
    st.add_layer(0.12e-6, eps=2.1**2)
    st.add_layer(0.20e-6, eps=1.46**2)
    st.add_layer(0.08e-6, eps=2.35**2)
    for ang in (0.0, 40.0):
        st.set_source(wl, theta=np.deg2rad(ang))
        res=st.solve()
        Jr=res.jones_reflection(); Jt=res.jones_transmission()
        tm=tmm_jones([1.0,2.1,1.46,2.35,1.5],[0.12e-6,0.20e-6,0.08e-6],wl,np.deg2rad(ang))
        print(f" ang={ang}")
        print(f"   Jr xx={fmt(Jr[0,0])} yy={fmt(Jr[1,1])} | TMM rxx={fmt(tm['rxx'])} ryy={fmt(tm['ryy'])}")
        print(f"   Jt xx={fmt(Jt[0,0])} yy={fmt(Jt[1,1])} | TMM txx={fmt(tm['txx'])} tyy={fmt(tm['tyy'])}")
        print(f"   |dr_xx|={min(abs(Jr[0,0]-tm['rxx']),abs(Jr[0,0]+tm['rxx'])):.3e} "
              f"|dr_yy|={min(abs(Jr[1,1]-tm['ryy']),abs(Jr[1,1]+tm['ryy'])):.3e} "
              f"|dt_xx|={min(abs(Jt[0,0]-tm['txx']),abs(Jt[0,0]+tm['txx'])):.3e} "
              f"|dt_yy|={min(abs(Jt[1,1]-tm['tyy']),abs(Jt[1,1]+tm['tyy'])):.3e}", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()

print("\n===== C) thick absorbing layer stability (100 um) =====", flush=True)
for dep in (1e-6, 1e-5, 1e-4):
    try:
        o,R,T=rcwa_efficiency_1d(1.0e-6,1.5+0.02j,1.0,1.5,1.0,dep,0.5,0.633e-6,
                                 polarization='tm',n_orders=15)
        print(f"   depth={dep*1e6:8.1f} um  sumR={R.sum():.9f} sumT={T.sum():.3e} "
              f"finite={np.all(np.isfinite(R))and np.all(np.isfinite(T))}", flush=True)
    except Exception as e:
        print("   depth",dep,"RAISED",type(e).__name__,str(e)[:120], flush=True)

print("\n===== D) reciprocity (1-D lossless): DE_t(m, sup->sub) vs reversed =====", flush=True)
# Reciprocity for a grating: t_{0->m} (from sup) and t_{-m->0} (from sub, at the
# conjugate angle).  Check via the symmetric relation n_I kz_I |t|^2 relation.
args=dict(period=1.3e-6,n_ridge=2.1,n_groove=1.0,depth=0.45e-6,duty_cycle=0.4,wavelength=0.633e-6)
nI,nII=1.0,1.62
for pol in ('te','tm'):
  th=np.deg2rad(18.0)
  o,R,T=rcwa_efficiency_1d(n_substrate=nII,n_superstrate=nI,angle=th,polarization=pol,n_orders=25,**args)
  # order m transmitted into substrate at angle th_m: n_II sin(th_m) = n_I sin(th) + m*wl/P
  wl0=args['wavelength']; Pp=args['period']
  for m in (-1,0,1):
    i=25+m
    s_m=(nI*np.sin(th)+m*wl0/Pp)/nII
    if abs(s_m)>=1: continue
    th_m=np.arcsin(s_m)
    # reverse: incidence from substrate at -th_m, look at order -m
    o2,R2,T2=rcwa_efficiency_1d(n_substrate=nI,n_superstrate=nII,angle=-th_m,polarization=pol,n_orders=25,**args)
    j=25-m
    print(f"   pol={pol} m={m:+d}: T_fwd={T[i]:.12f}  T_rev={T2[j]:.12f}  rel={abs(T[i]-T2[j])/max(T[i],1e-30):.3e}", flush=True)

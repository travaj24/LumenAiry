import sys, numpy as np
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/RCWA-EME-BOR")
from oracle1d import oracle_1d
from lumenairy.elements.rcwa import rcwa_efficiency_1d
P=1.0e-6; nr=2.04; ng=1.0; ns=1.0; nsup=1.0; d=1.0e-6; duty=0.5; M=21
def lib(wl,pol):
    o,R,T=rcwa_efficiency_1d(P,nr,ng,ns,nsup,d,duty,wl,polarization=pol,n_orders=M,formulation="li")
    return R[M],T[M],R.sum()+T.sum()
def orc(wl,pol):
    m,R,T,_,_=oracle_1d(P,nr,ng,ns,nsup,d,duty,wl,pol=pol,M=M)
    return R[M],T[M],R.sum()+T.sum()
print("=== exact Wood point lam=Lam=1um, normal incidence ===")
for pol in ("te","tm"):
    print(f"  pol={pol}")
    for lam,tag in [(1.0e-6,"lam=Lam EXACT"),(1.0e-6*(1+1e-7),"lam*(1+1e-7)"),
                    (1.0e-6*(1+1e-6),"lam*(1+1e-6)"),(1.0e-6*(1-1e-6),"lam*(1-1e-6)"),
                    (1.0e-6*(1+1e-4),"lam*(1+1e-4)"),(1.0e-6*(1-1e-4),"lam*(1-1e-4)")]:
        a=lib(lam,pol); b=orc(lam,pol)
        print(f"    {tag:14s} libR0={a[0]:.12f} orcR0={b[0]:.12f} d={a[0]-b[0]:+.3e} | libclos={a[2]-1:+.2e} orcclos={b[2]-1:+.2e}")
print("=== continuity of the LIBRARY answer in lambda across the anomaly ===")
for pol in ("te","tm"):
    print(f"  pol={pol}")
    for e in (-1e-5,-3e-6,-1e-6,-3e-7,-1e-7,-3e-8,0.0,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5):
        lam=1.0e-6*(1+e)
        a=lib(lam,pol); b=orc(lam,pol)
        print(f"    d_rel={e:+.1e} libR0={a[0]:.12f}  orcR0={b[0]:.12f}   diff={a[0]-b[0]:+.3e}")

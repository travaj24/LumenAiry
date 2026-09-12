"""TR-INFRA probe 12: _Cheb2DEvaluator fit + ev performance."""
import time, tracemalloc
import numpy as np
from lumenairy.elements import _lens_traced as T

def t(fn, rep=3):
    fn()  # warm
    best=1e9
    for _ in range(rep):
        t0=time.perf_counter(); fn(); best=min(best,time.perf_counter()-t0)
    return best

n=257; xs=np.linspace(-1,1,n); Xi,Yi=np.meshgrid(xs,xs,indexing='ij')
V=np.exp(-(Xi**2+Yi**2))+0.3*Xi**3*Yi
print("=== FIT cost: deterministic vs BLAS normal equations (order 6 / 10) ===")
for order in (6,10):
    T.DETERMINISTIC_TRACED_FIT=True
    td=t(lambda: T._Cheb2DEvaluator(xs,xs,V,order=order))
    T.DETERMINISTIC_TRACED_FIT=False
    tb=t(lambda: T._Cheb2DEvaluator(xs,xs,V,order=order))
    T.DETERMINISTIC_TRACED_FIT=True
    print(f"  order {order:2d} ({n*n} samples): det {td*1e3:8.2f} ms   BLAS {tb*1e3:8.2f} ms   {td/tb:5.2f}x")

print("\n=== EVAL cost at 4096^2 = 16.8 M points (numba vs numpy) ===")
ev=T._Cheb2DEvaluator(xs,xs,V,order=6)
st=T._cheb_fit_state(ev)
M=4096
q=np.linspace(-0.99,0.99,M); QX,QY=np.meshgrid(q,q,indexing='ij')
qx=QX.ravel(); qy=QY.ravel()
for backend in ('numba','numpy'):
    e=T._Cheb2DEvaluator.from_state(st,backend=backend)
    tt=t(lambda: e.ev_value_and_grad(qx,qy), rep=2)
    print(f"  backend={backend:6s} ev_value_and_grad: {tt:7.3f} s  "
          f"({M*M/tt/1e6:7.2f} Mpt/s)")
# how much does the numba kernel's per-sample np.empty cost?
print("\n=== numba kernel: cost vs order (per-sample np.empty(order+1) x4) ===")
for order in (4,6,8,10,12):
    ev2=T._Cheb2DEvaluator(xs,xs,V,order=order)
    e=T._Cheb2DEvaluator.from_state(T._cheb_fit_state(ev2),backend='numba')
    sub=qx[:4_000_000]; subx=qy[:4_000_000]
    tt=t(lambda: e.ev_value_and_grad(sub,subx), rep=2)
    M_terms=(order+1)*(order+2)//2
    print(f"  order {order:2d} (M={M_terms:3d} terms): {tt*1e3:8.2f} ms for 4 Mpt "
          f"-> {tt*1e9/4e6:7.1f} ns/pt, {tt*1e9/4e6/M_terms:6.2f} ns/pt/term")

print("\n=== ev(dx=1) computes ALL THREE quantities (3x waste when one is wanted) ===")
e=T._Cheb2DEvaluator.from_state(st,backend='numba')
sub=qx[:4_000_000]; suby=qy[:4_000_000]
t_all=t(lambda: e.ev_value_and_grad(sub,suby), rep=2)
t_one=t(lambda: e.ev(sub,suby), rep=2)
print(f"  ev_value_and_grad  {t_all*1e3:7.2f} ms;  ev() [value only] {t_one*1e3:7.2f} ms"
      f"  -> value-only pays {t_one/t_all:.2f}x of the full call")

print("\n=== _decentred_fit_score / _decentred_fit_spectrum rebuild the Vandermonde ===")
opl=(Xi**2+Yi**2)*1e-3
disc=(Xi**2+Yi**2)<=0.25
wt=T._decentred_fit_score_weight(xs,0.0,0.0,0.4)
ts=t(lambda: T._decentred_fit_score(xs,opl,wt,disc,None,6),rep=3)
print(f"  _decentred_fit_score (one candidate, n={n}^2): {ts*1e3:7.2f} ms"
      f"  -- the C11 arbiter runs TWO of these per call on top of the real fit")

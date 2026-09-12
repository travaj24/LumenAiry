import time, tracemalloc
import numpy as np
from lumenairy.elements import _lens_traced as T
def t(fn, rep=2):
    fn(); best=1e9
    for _ in range(rep):
        t0=time.perf_counter(); fn(); best=min(best,time.perf_counter()-t0)
    return best
n=257; xs=np.linspace(-1,1,n); Xi,Yi=np.meshgrid(xs,xs,indexing='ij')
V=np.exp(-(Xi**2+Yi**2))+0.3*Xi**3*Yi
print("=== FIT: deterministic vs BLAS normal equations ===", flush=True)
for order in (6,10):
    T.DETERMINISTIC_TRACED_FIT=True;  td=t(lambda: T._Cheb2DEvaluator(xs,xs,V,order=order))
    T.DETERMINISTIC_TRACED_FIT=False; tb=t(lambda: T._Cheb2DEvaluator(xs,xs,V,order=order))
    T.DETERMINISTIC_TRACED_FIT=True
    print(f"  order {order:2d} ({n*n} samples): det {td*1e3:8.2f} ms  BLAS {tb*1e3:8.2f} ms  {td/tb:5.2f}x", flush=True)

ev=T._Cheb2DEvaluator(xs,xs,V,order=6); st=T._cheb_fit_state(ev)
print("\n=== EVAL throughput + PEAK MEMORY, numba vs pure-numpy fallback ===", flush=True)
for npts in (100_000, 1_000_000):
    q=np.linspace(-0.99,0.99,npts); qy=q[::-1].copy()
    for backend in ('numba','numpy'):
        e=T._Cheb2DEvaluator.from_state(st,backend=backend)
        e.ev_value_and_grad(q[:1000],qy[:1000])
        tracemalloc.start()
        t0=time.perf_counter(); e.ev_value_and_grad(q,qy); dt=time.perf_counter()-t0
        cur,pk=tracemalloc.get_traced_memory(); tracemalloc.stop()
        print(f"  n={npts:9d} backend={backend:6s}: {dt*1e3:8.2f} ms "
              f"({npts/dt/1e6:7.2f} Mpt/s)  peak alloc {pk/1e6:9.2f} MB "
              f"= {pk/(8*npts):6.1f} float64/point", flush=True)
print("\n=== numba kernel cost vs order (per-sample np.empty x4 inside prange) ===", flush=True)
q=np.linspace(-0.99,0.99,2_000_000); qy=q[::-1].copy()
for order in (4,6,8,12):
    ev2=T._Cheb2DEvaluator(xs,xs,V,order=order)
    e=T._Cheb2DEvaluator.from_state(T._cheb_fit_state(ev2),backend='numba')
    tt=t(lambda: e.ev_value_and_grad(q,qy))
    M=(order+1)*(order+2)//2
    print(f"  order {order:2d} (M={M:3d}): {tt*1e3:8.2f} ms / 2 Mpt = {tt*1e9/2e6:7.1f} ns/pt"
          f"  ({tt*1e9/2e6/M:6.2f} ns/pt/term)", flush=True)
print("\n=== value-only ev() pays the full 3-quantity cost ===", flush=True)
e=T._Cheb2DEvaluator.from_state(st,backend='numba')
print(f"  ev_value_and_grad {t(lambda: e.ev_value_and_grad(q,qy))*1e3:7.2f} ms"
      f"   ev() [value only] {t(lambda: e.ev(q,qy))*1e3:7.2f} ms", flush=True)
print("\n=== _decentred_fit_score cost (arbiter runs 2 of these) ===", flush=True)
opl=(Xi**2+Yi**2)*1e-3; disc=(Xi**2+Yi**2)<=0.25
wt=T._decentred_fit_score_weight(xs,0.0,0.0,0.4)
print(f"  one candidate at n={n}^2: {t(lambda: T._decentred_fit_score(xs,opl,wt,disc,None,6))*1e3:7.2f} ms", flush=True)

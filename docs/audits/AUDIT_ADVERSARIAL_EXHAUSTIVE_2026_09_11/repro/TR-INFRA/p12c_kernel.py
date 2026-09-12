"""TR-INFRA 12c: is the shipped numba Chebyshev kernel allocation-bound?
Compare it against a BLOCKED variant that allocates the T/U scratch once per
block of 512 samples instead of once per sample."""
import time
import numpy as np
import numba
from numba import njit, prange
from lumenairy.elements import _lens_traced as T

@njit(cache=True, parallel=True, fastmath=True)
def blocked(coeffs, K1, K2, u_flat, v_flat, max_order):
    N = u_flat.shape[0]; M = coeffs.shape[0]
    f = np.zeros(N); fx = np.zeros(N); fy = np.zeros(N)
    B = 512
    nb = (N + B - 1) // B
    for b in prange(nb):
        i0 = b*B; i1 = min(i0+B, N)
        Tu = np.empty(max_order+1); Tv = np.empty(max_order+1)
        dTu = np.zeros(max_order+1); dTv = np.zeros(max_order+1)
        for i in range(i0, i1):
            u = u_flat[i]; v = v_flat[i]
            Tu[0] = 1.0; Tv[0] = 1.0
            if max_order >= 1:
                Tu[1] = u; Tv[1] = v
            for n in range(2, max_order+1):
                Tu[n] = 2.0*u*Tu[n-1]-Tu[n-2]
                Tv[n] = 2.0*v*Tv[n-1]-Tv[n-2]
            if max_order >= 1:
                dTu[1] = 1.0; dTv[1] = 1.0
                if max_order >= 2:
                    Upu = 1.0; Uu = 2.0*u; Upv = 1.0; Uv = 2.0*v
                    dTu[2] = 2.0*Uu; dTv[2] = 2.0*Uv
                    for n in range(3, max_order+1):
                        Unu = 2.0*u*Uu-Upu; Unv = 2.0*v*Uv-Upv
                        Upu = Uu; Uu = Unu; Upv = Uv; Uv = Unv
                        dTu[n] = n*Uu; dTv[n] = n*Uv
            af = 0.0; ax = 0.0; ay = 0.0
            for m in range(M):
                kx = K1[m]; ky = K2[m]; c = coeffs[m]
                tu = Tu[kx]; tv = Tv[ky]
                af += c*tu*tv; ax += c*dTu[kx]*tv; ay += c*tu*dTv[ky]
            f[i] = af; fx[i] = ax; fy[i] = ay
    return f, fx, fy

n=257; xs=np.linspace(-1,1,n); Xi,Yi=np.meshgrid(xs,xs,indexing='ij')
V=np.exp(-(Xi**2+Yi**2))+0.3*Xi**3*Yi
kern = T._get_cheb2d_val_grad_numba()
q=np.linspace(-0.99,0.99,2_000_000); qy=q[::-1].copy()
print(f"  numba threads = {numba.get_num_threads()}")
print("  order   shipped(per-sample alloc)   blocked(per-512 alloc)   speedup   max|df|")
for order in (4,6,8,10,12):
    ev=T._Cheb2DEvaluator(xs,xs,V,order=order)
    c=np.ascontiguousarray(ev.coeffs); K1=np.ascontiguousarray(ev._K1); K2=np.ascontiguousarray(ev._K2)
    kern(c,K1,K2,q[:1000],qy[:1000],order); blocked(c,K1,K2,q[:1000],qy[:1000],order)
    def t(fn):
        best=1e9
        for _ in range(3):
            t0=time.perf_counter(); r=fn(); best=min(best,time.perf_counter()-t0)
        return best,r
    t1,r1=t(lambda: kern(c,K1,K2,q,qy,order))
    t2,r2=t(lambda: blocked(c,K1,K2,q,qy,order))
    d=max(np.abs(r1[0]-r2[0]).max(), np.abs(r1[1]-r2[1]).max(), np.abs(r1[2]-r2[2]).max())
    print(f"  {order:5d}   {t1*1e3:9.2f} ms ({t1*1e9/2e6:6.1f} ns/pt)   "
          f"{t2*1e3:9.2f} ms ({t2*1e9/2e6:6.1f} ns/pt)   {t1/t2:6.2f}x   {d:.3e}")

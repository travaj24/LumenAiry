"""Clean interleaved kernel benchmark: shipped (per-sample np.empty x4 inside
prange) vs a blocked variant that allocates once per 512 samples."""
import time
import numpy as np, numba
from numba import njit, prange
from lumenairy.elements import _lens_traced as T

@njit(cache=True, parallel=True, fastmath=True)
def blocked(coeffs, K1, K2, u_flat, v_flat, max_order):
    N=u_flat.shape[0]; M=coeffs.shape[0]
    f=np.zeros(N); fx=np.zeros(N); fy=np.zeros(N)
    B=512; nb=(N+B-1)//B
    for b in prange(nb):
        i0=b*B; i1=min(i0+B,N)
        Tu=np.empty(max_order+1); Tv=np.empty(max_order+1)
        dTu=np.zeros(max_order+1); dTv=np.zeros(max_order+1)
        for i in range(i0,i1):
            u=u_flat[i]; v=v_flat[i]
            Tu[0]=1.0; Tv[0]=1.0
            if max_order>=1: Tu[1]=u; Tv[1]=v
            for n in range(2,max_order+1):
                Tu[n]=2.0*u*Tu[n-1]-Tu[n-2]; Tv[n]=2.0*v*Tv[n-1]-Tv[n-2]
            if max_order>=1:
                dTu[1]=1.0; dTv[1]=1.0
                if max_order>=2:
                    Upu=1.0; Uu=2.0*u; Upv=1.0; Uv=2.0*v
                    dTu[2]=2.0*Uu; dTv[2]=2.0*Uv
                    for n in range(3,max_order+1):
                        Unu=2.0*u*Uu-Upu; Unv=2.0*v*Uv-Upv
                        Upu=Uu; Uu=Unu; Upv=Uv; Uv=Unv
                        dTu[n]=n*Uu; dTv[n]=n*Uv
            af=0.0; ax=0.0; ay=0.0
            for m in range(M):
                kx=K1[m]; ky=K2[m]; c=coeffs[m]
                tu=Tu[kx]; tv=Tv[ky]
                af+=c*tu*tv; ax+=c*dTu[kx]*tv; ay+=c*tu*dTv[ky]
            f[i]=af; fx[i]=ax; fy[i]=ay
    return f,fx,fy

n=129; xs=np.linspace(-1,1,n); Xi,Yi=np.meshgrid(xs,xs,indexing='ij')
V=np.exp(-(Xi**2+Yi**2))+0.3*Xi**3*Yi
kern=T._get_cheb2d_val_grad_numba()
q=np.linspace(-0.99,0.99,4_000_000); qy=q[::-1].copy()
print(f"numba threads={numba.get_num_threads()}   4 Mpt, 9 interleaved reps, min wall")
for order in (6,10):
    ev=T._Cheb2DEvaluator(xs,xs,V,order=order)
    c=np.ascontiguousarray(ev.coeffs); K1=np.ascontiguousarray(ev._K1); K2=np.ascontiguousarray(ev._K2)
    kern(c,K1,K2,q[:1000],qy[:1000],order); blocked(c,K1,K2,q[:1000],qy[:1000],order)
    ta=tb=1e9
    for _ in range(9):
        t0=time.perf_counter(); r1=kern(c,K1,K2,q,qy,order); ta=min(ta,time.perf_counter()-t0)
        t0=time.perf_counter(); r2=blocked(c,K1,K2,q,qy,order); tb=min(tb,time.perf_counter()-t0)
    d=max(np.abs(r1[i]-r2[i]).max() for i in range(3))
    print(f"  order {order:2d} (M={(order+1)*(order+2)//2:3d}): shipped {ta*1e3:7.1f} ms "
          f"({ta*1e9/4e6:6.1f} ns/pt)   blocked {tb*1e3:7.1f} ms ({tb*1e9/4e6:6.1f} ns/pt)"
          f"   {ta/tb:5.2f}x   max|diff|={d:.3e}")

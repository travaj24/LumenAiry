"""TR-INFRA probe 2c: is the 'deterministic' solve actually byte-reproducible
across BLAS thread counts?  Run under different OMP/OPENBLAS thread settings."""
import hashlib, os, sys, warnings
import numpy as np
from lumenairy.elements import _lens_traced as T

def build_A(xs, order):
    xmin,xmax=xs.min(),xs.max()
    X,Y=np.meshgrid(xs,xs,indexing='ij')
    u=(2.0*X-(xmin+xmax))/(xmax-xmin); v=(2.0*Y-(xmin+xmax))/(xmax-xmin)
    mi=[(kx,ky) for kx in range(order+1) for ky in range(order+1-kx)]
    K1=np.array([m[0] for m in mi]); K2=np.array([m[1] for m in mi])
    Tu=T._cheb_vand_2d(u,order,np); Tv=T._cheb_vand_2d(v,order,np)
    return (Tu[K1]*Tv[K2]).reshape(len(mi),-1).T, X, Y

n=257
xs=np.linspace(-1.0,1.0,n); rng=np.random.default_rng(0)
out={}
for order,label in ((6,'M=28 hard-disc (screen FIRES -> refine)'),
                    (10,'M=66 hard-disc (rcond 0 -> QR fallback)'),
                    (6,'M=28 FULL square (screen PASSES)')):
    A,X,Y=build_A(xs,order)
    if 'FULL' in label:
        Am=np.ascontiguousarray(A); sel=slice(None)
    else:
        d=((X**2+Y**2)<=0.25).ravel(); Am=np.ascontiguousarray(A[d,:]); sel=d
    r2=(X**2+Y**2).ravel()[sel]
    b=1e-3*r2-4e-5*r2**2+9e-7*r2**3+1e-12*rng.standard_normal(Am.shape[0])
    for det in (False,True):
        with warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter('always')
            x=T._solve_lstsq_thread_safe(Am,b,deterministic=det)
            nw=len([w for w in wlist if 'deterministic least-squares' in str(w.message)])
        h=hashlib.blake2b(np.ascontiguousarray(x).tobytes(),digest_size=8).hexdigest()
        print(f"THREADS={os.environ.get('OMP_NUM_THREADS','default')} {label:42s} det={det!s:5s} hash={h} stepdown_warns={nw}")

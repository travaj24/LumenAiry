"""TR-INFRA probe 2b: does the in-disc residual criterion bound the
EXTRAPOLATION error the Newton loop actually sees over the launch square?"""
import math, warnings
import numpy as np
import scipy.linalg as sla
from lumenairy.elements import _lens_traced as T

def build_A(xs, order):
    xmin,xmax=xs.min(),xs.max()
    X,Y=np.meshgrid(xs,xs,indexing='ij')
    u=(2.0*X-(xmin+xmax))/(xmax-xmin); v=(2.0*Y-(xmin+xmax))/(xmax-xmin)
    mi=[(kx,ky) for kx in range(order+1) for ky in range(order+1-kx)]
    K1=np.array([m[0] for m in mi]); K2=np.array([m[1] for m in mi])
    Tu=T._cheb_vand_2d(u,order,np); Tv=T._cheb_vand_2d(v,order,np)
    return (Tu[K1]*Tv[K2]).reshape(len(mi),-1).T, mi, X, Y

n=129; R=1.0
xs=np.linspace(-R,R,n); rng=np.random.default_rng(0)
print("=== 2b  field difference between candidate solves, IN-disc vs WHOLE square ===")
for order in (6, 8, 10):
    A, mi, X, Y = build_A(xs, order)
    disc=((X**2+Y**2)<=(0.5*R)**2)
    dflat=disc.ravel(); Am=np.ascontiguousarray(A[dflat,:])
    r2=(X**2+Y**2).ravel()[dflat]
    b=1e-3*r2-4e-5*r2**2+9e-7*r2**3+1e-12*rng.standard_normal(int(dflat.sum()))
    x_qr=T._solve_lstsq_qr(Am,b)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x_det=T._solve_lstsq_thread_safe(Am,b,deterministic=True)
        x_ne_only = None
        old=T.LSTSQ_CONDITIONING_STEPDOWN
        T.LSTSQ_CONDITIONING_STEPDOWN=False
        x_ne_only=T._solve_lstsq_thread_safe(Am,b,deterministic=False)
        T.LSTSQ_CONDITIONING_STEPDOWN=old
    # evaluate on the WHOLE launch square (what Newton iterates over)
    f_qr=(A@x_qr).reshape(X.shape); f_det=(A@x_det).reshape(X.shape)
    f_ne =(A@x_ne_only).reshape(X.shape)
    scale=float(np.abs(f_qr[disc]).max())
    print(f"  order {order}:  peak |f| in disc = {scale:.4e}")
    for nm,f in (('det(refined) vs QR', f_det), ('raw-NE       vs QR', f_ne)):
        print(f"    {nm}: max|df| in disc = {np.abs(f-f_qr)[disc].max():.4e}"
              f"   max|df| whole square = {np.abs(f-f_qr).max():.4e}"
              f"   ({np.abs(f-f_qr).max()/scale:.3e} of peak)")

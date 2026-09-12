"""TR-INFRA probe 2: _solve_lstsq_thread_safe accuracy vs QR + determinism.
Oracle = scipy gelsd (SVD, backward stable) with one step of extra-precision
iterative refinement using math.fsum on the residual rows."""
import math
import numpy as np
import scipy.linalg as sla
from lumenairy.elements import _lens_traced as T

def build_A(xs, order):
    xmin, xmax = xs.min(), xs.max()
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    u = (2.0*X-(xmin+xmax))/(xmax-xmin); v=(2.0*Y-(xmin+xmax))/(xmax-xmin)
    mi = [(kx,ky) for kx in range(order+1) for ky in range(order+1-kx)]
    K1=np.array([m[0] for m in mi]); K2=np.array([m[1] for m in mi])
    Tu=T._cheb_vand_2d(u,order,np); Tv=T._cheb_vand_2d(v,order,np)
    return (Tu[K1]*Tv[K2]).reshape(len(mi),-1).T, mi, X, Y

def fsum_resid(A,b,x):
    """||b - A x|| with each row's dot product accumulated by math.fsum."""
    Ax = np.array([math.fsum((A[i]*x).tolist()) for i in range(A.shape[0])])
    return b - Ax

n=129; R=1.0
xs=np.linspace(-R,R,n)
rng=np.random.default_rng(0)

print("=== 2a  normal-equations vs QR vs gelsd+refinement, disc-masked ===")
for order in (6, 8, 10):
    A, mi, X, Y = build_A(xs, order)
    disc=((X**2+Y**2)<=(0.5*R)**2).ravel()
    Am=np.ascontiguousarray(A[disc,:])
    r2=(X**2+Y**2).ravel()[disc]
    b = 1e-3*r2 - 4e-5*r2**2 + 9e-7*r2**3 + 1e-12*rng.standard_normal(int(disc.sum()))
    # oracle: gelsd + 2 refinement steps with fsum residuals
    xr = sla.lstsq(Am,b,lapack_driver='gelsd')[0]
    for _ in range(3):
        rres = fsum_resid(Am,b,xr)
        xr = xr + sla.lstsq(Am,rres,lapack_driver='gelsd')[0]
    rref = float(np.linalg.norm(fsum_resid(Am,b,xr)))
    G=Am.T@Am; rhs=Am.T@b
    from scipy.linalg import cho_factor, cho_solve
    try:
        cf=cho_factor(G, check_finite=False); x_ne=cho_solve(cf,rhs,check_finite=False)
    except Exception:
        x_ne=np.linalg.solve(G,rhs)
    x_qr=T._solve_lstsq_qr(Am,b)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x_ship=T._solve_lstsq_thread_safe(Am,b,deterministic=False)
        x_det =T._solve_lstsq_thread_safe(Am,b,deterministic=True)
    scale=float(np.max(np.abs(xr)))
    print(f"  order {order} M={len(mi)} n_in={int(disc.sum())} "
          f"cond(A)={np.linalg.cond(Am):.3e} gram_rcond={T._gram_rcond(G):.3e} "
          f"oracle ||r||={rref:.6e}")
    for nm,x in (('raw normal-eq',x_ne),('QR (geqrf)   ',x_qr),
                 ('shipped det=F',x_ship),('shipped det=T',x_det)):
        rr=float(np.linalg.norm(fsum_resid(Am,b,x)))
        ce=float(np.max(np.abs(x-xr)))/max(scale,1e-300)
        print(f"    {nm}: ||b-Ax||={rr:.6e} ({rr/rref:9.4f}x)  max rel coeff err={ce:.3e}")
    print()

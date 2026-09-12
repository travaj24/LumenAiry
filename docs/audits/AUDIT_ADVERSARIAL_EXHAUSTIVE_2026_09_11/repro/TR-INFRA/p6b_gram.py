"""TR-INFRA 6b': does the EQUILIBRATED _gram_rcond screen correctly predict the
accuracy of the UNSCALED Cholesky solve on a raw-monomial (metre) basis?"""
import warnings
import numpy as np, scipy.linalg as sla
from lumenairy.elements import _lens_traced as T

def auto_fit_matrix(N, dx, deg, w0):
    x=(np.arange(N)-N/2)*dx
    Xg,Yg=np.meshgrid(x,x,indexing='xy')
    r2=Xg**2+Yg**2
    m=np.exp(-r2/w0**2)>0.05
    xL=Xg[m]; yL=Yg[m]
    terms=[(i,j) for d in range(1,deg+1) for i in range(d+1) for j in [d-i]]
    n=xL.size; A=np.zeros((2*n,len(terms)))
    for k,(i,j) in enumerate(terms):
        A[:n,k]=(i*xL**(i-1)*yL**j) if i>=1 else 0.0
        A[n:,k]=(j*xL**i*yL**(j-1)) if j>=1 else 0.0
    return A, terms, xL, yL

print(" deg  w0[m]   cond(A)   cond(G)  gram_rcond  ||r_ne||/||r_qr||  max|dc|/|c|")
for w0 in (6e-4, 3e-5):
  for deg in (2,3,4,6):
    N, dx = 256, (8e-6 if w0>1e-4 else 4e-7)
    A,terms,xL,yL = auto_fit_matrix(N,dx,deg,w0)
    rng=np.random.default_rng(1)
    ctrue=rng.standard_normal(len(terms))*np.array([1e0/ (1e-3)**(i+j-1) for (i,j) in terms])
    b=A@ctrue + 1e-9*rng.standard_normal(A.shape[0])
    G=A.T@A; rhs=A.T@b
    try:
        cf=sla.cho_factor(G,check_finite=False); x_ne=sla.cho_solve(cf,rhs,check_finite=False)
        ok='chol'
    except Exception:
        x_ne=np.linalg.solve(G,rhs); ok='lu'
    x_qr=T._solve_lstsq_qr(A,b)
    rne=np.linalg.norm(b-A@x_ne); rqr=np.linalg.norm(b-A@x_qr)
    dc=np.max(np.abs(x_ne-x_qr)/np.maximum(np.abs(x_qr),1e-300))
    print(f"  {deg}  {w0:.1e}  {np.linalg.cond(A):9.3e} {np.linalg.cond(G):9.3e} "
          f"{T._gram_rcond(G):10.3e}   {rne/rqr:12.5f}    {dc:9.3e}  [{ok}]")

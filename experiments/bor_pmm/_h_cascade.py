"""Decisive test: build H-form layer modal basis (W=tangential E, V=tangential H
recovered from h-fields) and run the cascade energy gate on a STRUCTURED ring,
comparing the spurious-real-q leakage floor vs the E-form.

For the cascade we still need tangential (E_t, H_t) at z-interfaces. From the
H-eigenvectors: H_t=(h_r,h_phi) directly; E_t recovered via E=(i/(k0 eps))curl h:
  E_r   = (i/(k0 eps_n))[(i m/r) h_z - i q h_phi]      (normal -> inverse rule)
  E_phi = (i/(k0 eps  ))[ i q h_r - dh_z/dr ]          (tangential -> pointwise)
"""
import numpy as np; import sys; sys.path.insert(0,".")
from coupled_radial_eigensolver import _fd_grid, _normal_eps
from scipy.linalg import eig


def h_layer_modes(m,Rbig,N,eps_profile,k0):
    r,D,h=_fd_grid(Rbig,N)
    eps=np.asarray(eps_profile(r),dtype=complex); eps_n=_normal_eps(eps)
    I=np.eye(N); ir=np.diag(1/r); mr=m*ir; A=D+ir
    pinv=np.diag(1/eps); pinv_n=np.diag(1/eps_n)
    Z=np.zeros((N,N),dtype=complex)
    C0=np.vstack([np.hstack([Z,Z,1j*mr]),np.hstack([Z,Z,-D]),np.hstack([-1j*mr,A,Z])])
    Cq=np.vstack([np.hstack([Z,-1j*I,Z]),np.hstack([1j*I,Z,Z]),np.hstack([Z,Z,Z])])
    P=np.block([[pinv_n,Z,Z],[Z,pinv,Z],[Z,Z,pinv]])
    M0=C0@P@C0; M1=C0@P@Cq+Cq@P@C0; M2=Cq@P@Cq
    n3=3*N; I3=np.eye(n3,dtype=complex); M0c=M0-k0**2*I3
    Abig=np.vstack([np.hstack([np.zeros((n3,n3),dtype=complex),I3]),np.hstack([-M0c,-M1])])
    Bbig=np.vstack([np.hstack([I3,np.zeros((n3,n3),dtype=complex)]),
                    np.hstack([np.zeros((n3,n3),dtype=complex),M2])])
    qall,Vall=eig(Abig,Bbig)
    fin=np.isfinite(qall); qall=qall[fin]; Vall=Vall[:,fin]
    # keep finite, pick 2N with smallest |q| ... but companion gives ~ up to 4N finite.
    # Sort by |q|, dedupe near-duplicates, keep the genuine modes.
    nm=len(qall)
    wq=(r*h).astype(complex)
    # build tangential fields, orient forward by flux
    recs=[]
    for j in range(nm):
        q=qall[j]
        hr=Vall[:N,j];hp=Vall[N:2*N,j];hz=Vall[2*N:3*N,j]
        nn=np.sqrt(np.sum(np.abs(hr)**2+np.abs(hp)**2+np.abs(hz)**2))
        if nn<1e-9 or not np.isfinite(q): continue
        def fields(qq):
            Er=(1j/(k0))*(pinv_n@((1j*mr@hz)-1j*qq*hp))
            Ephi=(1j/(k0))*(pinv@(1j*qq*hr-(D@hz)))
            return Er,Ephi
        # orient forward: Pz=Re sum(Er hphi* - Ephi hr*) r dr
        def flux(qq):
            Er,Ephi=fields(qq)
            return np.real(np.sum((Er*np.conj(hp)-Ephi*np.conj(hr))*wq))
        if abs(q.imag)<1e-9*max(abs(q.real),1e-300):
            qf=q if flux(q)>=0 else -q
        else:
            qf=q if q.imag>0 else -q
        # recompute hz sign consistency under q->qf: hz is from eigenvector tied to q.
        # For q->-q the physical h_z flips; but here vector is fixed -> approximate:
        Er,Ephi=fields(qf)
        recs.append((qf,hr,hp,Er,Ephi,nn))
    # dedupe to 2N: sort by real(q) then take unique
    return recs,r,wq,N

# structured ring (high contrast)
m,Rbig,N,k0=1,3.0,150,2.0
def eps_struct(rr):
    e=np.full_like(rr,2.25,dtype=complex)
    e[rr<0.6]=12.0; e[(rr>=1.0)&(rr<1.3)]=12.0
    return e
def eps_unif(rr): return np.full_like(rr,2.25,dtype=complex)

recsH,r,wq,N=h_layer_modes(m,Rbig,N,eps_struct,k0)
qs=np.array([x[0] for x in recsH])
realq=np.array([abs(q.imag)<1e-3 and q.real>1e-2 for q in qs])
print("H-form structured layer: finite modes",len(recsH),
      " real-q (propagating):",int(realq.sum()))

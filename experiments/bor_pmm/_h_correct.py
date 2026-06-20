"""CORRECT H-form: build curl(1/eps)curl h = k0^2 h EXACTLY (full 3-component),
then reduce to transverse linear-in-q^2 by eliminating h_z PROPERLY, and
validate guided q vs the exact oracle (2.425231)."""
import numpy as np; import sys; sys.path.insert(0,".")
from coupled_radial_eigensolver import _fd_grid, _normal_eps
from fiber_oracle import fiber_modes
from scipy.linalg import eig

m,Rbig,N,k0=1,6.0,300,2.0; a=1.0; eps1,eps2=1.5**2,1.0**2
def eps_profile(rr): return np.where(rr<=a,eps1,eps2).astype(complex)

r,D,h=_fd_grid(Rbig,N)
eps=np.asarray(eps_profile(r),dtype=complex); eps_n=_normal_eps(eps)
I=np.eye(N); ir=np.diag(1/r); mr=m*ir
pinv=np.diag(1/eps); pinv_n=np.diag(1/eps_n)

# curl in cylindrical (fields exp(i m phi + i q z)), unknown (h_r,h_phi,h_z):
#  (curl h)_r   = (i m/r) h_z - i q h_phi
#  (curl h)_phi = i q h_r - D h_z
#  (curl h)_z   = (1/r)d/dr(r h_phi) - (i m/r) h_r = A h_phi - (i m/r) h_r,  A=D+1/r
# F = (1/eps) curl h component-wise. NORMAL component of curl h is the r-comp ->
#   D_r-like; but careful: it's E=(i/(k0 eps)) curl h. The component of E that is
#   NORMAL to the rings is E_r -> uses inverse rule. (curl h)_r feeds E_r -> pinv_n.
#   (curl h)_phi, (curl h)_z feed E_phi,E_z (tangential) -> pinv pointwise.
# Then curl F = k0^2 h.

def blocks(q):
    Z=np.zeros((N,N),dtype=complex)
    # curl h as linear op on (h_r,h_phi,h_z):
    Cr = np.hstack([Z,      -1j*q*I,  1j*mr])   # (curl h)_r
    Cp = np.hstack([1j*q*I, Z,        -D    ])  # (curl h)_phi
    Cz = np.hstack([-1j*mr, A,        Z     ])  # (curl h)_z  -- A defined below
    return Cr,Cp,Cz
A=D+ir
# We must redefine blocks with A in scope:
def curlmat(q):
    Z=np.zeros((N,N),dtype=complex)
    Cr=np.hstack([Z,-1j*q*I,1j*mr])
    Cp=np.hstack([1j*q*I,Z,-D])
    Cz=np.hstack([-1j*mr,A,Z])
    return np.vstack([Cr,Cp,Cz])

# (1/eps) diag with inverse rule on r-row:
P=np.block([[pinv_n,np.zeros((N,N)),np.zeros((N,N))],
            [np.zeros((N,N)),pinv,np.zeros((N,N))],
            [np.zeros((N,N)),np.zeros((N,N)),pinv]])

# Master M(q) = curl(q) P curl(q) - k0^2 I3.  Solve det via scanning is hard;
# instead build full polynomial eigenproblem in q (quadratic) for the FULL 3N
# unknown, then identify guided q. Reuse companion from before.
def Cmats():
    Z=np.zeros((N,N),dtype=complex)
    C0=np.vstack([np.hstack([Z,Z,1j*mr]),
                  np.hstack([Z,Z,-D]),
                  np.hstack([-1j*mr,A,Z])])
    Cq=np.vstack([np.hstack([Z,-1j*I,Z]),
                  np.hstack([1j*I,Z,Z]),
                  np.hstack([Z,Z,Z])])
    return C0,Cq
C0,Cq=Cmats()
M0=C0@P@C0; M1=C0@P@Cq+Cq@P@C0; M2=Cq@P@Cq
n3=3*N; I3=np.eye(n3,dtype=complex); M0c=M0-k0**2*I3
top=np.hstack([np.zeros((n3,n3),dtype=complex),I3])
bot=np.hstack([-M0c,-M1])
Abig=np.vstack([top,bot])
Bbig=np.vstack([np.hstack([I3,np.zeros((n3,n3),dtype=complex)]),
                np.hstack([np.zeros((n3,n3),dtype=complex),M2])])
qall,Vall=eig(Abig,Bbig)
fin=np.isfinite(qall); qall=qall[fin]; Vall=Vall[:,fin]
# guided window
qlo,qhi=np.sqrt(eps2)*k0,np.sqrt(eps1)*k0
hr=Vall[:N];hphi=Vall[N:2*N];hz=Vall[2*N:3*N]
g=[]
for j in range(len(qall)):
    q=qall[j]
    if not(qlo+1e-2<q.real<qhi-1e-2 and abs(q.imag)<1e-2): continue
    Hr=hr[:,j];Hp=hphi[:,j];Hz=hz[:,j]
    En=np.sqrt(np.sum(np.abs(Hr)**2+np.abs(Hp)**2+np.abs(Hz)**2))
    if En<1e-9: continue
    div=A@Hr+1j*mr@Hp+1j*q*Hz; rd=np.sqrt(np.sum(np.abs(div)**2))/(k0*En)
    # check bound (decays): amplitude near outer edge small
    amp=np.abs(Hr)+np.abs(Hp); amp=amp/amp.max()
    tail=amp[r>0.8*Rbig].max()
    g.append((q.real,float(rd.real),float(tail)))
g=sorted(set([(round(x[0],5),round(x[1],3),round(x[2],3)) for x in g]),reverse=True)
print("H-form (correct curl(1/eps)curl) guided-window modes (q, reldiv, tail):")
for x in g[:10]: print("  ",x)
print("EXACT oracle q:", [f"{q:.6f}" for q in fiber_modes(m,a,eps1,eps2,k0)])

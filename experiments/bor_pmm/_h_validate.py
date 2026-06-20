"""Validate H-form physical modes vs the fiber oracle (guided), and inspect the
REAL-Q spurious modes that drive the cascade energy floor."""
import numpy as np
import sys; sys.path.insert(0, ".")
from coupled_radial_eigensolver import _fd_grid, _normal_eps
from scipy.linalg import eig

# Step-index fiber to validate H-form against the EXACT oracle
m, Rbig, N, k0 = 1, 6.0, 300, 2.0
a, ncore, nclad = 1.0, 1.5, 1.0
def eps_profile(rr):
    return np.where(rr<=a, ncore**2, nclad**2).astype(complex)

def build_H(eps_profile):
    r,D,h=_fd_grid(Rbig,N)
    eps=np.asarray(eps_profile(r),dtype=complex); eps_n=_normal_eps(eps)
    I=np.eye(N); ir=np.diag(1/r); mr=m*ir; m2r2=(m**2)*np.diag(1/r**2)
    A=D+ir; Lm=D@D+ir@D-m2r2; dA=D@A
    pinv=np.diag(1/eps); pinv_n=np.diag(1/eps_n)
    LeiH=np.linalg.inv(Lm+k0**2*I)
    Phi_r=LeiH@(1j*A); Phi_p=LeiH@(-mr)
    B=np.block([[I+1j*D@Phi_r,1j*D@Phi_p],[-mr@Phi_r,I-mr@Phi_p]])
    K=np.block([[k0**2*I-pinv_n@m2r2,-1j*pinv_n@(mr@A)],
                [-1j*pinv@(D@mr),k0**2*I+pinv@dA]])
    q2,Vm=eig(K,B); q=np.sqrt(q2)
    out=[]
    for j in range(len(q)):
        Hr=Vm[:N,j];Hp=Vm[N:,j];Hz=q[j]*(LeiH@(1j*A@Hr-mr@Hp))
        En=np.sqrt(np.sum(np.abs(Hr)**2+np.abs(Hp)**2+np.abs(Hz)**2))
        div=A@Hr+1j*mr@Hp+1j*q[j]*Hz
        rd=np.sqrt(np.sum(np.abs(div)**2))/(k0*max(En,1e-300))
        out.append(dict(q=q[j],reldiv=float(rd.real),Hr=Hr,Hp=Hp,r=r))
    return out

modesH=build_H(eps_profile)
qlo,qhi=nclad*k0, ncore*k0
guidedH=sorted([md for md in modesH if qlo+1e-2<md["q"].real<qhi-1e-2 and abs(md["q"].imag)<1e-3 and md["reldiv"]<1.0],
               key=lambda md:-md["q"].real)
print("H-form guided modes (m=1, step fiber), q values:")
for md in guidedH[:6]:
    print(f"   q={md['q'].real:.6f}  reldiv={md['reldiv']:.3f}")

# E-form oracle comparison
from coupled_radial_eigensolver import guided_modes

gE=guided_modes(1,a,Rbig,N,ncore**2,nclad**2,k0)
print("E-form guided modes q (validated vs oracle ~1e-4):")
for md in gE[:6]:
    print(f"   q={md['q'].real:.6f}  reldiv={md['reldiv']:.3f}")

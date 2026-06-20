"""Mixed (E_z,H_z) longitudinal formulation: is it linear in q^2 (no companion)?
Standard fiber/BOR longitudinal formulation: (E_z,H_z) satisfy coupled scalar
Helmholtz. With gamma^2=eps k0^2-q^2, the transverse fields are algebraic in
(E_z,H_z) with 1/gamma^2 prefactors. The eigenproblem in (E_z,H_z):

  (L_m + eps k0^2) E_z = q^2 E_z + coupling(H_z)
The coupling enters through transverse-field continuity, and the transverse
recovery E_t,H_t ~ (1/gamma^2)[...] = (1/(eps k0^2 - q^2))[...]. That 1/gamma^2
is the q-NONLINEARITY. As an INTERIOR (no interface) operator on a single layer,
the bulk equation IS linear in q^2:
   L_m E_z = -(eps k0^2 - q^2) E_z  ->  (L_m + eps k0^2) E_z = q^2 E_z   (LINEAR!)
   (L_m + eps k0^2) H_z = q^2 H_z
DECOUPLED in the bulk (m,q). The coupling is ONLY at eps-interfaces (boundary
conditions). So a GENERALIZED linear eigenproblem K[Ez;Hz]=q^2 [Ez;Hz] with the
interface coupling folded into K avoids the 1/gamma^2 nonlinearity IFF the
interface coupling can be written q-independently. Check: does (L_m+eps k0^2)
give the guided q as a plain eigenproblem when eps is the inverse-rule-averaged
profile? (TE/TM-like, m=0 decouples cleanly.)"""
import numpy as np; import sys; sys.path.insert(0,".")
from coupled_radial_eigensolver import _fd_grid
from fiber_oracle import fiber_modes
from scipy.linalg import eig

m,Rbig,N,k0=1,6.0,400,2.0; a=1.0; eps1,eps2=1.5**2,1.0**2
r,D,h=_fd_grid(Rbig,N)
eps=np.where(r<=a,eps1,eps2).astype(complex)
I=np.eye(N); ir=np.diag(1/r); m2r2=(m**2)*np.diag(1/r**2)
Lm=D@D+ir@D-m2r2
# bulk decoupled longitudinal operator: (L_m + eps k0^2) f = q^2 f
Kez=Lm+k0**2*np.diag(eps)
q2,V=eig(Kez)   # standard eig
q=np.sqrt(q2)
qlo,qhi=np.sqrt(eps2)*k0,np.sqrt(eps1)*k0
gw=sorted([qq.real for qq in q if qlo+1e-2<qq.real<qhi-1e-2 and abs(qq.imag)<1e-2],reverse=True)
print("Bulk-decoupled (L_m+eps k0^2) E_z eig, guided window q:", [f"{x:.5f}" for x in gw[:6]])
print("EXACT oracle (full vector HE11):", [f"{qq:.6f}" for qq in fiber_modes(m,a,eps1,eps2,k0)])
print("-> bulk-decoupled MISSES the vector coupling (gives scalar/LP q, not HE/EH).")

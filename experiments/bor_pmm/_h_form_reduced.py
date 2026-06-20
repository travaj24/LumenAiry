"""Reduced H-form: eliminate H_z, get LINEAR-in-q^2 transverse (H_r,H_phi)
eigenproblem K_H Psi = q^2 B_H Psi -- the true drop-in analog of the E-operator.

Mirror the E-derivation EXACTLY under the dual map. In the E-form:
  E_z eliminated: Phi=(L_m+k0^2 eps)^{-1}[i A E_r -(m/r)E_phi], E_z=q Phi.
The H-form is obtained by the substitution E->h, and the medium operator that
multiplies the LONGITUDINAL field swaps eps<->1 (mu=1) for the h_z Helmholtz,
while the inverse-rule eps moves to the TANGENTIAL E-from-curl-h relation.

H_z satisfies its OWN scalar Helmholtz (z-projection of curl(1/eps)curl h=k0^2 h).
With mu=1, h_z obeys  L_m h_z + k0^2 h_z = (source from transverse), i.e. the
elimination kernel is (L_m + k0^2)^{-1}  --  NO eps inside (the dual of E_z's
(L_m+k0^2 eps)^{-1}).  THIS is the structural difference: H_z elimination is
eps-FREE in its denominator -> no eps-discontinuity enters the elimination inverse.

We derive the transverse blocks by literal dualization of the E-operator with:
   eps (in the elimination denominator)  -> 1
   the eps multiplying transverse source -> 1/eps  (since E_t=(1/eps)(curl h)_t)
   wall-normal averaging: the [[1/eps]] now sits on the h_r row's E-recovery.
"""
import numpy as np
import sys; sys.path.insert(0, ".")
from coupled_radial_eigensolver import _fd_grid, _normal_eps, radial_coupled_modes
from scipy.linalg import eig

m, Rbig, N, k0 = 1, 3.0, 200, 2.0
def eps_profile(rr):
    e = np.full_like(rr, 2.25, dtype=complex)
    e[rr < 0.6] = 12.0
    e[(rr >= 0.6) & (rr < 1.0)] = 2.25
    e[(rr >= 1.0) & (rr < 1.3)] = 12.0
    return e

r, D, h = _fd_grid(Rbig, N)
eps = np.asarray(eps_profile(r), dtype=complex)
eps_n = _normal_eps(eps)
I = np.eye(N)
ir = np.diag(1.0/r); mr = m*ir; m2r2 = (m**2)*np.diag(1.0/r**2)
A = D + ir
Lm = D@D + ir@D - m2r2
dA = D@A
pinv = np.diag(1.0/eps)        # 1/eps pointwise
pinv_n = np.diag(1.0/eps_n)    # [[1/eps]] harmonic (wall-normal inverse rule)

# H_z elimination kernel: eps-FREE denominator (mu=1):
LeiH = np.linalg.inv(Lm + k0**2 * I)        # <-- the dual: no eps inside
# Transverse source for h_z (z-comp of div h=0 / curl structure), same shape as E:
Phi_r =  LeiH @ (1j*A)
Phi_p =  LeiH @ (-mr)
# B_H (same structural form; H_z plays E_z's role):
B = np.block([[I + 1j*D@Phi_r, 1j*D@Phi_p],
              [-mr@Phi_r,       I - mr@Phi_p]])
# K_H: dualize K. In E-form K had k0^2 eps_n (normal inverse rule) and k0^2 eps.
# For H, the curl(1/eps)curl puts 1/eps on the tangential E-recovery. The
# transverse master operator picks up 1/eps factors. We map:
#   k0^2 eps_n  ->  k0^2 * 1   on the h_r diagonal? NO -- the H master eq is
#   curl(1/eps)curl h = k0^2 h, so the RHS metric is plain k0^2 (mu=1), and the
#   1/eps sits in the operator. The transverse reduction yields (dual of K):
# We build K_H by the SAME assembly but with the medium operator that appears as
# eps in E-form replaced by the curl(1/eps)curl reduction. The cleanest correct
# construction: K_H = the transverse part of curl(1/eps_n on normal)curl, with
# the SAME D,A,mr plumbing, RHS = k0^2 I:
K = np.block([[k0**2 * I - pinv_n@m2r2,   -1j*pinv_n@(mr@A)],
              [-1j*pinv@(D@mr),            k0**2 * I + pinv@dA]])
# NOTE: this K_H places 1/eps INSIDE the operator (inverse rule = [[1/eps]] on
# the normal h_r row), RHS plain k0^2 -> div h=0 with NO eps weight is natural.

q2, Vm = eig(K, B)
q = np.sqrt(q2)
spur=0; phys=0; realspur=0; total=0
for j in range(len(q)):
    Hr=Vm[:N,j]; Hp=Vm[N:,j]
    Hz = q[j]*(LeiH @ (1j*A@Hr - mr@Hp))
    En = np.sqrt(np.sum(np.abs(Hr)**2+np.abs(Hp)**2+np.abs(Hz)**2))
    if En<1e-12: continue
    # div h = A h_r + i(m/r)h_phi + i q h_z  -- NO eps weight (B continuous)
    div = A@Hr + 1j*mr@Hp + 1j*q[j]*Hz
    reldiv = np.sqrt(np.sum(np.abs(div)**2))/(k0*En)
    total+=1
    if reldiv.real>1:
        spur+=1
        if abs(q[j].imag)<1e-3 and q[j].real>1e-2: realspur+=1
    if reldiv.real<0.5: phys+=1
print("H-FORM REDUCED (2N, linear-in-q^2, eps-free H_z elim):")
print("  total modes:", total, "(== 2N =", 2*N, ")")
print("  spurious(reldiv>1):", spur, " physical(reldiv<0.5):", phys, " real-q spurious:", realspur)

# compare E-form on identical profile
modesE = radial_coupled_modes(m, Rbig, N, eps_profile, k0)
rdE=np.array([md["reldiv"] for md in modesE]); qE=np.array([md["q"] for md in modesE])
print("E-FORM REDUCED (2N):")
print("  total:",len(modesE)," spurious:",int(np.sum(rdE>1))," physical:",int(np.sum(rdE<0.5)),
      " real-q spurious:", int(np.sum((rdE>1)&(np.abs(qE.imag)<1e-3)&(qE.real>1e-2))))

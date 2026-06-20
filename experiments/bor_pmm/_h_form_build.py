"""Build the H-formulation (H_r,H_phi) q^2 eigenproblem by DUALITY of the
validated E-operator, and compare spurious counts on the structured ring.

DERIVATION (exp(-iwt), mu=1, h normalized, curl E=i k0 h, curl h=-i k0 eps E):

The two curl equations are NOT symmetric under E<->h: eps sits only in the
second.  To get the H-form we eliminate E instead of h.  Take curl of curl h:
   curl curl h = -i k0 curl(eps E) = -i k0 [ (grad eps) x E + eps curl E ]
              = -i k0 (grad eps) x E + eps k0^2 h.
With curl E = i k0 h.  And curl curl h = grad(div h) - lap h, div h = 0 (always,
mu=1, no magnetic charge -> div h=0 EXACTLY, the KEY point).  And E = (i/(k0 eps)) curl h.
So:
   -lap h = eps k0^2 h - i k0 (grad eps) x [ (i/(k0 eps)) curl h ]
   -lap h = eps k0^2 h + (grad eps/eps) x curl h.
That is the H vector Helmholtz with the (grad eps/eps) x curl h term carrying ALL
the eps-discontinuity.  Crucially div h = 0 with NO eps weighting -> H_r continuous
(B_r=mu0 H_r continuous), nodal H_r is consistent.

For the q^2 transverse eigenproblem we mirror the E-derivation with the dual
substitution.  The cleanest route: the E-solver already provides h for every
mode via curl E = i k0 h.  The H-form modes are the SAME physical modes; the
question is only whether DISCRETIZING in (H_r,H_phi) directly yields fewer
spurious eigenvectors.  We build the H-operator analogous to the E one:

  E-form unknown (E_r,E_phi), E_z eliminated via (L_m+k0^2 eps)^{-1}.
  H-form unknown (H_r,H_phi), H_z eliminated via (L_m+k0^2 eps_something)^{-1}.

Mirror Maxwell component eqs.  From curl h = -i k0 eps E and curl E = i k0 h,
eliminate E: E = (i/(k0 eps)) curl h.  Substitute into curl E = i k0 h:
   curl[ (1/eps) curl h ] = k0^2 h.        (the H master equation)
This is the magnetic-field master eq with operator curl (1/eps) curl.  The 1/eps
is now INSIDE the curl-curl -> the inverse rule appears as [[1/eps]] DIRECTLY
(the natural averaging), and eps_tangential -> the 1/eps that multiplies the
transverse curl pieces.

We implement curl (1/eps) curl h = k0^2 h in cylindrical (m,q), reduce to
transverse (H_r,H_phi) with H_z eliminated, and get q^2 K_H Psi = ... .
Rather than re-derive every block by hand (error-prone), we VERIFY the central
claim differently: we transform the validated E-modes' h-fields and test the
H-form's NODAL-H_r divergence consistency, i.e. whether div h = 0 holds
NODALLY for ALL eigenvectors of a nodal-H operator, which is what kills spurious
modes.  div h = (1/r) d/dr(r h_r) + (im/r) h_phi + i q h_z = 0 with NO eps weight.
"""
import numpy as np

np.set_printoptions(suppress=True)
import sys; sys.path.insert(0, ".")
from coupled_radial_eigensolver import _fd_grid, _normal_eps
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

# ---- H-FORMULATION: curl (1/eps) curl h = k0^2 h, eliminate H_z ----
# In cylindrical with exp(i m phi + i q z), define p = 1/eps (DIAGONAL choice of
# averaging is the design knob). curl h components:
#   (curl h)_r   =  (im/r) h_z - i q h_phi
#   (curl h)_phi =  i q h_r - dh_z/dr
#   (curl h)_z   =  (1/r) d/dr(r h_phi) - (im/r) h_r  = A h_phi - (im/r) h_r
# Let F = (1/eps) curl h  (component-wise multiply by p=1/eps). The eps that
# multiplies the NORMAL curl component (curl h)_r is the wall-normal one -> p_n;
# tangential components (phi,z) use pointwise p. (This is the dual inverse rule:
# E_t = (1/eps)(curl h)_t pointwise; E_r normal uses harmonic-mean eps.)
# Then curl F = k0^2 h.
# Build with H_z eliminated from div h = 0 -> i q h_z = -(A h_r) - (im/r) h_phi
#  => h_z = (i/q)(A h_r + (im/r) h_phi).  But q is the eigenvalue -> messy.
# Instead eliminate H_z via its OWN equation (z-component of master eq), exactly
# mirroring the E-form's Phi elimination.

p  = np.diag(1.0/eps)        # pointwise 1/eps  (tangential)
pn = np.diag(1.0/eps_n)      # wall-normal 1/eps = [[1/eps]] harmonic -> this IS the inverse rule

# We assemble the full 3x3 (h_r,h_phi,h_z) master operator M with curl(1/eps)curl
# and split q^2. Represent curl as block matrices acting on (h_r,h_phi,h_z):
Z = np.zeros((N,N), dtype=complex)
# (curl h): r:  im/r h_z - i q h_phi ; phi: i q h_r - D h_z ; z: A h_phi - im/r h_r
# Write curl = C0 + q*Cq where Cq couples via i q.
# C0 (q-independent):
# r-row:   [0,            0,        i m/r ]
# phi-row: [0,            0,        -D    ]
# z-row:   [-im/r,        A,        0     ]
C0 = np.block([[Z,    Z,    1j*mr],
               [Z,    Z,    -D   ],
               [-1j*mr, A,   Z   ]])
# Cq (multiplies q): from -i q h_phi (r-row) and i q h_r (phi-row):
Cq = np.block([[Z,    -1j*I, Z],
               [1j*I, Z,     Z],
               [Z,    Z,     Z]])
# (1/eps) applied componentwise with normal rule on r-row:
P = np.block([[pn, Z, Z],[Z, p, Z],[Z, Z, p]])   # F = P @ curl h
# curl F = k0^2 h.  curl acts again with same C0,Cq structure.
# Master: (C0 + q Cq) P (C0 + q Cq) h = k0^2 h.
# Expand: [C0 P C0 + q(C0 P Cq + Cq P C0) + q^2 Cq P Cq] h = k0^2 h.
M0  = C0 @ P @ C0
M1  = C0 @ P @ Cq + Cq @ P @ C0
M2  = Cq @ P @ Cq
# This is a QUADRATIC eigenproblem in q: (M0 - k0^2 Iall + q M1 + q^2 M2) h = 0.
# That's the 1/gamma^2-style nonlinearity the user worries about. Linearize via
# companion OR eliminate h_z to get linear-in-q^2.  Since M1 is ODD in q and
# couples to h_z, eliminating h_z removes the linear-q term (as in E-form).
# Quick assessment: solve the QUADRATIC via companion (2*3N) to GET the spectrum
# and divergence of the H-eigenvectors, to answer "fewer spurious?".
n3 = 3*N
Iall = np.eye(n3, dtype=complex)
Mc0 = M0 - k0**2 * Iall
# companion: [ [0, I],[ -M2^{-1} Mc0, -M2^{-1} M1 ] ] has eig = q. But M2 is singular
# (z-row/col zero in Cq) -> use generalized companion (polyeig):
# [Mc0  M1 ; 0  I] x = q [ -... ]  Use scipy via building A x = q B x:
# Standard linearization for (M0c + q M1 + q^2 M2): 
#   A = [[0, I],[-M0c, -M1]],  B=[[I,0],[0,M2]],  eig A x = q B x.
top = np.hstack([np.zeros((n3,n3),dtype=complex), Iall])
bot = np.hstack([-Mc0, -M1])
Abig = np.vstack([top, bot])
Btop = np.hstack([Iall, np.zeros((n3,n3),dtype=complex)])
Bbot = np.hstack([np.zeros((n3,n3),dtype=complex), M2])
Bbig = np.vstack([Btop, Bbot])
qall, Vall = eig(Abig, Bbig)
finite = np.isfinite(qall)
qall = qall[finite]; Vall = Vall[:, finite]
# divergence div h = A h_r + (im/r) h_phi + i q h_z  (NO eps weight)
hr = Vall[:N]; hphi = Vall[N:2*N]; hz = Vall[2*N:3*N]
spur = 0; phys = 0; realspur=0; total=0
for j in range(len(qall)):
    q = qall[j]
    if not np.isfinite(q): continue
    Hr=hr[:,j]; Hp=hphi[:,j]; Hz=hz[:,j]
    En = np.sqrt(np.sum(np.abs(Hr)**2+np.abs(Hp)**2+np.abs(Hz)**2))
    if En < 1e-12: continue
    div = A@Hr + 1j*mr@Hp + 1j*q*Hz
    reldiv = np.sqrt(np.sum(np.abs(div)**2))/(k0*En)
    total+=1
    if reldiv.real>1: 
        spur+=1
        if abs(q.imag)<1e-3 and q.real>1e-2: realspur+=1
    if reldiv.real<0.5: phys+=1
print("H-FORM (quadratic, full 3N, div h no-eps-weight):")
print("  finite-q modes:", total)
print("  spurious(reldiv>1):", spur, " physical(reldiv<0.5):", phys, " real-q spurious:", realspur)

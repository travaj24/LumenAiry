from common import *
from lumenairy.elements.pmm import pmm_jones_2d
wl = 1.0e-6; Px = Py = 0.9e-6; dep = 0.35e-6
nsup, nsub = 1.0, 1.45
def iso(e): return e*np.eye(3, dtype=complex)
S = 24
cell = np.zeros((S,S,3,3), dtype=complex); cell[...] = iso(1.0); cell[6:18,6:18] = iso(12.11)  # Si pillar in air
print("== energy closure: lossless Si pillar / SiO2 substrate, Px=Py=0.9um ==")
print(" theta phi   n_or  form     sum(R+T) row0        row1         max|1-E|")
for (thd, phid) in ((0,0),(20,0),(20,30)):
    for nn in (5,9,11):
        for form in ("laurent","li"):
            th, ph = np.deg2rad(thd), np.deg2rad(phid)
            o,R,T,J = pmm_jones_2d(Px,Py,cell,nsub,nsup,dep,wl,theta=th,phi=ph,
                                   degree=11,n_orders=nn,formulation=form)
            E = R.sum(1)+T.sum(1)
            print(f" {thd:4d} {phid:4d} {nn:4d}  {form:8s} {E[0]:.10f} {E[1]:.10f}  {np.max(np.abs(E-1)):.2e}", flush=True)
print()
print("== RECIPROCITY: J(theta,phi)^T  vs  J(theta, phi+pi) (reflection, lossless) ==")
# For a reflection Jones in the lab basis, reciprocity relates the (0,0) reflection at
# (theta,phi) to that at the reversed in-plane wavevector (theta, phi+pi) with the
# standard sign flip on the component along the reversal.
D = np.diag([-1.0, 1.0])   # x-component flips under phi -> phi+pi about y? tested both
for (thd, phid) in ((20,0),(20,30)):
    th, ph = np.deg2rad(thd), np.deg2rad(phid)
    o,R,T,Ja = pmm_jones_2d(Px,Py,cell,nsub,nsup,dep,wl,theta=th,phi=ph,degree=11,n_orders=9)
    o,R,T,Jb = pmm_jones_2d(Px,Py,cell,nsub,nsup,dep,wl,theta=th,phi=ph+np.pi,degree=11,n_orders=9)
    c,s = np.cos(ph), np.sin(ph)
    Rot = np.array([[c,s],[-s,c]])           # lab -> (p,s) of the incident plane
    Ja_ps = Rot@Ja@Rot.T
    c2,s2 = np.cos(ph+np.pi), np.sin(ph+np.pi)
    Rot2 = np.array([[c2,s2],[-s2,c2]])
    Jb_ps = Rot2@Jb@Rot2.T
    G = np.diag([1.0,-1.0])
    print(f"  th={thd} phi={phid}:")
    print(f"    max|Ja_ps - Jb_ps^T|            = {np.max(np.abs(Ja_ps-Jb_ps.T)):.3e}")
    print(f"    max|Ja_ps - G Jb_ps^T G|        = {np.max(np.abs(Ja_ps-G@Jb_ps.T@G)):.3e}")
    print(f"    max|Ja_ps - Jb_ps|              = {np.max(np.abs(Ja_ps-Jb_ps)):.3e}")
    print(f"    Ja_ps=\n{Ja_ps}\n    Jb_ps=\n{Jb_ps}", flush=True)
print()
print("== RECIPROCITY on a CHIRAL (no mirror symmetry) cell: J^T symmetry at normal incidence ==")
ch = np.zeros((S,S,3,3), dtype=complex); ch[...] = iso(1.0)
ch[4:12,4:20] = iso(12.11); ch[12:20,4:12] = iso(12.11)   # L-shape (no mirror line)
for nn in (5,9):
    o,R,T,J = pmm_jones_2d(Px,Py,ch,nsub,nsup,dep,wl,theta=0.0,phi=0.0,degree=9,n_orders=nn)
    print(f"  n_orders={nn}: |Jxy-Jyx|={abs(J[0,1]-J[1,0]):.3e}  |Jxy|={abs(J[0,1]):.3e} "
          f"|Jxy+Jyx|={abs(J[0,1]+J[1,0]):.3e}  E={R.sum(1)[0]+T.sum(1)[0]:.8f}", flush=True)

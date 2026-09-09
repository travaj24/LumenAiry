"""Do the PROBE gauge (tau = exp(+i a p)) and the SHIPPED gauge
(tau = exp(-i a p)) give the SAME per-order answer on a CHIRAL cell at
oblique?  If they do, the Bloch-sign difference is fully internal and the
out-of-plane blocks transplant with no compensating sign; if they differ by an
order mirror, the gauge is load-bearing and must be settled before anything is
integrated.  Oracle: rcwa_jones_1d_segments (independent engine)."""
import os
import sys

for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import probe_common as pc

from lumenairy.elements.pmm.twod_staggered import pmm_efficiency_2d_staggered
from lumenairy.elements.rcwa.oned import rcwa_jones_1d_segments

WL, P, DEP = 1.0, 1.7, 0.35
prof = [1.0, 2.25, 4.0]                    # chiral under x -> -x
cell4 = np.zeros((3,3,3,3), dtype=complex)
for i, e in enumerate(prof):
    cell4[i, :] = e * np.eye(3)
cell2 = np.tile(np.array(prof, dtype=complex)[:, None], (1,3))
th = np.deg2rad(25.0)
o1, R1, T1, _ = rcwa_jones_1d_segments(P, [(1/3., e) for e in prof], 1.5, 1.0,
                                       DEP, WL, n_orders=41, theta=th)
o1 = np.asarray(o1)
idx = {int(m): j for j, m in enumerate(o1)}
o2, R2, T2, J2 = pc.solve_slab(P, P, cell4, 1.5, 1.0, DEP, WL, M=8,
                               theta=th, phi=0.0, candidate="eform")
o2 = np.asarray(o2)
sel = o2[:,1] == 0
ox = o2[sel,0]
Rp = np.asarray(R2)[:,sel]
Tp = np.asarray(T2)[:,sel]
os3, Rs, Ts = pmm_efficiency_2d_staggered(P, P, cell2, 1.5, 1.0, DEP, WL,
                                          degree=8, n_orders=5,
                                          polarization="te", theta=th, phi=0.0)
os3 = np.asarray(os3)
ss = os3[:,1]==0
oxs = os3[ss,0]
Rsh = np.asarray(Rs)[ss]
print("  m  |  oracle R(+m)   oracle R(-m) | PROBE(eform) row1  SHIPPED te")
for m in (-2,-1,0,1,2):
    j = int(np.where(ox==m)[0][0])
    k = int(np.where(oxs==m)[0][0])
    print(f" {m:+d}  |  {R1[1,idx[m]]:.8f}  {R1[1,idx[-m]]:.8f} | "
          f"{Rp[1,j]:.8f}       {Rsh[k]:.8f}")
d_same = max(abs(Rp[1,int(np.where(ox==m)[0][0])] - R1[1,idx[m]]) for m in (-2,-1,1,2))
d_mirr = max(abs(Rp[1,int(np.where(ox==m)[0][0])] - R1[1,idx[-m]]) for m in (-2,-1,1,2))
print(f"  PROBE vs oracle(+m) = {d_same:.3e}   vs oracle(-m) = {d_mirr:.3e}")
d_ps = max(abs(Rp[1,int(np.where(ox==m)[0][0])] - Rsh[int(np.where(oxs==m)[0][0])])
           for m in (-2,-1,0,1,2))
print(f"  PROBE(eform) vs SHIPPED(te), same label m: {d_ps:.3e}")

"""Is the prototype's (3,3) L-cell 'open item' a 180-degree PATTERN rotation?

Hypothesis from the chiral 1-D measurement: the PROBE's eps_cell indexing is
reversed relative to its basis, so it solves the pattern rotated by 180 deg
about z (rho).  A cell whose pattern is rho-symmetric up to a lattice shift
(every stripe, the (2,2) single pillar) is blind to it; the (3,3) L is not.
TEST: feed the probe the rho-ROTATED cell and compare PER ORDER against both
Fourier oracles.  If the 2.2e-03 collapses, the open item is a convention, not
a corner.
"""
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import probe_common as pc

from lumenairy.elements.pmm import pmm_jones_2d
from lumenairy.elements.rcwa import rcwa_jones_2d

WL=1.0
PX=PY=1.2
DEP=0.4
NSUB,NSUP=1.5,1.0
er = pc.uniaxial(1.5,1.7,35.0,azim_deg=25.0)
eg=np.eye(3,dtype=complex)
def L(): 
    e=np.zeros((3,3,3,3),dtype=complex)
    e[:,:]=eg
    for i, j in ((0, 0), (1, 0), (0, 1)):
        e[i, j] = er
    return e
def rho(e):                       # eps'[s] = eps[N-1-s] on both axes
    return np.ascontiguousarray(e[::-1, ::-1])
def rho_tensor(e):                # 180-deg rotation about z of every tensor
    o=np.array(e,dtype=complex)
    o[...,0,2]*=-1
    o[...,1,2]*=-1
    o[...,2,0]*=-1
    o[...,2,1]*=-1
    return o
def upsample(ec,n): return np.repeat(np.repeat(np.asarray(ec),n,axis=0),n,axis=1)
def cmp(tag,o_s,R_s,T_s,J_s,o_r,R_r,T_r,J_r):
    idx={tuple(int(v) for v in r):j for j,r in enumerate(np.asarray(o_r))}
    dR=dT=0.
    for i,r in enumerate(np.asarray(o_s)):
        j=idx.get(tuple(int(v) for v in r))
        if j is None:
            continue
        dR=max(dR,float(np.max(np.abs(np.asarray(R_s)[:,i]-np.asarray(R_r)[:,j]))))
        dT=max(dT,float(np.max(np.abs(np.asarray(T_s)[:,i]-np.asarray(T_r)[:,j]))))
    dJ=float(np.max(np.abs(np.asarray(J_s)-np.asarray(J_r))))
    print(f"    {tag:44s} dR={dR:.3e}  dT={dT:.3e}  dJ={dJ:.3e}")
    return dR,dT,dJ

base = L()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    orc = rcwa_jones_2d(PX,PY,upsample(base,13),NSUB,NSUP,DEP,WL,
                        n_orders_x=9,n_orders_y=9)
    orc7 = rcwa_jones_2d(PX,PY,upsample(base,11),NSUB,NSUP,DEP,WL,
                         n_orders_x=7,n_orders_y=7)
    hyb = pmm_jones_2d(PX,PY,base,NSUB,NSUP,DEP,WL,degree=9,n_orders=13,
                       stabilize=True)
print("ORACLES on the ORIGINAL L cell (the reference configuration):")
cmp("rcwa(7) vs rcwa(9)",*orc7,*orc)
cmp("hybrid(13) vs rcwa(9)",*hyb,*orc)
print("\nPROBE candidate (a), M=6, on variants of the SAME physical cell:")
for tag, cell in (("as-is", base),
                  ("rho(pattern)", rho(base)),
                  ("rho(tensor) only", rho_tensor(base)),
                  ("rho(pattern) + rho(tensor)", rho_tensor(rho(base)))):
    st = pc.solve_slab(PX,PY,cell,NSUB,NSUP,DEP,WL,M=6,candidate="a")
    cmp(f"stag[{tag}] vs rcwa(9)",*st,*orc)

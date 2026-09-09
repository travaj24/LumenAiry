"""Does the SHIPPED in-plane staggered path mirror the (3,3) L pattern?

The probe's in-plane control on this cell has its orders (m,n) -> (-m,-n)
against both Fourier oracles at NORMAL incidence (GATE 0 table).  The library
uses the OPPOSITE Bloch/kernel sign, so it must be measured separately -- this
is the two-arm, same-build check that decides what the out-of-plane
integration has to compensate for.
"""
import os, warnings
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import numpy as np
from lumenairy.elements.pmm.twod_staggered import pmm_jones_2d_staggered
from lumenairy.elements.rcwa import rcwa_jones_2d
from lumenairy.elements.rcwa._core import uniaxial_tensor

WL=1.0; PX=PY=1.2; DEP=0.4; NSUB,NSUP=1.5,1.0
er = uniaxial_tensor(1.5,1.7,np.pi/2, phi=0.42)     # IN-PLANE (tilt 90 deg)
eg = np.eye(3,dtype=complex)
e=np.zeros((3,3,3,3),dtype=complex); e[:,:]=eg
for i,j in ((0,0),(1,0),(0,1)): e[i,j]=er
def up(ec,n): return np.repeat(np.repeat(np.asarray(ec),n,axis=0),n,axis=1)
for th,ph in ((0.0,0.0),(np.deg2rad(20.),np.deg2rad(35.))):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o_r,R_r,T_r,J_r = rcwa_jones_2d(PX,PY,up(e,13),NSUB,NSUP,DEP,WL,
                                        theta=th,phi=ph,n_orders_x=9,n_orders_y=9)
        o_s,R_s,T_s,J_s = pmm_jones_2d_staggered(PX,PY,e,NSUB,NSUP,DEP,WL,
                                                 degree=6,n_orders=4,
                                                 theta=th,phi=ph)
    idx={tuple(int(v) for v in r):j for j,r in enumerate(np.asarray(o_r))}
    dS=dM=0.
    print(f"\n theta={np.rad2deg(th):.0f} phi={np.rad2deg(ph):.0f}   "
          f"(m,n)    stagT        rcwaT(+m,+n)   rcwaT(-m,-n)")
    for i,r in enumerate(np.asarray(o_s)):
        k=tuple(int(v) for v in r); km=(-k[0],-k[1])
        if k not in idx or km not in idx: continue
        a=float(np.max(np.abs(T_s[:,i]-T_r[:,idx[k]])))
        b=float(np.max(np.abs(T_s[:,i]-T_r[:,idx[km]])))
        dS=max(dS,a); dM=max(dM,b)
        if abs(k[0])<=1 and abs(k[1])<=1:
            print(f"            {str(k):9s} {T_s[0,i]:.7f}   {T_r[0,idx[k]]:.7f}"
                  f"    {T_r[0,idx[km]]:.7f}")
    print(f"   max|stag(m,n) - rcwa(m,n)| = {dS:.3e}    "
          f"max|stag(m,n) - rcwa(-m,-n)| = {dM:.3e}")
    print(f"   sum R stag {R_s.sum(axis=1)}  rcwa {R_r.sum(axis=1)}")

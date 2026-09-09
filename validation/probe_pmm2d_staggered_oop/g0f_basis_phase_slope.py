"""GATE 0f -- which transverse momentum does the SHIPPED staggered order-m
field actually carry?

Reconstructs the real-space E1 profile of the library's own order-m mode
combination in a UNIFORM superstrate at oblique and fits its phase slope
against -(alpha0 + m G) and +(alpha0 + m G).  Measured, not read off a
comment: the shipped basis carries exp(-i alpha0 x), so its order label m
sits on the harmonic exp(-i(alpha0 + m G) x).
"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
import numpy as np, lumenairy
assert lumenairy.__file__.startswith(os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..")) + os.sep)
from lumenairy.elements.pmm.twod_staggered import (
    Granet2DTransverseE, _far_projector_2d, _homog_geom_cache,
    _homog_region_modes, _pmm2d_project_orders, _modleg_value_deriv)
from lumenairy.elements.pmm._core import _guarded_lstsq
wl,P=1.0,1.7; k0=2*np.pi/wl; G=2*np.pi/P
th=np.deg2rad(25.); kx0=np.sin(th); a0x=kx0*k0
Nx=Ny=2; M=9
sol=Granet2DTransverseE(P,P,Nx,Ny,M,np.full((Nx,Ny),1.0+0j),alpha0x=a0x,alpha0y=0.0,k0=k0)
geom=_homog_geom_cache(sol); W,V,lam=_homog_region_modes(geom,1.0+0j)
no=4; ox=np.arange(-no,no+1); oy=ox
order_x=np.tile(ox,len(oy)); order_y=np.repeat(oy,len(ox))
P1,P2=_far_projector_2d(sol.bx,sol.by,ox,oy,a0x,0.0)
qq=sol.q*sol.q; H=_pmm2d_project_orders(P1,P2,W,qq); Nfo=len(order_x)
bx,by=sol.bx,sol.by
xs=np.linspace(0.005,0.995,120)*P; y0=0.137*P
def ev(basis,sets,xarr):
    seg=np.clip((xarr/basis.h).astype(int),0,basis.N-1)
    u=2.0*(xarr-basis.xb[seg])/basis.h-1.0
    S=np.array(sets); out=np.zeros((S.shape[0],len(xarr)),dtype=complex)
    for i,(s,uu) in enumerate(zip(seg,u)):
        Vv,_=_modleg_value_deriv(basis.M,np.array([uu])); out[:,i]=S[:,s,:]@Vv[:,0]
    return out
fx=ev(bx,bx.B,xs); fy=ev(by,by.Btilde,np.array([y0]))[:,0]; qx=bx.dim
for mtarget in (0,1,-1):
    sel=((order_x==mtarget)&(order_y==0)).astype(complex)
    rhs=np.concatenate([sel,0.0*sel])
    cinc=_guarded_lstsq(H,rhs,"probe"); e1=(W@cinc)[:qq]
    coef=(e1.reshape(by.dim,qx)*fy[:,None]).sum(axis=0); E1x=coef@fx
    slope=np.polyfit(xs,np.unwrap(np.angle(E1x)),1)[0]
    print(f" order m={mtarget:+d}: slope={slope:+.5f}   -(a+mG)={-(a0x+mtarget*G):+.5f}"
          f"   -(a-mG)={-(a0x-mtarget*G):+.5f}   +(a+mG)={+(a0x+mtarget*G):+.5f}")

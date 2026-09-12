"""Downstream OPD impact of the n1 double-count on carrier='auto' / ndarray."""
import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import _screen_obliquity_delta, _facet_axial_momenta
from lumenairy.elements.lenses import surface_sag_general as sg
lam=0.55e-6
N=257; dx=8e-6
ax=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(ax,ax)
R=25e-3; n1=1.5168; n2=1.0          # immersed FIRST medium, exit into air
sag=np.nan_to_num(sg(X**2+Y**2,R,0.0,None))
gy,gx=np.gradient(sag,dx,dx)
print(" theta  |  correction with the TRUE q = n1 sin(t)  |  with the shipped n1*q "
      "|  ratio | vs the exact (T1) target")
for th in (0.02,0.05,0.10):
    q_true=n1*np.sin(th)
    q_ship=n1*q_true                                  # what 'auto'/ndarray give
    d_t=_screen_obliquity_delta(sag,gx,gy,0.,0.,q_true,0.,n1,n2,np)
    d_s=_screen_obliquity_delta(sag,gx,gy,0.,0.,q_ship,0.,n1,n2,np)
    dz,_=_facet_axial_momenta(np.full_like(sag,q_true),np.zeros_like(sag),gx,gy,n1,n2,np)
    dz0,_=_facet_axial_momenta(np.zeros_like(sag),np.zeros_like(sag),gx,gy,n1,n2,np)
    tgt=(dz-dz0)*sag
    m=(X**2+Y**2)<=(0.9*N*dx/2)**2
    r_t=float(np.sqrt(np.mean((d_t-tgt)[m]**2)))/lam
    r_s=float(np.sqrt(np.mean((d_s-tgt)[m]**2)))/lam
    print(f" {th:5.3f}  | rms {float(np.sqrt(np.mean(d_t[m]**2)))/lam:9.5f} wv "
          f"| rms {float(np.sqrt(np.mean(d_s[m]**2)))/lam:9.5f} wv "
          f"| {float(np.sqrt(np.mean(d_s[m]**2))/max(np.sqrt(np.mean(d_t[m]**2)),1e-30)):5.2f} "
          f"| err_true {r_t:.2e} wv  err_shipped {r_s:.4f} wv")

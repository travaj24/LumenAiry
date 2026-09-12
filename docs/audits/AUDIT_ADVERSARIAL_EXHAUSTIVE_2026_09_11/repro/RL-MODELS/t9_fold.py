"""B: is the tangent_facet_remap fold / pull-back guard driven by pixels that
carry NO energy?  A converging beam normally needs a PADDED grid."""
import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import (apply_real_lens,
    _tangent_facet_remap_screen, _TF_REMAP_MIN_DET)
lam = 0.55e-6
ap = 2.0e-3
print(" pad  N      dx[um] window[mm] corner[mm]  result")
for N, dx in [(1024,4e-6), (2048,4e-6), (4096,4e-6), (1024,8e-6), (2048,8e-6)]:
    ax=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(ax,ax)
    E=np.exp(-(X**2+Y**2)/(ap/3)**2).astype(np.complex128)
    rx=dict(surfaces=[dict(radius=+19.6e-3, glass_before='AIR', glass_after='N-BK7'),
                      dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='AIR')],
            thicknesses=[2.5e-3], aperture_diameter=ap)
    corner=0.5*np.hypot(N*dx,N*dx)
    try:
        R=apply_real_lens(E.copy(),prescription=rx,wavelength=lam,dx=dx,
                          surface_model='tangent_facet_remap')
        msg="OK"
    except ValueError as e:
        msg="REFUSED: "+str(e).split('REFUSES')[1][:110]
    print(f" {N*dx/ap:4.1f} {N:5d} {dx*1e6:7.1f} {N*dx*1e3:9.2f} {corner*1e3:9.2f}  {msg}")

print()
print("min det over WHOLE grid vs over the ILLUMINATED PUPIL (surface 1 of the")
print("same singlet, N=1024, various dx) -- the guard uses the whole-grid min:")
from lumenairy.elements.lenses import surface_sag_general as sg
for N, dx in [(1024,4e-6),(1024,8e-6),(1024,16e-6),(1024,24e-6)]:
    ax=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(ax,ax)
    R2=-27.4e-3
    sag=np.nan_to_num(sg(X**2+Y**2, R2, 0.0, None))
    gy,gx=np.gradient(sag,dx,dx)
    hxy,hxx=np.gradient(gx,dx,dx); hyy,hyx=np.gradient(gy,dx,dx)
    px=np.zeros_like(sag); py=np.zeros_like(sag)
    opd,wx,wy,pox,poy,ok=_tangent_facet_remap_screen(sag,gx,gy,hxx,hxy,hyx,hyy,
                                                     px,py,1.5168,1.0,np)
    wxy,wxx=np.gradient(wx,dx,dx); wyy,wyx=np.gradient(wy,dx,dx)
    det=(1+wxx)*(1+wyy)-wxy*wyx
    pup=(X**2+Y**2)<=(ap/2)**2
    print(f"   N={N} dx={dx*1e6:5.1f}um  min det whole-grid={float(det.min()):10.4f} "
          f"| min det inside pupil={float(det[pup].min()):10.4f}  "
          f"(bar {_TF_REMAP_MIN_DET})")

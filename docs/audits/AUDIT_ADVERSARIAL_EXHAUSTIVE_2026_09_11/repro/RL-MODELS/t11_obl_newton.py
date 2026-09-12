import numpy as np, sys, time
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import (_screen_obliquity_delta,
    _facet_axial_momenta, _build_displaced_cos_grid, _disp_surface_z_grad)
from lumenairy.elements.lenses import surface_sag_general as sg
lam = 0.55e-6

print("== probe 5: on ONE surface, thin + eq(4) == (T1) exactly? ==")
N=257; dx=8e-6
ax=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(ax,ax)
R=19.6e-3; n1,n2=1.0,1.62
sag=np.nan_to_num(sg(X**2+Y**2,R,0.0,None))
gy,gx=np.gradient(sag,dx,dx)
for th in (10e-3, 55e-3, 150e-3):
    qx=n1*np.sin(th); qy=0.0
    d=_screen_obliquity_delta(sag,gx,gy,0.0,0.0,qx,qy,n1,n2,np)
    corrected=(n2-n1)*sag+d
    dz,_=_facet_axial_momenta(np.full_like(sag,qx),np.zeros_like(sag),gx,gy,n1,n2,np)
    t1=dz*sag
    print(f"   theta={th*1e3:6.1f} mrad: max|thin+eq4 - (T1)| = "
          f"{float(np.max(np.abs(corrected-t1))):.3e} m  "
          f"(sag scale {float(np.max(np.abs(sag))):.3e} m)")

print()
print("== probe 11: Newton iteration count actually needed in the pointwise trace ==")
s=[dict(radius=+19.6e-3, glass_before='AIR', glass_after='N-BK7', decenter=(1e-4,0.0)),
   dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='AIR', decenter=(1e-4,0.0))]
# replicate the Newton loop and report the residual per iteration
nl=257; r_max=1.5e-3
axl=np.linspace(-r_max,r_max,nl); LX,LY=np.meshgrid(axl,axl)
px=LX.ravel().copy(); py=LY.ravel().copy(); pz=np.zeros(px.size)
dxr=np.zeros(px.size); dyr=np.zeros(px.size); dzr=np.ones(px.size)
t=(0.0-pz)/dzr
for k in range(24):
    xq=px+t*dxr; yq=py+t*dyr
    f,fx,fy=_disp_surface_z_grad(s[0],xq,yq)
    f=np.nan_to_num(f); fx=np.nan_to_num(fx); fy=np.nan_to_num(fy)
    g=pz+t*dzr-0.0-f
    dg=dzr-(fx*dxr+fy*dyr)
    t=t-g/np.where(np.abs(dg)<1e-30,1e-30,dg)
    if k<10 or k==23:
        print(f"   iter {k+1:2d}: max|residual| = {float(np.max(np.abs(g))):.3e} m")

t0=time.perf_counter()
grids=_build_displaced_cos_grid(s,[2.5e-3],lam,r_max,1024,1024,4e-6,4e-6)
print(f"   _build_displaced_cos_grid(n_launch=257, N=1024): {time.perf_counter()-t0:.3f} s")
t0=time.perf_counter()
grids=_build_displaced_cos_grid(s,[2.5e-3],lam,r_max,1024,1024,4e-6,4e-6,
                                interp_method='delaunay')
print(f"   ... interp_method='delaunay'                    : {time.perf_counter()-t0:.3f} s")

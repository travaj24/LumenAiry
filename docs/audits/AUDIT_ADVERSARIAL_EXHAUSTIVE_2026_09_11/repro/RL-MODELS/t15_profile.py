"""Where does tangent_facet_remap's wall clock go?"""
import numpy as np, sys, time
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy.elements._lens_real as LR
from lumenairy.elements._lens_real import apply_real_lens
from scipy.ndimage import map_coordinates, spline_filter
lam=0.55e-6
N=1024; dx=4e-6; ap=2.0e-3
ax=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(ax,ax)
E0=np.exp(-(X**2+Y**2)/(ap/3)**2).astype(np.complex128)
rx=dict(surfaces=[dict(radius=+12.6e-3, glass_before='AIR', glass_after='N-SSK2'),
                  dict(radius=-12.6e-3, glass_before='N-SSK2', glass_after='AIR')],
        thicknesses=[3.0e-3], aperture_diameter=ap)

# count the pull-back iterations actually used, and time the pieces
orig = LR._tangent_facet_remap_apply
stats = []
def patched(E, wx, wy, pox, poy, dxx, dyy, k0, order, xp, si):
    t0=time.perf_counter()
    # count iterations by replicating just the pull-back
    ny,nx = E.shape
    sx = wx/dxx; sy = wy/dyy
    if order>1:
        sx=spline_filter(sx,order=order,output=np.float64)
        sy=spline_filter(sy,order=order,output=np.float64)
    iu=np.arange(nx,dtype=np.float64)[None,:]+np.zeros((ny,1))
    iv=np.arange(ny,dtype=np.float64)[:,None]+np.zeros((1,nx))
    ix=iu.copy(); iy=iv.copy(); nit=0
    tpb=time.perf_counter()
    for _ in range(64):
        nit+=1
        crd=np.stack([iy.ravel(),ix.ravel()])
        nix=iu-map_coordinates(sx,crd,order=order,mode='nearest',prefilter=False).reshape(ny,nx)
        niy=iv-map_coordinates(sy,crd,order=order,mode='nearest',prefilter=False).reshape(ny,nx)
        step=max(float(np.max(np.abs(nix-ix))),float(np.max(np.abs(niy-iy))))
        ix,iy=nix,niy
        if step<1e-9: break
    tpb=time.perf_counter()-tpb
    r = orig(E, wx, wy, pox, poy, dxx, dyy, k0, order, xp, si)
    tot=time.perf_counter()-t0
    stats.append((si, nit, tpb, tot))
    return r
LR._tangent_facet_remap_apply = patched
t0=time.perf_counter()
R=apply_real_lens(E0.copy(),prescription=rx,wavelength=lam,dx=dx,sag_chunk_rows=0,
                  surface_model='tangent_facet_remap')
tt=time.perf_counter()-t0
LR._tangent_facet_remap_apply = orig
print(f"total call {tt:.3f}s  (includes my duplicated pull-back instrumentation)")
for si,nit,tpb,tot in stats:
    print(f"  surface {si}: pull-back converged in {nit} iterations; "
          f"pull-back alone {tpb:.3f}s; remap_apply {tot-tpb:.3f}s (real)")
# order comparison
for o in (1,3,5):
    t0=time.perf_counter()
    apply_real_lens(E0.copy(),prescription=rx,wavelength=lam,dx=dx,sag_chunk_rows=0,
                    surface_model='tangent_facet_remap', remap_order=o)
    print(f"  remap_order={o}: {time.perf_counter()-t0:.3f}s")

"""Probe 2 + 10: exit WAVEFRONT of every model vs an EXACT meridional ray
trace, on a fast plano-convex singlet in BOTH orientations (the 4x spherical-
aberration orientation split the paraxial screen cannot see)."""
import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.glass import get_glass_index
from lumenairy.elements.lenses import surface_sag_general as sg

lam = 0.55e-6; k0 = 2*np.pi/lam
n_g = get_glass_index('N-BK7', lam)
Rc = 12.0e-3                 # f = R/(n-1) = 23.2 mm ; ap 6 mm -> f/3.9
ap = 6.0e-3; t = 4.0e-3
CURVED_FIRST = [dict(radius=+Rc, glass_before='AIR', glass_after='N-BK7'),
                dict(radius=np.inf, glass_before='N-BK7', glass_after='AIR')]
FLAT_FIRST   = [dict(radius=np.inf, glass_before='AIR', glass_after='N-BK7'),
                dict(radius=-Rc, glass_before='N-BK7', glass_after='AIR')]

def exact_radial(surfaces, thick, hmax, nf=6001):
    h=np.linspace(1e-9, hmax, nf)
    pz=np.zeros(nf); py=h.copy(); dz=np.ones(nf); dy=np.zeros(nf)
    opl=np.zeros(nf); z_v=0.0
    for i,s in enumerate(surfaces):
        n1=get_glass_index(s['glass_before'],lam); n2=get_glass_index(s['glass_after'],lam)
        R=s['radius']
        if not np.isfinite(R):
            tt=(z_v-pz)/dz; pz=pz+tt*dz; py=py+tt*dy; nz=np.ones(nf); ny=np.zeros(nf)
        else:
            tt=(z_v-pz)/dz
            for _ in range(50):
                y=py+tt*dy; r=np.abs(y)
                sagv=np.nan_to_num(sg(r*r,R,0.0,None))
                dsag=r/(R*np.sqrt(np.maximum(1-r*r/R**2,1e-300)))
                g=pz+tt*dz-z_v-sagv
                tt=tt-g/(dz-dsag*np.sign(y)*dy)
            pz=pz+tt*dz; py=py+tt*dy; r=np.abs(py)
            dsag=r/(R*np.sqrt(np.maximum(1-r*r/R**2,1e-300)))
            nz=np.ones(nf); ny=-dsag*np.sign(py)
            nn=np.hypot(nz,ny); nz,ny=nz/nn,ny/nn
        opl=opl+n1*tt
        ci=dz*nz+dy*ny; eta=n1/n2
        ct=np.sqrt(np.maximum(1-eta*eta*(1-ci*ci),0))
        ndz=eta*dz+(ct-eta*ci)*nz; ndy=eta*dy+(ct-eta*ci)*ny
        nn2=np.hypot(ndz,ndy); dz,dy=ndz/nn2,ndy/nn2
        if i<len(surfaces)-1: z_v+=thick[i]
    z_ex=sum(thick); tf=(z_ex-pz)/dz
    n_last=get_glass_index(surfaces[-1]['glass_after'],lam)
    opl=opl+n_last*tf
    return h, py+tf*dy, opl

N=2048; dx=4.0e-6
ax=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(ax,ax)
E0=np.exp(-(X*X+Y*Y)/(ap/2.2)**2).astype(np.complex128)
row=N//2
rr = np.abs(ax)

def model_wavefront(surfaces, kw):
    rx=dict(surfaces=surfaces, thicknesses=[t], aperture_diameter=ap)
    E=apply_real_lens(E0.copy(),prescription=rx,wavelength=lam,dx=dx,
                      sag_chunk_rows=0, **kw)
    ph=np.unwrap(np.angle(E[row]))
    return ph/k0

def score(S, S_ex, mask):
    A=np.stack([np.ones(mask.sum()), ax[mask]],axis=1)
    d=S[mask]-S_ex[mask]
    c,*_=np.linalg.lstsq(A,d,rcond=None)
    return float(np.sqrt(np.mean((d-A@c)**2))/lam)

print(f"plano-convex R={Rc*1e3}mm t={t*1e3}mm ap={ap*1e3}mm n={n_g:.4f}  N={N} dx={dx*1e6}um")
print(f"{'orientation':16s} {'model':22s} {'exit wavefront rms err [waves]':>32s}")
for name, surfaces in (('curved-first', CURVED_FIRST), ('flat-first', FLAT_FIRST)):
    h, ho, opl = exact_radial(surfaces, [t], ap/2)
    o = np.argsort(ho)
    S_ex = np.interp(rr, ho[o], opl[o], left=np.nan, right=np.nan)
    mask = (rr <= 0.85*ho.max()) & np.isfinite(S_ex)
    for mn, kw in [('thin', {}),
                   ('displaced', dict(surface_model='displaced')),
                   ('tangent_facet', dict(surface_model='tangent_facet')),
                   ('tangent_facet_remap', dict(surface_model='tangent_facet_remap'))]:
        try:
            S = model_wavefront(surfaces, kw)
            print(f"{name:16s} {mn:22s} {score(S,S_ex,mask):32.5f}")
        except Exception as e:
            print(f"{name:16s} {mn:22s}  RAISED {str(e)[:60]}")

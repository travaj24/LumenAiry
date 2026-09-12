import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy.elements._lens_real as LR
from lumenairy.elements._lens_real import (
    _build_displaced_cos_grid, _build_displaced_ray_map,
    _build_displaced_cos_luts, apply_real_lens, _displaced_opd)
lam = 0.55e-6

print("C2. does _build_displaced_cos_grid actually depend on sag_callable's state?")
class FF:
    def __init__(self, a): self.a = a
    def __call__(self, xs, ys): return self.a*(xs**2+ys**2)
for a in (5.0, -5000.0):
    ff = FF(a)
    s = [dict(radius=np.inf, glass_before='AIR', glass_after='N-BK7',
              sag_callable=ff, decenter=(1e-9, 0.0)),
         dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='AIR',
              decenter=(1e-9, 0.0))]
    g = _build_displaced_cos_grid(s, [2.5e-3], lam, 1.0e-3, 128, 128, 4e-6, 4e-6)
    print(f"   a={a:9.1f}: surf0 cos_out mean={g[0][1].mean():.9f}  "
          f"surf1 cos_in mean={g[1][0].mean():.9f}")

print()
print("C3. same via the PUBLIC api (apply_real_lens, pointwise screen)")
N=256; dx=6e-6; ap=1.2e-3
ax=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(ax,ax)
E=np.exp(-(X**2+Y**2)/(ap/3)**2).astype(np.complex128)
LR.set_pointwise_cos_grid_cache_budget(64)
ff = FF(5.0)
rx = dict(surfaces=[dict(radius=np.inf, glass_before='AIR', glass_after='N-BK7',
                         sag_callable=ff, decenter=(1e-9,0.0)),
                    dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='AIR',
                         decenter=(1e-9,0.0))],
          thicknesses=[2.5e-3], aperture_diameter=ap)
E1 = apply_real_lens(E.copy(), prescription=rx, wavelength=lam, dx=dx,
                     surface_model='displaced', displaced_obliquity='pointwise')
ff.a = -5000.0
E2 = apply_real_lens(E.copy(), prescription=rx, wavelength=lam, dx=dx,
                     surface_model='displaced', displaced_obliquity='pointwise')
LR.clear_pointwise_cos_grid_cache()
E3 = apply_real_lens(E.copy(), prescription=rx, wavelength=lam, dx=dx,
                     surface_model='displaced', displaced_obliquity='pointwise')
m = (X**2+Y**2) <= (0.8*ap/2)**2
print(f"   |E2-E3|max/|E3|max (stale-cache vs cold rebuild, SAME physics) = "
      f"{float(np.max(np.abs(E2-E3)[m])/np.max(np.abs(E3)[m])):.4e}")
print(f"   |E1-E2|max/|E1|max (a=5 vs a=-5000, should be LARGE)          = "
      f"{float(np.max(np.abs(E1-E2)[m])/np.max(np.abs(E1)[m])):.4e}")
LR.set_pointwise_cos_grid_cache_budget(0)

print()
print("D2. _build_displaced_ray_map exit leg: n=1 vs the true glass_after")
def manual_map(surfaces, thick, lam, r_max, n_fan=1025, n_exit=None):
    from lumenairy.glass import get_glass_index
    from lumenairy.elements.lenses import surface_sag_general as sg
    h = np.linspace(r_max/n_fan, r_max, n_fan)
    pz=np.zeros(n_fan); py=h.copy(); dz=np.ones(n_fan); dy=np.zeros(n_fan)
    opl=np.zeros(n_fan); z_v=0.0
    for i,s in enumerate(surfaces):
        n1=float(get_glass_index(s['glass_before'],lam)); n2=float(get_glass_index(s['glass_after'],lam))
        R=s['radius']
        if not np.isfinite(R):
            t=(z_v-pz)/dz; pz=pz+t*dz; py=py+t*dy; nz=np.ones(n_fan); ny=np.zeros(n_fan)
        else:
            t=(z_v-pz)/dz
            for _ in range(60):
                y=py+t*dy; r=np.abs(y)
                sag=np.nan_to_num(sg(r*r,R,0.0,None))
                e=np.maximum(1e-9,1e-6*r)
                sp=np.nan_to_num(sg((r+e)**2,R,0.0,None)); sm=np.nan_to_num(sg((r-e)**2,R,0.0,None))
                sp_=(sp-sm)/(2*e)
                g=pz+t*dz-z_v-sag; dg=dz-sp_*np.sign(y)*dy
                t=t-g/dg
            pz=pz+t*dz; py=py+t*dy; r=np.abs(py)
            e=np.maximum(1e-9,1e-6*r)
            sp=np.nan_to_num(sg((r+e)**2,R,0.0,None)); sm=np.nan_to_num(sg((r-e)**2,R,0.0,None))
            sp_=(sp-sm)/(2*e); nz=np.ones(n_fan); ny=-sp_*np.sign(py)
            nn=np.hypot(nz,ny); nz,ny=nz/nn,ny/nn
        opl=opl+n1*t
        ci=dz*nz+dy*ny; eta=n1/n2
        ct=np.sqrt(np.maximum(1-eta*eta*(1-ci*ci),0))
        ndz=eta*dz+(ct-eta*ci)*nz; ndy=eta*dy+(ct-eta*ci)*ny
        nn2=np.hypot(ndz,ndy); dz,dy=ndz/nn2,ndy/nn2
        if i<len(surfaces)-1: z_v+=thick[i]
    z_exit=sum(thick); t_f=(z_exit-pz)/dz
    n_last = n_exit if n_exit is not None else float(get_glass_index(surfaces[-1]['glass_after'],lam))
    opl=opl+n_last*t_f
    return h, py+t_f*dy, opl
sB = [dict(radius=+19.6e-3, glass_before='AIR', glass_after='N-BK7'),
      dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='N-BK7')]
h, ho, opl_model = _build_displaced_ray_map(sB, [2.5e-3], lam, 1.0e-3)
h2, ho2, opl_true = manual_map(sB, [2.5e-3], lam, 1.0e-3)
h3, ho3, opl_n1   = manual_map(sB, [2.5e-3], lam, 1.0e-3, n_exit=1.0)
d_true = (opl_model-opl_model[0]) - (opl_true-opl_true[0])
d_n1   = (opl_model-opl_model[0]) - (opl_n1  -opl_n1[0])
print(f"   model OPL vs manual with n_exit=n(glass_after)=1.5168 : max|d|={np.max(np.abs(d_true))*1e6:9.4f} um = {np.max(np.abs(d_true))/lam:8.2f} waves")
print(f"   model OPL vs manual with n_exit=1.0                   : max|d|={np.max(np.abs(d_n1))*1e6:9.4e} um")

print()
print("E2. LUT crossing-height fold for a strongly converging conjugate")
sA = [dict(radius=+19.6e-3, glass_before='AIR', glass_after='N-BK7'),
      dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='AIR')]
for conj in (-0.0030, -0.0020, -0.0012):
    slope = lambda hh, s=conj: np.asarray(hh)/s
    luts = _build_displaced_cos_luts(sA, [2.5e-3], lam, 1.0e-3, carrier_slope=slope)
    for i,(hh,ci,co) in enumerate(luts):
        d=np.diff(hh); nd=int(np.sum(d<=0))
        # how badly does interp mis-read?  check cos_in monotone consistency
        print(f"   conj={conj}: surf{i}: n={hh.size} non-increasing={nd} "
              f"h=[{hh.min()*1e6:.2f},{hh.max()*1e6:.2f}]um  cos_in ptp={float(np.ptp(ci)):.4f}")

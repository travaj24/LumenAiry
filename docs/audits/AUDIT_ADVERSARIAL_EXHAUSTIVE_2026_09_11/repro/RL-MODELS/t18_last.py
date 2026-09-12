import numpy as np, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import (apply_real_lens,
    _build_displaced_ray_map, _apply_displaced_remap)
lam=0.55e-6
print("== A: 2-D remap transverse resolution vs the hard-coded n_side=181 fan ==")
N=1024; dx=2e-6; ap=2.0e-3
ax=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(ax,ax); Rr=np.hypot(X,Y)
rx=dict(surfaces=[dict(radius=+19.6e-3, glass_before='AIR', glass_after='N-BK7', decenter=(1e-9,0.)),
                  dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='AIR', decenter=(1e-9,0.))],
        thicknesses=[2.5e-3], aperture_diameter=ap)
print(f"   launch pitch = {2*1.03*(ap/2)/180*1e6:.2f} um, grid dx = {dx*1e6:.1f} um")
for Lam in (200e-6, 60e-6, 25e-6):
    E0=(np.exp(-(X*X+Y*Y)/(ap/3)**2)*(1.0+0.5*np.cos(2*np.pi*Rr/Lam))).astype(np.complex128)
    out={}
    for lab,kw in [('screen','pointwise'),('remap',None)]:
        kk=dict(surface_model='displaced')
        if kw: kk['displaced_obliquity']=kw
        E=apply_real_lens(E0.copy(),prescription=rx,wavelength=lam,dx=dx,sag_chunk_rows=0,**kk)
        row=np.abs(E[N//2]); m=np.abs(ax)<=0.7*ap/2
        env=np.convolve(row,np.ones(41)/41,mode='same')
        out[lab]=float(np.max(np.abs(row[m]-env[m]))/max(np.max(env[m]),1e-30))
    print(f"   ripple {Lam*1e6:6.1f} um ({Lam/(2*1.03*(ap/2)/180):4.1f} launch samples/period): "
          f"screen contrast {out['screen']:.4f}  remap contrast {out['remap']:.4f}")

print()
print("== B: 1-D displaced remap silently DROPS non-monotone rays (no warning) ==")
sA=[dict(radius=+19.6e-3, glass_before='AIR', glass_after='N-BK7'),
    dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='AIR')]
for conj in (None, -0.0030, -0.0015):
    slope = None if conj is None else (lambda h,s=conj: np.asarray(h)/s)
    eik   = None if conj is None else (lambda h,s=conj: np.asarray(h)**2/(2*s))
    h,ho,opl=_build_displaced_ray_map(sA,[2.5e-3],lam,1.0e-3,
                                      carrier_slope=slope, eikonal_fn=eik)
    o=np.argsort(ho); hs=ho[o]
    kept=int(np.sum(np.concatenate(([True], np.diff(hs)>0))))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        E=_apply_displaced_remap(np.ones((64,64),dtype=np.complex128), h, ho, lam,
                                 4e-6, 4e-6, opl)
    print(f"   conjugate={conj}: fan traced {h.size} rays, {kept} kept after the "
          f"strictly-increasing filter ({h.size-kept} DROPPED), warnings={len(w)}")

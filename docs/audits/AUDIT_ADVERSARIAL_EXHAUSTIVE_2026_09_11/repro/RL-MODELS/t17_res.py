"""Probe 4/7: the 2-D displaced remap reconstructs the WHOLE exit field from a
HARD-CODED 181x181 scattered ray set (n_side=181), so its transverse
resolution is independent of N."""
import numpy as np, sys, time
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import apply_real_lens
lam=0.55e-6
N=1024; dx=4e-6; ap=2.0e-3
ax=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(ax,ax); Rr=np.sqrt(X*X+Y*Y)
Lam=60e-6
E0=(np.exp(-(X*X+Y*Y)/(ap/3)**2)*(1.0+0.5*np.cos(2*np.pi*Rr/Lam))).astype(np.complex128)
rx=dict(surfaces=[dict(radius=+19.6e-3, glass_before='AIR', glass_after='N-BK7', decenter=(1e-9,0.)),
                  dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='AIR', decenter=(1e-9,0.))],
        thicknesses=[2.5e-3], aperture_diameter=ap)
print(f"input amplitude ripple period {Lam*1e6:.0f} um ; grid dx {dx*1e6:.0f} um ;"
      f" 2-D remap launch pitch = {2*1.03*(ap/2)/180*1e6:.2f} um")
for label, kw in [('thin', {}),
                  ('pointwise SCREEN', dict(surface_model='displaced',
                                            displaced_obliquity='pointwise')),
                  ('2-D remap (DEFAULT for asym)', dict(surface_model='displaced'))]:
    t0=time.perf_counter()
    E=apply_real_lens(E0.copy(),prescription=rx,wavelength=lam,dx=dx,sag_chunk_rows=0,**kw)
    t=time.perf_counter()-t0
    # ripple contrast of |E| on the central row inside the pupil
    row=np.abs(E[N//2]); m=np.abs(ax)<=0.8*ap/2
    env=np.convolve(row, np.ones(31)/31, mode='same')
    c=float(np.max(np.abs(row[m]-env[m]))/max(np.max(env[m]),1e-30))
    print(f"  {label:30s} t={t:6.2f}s  residual ripple contrast = {c:.4f}")
print()
print("float32 sag helper: does the DEFAULT call ever trip on_partial_aperture?")
from lumenairy.elements._lens_real import lens_sag_float32_opd_error
rx2=dict(surfaces=[dict(radius=+19.6e-3, glass_before='AIR', glass_after='N-BK7'),
                   dict(radius=-27.4e-3, glass_before='N-BK7', glass_after='AIR')],
         thicknesses=[2.5e-3], aperture_diameter=ap)
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    r=lens_sag_float32_opd_error(rx2, lam)
    print(f"  default call: aperture_cover={r['aperture_cover']:.3f} "
          f"covers={r['field_check_covers_aperture']} warnings={len(w)} "
          f"ok={r['ok']} field_check_dx={r['field_check_dx']*1e6:.3f} um")
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter('always')
    r=lens_sag_float32_opd_error(rx2, lam, field_check_dx=0.9e-6)
    print(f"  production dx: aperture_cover={r['aperture_cover']:.3f} "
          f"covers={r['field_check_covers_aperture']} warnings={len(w)} ok={r['ok']}")
# does the global sag dtype leak on exception?
import lumenairy.elements._lens_real as LR
LR.set_lens_sag_dtype(None)
before=LR.get_lens_sag_dtype()
try:
    lens_sag_float32_opd_error(dict(surfaces=[], thicknesses=[]), lam, aperture=1e-3,
                               field_check_n=0)
except Exception as e:
    pass
print(f"  global sag dtype before={before} after={LR.get_lens_sag_dtype()} (no leak expected)")

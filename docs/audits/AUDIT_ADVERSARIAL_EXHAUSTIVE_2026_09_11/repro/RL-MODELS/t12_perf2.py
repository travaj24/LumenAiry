"""Careful perf/memory: warm every path at the SAME N first so FFT plans /
module imports are not charged to the first model measured."""
import numpy as np, sys, time, tracemalloc, gc
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.elements._lens_traced import TiltedCarrier
lam=0.55e-6
rx = dict(surfaces=[dict(radius=+12.6e-3, glass_before='AIR', glass_after='N-SSK2'),
                    dict(radius=-12.6e-3, glass_before='N-SSK2', glass_after='AIR')],
          thicknesses=[3.0e-3], aperture_diameter=2.0e-3)
CASES = [('thin (baseline)', {}),
         ('thin + collimated carrier', dict(carrier=TiltedCarrier(L=0.05, M=0.0, R=np.inf),
                                            on_screen_obliquity='silent')),
         ('thin + finite-R carrier', dict(carrier=TiltedCarrier(L=0.05, M=0.0, R=0.5),
                                          on_screen_obliquity='silent')),
         ('displaced (meridional)', dict(surface_model='displaced')),
         ('tangent_facet', dict(surface_model='tangent_facet')),
         ('tangent_facet_remap', dict(surface_model='tangent_facet_remap'))]
for N, dx in [(512, 8e-6), (1024, 4e-6), (2048, 2e-6)]:
    ax=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(ax,ax)
    E0=np.exp(-(X**2+Y**2)/(2.0e-3/3)**2).astype(np.complex128)
    # WARM every path at this exact N
    for _, kw in CASES:
        try: apply_real_lens(E0.copy(), prescription=rx, wavelength=lam, dx=dx,
                             sag_chunk_rows=0, **kw)
        except Exception: pass
    g = 8.0*N*N
    print(f"N={N} dx={dx*1e6:.0f}um   (1 float64 grid = {g/2**20:.1f} MiB)")
    base_t = base_m = None
    for name, kw in CASES:
        gc.collect(); tracemalloc.start(); t0=time.perf_counter()
        R = apply_real_lens(E0.copy(), prescription=rx, wavelength=lam, dx=dx,
                            sag_chunk_rows=0, **kw)
        t=time.perf_counter()-t0
        _,peak=tracemalloc.get_traced_memory(); tracemalloc.stop(); del R; gc.collect()
        if base_t is None: base_t, base_m = t, peak
        print(f"   {name:28s} t={t:7.3f}s ({t/base_t:5.2f}x)  peak={peak/g:6.2f} grids "
              f"(+{(peak-base_m)/g:6.2f})")

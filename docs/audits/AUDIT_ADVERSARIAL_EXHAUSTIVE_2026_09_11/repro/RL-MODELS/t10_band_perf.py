"""Probe 6 (halo byte-identity) + 11 (perf/memory)."""
import numpy as np, sys, time, tracemalloc, gc
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.elements._lens_traced import TiltedCarrier
lam = 0.55e-6

def make(N, dx, ap):
    ax=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(ax,ax)
    E=np.exp(-(X**2+Y**2)/(ap/3)**2).astype(np.complex128)
    return E
rx3 = dict(surfaces=[
    dict(radius=+19.6e-3, glass_before='AIR',    glass_after='N-BK7'),
    dict(radius=-27.4e-3, glass_before='N-BK7',  glass_after='N-SF11'),
    dict(radius=-60.0e-3, glass_before='N-SF11', glass_after='AIR')],
    thicknesses=[2.0e-3, 1.5e-3], aperture_diameter=2.0e-3)

print("== BYTE-IDENTITY: banded (explicit sag_chunk_rows) vs whole-grid ==")
N, dx, ap = 1024, 4e-6, 2.0e-3
E0 = make(N, dx, ap)
for model in ('thin', 'tangent_facet', 'tangent_facet_remap'):
    for carrier in (None, TiltedCarrier(L=0.03, M=0.0, R=np.inf) if 'Tilted' else None):
        kw = dict(surface_model=model)
        if carrier is not None and model == 'thin':
            kw['carrier'] = carrier; kw['on_screen_obliquity']='silent'
        elif carrier is not None:
            continue
        try:
            Ew = apply_real_lens(E0.copy(), prescription=rx3, wavelength=lam,
                                 dx=dx, sag_chunk_rows=0, **kw)
            outs=[]
            for cr in (37, 128, 256):
                Eb = apply_real_lens(E0.copy(), prescription=rx3, wavelength=lam,
                                     dx=dx, sag_chunk_rows=cr, **kw)
                outs.append((cr, np.array_equal(Ew.view(np.float64), Eb.view(np.float64)),
                             float(np.max(np.abs(Ew-Eb)))))
            tag = f"{model}{' +carrier' if carrier is not None else ''}"
            print(f"  {tag:28s} " + "  ".join(
                f"band={c}: identical={i} maxdiff={d:.3e}" for c,i,d in outs))
        except Exception as e:
            print(f"  {model}: RAISED {type(e).__name__}: {str(e)[:100]}")

print()
print("== PERF / MEMORY (3-surface element, whole-grid path) ==")
for N, dx in [(1024, 4e-6), (2048, 2e-6)]:
    E0 = make(N, dx, ap)
    base_bytes = N*N*8          # one float64 grid
    print(f"  N={N}:  1 float64 grid = {base_bytes/2**20:.1f} MiB")
    for model, kw in [('thin', {}),
                      ('displaced', dict(surface_model='displaced')),
                      ('tangent_facet', dict(surface_model='tangent_facet')),
                      ('tangent_facet_remap', dict(surface_model='tangent_facet_remap'))]:
        # warm
        try:
            apply_real_lens(E0[:64,:64].copy(), prescription=rx3, wavelength=lam, dx=dx, **kw)
        except Exception: pass
        gc.collect(); tracemalloc.start()
        t0=time.perf_counter()
        try:
            R = apply_real_lens(E0.copy(), prescription=rx3, wavelength=lam, dx=dx,
                                sag_chunk_rows=0, **kw)
        except Exception as e:
            tracemalloc.stop(); print(f"    {model:22s} RAISED {str(e)[:70]}"); continue
        t=time.perf_counter()-t0
        cur,peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        del R; gc.collect()
        print(f"    {model:22s} t={t:7.3f}s  peak={peak/2**20:8.1f} MiB = "
              f"{peak/base_bytes:6.2f} float64 grids")

import sys, warnings, time
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.elements._lens_real import apply_real_lens
lam = 632.8e-9
N, dx = 256, 20e-6
x = (np.arange(N) - N/2)*dx
X, Y = np.meshgrid(x, x)
E = np.ones((N, N), dtype=np.complex128)

def rx(stop=None, ap=3e-3):
    r = dict(surfaces=[dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'),
                       dict(radius=-50e-3, glass_before='N-BK7', glass_after='AIR')],
             thicknesses=[3e-3], aperture_diameter=ap)
    if stop is not None: r['stop_index'] = stop
    return r

print("=== 7. stop_index handling ===")
for stop in (None, 0, 1, 2, 5, -1):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        Eo = apply_real_lens(E.copy(), prescription=rx(stop), wavelength=lam, dx=dx)
    frac = float(np.count_nonzero(np.abs(Eo) > 1e-12))/Eo.size
    exp = float(np.count_nonzero(X**2+Y**2 <= (1.5e-3)**2))/Eo.size
    print(f"  stop_index={str(stop):<5} non-zero fraction {frac:.4f} "
          f"(aperture would give {exp:.4f})  warnings={len(w)}")

print()
print("=== 7b. aperture edge: hard binary mask? ===")
Eo = apply_real_lens(E.copy(), prescription=rx(), wavelength=lam, dx=dx)
a = np.abs(Eo[N//2])
i = np.argmax(np.abs(x) > 1.5e-3)
print(f"  |E| across the rim (x={x[i-3]*1e3:.3f}..{x[i+2]*1e3:.3f} mm): "
      + ' '.join(f'{v:.4f}' for v in a[i-3:i+3]))
print("  (a hard 0/1 mask + ASM ringing; no apodisation / sub-pixel edge)")

print()
print("=== 17b. cost of a FLAT surface on the default path ===")
def timeit(r, N=1024):
    dxl = 4e-3/(0.8*N)
    El = np.ones((N, N), dtype=np.complex128)
    apply_real_lens(El.copy(), prescription=r, wavelength=lam, dx=dxl, sag_chunk_rows=0)
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        apply_real_lens(El.copy(), prescription=r, wavelength=lam, dx=dxl, sag_chunk_rows=0)
        ts.append(time.perf_counter()-t0)
    return min(ts)
r_curved = dict(surfaces=[dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'),
                          dict(radius=-50e-3, glass_before='N-BK7', glass_after='AIR')],
                thicknesses=[3e-3])
r_flat = dict(surfaces=[dict(radius=float('inf'), glass_before='AIR', glass_after='N-BK7'),
                        dict(radius=float('inf'), glass_before='N-BK7', glass_after='AIR')],
              thicknesses=[3e-3])
print(f"  2 curved surfaces (N=1024): {timeit(r_curved)*1e3:8.2f} ms")
print(f"  2 FLAT   surfaces (N=1024): {timeit(r_flat)*1e3:8.2f} ms  "
      f"<- the two exp(-i*0) screens are pure waste")

print()
print("=== 17c. per-call fixed overhead breakdown at N=64 ===")
import cProfile, pstats, io
rc = dict(surfaces=[dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'),
                    dict(radius=-40e-3, glass_before='N-BK7', glass_after='N-SF11'),
                    dict(radius=-200e-3, glass_before='N-SF11', glass_after='AIR')],
          thicknesses=[3e-3, 2e-3])
Et = np.ones((64, 64), dtype=np.complex128)
apply_real_lens(Et.copy(), prescription=rc, wavelength=lam, dx=1e-5)
pr = cProfile.Profile(); pr.enable()
for _ in range(100):
    apply_real_lens(Et.copy(), prescription=rc, wavelength=lam, dx=1e-5)
pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(14)
print(s.getvalue()[:2600])

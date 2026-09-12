import sys, warnings, time
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.elements._lens_real import apply_real_lens
lam = 632.8e-9
N, dx = 256, 20e-6
E = np.ones((N, N), dtype=np.complex128)
def rx(stop=None, ap=3e-3):
    r = dict(surfaces=[dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'),
                       dict(radius=-50e-3, glass_before='N-BK7', glass_after='AIR')],
             thicknesses=[3e-3], aperture_diameter=ap)
    if stop is not None: r['stop_index'] = stop
    return r
P0 = float(np.sum(np.abs(E)**2))
print("=== stop_index: transmitted POWER fraction (aperture 3 mm on a 5.12 mm grid) ===")
for stop in (None, 0, 1, 2, 5, -1, -2):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        Eo = apply_real_lens(E.copy(), prescription=rx(stop), wavelength=lam, dx=dx)
    print(f"  stop_index={str(stop):<5} P/P0 = {float(np.sum(np.abs(Eo)**2))/P0:.5f}"
          f"   warnings={[str(x.message)[:40] for x in w]}")
print("  (no aperture at all -> 1.0;  3 mm stop -> ~0.27)")

print()
print("=== flat-surface screen waste, N=2048 complex128 ===")
a = np.ones((2048, 2048), dtype=np.complex128)
opd = np.zeros((2048, 2048))
t0=time.perf_counter(); ph = np.exp(-1j*2*np.pi/lam*opd); t1=time.perf_counter()
b = a*ph; t2=time.perf_counter()
print(f"  np.exp(-1j*k0*0) on a 2048^2 grid: {(t1-t0)*1e3:7.2f} ms")
print(f"  E * ones                        : {(t2-t1)*1e3:7.2f} ms")
print(f"  -> per FLAT surface the default path burns {(t2-t0)*1e3:.1f} ms "
      f"+ {2*a.nbytes/1e6:.0f} MB of temporaries for an identity operation")

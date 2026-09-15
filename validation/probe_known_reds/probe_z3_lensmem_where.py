import gc, sys, warnings, tracemalloc
import numpy as np
import lumenairy as la
N, dt = 512, np.complex128
wl, dx = 633e-9, 30e-3 / N
rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7', aperture=25e-3)
E, _, _ = la.create_gaussian_beam(N, dx, wl, w0=5e-3, dtype=dt)
E = np.ascontiguousarray(E)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    la.apply_real_lens(E[:64, :64].copy(), prescription=rx, wavelength=wl, dx=dx)
gc.collect()
tracemalloc.start(12)
tracemalloc.reset_peak()
before = tracemalloc.take_snapshot()
res = la.apply_real_lens(E, prescription=rx, wavelength=wl, dx=dx)
after = tracemalloc.take_snapshot()
cur, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(f"platform={sys.platform} peak={peak/1e6:.1f} MB retained={cur/1e6:.1f} MB")
print("--- RETAINED after the first call (top 12) ---")
for s in after.compare_to(before, 'lineno')[:12]:
    print(f"  {s.size_diff/1e6:8.2f} MB  {s.traceback[0]}")

"""Probe 12: memory + profile at N=1024, ray_subsample=4."""
import warnings, cProfile, pstats, io, tracemalloc, time, numpy as np
import common
common.register_glass()
from lumenairy.elements import apply_real_lens_traced, apply_real_lens
WL = common.WL
N = 1024; AP = 8e-3; dx = 1.3*AP/N
rx = common.plano_convex(R=60e-3, t=4e-3, ap=AP)
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
E_in = np.exp(-(X**2+Y**2)/(2e-3)**2).astype(np.complex128)
kw = dict(prescription=rx, wavelength=WL, dx=dx, ray_subsample=4, n_workers=1,
          on_undersample='silent', min_coarse_samples_per_aperture=0,
          on_pool_memory='silent', parallel_amp=False)
G = 8.0*N*N        # one float64 full grid, bytes
warnings.simplefilter('ignore')
apply_real_lens_traced(E_in, **kw)                 # warm caches

tracemalloc.start()
t0 = time.perf_counter(); E = apply_real_lens_traced(E_in, **kw)
t1 = time.perf_counter()
cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"traced  N={N} sub=4: {t1-t0:.3f} s  peak={peak/1e6:.1f} MB "
      f"= {peak/G:.2f} x (8 N^2)")

tracemalloc.start(); t0 = time.perf_counter()
Ea = apply_real_lens(E_in, prescription=rx, wavelength=WL, dx=dx)
t1 = time.perf_counter()
cur2, peak2 = tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"analytic         : {t1-t0:.3f} s  peak={peak2/1e6:.1f} MB "
      f"= {peak2/G:.2f} x (8 N^2)")

# parallel_amp on
kw2 = dict(kw); kw2['parallel_amp'] = True
tracemalloc.start(); t0 = time.perf_counter()
apply_real_lens_traced(E_in, **kw2); t1 = time.perf_counter()
cur3, peak3 = tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"traced parallel_amp=True: {t1-t0:.3f} s peak={peak3/1e6:.1f} MB "
      f"= {peak3/G:.2f} x (8 N^2)")

pr = cProfile.Profile(); pr.enable()
apply_real_lens_traced(E_in, **kw)
pr.disable()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('tottime').print_stats(14)
print("\n".join(s.getvalue().splitlines()[:26]))

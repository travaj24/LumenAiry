import fixt, numpy as np, warnings, tracemalloc, time, sys
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced import apply_real_lens_traced
p = fixt.small_singlet()
N, dx = 1024, 3e-6
E = fixt.gauss(N, dx, 0.4e-3)
G = 8.0*N*N            # one float64 full grid, bytes
base = dict(prescription=p, wavelength=fixt.WL, dx=dx, ray_subsample=4,
            on_undersample='silent', on_pool_memory='silent',
            on_aperture_beam='silent')
for lbl, extra in [('screen, imap default', {}),
                   ('screen, imap=False', dict(inverse_map=False)),
                   ('ray_density, imap default', dict(amplitude_model='ray_density')),
                   ('ray_density, imap=False', dict(amplitude_model='ray_density', inverse_map=False))]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        apply_real_lens_traced(E, **base, **extra)   # warm caches
        tracemalloc.start()
        t = time.perf_counter()
        o = apply_real_lens_traced(E, **base, **extra)
        dt = time.perf_counter()-t
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    print('%-28s %.3fs  peak=%.1f MB = %.2f full float64 grids (8N^2=%.1f MB)'
          % (lbl, dt, peak/1e6, peak/G, G/1e6))

"""inversion_method='fit' at ray_subsample=1: full-grid Chebyshev design matrix."""
import fixt, numpy as np, warnings, tracemalloc, time
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced import apply_real_lens_traced
from lumenairy.elements._lens_real import apply_real_lens
p = fixt.small_singlet()
for N, dx in ((512, 6e-6), (1024, 3e-6)):
    E = fixt.gauss(N, dx, 0.4e-3)
    G = 8.0*N*N
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tracemalloc.start(); apply_real_lens(E, prescription=p, wavelength=fixt.WL, dx=dx)
        _, pk_a = tracemalloc.get_traced_memory(); tracemalloc.stop()
        for sub in (1, 8):
            tracemalloc.start(); t=time.perf_counter()
            apply_real_lens_traced(E, prescription=p, wavelength=fixt.WL, dx=dx,
                                   inversion_method='fit', ray_subsample=sub,
                                   on_undersample='silent', on_pool_memory='silent',
                                   on_aperture_beam='silent')
            dt=time.perf_counter()-t
            _, pk = tracemalloc.get_traced_memory(); tracemalloc.stop()
            print('N=%5d fit sub=%d  %.2fs  peak %.1f MB = %.2f float64 grids '
                  '(analytic leg alone %.2f grids)'
                  % (N, sub, dt, pk/1e6, pk/G, pk_a/G))

import fixt, numpy as np, warnings, time, cProfile, pstats, io, tracemalloc
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced import apply_real_lens_traced
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.elements import _lens_imap as IM
p = fixt.small_singlet()
N, dx = 1024, 3e-6
E = fixt.gauss(N, dx, 0.4e-3)
G = 8.0*N*N
base = dict(prescription=p, wavelength=fixt.WL, dx=dx, ray_subsample=4,
            on_undersample='silent', on_pool_memory='silent',
            on_aperture_beam='silent')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    tracemalloc.start(); apply_real_lens(E, prescription=p, wavelength=fixt.WL, dx=dx)
    c,pk = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print('apply_real_lens alone: peak %.1f MB = %.2f grids' % (pk/1e6, pk/G))
    # cold vs warm imap
    for lbl, im in (('imap=True', True), ('imap=False', False)):
        IM.inverse_map_cache_clear()
        t=time.perf_counter(); apply_real_lens_traced(E, inverse_map=im, **base); c1=time.perf_counter()-t
        t=time.perf_counter(); apply_real_lens_traced(E, inverse_map=im, **base); w1=time.perf_counter()-t
        print('%-10s cold %.3fs  warm %.3fs' % (lbl, c1, w1))
    pr = cProfile.Profile(); pr.enable()
    apply_real_lens_traced(E, **base); pr.disable()
s = io.StringIO(); ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(22); out = s.getvalue()
print('\n'.join(out.splitlines()[4:32]))

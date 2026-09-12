import warnings, time, numpy as np
import common
common.register_glass()
from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements import _lens_traced as LT
warnings.simplefilter('ignore')
print("cheb backend:", LT._resolved_cheb_backend('polynomial'))
print("numba loaded:", LT._load_numba() is not None)
print("_LENS_PARALLEL_AMP_DEFAULT =", LT._LENS_PARALLEL_AMP_DEFAULT)
WL = common.WL; N = 1024; AP = 8e-3; dx = 1.3*AP/N
rx = common.plano_convex(R=60e-3, t=4e-3, ap=AP)
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
E_in = np.exp(-(X**2+Y**2)/(2e-3)**2).astype(np.complex128)
base = dict(prescription=rx, wavelength=WL, dx=dx, ray_subsample=4, n_workers=1,
            on_undersample='silent', min_coarse_samples_per_aperture=0,
            on_pool_memory='silent')
for par in (False, True):
    ts = []
    for i in range(4):
        t0 = time.perf_counter()
        apply_real_lens_traced(E_in, parallel_amp=par, **base)
        ts.append(time.perf_counter()-t0)
    print(f"parallel_amp={par}: times {['%.3f'%t for t in ts]}  median={np.median(ts):.3f} s")
# cost of the F1/tilt diagnostic
for noncol in ('warn', 'off'):
    ts = []
    for i in range(4):
        t0 = time.perf_counter()
        apply_real_lens_traced(E_in, parallel_amp=False,
                               on_noncollimated=noncol, **base)
        ts.append(time.perf_counter()-t0)
    print(f"on_noncollimated={noncol!r}: median={np.median(ts):.3f} s")

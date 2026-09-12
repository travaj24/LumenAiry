import fixt, numpy as np, warnings, tracemalloc
import lumenairy as la
fixt.register_glass()
from lumenairy.elements import _lens_traced as LT
p = fixt.small_singlet()
N, dx = 512, 6e-6
E = fixt.gauss(N, dx, 0.4e-3)

print('--- segmented with dy != dx (traced itself refuses anamorphic grids) ---')
try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        o = LT.apply_real_lens_traced_segmented(E, prescription=p, wavelength=fixt.WL,
                                                dx=dx, dy=2*dx, on_undersample='silent',
                                                on_pool_memory='silent',
                                                on_aperture_beam='silent')
    print('  segmented(dy=2dx) returned OK, shape', o.shape)
except Exception as e:
    print('  segmented(dy=2dx) ->', type(e).__name__, str(e)[:100])
try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        LT.apply_real_lens_traced(E, prescription=p, wavelength=fixt.WL, dx=dx, dy=2*dx)
except Exception as e:
    print('  direct traced(dy=2dx) ->', type(e).__name__, str(e)[:80])

print('--- self-check upcast: np.asarray(E_in, complex128) on a complex64 field ---')
Ec = E.astype(np.complex64)
tracemalloc.start(); a = np.abs(np.asarray(Ec, dtype=np.complex128))**2
c, pk = tracemalloc.get_traced_memory(); tracemalloc.stop()
print('  shipped expression      peak %.2f MB  (field is %.2f MB)' % (pk/1e6, Ec.nbytes/1e6))
del a
tracemalloc.start(); b = np.abs(Ec).astype(np.float64); b *= b
c2, pk2 = tracemalloc.get_traced_memory(); tracemalloc.stop()
print('  |E|.astype(f8); b*=b    peak %.2f MB' % (pk2/1e6,))
del b
Z = (np.random.randn(N,N)+1j*np.random.randn(N,N))
tracemalloc.start(); s1 = float((np.abs(Z)**2).sum()); c3,pk3 = tracemalloc.get_traced_memory(); tracemalloc.stop()
tracemalloc.start(); s2 = float(np.vdot(Z, Z).real);   c4,pk4 = tracemalloc.get_traced_memory(); tracemalloc.stop()
print('  power sum (np.abs**2).sum() peak %.2f MB  vs vdot peak %.4f MB   (agree: %.3e)'
      % (pk3/1e6, pk4/1e6, abs(s1-s2)/s1))

import fixt, numpy as np, warnings
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced_multibranch import apply_real_lens_traced_multibranch as MB
p = fixt.small_singlet()
N, dx = 512, 4e-6
E = fixt.gauss(N, dx, 0.30e-3)
Pin = float((np.abs(E)**2).sum())
print('P_in', Pin)
for z in [0.0, 1e-3, 5e-3, 10e-3, 15e-3, 20e-3, 22e-3, 24e-3, 24.834e-3, 26e-3, 30e-3]:
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter('always')
        F = MB(E, prescription=p, wavelength=fixt.WL, dx=dx,
               output_plane_distance=z, ray_subsample=4, min_area_ratio=1e-6,
               caustic_band='ludwig', input_carrier=None)
    P = float((np.abs(F)**2).sum())
    nz = int((F != 0).sum())
    print('z=%8.4f mm  P/Pin=%.4e  nonzero=%7d  peak=%.4e   warns=%d %s'
          % (z*1e3, P/Pin, nz, float(np.abs(F).max()), len(W),
             '; '.join(sorted({str(w.message)[:60] for w in W}))))

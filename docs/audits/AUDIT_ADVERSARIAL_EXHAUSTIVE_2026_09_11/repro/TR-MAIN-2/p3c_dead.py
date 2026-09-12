import fixt, numpy as np, warnings
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced_multibranch import apply_real_lens_traced_multibranch as MB
from lumenairy.elements._lens_traced import apply_real_lens_traced
p = fixt.small_singlet()
N, dx = 512, 4e-6
E = fixt.gauss(N, dx, 0.30e-3)
Pin = float((np.abs(E)**2).sum())
print('--- dead-zone width around the paraxial focus (24.8341 mm) ---')
for z in np.arange(24.70, 24.98, 0.02)*1e-3:
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter('always')
        F = MB(E, prescription=p, wavelength=fixt.WL, dx=dx,
               output_plane_distance=float(z), ray_subsample=4,
               min_area_ratio=1e-6, caustic_band='ludwig', input_carrier=None)
    print('  z=%.4f mm  P/Pin=%.4e  nwarn=%d' % (z*1e3, float((np.abs(F)**2).sum())/Pin, len(W)))
print('--- min_area_ratio sweep AT z = 24.834 mm ---')
for mar in (1e-6, 1e-8, 1e-10, 1e-12, 0.0):
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter('always')
        F = MB(E, prescription=p, wavelength=fixt.WL, dx=dx,
               output_plane_distance=24.834e-3, ray_subsample=4,
               min_area_ratio=mar, caustic_band='ludwig', input_carrier=None)
    print('  min_area_ratio=%-8g  P/Pin=%.4e  peak=%.4e  nwarn=%d  %s'
          % (mar, float((np.abs(F)**2).sum())/Pin, float(np.abs(F).max()), len(W),
             '; '.join(sorted({str(w.message)[:70] for w in W}))))
print('--- through the public entry point (caustic="multibranch") ---')
for z in (24.0e-3, 24.834e-3):
    with warnings.catch_warnings(record=True) as W:
        warnings.simplefilter('always')
        F = apply_real_lens_traced(E, prescription=p, wavelength=fixt.WL, dx=dx,
                                   amplitude_model='ray_density', caustic='multibranch',
                                   output_plane_distance=z, caustic_ray_subsample=4,
                                   on_undersample='silent')
    print('  z=%.4f mm  P/Pin=%.4e  nwarn=%d' % (z*1e3, float((np.abs(F)**2).sum())/Pin, len(W)))

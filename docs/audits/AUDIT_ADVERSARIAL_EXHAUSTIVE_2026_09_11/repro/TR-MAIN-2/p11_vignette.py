"""Vignetted prescription (dead rays -> NaN in the launch grid) on both fits."""
import fixt, numpy as np, warnings
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced import apply_real_lens_traced
p = fixt.small_singlet()
p['surfaces'][1]['semi_diameter'] = 0.30e-3     # hard vignette at the rear
N, dx = 512, 6e-6
E = fixt.gauss(N, dx, 0.4e-3)
Pin = float((np.abs(E)**2).sum())
for fit in ('polynomial', 'spline'):
    for sub in (8,):
        with warnings.catch_warnings(record=True) as W:
            warnings.simplefilter('always')
            try:
                o = apply_real_lens_traced(E, prescription=p, wavelength=fixt.WL,
                                           dx=dx, newton_fit=fit, ray_subsample=sub,
                                           on_undersample='silent',
                                           on_pool_memory='silent',
                                           on_aperture_beam='silent')
                print('newton_fit=%-11s P/Pin=%.6e  nonzero=%7d  NaN=%d  peak=%.4e  warns=%d'
                      % (fit, float((np.abs(o)**2).sum())/Pin, int((o!=0).sum()),
                         int(np.isnan(o).sum()), float(np.abs(o).max()), len(W)))
                for m in sorted({str(w.message)[:90] for w in W}):
                    print('     warn:', m)
            except Exception as e:
                print('newton_fit=%-11s RAISED %s: %s' % (fit, type(e).__name__, str(e)[:110]))

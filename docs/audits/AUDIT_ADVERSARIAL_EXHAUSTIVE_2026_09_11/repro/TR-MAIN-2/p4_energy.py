import fixt, numpy as np, warnings
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced import apply_real_lens_traced
p = fixt.small_singlet()
N, dx = 512, 6e-6
E = fixt.gauss(N, dx, 0.4e-3)
x = (np.arange(N)-N/2)*dx; X,Y = np.meshgrid(x,x)
Pap = float((np.abs(E)**2)[(X**2+Y**2) <= (1.0e-3)**2].sum())
print('P_in(full)=%.4f  P_in(inside aperture)=%.4f' % (float((np.abs(E)**2).sum()), Pap))
base = dict(prescription=p, wavelength=fixt.WL, dx=dx, on_undersample='silent',
            on_pool_memory='silent', on_aperture_beam='silent')
for am in ('screen', 'ray_density'):
    for sub in (1, 2, 4, 8, 16):
        for im in (True, False):
            with warnings.catch_warnings(record=True) as W:
                warnings.simplefilter('always')
                o = apply_real_lens_traced(E, amplitude_model=am, ray_subsample=sub,
                                           inverse_map=im, **base)
            nw = len([w for w in W if 'energy self-check' in str(w.message)
                      or 'HALO' in str(w.message) or 'SUPPORT-BAND' in str(w.message)])
            print('  %-11s sub=%-3d imap=%-5s  P_out/P_ap = %.6f   selfcheck_warn=%d'
                  % (am, sub, im, float((np.abs(o)**2).sum())/Pap, nw))

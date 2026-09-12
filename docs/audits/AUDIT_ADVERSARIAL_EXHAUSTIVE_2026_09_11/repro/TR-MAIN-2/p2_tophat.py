import fixt, numpy as np, warnings
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced import apply_real_lens_traced
p = fixt.small_singlet()
N, dx = 512, 6e-6
kw = dict(prescription=p, wavelength=fixt.WL, dx=dx, on_undersample='silent',
          on_pool_memory='silent', on_aperture_beam='silent')
for label, E in [('tophat r=0.9mm', fixt.tophat(N, dx, 0.9e-3)),
                 ('tophat r=1.4mm (> aperture/2)', fixt.tophat(N, dx, 1.4e-3)),
                 ('gauss w=0.4mm', fixt.gauss(N, dx, 0.4e-3))]:
    for am in ('screen', 'ray_density'):
        with warnings.catch_warnings(record=True) as W:
            warnings.simplefilter('always')
            o = apply_real_lens_traced(E, amplitude_model=am, **kw)
        nnan = int(np.isnan(o).sum()); ninf = int(np.isinf(o).sum())
        pin = float((np.abs(E)**2).sum()); pout = float((np.abs(o)**2).sum())
        nz  = int((o == 0).sum())
        msgs = sorted({str(w.category.__name__)+':'+str(w.message)[:60] for w in W})
        print(f'{label:32s} {am:12s} NaN={nnan:6d} Inf={ninf:6d} zeros={nz:7d} '
              f'P_out/P_in={pout/pin:.6f}')
        for m in msgs: print('      warn:', m)

import fixt, numpy as np, warnings, traceback
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced import apply_real_lens_traced, apply_real_lens
p = fixt.small_singlet()
N, dx = 256, 12e-6
Ereal = np.real(fixt.gauss(N, dx, 0.4e-3)).copy()   # float64 2-D field
print('E dtype', Ereal.dtype, 'iscomplexobj', np.iscomplexobj(Ereal))
kw = dict(prescription=p, wavelength=fixt.WL, dx=dx, on_undersample='silent',
          on_pool_memory='silent', on_aperture_beam='silent')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        o = apply_real_lens(Ereal, prescription=p, wavelength=fixt.WL, dx=dx)
        print('apply_real_lens OK ->', o.dtype)
    except Exception as e:
        print('apply_real_lens RAISED', type(e).__name__, e)
    try:
        o = apply_real_lens_traced(Ereal, **kw)
        print('traced OK ->', o.dtype)
    except Exception as e:
        print('traced RAISED', type(e).__name__, ':', e)
        traceback.print_exc()

import fixt, numpy as np, warnings, traceback
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced import apply_real_lens_traced_multi
p = fixt.small_singlet()
N, dx = 512, 6e-6
E1 = fixt.gauss(N, dx, 0.4e-3); E2 = fixt.gauss(N, dx, 0.4e-3, tilt=(5e-3,0))
common = dict(prescription=p, wavelength=fixt.WL, dx=dx)
for kwname, val in [('on_pool_memory','silent'), ('newton_mask_dilate_coarse_px', 3), ('dy', dx),
                    ('origin', (0.0, 0.0))]:
    for reuse in (True, False):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                apply_real_lens_traced_multi([E1,E2], carriers=None,
                    reuse_prepared=reuse, **{kwname: val}, **common)
            print(f'{kwname:32s} reuse={reuse!s:5s} OK')
        except Exception as e:
            print(f'{kwname:32s} reuse={reuse!s:5s} {type(e).__name__}: {str(e)[:90]}')

import fixt, numpy as np, warnings, copy
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced import (apply_real_lens_traced,
                                             prepare_real_lens_traced,
                                             PreparedTracedLens)
p = fixt.small_singlet()
N, dx = 512, 6e-6
kw = dict(prescription=p, wavelength=fixt.WL, dx=dx, on_undersample='silent')
def rel(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    prep = prepare_real_lens_traced(N=N, carrier=None, **kw)
    print('screen dtype', prep.screen.dtype, 'shape', prep.screen.shape)
    print('slots:', {s: (type(getattr(prep,s,None)).__name__) for s in PreparedTracedLens.__slots__})
    # 1. equivalence on a field with the SAME footprint used to prepare
    for lbl, E in [('gauss w=0.4mm', fixt.gauss(N, dx, 0.4e-3)),
                   ('gauss w=0.15mm (different footprint)', fixt.gauss(N, dx, 0.15e-3)),
                   ('tilted 8 mrad', fixt.gauss(N, dx, 0.4e-3, tilt=(8e-3,0))),
                   ('complex64', fixt.gauss(N, dx, 0.4e-3).astype(np.complex64)),
                   ('tophat', fixt.tophat(N, dx, 0.9e-3))]:
        direct = apply_real_lens_traced(E, carrier=None, on_pool_memory='silent',
                                        on_aperture_beam='silent',
                                        tilt_aware_rays=False, parallel_amp=False,
                                        newton_amp_mask_rel=0.0, **kw)
        pre = prep(E)
        print('  %-38s rel(prep,direct) = %.3e  dtypes %s/%s'
              % (lbl, rel(pre, direct), pre.dtype, direct.dtype))
    # 2. staleness: mutate the prescription IN PLACE after preparing
    p['surfaces'][0]['radius'] = 20.0e-3
    E = fixt.gauss(N, dx, 0.4e-3)
    after = prep(E)
    prep2 = prepare_real_lens_traced(N=N, carrier=None, **dict(kw, prescription=p))
    print('  after in-place R edit: prep_old vs prep_new = %.4e' % rel(after, prep2(E)))
    p['surfaces'][0]['radius'] = 25.84e-3
    # 3. wrong shape / dtype
    try:
        prep(fixt.gauss(256, dx, 0.4e-3))
    except Exception as e:
        print('  wrong shape ->', type(e).__name__, str(e)[:70])
    # 4. does a FLOAT (real) input work through the prepared object?
    try:
        r = prep(np.real(E).copy())
        print('  real input through prepared -> OK, dtype', r.dtype)
    except Exception as e:
        print('  real input through prepared ->', type(e).__name__, str(e)[:70])

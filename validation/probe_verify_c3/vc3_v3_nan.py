"""The defect WP-C3 says the fallback change fixes: _carrier_step_fast on a
COLLIMATED carrier.  Measured directly, with no test in the way."""
import warnings
import numpy as np
import lumenairy.propagators.carrier as CA
LAM = 1.064e-6
x = (np.arange(512) - 256) * 8e-6
env = np.exp(-(x[None, :]**2 + x[:, None]**2) / (0.5e-3)**2).astype(complex)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    old = CA._carrier_step_fast(env, np.inf, 5e-3, LAM, 8e-6, 8e-6,
                                gap_kernel='auto')
    new = CA.propagate_carrier_referenced(env, np.inf, 5e-3, LAM, 8e-6,
                                          transport='sziklas')
    diag = {}
    leg = CA._collins_carrier_leg(env, np.inf, 5e-3, LAM, 8e-6, 8e-6,
                                  gap_kernel='auto',
                                  on_collins_sampling='ignore', diag=diag)
def rep(t, cr):
    a = np.asarray(cr.env)
    print('%-34s dx=%-22r R=%-8r nan=%s  |E|max=%s'
          % (t, cr.dx, cr.R, bool(np.isnan(a).any()),
             ('nan' if np.isnan(a).all() else '%.6g' % np.abs(a).max())))
rep('_carrier_step_fast (pre-C3 call)', old)
rep("propagate_carrier_referenced szik", new)
rep('_collins_carrier_leg (shipped)', leg)
print('leg resolved:', diag.get('collins_form'), 'k1=', diag.get('collins_k1'),
      'k3=', diag.get('collins_k3'))

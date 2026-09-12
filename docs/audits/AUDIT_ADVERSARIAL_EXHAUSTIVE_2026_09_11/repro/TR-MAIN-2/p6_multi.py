import fixt, numpy as np, warnings
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced import (apply_real_lens_traced,
                                             apply_real_lens_traced_multi)
from lumenairy.elements._lens_real import apply_real_lens
p = fixt.small_singlet()
N, dx = 512, 6e-6
th = 5e-3
E1 = fixt.gauss(N, dx, 0.4e-3, tilt=(+th, 0.0))
E2 = fixt.gauss(N, dx, 0.4e-3, tilt=(-th, 0.0))
Esum = E1 + E2
common = dict(prescription=p, wavelength=fixt.WL, dx=dx)
tkw = dict(on_undersample='silent', on_pool_memory='silent',
           on_aperture_beam='silent')

def rel(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))

with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter('always')
    A_sum  = apply_real_lens(Esum, **common)
    A_1    = apply_real_lens(E1, **common)
    A_2    = apply_real_lens(E2, **common)
    T_multi_auto = apply_real_lens_traced_multi([E1, E2], carriers='auto', **common, **tkw)
    T_multi_none = apply_real_lens_traced_multi([E1, E2], carriers=None, **common, on_undersample='silent')
    T_multi_nr   = apply_real_lens_traced_multi([E1, E2], carriers=None, reuse_prepared=False, **common, **tkw)
    T_1 = apply_real_lens_traced(E1, carrier='auto', **common, **tkw)
    T_2 = apply_real_lens_traced(E2, carrier='auto', **common, **tkw)
    T_sum_direct = apply_real_lens_traced(Esum, carrier=None, **common, **tkw)
print('analytic linearity  |A(E1+E2)-(A1+A2)|/|A| = %.3e' % rel(A_sum, A_1+A_2))
print('multi(auto)  vs analytic(sum): %.4e' % rel(T_multi_auto, A_sum))
print('multi(None)  vs analytic(sum): %.4e' % rel(T_multi_none, A_sum))
print('multi(None,noreuse) vs analytic: %.4e' % rel(T_multi_nr, A_sum))
print('multi(None) vs multi(None,noreuse): %.4e' % rel(T_multi_none, T_multi_nr))
print('T1+T2        vs analytic(sum): %.4e' % rel(T_1+T_2, A_sum))
print('multi(auto)  vs  T1+T2       : %.4e' % rel(T_multi_auto, T_1+T_2))
print('traced(sum)  vs analytic(sum): %.4e' % rel(T_sum_direct, A_sum))
for nm, arr in [('A_sum', A_sum), ('multi_auto', T_multi_auto),
                ('multi_none', T_multi_none), ('T1+T2', T_1+T_2)]:
    print('  %-12s P=%.6f  peak=%.6f' % (nm, float((np.abs(arr)**2).sum()),
                                          float(np.abs(arr).max())))
# per-branch piston consistency: compare each branch's phase against analytic
for nm, (t, a) in [('branch1', (T_1, A_1)), ('branch2', (T_2, A_2))]:
    m = np.abs(a) > 1e-2*np.abs(a).max()
    d = np.angle(t[m]) - np.angle(a[m]); d = (d+np.pi)%(2*np.pi)-np.pi
    print('  %s: mean phase offset %.4f rad, rms about mean %.4e rad'
          % (nm, float(np.angle(np.mean(np.exp(1j*d)))), float(np.std(d))))

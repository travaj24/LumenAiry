"""Asymmetric two-branch coherent sum: branch pistons must be consistent."""
import fixt, numpy as np, warnings
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced import (apply_real_lens_traced,
                                             apply_real_lens_traced_multi,
                                             TiltedCarrier)
from lumenairy.elements._lens_real import apply_real_lens
p = fixt.small_singlet()
N, dx = 512, 6e-6
w = 0.35e-3
E1 = fixt.gauss(N, dx, w, tilt=(0.0, 0.0))          # on-axis
E2 = fixt.gauss(N, dx, w, tilt=(12e-3, 0.0))        # 12 mrad tilt
common = dict(prescription=p, wavelength=fixt.WL, dx=dx)
tkw = dict(on_undersample='silent', on_pool_memory='silent',
           on_aperture_beam='silent')
def rel(a,b): return float(np.linalg.norm(a-b)/np.linalg.norm(b))
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    A1 = apply_real_lens(E1, **common); A2 = apply_real_lens(E2, **common)
    Asum = apply_real_lens(E1+E2, **common)
    c1 = TiltedCarrier(np.inf, 0.0,   0.0, 0.0, 0.0)
    c2 = TiltedCarrier(np.inf, 12e-3, 0.0, 0.0, 0.0)
    Tm = apply_real_lens_traced_multi([E1, E2], carriers=[c1, c2],
                                      reuse_prepared=False, **common, **tkw)
    T1 = apply_real_lens_traced(E1, carrier=c1, **common, **tkw)
    T2 = apply_real_lens_traced(E2, carrier=c2, **common, **tkw)
print('TiltedCarrier fields:', TiltedCarrier._fields)
print('multi == T1+T2 :', rel(Tm, T1+T2))
print('multi vs analytic(sum): %.4e' % rel(Tm, Asum))
print('T1 vs A1: %.4e ; T2 vs A2: %.4e' % (rel(T1,A1), rel(T2,A2)))
for nm, (t,a) in [('branch1 (0 mrad)',(T1,A1)), ('branch2 (12 mrad)',(T2,A2))]:
    m = np.abs(a) > 3e-2*np.abs(a).max()
    d = np.angle(t[m]) - np.angle(a[m]); d = (d+np.pi)%(2*np.pi)-np.pi
    off = float(np.angle(np.mean(np.exp(1j*d))))
    print('  %-18s piston vs analytic = %+.5f rad ; rms about it = %.3e rad'
          % (nm, off, float(np.std((d-off+np.pi)%(2*np.pi)-np.pi))))
# interference fringe position test: fringe phase = arg(T1) - arg(T2)
m = (np.abs(A1) > 0.3*np.abs(A1).max()) & (np.abs(A2) > 0.3*np.abs(A2).max())
print('overlap pixels:', int(m.sum()))
if m.any():
    fa = np.angle(A1[m]) - np.angle(A2[m])
    ft = np.angle(T1[m]) - np.angle(T2[m])
    d = (ft-fa+np.pi)%(2*np.pi)-np.pi
    print('  fringe-phase (T) - (A): mean %+.4f rad  rms %.4f rad' % (d.mean(), d.std()))

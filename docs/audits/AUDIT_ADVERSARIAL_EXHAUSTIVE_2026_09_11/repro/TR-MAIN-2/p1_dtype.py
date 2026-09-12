import fixt, numpy as np, warnings
import lumenairy as la
fixt.register_glass()
from lumenairy.elements._lens_traced import apply_real_lens_traced

# 5 mm-thick element as the probe asks
p = dict(fixt.small_singlet(ap=2.0e-3, t=5.0e-3, R=25.84e-3))
N, dx = 512, 6e-6
E64 = fixt.gauss(N, dx, 0.4e-3, dtype=np.complex128)
E32 = E64.astype(np.complex64)
kw = dict(prescription=p, wavelength=fixt.WL, dx=dx,
          on_undersample='silent', on_pool_memory='silent',
          on_aperture_beam='silent')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    o64 = apply_real_lens_traced(E64, **kw)
    o32 = apply_real_lens_traced(E32, **kw)
print('dtypes', o64.dtype, o32.dtype)
d = o32.astype(np.complex128) - o64
pk = np.abs(o64).max()
print('max|diff|/peak = %.3e' % (np.abs(d).max()/pk))
print('rms|diff|/peak = %.3e' % (np.sqrt((np.abs(d)**2).mean())/pk))
m = np.abs(o64) > 0.05*pk
ph = np.angle(o32.astype(np.complex128)[m]) - np.angle(o64[m])
ph = (ph + np.pi) % (2*np.pi) - np.pi
print('core phase diff: max %.3e rms %.3e rad' % (np.abs(ph).max(), np.sqrt((ph**2).mean())))
print('P64', float((np.abs(o64)**2).sum()), 'P32', float((np.abs(o32.astype(np.complex128))**2).sum()))
# what is the magnitude of the raw traced phase before wrap?

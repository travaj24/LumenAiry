"""Probe 5a: does _lens_jax enforce float64?  Run with DEFAULT jax config
(x64 OFF -- JAX's default, which lumenairy never sets) and compare the phase
screen against the NumPy apply_real_lens_traced."""
import os, sys, warnings, numpy as np
X64 = os.environ.get('X64', '0') == '1'
import jax
if X64:
    jax.config.update('jax_enable_x64', True)
print('jax', jax.__version__, 'x64 =', jax.config.read('jax_enable_x64'))
import lumenairy
from lumenairy.elements._lens_jax import apply_real_lens_traced_jax, apply_real_lens_maslov_jax
from lumenairy.elements.lenses import apply_real_lens_traced
import jax.numpy as jnp

lam = 0.5876e-6; k0 = 2*np.pi/lam
N, dx = 256, 30e-6
rx = lumenairy.make_singlet(R1=25e-3, R2=float('inf'), d=3e-3, glass='N-BK7', aperture=6e-3)
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x); r = np.hypot(X, Y)
E = np.exp(-(r/2.0e-3)**2).astype(np.complex128)
print('E dtype numpy', E.dtype, ' -> jnp.asarray ->', jnp.asarray(E).dtype)
print('x_wave dtype', ((jnp.arange(N)-N/2)*float(dx)).dtype)

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    Ej = np.asarray(apply_real_lens_traced_jax(E, prescription=rx, wavelength=lam,
                                               dx=dx, ray_subsample=4, cheb_order=10))
    En = np.asarray(apply_real_lens_traced(E, prescription=rx, wavelength=lam, dx=dx,
                                           amplitude_model='screen',
                                           preserve_input_phase=True,
                                           on_undersample='silent'))
print('jax out dtype', Ej.dtype)
m = (np.abs(E) > 1e-3) & (np.abs(Ej) > 0) & np.isfinite(En) & (np.abs(En) > 0)
ph = np.angle(Ej[m]*np.conj(E[m]))           # the jax OPD screen phase
# reference screen phase from the numpy traced path
phn = np.angle(En[m]*np.conj(E[m]))
d = np.angle(np.exp(1j*(ph-phn)))
d = np.angle(np.exp(1j*(d-np.angle(np.sum(np.exp(1j*d))))))
print('pixels %d  |phase(jax) - phase(numpy)| rms %.4g rad = %.4g waves, max %.4g rad'
      % (m.sum(), np.sqrt(np.mean(d**2)), np.sqrt(np.mean(d**2))/(2*np.pi), np.abs(d).max()))

"""Probe 5b/c/d: jax.grad, jit and timing through _lens_jax."""
import os, time, warnings, numpy as np, jax
if os.environ.get('X64','1')=='1':
    jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
print('x64', jax.config.read('jax_enable_x64'))
import lumenairy
from lumenairy.elements._lens_jax import apply_real_lens_traced_jax as TJ
from lumenairy.elements._lens_jax import apply_real_lens_maslov_jax as MJ

lam=0.5876e-6; N,dx=128,60e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x); r=np.hypot(X,Y)
E=jnp.asarray(np.exp(-(r/2.0e-3)**2).astype(np.complex128))
def rx_of(R1):
    return {'name':'s','aperture_diameter':6e-3,
            'surfaces':[{'radius':R1,'conic':0.0,'aspheric_coeffs':None,'radius_y':None,
                         'conic_y':None,'aspheric_coeffs_y':None,
                         'glass_before':'air','glass_after':'N-BK7'},
                        {'radius':float('inf'),'conic':0.0,'aspheric_coeffs':None,
                         'radius_y':None,'conic_y':None,'aspheric_coeffs_y':None,
                         'glass_before':'N-BK7','glass_after':'air'}],
            'thicknesses':[3e-3]}

# ---------- (b1) grad w.r.t. E_in ----------
def merit_E(Ein, fn):
    Eo = fn(Ein, prescription=rx_of(25e-3), wavelength=lam, dx=dx,
            ray_subsample=8, cheb_order=8, newton_iters=8)
    return jnp.sum(jnp.abs(Eo)**2).real
for nm, fn in (('traced_jax', TJ), ('maslov_jax', MJ)):
    try:
        g = jax.grad(lambda v: merit_E(v, fn))(E)
        print('%-11s grad wrt E_in: finite=%s  max|g|=%.4g  nonzero=%d/%d'
              % (nm, bool(np.all(np.isfinite(np.asarray(g)))), float(jnp.max(jnp.abs(g))),
                 int(jnp.sum(jnp.abs(g) > 0)), g.size))
    except Exception as e:
        print('%-11s grad wrt E_in FAILED: %s: %s' % (nm, type(e).__name__, str(e)[:170]))

# ---------- (b2) grad w.r.t. a RADIUS via radii= (the design lever) ----------
def merit_R(Rv):
    radii = jnp.stack([Rv, jnp.asarray(np.inf)])
    Eo = TJ(E, prescription=rx_of(25e-3), wavelength=lam, dx=dx,
            ray_subsample=8, cheb_order=8, newton_iters=10, radii=radii)
    I = jnp.abs(Eo)**2
    return (jnp.sum(I * jnp.exp(-(jnp.asarray(r)/3e-4)**2)) / jnp.sum(I)).real
R0 = jnp.asarray(25e-3)
try:
    v = merit_R(R0); g = jax.grad(merit_R)(R0)
    print('merit(R0)=%.8g   jax.grad=%.8g' % (float(v), float(g)))
    for h in (1e-6, 1e-5, 1e-4):
        fd = (float(merit_R(R0+h)) - float(merit_R(R0-h))) / (2*h)
        print('   central FD h=%.0e -> %.8g   rel diff %.3e'
              % (h, fd, abs(g-fd)/max(abs(fd), 1e-300)))
except Exception as e:
    import traceback; traceback.print_exc()

# ---------- (c) NumPy/JAX parity ----------
from lumenairy.elements.lenses import apply_real_lens_traced
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    Ej = np.asarray(TJ(E, prescription=rx_of(25e-3), wavelength=lam, dx=dx,
                       ray_subsample=8, cheb_order=10, newton_iters=12))
    En = np.asarray(apply_real_lens_traced(np.asarray(E), prescription=rx_of(25e-3),
                    wavelength=lam, dx=dx, amplitude_model='screen',
                    preserve_input_phase=True, on_undersample='silent'))
m = (np.abs(Ej) > 0) & np.isfinite(En) & (np.abs(En) > 0)
d = np.angle(Ej[m]*np.conj(En[m])); d = np.angle(np.exp(1j*(d-np.angle(np.sum(np.exp(1j*d))))))
print('parity numpy-vs-jax phase rms %.4g rad, amp rel rms %.4g'
      % (np.sqrt(np.mean(d**2)),
         np.sqrt(np.mean(((np.abs(Ej[m])-np.abs(En[m]))/np.abs(En[m]).max())**2))))

# ---------- (d) jit ----------
import functools
try:
    f = jax.jit(functools.partial(TJ, prescription=rx_of(25e-3), wavelength=lam, dx=dx,
                                  ray_subsample=8, cheb_order=8, newton_iters=8))
    t0=time.time(); out=f(E); out.block_until_ready(); print('jit compile+run %.2fs' % (time.time()-t0))
    t0=time.time()
    for _ in range(5): f(E).block_until_ready()
    print('jit steady %.4fs/call' % ((time.time()-t0)/5))
except Exception as e:
    print('jax.jit FAILED: %s: %s' % (type(e).__name__, str(e)[:250]))
t0=time.time()
for _ in range(3):
    np.asarray(TJ(E, prescription=rx_of(25e-3), wavelength=lam, dx=dx,
                  ray_subsample=8, cheb_order=8, newton_iters=8))
print('eager %.3fs/call (N=%d)' % ((time.time()-t0)/3, N))

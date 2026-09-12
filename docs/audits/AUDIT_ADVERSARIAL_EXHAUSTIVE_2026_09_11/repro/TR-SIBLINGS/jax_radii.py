"""Does apply_real_lens_traced_jax(radii=...) actually change the output, and
does jax.grad through it match finite differences?  Also: jit with radii."""
import time, numpy as np, jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from lumenairy.elements._lens_jax import apply_real_lens_traced_jax as TJ

lam=0.5876e-6; N,dx=192,20e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x); r=np.hypot(X,Y)
E=jnp.asarray(np.exp(-(r/1.2e-3)**2).astype(np.complex128))
rx={'name':'s','aperture_diameter':3.4e-3,
    'surfaces':[{'radius':25e-3,'conic':0.0,'aspheric_coeffs':None,'radius_y':None,
                 'conic_y':None,'aspheric_coeffs_y':None,
                 'glass_before':'air','glass_after':'N-BK7'},
                {'radius':float('inf'),'conic':0.0,'aspheric_coeffs':None,'radius_y':None,
                 'conic_y':None,'aspheric_coeffs_y':None,
                 'glass_before':'N-BK7','glass_after':'air'}],
    'thicknesses':[3e-3]}
base = dict(prescription=rx, wavelength=lam, dx=dx, ray_subsample=2,
            cheb_order=8, newton_iters=14)

print('--- does radii= change the OUTPUT at all? ---')
E0 = np.asarray(TJ(E, **base))
for Rv in (25e-3, 24e-3, 20e-3):
    Er = np.asarray(TJ(E, radii=jnp.asarray([Rv, np.inf]), **base))
    m = np.abs(E0) > 1e-6
    print('  radii=[%.4g, inf]  max|E_radii - E_static| = %.4g   phase rms diff %.4g rad'
          % (Rv, np.max(np.abs(Er-E0)),
             np.sqrt(np.mean(np.angle(Er[m]*np.conj(E0[m]))**2))))
print()
print('--- jax.grad vs central FD of a SMOOTH merit (encircled energy) ---')
rj = jnp.asarray(r)
def merit(Rv):
    Eo = TJ(E, radii=jnp.stack([Rv, jnp.asarray(jnp.inf)]), **base)
    I = jnp.abs(Eo)**2
    return (jnp.sum(I*jnp.exp(-(rj/6e-4)**2))/jnp.sum(I)).real
R0 = jnp.asarray(25e-3)
t0=time.time(); v = float(merit(R0)); print('  merit(R0) = %.10g   (%.1f s)'%(v, time.time()-t0))
t0=time.time(); g = float(jax.grad(merit)(R0)); print('  jax.grad  = %.10g   (%.1f s)'%(g, time.time()-t0))
for h in (2e-5, 1e-4, 5e-4):
    fd = (float(merit(R0+h))-float(merit(R0-h)))/(2*h)
    print('  FD h=%.0e -> %.10g    rel diff %.3g' % (h, fd, abs(g-fd)/max(abs(fd),1e-300)))
print()
print('--- jit WITH radii (tracer-safe branch) ---')
import functools
try:
    f = jax.jit(lambda Rv, Ein: TJ(Ein, radii=jnp.stack([Rv, jnp.asarray(jnp.inf)]), **base))
    t0=time.time(); o=f(R0,E); o.block_until_ready(); print('  compile+run %.2f s'%(time.time()-t0))
    t0=time.time()
    for _ in range(3): f(R0,E).block_until_ready()
    print('  steady %.4f s/call' % ((time.time()-t0)/3))
except Exception as e:
    print('  FAILED %s: %s' % (type(e).__name__, str(e)[:200]))
print()
print('--- n_launch adequacy vs cheb_order (silent under-determination) ---')
for sub, order in ((8,10),(8,8),(4,10),(2,10)):
    lr = 0.5*3.4e-3*1.02
    nl = max(8, int(2*lr/(dx*sub)));  nl += (nl%2==0)
    terms = (order+1)*(order+2)//2
    print('  ray_subsample=%d cheb_order=%d -> n_launch=%d (%d samples) for %d terms  -> %.1f samples/term'
          % (sub, order, nl, nl*nl, terms, nl*nl/terms))

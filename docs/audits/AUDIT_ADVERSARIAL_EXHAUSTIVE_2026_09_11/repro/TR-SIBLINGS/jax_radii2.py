"""jax.grad w.r.t. a radius, with a PHASE-SENSITIVE merit (Strehl / on-axis
far-field), vs central finite differences."""
import time, numpy as np, jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from lumenairy.elements._lens_jax import apply_real_lens_traced_jax as TJ
from lumenairy.elements._lens_jax import apply_real_lens_maslov_jax as MJ

lam=0.5876e-6; N,dx=192,20e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x); r=np.hypot(X,Y)
E=jnp.asarray(np.exp(-(r/1.2e-3)**2).astype(np.complex128))
def rx(R1=25e-3):
    return {'name':'s','aperture_diameter':3.4e-3,
     'surfaces':[{'radius':R1,'conic':0.0,'aspheric_coeffs':None,'radius_y':None,
                  'conic_y':None,'aspheric_coeffs_y':None,
                  'glass_before':'air','glass_after':'N-BK7'},
               {'radius':float('inf'),'conic':0.0,'aspheric_coeffs':None,'radius_y':None,
                'conic_y':None,'aspheric_coeffs_y':None,
                'glass_before':'N-BK7','glass_after':'air'}],
     'thicknesses':[3e-3]}
base=dict(prescription=rx(), wavelength=lam, dx=dx, ray_subsample=2,
          cheb_order=8, newton_iters=14)
# defocus-free Strehl-like merit: |sum E|^2 / (Npix * sum|E|^2) == on-axis
# far-field intensity, which is PHASE sensitive.
def strehl(Eo):
    s = jnp.sum(Eo)
    return (jnp.abs(s)**2/(Eo.size*jnp.sum(jnp.abs(Eo)**2))).real
for nm, F in (('traced_jax', TJ),):
    def merit(Rv):
        return strehl(F(E, radii=jnp.stack([Rv, jnp.asarray(jnp.inf)]), **base))
    R0=jnp.asarray(25e-3)
    t0=time.time(); v=float(merit(R0)); print('%s merit(R0)=%.10g (%.1fs)'%(nm,v,time.time()-t0))
    t0=time.time(); g=float(jax.grad(merit)(R0)); print('  jax.grad = %.8g  (%.1fs)'%(g,time.time()-t0))
    for h in (1e-6,1e-5,5e-5,2e-4):
        fd=(float(merit(R0+h))-float(merit(R0-h)))/(2*h)
        print('  FD h=%.0e -> %.8g   rel |g-fd|/|fd| = %.4g'%(h,fd,abs(g-fd)/max(abs(fd),1e-300)))

print()
print('--- where does the gradient die?  d(opl_grid)/dR through trace_jax_with_params ---')
from lumenairy.raytrace.jax_trace import make_jax_ray_state, trace_jax_with_params
from lumenairy.glass import get_glass_index
hs=jnp.linspace(-1.7e-3,1.7e-3,21); HX,HY=jnp.meshgrid(hs,hs,indexing='ij')
hx,hy=HX.ravel(),HY.ravel()
def opl_sum(Rv):
    st=make_jax_ray_state(x=hx,y=hy,z=jnp.zeros_like(hx),L=jnp.zeros_like(hx),
                          M=jnp.zeros_like(hx),N=jnp.ones_like(hx))
    fin=trace_jax_with_params(st, rx(), lam, radii=jnp.stack([Rv,jnp.asarray(jnp.inf)]))
    ne=float(get_glass_index('air',lam))
    t=jnp.where(fin.alive, -fin.z/jnp.where(jnp.abs(fin.N)>1e-30,fin.N,1e-30),0.0)
    op=fin.opd+ne*t
    return jnp.sum(jnp.where(fin.alive, op, 0.0))
R0=jnp.asarray(25e-3)
g=float(jax.grad(opl_sum)(R0))
fd=(float(opl_sum(R0+1e-6))-float(opl_sum(R0-1e-6)))/2e-6
print('  d(sum OPL)/dR: jax.grad %.8g   FD %.8g   rel %.4g'%(g,fd,abs(g-fd)/max(abs(fd),1e-300)))

print()
print('--- d(coeffs)/dR through _cheb_fit_2d_jax ---')
from lumenairy.elements._lens_jax import _cheb_fit_2d_jax
def coef_sum(Rv):
    st=make_jax_ray_state(x=hx,y=hy,z=jnp.zeros_like(hx),L=jnp.zeros_like(hx),
                          M=jnp.zeros_like(hx),N=jnp.ones_like(hx))
    fin=trace_jax_with_params(st, rx(), lam, radii=jnp.stack([Rv,jnp.asarray(jnp.inf)]))
    t=jnp.where(fin.alive, -fin.z/jnp.where(jnp.abs(fin.N)>1e-30,fin.N,1e-30),0.0)
    op=(fin.opd+t).reshape(21,21)
    op=op-op[10,10]
    op=jnp.where(fin.alive.reshape(21,21), op, jnp.nan)
    c,K1,K2,a,b,cc,ddd=_cheb_fit_2d_jax(hs,hs,op,8)
    return jnp.sum(c)
g=float(jax.grad(coef_sum)(R0))
fd=(float(coef_sum(R0+1e-6))-float(coef_sum(R0-1e-6)))/2e-6
print('  d(sum coeffs)/dR: jax.grad %.8g   FD %.8g   rel %.4g'%(g,fd,abs(g-fd)/max(abs(fd),1e-300)))

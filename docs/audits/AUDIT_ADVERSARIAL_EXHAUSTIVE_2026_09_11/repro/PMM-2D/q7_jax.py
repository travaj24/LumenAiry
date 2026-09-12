from common import *
import jax, jax.numpy as jnp
print("jax", jax.__version__, "x64 flag:", jax.config.jax_enable_x64, flush=True)
from lumenairy.elements.pmm import pmm_jones_2d, pmm_efficiency_2d_cell, PMM2DStackHybrid

wl, Px, Py, dep = 1.0e-6, 0.47e-6, 0.47e-6, 0.3e-6
nsup, nsub = 1.0, 1.5
S = 12
lay = np.zeros((S,S), dtype=int); lay[3:9,3:9] = 1        # both-axes patterned
def build(eps_p, eps_h=1.0):
    c = np.zeros((S,S,3,3), dtype=complex)
    c[...] = eps_h*np.eye(3); c[3:9,3:9] = eps_p*np.eye(3)
    return c

print("== 1) x64 enforcement: does it RAISE without x64? ==")
try:
    jax.config.update("jax_enable_x64", False)
    cj = jnp.asarray(build(12.25))
    pmm_jones_2d(Px,Py,cj,nsub,nsup,dep,wl,degree=5,n_orders=2,region_layout=lay)
    print("  !! no raise at x32")
except Exception as e:
    print("  raised:", type(e).__name__, str(e)[:140])
jax.config.update("jax_enable_x64", True)

print("== 2) numpy/jax forward parity ==")
cn = build(12.25); cj = jnp.asarray(cn)
o1,R1,T1,J1 = pmm_jones_2d(Px,Py,cn,nsub,nsup,dep,wl,degree=5,n_orders=3)
o2,R2,T2,J2 = pmm_jones_2d(Px,Py,cj,nsub,nsup,dep,wl,degree=5,n_orders=3,region_layout=lay)
J2 = np.asarray(J2); R2 = np.asarray(R2); T2 = np.asarray(T2)
def rms_rel(a,b):
    a=np.asarray(a); b=np.asarray(b)
    return float(np.sqrt(np.mean(np.abs(a-b)**2))/max(np.sqrt(np.mean(np.abs(a)**2)),1e-300))
print(f"  RMS rel: R={rms_rel(R1,R2):.3e}  T={rms_rel(T1,T2):.3e}  J={rms_rel(J1,J2):.3e}")
print(f"  J numpy={J1.ravel()}\n  J jax  ={J2.ravel()}", flush=True)
print("  numpy symmetry=False:", end=" ")
o3,R3,T3,J3 = pmm_jones_2d(Px,Py,cn,nsub,nsup,dep,wl,degree=5,n_orders=3,symmetry=False)
print(f"J rel vs jax = {rms_rel(J3,J2):.3e}", flush=True)

print("== 3) jax.grad vs central FD on a PILLAR-WIDTH-like parameter (eps value) ==")
def f_eps(ep):
    c = jnp.zeros((S,S,3,3), dtype=jnp.complex128)
    c = c.at[...].set(1.0*jnp.eye(3, dtype=jnp.complex128))
    c = c.at[3:9,3:9].set(ep.astype(jnp.complex128)*jnp.eye(3, dtype=jnp.complex128))
    o,R,T,J = pmm_jones_2d(Px,Py,c,nsub,nsup,dep,wl,degree=5,n_orders=3,region_layout=lay)
    return jnp.real(T.sum())
g = jax.grad(f_eps)(jnp.asarray(12.25))
h = 1e-5
fd = (f_eps(jnp.asarray(12.25+h)) - f_eps(jnp.asarray(12.25-h)))/(2*h)
print(f"  d(sumT)/d(eps): AD={float(g):.10e}  FD={float(fd):.10e}  rel={abs(float(g)-float(fd))/max(abs(float(fd)),1e-300):.3e}", flush=True)

print("== 4) jax.grad w.r.t. depth and theta ==")
for name, fn, x0 in (
    ("depth", lambda d: jnp.real(pmm_jones_2d(Px,Py,jnp.asarray(build(12.25)),nsub,nsup,d,wl,degree=5,n_orders=3,region_layout=lay)[2].sum()), jnp.asarray(dep)),
    ("theta", lambda t: jnp.real(pmm_jones_2d(Px,Py,jnp.asarray(build(12.25)),nsub,nsup,dep,wl,theta=t,degree=5,n_orders=3,region_layout=lay)[2].sum()), jnp.asarray(0.30)),
):
    g = float(jax.grad(fn)(x0)); hh = float(x0)*1e-5 if float(x0)!=0 else 1e-7
    fd = float((fn(x0+hh)-fn(x0-hh))/(2*hh))
    print(f"  {name}: AD={g:.8e} FD={fd:.8e} rel={abs(g-fd)/max(abs(fd),1e-300):.3e} finite={np.isfinite(g)}", flush=True)

print("== 5) grad at EXACTLY normal incidence (branch-cut census claim) ==")
for t0 in (0.0, 1e-7, 1e-5):
    fn = lambda t: jnp.real(pmm_jones_2d(Px,Py,jnp.asarray(build(12.25)),nsub,nsup,dep,wl,theta=t,degree=5,n_orders=3,region_layout=lay)[2].sum())
    g = float(jax.grad(fn)(jnp.asarray(t0)))
    hh = 1e-7
    fd = float((fn(jnp.asarray(t0+hh))-fn(jnp.asarray(max(t0-hh,0.0))))/((t0+hh)-max(t0-hh,0.0)))
    print(f"  theta0={t0:g}: AD={g:+.6e} FD={fd:+.6e} finite={np.isfinite(g)}", flush=True)

print("== 6) recompilation per call? ==")
import time
c = jnp.asarray(build(12.25))
for i in range(3):
    t=time.time(); pmm_jones_2d(Px,Py,c,nsub,nsup,dep,wl,degree=5,n_orders=3,region_layout=lay); print(f"   call {i}: {time.time()-t:.3f}s", flush=True)

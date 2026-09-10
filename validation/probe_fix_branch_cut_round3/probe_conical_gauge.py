"""Is the conical-at-EXACTLY-normal gradient defect a branch-cut issue?"""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(_v,"1")
os.environ.setdefault("JAX_ENABLE_X64","true")
import numpy as np, jax, jax.numpy as jnp, lumenairy
jax.config.update("jax_enable_x64", True)
from lumenairy.elements.pmm import pmm_efficiency_2d
P, WL, DEP = 0.6e-6, 0.55e-6, 0.25e-6
XB = (0.2*P, 0.6*P)
print("lumenairy from", lumenairy.__file__)
def f(th, phi):
    o,R,T = pmm_efficiency_2d(P,P,jnp.asarray(6.0+0j),1.0,XB,XB,1.5,1.0,
                              jnp.asarray(DEP),WL,theta=th,phi=phi,
                              degree=5,n_orders=2,polarization="te")
    return jnp.sum(T)
for phi in (0.0, 0.7):
    for th0 in (0.0, 1e-8, 1e-5, 0.4):
        ad = float(jax.grad(lambda t: f(t, phi))(jnp.asarray(th0)))
        h = 1e-4
        fd = (float(f(jnp.asarray(th0+h),phi))-float(f(jnp.asarray(th0-h),phi)))/(2*h)
        print(f"  phi={phi}  theta0={th0:<8g} AD={ad: .6e} FD={fd: .6e} "
              f"rel={abs(ad-fd)/max(abs(fd),1e-300):.3e}")

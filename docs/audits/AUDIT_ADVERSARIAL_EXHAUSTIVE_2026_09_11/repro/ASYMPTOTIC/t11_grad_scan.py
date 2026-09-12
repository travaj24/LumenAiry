"""T11: is jax.grad wrt s2_image correct?  FD step scan."""
import sys
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import lumenairy as lm
from lumenairy.propagators.asymptotic import (
    fit_canonical_polynomials, aberration_tensor_lg00_jax,
    solve_envelope_stationary)

lam = 1.31e-6
rx = lm.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7', aperture=10e-3)
rx['object_distance'] = 0.1
fit = fit_canonical_polynomials(rx, lam, source_box_half=20e-6,
                                pupil_box_half=0.02, n_field=8, n_pupil=8,
                                poly_order=6)
vc = (fit.v2x_centre, fit.v2y_centre); w_s, w_p = 20e-6, 0.02
v, _, _ = solve_envelope_stationary(fit, (0.0, 0.0), (0.0, 0.0), w_s=w_s,
                                    w_p=w_p, v2_centre=vc)

def L_of(s2x):
    return aberration_tensor_lg00_jax(fit, (s2x, 0.0), v, source_point=(0.0, 0.0),
                                      w_s=w_s, w_p=w_p, v2_centre=vc)
def loss(s2x):
    return jnp.abs(L_of(s2x))**2

g = float(jax.grad(loss)(0.0))
print(f"jax.grad = {g:.10e}")
print(f"{'h [m]':>10} {'central FD':>18} {'rel diff':>12}")
for h in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11):
    fd = (float(loss(h)) - float(loss(-h)))/(2*h)
    print(f"{h:10.1e} {fd:18.10e} {abs(g-fd)/max(abs(fd),1e-300):12.3e}")

# is w_o (which depends on s2x via M) the culprit?  fix w_o explicitly
w_o_fixed = 1.011e-4
def loss_fixed(s2x):
    return jnp.abs(aberration_tensor_lg00_jax(
        fit, (s2x, 0.0), v, source_point=(0.0, 0.0), w_s=w_s, w_p=w_p,
        w_o=w_o_fixed, v2_centre=vc))**2
g2 = float(jax.grad(loss_fixed)(0.0))
print(f"\nwith explicit w_o: jax.grad = {g2:.10e}")
for h in (1e-6, 1e-7, 1e-8, 1e-9):
    fd = (float(loss_fixed(h)) - float(loss_fixed(-h)))/(2*h)
    print(f"{h:10.1e} {fd:18.10e} {abs(g2-fd)/max(abs(fd),1e-300):12.3e}")

# component-wise: grad of Re and Im of L
def reL(s2x): return jnp.real(L_of(s2x))
def imL(s2x): return jnp.imag(L_of(s2x))
gr = float(jax.grad(reL)(0.0)); gi = float(jax.grad(imL)(0.0))
for h in (1e-7, 1e-8, 1e-9):
    fdr = (float(reL(h))-float(reL(-h)))/(2*h)
    fdi = (float(imL(h))-float(imL(-h)))/(2*h)
    print(f"h={h:.0e}  dRe: grad {gr:.6e} fd {fdr:.6e} rel {abs(gr-fdr)/abs(fdr):.2e}"
          f"   dIm: grad {gi:.6e} fd {fdi:.6e} rel {abs(gi-fdi)/abs(fdi):.2e}")

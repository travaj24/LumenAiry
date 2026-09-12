import sys
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import jax; jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import lumenairy as lm
from lumenairy.propagators.asymptotic import (
    fit_canonical_polynomials, aberration_tensor_lg00_jax,
    solve_envelope_stationary, _compute_M_b)

lam = 1.31e-6
rx = lm.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7', aperture=10e-3)
rx['object_distance'] = 0.1
fit = fit_canonical_polynomials(rx, lam, source_box_half=20e-6,
                                pupil_box_half=0.02, n_field=8, n_pupil=8, poly_order=6)
vc = (fit.v2x_centre, fit.v2y_centre); w_s, w_p = 20e-6, 0.02
v, _, _ = solve_envelope_stationary(fit, (0.0,0.0), (0.0,0.0), w_s=w_s, w_p=w_p, v2_centre=vc)
M, b, s1s, J, phis, G0, detJ = _compute_M_b(fit, 0.0, 0.0, v[0], v[1], 0.0, 0.0,
                                            w_s, w_p, vc[0], vc[1])
ev = np.linalg.eigvalsh(np.real(M))
print("Re M eigenvalues:", ev, " gap/mean:", (ev[1]-ev[0])/ev.mean())
print("Re M =\n", np.real(M))

w_o_fixed = 1.0/np.sqrt(ev[-1])
def loss_fixed(s2x):
    return jnp.abs(aberration_tensor_lg00_jax(
        fit, (s2x, 0.0), v, source_point=(0.0,0.0), w_s=w_s, w_p=w_p,
        w_o=w_o_fixed, v2_centre=vc))**2
g2 = float(jax.grad(loss_fixed)(0.0))
# 5-point FD
for h in (3e-6, 1e-6, 3e-7):
    f = lambda t: float(loss_fixed(t))
    fd5 = (-f(2*h) + 8*f(h) - 8*f(-h) + f(-2*h))/(12*h)
    print(f"explicit w_o  h={h:.0e}: grad {g2:.8e}  5pt-FD {fd5:.8e} rel {abs(g2-fd5)/abs(fd5):.3e}")

# Now default w_o (eigvalsh inside the graph)
def loss_def(s2x):
    return jnp.abs(aberration_tensor_lg00_jax(
        fit, (s2x, 0.0), v, source_point=(0.0,0.0), w_s=w_s, w_p=w_p,
        v2_centre=vc))**2
g1 = float(jax.grad(loss_def)(0.0))
for h in (3e-6, 1e-6, 3e-7):
    f = lambda t: float(loss_def(t))
    fd5 = (-f(2*h) + 8*f(h) - 8*f(-h) + f(-2*h))/(12*h)
    print(f"default  w_o  h={h:.0e}: grad {g1:.8e}  5pt-FD {fd5:.8e} rel {abs(g1-fd5)/abs(fd5):.3e}")

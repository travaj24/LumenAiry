"""T13: jax.grad of the LG00 Strehl merit wrt every documented
differentiable slot, default w_o (eigvalsh in graph) vs explicit w_o."""
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
vc = (fit.v2x_centre, fit.v2y_centre); W_S, W_P = 20e-6, 0.02
v, _, _ = solve_envelope_stationary(fit, (0.0,0.0), (0.0,0.0), w_s=W_S, w_p=W_P, v2_centre=vc)
M, *_ = _compute_M_b(fit, 0.0, 0.0, v[0], v[1], 0.0, 0.0, W_S, W_P, vc[0], vc[1])
WO = 1.0/np.sqrt(np.linalg.eigvalsh(np.real(M))[-1])

def merit(s2x, ws, wp, vx, srcx, wo):
    kw = dict(source_point=(srcx, 0.0), w_s=ws, w_p=wp, v2_centre=vc)
    if wo is not None:
        kw['w_o'] = wo
    return 1.0 - jnp.abs(aberration_tensor_lg00_jax(
        fit, (s2x, 0.0), (vx, v[1]), **kw))**2

base = dict(s2x=0.0, ws=W_S, wp=W_P, vx=v[0], srcx=0.0)
names = ['s2x', 'ws', 'wp', 'vx', 'srcx']
steps = {'s2x': 1e-6, 'ws': W_S*1e-4, 'wp': W_P*1e-4, 'vx': 1e-5, 'srcx': 1e-7}
for wo_label, wo in (('default w_o (eigvalsh)', None), ('explicit w_o', WO)):
    print(f"\n--- {wo_label} ---")
    for i, nm in enumerate(names):
        def f(t, i=i):
            a = [base[k] for k in names]; a[i] = t
            return merit(*a, wo)
        g = float(jax.grad(f)(base[nm]))
        h = steps[nm]
        ff = lambda t: float(f(t))
        t0 = base[nm]
        fd5 = (-ff(t0+2*h) + 8*ff(t0+h) - 8*ff(t0-h) + ff(t0-2*h))/(12*h)
        rel = abs(g-fd5)/max(abs(fd5), 1e-300)
        flag = "   <-- WRONG" if rel > 1e-2 else ""
        print(f"  d/d{nm:5s}: grad {g:15.8e}   5pt-FD {fd5:15.8e}   rel {rel:.3e}{flag}")

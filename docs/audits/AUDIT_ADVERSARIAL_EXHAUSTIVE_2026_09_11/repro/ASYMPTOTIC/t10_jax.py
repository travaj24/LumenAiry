"""T10: JAX twin -- parity with NumPy, x64 enforcement, grad vs FD,
and the missing validity guards."""
import sys, time, math, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import lumenairy as lm
from lumenairy.propagators.asymptotic import (
    fit_canonical_polynomials, propagate_modal_asymptotic,
    aberration_tensor, aberration_tensor_lg00_jax,
    propagate_modal_asymptotic_lg00_jax, solve_envelope_stationary,
    solve_envelope_stationary_jax_ift, _solve_envelope_stationary_batch)

lam = 1.31e-6
rx = lm.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7', aperture=10e-3)
rx['object_distance'] = 0.1
fit = fit_canonical_polynomials(rx, lam, source_box_half=20e-6,
                                pupil_box_half=0.02, n_field=8, n_pupil=8,
                                poly_order=6)
vc = (fit.v2x_centre, fit.v2y_centre)
w_s, w_p = 20e-6, 0.02

# ---- 1. aberration_tensor(0,0) NumPy vs JAX -----------------------
s2 = (0.0, 0.0)
v, _, _ = solve_envelope_stationary(fit, s2, (0.0, 0.0), w_s=w_s, w_p=w_p,
                                    v2_centre=vc)
Lnp = aberration_tensor(fit, s2, source_point=(0.0, 0.0),
                        output_modes=[(0, 0)], w_s=w_s, w_p=w_p,
                        v2_centre=vc).L[0, 0]
Ljx = complex(aberration_tensor_lg00_jax(fit, s2, v, source_point=(0.0, 0.0),
                                         w_s=w_s, w_p=w_p, v2_centre=vc))
print(f"aberration_tensor (0,0):  numpy {Lnp:.10e}\n"
      f"                          jax   {Ljx:.10e}\n"
      f"   rel {abs(Lnp-Ljx)/abs(Lnp):.3e}")

# ---- 2. propagate parity over a grid -------------------------------
n = 17; L = fit.s2x_halfrange*0.4
ax = np.linspace(-L, L, n)+fit.s2x_centre; ay = np.linspace(-L, L, n)+fit.s2y_centre
X, Y = np.meshgrid(ax, ay, indexing='xy')
Enp = propagate_modal_asymptotic(fit, w_s=w_s, w_p=w_p, v2_centre=vc,
                                 s2_grid_x=X, s2_grid_y=Y)
vx, vy, _ = _solve_envelope_stationary_batch(fit, X.ravel(), Y.ravel(), 0.0, 0.0,
                                             w_s=w_s, w_p=w_p, v_cx=vc[0], v_cy=vc[1])
vg = np.stack([vx.reshape(X.shape), vy.reshape(X.shape)], axis=-1)
Ejx = np.asarray(propagate_modal_asymptotic_lg00_jax(
    fit, X, Y, vg, w_s=w_s, w_p=w_p, v2_centre=vc))
m = np.abs(Enp) > 0
print(f"\npropagate parity (in-box pixels): RMS rel "
      f"{np.sqrt(np.mean(np.abs(Ejx[m]-Enp[m])**2))/np.sqrt(np.mean(np.abs(Enp[m])**2)):.3e}"
      f"  worst {np.max(np.abs(Ejx[m]-Enp[m])/np.abs(Enp[m])):.3e}")
print(f"   numpy zeroed pixels: {(~m).sum()} ; jax value there: "
      f"{np.abs(Ejx[~m])}" if (~m).any() else "   (all pixels in box)")

# ---- 3. grid that leaves the fit box (guard parity) -----------------
Lb = fit.s2x_halfrange*3.0
axb = np.linspace(-Lb, Lb, 9)+fit.s2x_centre
Xb, Yb = np.meshgrid(axb, axb, indexing='xy')
Enp_b = propagate_modal_asymptotic(fit, w_s=w_s, w_p=w_p, v2_centre=vc,
                                   s2_grid_x=Xb, s2_grid_y=Yb)
vxb, vyb, _ = _solve_envelope_stationary_batch(fit, Xb.ravel(), Yb.ravel(), 0.0, 0.0,
                                               w_s=w_s, w_p=w_p, v_cx=vc[0], v_cy=vc[1])
vgb = np.stack([vxb.reshape(Xb.shape), vyb.reshape(Xb.shape)], axis=-1)
Ejx_b = np.asarray(propagate_modal_asymptotic_lg00_jax(
    fit, Xb, Yb, vgb, w_s=w_s, w_p=w_p, v2_centre=vc))
out = np.abs(Enp_b) == 0
print(f"\nout-of-box grid: numpy zeroes {out.sum()}/{out.size} pixels; "
      f"max |E_jax| on those pixels = {np.nanmax(np.abs(Ejx_b[out])):.6e}, "
      f"n_nonfinite = {int((~np.isfinite(Ejx_b[out])).sum())}")

# ---- 4. jax.grad vs central finite differences ---------------------
def loss(s2x):
    return jnp.abs(aberration_tensor_lg00_jax(
        fit, (s2x, 0.0), v, source_point=(0.0, 0.0),
        w_s=w_s, w_p=w_p, v2_centre=vc))**2
g = float(jax.grad(loss)(0.0))
h = 1e-7
fd = (float(loss(h)) - float(loss(-h)))/(2*h)
print(f"\njax.grad d|L|^2/ds2x = {g:.8e}   central FD = {fd:.8e}   "
      f"rel {abs(g-fd)/max(abs(fd),1e-300):.3e}")

# grad wrt w_s
def loss_ws(ws):
    return jnp.abs(aberration_tensor_lg00_jax(
        fit, s2, v, source_point=(0.0, 0.0), w_s=ws, w_p=w_p, v2_centre=vc))**2
g2 = float(jax.grad(loss_ws)(w_s))
h2 = w_s*1e-6
fd2 = (float(loss_ws(w_s+h2)) - float(loss_ws(w_s-h2)))/(2*h2)
print(f"jax.grad d|L|^2/dw_s = {g2:.8e}   central FD = {fd2:.8e}   "
      f"rel {abs(g2-fd2)/max(abs(fd2),1e-300):.3e}")

# ---- 5. IFT solver vs numpy solve ----------------------------------
vj = np.asarray(solve_envelope_stationary_jax_ift(
    fit, (0.0, 0.0), (0.0, 0.0), w_s=w_s, w_p=w_p, v2_centre=vc))
print(f"\nIFT solver v* {vj}   numpy v* {np.asarray(v)}  "
      f"diff {np.max(np.abs(vj-np.asarray(v))):.3e}")

# ---- 6. recompilation / timing -------------------------------------
t = time.perf_counter(); _ = propagate_modal_asymptotic_lg00_jax(
    fit, X, Y, vg, w_s=w_s, w_p=w_p, v2_centre=vc); t1 = time.perf_counter()-t
t = time.perf_counter(); _ = propagate_modal_asymptotic_lg00_jax(
    fit, X, Y, vg, w_s=w_s, w_p=w_p, v2_centre=vc); t2 = time.perf_counter()-t
t = time.perf_counter(); _ = propagate_modal_asymptotic(
    fit, w_s=w_s, w_p=w_p, v2_centre=vc, s2_grid_x=X, s2_grid_y=Y); t3 = time.perf_counter()-t
print(f"\ntiming 17x17: jax 1st {t1*1e3:.1f} ms  2nd {t2*1e3:.1f} ms   numpy {t3*1e3:.1f} ms")

"""T7: brute-force quadrature of the phase-space v2 integral the
propagator claims to evaluate, with and without the extracted linear
phase, vs propagate_modal_asymptotic."""
import sys, math
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as lm
from lumenairy.propagators.asymptotic import (
    fit_canonical_polynomials, propagate_modal_asymptotic,
    solve_envelope_stationary, _compute_M_b_batch)

lam = 1.31e-6
def base():
    r = lm.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7', aperture=10e-3)
    r['object_distance'] = 0.1
    return r

def brute(fit, s2, src, w_s, w_p, vc, include_linear, nq=401, K=8.0):
    s2x, s2y = s2
    v, _, _ = solve_envelope_stationary(fit, (s2x, s2y), src, w_s=w_s, w_p=w_p,
                                        v2_centre=vc)
    vsx, vsy = v
    M, b, s1s, J, phis, G0, detJ = _compute_M_b_batch(
        fit, np.array([s2x]), np.array([s2y]), np.array([vsx]), np.array([vsy]),
        src[0], src[1], w_s, w_p, vc[0], vc[1])
    R = np.real(M[0])
    ev = np.linalg.eigvalsh(R)
    sig = 1.0/np.sqrt(max(ev.min(), 1e-30))   # widest Gaussian 1/e width
    half = K*sig
    g = np.linspace(-half, half, nq)
    GX, GY = np.meshgrid(vsx+g, vsy+g, indexing='xy')
    s1x, s1y = fit.eval_s1(np.full_like(GX, s2x), np.full_like(GX, s2y), GX, GY)
    phi = fit.eval_phi(np.full_like(GX, s2x), np.full_like(GX, s2y), GX, GY,
                       include_linear=include_linear)
    Ns = math.sqrt(2.0/(math.pi*w_s*w_s)); Np = math.sqrt(2.0/(math.pi*w_p*w_p))
    env = np.exp(-((s1x-src[0])**2+(s1y-src[1])**2)/(w_s*w_s)
                 -((GX-vc[0])**2+(GY-vc[1])**2)/(w_p*w_p))
    integ = float(detJ[0])*Ns*Np*env*np.exp(2j*np.pi*phi)
    dv = (g[1]-g[0])**2
    return integ.sum()*dv, sig

for label, fitkw, src in [
    ("A on-axis", {}, (0.0, 0.0)),
    ("B source_centre=100um", dict(source_centre=(100e-6, 0.0)), (100e-6, 0.0)),
]:
    fit = fit_canonical_polynomials(base(), lam, source_box_half=20e-6,
                                    pupil_box_half=0.02, n_field=8, n_pupil=8,
                                    poly_order=6, **fitkw)
    vc = (fit.v2x_centre, fit.v2y_centre)
    w_s, w_p = 20e-6, 0.02
    # pick the pixel where the code has its peak
    n = 21; L = fit.s2x_halfrange*0.5
    ax = np.linspace(-L, L, n)+fit.s2x_centre; ay = np.linspace(-L, L, n)+fit.s2y_centre
    X, Y = np.meshgrid(ax, ay, indexing='xy')
    Efull = propagate_modal_asymptotic(fit, source_point=src, w_s=w_s, w_p=w_p,
                                       v2_centre=vc, s2_grid_x=X, s2_grid_y=Y)
    k = np.unravel_index(np.argmax(np.abs(Efull)), Efull.shape)
    s2 = (float(X[k]), float(Y[k]))
    E_code = complex(Efull[k])
    bf_res, sig = brute(fit, s2, src, w_s, w_p, vc, include_linear=False)
    bf_full, _ = brute(fit, s2, src, w_s, w_p, vc, include_linear=True)
    bf_res2, _ = brute(fit, s2, src, w_s, w_p, vc, include_linear=False, nq=601, K=10.0)
    bf_full2, _ = brute(fit, s2, src, w_s, w_p, vc, include_linear=True, nq=601, K=10.0)
    print(f"\n{label}   pixel s2={s2}   a3={fit.linear_coeffs_phi[3]:.4e} waves")
    print(f"   sigma_v = {sig:.3e}")
    print(f"   code                              |E| = {abs(E_code):.6e}")
    print(f"   brute force, include_linear=False |E| = {abs(bf_res):.6e}"
          f"   (converged: {abs(bf_res2):.6e})")
    print(f"   brute force, include_linear=True  |E| = {abs(bf_full):.6e}"
          f"   (converged: {abs(bf_full2):.6e})")
    print(f"   code / bf(False) = {abs(E_code)/abs(bf_res):.6f}"
          f"   |  code / bf(True) = {abs(E_code)/max(abs(bf_full),1e-300):.6e}")

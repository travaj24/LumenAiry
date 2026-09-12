"""T17: Maslov branch claims + robustness/edge cases."""
import sys, math, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as lm
from lumenairy.propagators.asymptotic import (
    fit_canonical_polynomials, propagate_modal_asymptotic,
    _solve_envelope_stationary_batch, _compute_M_b_batch,
    _maslov_branch_corrected_sqrt)

lam = 1.31e-6
rx = lm.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7', aperture=10e-3)
rx['object_distance'] = 0.1
fit = fit_canonical_polynomials(rx, lam, source_box_half=20e-6,
                                pupil_box_half=0.02, n_field=8, n_pupil=8, poly_order=6)
vc = (fit.v2x_centre, fit.v2y_centre)

# ---- 1. arg det M identity + positive definiteness of Re M ----------
for (w_s, w_p) in ((20e-6, 0.02), (1e-3, 0.1), (1e-2, 1.0)):
    n = 33; L = fit.s2x_halfrange*0.8
    a = np.linspace(-L, L, n)
    X, Y = np.meshgrid(a+fit.s2x_centre, a+fit.s2y_centre, indexing='xy')
    vx, vy, _ = _solve_envelope_stationary_batch(fit, X.ravel(), Y.ravel(), 0.0, 0.0,
                                                 w_s=w_s, w_p=w_p, v_cx=vc[0], v_cy=vc[1])
    M, *_ = _compute_M_b_batch(fit, X.ravel(), Y.ravel(), vx, vy, 0.0, 0.0,
                               w_s, w_p, vc[0], vc[1])
    R = np.real(M); Hp = -np.imag(M)/math.pi   # M = R - i pi H
    det = M[:,0,0]*M[:,1,1]-M[:,0,1]*M[:,1,0]
    args = np.angle(det)
    ident = []
    mineig = []
    for i in range(M.shape[0]):
        r = R[i]
        ev = np.linalg.eigvalsh(r); mineig.append(ev.min())
        if ev.min() <= 0: continue
        rs = np.linalg.cholesky(r)
        Ri = np.linalg.inv(rs)
        Kk = Ri @ (math.pi*Hp[i]) @ Ri.T
        kk = np.linalg.eigvalsh(Kk)
        ident.append(abs(args[i] - (-math.atan(kk[0])-math.atan(kk[1]))))
    print(f"w_s={w_s:.0e} w_p={w_p:.2g}: min eig Re M = {min(mineig):.3e}, "
          f"|arg detM| max = {np.abs(args).max():.6f} (pi={math.pi:.4f}), "
          f"identity max dev = {max(ident) if ident else float('nan'):.2e}")

# ---- 2. legacy maslov modes: does the warning fire? -----------------
w_s, w_p = 1e-3, 0.1
n = 65; L = fit.s2x_halfrange*0.95
a = np.linspace(-L, L, n)
X, Y = np.meshgrid(a+fit.s2x_centre, a+fit.s2y_centre, indexing='xy')
E0 = propagate_modal_asymptotic(fit, w_s=w_s, w_p=w_p, v2_centre=vc,
                                s2_grid_x=X, s2_grid_y=Y)
for mode in ('1d_raster', 'row_reset'):
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        Em = propagate_modal_asymptotic(fit, w_s=w_s, w_p=w_p, v2_centre=vc,
                                        s2_grid_x=X, s2_grid_y=Y,
                                        maslov_tracking=mode)
    nz = np.abs(E0) > 0
    flips = int(np.sum(np.abs(Em[nz]+E0[nz]) < 1e-12*np.abs(E0[nz])))
    print(f"  {mode:10s}: warnings={len(wl)} flipped(sign-inverted) pixels={flips}"
          f" of {int(nz.sum())}, max|dE|/max|E|="
          f"{float(np.max(np.abs(Em-E0))/np.max(np.abs(E0))):.4f}")

# ---- 3. edge cases --------------------------------------------------
print("\nedge cases:")
try:
    r = propagate_modal_asymptotic(fit, w_s=20e-6, w_p=0.02, v2_centre=vc,
                                   s2_grid_x=np.float64(0.0), s2_grid_y=np.float64(0.0))
    print("  0-D grid ->", type(r), np.shape(r), r)
except Exception as e:
    print("  0-D grid raised", type(e).__name__, e)
try:
    r = propagate_modal_asymptotic(fit, w_s=20e-6, w_p=0.02, v2_centre=vc,
                                   s2_grid_x=np.array([np.nan, 0.0]),
                                   s2_grid_y=np.array([0.0, 0.0]))
    print("  NaN pixel ->", r)
except Exception as e:
    print("  NaN pixel raised", type(e).__name__, e)
try:
    r = propagate_modal_asymptotic(fit, w_s=0.0, w_p=0.02, v2_centre=vc,
                                   s2_grid_x=np.array([0.0]), s2_grid_y=np.array([0.0]))
    print("  w_s=0 ->", r)
except Exception as e:
    print("  w_s=0 raised", type(e).__name__, e)
try:
    r = propagate_modal_asymptotic(fit, w_s=-20e-6, w_p=0.02, v2_centre=vc,
                                   s2_grid_x=np.array([0.0]), s2_grid_y=np.array([0.0]))
    print("  w_s<0 ->", r)
except Exception as e:
    print("  w_s<0 raised", type(e).__name__, e)
try:
    r = propagate_modal_asymptotic(fit, w_s=20e-6, w_p=0.02, v2_centre=vc,
                                   source_amplitudes={(-1, 0): 1.0},
                                   s2_grid_x=np.array([0.0]), s2_grid_y=np.array([0.0]))
    print("  p=-1 source mode ->", r)
except Exception as e:
    print("  p=-1 source mode raised", type(e).__name__, e)

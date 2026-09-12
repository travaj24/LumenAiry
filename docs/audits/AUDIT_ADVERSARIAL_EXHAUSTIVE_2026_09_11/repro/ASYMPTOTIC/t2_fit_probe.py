"""T2: build a canonical fit, probe det J variation + basic structure."""
import sys, time
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as lm
from lumenairy.propagators.asymptotic import (
    fit_canonical_polynomials, propagate_modal_asymptotic,
    _solve_envelope_stationary_batch, _compute_M_b_batch,
)

def make_rx(R1=20e-3, R2=-20e-3, d=2e-3, obj=0.1):
    rx = lm.make_singlet(R1=R1, R2=R2, d=d, glass='N-BK7', aperture=10e-3)
    rx['object_distance'] = obj
    return rx

t0 = time.time()
lam = 1.31e-6
rx = make_rx()
fit = fit_canonical_polynomials(rx, wavelength=lam,
                                source_box_half=20e-6, pupil_box_half=0.02,
                                n_field=8, n_pupil=8, poly_order=6)
print(f"fit built in {time.time()-t0:.2f}s  res_phi={fit.res_phi_rms_waves:.3e} waves  "
      f"res_s1={fit.res_s1_rms_m:.3e} m  n_rays={fit.n_rays}")
print("s2x c/h:", fit.s2x_centre, fit.s2x_halfrange)
print("s2y c/h:", fit.s2y_centre, fit.s2y_halfrange)
print("v2x c/h:", fit.v2x_centre, fit.v2x_halfrange)
print("v2y c/h:", fit.v2y_centre, fit.v2y_halfrange)
print("linear coeffs:", fit.linear_coeffs_phi)

# output grid
n = 33
L = fit.s2x_halfrange*0.3
ax = np.linspace(-L, L, n) + fit.s2x_centre
ay = np.linspace(-L, L, n) + fit.s2y_centre
X, Y = np.meshgrid(ax, ay, indexing='xy')
w_s, w_p = 20e-6, 0.02
vx, vy, conv = _solve_envelope_stationary_batch(
    fit, X.ravel(), Y.ravel(), 0.0, 0.0, w_s=w_s, w_p=w_p,
    v_cx=fit.v2x_centre, v_cy=fit.v2y_centre)
M, b, s1s, J, phis, G0, detJ = _compute_M_b_batch(
    fit, X.ravel(), Y.ravel(), vx, vy, 0.0, 0.0, w_s, w_p,
    fit.v2x_centre, fit.v2y_centre)
E = propagate_modal_asymptotic(fit, source_point=(0.0,0.0), w_s=w_s, w_p=w_p,
                               v2_centre=(fit.v2x_centre, fit.v2y_centre),
                               s2_grid_x=X, s2_grid_y=Y)
amp = np.abs(E).ravel()
nz = amp > 0
print("\nnon-zero pixels:", nz.sum(), "of", amp.size)
dJ = detJ[nz]
print("detJ  min/max/ratio :", dJ.min(), dJ.max(), dJ.max()/dJ.min())
# amplitude-weighted spread
wgt = amp[nz]/amp[nz].sum()
mean = (wgt*dJ).sum()
print("detJ amplitude-weighted mean:", mean, " rel ptp:", (dJ.max()-dJ.min())/mean)
print("sqrt(detJ) ratio max/min:", np.sqrt(dJ.max()/dJ.min()))
# where is the brightest?
imax = np.argmax(amp)
print("brightest pixel detJ:", detJ[imax], " |E|:", amp[imax])
# how much does detJ vary over the *bright* half?
bright = amp > 0.5*amp.max()
print("over |E|>0.5max:", bright.sum(), "pixels, detJ ratio",
      detJ[bright].max()/detJ[bright].min(),
      " sqrt ratio", np.sqrt(detJ[bright].max()/detJ[bright].min()))
np.save(r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/ASYMPTOTIC/E33.npy", E)

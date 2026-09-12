"""T9: validate the proposed fix -- keep a3,a4 (the v2-linear phase)
inside the integral while still dropping the s2 piston/tilt a0,a1,a2."""
import sys
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as lm
from lumenairy.propagators.asymptotic import (
    fit_canonical_polynomials, propagate_modal_asymptotic, CanonicalPolyFit)
from lumenairy.elements.lenses import _evaluate_polynomial_4d_and_grad34

_orig = CanonicalPolyFit.eval_phi_with_v2_grad
def patched(self, s2x, s2y, v2x, v2y, *, include_linear=False):
    u1, u2, u3, u4 = self.to_normalised(s2x, s2y, v2x, v2y)
    phi, d3, d4 = _evaluate_polynomial_4d_and_grad34(
        self.coef_phi, self.multi_indices, u1, u2, u3, u4, self.poly_order)
    if self.linear_coeffs_phi is not None:
        a0, a1, a2, a3, a4 = self.linear_coeffs_phi
        if include_linear:
            phi = phi + (a0 + a1*u1 + a2*u2)
        # v2-linear part is INSIDE the integral -- always keep it.
        phi = phi + a3*u3 + a4*u4
        d3 = d3 + a3
        d4 = d4 + a4
    return phi, d3/self.v2x_halfrange, d4/self.v2y_halfrange

lam = 1.31e-6
rx = lm.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7', aperture=10e-3)
rx['object_distance'] = 0.1
src = (100e-6, 0.0)
grid_kw = dict(w_s=20e-6, w_p=0.02)

def run(flag, patch):
    fit = fit_canonical_polynomials(rx, lam, source_box_half=20e-6,
                                    pupil_box_half=0.02, n_field=8, n_pupil=8,
                                    poly_order=6, source_centre=src,
                                    extract_linear_phase=flag)
    n = 161; L = fit.s2x_halfrange*0.9
    ax = np.linspace(-L, L, n)+fit.s2x_centre; ay = np.linspace(-L, L, n)+fit.s2y_centre
    X, Y = np.meshgrid(ax, ay, indexing='xy')
    CanonicalPolyFit.eval_phi_with_v2_grad = patched if patch else _orig
    try:
        E = propagate_modal_asymptotic(fit, source_point=src,
                                       v2_centre=(fit.v2x_centre, fit.v2y_centre),
                                       s2_grid_x=X, s2_grid_y=Y, **grid_kw)
    finally:
        CanonicalPolyFit.eval_phi_with_v2_grad = _orig
    return X, Y, E

for label, flag, patch in [("default (extract=True, as shipped)", True, False),
                            ("extract=False reference", False, False),
                            ("extract=True + v2-linear restored (FIX)", True, True)]:
    X, Y, E = run(flag, patch)
    A = np.abs(E); k = np.unravel_index(np.argmax(A), A.shape)
    print(f"{label:42s} peak={A.max():.6e} at ({float(X[k]):.5e},{float(Y[k]):.5e})")

# shape comparison fix vs extract=False
X, Y, Eref = run(False, False)
_, _, Efix = run(True, True)
m = (np.abs(Eref) > 0.02*np.abs(Eref).max())
r = np.abs(Efix[m])/np.abs(Eref[m])
print(f"\nFIX vs extract=False over bright pixels: |E| ratio max {r.max():.6f} "
      f"min {r.min():.6f}   (n={m.sum()})")

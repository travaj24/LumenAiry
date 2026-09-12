"""T8: for the off-axis source_centre fit, which of extract=True/False
puts the PSF where the chief ray actually lands?"""
import sys
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as lm
from lumenairy.propagators.asymptotic import (
    fit_canonical_polynomials, propagate_modal_asymptotic)
from lumenairy.raytrace import _make_bundle, surfaces_from_prescription, trace

lam = 1.31e-6
rx = lm.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7', aperture=10e-3)
rx['object_distance'] = 0.1
src = (100e-6, 0.0)

# ---- independent oracle: chief-ray landing of the off-axis source ----
surfaces = surfaces_from_prescription(rx)
b = _make_bundle(x=np.array([src[0]]), y=np.array([src[1]]),
                 L=np.array([0.0]), M=np.array([0.0]), wavelength=lam)
b.z = np.full(1, -0.1)
r = trace(b, surfaces, lam, output_filter='last')
print("chief-ray landing (independent ray trace):",
      float(r.image_rays.x[0]), float(r.image_rays.y[0]))

for flag in (True, False):
    fit = fit_canonical_polynomials(rx, lam, source_box_half=20e-6,
                                    pupil_box_half=0.02, n_field=8, n_pupil=8,
                                    poly_order=6, source_centre=src,
                                    extract_linear_phase=flag)
    n = 161; L = fit.s2x_halfrange*0.9
    ax = np.linspace(-L, L, n)+fit.s2x_centre
    ay = np.linspace(-L, L, n)+fit.s2y_centre
    X, Y = np.meshgrid(ax, ay, indexing='xy')
    E = propagate_modal_asymptotic(fit, source_point=src, w_s=20e-6, w_p=0.02,
                                   v2_centre=(fit.v2x_centre, fit.v2y_centre),
                                   s2_grid_x=X, s2_grid_y=Y)
    A = np.abs(E)
    k = np.unravel_index(np.argmax(A), A.shape)
    tot = A.sum()
    cx = float((A*X).sum()/tot); cy = float((A*Y).sum()/tot)
    print(f"extract_linear_phase={flag!s:5s}  peak |E|={A.max():.5e} at "
          f"({float(X[k]):.6e}, {float(Y[k]):.6e})   centroid ({cx:.3e},{cy:.3e})")

"""T6: extract_linear_phase=True (default) vs False on the SAME system.
Both fits describe the same physical Phi; the propagator reads Phi with
include_linear=False, so any v2-linear content removed by the prefit is
LOST.  W6-A4 claims the difference is phase-only."""
import sys
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as lm
from lumenairy.propagators.asymptotic import (
    fit_canonical_polynomials, propagate_modal_asymptotic)

lam = 1.31e-6
def base(ap=10e-3, obj=0.1):
    r = lm.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7', aperture=ap)
    r['object_distance'] = obj
    return r

def compare(label, **fitkw):
    kwA = dict(source_box_half=20e-6, pupil_box_half=0.02, n_field=8,
               n_pupil=8, poly_order=6)
    kwA.update(fitkw)
    fT = fit_canonical_polynomials(base(), lam, extract_linear_phase=True, **kwA)
    fF = fit_canonical_polynomials(base(), lam, extract_linear_phase=False, **kwA)
    a = fT.linear_coeffs_phi
    n = 21
    L = fT.s2x_halfrange*0.5
    ax = np.linspace(-L, L, n) + fT.s2x_centre
    ay = np.linspace(-L, L, n) + fT.s2y_centre
    X, Y = np.meshgrid(ax, ay, indexing='xy')
    kw = dict(source_point=(fitkw.get('source_centre', (0.0, 0.0))),
              w_s=20e-6, w_p=0.02,
              v2_centre=(fT.v2x_centre, fT.v2y_centre),
              s2_grid_x=X, s2_grid_y=Y)
    ET = propagate_modal_asymptotic(fT, **kw)
    EF = propagate_modal_asymptotic(fF, **kw)
    m = (np.abs(ET) > 0) | (np.abs(EF) > 0)
    aT, aF = np.abs(ET[m]), np.abs(EF[m])
    peakT, peakF = aT.max(), aF.max()
    print(f"{label}")
    print(f"   res T/F = {fT.res_phi_rms_waves:.3e} / {fF.res_phi_rms_waves:.3e} waves")
    print(f"   a1={a[1]:.4e}  a2={a[2]:.4e}  a3={a[3]:.4e}  a4={a[4]:.4e}  waves")
    print(f"   peak|E| extract=True {peakT:.6e}   extract=False {peakF:.6e}   "
          f"ratio {peakT/max(peakF,1e-300):.6g}")
    both = (np.abs(ET) > 0.05*peakT) & (np.abs(EF) > 0.05*peakF)
    if both.any():
        r = (np.abs(ET[both])/peakT)/(np.abs(EF[both])/peakF)
        print(f"   normalised-shape ratio over bright pixels: "
              f"max {r.max():.6g} min {r.min():.6g}  (1.0 = same shape)")
    print(f"   alive pixels T/F: {(np.abs(ET)>0).sum()} / {(np.abs(EF)>0).sum()}")

compare("A. on-axis, no grating")
compare("B. source_centre=(100 um, 0)", source_centre=(100e-6, 0.0))
compare("C. grating surf 0, m=1, 2 um",
        surface_diffraction={0: (1.0, 0.0, 2e-6, 2e-6)})
compare("D. grating surf 1, m=1, 2 um",
        surface_diffraction={1: (1.0, 0.0, 2e-6, 2e-6)})

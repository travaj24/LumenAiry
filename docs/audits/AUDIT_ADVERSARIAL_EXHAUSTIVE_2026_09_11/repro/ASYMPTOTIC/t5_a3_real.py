"""T5: how big is the v2-linear prefit coefficient a3/a4 on REAL fits?
(The W6-A4 note claims |a3|+|a4| <= 1.7e-9 waves 'on every case
measured', so the removal is 'phase-only'.)"""
import sys
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as lm
from lumenairy.propagators.asymptotic import (
    fit_canonical_polynomials, propagate_modal_asymptotic,
)

lam = 1.31e-6
def rx_singlet(obj=0.1, ap=10e-3):
    r = lm.make_singlet(R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7', aperture=ap)
    r['object_distance'] = obj
    return r

def show(name, fit):
    a = fit.linear_coeffs_phi
    print(f"{name:44s} a0={a[0]:12.5e} a1={a[1]:11.4e} a2={a[2]:11.4e} "
          f"a3={a[3]:11.4e} a4={a[4]:11.4e}  (waves)")
    return a

cases = {}
f = fit_canonical_polynomials(rx_singlet(), lam, source_box_half=20e-6,
                              pupil_box_half=0.02, n_field=8, n_pupil=8, poly_order=6)
cases['on-axis, source_centre=(0,0)'] = f
show('on-axis, source_centre=(0,0)', f)

for cx in (100e-6, 500e-6, 2e-3):
    f = fit_canonical_polynomials(rx_singlet(), lam, source_box_half=20e-6,
                                  pupil_box_half=0.02, n_field=8, n_pupil=8,
                                  poly_order=6, source_centre=(cx, 0.0))
    cases[f'source_centre=({cx*1e6:.0f} um, 0)'] = f
    show(f'source_centre=({cx*1e6:.0f} um, 0)', f)

# grating on surface 1, 1st order, 2 um period  (the W6-A4 case)
try:
    f = fit_canonical_polynomials(rx_singlet(), lam, source_box_half=20e-6,
                                  pupil_box_half=0.02, n_field=8, n_pupil=8,
                                  poly_order=6,
                                  surface_diffraction={0: (1.0, 0.0, 2e-6, 2e-6)})
    cases['grating m=1 @2um on surf 0'] = f
    show('grating m=1 @2um on surf 0', f)
except Exception as e:
    print("grating case failed:", type(e).__name__, e)

# off-centre pupil: tilt the whole system by using a wedge-like offset --
# emulate by shifting the source far off axis with a bigger aperture
f = fit_canonical_polynomials(rx_singlet(obj=0.1, ap=30e-3), lam,
                              source_box_half=50e-6, pupil_box_half=0.05,
                              n_field=8, n_pupil=8, poly_order=6,
                              source_centre=(5e-3, 0.0))
cases['source_centre=(5 mm,0), NA .05'] = f
show('source_centre=(5 mm,0), NA .05', f)

# ---- amplitude impact: rebuild each fit with the linear term folded back
from lumenairy.propagators.asymptotic import CanonicalPolyFit
def fold_back(fit):
    """Return an equivalent fit with the 5 linear terms folded into
    coef_phi (T0=1, T1=u makes this exact) and linear_coeffs_phi=None."""
    idx = {k: i for i, k in enumerate(fit.multi_indices)}
    c = np.array(fit.coef_phi, dtype=float).copy()
    a0, a1, a2, a3, a4 = fit.linear_coeffs_phi
    c[idx[(0,0,0,0)]] += a0
    c[idx[(1,0,0,0)]] += a1
    c[idx[(0,1,0,0)]] += a2
    c[idx[(0,0,1,0)]] += a3
    c[idx[(0,0,0,1)]] += a4
    g = CanonicalPolyFit(**{**fit.__dict__})
    g.coef_phi = c
    g.linear_coeffs_phi = None
    g.extract_linear_phase = False
    return g

print("\namplitude impact of dropping (a3,a4):")
for name, fit in cases.items():
    g = fold_back(fit)
    n = 17
    L = fit.s2x_halfrange*0.5
    ax = np.linspace(-L, L, n) + fit.s2x_centre
    ay = np.linspace(-L, L, n) + fit.s2y_centre
    X, Y = np.meshgrid(ax, ay, indexing='xy')
    src = (0.0, 0.0)
    kw = dict(source_point=src, w_s=20e-6, w_p=0.02,
              v2_centre=(fit.v2x_centre, fit.v2y_centre),
              s2_grid_x=X, s2_grid_y=Y)
    E1 = propagate_modal_asymptotic(fit, **kw)
    E2 = propagate_modal_asymptotic(g, **kw)
    m = (np.abs(E1) > 0) & (np.abs(E2) > 0)
    if not m.any():
        print(f"  {name:44s}  no alive pixels"); continue
    r = np.abs(E1[m])/np.abs(E2[m])
    print(f"  {name:44s} |E|_asfitted/|E|_full  max {r.max():.6f}  min {r.min():.6f}")

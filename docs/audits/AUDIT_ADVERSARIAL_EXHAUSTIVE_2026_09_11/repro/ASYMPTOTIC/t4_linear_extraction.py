"""T4: does the linear-phase extraction change the PHYSICS?

Construct two CanonicalPolyFit objects that represent EXACTLY the same
Phi(s2, v2):
  A: everything in coef_phi, linear_coeffs_phi = None
  B: the 5 linear terms moved into linear_coeffs_phi
(The Chebyshev basis has T0=1, T1=u so the map is exact.)

propagate_modal_asymptotic reads Phi with include_linear=False, so B
drops a0..a4.  a0,a1,a2 are s2-only -> pure per-pixel phase.  a3,a4
multiply v2 -> they live INSIDE the integral, so dropping them must
change the amplitude too.
"""
import sys, math, copy
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.asymptotic import (
    CanonicalPolyFit, propagate_modal_asymptotic,
)
from lumenairy.elements.lenses import _multi_indices_total_degree

lam = 1.31e-6
order = 4
mi = _multi_indices_total_degree(4, order)
idx = {k: i for i, k in enumerate(mi)}

# ---- synthetic "free space + weak aberration" fit -------------------
z = 0.05                      # m
s2h = 1.0e-3                  # s2 half range [m]
v2h = 0.05                    # v2 half range
coef_s1x = np.zeros(len(mi)); coef_s1y = np.zeros(len(mi))
# s1 = s2 - z*v2   (exact in Chebyshev: T1(u)=u)
coef_s1x[idx[(1,0,0,0)]] = s2h        # s2x
coef_s1x[idx[(0,0,1,0)]] = -z*v2h     # -z v2x
coef_s1y[idx[(0,1,0,0)]] = s2h
coef_s1y[idx[(0,0,0,1)]] = -z*v2h

# Phi = (z + z|v2|^2/2)/lam  + a small quartic aberration in v2
#  z|v2|^2/2/lam in u:  z*v2h^2*(u3^2+u4^2)/(2 lam); u^2 = (T0+T2)/2
c2 = z*v2h*v2h/(2.0*lam)
coef_phi = np.zeros(len(mi))
coef_phi[idx[(0,0,0,0)]] += z/lam + c2*0.5*2       # T0 parts of u3^2 and u4^2
coef_phi[idx[(0,0,2,0)]] += c2*0.5
coef_phi[idx[(0,0,0,2)]] += c2*0.5
# genuine linear-in-v2 phase term (e.g. residual tilt / grating order):
A3 = 0.0   # set below

def build(a3, a4, split):
    """Return a fit whose TOTAL Phi has linear v2 coefficients a3,a4
    (in waves, per normalised u).  split=False -> all in coef_phi;
    split=True -> carried in linear_coeffs_phi (what the real fitter
    produces)."""
    c = coef_phi.copy()
    lin = None
    if split:
        lin = np.array([0.0, 0.0, 0.0, a3, a4])
    else:
        c[idx[(0,0,1,0)]] += a3
        c[idx[(0,0,0,1)]] += a4
    return CanonicalPolyFit(
        poly_order=order, multi_indices=mi,
        coef_phi=c, coef_s1x=coef_s1x.copy(), coef_s1y=coef_s1y.copy(),
        s2x_centre=0.0, s2x_halfrange=s2h,
        s2y_centre=0.0, s2y_halfrange=s2h,
        v2x_centre=0.0, v2x_halfrange=v2h,
        v2y_centre=0.0, v2y_halfrange=v2h,
        wavelength=lam, linear_coeffs_phi=lin,
        extract_linear_phase=bool(split))

n = 9
ax = np.linspace(-0.5*s2h, 0.5*s2h, n)
X, Y = np.meshgrid(ax, ax, indexing='xy')
w_s, w_p = 50e-6, 0.02

print(f"{'a3 [waves]':>12} {'|E|B/|E|A max':>16} {'|E|B/|E|A min':>16} "
      f"{'phase spread B-A [rad]':>24}")
for a3 in (0.0, 1e-3, 1e-2, 0.1, 1.0, 10.0):
    fA = build(a3, 0.0, split=False)
    fB = build(a3, 0.0, split=True)
    EA = propagate_modal_asymptotic(fA, w_s=w_s, w_p=w_p, s2_grid_x=X, s2_grid_y=Y)
    EB = propagate_modal_asymptotic(fB, w_s=w_s, w_p=w_p, s2_grid_x=X, s2_grid_y=Y)
    m = np.abs(EA) > 0
    r = np.abs(EB[m])/np.abs(EA[m])
    dp = np.angle(EB[m]/EA[m])
    dp = np.unwrap(dp.ravel())
    print(f"{a3:12.4g} {r.max():16.8f} {r.min():16.8f} "
          f"{float(dp.max()-dp.min()):24.3e}")

print()
print("Now the s2-linear terms (a1) -- expected pure phase:")
def build_s2lin(a1, split):
    c = coef_phi.copy(); lin = None
    if split:
        lin = np.array([0.0, a1, 0.0, 0.0, 0.0])
    else:
        c[idx[(1,0,0,0)]] += a1
    return CanonicalPolyFit(
        poly_order=order, multi_indices=mi, coef_phi=c,
        coef_s1x=coef_s1x.copy(), coef_s1y=coef_s1y.copy(),
        s2x_centre=0.0, s2x_halfrange=s2h, s2y_centre=0.0, s2y_halfrange=s2h,
        v2x_centre=0.0, v2x_halfrange=v2h, v2y_centre=0.0, v2y_halfrange=v2h,
        wavelength=lam, linear_coeffs_phi=lin, extract_linear_phase=bool(split))
for a1 in (0.0, 1.0, 100.0, 2017.0):
    fA = build_s2lin(a1, False); fB = build_s2lin(a1, True)
    EA = propagate_modal_asymptotic(fA, w_s=w_s, w_p=w_p, s2_grid_x=X, s2_grid_y=Y)
    EB = propagate_modal_asymptotic(fB, w_s=w_s, w_p=w_p, s2_grid_x=X, s2_grid_y=Y)
    m = np.abs(EA) > 0
    r = np.abs(EB[m])/np.abs(EA[m])
    print(f"  a1={a1:10.4g}  |E| ratio  max {r.max():.10f} min {r.min():.10f}")

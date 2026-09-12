"""T16: absolute-amplitude test of propagate_modal_asymptotic against the
analytic Fresnel/Gaussian-beam answer, on a synthetic free-space
CanonicalPolyFit (s1 = s2 - z v2, Phi = (z + z|v2|^2/2)/lam).

Predicted from the Van Vleck/Maslov normalisation:
   correct v2-integrand weight = -i * sqrt(|det J|)/lam
   code's weight               =      |det J|
=> E_code = (i * lam * sqrt(|det J|)) * E_true = (i lam z) * E_true.
"""
import sys, math
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.asymptotic import (
    CanonicalPolyFit, propagate_modal_asymptotic)
from lumenairy.elements.lenses import _multi_indices_total_degree

lam = 1.0e-6; k = 2*np.pi/lam
z = 0.02
w_s = 200e-6              # source LG00 waist
w_p = 1.0e3               # essentially no pupil apodisation
s2h = 2.0e-3; v2h = 0.05

order = 2
mi = _multi_indices_total_degree(4, order)
idx = {m: i for i, m in enumerate(mi)}
cs1x = np.zeros(len(mi)); cs1y = np.zeros(len(mi)); cphi = np.zeros(len(mi))
cs1x[idx[(1,0,0,0)]] = s2h;  cs1x[idx[(0,0,1,0)]] = -z*v2h
cs1y[idx[(0,1,0,0)]] = s2h;  cs1y[idx[(0,0,0,1)]] = -z*v2h
# Phi = z/lam + z(v2x^2+v2y^2)/(2 lam);  v2 = v2h*u ; u^2 = (T0+T2)/2
B = z*v2h*v2h/(2.0*lam)
cphi[idx[(0,0,0,0)]] = z/lam + B*0.5 + B*0.5
cphi[idx[(0,0,2,0)]] = B*0.5
cphi[idx[(0,0,0,2)]] = B*0.5

fit = CanonicalPolyFit(
    poly_order=order, multi_indices=mi, coef_phi=cphi,
    coef_s1x=cs1x, coef_s1y=cs1y,
    s2x_centre=0.0, s2x_halfrange=s2h, s2y_centre=0.0, s2y_halfrange=s2h,
    v2x_centre=0.0, v2x_halfrange=v2h, v2y_centre=0.0, v2y_halfrange=v2h,
    wavelength=lam, linear_coeffs_phi=None, extract_linear_phase=False)

n = 25
ax = np.linspace(-0.3*s2h, 0.3*s2h, n)
X, Y = np.meshgrid(ax, ax, indexing='xy')
E = propagate_modal_asymptotic(fit, w_s=w_s, w_p=w_p, s2_grid_x=X, s2_grid_y=Y)

N_s = math.sqrt(2.0/(math.pi*w_s*w_s)); N_p = math.sqrt(2.0/(math.pi*w_p*w_p))
q0 = 1.0/(1j*lam/(np.pi*w_s**2)); qz = q0 + z
E_true = N_p*N_s*(q0/qz)*np.exp(1j*k*(X**2+Y**2)/(2*qz))*np.exp(1j*k*z)

r = E/E_true
print("E_code / E_true  (Fresnel of the same source, with the pupil N_p):")
print("   |ratio| mean", float(np.abs(r).mean()), " spread",
      float(np.abs(r).max()/np.abs(r).min()-1))
print("   arg(ratio) mean [rad]", float(np.angle(r).mean()), " spread",
      float(np.ptp(np.angle(r))))
pred = 1j*lam*z
print(f"   predicted ratio i*lam*sqrt(detJ) = i*lam*z = {pred}")
print(f"   measured mean ratio              = {complex(r.mean())}")
print(f"   |measured/predicted - 1|         = {abs(r.mean()/pred - 1):.3e}")

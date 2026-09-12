"""T15: absolute-amplitude test of propagate_hf_chebyshev_quadrature.
Synthesise an HFPolyFit whose Phi(s1,s2) is EXACTLY the Fresnel
quadratic for free-space distance z, propagate a Gaussian, and compare
with the analytic ABCD q-parameter result.
Tests the '-1j' Maslov factor + sqrt|det d2Phi/ds1ds2| units claim
(asymptotic_canonical_fit.py:1264-1274)."""
import sys, math
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.asymptotic import (
    HFPolyFit, propagate_hf_chebyshev_quadrature)
from lumenairy.elements.lenses import _multi_indices_total_degree

lam = 1.0e-6
k = 2*np.pi/lam
z = 0.02
w0 = 200e-6
h1 = 1.2e-3      # s1 half-range
h2 = 1.2e-3      # s2 half-range

order = 2
mi = _multi_indices_total_degree(4, order)
idx = {m: i for i, m in enumerate(mi)}
c = np.zeros(len(mi))
# Phi = z/lam + [ (h2 u3 - h1 u1)^2 + (h2 u4 - h1 u2)^2 ] / (2 z lam)
A = 1.0/(2.0*z*lam)
c[idx[(0,0,0,0)]] += z/lam
for (iu, iv, hu, hv) in ((0, 2, h1, h2), (1, 3, h1, h2)):
    # (hv u_v - hu u_u)^2 = hv^2 u_v^2 - 2 hu hv u_u u_v + hu^2 u_u^2
    ku = [0,0,0,0]; ku[iu] = 2; c[idx[tuple(ku)]] += A*hu*hu*0.5
    c[idx[(0,0,0,0)]] += A*hu*hu*0.5
    kv = [0,0,0,0]; kv[iv] = 2; c[idx[tuple(kv)]] += A*hv*hv*0.5
    c[idx[(0,0,0,0)]] += A*hv*hv*0.5
    kuv = [0,0,0,0]; kuv[iu] = 1; kuv[iv] = 1
    c[idx[tuple(kuv)]] += -2.0*A*hu*hv

fit = HFPolyFit(poly_order=order, multi_indices=mi, coef_phi=c,
                s1x_centre=0.0, s1x_halfrange=h1,
                s1y_centre=0.0, s1y_halfrange=h1,
                s2x_centre=0.0, s2x_halfrange=h2,
                s2y_centre=0.0, s2y_halfrange=h2,
                wavelength=lam, linear_coeffs_phi=None,
                extract_linear_phase=False)

# sanity: Phi at a few points vs the closed form
rng = np.random.default_rng(0)
p = rng.uniform(-0.8, 0.8, (4, 5))
s1x, s1y, s2x, s2y = p[0]*h1, p[1]*h1, p[2]*h2, p[3]*h2
exact = (z + ((s2x-s1x)**2 + (s2y-s1y)**2)/(2*z))/lam
got = fit.eval_phi(s1x, s1y, s2x, s2y)
print("Phi synthesis check, max rel:", np.max(np.abs(got-exact)/np.abs(exact)))
vv = fit.eval_van_vleck_density(s1x, s1y, s2x, s2y)
print("Van Vleck sqrt|det| :", vv[:3], "  expected 1/(lam z) =", 1.0/(lam*z))

N = 513
ax = np.linspace(-h1, h1, N)
X, Y = np.meshgrid(ax, ax, indexing='xy')
E_in = np.exp(-(X**2+Y**2)/w0**2).astype(np.complex128)

nout = 33
axo = np.linspace(-0.4*h2, 0.4*h2, nout)
E = propagate_hf_chebyshev_quadrature(fit, E_in, ax, ax, axo, axo)

# analytic Gaussian-beam reference
q0 = 1.0/(1j*lam/(np.pi*w0**2))
qz = q0 + z
XO, YO = np.meshgrid(axo, axo, indexing='xy')
E_ref = (q0/qz)*np.exp(1j*k*(XO**2+YO**2)/(2*qz))*np.exp(1j*k*z)

r = E/E_ref
print("\nratio code/analytic:  |mean| =", abs(r.mean()),
      " std/|mean| =", float(np.std(np.abs(r))/abs(r).mean()))
print("  amplitude ratio range:", float(np.abs(r).min()), float(np.abs(r).max()))
print("  phase(ratio) spread [rad]:", float(np.ptp(np.angle(r))))
print("  max |E_code - E_ref| / max|E_ref| =",
      float(np.max(np.abs(E-E_ref))/np.max(np.abs(E_ref))))

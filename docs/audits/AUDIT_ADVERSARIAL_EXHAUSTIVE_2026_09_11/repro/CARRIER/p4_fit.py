"""Probe 4: carrier_referenced_fit_radius under tilt / decentre / R->inf."""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.carrier import (
    carrier_referenced_fit_radius, carrier_referenced_aperture,
    carrier_referenced_reconstruct, propagate_carrier_referenced)

wl = 1.31e-6; k = 2*np.pi/wl
N = 512; dx = 2e-6
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x, indexing='xy')
w = 100e-6

def field(R=np.inf, L=0.0, M=0.0, x0=0.0, y0=0.0):
    r2 = (X-x0)**2 + (Y-y0)**2
    ph = 0.0 if not np.isfinite(R) else k*r2/(2*R)
    return np.exp(-r2/w**2)*np.exp(1j*(ph + k*(L*X + M*Y)))

print("=== 1. clean parabolic carrier, on axis, untilted ===")
for R in (0.05, -0.05, 0.5, 1e9, np.inf):
    for est in ('gradient', 'increment'):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Rf = carrier_referenced_fit_radius(field(R=R), wl, dx, estimator=est)
        print(f"  R={R:11.4g}  est={est:9s} -> R_fit={Rf:14.6g}  ratio={Rf/R if np.isfinite(R) and R!=0 else float('nan'):.6f}")

print("\n=== 2. pure TILT, no curvature: is tilt part of the carrier? ===")
for L in (0.0, 0.002, 0.02):
    for x0 in (0.0, 50e-6, 200e-6):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Rf = carrier_referenced_fit_radius(field(R=np.inf, L=L, x0=x0), wl, dx)
            Rf2 = carrier_referenced_fit_radius(field(R=np.inf, L=L, x0=x0), wl, dx, estimator='increment')
        print(f"  L={L:6.3f} x0={x0*1e6:6.1f}um -> R_fit(grad)={Rf:14.6g}  R_fit(incr)={Rf2:14.6g}")

print("\n=== 3. DECENTRED beam with a true curvature about its OWN centre ===")
R = 0.05
for x0 in (0.0, 50e-6, 100e-6, 200e-6):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Rf = carrier_referenced_fit_radius(field(R=R, x0=x0), wl, dx)
        Rfi = carrier_referenced_fit_radius(field(R=R, x0=x0), wl, dx, estimator='increment')
    print(f"  x0={x0*1e6:6.1f}um ({x0/w:.1f} w) -> R_fit(grad)={Rf:12.6g} ({Rf/R:7.4f}x)  "
          f"R_fit(incr)={Rfi:12.6g} ({Rfi/R:7.4f}x)")

print("\n=== 4. near-collimated: R -> inf division guards ===")
for R in (1e3, 1e5, 1e7, 1e9, 1e12):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Rf = carrier_referenced_fit_radius(field(R=R), wl, dx)
    print(f"  R={R:9.1e} -> R_fit={Rf:14.6g}  ratio={Rf/R:.6f}")
# and a truly flat field
Rf = carrier_referenced_fit_radius(np.exp(-(X**2+Y**2)/w**2).astype(complex), wl, dx)
print(f"  flat field -> R_fit={Rf}")
# m -> 1 / R=inf propagation step
env = np.exp(-(X**2+Y**2)/w**2).astype(complex)
for R in (np.inf, 1e12, 1e15):
    e, Ro, dxo = propagate_carrier_referenced(env, R, 1e-3, wl, dx)
    print(f"  propagate R={R:9.3g}: R_out={Ro:12.6g} dx_out/dx={dxo/dx:.12f} "
          f"maxdiff vs R=inf: {np.abs(e-propagate_carrier_referenced(env, np.inf, 1e-3, wl, dx).env).max():.3e}")

print("\n=== 5. aperture + refit_carrier ===")
E = field(R=0.05)
env = E*np.exp(-1j*k*(X**2+Y**2)/(2*0.05))
for refit in (False, True):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = carrier_referenced_aperture(env, 0.05, wl, dx, radius=80e-6,
                                          refit_carrier=refit, return_transmission=True)
    (e2, R2, dx2), t = out
    print(f"  refit={refit}: R_out={R2:.8g} transmission={t:.6f} "
          f"power ratio={(np.abs(e2)**2).sum()/(np.abs(env)**2).sum():.6f}")

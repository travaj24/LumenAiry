"""ANALYSIS probe 6: end-to-end demo that through_focus_scan reports Strehl >> 1
for a PERFECT (exact spherical) converging wavefront, because
diffraction_limited_peak's reference is only paraxial-quadratic."""
import sys, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.analysis.through_focus import (diffraction_limited_peak,
                                              through_focus_scan, find_best_focus)

lam = 600e-9; k0 = 2*np.pi/lam
N, dxg = 4096, 0.2e-6
D, f = 614e-6, 1.228e-3          # f/2.0 ; W040 = (D/2)^4/(8 f^3 lam)
x = (np.arange(N)-N/2)*dxg
X, Y = np.meshgrid(x, x)
R2 = X**2+Y**2
ap = (R2 <= (D/2)**2).astype(float)
W040 = (D/2)**4/(8*f**3*lam)
print(f"grid N={N} dx={dxg*1e6:.2f} um extent={N*dxg*1e6:.1f} um")
print(f"D={D*1e6:.0f} um  f={f*1e6:.1f} um  f/# = {f/D:.2f}  D/lam = {D/lam:.0f}")
print(f"paraxial-vs-sphere W040 = {W040:.4f} waves")

# PERFECT converging wavefront (exact sphere) -- zero aberration by construction
E_exit = ap*np.exp(-1j*k0*(np.sqrt(R2+f*f)-f))
ideal = diffraction_limited_peak(E_exit, lam, f, dxg)
z = np.linspace(f*0.97, f*1.03, 25)
scan = through_focus_scan(E_exit, dxg, lam, z, ideal_peak=ideal, verbose=False)
zb, sb = find_best_focus(scan, 'strehl')
print()
print(f"diffraction_limited_peak (paraxial ref)      = {ideal:.6e}")
print(f"peak of the PERFECT sphere at z=f            = "
      f"{scan.peak_I[np.argmin(np.abs(z-f))]:.6e}")
print(f"through_focus_scan best Strehl for a PERFECT lens = {sb:.4f}   at z={zb*1e6:.2f} um")
print(f"Strehl at z = f exactly                          = "
      f"{scan.strehl[np.argmin(np.abs(z-f))]:.4f}")
print()
print("A physically-correct Strehl for an aberration-free pupil is exactly 1.0.")

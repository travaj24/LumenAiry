"""Probe 5b: the slant coefficient uses NORMAL-referenced cosines where the
module's own eq.(3) requires Z-AXIS-referenced ones.  Predicted penalty
n2/(n2-1); measured on a SINGLE refracting surface."""
import sys, os
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from oracle import trace_meridional, sag_of, dsag_dh
from lumenairy.glass import get_glass_index
lam = 632.8e-9
n1, n2 = 1.0, float(get_glass_index('N-BK7', lam))
print(f"n1={n1}, n2={n2:.6f}")
R = 30e-3
h = np.linspace(1e-9, 3e-3, 3001)
sag = sag_of(h*h, R)
g = dsag_dh(h, R)                    # d sag / d h
th_i = np.arctan(g)                  # AOI for an axial incident ray
th_t = np.arcsin(np.sin(th_i)*n1/n2)  # refraction angle from the normal
# three candidate screen coefficients
c_par   = (n2 - n1)*np.ones_like(h)
c_slant = n2*np.cos(th_t) - n1*np.cos(th_i)          # what the code applies
c_eq3   = n2*np.cos(th_i - th_t) - n1*np.cos(0.0)    # module eq.(3): to the Z AXIS
# exact eikonal at the vertex plane for a single refracting surface:
rx = dict(surfaces=[dict(radius=R, glass_before='AIR', glass_after='N-BK7')],
          thicknesses=[])
r = trace_meridional(rx, lam, h)
W_exact = r['opl'] - r['opl'][0]
for nm, c in (('paraxial (n2-n1)', c_par), ('slant (normal cosines)', c_slant),
              ('eq.(3) (z-axis cosines)', c_eq3)):
    W = c*sag; W = W - W[0]
    err = W_exact - W
    print(f"  {nm:<26} OPD err at h=3mm: {err[-1]*1e9:12.4f} nm   "
          f"rms {np.sqrt(np.mean((err-err.mean())**2))*1e9:10.4f} nm")
th = th_i[-1]
print(f"\n  theta_i at h=3mm = {th:.5f} rad")
print(f"  predicted slant/paraxial error ratio = n2/(n2-1) = {n2/(n2-1):.4f}")
e_par = (W_exact - (c_par*sag - (c_par*sag)[0]))
e_sl  = (W_exact - (c_slant*sag - (c_slant*sag)[0]))
print(f"  measured  |slant err| / |paraxial err| at the rim = "
      f"{abs(e_sl[-1]/e_par[-1]):.4f}")
print(f"  signs: paraxial err {np.sign(e_par[-1]):+.0f}, "
      f"slant err {np.sign(e_sl[-1]):+.0f}  (opposite-sign overshoot)")

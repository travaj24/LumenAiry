"""Units check on _screen_obliquity_angle_field: is grad(W) a DIRECTION COSINE
(so the n1 multiply is right) or already the TRANSVERSE OPTICAL MOMENTUM
(so the n1 multiply double-counts)?  Test each carrier vocabulary."""
import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import _screen_obliquity_angle_field
from lumenairy.elements._lens_traced import TiltedCarrier
lam=0.55e-6; k0=2*np.pi/lam
N=512; dx=1.0e-6
ax=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(ax,ax)
n1=1.5168
theta=0.050                       # ray angle IN the glass (rad)
L_dircos=np.sin(theta)            # direction cosine
p_true=n1*L_dircos                # TRUE transverse optical momentum
# a field that really IS a plane wave in medium n1 at that angle:
E=(np.exp(-(X*X+Y*Y)/(120e-6)**2)*np.exp(1j*k0*p_true*X)).astype(np.complex128)
print(f"n1={n1}, theta_in_glass={theta} rad -> direction cosine {L_dircos:.6f}, "
      f"true transverse OPTICAL momentum p = n1*sin(theta) = {p_true:.6f}")
for label, carrier in [("TiltedCarrier(L=sin(theta)) [dircos]", TiltedCarrier(L=L_dircos, M=0.0, R=np.inf)),
                       ("TiltedCarrier(L=n1*sin(theta)) [p]",  TiltedCarrier(L=p_true,   M=0.0, R=np.inf)),
                       ("'auto' (fit of the real in-glass field)", 'auto'),
                       ("ndarray W = p_true*X (phase/k0)", p_true*X)]:
    qx, qy = _screen_obliquity_angle_field(carrier, E, lam, dx, dx, N, N,
                                           n_medium=n1)
    qxv = float(qx) if np.ndim(qx)==0 else float(np.median(qx))
    print(f"  {label:42s} -> qx = {qxv:9.6f}   "
          f"{'OK' if abs(qxv-p_true)<2e-4 else 'MISMATCH (x%.4f)'%(qxv/p_true)}")

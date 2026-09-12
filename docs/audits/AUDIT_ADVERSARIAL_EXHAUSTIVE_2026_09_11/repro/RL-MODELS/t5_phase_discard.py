"""Probe 4/7: do the 'displaced' REMAP paths discard the INPUT PHASE?
The 2-D remap is the DEFAULT for an asymmetric element with
surface_model='displaced' (displaced_mode='screen', displaced_obliquity='auto').
"""
import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import apply_real_lens

lam = 0.55e-6; k0 = 2*np.pi/lam
N = 512; dx = 6.0e-6; ap = 2.4e-3
axis = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(axis, axis)
w0 = ap/3.0
amp = np.exp(-(X**2+Y**2)/w0**2)

def prescr(dec=None):
    s0 = dict(radius=+19.6e-3, conic=0.0, glass_before='AIR', glass_after='N-BK7')
    s1 = dict(radius=-27.4e-3, conic=0.0, glass_before='N-BK7', glass_after='AIR')
    if dec is not None:
        s0 = dict(s0, decenter=dec); s1 = dict(s1, decenter=dec)
    return dict(surfaces=[s0, s1], thicknesses=[2.5e-3], aperture_diameter=ap)

# Two inputs: flat phase, and a strong DEFOCUS + TILT phase
E_flat = amp.astype(np.complex128)
W_extra = (X**2+Y**2)/(2*0.20) + 0.004*X      # 200 mm defocus + 4 mrad tilt [m]
E_ph = (amp*np.exp(1j*k0*W_extra)).astype(np.complex128)
print("input phase p-v over pupil (waves):",
      float((W_extra.max()-W_extra.min())/lam))

for label, kw, rx in [
    ("1-D remap  (displaced_mode='remap', symmetric)",
     dict(surface_model='displaced', displaced_mode='remap'), prescr()),
    ("2-D remap  (DEFAULT for decentered element)",
     dict(surface_model='displaced'), prescr(dec=(0.3e-3, 0.0))),
    ("pointwise SCREEN (displaced_obliquity='pointwise')",
     dict(surface_model='displaced', displaced_obliquity='pointwise'),
     prescr(dec=(0.3e-3, 0.0))),
    ("thin (reference)", {}, prescr()),
    ("displaced screen (symmetric)", dict(surface_model='displaced'), prescr()),
]:
    try:
        A = apply_real_lens(E_flat.copy(), prescription=rx, wavelength=lam, dx=dx, **kw)
        B = apply_real_lens(E_ph.copy(),  prescription=rx, wavelength=lam, dx=dx, **kw)
    except Exception as e:
        print(f"{label}: RAISED {type(e).__name__}: {str(e)[:120]}"); continue
    m = (X**2+Y**2) <= (0.8*ap/2)**2
    d = np.max(np.abs(A-B)[m])/max(np.max(np.abs(A)[m]), 1e-30)
    # phase difference that SHOULD appear
    ph = np.angle(B[m]*np.conj(A[m]))
    print(f"{label}\n    max|E(flat) - E(phased)| / max|E| = {d:.3e}   "
          f"phase-diff ptp = {float(np.ptp(ph)):.4f} rad")

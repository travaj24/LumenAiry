"""Quantify the remap input-phase discard: does conjugate='auto' recover it?
Physical consequence: focal shift of a DIVERGING input through a decentered
singlet, 2-D remap (the DEFAULT routing) vs the 'thin' screen."""
import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.propagators.propagation import angular_spectrum_propagate

lam = 0.55e-6; k0 = 2*np.pi/lam
N = 1024; dx = 4.0e-6; ap = 2.4e-3
axis = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(axis, axis)
amp = np.exp(-(X**2+Y**2)/(ap/3.0)**2)
s_src = 0.150                      # 150 mm diverging source
E_div = (amp*np.exp(1j*k0*(X**2+Y**2)/(2*s_src))).astype(np.complex128)

s0 = dict(radius=+19.6e-3, conic=0.0, glass_before='AIR', glass_after='N-BK7',
          decenter=(0.2e-3, 0.0))
s1 = dict(radius=-27.4e-3, conic=0.0, glass_before='N-BK7', glass_after='AIR',
          decenter=(0.2e-3, 0.0))
rx = dict(surfaces=[s0, s1], thicknesses=[2.5e-3], aperture_diameter=ap)

def peak_z(E, zs):
    out = []
    for z in zs:
        Ez = angular_spectrum_propagate(E.copy(), z, lam, dx, bandlimit=True)
        out.append(float(np.max(np.abs(Ez)**2)))
    return zs[int(np.argmax(out))], out

f = 22.11e-3
zs = np.linspace(0.010, 0.045, 36)
for label, kw in [('thin', {}),
                  ('displaced 2-D remap, conjugate=None (DEFAULT)',
                   dict(surface_model='displaced')),
                  ("displaced 2-D remap, conjugate='auto'",
                   dict(surface_model='displaced', conjugate='auto')),
                  ('displaced 2-D remap, conjugate=+0.150',
                   dict(surface_model='displaced', conjugate=s_src)),
                  ("pointwise SCREEN, conjugate=None",
                   dict(surface_model='displaced', displaced_obliquity='pointwise'))]:
    try:
        E = apply_real_lens(E_div.copy(), prescription=rx, wavelength=lam, dx=dx, **kw)
    except Exception as e:
        print(f"{label}: RAISED {type(e).__name__}: {str(e)[:110]}"); continue
    zb, prof = peak_z(E, zs)
    print(f"{label:52s} best focus z = {zb*1e3:7.3f} mm  (thin-lens pred "
          f"{1/(1/f - 1/ s_src)*1e3:.3f} mm for diverging, {f*1e3:.3f} mm collimated)")

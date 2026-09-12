"""Probe 2: in-glass propagation -- medium wavelength, axial piston,
evanescent cutoff at the medium k."""
import sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.elements._lens_real import apply_real_lens, _propagate_through_glass
from lumenairy.propagators.propagation import angular_spectrum_propagate as asm
lam = 632.8e-9; k0 = 2*np.pi/lam
N, dx = 256, 2e-6
E = np.ones((N, N), dtype=np.complex128)

print("=== axial piston: flat plate of BK7, thickness t ===")
for t in (1e-3, 3e-3):
    rx = dict(surfaces=[dict(radius=float('inf'), glass_before='AIR', glass_after='N-BK7'),
                        dict(radius=float('inf'), glass_before='N-BK7', glass_after='AIR')],
              thicknesses=[t])
    Eo = apply_real_lens(E.copy(), prescription=rx, wavelength=lam, dx=dx)
    n = 1.5150891983370924
    got = float(np.angle(Eo[N//2, N//2]))
    want = (n*k0*t) % (2*np.pi)
    want = want if want <= np.pi else want - 2*np.pi
    print(f"  t={t*1e3:.1f}mm: arg(E) = {got:+.6f} rad   expected n*k0*t mod 2pi "
          f"= {want:+.6f}   diff = {abs(((got-want+np.pi)%(2*np.pi))-np.pi):.3e}")
print("  (=> the ASM carries the absolute in-medium piston n*k0*t)")

print()
print("=== medium wavelength: a tilted plane wave inside glass ===")
# launch a plane wave at angle theta in AIR; inside glass it must refract
theta = 0.20
x = (np.arange(N) - N/2)*dx
X, Y = np.meshgrid(x, x)
Ein = np.exp(1j*k0*np.sin(theta)*X)
t = 5e-3
Eg = _propagate_through_glass(Ein.copy(), t, lam, 1.5150891983370924, 0.0,
                              dx, dx, True, 'asm', False, k0, np)
ph = np.unwrap(np.angle(Eg[N//2]))
kx = np.polyfit(x, ph, 1)[0]
print(f"  transverse k preserved across the slab: kx/k0 = {kx/k0:.6f} "
      f"(input sin(theta) = {np.sin(theta):.6f})")
n = 1.5150891983370924
kzn = np.sqrt((n*k0)**2 - (k0*np.sin(theta))**2)
pist = (kzn*t) % (2*np.pi); pist = pist if pist <= np.pi else pist-2*np.pi
got = float(np.angle(Eg[N//2, N//2]))
print(f"  on-axis phase {got:+.6f} vs sqrt((n k0)^2-kx^2)*t mod 2pi "
      f"{pist:+.6f}  diff {abs(((got-pist+np.pi)%(2*np.pi))-np.pi):.3e}")

print()
print("=== evanescent cutoff uses the MEDIUM k ? ===")
# a transverse frequency between k0 and n*k0: propagating in glass, evanescent in air
f_between = 1.25/lam        # |kx| = 1.25 k0  ->  evanescent in air, propagating in n=1.515
Ein2 = np.exp(2j*np.pi*f_between*X)
for nmed, tag in ((1.0, 'air  '), (1.515089, 'glass')):
    Eo2 = _propagate_through_glass(Ein2.copy(), 20e-6, lam, nmed, 0.0, dx, dx,
                                   False, 'asm', False, k0, np)
    print(f"  n={nmed:<8} ({tag}) |E| after 20 um = {float(np.abs(Eo2).mean()):.6e} "
          f"(0 => zeroed as evanescent)")
print("  (dx = 2 um so 1.25/lam is representable: f_nyq = "
      f"{1/(2*dx)*lam:.3f}/lam)")

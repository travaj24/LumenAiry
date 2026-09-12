import sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.elements._lens_real import _propagate_through_glass
lam = 632.8e-9; k0 = 2*np.pi/lam; n = 1.5150891983370924
N, dx = 512, 0.15e-6      # dx < lambda/4 so |f| up to 3.33/lam is representable
x = (np.arange(N) - N/2)*dx
X, Y = np.meshgrid(x, x)
print(f"dx={dx*1e6:.3f} um -> f_nyq = {(1/(2*dx))*lam:.3f}/lam")
print()
print("=== tilted plane wave through a glass slab (ASM at lambda/n) ===")
for st in (0.2, 0.6):
    Ein = np.exp(1j*k0*st*X).astype(np.complex128)
    t = 2e-6
    Eg = _propagate_through_glass(Ein.copy(), t, lam, n, 0.0, dx, dx, False,
                                  'asm', False, k0, np)
    ph = np.unwrap(np.angle(Eg[N//2]))
    kx = np.polyfit(x, ph, 1)[0]
    kz = np.sqrt((n*k0)**2 - (k0*st)**2)
    want = (kz*t) % (2*np.pi); want = want if want <= np.pi else want-2*np.pi
    got = float(np.angle(Eg[N//2, N//2]))
    print(f"  sin(theta_air)={st}:  kx/k0 out = {kx/k0:+.6f} (in {st:+.6f});  "
          f"axial phase {got:+.6f} vs sqrt((n k0)^2-kx^2) t = {want:+.6f}  "
          f"diff {abs(((got-want+np.pi)%(2*np.pi))-np.pi):.3e}")
print()
print("=== evanescent cutoff: uses the MEDIUM k (n*k0), not k0 ===")
for fx in (0.9, 1.25, 1.8):
    Ein = np.exp(2j*np.pi*(fx/lam)*X).astype(np.complex128)
    for nm in (1.0, n):
        Eo = _propagate_through_glass(Ein.copy(), 5e-6, lam, nm, 0.0, dx, dx,
                                      False, 'asm', False, k0, np)
        print(f"  |kx|={fx:4.2f} k0, n={nm:8.6f}: |E| after 5 um = "
              f"{float(np.abs(Eo).mean()):.4e}  "
              f"(propagating iff {fx:.2f} < {nm:.3f})")

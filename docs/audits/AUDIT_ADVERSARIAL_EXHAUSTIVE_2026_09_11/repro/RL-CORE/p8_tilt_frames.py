"""Probe 8: decenter / tilt field-frame vs surface_frame.

(a) does a TILTED FLAT plate deviate the beam by (n-1)*theta?
(b) which axis does tilt[0] rotate about in each branch?
(c) tilted sphere vs exact ray trace.
"""
import sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.glass import get_glass_index

lam = 632.8e-9
k0 = 2 * np.pi / lam
n = float(get_glass_index('N-BK7', lam))
print(f"n = {n:.6f}")

N, dx = 512, 4e-6
x = (np.arange(N) - N / 2) * dx
X, Y = np.meshgrid(x, x)
E = np.ones((N, N), dtype=np.complex128)

theta = 5e-3          # 5 mrad tilt


def slopes(Eo):
    """Mean transverse phase gradient of the exit field -> ray direction."""
    ph = np.unwrap(np.angle(Eo[N // 2]))
    kx = np.polyfit(x, ph, 1)[0]
    phc = np.unwrap(np.angle(Eo[:, N // 2]))
    ky = np.polyfit(x, phc, 1)[0]
    return kx / k0, ky / k0        # direction cosines (sin theta)


def plate(tilt, sf):
    rx = dict(surfaces=[dict(radius=float('inf'), glass_before='AIR',
                             glass_after='N-BK7', tilt=tilt),
                        dict(radius=float('inf'), glass_before='N-BK7',
                             glass_after='AIR')],
              thicknesses=[1e-3])
    return apply_real_lens(E.copy(), prescription=rx, wavelength=lam, dx=dx,
                           surface_frame=sf)


print()
print("=== (a) TILTED FLAT first face of a plate, tilt = (theta, 0) ===")
print("    exact deviation of a thin tilted refracting FLAT surface: "
      f"(n-1)*theta = {(n-1)*theta:+.6e} rad")
for sf in (False, True):
    Eo = plate((theta, 0.0), sf)
    sx, sy = slopes(Eo)
    print(f"  surface_frame={sf!s:<5}  deflection (Lx, Ly) = "
          f"({sx:+.6e}, {sy:+.6e})")

print()
print("=== (b) same, tilt = (0, theta) ===")
for sf in (False, True):
    Eo = plate((0.0, theta), sf)
    sx, sy = slopes(Eo)
    print(f"  surface_frame={sf!s:<5}  deflection (Lx, Ly) = "
          f"({sx:+.6e}, {sy:+.6e})")

print()
print("=== (c) TILTED SPHERE R=+50mm, tilt=(theta,0), compare OPD maps ===")


def sphere(tilt, sf):
    rx = dict(surfaces=[dict(radius=50e-3, glass_before='AIR',
                             glass_after='N-BK7', tilt=tilt),
                        dict(radius=float('inf'), glass_before='N-BK7',
                             glass_after='AIR')],
              thicknesses=[2e-3])
    return apply_real_lens(E.copy(), prescription=rx, wavelength=lam, dx=dx,
                           surface_frame=sf)


base = sphere((0.0, 0.0), False)
for sf in (False, True):
    Eo = sphere((theta, 0.0), sf)
    dphi = np.unwrap(np.angle(Eo[N // 2])) - np.unwrap(np.angle(base[N // 2]))
    dphi -= dphi[N // 2]
    opd = dphi / k0
    # exact rigid-body: surface z_f(x) = sag(x) - b*x  (rotation about +y by b)
    print(f"  surface_frame={sf!s:<5} d(OPD)/dx over +-1mm = "
          f"{np.polyfit(x[abs(x) < 1e-3], opd[abs(x) < 1e-3], 1)[0]:+.6e}"
          f"   (expected +-(n-1)*theta = {(n-1)*theta:.3e})")

print()
print("=== (d) surface_frame with a tilted flat: is the OPD IDENTICALLY zero? ===")
rx = dict(surfaces=[dict(radius=float('inf'), glass_before='AIR',
                         glass_after='N-BK7', tilt=(theta, 2 * theta)),
                    dict(radius=float('inf'), glass_before='N-BK7',
                         glass_after='AIR')],
          thicknesses=[1e-3])
A = apply_real_lens(E.copy(), prescription=rx, wavelength=lam, dx=dx,
                    surface_frame=True)
rx0 = dict(surfaces=[dict(radius=float('inf'), glass_before='AIR',
                          glass_after='N-BK7'),
                     dict(radius=float('inf'), glass_before='N-BK7',
                          glass_after='AIR')],
           thicknesses=[1e-3])
B = apply_real_lens(E.copy(), prescription=rx0, wavelength=lam, dx=dx,
                    surface_frame=True)
print(f"  max|E(tilted, surface_frame) - E(untilted)| = {np.abs(A-B).max():.3e}")
print(f"  bytes identical: {np.array_equal(A, B)}")

print()
print("=== (e) DECENTER: does surface_frame agree with field frame? ===")
d0 = 0.3e-3
for sf in (False, True):
    rx = dict(surfaces=[dict(radius=50e-3, glass_before='AIR',
                             glass_after='N-BK7', decenter=(d0, 0.0)),
                        dict(radius=float('inf'), glass_before='N-BK7',
                             glass_after='AIR')],
              thicknesses=[2e-3])
    Eo = apply_real_lens(E.copy(), prescription=rx, wavelength=lam, dx=dx,
                         surface_frame=sf)
    sx, sy = slopes(Eo)
    print(f"  surface_frame={sf!s:<5}  (Lx,Ly)=({sx:+.6e},{sy:+.6e})  "
          f"expected tilt ~ -d/f = {-d0/((50e-3)/(n-1)):+.6e}")

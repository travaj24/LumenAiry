"""Probe 4: absorption=True -- amplitude factor, path length, kappa source."""
import sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.glass import (get_glass_index, get_glass_index_complex,
                             register_glass)

lam = 1.0e-6
k0 = 2 * np.pi / lam

# find a glass in the catalogue with a non-zero kappa
import lumenairy.glass as G
cands = []
for nm in ('SILICON', 'SI', 'GE', 'GERMANIUM', 'ZNSE', 'GAAS', 'N-BK7',
           'FUSED_SILICA', 'SILICA', 'CDTE', 'INP'):
    try:
        nc = get_glass_index_complex(nm, lam)
        if abs(nc.imag) > 0:
            cands.append((nm, nc))
    except Exception:
        pass
print("catalogue glasses with kappa != 0 at 1 um:", cands)

# register a synthetic absorbing glass if the API allows it
kappa = 1e-4
try:
    register_glass('AUDITABS', n=1.5, kappa=kappa)
    ok = True
except Exception as e:
    print("register_glass signature issue:", e)
    ok = False
    import inspect
    print(inspect.signature(register_glass))

if ok:
    nc = get_glass_index_complex('AUDITABS', lam)
    print("AUDITABS n_complex =", nc)
    N = 128
    dx = 5e-6
    E = np.ones((N, N), dtype=np.complex128)
    t = 2e-3
    rx = dict(surfaces=[dict(radius=float('inf'), glass_before='AIR',
                             glass_after='AUDITABS'),
                        dict(radius=float('inf'), glass_before='AUDITABS',
                             glass_after='AIR')],
              thicknesses=[t])
    Eo = apply_real_lens(E.copy(), prescription=rx, wavelength=lam, dx=dx,
                         absorption=True)
    amp = float(np.abs(Eo).mean())
    print(f"  |E_out| = {amp:.9f}")
    print(f"  exp(-k0*kappa*t)        = {np.exp(-k0*kappa*t):.9f}  "
          f"(correct AMPLITUDE factor)")
    print(f"  exp(-2*k0*kappa*t)      = {np.exp(-2*k0*kappa*t):.9f}  "
          f"(would be wrong: that is the INTENSITY factor)")
    print(f"  intensity ratio |E|^2   = {amp**2:.9f} vs "
          f"exp(-4 pi kappa t/lam) = {np.exp(-4*np.pi*kappa*t/lam):.9f}")
    # path length: axial t (not local).  Build a thick meniscus so the local
    # geometric path differs strongly from the axial one and check the field
    # attenuation is uniform (axial-only).
    rx2 = dict(surfaces=[dict(radius=20e-3, glass_before='AIR',
                              glass_after='AUDITABS'),
                         dict(radius=20e-3, glass_before='AUDITABS',
                              glass_after='AIR')],
               thicknesses=[t], aperture_diameter=8e-3)
    Ns, dxs = 512, 8e-3 / 0.8 / 512
    E2 = np.ones((Ns, Ns), dtype=np.complex128)
    Eo2 = apply_real_lens(E2, prescription=rx2, wavelength=lam, dx=dxs,
                          absorption=True)
    a = np.abs(Eo2[Ns // 2])
    print(f"  meniscus: |E| axis={a[Ns//2]:.9f} edge(3.5mm)="
          f"{a[Ns//2 + int(3.5e-3/dxs)]:.9f}   "
          f"(equal => axial-only path, local thickness ignored)")

    # LAST-medium absorption: does a prescription ending IN the absorber
    # attenuate at all?  (No trailing thickness exists, so it should not.)
    rx3 = dict(surfaces=[dict(radius=float('inf'), glass_before='AIR',
                              glass_after='AUDITABS')],
               thicknesses=[])
    Eo3 = apply_real_lens(np.ones((N, N), dtype=np.complex128),
                          prescription=rx3, wavelength=lam, dx=dx,
                          absorption=True)
    print(f"  single-interface into absorber: |E| = {float(np.abs(Eo3).mean()):.9f} "
          f"(1.0 expected: no thickness after last surface)")

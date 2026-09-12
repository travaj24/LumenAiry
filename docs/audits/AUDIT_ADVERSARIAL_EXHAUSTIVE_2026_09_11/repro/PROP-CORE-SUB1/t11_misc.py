"""Misc: RW dtype path, RW symmetric-pupil blindness, MHS validation gaps."""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.vector_diffraction import richards_wolf_focus, debye_wolf_psf
from lumenairy.propagators import propagation as P
import lumenairy.propagators.mhs as M

lam = 633e-9
print("="*72)
print("(1) richards_wolf_focus dtype path under precision='single'")
print("="*72)
NA, f, Np, dxp = 0.3, 2e-3, 128, 12e-6
x = (np.arange(Np)-Np/2)*dxp
X, Y = np.meshgrid(x, x)
pup = (np.hypot(X, Y) <= f*NA).astype(np.complex64)
print(f"  global default complex dtype = {P.get_default_complex_dtype()}")
Ex, Ey, Ez, _, _ = richards_wolf_focus(pup, lam, NA, f, dxp)
print(f"  complex64 pupil, global default complex128 -> out dtype {Ex.dtype}")
try:
    P.set_default_complex_dtype(np.complex64)
    Ex2, _, _, _, _ = richards_wolf_focus(pup, lam, NA, f, dxp)
    print(f"  after set_default_complex_dtype(complex64) -> out dtype {Ex2.dtype}")
    # relative difference of the two
    print(f"  relL2(complex64 result, complex128 result) = "
          f"{np.linalg.norm(Ex2.astype(np.complex128)-Ex)/np.linalg.norm(Ex):.3e}")
finally:
    P.set_default_complex_dtype(np.complex128)

print()
print("="*72)
print("(2) the focal-field 180-deg rotation is INVISIBLE for a symmetric")
print("    pupil and VISIBLE the moment the pupil is not 180-deg symmetric")
print("="*72)


def flip(A):
    return np.roll(A[::-1, ::-1], (1, 1), (0, 1))


for label, pupil in (
        ("uniform disc (180-deg symmetric)",
         (np.hypot(X, Y) <= f*NA).astype(np.complex128)),
        ("+ defocus (symmetric)",
         np.where(np.hypot(X, Y) <= f*NA,
                  np.exp(2j*np.pi*0.5*(np.hypot(X, Y)/(f*NA))**2), 0)),
        ("+ 0.3 wv coma (NOT symmetric)",
         np.where(np.hypot(X, Y) <= f*NA,
                  np.exp(2j*np.pi*0.3*(3*(np.hypot(X, Y)/(f*NA))**3
                                       - 2*np.hypot(X, Y)/(f*NA))
                         * np.cos(np.arctan2(Y, X))), 0)),
        ("+ 0.3 wv x-tilt (NOT symmetric)",
         np.where(np.hypot(X, Y) <= f*NA,
                  np.exp(2j*np.pi*0.3*X/(f*NA)), 0))):
    psf, _, _ = debye_wolf_psf(np.asarray(pupil, np.complex128), lam, NA, f, dxp)
    d = np.linalg.norm(psf-flip(psf))/np.linalg.norm(psf)
    print(f"  {label:<34s} ||PSF - rot180(PSF)|| / ||PSF|| = {d:.4e}")

print()
print("="*72)
print("(3) MhsPipeline._validate ignores HuygensSurface.centre")
print("="*72)
s0 = M.HuygensSurface(z=0.0, Ny=32, Nx=32, dx=2e-6, centre=(0.0, 0.0), label='a')
s1 = M.HuygensSurface(z=1e-3, Ny=32, Nx=32, dx=2e-6, centre=(0.0, 0.0), label='b')
s1b = M.HuygensSurface(z=1e-3, Ny=32, Nx=32, dx=2e-6, centre=(50e-6, 0.0), label='b-shifted')
s2 = M.HuygensSurface(z=2e-3, Ny=32, Nx=32, dx=2e-6, centre=(0.0, 0.0), label='c')
try:
    pipe = M.MhsPipeline([M.asm_subdomain(s0, s1, wavelength=lam),
                          M.asm_subdomain(s1b, s2, wavelength=lam)])
    print(f"  chain with a 50 um transverse centre JUMP between subdomain 0's")
    print(f"  out_surface (centre={s1.centre}) and subdomain 1's in_surface")
    print(f"  (centre={s1b.centre}) was ACCEPTED: n_subdomains={pipe.n_subdomains}")
    E = np.ones((32, 32), dtype=np.complex128)
    out = pipe.run(E, return_intermediate=False)
    print(f"  ...and ran to completion, shape {out.shape}; the 50 um offset is")
    print(f"     silently ignored (ASM propagates on the in_surface grid only).")
except ValueError as e:
    print(f"  rejected: {e}")

# same z/N/dx mismatch IS caught
try:
    M.MhsPipeline([M.asm_subdomain(s0, s1, wavelength=lam),
                   M.asm_subdomain(M.HuygensSurface(z=1e-3, Ny=64, Nx=64,
                                                    dx=2e-6), s2,
                                   wavelength=lam)])
    print("  shape mismatch NOT caught (unexpected)")
except ValueError:
    print("  shape/z/dx mismatch IS caught (control)")

print()
print("="*72)
print("(4) MhsPipeline.run default return type")
print("="*72)
pipe = M.MhsPipeline([M.asm_subdomain(s0, s1, wavelength=lam),
                      M.asm_subdomain(s1, s2, wavelength=lam)])
r = pipe.run(np.ones((32, 32), dtype=np.complex128))
print(f"  run(E) default -> {type(r).__name__} of len {len(r)}; "
      f"element 0 is {tuple(type(t).__name__ for t in r[0])}")
rr = pipe.run(np.ones((32, 32), dtype=np.complex128), return_result=True)
print(f"  run(E, return_result=True).wavelength = {rr.wavelength}  "
      f"(wavelength was never asked for; subdomains carry it in kwargs)")

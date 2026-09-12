"""Probe 3: fresnel=True amplitude transmission -- is it sqrt(power T) or
sqrt(mean |t|^2)?  Energy audit at normal incidence.
"""
import sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.glass import get_glass_index

lam = 632.8e-9
n = float(get_glass_index('N-BK7', lam))
print(f"n(N-BK7) @ {lam*1e9:.1f} nm = {n:.6f}")

T_correct = 1 - ((n - 1) / (n + 1)) ** 2
t_amp = 2.0 / (1.0 + n)
print(f"  correct normal-incidence power transmittance T = 1-((n-1)/(n+1))^2 "
      f"= {T_correct:.6f}")
print(f"  amplitude t = 2/(1+n) = {t_amp:.6f},  |t|^2 = {t_amp**2:.6f}")
print(f"  ratio T/|t|^2 = n2/n1 = {T_correct/t_amp**2:.6f} (should be n = {n:.6f})")
print()

N = 256
dx = 4e-6
E = np.ones((N, N), dtype=np.complex128)


def power(F):
    return float(np.sum(np.abs(F) ** 2))


print("=== A. SINGLE flat air->glass interface, zero thickness chain ===")
# one refracting surface only: no thicknesses needed (surfaces-1 = 0)
rx1 = dict(surfaces=[dict(radius=float('inf'), glass_before='AIR',
                          glass_after='N-BK7')],
           thicknesses=[])
Eo = apply_real_lens(E.copy(), prescription=rx1, wavelength=lam, dx=dx,
                     fresnel=True)
print(f"  |E_out|^2 / |E_in|^2 = {power(Eo)/power(E):.6f}")
print(f"  expected (power-correct)   : {T_correct:.6f}")
print(f"  expected (amplitude-only)  : {t_amp**2:.6f}")
print()

print("=== B. flat plate AIR->BK7->AIR (both faces) ===")
rx2 = dict(surfaces=[dict(radius=float('inf'), glass_before='AIR',
                          glass_after='N-BK7'),
                     dict(radius=float('inf'), glass_before='N-BK7',
                          glass_after='AIR')],
           thicknesses=[3e-3])
Eo2 = apply_real_lens(E.copy(), prescription=rx2, wavelength=lam, dx=dx,
                      fresnel=True)
print(f"  |E_out|^2 / |E_in|^2 = {power(Eo2)/power(E):.6f}")
print(f"  expected T1*T2       = {T_correct**2:.6f}  "
      f"(2 uncoated faces, ~{100*(1-T_correct**2):.2f} % loss)")
print()

print("=== C. prescription that ENDS IN GLASS (immersed / half element) ===")
rx3 = dict(surfaces=[dict(radius=float('inf'), glass_before='AIR',
                          glass_after='N-BK7'),
                     dict(radius=float('inf'), glass_before='N-BK7',
                          glass_after='N-BK7')],
           thicknesses=[3e-3])
Eo3 = apply_real_lens(E.copy(), prescription=rx3, wavelength=lam, dx=dx,
                      fresnel=True)
print(f"  |E_out|^2 / |E_in|^2 = {power(Eo3)/power(E):.6f}  "
      f"expected {T_correct:.6f}")
print()

print("=== D. cemented interface BK7 -> SF11 (index step inside glass) ===")
n2 = float(get_glass_index('N-SF11', lam))
T12 = 4 * n * n2 / (n + n2) ** 2
print(f"  n(N-SF11) = {n2:.6f}; correct T(BK7->SF11) = {T12:.6f}")
rx4 = dict(surfaces=[dict(radius=float('inf'), glass_before='N-BK7',
                          glass_after='N-SF11')],
           thicknesses=[])
Eo4 = apply_real_lens(E.copy(), prescription=rx4, wavelength=lam, dx=dx,
                      fresnel=True)
print(f"  |E_out|^2/|E_in|^2 = {power(Eo4)/power(E):.6f}  "
      f"amplitude-only |t|^2 = {(2*n/(n+n2))**2:.6f}")
print()

print("=== E. real singlet, oblique-angle check (curved surfaces) ===")
ap = 8e-3
Ns = 1024
dxs = ap / (0.8 * Ns)
x = (np.arange(Ns) - Ns / 2) * dxs
X, Y = np.meshgrid(x, x)
mask = (X ** 2 + Y ** 2) <= (ap / 2) ** 2
Ein = mask.astype(np.complex128)
rx5 = dict(surfaces=[dict(radius=20e-3, glass_before='AIR', glass_after='N-BK7'),
                     dict(radius=-20e-3, glass_before='N-BK7', glass_after='AIR')],
           thicknesses=[6e-3])
Eo5 = apply_real_lens(Ein.copy(), prescription=rx5, wavelength=lam, dx=dxs,
                      fresnel=True)
Eo5n = apply_real_lens(Ein.copy(), prescription=rx5, wavelength=lam, dx=dxs,
                       fresnel=False)
print(f"  fresnel on : P = {power(Eo5):.4f}")
print(f"  fresnel off: P = {power(Eo5n):.4f}")
print(f"  ratio      = {power(Eo5)/power(Eo5n):.6f}   "
      f"(2 uncoated faces -> expect ~{T_correct**2:.4f})")

"""PROP-HF p2b: corrected continuum oracle for the HF kernel + Gaussian +
ASM cut + finite-difference step scan.

Corrected oracle B (rho drho = r dr, kernel (1/(i lam)) cos(th) e^{ikr}/r):
    U = (2 pi z/(i lam)) Int_z^{ra} e^{ikr}/r dr
Integration by parts shows this = e^{ikz} - (z/ra) e^{ik ra} + O(1/(k r)),
i.e. exactly RS-I minus the (1 - 1/(ikr)) near-field term.
"""
import sys
import time
import numpy as np
from scipy.integrate import simpson

sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.hf import (  # noqa: E402
    propagate_huygens_fresnel_with_opl_callable as hf_opl)
from lumenairy.propagators.propagation import angular_spectrum_propagate  # noqa: E402

LAM = 633e-9
A = 20e-6
K = 2 * np.pi / LAM


def oracle_rs1(z):
    ra = np.hypot(z, A)
    return np.exp(1j * K * z) - (z / ra) * np.exp(1j * K * ra)


def oracle_no_nearfield(z, n=2000001):
    ra = np.hypot(z, A)
    r = np.linspace(z, ra, n)
    return (2 * np.pi * z / (1j * LAM)) * simpson(np.exp(1j * K * r) / r, x=r)


def circ(N, dx):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    return (X ** 2 + Y ** 2 <= A * A).astype(np.complex128)


def spherical_opl(z):
    def f(a1, b1, a2, b2):
        return np.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2 + z * z) / LAM
    return f


print("=" * 78)
print("(A) on-axis circular aperture: which kernel does HF converge to?")
print("=" * 78)
for z in (100e-6, 500e-6, 2000e-6):
    u_rs = oracle_rs1(z)
    u_nn = oracle_no_nearfield(z)
    print(f"\nz={z*1e6:7.1f} um  N_F={A*A/(LAM*z):6.3f}")
    print(f"  RS-I exact              U = {u_rs:.6f}   |U|^2={abs(u_rs)**2:.6f}")
    print(f"  RS-I minus near-field   U = {u_nn:.6f}   |U|^2={abs(u_nn)**2:.6f}"
          f"   (differs from RS-I by {abs(u_nn-u_rs)/abs(u_rs):.3e})")
    for N in (256, 512, 1024, 2048):
        dx = 5.0 * A / N
        out = hf_opl(circ(N, dx), opl_fn=spherical_opl(z),
                     output_grid_x=np.array([0.0]), output_grid_y=np.array([0.0]),
                     input_grid_dx=dx, apply_van_vleck=True)
        v = complex(out[0, 0])
        print(f"    N={N:5d} dx={dx*1e9:7.1f}nm  U={v:.6f}  "
              f"err_vs_noNF={abs(v-u_nn)/abs(u_nn):.3e}  "
              f"err_vs_RS1={abs(v-u_rs)/abs(u_rs):.3e}")

print()
print("=" * 78)
print("(B) Gaussian beam vs analytic, on a 1-D cut, and vs ASM")
print("=" * 78)
w0 = 8e-6
zR = np.pi * w0 ** 2 / LAM
print(f"  w0={w0*1e6} um  z_R={zR*1e6:.3f} um  (NA ~ lam/(pi w0) = {LAM/(np.pi*w0):.4f})")
N, dx = 512, 0.25e-6
x = (np.arange(N) - N / 2) * dx
X, Y = np.meshgrid(x, x, indexing='xy')
E0 = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
ncut = 25
xo = x[N // 2: N // 2 + ncut * 4: 4]


def gauss_analytic(z, xs):
    wz = w0 * np.sqrt(1 + (z / zR) ** 2)
    Rinv = z / (z ** 2 + zR ** 2)
    gouy = np.arctan2(z, zR)
    r2 = xs ** 2
    return ((w0 / wz) * np.exp(-r2 / wz ** 2)
            * np.exp(1j * (K * z + K * r2 * Rinv / 2 - gouy)))


for fac in (0.5, 1.0, 3.0):
    z = fac * zR
    t0 = time.time()
    out = hf_opl(E0, opl_fn=spherical_opl(z),
                 output_grid_x=xo, output_grid_y=np.array([0.0]),
                 input_grid_dx=dx, apply_van_vleck=True)
    dt = time.time() - t0
    hf_line = np.asarray(out)[0]
    an = gauss_analytic(z, xo)
    asm = angular_spectrum_propagate(E0, z, LAM, dx)
    asm_line = np.asarray(asm)[N // 2, N // 2: N // 2 + ncut * 4: 4]
    def l2(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b)
    print(f"  z={fac:4.1f} z_R = {z*1e6:8.3f} um   "
          f"L2(HF, analytic)={l2(hf_line, an):.3e}   "
          f"L2(ASM, analytic)={l2(asm_line, an):.3e}   "
          f"L2(HF, ASM)={l2(hf_line, asm_line):.3e}   [{dt:.1f}s, {ncut} out px]")

print()
print("=" * 78)
print("(C) finite_diff_step scan on the EXACT SPHERICAL OPL (not the quadratic")
print("    Fresnel OPL the docstring measured on)")
print("=" * 78)
z = 50e-3
lam = 1e-6
for h in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
    def opl(a1, b1, a2, b2):
        return np.sqrt((a1 - a2) ** 2 + (b1 - b2) ** 2 + z * z) / lam
    u, v = 2e-3, 1e-3
    s1x, s1y, s2x, s2y = u, v, 0.0, 0.0
    pxx = (opl(s1x + h, s1y, s2x + h, s2y) - opl(s1x + h, s1y, s2x - h, s2y)
           - opl(s1x - h, s1y, s2x + h, s2y) + opl(s1x - h, s1y, s2x - h, s2y)) / (4 * h * h)
    pyy = (opl(s1x, s1y + h, s2x, s2y + h) - opl(s1x, s1y + h, s2x, s2y - h)
           - opl(s1x, s1y - h, s2x, s2y + h) + opl(s1x, s1y - h, s2x, s2y - h)) / (4 * h * h)
    pxy = (opl(s1x + h, s1y, s2x, s2y + h) - opl(s1x + h, s1y, s2x, s2y - h)
           - opl(s1x - h, s1y, s2x, s2y + h) + opl(s1x - h, s1y, s2x, s2y - h)) / (4 * h * h)
    pyx = (opl(s1x, s1y + h, s2x + h, s2y) - opl(s1x, s1y + h, s2x - h, s2y)
           - opl(s1x, s1y - h, s2x + h, s2y) + opl(s1x, s1y - h, s2x - h, s2y)) / (4 * h * h)
    r = np.sqrt(u * u + v * v + z * z)
    exact = (z / r) / (lam * r)
    got = np.sqrt(abs(pxx * pyy - pxy * pyx))
    print(f"  h={h:8.1e} m   sqrt|det| rel err = {got/exact - 1.0:+.4e}")

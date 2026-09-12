"""PROP-HF p2: what kernel does hf.propagate_huygens_fresnel_with_opl_callable
actually implement, and does it converge to it?

Oracles
-------
A. EXACT RS-I on-axis closed form for a circular aperture under a unit plane
   wave.  Derived here (not quoted):
       U = -(1/2pi) Int U0 d/dz(e^{ikr}/r) dS,  dS = 2 pi rho drho, rho drho = r dr
         = -z [e^{ikr}/r]_{z}^{ra}
         = e^{ikz} - (z/ra) e^{ik ra},     ra = sqrt(z^2+a^2)
   |U|^2 = 1 + (z/ra)^2 - 2 (z/ra) cos(k (ra - z)).
   The textbook 4 sin^2(k/2 (ra - z)) is the SAME expression with the
   obliquity factor z/ra set to 1 (Kirchhoff, no-obliquity limit).
B. CONTINUUM value of the RS-I-without-near-field-term kernel
       K = (1/(i lam)) cos(theta) e^{ikr} / r
   on the same aperture, by 1-D radial quadrature:
       U = (2 pi z/(i lam)) Int_z^{ra} e^{ikr}/r^2 dr.
   Oracle B is what a kernel of that form MUST converge to as dx -> 0.

Also derives, symbolically-by-substitution, that the Van Vleck density
sqrt|det d2Phi/ds1 ds2| for Phi = r/lam equals cos(theta)/(lam r):
    A = d2Phi/dx1dx2 = (1/(lam r))(u^2/r^2 - 1),  D likewise in v
    B = C = u v /(lam r^3)
    det = (1/(lam^2 r^2)) (1 - (u^2+v^2)/r^2) = z^2/(lam^2 r^4)
    sqrt|det| = z/(lam r^2) = cos(theta)/(lam r)
so kernel = (-1j) * sqrt|det| * exp(2 pi i Phi) = (1/(i lam)) cos(theta) e^{ikr}/r.
"""
import sys
import time
import numpy as np

sys.path.insert(0, r"D:/Metacept/Neurophos/Free_Space_Optics/Lumenairy")
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.hf import (  # noqa: E402
    propagate_huygens_fresnel_with_opl_callable as hf_opl,
)

LAM = 633e-9
A = 20e-6
K = 2 * np.pi / LAM


def oracle_rs1(z):
    ra = np.hypot(z, A)
    return np.exp(1j * K * z) - (z / ra) * np.exp(1j * K * ra)


def oracle_kirchhoff_4sin2(z):
    ra = np.hypot(z, A)
    return 4.0 * np.sin(0.5 * K * (ra - z)) ** 2


def oracle_no_nearfield(z, n=400001):
    """(2 pi z/(i lam)) Int_z^{ra} e^{ikr}/r^2 dr by fine Simpson."""
    ra = np.hypot(z, A)
    r = np.linspace(z, ra, n)
    g = np.exp(1j * K * r) / r ** 2
    from scipy.integrate import simpson
    return (2 * np.pi * z / (1j * LAM)) * simpson(g, x=r)


def circ(N, dx):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    return (X ** 2 + Y ** 2 <= A * A).astype(np.complex128)


def main():
    print("Van Vleck density identity check (numeric, exact spherical OPL):")
    z = 300e-6
    def opl(s1x, s1y, s2x, s2y):
        return np.sqrt((s1x - s2x) ** 2 + (s1y - s2y) ** 2 + z * z) / LAM
    h = 1e-6
    u, v = 7e-6, -3e-6
    s1x, s1y, s2x, s2y = u, v, 0.0, 0.0
    def d2(ax, ay):
        return (opl(s1x + ax * h, s1y + ay * h, s2x + h * (1 - ax), s2y + h * (1 - ay))
                if False else None)
    pxx = (opl(s1x + h, s1y, s2x + h, s2y) - opl(s1x + h, s1y, s2x - h, s2y)
           - opl(s1x - h, s1y, s2x + h, s2y) + opl(s1x - h, s1y, s2x - h, s2y)) / (4 * h * h)
    pyy = (opl(s1x, s1y + h, s2x, s2y + h) - opl(s1x, s1y + h, s2x, s2y - h)
           - opl(s1x, s1y - h, s2x, s2y + h) + opl(s1x, s1y - h, s2x, s2y - h)) / (4 * h * h)
    pxy = (opl(s1x + h, s1y, s2x, s2y + h) - opl(s1x + h, s1y, s2x, s2y - h)
           - opl(s1x - h, s1y, s2x, s2y + h) + opl(s1x - h, s1y, s2x, s2y - h)) / (4 * h * h)
    pyx = (opl(s1x, s1y + h, s2x + h, s2y) - opl(s1x, s1y + h, s2x - h, s2y)
           - opl(s1x, s1y - h, s2x + h, s2y) + opl(s1x, s1y - h, s2x - h, s2y)) / (4 * h * h)
    det = pxx * pyy - pxy * pyx
    r = np.sqrt(u * u + v * v + z * z)
    print(f"  sqrt|det| = {np.sqrt(abs(det)):.10e}   cos(th)/(lam r) = {(z / r) / (LAM * r):.10e}")
    print(f"  ratio = {np.sqrt(abs(det)) / ((z / r) / (LAM * r)):.12f}\n")

    print("On-axis circular aperture, a=20 um, lam=633 nm.")
    print("Oracles: RS-I exact | Kirchhoff 4sin^2 | RS-I-no-near-field (kernel the code implements)\n")
    for z in (100e-6, 500e-6, 2000e-6):
        u_rs = oracle_rs1(z)
        u_nn = oracle_no_nearfield(z)
        print(f"z = {z*1e6:7.1f} um   N_F = {A*A/(LAM*z):6.3f}")
        print(f"   |U|^2  RS-I exact        = {abs(u_rs)**2:.6f}")
        print(f"   |U|^2  Kirchhoff 4sin^2  = {oracle_kirchhoff_4sin2(z):.6f}")
        print(f"   |U|^2  no-near-field     = {abs(u_nn)**2:.6f}   (phase {np.angle(u_nn):+.5f})")
        for N in (256, 512, 1024):
            dx = 2.5 * A / N * 2          # half-width = 2.5a
            E = circ(N, dx)
            t0 = time.time()
            out = hf_opl(E,
                         opl_fn=lambda a1, b1, a2, b2, _z=z: np.sqrt(
                             (a1 - a2) ** 2 + (b1 - b2) ** 2 + _z * _z) / LAM,
                         output_grid_x=np.array([0.0]),
                         output_grid_y=np.array([0.0]),
                         input_grid_dx=dx,
                         apply_van_vleck=True)
            dt = time.time() - t0
            val = complex(out[0, 0])
            print(f"     N={N:5d} dx={dx*1e9:7.1f} nm  |U|^2={abs(val)**2:.6f} "
                  f"phase={np.angle(val):+.5f}  relerr vs no-NF = "
                  f"{abs(val - u_nn)/abs(u_nn):.3e}  relerr vs RS-I = "
                  f"{abs(val - u_rs)/abs(u_rs):.3e}   [{dt:.1f}s]")
        print()


if __name__ == '__main__':
    main()

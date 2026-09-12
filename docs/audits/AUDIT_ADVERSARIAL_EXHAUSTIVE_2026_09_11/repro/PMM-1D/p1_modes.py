"""PROBE 1: 1-D lamellar mode finding.  Compare the SEM (Galerkin) modal
spectrum to the EXACT Botten-1981 transcendental dispersion relation.
"""
import sys, os
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import _core as pc


def exact_roots(eps1, eps2, f, period, wl, kx0, pol, umin, umax, n=400000):
    """Dense real-line scan + bisection for the roots of the exact lamellar
    dispersion relation in u = (gamma/k0)^2."""
    k0 = 2 * np.pi / wl
    d1, d2 = f * period, (1 - f) * period

    def F(u):
        k1 = np.sqrt(complex(k0 * k0 * (eps1 - u)))
        k2 = np.sqrt(complex(k0 * k0 * (eps2 - u)))
        if pol == "te":
            a = k1 / k2
        else:
            a = (k1 * eps2) / (k2 * eps1)
        v = (np.cos(k1 * d1) * np.cos(k2 * d2)
             - 0.5 * (a + 1.0 / a) * np.sin(k1 * d1) * np.sin(k2 * d2)
             - np.cos(kx0 * period))
        return v.real                       # real for real eps, real u

    us = np.linspace(umin, umax, n)
    fv = np.array([F(u) for u in us])
    roots = []
    for i in range(n - 1):
        if not np.isfinite(fv[i]) or not np.isfinite(fv[i + 1]):
            continue
        if fv[i] == 0.0:
            roots.append(us[i]); continue
        if fv[i] * fv[i + 1] < 0:
            a, b = us[i], us[i + 1]
            fa = fv[i]
            for _ in range(200):
                m = 0.5 * (a + b)
                fm = F(m)
                if fa * fm <= 0:
                    b = m
                else:
                    a, fa = m, fm
            roots.append(0.5 * (a + b))
    return np.array(roots)


def sem_spectrum(eps_r, eps_g, duty, period, wl, kx0, pol, degree, nel=1,
                 grade=False):
    mats = pc._build_sem(period, duty * period, eps_r, eps_g, degree,
                         nel, nel, grade)
    k0 = 2 * np.pi / wl
    A, lam, q, invop = pc._sem_modes(mats, k0, pol, kx0, False)
    return (q ** 2).real if np.max(np.abs((q ** 2).imag)) < 1e-8 else q ** 2, q


CASES = [
    # label, eps_ridge, eps_groove, duty, period/wl, angle_deg
    ("Si/air  P=lam,  f=0.5", 3.48**2, 1.0, 0.5, 1.0, 0.0),
    ("Si/air  P=lam,  f=0.5, 25deg", 3.48**2, 1.0, 0.5, 1.0, 25.0),
    ("Si/air  P=3lam, f=0.5 (dense)", 3.48**2, 1.0, 0.5, 3.0, 0.0),
    ("Si/SiO2 P=lam,  f=0.3", 3.48**2, 1.444**2, 0.3, 1.0, 0.0),
]

wl = 1.0e-6
for label, e1, e2, f, pl, angd in CASES:
    period = pl * wl
    k0 = 2 * np.pi / wl
    kx0 = np.sin(np.deg2rad(angd)) * k0
    print("=" * 78)
    print(f"{label}   (kx0*P/2pi = {kx0*period/2/np.pi:.4f})")
    for pol in ("te", "tm"):
        emax = max(e1, e2)
        ex = exact_roots(e1, e2, f, period, wl, kx0, pol, -60.0, emax + 0.001)
        for degree in (12, 20, 32):
            u_sem, q = sem_spectrum(e1, e2, f, period, wl, kx0, pol, degree)
            us = np.sort(np.real(u_sem))[::-1]
            exs = np.sort(ex)[::-1]
            ncmp = min(len(exs), 10)
            err = []
            for i in range(ncmp):
                # nearest SEM eigenvalue to each exact root
                err.append(np.min(np.abs(us - exs[i])))
            err = np.array(err)
            # also: duplicates?  count SEM eigenvalues within 1e-9 of each other
            dup = np.sum(np.abs(np.diff(np.sort(us))) < 1e-10)
            print(f"  {pol} deg={degree:3d} n_modes={len(us):3d} "
                  f"n_exact(u>-60)={len(exs):3d} "
                  f"max|du| over lowest {ncmp} = {err.max():.3e} "
                  f"(mean {err.mean():.2e}) exact-dup-pairs={dup}")
        # print the lowest few for the highest degree
        u_sem, q = sem_spectrum(e1, e2, f, period, wl, kx0, pol, 32)
        us = np.sort(np.real(u_sem))[::-1]
        exs = np.sort(ex)[::-1]
        print(f"    exact[:8] = {np.array2string(exs[:8], precision=9)}")
        print(f"    sem  [:8] = {np.array2string(us[:8], precision=9)}")

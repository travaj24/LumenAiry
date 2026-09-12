"""Probe 6c: independent confirmation that seidel_correction=True DEGRADES the
field -- via focal-plane Strehl (no phase unwrapping involved)."""
import sys, os
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from oracle import trace_meridional
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.propagators.propagation import angular_spectrum_propagate as asm

lam = 632.8e-9


def pc(R=50e-3, t=3e-3, ap=4e-3):
    return dict(surfaces=[dict(radius=R, glass_before='AIR', glass_after='N-BK7'),
                          dict(radius=float('inf'), glass_before='N-BK7',
                               glass_after='AIR')],
                thicknesses=[t], aperture_diameter=ap)


def dbl(ap=8e-3):
    return dict(surfaces=[
        dict(radius=33.3e-3, glass_before='AIR', glass_after='N-BAF10'),
        dict(radius=-22.28e-3, glass_before='N-BAF10', glass_after='N-SF6HT'),
        dict(radius=-291.07e-3, glass_before='N-SF6HT', glass_after='AIR')],
        thicknesses=[9.0e-3, 2.5e-3], aperture_diameter=ap)


def peak_scan(rx, N, dx, zs, **kw):
    E = np.ones((N, N), dtype=np.complex128)
    Eo = apply_real_lens(E, prescription=rx, wavelength=lam, dx=dx, **kw)
    best = (-1, None)
    for z in zs:
        Ez = asm(Eo.copy(), z, lam, dx)
        p = float(np.abs(Ez).max() ** 2)
        if p > best[0]:
            best = (p, z)
    return best


for nm, rx, N in (('plano-convex 4mm', pc(), 1024), ('doublet 8mm', dbl(), 2048)):
    ap = rx['aperture_diameter']
    h0 = np.linspace(-0.995 * ap / 2, 0.995 * ap / 2, 2001)
    r = trace_meridional(rx, lam, h0)
    NA = float(np.max(np.abs(r['Lx'])))
    dx = 0.30 * lam / NA
    N = int(2 ** np.ceil(np.log2(1.45 * ap / dx)))
    # paraxial focus from the marginal ray
    f_est = float(np.abs(r['x'][-1] / r['Lx'][-1]))
    zs = f_est * np.linspace(0.90, 1.10, 41)
    print(f"== {nm}: N={N} dx={dx*1e6:.3f}um f~{f_est*1e3:.3f}mm ==")
    for kw in ({}, dict(seidel_correction=True)):
        p, z = peak_scan(rx, N, dx, zs, **kw)
        print(f"   {str(kw) or 'default':<32} peak |E|^2 = {p:10.2f} at "
              f"z = {z*1e3:8.4f} mm")

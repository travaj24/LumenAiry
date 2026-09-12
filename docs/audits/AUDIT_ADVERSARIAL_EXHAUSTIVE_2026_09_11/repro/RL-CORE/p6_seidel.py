"""Probe 6: seidel_correction -- does it improve or degrade agreement with an
INDEPENDENT ray oracle?  Also check radial normalisation, threshold, and
whether it double-counts slant_correction."""
import sys, os, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from oracle import trace_meridional
from lumenairy.elements._lens_real import apply_real_lens

lam = 632.8e-9
k0 = 2 * np.pi / lam


def bx(R=60e-3, t=4e-3, ap=4e-3, glass='N-BK7'):
    return dict(surfaces=[dict(radius=R, glass_before='AIR', glass_after=glass),
                          dict(radius=-R, glass_before=glass, glass_after='AIR')],
                thicknesses=[t], aperture_diameter=ap)


def dbl(ap=4e-3):
    return dict(surfaces=[
        dict(radius=33.3e-3, glass_before='AIR', glass_after='N-BAF10'),
        dict(radius=-22.28e-3, glass_before='N-BAF10', glass_after='N-SF6HT'),
        dict(radius=-291.07e-3, glass_before='N-SF6HT', glass_after='AIR')],
        thicknesses=[9.0e-3, 2.5e-3], aperture_diameter=ap)


def pc(R=50e-3, t=3e-3, ap=4e-3, glass='N-BK7'):
    return dict(surfaces=[dict(radius=R, glass_before='AIR', glass_after=glass),
                          dict(radius=float('inf'), glass_before=glass,
                               glass_after='AIR')],
                thicknesses=[t], aperture_diameter=ap)


def residual(name, rx, nyq=0.30, frac=0.85, **kw):
    ap = rx['aperture_diameter']
    h0 = np.linspace(-0.995 * ap / 2, 0.995 * ap / 2, 8001)
    r = trace_meridional(rx, lam, h0)
    NA = float(np.max(np.abs(r['Lx'])))
    dx = nyq * lam / max(NA, 1e-6)
    N = int(2 ** np.ceil(np.log2(1.45 * ap / dx)))
    x = (np.arange(N) - N / 2) * dx
    E = np.ones((N, N), dtype=np.complex128)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        Eo = apply_real_lens(E, prescription=rx, wavelength=lam, dx=dx, **kw)
        wl = [str(ww.message)[:90] for ww in w]
    ph = np.unwrap(np.angle(Eo[N // 2]))
    m = np.abs(x) <= frac * ap / 2
    xs = x[m]
    W = ph[m] / k0
    order = np.argsort(r['x'])
    Wr = np.interp(xs, r['x'][order], r['opl'][order])
    d = W - Wr
    d0 = d - d.mean()
    print(f"  {name:<42} rms {np.sqrt(np.mean(d0**2))*1e9:9.4f} nm  "
          f"PV {np.ptp(d)*1e9:10.4f} nm  N={N}")
    for s in wl:
        print(f"        warn: {s}")
    return np.sqrt(np.mean(d0 ** 2))


for nm, rx in (('plano-convex 4mm', pc()), ('biconvex R=+-60 4mm', bx()),
               ('biconvex R=+-30 4mm', bx(R=30e-3)),
               ('cemented doublet 4mm', dbl()),
               ('cemented doublet 8mm', dbl(ap=8e-3))):
    print(f"== {nm} ==")
    a = residual('seidel OFF', rx)
    b = residual('seidel ON  (order 6)', rx, seidel_correction=True)
    c = residual('seidel ON  (order 4)', rx, seidel_correction=True,
                 seidel_poly_order=4)
    print(f"    -> seidel gain (order 6): {a/b:.3f}x   (order 4): {a/c:.3f}x")
    print()

print("== slant + seidel together (is the analytic reference consistent?) ==")
rx = dbl(ap=8e-3)
residual('slant OFF seidel OFF', rx)
residual('slant ON  seidel OFF', rx, slant_correction=True)
residual('slant OFF seidel ON ', rx, seidel_correction=True)
residual('slant ON  seidel ON ', rx, slant_correction=True,
         seidel_correction=True)

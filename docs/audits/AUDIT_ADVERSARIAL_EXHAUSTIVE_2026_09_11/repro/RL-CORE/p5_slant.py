"""Probe 5: slant_correction -- formula and the angle it uses."""
import sys, os
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from oracle import trace_meridional
from lumenairy.elements._lens_real import apply_real_lens
lam = 632.8e-9; k0 = 2*np.pi/lam
def pc(R=50e-3, t=3e-3, ap=4e-3, flip=False):
    s = ([dict(radius=R, glass_before='AIR', glass_after='N-BK7'),
          dict(radius=float('inf'), glass_before='N-BK7', glass_after='AIR')]
         if not flip else
         [dict(radius=float('inf'), glass_before='AIR', glass_after='N-BK7'),
          dict(radius=-R, glass_before='N-BK7', glass_after='AIR')])
    return dict(surfaces=s, thicknesses=[t], aperture_diameter=ap)
def bx(R=60e-3, t=4e-3, ap=4e-3):
    return dict(surfaces=[dict(radius=R, glass_before='AIR', glass_after='N-BK7'),
                          dict(radius=-R, glass_before='N-BK7', glass_after='AIR')],
                thicknesses=[t], aperture_diameter=ap)
def men(R1=20e-3, R2=25e-3, t=4e-3, ap=4e-3):
    return dict(surfaces=[dict(radius=R1, glass_before='AIR', glass_after='N-BK7'),
                          dict(radius=R2, glass_before='N-BK7', glass_after='AIR')],
                thicknesses=[t], aperture_diameter=ap)
def asph(ap=4e-3):
    return dict(surfaces=[dict(radius=25e-3, conic=-1.0,
                               aspheric_coeffs={4: -2e-5, 6: 1e-6},
                               glass_before='AIR', glass_after='N-BK7'),
                          dict(radius=float('inf'), glass_before='N-BK7',
                               glass_after='AIR')],
                thicknesses=[4e-3], aperture_diameter=ap)
def resid(rx, **kw):
    ap = rx['aperture_diameter']
    h0 = np.linspace(-0.995*ap/2, 0.995*ap/2, 4001)
    r = trace_meridional(rx, lam, h0)
    NA = float(np.max(np.abs(r['Lx']))); dx = 0.30*lam/max(NA, 1e-6)
    N = int(2**np.ceil(np.log2(1.45*ap/dx)))
    E = np.ones((N, N), dtype=np.complex128)
    Eo = apply_real_lens(E, prescription=rx, wavelength=lam, dx=dx, **kw)
    x = (np.arange(N)-N/2)*dx
    ph = np.unwrap(np.angle(Eo[N//2])); m = np.abs(x) <= 0.85*ap/2
    W = ph[m]/k0; o = np.argsort(r['x'])
    d = W - np.interp(x[m], r['x'][o], r['opl'][o]); d -= d.mean()
    return float(np.sqrt(np.mean(d**2)))*1e9, N
print(f"{'case':<28}{'default[nm]':>13}{'slant[nm]':>12}{'gain':>8}   N")
for nm, rx in (('plano-cvx curved-first', pc()), ('plano-cvx flat-first', pc(flip=True)),
               ('biconvex R=+-60', bx()), ('biconvex R=+-25', bx(R=25e-3)),
               ('meniscus R=20/25', men()), ('parabolic asphere', asph())):
    a, N = resid(rx); b, _ = resid(rx, slant_correction=True)
    print(f"{nm:<28}{a:13.4f}{b:12.4f}{a/b:8.3f}   {N}")

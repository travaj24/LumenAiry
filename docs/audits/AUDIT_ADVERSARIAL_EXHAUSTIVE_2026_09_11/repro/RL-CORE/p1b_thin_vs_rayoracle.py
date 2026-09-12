"""Probe 1b: default 'thin' path exit-plane OPD vs independent ray oracle,
with Nyquist-safe sampling chosen from the traced exit NA.
"""
import sys, os
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from oracle import trace_meridional
from lumenairy.elements._lens_real import apply_real_lens

lam = 632.8e-9
k0 = 2 * np.pi / lam


def pc(R=50e-3, t=3e-3, ap=4e-3, glass='N-BK7', flip=False):
    if not flip:
        s = [dict(radius=R, glass_before='AIR', glass_after=glass),
             dict(radius=float('inf'), glass_before=glass, glass_after='AIR')]
    else:
        s = [dict(radius=float('inf'), glass_before='AIR', glass_after=glass),
             dict(radius=-R, glass_before=glass, glass_after='AIR')]
    return dict(surfaces=s, thicknesses=[t], aperture_diameter=ap)


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


def run(name, rx, nyq=0.30, frac=0.85, N=None, verbose=True, **kw):
    ap = rx['aperture_diameter']
    h0 = np.linspace(-0.995 * ap / 2, 0.995 * ap / 2, 8001)
    r = trace_meridional(rx, lam, h0)
    NA = float(np.max(np.abs(r['Lx'])))
    dx = nyq * lam / max(NA, 1e-6)
    if N is None:
        N = int(2 ** np.ceil(np.log2(1.45 * ap / dx)))
    x = (np.arange(N) - N / 2) * dx
    E = np.ones((N, N), dtype=np.complex128)
    Eo = apply_real_lens(E, prescription=rx, wavelength=lam, dx=dx, **kw)
    row = Eo[N // 2]
    ph = np.unwrap(np.angle(row))
    m = np.abs(x) <= frac * ap / 2
    xs = x[m]
    W_model = ph[m] / k0
    order = np.argsort(r['x'])
    W_ray = np.interp(xs, r['x'][order], r['opl'][order])
    d = W_model - W_ray
    d0 = d - d.mean()
    # also fit-and-remove piston+tilt (tilt is meaningless here, sym system)
    if verbose:
        print(f"--- {name}: N={N} dx={dx*1e6:.4f}um NA={NA:.5f} ap={ap*1e3:.2f}mm {kw}")
        print(f"    PV  {np.ptp(d)*1e9:10.4f} nm ({np.ptp(d)/lam:9.6f} w)   "
              f"RMS {np.sqrt(np.mean(d0**2))*1e9:9.4f} nm "
              f"({np.sqrt(np.mean(d0**2))/lam:9.6f} w)")
    return np.sqrt(np.mean(d0 ** 2)), np.ptp(d), NA, dx, N


if __name__ == '__main__':
    print("=== absolute OPL agreement (includes n*t piston) ===")
    run('plano-convex curved-first R=+50, 4mm', pc())
    run('plano-convex flat-first R=-50, 4mm', pc(flip=True))
    run('biconvex R=+-60, 4mm', bx())
    run('cemented doublet, 4mm', dbl())
    print()
    print("=== sag*theta^2 scaling: biconvex, fixed 4mm aperture, R varied ===")
    print("   (theta_max ~ NA; sag_max ~ ap^2/8R ; predicted ~ sag*theta^2)")
    rows = []
    for R in (240e-3, 120e-3, 60e-3, 30e-3, 15e-3):
        rms, pv, NA, dx, N = run(f'  R={R*1e3:.0f}mm', bx(R=R, t=4e-3, ap=4e-3))
        sag = (2e-3) ** 2 / (2 * R)
        rows.append((R, rms, NA, sag, sag * NA ** 2))
    print()
    print(f"{'R[mm]':>8} {'rms[nm]':>12} {'NA':>9} {'sag[um]':>10} "
          f"{'sag*NA^2[nm]':>14} {'ratio':>10}")
    for R, rms, NA, sag, pred in rows:
        print(f"{R*1e3:8.0f} {rms*1e9:12.5f} {NA:9.5f} {sag*1e6:10.4f} "
              f"{pred*1e9:14.5f} {rms/pred:10.4f}")

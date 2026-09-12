"""Probe 1: default 'thin' path exit-plane OPD vs an independent ray-trace OPL.

Also checks the SIGN of the imprinted phase (converging lens => phase
decreases with r under exp(+ikz)).
"""
import sys, os
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from oracle import trace_meridional
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.glass import get_glass_index

lam = 632.8e-9
k0 = 2 * np.pi / lam


def rx_planoconvex(R=50e-3, t=3e-3, ap=9e-3, glass='N-BK7'):
    return dict(surfaces=[dict(radius=R, glass_before='AIR', glass_after=glass),
                          dict(radius=float('inf'), glass_before=glass,
                               glass_after='AIR')],
                thicknesses=[t], aperture_diameter=ap)


def rx_biconvex(R=60e-3, t=4e-3, ap=9e-3, glass='N-BK7'):
    return dict(surfaces=[dict(radius=R, glass_before='AIR', glass_after=glass),
                          dict(radius=-R, glass_before=glass, glass_after='AIR')],
                thicknesses=[t], aperture_diameter=ap)


def rx_doublet(ap=9e-3):
    # loosely AC254-050-A-ish cemented doublet
    return dict(surfaces=[
        dict(radius=33.3e-3, glass_before='AIR', glass_after='N-BAF10'),
        dict(radius=-22.28e-3, glass_before='N-BAF10', glass_after='N-SF6HT'),
        dict(radius=-291.07e-3, glass_before='N-SF6HT', glass_after='AIR')],
        thicknesses=[9.0e-3, 2.5e-3], aperture_diameter=ap)


def model_phase(rx, N=2048, dx=None, ap=None, **kw):
    ap = ap or rx['aperture_diameter']
    if dx is None:
        dx = ap / (0.8 * N)
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.ones((N, N), dtype=np.complex128)
    Eo = apply_real_lens(E, prescription=rx, wavelength=lam, dx=dx, **kw)
    return x, Eo, dx


def run(name, rx, N=2048, frac=0.9, **kw):
    ap = rx['aperture_diameter']
    x, Eo, dx = model_phase(rx, N=N, **kw)
    row = Eo[N // 2]                       # y = 0 cut... careful: y index N/2 => y=0
    ph = np.unwrap(np.angle(row))
    # restrict to inside the aperture
    m = np.abs(x) <= frac * ap / 2
    xs = x[m]
    W_model = ph[m] / k0                   # OPL (m), up to a constant

    # ray oracle: launch rays at many heights, record exit height + OPL
    h0 = np.linspace(-0.98 * ap / 2, 0.98 * ap / 2, 4001)
    r = trace_meridional(rx, lam, h0)
    # eikonal at exit plane, as a function of the landing coordinate
    order = np.argsort(r['x'])
    W_ray = np.interp(xs, r['x'][order], r['opl'][order])

    # remove piston only (both are absolute OPL up to a constant)
    d = (W_model - W_model[len(W_model) // 2]) - (W_ray - W_ray[len(W_ray) // 2])
    # also remove best-fit piston over the window for an rms number
    d0 = d - d.mean()
    print(f"--- {name}  N={N} dx={dx*1e6:.4f}um  ap={ap*1e3:.2f}mm  kw={kw}")
    print(f"    peak-valley residual  : {np.ptp(d)*1e9:12.4f} nm  "
          f"({np.ptp(d)/lam:8.5f} waves)")
    print(f"    rms residual (piston-free): {np.sqrt(np.mean(d0**2))*1e9:8.4f} nm "
          f"({np.sqrt(np.mean(d0**2))/lam:8.5f} waves)")
    # sign check: converging lens phase must DECREASE with |x|
    i0 = np.argmin(np.abs(xs))
    iedge = np.argmin(np.abs(xs - 0.8 * ap / 2))
    print(f"    phase(edge)-phase(axis)   : {(ph[m][iedge]-ph[m][i0]):+.4f} rad "
          f"(negative => converging, per CONVENTIONS exp(+ikz))")
    return xs, d


if __name__ == '__main__':
    np.set_printoptions(precision=6)
    print("n(N-BK7) =", get_glass_index('N-BK7', lam))
    run('plano-convex R=+50 flat', rx_planoconvex())
    run('biconvex R=+-60', rx_biconvex())
    run('cemented doublet', rx_doublet())
    # sag*theta^2 scaling claim: shrink R (faster) and watch residual grow
    print()
    print("== sag*theta^2 scaling: biconvex, aperture fixed, R varied ==")
    for R in (120e-3, 60e-3, 30e-3, 20e-3):
        rx = rx_biconvex(R=R, t=4e-3, ap=6e-3)
        xs, d = run(f'biconvex R={R*1e3:.0f}mm', rx, N=2048)

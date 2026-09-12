"""Probe 6b: replicate the Seidel block's internals verbatim and diagnose."""
import sys, os
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from oracle import trace_meridional
from lumenairy.elements.lenses import surface_sag_general as _sg
from lumenairy.glass import get_glass_index
from lumenairy.raytrace import (_make_bundle as mkb,
                                surfaces_from_prescription as sfp, trace as rt)

lam = 632.8e-9


def pc(R=50e-3, t=3e-3, ap=4e-3, glass='N-BK7'):
    return dict(surfaces=[dict(radius=R, glass_before='AIR', glass_after=glass),
                          dict(radius=float('inf'), glass_before=glass,
                               glass_after='AIR')],
                thicknesses=[t], aperture_diameter=ap)


def dbl(ap=8e-3):
    return dict(surfaces=[
        dict(radius=33.3e-3, glass_before='AIR', glass_after='N-BAF10'),
        dict(radius=-22.28e-3, glass_before='N-BAF10', glass_after='N-SF6HT'),
        dict(radius=-291.07e-3, glass_before='N-SF6HT', glass_after='AIR')],
        thicknesses=[9.0e-3, 2.5e-3], aperture_diameter=ap)


def seidel_internals(rx, order=6):
    aperture = rx['aperture_diameter']
    surfaces = rx['surfaces']
    r_pupil = 0.5 * aperture
    n_fan = 41
    h_fan = np.linspace(-0.9 * r_pupil, 0.9 * r_pupil, n_fan)
    z = np.zeros_like(h_fan)
    fan = mkb(x=h_fan, y=z, L=z, M=z, wavelength=lam)
    res = rt(fan, sfp(rx), lam)
    fr = res.image_rays
    alive = fr.alive
    opl_ray = fr.opd[alive]
    h_alive = h_fan[alive]
    opl_analytic = np.zeros_like(h_alive)
    for s in surfaces:
        n1 = get_glass_index(s['glass_before'], lam)
        n2 = get_glass_index(s['glass_after'], lam)
        opl_analytic = opl_analytic + (n2 - n1) * _sg(
            h_alive * h_alive, s['radius'], s.get('conic', 0.0),
            s.get('aspheric_coeffs'))
    i_ax = int(np.argmin(np.abs(h_alive)))
    delta_ray = opl_ray - opl_ray[i_ax]
    opl_wave_rel = -(opl_analytic - opl_analytic[i_ax])
    correction = delta_ray - opl_wave_rel
    rho = h_alive / r_pupil
    even = np.arange(2, max(2, order) + 2, 2)
    A = np.column_stack([rho ** p for p in even])
    coeffs, *_ = np.linalg.lstsq(A, correction, rcond=None)
    return dict(h=h_alive, opl_ray=opl_ray, delta_ray=delta_ray,
                opl_wave_rel=opl_wave_rel, correction=correction,
                coeffs=coeffs, even=even, rho=rho, r_pupil=r_pupil,
                rms=float(np.sqrt(np.mean(correction ** 2))))


for nm, rx in (('plano-convex 4mm', pc()), ('doublet 8mm', dbl())):
    d = seidel_internals(rx)
    print(f"=== {nm} ===")
    print(f"  fan half-width 0.9*r_pupil = {0.9*d['r_pupil']*1e3:.3f} mm")
    print(f"  delta_ray   edge = {d['delta_ray'][-1]*1e9:14.3f} nm")
    print(f"  opl_wave_rel edge= {d['opl_wave_rel'][-1]*1e9:14.3f} nm")
    print(f"  correction  edge = {d['correction'][-1]*1e9:14.3f} nm   "
          f"rms = {d['rms']*1e9:.3f} nm  (gate is 5 nm)")
    print(f"  fitted coeffs (powers {list(d['even'])}) [m]: "
          + ', '.join(f'{c:+.6e}' for c in d['coeffs']))
    print(f"    -> defocus (rho^2) term at rim: "
          f"{d['coeffs'][0]*1e9:+.3f} nm")
    # independent oracle: ray OPL back-projected to the EXIT VERTEX plane
    r = trace_meridional(rx, lam, d['h'])
    true_rel = (r['opl'] - r['opl'][int(np.argmin(np.abs(d['h'])))])
    print(f"  INDEPENDENT oracle: OPL(exit vertex plane) rel edge = "
          f"{true_rel[-1]*1e9:14.3f} nm   (vs delta_ray "
          f"{d['delta_ray'][-1]*1e9:.3f} nm)")
    # what the library's own traced OPD is referenced to
    print(f"  difference (library delta_ray - oracle) edge = "
          f"{(d['delta_ray'][-1]-true_rel[-1])*1e9:.3f} nm")
    # true residual of the thin model, from the oracle: model OPL is
    # -(opl_analytic) + pistons; residual = true - model
    resid = true_rel - d['opl_wave_rel']
    print(f"  TRUE thin-model residual (oracle) edge = {resid[-1]*1e9:.3f} nm, "
          f"rms = {np.sqrt(np.mean((resid-resid.mean())**2))*1e9:.3f} nm")
    print()

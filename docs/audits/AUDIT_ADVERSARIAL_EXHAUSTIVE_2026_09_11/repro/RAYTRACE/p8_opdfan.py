"""RAYTRACE probe 5/12: opd_fan_data reference convention.

(a) on-axis: OPL-difference-at-the-final-plane vs the reference-sphere
    wavefront W(rho) = OPL(->image point) - OPL_chief(->image point).
(b) off-axis: is there a spurious y*sin(field_angle) input-wavefront tilt?
"""
import sys
import numpy as np
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.raytrace import (trace, Surface, system_abcd, opd_fan_data,
                                 ray_fan_data, make_fan)
from lumenairy.raytrace.surface import RayBundle

WL = 587.6e-9


def singlet(R1, R2, t=3.6e-3, sd=12.5e-3, img=True):
    M, efl, bfl, ffl = system_abcd(
        [Surface(radius=R1, glass_before='air', glass_after='N-BK7',
                 thickness=t),
         Surface(radius=R2, glass_before='N-BK7', glass_after='air',
                 thickness=0.0)], WL)
    s = [Surface(radius=R1, semi_diameter=sd, glass_before='air',
                 glass_after='N-BK7', thickness=t, is_stop=True),
         Surface(radius=R2, semi_diameter=sd, glass_before='N-BK7',
                 glass_after='air', thickness=bfl if img else 0.0)]
    if img:
        s.append(Surface(radius=np.inf, semi_diameter=np.inf,
                         glass_before='air', glass_after='air',
                         label='image'))
    return s, efl, bfl


print('=' * 78)
print('(a) ON-AXIS: library opd_fan_data vs the reference-sphere wavefront')
print('=' * 78)
for R1, R2 in [(51.68e-3, np.inf), (np.inf, -51.68e-3)]:
    surfs, efl, bfl = singlet(R1, R2, img=True)
    n = 41
    py, opd_y, px, opd_x = opd_fan_data(surfs, WL, 12.5e-3, 0.0, n)
    # reference-sphere oracle on the SAME surfaces minus the image plane
    s2, efl2, bfl2 = singlet(R1, R2, img=False)
    ys = np.linspace(-12.5e-3, 12.5e-3, n)
    rb = RayBundle(x=np.zeros(n), y=ys, z=np.zeros(n), L=np.zeros(n),
                   M=np.zeros(n), N=np.ones(n), wavelength=WL,
                   alive=np.ones(n, bool), opd=np.zeros(n))
    im = trace(rb, s2, WL).image_rays
    Pimg = np.array([0.0, 0.0, bfl2])
    P = np.stack([im.x, im.y, im.z], axis=-1)
    seg = np.linalg.norm(Pimg[None, :] - P, axis=-1)
    W = (im.opd + seg)
    W = (W - W[n // 2]) / WL
    print(f'R1={R1:+.5g} R2={R2:+.5g}  f={efl*1e3:.3f} mm')
    print(f'  library opd_fan   at rho=1 : {opd_y[-1]:+10.4f} waves')
    print(f'  ref-sphere oracle at rho=1 : {W[-1]:+10.4f} waves')
    print(f'  difference                 : {opd_y[-1]-W[-1]:+10.4f} waves '
          f'({abs((opd_y[-1]-W[-1])/max(abs(W[-1]),1e-30))*100:.1f} % of the '
          f'true WFE)')
    # transverse aberration for context
    py2, ey, px2, ex = ray_fan_data(surfs, WL, 12.5e-3, 0.0, n)
    print(f'  transverse aberration at rho=1: {ey[-1]*1e6:+.1f} um ; '
          f'eps^2/(2L) = {(ey[-1]**2/(2*bfl))/WL:+.3f} waves')
    print()

print('=' * 78)
print('(b) OFF-AXIS: spurious input-wavefront tilt  y*sin(theta)')
print('=' * 78)
surfs, efl, bfl = singlet(51.68e-3, np.inf, img=True)
for fa_deg in (0.0, 0.5, 2.0, 5.0):
    fa = np.radians(fa_deg)
    py, opd_y, px, opd_x = opd_fan_data(surfs, WL, 12.5e-3, fa, 41)
    ok = np.isfinite(opd_y)
    # linear (tilt) term of the fan
    c = np.polyfit(py[ok], opd_y[ok], 4)
    tilt_pred = 12.5e-3 * np.sin(fa) / WL
    print(f'  field={fa_deg:4.1f} deg: fitted LINEAR term of the OPD fan = '
          f'{c[3]:+12.2f} waves ;  y_max*sin(theta)/lambda = '
          f'{tilt_pred:+12.2f} waves ; ratio = '
          f'{c[3]/tilt_pred if tilt_pred else float("nan"):+.4f}')
    print(f'              quartic term = {c[0]:+9.3f} waves, '
          f'quadratic = {c[2]:+9.3f} waves, PV of fan = '
          f'{np.nanmax(opd_y)-np.nanmin(opd_y):.2f} waves')

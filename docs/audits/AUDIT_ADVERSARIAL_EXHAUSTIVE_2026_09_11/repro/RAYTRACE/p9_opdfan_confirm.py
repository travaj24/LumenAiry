"""RAYTRACE probe 5b: confirm the opd_fan_data defect quantitatively.

Prediction: opd_fan_data(rho) - W_refsphere(rho) = -n' * eps(rho) * sin(theta')
            (first order in the transverse aberration eps).
Controls: (i) an aberration-free system -> the two agree exactly;
          (ii) a very slow (low-eps) system -> agreement improves as eps^1.
"""
import sys
import numpy as np
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.raytrace import (trace, Surface, system_abcd, opd_fan_data,
                                 ray_fan_data)
from lumenairy.raytrace.surface import RayBundle

WL = 587.6e-9


def refsphere_W(s_nofocus, bfl, ys):
    n = len(ys)
    rb = RayBundle(x=np.zeros(n), y=ys.copy(), z=np.zeros(n), L=np.zeros(n),
                   M=np.zeros(n), N=np.ones(n), wavelength=WL,
                   alive=np.ones(n, bool), opd=np.zeros(n))
    im = trace(rb, s_nofocus, WL).image_rays
    P = np.stack([im.x, im.y, im.z], axis=-1)
    Pimg = np.array([0.0, 0.0, bfl])
    seg = np.linalg.norm(Pimg[None, :] - P, axis=-1)
    W = im.opd + seg
    return (W - W[n // 2]) / WL, im


print('=' * 78)
print('Prediction check: d = opd_fan - W_refsphere  ==  -eps * sin(theta_out)')
print('=' * 78)
for (R1, R2, sd, lbl) in [(51.68e-3, np.inf, 12.5e-3, 'f/4  plano-convex'),
                          (np.inf, -51.68e-3, 12.5e-3, 'f/4  convex-last'),
                          (51.68e-3, np.inf, 2.5e-3, 'f/20 plano-convex'),
                          (51.68e-3, np.inf, 1.0e-3, 'f/50 plano-convex')]:
    t = 3.6e-3
    base = [Surface(radius=R1, glass_before='air', glass_after='N-BK7',
                    thickness=t),
            Surface(radius=R2, glass_before='N-BK7', glass_after='air',
                    thickness=0.0)]
    _, efl, bfl, _ = system_abcd(base, WL)
    s_img = [Surface(radius=R1, semi_diameter=sd, glass_before='air',
                     glass_after='N-BK7', thickness=t, is_stop=True),
             Surface(radius=R2, semi_diameter=sd, glass_before='N-BK7',
                     glass_after='air', thickness=bfl),
             Surface(radius=np.inf, semi_diameter=np.inf, glass_before='air',
                     glass_after='air')]
    n = 41
    py, opd_y, px, opd_x = opd_fan_data(s_img, WL, sd, 0.0, n)
    ys = np.linspace(-sd, sd, n)
    W, im = refsphere_W(base, bfl, ys)
    # transverse aberration and exit angle of each marginal ray
    py2, ey, _, _ = ray_fan_data(s_img, WL, sd, 0.0, n)
    sin_out = im.M / np.sqrt(im.L**2 + im.M**2 + im.N**2)
    pred = -ey * sin_out / WL
    d = opd_y - W
    i = -1
    print(f'{lbl}: eps(rho=1)={ey[i]*1e6:+9.2f} um  sin(theta_out)={sin_out[i]:+.5f}')
    print(f'   opd_fan={opd_y[i]:+10.4f} w   W_ref={W[i]:+10.4f} w   '
          f'diff={d[i]:+10.4f} w   prediction -eps*sin={pred[i]:+10.4f} w '
          f'(match {abs(d[i]-pred[i]):.3f} w)')

print()
print('=' * 78)
print('CONTROL: aberration-free parabolic mirror -> both definitions agree')
print('=' * 78)
Rm = -200e-3
s_img = [Surface(radius=Rm, conic=-1.0, semi_diameter=25e-3,
                 glass_before='air', glass_after='air', is_mirror=True,
                 is_stop=True, thickness=-100e-3),
         Surface(radius=np.inf, semi_diameter=np.inf, glass_before='air',
                 glass_after='air')]
py, opd_y, px, opd_x = opd_fan_data(s_img, WL, 25e-3, 0.0, 21)
print(f'   parabola opd_fan PV = {np.nanmax(opd_y)-np.nanmin(opd_y):.3e} waves'
      '   (correctly ~0)')
s_img[0] = Surface(radius=Rm, conic=0.0, semi_diameter=25e-3,
                   glass_before='air', glass_after='air', is_mirror=True,
                   is_stop=True, thickness=-100e-3)
py, opd_y, px, opd_x = opd_fan_data(s_img, WL, 25e-3, 0.0, 21)
# reference-sphere oracle for the sphere mirror
ys = np.linspace(-25e-3, 25e-3, 21)
base = [Surface(radius=Rm, conic=0.0, semi_diameter=np.inf,
                glass_before='air', glass_after='air', is_mirror=True,
                thickness=0.0)]
W, im = refsphere_W(base, -100e-3, ys)
# the segment sign: rays travel -z, so use the signed distance
print(f'   SPHERICAL mirror opd_fan at rho=1 = {opd_y[-1]:+10.4f} waves')
print(f'   ref-sphere oracle  at rho=1 = {-W[-1]:+10.4f} waves '
      '(sign: rays travel -z so |seg| enters with the opposite sign)')
print(f'   Seidel -S1/8 (probe p4) = -12.207 um = '
      f'{-12.207e-6/WL:+10.4f} waves')

"""RAYTRACE probe 8c: the missing aspheric/conic Seidel term, quantified,
plus the proposed one-line fix validated against real rays.

POST-FIX READING NOTE (VERIFY-WP-A1, open item 10) -- the MEASUREMENTS below
are correct on the fixed library, but the NARRATIVE is pre-fix: WP-A1's R3
now adds the aspheric term inside ``seidel_coefficients``, so the ``S1_lib``
column already contains it and the ``S1_fixed`` column DOUBLE-COUNTS it.
Read ``S1_lib`` as the shipped value; ignore ``dS1_pred`` and ``S1_fixed``.
This file is audit evidence, so it is annotated rather than rewritten.

Proposed fix (library sign convention, code = -S_Welford):
    A4_eff = conic / (8 R^3) + aspheric_coeffs.get(4, 0.0)
    S1[i] += 8 * (n2 - n1) * A4_eff * y_marginal[i]**4
    S2[i] += that * (y_c/y_m)
    S3[i] += that * (y_c/y_m)**2
    S5[i] += that * (y_c/y_m)**3
"""
import sys
import numpy as np
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.raytrace import trace, Surface, system_abcd, seidel_coefficients
from lumenairy.raytrace.surface import RayBundle
from lumenairy.glass import get_glass_index

WL = 587.6e-9


def bundle(ys):
    n = len(ys)
    return RayBundle(x=np.zeros(n), y=np.asarray(ys, float), z=np.zeros(n),
                     L=np.zeros(n), M=np.zeros(n), N=np.ones(n),
                     wavelength=WL, alive=np.ones(n, bool), opd=np.zeros(n))


def realray_a4(surfs, semi_ap, n_img=1.0, n_rays=61, bfl_override=None):
    M, efl, bfl, ffl = system_abcd(surfs, WL)
    if bfl_override is not None:
        bfl = bfl_override
    s2 = [Surface(radius=s.radius, conic=s.conic,
                  aspheric_coeffs=s.aspheric_coeffs, semi_diameter=np.inf,
                  glass_before=s.glass_before, glass_after=s.glass_after,
                  is_mirror=s.is_mirror, thickness=s.thickness)
          for s in surfs]
    ys = np.linspace(1e-9, semi_ap, n_rays)
    res = trace(bundle(ys), s2, WL)
    img = res.image_rays
    Pimg = np.array([0.0, 0.0, bfl])
    P = np.stack([img.x, img.y, img.z], axis=-1)
    sgn = np.sign(bfl - img.z)
    seg = sgn * np.linalg.norm(Pimg[None, :] - P, axis=-1)
    opl = img.opd + n_img * seg
    W = opl - opl[0]
    rho = ys / semi_ap
    A = np.stack([rho**2, rho**4, rho**6], axis=-1)
    coef, *_ = np.linalg.lstsq(A, W, rcond=None)
    return coef[1], efl, bfl


n_bk7 = get_glass_index('N-BK7', WL)
print('=' * 78)
print('Aspheric A4 contribution:  predicted dS1_code = 8 (n2-n1) A4 h^4')
print('=' * 78)
semi_ap = 12.5e-3
for A4 in (0.0, -2.5e2, -5e2, -1e3, -2e3, +5e2):
    surfs = [Surface(radius=51.68e-3,
                     aspheric_coeffs=({4: A4} if A4 else None),
                     semi_diameter=12.5e-3, glass_before='air',
                     glass_after='N-BK7', thickness=3.6e-3, is_stop=True),
             Surface(radius=np.inf, semi_diameter=12.5e-3,
                     glass_before='N-BK7', glass_after='air', thickness=0.0)]
    sd, _ = seidel_coefficients(surfs, WL, field_angle=1e-6)
    a4, efl, bfl = realray_a4(surfs, semi_ap)
    h = sd['y_marginal'][0]
    pred = 8.0 * (n_bk7 - 1.0) * A4 * h**4
    S1_lib = sd['total']['S1']
    S1_fixed = S1_lib + pred
    print(f'A4={A4:+8.1f}: h_marg={h*1e3:.4f} mm  S1_lib={S1_lib:+.6e} '
          f'dS1_pred={pred:+.6e}  S1_fixed={S1_fixed:+.6e}\n'
          f'            -> -S1_fixed/8 = {-S1_fixed/8*1e6:+9.4f} um  vs  '
          f'real-ray a4 = {a4*1e6:+9.4f} um   (rel err '
          f'{abs(-S1_fixed/8 - a4)/max(abs(a4),1e-12)*100:6.2f} %)')

print()
print('=' * 78)
print('Conic contribution: A4_eff = k/(8 R^3).  Parabolic mirror -> S1 = 0.')
print('=' * 78)
for k in (0.0, -0.5, -1.0, -1.5):
    R = -200e-3
    ms = [Surface(radius=R, conic=k, semi_diameter=25e-3, glass_before='air',
                  glass_after='air', is_mirror=True, is_stop=True,
                  thickness=0.0)]
    sd, _ = seidel_coefficients(ms, WL, field_angle=1e-6)
    h = sd['y_marginal'][0]
    n1, n2 = 1.0, -1.0          # Welford mirror
    A4_eff = k / (8 * R**3)
    pred = 8.0 * (n2 - n1) * A4_eff * h**4
    S1_lib = sd['total']['S1']
    # real-ray: reflected rays go -z, so the image plane is a NEGATIVE
    # thickness away (Zemax post-mirror convention).
    ms2 = [Surface(radius=R, conic=k, semi_diameter=np.inf,
                   glass_before='air', glass_after='air', is_mirror=True,
                   thickness=-100e-3),
           Surface(radius=np.inf, semi_diameter=np.inf, glass_before='air',
                   glass_after='air', label='img')]
    ys = np.linspace(1e-9, 25e-3, 61)
    res = trace(bundle(ys), ms2, WL)
    img = res.image_rays
    spread = (img.y.max() - img.y.min()) * 1e6
    # W(rho) at the paraxial focus (z = -100 mm in the mirror's frame)
    res1 = trace(bundle(ys), [Surface(radius=R, conic=k, semi_diameter=np.inf,
                                      glass_before='air', glass_after='air',
                                      is_mirror=True, thickness=0.0)], WL)
    im1 = res1.image_rays
    P = np.stack([im1.x, im1.y, im1.z], axis=-1)
    Pimg = np.array([0.0, 0.0, -100e-3])
    seg = np.linalg.norm(Pimg[None, :] - P, axis=-1)
    W = (im1.opd + seg) - (im1.opd[0] + np.linalg.norm(Pimg - P[0]))
    rho = ys / 25e-3
    A = np.stack([rho**2, rho**4, rho**6], axis=-1)
    coef, *_ = np.linalg.lstsq(A, W, rcond=None)
    print(f'k={k:+5.2f}: S1_lib={S1_lib:+.6e}  dS1_pred={pred:+.6e}  '
          f'S1_fixed={S1_lib+pred:+.6e}')
    print(f'         -> -S1_fixed/8 = {-(S1_lib+pred)/8*1e6:+9.4f} um ; '
          f'real-ray a4 = {coef[1]*1e6:+9.4f} um ; '
          f'spot spread at focus = {spread:.3f} um')

print()
print('=' * 78)
print('Mirror handling: post-mirror NEGATIVE thickness, alive/OPL sanity')
print('=' * 78)
R = -200e-3
ms2 = [Surface(radius=R, conic=-1.0, semi_diameter=np.inf,
               glass_before='air', glass_after='air', is_mirror=True,
               thickness=-100e-3),
       Surface(radius=np.inf, semi_diameter=np.inf, glass_before='air',
               glass_after='air')]
ys = np.array([0.0, 5e-3, 15e-3, 25e-3])
res = trace(bundle(ys), ms2, WL)
im = res.image_rays
print(f'  y at image plane : {np.round(im.y*1e9, 4)} nm')
print(f'  z                : {np.round(im.z*1e9, 4)} nm')
print(f'  N (direction)    : {im.N}')
print(f'  opd              : {im.opd}')
print(f'  opd - opd[0]     : {(im.opd-im.opd[0])}')
print('  (parabolic mirror, collimated in -> perfect focus: y == 0 and')
print('   equal OPL for every ray)')

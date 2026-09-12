"""RAYTRACE probe 7+8: system_abcd closed forms; Seidel vs a real-ray OPD fit;
the missing conic/aspheric Seidel contribution.
"""
import sys
import numpy as np
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.raytrace import (trace, Surface, make_fan, system_abcd,
                                 seidel_coefficients, find_paraxial_focus)
from lumenairy.raytrace.surface import RayBundle
from lumenairy.glass import get_glass_index

WL = 587.6e-9
np.set_printoptions(precision=10, suppress=False)


def bundle(ys, M0=0.0):
    n = len(ys)
    N0 = float(np.sqrt(1 - M0 ** 2))
    return RayBundle(x=np.zeros(n), y=np.asarray(ys, float), z=np.zeros(n),
                     L=np.zeros(n), M=np.full(n, M0), N=np.full(n, N0),
                     wavelength=WL, alive=np.ones(n, bool), opd=np.zeros(n))


print('=' * 76)
print('7. system_abcd EFL/BFL/FFL vs closed forms')
print('=' * 76)
n_bk7 = get_glass_index('N-BK7', WL)
print(f'n(N-BK7, 587.6 nm) = {n_bk7:.8f}')
for (R1, R2, t) in [(50e-3, -50e-3, 6e-3), (50e-3, np.inf, 4e-3),
                    (np.inf, -50e-3, 4e-3), (25e-3, 40e-3, 5e-3),
                    (-60e-3, 60e-3, 3e-3)]:
    surfs = [Surface(radius=R1, glass_before='air', glass_after='N-BK7',
                     thickness=t),
             Surface(radius=R2, glass_before='N-BK7', glass_after='air',
                     thickness=0.0)]
    M, efl, bfl, ffl = system_abcd(surfs, WL)
    n = n_bk7
    c1 = 0.0 if np.isinf(R1) else 1 / R1
    c2 = 0.0 if np.isinf(R2) else 1 / R2
    inv_f = (n - 1) * (c1 - c2 + (n - 1) * t * c1 * c2 / n)
    f = 1 / inv_f
    bfl_cf = f * (1 - (n - 1) * t * c1 / n)
    ffl_cf = f * (1 + (n - 1) * t * c2 / n)
    print(f'R1={R1:+.4g} R2={R2:+.4g} t={t*1e3:.1f}mm | '
          f'efl {efl*1e3:+11.6f} vs {f*1e3:+11.6f} '
          f'(d={abs(efl-f):.2e}) | bfl {bfl*1e3:+11.6f} vs {bfl_cf*1e3:+11.6f} '
          f'(d={abs(bfl-bfl_cf):.2e}) | ffl {ffl*1e3:+11.6f} vs '
          f'{ffl_cf*1e3:+11.6f} (d={abs(ffl-ffl_cf):.2e})')

print('\n  mirror: single concave R=-100 mm -> f = R/2 = -50 mm (or +50 by')
print('  the library note).  system_abcd says:')
ms = [Surface(radius=-100e-3, glass_before='air', glass_after='air',
              is_mirror=True, thickness=0.0)]
M, efl, bfl, ffl = system_abcd(ms, WL)
print(f'    efl={efl*1e3:+.6f} mm  bfl={bfl*1e3:+.6f} mm  ffl={ffl*1e3:+.6f} mm')

print('\n  two thin lenses f1=100, f2=100, d=50 mm (thin-lens combo):')
print('    expected 1/f = 1/f1 + 1/f2 - d/(f1 f2) -> f = 66.6667 mm,')
print('    BFL = f (1 - d/f1) = 33.3333 mm')
# build each thin lens as a symmetric biconvex with n=1.5 and t->0
n_t = 1.5
from lumenairy.raytrace.trace import _register_fixed_index
_register_fixed_index('__p4glass__', n_t, WL)
def thin(f):
    R = 2 * f * (n_t - 1)
    return [Surface(radius=R, glass_before='air', glass_after='__p4glass__',
                    thickness=0.0),
            Surface(radius=-R, glass_before='__p4glass__', glass_after='air',
                    thickness=0.0)]
tl = thin(100e-3)
tl[1].thickness = 50e-3
surfs = tl + thin(100e-3)
M, efl, bfl, ffl = system_abcd(surfs, WL)
print(f'    efl={efl*1e3:.6f} mm  bfl={bfl*1e3:.6f} mm')

print()
print('=' * 76)
print('8. Seidel S1 vs a REAL-RAY OPD fit (plano-convex, both orientations)')
print('=' * 76)


def wfe_from_trace(surfs, semi_ap, n_rays=41, n_img=1.0):
    """W(rho) referenced to the paraxial image point, from real rays."""
    M, efl, bfl, ffl = system_abcd(surfs, WL)
    s2 = [Surface(radius=s.radius, conic=s.conic,
                  aspheric_coeffs=s.aspheric_coeffs,
                  semi_diameter=np.inf, glass_before=s.glass_before,
                  glass_after=s.glass_after, is_mirror=s.is_mirror,
                  thickness=s.thickness) for s in surfs]
    ys = np.linspace(0, semi_ap, n_rays)
    res = trace(bundle(ys), s2, WL)
    img = res.image_rays
    # paraxial image point in the LAST surface's local frame: z = bfl
    Pimg = np.array([0.0, 0.0, bfl])
    P = np.stack([img.x, img.y, img.z], axis=-1)
    seg = np.linalg.norm(Pimg[None, :] - P, axis=-1)
    opl = img.opd + n_img * seg
    W = opl - opl[0]
    rho = ys / semi_ap
    return rho, W, efl, bfl


for label, (R1, R2) in [('convex-first (R1=+51.68, R2=inf)', (51.68e-3, np.inf)),
                        ('flat-first   (R1=inf, R2=-51.68)', (np.inf, -51.68e-3))]:
    t = 3.6e-3
    surfs = [Surface(radius=R1, semi_diameter=12.5e-3, glass_before='air',
                     glass_after='N-BK7', thickness=t, is_stop=True),
             Surface(radius=R2, semi_diameter=12.5e-3, glass_before='N-BK7',
                     glass_after='air', thickness=0.0)]
    semi_ap = 12.5e-3
    rho, W, efl, bfl = wfe_from_trace(surfs, semi_ap)
    # fit W = a2 rho^2 + a4 rho^4 + a6 rho^6
    A = np.stack([rho ** 2, rho ** 4, rho ** 6], axis=-1)
    coef, *_ = np.linalg.lstsq(A, W, rcond=None)
    a2, a4, a6 = coef
    sd, abcd = seidel_coefficients(surfs, WL, field_angle=1e-6)
    S1 = sd['total']['S1']
    W040_seidel = -S1 / 8.0          # library convention: S_Welford = -S1_code
    print(f'{label}:  f={efl*1e3:.4f} mm  bfl={bfl*1e3:.4f} mm')
    print(f'   real-ray fit : a2={a2*1e6:+10.4f} um  a4={a4*1e6:+10.4f} um  '
          f'a6={a6*1e6:+.4f} um')
    print(f'   Seidel S1={S1:+.6e} m  ->  S1/8 = {S1/8*1e6:+10.4f} um ; '
          f'-S1/8 = {W040_seidel*1e6:+10.4f} um')
    print(f'   ratio a4 / (S1/8) = {a4/(S1/8):+.4f}')
    print()

print('  Classic check: the ratio of a4 between the two orientations should be')
print('  ~4 (n=1.52 plano-convex).  Compare the two a4 rows above.')

print()
print('=' * 76)
print('8c. CONIC / ASPHERIC contribution to the Seidel sums')
print('=' * 76)
print('A PARABOLIC mirror (k = -1) at infinite conjugate has EXACTLY zero')
print('spherical aberration.  seidel_coefficients should report S1 = 0.')
for k in (0.0, -1.0, -2.0):
    ms = [Surface(radius=-200e-3, conic=k, semi_diameter=25e-3,
                  glass_before='air', glass_after='air', is_mirror=True,
                  is_stop=True, thickness=0.0)]
    sd, _ = seidel_coefficients(ms, WL, field_angle=1e-6)
    # real-ray check: spot radius at paraxial focus
    ys = np.linspace(1e-4, 25e-3, 25)
    s2 = [Surface(radius=-200e-3, conic=k, semi_diameter=np.inf,
                  glass_before='air', glass_after='air', is_mirror=True,
                  thickness=100e-3),
          Surface(radius=np.inf, semi_diameter=np.inf, glass_before='air',
                  glass_after='air', label='img')]
    res = trace(bundle(ys), s2, WL)
    img = res.image_rays
    print(f'  conic k={k:+.1f}:  seidel S1 = {sd["total"]["S1"]:+.6e} m ; '
          f'real-ray transverse spread at paraxial focus = '
          f'{(img.y.max()-img.y.min())*1e6:9.3f} um')

print()
print('An A4 aspheric term on a singlet front surface changes the real')
print('spherical aberration but not the reported Seidel S1:')
for A4 in (0.0, -5e2, -2e3):
    surfs = [Surface(radius=51.68e-3, aspheric_coeffs=({4: A4} if A4 else None),
                     semi_diameter=12.5e-3, glass_before='air',
                     glass_after='N-BK7', thickness=3.6e-3, is_stop=True),
             Surface(radius=np.inf, semi_diameter=12.5e-3,
                     glass_before='N-BK7', glass_after='air', thickness=0.0)]
    sd, _ = seidel_coefficients(surfs, WL, field_angle=1e-6)
    rho, W, efl, bfl = wfe_from_trace(surfs, 12.5e-3)
    A = np.stack([rho ** 2, rho ** 4, rho ** 6], axis=-1)
    coef, *_ = np.linalg.lstsq(A, W, rcond=None)
    print(f'  A4={A4:+9.1f} m^-3 : seidel S1={sd["total"]["S1"]:+.6e} '
          f'(S1/8={sd["total"]["S1"]/8*1e6:+9.4f} um)  real-ray '
          f'a4={coef[1]*1e6:+9.4f} um')

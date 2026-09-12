"""RAYTRACE probe: misc robustness / convention checks."""
import sys
import numpy as np
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.raytrace import trace, Surface
from lumenairy.raytrace.surface import RayBundle, RAY_OK
from lumenairy.raytrace.intersection import (_transfer, _intersect_surface,
                                              _refract)
from lumenairy.glass import get_glass_index

WL = 1.31e-6


def mk(x, y, z, L, M, N):
    n = len(x)
    return RayBundle(x=np.asarray(x, float), y=np.asarray(y, float),
                     z=np.asarray(z, float), L=np.asarray(L, float),
                     M=np.asarray(M, float), N=np.asarray(N, float),
                     wavelength=WL, alive=np.ones(n, bool), opd=np.zeros(n))


print('=' * 76)
print('A. _transfer with a GRAZING ray (|N| <= 1e-30): teleport + immortal?')
print('=' * 76)
r = mk([0.0, 0.0], [0.0, 1e-3], [1e-4, 1e-4], [1.0, 0.0], [0.0, 1.0],
       [0.0, 0.0])
print(f'   before: z={r.z}  alive={r.alive}  opd={r.opd}')
_transfer(r, 10e-3, 1.0)
print(f'   after _transfer(t=10mm): z={r.z}  x={r.x}  alive={r.alive}  '
      f'opd={r.opd}  error_code={r.error_code}')
print('   -> the grazing ray was TELEPORTED to the next vertex plane (z=0)')
print('      with ZERO OPL and is still alive.  _intersect_surface has the')
print('      R-4 graze guard; _transfer does not.')

print()
print('=' * 76)
print('B. error_code "first-failure-wins" claim in _refract')
print('=' * 76)
import inspect
src = inspect.getsource(_refract)
i = src.index('newly_tir')
print('   ' + '\n   '.join(src[i:i + 330].split('\n')))
print('   -> the comment promises RAY_TIR overwrites only RAY_OK, but the')
print('      np.where(newly_tir, ...) is unconditional (cf. the corrected')
print('      aperture block 30 lines below, which DOES use')
print('      `first_failure = clipped & (error_code == RAY_OK)`).')

print()
print('=' * 76)
print('C. spherical FAST PATH vs the Newton path -- same answer?')
print('=' * 76)
rng = np.random.default_rng(4)
n = 5000
h = 10e-3 * np.sqrt(rng.random(n))
th = 2 * np.pi * rng.random(n)
L = 0.1 * (rng.random(n) - .5)
M = 0.1 * (rng.random(n) - .5)
N = np.sqrt(1 - L**2 - M**2)
for R in (50e-3, -50e-3, 200e-3):
    a = mk(h * np.cos(th), h * np.sin(th), np.zeros(n), L, M, N)
    b = a.copy()
    _intersect_surface(a, Surface(radius=R, conic=0.0), 1.0)      # fast path
    # force the Newton path with a negligible conic
    _intersect_surface(b, Surface(radius=R, conic=1e-300), 1.0)
    print(f'   R={R:+.3f}: max |t_fast - t_newton| = '
          f'{np.abs(a.opd-b.opd).max():.3e} m  '
          f'(alive {a.alive.sum()} vs {b.alive.sum()})')

print()
print('=' * 76)
print('D. DOE grating kick INSIDE a medium: is the n2 factor applied?')
print('=' * 76)
# A grating written on the glass side of an air->BK7 interface.  The
# correct law is  n2 L2 = n1 L1 + m lam / Lambda  (vacuum lam), i.e.
# dL = m lam / (n2 Lambda).  The library adds m lam / Lambda.
n2 = get_glass_index('N-BK7', WL)
surfs = [Surface(radius=np.inf, semi_diameter=np.inf, glass_before='air',
                 glass_after='N-BK7', thickness=1e-3),
         Surface(radius=np.inf, semi_diameter=np.inf, glass_before='N-BK7',
                 glass_after='N-BK7', thickness=0.0)]
period = 5e-6
rb = mk([0.0], [0.0], [0.0], [0.0], [0.0], [1.0])
res = trace(rb, surfs, WL, surface_diffraction={0: (1, 0, period, np.inf)})
L_out = res.image_rays.L[0]
print(f'   n(BK7)={n2:.6f}  period={period*1e6:.1f} um  lambda={WL*1e6:.3f} um')
print(f'   library L inside the glass       = {L_out:.8f}')
print(f'   grating eq.  m lam/(n2 Lambda)   = {WL/(n2*period):.8f}')
print(f'   library value / correct value    = {L_out/(WL/(n2*period)):.6f}'
      f'   (= n2 if the index factor is missing)')

print()
print('=' * 76)
print('E. make_rings pupil-area weighting (documented) -- measured bias')
print('=' * 76)
from lumenairy.raytrace import make_rings
rb = make_rings(12.5e-3, 6, 36, 0.0, WL)
rr = np.sqrt(rb.x**2 + rb.y**2)
print(f'   mean r / R = {rr.mean()/12.5e-3:.4f}  (area-uniform would be 2/3 '
      f'= 0.6667)')
print(f'   mean r^2 / R^2 = {(rr**2).mean()/12.5e-3**2:.4f} '
      f'(area-uniform = 0.5)')

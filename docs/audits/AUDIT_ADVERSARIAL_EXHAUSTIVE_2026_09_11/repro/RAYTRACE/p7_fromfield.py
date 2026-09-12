"""RAYTRACE probe 11: rays_from_field -- sign, band limit, edge pixels,
converging spherical wave, and the wrapped-OPD seed.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.raytrace import rays_from_field

WL = 1.0e-6
k0 = 2 * np.pi / WL


def grid(N, dx):
    c = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(c, c, indexing='xy')     # rows = y, cols = x
    return X, Y


print('=' * 76)
print('1. SIGN: tilted plane wave E = exp(+i k (L x + M y))  ->  rays must')
print('   come back with the SAME (L, M)  [exp(-i w t) / exp(+ikz) convention]')
print('=' * 76)
N, dx = 128, 2e-6
X, Y = grid(N, dx)
for (Lt, Mt) in [(0.05, 0.0), (0.0, -0.08), (0.03, 0.04)]:
    E = np.exp(1j * k0 * (Lt * X + Mt * Y)) * np.exp(-(X**2 + Y**2) / (2 * (60e-6)**2))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rb = rays_from_field(E, dx=dx, wavelength=WL, n_rays=200,
                             placement='uniform')
    print(f'   true (L,M)=({Lt:+.3f},{Mt:+.3f})  recovered median '
          f'({np.median(rb.L):+.6f},{np.median(rb.M):+.6f})  '
          f'max dev L {np.abs(rb.L-Lt).max():.2e}  M {np.abs(rb.M-Mt).max():.2e}')

print()
print('=' * 76)
print('2. BAND LIMIT of the 2-pixel central phase-ratio estimator')
print('   grid supports |L| up to  lambda/(2 dx);  estimator aliases above')
print('   lambda/(4 dx)  because it divides arg(E[j+1] conj(E[j-1])) by 2 dx')
print('=' * 76)
dx = 2e-6
Lmax_grid = WL / (2 * dx)
Lmax_est = WL / (4 * dx)
print(f'   dx={dx*1e6:.1f} um : grid Nyquist |L| <= {Lmax_grid:.4f} ; '
      f'estimator valid |L| < {Lmax_est:.4f}')
X, Y = grid(96, dx)
env = np.exp(-(X**2 + Y**2) / (2 * (40e-6)**2))
print('   L_true    L_recovered   note')
for Lt in (0.05, 0.10, 0.15, 0.20, 0.24, 0.26, 0.30, 0.40, 0.49):
    E = np.exp(1j * k0 * Lt * X) * env
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        rb = rays_from_field(E, dx=dx, wavelength=WL, n_rays=120,
                             placement='uniform')
    Lr = np.median(rb.L)
    flag = '' if abs(Lr - Lt) < 1e-3 else '   <-- ALIASED'
    print(f'   {Lt:6.3f}   {Lr:+10.6f}  {flag}')
print('   (a 1-pixel forward difference, arg(E[j+1] conj(E[j]))/dx, would')
print('    stay exact all the way to the grid Nyquist limit)')

print()
print('=' * 76)
print('3. EDGE PIXELS: the clip(ix+-1) boundary halves the recovered k')
print('=' * 76)
dx = 2e-6
N = 64
X, Y = grid(N, dx)
Lt = 0.05
E = np.exp(1j * k0 * Lt * X)              # uniform amplitude -> edges sampled
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    rb = rays_from_field(E, dx=dx, wavelength=WL, n_rays=N * N,
                         placement='uniform', intensity_threshold=0.0)
xs = np.round((rb.x / dx) + N // 2).astype(int)
edge = (xs == 0) | (xs == N - 1)
inner = ~edge
print(f'   inner pixels : L = {np.median(rb.L[inner]):+.6f}  (true {Lt})')
if edge.any():
    print(f'   EDGE columns : L = {np.median(rb.L[edge]):+.6f}  '
          f'ratio to truth = {np.median(rb.L[edge])/Lt:.4f}  (n={edge.sum()})')
else:
    print('   (no edge pixels in this sampling)')

print()
print('=' * 76)
print('4. CONVERGING spherical wave: rays must point at the focus')
print('=' * 76)
f = 5e-3
dx = 1e-6
N = 128
X, Y = grid(N, dx)
R = np.sqrt(X**2 + Y**2 + f**2)
E = np.exp(-1j * k0 * R) * np.exp(-(X**2 + Y**2) / (2 * (40e-6)**2))
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    rb = rays_from_field(E, dx=dx, wavelength=WL, n_rays=400,
                         placement='uniform')
# each ray: where does it cross z = f ?
xf = rb.x + rb.L / rb.N * f
yf = rb.y + rb.M / rb.N * f
print(f'   converging wave exp(-i k R): crossing at z=+f  rms radius = '
      f'{np.sqrt(np.mean(xf**2+yf**2))*1e9:.3f} nm  (should be ~0)')
xb = rb.x - rb.L / rb.N * f
yb = rb.y - rb.M / rb.N * f
print(f'                                 crossing at z=-f  rms radius = '
      f'{np.sqrt(np.mean(xb**2+yb**2))*1e6:.3f} um')
E2 = np.exp(+1j * k0 * R) * np.exp(-(X**2 + Y**2) / (2 * (40e-6)**2))
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    rb2 = rays_from_field(E2, dx=dx, wavelength=WL, n_rays=400,
                          placement='uniform')
xf2 = rb2.x + rb2.L / rb2.N * f
print(f'   diverging  wave exp(+i k R): crossing at z=+f rms radius = '
      f'{np.sqrt(np.mean(xf2**2+(rb2.y+rb2.M/rb2.N*f)**2))*1e6:.3f} um '
      f'(should be LARGE)')

print()
print('=' * 76)
print('5. OPD SEED is the WRAPPED phase angle(E)/k0')
print('=' * 76)
print(f'   converging-wave bundle: opd range = [{rb.opd.min()*1e9:.3f}, '
      f'{rb.opd.max()*1e9:.3f}] nm ; lambda/2 = {WL/2*1e9:.1f} nm')
print(f'   true OPL spread across the sampled aperture = '
      f'{(R.max()-f)*1e6:.3f} um = {(R.max()-f)/WL:.1f} waves')
print('   => opd is wrapped into (-lambda/2, lambda/2]; any consumer that')
print('      treats RayBundle.opd as a GEOMETRIC path (OPD fan, wavefront')
print('      fit, unwrap) sees sawtooth, not the true path.  exp(i k0 opd)')
print('      consumers (ray_to_beamlet / HFPI) are unaffected (mod 2 pi).')

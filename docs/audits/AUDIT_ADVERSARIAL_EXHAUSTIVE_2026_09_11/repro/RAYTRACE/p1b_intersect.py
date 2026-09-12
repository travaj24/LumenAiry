"""RAYTRACE probe 1b: corrected stable root + the h>|R| conic false-miss."""
import sys
import numpy as np
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.raytrace.surface import Surface, RayBundle
from lumenairy.raytrace.intersection import _intersect_surface
from lumenairy.raytrace import trace, make_fan, make_ray

rng = np.random.default_rng(7)


def mb(x, y, z, L, M, N):
    n = len(x)
    return RayBundle(x=x.copy(), y=y.copy(), z=z.copy(), L=L.copy(), M=M.copy(),
                     N=N.copy(), wavelength=1.31e-6,
                     alive=np.ones(n, bool), opd=np.zeros(n))


def stable_root(x0, y0, z0, L, M, N, R):
    """t^2 - 2 G t + F = 0 ; near root = F/(G + sign(G) sqrt(G^2-F))."""
    F = x0**2 + y0**2 + z0**2 - 2.0 * R * z0
    G = R * N - (x0 * L + y0 * M + z0 * N)
    disc = G * G - F
    root = np.sqrt(np.maximum(disc, 0.0))
    sg = np.where(G >= 0, 1.0, -1.0)
    denom = G + sg * root
    t_near = F / denom
    t_far = denom            # = G + sign(G) sqrt  (the large root)
    return t_near, t_far, disc


print('=' * 74)
print('A. pure-sphere fast path vs STABLE near root  (max |dt| over 20k rays)')
print('=' * 74)
worst = 0.0
for R in (0.1, 0.05, 0.01, 0.005, -0.05, -0.005, 1.0, 0.002):
    for hf in (0.3, 0.7, 0.95):
        h_max = abs(R) * hf
        n = 20000
        r = h_max * np.sqrt(rng.random(n))
        th = 2 * np.pi * rng.random(n)
        x0 = r * np.cos(th); y0 = r * np.sin(th); z0 = np.zeros(n)
        L = 0.05 * (rng.random(n) - .5); M = 0.05 * (rng.random(n) - .5)
        N = np.sqrt(1 - L**2 - M**2)
        rb = mb(x0, y0, z0, L, M, N)
        _intersect_surface(rb, Surface(radius=R, conic=0.0), n_medium=1.0)
        t_lib = rb.opd
        t_ref, t_far, disc = stable_root(x0, y0, z0, L, M, N, R)
        dt = np.abs(t_lib - t_ref)
        worst = max(worst, dt.max())
        # sag for comparison
        sag = h_max**2 / (2 * R)
        print(f'R={R:+8.4f} h<={hf:4.2f}|R| : max|dt|={dt.max():.3e} m  '
              f'(sag at edge {sag:+.3e} m, rel {dt.max()/abs(sag):.2e})')
print(f'\nWORST absolute dt over all cases: {worst:.3e} m\n')

print('=' * 74)
print('B. CONIC false-miss for h > |R| : real aspheric condenser geometry')
print('=' * 74)
# Thorlabs ACL2520U-ish: R = 10.84 mm, k = -0.6, clear aperture 23 mm
R, k, sd = 10.84e-3, -0.6, 11.5e-3
h_valid = abs(R) / np.sqrt(1 + k)
print(f'R={R*1e3:.2f} mm  k={k}  conic valid to h={h_valid*1e3:.3f} mm ; '
      f'clear semi-dia {sd*1e3:.2f} mm')
hs = np.array([2e-3, 5e-3, 9e-3, 10.0e-3, 10.83e-3, 10.9e-3, 11.4e-3])
rb = mb(hs, np.zeros_like(hs), np.zeros_like(hs),
        np.zeros_like(hs), np.zeros_like(hs), np.ones_like(hs))
surf = Surface(radius=R, conic=k, semi_diameter=sd)
_intersect_surface(rb, surf, n_medium=1.0)
true_sag = hs**2 / (R * (1 + np.sqrt(1 - (1 + k) * hs**2 / R**2)))
print(' h[mm]   alive  err  t_lib[mm]     true_sag[mm]   dz[mm]')
for i, h in enumerate(hs):
    print(f'{h*1e3:7.3f}  {str(bool(rb.alive[i])):5s} {rb.error_code[i]:3d} '
          f'{rb.opd[i]*1e3:12.6f}  {true_sag[i]*1e3:12.6f}  '
          f'{(rb.opd[i]-true_sag[i])*1e3:+.3e}')

print()
print('C. same through the PUBLIC trace() API (aspheric singlet front surface)')
surfs = [Surface(radius=R, conic=k, semi_diameter=sd,
                 glass_before='air', glass_after='N-BK7', thickness=8e-3),
         Surface(radius=np.inf, semi_diameter=sd,
                 glass_before='N-BK7', glass_after='air', thickness=10e-3),
         Surface(radius=np.inf, semi_diameter=np.inf, label='image')]
rays = make_fan('y', 11.4e-3, 41, 0.0, 1.31e-6)
res = trace(rays, surfs, 1.31e-6)
img = res.image_rays
alive = img.alive
print(f'  rays alive at image: {alive.sum()} / {len(alive)}')
ys = rays.y
dead = ~alive
print(f'  dead ray heights [mm]: {np.round(ys[dead]*1e3,3)}')
print(f'  error codes of dead: {np.unique(img.error_code[dead])}  '
      f'(3 = RAY_MISSED_SURFACE)')
print(f'  |y| of first dead ray / |R| = '
      f'{np.abs(ys[dead]).min()/abs(R) if dead.any() else float("nan"):.4f}')

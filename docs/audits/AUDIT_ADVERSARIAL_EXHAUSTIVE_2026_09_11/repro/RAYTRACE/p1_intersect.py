"""RAYTRACE probe 1: ray-conic intersection accuracy + root selection.

Independent oracle: mpmath-free high-precision via Python `fractions`-free
approach -- we use scipy.optimize.brentq on F(t) = z0+N t - sag(x0+L t, y0+M t)
with a bracket found by scanning, plus the analytic stable quadratic root for
the pure sphere (Spencer&Murty / Welford form).
"""
import sys, os
import numpy as np
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')

from lumenairy.raytrace.surface import Surface, RayBundle
from lumenairy.raytrace.intersection import _intersect_surface

rng = np.random.default_rng(12345)


def make_bundle(x, y, z, L, M, N):
    n = len(x)
    return RayBundle(x=x.copy(), y=y.copy(), z=z.copy(),
                     L=L.copy(), M=M.copy(), N=N.copy(),
                     wavelength=1.31e-6,
                     alive=np.ones(n, bool), opd=np.zeros(n))


def sag_conic(h_sq, R, k):
    """exact conic sag (numpy float64), NaN out of domain."""
    if np.isinf(R):
        return np.zeros_like(h_sq)
    norm = (1 + k) * h_sq / R**2
    out = np.where(norm < 1.0, h_sq / (R * (1 + np.sqrt(np.maximum(1 - norm, 0.0)))), np.nan)
    return out


def stable_sphere_root(x0, y0, z0, L, M, N, R):
    """Numerically stable near-vertex root of the ray-sphere quadratic.

    Sphere:  x^2+y^2+(z-R)^2 = R^2  i.e.  x^2+y^2+z^2 - 2 R z = 0
    Sub P = P0 + t D:  t^2 (|D|^2=1) + 2 t (P0.D - R N) + (|P0|^2 - 2 R z0) = 0
    F = |P0|^2 - 2 R z0     (small near the vertex)
    G = -(P0.D - R N) = R N - (P0.D)
    t = F / (G + sqrt(G^2 - F))     <- no cancellation for the near root
    """
    F = x0**2 + y0**2 + z0**2 - 2.0 * R * z0
    G = R * N - (x0 * L + y0 * M + z0 * N)
    disc = G * G - F
    with np.errstate(invalid='ignore'):
        root = np.sqrt(disc)
    t = F / (G + root)
    return t, disc


print('=' * 72)
print('A. pure-sphere fast path vs stable (Spencer-Murty style) root')
print('=' * 72)
for R in (0.1, 0.05, 0.01, 0.005, -0.05, 1.0):
    for hmax_frac in (0.3, 0.7, 0.95):
        h_max = abs(R) * hmax_frac
        n = 20000
        r = h_max * np.sqrt(rng.random(n))
        th = 2 * np.pi * rng.random(n)
        x0 = r * np.cos(th); y0 = r * np.sin(th); z0 = np.zeros(n)
        # random small tilts
        L = 0.05 * (rng.random(n) - .5); M = 0.05 * (rng.random(n) - .5)
        N = np.sqrt(1 - L**2 - M**2)
        surf = Surface(radius=R, conic=0.0, semi_diameter=np.inf)
        rb = make_bundle(x0, y0, z0, L, M, N)
        _intersect_surface(rb, surf, n_medium=1.0)
        t_lib = rb.opd.copy()          # n_medium = 1 so opd == t
        t_ref, disc = stable_sphere_root(x0, y0, z0, L, M, N, R)
        ok = np.isfinite(t_ref) & rb.alive
        if not ok.any():
            continue
        dt = np.abs(t_lib[ok] - t_ref[ok])
        rel = dt / np.maximum(np.abs(t_ref[ok]), 1e-300)
        # residual of the library point on the exact sphere
        res_lib = (rb.x[ok]**2 + rb.y[ok]**2 + rb.z[ok]**2
                   - 2 * R * rb.z[ok])
        print(f'R={R:+.4f} h/|R|<={hmax_frac:4.2f}  max|dt|={dt.max():.3e} m '
              f' max rel={rel.max():.3e}  max |sphere residual|={np.abs(res_lib).max():.3e}')
print()

print('=' * 72)
print('B. CONIC (Newton path) vs brentq oracle; including h > |R| misses')
print('=' * 72)
from scipy.optimize import brentq


def oracle_t(x0, y0, z0, L, M, N, R, k, asph=None):
    def F(t):
        h_sq = (x0 + L * t)**2 + (y0 + M * t)**2
        s = sag_conic(np.array(h_sq), R, k)
        if asph:
            for p, c in asph.items():
                s = s + c * h_sq**(p // 2)
        return (z0 + N * t) - s
    # bracket: scan
    lo, hi = -3 * abs(R) if np.isfinite(R) else -1.0, 3 * abs(R) if np.isfinite(R) else 1.0
    ts = np.linspace(lo, hi, 20001)
    vals = np.array([F(t) for t in ts])
    good = np.isfinite(vals)
    sgn = np.sign(vals)
    roots = []
    for i in range(len(ts) - 1):
        if good[i] and good[i + 1] and sgn[i] * sgn[i + 1] < 0:
            roots.append(brentq(F, ts[i], ts[i + 1], xtol=1e-18, rtol=8.9e-16))
    if not roots:
        return None
    return min(roots, key=abs)


cases = [
    # (R, k, label)
    (0.05, 0.0, 'sphere'),
    (0.05, -1.0, 'parabola'),
    (0.05, -2.0, 'hyperbola k=-2'),
    (0.05, -6.0, 'hyperbola k=-6'),
    (0.05, +2.0, 'oblate ellipsoid k=+2'),
    (-0.05, -1.5, 'concave hyperbola'),
]
for R, k, lab in cases:
    for hfrac in (0.5, 0.9, 1.2, 1.8):
        h = abs(R) * hfrac
        x0 = np.array([h]); y0 = np.array([0.0]); z0 = np.array([0.0])
        L = np.array([0.0]); M = np.array([0.0]); N = np.array([1.0])
        surf = Surface(radius=R, conic=k, semi_diameter=np.inf)
        rb = make_bundle(x0, y0, z0, L, M, N)
        _intersect_surface(rb, surf, n_medium=1.0)
        t_ref = oracle_t(h, 0.0, 0.0, 0.0, 0.0, 1.0, R, k)
        s_true = sag_conic(np.array([h**2]), R, k)[0]
        print(f'{lab:24s} h/|R|={hfrac:4.2f} alive={bool(rb.alive[0])!s:5s} '
              f't_lib={rb.opd[0]: .9e}  t_oracle='
              f'{"None" if t_ref is None else f"{t_ref: .9e}"}  '
              f'true_sag={s_true: .6e}')
    print()

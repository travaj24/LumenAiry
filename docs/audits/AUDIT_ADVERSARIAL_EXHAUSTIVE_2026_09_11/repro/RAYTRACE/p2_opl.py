"""RAYTRACE probe 4: OPL bookkeeping vs an independent geometric oracle.

Oracle: re-trace the same system by hand in the GLOBAL (cumulative-z) frame
with exact vector Snell, and accumulate sum_i n_i * |P_{i+1} - P_i|.
No lumenairy geometry is used in the oracle.
"""
import sys
import numpy as np
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.raytrace import trace, Surface, make_fan
from lumenairy.glass import get_glass_index

WL = 1.31e-6


def sag(h_sq, R, k=0.0, asph=None):
    if np.isinf(R):
        z = np.zeros_like(np.asarray(h_sq, float))
    else:
        z = h_sq / (R * (1 + np.sqrt(1 - (1 + k) * h_sq / R**2)))
    if asph:
        for p, c in asph.items():
            z = z + c * h_sq ** (p // 2)
    return z


def grad(x, y, R, k=0.0, asph=None):
    h_sq = x * x + y * y
    if np.isinf(R):
        dz = 0.0
    else:
        dz = 1.0 / (R * np.sqrt(1 - (1 + k) * h_sq / R**2))
    g = 2.0 * dz  # d(h_sq)/dx = 2x  -> dz/dx = dsag/dh_sq * 2x
    # dsag/dh_sq for conic: derive numerically-free:
    # sag = h2/(R(1+s)), s=sqrt(1-(1+k)h2/R^2)
    # d/dh2 = [R(1+s) - h2*R*ds/dh2] / (R(1+s))^2, ds/dh2 = -(1+k)/(2 R^2 s)
    if np.isinf(R):
        dsdh2 = 0.0
    else:
        s = np.sqrt(1 - (1 + k) * h_sq / R**2)
        dsd = -(1 + k) / (2 * R**2 * s)
        dsdh2 = (R * (1 + s) - h_sq * R * dsd) / (R * (1 + s))**2
    if asph:
        for p, c in asph.items():
            m = p // 2
            dsdh2 = dsdh2 + c * m * h_sq ** (m - 1)
    return 2 * x * dsdh2, 2 * y * dsdh2


def oracle(P0, D0, surfs, ns_before, ns_after, verts):
    """Exact global-frame trace.  Returns (points, total_opl)."""
    P = np.array(P0, float)
    D = np.array(D0, float)
    opl = 0.0
    pts = [P.copy()]
    for i, s in enumerate(surfs):
        zv = verts[i]
        # solve for t:  Pz + Dz t - zv = sag(|P_xy + D_xy t|^2)
        from scipy.optimize import brentq

        def F(t):
            p = P + D * t
            return (p[2] - zv) - sag(p[0]**2 + p[1]**2, s['R'], s.get('k', 0.0),
                                      s.get('asph'))
        # bracket around the nominal
        lo, hi = -0.5, 2.0
        ts = np.linspace(lo, hi, 400001)
        vals = np.array([F(t) for t in ts])
        sgn = np.sign(vals)
        idx = np.where(np.isfinite(vals[:-1]) & np.isfinite(vals[1:])
                        & (sgn[:-1] * sgn[1:] < 0))[0]
        assert len(idx) > 0, f'no root at surface {i}'
        # take the smallest positive root
        roots = [brentq(F, ts[j], ts[j + 1], xtol=1e-17, rtol=8.9e-16)
                 for j in idx]
        roots = [t for t in roots if t > 1e-12]
        t = min(roots)
        Pn = P + D * t
        opl += ns_before[i] * t     # |D| = 1
        pts.append(Pn.copy())
        # refract
        gx, gy = grad(Pn[0], Pn[1], s['R'], s.get('k', 0.0), s.get('asph'))
        nv = np.array([-gx, -gy, 1.0])
        nv /= np.linalg.norm(nv)
        if np.dot(D, nv) > 0:
            nv = -nv
        ci = -np.dot(D, nv)
        mu = ns_before[i] / ns_after[i]
        disc = 1 - mu**2 * (1 - ci**2)
        assert disc >= 0, 'TIR in oracle'
        D = mu * D + (mu * ci - np.sqrt(disc)) * nv
        D /= np.linalg.norm(D)
        P = Pn
    return pts, opl


def run_case(name, surf_specs, thicks, glasses, y0s, M0=0.0):
    n_pre = [get_glass_index(g[0], WL) for g in glasses]
    n_post = [get_glass_index(g[1], WL) for g in glasses]
    surfs = []
    for i, s in enumerate(surf_specs):
        surfs.append(Surface(radius=s['R'], conic=s.get('k', 0.0),
                             aspheric_coeffs=s.get('asph'),
                             semi_diameter=np.inf,
                             glass_before=glasses[i][0],
                             glass_after=glasses[i][1],
                             thickness=thicks[i]))
    verts = np.concatenate([[0.0], np.cumsum(thicks)[:-1]])
    from lumenairy.raytrace.surface import RayBundle
    n = len(y0s)
    N0 = np.sqrt(1 - M0**2)
    rb = RayBundle(x=np.zeros(n), y=np.array(y0s, float), z=np.zeros(n),
                   L=np.zeros(n), M=np.full(n, M0), N=np.full(n, N0),
                   wavelength=WL, alive=np.ones(n, bool), opd=np.zeros(n))
    res = trace(rb, surfs, WL)
    img = res.image_rays
    print(f'--- {name} ---  verts={np.round(verts*1e3,4)} mm')
    worst = 0.0
    for j, y0 in enumerate(y0s):
        pts, opl_ref = oracle([0.0, y0, 0.0], [0.0, M0, N0],
                              surf_specs, n_pre, n_post, verts)
        d = img.opd[j] - opl_ref
        worst = max(worst, abs(d))
        print(f'  y0={y0*1e3:+7.3f} mm  opd_lib={img.opd[j]:.15e}  '
              f'opl_ref={opl_ref:.15e}  d={d:+.3e} m'
              f'  alive={bool(img.alive[j])}')
    print(f'  WORST |d| = {worst:.3e} m\n')
    return worst


# Case 1: biconvex singlet, N-BK7, on-axis + field
w = 0.0
w = max(w, run_case(
    'biconvex singlet on-axis',
    [{'R': 50e-3}, {'R': -50e-3}, {'R': np.inf}],
    [6e-3, 95e-3, 0.0],
    [('air', 'N-BK7'), ('N-BK7', 'air'), ('air', 'air')],
    [0.0, 4e-3, 10e-3, 15e-3]))

w = max(w, run_case(
    'biconvex singlet, 5 deg field',
    [{'R': 50e-3}, {'R': -50e-3}, {'R': np.inf}],
    [6e-3, 95e-3, 0.0],
    [('air', 'N-BK7'), ('N-BK7', 'air'), ('air', 'air')],
    [0.0, 4e-3, 10e-3, -10e-3], M0=np.sin(np.radians(5.0))))

# Case 3: concave-exit meniscus (both R same sign) -> backward t at exit
w = max(w, run_case(
    'meniscus (R1=+25, R2=+40) concave exit',
    [{'R': 25e-3}, {'R': 40e-3}, {'R': np.inf}],
    [5e-3, 80e-3, 0.0],
    [('air', 'N-BK7'), ('N-BK7', 'air'), ('air', 'air')],
    [0.0, 5e-3, 10e-3]))

# Case 4: aspheric front
w = max(w, run_case(
    'asphere front (k=-0.6, A4)',
    [{'R': 25e-3, 'k': -0.6, 'asph': {4: 1.2e3}}, {'R': np.inf},
     {'R': np.inf}],
    [8e-3, 40e-3, 0.0],
    [('air', 'N-BK7'), ('N-BK7', 'air'), ('air', 'air')],
    [0.0, 4e-3, 8e-3]))

print(f'GLOBAL WORST |OPL error| = {w:.3e} m')

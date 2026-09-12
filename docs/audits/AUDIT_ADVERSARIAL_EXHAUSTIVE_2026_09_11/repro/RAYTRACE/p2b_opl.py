"""RAYTRACE probe 4 (v2): OPL bookkeeping vs a 60-digit Decimal oracle.

Oracle uses the IMPLICIT conic form  G(P) = c(x^2+y^2+(1+k)z^2) - 2z = 0
(exactly equivalent to the explicit sag), its exact gradient, and exact
vector Snell.  Optical path = sum_i n_i * t_i with |D| = 1.
Nothing from lumenairy is used in the oracle.
"""
import sys
from decimal import Decimal as D, getcontext
import numpy as np
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
getcontext().prec = 60

from lumenairy.raytrace import trace, Surface
from lumenairy.raytrace.surface import RayBundle
from lumenairy.glass import get_glass_index

WL = 1.31e-6


def DD(v):
    return D(repr(float(v)))


def dsqrt(a):
    return a.sqrt()


def oracle(P0, D0, surfs, n_pre, n_post, verts):
    """surfs: list of dicts with 'R','k' (asph unsupported here).
    verts: cumulative z of each surface vertex (global frame)."""
    P = [DD(v) for v in P0]
    Dv = [DD(v) for v in D0]
    opl = D(0)
    for i, s in enumerate(surfs):
        R = s['R']
        k = DD(s.get('k', 0.0))
        zv = DD(verts[i])
        c = D(0) if np.isinf(R) else D(1) / DD(R)
        # shift to local frame
        z0 = P[2] - zv
        x0, y0 = P[0], P[1]
        dx, dy, dz = Dv
        one_k = D(1) + k
        A = c * (dx * dx + dy * dy + one_k * dz * dz)
        B = D(2) * c * (x0 * dx + y0 * dy + one_k * z0 * dz) - D(2) * dz
        C = c * (x0 * x0 + y0 * y0 + one_k * z0 * z0) - D(2) * z0
        if A == 0:
            t = -C / B
        else:
            disc = B * B - D(4) * A * C
            assert disc >= 0, f'oracle: ray misses surface {i}'
            sq = dsqrt(disc)
            sgn = D(1) if B >= 0 else D(-1)
            q = -(B + sgn * sq) / D(2)
            t1, t2 = q / A, C / q
            # near-vertex root = smaller |t|
            t = t1 if abs(t1) <= abs(t2) else t2
        Pn = [P[0] + dx * t, P[1] + dy * t, P[2] + dz * t]
        opl += DD(n_pre[i]) * t
        # normal from grad G (local coords)
        zl = Pn[2] - zv
        gx = D(2) * c * Pn[0]
        gy = D(2) * c * Pn[1]
        gz = D(2) * c * one_k * zl - D(2)
        gm = dsqrt(gx * gx + gy * gy + gz * gz)
        nx, ny, nz = gx / gm, gy / gm, gz / gm
        dn = dx * nx + dy * ny + dz * nz
        if dn > 0:
            nx, ny, nz = -nx, -ny, -nz
            dn = -dn
        ci = -dn
        mu = DD(n_pre[i]) / DD(n_post[i])
        rad = D(1) - mu * mu * (D(1) - ci * ci)
        assert rad >= 0, 'oracle TIR'
        ct = dsqrt(rad)
        dx = mu * dx + (mu * ci - ct) * nx
        dy = mu * dy + (mu * ci - ct) * ny
        dz = mu * dz + (mu * ci - ct) * nz
        nrm = dsqrt(dx * dx + dy * dy + dz * dz)
        dx, dy, dz = dx / nrm, dy / nrm, dz / nrm
        Dv = [dx, dy, dz]
        P = Pn
    return P, opl


def run(name, specs, thicks, glasses, y0s, M0=0.0):
    n_pre = [get_glass_index(g[0], WL) for g in glasses]
    n_post = [get_glass_index(g[1], WL) for g in glasses]
    surfs = [Surface(radius=s['R'], conic=s.get('k', 0.0),
                     semi_diameter=np.inf,
                     glass_before=glasses[i][0], glass_after=glasses[i][1],
                     thickness=thicks[i]) for i, s in enumerate(specs)]
    verts = np.concatenate([[0.0], np.cumsum(thicks)[:-1]])
    n = len(y0s)
    N0 = float(np.sqrt(1 - M0**2))
    rb = RayBundle(x=np.zeros(n), y=np.array(y0s, float), z=np.zeros(n),
                   L=np.zeros(n), M=np.full(n, M0), N=np.full(n, N0),
                   wavelength=WL, alive=np.ones(n, bool), opd=np.zeros(n))
    res = trace(rb, surfs, WL)
    img = res.image_rays
    print(f'--- {name} ---  vertices (mm): {np.round(verts*1e3, 4)}')
    worst = 0.0
    worstp = 0.0
    for j, y0 in enumerate(y0s):
        Pf, opl = oracle([0.0, y0, 0.0], [0.0, M0, N0], specs,
                         n_pre, n_post, verts)
        dopl = img.opd[j] - float(opl)
        # library leaves rays in the LAST surface's local frame
        dy = img.y[j] - float(Pf[1])
        dz = img.z[j] - (float(Pf[2]) - verts[-1])
        worst = max(worst, abs(dopl))
        worstp = max(worstp, abs(dy), abs(dz))
        print(f'  y0={y0*1e3:+7.3f} mm  opd={img.opd[j]:.13e} '
              f'ref={float(opl):.13e}  dOPL={dopl:+.3e} m   '
              f'dy={dy:+.2e}  dz={dz:+.2e}')
    print(f'  worst |dOPL| = {worst:.3e} m ; worst |dpos| = {worstp:.3e} m\n')
    return worst


W = 0.0
W = max(W, run('biconvex singlet (R=+50/-50, d=6mm) on-axis',
               [{'R': 50e-3}, {'R': -50e-3}, {'R': np.inf}],
               [6e-3, 95e-3, 0.0],
               [('air', 'N-BK7'), ('N-BK7', 'air'), ('air', 'air')],
               [0.0, 4e-3, 10e-3, 15e-3]))
W = max(W, run('biconvex singlet, 5 deg field',
               [{'R': 50e-3}, {'R': -50e-3}, {'R': np.inf}],
               [6e-3, 95e-3, 0.0],
               [('air', 'N-BK7'), ('N-BK7', 'air'), ('air', 'air')],
               [0.0, 4e-3, 10e-3, -10e-3], M0=float(np.sin(np.radians(5.0)))))
W = max(W, run('meniscus R1=+25 R2=+40 (concave exit -> backward t)',
               [{'R': 25e-3}, {'R': 40e-3}, {'R': np.inf}],
               [5e-3, 80e-3, 0.0],
               [('air', 'N-BK7'), ('N-BK7', 'air'), ('air', 'air')],
               [0.0, 5e-3, 10e-3]))
W = max(W, run('deep biconvex, edge-thickness NEGATIVE geometry (R=20, d=1mm)',
               [{'R': 20e-3}, {'R': -20e-3}, {'R': np.inf}],
               [1e-3, 30e-3, 0.0],
               [('air', 'N-BK7'), ('N-BK7', 'air'), ('air', 'air')],
               [0.0, 4e-3, 8e-3]))
W = max(W, run('conic singlet k=-1 front, 3 deg field',
               [{'R': 30e-3, 'k': -1.0}, {'R': -80e-3, 'k': -0.5},
                {'R': np.inf}],
               [7e-3, 60e-3, 0.0],
               [('air', 'N-BK7'), ('N-BK7', 'air'), ('air', 'air')],
               [0.0, 5e-3, 10e-3], M0=float(np.sin(np.radians(3.0)))))
print(f'GLOBAL WORST |OPL error| = {W:.3e} m  ({W/WL:.3e} waves @1.31um)')

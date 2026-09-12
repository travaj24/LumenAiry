"""Independent meridional ray-trace oracle for a spherical singlet in air.

Written from scratch (Snell in vector form on exact spheres) -- no lumenairy
code -- to check _lens_traced_* exit OPL / landing / caustic geometry.
CONVENTIONS.md sec.7: R>0 = centre of curvature downstream.
"""
import numpy as np


def _sphere_hit(x, z, L, Nz, R, zv):
    """Intersect ray (x,z,dir=(L,Nz)) with sphere of radius R whose vertex is at
    z=zv on axis (centre at zv+R).  Returns (t, xh, zh).  R=inf -> plane."""
    if not np.isfinite(R):
        t = (zv - z) / Nz
        return t, x + t * L, z + t * Nz
    cz = zv + R
    # |P + t d - C|^2 = R^2 ; C = (0, cz)
    ox = x - 0.0
    oz = z - cz
    b = 2.0 * (ox * L + oz * Nz)
    c = ox * ox + oz * oz - R * R
    disc = b * b - 4.0 * c            # a = 1 (unit dir)
    disc = np.where(disc < 0, np.nan, disc)
    sq = np.sqrt(disc)
    t1 = (-b - sq) / 2.0
    t2 = (-b + sq) / 2.0
    # take the root nearest the vertex (smallest |z_hit - zv|)
    z1 = z + t1 * Nz
    z2 = z + t2 * Nz
    pick1 = np.abs(z1 - zv) <= np.abs(z2 - zv)
    t = np.where(pick1, t1, t2)
    return t, x + t * L, z + t * Nz


def trace_singlet(h, R1, R2, d, n_glass, z_out, n_out=1.0):
    """Meridional trace of collimated rays at heights h through a singlet whose
    surface-1 vertex is z=0 and surface-2 vertex is z=d, then to plane
    z = d + z_out.  Returns (x_out, OPL) with OPL measured from z=0 plane."""
    h = np.asarray(h, float)
    x = h.copy()
    z = np.zeros_like(h)
    L = np.zeros_like(h)
    Nz = np.ones_like(h)
    opl = np.zeros_like(h)
    for (R, zv, n1, n2) in ((R1, 0.0, 1.0, n_glass), (R2, d, n_glass, 1.0)):
        t, xh, zh = _sphere_hit(x, z, L, Nz, R, zv)
        opl = opl + n1 * t
        x, z = xh, zh
        # outward normal of the sphere at the hit point (pointing along +z-ish)
        if np.isfinite(R):
            cz = zv + R
            nx = (x - 0.0) / R
            nz = (z - cz) / R           # unit, sign chosen so nz>0 for R>0
        else:
            nx = np.zeros_like(x)
            nz = np.ones_like(x)
        # Snell, vector form: t = mu*i + (mu*cosi - cost)*n_hat, with n_hat
        # oriented against the incident direction (cosi > 0)
        ci = -(L * nx + Nz * nz)
        flip = ci < 0
        nx = np.where(flip, -nx, nx)
        nz = np.where(flip, -nz, nz)
        ci = np.abs(ci)
        mu = n1 / n2
        k = 1.0 - mu * mu * (1.0 - ci * ci)
        k = np.where(k < 0, np.nan, k)
        ct = np.sqrt(k)
        L = mu * L + (mu * ci - ct) * nx
        Nz = mu * Nz + (mu * ci - ct) * nz
    # free propagate to z = d + z_out
    zt = d + z_out
    t = (zt - z) / Nz
    opl = opl + n_out * t
    x = x + t * L
    return x, opl, L, Nz

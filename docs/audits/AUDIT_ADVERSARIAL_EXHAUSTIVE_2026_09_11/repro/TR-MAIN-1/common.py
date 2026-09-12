import os, sys, warnings
import numpy as np
REPO = r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy"
if REPO not in sys.path:
    sys.path.insert(0, REPO)

WL = 1.31e-6
NG = 1.5168

def register_glass(name='_AUD_GLASS', n=NG):
    from lumenairy import glass as _g
    _g.GLASS_REGISTRY[name] = (lambda wl, _n=n: _n)
    return name

def singlet_f5(name='_AUD_GLASS'):
    return {
        'wavelength': WL,
        'aperture_diameter': 24e-3,
        'surfaces': [
            {'radius': 51.68e-3, 'thickness': 5e-3,
             'glass_before': 'air', 'glass_after': name,
             'semi_diameter': 12e-3},
            {'radius': -51.68e-3, 'thickness': 0.0,
             'glass_before': name, 'glass_after': 'air',
             'semi_diameter': 12e-3},
        ],
        'thicknesses': [5e-3],
        'stop_index': 0,
    }

def plano_convex(R=100e-3, t=4e-3, ap=20e-3, name='_AUD_GLASS'):
    """Plano-convex: curved first surface (R>0 convex toward input), flat exit."""
    return {
        'wavelength': WL,
        'aperture_diameter': ap,
        'surfaces': [
            {'radius': R, 'thickness': t,
             'glass_before': 'air', 'glass_after': name,
             'semi_diameter': ap/2},
            {'radius': np.inf, 'thickness': 0.0,
             'glass_before': name, 'glass_after': 'air',
             'semi_diameter': ap/2},
        ],
        'thicknesses': [t],
        'stop_index': 0,
    }

def plate(t=4e-3, ap=20e-3, name='_AUD_GLASS'):
    return {
        'wavelength': WL,
        'aperture_diameter': ap,
        'surfaces': [
            {'radius': np.inf, 'thickness': t,
             'glass_before': 'air', 'glass_after': name,
             'semi_diameter': ap/2},
            {'radius': np.inf, 'thickness': 0.0,
             'glass_before': name, 'glass_after': 'air',
             'semi_diameter': ap/2},
        ],
        'thicknesses': [t],
        'stop_index': 0,
    }

def neg_meniscus(R1=-40e-3, R2=-25e-3, t=3e-3, ap=16e-3, name='_AUD_GLASS'):
    """Negative meniscus with CONVEX exit surface (R2<0 -> exit sag > 0)."""
    return {
        'wavelength': WL,
        'aperture_diameter': ap,
        'surfaces': [
            {'radius': R1, 'thickness': t,
             'glass_before': 'air', 'glass_after': name,
             'semi_diameter': ap/2},
            {'radius': R2, 'thickness': 0.0,
             'glass_before': name, 'glass_after': 'air',
             'semi_diameter': ap/2},
        ],
        'thicknesses': [t],
        'stop_index': 0,
    }

# ---------------- INDEPENDENT ORACLE: sequential exact-sphere ray trace -------
def _sphere_intersect(p, d, R, zv):
    """Exact intersection of ray p + t d with sphere of radius R whose VERTEX is
    at z = zv on the axis (centre at z = zv + R).  R = inf -> plane z = zv.
    Sign convention (CONVENTIONS.md S7): R > 0 = centre of curvature downstream.
    Returns t (path length, |d| = 1)."""
    p = np.asarray(p, float); d = np.asarray(d, float)
    if not np.isfinite(R):
        return (zv - p[..., 2]) / d[..., 2]
    c = np.array([0.0, 0.0, zv + R])
    oc = p - c
    b = 2.0 * np.einsum('...i,...i->...', oc, d)
    cc = np.einsum('...i,...i->...', oc, oc) - R * R
    disc = b * b - 4.0 * cc
    disc = np.where(disc < 0, np.nan, disc)
    sq = np.sqrt(disc)
    t1 = (-b - sq) / 2.0
    t2 = (-b + sq) / 2.0
    # choose the root whose z is closest to the vertex (the near cap)
    z1 = p[..., 2] + t1 * d[..., 2]
    z2 = p[..., 2] + t2 * d[..., 2]
    pick1 = np.abs(z1 - zv) <= np.abs(z2 - zv)
    return np.where(pick1, t1, t2)

def _sphere_normal(p, R, zv):
    if not np.isfinite(R):
        n = np.zeros_like(p); n[..., 2] = 1.0
        return n
    c = np.array([0.0, 0.0, zv + R])
    n = (p - c) / R          # unit outward-ish normal pointing +z near vertex
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    # orient along +z
    s = np.sign(n[..., 2])[..., None]
    return n * np.where(s == 0, 1.0, s)

def _refract(d, n, n1, n2):
    """Vector Snell.  d, n unit; n oriented so d.n > 0."""
    mu = n1 / n2
    ci = np.einsum('...i,...i->...', d, n)
    k = 1.0 - mu * mu * (1.0 - ci * ci)
    k = np.where(k < 0, np.nan, k)
    ct = np.sqrt(k)
    return mu * d + (ct - mu * ci)[..., None] * n

def oracle_trace(rx, xs, ys, L=None, M=None, n_of=None):
    """Sequential exact ray trace through the prescription.
    Returns (x_exit, y_exit, OPL) at the EXIT VERTEX PLANE (z of last vertex).
    n_of(glass_name) -> index."""
    surfs = rx['surfaces']
    xs = np.atleast_1d(np.asarray(xs, float))
    ys = np.atleast_1d(np.asarray(ys, float))
    sh = xs.shape
    p = np.stack([xs, ys, np.zeros_like(xs)], axis=-1)
    if L is None:
        d = np.zeros(sh + (3,)); d[..., 2] = 1.0
    else:
        L = np.broadcast_to(np.asarray(L, float), sh)
        M = np.broadcast_to(np.asarray(M, float), sh)
        d = np.stack([L, M, np.sqrt(np.maximum(0.0, 1 - L**2 - M**2))], axis=-1)
    opl = np.zeros(sh)
    zv = 0.0
    for i, s in enumerate(surfs):
        R = float(s['radius'])
        n1 = n_of(s.get('glass_before', 'air'))
        n2 = n_of(s.get('glass_after', 'air'))
        t = _sphere_intersect(p, d, R, zv)
        p = p + t[..., None] * d
        opl = opl + n1 * t
        nvec = _sphere_normal(p, R, zv)
        d = _refract(d, nvec, n1, n2)
        zv = zv + float(s['thickness'])
    # transfer to exit vertex plane z = zv_last (= sum of thicknesses)
    z_exit = sum(float(s['thickness']) for s in surfs)
    n_last = n_of(surfs[-1].get('glass_after', 'air'))
    tt = (z_exit - p[..., 2]) / d[..., 2]
    p = p + tt[..., None] * d
    opl = opl + n_last * tt
    return p[..., 0], p[..., 1], opl, d

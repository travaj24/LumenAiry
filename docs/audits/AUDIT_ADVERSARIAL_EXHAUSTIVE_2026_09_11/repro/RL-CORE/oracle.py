"""Independent sequential ray-trace oracle for RL-CORE audit.

Self-contained: exact conic intersection by Newton on the sag equation,
vector Snell, OPL accumulation.  Does NOT use lumenairy.raytrace.

Surface convention (matches lumenairy docs): sag(h) = h^2/R / (1 + sqrt(1-(1+k)h^2/R^2))
so R > 0 => sag > 0 off-axis (surface bulges toward +z downstream... check).
Surface z position = vertex_z + sag(h).
"""
import numpy as np


def sag_of(h_sq, R, k=0.0, asph=None):
    if not np.isfinite(R) or R == 0:
        s = np.zeros_like(np.asarray(h_sq, dtype=float))
    else:
        c = 1.0 / R
        arg = 1.0 - (1.0 + k) * c * c * h_sq
        arg = np.where(arg < 0, np.nan, arg)
        s = c * h_sq / (1.0 + np.sqrt(arg))
    if asph:
        h = np.sqrt(h_sq)
        for p, a in asph.items():
            s = s + a * h ** p
    return s


def dsag_dh(h, R, k=0.0, asph=None, eps=1e-9):
    """d sag / d h, analytic for the conic + numeric-free for asphere."""
    h = np.asarray(h, dtype=float)
    if not np.isfinite(R) or R == 0:
        d = np.zeros_like(h)
    else:
        c = 1.0 / R
        rt = np.sqrt(1.0 - (1.0 + k) * c * c * h * h)
        d = c * h / rt
    if asph:
        for p, a in asph.items():
            d = d + a * p * h ** (p - 1)
    return d


def trace_meridional(prescription, wavelength, h0, n_of=None, z_start=0.0):
    """Trace a set of rays parallel to +z entering at height h0 (x only, y=0).

    Returns dict with per-ray: exit height x, exit direction (Lx, Lz),
    and OPL accumulated from the plane z=z_start to the exit vertex plane
    (z of the LAST surface vertex).
    """
    surfaces = prescription['surfaces']
    thick = prescription['thicknesses']
    if n_of is None:
        from lumenairy.glass import get_glass_index
        n_of = lambda g: float(get_glass_index(g, wavelength))

    h0 = np.atleast_1d(np.asarray(h0, dtype=float))
    x = h0.copy()
    z = np.full_like(x, float(z_start))
    Lx = np.zeros_like(x)          # direction cosine x
    Lz = np.ones_like(x)           # direction cosine z
    opl = np.zeros_like(x)

    # vertex z of surface i
    vz = [0.0]
    for t in thick:
        vz.append(vz[-1] + float(t))

    n_before = n_of(surfaces[0]['glass_before'])

    for i, s in enumerate(surfaces):
        R = s['radius']
        k = s.get('conic', 0.0)
        asph = s.get('aspheric_coeffs')
        zv = vz[i]
        n1 = n_of(s['glass_before'])
        n2 = n_of(s['glass_after'])
        # Newton solve for t s.t. z + t*Lz == zv + sag((x + t*Lx)^2)
        t = (zv - z) / Lz
        for _ in range(80):
            xx = x + t * Lx
            zz = z + t * Lz
            f = zz - (zv + sag_of(xx * xx, R, k, asph))
            dfd = Lz - dsag_dh(xx, R, k, asph) * Lx
            step = f / dfd
            t = t - step
            if np.max(np.abs(step)) < 1e-16:
                break
        xh = x + t * Lx
        zh = z + t * Lz
        opl = opl + n1 * t
        # surface normal: surface F(x,z) = z - zv - sag(x^2) = 0
        # grad F = (-dsag/dx, 1) -> normal (unnormalised)
        nx = -dsag_dh(xh, R, k, asph)
        nz = np.ones_like(nx)
        nn = np.sqrt(nx * nx + nz * nz)
        nx, nz = nx / nn, nz / nn
        # vector Snell: n1 * d - ((n1 * d.N) - n2*cos_t) N  ... use standard form
        mu = n1 / n2
        cosi = -(Lx * nx + Lz * nz)     # N chosen pointing toward -incoming
        # make normal point against the incoming ray
        sgn = np.where(cosi < 0, -1.0, 1.0)
        nx_, nz_ = nx * sgn, nz * sgn
        cosi = -(Lx * nx_ + Lz * nz_)
        sin2t = mu * mu * (1.0 - cosi * cosi)
        cost = np.sqrt(np.maximum(1.0 - sin2t, 0.0))
        Lx = mu * Lx + (mu * cosi - cost) * nx_
        Lz = mu * Lz + (mu * cosi - cost) * nz_
        x, z = xh, zh
        n_before = n2

    # bring all rays back to the LAST vertex plane z = vz[-1] along their
    # direction (negative distance for rays that overshot)
    z_exit = vz[-1]
    t_end = (z_exit - z) / Lz
    opl = opl + n_before * t_end
    x_exit = x + t_end * Lx
    return dict(x=x_exit, Lx=Lx, Lz=Lz, opl=opl, n_exit=n_before,
                z_exit=z_exit)

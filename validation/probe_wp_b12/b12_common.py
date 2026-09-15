"""WP-B12 -- shared fixtures and the independent diffraction oracle.

The oracle imports NOTHING from ``lumenairy`` for its physics: its glass
dispersion is a Sellmeier evaluation from typed-in Schott coefficients, its
intersection is a Newton solve on the implicit sag (not a closed-form quadric),
its refraction is vector Snell with the normal oriented AGAINST the incident
ray, and it propagates by a brute-force Rayleigh-Sommerfeld-I sum over
(exit ray) x (azimuth).  ``lumenairy`` is imported only to BUILD the
prescriptions the fixtures name and to score the library members against the
oracle.

Every fixture is rotationally symmetric with a collimated Gaussian input, so
the oracle's radial field determines the whole 2-D field.

Author: WP-B12
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Sellmeier coefficients (Schott catalogue), typed in here.  The oracle never
# calls lumenairy.glass; ``control_glass`` below measures the two against each
# other and the probes record the difference.
# ---------------------------------------------------------------------------
SELLMEIER = {
    'N-SF11': ((1.73759695, 0.313747346, 1.89878101),
               (0.013188707, 0.0623068142, 155.23629)),
    'N-BK7': ((1.03961212, 0.231792344, 1.01046945),
              (0.00600069867, 0.0200179144, 103.560653)),
    'N-LASF9': ((2.00029547, 0.298926886, 1.80691843),
                (0.0121426017, 0.0538736236, 156.530829)),
    'N-BAF10': ((1.5851495, 0.143559385, 1.08521269),
                (0.00926681282, 0.0424489805, 105.613573)),
    'N-LAK22': ((1.14229781, 0.535138441, 1.04088385),
                (0.00585778594, 0.0198546147, 100.834017)),
    'N-SF6': ((1.77931763, 0.338149866, 2.08734474),
              (0.0133714182, 0.0617533621, 174.01759)),
}


def n_sellmeier(glass: str, wavelength_m: float) -> float:
    """Refractive index from the typed-in Sellmeier coefficients."""
    b, c = SELLMEIER[glass]
    l2 = (wavelength_m * 1e6) ** 2
    s = 1.0
    for bi, ci in zip(b, c):
        s += bi * l2 / (l2 - ci)
    return math.sqrt(s)


# ---------------------------------------------------------------------------
# Exact conic / even-aspheric meridional trace.
# ---------------------------------------------------------------------------
def _sag(h, R, conic, asph):
    """Surface sag at radial height ``h`` (scalar or array)."""
    h2 = np.asarray(h, float) ** 2
    if np.isinf(R):
        z = np.zeros_like(h2)
    else:
        c = 1.0 / R
        z = c * h2 / (1.0 + np.sqrt(np.maximum(
            1.0 - (1.0 + conic) * c * c * h2, 0.0)))
    for p, a in (asph or ()):
        z = z + a * h2 ** (p // 2)
    return z


def _dsag(h, R, conic, asph):
    """d(sag)/dh."""
    h = np.asarray(h, float)
    h2 = h * h
    if np.isinf(R):
        d = np.zeros_like(h)
    else:
        c = 1.0 / R
        rt = np.sqrt(np.maximum(1.0 - (1.0 + conic) * c * c * h2, 0.0))
        d = c * h / np.maximum(rt, 1e-300)
    for p, a in (asph or ()):
        m = p // 2
        d = d + a * m * h2 ** (m - 1) * 2.0 * h
    return d


def trace_meridional(h0, surfaces, wavelength, *, n_air=1.0):
    """Exact meridional trace of collimated rays, launched at height ``h0``
    on the z = 0 plane, through ``surfaces``.

    ``surfaces`` is a list of dicts with keys ``radius`` / ``conic`` /
    ``asph`` / ``zv`` (vertex z) / ``n_after`` (index after the surface).
    Intersection by Newton on ``F(t) = (z + t uz) - zv - sag(|x + t ux|)``,
    refraction by vector Snell with the normal oriented against the incident
    ray.

    Returns ``(x_surf, u_surf, opl_surf, x_vert, u_vert, opl_vert)`` -- the
    state ON the last surface and on its VERTEX plane, with ``u = ux/uz``.
    """
    x = np.asarray(h0, float).copy()
    z = np.zeros_like(x)
    ux = np.zeros_like(x)
    uz = np.ones_like(x)
    opl = np.zeros_like(x)
    n_cur = n_air
    for s in surfaces:
        zv, R, k, asph = s['zv'], s['radius'], s['conic'], s.get('asph')
        # Newton on the implicit sag; seed with the flat-plane crossing.
        t = (zv - z) / uz
        for _ in range(60):
            xx = x + t * ux
            zz = z + t * uz
            F = (zz - zv) - _sag(np.abs(xx), R, k, asph)
            dF = uz - _dsag(np.abs(xx), R, k, asph) * np.sign(xx) * ux
            step = F / np.where(np.abs(dF) < 1e-300, 1e-300, dF)
            t = t - step
            if np.max(np.abs(step)) < 1e-16:
                break
        xx = x + t * ux
        zz = z + t * uz
        resid = np.max(np.abs((zz - zv) - _sag(np.abs(xx), R, k, asph)))
        assert resid < 1e-14, f'intersection Newton did not converge: {resid}'
        opl = opl + n_cur * t
        x, z = xx, zz
        # surface normal: grad(z - zv - sag(|x|)) = (-dsag*sign(x), 1)
        gx = -_dsag(np.abs(x), R, k, asph) * np.sign(x)
        gz = np.ones_like(gx)
        g = np.sqrt(gx * gx + gz * gz)
        nx, nz = gx / g, gz / g
        # orient AGAINST the incident ray: cos(theta_i) = -n.u > 0
        ci = -(nx * ux + nz * uz) / np.sqrt(ux * ux + uz * uz)
        flip = ci < 0.0
        nx = np.where(flip, -nx, nx)
        nz = np.where(flip, -nz, nz)
        ci = np.abs(ci)
        n_next = s['n_after']
        mu = n_cur / n_next
        disc = 1.0 - mu * mu * (1.0 - ci * ci)
        assert np.all(disc > 0.0), 'total internal reflection in the oracle'
        # normalise the incident direction before Snell
        nn = np.sqrt(ux * ux + uz * uz)
        uxn, uzn = ux / nn, uz / nn
        f = mu * ci - np.sqrt(disc)
        ux = mu * uxn + f * nx
        uz = mu * uzn + f * nz
        n_cur = n_next
    x_s, u_s, opl_s = x.copy(), (ux / uz), opl.copy()
    # Project to the LAST surface's vertex plane, in the exit medium.  After
    # every Snell step (mu*u_hat + f*n_hat with two unit vectors) the direction
    # stays a UNIT vector, so the parametric step IS the geometric path.
    zv_last = surfaces[-1]['zv']
    t = (zv_last - z) / uz
    x_v = x + t * ux
    opl_v = opl + n_cur * t
    return x_s, u_s, opl_s, x_v, (ux / uz), opl_v, n_cur


def rs_radial(x_v, u_v, opl_v, weight, rho, z, wavelength, n_phi=1024):
    """Brute-force Rayleigh-Sommerfeld-I sum of the exit-vertex-plane ray
    field onto a radial readout ``rho`` at axial distance ``z``.

    ``weight`` already carries the amplitude, the Jacobian |dx_e/dh| and the
    quadrature step (see ``ring_weights``).
    """
    k = 2.0 * np.pi / wavelength
    phi = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    cph = np.cos(phi)
    pre = np.exp(1j * k * opl_v) * weight
    out = np.empty(np.asarray(rho).size, complex)
    rho = np.asarray(rho, float)
    for i0 in range(0, rho.size, 16):
        rr = rho[i0:i0 + 16][:, None, None]
        r2 = (z * z + rr * rr + x_v[None, :, None] ** 2
              - 2.0 * rr * x_v[None, :, None] * cph[None, None, :])
        r = np.sqrt(r2)
        out[i0:i0 + 16] = ((np.exp(1j * k * r) * (z / r2))
                           * pre[None, :, None]).sum(axis=(1, 2))
    return out * (2.0 * np.pi / n_phi) / (1j * wavelength)


def ring_weights(h, x_v, w0):
    """Quadrature weight for the RS sum under the change of variable to the
    launch height ``h``: ``U x dx = E_in(h) sqrt(h x |dx/dh|) dh``."""
    dh = h[1] - h[0]
    dxe = np.gradient(x_v, h, edge_order=2)
    assert np.all(dxe > 0), 'premise: the exit-vertex plane is not a caustic'
    return np.exp(-(h / w0) ** 2) * np.sqrt(h * x_v * dxe) * dh


def fidelity(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(abs(np.vdot(a, b)) ** 2
                 / (np.vdot(a, a).real * np.vdot(b, b).real))


def rel_l2(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    s = np.vdot(b, a) / np.vdot(b, b)          # best complex scale
    return float(np.linalg.norm(a - s * b) / np.linalg.norm(a))


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------
class Fixture:
    def __init__(self, key, glass, R1, R2, t, semi, lam, w0, N, dx,
                 note='', conic1=0.0, conic2=0.0, model_index=None):
        self.key = key
        self.glass = glass
        self.R1, self.R2, self.t = R1, R2, t
        self.semi, self.lam, self.w0 = semi, lam, w0
        self.N, self.dx = N, dx
        self.note = note
        self.conic1, self.conic2 = conic1, conic2
        # ``model_index`` makes the glass a DISPERSIONLESS model index of that
        # value, registered probe-locally under ``glass``; the oracle uses the
        # same constant, so the two still share no code.
        self.model_index = model_index

    def index(self):
        if self.model_index is not None:
            return float(self.model_index)
        return n_sellmeier(self.glass, self.lam)

    def register(self):
        if self.model_index is None:
            return
        from lumenairy import glass as _g
        _g.GLASS_REGISTRY[self.glass] = (
            lambda wl, _n=float(self.model_index): _n)

    # -- lumenairy side ---------------------------------------------------
    def prescription(self):
        import lumenairy as la
        self.register()
        p = la.make_singlet(self.R1, self.R2, self.t, self.glass,
                            aperture=2.0 * self.semi)
        if self.conic1 or self.conic2:
            s = [dict(p['surfaces'][0]), dict(p['surfaces'][1])]
            s[0]['conic'] = float(self.conic1)
            s[1]['conic'] = float(self.conic2)
            p = {**p, 'surfaces': s}
        return p

    def grid(self):
        x1 = (np.arange(self.N) - self.N / 2) * self.dx
        return np.meshgrid(x1, x1)

    def beam(self):
        X, Y = self.grid()
        return np.exp(-(X ** 2 + Y ** 2) / self.w0 ** 2).astype(np.complex128)

    # -- oracle side ------------------------------------------------------
    def oracle_surfaces(self):
        ng = self.index()
        return [dict(zv=0.0, radius=self.R1, conic=self.conic1, asph=None,
                     n_after=ng),
                dict(zv=self.t, radius=self.R2, conic=self.conic2, asph=None,
                     n_after=1.0)]

    def oracle_exit(self, n_h=1201):
        """Exit-vertex-plane ray field of the oracle."""
        h = np.linspace(self.semi / (2 * n_h),
                        self.semi * (1.0 - 1.0 / (2 * n_h)), n_h)
        xs, us, ols, xv, uv, olv, n_out = trace_meridional(
            h, self.oracle_surfaces(), self.lam)
        return h, xs, us, ols, xv, uv, olv, n_out

    def best_focus(self, n_h=4001):
        """Intensity-weighted geometric best focus past the exit vertex, from
        the oracle's own trace (derived, never pinned)."""
        h, _xs, _us, _ols, xv, uv, _olv, _n = self.oracle_exit(n_h=n_h)
        f0 = float(-xv[0] / uv[0])
        zs = np.linspace(0.5 * f0, 1.2 * f0, 4001)
        wgt = np.exp(-2.0 * (h / self.w0) ** 2) * h
        xz = xv[None, :] + uv[None, :] * zs[:, None]
        ctr = (wgt * xz).sum(1) / wgt.sum()
        var = (wgt * (xz - ctr[:, None]) ** 2).sum(1) / wgt.sum()
        return float(zs[int(np.argmin(var))])

    def oracle_field(self, z, n_h=1201, n_phi=1024, n_rho=None):
        """Oracle field on the fixture's own 2-D grid at distance ``z`` past
        the exit vertex.

        ``z == 0`` is the exit-vertex PLANE itself, where the
        Rayleigh-Sommerfeld kernel ``z / r**2`` degenerates: there the oracle
        IS its own boundary field -- the geometrical-optics exit field
        ``E_in(h) sqrt(h / (x_v |dx_v/dh|)) exp(i k opl_v)`` that the RS sum
        integrates -- so that branch is returned directly."""
        h, _xs, _us, _ols, xv, _uv, olv, _n = self.oracle_exit(n_h=n_h)
        X, Y = self.grid()
        rr = np.sqrt(X ** 2 + Y ** 2)
        k = 2.0 * np.pi / self.lam
        if z == 0.0:
            dxe = np.gradient(xv, h, edge_order=2)
            amp = np.exp(-(h / self.w0) ** 2) * np.sqrt(
                np.abs(h / (xv * dxe)))
            e = amp * np.exp(1j * k * olv)
            rho, src = xv, e
            out = np.where(
                (rr.ravel() >= xv[0]) & (rr.ravel() <= xv[-1]),
                np.interp(rr.ravel(), rho, src.real)
                + 1j * np.interp(rr.ravel(), rho, src.imag), 0.0)
            return out.reshape(rr.shape)
        w = ring_weights(h, xv, self.w0)
        rho = self.readout_radii(float(rr.max()), fine=n_rho)
        e = rs_radial(xv, _uv, olv, w, rho, z, self.lam, n_phi=n_phi)
        return (np.interp(rr.ravel(), rho, e.real)
                + 1j * np.interp(rr.ravel(), rho, e.imag)).reshape(rr.shape)

    def airy_radius(self):
        """0.61 lambda / NA from the oracle's own marginal exit ray."""
        h, _xs, _us, _ols, xv, uv, _olv, _n = self.oracle_exit(n_h=201)
        na = abs(uv[-1]) / np.sqrt(1.0 + uv[-1] ** 2)
        return float(0.61 * self.lam / na)

    def readout_radii(self, r_max, fine=None):
        """Radial readout grid for the RS sum: 1/20 of an Airy radius out to
        12 Airy radii (where the whole focal structure lives), then 1 grid
        pitch to the corner.  Interpolation error is therefore set by the FINE
        region, not by the number of points."""
        ra = self.airy_radius()
        dr = ra / (fine or 20)
        r1 = min(12.0 * ra, r_max)
        rho = np.concatenate([
            np.arange(0.0, r1, dr),
            np.arange(r1, r_max * 1.001 + self.dx, self.dx)])
        rho[0] = 1e-12
        return rho


FIXTURES = {
    # WP-B7b's own fixture (curved last surface), at its report grid.
    'b7b_biconvex': Fixture(
        'b7b_biconvex', 'N-SF11', 1.6e-3, -1.6e-3, 0.60e-3, 0.15e-3,
        0.633e-6, 80e-6, 256, 1.4e-6,
        note="WP-B7b's N-SF11 R=+/-1.6 mm biconvex, 633 nm"),
    # WP-B7b's fixture at its TEST grid (the grid the pinned tests use).
    'b7b_biconvex_testgrid': Fixture(
        'b7b_biconvex_testgrid', 'N-SF11', 1.6e-3, -1.6e-3, 0.60e-3, 0.15e-3,
        0.633e-6, 80e-6, 192, 1.8e-6,
        note="WP-B7b's fixture at the pinned tests' grid"),
    # Mine #1: a different glass, wavelength and shape -- biconvex.
    'm_biconvex_baf10': Fixture(
        'm_biconvex_baf10', 'N-BAF10', 2.10e-3, -2.10e-3, 0.70e-3, 0.20e-3,
        1.064e-6, 105e-6, 256, 1.9e-6,
        note='mine: N-BAF10 biconvex, 1.064 um'),
    # Mine #2: a BENT singlet (both centres on the same side).
    'm_bent_lak22': Fixture(
        'm_bent_lak22', 'N-LAK22', 2.05e-3, -1.40e-3, 0.55e-3, 0.17e-3,
        1.55e-6, 90e-6, 256, 2.6e-6,
        note='mine: N-LAK22 bent singlet, 1.55 um'),
    # Mine #3: a converging MENISCUS (last surface curved the other way).
    'm_meniscus_sf6': Fixture(
        'm_meniscus_sf6', 'N-SF6', 1.30e-3, 9.0e-3, 0.60e-3, 0.16e-3,
        0.850e-6, 85e-6, 256, 1.8e-6,
        note='mine: N-SF6 converging meniscus, 850 nm'),
    # Mine #4: the FLAT-last-surface control -- the defect must read zero.
    'm_flat_last_lasf9': Fixture(
        'm_flat_last_lasf9', 'N-LASF9', 1.45e-3, float('inf'), 0.55e-3,
        0.262e-3, 0.850e-6, 190e-6, 256, 2.0e-6,
        note='mine: N-LASF9 plano-convex, curved side FIRST (flat last '
             'surface) -- the control'),
}


# VERIFY-B7b's fixture V, at ITS grid and its two beams (the sag-screen
# estimate is swept across the aberration envelope by the beam radius alone:
# 1.54 rad inside, 2.03 rad outside).
FIXTURES['v_planoconvex_w190'] = Fixture(
    'v_planoconvex_w190', 'N-LASF9', 1.45e-3, float('inf'), 0.55e-3, 0.262e-3,
    0.850e-6, 190e-6, 384, 1.6e-6,
    note="VERIFY-B7b's fixture V (flat last surface), w0 = 190 um -- inside "
         'the aberration envelope')
FIXTURES['v_planoconvex_w205'] = Fixture(
    'v_planoconvex_w205', 'N-LASF9', 1.45e-3, float('inf'), 0.55e-3, 0.262e-3,
    0.850e-6, 205e-6, 384, 1.6e-6,
    note="VERIFY-B7b's fixture V, w0 = 205 um -- OVER the aberration budget")
# The H2 f/5 dual-oracle biconvex (docs/audit_real_lens_hammer_2026_07_19.md):
# R = +/-51.68 mm, t = 5 mm, n = 1.5168 dispersionless, lambda = 1.31 um,
# w0 = 5 mm, image at 49.163 mm.  Its routing quantities are tractable; its
# CAUSTIC is not resolvable on any grid this probe can run (see probe C).
FIXTURES['h2_f5'] = Fixture(
    'h2_f5', '_B12_N1p5168', 51.68e-3, -51.68e-3, 5.0e-3, 5.0e-3,
    1.31e-6, 5.0e-3, 256, 2 * 12.8e-3 / 256,
    note='the H2 f/5 dual-oracle biconvex (model glass n = 1.5168)',
    model_index=1.5168)


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(obj, fh, indent=1, default=float)
    print(f'[wrote] {path}')


def build_tag():
    import numpy
    import scipy
    return {
        'python': sys.version.split()[0],
        'numpy': numpy.__version__,
        'scipy': scipy.__version__,
        'platform': sys.platform,
    }


def assert_tree(root):
    """Pin the tree the probe is measuring."""
    import lumenairy
    f = os.path.abspath(lumenairy.__file__)
    assert os.path.abspath(root) in f, (f, root)
    print('lumenairy.__file__ =', f, '| version', lumenairy.__version__)
    return f

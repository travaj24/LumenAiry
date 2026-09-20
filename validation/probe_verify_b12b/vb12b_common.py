"""VERIFY-WP-B12b -- my own fixtures, my own fully 3-D tracer, my own
band-limited angular-spectrum diffraction oracle.

Independent of the library path under test, of WP-B12b's own probe package
(``validation/probe_gbd_projection``, whose oracle is the rotationally
symmetric Rayleigh-Sommerfeld ring sum imported from ``probe_wp_b12``), and of
VERIFY-WP-B12's probe package -- every fixture here is new (new glasses, new
radii, new wavelength, new grid), and the surface classes WP-B12b could only
report as "movement" get a diffraction reference:

* the conic sag is evaluated from the **quadric root**
  ``z = (1 - sqrt(1 - (1+k) c^2 u^2)) / ((1+k) c)`` rather than the
  rationalised ``c u^2 / (1 + sqrt(...))`` the library uses -- algebraically
  the same surface, a different expression;
* the intersection is a **damped Newton on the implicit 3-D surface
  equation** with the analytic transverse gradient, so a BICONIC, an XY
  POLYNOMIAL FREEFORM and a FIELD-FRAME decentred last surface are traced
  exactly, not approximated;
* refraction is vector Snell in the ``(n1/n2)`` form with the normal oriented
  against the incident ray; reflection is ``d - 2 (d.n) n``;
* propagation is a **band-limited angular spectrum** (Matsushima & Shimobaba
  2009) of the geometrical-optics exit-vertex boundary field, resampled by
  radius where the optic is rotationally symmetric and by a C1 Clough-Tocher
  interpolation of the two smooth functions ``opl`` and ``amp`` otherwise.

``lumenairy`` is imported only to build the prescriptions the fixtures name
and to score the library; never for the oracle's physics.

Run with ``OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`` and
``LUMENAIRY_MEM_BUDGET_MB`` pinned, with ``PYTHONPATH`` set to the tree under
test (``assert_tree`` enforces it).

Author: VERIFY-WP-B12b
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Glass.  Dispersionless MODEL indices registered probe-locally, so the oracle
# and the library read the SAME index by construction and glass modelling is
# out of scope for this package.  None of these names is used by WP-B12b's
# probes (N-BAF10 / N-LASF9) or by VERIFY-WP-B12's (VB12-M*).
# ---------------------------------------------------------------------------
MODEL_INDICES = {
    'VB12B-M158': 1.58,
    'VB12B-M172': 1.72,
}


def register_model_glasses():
    from lumenairy import glass as _g
    for nm, nv in MODEL_INDICES.items():
        _g.GLASS_REGISTRY[nm] = (lambda wl, _n=float(nv): _n)


#: Schott Sellmeier coefficients typed in HERE for the one CATALOGUE glass a
#: cross-package fixture needs (WP-B12b's own optic is N-BAF10).  Control:
#: ``probe_w4_frame`` compares these against ``lumenairy.get_glass_index``.
SELLMEIER = {
    'N-BAF10': ((1.5851495, 0.143559385, 1.08521269),
                (0.00926681282, 0.0424489805, 105.613573)),
}


def n_sellmeier(name, wavelength_m):
    b, c = SELLMEIER[name]
    l2 = (wavelength_m * 1e6) ** 2
    s = 1.0
    for bi, ci in zip(b, c):
        s += bi * l2 / (l2 - ci)
    return math.sqrt(s)


def glass_index(name, lam=None):
    if name is None or str(name).lower() in ('air', 'vacuum', ''):
        return 1.0
    if name in MODEL_INDICES:
        return float(MODEL_INDICES[name])
    return float(n_sellmeier(name, lam))


# ===========================================================================
# My surface model.  A spec is a plain dict:
#   zv       vertex z [m] in this tracer's running global frame
#   Rx, kx   x-axis (or rotationally symmetric) radius / conic
#   ax       even-aspheric {power: coeff}, sum_p a_p r^p  (rot-sym branch)
#   Ry, ky   y-axis radius / conic  (None -> rotationally symmetric)
#   ay       even-aspheric {power: coeff} on the y branch
#   xy       XY-polynomial freeform {(i, j): c_ij} on a rot-sym conic base
#   dec      (dx, dy) field-frame decenter: the sag is evaluated at (x-dx,y-dy)
#   tilt     (tx, ty) field-frame small-angle tilt ramp
#   n_after  index after the surface (ignored for a mirror)
#   mirror   bool
# ===========================================================================
def _conic_axis(u2, R, k):
    """Conic sag of one axis from the QUADRIC ROOT.

    ``(1+k) c z^2 - 2 z + c u2 = 0`` has the physical root
    ``z = (1 - sqrt(1 - (1+k) c^2 u2)) / ((1+k) c)``; at ``k = -1`` the
    quadratic degenerates to ``z = c u2 / 2``.  Algebraically the same surface
    as the library's ``c u2 / (1 + sqrt(...))``, a different expression -- so
    the two agree only if both are right.
    """
    u2 = np.asarray(u2, float)
    if R is None or not np.isfinite(R):
        return np.zeros_like(u2)
    c = 1.0 / R
    kp = 1.0 + k
    if abs(kp) < 1e-12:
        return 0.5 * c * u2
    rad = 1.0 - kp * c * c * u2
    rad = np.where(rad < 0.0, np.nan, rad)
    return (1.0 - np.sqrt(rad)) / (kp * c)


def _conic_axis_d(u, R, k):
    """d/du of ``_conic_axis(u**2, R, k)``."""
    u = np.asarray(u, float)
    if R is None or not np.isfinite(R):
        return np.zeros_like(u)
    c = 1.0 / R
    kp = 1.0 + k
    if abs(kp) < 1e-12:
        return c * u
    rad = 1.0 - kp * c * c * u * u
    rad = np.where(rad < 0.0, np.nan, rad)
    return c * u / np.sqrt(rad)


def _base_sag(x, y, s):
    """Sag WITHOUT the field-frame decenter / tilt (the shifted coordinates
    are applied by :func:`sag`)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xy = s.get('xy')
    if xy:
        # Freeform: a rotationally-symmetric conic BASE plus the polynomial
        # departure -- the library's own "instead of, not on top of the
        # biconic" convention.
        z = _conic_axis(x * x + y * y, s['Rx'], s.get('kx', 0.0))
        for (i, j), c in xy.items():
            z = z + c * (x ** i) * (y ** j)
        return z
    if s.get('Ry') is None:
        r2 = x * x + y * y
        z = _conic_axis(r2, s['Rx'], s.get('kx', 0.0))
        for p, a in (s.get('ax') or {}).items():
            z = z + a * r2 ** (p // 2)
        return z
    z = (_conic_axis(x * x, s['Rx'], s.get('kx', 0.0))
         + _conic_axis(y * y, s['Ry'], s.get('ky', 0.0)))
    for p, a in (s.get('ax') or {}).items():
        z = z + a * (x * x) ** (p // 2)
    for p, a in (s.get('ay') or {}).items():
        z = z + a * (y * y) ** (p // 2)
    return z


def _base_sag_grad(x, y, s):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xy = s.get('xy')
    if xy:
        h = np.sqrt(x * x + y * y)
        hs = np.where(h > 0, h, 1.0)
        d = _conic_axis_d(h, s['Rx'], s.get('kx', 0.0))
        gx = np.where(h > 0, d * x / hs, 0.0)
        gy = np.where(h > 0, d * y / hs, 0.0)
        for (i, j), c in xy.items():
            if i > 0:
                gx = gx + c * i * (x ** (i - 1)) * (y ** j)
            if j > 0:
                gy = gy + c * j * (x ** i) * (y ** (j - 1))
        return gx, gy
    if s.get('Ry') is None:
        h = np.sqrt(x * x + y * y)
        hs = np.where(h > 0, h, 1.0)
        d = _conic_axis_d(h, s['Rx'], s.get('kx', 0.0))
        for p, a in (s.get('ax') or {}).items():
            d = d + a * p * h ** (p - 1)
        gx = np.where(h > 0, d * x / hs, 0.0)
        gy = np.where(h > 0, d * y / hs, 0.0)
        return gx, gy
    gx = _conic_axis_d(x, s['Rx'], s.get('kx', 0.0))
    gy = _conic_axis_d(y, s['Ry'], s.get('ky', 0.0))
    for p, a in (s.get('ax') or {}).items():
        gx = gx + a * p * x ** (p - 1)
    for p, a in (s.get('ay') or {}).items():
        gy = gy + a * p * y ** (p - 1)
    return gx, gy


def sag(x, y, s):
    dcx, dcy = s.get('dec') or (0.0, 0.0)
    tx, ty = s.get('tilt') or (0.0, 0.0)
    xs = np.asarray(x, float) - dcx
    ys = np.asarray(y, float) - dcy
    z = _base_sag(xs, ys, s)
    if tx or ty:
        z = z + tx * xs + ty * ys
    return z


def sag_grad(x, y, s):
    dcx, dcy = s.get('dec') or (0.0, 0.0)
    tx, ty = s.get('tilt') or (0.0, 0.0)
    xs = np.asarray(x, float) - dcx
    ys = np.asarray(y, float) - dcy
    gx, gy = _base_sag_grad(xs, ys, s)
    if tx or ty:
        gx = gx + tx
        gy = gy + ty
    return gx, gy


def trace3d(x, y, z, L, M, N, opl, surfaces, n0=1.0, iters=120):
    """Exact 3-D sequential trace.  Returns the state ON the last surface.

    Damped Newton on ``F(t) = (z + t N) - zv - sag(x + t L, y + t M)``, then
    vector Snell / mirror reflection.  ``opl`` accumulates ``n * t``.
    """
    x = np.array(x, float, copy=True)
    y = np.array(y, float, copy=True)
    z = np.array(z, float, copy=True)
    L = np.array(L, float, copy=True)
    M = np.array(M, float, copy=True)
    N = np.array(N, float, copy=True)
    opl = np.array(opl, float, copy=True)
    n_cur = float(n0)
    for s in surfaces:
        zv = s['zv']
        t = (zv - z) / N
        for _ in range(iters):
            xx, yy, zz = x + t * L, y + t * M, z + t * N
            F = (zz - zv) - sag(xx, yy, s)
            gx, gy = sag_grad(xx, yy, s)
            dF = N - gx * L - gy * M
            dF = np.where(np.abs(dF) < 1e-300, 1e-300, dF)
            step = F / dF
            t = t - step
            if np.nanmax(np.abs(step)) < 1e-17:
                break
        x = x + t * L
        y = y + t * M
        z = z + t * N
        opl = opl + n_cur * t
        gx, gy = sag_grad(x, y, s)
        nx, ny, nz = -gx, -gy, np.ones_like(gx)
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / nn, ny / nn, nz / nn
        cosi = L * nx + M * ny + N * nz
        sgn = np.where(cosi > 0.0, -1.0, 1.0)
        nx, ny, nz = nx * sgn, ny * sgn, nz * sgn
        cosi = L * nx + M * ny + N * nz          # now <= 0
        if s.get('mirror', False):
            L = L - 2.0 * cosi * nx
            M = M - 2.0 * cosi * ny
            N = N - 2.0 * cosi * nz
        else:
            n_next = float(s['n_after'])
            mu = n_cur / n_next
            kk = 1.0 - mu * mu * (1.0 - cosi * cosi)
            if np.any(kk < 0):
                raise RuntimeError('TIR in the oracle trace')
            f = mu * cosi + np.sqrt(kk)
            L = mu * L - f * nx
            M = mu * M - f * ny
            N = mu * N - f * nz
            n_cur = n_next
    return dict(x=x, y=y, z=z, L=L, M=M, N=N, opl=opl, n=n_cur)


def to_vertex(st, n_exit, zv_last=0.0):
    """Project a last-surface state onto that surface's VERTEX plane.

    ``t = -z_local / N``, so the sign of ``N`` -- which a mirror flips -- is
    carried by the arithmetic rather than by a convention.
    """
    z_loc = st['z'] - zv_last
    t = -z_loc / st['N']
    return dict(x=st['x'] + st['L'] * t, y=st['y'] + st['M'] * t,
                z=np.zeros_like(z_loc), z_local=z_loc, t=t,
                L=st['L'], M=st['M'], N=st['N'],
                opl=st['opl'] + n_exit * t, n=st['n'])


# ---------------------------------------------------------------------------
# Band-limited angular-spectrum propagation (Matsushima & Shimobaba 2009).
# ---------------------------------------------------------------------------
def asm_propagate(E, dx, dy, lam, z, pad=2):
    ny, nx = E.shape
    py, px = ny * pad, nx * pad
    buf = np.zeros((py, px), complex)
    oy, ox = (py - ny) // 2, (px - nx) // 2
    buf[oy:oy + ny, ox:ox + nx] = E
    fx = np.fft.fftfreq(px, dx)
    fy = np.fft.fftfreq(py, dy)
    FX, FY = np.meshgrid(fx, fy)
    arg = 1.0 / lam ** 2 - FX ** 2 - FY ** 2
    prop = arg > 0.0
    kz = 2.0 * np.pi * np.sqrt(np.where(prop, arg, 0.0))
    Lx, Ly = px * dx, py * dy
    with np.errstate(divide='ignore', invalid='ignore'):
        fx_lim = 1.0 / (lam * np.sqrt((2.0 * abs(z) / Lx) ** 2 + 1.0))
        fy_lim = 1.0 / (lam * np.sqrt((2.0 * abs(z) / Ly) ** 2 + 1.0))
    band = (np.abs(FX) <= fx_lim) & (np.abs(FY) <= fy_lim) & prop
    H = np.where(band, np.exp(1j * kz * z), 0.0)
    out = np.fft.ifft2(np.fft.fft2(buf) * H)
    return out[oy:oy + ny, ox:ox + nx]


# ---------------------------------------------------------------------------
# Field metrics and bookkeeping.
# ---------------------------------------------------------------------------
def fidelity(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    num = abs(np.vdot(b, a)) ** 2
    den = float(np.vdot(a, a).real * np.vdot(b, b).real)
    return float(num / den) if den > 0 else 0.0


def rel_l2(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    d = float(np.linalg.norm(a - b))
    n = float(np.linalg.norm(b))
    return d / n if n > 0 else float('nan')


def sha(arr):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(arr)).tobytes()).hexdigest()


def build_tag():
    return f"{sys.platform}_{sys.version_info.major}{sys.version_info.minor}"


def assert_tree(root_env='VB12B_TREE'):
    """Print and assert the resolved ``lumenairy.__file__``."""
    import lumenairy as _la
    p = os.path.abspath(_la.__file__)
    print(f'lumenairy.__file__ = {p}')
    print(f'lumenairy.__version__ = {_la.__version__}')
    want = os.environ.get(root_env)
    if want:
        w = os.path.abspath(want)
        assert os.path.normcase(p).startswith(os.path.normcase(w)), (
            f'lumenairy resolved to {p}, not under {w}')
    return p


def env_block():
    import numpy as _np
    import scipy as _sp

    import lumenairy as _la
    return dict(platform=sys.platform, python=platform.python_version(),
                numpy=_np.__version__, scipy=_sp.__version__,
                lumenairy_version=_la.__version__,
                lumenairy_file=os.path.abspath(_la.__file__),
                mem_budget=os.environ.get('LUMENAIRY_MEM_BUDGET_MB'),
                threads=[os.environ.get(k) for k in
                         ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                          'MKL_NUM_THREADS')])


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    return str(o)


def dump(obj, name, outdir=None, tag=None):
    outdir = outdir or os.path.dirname(os.path.abspath(__file__))
    tag = tag or build_tag()
    path = os.path.join(outdir, f'{name}_{tag}.json')
    with open(path, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(obj, fh, indent=1, default=_jsonable)
    print(f'WROTE {path}')
    return path


# ---------------------------------------------------------------------------
# The in-line conic-sag copy the package DELETES, transcribed for measurement
# only (never a second live implementation).  Verbatim from
# ``lumenairy/propagators/gbd.py`` at 1218b24f.
# ---------------------------------------------------------------------------
def inline_conic_sag(surface, x, y):
    _Rl = float(getattr(surface, 'radius', np.inf))
    _kl = float(getattr(surface, 'conic', 0.0) or 0.0)
    if np.isfinite(_Rl) and _Rl != 0.0:
        _cl = 1.0 / _Rl
        _r2 = np.asarray(x) ** 2 + np.asarray(y) ** 2
        return _cl * _r2 / (1.0 + np.sqrt(np.maximum(
            1.0 - (1.0 + _kl) * _cl * _cl * _r2, 0.0)))
    return np.zeros_like(np.asarray(x))


def gbd_surfaces(prescription):
    """The surface list ``apply_prescription_persurface_to_beamlets`` traces
    on its LOCAL branch: ``surfaces_from_prescription`` with the last
    surface's thickness zeroed."""
    from lumenairy.raytrace import surfaces_from_prescription
    s = list(surfaces_from_prescription(prescription))
    s[-1] = copy.copy(s[-1])
    s[-1].thickness = 0.0
    return s


# ===========================================================================
# Fixtures.  ONE optic geometry -- biconvex, model glass n = 1.58,
# R1 = +6.0 mm, t = 1.1 mm, semi = 0.25 mm, 780 nm -- with only the LAST
# surface varied, so every difference between two rows is the last surface
# and nothing else.  NONE of these is a WP-B12b fixture (that package uses
# N-BAF10 R = 11 mm at 1.064 um, N-LASF9 at 850 nm and a R = -20 mm mirror)
# nor a VERIFY-WP-B12 one.
#
# NA 0.048, Airy radius 9.85 um against a 3.2 um pitch (3.1 px), so the focal
# structure is resolved and a fidelity against the diffraction oracle means
# something.  The 192 x 3.2 um grid spans 0.614 mm against a 0.500 mm clear
# aperture.
# ===========================================================================
LAM = 780e-9
R1 = 6.0e-3
R2 = -6.0e-3
TH = 1.1e-3
SEMI = 0.25e-3
W0 = 0.15e-3
NGRID = 192
DX = 3.2e-6
GLASS = 'VB12B-M158'


class VFixture:
    """One optic + one input beam, described TWICE -- as a lumenairy
    prescription and as my own surface list -- with no shared code."""

    def __init__(self, key, note, last=None, *, R1=R1, R2=R2, t=TH,
                 semi=SEMI, glass=GLASS, lam=LAM, w0=W0, N=NGRID, dx=DX,
                 first=None, mirror=False, immersed=None):
        self.key = key
        self.note = note
        self.last = dict(last or {})
        self.first = dict(first or {})
        self.R1, self.R2, self.t = R1, R2, t
        self.semi, self.glass = semi, glass
        self.lam, self.w0 = lam, w0
        self.N, self.dx = N, dx
        self.mirror = mirror
        self.immersed = immersed        # glass name AFTER the last surface

    # -- lumenairy side ---------------------------------------------------
    def prescription(self):
        register_model_glasses()
        if self.mirror:
            s = {'radius': self.R2, 'conic': 0.0, 'thickness': 0.0,
                 'glass_before': 'air', 'glass_after': 'MIRROR',
                 'semi_diameter': self.semi}
            s.update(self.last)
            return {'name': self.key, 'aperture_diameter': 2 * self.semi,
                    'surfaces': [s], 'thicknesses': [0.0], 'stop_index': 0}
        s0 = {'radius': self.R1, 'conic': 0.0, 'thickness': self.t,
              'glass_before': 'air', 'glass_after': self.glass,
              'semi_diameter': self.semi}
        s0.update(self.first)
        s1 = {'radius': self.R2, 'conic': 0.0, 'thickness': 0.0,
              'glass_before': self.glass,
              'glass_after': (self.immersed or 'air'),
              'semi_diameter': self.semi}
        s1.update(self.last)
        return {'name': self.key, 'aperture_diameter': 2 * self.semi,
                'surfaces': [s0, s1], 'thicknesses': [self.t],
                'stop_index': 0}

    # -- oracle side ------------------------------------------------------
    def _spec(self, d, zv, n_after, mirror=False):
        out = dict(zv=zv, Rx=float(d.get('radius', np.inf)),
                   kx=float(d.get('conic', 0.0) or 0.0),
                   ax=(d.get('aspheric_coeffs') or None),
                   Ry=(None if d.get('radius_y') is None
                       else float(d['radius_y'])),
                   ky=float(d.get('conic_y', 0.0) or 0.0),
                   ay=(d.get('aspheric_coeffs_y') or None),
                   xy=(d.get('xy_coeffs') or None),
                   dec=(tuple(float(v) for v in d['decenter'])
                        if d.get('decenter') else None),
                   tilt=(tuple(float(v) for v in d['tilt'])
                         if d.get('tilt') else None),
                   n_after=float(n_after), mirror=bool(mirror))
        return out

    def oracle_surfaces(self):
        p = self.prescription()
        ss = p['surfaces']
        if self.mirror:
            return [self._spec(ss[0], 0.0, 1.0, mirror=True)]
        ng = glass_index(self.glass, self.lam)
        return [self._spec(ss[0], 0.0, ng),
                self._spec(ss[1], self.t,
                           glass_index(self.immersed, self.lam)
                           if self.immersed else 1.0)]

    def n_exit(self):
        return float(self.oracle_surfaces()[-1]['n_after'])

    def rot_sym(self):
        d = self.last
        return not (d.get('radius_y') is not None or d.get('xy_coeffs')
                    or d.get('decenter') or d.get('tilt'))

    # -- grids ------------------------------------------------------------
    def axis(self, refine=1):
        n = self.N * refine
        return (np.arange(n) - n / 2) * (self.dx / refine)

    def grid(self, refine=1):
        a = self.axis(refine)
        return np.meshgrid(a, a)

    def E_in(self, refine=1):
        X, Y = self.grid(refine)
        return np.exp(-(X ** 2 + Y ** 2) / self.w0 ** 2).astype(np.complex128)

    # -- the oracle -------------------------------------------------------
    def exit_rays(self, a, b):
        surfs = self.oracle_surfaces()
        z0 = np.zeros_like(np.asarray(a, float))
        st = trace3d(a, b, z0, z0, z0, np.ones_like(z0), z0, surfs)
        return to_vertex(st, self.n_exit(), surfs[-1]['zv'])

    def exit_field(self, refine=2, n_src=None):
        """Geometrical-optics boundary field on the exit-VERTEX plane.

        ``E(xv, yv) = E_in(a, b) / sqrt|det d(xv,yv)/d(a,b)| * exp(i k opl)``
        -- intensity times ray-tube area is conserved.  Resampled by RADIUS
        when the optic is rotationally symmetric and by a C1 Clough-Tocher
        scattered interpolation of the two SMOOTH functions ``opl`` and
        ``amp`` otherwise.
        """
        X, Y = self.grid(refine)
        k = 2.0 * np.pi / self.lam
        rr = np.hypot(X, Y)
        n_src = int(n_src or 6 * self.N)
        if self.rot_sym():
            h = np.linspace(0.0, self.semi, n_src)
            v = self.exit_rays(h, np.zeros_like(h))
            xv, opl = v['x'], v['opl']
            dxe = np.gradient(xv, h, edge_order=2)
            with np.errstate(divide='ignore', invalid='ignore'):
                amp = (np.exp(-(h / self.w0) ** 2)
                       * np.sqrt(np.abs(h / (xv * dxe))))
            amp[0] = amp[1]
            inside = (rr >= xv.min()) & (rr <= xv.max())
            A = np.interp(rr.ravel(), xv, amp)
            P = np.interp(rr.ravel(), xv, opl)
            E = np.where(inside.ravel(), A * np.exp(1j * k * P), 0.0)
            return E.reshape(X.shape), dict(mode='radial', n_src=n_src)
        from scipy.interpolate import CloughTocher2DInterpolator
        m = max(int(n_src // 3), 129)
        s1 = np.linspace(-self.semi * 1.10, self.semi * 1.10, m)
        A0, B0 = np.meshgrid(s1, s1)
        v = self.exit_rays(A0, B0)
        da = s1[1] - s1[0]
        jx_a, jx_b = np.gradient(v['x'], da, da, edge_order=2)
        jy_a, jy_b = np.gradient(v['y'], da, da, edge_order=2)
        det = jx_a * jy_b - jx_b * jy_a
        amp = (np.exp(-(A0 ** 2 + B0 ** 2) / self.w0 ** 2)
               / np.sqrt(np.abs(det)))
        phase = k * v['opl']
        rin = np.hypot(A0, B0)
        ok = np.isfinite(amp) & np.isfinite(phase)
        pts = np.column_stack([v['x'][ok].ravel(), v['y'][ok].ravel()])
        vals = np.column_stack([amp[ok].ravel(), phase[ok].ravel(),
                                rin[ok].ravel()])
        itp = CloughTocher2DInterpolator(pts, vals, fill_value=np.nan)
        got = itp(np.column_stack([X.ravel(), Y.ravel()]))
        A, P, Rr = got[:, 0], got[:, 1], got[:, 2]
        good = (np.isfinite(A) & np.isfinite(P) & np.isfinite(Rr)
                & (Rr <= self.semi))
        E = np.where(good, A * np.exp(1j * P), 0.0)
        return E.reshape(X.shape), dict(mode='scattered', n_src=int(m))

    def oracle_field(self, z, refine=2, n_src=None):
        E, info = self.exit_field(refine=refine, n_src=n_src)
        if z == 0.0:
            out = E
        else:
            out = asm_propagate(E, self.dx / refine, self.dx / refine,
                                self.lam, z)
        return out[::refine, ::refine].copy(), info

    def oracle_converge(self, z, refine=2):
        """The oracle's OWN floor, measured: halve the ray quadrature, and
        halve the propagation grid refinement."""
        a, _ = self.oracle_field(z, refine=refine, n_src=3 * self.N)
        b, _ = self.oracle_field(z, refine=refine, n_src=6 * self.N)
        c, _ = self.oracle_field(z, refine=max(refine // 2, 1),
                                 n_src=6 * self.N)
        return dict(src_halved_infidelity=1.0 - fidelity(a, b),
                    grid_halved_infidelity=1.0 - fidelity(c, b))

    def best_focus(self):
        """Geometric best focus past the exit vertex, from MY own 3-D trace:
        the z minimising the intensity-weighted transverse variance of the
        exit bundle about its own centroid.  Derived at run time."""
        m = 101
        s1 = np.linspace(-self.semi * 0.995, self.semi * 0.995, m)
        A0, B0 = np.meshgrid(s1, s1)
        keep = np.hypot(A0, B0) <= self.semi
        A0, B0 = A0[keep], B0[keep]
        v = self.exit_rays(A0, B0)
        ux, uy = v['L'] / v['N'], v['M'] / v['N']
        w = np.exp(-2.0 * (A0 ** 2 + B0 ** 2) / self.w0 ** 2)
        w = w / w.sum()

        def lsq(p, u):
            pc = p - (w * p).sum()
            uc = u - (w * u).sum()
            den = (w * uc * uc).sum()
            return -(w * pc * uc).sum() / den if den > 0 else 0.0

        f0 = 0.5 * (lsq(v['x'], ux) + lsq(v['y'], uy))
        zs = np.linspace(0.70 * f0, 1.30 * f0, 2401)
        xz = v['x'][None, :] + ux[None, :] * zs[:, None]
        yz = v['y'][None, :] + uy[None, :] * zs[:, None]
        cx = (w[None, :] * xz).sum(1)
        cy = (w[None, :] * yz).sum(1)
        var = ((w[None, :] * (xz - cx[:, None]) ** 2).sum(1)
               + (w[None, :] * (yz - cy[:, None]) ** 2).sum(1))
        return float(zs[int(np.argmin(var))])

    def numerical_aperture(self):
        h = np.array([self.semi * 0.999])
        v = self.exit_rays(h, np.zeros_like(h))
        return float(abs(v['L'][0]))

    def airy_radius(self):
        na = self.numerical_aperture()
        return float(0.61 * self.lam / na) if na > 0 else float('inf')


# The fixture table.  ``mode`` says what each one is FOR.
def fixtures():
    F = {}
    F['conic'] = VFixture(
        'conic', 'CONTROL -- conic last surface (R = -6.0 mm): the in-line '
                 'copy was EXACT here, so the field may move only by the '
                 'Jacobian projection and floating-point reassociation')
    F['conic_k'] = VFixture(
        'conic_k', 'CONTROL -- the same optic with conic k = -0.80 on the '
                   'last surface; the in-line copy carried k',
        last={'conic': -0.80})
    F['asph'] = VFixture(
        'asph', 'even-aspheric last surface, A4 = 6.0e8 / A6 = -9.0e15 on a '
                'CURVED base -- the departure the in-line copy dropped',
        last={'aspheric_coeffs': {4: 6.0e8, 6: -9.0e15}})
    F['flatbase_asph'] = VFixture(
        'flatbase_asph', 'FLAT-BASE aspheric last surface (R = inf, power in '
                         'A2 = -8.333e1 / A4 = 5.0e8): the in-line copy '
                         'guarded on np.isfinite(radius) and read its sag as '
                         'EXACTLY zero -- 100 % of the sag',
        last={'radius': float('inf'),
              'aspheric_coeffs': {2: -83.3333333333333, 4: 5.0e8}})
    F['bicon'] = VFixture(
        'bicon', 'BICONIC last surface (Rx = -6.0 mm, Ry = -9.5 mm): the '
                 'in-line copy evaluated the x radius on both axes',
        last={'radius_y': -9.5e-3})
    F['freeform'] = VFixture(
        'freeform', 'FREEFORM (XY polynomial) last surface on the conic '
                    'base: the in-line copy dropped the whole departure',
        last={'freeform_type': 'xy_polynomial',
              'xy_coeffs': {(2, 0): 3.0e1, (0, 2): -1.8e1, (4, 0): 6.0e7},
              'norm_x': 1.0, 'norm_y': 1.0})
    F['fieldframe'] = VFixture(
        'fieldframe', 'FIELD-FRAME decentred last surface (50 um, -35 um): '
                      'the in-line copy evaluated the sag at the '
                      'undecentred coordinates',
        last={'decenter': (5.0e-5, -3.5e-5)})
    F['flat_last'] = VFixture(
        'flat_last', 'CONTROL -- plano-convex with the CURVED side FIRST, so '
                     'the last surface is FLAT and the projection '
                     'short-circuits structurally (byte identity)',
        R1=3.4e-3, R2=float('inf'), t=0.9e-3, glass='VB12B-M172',
        semi=0.25e-3)
    F['mirror'] = VFixture(
        'mirror', 'concave MIRROR last surface (R = -15.0 mm): the in-line '
                  'copy assumed a forward-going exit ray, so it applied '
                  '-sag with the wrong SIGN and DOUBLED the error',
        R2=-15.0e-3, semi=0.40e-3, w0=0.24e-3, N=160, dx=6.0e-6,
        mirror=True)
    F['immersed'] = VFixture(
        'immersed', 'the same optic exiting into GLASS (n = 1.72), for the '
                    'vacuum-exit-medium open item',
        immersed='VB12B-M172')
    for k, f in F.items():
        f.mode = k
    return F


def fixture(key):
    return fixtures()[key]


# The freeform coefficients are written into the prescription under the
# library's own keys; my oracle reads them from the SAME dict through
# ``_spec``, so the two sides share the data and not the evaluation.
def _last_dict(fx):
    return fx.prescription()['surfaces'][-1]

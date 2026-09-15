"""VERIFY-WP-B12 -- my own fixtures, my own 3-D tracer, my own diffraction
oracle.

Independent of both the library path under test and of WP-B12's own probe:

* the tracer is **fully 3-D** (the WP-B12 probe's is meridional and
  rotationally symmetric), so it covers a BICONIC last surface, an OBLIQUE
  input and a MIRROR-terminated prescription -- the three classes the WP-B12
  oracle structurally cannot represent;
* the conic sag is evaluated from the **quadric root**
  ``z = (1 - sqrt(1 - (1+k) c^2 h^2)) / ((1+k) c)`` rather than from the
  ``c h^2 / (1 + sqrt(...))`` rationalised form both the library and the
  WP-B12 probe use -- algebraically the same surface, a different expression;
* the intersection is a **damped Newton on the implicit surface equation in
  3-D** with the analytic transverse gradient;
* refraction is vector Snell in the ``(n1/n2)`` form with the normal oriented
  against the incident ray; reflection is ``d - 2 (d.n) n``;
* propagation is a **band-limited angular spectrum** (Matsushima) of the
  exit-vertex-plane boundary field -- the WP-B12 probe uses a brute-force
  Rayleigh-Sommerfeld-I ring sum, so the two oracles disagree in method as
  well as in code.

``lumenairy`` is imported only to build the prescriptions the fixtures name
and to score the library members; never for the oracle's physics.

Author: VERIFY-WP-B12
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Glass.  Most fixtures use a DISPERSIONLESS model index registered
# probe-locally, so the index is the same number on both sides by
# construction and glass modelling is out of this package's scope.  One
# fixtures use real catalogue glasses with the Sellmeier coefficients typed in
# here, and ``probe_v0_controls.py`` control 1 measures those against
# ``lumenairy.glass.get_glass_index`` (2.2e-16 on Windows, 4.4e-16 on WSL).
# ---------------------------------------------------------------------------
SELLMEIER = {
    # Schott N-SF10 (not used by WP-B12's probe).
    'N-SF10': ((1.62153902, 0.256287842, 1.64447552),
               (0.0122241457, 0.0595736775, 147.468793)),
    # Schott N-SSK8.
    'N-SSK8': ((1.44857867, 0.117965926, 1.06937528),
               (0.00869310149, 0.0421566593, 111.300666)),
}


def n_sellmeier(name, wavelength_m):
    b, c = SELLMEIER[name]
    l2 = (wavelength_m * 1e6) ** 2
    s = 1.0
    for bi, ci in zip(b, c):
        s += bi * l2 / (l2 - ci)
    return math.sqrt(s)


# ---------------------------------------------------------------------------
# My surface model.  ``spec`` is a plain dict:
#   zv      vertex z [m]
#   Rx, Ry  x / y radii (Ry None -> rotationally symmetric in h^2)
#   kx, ky  conics
#   ax, ay  even-aspheric {power: coeff} (x branch / y branch)
#   n_after index after the surface (ignored for a mirror)
#   mirror  bool
# ---------------------------------------------------------------------------
def _conic_axis(u2, R, k):
    """Conic sag of one axis from the QUADRIC ROOT.

    ``(1+k) c z^2 - 2 z + c u2 = 0`` has the physical root
    ``z = (1 - sqrt(1 - (1+k) c^2 u2)) / ((1+k) c)``; at ``k = -1`` the
    quadratic degenerates and the root is ``c u2 / 2``.  Algebraically the
    same surface as the library's ``c u2 / (1 + sqrt(...))``, a different
    expression -- so the two agree only if both are right.
    """
    if R is None or np.isinf(R):
        return np.zeros_like(np.asarray(u2, float))
    c = 1.0 / R
    kp = 1.0 + k
    rad = 1.0 - kp * c * c * np.asarray(u2, float)
    rad = np.where(rad < 0.0, np.nan, rad)
    if abs(kp) < 1e-12:
        return 0.5 * c * np.asarray(u2, float)
    return (1.0 - np.sqrt(rad)) / (kp * c)


def _conic_axis_d(u, R, k):
    """d/du of ``_conic_axis(u**2, R, k)``."""
    u = np.asarray(u, float)
    if R is None or np.isinf(R):
        return np.zeros_like(u)
    c = 1.0 / R
    kp = 1.0 + k
    rad = 1.0 - kp * c * c * u * u
    rad = np.where(rad < 0.0, np.nan, rad)
    if abs(kp) < 1e-12:
        return c * u
    # d/du (1 - sqrt(rad))/(kp c) = (kp c^2 u / sqrt(rad)) / (kp c) = c u/sqrt
    return c * u / np.sqrt(rad)


def sag(x, y, s):
    """Surface sag (relative to its own vertex)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if s.get('Ry') is None:
        z = _conic_axis(x * x + y * y, s['Rx'], s.get('kx', 0.0))
        for p, a in (s.get('ax') or {}).items():
            z = z + a * (x * x + y * y) ** (p // 2)
        return z
    # Separable biconic (the library's documented RT-2 form).
    z = (_conic_axis(x * x, s['Rx'], s.get('kx', 0.0))
         + _conic_axis(y * y, s['Ry'], s.get('ky', 0.0)))
    for p, a in (s.get('ax') or {}).items():
        z = z + a * (x * x) ** (p // 2)
    for p, a in (s.get('ay') or {}).items():
        z = z + a * (y * y) ** (p // 2)
    return z


def sag_grad(x, y, s):
    """(dz/dx, dz/dy)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if s.get('Ry') is None:
        h = np.sqrt(x * x + y * y)
        hs = np.where(h > 0, h, 1.0)
        d = _conic_axis_d(h, s['Rx'], s.get('kx', 0.0))
        for p, a in (s.get('ax') or {}).items():
            m = p // 2
            d = d + a * m * h ** (2 * m - 1) * 2.0
        gx = np.where(h > 0, d * x / hs, 0.0)
        gy = np.where(h > 0, d * y / hs, 0.0)
        return gx, gy
    gx = _conic_axis_d(x, s['Rx'], s.get('kx', 0.0))
    gy = _conic_axis_d(y, s['Ry'], s.get('ky', 0.0))
    for p, a in (s.get('ax') or {}).items():
        m = p // 2
        gx = gx + a * m * 2.0 * x * (x * x) ** (m - 1)
    for p, a in (s.get('ay') or {}).items():
        m = p // 2
        gy = gy + a * m * 2.0 * y * (y * y) ** (m - 1)
    return gx, gy


def trace3d(x, y, z, L, M, N, opl, surfaces, n0=1.0, iters=80):
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
            step = np.clip(step, -abs(zv - 0.0) - 1e-2, abs(zv - 0.0) + 1e-2)
            t = t - step
            if np.nanmax(np.abs(step)) < 1e-17:
                break
        x = x + t * L
        y = y + t * M
        z = z + t * N
        opl = opl + n_cur * t
        gx, gy = sag_grad(x, y, s)
        # Outward normal (pointing against +z convention: (-gx, -gy, 1)).
        nx, ny, nz = -gx, -gy, np.ones_like(gx)
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / nn, ny / nn, nz / nn
        cosi = L * nx + M * ny + N * nz
        # Orient the normal AGAINST the incident ray.
        sgn = np.where(cosi > 0.0, -1.0, 1.0)
        nx, ny, nz = nx * sgn, ny * sgn, nz * sgn
        cosi = L * nx + M * ny + N * nz          # now <= 0
        if s.get('mirror', False):
            L = L - 2.0 * cosi * nx
            M = M - 2.0 * cosi * ny
            N = N - 2.0 * cosi * nz
            # medium unchanged
        else:
            n_next = float(s['n_after'])
            mu = n_cur / n_next
            k = 1.0 - mu * mu * (1.0 - cosi * cosi)
            if np.any(k < 0):
                raise RuntimeError('TIR in the oracle trace')
            f = mu * cosi + np.sqrt(k)
            L = mu * L - f * nx
            M = mu * M - f * ny
            N = mu * N - f * nz
            n_cur = n_next
    return dict(x=x, y=y, z=z, L=L, M=M, N=N, opl=opl, n=n_cur)


def to_vertex(st, n_exit, zv_last=0.0):
    """Project a last-surface state onto the last surface's vertex plane.

    ``zv_last`` is that surface's vertex z in THIS tracer's running global
    frame; ``lumenairy.raytrace.trace`` resets the frame at every transfer,
    so its ``image_rays.z`` is already the local ``sag`` (measured: the two
    differ by exactly the cumulative vertex position, probe V0).
    """
    z_loc = st['z'] - zv_last
    t = -z_loc / st['N']
    return dict(x=st['x'] + st['L'] * t, y=st['y'] + st['M'] * t,
                z=np.zeros_like(z_loc), z_local=z_loc,
                L=st['L'], M=st['M'], N=st['N'],
                opl=st['opl'] + n_exit * t, n=st['n'])


# ---------------------------------------------------------------------------
# Band-limited angular-spectrum propagation (Matsushima & Shimobaba 2009).
# ---------------------------------------------------------------------------
def asm_propagate(E, dx, dy, lam, z, pad=2):
    """Exact (band-limited) angular-spectrum propagation of ``E`` by ``z``."""
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
    # Matsushima band limit: the local fringe of H must stay inside the
    # window, |z lam fx / sqrt(1-(lam f)^2)| <= L/2.
    Lx, Ly = px * dx, py * dy
    with np.errstate(divide='ignore', invalid='ignore'):
        fx_lim = 1.0 / (lam * np.sqrt((2.0 * abs(z) / Lx) ** 2 + 1.0))
        fy_lim = 1.0 / (lam * np.sqrt((2.0 * abs(z) / Ly) ** 2 + 1.0))
    band = (np.abs(FX) <= fx_lim) & (np.abs(FY) <= fy_lim) & prop
    H = np.where(band, np.exp(1j * kz * z), 0.0)
    out = np.fft.ifft2(np.fft.fft2(buf) * H)
    return out[oy:oy + ny, ox:ox + nx]


# ---------------------------------------------------------------------------
# Field metrics.
# ---------------------------------------------------------------------------
def fidelity(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    num = abs(np.vdot(b, a)) ** 2
    den = float(np.vdot(a, a).real * np.vdot(b, b).real)
    return float(num / den) if den > 0 else 0.0


def power_ratio(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    pb = float((np.abs(b) ** 2).sum())
    return float((np.abs(a) ** 2).sum() / pb) if pb > 0 else float('nan')


def sha(arr):
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(a.tobytes()).hexdigest()


def build_tag():
    return f"{sys.platform}_{sys.version_info.major}{sys.version_info.minor}"


def env_block():
    import numpy as _np
    import scipy as _sp

    import lumenairy as _la
    d = dict(platform=sys.platform, python=platform.python_version(),
             numpy=_np.__version__, scipy=_sp.__version__,
             lumenairy_version=_la.__version__,
             lumenairy_file=os.path.abspath(_la.__file__))
    try:
        import numba
        d['numba'] = numba.__version__
    except Exception:
        d['numba'] = None
    try:
        import jax
        d['jax'] = jax.__version__
    except Exception:
        d['jax'] = None
    return d


def dump(obj, name, outdir=None):
    outdir = outdir or os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(outdir, f'{name}_{build_tag()}.json')
    with open(path, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump(obj, fh, indent=1, default=_jsonable)
    print(f'WROTE {path}')
    return path


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    return str(o)


# ===========================================================================
# Fixtures.  None of these is a WP-B12 fixture: different glasses, different
# radii, different wavelengths, and three surface / input classes the WP-B12
# oracle cannot represent at all.
# ===========================================================================
MODEL_INDICES = {
    'VB12-M167': 1.67,        # dispersionless model glasses, registered
    'VB12-M152': 1.52,        # probe-locally so the oracle and the library
    'VB12-M178': 1.78,        # read the SAME index by construction
}


def register_model_glasses():
    from lumenairy import glass as _g
    for nm, nv in MODEL_INDICES.items():
        _g.GLASS_REGISTRY[nm] = (lambda wl, _n=float(nv): _n)


def glass_index(name, lam):
    if name is None or str(name).lower() in ('air', 'vacuum', ''):
        return 1.0
    if name in MODEL_INDICES:
        return float(MODEL_INDICES[name])
    return n_sellmeier(name, lam)


class Fixture:
    """One optic + one input field, described twice: as a lumenairy
    prescription and as my own surface list."""

    def __init__(self, key, note, surfaces, thicknesses, glasses, lam,
                 semi, N, dx, w0, tilt_deg=0.0, flat_last=False,
                 mirror_last=False, conics=None, asph=None, radii_y=None,
                 conics_y=None):
        self.key = key
        self.note = note
        self.radii = surfaces            # list of x-radii, one per surface
        self.radii_y = radii_y           # list or None
        self.conics = conics or [0.0] * len(surfaces)
        self.conics_y = conics_y
        self.asph = asph or [None] * len(surfaces)
        self.thicknesses = thicknesses
        self.glasses = glasses           # glass AFTER each surface
        self.lam = lam
        self.semi = semi
        self.N = N
        self.dx = dx
        self.w0 = w0
        self.tilt_deg = tilt_deg
        self.flat_last = flat_last
        self.mirror_last = mirror_last

    # -- lumenairy side ---------------------------------------------------
    def prescription(self):
        register_model_glasses()
        surfs = []
        gb = 'air'
        for i, R in enumerate(self.radii):
            ga = self.glasses[i]
            d = {'radius': float(R), 'conic': float(self.conics[i]),
                 'aspheric_coeffs': self.asph[i],
                 'radius_y': (None if self.radii_y is None
                              else self.radii_y[i]),
                 'conic_y': (None if self.conics_y is None
                             else self.conics_y[i]),
                 'aspheric_coeffs_y': None,
                 'glass_before': gb, 'glass_after': ga}
            if self.mirror_last and i == len(self.radii) - 1:
                d['is_mirror'] = True
                d['glass_after'] = gb
            surfs.append(d)
            gb = ga
        return {'name': self.key, 'aperture_diameter': 2.0 * self.semi,
                'surfaces': surfs, 'thicknesses': list(self.thicknesses)}

    # -- oracle side ------------------------------------------------------
    def oracle_surfaces(self):
        out = []
        zv = 0.0
        gb = 'air'
        for i, R in enumerate(self.radii):
            ga = self.glasses[i]
            mir = self.mirror_last and i == len(self.radii) - 1
            out.append(dict(
                zv=zv, Rx=float(R),
                Ry=(None if self.radii_y is None else self.radii_y[i]),
                kx=float(self.conics[i]),
                ky=(0.0 if self.conics_y is None else
                    float(self.conics_y[i] or 0.0)),
                ax=self.asph[i], ay=None,
                n_after=(glass_index(gb, self.lam) if mir
                         else glass_index(ga, self.lam)),
                mirror=mir))
            if i < len(self.thicknesses):
                zv = zv + float(self.thicknesses[i])
            gb = ga
        return out

    def n_exit(self):
        s = self.oracle_surfaces()[-1]
        return float(s['n_after'])

    # -- grids / fields ---------------------------------------------------
    def axis(self, refine=1):
        n = self.N * refine
        d = self.dx / refine
        return (np.arange(n) - n / 2) * d

    def grid(self, refine=1):
        a = self.axis(refine)
        return np.meshgrid(a, a)

    def E_in(self, refine=1):
        X, Y = self.grid(refine)
        k = 2.0 * np.pi / self.lam
        ph = k * math.sin(math.radians(self.tilt_deg)) * X
        return (np.exp(-(X ** 2 + Y ** 2) / self.w0 ** 2)
                * np.exp(1j * ph)).astype(np.complex128)

    def input_dirs(self, shape):
        th = math.radians(self.tilt_deg)
        L = np.full(shape, math.sin(th))
        M = np.zeros(shape)
        N = np.full(shape, math.cos(th))
        return L, M, N

    def coarse(self, factor=2):
        """The same optic and the same window on a coarser grid.

        GBD's per-surface path costs ~N^4 here (27 s at N = 96 against 5.6 s
        at N = 64 on this box), so the GBD arms of probe V4 run at half the
        linear resolution; the OPTIC, the beam and the window are unchanged,
        so the oracle comparison is like for like."""
        import copy as _c
        g = _c.copy(self)
        g.N = self.N // factor
        g.dx = self.dx * factor
        return g

    # -- oracle -----------------------------------------------------------
    def _rot_sym(self):
        return self.tilt_deg == 0.0 and self.radii_y is None

    def _exit_rays(self, a, b):
        surfs = self.oracle_surfaces()
        L, M, N = self.input_dirs(a.shape)
        st = trace3d(a, b, np.zeros_like(a), L, M, N, np.zeros_like(a), surfs)
        return to_vertex(st, self.n_exit(), surfs[-1]['zv'])

    def exit_field(self, refine=4, n_src=None):
        """Exit-vertex-plane boundary field on the refined grid.

        Geometrical optics: the ray tube through ``(a, b)`` on the entrance
        plane lands at ``(xv, yv)`` with optical path ``opl``, so

            E_exit(xv, yv) = E_in(a, b) / sqrt|det d(xv,yv)/d(a,b)|
                             * exp(i k opl)

        (intensity times area is conserved along the tube).  The map is
        resampled onto the refined output grid -- by RADIUS when the optic
        and the input are rotationally symmetric, and by a C1 cubic
        (Clough-Tocher) scattered interpolation of the two SMOOTH functions
        ``opl`` and ``amp`` otherwise.  The interpolation error is bounded by
        the source spacing times the curvature of ``opl``; ``oracle_converge``
        measures it by halving the spacing.
        """
        X, Y = self.grid(refine)
        k = 2.0 * np.pi / self.lam
        th = math.radians(self.tilt_deg)
        rr = np.sqrt(X * X + Y * Y)
        n_src = n_src or (8 * self.N)
        if self._rot_sym():
            h = np.linspace(0.0, self.semi, n_src)
            b0 = np.zeros_like(h)
            v = self._exit_rays(h, b0)
            xv = v['x']
            opl = v['opl']
            # radial ray-tube area element: h dh / (xv dxv)
            dxe = np.gradient(xv, h, edge_order=2)
            with np.errstate(divide='ignore', invalid='ignore'):
                amp = (np.exp(-(h / self.w0) ** 2)
                       * np.sqrt(np.abs(h / (xv * dxe))))
            amp[0] = amp[1]                     # the axis is a removable 0/0
            inside = (rr >= xv.min()) & (rr <= xv.max())
            A = np.interp(rr.ravel(), xv, amp)
            P = np.interp(rr.ravel(), xv, opl)
            E = np.where(inside.ravel(), A * np.exp(1j * k * P), 0.0)
            return E.reshape(X.shape), dict(mode='radial', n_src=int(n_src))
        # -- general 2-D --------------------------------------------------
        from scipy.interpolate import CloughTocher2DInterpolator
        m = max(int(n_src // 2), 129)
        # a source grid extending past the aperture so the hull covers the
        # readout region; the aperture mask is applied in INPUT coordinates.
        s1 = np.linspace(-self.semi * 1.12, self.semi * 1.12, m)
        A0, B0 = np.meshgrid(s1, s1)
        v = self._exit_rays(A0, B0)
        da = s1[1] - s1[0]
        jx_a, jx_b = np.gradient(v['x'], da, da, edge_order=2)
        jy_a, jy_b = np.gradient(v['y'], da, da, edge_order=2)
        det = jx_a * jy_b - jx_b * jy_a
        amp = (np.exp(-(A0 ** 2 + B0 ** 2) / self.w0 ** 2)
               / np.sqrt(np.abs(det)))
        phase = k * (math.sin(th) * A0 + v['opl'])
        rin = np.sqrt(A0 ** 2 + B0 ** 2)
        ok = np.isfinite(amp) & np.isfinite(phase)
        pts = np.column_stack([v['x'][ok].ravel(), v['y'][ok].ravel()])
        vals = np.column_stack([amp[ok].ravel(), phase[ok].ravel(),
                                rin[ok].ravel()])
        itp = CloughTocher2DInterpolator(pts, vals, fill_value=np.nan)
        got = itp(np.column_stack([X.ravel(), Y.ravel()]))
        A, P, R = got[:, 0], got[:, 1], got[:, 2]
        good = np.isfinite(A) & np.isfinite(P) & np.isfinite(R)             & (R <= self.semi)
        E = np.where(good, A * np.exp(1j * P), 0.0)
        return E.reshape(X.shape), dict(mode='scattered', n_src=int(m))

    def oracle_field(self, z, refine=4, n_src=None):
        """Oracle field on the FIXTURE's own grid at ``z`` past the exit
        vertex (exact subsample of the refined ASM result)."""
        E, info = self.exit_field(refine=refine, n_src=n_src)
        if z == 0.0:
            out = E
        else:
            out = asm_propagate(E, self.dx / refine, self.dx / refine,
                                self.lam, z)
        return out[::refine, ::refine].copy(), info

    def best_focus(self):
        """Geometric best focus past the exit vertex, from my own trace of a
        2-D ray grid: the z that minimises the intensity-weighted transverse
        variance of the exit bundle about its own centroid.

        Derived at run time from the fixture's own geometry (never pinned),
        and general -- it handles the tilted input (the centroid moves with
        the chief ray) and the biconic (whose x and y line foci differ, so
        the minimiser of the COMBINED variance is the compromise plane).
        """
        m = 121
        s1 = np.linspace(-self.semi * 0.995, self.semi * 0.995, m)
        A0, B0 = np.meshgrid(s1, s1)
        keep = np.sqrt(A0 ** 2 + B0 ** 2) <= self.semi
        A0, B0 = A0[keep], B0[keep]
        v = self._exit_rays(A0, B0)
        ux, uy = v['L'] / v['N'], v['M'] / v['N']
        w = np.exp(-2.0 * (A0 ** 2 + B0 ** 2) / self.w0 ** 2)
        w = w / w.sum()

        def lsq(p, u):
            pc = p - (w * p).sum()
            uc = u - (w * u).sum()
            den = (w * uc * uc).sum()
            return -(w * pc * uc).sum() / den if den > 0 else 0.0

        f0 = 0.5 * (lsq(v['x'], ux) + lsq(v['y'], uy))
        zs = np.linspace(0.65 * f0, 1.35 * f0, 2401)
        xz = v['x'][None, :] + ux[None, :] * zs[:, None]
        yz = v['y'][None, :] + uy[None, :] * zs[:, None]
        cx = (w[None, :] * xz).sum(1)
        cy = (w[None, :] * yz).sum(1)
        var = ((w[None, :] * (xz - cx[:, None]) ** 2).sum(1)
               + (w[None, :] * (yz - cy[:, None]) ** 2).sum(1))
        return float(zs[int(np.argmin(var))])

    def oracle_converge(self, z, refine=4):
        """Halve the oracle's source spacing and report the change: the
        oracle's OWN floor, stated rather than assumed."""
        a, _ = self.oracle_field(z, refine=refine, n_src=4 * self.N)
        b, _ = self.oracle_field(z, refine=refine, n_src=8 * self.N)
        c, _ = self.oracle_field(z, refine=max(refine // 2, 1),
                                 n_src=8 * self.N)
        return dict(src_halved_fidelity=fidelity(a, b),
                    grid_halved_fidelity=fidelity(c, b))


def fixtures():
    """The eight fixtures.  Six have a CURVED last surface, two are FLAT
    controls; one is aspheric, one biconic, one a mirror, one oblique.

    Common regime: a 0.256 mm aperture on a 192 x 192 grid, NA 0.08-0.11,
    so the Airy radius is three or more pixels and the oracle's own readout
    is not the limit.  None of these is a WP-B12 fixture."""
    N, DX, SEMI, W0 = 192, 1.6e-6, 128e-6, 57e-6
    F = []
    # 1. Aspheric LAST surface (conic + A4 + A6) -- a 0.56 um polynomial
    #    departure at the rim, which a conic-only sag copy drops entirely.
    F.append(Fixture(
        key='asph', note='plano-aspheric singlet, aspheric LAST surface '
                         '(R=-0.911 mm, k=-0.6, A4=2.0e9, A6=-4.0e16), '
                         'N-SF10, 780 nm',
        surfaces=[np.inf, -0.911e-3], thicknesses=[0.55e-3],
        glasses=['N-SF10', 'air'], conics=[0.0, -0.60],
        asph=[None, {4: 2.0e9, 6: -4.0e16}],
        lam=780e-9, semi=SEMI, N=N, dx=DX, w0=W0))
    # 2. BICONIC last surface (Rx != Ry, so the optic is astigmatic) -- the
    #    class the WP-B12 oracle cannot represent.
    F.append(Fixture(
        key='bicon', note='biconic singlet, LAST surface Rx=-0.858 mm / '
                          'Ry=-1.17 mm, model n=1.67, 1.03 um',
        surfaces=[np.inf, -0.858e-3], radii_y=[np.inf, -1.17e-3],
        conics=[0.0, 0.0], conics_y=[0.0, 0.0],
        thicknesses=[0.55e-3], glasses=['VB12-M167', 'air'],
        lam=1.03e-6, semi=SEMI, N=N, dx=DX, w0=W0))
    # 3. Converging MENISCUS -- last surface sag POSITIVE (the sign check),
    #    and the smallest defect of the set.
    F.append(Fixture(
        key='menisc', note='converging meniscus R1=+1.05 mm R2=+5.4 mm, '
                           'model n=1.78, 1.31 um',
        surfaces=[1.05e-3, 5.4e-3], thicknesses=[0.60e-3],
        glasses=['VB12-M178', 'air'],
        lam=1.31e-6, semi=SEMI, N=N, dx=DX, w0=W0))
    # 4. Cemented DOUBLET -- three surfaces, two glasses, curved last.
    F.append(Fixture(
        key='doublet', note='cemented doublet R=+1.20/-0.62/-3.00 mm, '
                            'N-SSK8 + N-SF10, 633 nm',
        surfaces=[1.20e-3, -0.62e-3, -3.00e-3],
        thicknesses=[0.40e-3, 0.25e-3],
        glasses=['N-SSK8', 'N-SF10', 'air'],
        lam=633e-9, semi=SEMI, N=N, dx=DX, w0=W0))
    # 5. OBLIQUE incidence -- a 6 deg tilted plane wave on a biconvex.
    F.append(Fixture(
        key='oblique', note='biconvex R=+/-1.40 mm, model n=1.52, 850 nm, '
                            'input tilted 6 deg in x',
        surfaces=[1.40e-3, -1.40e-3], thicknesses=[0.55e-3],
        glasses=['VB12-M152', 'air'], tilt_deg=6.0,
        lam=850e-9, semi=SEMI, N=N, dx=DX, w0=W0))
    # 6. Concave MIRROR last surface -- where _exit_direction_sign matters.
    #    Primitive-level only: FGA's image leg runs +z past the vertex, so a
    #    mirror-terminated prescription is not an FGA input.
    F.append(Fixture(
        key='mirror', note='plate + concave MIRROR R=-2.5 mm as the last '
                           'surface, 633 nm -- primitive-level only',
        surfaces=[np.inf, -2.5e-3], thicknesses=[0.90e-3],
        glasses=['air', 'air'], mirror_last=True,
        lam=633e-9, semi=SEMI, N=N, dx=DX, w0=W0))
    # 7. FLAT last surface control -- convex FIRST, plano LAST.
    F.append(Fixture(
        key='flat_planoconvex', note='plano-convex, curved FIRST / FLAT '
                                     'last, model n=1.67, 780 nm (CONTROL)',
        surfaces=[1.00e-3, np.inf], thicknesses=[0.55e-3],
        glasses=['VB12-M167', 'air'], flat_last=True,
        lam=780e-9, semi=SEMI, N=N, dx=DX, w0=W0))
    # 8. FLAT last surface control -- an air-spaced pair, four surfaces.
    F.append(Fixture(
        key='flat_pair', note='air-spaced pair, FLAT last surface, N-SSK8, '
                              '1.03 um (CONTROL)',
        surfaces=[1.65e-3, -2.55e-3, 4.50e-3, np.inf],
        thicknesses=[0.38e-3, 0.19e-3, 0.34e-3],
        glasses=['N-SSK8', 'air', 'N-SSK8', 'air'], flat_last=True,
        lam=1.03e-6, semi=SEMI, N=N, dx=DX, w0=W0))
    return F


def fixture(key):
    for f in fixtures():
        if f.key == key:
            return f
    raise KeyError(key)

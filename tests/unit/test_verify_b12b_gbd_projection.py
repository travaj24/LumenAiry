"""VERIFY-WP-B12b: the gaps WP-B12b's own suite leaves open, each closed as a
two-sided decision against an INDEPENDENT 3-D diffraction oracle.

Added 2026-09-19 by the independent re-verification of WP-B12b
(``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/
VERIFY_WP-B12b.md``).

WP-B12b deleted ``propagators.gbd.apply_prescription_persurface_to_beamlets``'s
in-line conic-sag copy and had the local-frame branch ask the differential
primitive for ``reference='exit_vertex'``.  Its own suite
(``test_audit2609_b12b_gbd_projection.py``, 14 ids) pins the primitive's state
on every surface class, the flat-surface bit-identity, and ONE field decision
-- on the flat-base aspheric fixture, against a rotationally-symmetric
Rayleigh-Sommerfeld oracle.  Re-verification found four properties that suite
does not hold, confirmed by in-memory mutation (2026-09-19,
``validation/probe_verify_b12b/vb12b_mutate.py``: all fourteen of its ids stay
GREEN under three of the five mutations tried):

* **the deleted ``-sag`` fold, put back** while the projection stays, is caught
  only by that file's SOURCE-token check, so any spelling of the same double
  correction that avoids the names ``_Rl`` / ``_kl`` / ``_cl`` passes the whole
  suite.  A CONIC last surface is where that matters -- it is exactly the class
  the in-line copy got right, so the builder's suite has no diffraction row
  there at all;
* **the biconic, the freeform and the field-frame classes** were reported as
  field MOVEMENT only, because the imported oracle is a rotationally-symmetric
  ring sum.  The oracle here is a fully 3-D trace plus a band-limited angular
  spectrum, so those three get scored;
* **the ``world_output_plane`` branch's ``reference='surface'``** is pinned as
  *which keyword it asks for*, never as *the right one*;
* the **mirror-terminated local branch** and the **immersed exit** are recorded
  as open items with no pin at all.

Every oracle here is built from this file's own geometry -- the conic sag from
the QUADRIC ROOT rather than the library's rationalised form, the intersection
from a damped Newton on the implicit 3-D surface, refraction from vector Snell
-- and is checked against the library's own ``TraceResult.at_exit_vertex()``
before it is used.  Every bar is derived at run time from a quantity this build
measures, with the gap to the signal printed in the assertion.  No wall-clock
assertion.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators import gbd as G
from lumenairy.raytrace import differential as D

_LAM = 780e-9
_GLASS_N = 1.58
_GLASS = 'VB12B-T158'
_R1, _R2, _T = 6.0e-3, -6.0e-3, 1.1e-3
# The grid every field test uses.  112 x 3.6 um spans 0.403 mm against a
# 0.340 mm clear aperture, and the Airy radius (13 .. 19 um across the
# fixtures) is four to five pixels, so the focal structure is resolved and a
# fidelity against the oracle means something.  The size is a COST choice:
# the per-surface GBD path is ~N^4 here, and 112 keeps every id in this file
# inside the 60 s budget with the box loaded (measured 2026-09-19).
_N, _DX, _SEMI, _W0 = 112, 3.6e-6, 0.17e-3, 0.105e-3
# The beamlet frame.  Named rather than left to ``_auto_sample_step`` because
# the auto frame is one beamlet per pixel, whose beamlets are wider than the
# grid at the image plane, so the windowed reconstruction degenerates to the
# dense sum.  ``test_the_named_beamlet_frame_is_a_cost_choice_not_a_result``
# re-derives the choice at run time instead of trusting this comment.
_FRAME = dict(sample_step=4, waist_factor=4.0)


#: VERIFY-WP-B12b D-6 (2026-09-19).  A GBD field's SHA-256 depends on the
#: memory budget through the PUBLIC entry: ``_reconstruct_windowed`` chunks
#: each bucket of the coherent beamlet sum to stay under it, and the chunk
#: boundaries change the grouping of a ``bincount`` scatter-add.  Measured on
#: one fixture with only ``LUMENAIRY_MEM_BUDGET_MB`` varied -- unset / 4096 /
#: 2048 / 512 give ONE digest, 64 and 8 give two others, while the fields
#: agree to ~1e-15 (``validation/probe_wp_b12b_round2/probe_r3_budget.py``,
#: both builds).  The environment variable is a CEILING on the kwarg, so a
#: byte-identity id must pin BOTH or it is only reproducible in the
#: environment it was written in.
_MEM_BUDGET_MB = 2048.0


@pytest.fixture(autouse=True)
def _pin_mem_budget(monkeypatch):
    """Every id in this file runs at ONE memory budget (D-6), so its digests
    are a property of the library and not of the shell that invoked it."""
    monkeypatch.setenv('LUMENAIRY_MEM_BUDGET_MB', str(int(_MEM_BUDGET_MB)))



# ===========================================================================
# Glass: a dispersionless MODEL index registered here, so the oracle and the
# library read the same number by construction and glass modelling is out of
# this file's scope.
# ===========================================================================
def _register_glass():
    from lumenairy import glass as _g
    _g.GLASS_REGISTRY[_GLASS] = lambda wl: _GLASS_N


# ===========================================================================
# My own surface model and 3-D tracer -- no library physics.
# ===========================================================================
def _conic_axis(u2, R, k):
    """Conic sag of one axis from the QUADRIC ROOT
    ``z = (1 - sqrt(1 - (1+k) c^2 u2)) / ((1+k) c)`` -- algebraically the same
    surface as the library's ``c u2 / (1 + sqrt(...))``, a different
    expression, so the two agree only if both are right."""
    u2 = np.asarray(u2, float)
    if R is None or not np.isfinite(R):
        return np.zeros_like(u2)
    c, kp = 1.0 / R, 1.0 + k
    if abs(kp) < 1e-12:
        return 0.5 * c * u2
    rad = np.where(1.0 - kp * c * c * u2 < 0.0, np.nan,
                   1.0 - kp * c * c * u2)
    return (1.0 - np.sqrt(rad)) / (kp * c)


def _conic_axis_d(u, R, k):
    u = np.asarray(u, float)
    if R is None or not np.isfinite(R):
        return np.zeros_like(u)
    c, kp = 1.0 / R, 1.0 + k
    if abs(kp) < 1e-12:
        return c * u
    rad = np.where(1.0 - kp * c * c * u * u < 0.0, np.nan,
                   1.0 - kp * c * c * u * u)
    return c * u / np.sqrt(rad)


def _sag(x, y, s):
    dcx, dcy = s.get('dec') or (0.0, 0.0)
    x = np.asarray(x, float) - dcx
    y = np.asarray(y, float) - dcy
    if s.get('xy'):
        z = _conic_axis(x * x + y * y, s['Rx'], s.get('kx', 0.0))
        for (i, j), c in s['xy'].items():
            z = z + c * x ** i * y ** j
        return z
    if s.get('Ry') is None:
        r2 = x * x + y * y
        z = _conic_axis(r2, s['Rx'], s.get('kx', 0.0))
        for p, a in (s.get('ax') or {}).items():
            z = z + a * r2 ** (p // 2)
        return z
    return (_conic_axis(x * x, s['Rx'], s.get('kx', 0.0))
            + _conic_axis(y * y, s['Ry'], s.get('ky', 0.0)))


def _sag_grad(x, y, s):
    dcx, dcy = s.get('dec') or (0.0, 0.0)
    x = np.asarray(x, float) - dcx
    y = np.asarray(y, float) - dcy
    if s.get('xy'):
        h = np.hypot(x, y)
        hs = np.where(h > 0, h, 1.0)
        d = _conic_axis_d(h, s['Rx'], s.get('kx', 0.0))
        gx = np.where(h > 0, d * x / hs, 0.0)
        gy = np.where(h > 0, d * y / hs, 0.0)
        for (i, j), c in s['xy'].items():
            if i:
                gx = gx + c * i * x ** (i - 1) * y ** j
            if j:
                gy = gy + c * j * x ** i * y ** (j - 1)
        return gx, gy
    if s.get('Ry') is None:
        h = np.hypot(x, y)
        hs = np.where(h > 0, h, 1.0)
        d = _conic_axis_d(h, s['Rx'], s.get('kx', 0.0))
        for p, a in (s.get('ax') or {}).items():
            d = d + a * p * h ** (p - 1)
        return (np.where(h > 0, d * x / hs, 0.0),
                np.where(h > 0, d * y / hs, 0.0))
    return (_conic_axis_d(x, s['Rx'], s.get('kx', 0.0)),
            _conic_axis_d(y, s['Ry'], s.get('ky', 0.0)))


def _trace3d(x, y, L, M, N, surfaces, iters=80):
    """Damped Newton on ``F(t) = (z + tN) - zv - sag(x + tL, y + tM)``, then
    vector Snell / mirror reflection.  Returns the state ON the last surface
    with the accumulated optical path."""
    x = np.array(x, float, copy=True)
    y = np.array(y, float, copy=True)
    z = np.zeros_like(x)
    L = np.array(L, float, copy=True)
    M = np.array(M, float, copy=True)
    N = np.array(N, float, copy=True)
    opl = np.zeros_like(x)
    n_cur = 1.0
    for s in surfaces:
        zv = s['zv']
        t = (zv - z) / N
        for _ in range(iters):
            F = (z + t * N - zv) - _sag(x + t * L, y + t * M, s)
            gx, gy = _sag_grad(x + t * L, y + t * M, s)
            dF = N - gx * L - gy * M
            t = t - F / np.where(np.abs(dF) < 1e-300, 1e-300, dF)
            if np.nanmax(np.abs(F)) < 1e-18:
                break
        x, y, z = x + t * L, y + t * M, z + t * N
        opl = opl + n_cur * t
        gx, gy = _sag_grad(x, y, s)
        nx, ny, nz = -gx, -gy, np.ones_like(gx)
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx, ny, nz = nx / nn, ny / nn, nz / nn
        ci = L * nx + M * ny + N * nz
        sg = np.where(ci > 0.0, -1.0, 1.0)
        nx, ny, nz = nx * sg, ny * sg, nz * sg
        ci = L * nx + M * ny + N * nz
        if s.get('mirror'):
            L, M, N = L - 2 * ci * nx, M - 2 * ci * ny, N - 2 * ci * nz
        else:
            mu = n_cur / float(s['n_after'])
            kk = 1.0 - mu * mu * (1.0 - ci * ci)
            f = mu * ci + np.sqrt(kk)
            L, M, N = mu * L - f * nx, mu * M - f * ny, mu * N - f * nz
            n_cur = float(s['n_after'])
    return dict(x=x, y=y, z=z, L=L, M=M, N=N, opl=opl, n=n_cur)


def _to_vertex(st, n_exit, zv_last):
    """Straight-line transfer of each ray to ``z = zv_last``; the sign of
    ``N`` -- which a mirror flips -- is carried by the arithmetic."""
    t = -(st['z'] - zv_last) / st['N']
    return dict(x=st['x'] + st['L'] * t, y=st['y'] + st['M'] * t,
                L=st['L'], M=st['M'], N=st['N'], t=t,
                opl=st['opl'] + n_exit * t)


def _asm(E, dx, lam, z, pad=2):
    """Band-limited angular spectrum (Matsushima & Shimobaba 2009)."""
    ny, nx = E.shape
    py, px = ny * pad, nx * pad
    buf = np.zeros((py, px), complex)
    oy, ox = (py - ny) // 2, (px - nx) // 2
    buf[oy:oy + ny, ox:ox + nx] = E
    fx = np.fft.fftfreq(px, dx)
    FX, FY = np.meshgrid(fx, np.fft.fftfreq(py, dx))
    arg = 1.0 / lam ** 2 - FX ** 2 - FY ** 2
    prop = arg > 0.0
    kz = 2.0 * np.pi * np.sqrt(np.where(prop, arg, 0.0))
    Lx = px * dx
    with np.errstate(divide='ignore', invalid='ignore'):
        lim = 1.0 / (lam * np.sqrt((2.0 * abs(z) / Lx) ** 2 + 1.0))
    H = np.where((np.abs(FX) <= lim) & (np.abs(FY) <= lim) & prop,
                 np.exp(1j * kz * z), 0.0)
    return np.fft.ifft2(np.fft.fft2(buf) * H)[oy:oy + ny, ox:ox + nx]


def _fid(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    den = float(np.vdot(a, a).real * np.vdot(b, b).real)
    return float(abs(np.vdot(b, a)) ** 2 / den) if den > 0 else 0.0


# ===========================================================================
# Fixtures: ONE optic with only the LAST surface varied.
# ===========================================================================
def _presc(last=None, *, R1=_R1, R2=_R2, t=_T, semi=_SEMI, glass=None,
           mirror=False, exit_glass='air'):
    _register_glass()
    glass = glass or _GLASS
    if mirror:
        s = {'radius': R2, 'conic': 0.0, 'thickness': 0.0,
             'glass_before': 'air', 'glass_after': 'MIRROR',
             'semi_diameter': semi}
        s.update(last or {})
        return {'name': 'vb12b', 'aperture_diameter': 2 * semi,
                'surfaces': [s], 'thicknesses': [0.0], 'stop_index': 0}
    s0 = {'radius': R1, 'conic': 0.0, 'thickness': t, 'glass_before': 'air',
          'glass_after': glass, 'semi_diameter': semi}
    s1 = {'radius': R2, 'conic': 0.0, 'thickness': 0.0,
          'glass_before': glass, 'glass_after': exit_glass,
          'semi_diameter': semi}
    s1.update(last or {})
    return {'name': 'vb12b', 'aperture_diameter': 2 * semi,
            'surfaces': [s0, s1], 'thicknesses': [t], 'stop_index': 0}


def _osurfs(presc, n_glass=_GLASS_N, n_exit=1.0, mirror=False):
    ss = presc['surfaces']

    def spec(d, zv, na, mir=False):
        return dict(zv=zv, Rx=float(d.get('radius', np.inf)),
                    kx=float(d.get('conic', 0.0) or 0.0),
                    ax=d.get('aspheric_coeffs') or None,
                    Ry=(None if d.get('radius_y') is None
                        else float(d['radius_y'])),
                    ky=float(d.get('conic_y', 0.0) or 0.0),
                    xy=d.get('xy_coeffs') or None,
                    dec=(tuple(float(v) for v in d['decenter'])
                         if d.get('decenter') else None),
                    n_after=float(na), mirror=bool(mir))
    if mirror:
        return [spec(ss[0], 0.0, 1.0, True)]
    return [spec(ss[0], 0.0, n_glass), spec(ss[1], _T, n_exit)]


_FIXTURES = {
    'conic': dict(last=None, rot_sym=True,
                  note='CONIC last surface -- the class the in-line copy got '
                       'RIGHT, so the builder scores no field here'),
    'bicon': dict(last={'radius_y': -9.5e-3}, rot_sym=False,
                  note='BICONIC last surface (Rx = -6.0, Ry = -9.5 mm)'),
    'freeform': dict(last={'freeform_type': 'xy_polynomial',
                           'xy_coeffs': {(2, 0): 3.0e1, (0, 2): -1.8e1,
                                         (4, 0): 6.0e7},
                           'norm_x': 1.0, 'norm_y': 1.0},
                     rot_sym=False,
                     note='FREEFORM (XY polynomial) last surface'),
    'fieldframe': dict(last={'decenter': (5.0e-5, -3.5e-5)}, rot_sym=False,
                       note='FIELD-FRAME decentred last surface'),
}


def _grid(N=_N, dx=_DX):
    a = (np.arange(N) - N / 2) * dx
    return np.meshgrid(a, a)


def _E_in(N=_N, dx=_DX, w0=_W0):
    X, Y = _grid(N, dx)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def _exit_rays(osurfs, a, b, n_exit=1.0):
    z = np.zeros_like(np.asarray(a, float))
    st = _trace3d(a, b, z, z, np.ones_like(z), osurfs)
    return st, _to_vertex(st, n_exit, osurfs[-1]['zv'])


def _best_focus(osurfs, semi=_SEMI, w0=_W0):
    """The traced best focus: the z minimising the intensity-weighted
    transverse variance of the exit bundle.  Derived, never pinned."""
    s1 = np.linspace(-semi * 0.995, semi * 0.995, 81)
    A, B = np.meshgrid(s1, s1)
    keep = np.hypot(A, B) <= semi
    A, B = A[keep], B[keep]
    _st, v = _exit_rays(osurfs, A, B)
    ux, uy = v['L'] / v['N'], v['M'] / v['N']
    w = np.exp(-2.0 * (A ** 2 + B ** 2) / w0 ** 2)
    w = w / w.sum()

    def lsq(p, u):
        pc, uc = p - (w * p).sum(), u - (w * u).sum()
        den = (w * uc * uc).sum()
        return -(w * pc * uc).sum() / den if den > 0 else 0.0

    f0 = 0.5 * (lsq(v['x'], ux) + lsq(v['y'], uy))
    zs = np.linspace(0.75 * f0, 1.25 * f0, 1201)
    xz = v['x'][None, :] + ux[None, :] * zs[:, None]
    yz = v['y'][None, :] + uy[None, :] * zs[:, None]
    cx, cy = (w * xz).sum(1), (w * yz).sum(1)
    var = ((w * (xz - cx[:, None]) ** 2).sum(1)
           + (w * (yz - cy[:, None]) ** 2).sum(1))
    return float(zs[int(np.argmin(var))])


def _exit_field(osurfs, rot_sym, refine=2, n_src=None, semi=_SEMI,
                w0=_W0, N=_N, dx=_DX, lam=_LAM):
    """Geometrical-optics boundary field on the exit-VERTEX plane:
    ``E = E_in / sqrt|det d(xv,yv)/d(a,b)| * exp(i k opl)``."""
    a = (np.arange(N * refine) - N * refine / 2) * (dx / refine)
    X, Y = np.meshgrid(a, a)
    k = 2.0 * np.pi / lam
    rr = np.hypot(X, Y)
    n_src = int(n_src or 4 * N)
    if rot_sym:
        h = np.linspace(0.0, semi, n_src)
        _st, v = _exit_rays(osurfs, h, np.zeros_like(h))
        xv, opl = v['x'], v['opl']
        dxe = np.gradient(xv, h, edge_order=2)
        with np.errstate(divide='ignore', invalid='ignore'):
            amp = np.exp(-(h / w0) ** 2) * np.sqrt(np.abs(h / (xv * dxe)))
        amp[0] = amp[1]
        inside = (rr >= xv.min()) & (rr <= xv.max())
        E = np.where(inside.ravel(),
                     np.interp(rr.ravel(), xv, amp)
                     * np.exp(1j * k * np.interp(rr.ravel(), xv, opl)), 0.0)
        return E.reshape(X.shape)
    from scipy.interpolate import CloughTocher2DInterpolator
    m = max(int(n_src // 3), 129)
    s1 = np.linspace(-semi * 1.10, semi * 1.10, m)
    A0, B0 = np.meshgrid(s1, s1)
    _st, v = _exit_rays(osurfs, A0, B0)
    da = s1[1] - s1[0]
    jxa, jxb = np.gradient(v['x'], da, da, edge_order=2)
    jya, jyb = np.gradient(v['y'], da, da, edge_order=2)
    det = jxa * jyb - jxb * jya
    amp = np.exp(-(A0 ** 2 + B0 ** 2) / w0 ** 2) / np.sqrt(np.abs(det))
    ph = k * v['opl']
    ok = np.isfinite(amp) & np.isfinite(ph)
    itp = CloughTocher2DInterpolator(
        np.column_stack([v['x'][ok].ravel(), v['y'][ok].ravel()]),
        np.column_stack([amp[ok].ravel(), ph[ok].ravel(),
                         np.hypot(A0, B0)[ok].ravel()]), fill_value=np.nan)
    got = itp(np.column_stack([X.ravel(), Y.ravel()]))
    A, P, Rr = got[:, 0], got[:, 1], got[:, 2]
    good = np.isfinite(A) & np.isfinite(P) & np.isfinite(Rr) & (Rr <= semi)
    return np.where(good, A * np.exp(1j * P), 0.0).reshape(X.shape)


def _oracle(osurfs, rot_sym, z, refine=2, n_src=None, **kw):
    E = _exit_field(osurfs, rot_sym, refine=refine, n_src=n_src, **kw)
    out = E if z == 0.0 else _asm(E, _DX / refine, _LAM, z)
    return out[::refine, ::refine].copy()


def _gbd(presc, z, **extra):
    kw = dict(_FRAME)
    kw.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_gbd(
            _E_in(), prescription=presc, wavelength=_LAM, dx=_DX,
            output_plane_distance=float(z),
            mem_budget_mb=_MEM_BUDGET_MB, **kw))


class _Pin:
    """Force BOTH differential primitives onto one reference plane, through
    the shipped public keyword -- never through a copy of deleted code.

    ``'surface'`` is the local branch's repair undone (pre-v5.22 on a curved
    base).  ``'exit_vertex'`` forced onto the WORLD branch is the mirrored
    defect: that branch starts its own leg at the last-surface INTERSECTION,
    so the sag is then counted twice.
    """

    def __init__(self, reference):
        self.reference = reference

    def __enter__(self):
        self.saved = (D.ray_transfer_jacobian,
                      D.ray_transfer_jacobian_analytic)
        ref = self.reference

        def pin(base):
            def wrapped(*a, **kw):
                kw['reference'] = ref
                return base(*a, **kw)
            return wrapped
        D.ray_transfer_jacobian = pin(self.saved[0])
        D.ray_transfer_jacobian_analytic = pin(self.saved[1])
        return self

    def __exit__(self, *exc):
        (D.ray_transfer_jacobian,
         D.ray_transfer_jacobian_analytic) = self.saved
        return False


class _DoubleProject:
    """Apply the exit-vertex projection TWICE.

    This is the deleted ``-sag`` fold RESTORED, in the arithmetic that matters
    rather than in its original spelling: the fold contributed
    ``-sag_inline * sec`` of optical path and ``-sag_inline * u`` of height on
    top of the projection, and on a CONIC last surface exiting forwards into
    air ``sag_inline == sag_true`` and ``n_exit * sign(N) == 1``, so a second
    projection is the same operator.  Reaching it this way means the check
    cannot be satisfied by renaming a variable.
    """

    def __enter__(self):
        self.saved = D._project_to_exit_vertex_plane

        def twice(transfer, *a, **kw):
            out = self.saved(transfer, *a, **kw)
            return out if out is transfer else self.saved(out, *a, **kw)
        D._project_to_exit_vertex_plane = twice
        return self

    def __exit__(self, *exc):
        D._project_to_exit_vertex_plane = self.saved
        return False


def _oracle_is_sound(osurfs, presc, semi=_SEMI):
    """Assert MY tracer's exit-vertex state against the library's OWN vertex
    operator before the oracle built on it is used.  Returns the agreement and
    the slope-vs-direction-cosine contrast that shows the comparison is real.
    """
    import copy as _c

    from lumenairy.raytrace import surfaces_from_prescription
    surfs = list(surfaces_from_prescription(presc))
    surfs[-1] = _c.copy(surfs[-1])
    surfs[-1].thickness = 0.0
    n_r, n_az = 13, 8
    r = np.linspace(semi / (2 * n_r), semi * 0.98, n_r)
    az = (np.arange(n_az) + 0.5) * (np.pi / n_az)
    RR, AA = np.meshgrid(r, az, indexing='ij')
    h, y = (RR * np.cos(AA)).ravel(), (RR * np.sin(AA)).ravel()
    z = np.zeros_like(h)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        tr = la.raytrace.trace(la.raytrace.RayBundle(
            x=h.copy(), y=y.copy(), z=z.copy(), L=z.copy(), M=z.copy(),
            N=np.ones_like(h), wavelength=_LAM, alive=np.ones(h.size, bool),
            opd=z.copy()), surfs, _LAM)
        ev = tr.at_exit_vertex()
    st, v = _exit_rays(osurfs, h, y)
    ok = np.asarray(ev.alive, bool)
    dx = float(np.nanmax(np.abs((v['x'] - ev.x)[ok])))
    do = float(np.nanmax(np.abs((v['opl'] - ev.opd)[ok])))
    trap = float(np.nanmax(np.abs((st['L'] / st['N'] - st['L'])[ok])))
    sag = float(np.nanmax(np.abs((st['z'] - osurfs[-1]['zv'])[ok])))
    return dx, do, trap, sag


# ===========================================================================
# 1.  A CONIC last surface -- the class the in-line copy got right, and the
#     one a restored fold destroys.
# ===========================================================================
def test_a_conic_last_surface_reproduces_a_diffraction_oracle_and_a_double_vertex_correction_does_not():
    """DECISION: on a CONIC last surface -- where the deleted in-line copy
    computed the sag EXACTLY, so WP-B12b's own suite scores no field at all --
    the shipped GBD field reproduces an independent diffraction oracle, and
    BOTH plausible wrong versions of the repair fail it:

    * the projection applied TWICE (the deleted ``-sag`` fold put back on top
      of the projection -- the same operator, since on a conic transmissive
      exit in air ``sag_inline == sag_true`` and ``n_exit * sign(N) == 1``);
    * the projection not applied at all (``reference='surface'``).

    WP-B12b's suite catches the first only through a SOURCE-token check for
    the names ``_Rl`` / ``_kl`` / ``_cl``, so any respelling of the same
    double correction passes it; this is the numerical claim.

    Bars.  ``> 0.99`` for "reproduces" and ``< 0.95`` for "does not", with the
    whole distance between them empty: the defect either arm injects is a
    ``k * sag`` pupil-phase ramp reaching the last surface's own sag, which is
    RE-MEASURED below and asserted to exceed one wave (2.58 waves here on
    2026-09-19, both builds), and no global piston absorbs a ramp.  The
    oracle's own floor is asserted first, by halving its ray quadrature.
    """
    presc = _presc()
    osurfs = _osurfs(presc)
    dx_g, do_g, trap, sag = _oracle_is_sound(osurfs, presc)
    assert dx_g < 1e-15 and do_g < 1e-14, (
        f'my oracle disagrees with at_exit_vertex by {dx_g:.3e} m of height '
        f'and {do_g:.3e} m of path; it is not fit to score anything')
    assert trap > 1e4 * max(dx_g, 1e-30), (
        f'premise: the slope/direction-cosine contrast is only {trap:.3e}, so '
        f'the agreement above is not a real comparison')
    sag_w = sag / _LAM
    assert sag_w > 1.0, f'premise: the last-surface sag is {sag_w:.3f} waves'

    z = _best_focus(osurfs)
    hi = _oracle(osurfs, True, z, n_src=4 * _N)
    lo = _oracle(osurfs, True, z, n_src=2 * _N)
    conv = 1.0 - _fid(hi, lo)
    assert conv < 1e-6, f'oracle not converged in its quadrature: {conv:.2e}'

    now = _fid(hi, _gbd(presc, z))
    with _DoubleProject():
        dbl = _fid(hi, _gbd(presc, z))
    with _Pin('surface'):
        none_ = _fid(hi, _gbd(presc, z))
    assert now > 0.99, (
        f'the shipped conic-last-surface field scores {now:.6f} against my '
        f'oracle (quadrature floor {conv:.2e}, sag {sag_w:.2f} waves)')
    assert dbl < 0.95, (
        f'the DOUBLE vertex correction -- the deleted -sag fold restored -- '
        f'scores {dbl:.6f}, so this fixture does not separate it from the '
        f'shipped {now:.6f}')
    assert none_ < 0.95, (
        f'the un-projected arm scores {none_:.6f}; the reference-plane '
        f'keyword is inert on this build')


# ===========================================================================
# 2.  The three classes WP-B12b could only report as movement.
# ===========================================================================
@pytest.mark.parametrize('name', ['bicon', 'freeform', 'fieldframe'])
def test_a_non_rotationally_symmetric_last_surface_is_scored_against_a_three_d_oracle(name):
    """DECISION: on a BICONIC, an XY-polynomial FREEFORM and a FIELD-FRAME
    decentred last surface, the repaired GBD field reproduces an independent
    diffraction oracle and the pre-repair arm does not.

    WP-B12b reports these three as field MOVEMENT only -- its oracle is a
    rotationally-symmetric ring sum and cannot represent them (its own open
    item 4).  The oracle here is a fully 3-D trace whose exit-vertex state is
    checked against ``TraceResult.at_exit_vertex()`` first, propagated by a
    band-limited angular spectrum, so these are accuracy readings.

    Bars: the same empty corridor as the conic test, ``> 0.99`` against
    ``< 0.95``.  The defect the pre-repair arm carries is the whole
    last-surface sag, which is RE-MEASURED here from my own trace and asserted
    to exceed one wave before either bar is used (2.55 / 3.08 / 5.01 waves for
    the biconic, the freeform and the field-frame fixtures on 2026-09-19, both
    builds); the equivalent readings on the probe package's larger-aperture
    fixtures are 1.98 / 2.08 / 3.39 waves
    (``validation/probe_verify_b12b/probe_w1_mechanism.py``).  The oracle's own
    quadrature convergence is asserted too.
    """
    spec = _FIXTURES[name]
    presc = _presc(spec['last'])
    osurfs = _osurfs(presc)
    dx_g, do_g, trap, sag = _oracle_is_sound(osurfs, presc)
    assert dx_g < 1e-15 and do_g < 1e-14, (
        f'{name}: my oracle disagrees with at_exit_vertex by {dx_g:.3e} m / '
        f'{do_g:.3e} m; it is not fit to score anything')
    assert trap > 1e4 * max(dx_g, 1e-30), (
        f'{name}: premise -- the slope/cosine contrast is only {trap:.3e}')
    assert sag / _LAM > 1.0, f'{name}: premise -- sag {sag / _LAM:.3f} waves'
    assert not D._last_surface_sag_vanishes(
        list(la.raytrace.surfaces_from_prescription(presc))[-1]), (
        f'{name}: premise -- the projection must be active on this surface')

    z = _best_focus(osurfs)
    hi = _oracle(osurfs, False, z, n_src=4 * _N)
    lo = _oracle(osurfs, False, z, n_src=2 * _N)
    conv = 1.0 - _fid(hi, lo)
    assert conv < 1e-4, (
        f'{name}: oracle not converged in its quadrature: {conv:.2e}')

    now = _fid(hi, _gbd(presc, z))
    with _Pin('surface'):
        old = _fid(hi, _gbd(presc, z))
    assert now > 0.99, (
        f'{name}: the repaired field scores {now:.6f} against my 3-D oracle '
        f'(quadrature floor {conv:.2e}, sag {sag / _LAM:.2f} waves)')
    assert old < 0.95, (
        f'{name}: the pre-repair arm scores {old:.6f}, so this fixture does '
        f'not separate the two reference planes on this build')


# ===========================================================================
# 3.  The world branch's plane is the RIGHT one, not merely the one it asks
#     for.
# ===========================================================================
def test_the_world_branch_keeps_the_surface_plane_because_the_exit_vertex_one_double_counts():
    """DECISION: the ``world_output_plane`` branch is CORRECT to keep
    ``reference='surface'``.

    WP-B12b pins which keyword each branch asks for; it never scores the world
    branch, so a build that moved it to ``'exit_vertex'`` would pass its whole
    suite on the strength of one behavioural recording.  That branch measures
    its own leg from the last-surface INTERSECTION (``t = -p_l[2]/d_l[2]`` on
    the world trace), so moving the primitive under it subtracts the sag in
    ``dt.opd`` and never adds it back -- the repaired defect, mirrored.

    Measured here: the world branch reconstructed on an explicit world plane
    at the fixture's own traced best focus, scored against my oracle, under
    the shipped plane and under the forced ``'exit_vertex'`` one.
    """
    presc = _presc()
    osurfs = _osurfs(presc)
    dx_g, _do, _t, sag = _oracle_is_sound(osurfs, presc)
    assert dx_g < 1e-15, 'premise: my oracle must agree with at_exit_vertex'
    z = _best_focus(osurfs)
    orc = _oracle(osurfs, True, z, n_src=4 * _N)
    # The world plane is named in WORLD coordinates, where the last vertex
    # sits at the accumulated thickness -- taken from the library's own world
    # surfaces rather than assumed, so the two branches are compared on the
    # SAME physical plane.
    from lumenairy.raytrace.world import world_surfaces_from_prescription
    _last_w = world_surfaces_from_prescription(presc)[-1]
    assert abs(float(_last_w.world_origin[2]) - _T) < 1e-15, (
        'premise: the last vertex must sit at the accumulated thickness in '
        f'world coordinates; it is at {float(_last_w.world_origin[2]):.6e}')
    plane = (_last_w.world_origin + float(z) * _last_w.world_R[:, 2],
             _last_w.world_R.copy())

    def world(**pin):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            b = G.decompose_field_to_beamlets(_E_in(), _DX, wavelength=_LAM,
                                              **_FRAME)
            w = G.apply_prescription_persurface_to_beamlets(
                b, presc, _LAM, world_output_plane=plane)
            return np.asarray(G.reconstruct_field_from_beamlets(
                w, Ny=_N, Nx=_N, dx=_DX, wavelength=_LAM, window=5.0))

    shipped = _fid(orc, world())
    local = _fid(orc, _gbd(presc, z))
    with _Pin('exit_vertex'):
        moved = _fid(orc, world())
    assert abs(shipped - local) < 1e-5, (
        f'premise: on an unfolded system the world branch must reduce to the '
        f'local one; they read {shipped:.8f} and {local:.8f}')
    assert shipped > 0.99, (
        f'the shipped world branch scores {shipped:.6f} against my oracle; '
        f'this fixture cannot decide the question')
    assert moved < 0.95, (
        f"forcing the world branch onto 'exit_vertex' scores {moved:.6f} "
        f'against the shipped {shipped:.6f}: the sag ({sag / _LAM:.2f} waves) '
        f'is NOT being double-counted there, so the branch is free to move '
        f'and the code comment that says otherwise is wrong')


# ===========================================================================
# 4.  The two open items, pinned so the edits that close them are forced to
#     re-derive these numbers.
# ===========================================================================
def test_a_mirror_terminated_local_branch_is_refused_or_carries_the_direction():
    """DECISION: a mirror-terminated prescription through the LOCAL branch
    must EITHER be refused, OR return an exit direction whose ``z`` sign
    matches the traced one.  It is now REFUSED.

    HISTORY, because the shape of the gate matters.  Until 2026-09-19 this id
    was ``xfail(raises=AssertionError, strict=True)`` with the note "this
    xfail turns RED the day either remedy lands".  It would NOT have: both
    ends of the id raise ``AssertionError`` -- the served arm through its
    final assertion, the refused arm through the ``except`` clause that
    re-raises as one -- so a landing remedy would have left it quietly
    xfailed, and ``strict=True`` only catches an XPASS.  A gate that cannot
    change state is not a gate; the xfail is removed and the refusal is
    asserted directly.

    WHAT IS ASSERTED.  WP-B12b repaired the SIGN of the sag term on this
    class (``_exit_direction_sign``, 13.15 waves on this fixture) and left the
    branch's own direction convention alone, recording it as open item 2 with
    the recommendation "should probably be refused with a message naming
    ``world_output_plane``".  Measured here as a PREMISE first: that
    recommendation has no remedy to name.  The ``world_output_plane`` branch
    REFUSES a curved terminating mirror outright
    (``_unfolded_equivalent_surfaces``: "curved (powered) fold mirrors are not
    yet supported"), so on exactly the class where the local branch is wrong
    there is nowhere to send the caller -- and the shipped message must not
    pretend otherwise.  The refusal itself, its two-sided control and the
    message's content are pinned in ``tests/unit/test_wp_b12b_round2.py``;
    what this id keeps is the DECISION on the verifier's own fixture, against
    the verifier's own 3-D trace.
    """
    presc = _presc(mirror=True, R2=-15.0e-3, semi=0.40e-3)
    osurfs = _osurfs(presc, mirror=True)
    n_r = 9
    h = np.linspace(0.40e-3 / (2 * n_r), 0.40e-3 * 0.98, n_r)
    y = np.zeros_like(h)
    st, _v = _exit_rays(osurfs, h, y)
    true_sign = float(np.sign(np.median(st['N'])))
    assert true_sign == -1.0, 'premise: one mirror must reverse the ray'

    b = G.decompose_field_to_beamlets(_E_in(), _DX, wavelength=_LAM,
                                      **_FRAME)
    # PREMISE: the branch WP-B12b's open item points at cannot serve this
    # class either.  Recorded as its own expectation so the day it CAN, this
    # test says so.
    with pytest.raises(NotImplementedError) as ei:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            G.apply_prescription_persurface_to_beamlets(
                b, presc, _LAM,
                world_output_plane=(np.array([0.0, 0.0, -7.5e-3]), np.eye(3)))
    assert 'mirror' in str(ei.value).lower(), (
        f'premise: the world branch was expected to refuse a curved '
        f'terminating mirror; it raised {ei.value}')

    refused = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = G.apply_prescription_persurface_to_beamlets(
                b, presc, _LAM, z_image=7.5e-3)
    except (NotImplementedError, ValueError) as e:
        refused = e
    if refused is None:                                 # pragma: no cover
        got = float(np.sign(np.median(np.asarray(r.directions)[:, 2])))
        assert got == true_sign, (
            f'the local branch returned a direction with z sign {got:+.0f} '
            f'for a prescription whose light travels toward '
            f'{true_sign:+.0f}, and did not refuse')
        return
    msg = str(refused)
    assert 'MIRROR' in msg, (
        f'the refusal must name the class it refuses; it reads {msg[:160]!r}')
    # and it must NOT send the caller to the branch that refuses this same
    # class: the premise above measured that ``world_output_plane`` raises on
    # a CURVED terminating fold, so a message offering it unconditionally
    # would be a dead end (VERIFY-WP-B12b D-4).
    low = msg.lower()
    assert 'world_output_plane' not in low or (
        'curved' in low and 'flat' in low), (
        f'the refusal names world_output_plane without distinguishing the '
        f'CURVED case (which that branch refuses) from the FLAT one (which '
        f'it serves): {msg[:400]!r}')


def test_an_immersed_exit_is_refused_or_served_with_a_vacuum_image_leg():
    """DECISION: a prescription whose last surface exits into GLASS is either
    REFUSED, or served with an image leg that is missing its exit index -- and
    if it is served, the size of the omission is measured here rather than
    described.

    ``_project_to_exit_vertex_plane`` resolves ``n_exit`` and weights the sag
    term with it; GBD's own leg is ``exp(i k0 * z_image * sec)``, index-free.
    In a medium of index ``n`` that omits ``|n - 1| * z_image * sec`` of
    optical path.  WP-B12b records this as open item 1 and does not measure
    it; the sibling ``propagators.fga`` sites are checked here for a guard, so
    the report can say whether GBD is the only remaining one.

    Two-sided: the same measurement on an AIR-exit control must read zero.
    """
    from lumenairy.raytrace import surfaces_from_prescription
    _register_glass()
    from lumenairy import glass as _g
    _g.GLASS_REGISTRY['VB12B-T172'] = lambda wl: 1.72
    z_img = 2.0e-3
    k0 = 2.0 * np.pi / _LAM
    out = {}
    for label, exit_glass, n_exit in (('immersed', 'VB12B-T172', 1.72),
                                      ('air control', 'air', 1.0)):
        presc = _presc(exit_glass=exit_glass)
        surfs = list(surfaces_from_prescription(presc))
        assert abs(float(la.raytrace.exit_vertex.resolve_exit_index(
            surfs, _LAM, fn_name='vb12b')) - n_exit) < 1e-12, (
            f'premise: {label} must resolve n_exit = {n_exit}')
        osurfs = _osurfs(presc, n_exit=n_exit)
        n_r = 11
        h = np.linspace(_SEMI / (2 * n_r), _SEMI * 0.98, n_r)
        y = np.zeros_like(h)
        st, v = _exit_rays(osurfs, h, y, n_exit=n_exit)
        sec = np.sqrt(1.0 + (st['L'] / st['N']) ** 2
                      + (st['M'] / st['N']) ** 2)
        bl = G.BeamletBundle(
            positions=np.stack([h, y, np.zeros_like(h)], -1),
            directions=np.stack([np.zeros_like(h), np.zeros_like(h),
                                 np.ones_like(h)], -1),
            Q=np.full(h.size, -1j * _LAM / (np.pi * (4 * _DX) ** 2),
                      dtype=np.complex128),
            amplitude=np.ones(h.size, dtype=np.complex128),
            waist0=np.full(h.size, 4.0 * _DX))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                a0 = np.asarray(G.apply_prescription_persurface_to_beamlets(
                    bl, presc, _LAM, z_image=0.0).amplitude)
                a1 = np.asarray(G.apply_prescription_persurface_to_beamlets(
                    bl, presc, _LAM, z_image=z_img).amplitude)
        except (NotImplementedError, ValueError) as e:
            out[label] = ('refused', str(e))
            continue
        leg = np.angle(a1 * np.conj(a0))
        res = {}
        for nm, pred in (('indexed', k0 * n_exit * z_img * sec),
                         ('vacuum', k0 * 1.0 * z_img * sec)):
            d = np.angle(np.exp(1j * (leg - pred)))
            p = np.angle(np.mean(np.exp(1j * d)))
            res[nm] = float(np.nanmax(np.abs(
                np.angle(np.exp(1j * (d - p)))))) / (2 * np.pi)
        out[label] = ('served', res,
                      float(np.nanmax((n_exit - 1.0) * z_img * sec) / _LAM))
    if out['immersed'][0] == 'refused':
        assert 'immersed' in out['immersed'][1].lower() \
            or 'exit' in out['immersed'][1].lower(), out['immersed'][1]
        return
    _s, res, missing = out['immersed']
    _sc, resc, missc = out['air control']
    # The bar is DERIVED: the AIR control runs the identical measurement on a
    # prescription with nothing to miss, so its own residual -- the beamlet
    # amplitude's Collins factors, which do not cancel exactly between the two
    # z_image arms -- IS the floor of this method.  Three decades of slack,
    # against a signal that is the whole index of the leg.
    floor = max(resc['vacuum'], 1e-12)
    assert missc == 0.0, (
        f'premise: the AIR control must have nothing to miss; it reads '
        f'{missc:.3e} waves')
    assert missing > 1.0, (
        f'premise: the immersed fixture must omit more than a wave; it omits '
        f'{missing:.3f}')
    assert res['vacuum'] < 1e3 * floor, (
        f"GBD's image leg does not match a VACUUM leg either: residual "
        f"{res['vacuum']:.3e} waves against a {floor:.3e} derived floor")
    assert abs(resc['indexed'] - resc['vacuum']) <= 0.0, (
        'premise: on the AIR control the two predictions are the same '
        'expression, so they must read identically; they read '
        f"{resc['indexed']:.6e} and {resc['vacuum']:.6e}")
    # 100x, with the measured ratio at 895x on both builds (2026-09-19,
    # z_image = 2 mm, n_exit = 1.72): the two predictions differ by the
    # obliquity spread of the omitted (n-1)*z*sec, so a longer leg widens the
    # gap and a shorter one narrows it -- nine decades of slack at this leg.
    assert res['indexed'] > 1e2 * res['vacuum'], (
        f"the indexed prediction is only {res['indexed']:.3e} waves off "
        f"against the vacuum one's {res['vacuum']:.3e} (ratio "
        f"{res['indexed'] / max(res['vacuum'], 1e-30):.0f}x); this fixture "
        f'does not separate them')


# ===========================================================================
# 5.  The named beamlet frame is a cost choice, re-derived at run time.
# ===========================================================================
def test_the_named_beamlet_frame_is_a_cost_choice_not_a_result():
    """INVARIANT: the explicit beamlet frame these tests name reproduces the
    field a DENSER frame produces, so naming it is a cost decision and not a
    thumb on the scale.

    WP-B12b's own file records a single number in a comment ("reproduces the
    dense-frame field at a fidelity of 0.999587").  A number in a comment is
    not a gate: this re-derives the property on the running build by refining
    the frame and asserting BOTH that the two fields agree far better than the
    decision bars those tests use, AND that the decision itself is unchanged
    -- the denser frame still scores above 0.99 against the oracle and the
    pre-repair arm still below 0.95.
    """
    presc = _presc()
    osurfs = _osurfs(presc)
    z = _best_focus(osurfs)
    orc = _oracle(osurfs, True, z, n_src=4 * _N)
    coarse = _gbd(presc, z, **_FRAME)
    dense = _gbd(presc, z, sample_step=3, waist_factor=3.0)
    agree = _fid(coarse, dense)
    f_coarse, f_dense = _fid(orc, coarse), _fid(orc, dense)
    # the bar: the two frames must agree at least two decades of INFIDELITY
    # better than the distance between the decision bars (0.99 and 0.95), so
    # the frame cannot be what carries a decision.
    corridor = 0.99 - 0.95
    assert 1.0 - agree < corridor / 100.0, (
        f'the named frame reproduces the denser one only to {agree:.6f}; the '
        f'decision corridor is {corridor:.2f} wide, so the frame is carrying '
        f'the result')
    assert f_dense > 0.99 and f_coarse > 0.99, (
        f'the decision moves with the frame: coarse {f_coarse:.6f}, dense '
        f'{f_dense:.6f} against the oracle')
    assert abs(f_dense - f_coarse) < corridor / 100.0, (
        f'refining the frame moves the oracle score by '
        f'{abs(f_dense - f_coarse):.2e}, which is not small against the '
        f'{corridor:.2f} decision corridor')


def test_my_oracle_agrees_with_the_library_vertex_operator_on_every_class():
    """CONTROL: the 3-D tracer these tests are scored against reproduces
    ``TraceResult.at_exit_vertex()`` -- a different implementation, on
    direction cosines and the traced ``z`` rather than on a sag kernel and
    unreduced slopes -- on every surface class used here, and the
    slope-versus-direction-cosine contrast shows the comparison is real.

    Stated as its own id so a regression in the oracle is not mistaken for a
    regression in the library.
    """
    rows = []
    for name, spec in _FIXTURES.items():
        presc = _presc(spec['last'])
        dx_g, do_g, trap, sag = _oracle_is_sound(_osurfs(presc), presc)
        rows.append((name, dx_g, do_g, trap, sag))
    for name, dx_g, do_g, trap, sag in rows:
        assert dx_g < 1e-15, f'{name}: height gap {dx_g:.3e} m'
        assert do_g < 1e-14, f'{name}: optical-path gap {do_g:.3e} m'
        assert sag / _LAM > 1.0, f'{name}: sag only {sag / _LAM:.3f} waves'
        assert trap > 1e4 * max(dx_g, 1e-30), (
            f'{name}: slope/cosine contrast {trap:.3e} against a {dx_g:.3e} '
            f'agreement -- not a real comparison')
    assert math.isfinite(rows[0][4])

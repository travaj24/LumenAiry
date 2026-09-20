"""VERIFY-WP-C3 -- an INDEPENDENT harness, written from scratch.

Nothing here imports ``validation/probe_c3_collins_default/clib.py`` or
``probe_wave5_hyg2/hlib.py``: the point of this directory is a second oracle
and a second set of fixtures, so sharing the first one's code would defeat it.

CONVENTION.  ``exp(-i omega t)`` / ``exp(+i k z)`` (CONVENTIONS sec. 7), so

    1/q = 1/R + i lambda/(pi w^2)

which puts ``Im q < 0`` STRICTLY for every finite beam.  Two consequences that
are used below rather than asserted:

* ``q`` and ``q2 = q + z`` both lie in the OPEN lower half-plane, so
  ``arg(q/q2)`` is in ``(-pi, pi)`` and ``np.sqrt`` of that RATIO is the
  continuous branch at every ``z``, waist crossings included.  (The trap is
  ``1/sqrt((1 + z/q)**2)``: squaring first doubles the argument into
  ``(-2pi, 2pi)``, which wraps and costs exactly ``pi`` past the waist.)
* the 2-D isotropic prefactor is the ratio ``q/q2`` itself, which is the
  square of the 1-D one and therefore needs no branch care at all.

ORACLE VALIDATION lives in :func:`validate_oracle` and is run by the probes
before the oracle is used for anything.
"""
from __future__ import annotations

import json
import os
import platform
import sys

import numpy as np

# ---------------------------------------------------------------------------
# anchor / io
# ---------------------------------------------------------------------------


def anchor(tree):
    """Refuse to measure a tree other than ``tree``; print what bound."""
    import lumenairy
    got = os.path.abspath(lumenairy.__file__)
    want = os.path.abspath(tree)
    if not got.lower().startswith(want.lower()):
        raise SystemExit(f'ANCHOR FAIL: lumenairy.__file__={got} not under {want}')
    print(f'[anchor] lumenairy.__file__ = {got}', flush=True)
    return got


def build_tag():
    return {
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'platform': platform.platform(),
        'machine': platform.machine(),
    }


def write_json(path, payload):
    payload = dict(payload)
    payload['_build'] = build_tag()
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=1, default=_jd, sort_keys=True)
    print(f'[write] {path}', flush=True)


def _jd(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return {'re': o.real, 'im': o.imag}
    return str(o)


# ---------------------------------------------------------------------------
# grid / metrics
# ---------------------------------------------------------------------------


def axis(n, d):
    """The package's centred axis, ``(i - n/2) * d``."""
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2.0) * float(d)


def rel_l2(got, ref):
    got = np.asarray(got)
    ref = np.asarray(ref)
    den = float(np.linalg.norm(ref))
    if den == 0.0:
        return float('inf')
    return float(np.linalg.norm(got - ref) / den)


def pitch2(dx):
    if isinstance(dx, (tuple, list)):
        return float(dx[0]), float(dx[1])
    return float(dx), float(dx)


def rad2(R):
    if isinstance(R, (tuple, list)):
        return float(R[0]), float(R[1])
    return float(R), float(R)


def refield(env, R, dx, dy=None, lam=None):
    """envelope -> FIELD: multiply by ``exp(i k u^2/2R)`` per axis."""
    env = np.asarray(env)
    ny, nx = env.shape[-2], env.shape[-1]
    dx, dy = (float(dx), float(dx) if dy is None else float(dy))
    Rx, Ry = rad2(R)
    k = 2.0 * np.pi / lam
    x = axis(nx, dx)
    y = axis(ny, dy)
    ph = np.zeros((ny, nx), dtype=np.float64)
    if np.isfinite(Rx):
        ph = ph + k * (x * x)[None, :] / (2.0 * Rx)
    if np.isfinite(Ry):
        ph = ph + k * (y * y)[:, None] / (2.0 * Ry)
    return env * np.exp(1j * ph)


# ---------------------------------------------------------------------------
# THE ORACLE
# ---------------------------------------------------------------------------


def qpar(R, w, lam):
    """``q`` at the reference plane from ``(R, w)``.  ``R = +/-inf`` allowed."""
    inv = (0.0 if not np.isfinite(R) else 1.0 / float(R)) \
        + 1j * float(lam) / (np.pi * float(w) * float(w))
    return 1.0 / inv


def gauss_env(x, y, w):
    """``exp(-r^2/w^2)`` -- the ENVELOPE the carrier API takes (amplitude
    only; the carrier ``exp(i k r^2/2R)`` is the API's own reference)."""
    r2 = x[None, :] ** 2 + y[:, None] ** 2
    return np.exp(-r2 / (float(w) ** 2)).astype(np.complex128)


def gauss_field_at(x, y, w, R, lam, z):
    """THE ORACLE.  The FIELD of a Gaussian that has ``(w, R)`` at ``z = 0``,
    evaluated a distance ``z`` on, with its ABSOLUTE phase -- the ``exp(i k z)``
    piston and the Gouy term both included, no alignment permitted afterwards.

    Astigmatic when ``w`` or ``R`` is a 2-tuple: the per-axis prefactor is
    ``sqrt(q/q2)`` on the principal branch, which is continuous here for the
    reason in the module docstring.
    """
    lam = float(lam)
    k = 2.0 * np.pi / lam
    wx, wy = (w if isinstance(w, (tuple, list)) else (w, w))
    Rx, Ry = rad2(R)
    qx, qy = qpar(Rx, wx, lam), qpar(Ry, wy, lam)
    qx2, qy2 = qx + z, qy + z
    if (Rx == Ry) and (wx == wy):
        pre = qx / qx2                       # the ratio, once: no branch
    else:
        pre = np.sqrt(qx / qx2) * np.sqrt(qy / qy2)
    ph = k * (x[None, :] ** 2 / (2.0 * qx2) + y[:, None] ** 2 / (2.0 * qy2))
    return (np.exp(1j * k * z) * pre * np.exp(1j * ph)).astype(np.complex128)


def gauss_w_at(w, R, lam, z):
    """``w(z)`` per axis of the same beam (1/e amplitude radius)."""
    wx, wy = (w if isinstance(w, (tuple, list)) else (w, w))
    Rx, Ry = rad2(R)
    out = []
    for ww, RR in ((wx, Rx), (wy, Ry)):
        q2 = qpar(RR, ww, lam) + z
        out.append(float(np.sqrt(float(lam) / (np.pi * (1.0 / q2).imag))))
    return tuple(out)


# ---------------------------------------------------------------------------
# ORACLE VALIDATION -- two independent checks
# ---------------------------------------------------------------------------


def _lap5(u, d):
    """4th-order 5-point Laplacian on the interior of ``u`` (square pitch)."""
    c = (-1.0, 16.0, -30.0, 16.0, -1.0)
    lx = (c[0] * u[2:-2, 0:-4] + c[1] * u[2:-2, 1:-3] + c[2] * u[2:-2, 2:-2]
          + c[3] * u[2:-2, 3:-1] + c[4] * u[2:-2, 4:]) / (12.0 * d * d)
    ly = (c[0] * u[0:-4, 2:-2] + c[1] * u[1:-3, 2:-2] + c[2] * u[2:-2, 2:-2]
          + c[3] * u[3:-1, 2:-2] + c[4] * u[4:, 2:-2]) / (12.0 * d * d)
    return lx + ly


def validate_oracle_pde(w=0.30e-3, R=-40.0e-3, lam=1.064e-6, z=20.0e-3,
                        N=256, span_w=4.0, h=2.0e-6):
    """(i) the oracle satisfies the PARAXIAL HELMHOLTZ equation.

    ``E = exp(i k z) u`` with ``2 i k du/dz + lap_perp u = 0``.  ``du/dz`` by a
    4th-order central difference in ``z``, ``lap_perp`` by a 4th-order stencil
    in ``x, y``.  Returned as ``max|residual| / max|2 i k du/dz|``, which is the
    only scaling that makes it a statement about the PDE rather than about the
    units.
    """
    k = 2.0 * np.pi / lam
    wz = gauss_w_at(w, R, lam, z)[0]
    d = 2.0 * span_w * wz / N
    x = axis(N, d)

    def u_at(zz):
        return gauss_field_at(x, x, w, R, lam, zz) * np.exp(-1j * k * zz)

    um2, um1, u0, up1, up2 = (u_at(z - 2 * h), u_at(z - h), u_at(z),
                              u_at(z + h), u_at(z + 2 * h))
    duz = (um2 - 8.0 * um1 + 8.0 * up1 - up2) / (12.0 * h)
    res = 2j * k * duz[2:-2, 2:-2] + _lap5(u0, d)
    scale = float(np.max(np.abs(2j * k * duz[2:-2, 2:-2])))
    return {'residual_max': float(np.max(np.abs(res))), 'scale': scale,
            'residual_rel': float(np.max(np.abs(res))) / scale,
            'dx': d, 'N': N, 'h': h, 'w_at_z': wz}


def _fresnel_tf(E, d, lam, z, exact=False):
    """One transfer-function step on a grid that fully resolves ``E``.

    ``exact=False`` is the PARAXIAL kernel ``exp(i k z) exp(-i pi lam z f^2)``
    -- the propagator the paraxial oracle is a solution of, so the two agree to
    round-off.  ``exact=True`` is ``exp(i k z sqrt(1 - (lam f)^2))``, which
    differs from the oracle by its own dropped quartic ``k z theta^4/8``; both
    are reported so the gap between them is visible and expected.
    """
    N = E.shape[-1]
    f = np.fft.fftfreq(N, d=d)
    F2 = f[None, :] ** 2 + f[:, None] ** 2
    k = 2.0 * np.pi / lam
    if exact:
        arg = 1.0 - (lam ** 2) * F2
        H = np.where(arg > 0.0, np.exp(1j * k * z * np.sqrt(np.abs(arg))), 0.0)
    else:
        H = np.exp(1j * k * z) * np.exp(-1j * np.pi * lam * z * F2)
    return np.fft.ifft2(np.fft.fft2(E) * H)


def validate_oracle_prop(w=0.30e-3, R=-40.0e-3, lam=1.064e-6, z0=0.0,
                         legs=(5.0e-3, 20.0e-3, 41.0e-3, 60.0e-3),
                         N=2048, span_w=8.0):
    """(ii) the oracle is reproduced by a heavily oversampled transfer-function
    propagation of the oracle's OWN field at ``z0``, over each leg.

    Compared on a CENTRAL sub-window (half the grid) so the periodic wrap of
    the truncated tail is not what is being measured.
    """
    wmax = max(gauss_w_at(w, R, lam, z0 + zz)[0] for zz in legs)
    w0 = gauss_w_at(w, R, lam, z0)[0]
    d = 2.0 * span_w * max(wmax, w0) / N
    x = axis(N, d)
    E0 = gauss_field_at(x, x, w, R, lam, z0)
    s = slice(N // 4, 3 * N // 4)
    rows = []
    for zz in legs:
        ref = gauss_field_at(x, x, w, R, lam, z0 + zz)
        par = _fresnel_tf(E0, d, lam, zz, exact=False)
        exa = _fresnel_tf(E0, d, lam, zz, exact=True)
        th = lam / (np.pi * gauss_w_at(w, R, lam, z0)[0])
        rows.append({
            'leg': zz, 'A': 1.0 + zz / R if np.isfinite(R) else 1.0,
            'rel_l2_paraxial_tf': rel_l2(par[s, s], ref[s, s]),
            'rel_l2_exact_tf': rel_l2(exa[s, s], ref[s, s]),
            'quartic_rad': float(2 * np.pi / lam * abs(zz) * th ** 4 / 8.0)})
    return {'N': N, 'dx': d, 'span_w': span_w, 'rows': rows}


# ---------------------------------------------------------------------------
# truncation floor, by quadrature, refined
# ---------------------------------------------------------------------------


_GL16 = np.polynomial.legendre.leggauss(16)


def _panel_int(f, a, b, panels):
    """Composite 16-point Gauss-Legendre of ``f`` over ``[a, b]``."""
    t, wt = _GL16
    edges = np.linspace(a, b, int(panels) + 1)
    lo, hi = edges[:-1, None], edges[1:, None]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
    return float(np.sum(half * wt[None, :] * f(mid + half * t[None, :])))


def truncation_floor(w, lo, hi, refine=(8, 16, 32, 64, 128)):
    """The relative-L2 floor a SQUARE window ``[lo, hi]^2`` imposes on a
    Gaussian of 1/e AMPLITUDE radius ``w``: ``sqrt(P_outside / P_total)``,
    by composite Gauss-Legendre quadrature refined over panel count.

    The lattice this library uses is ``(i - N/2) dx``, so a window is
    ``[-N dx/2, (N/2 - 1) dx]`` -- NOT symmetric, which is why both ends are
    taken rather than a half-width.

    The OUTSIDE power is integrated DIRECTLY (two tails) rather than as
    ``total - inside``: at four 1/e radii the difference is 1e-19 of the total
    and the subtraction has no significant digits left.  The closed form is
    written with ``erfc`` for the same reason -- ``1 - erf(x)**2`` underflows
    to exactly 0 past ~5.8 radii while the true floor is still 1e-16.
    """
    from math import erfc, sqrt
    lo, hi, w = float(lo), float(hi), float(w)

    def g(u):
        return np.exp(-2.0 * u ** 2 / (w * w))

    # closed form:  frac = erfc(hi sqrt2/w) + erfc(|lo| sqrt2/w) to leading
    # order, and floor = sqrt(frac (1 - frac/4)) exactly for the symmetric case
    e_hi = erfc(abs(hi) * sqrt(2.0) / w)
    e_lo = erfc(abs(lo) * sqrt(2.0) / w)
    frac = 0.5 * (e_hi + e_lo)                 # 1 - inside/total, per axis
    closed = sqrt(max(0.0, frac * (2.0 - frac)))
    L = max(abs(lo), abs(hi)) + 14.0 * w
    levs = []
    for p in refine:
        tot1 = _panel_int(g, -L, L, p)
        out1 = _panel_int(g, hi, L, p) + _panel_int(g, -L, lo, p)
        f = out1 / tot1
        levs.append({'panels': p, 'P_total_1d': tot1, 'P_outside_1d': out1,
                     'frac_1d': f,
                     'floor': float(np.sqrt(max(0.0, f * (2.0 - f))))})
    return {'closed_form': closed, 'lo': lo, 'hi': hi, 'w': w,
            'levels': levs, 'converged': levs[-1]['floor'],
            'ratio_last': (levs[-1]['floor'] / levs[-2]['floor']
                           if len(levs) > 1 and levs[-2]['floor'] else None),
            'quad_vs_closed': (levs[-1]['floor'] / closed if closed else None)}


# ---------------------------------------------------------------------------
# INDEPENDENT REFERENCE for the one-step readout (claim 1c)
# ---------------------------------------------------------------------------


def _sinc_upsample_axis(A, F, axisno):
    """Band-limited (Fourier zero-pad) upsampling of ``A`` by an INTEGER factor
    ``F`` along ``axisno``, on the package's centred lattice.

    The lattice convention is ``u_i = (i - n/2) d``, so the array's sample 0 is
    at ``-n d / 2``, not at 0.  Zero padding interpolates a periodic
    band-limited function about the ARRAY origin, so the phase ramp that
    re-centres the fine lattice on the same physical points is applied in the
    frequency domain: a shift of ``-n d/2`` before and ``+n_f d_f/2`` after,
    which are the same physical offset and therefore cancel exactly.  The
    upshot is the plain "fftshift-free" identity below; it is written out
    because getting it wrong moves the reference by half a coarse pitch.
    """
    A = np.moveaxis(np.asarray(A), axisno, -1)
    n = A.shape[-1]
    nf = n * int(F)
    S = np.fft.fft(A, axis=-1)
    out = np.zeros(A.shape[:-1] + (nf,), dtype=np.complex128)
    h = n // 2
    out[..., :h] = S[..., :h]
    out[..., nf - (n - h):] = S[..., h:]
    if n % 2 == 0:                     # split the Nyquist bin evenly
        out[..., nf - h] = S[..., h] * 0.5
        out[..., h] = S[..., h] * 0.5
    fine = np.fft.ifft(out, axis=-1) * int(F)
    return np.moveaxis(fine, -1, axisno)


def upsampled_fresnel_reference(env, R, lam, dx, dy, z, dx_out, N_out, F,
                                centre_out=(0.0, 0.0)):
    """MY reference for the field ``z`` past a carrier-referenced plane, at the
    readout points -- direct Fresnel quadrature over the exit field SINC-
    UPSAMPLED by ``F`` per axis, evaluated in two separable passes so the fine
    2-D field is never formed.

    The escape the report says a dense sum on the ORIGINAL pitch does not have:
    the ENVELOPE is smooth and band-limited (it is what the chain transports),
    so it may be interpolated; the CARRIER ``exp(i k u^2/2R)`` is analytic and
    is re-applied on the FINE lattice, where it is sampled.  The product --
    which is the thing the one-step readout aliases -- is therefore resolved.

    Pass 1 refines x, applies the x carrier and the x half of the Fresnel
    kernel, contracting to ``N_out`` columns.  What survives is a function of
    the COARSE y that is smooth in y (it is a linear functional of a smooth
    envelope row and the y carrier has NOT been applied yet), so pass 2 refines
    THAT in y, applies the y carrier on the fine y lattice, and contracts.
    """
    env = np.asarray(env, dtype=np.complex128)
    ny, nx = env.shape[-2], env.shape[-1]
    k = 2.0 * np.pi / lam
    Rx, Ry = rad2(R)
    xo = axis(N_out, dx_out) + float(centre_out[0])
    yo = axis(N_out, dx_out) + float(centre_out[1])

    # --- pass 1: x -------------------------------------------------------
    ef = _sinc_upsample_axis(env, F, -1)                 # (ny, nx*F)
    dxf = dx / F
    xf = axis(nx * F, dxf)
    cx = np.ones_like(xf, dtype=np.complex128)
    if np.isfinite(Rx):
        cx = np.exp(1j * k * xf * xf / (2.0 * Rx))
    Kx = np.exp(1j * k * (xf[:, None] - xo[None, :]) ** 2 / (2.0 * z)) \
        * cx[:, None] * dxf                              # (nx*F, N_out)
    M = ef @ Kx                                          # (ny, N_out)

    # --- pass 2: y -------------------------------------------------------
    Mf = _sinc_upsample_axis(M, F, 0)                    # (ny*F, N_out)
    dyf = dy / F
    yf = axis(ny * F, dyf)
    cy = np.ones_like(yf, dtype=np.complex128)
    if np.isfinite(Ry):
        cy = np.exp(1j * k * yf * yf / (2.0 * Ry))
    Ky = np.exp(1j * k * (yf[:, None] - yo[None, :]) ** 2 / (2.0 * z)) \
        * cy[:, None] * dyf                              # (ny*F, N_out)
    out = Ky.T @ Mf                                      # (N_out, N_out)
    return (np.exp(1j * k * z) / (1j * lam * z)) * out


def _two_pass(env, R, lam, dx, dy, z, dx_out, N_out, F, px=0, py=0,
              centre_out=(0.0, 0.0)):
    """The two-pass separable Fresnel quadrature of
    :func:`upsampled_fresnel_reference`, with the optional extra polynomial
    weights ``(x - x0)**px`` and ``(y - y0)**py`` inside the integral.

    ``px = py = 0`` IS that function.  The weights exist so the leading dropped
    term of the EXACT kernel -- ``exp(-i k rho^4/(8 z^3))`` with
    ``rho^4 = (x-x0)^4 + 2 (x-x0)^2 (y-y0)^2 + (y-y0)^4`` -- can be evaluated
    to first order as three separable pieces, which is how this reference
    measures its own paraxiality instead of assuming it.
    """
    env = np.asarray(env, dtype=np.complex128)
    ny, nx = env.shape[-2], env.shape[-1]
    k = 2.0 * np.pi / lam
    Rx, Ry = rad2(R)
    xo = axis(N_out, dx_out) + float(centre_out[0])
    yo = axis(N_out, dx_out) + float(centre_out[1])

    ef = _sinc_upsample_axis(env, F, -1)
    dxf = dx / F
    xf = axis(nx * F, dxf)
    cx = (np.exp(1j * k * xf * xf / (2.0 * Rx)) if np.isfinite(Rx)
          else np.ones_like(xf, dtype=np.complex128))
    dX = xf[:, None] - xo[None, :]
    Kx = np.exp(1j * k * dX ** 2 / (2.0 * z)) * cx[:, None] * dxf
    if px:
        Kx = Kx * dX ** px
    M = ef @ Kx
    del ef, Kx, dX

    Mf = _sinc_upsample_axis(M, F, 0)
    dyf = dy / F
    yf = axis(ny * F, dyf)
    cy = (np.exp(1j * k * yf * yf / (2.0 * Ry)) if np.isfinite(Ry)
          else np.ones_like(yf, dtype=np.complex128))
    dY = yf[:, None] - yo[None, :]
    Ky = np.exp(1j * k * dY ** 2 / (2.0 * z)) * cy[:, None] * dyf
    if py:
        Ky = Ky * dY ** py
    out = Ky.T @ Mf
    return (np.exp(1j * k * z) / (1j * lam * z)) * out


def quartic_correction_size(env, R, lam, dx, dy, z, dx_out, N_out, F):
    """How paraxial is the reference?  The first-order effect of the exact
    kernel's dropped quartic, as a fraction of the Fresnel answer.

    A SMALL number here is what licenses comparing the Fresnel reference with
    the library's own (also paraxial) transports; a large one would mean the
    reference and the library agree only on a paraxial answer, which is a
    different statement and would have to be said.
    """
    k = 2.0 * np.pi / lam
    E0 = _two_pass(env, R, lam, dx, dy, z, dx_out, N_out, F)
    t1 = _two_pass(env, R, lam, dx, dy, z, dx_out, N_out, F, px=4)
    t2 = _two_pass(env, R, lam, dx, dy, z, dx_out, N_out, F, px=2, py=2)
    t3 = _two_pass(env, R, lam, dx, dy, z, dx_out, N_out, F, py=4)
    dE = (-1j * k / (8.0 * z ** 3)) * (t1 + 2.0 * t2 + t3)
    c = (E0.shape[-2] // 2, E0.shape[-1] // 2)
    return {'F': F,
            'rel_l2_correction': float(np.linalg.norm(dE)
                                       / np.linalg.norm(E0)),
            'centre_rel': float(abs(dE[c]) / abs(E0[c])),
            'centre_phase_rad': float(np.angle(1.0 + dE[c] / E0[c])),
            'on_axis_fresnel': float(abs(E0[c]) ** 2),
            'on_axis_plus_quartic': float(abs(E0[c] + dE[c]) ** 2)}

"""Differential ray transfer -- per-beamlet / per-ray ABCD Jacobians.

The reusable primitive underneath the *per-surface* forms of the phase-space
propagators.  Along each base ray of a bundle it returns the ray-transfer
Jacobian ``[[A, B], [C, D]]`` (2x2 blocks) of the real (aberrated) trace, by
central finite differences.  The complex-beam-parameter propagators consume it:

* Gaussian Beam Decomposition (``propagators.gbd``): evolves each beamlet's
  tensor ``Q`` via the generalized Collins relation
  ``Q_out = (C + D Q)(A + B Q)^{-1}`` and its amplitude via
  ``1/sqrt(det(A + B Q))`` -- capturing off-axis astigmatism / higher-order
  aberrations that a single whole-system paraxial ABCD cannot.
* (future) Maslov phase-space: the same differential transfer supplies the
  Hessian propagation.

Phase space is **unreduced** ``(x, y, ux, uy)`` with slopes ``ux = L/N``,
``uy = M/N`` (matching ``propagators.gbd.apply_abcd_to_beamlets``), referenced
to surface **vertex** planes.  The finite-difference Jacobian bakes Snell's law
and the glass indices in automatically (no reduced ``n*u`` bookkeeping), so the
on-axis 2x2 meridional block reproduces
``raytrace.system_abcd_prescription`` to ~1e-8.

Validated in ``tests/unit/test_gbd_feature_complete.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ._conic_core import reflect_mirror, refract_snell
from .trace import _make_bundle, trace


@dataclass
class DifferentialTransfer:
    """Result of :func:`ray_transfer_jacobian`.

    Attributes
    ----------
    jacobian : ndarray
        ``(N_rays, 4, 4)`` composite input->output ray-transfer Jacobian, OR
        ``(N_surfaces, N_rays, 4, 4)`` per-surface local transfers when
        ``per_surface=True``.  Row/col order is ``(x, y, ux, uy)``; the 2x2
        blocks are ``A = J[..., 0:2, 0:2]`` (dx_out/dx_in),
        ``B = J[..., 0:2, 2:4]``, ``C = J[..., 2:4, 0:2]``,
        ``D = J[..., 2:4, 2:4]``.
    x, y, ux, uy : ndarray
        ``(N_rays,)`` base-ray state at the output vertex (unreduced slopes).
    opd : ndarray
        ``(N_rays,)`` base-ray accumulated optical path length [m] at output.
    alive : ndarray of bool
        ``(N_rays,)`` False for rays that vignetted / TIR'd / missed.
    """
    jacobian: np.ndarray
    x: np.ndarray
    y: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    opd: np.ndarray
    alive: np.ndarray


def _slopes_to_dirs(ux, uy):
    inv = 1.0 / np.sqrt(1.0 + ux * ux + uy * uy)
    return ux * inv, uy * inv, inv


def ray_transfer_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    surfaces: List,
    wavelength: float,
    *,
    per_surface: bool = False,
    h_pos: float = 1e-6,
    h_slope: float = 5e-5,
) -> DifferentialTransfer:
    """Differential ray-transfer Jacobian along each base ray.

    Parameters
    ----------
    x, y : ndarray ``(N,)``
        Base-ray transverse positions at the input vertex [m].
    ux, uy : ndarray ``(N,)``
        Base-ray paraxial slopes ``L/N``, ``M/N`` at the input.
    surfaces : list of Surface
        As from :func:`raytrace.surfaces_from_prescription`.  Each surface's
        ``thickness`` is the axial distance to the next vertex.
    wavelength : float
        Vacuum wavelength [m].
    per_surface : bool, default False
        ``False`` -> composite input->output ``(N, 4, 4)`` Jacobian.
        ``True``  -> per-surface local transfers ``(N_surf, N, 4, 4)`` obtained
        from the cumulative Jacobians ``J_local_k = J_cum_k @ inv(J_cum_{k-1})``
        (needed for the per-surface complex-parameter amplitude accumulation,
        where each near-identity local factor keeps the ``sqrt(det)`` branch
        unambiguous).
    h_pos, h_slope : float
        Central-FD steps for the position and slope perturbations.  The result
        is insensitive across several decades (the trace is smooth); the
        defaults sit in the truncation-limited regime for mm-scale optics.

    Returns
    -------
    DifferentialTransfer
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ux = np.asarray(ux, dtype=np.float64)
    uy = np.asarray(uy, dtype=np.float64)
    n = x.shape[0]
    steps = (h_pos, h_pos, h_slope, h_slope)

    # Build 9N rays: base + (+/-h) in each of x, y, ux, uy.  Group g runs
    # 0=base, then (dim, sign) pairs 1..8.
    base = (x, y, ux, uy)
    cols = [base]
    for dim in range(4):
        for sign in (+1.0, -1.0):
            pert = [a.copy() for a in base]
            pert[dim] = pert[dim] + sign * steps[dim]
            cols.append(tuple(pert))
    xx = np.concatenate([c[0] for c in cols])
    yy = np.concatenate([c[1] for c in cols])
    uxx = np.concatenate([c[2] for c in cols])
    uyy = np.concatenate([c[3] for c in cols])
    L, M, _ = _slopes_to_dirs(uxx, uyy)
    rb = _make_bundle(xx, yy, L, M, wavelength)

    of = 'all' if per_surface else 'last'
    res = trace(rb, surfaces, wavelength, output_filter=of)

    def _state(bundle):
        return (bundle.x, bundle.y, bundle.L / bundle.N, bundle.M / bundle.N,
                bundle.opd, bundle.alive)

    def _cum_jac(bundle):
        sx, sy, sux, suy, _, _ = _state(bundle)
        g = [(sx[i * n:(i + 1) * n], sy[i * n:(i + 1) * n],
              sux[i * n:(i + 1) * n], suy[i * n:(i + 1) * n])
             for i in range(9)]
        J = np.zeros((n, 4, 4))
        for dim in range(4):
            gp = g[1 + 2 * dim]
            gm = g[2 + 2 * dim]
            d = 2.0 * steps[dim]
            for outr in range(4):
                J[:, outr, dim] = (gp[outr] - gm[outr]) / d
        return J

    def _companion_alive(balive):
        """A base ray is only usable if ALL its 9 finite-difference companions
        survived: a +/-h companion that vignettes/TIRs at a rim yields NaN
        Jacobian rows while the base ray's own ``alive`` is True (D2).  AND the
        companion-aliveness into the returned mask so no NaN-Jacobian row
        reaches a downstream coherent sum -- the analytic backend already
        scrubs (nan_to_num); the FD backend must too."""
        return np.asarray(balive, bool).reshape(9, n).all(axis=0)

    if not per_surface:
        img = res.image_rays
        Jc = _cum_jac(img)
        bx, by, bux, buy, bopd, balive = _state(img)
        alive = np.asarray(balive[:n], bool) & _companion_alive(balive)
        Jc = np.nan_to_num(Jc, nan=0.0, posinf=0.0, neginf=0.0)
        return DifferentialTransfer(
            jacobian=Jc, x=bx[:n], y=by[:n], ux=bux[:n], uy=buy[:n],
            opd=bopd[:n], alive=alive)

    # per-surface: cumulative J at each surface -> local transfers
    hist = res.ray_history
    cum = [np.broadcast_to(np.eye(4), (n, 4, 4)).copy()]  # J at input = I
    for hb in hist:
        cum.append(_cum_jac(hb))
    locals_ = np.stack([cum[k + 1] @ np.linalg.inv(cum[k])
                        for k in range(len(hist))], axis=0)  # (Nsurf,n,4,4)
    locals_ = np.nan_to_num(locals_, nan=0.0, posinf=0.0, neginf=0.0)
    final = hist[-1]
    bx, by, bux, buy, bopd, balive = _state(final)
    alive = np.asarray(balive[:n], bool) & _companion_alive(balive)
    return DifferentialTransfer(
        jacobian=locals_, x=bx[:n], y=by[:n], ux=bux[:n], uy=buy[:n],
        opd=bopd[:n], alive=alive)


def ray_transfer_jacobian_jax(x, y, ux, uy, prescription, wavelength):
    """Differentiable per-ray composite ABCD Jacobian via ``jax`` autodiff.

    The JAX twin of :func:`ray_transfer_jacobian` (composite input->output):
    the 4x4 ``(x, y, ux, uy)`` ray-transfer Jacobian of the JAX-traceable
    :func:`raytrace.trace_jax`, computed by ``jax.jacfwd`` (EXACT, not finite
    differences) and vmapped over rays.  Because it is built on ``trace_jax``,
    the whole thing is ``jax.grad`` / ``jax.jit`` friendly and differentiable
    with respect to the ray state -- and, via
    :func:`raytrace.trace_jax_with_params`, with respect to prescription
    parameters (radii, thicknesses) -- enabling gradient-based lens design on
    the per-surface GBD.

    .. note::
       ``jax_trace._transfer_jax`` propagates ``x_new = x + L * thickness``
       using the direction cosine ``L`` rather than the slope ``u = L / N`` --
       paraxial *for that step in isolation*.  BUT it does **not** reset ``z``
       afterwards, so the next surface's intersection re-propagates the ray to
       the true vertex plane and *undoes* the under-count.  Empirically the
       **composed** output Jacobian therefore matches the NumPy FD
       :func:`ray_transfer_jacobian` (and the exact
       :func:`ray_transfer_jacobian_analytic`) to ~1e-8 even at high NA (e.g.
       slope ``u = 0.9`` through a powered lens), because every transfer is
       followed by an intersection.  (The only way to expose the isolated
       paraxial step would be a final transfer with no following surface, which
       this primitive never emits -- its output is at the last vertex.)

    Returns
    -------
    jax array ``(N, 4, 4)``
        Per-ray composite ray-transfer Jacobian (row/col order (x, y, ux, uy)).
    """
    import jax
    import jax.numpy as jnp

    from .jax_trace import make_jax_ray_state, trace_jax

    def _out_state(s4):
        xx, yy, uxx, uyy = s4[0], s4[1], s4[2], s4[3]
        inv = 1.0 / jnp.sqrt(1.0 + uxx * uxx + uyy * uyy)
        st = make_jax_ray_state(
            jnp.reshape(xx, (1,)), jnp.reshape(yy, (1,)),
            jnp.zeros((1,)), jnp.reshape(uxx * inv, (1,)),
            jnp.reshape(uyy * inv, (1,)), jnp.reshape(inv, (1,)))
        r = trace_jax(st, prescription, wavelength)
        return jnp.stack([r.x[0], r.y[0], r.L[0] / r.N[0], r.M[0] / r.N[0]])

    s4 = jnp.stack([jnp.asarray(x), jnp.asarray(y),
                    jnp.asarray(ux), jnp.asarray(uy)], axis=-1)
    return jax.vmap(jax.jacfwd(_out_state))(s4)


# ---------------------------------------------------------------------------
# Analytic differential ray transfer (forward-mode AD over the EXACT conic
# trace).  Forward-mode AD of the exact intersection + Snell map *is* the
# analytic differential ray tracing of Stone & Forbes (Volatier, JOSA A 34,
# 1146 (2017)); it produces the same 4x4 (x, y, ux, uy) Jacobian as the finite-
# difference :func:`ray_transfer_jacobian` but EXACTLY (no truncation), in pure
# NumPy, and -- on the JAX backend -- differentiably (jax.jacfwd / grad / jit).
# Conic surfaces (sphere / conic), refraction and reflection.
# See docs/ANALYTIC_DIFFERENTIAL_RAY_TRACING_LITERATURE.md.
# ---------------------------------------------------------------------------


class _AdrtDual:
    """Minimal forward-mode AD dual: value ``v`` ``(N,)`` and derivative ``d``
    ``(N, 4)`` w.r.t. the seed state ``(x, y, ux, uy)``."""
    __slots__ = ('v', 'd')
    # Defer to our __r*__ so ``ndarray * dual`` (sign selectors) doesn't get
    # absorbed into an object-array by NumPy.
    __array_ufunc__ = None

    def __init__(self, v, d):
        self.v = v
        self.d = d

    def __add__(s, o):
        o = _as_dual(o, s)
        return _AdrtDual(s.v + o.v, s.d + o.d)
    __radd__ = __add__

    def __sub__(s, o):
        o = _as_dual(o, s)
        return _AdrtDual(s.v - o.v, s.d - o.d)

    def __rsub__(s, o):
        o = _as_dual(o, s)
        return _AdrtDual(o.v - s.v, o.d - s.d)

    def __mul__(s, o):
        o = _as_dual(o, s)
        return _AdrtDual(s.v * o.v, s.v[:, None] * o.d + o.v[:, None] * s.d)
    __rmul__ = __mul__

    def __truediv__(s, o):
        o = _as_dual(o, s)
        iv = 1.0 / o.v
        return _AdrtDual(s.v * iv,
                         (s.d * o.v[:, None] - s.v[:, None] * o.d)
                         * (iv * iv)[:, None])

    def __rtruediv__(s, o):
        return _as_dual(o, s).__truediv__(s)

    def __neg__(s):
        return _AdrtDual(-s.v, -s.d)


def _as_dual(o, ref):
    if isinstance(o, _AdrtDual):
        return o
    n = ref.v.shape[0]
    return _AdrtDual(np.broadcast_to(np.asarray(o, np.float64), (n,)).copy(),
                     np.zeros((n, 4)))


def _dual_sqrt(a):
    # clamp the radicand at 0 so a dead ray (missed surface / TIR: negative
    # discriminant) yields a FINITE (masked) value instead of a NaN that would
    # contaminate live neighbours; live rays (radicand > 0) are unaffected.
    vc = np.maximum(a.v, 0.0)
    v = np.sqrt(vc)
    return _AdrtDual(v, a.d / (2.0 * np.maximum(v, 1e-300))[:, None])


def _dual_where(m, a, b):
    a = _as_dual(a, b if isinstance(b, _AdrtDual) else a)
    b = _as_dual(b, a)
    return _AdrtDual(np.where(m, a.v, b.v), np.where(m[:, None], a.d, b.d))


_DUAL_OPS = {'sqrt': _dual_sqrt, 'val': lambda a: a.v,
             'pwhere': np.where, 'dwhere': _dual_where}


def _adrt_coordbreak(x, y, ux, uy, surf, wavelength, apply_transfer,
                     sqrt, val, compute_dead):
    """Zemax-style coordinate break (decenter + intrinsic X,Y,Z tilts) then the
    gap transfer -- no intersection.  A smooth frame rotation, so it is exactly
    differentiable; replicates ``raytrace.intersection._apply_coord_break``
    (decenter-then-tilt for ``coordbrk_order=0``, tilt-then-decenter for 1).
    The ray starts and ends at ``z = 0`` (the vertex planes), so no cross-step
    ``z`` state is needed -- the intermediate ``z`` from the tilt is consumed by
    the gap transfer.  Small tilts / decenters give differentiable alignment
    (tolerancing) sensitivity; large folds share the world-frame slope-space
    caveat (``u = L/N`` degenerates as ``N -> 0``)."""
    import math

    from ..glass import get_glass_index
    tx = math.radians(float(getattr(surf, 'tilt_x_deg', 0.0) or 0.0))
    ty = math.radians(float(getattr(surf, 'tilt_y_deg', 0.0) or 0.0))
    tz = math.radians(float(getattr(surf, 'tilt_z_deg', 0.0) or 0.0))
    dcx = float(getattr(surf, 'decenter_x_m', 0.0) or 0.0)
    dcy = float(getattr(surf, 'decenter_y_m', 0.0) or 0.0)
    order = int(getattr(surf, 'coordbrk_order', 0) or 0)

    sden = sqrt(1.0 + ux * ux + uy * uy)
    L = ux / sden
    M = uy / sden
    Nn = 1.0 / sden
    px, py, pz = x, y, 0.0 * ux            # z = 0 at the vertex plane

    def _decenter(px, py):
        return px - dcx, py - dcy

    def _tilts(px, py, pz, L, M, Nn):
        cx, sx = math.cos(tx), math.sin(tx)      # Rx (optical convention)
        py, pz = cx * py - sx * pz, sx * py + cx * pz
        M, Nn = cx * M - sx * Nn, sx * M + cx * Nn
        cy, sy = math.cos(ty), math.sin(ty)      # Ry
        px, pz = cy * px + sy * pz, -sy * px + cy * pz
        L, Nn = cy * L + sy * Nn, -sy * L + cy * Nn
        cz, sz = math.cos(tz), math.sin(tz)      # Rz
        px, py = cz * px - sz * py, sz * px + cz * py
        L, M = cz * L - sz * M, sz * L + cz * M
        return px, py, pz, L, M, Nn

    if order == 1:
        px, py, pz, L, M, Nn = _tilts(px, py, pz, L, M, Nn)
        px, py = _decenter(px, py)
    else:
        px, py = _decenter(px, py)
        px, py, pz, L, M, Nn = _tilts(px, py, pz, L, M, Nn)

    opd = 0.0 * val(ux)
    if apply_transfer:
        t = float(getattr(surf, 'thickness', 0.0) or 0.0)
        n2 = float(get_glass_index(
            getattr(surf, 'glass_after', None) or 'air', wavelength))
        tau = (t - pz) / Nn
        px = px + L * tau
        py = py + M * tau
        # SIGNED transfer leg (RT-1), matching raytrace._transfer's n*t used by
        # the main-trace coord-break path (trace._apply_coord_break -> _transfer):
        # a negative gap tau must SUBTRACT its OPL, not add via abs (S3-9).
        opd = n2 * val(tau)
    ux_out = L / Nn
    uy_out = M / Nn
    dead = None
    if compute_dead:
        dead = np.zeros(val(ux).shape, dtype=bool)
    return px, py, ux_out, uy_out, opd, dead


def _adrt_step(x, y, ux, uy, surf, wavelength, apply_transfer, O,
               compute_dead=True):
    """One surface: exact intersect (conic) + refract/reflect + optional
    transfer, OR a coordinate-break frame transform (``is_coordbrk``), on
    dual-or-jnp ``(x, y, ux, uy)``.  Returns the updated state plus the
    plain-array OPL increment and dead-ray mask."""
    from ..glass import get_glass_index
    sqrt, val = O['sqrt'], O['val']
    pwhere, dwhere = O['pwhere'], O['dwhere']
    if bool(getattr(surf, 'is_coordbrk', False)):
        return _adrt_coordbreak(x, y, ux, uy, surf, wavelength,
                                apply_transfer, sqrt, val, compute_dead)
    R = float(getattr(surf, 'radius', np.inf))
    c = 0.0 if not np.isfinite(R) else 1.0 / R
    k = float(getattr(surf, 'conic', 0.0) or 0.0)
    is_mir = bool(getattr(surf, 'is_mirror', False))
    n1 = float(get_glass_index(getattr(surf, 'glass_before', None) or 'air',
                               wavelength))
    n2 = float(get_glass_index(getattr(surf, 'glass_after', None) or 'air',
                               wavelength))
    sden = sqrt(1.0 + ux * ux + uy * uy)
    L = ux / sden
    M = uy / sden
    Nn = 1.0 / sden
    # intersect conic F = c(x^2+y^2) - 2z + (1+k) c z^2 = 0, ray from (x, y, 0)
    a = c * (L * L + M * M) + (1.0 + k) * c * (Nn * Nn)
    b = 2.0 * (c * (x * L + y * M) - Nn)
    e = c * (x * x + y * y)
    disc = b * b - 4.0 * a * e
    sq = sqrt(disc)
    sgn = pwhere(val(b) >= 0.0, 1.0, -1.0)
    q = -0.5 * (b + sgn * sq)
    # stable near-vertex root tau = e/q; flat surface (a == 0) -> tau = -e/b
    tau = dwhere(abs(val(a)) < 1e-14, (0.0 - e) / b, e / q)
    xi = x + tau * L
    yi = y + tau * M
    zi = tau * Nn
    # surface normal grad F = (2c x, 2c y, -2 + 2(1+k)c z), oriented against ray
    gx = (2.0 * c) * xi
    gy = (2.0 * c) * yi
    gz = -2.0 + (2.0 * (1.0 + k) * c) * zi
    gn = sqrt(gx * gx + gy * gy + gz * gz)
    nx = gx / gn
    ny = gy / gn
    nz = gz / gn
    # S3-10: vector Snell / reflection via the backend-agnostic shared
    # core (raytrace._conic_core).  The core orients the (un-oriented)
    # grad-F normal against the ray and applies the same law this site
    # used; ``eta_sq = eta * eta`` preserves this site's PRODUCT form
    # (NOT the scalar-power ``mu**2`` the NumPy / JAX sites use -- IEEE
    # gives ``x**2 != x*x`` for ~0.05% of ratios; see the shared-core
    # docstring), and the injected ADRT ``sqrt`` (``O['sqrt']`` --
    # ``_dual_sqrt`` on the dual backend, a clamping jnp sqrt on the JAX
    # backend) clamps the radicand so the default no-op TIR guard
    # reproduces the former ``root = sqrt(disc_r)`` exactly.  Pure /
    # dual-aware: no in-place writes.
    if is_mir:
        Lp, Mp, Np, nx, ny, nz, _cos_i = reflect_mirror(
            L, M, Nn, nx, ny, nz, where=pwhere, val=val)
        disc_r = None
    else:
        eta = n1 / n2
        Lp, Mp, Np, nx, ny, nz, _cos_i, disc_r, _tir = refract_snell(
            L, M, Nn, nx, ny, nz, eta, eta * eta,
            sqrt=sqrt, where=pwhere, val=val)
    if apply_transfer:
        t = float(getattr(surf, 'thickness', 0.0) or 0.0)
        tau2 = (t - zi) / Np
        x_out = xi + tau2 * Lp
        y_out = yi + tau2 * Mp
        # OPL: SIGNED intersection leg (a backtrack on a concave surface
        # subtracts over-counted OPL, matching raytrace._intersect_surface) plus
        # the SIGNED transfer leg (matching raytrace._transfer's RT-1 signed
        # n*t: a negative tau2 -- overlapping sag / post-mirror fold -- means the
        # ray already crossed the next vertex plane, so the over-counted OPL must
        # be SUBTRACTED, not added via abs; abs here diverged from the base-ray
        # OPL the FD/main trace produces whenever tau2 < 0, S3-9).
        opd = n1 * val(tau) + n2 * val(tau2)
    else:
        x_out = xi
        y_out = yi
        opd = n1 * val(tau)
    ux_out = Lp / Np
    uy_out = Mp / Np
    # dead: aperture vignette + TIR (NumPy path only; the JAX path returns
    # alive=True and would trip a tracer->ndarray conversion here).
    dead = None
    if compute_dead:
        dead = val(disc) < 0.0                    # missed the surface
        sd = float(getattr(surf, 'semi_diameter', np.inf))
        if np.isfinite(sd):
            dead = dead | (val(xi) ** 2 + val(yi) ** 2 > sd * sd)
        if disc_r is not None:
            dead = dead | (val(disc_r) < 0.0)     # TIR
    return x_out, y_out, ux_out, uy_out, opd, dead


def _adrt_numpy(x, y, ux, uy, surfaces, wavelength, per_surface):
    x = np.asarray(x, np.float64)
    n = x.shape[0]
    y = np.broadcast_to(np.asarray(y, np.float64), (n,)).copy()
    ux = np.broadcast_to(np.asarray(ux, np.float64), (n,)).copy()
    uy = np.broadcast_to(np.asarray(uy, np.float64), (n,)).copy()
    x = np.broadcast_to(x, (n,)).copy()
    eye = np.eye(4)

    def _seed(xv, yv, uxv, uyv):
        return (_AdrtDual(xv.copy(), np.tile(eye[0], (n, 1))),
                _AdrtDual(yv.copy(), np.tile(eye[1], (n, 1))),
                _AdrtDual(uxv.copy(), np.tile(eye[2], (n, 1))),
                _AdrtDual(uyv.copy(), np.tile(eye[3], (n, 1))))

    opd = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    nsurf = len(surfaces)
    # dead rays (missed / TIR) can overflow the dual arithmetic; the result is
    # masked (alive) and nan_to_num'd below, so silence the expected noise.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        if per_surface:
            locals_ = []
            cx, cy, cux, cuy = x, y, ux, uy
            for si, s in enumerate(surfaces):
                X, Y, UX, UY = _seed(cx, cy, cux, cuy)
                X, Y, UX, UY, dopd, dead = _adrt_step(
                    X, Y, UX, UY, s, wavelength, si < nsurf - 1, _DUAL_OPS)
                Jk = np.stack([X.d, Y.d, UX.d, UY.d], axis=1)   # (n, 4, 4)
                locals_.append(Jk)
                opd = opd + dopd
                alive = alive & ~dead
                cx, cy, cux, cuy = X.v, Y.v, UX.v, UY.v
            jac = np.stack(locals_, axis=0)                 # (nsurf, n, 4, 4)
            bx, by, bux, buy = cx, cy, cux, cuy
        else:
            X, Y, UX, UY = _seed(x, y, ux, uy)
            for si, s in enumerate(surfaces):
                X, Y, UX, UY, dopd, dead = _adrt_step(
                    X, Y, UX, UY, s, wavelength, si < nsurf - 1, _DUAL_OPS)
                opd = opd + dopd
                alive = alive & ~dead
            jac = np.stack([X.d, Y.d, UX.d, UY.d], axis=1)  # (n, 4, 4)
            bx, by, bux, buy = X.v, Y.v, UX.v, UY.v
    # A dead ray (missed / TIR) can leave non-finite entries (e.g. a grazing
    # 1/N'); zero them so they cannot contaminate array-wide ops downstream.
    # Live rays are already finite, so this is a no-op for them.
    jac = np.nan_to_num(jac, nan=0.0, posinf=0.0, neginf=0.0)
    bx, by, bux, buy, opd = (np.nan_to_num(v) for v in (bx, by, bux, buy, opd))
    return DifferentialTransfer(jacobian=jac, x=bx, y=by, ux=bux, uy=buy,
                                opd=opd, alive=alive)


def ray_transfer_jacobian_analytic(
    x, y, ux, uy, surfaces, wavelength, *, per_surface: bool = False,
):
    """Analytic (exact) differential ray-transfer Jacobian -- the closed-form /
    autodiff twin of the finite-difference :func:`ray_transfer_jacobian`.

    Forward-mode AD over the EXACT conic trace (intersection + vector Snell /
    reflection + vertex transfer), so the 4x4 ``(x, y, ux, uy)`` ray-transfer
    Jacobian is computed WITHOUT finite-difference truncation (the ``h -> 0``
    limit) and is correct at all NA.  On the JAX backend (``x`` a jax array) the
    trace is differentiated by ``jax.jacfwd`` and is itself ``jax.grad`` /
    ``jit`` friendly.

    Value vs the two existing primitives: it is exact where the FD
    :func:`ray_transfer_jacobian` carries ~1e-8 truncation, and pure NumPy (no
    JAX dependency, unlike :func:`ray_transfer_jacobian_jax`).  It is the
    closed-form realization of the differential ray tracing of Stone & Forbes
    (forward-mode AD == analytic differential ray tracing, Volatier 2017); see
    ``docs/ANALYTIC_DIFFERENTIAL_RAY_TRACING_LITERATURE.md``.  (Note: the
    *composed output* of :func:`ray_transfer_jacobian_jax` is empirically also
    exact at high NA -- its per-surface paraxial transfer under-count is undone
    by the next surface's intersection -- so this is not a high-NA correction of
    that path, just a NumPy-native, truncation-free one with a cleaner
    forward-AD structure.)

    Same signature / return (:class:`DifferentialTransfer`) and ``per_surface``
    semantics as :func:`ray_transfer_jacobian`; agrees with it to the FD
    truncation floor (~1e-8).  On axis the 2x2 meridional block equals
    ``system_abcd_prescription`` (exactly for air-to-air prescriptions, where
    the unreduced slope ``u = L/N`` coincides with the reduced ``n*u`` momentum
    at the ``n = 1`` endpoints).  Refracting / reflecting surfaces must be conic
    (sphere / conic ``+`` thickness ``+`` glass ``+`` ``is_mirror``); **Zemax
    coordinate breaks** (``is_coordbrk`` -- decenter + X/Y/Z tilts, a smooth
    frame transform) ARE handled and differentiable, giving alignment /
    tolerancing sensitivity through a fold (a *large* tilt shares the slope-
    space caveat: ``u = L/N`` degenerates as the folded ``N -> 0``).
    Aspheric-polynomial departures, freeforms and biconics are not yet handled
    (use the FD primitive there).

    Returns
    -------
    DifferentialTransfer
    """
    from ..backend.array import is_jax_array
    for s in surfaces:
        # _adrt_step reads only ``radius`` / ``conic`` (rotationally symmetric)
        # for refracting/reflecting surfaces (coordinate breaks are handled
        # separately), so a biconic ``radius_y`` / ``conic_y`` /
        # ``aspheric_coeffs_y``, an asphere, or a freeform must be rejected
        # (else a biconic would be silently traced as if it were rotationally
        # symmetric, giving a wrong y-axis power).
        if (getattr(s, 'aspheric_coeffs', None)
                or getattr(s, 'freeform', None)
                or getattr(s, 'radius_y', None) is not None
                or getattr(s, 'conic_y', None) is not None
                or getattr(s, 'aspheric_coeffs_y', None) is not None):
            raise NotImplementedError(
                'ray_transfer_jacobian_analytic handles rotationally-symmetric '
                'conic surfaces (plus coordinate breaks) only; aspheric-'
                'polynomial departures, freeforms and biconic (radius_y / '
                'conic_y) surfaces are not yet supported -- use '
                'ray_transfer_jacobian (FD) for those.')
    if is_jax_array(x) or is_jax_array(y) or is_jax_array(ux) \
            or is_jax_array(uy):
        return _adrt_jax(x, y, ux, uy, surfaces, wavelength, per_surface)
    return _adrt_numpy(x, y, ux, uy, surfaces, wavelength, per_surface)


def _adrt_jax(x, y, ux, uy, surfaces, wavelength, per_surface):
    import jax
    import jax.numpy as jnp
    if per_surface:
        raise NotImplementedError(
            'ray_transfer_jacobian_analytic: per_surface=True is NumPy-only; '
            'the JAX path returns the composite Jacobian (use jax.jacfwd on the '
            'per-surface steps if per-surface gradients are needed).')
    jnp_ops = {'sqrt': lambda z: jnp.sqrt(jnp.maximum(z, 0.0)),
               'val': lambda a: a, 'pwhere': jnp.where, 'dwhere': jnp.where}
    nsurf = len(surfaces)

    def _state(s4):
        xx, yy, uxx, uyy = s4[0], s4[1], s4[2], s4[3]
        for si, s in enumerate(surfaces):
            xx, yy, uxx, uyy, _dopd, _dead = _adrt_step(
                xx, yy, uxx, uyy, s, wavelength, si < nsurf - 1, jnp_ops,
                compute_dead=False)
        return jnp.stack([xx, yy, uxx, uyy])

    def _full(s4):
        xx, yy, uxx, uyy = s4[0], s4[1], s4[2], s4[3]
        opd = jnp.zeros(())
        for si, s in enumerate(surfaces):
            xx, yy, uxx, uyy, dopd, _dead = _adrt_step(
                xx, yy, uxx, uyy, s, wavelength, si < nsurf - 1, jnp_ops,
                compute_dead=False)
            opd = opd + dopd
        return jnp.stack([xx, yy, uxx, uyy]), opd

    s4 = jnp.stack([jnp.reshape(jnp.asarray(x), (-1,)),
                    jnp.reshape(jnp.asarray(y), (-1,)),
                    jnp.reshape(jnp.asarray(ux), (-1,)),
                    jnp.reshape(jnp.asarray(uy), (-1,))], axis=0)
    n = s4.shape[1]
    jac = jax.vmap(jax.jacfwd(_state), in_axes=1, out_axes=0)(s4)
    st, opd = jax.vmap(_full, in_axes=1, out_axes=(0, 0))(s4)
    return DifferentialTransfer(
        jacobian=jac, x=st[:, 0], y=st[:, 1], ux=st[:, 2], uy=st[:, 3],
        opd=opd, alive=jnp.ones((n,), dtype=bool))


__all__ = ['DifferentialTransfer', 'ray_transfer_jacobian',
           'ray_transfer_jacobian_analytic', 'ray_transfer_jacobian_jax']

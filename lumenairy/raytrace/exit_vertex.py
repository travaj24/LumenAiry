"""
lumenairy.raytrace.exit_vertex -- the shared exit-vertex transfer.

:func:`lumenairy.raytrace.trace` leaves every ray at its intersection
with the LAST surface, i.e. at ``z = sag(rho)`` in that surface's local
frame -- NOT on the exit vertex plane ``z = 0``.  Every consumer that
wants the exit OPL / transverse position on the vertex plane therefore
has to apply the signed straight-line transfer::

    t    = -z / N                 (alive, non-grazing rays only)
    opd += n_exit * t
    x   += L * t
    y   += M * t
    z    = 0

This module is the ONE implementation of that operator.  Consumers
should call :meth:`lumenairy.raytrace.TraceResult.at_exit_vertex` (which
resolves ``n_exit`` from the prescription) or :func:`exit_vertex_transfer`
(when they hold a bare bundle and already know the index).  The
JAX-traceable twin is
:func:`lumenairy.raytrace.jax_trace.exit_vertex_transfer_jax`.

Why it is a shared primitive
----------------------------
Two properties are easy to get wrong and were got wrong differently in
each of the hand-written copies the 2026-09-11 audit found (§15.1, seven
sites):

* **Signed, not absolute.**  ``t`` is negative whenever the last surface
  is concave to the outgoing ray (the ray has already crossed the vertex
  plane), and the over-counted OPL must be SUBTRACTED.  ``abs(t)``
  silently doubles the error instead of removing it.
* **Grazing rays are killed, not teleported.**  A ray with ``|N|`` at or
  below :data:`EXIT_VERTEX_GRAZING_TOL` never reaches the vertex plane.
  The NumPy copies masked it to ``t = 0`` but still wrote ``z = 0``,
  teleporting the ray with zero OPL while leaving it ``alive``; the two
  JAX copies clamped ``N`` to ``1e-30`` and produced ``t = -z/1e-30``.
  Here such a ray is killed with ``RAY_MISSED_SURFACE`` and its state is
  frozen -- the same vocabulary and the same policy the flat branch of
  :func:`lumenairy.raytrace.intersection._intersect_surface` uses.

Dead rays (``alive == False`` on input, including newly-killed grazing
rays) keep their position, direction and OPL exactly.  The operator is
therefore **idempotent**: applying it twice is a no-op, because every ray
it moved now sits at ``z = 0`` and gets ``t = 0`` on the second pass.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .surface import RAY_MISSED_SURFACE, RAY_OK

# Grazing-ray tolerance on |N|.  Shared with ``intersection._transfer``
# and the flat branch of ``intersection._intersect_surface`` so the three
# "advance to a z = const plane" primitives agree on what "parallel to
# the plane" means (a ray with |N| <= this value cannot reach the plane
# in any finite path length: |t| = |z| / |N| >= |z| * 1e30).
EXIT_VERTEX_GRAZING_TOL = 1e-30

__all__ = [
    'EXIT_VERTEX_GRAZING_TOL',
    'exit_vertex_transfer',
    'resolve_exit_index',
    'vertex_plane_transfer_t',
]


def vertex_plane_transfer_t(
    z,
    N,
    alive,
    *,
    z_target: float = 0.0,
    tol: float = EXIT_VERTEX_GRAZING_TOL,
) -> Tuple[np.ndarray, np.ndarray]:
    """Signed parametric distance to the plane ``z = z_target``.

    The low-level kernel shared by :func:`exit_vertex_transfer`,
    :func:`lumenairy.raytrace.intersection._transfer`, the flat branch of
    :func:`lumenairy.raytrace.intersection._intersect_surface` and
    :func:`lumenairy.raytrace.ray_fan.refocus`, so that all four agree on
    the arithmetic AND on the grazing-ray definition.

    Parameters
    ----------
    z, N : ndarray
        Current axial position [m] and longitudinal direction cosine.
    alive : ndarray of bool
        Rays that are still propagating.  Dead rays get ``t = 0``.
    z_target : float, default 0.0
        Plane to advance to, in the same frame as ``z``.
    tol : float, default :data:`EXIT_VERTEX_GRAZING_TOL`
        Rays with ``|N| <= tol`` are grazing: they never reach the plane.

    Returns
    -------
    t : ndarray
        ``(z_target - z) / N`` where the ray is alive and non-grazing,
        ``0.0`` everywhere else.  Signed: negative means the ray has
        already passed the plane and must back-track.
    graze : ndarray of bool
        Alive rays whose ``|N| <= tol`` -- the caller decides whether to
        kill them (every caller in this package does).
    """
    z = np.asarray(z)
    N = np.asarray(N)
    alive = np.asarray(alive, dtype=bool)
    propagating = np.abs(N) > tol
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.where(alive & propagating, (z_target - z) / N, 0.0)
    return t, alive & ~propagating


def _kill_grazing(bundle, graze) -> None:
    """Mark ``graze`` rays dead on ``bundle`` with ``RAY_MISSED_SURFACE``.

    First-failure-wins on ``error_code`` (only ``RAY_OK`` entries are
    overwritten), matching every other kill site in this package.
    """
    if not graze.any():
        return
    bundle.alive = np.asarray(bundle.alive, dtype=bool) & ~graze
    ec = getattr(bundle, 'error_code', None)
    if ec is not None:
        first_failure = graze & (ec == RAY_OK)
        bundle.error_code = np.where(first_failure, RAY_MISSED_SURFACE, ec)


def exit_vertex_transfer(bundle, n_exit, *, fn_name: str = 'exit_vertex_transfer'):
    """Transfer a post-trace ray bundle to its exit VERTEX plane.

    Straight-line transfer of every ALIVE ray from its intersection with
    the last surface (``z = sag(rho)``) to that surface's vertex plane
    (``z = 0``), accumulating the signed optical path in the exit medium.

    Parameters
    ----------
    bundle : RayBundle
        Ray bundle in the LAST SURFACE's local frame -- normally
        ``TraceResult.image_rays``.  **Not modified**; a copy is
        returned.
    n_exit : float or ndarray
        Refractive index of the medium the rays are in after the last
        surface, at the trace wavelength.  Scalar, or per-ray.  Use
        :meth:`lumenairy.raytrace.TraceResult.at_exit_vertex` to have it
        resolved from the prescription automatically.
    fn_name : str, optional
        Name used in error messages (so the caller that resolved
        ``n_exit`` can own the diagnostic).

    Returns
    -------
    RayBundle
        A new bundle of the same class as ``bundle`` with

        * ``z = 0`` for every ray that was transferred,
        * ``x``, ``y`` advanced along ``(L, M)``,
        * ``opd`` increased by the SIGNED ``n_exit * t``,
        * directions ``(L, M, N)`` untouched (this is a transfer, not a
          refraction),
        * grazing rays (``|N| <=``
          :data:`EXIT_VERTEX_GRAZING_TOL`) killed with
          ``error_code = RAY_MISSED_SURFACE`` and their state frozen,
        * rays that were already dead frozen exactly as they were
          (position, direction and OPL), so a vignetted ray never
          reports a vertex-plane coordinate it did not reach.

    Notes
    -----
    * **Idempotent.**  ``exit_vertex_transfer(exit_vertex_transfer(b, n),
      n)`` equals ``exit_vertex_transfer(b, n)`` bit-for-bit: transferred
      rays sit at ``z = 0`` so the second pass computes ``t = 0``.
    * **Exact on the analytic OPL.**  For a ray leaving a surface of sag
      ``s`` at direction cosine ``N``, the vertex-plane optical path is
      ``opl_sag - n_exit * s / N`` -- which is what this returns, with no
      paraxial approximation and no ``abs()``.
    * ``n_exit`` is a physical (positive) index even for a system whose
      last surface is a mirror: the Welford ``n' = -n`` convention lives
      only in the paraxial/Seidel algebra, never in the traced OPL.  The
      signed ``t`` already carries the propagation direction.

    Examples
    --------
    >>> import numpy as np, lumenairy.raytrace as rt
    >>> surfs = [rt.Surface(radius=0.05, glass_before='air',
    ...                     glass_after='N-BK7', thickness=3e-3),
    ...          rt.Surface(radius=-0.05, glass_before='N-BK7',
    ...                     glass_after='air')]
    >>> res = rt.trace(rt.make_fan('y', 5e-3, 5, 0.0, 587.6e-9),
    ...                surfs, 587.6e-9)
    >>> ex = res.at_exit_vertex()
    >>> bool(np.allclose(ex.z, 0.0))
    True
    """
    n_exit_arr = np.asarray(n_exit, dtype=np.float64)
    if not np.all(np.isfinite(n_exit_arr)) or np.any(n_exit_arr <= 0.0):
        raise ValueError(
            f"{fn_name}: n_exit must be a positive, finite refractive "
            f"index (scalar or per-ray); got {n_exit!r}.  The traced OPL "
            f"is a physical path length, so a mirror's Welford 'n2 = -n1' "
            f"paraxial sign must NOT be passed here -- the signed transfer "
            f"parameter already carries the propagation direction.")

    out = bundle.copy()
    t, graze = vertex_plane_transfer_t(out.z, out.N, out.alive)
    _kill_grazing(out, graze)
    # Recompute the mask AFTER the kill so a grazing ray neither moves
    # nor accrues OPL (``t`` is already 0 there, but ``z`` must not be
    # forced to the vertex plane either -- that was the "immortal
    # phantom" teleport the audit found in five of the seven copies).
    moved = np.asarray(out.alive, dtype=bool)
    out.opd = out.opd + n_exit_arr * t
    out.x = out.x + out.L * t
    out.y = out.y + out.M * t
    out.z = np.where(moved, 0.0, out.z)
    return out


def resolve_exit_index(
    surfaces,
    wavelength: float,
    *,
    fn_name: str = 'at_exit_vertex',
    n_exit: Optional[float] = None,
) -> float:
    """Refractive index of the medium after ``surfaces[-1]``.

    The default ``n_exit`` for :meth:`TraceResult.at_exit_vertex`.  Reads
    ``surfaces[-1].glass_after`` at ``wavelength``; for a last surface
    that is a MIRROR the reflected ray travels back through
    ``glass_before``, so that is used instead (a mirror's ``glass_after``
    is frequently the ``'MIRROR'`` marker rather than a real medium).

    Raises
    ------
    ValueError
        When the glass name cannot be resolved -- the caller must then
        pass ``n_exit`` explicitly.  Naming the failure here (rather
        than letting a ``KeyError`` escape from the glass registry) is
        the §2 convention.
    """
    if n_exit is not None:
        return float(n_exit)
    from ..glass import get_glass_index
    if not surfaces:
        raise ValueError(
            f"{fn_name}: the trace result carries no surfaces, so the exit "
            f"medium index cannot be inferred; pass n_exit explicitly.")
    last = surfaces[-1]
    name = getattr(last, 'glass_after', 'air')
    if getattr(last, 'is_mirror', False) or (
            isinstance(name, str) and name.upper() in ('MIRROR', '__MIRROR__')):
        # A reflected ray leaves through the medium it arrived in.
        name = getattr(last, 'glass_before', 'air')
    try:
        return float(get_glass_index(name, wavelength))
    except Exception as exc:          # noqa: BLE001 -- re-raised with context
        raise ValueError(
            f"{fn_name}: could not resolve the exit-medium index for "
            f"surfaces[-1].glass_after={name!r} at wavelength="
            f"{wavelength!r} m ({type(exc).__name__}: {exc}).  Pass "
            f"n_exit=<index> explicitly.") from exc

"""
v5.1.0 split: world-frame sequential trace + world<->local helpers.

Extracted from ``lumenairy/raytrace/core.py`` as part of the v5.1.0
6-file split (ROADMAP Agent B).  Hosts the world-frame trace engine
introduced in v3.7.5 along with the small ``_world_to_local_state`` /
``_local_to_world_state`` helpers that rotate a ray bundle between
the surface's local frame and absolute world coordinates.

Every public name here is re-exported from
``lumenairy.raytrace.core`` so existing imports continue to resolve.

No physics change: contents are bit-for-bit copies of the original
implementations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from ..glass import get_glass_index
from .intersection import (
    _intersect_surface,
    _reflect,
    _refract,
)
from .surface import (
    RAY_EVANESCENT,
    RAY_OK,
    RayBundle,
    Surface,
    TraceResult,
)

# ============================================================================
# 3.7.5: World-frame sequential trace
# ============================================================================

def _world_to_local_state(rays, origin, R):
    """Transform a ray bundle's (x, y, z) and (L, M, N) from world to
    a surface-local frame defined by (origin, R).

    ``R`` is the surface's local-to-world rotation matrix (i.e. the
    columns of ``R`` are the local +x, +y, +z axes expressed in world
    coords).  The inverse transform is ``R.T``.
    """
    Rt = R.T
    dx = rays.x - origin[0]
    dy = rays.y - origin[1]
    dz = rays.z - origin[2]
    rays.x = Rt[0, 0] * dx + Rt[0, 1] * dy + Rt[0, 2] * dz
    rays.y = Rt[1, 0] * dx + Rt[1, 1] * dy + Rt[1, 2] * dz
    rays.z = Rt[2, 0] * dx + Rt[2, 1] * dy + Rt[2, 2] * dz
    Lw, Mw, Nw = rays.L, rays.M, rays.N
    rays.L = Rt[0, 0] * Lw + Rt[0, 1] * Mw + Rt[0, 2] * Nw
    rays.M = Rt[1, 0] * Lw + Rt[1, 1] * Mw + Rt[1, 2] * Nw
    rays.N = Rt[2, 0] * Lw + Rt[2, 1] * Mw + Rt[2, 2] * Nw


def _local_to_world_state(rays, origin, R):
    """Inverse of :func:`_world_to_local_state`."""
    lx, ly, lz = rays.x, rays.y, rays.z
    rays.x = origin[0] + R[0, 0] * lx + R[0, 1] * ly + R[0, 2] * lz
    rays.y = origin[1] + R[1, 0] * lx + R[1, 1] * ly + R[1, 2] * lz
    rays.z = origin[2] + R[2, 0] * lx + R[2, 1] * ly + R[2, 2] * lz
    Ll, Ml, Nl = rays.L, rays.M, rays.N
    rays.L = R[0, 0] * Ll + R[0, 1] * Ml + R[0, 2] * Nl
    rays.M = R[1, 0] * Ll + R[1, 1] * Ml + R[1, 2] * Nl
    rays.N = R[2, 0] * Ll + R[2, 1] * Ml + R[2, 2] * Nl


def trace_world(
    rays: 'RayBundle',
    surfaces: List['Surface'],
    wavelength: float,
    output_filter: Union[str, Callable[..., Any]] = 'all',
    surface_diffraction: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
) -> 'TraceResult':
    """Sequential ray trace in world coordinates.

    Each Surface carries its own ``world_origin`` (vertex world
    position, metres) and ``world_R`` (3x3 local-to-world rotation).
    Rays propagate in world coordinates between surfaces; at each
    surface they are transformed into the local frame for the
    existing intersect / refract / reflect path, then transformed
    back to world for the leg to the next surface.

    The trace is strictly sequential: surface i sees the ray bundle
    that came out of surface i-1, and produces the bundle that goes
    into surface i+1.  No surface intersection happens "out of
    order".

    Coordinate breaks are NOT emitted on this path -- the surface
    list contains one Surface per actual optical surface, and any
    tilts / decenters are baked into each surface's ``world_R`` and
    ``world_origin``.  ``surface.thickness`` is unused: the gap to
    the next surface is implicit in the next surface's
    ``world_origin``.

    Parameters
    ----------
    rays : RayBundle
        Input rays expressed in WORLD coordinates.  The bundle is
        not modified; a copy is traced.
    surfaces : list of Surface
        Each must have ``world_origin`` and ``world_R`` populated.
    wavelength : float
        Vacuum wavelength [m].
    output_filter : ``'all'`` | ``'last'`` | callable
        Identical semantics to :func:`trace`.  History is recorded
        in each surface's LOCAL frame (so that downstream consumers
        and the GUI's ``surface_frames_*_mm`` rendering --
        ``world_pos = surf.world_origin + surf.world_R @ ray.local``
        -- stay aligned with the legacy path).
    surface_diffraction : dict or None
        Same {surface_index: (mx, my, period_x_m, period_y_m)}
        spec accepted by :func:`trace`.

    Returns
    -------
    TraceResult
    """
    r = rays.copy()
    history = []
    final = None
    _diff = surface_diffraction or {}

    _n_pre = [get_glass_index(s.glass_before, wavelength) for s in surfaces]
    _n_post = [get_glass_index(s.glass_after, wavelength) for s in surfaces]

    for i, surf in enumerate(surfaces):
        if surf.world_origin is None or surf.world_R is None:
            raise ValueError(
                f'trace_world: surface {i} ({surf.label!r}) is missing '
                'world_origin / world_R.  Use trace() for surfaces '
                'without world frames, or populate them via '
                '_build_trace_surfaces_world().')

        n1 = _n_pre[i]
        n2 = _n_post[i]

        # 1. Bring the ray bundle into this surface's local frame.
        # After this, ``rays.z`` is the signed distance from the
        # surface vertex along the local +z axis -- the same state
        # the legacy ``_transfer`` would have left rays in just
        # before the next ``_intersect_surface`` call.
        _world_to_local_state(r, surf.world_origin, surf.world_R)

        # 2. Intersect with the surface in local coords.  This
        # accumulates the inter-surface OPL via the SIGNED ``n*t`` leg
        # (not |t|; cf. RT-1) -- a negative t back-tracks and subtracts
        # its over-counted OPL.
        _intersect_surface(r, surf, n_medium=n1)

        # 3. Refract or reflect at the surface.
        if surf.is_mirror:
            _reflect(r, surf)
        else:
            _refract(r, surf, n1, n2)

        # 3.5. Diffractive-order kick (DOE) -- applied in the local
        # frame where (L, M) are the in-plane direction cosines that
        # match how the grating period is specified.
        _diff_spec = _diff.get(i)
        if _diff_spec is not None:
            _mx, _my, _px, _py = _diff_spec
            # R-13 (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25), world twin:
            # a zero or non-finite period means "no grating along that
            # axis" and yields a ZERO kick -- the same contract the JAX
            # path documents (``jax_trace._apply_doe_kick_jax._kick``:
            # "Returns 0.0 when ``period`` is non-finite or zero") and
            # the sibling numpy loop (``trace.py``) now enforces.  Pre-fix
            # this site divided unguarded, so ``period=0.0`` raised
            # ``ZeroDivisionError`` mid-trace and ``period=nan`` silently
            # NaN-poisoned (L, M).  ``inf`` already gave 0.0 by IEEE
            # division, so that case is bit-identical.
            _px_f = float(_px)
            _py_f = float(_py)
            _dL = (float(_mx) * wavelength / _px_f
                   if (np.isfinite(_px_f) and _px_f != 0.0) else 0.0)
            _dM = (float(_my) * wavelength / _py_f
                   if (np.isfinite(_py_f) and _py_f != 0.0) else 0.0)
            r.L = r.L + _dL
            r.M = r.M + _dM
            _sumsq = r.L * r.L + r.M * r.M
            _evan = _sumsq > 1.0
            _propagating = ~_evan
            _N_new = np.zeros_like(r.N)
            np.sqrt(np.maximum(1.0 - _sumsq, 0.0),
                    out=_N_new, where=_propagating)
            r.N = np.where(r.N < 0, -_N_new, _N_new)
            r.opd = r.opd + _dL * r.x + _dM * r.y
            if np.any(_evan) and r.alive is not None:
                r.alive = r.alive & _propagating
                if r.error_code is not None:
                    r.error_code = np.where(
                        _evan & (r.error_code == RAY_OK),
                        np.uint8(RAY_EVANESCENT),
                        r.error_code,
                    )

        # 4. Record post-surface state in the LOCAL frame so the
        # GUI's ``world_pos = surf.world_origin + surf.world_R @
        # ray.local`` rendering matches without changes.
        if output_filter == 'all':
            history.append(r.copy())
        elif callable(output_filter):
            item = output_filter(r, surf, i)
            if item is not None:
                history.append(item)

        if i == len(surfaces) - 1:
            final = r.copy() if output_filter == 'last' else None

        # 5. Transform back to world for the leg to the next surface
        # (skipped on the last surface -- callers consume
        # ``image_rays`` in local coords for index-i access, but if
        # they want world coords they can reproject via the
        # surface's world frame).
        if i < len(surfaces) - 1:
            _local_to_world_state(r, surf.world_origin, surf.world_R)

    if output_filter == 'last':
        history = [final] if final is not None else []

    return TraceResult(
        surfaces=surfaces,
        ray_history=history,
        input_rays=rays,
        wavelength=wavelength,
    )


__all__ = [
    '_world_to_local_state',
    '_local_to_world_state',
    'trace_world',
]

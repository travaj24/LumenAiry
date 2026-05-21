"""
lumenairy.raytrace.bundles -- common protocol for ray-style bundles
plus inter-bundle conversion utilities.

Three bundle types share a vocabulary in lumenairy:

* :class:`lumenairy.raytrace.RayBundle` -- pure geometric rays
  stored as per-component arrays (``x, y, z, L, M, N, opd, alive``).
* :class:`lumenairy.propagators.hfpi.PathBundle` -- ray bundle
  with a complex amplitude weight per path (HFPI Monte-Carlo).
  Stored as ``positions``/``directions`` ``(N, 3)`` stacked arrays.
* :class:`lumenairy.propagators.gbd.BeamletBundle` -- ray bundle
  with a complex beam parameter (Q) and amplitude per beamlet
  (Gaussian Beamlet Decomposition).  Also ``positions``/``directions``.

PathBundle and BeamletBundle both carry the stacked
``positions`` / ``directions`` form; RayBundle uses per-component
arrays.  The conversion helpers below bridge those two layouts.

History note (4.11.2): pre-4.11.2 the helpers below referenced
``ray_bundle.positions`` / ``.directions`` -- attributes that
``RayBundle`` does not expose -- so they raised ``AttributeError`` on
the first call.  They are still re-exported from
:mod:`lumenairy.raytrace.__init__` but no internal caller used them,
so the bug never surfaced in CI.  4.11.2 fixes the attribute access
to match the actual ``RayBundle`` schema (``x, y, z, L, M, N``).

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    # Forward references for type-checking only.  These imports cause
    # a circular dependency at runtime (RayBundle lives in
    # :mod:`lumenairy.raytrace.core` which itself depends on this
    # module), so they are only loaded when a type checker
    # (mypy / pyright / pylance) is processing the file.
    from ..propagators.gbd import BeamletBundle
    from ..propagators.hfpi import PathBundle
    from .core import RayBundle


@runtime_checkable
class BundleProtocol(Protocol):
    """Structural type satisfied by PathBundle and BeamletBundle
    (the GBD/HFPI-side bundles).

    Required attributes
    -------------------
    positions : array (N, 3)
        Per-element ``(x, y, z)`` coordinate in metres.
    directions : array (N, 3)
        Per-element ``(L, M, N)`` unit direction cosines.

    :class:`lumenairy.raytrace.RayBundle` is NOT structurally a
    BundleProtocol -- it uses per-component arrays
    (``x, y, z, L, M, N``).  Convert via :func:`ray_to_path` or
    :func:`ray_to_beamlet` to obtain a BundleProtocol-compatible
    object.
    """

    positions: object
    directions: object


def _raybundle_positions(ray_bundle: 'RayBundle') -> np.ndarray:
    """Stack ``RayBundle.(x, y, z)`` into an ``(N, 3)`` positions
    array.  Helper for the converters below; not part of the public
    API."""
    return np.stack([np.asarray(ray_bundle.x),
                     np.asarray(ray_bundle.y),
                     np.asarray(ray_bundle.z)], axis=-1)


def _raybundle_directions(ray_bundle: 'RayBundle') -> np.ndarray:
    """Stack ``RayBundle.(L, M, N)`` into an ``(N, 3)`` directions
    array.  Helper for the converters below; not part of the public
    API."""
    return np.stack([np.asarray(ray_bundle.L),
                     np.asarray(ray_bundle.M),
                     np.asarray(ray_bundle.N)], axis=-1)


def ray_to_path(
    ray_bundle: 'RayBundle',
    *,
    weights: Optional[np.ndarray] = None,
) -> 'PathBundle':
    """Convert a :class:`RayBundle` into a :class:`PathBundle`.

    The ray bundle's geometric state (positions, directions, OPL,
    alive) is preserved; ``weights`` is added as a complex
    amplitude per path.  If ``weights`` is None, all paths are
    initialised with weight 1+0j.

    This is the standard hand-off when switching from a geometric
    raytrace to an HFPI Monte-Carlo accumulation.
    """
    from ..propagators.hfpi import PathBundle

    positions = _raybundle_positions(ray_bundle)
    directions = _raybundle_directions(ray_bundle)
    n = positions.shape[0]
    if weights is None:
        weights = np.ones(n, dtype=np.complex128)
    return PathBundle(
        positions=positions,
        directions=directions,
        weights=weights,
        opl=np.asarray(getattr(ray_bundle, 'opd', np.zeros(n))),
        alive=np.asarray(getattr(ray_bundle, 'alive',
                                  np.ones(n, dtype=bool))),
    )


def ray_to_beamlet(
    ray_bundle: 'RayBundle',
    *,
    wavelength: float,
    waist0: float,
    amplitude: Optional[np.ndarray] = None,
) -> 'BeamletBundle':
    """Convert a :class:`RayBundle` into a :class:`BeamletBundle`.

    Each ray becomes the central base ray of a Gaussian beamlet
    with waist ``waist0`` (metres) at its current position.  The
    complex Q-parameter is initialised to ``-i / z_R`` (waist
    plane) where ``z_R = pi w0^2 / lambda`` is the Rayleigh range.

    Use this when switching from a geometric raytrace into a GBD
    coherent-recombination workflow.
    """
    from ..propagators.gbd import BeamletBundle

    positions = _raybundle_positions(ray_bundle)
    directions = _raybundle_directions(ray_bundle)
    n = positions.shape[0]
    z_R = float(np.pi) * (waist0 ** 2) / wavelength
    Q = np.full(n, -1j / z_R, dtype=np.complex128)
    if amplitude is None:
        amplitude = np.ones(n, dtype=np.complex128)
    waist0_arr = np.full(n, float(waist0))
    return BeamletBundle(
        positions=positions,
        directions=directions,
        Q=Q,
        amplitude=amplitude,
        waist0=waist0_arr,
    )


def path_to_ray(path_bundle: 'PathBundle') -> 'RayBundle':
    """Discard the complex weight of a :class:`PathBundle` and
    return its geometric :class:`RayBundle` core.

    Useful for piping HFPI paths into a downstream geometric raytrace
    (where complex amplitude is irrelevant).  ``PathBundle.opl`` maps
    to ``RayBundle.opd``; both are accumulated optical path length in
    metres.  ``PathBundle`` does not carry a wavelength, so the
    returned ``RayBundle`` is constructed with ``wavelength=0.0``
    unless the path bundle has a ``wavelength`` attribute (a forward-
    compatible fallback for future schemas).
    """
    from .core import RayBundle

    positions = np.asarray(path_bundle.positions)
    directions = np.asarray(path_bundle.directions)
    n = positions.shape[0]
    x = positions[..., 0]
    y = positions[..., 1]
    z = positions[..., 2]
    L = directions[..., 0]
    M = directions[..., 1]
    N = directions[..., 2]
    opl = np.asarray(getattr(path_bundle, 'opl', np.zeros(n)))
    alive = np.asarray(getattr(path_bundle, 'alive',
                                np.ones(n, dtype=bool)))
    return RayBundle(
        x=x, y=y, z=z, L=L, M=M, N=N,
        wavelength=float(getattr(path_bundle, 'wavelength', 0.0)),
        alive=alive,
        opd=opl,
    )


__all__ = [
    'BundleProtocol',
    'ray_to_path',
    'ray_to_beamlet',
    'path_to_ray',
]

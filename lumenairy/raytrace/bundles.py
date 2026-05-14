"""
lumenairy.raytrace.bundles -- common protocol for ray-style bundles
plus inter-bundle conversion utilities.

Three bundle types share a vocabulary in lumenairy:

* :class:`lumenairy.raytrace.RayBundle` -- pure geometric rays
  with positions / directions / OPL / alive flag.
* :class:`lumenairy.propagators.hfpi.PathBundle` -- ray bundle
  with a complex amplitude weight per path (HFPI Monte-Carlo).
* :class:`lumenairy.propagators.gbd.BeamletBundle` -- ray bundle
  with a complex beam parameter (Q) and amplitude per beamlet
  (Gaussian Beamlet Decomposition).

All three carry ``positions``, ``directions``, and a per-element
``alive`` mask.  This module formalises that contract via
:class:`BundleProtocol` and provides conversion helpers so users
can switch between propagation methods mid-pipeline.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    # Forward references for type-checking only.  These imports cause
    # a circular dependency at runtime (RayBundle lives in
    # :mod:`lumenairy.raytrace.core` which itself depends on this
    # module), so they are only loaded when a type checker
    # (mypy / pyright / pylance) is processing the file.
    from .core import RayBundle
    from ..propagators.hfpi import PathBundle
    from ..propagators.gbd import BeamletBundle


@runtime_checkable
class BundleProtocol(Protocol):
    """Structural type satisfied by every ray-style bundle in
    lumenairy: ``RayBundle``, ``PathBundle``, ``BeamletBundle``.

    Required attributes
    -------------------
    positions : array (N, 3)
        Per-element ``(x, y, z)`` coordinate in metres.
    directions : array (N, 3)
        Per-element ``(L, M, N)`` unit direction cosines.

    Other attributes (``opl``, ``alive``, ``weights``,
    ``amplitude``, ``Q``, ``waist0``) are optional and bundle-type
    specific.
    """

    positions: object
    directions: object


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

    n = ray_bundle.positions.shape[0]
    if weights is None:
        weights = np.ones(n, dtype=np.complex128)
    return PathBundle(
        positions=ray_bundle.positions,
        directions=ray_bundle.directions,
        weights=weights,
        opl=getattr(ray_bundle, 'opl', np.zeros(n)),
        alive=getattr(ray_bundle, 'alive', np.ones(n, dtype=bool)),
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

    n = ray_bundle.positions.shape[0]
    z_R = float(np.pi) * (waist0 ** 2) / wavelength
    Q = np.full(n, -1j / z_R, dtype=np.complex128)
    if amplitude is None:
        amplitude = np.ones(n, dtype=np.complex128)
    waist0_arr = np.full(n, float(waist0))
    return BeamletBundle(
        positions=ray_bundle.positions,
        directions=ray_bundle.directions,
        Q=Q,
        amplitude=amplitude,
        waist0=waist0_arr,
    )


def path_to_ray(path_bundle: 'PathBundle') -> 'RayBundle':
    """Discard the complex weight of a :class:`PathBundle` and
    return its geometric :class:`RayBundle` core.

    Useful for piping HFPI paths into a downstream geometric raytrace
    (where complex amplitude is irrelevant).
    """
    from .core import RayBundle

    n = path_bundle.positions.shape[0]
    return RayBundle(
        positions=path_bundle.positions,
        directions=path_bundle.directions,
        opl=getattr(path_bundle, 'opl', np.zeros(n)),
        alive=getattr(path_bundle, 'alive', np.ones(n, dtype=bool)),
        wavelength=getattr(path_bundle, 'wavelength', 0.0),
    )


__all__ = [
    'BundleProtocol',
    'ray_to_path',
    'ray_to_beamlet',
    'path_to_ray',
]

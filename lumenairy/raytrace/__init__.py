"""
lumenairy.raytrace -- geometric ray tracing.

Submodules:

* :mod:`lumenairy.raytrace.core` -- ``RayBundle``, ``Surface``,
  ``trace``, ``surfaces_from_prescription``, ABCD, paraxial trace,
  fan / ring / grid bundle factories.
* :mod:`lumenairy.raytrace.bundles` -- ``BundleProtocol`` shared
  type and ``ray_to_path`` / ``ray_to_beamlet`` / ``path_to_ray``
  inter-bundle conversion utilities.
"""
from . import core as _impl
globals().update({k: v for k, v in _impl.__dict__.items() if not k.startswith('__')})

# Bundle protocol + conversions
from .bundles import (
    BundleProtocol,
    ray_to_path,
    ray_to_beamlet,
    path_to_ray,
)

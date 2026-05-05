"""
lumenairy.propagators -- diffraction-propagator family.

Subpackage grouping every plane-to-plane propagator the library
ships:

* :mod:`lumenairy.propagators.propagation` -- free-space ASM,
  Fresnel, Fraunhofer, Rayleigh-Sommerfeld.
* :mod:`lumenairy.propagators.asymptotic` -- phase-space asymptotic
  / LG aberration tensor.
* :mod:`lumenairy.propagators.gbd` -- Gaussian Beamlet Decomposition.
* :mod:`lumenairy.propagators.hfpi` -- Huygens-Fresnel Path
  Integration (Monte Carlo).
* :mod:`lumenairy.propagators.hf` -- Van-Vleck-corrected
  Huygens-Fresnel.
* :mod:`lumenairy.propagators.dispatch` -- top-level smart-method
  ``propagate(...)``.
* :mod:`lumenairy.propagators.subaperture` -- patch decomposition.

The :mod:`propagation` and :mod:`asymptotic` sub-modules are NOT
eagerly re-exported here -- they have heavy circular dependencies
with :mod:`lumenairy.lenses` and :mod:`lumenairy.raytrace` and are
accessed via the top-level shims at ``lumenairy.propagation`` and
``lumenairy.asymptotic`` (or directly via
``lumenairy.propagators.propagation``).

The new propagators -- ``gbd``, ``hfpi``, ``subaperture`` -- have
no such cycle and are re-exported directly here.

Author: Andrew Traverso
"""

from __future__ import annotations

# New propagators (no cycles).
from .gbd import (
    BeamletBundle,
    decompose_field_to_beamlets,
    propagate_beamlets_freespace,
    apply_thin_lens_to_beamlets,
    reconstruct_field_from_beamlets,
    propagate_gbd_freespace,
    propagate_gbd_thin_lens,
)
from .hfpi import (
    PathBundle,
    init_paths_from_field,
    propagate_to_plane,
    apply_aperture_diffraction,
    accumulate_to_grid,
    propagate_hfpi_freespace_aperture,
)
from .subaperture import (
    PatchGrid,
    patches_for_box,
    patch_window,
    combine_patch_fields,
)
from .hf import (
    propagate_huygens_fresnel_freespace,
    propagate_huygens_fresnel_with_opl_callable,
)
from .dispatch import (
    propagate,
    VALID_METHODS,
)

__all__ = [
    'BeamletBundle',
    'decompose_field_to_beamlets',
    'propagate_beamlets_freespace',
    'apply_thin_lens_to_beamlets',
    'reconstruct_field_from_beamlets',
    'propagate_gbd_freespace',
    'propagate_gbd_thin_lens',
    'PathBundle',
    'init_paths_from_field',
    'propagate_to_plane',
    'apply_aperture_diffraction',
    'accumulate_to_grid',
    'propagate_hfpi_freespace_aperture',
    'PatchGrid',
    'patches_for_box',
    'patch_window',
    'combine_patch_fields',
    'propagate_huygens_fresnel_freespace',
    'propagate_huygens_fresnel_with_opl_callable',
    'propagate',
    'VALID_METHODS',
]

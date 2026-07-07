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

from .dispatch import (
    ASM_FAMILY,
    VALID_METHODS,
    asm_propagate,
    propagate,
    which_propagator,
)

# v4.16.1 (audit AUDIT_V4_16_0_DEEP P5/P0-1): partial-coherence
# ensemble propagator helper.  Closes the half-shipped Schell-model
# workflow by giving callers a one-line "propagate this ensemble and
# give me the partial-coherence intensity" entry point.
from .ensemble import propagate_ensemble

# New propagators (no cycles).
from .gbd import (
    BeamletBundle,
    apply_abcd_to_beamlets,
    apply_aperture_to_beamlets,
    apply_thin_lens_to_beamlets,
    asm_field_to_gbd,
    converge_gbd_sampling,
    csp_beamlet_field,
    decompose_field_adaptive,
    decompose_field_to_beamlets,
    gbd_asm_gouy_phase,
    gbd_field_to_asm,
    match_global_phase,
    propagate_beamlets_freespace,
    propagate_gbd,
    propagate_gbd_freespace,
    propagate_gbd_freespace_csp,
    propagate_gbd_thin_lens,
    propagate_gbd_through_prescription,
    reconstruct_field_from_beamlets,
    reconstruct_vector_field_with_ez,
)
from .hf import (
    propagate_huygens_fresnel,
    propagate_huygens_fresnel_freespace,
    propagate_huygens_fresnel_through_prescription,
    propagate_huygens_fresnel_with_opl_callable,
)
from .hfpi import (
    PathBundle,
    accumulate_to_grid,
    apply_aperture_diffraction,
    init_paths_from_field,
    init_paths_stratified,
    propagate_hfpi,
    propagate_hfpi_freespace_aperture,
    propagate_hfpi_through_prescription,
    propagate_to_plane,
)
from .mhs import (
    HuygensSurface,
    MhsPipeline,
    MhsSubdomain,
    aperture_subdomain,
    asm_subdomain,
    gbd_freespace_subdomain,
    prescription_subdomain,
)

# Unified result container (opt-in; native return shapes preserved).
from .result import PropagationResult
from .subaperture import (
    PatchGrid,
    combine_patch_fields,
    patch_window,
    patches_for_box,
    propagate_subaperture_asymptotic,
)
from .vectorial_hfpi import (
    VectorPathBundle,
    accumulate_vector_to_grid,
    apply_vector_aperture_diffraction,
    init_vector_paths_from_field,
    propagate_vector_hfpi_freespace_aperture,
    propagate_vector_to_plane,
)

__all__ = [
    'PropagationResult',
    'BeamletBundle',
    'decompose_field_to_beamlets',
    'decompose_field_adaptive',
    'converge_gbd_sampling',
    'csp_beamlet_field',
    'propagate_gbd_freespace_csp',
    'apply_aperture_to_beamlets',
    'gbd_asm_gouy_phase',
    'gbd_field_to_asm',
    'match_global_phase',
    'asm_field_to_gbd',
    'propagate_beamlets_freespace',
    'apply_thin_lens_to_beamlets',
    'reconstruct_field_from_beamlets',
    'reconstruct_vector_field_with_ez',
    'apply_abcd_to_beamlets',
    'propagate_gbd',
    'propagate_gbd_freespace',
    'propagate_gbd_thin_lens',
    'propagate_gbd_through_prescription',
    'PathBundle',
    'init_paths_from_field',
    'init_paths_stratified',
    'propagate_to_plane',
    'apply_aperture_diffraction',
    'accumulate_to_grid',
    'propagate_hfpi',
    'propagate_hfpi_freespace_aperture',
    'propagate_hfpi_through_prescription',
    'PatchGrid',
    'patches_for_box',
    'patch_window',
    'combine_patch_fields',
    'propagate_subaperture_asymptotic',
    'propagate_huygens_fresnel',
    'propagate_huygens_fresnel_freespace',
    'propagate_huygens_fresnel_with_opl_callable',
    'propagate_huygens_fresnel_through_prescription',
    'propagate',
    'VALID_METHODS',
    'asm_propagate',
    'which_propagator',
    'ASM_FAMILY',
    'propagate_ensemble',
    'VectorPathBundle',
    'init_vector_paths_from_field',
    'propagate_vector_to_plane',
    'apply_vector_aperture_diffraction',
    'accumulate_vector_to_grid',
    'propagate_vector_hfpi_freespace_aperture',
    'HuygensSurface',
    'MhsSubdomain',
    'MhsPipeline',
    'asm_subdomain',
    'aperture_subdomain',
    'gbd_freespace_subdomain',
    'prescription_subdomain',
]

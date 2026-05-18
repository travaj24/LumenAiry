"""
lumenairy.raytrace -- geometric ray tracing.

Submodules:

* :mod:`lumenairy.raytrace.core` -- ``RayBundle``, ``Surface``,
  ``trace``, ``surfaces_from_prescription``, ABCD, paraxial trace,
  fan / ring / grid bundle factories.
* :mod:`lumenairy.raytrace.bundles` -- ``BundleProtocol`` shared
  type and inter-bundle conversion utilities.
"""

from .core import (
    RayBundle,
    Surface,
    TraceResult,
    trace,
    trace_world,
    surfaces_from_prescription,
    validate_prescription,
    make_ray,
    make_fan,
    make_ring,
    make_grid,
    make_rings,
    apply_doe_phase_traced,
    trace_prescription,
    system_abcd,
    system_abcd_prescription,
    seidel_coefficients,
    seidel_prescription,
    spot_rms,
    spot_geo_radius,
    spot_diagram,
    ray_fan_data,
    ray_fan_plot,
    ray_fan_plot_prescription,
    opd_fan_data,
    ray_fan_data_world,
    opd_fan_data_world,
    through_focus_rms,
    refocus,
    find_stop,
    compute_pupils,
    lens_abcd,
    find_lenses,
    LensInfo,
    PupilInfo,
    FirstOrderData,
    first_order_data,
    RAY_OK,
    RAY_TIR,
    RAY_APERTURE,
    RAY_MISSED_SURFACE,
    RAY_NAN,
    RAY_EVANESCENT,
    find_paraxial_focus,
    trace_summary,
    prescription_summary,
    surfaces_from_elements,
    raytrace_system,
    # Private helpers needed by other subpackages (asymptotic uses
    # _make_bundle as a parametric ray-bundle constructor).
    _make_bundle,
)
from .bundles import (
    BundleProtocol,
    ray_to_path,
    ray_to_beamlet,
    path_to_ray,
)

# JAX-traceable trace (functional / immutable; differentiable via
# jax.grad and JIT-able via jax.jit).
from .jax_trace import (
    JaxRayState,
    make_jax_ray_state,
    trace_jax,
    jax_state_to_raybundle,
)

# Paraxial-design one-liner helpers.
from .paraxial import (
    field_of_view,
    optical_invariant,
    f_number,
    defocus_waves_to_zernike,
    astigmatism_waves_to_zernike,
)

# Field-dependent Seidel analysis (4.3.0).
from .seidel_analysis import (
    seidel_field_sweep,
    seidel_wfe,
)

# World-frame surface builder for folded prescriptions (4.4.0)
# plus paraxial-focus world helper (4.5.0).
from .world import (
    world_surfaces_from_prescription,
    paraxial_focus_world,
)

# v4.15.1 (Cluster B Item 6): bridge a coherent field into a RayBundle.
from .from_field import rays_from_field


__all__ = [
    'RayBundle',
    'Surface',
    'TraceResult',
    'trace',
    'surfaces_from_prescription',
    'validate_prescription',
    'make_ray',
    'make_fan',
    'make_ring',
    'make_grid',
    'make_rings',
    'apply_doe_phase_traced',
    'trace_prescription',
    'system_abcd',
    'system_abcd_prescription',
    'seidel_coefficients',
    'seidel_prescription',
    'seidel_field_sweep',
    'seidel_wfe',
    'world_surfaces_from_prescription',
    'paraxial_focus_world',
    'spot_rms',
    'spot_geo_radius',
    'spot_diagram',
    'ray_fan_data',
    'ray_fan_plot',
    'ray_fan_plot_prescription',
    'opd_fan_data',
    'through_focus_rms',
    'refocus',
    'find_stop',
    'compute_pupils',
    'lens_abcd',
    'find_lenses',
    'LensInfo',
    'PupilInfo',
    'FirstOrderData',
    'first_order_data',
    'RAY_OK',
    'RAY_TIR',
    'RAY_APERTURE',
    'RAY_MISSED_SURFACE',
    'RAY_NAN',
    'RAY_EVANESCENT',
    'find_paraxial_focus',
    'trace_summary',
    'prescription_summary',
    'surfaces_from_elements',
    'raytrace_system',
    'BundleProtocol',
    'ray_to_path',
    'ray_to_beamlet',
    'path_to_ray',
    'JaxRayState',
    'make_jax_ray_state',
    'trace_jax',
    'jax_state_to_raybundle',
    # paraxial helpers
    'field_of_view',
    'optical_invariant',
    'f_number',
    'defocus_waves_to_zernike',
    'astigmatism_waves_to_zernike',
    # v4.15.1 wave -> ray bridge (Cluster B Item 6)
    'rays_from_field',
]

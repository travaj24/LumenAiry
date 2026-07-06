"""
lumenairy.raytrace -- geometric ray tracing.

Submodules:

* :mod:`lumenairy.raytrace.core` -- ``RayBundle``, ``Surface``,
  ``trace``, ``surfaces_from_prescription``, ABCD, paraxial trace,
  fan / ring / grid bundle factories.
* :mod:`lumenairy.raytrace.bundles` -- ``BundleProtocol`` shared
  type and inter-bundle conversion utilities.
"""

from .bundles import (
    BundleProtocol,
    path_to_ray,
    ray_to_beamlet,
    ray_to_path,
)
from .core import (  # noqa: E402  (continuation of the .core import list)
    RAY_APERTURE,
    RAY_EVANESCENT,
    RAY_MISSED_SURFACE,
    RAY_NAN,
    RAY_OK,
    RAY_TIR,
    FirstOrderData,
    LensInfo,
    PupilInfo,
    RayBundle,
    Surface,
    TraceResult,
    # Private helpers needed by other subpackages (asymptotic uses
    # _make_bundle as a parametric ray-bundle constructor).
    _make_bundle,
    apply_doe_phase_traced,
    compute_pupils,
    find_lenses,
    find_paraxial_focus,
    find_stop,
    first_order_data,
    lens_abcd,
    make_fan,
    make_grid,
    make_ray,
    make_ring,
    make_rings,
    opd_fan_data,
    opd_fan_data_world,
    prescription_summary,
    ray_fan_data,
    ray_fan_data_world,
    ray_fan_plot,
    ray_fan_plot_prescription,
    raytrace_system,
    refocus,
    seidel_coefficients,
    seidel_prescription,
    spot_diagram,
    spot_geo_radius,
    spot_rms,
    surfaces_from_elements,
    surfaces_from_prescription,
    system_abcd,
    system_abcd_prescription,
    through_focus_rms,
    trace,
    trace_prescription,
    trace_summary,
    trace_world,
    validate_prescription,
)
from .differential import (  # noqa: E402
    DifferentialTransfer,
    ray_transfer_jacobian,
    ray_transfer_jacobian_analytic,
    ray_transfer_jacobian_jax,
)

# v4.15.1 (Cluster B Item 6): bridge a coherent field into a RayBundle.
from .from_field import rays_from_field

# JAX-traceable trace (functional / immutable; differentiable via
# jax.grad and JIT-able via jax.jit).
from .jax_trace import (
    JaxRayState,
    jax_state_to_raybundle,
    make_jax_ray_state,
    trace_jax,
)

# Paraxial-design one-liner helpers.
from .paraxial import (
    astigmatism_waves_to_zernike,
    defocus_waves_to_zernike,
    f_number,
    field_of_view,
    optical_invariant,
)

# Field-dependent Seidel analysis (4.3.0).
from .seidel_analysis import (
    seidel_field_sweep,
    seidel_wfe,
)

# World-frame surface builder for folded prescriptions (4.4.0)
# plus paraxial-focus world helper (4.5.0).
from .world import (
    paraxial_focus_world,
    world_surfaces_from_prescription,
)

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
    'ray_transfer_jacobian',
    'ray_transfer_jacobian_analytic',
    'ray_transfer_jacobian_jax',
    'DifferentialTransfer',
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

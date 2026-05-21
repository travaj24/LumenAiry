"""
lumenairy.analysis -- analysis & post-processing tools.

Submodules:

* :mod:`lumenairy.analysis.analysis` -- core beam analysis (Strehl,
  PSF, MTF, OTF, Zernike decomposition, OPD).
* :mod:`lumenairy.analysis.detector` -- detector models +
  Shack-Hartmann wavefront sensing.
* :mod:`lumenairy.analysis.ghost` -- ghost / parasitic-reflection
  enumeration and analysis.
* :mod:`lumenairy.analysis.interferometry` -- simulated
  interferograms, phase-shift extraction.
* :mod:`lumenairy.analysis.phase_retrieval` -- Gerchberg-Saxton,
  error reduction, hybrid input-output.
* :mod:`lumenairy.analysis.coherence` -- partially-coherent /
  Kohler / extended-source imaging.
* :mod:`lumenairy.analysis.through_focus` -- through-focus scans,
  best-focus search, tolerancing sweeps.
* :mod:`lumenairy.analysis.plotting` -- field / PSF / MTF / Stokes
  / Jones-pupil plotting helpers.
"""

from .aberration import (
    AberrationSummary,
    CausticDiagnostic,
    aberration_summary,
    caustic_diagnostic,
    format_aberration_summary,
    plot_caustic_diagnostic,
)
from .ao import (
    DeformableMirror,
    LeakyIntegrator,
    apply_dm,
    slope_to_modal,
    zernike_modal_basis,
)
from .coherence import (
    extended_source_image,
    koehler_image,
    mutual_coherence,
)
from .core import (
    M2,
    astigmatism_mag_angle,
    beam_centroid,
    beam_d4sigma,
    beam_diameter,
    beam_power,
    check_opd_sampling,
    check_sampling_conditions,
    chromatic_focal_shift,
    clear_zernike_basis_cache,
    compute_mtf,
    compute_otf,
    compute_psf,
    coupling_efficiency,
    coupling_efficiency_vector,
    depth_of_focus,
    ee_polychromatic,
    encircled_energy_curve,
    encircled_energy_radius,
    fwhm_resolution,
    mtf_cutoff,
    mtf_radial,
    opd_pv_rms,
    polychromatic_psf,
    polychromatic_strehl,
    radial_power_bands,
    rayleigh_resolution,
    remove_wavefront_modes,
    sparrow_resolution,
    strehl_marechal,
    strehl_phase_integral,
    strehl_ratio,
    strehl_vector,
    wave_opd_1d,
    wave_opd_2d,
    zernike_basis_matrix,
    zernike_decompose,
    zernike_index_to_nm,
    zernike_nm_to_index,
    zernike_polynomial,
    zernike_reconstruct,
)
from .detector import (
    apply_detector,
    shack_hartmann,
)
from .field import (
    DistortionGrid,
    DistortionVsField,
    FieldAberrationSweep,
    FieldFootprint,
    RelativeIllumination,
    SensitivityResult,
    SpotDiagramField,
    SurfaceFootprint,
    distortion_grid,
    distortion_vs_field,
    field_aberration_sweep,
    footprint_per_surface,
    petzval_radius,
    relative_illumination,
    sensitivity_ranking,
    spot_diagram_vs_field,
)
from .ghost import (
    enumerate_ghost_paths,
    ghost_analysis,
    non_sequential_stray_light,
)
from .image_plane_wfe import (
    ImagePlaneWFE,
    chebyshev_pupil_grid,
    eval_image_plane_wfe,
    field_grid_wfe,
    remove_low_order_aberrations,
    zemax_pupil_grid,
)
from .interferometry import (
    fringe_spacing,
    phase_shift_extract,
    simulate_interferogram,
)
from .phase_retrieval import (
    clear_phase_retrieval_caches,
    error_reduction,
    error_reduction_jax,
    gerchberg_saxton,
    gerchberg_saxton_jax,
    hybrid_input_output,
    hybrid_input_output_jax,
)
from .plotting import (
    abbe_diagram,
    compute_jones_pupil,
    plot_amplitude_phase,
    plot_beam_profile,
    plot_cross_section,
    plot_field,
    plot_glass_map,
    plot_intensity,
    plot_jones_pupil,
    plot_lens_layout,
    plot_mtf,
    plot_opd_fan,
    plot_opd_summary,
    plot_phase,
    plot_planes_grid,
    plot_polarization_ellipses,
    plot_psf,
    plot_stokes,
    plot_wavefront,
)
from .through_focus import (
    Perturbation,
    ThroughFocusResult,
    apply_perturbations,
    diffraction_limited_peak,
    find_best_focus,
    monte_carlo_tolerancing,
    monte_carlo_tolerancing_jax,
    monte_carlo_tolerancing_linearized,
    plot_through_focus,
    single_plane_metrics,
    through_focus_scan,
    through_focus_scan_jax,
    tolerancing_report,
    tolerancing_sweep,
)

__all__ = [
    # analysis
    'beam_centroid', 'beam_d4sigma', 'beam_diameter', 'beam_power',
    'strehl_ratio', 'strehl_marechal', 'strehl_phase_integral',
    'strehl_vector',
    'coupling_efficiency', 'coupling_efficiency_vector', 'M2',
    'caustic_diagnostic', 'plot_caustic_diagnostic', 'CausticDiagnostic',
    'check_sampling_conditions', 'compute_psf', 'compute_otf',
    'compute_mtf', 'mtf_radial', 'mtf_cutoff',
    'encircled_energy_curve', 'encircled_energy_radius',
    'ee_polychromatic',
    'rayleigh_resolution', 'sparrow_resolution', 'fwhm_resolution',
    'depth_of_focus',
    'remove_wavefront_modes',
    'opd_pv_rms', 'wave_opd_1d', 'wave_opd_2d', 'check_opd_sampling',
    'chromatic_focal_shift', 'polychromatic_strehl',
    'polychromatic_psf',
    'radial_power_bands',
    'zernike_polynomial', 'zernike_basis_matrix', 'zernike_decompose',
    'zernike_reconstruct', 'zernike_index_to_nm', 'zernike_nm_to_index',
    'astigmatism_mag_angle',
    'clear_zernike_basis_cache',
    # detector
    'apply_detector', 'shack_hartmann',
    # ghost
    'enumerate_ghost_paths', 'ghost_analysis', 'non_sequential_stray_light',
    # interferometry
    'simulate_interferogram', 'phase_shift_extract', 'fringe_spacing',
    # phase retrieval
    'gerchberg_saxton', 'error_reduction', 'hybrid_input_output',
    'gerchberg_saxton_jax', 'error_reduction_jax', 'hybrid_input_output_jax',
    'clear_phase_retrieval_caches',
    # coherence
    'koehler_image', 'extended_source_image', 'mutual_coherence',
    # through focus
    'single_plane_metrics', 'diffraction_limited_peak',
    'through_focus_scan', 'through_focus_scan_jax', 'find_best_focus',
    'plot_through_focus',
    'ThroughFocusResult', 'Perturbation', 'apply_perturbations',
    'tolerancing_sweep', 'monte_carlo_tolerancing',
    'monte_carlo_tolerancing_jax',
    'monte_carlo_tolerancing_linearized',
    'tolerancing_report',
    # aberration (unified Seidel + LG tensor)
    'AberrationSummary', 'aberration_summary', 'format_aberration_summary',
    # image-plane wavefront error (3.8.0 / off-axis + field-grid 4.0)
    'ImagePlaneWFE', 'eval_image_plane_wfe', 'field_grid_wfe',
    'zemax_pupil_grid', 'chebyshev_pupil_grid',
    'remove_low_order_aberrations',
    # plotting
    'plot_intensity', 'plot_phase', 'plot_field', 'plot_amplitude_phase',
    'plot_cross_section', 'plot_planes_grid', 'plot_psf', 'plot_mtf',
    'plot_stokes', 'plot_polarization_ellipses', 'plot_beam_profile',
    'plot_wavefront',
    'plot_opd_fan', 'plot_opd_summary',
    'plot_jones_pupil', 'compute_jones_pupil',
    'plot_lens_layout',
    'abbe_diagram', 'plot_glass_map',
    # adaptive optics (moved here in 4.3.0; lumenairy.ao still works
    # via a shim for back-compat)
    'DeformableMirror', 'apply_dm',
    'zernike_modal_basis', 'slope_to_modal',
    'LeakyIntegrator',
    # field-resolved analyses (4.4.0; lifted from ui/*_dock.py)
    'DistortionVsField', 'distortion_vs_field',
    'DistortionGrid', 'distortion_grid',
    'SurfaceFootprint', 'FieldFootprint', 'footprint_per_surface',
    'SpotDiagramField', 'spot_diagram_vs_field',
    'RelativeIllumination', 'relative_illumination',
    'FieldAberrationSweep', 'field_aberration_sweep',
    'petzval_radius',
    'SensitivityResult', 'sensitivity_ranking',
]

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

from .analysis import (
    beam_centroid,
    beam_d4sigma,
    beam_power,
    strehl_ratio,
    coupling_efficiency,
    M2,
    check_sampling_conditions,
    compute_psf,
    compute_otf,
    compute_mtf,
    mtf_radial,
    remove_wavefront_modes,
    opd_pv_rms,
    wave_opd_1d,
    wave_opd_2d,
    check_opd_sampling,
    chromatic_focal_shift,
    polychromatic_strehl,
    polychromatic_psf,
    radial_power_bands,
    zernike_polynomial,
    zernike_basis_matrix,
    zernike_decompose,
    zernike_reconstruct,
    zernike_index_to_nm,
    zernike_nm_to_index,
)
from .detector import (
    apply_detector,
    shack_hartmann,
)
from .ghost import (
    enumerate_ghost_paths,
    ghost_analysis,
    non_sequential_stray_light,
)
from .interferometry import (
    simulate_interferogram,
    phase_shift_extract,
    fringe_spacing,
)
from .phase_retrieval import (
    gerchberg_saxton,
    error_reduction,
    hybrid_input_output,
    gerchberg_saxton_jax,
    error_reduction_jax,
    hybrid_input_output_jax,
)
from .coherence import (
    koehler_image,
    extended_source_image,
    mutual_coherence,
)
from .through_focus import (
    single_plane_metrics,
    diffraction_limited_peak,
    through_focus_scan,
    through_focus_scan_jax,
    find_best_focus,
    plot_through_focus,
    ThroughFocusResult,
    Perturbation,
    apply_perturbations,
    tolerancing_sweep,
    monte_carlo_tolerancing,
    monte_carlo_tolerancing_jax,
    monte_carlo_tolerancing_linearized,
    tolerancing_report,
)
from .aberration import (
    AberrationSummary,
    aberration_summary,
    format_aberration_summary,
    CausticDiagnostic,
    caustic_diagnostic,
    plot_caustic_diagnostic,
)
from .image_plane_wfe import (
    ImagePlaneWFE,
    eval_image_plane_wfe,
    remove_low_order_aberrations,
)
from .plotting import (
    plot_intensity,
    plot_phase,
    plot_field,
    plot_amplitude_phase,
    plot_cross_section,
    plot_planes_grid,
    plot_psf,
    plot_mtf,
    plot_stokes,
    plot_polarization_ellipses,
    plot_beam_profile,
    plot_jones_pupil,
    compute_jones_pupil,
)


__all__ = [
    # analysis
    'beam_centroid', 'beam_d4sigma', 'beam_power', 'strehl_ratio',
    'coupling_efficiency', 'M2',
    'caustic_diagnostic', 'plot_caustic_diagnostic', 'CausticDiagnostic',
    'check_sampling_conditions', 'compute_psf', 'compute_otf',
    'compute_mtf', 'mtf_radial', 'remove_wavefront_modes',
    'opd_pv_rms', 'wave_opd_1d', 'wave_opd_2d', 'check_opd_sampling',
    'chromatic_focal_shift', 'polychromatic_strehl',
    'polychromatic_psf',
    'radial_power_bands',
    'zernike_polynomial', 'zernike_basis_matrix', 'zernike_decompose',
    'zernike_reconstruct', 'zernike_index_to_nm', 'zernike_nm_to_index',
    # detector
    'apply_detector', 'shack_hartmann',
    # ghost
    'enumerate_ghost_paths', 'ghost_analysis', 'non_sequential_stray_light',
    # interferometry
    'simulate_interferogram', 'phase_shift_extract', 'fringe_spacing',
    # phase retrieval
    'gerchberg_saxton', 'error_reduction', 'hybrid_input_output',
    'gerchberg_saxton_jax', 'error_reduction_jax', 'hybrid_input_output_jax',
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
    # image-plane wavefront error (3.8.0)
    'ImagePlaneWFE', 'eval_image_plane_wfe', 'remove_low_order_aberrations',
    # plotting
    'plot_intensity', 'plot_phase', 'plot_field', 'plot_amplitude_phase',
    'plot_cross_section', 'plot_planes_grid', 'plot_psf', 'plot_mtf',
    'plot_stokes', 'plot_polarization_ellipses', 'plot_beam_profile',
    'plot_jones_pupil', 'compute_jones_pupil',
]

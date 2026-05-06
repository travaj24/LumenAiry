"""
lumenairy — Coherent Optical Field Propagation Library
================================================================

A comprehensive library for simulating coherent optical beam propagation
using the Angular Spectrum Method (ASM) and related techniques.

Convention: exp(-i*omega*t) time dependence throughout.
Units: SI meters for all spatial quantities.

Usage::

    from lumenairy import angular_spectrum_propagate, apply_thin_lens
    # or
    import lumenairy as la
    E_out = la.angular_spectrum_propagate(E_in, z, wavelength, dx)

All public functions are available directly from the package namespace.
For more granular imports, use the submodules::

    from lumenairy.propagators.propagation import angular_spectrum_propagate
    from lumenairy.elements.lenses import apply_real_lens
    from lumenairy.glass import get_glass_index, GLASS_REGISTRY

Author: Andrew Traverso
"""

# ── Propagation ──────────────────────────────────────────────────────────
from .propagators.propagation import (
    angular_spectrum_propagate,
    angular_spectrum_propagate_tilted,
    scalable_angular_spectrum_propagate,
    fresnel_propagate,
    fraunhofer_propagate,
    set_default_complex_dtype,
    get_default_complex_dtype,
    set_asm_cache_size,
    set_fft_plan_cache_size,
    clear_asm_caches,
    reset_fft_backend,
    rayleigh_sommerfeld_propagate,
    resample_field,
    # FFT backend configuration
    PYFFTW_AVAILABLE,
    CUPY_AVAILABLE,
    set_fft_threads,
    set_fft_fallback,
    reset_fft_backend,
    # Precision configuration (complex64 vs complex128)
    DEFAULT_COMPLEX_DTYPE,
)

# ── Optional optimisation-backend availability flags ────────────────────
# Users / runners can inspect these before toggling tunables that
# depend on the optional backends (e.g. the numexpr-fused phase-screen
# path inside apply_real_lens).  Truthy if the package is importable
# in the current environment.
from .elements.lenses import NUMEXPR_AVAILABLE

# ── Backend / runtime helpers ────────────────────────────────────────────
from .memory import available_cpus

# ── Lenses ───────────────────────────────────────────────────────────────
from .elements.lenses import (
    apply_thin_lens,
    apply_spherical_lens,
    apply_aspheric_lens,
    apply_real_lens,
    apply_real_lens_traced,
    surface_sag_general,
    surface_sag_biconic,
    apply_cylindrical_lens,
    apply_grin_lens,
    apply_axicon,
    check_grid_vs_apertures,
    recommend_grid_for_prescription,
)
from .elements.lenses import (
    apply_real_lens_maslov,
    apply_real_lens_traced_jax,
    apply_real_lens_maslov_jax,
)

# ── Glass catalog ────────────────────────────────────────────────────────
from .glass import (
    get_glass_index,
    get_glass_index_complex,
    GLASS_REGISTRY,
)

# ── Optical elements ─────────────────────────────────────────────────────
from .elements import (
    apply_mirror,
    apply_aperture,
    apply_gaussian_aperture,
    apply_mask,
    zernike,
    apply_zernike_aberration,
    generate_turbulence_screen,
)

# ── Sources ──────────────────────────────────────────────────────────────
from .sources import (
    Source,
    create_gaussian_beam,
    create_hermite_gauss,
    create_laguerre_gauss,
    hermite_physicist,
    laguerre_generalized,
)

# ── Beam analysis ────────────────────────────────────────────────────────
from .analysis.aberration import (
    AberrationSummary,
    aberration_summary,
    format_aberration_summary,
)
from .analysis.phase_retrieval import (
    gerchberg_saxton_jax,
    error_reduction_jax,
    hybrid_input_output_jax,
)
from .analysis import (
    beam_centroid,
    beam_d4sigma,
    beam_power,
    strehl_ratio,
    coupling_efficiency,
    M2,
    caustic_diagnostic,
    plot_caustic_diagnostic,
    CausticDiagnostic,
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
    radial_power_bands,
    zernike_polynomial,
    zernike_basis_matrix,
    zernike_decompose,
    zernike_reconstruct,
    zernike_index_to_nm,
    zernike_nm_to_index,
)

# ── Off-axis + extended source helpers ─────────────────────────────────
from .sources import (
    create_tilted_plane_wave,
    create_point_source,
    create_multi_field_sources,
    create_top_hat_beam,
    create_annular_beam,
    create_fiber_mode,
    create_led_source,
    create_bessel_beam,
)

# ── High-NA vector diffraction (Richards-Wolf) ─────────────────────────
from .propagators.vector_diffraction import (
    richards_wolf_focus,
    debye_wolf_psf,
)

# ── Partial coherence / extended-source imaging ────────────────────────
from .analysis.coherence import (
    koehler_image,
    extended_source_image,
    mutual_coherence,
)

# ── Detector model / wavefront sensing ─────────────────────────────────
from .analysis.detector import (
    apply_detector,
    shack_hartmann,
)

# ── Thin-film coatings ────────────────────────────────────────────────
from .elements.coatings import (
    coating_reflectance,
    quarter_wave_ar,
    broadband_ar_v_coat,
)

# ── Interferometry ────────────────────────────────────────────────────
from .analysis.interferometry import (
    simulate_interferogram,
    phase_shift_extract,
    fringe_spacing,
)

# ── Freeform surfaces ─────────────────────────────────────────────────
from .elements.freeform import (
    surface_sag_xy_polynomial,
    surface_sag_zernike_freeform,
    surface_sag_chebyshev,
    surface_sag_freeform,
)

# ── Ghost analysis ────────────────────────────────────────────────────
from .analysis.ghost import (
    enumerate_ghost_paths,
    ghost_analysis,
)

# ── BSDF surface scatter (stray-light analysis) ─────────────────────────
from .elements.bsdf import (
    BSDFModel,
    LambertianBSDF,
    GaussianBSDF,
    HarveyShackBSDF,
    make_bsdf,
    sample_scatter_rays,
)

# ── RCWA (rigorous coupled-wave analysis) ─────────────────────────────
from .elements.rcwa import (
    rcwa_1d,
    grating_efficiency_vs_wavelength,
)

# ── Multi-configuration + afocal mode ─────────────────────────────────
from .optimize.multiconfig import (
    Configuration,
    multi_config_merit,
    create_zoom_configs,
    afocal_angular_magnification,
    beam_expander_prescription,
    keplerian_telescope,
)

# ── Through-focus / tolerancing ─────────────────────────────────────────
from .analysis.through_focus import (
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
    tolerancing_report,
)

# ── Hybrid wave/ray design optimization ────────────────────────────────
from .optimize import (
    DesignParameterization,
    MultiPrescriptionParameterization,
    MeritTerm,
    FocalLengthMerit,
    BackFocalLengthMerit,
    SphericalSeidelMerit,
    StrehlMerit,
    RMSWavefrontMerit,
    SpotSizeMerit,
    ChromaticFocalShiftMerit,
    MatchIdealThinLensMerit,
    MatchIdealSystemMerit,
    MatchTargetOPDMerit,
    ZernikeCoefficientMerit,
    LGAberrationMerit,
    CompositeMerit,
    CallableMerit,
    JaxMeritTerm,
    MultiWavelengthMerit,
    MultiFieldMerit,
    MinThicknessMerit,
    MaxThicknessMerit,
    MinBackFocalLengthMerit,
    MaxFNumberMerit,
    ToleranceAwareMerit,
    EvaluationContext,
    DesignResult,
    design_optimize,
    WAVE_PROPAGATOR_REGISTRY,
    register_wave_propagator,
    unregister_wave_propagator,
)

# ── Phase-space asymptotic propagator + LG aberration tensor ────────────
from .propagators.asymptotic import (
    CanonicalPolyFit,
    HFPolyFit,
    AberrationTensorResult,
    fit_canonical_polynomials,
    fit_hf_polynomials,
    aberration_tensor,
    propagate_modal_asymptotic,
    propagate_hf_chebyshev_quadrature,
    solve_envelope_stationary,
    lg_polynomial,
    hg_polynomial,
    evaluate_lg_mode,
    evaluate_hg_mode,
    decompose_lg,
    decompose_hg,
    lg_seidel_label,
    gaussian_moment_2d,
    gaussian_moment_table_2d,
    aberration_tensor_lg00_jax,
    propagate_modal_asymptotic_lg00_jax,
    JaxAberrationTensorResult,
    solve_envelope_stationary_jax_ift,
    fit_canonical_polynomials_jax,
)

# ── DOE / Gratings / Phase I/O ──────────────────────────────────────────
from .elements.doe import (
    create_periodic_phase_mask,
    create_microlens_array,
    makedammann2d,
    load_phase_file,
    save_phase_file,
    load_fits_field,
    save_fits_field,
)

# ── Phase retrieval ─────────────────────────────────────────────────────
from .analysis.phase_retrieval import (
    gerchberg_saxton,
    error_reduction,
    hybrid_input_output,
)

# ── Code generation: prescription -> standalone simulation script ───────
from .io.codegen import (
    generate_simulation_script,
    generate_script_from_zmx,
    generate_script_from_txt,
)

# ── Progress reporting (opt-in hook for long-running functions) ─────────
from .progress import ProgressCallback, ProgressScaler, call_progress

# ── Polarization / Jones calculus ───────────────────────────────────────
from .elements.polarization import (
    JonesField,
    apply_jones_matrix,
    apply_polarizer,
    apply_waveplate,
    apply_half_wave_plate,
    apply_quarter_wave_plate,
    apply_rotator,
    create_linear_polarized,
    create_circular_polarized,
    create_elliptical_polarized,
    stokes_parameters,
    degree_of_polarization,
    polarization_ellipse,
)

# ── Lens prescriptions ──────────────────────────────────────────────────
from .io.prescriptions import (
    make_singlet,
    make_doublet,
    make_cylindrical,
    make_biconic,
    thorlabs_lens,
    load_zmx_prescription,
    load_zemax_prescription_txt,
    export_zemax_lens_data,
    export_zemax_zmx,
    load_codev_seq,
    export_codev_seq,
    export_quadoa_qos,
    load_quadoa_qos,
    QUADOA_SCHEMA_VERSION,
    scale_prescription,
    THORLABS_CATALOG,
)

# ── Geometric ray tracing ────────────────────────────────────────────────
from .raytrace import (
    RayBundle,
    Surface,
    TraceResult,
    trace,
    surfaces_from_prescription,
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
    through_focus_rms,
    refocus,
    find_stop,
    compute_pupils,
    lens_abcd,
    find_lenses,
    LensInfo,
    PupilInfo,
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
)

# ── System propagation ──────────────────────────────────────────────────
from .system import propagate_through_system, propagate_through_system_jax

# ── Storage (unified HDF5 / Zarr) ────────────────────────────────────────
from .io.storage import (
    # Unified dispatch API
    set_storage_backend,
    get_storage_backend,
    default_extension,
    append_plane,
    load_planes,
    list_planes,
    load_plane_by_label,
    load_plane_slice,
    write_sim_metadata,
    read_sim_metadata,
    replay_run,
    # Backwards-compatible aliases
    list_planes_store,
    load_plane_by_label_store,
    load_plane_slice_store,
    write_metadata,
    read_metadata,
    # HDF5-specific functions
    save_field_h5,
    load_field_h5,
    save_planes_h5,
    load_planes_h5,
    save_jones_field_h5,
    load_jones_field_h5,
    append_plane_h5,
    list_h5_contents,
    TempFieldStore,
)

# ── Memory-aware batching helpers ───────────────────────────────────────
from .memory import (
    available_memory_bytes,
    total_memory_bytes,
    memory_info,
    bytes_per_element,
    array_bytes,
    estimate_op_memory,
    pick_batch_size,
    should_split,
    format_bytes,
    print_memory_report,
    get_ram_budget,
    set_max_ram,
)

# ── Plotting utilities ─────────────────────────────────────────────────
from .analysis.plotting import (
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

# ── Multi-backend infrastructure (3.4.0) ─────────────────────────────
# Foundation for first-class NumPy / CuPy / JAX backend support.
from .backend import (
    array_namespace,
    is_numpy_array,
    is_cupy_array,
    is_jax_array,
    backend_name,
    to_numpy,
    to_backend,
    JAX_AVAILABLE,
    RandomState,
)

# ── New propagators (3.4.0) ──────────────────────────────────────────
from .propagators.hfpi import (
    PathBundle,
    init_paths_from_field,
    init_paths_stratified,
    propagate_to_plane,
    apply_aperture_diffraction,
    accumulate_to_grid,
    propagate_hfpi_freespace_aperture,
    propagate_hfpi,                          # canonical-order alias
    propagate_hfpi_through_prescription,
)
from .propagators.gbd import (
    BeamletBundle,
    decompose_field_to_beamlets,
    propagate_beamlets_freespace,
    apply_thin_lens_to_beamlets,
    apply_abcd_to_beamlets,
    reconstruct_field_from_beamlets,
    propagate_gbd_freespace,
    propagate_gbd,                           # canonical-order alias
    propagate_gbd_thin_lens,
    propagate_gbd_through_prescription,
)
from .propagators.hf import (
    propagate_huygens_fresnel_freespace,
    propagate_huygens_fresnel_with_opl_callable,
    propagate_huygens_fresnel_through_prescription,
)
from .propagators.subaperture import (
    PatchGrid,
    patches_for_box,
    patch_window,
    combine_patch_fields,
    propagate_subaperture_asymptotic,
)
from .propagators.dispatch import (
    propagate,
    VALID_METHODS,
)
from .propagators.result import PropagationResult
from .propagators.mhs import (
    HuygensSurface,
    MhsSubdomain,
    MhsPipeline,
    asm_subdomain,
    aperture_subdomain,
    gbd_freespace_subdomain,
    prescription_subdomain,
)
from .propagators.vectorial_hfpi import (
    VectorPathBundle,
    init_vector_paths_from_field,
    propagate_vector_to_plane,
    apply_vector_aperture_diffraction,
    accumulate_vector_to_grid,
    propagate_vector_hfpi_freespace_aperture,
)
from .raytrace.jax_trace import (
    JaxRayState,
    make_jax_ray_state,
    trace_jax,
    jax_state_to_raybundle,
    raybundle_to_jax_state,
)

__version__ = "3.5.4"

#
# __all__ is grouped by user-journey tier:
#
#   Tier 1 -- Build a system (sources, prescriptions, elements, glass)
#   Tier 2 -- Propagate (dispatcher + propagator families)
#   Tier 3 -- Trace (geometric + JAX-traceable)
#   Tier 4 -- Analyze (beam analysis, aberration summary, through-focus,
#                       tolerancing, ghost, coherence, detector, etc.)
#   Tier 5 -- Optimize (parameterizations, merit terms, design_optimize)
#   Tier 6 -- Asymptotic / LG (polynomial fits, LG/HG, aberration tensor)
#   Tier 7 -- Specialized physics (RCWA, BSDF, coatings, vector
#                                    diffraction, vectorial HFPI)
#   Tier 8 -- I/O (prescription formats, HDF5/Zarr, code generation)
#   Tier 9 -- Plotting
#   Tier 10 - Infrastructure (backend, memory, progress, JAX flags)
#
__all__ = [

    # ============================================================
    # Tier 1 -- Build a system
    # ============================================================

    # Sources (field generators + Source object)
    'Source',
    'create_gaussian_beam',
    'create_hermite_gauss',
    'create_laguerre_gauss',
    'create_tilted_plane_wave',
    'create_point_source',
    'create_multi_field_sources',
    'create_top_hat_beam',
    'create_annular_beam',
    'create_fiber_mode',
    'create_led_source',
    'create_bessel_beam',
    'hermite_physicist',
    'laguerre_generalized',

    # Prescriptions (factories + format I/O)
    'make_singlet',
    'make_doublet',
    'make_cylindrical',
    'make_biconic',
    'thorlabs_lens',
    'THORLABS_CATALOG',
    'load_zmx_prescription',
    'load_zemax_prescription_txt',
    'export_zemax_lens_data',
    'export_zemax_zmx',
    'load_codev_seq',
    'export_codev_seq',
    'load_quadoa_qos',
    'export_quadoa_qos',
    'QUADOA_SCHEMA_VERSION',
    'scale_prescription',

    # Lens / element models (apply on a field)
    'apply_thin_lens',
    'apply_spherical_lens',
    'apply_aspheric_lens',
    'apply_real_lens',
    'apply_real_lens_traced',
    'apply_real_lens_maslov',
    'apply_cylindrical_lens',
    'apply_grin_lens',
    'apply_axicon',
    'apply_mirror',
    'apply_aperture',
    'apply_gaussian_aperture',
    'apply_mask',
    'apply_zernike_aberration',

    # Surface / sag helpers
    'surface_sag_general',
    'surface_sag_biconic',
    'surface_sag_xy_polynomial',
    'surface_sag_zernike_freeform',
    'surface_sag_chebyshev',
    'surface_sag_freeform',

    # Polarization / Jones calculus
    'JonesField',
    'apply_jones_matrix',
    'apply_polarizer',
    'apply_waveplate',
    'apply_half_wave_plate',
    'apply_quarter_wave_plate',
    'apply_rotator',
    'create_linear_polarized',
    'create_circular_polarized',
    'create_elliptical_polarized',
    'stokes_parameters',
    'degree_of_polarization',
    'polarization_ellipse',

    # Glass
    'get_glass_index',
    'get_glass_index_complex',
    'GLASS_REGISTRY',

    # DOE / phase mask helpers
    'create_periodic_phase_mask',
    'create_microlens_array',
    'makedammann2d',

    # Other field generators
    'zernike',
    'generate_turbulence_screen',

    # Grid / sampling helpers
    'check_grid_vs_apertures',
    'recommend_grid_for_prescription',

    # ============================================================
    # Tier 2 -- Propagate (dispatcher + propagator families)
    # ============================================================

    # Top-level smart-method propagator
    'propagate',
    'VALID_METHODS',
    'PropagationResult',

    # Free-space propagators (low-level)
    'angular_spectrum_propagate',
    'angular_spectrum_propagate_tilted',
    'scalable_angular_spectrum_propagate',
    'fresnel_propagate',
    'fraunhofer_propagate',
    'rayleigh_sommerfeld_propagate',
    'resample_field',

    # Huygens-Fresnel family
    'propagate_huygens_fresnel_freespace',
    'propagate_huygens_fresnel_with_opl_callable',
    'propagate_huygens_fresnel_through_prescription',

    # Gaussian Beamlet Decomposition
    'BeamletBundle',
    'decompose_field_to_beamlets',
    'propagate_beamlets_freespace',
    'apply_thin_lens_to_beamlets',
    'apply_abcd_to_beamlets',
    'reconstruct_field_from_beamlets',
    'propagate_gbd_freespace',
    'propagate_gbd',
    'propagate_gbd_thin_lens',
    'propagate_gbd_through_prescription',

    # Huygens-Fresnel Path Integration (Monte Carlo)
    'PathBundle',
    'init_paths_from_field',
    'init_paths_stratified',
    'propagate_to_plane',
    'apply_aperture_diffraction',
    'accumulate_to_grid',
    'propagate_hfpi_freespace_aperture',
    'propagate_hfpi',
    'propagate_hfpi_through_prescription',

    # Vectorial HFPI (Jones-vector paths)
    'VectorPathBundle',
    'init_vector_paths_from_field',
    'propagate_vector_to_plane',
    'apply_vector_aperture_diffraction',
    'accumulate_vector_to_grid',
    'propagate_vector_hfpi_freespace_aperture',

    # Subaperture decomposition
    'PatchGrid',
    'patches_for_box',
    'patch_window',
    'combine_patch_fields',
    'propagate_subaperture_asymptotic',

    # Multiple Huygens Surface (MHS) framework
    'HuygensSurface',
    'MhsSubdomain',
    'MhsPipeline',
    'asm_subdomain',
    'aperture_subdomain',
    'gbd_freespace_subdomain',
    'prescription_subdomain',

    # Element-walking system propagator
    'propagate_through_system',
    'propagate_through_system_jax',

    # ============================================================
    # Tier 3 -- Trace (geometric + JAX-traceable)
    # ============================================================

    # Geometric ray trace
    'RayBundle',
    'Surface',
    'TraceResult',
    'trace',
    'trace_prescription',
    'surfaces_from_prescription',
    'surfaces_from_elements',
    'raytrace_system',
    'apply_doe_phase_traced',

    # Ray generators
    'make_ray',
    'make_fan',
    'make_ring',
    'make_grid',
    'make_rings',

    # Ray-trace analysis
    'system_abcd',
    'system_abcd_prescription',
    'seidel_coefficients',
    'seidel_prescription',
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
    'find_paraxial_focus',
    'trace_summary',
    'prescription_summary',

    # Per-ray diagnostic codes
    'RAY_OK',
    'RAY_TIR',
    'RAY_APERTURE',
    'RAY_MISSED_SURFACE',
    'RAY_NAN',
    'RAY_EVANESCENT',

    # JAX-traceable ray trace
    'JaxRayState',
    'make_jax_ray_state',
    'trace_jax',
    'jax_state_to_raybundle',
    'raybundle_to_jax_state',

    # ============================================================
    # Tier 4 -- Analyze
    # ============================================================

    # Beam analysis (Strehl, PSF, MTF, OPD, Zernike)
    'beam_centroid',
    'beam_d4sigma',
    'coupling_efficiency',
    'M2',
    'caustic_diagnostic',
    'plot_caustic_diagnostic',
    'CausticDiagnostic',
    'beam_power',
    'strehl_ratio',
    'check_sampling_conditions',
    'compute_psf',
    'compute_otf',
    'compute_mtf',
    'mtf_radial',
    'remove_wavefront_modes',
    'opd_pv_rms',
    'wave_opd_1d',
    'wave_opd_2d',
    'check_opd_sampling',
    'chromatic_focal_shift',
    'polychromatic_strehl',
    'radial_power_bands',
    'zernike_polynomial',
    'zernike_basis_matrix',
    'zernike_decompose',
    'zernike_reconstruct',
    'zernike_index_to_nm',
    'zernike_nm_to_index',

    # Unified aberration analysis (Seidel + LG tensor)
    'AberrationSummary',
    'aberration_summary',
    'format_aberration_summary',

    # Through-focus / tolerancing
    'single_plane_metrics',
    'diffraction_limited_peak',
    'through_focus_scan',
    'through_focus_scan_jax',
    'find_best_focus',
    'plot_through_focus',
    'ThroughFocusResult',
    'Perturbation',
    'apply_perturbations',
    'tolerancing_sweep',
    'monte_carlo_tolerancing',

    # Coherence / extended-source imaging
    'koehler_image',
    'extended_source_image',
    'mutual_coherence',

    # Detector / wavefront sensing
    'apply_detector',
    'shack_hartmann',

    # Interferometry
    'simulate_interferogram',
    'phase_shift_extract',
    'fringe_spacing',

    # Phase retrieval
    'gerchberg_saxton',
    'error_reduction',
    'hybrid_input_output',
    'gerchberg_saxton_jax',
    'error_reduction_jax',
    'hybrid_input_output_jax',

    # Ghost analysis
    'enumerate_ghost_paths',
    'ghost_analysis',

    # ============================================================
    # Tier 5 -- Optimize
    # ============================================================

    'DesignParameterization',
    'MultiPrescriptionParameterization',
    'EvaluationContext',
    'DesignResult',
    'design_optimize',
    'WAVE_PROPAGATOR_REGISTRY',
    'register_wave_propagator',
    'unregister_wave_propagator',

    # Merit terms
    'MeritTerm',
    'FocalLengthMerit',
    'BackFocalLengthMerit',
    'SphericalSeidelMerit',
    'StrehlMerit',
    'RMSWavefrontMerit',
    'SpotSizeMerit',
    'ChromaticFocalShiftMerit',
    'MatchIdealThinLensMerit',
    'MatchIdealSystemMerit',
    'MatchTargetOPDMerit',
    'ZernikeCoefficientMerit',
    'LGAberrationMerit',
    'CompositeMerit',
    'CallableMerit',
    'JaxMeritTerm',
    'MultiWavelengthMerit',
    'MultiFieldMerit',
    'MinThicknessMerit',
    'MaxThicknessMerit',
    'MinBackFocalLengthMerit',
    'MaxFNumberMerit',
    'ToleranceAwareMerit',

    # Multi-configuration / afocal
    'Configuration',
    'multi_config_merit',
    'create_zoom_configs',
    'afocal_angular_magnification',
    'beam_expander_prescription',
    'keplerian_telescope',

    # ============================================================
    # Tier 6 -- Asymptotic / LG aberration tensor
    # ============================================================

    # Polynomial fits
    'CanonicalPolyFit',
    'HFPolyFit',
    'AberrationTensorResult',
    'fit_canonical_polynomials',
    'fit_hf_polynomials',

    # Newton solver / propagators / quadrature
    'solve_envelope_stationary',
    'aberration_tensor',
    'propagate_modal_asymptotic',
    'propagate_hf_chebyshev_quadrature',

    # LG / HG basis
    'lg_polynomial',
    'hg_polynomial',
    'evaluate_lg_mode',
    'evaluate_hg_mode',
    'decompose_lg',
    'decompose_hg',
    'lg_seidel_label',

    # Wick moments
    'gaussian_moment_2d',
    'gaussian_moment_table_2d',

    # JAX paths
    'aberration_tensor_lg00_jax',
    'propagate_modal_asymptotic_lg00_jax',
    'JaxAberrationTensorResult',
    'solve_envelope_stationary_jax_ift',
    'fit_canonical_polynomials_jax',
    'apply_real_lens_traced_jax',
    'apply_real_lens_maslov_jax',
    'monte_carlo_tolerancing_jax',
    'tolerancing_report',

    # ============================================================
    # Tier 7 -- Specialized physics
    # ============================================================

    # Vector diffraction (high-NA focus)
    'richards_wolf_focus',
    'debye_wolf_psf',

    # RCWA (rigorous coupled-wave analysis)
    'rcwa_1d',
    'grating_efficiency_vs_wavelength',

    # BSDF / surface scatter
    'BSDFModel',
    'LambertianBSDF',
    'GaussianBSDF',
    'HarveyShackBSDF',
    'make_bsdf',
    'sample_scatter_rays',

    # Thin-film coatings
    'coating_reflectance',
    'quarter_wave_ar',
    'broadband_ar_v_coat',

    # ============================================================
    # Tier 8 -- I/O (HDF5 / Zarr / phase / FITS / code-gen)
    # ============================================================

    # Unified storage (HDF5 / Zarr dispatch)
    'set_storage_backend',
    'get_storage_backend',
    'default_extension',
    'append_plane',
    'load_planes',
    'list_planes_store',
    'load_plane_by_label_store',
    'load_plane_slice_store',
    'write_metadata',
    'read_metadata',
    'replay_run',

    # HDF5 (low-level)
    'save_field_h5',
    'load_field_h5',
    'save_planes_h5',
    'load_planes_h5',
    'save_jones_field_h5',
    'load_jones_field_h5',
    'append_plane_h5',
    'list_h5_contents',
    'list_planes',
    'load_plane_by_label',
    'load_plane_slice',
    'TempFieldStore',
    'write_sim_metadata',
    'read_sim_metadata',

    # Phase / FITS file I/O
    'load_phase_file',
    'save_phase_file',
    'load_fits_field',
    'save_fits_field',

    # Code generation
    'generate_simulation_script',
    'generate_script_from_zmx',
    'generate_script_from_txt',

    # ============================================================
    # Tier 9 -- Plotting
    # ============================================================

    'plot_intensity',
    'plot_phase',
    'plot_field',
    'plot_amplitude_phase',
    'plot_cross_section',
    'plot_planes_grid',
    'plot_psf',
    'plot_mtf',
    'plot_stokes',
    'plot_polarization_ellipses',
    'plot_beam_profile',
    'plot_jones_pupil',
    'compute_jones_pupil',

    # ============================================================
    # Tier 10 -- Infrastructure (backend, memory, progress, flags)
    # ============================================================

    # Backend availability flags
    'PYFFTW_AVAILABLE',
    'CUPY_AVAILABLE',
    'NUMEXPR_AVAILABLE',
    'JAX_AVAILABLE',
    'available_cpus',
    'set_fft_threads',
    'set_fft_fallback',
    'reset_fft_backend',

    # Precision configuration
    'set_default_complex_dtype',
    'get_default_complex_dtype',
    'DEFAULT_COMPLEX_DTYPE',
    'set_asm_cache_size',
    'set_fft_plan_cache_size',
    'clear_asm_caches',

    # Memory-aware batching
    'available_memory_bytes',
    'total_memory_bytes',
    'memory_info',
    'bytes_per_element',
    'array_bytes',
    'estimate_op_memory',
    'pick_batch_size',
    'should_split',
    'format_bytes',
    'print_memory_report',
    'get_ram_budget',
    'set_max_ram',

    # Progress callback infrastructure
    'ProgressCallback',
    'ProgressScaler',
    'call_progress',
]

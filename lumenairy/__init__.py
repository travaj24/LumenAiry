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
    get_asm_cache_size,
    set_fft_plan_cache_size,
    set_pyfftw_planner,
    get_pyfftw_planner,
    set_fft_auto_promote,
    get_fft_auto_promote,
    warmup_fft_plans,
    clear_asm_caches,
    rayleigh_sommerfeld_propagate,
    resample_field,
    apply_fresnel_curvature,
    fresnel_propagate_mft,
    fraunhofer_propagate_mft,
    angular_spectrum_propagate_mft,
    # FFT backend configuration
    PYFFTW_AVAILABLE,
    CUPY_AVAILABLE,
    set_fft_threads,
    get_fft_threads,
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
    close_worker_pool,
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
    SELLMEIER_COEFFICIENTS,
    # v4.16.0 (ROADMAP #14): per-glass Sellmeier validity ranges.
    GLASS_VALIDITY,
    list_glasses,
    search_glasses,
)

# ── Optical elements ─────────────────────────────────────────────────────
from .elements import (
    apply_mirror,
    apply_aperture,
    apply_gaussian_aperture,
    apply_mask,
    zernike,
    apply_zernike_aberration,
    apply_lyot_focal_plane_mask,
    apply_vortex_phase_mask,
    apply_lyot_stop,
    apply_apodized_pupil,
    generate_turbulence_screen,
)

# Coronagraph contrast curve moved to analysis/coronagraph.py in
# 4.3.0; lumenairy.elements re-export still works via a deferred-
# import shim for back-compat.
from .analysis.coronagraph import coronagraph_contrast_curve

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
from .analysis.image_plane_wfe import (
    ImagePlaneWFE,
    eval_image_plane_wfe,
    field_grid_wfe,
    zemax_pupil_grid,
    chebyshev_pupil_grid,
    remove_low_order_aberrations,
)
from .analysis.phase_retrieval import (
    gerchberg_saxton_jax,
    error_reduction_jax,
    hybrid_input_output_jax,
    clear_phase_retrieval_caches,
)
from .analysis import (
    beam_centroid,
    beam_d4sigma,
    beam_diameter,
    beam_power,
    strehl_ratio,
    strehl_marechal,
    strehl_phase_integral,
    strehl_vector,
    coupling_efficiency,
    coupling_efficiency_vector,
    M2,
    caustic_diagnostic,
    plot_caustic_diagnostic,
    CausticDiagnostic,
    check_sampling_conditions,
    compute_psf,
    compute_otf,
    compute_mtf,
    mtf_radial,
    mtf_cutoff,
    encircled_energy_curve,
    encircled_energy_radius,
    ee_polychromatic,
    rayleigh_resolution,
    sparrow_resolution,
    fwhm_resolution,
    depth_of_focus,
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
    astigmatism_mag_angle,
    clear_zernike_basis_cache,
)
# v4.12.2: expose the through_focus_scan_jax kernel-cache clear helper
# alongside the other clear_*_cache exports.
from .analysis.through_focus import clear_through_focus_scan_jax_cache

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
    # v4.15 (ROADMAP v4.16 #9, #11): Schell-model + annular-incoherent
    # partial-coherence source factories.
    # v4.15.1 (P0-NEW-2): the factories now return ensembles (or a
    # PartialCoherenceMCF) and actually deliver partial coherence.
    create_gaussian_schell_source,
    create_schell_model_source,
    create_annular_incoherent_source,
    PartialCoherenceMCF,
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

# ── Adaptive optics primitives ─────────────────────────────────────────
# AO moved to analysis/ao.py in 4.3.0; lumenairy.ao still works via
# a back-compat shim.
from .analysis.ao import (
    DeformableMirror,
    apply_dm,
    zernike_modal_basis,
    slope_to_modal,
    LeakyIntegrator,
)

# ── Field-resolved analyses (4.4.0) ────────────────────────────────────
# Lifted from ui/<dock>.py so distortion / footprint / spot-by-field /
# sensitivity ranking are reachable from scripts.  GUI docks now call
# these public functions and render the result.
from .analysis.field import (
    DistortionVsField,
    distortion_vs_field,
    DistortionGrid,
    distortion_grid,
    SurfaceFootprint,
    FieldFootprint,
    footprint_per_surface,
    SpotDiagramField,
    spot_diagram_vs_field,
    RelativeIllumination,
    relative_illumination,
    FieldAberrationSweep,
    field_aberration_sweep,
    petzval_radius,
    SensitivityResult,
    sensitivity_ranking,
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
    surface_sag_q_bfs,
    surface_sag_q_con,
    surface_sag_freeform,
)

# ── Ghost analysis ────────────────────────────────────────────────────
from .analysis.ghost import (
    enumerate_ghost_paths,
    ghost_analysis,
    non_sequential_stray_light,
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

# ── Grating diffraction efficiency (thin-grating scalar approx) ───────
# The function is named for what it actually computes: an analytical
# scalar thin-phase-grating diffraction-efficiency formula.  A future
# release may add a separate full RCWA path.
from .elements.thin_grating import (
    thin_grating_efficiency_1d,
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
    monte_carlo_tolerancing_linearized,
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
    make_lg_aberration_merit_jax,
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
    Constraint,
    design_optimize,
    WAVE_PROPAGATOR_REGISTRY,
    register_wave_propagator,
    unregister_wave_propagator,
    # v4.16 (ROADMAP #11): multi-objective Pareto wrapper (pymoo-optional)
    ParetoResult,
    design_optimize_multi_objective,
    # v4.16.0 (Agent A __all__-symmetry walker): the pymoo
    # availability flag is the canonical "is the optional Pareto
    # backend installed?" probe.  Sibling to JAX_AVAILABLE /
    # CUPY_AVAILABLE / NUMEXPR_AVAILABLE / PYFFTW_AVAILABLE.  Was
    # in the multi_objective submodule __all__ but never re-
    # exported at top level.
    PYMOO_AVAILABLE,
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
    clear_lg_polynomial_cache,
    clear_lg_mode_stack_cache,
)

# ── DOE / Gratings / Phase I/O ──────────────────────────────────────────
from .elements.doe import (
    create_periodic_phase_mask,
    create_microlens_array,
    create_diffractive_lens,
    create_kinoform,
    create_fresnel_zone_plate,
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
from .progress import (
    ProgressCallback, ProgressScaler, call_progress,
    CancellableProgress, is_cancelled,
)

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
    make_off_axis_parabola,
    thorlabs_lens,
    load_zemax_zmx,
    load_zemax_prescription_data_txt,
    export_zemax_lens_data,
    export_zemax_zmx,
    load_codev_seq,
    export_codev_seq,
    export_quadoa_qos,
    load_quadoa_qos,
    QUADOA_SCHEMA_VERSION,
    scale_prescription,
    normalize_prescription,
    split_prescription_at_mirrors,
    has_mirrors,
    THORLABS_CATALOG,
)

# ── Geometric ray tracing ────────────────────────────────────────────────
from .raytrace import (
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
    seidel_field_sweep,
    seidel_wfe,
    world_surfaces_from_prescription,
    paraxial_focus_world,
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
    FirstOrderData,
    first_order_data,
    # Paraxial-design one-liner helpers (4.0+)
    field_of_view,
    optical_invariant,
    f_number,
    defocus_waves_to_zernike,
    astigmatism_waves_to_zernike,
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
    # v4.15.1 (Cluster B Item 6): wave -> ray bridge.
    rays_from_field,
)

# ── System propagation ──────────────────────────────────────────────────
from .system import (
    propagate_through_system,
    propagate_through_system_jax,
    clear_propagate_system_jax_cache,
    # v4.15 (ROADMAP v4.15 #3): ergonomic prescription + Source ->
    # PropagationResult one-call entry, exposed at top level so users
    # can also call ``lumenairy.evaluate(rx, source)`` directly.
    evaluate,
)

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
    get_max_ram,
)

# ── Scoped runtime-environment context (4.8.1) ──────────────────────────
from ._context import (
    lumenairy_context,
    snapshot_globals,
    apply_globals,
    install_atexit_restore,
)

# ── Central cache-clearer registry (4.16.0) ─────────────────────────────
# v4.16.0 (ROADMAP #15): retires the lazy-import fan-out in
# ``clear_asm_caches`` in favour of a central registry.  Each cache-
# owning module registers its clear function at import time;
# ``clear_asm_caches`` walks the registry rather than enumerating
# calls by hand.  Counter-measure to the "fix N, miss N+1" sibling-
# gap meta-pattern that recurred 5 ways inside v4.14.2 and again at
# v4.14.3.
from ._cache_registry import (
    register_cache_clearer,
    list_registered_cache_clearers,
)
# Snapshot the import-time defaults and register an atexit handler that
# restores them on process shutdown.  Catches the foot-gun where users
# call set_default_complex_dtype / set_pyfftw_planner / etc. at module
# scope inside a long-running process (Jupyter, test harnesses) and
# expect a "clean" library state on the next unrelated run -- the
# handler restores whatever the defaults were at the very first
# ``import lumenairy``.
install_atexit_restore()

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
    plot_wavefront,
    plot_opd_fan,
    plot_opd_summary,
    plot_jones_pupil,
    compute_jones_pupil,
    plot_lens_layout,
    abbe_diagram,
    plot_glass_map,
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
    propagate_huygens_fresnel,               # canonical-order alias
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
    asm_propagate,
    which_propagator,
    ASM_FAMILY,
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
    clear_trace_jax_cache,
)

# ── Operator algebra (4.15.1, Cluster B Item 2) ─────────────────────────
# Nazarathy/Shamir-style optical operator algebra over the existing
# array-first propagator infrastructure.  Composable optical primitives
# (FreeSpace, ThinLens, CylindricalLens, Magnify, FourierTransform,
# Aperture, GaussianAperture) with closed-form ABCDs and chain-and-
# delegate field application.  See ``lumenairy.algebra`` and
# ``docs/audits/CLUSTER_B_SPEC.md`` §3.
from .algebra import (
    Operator,
    CompositeOperator,
    FreeSpace,
    ThinLens,
    CylindricalLens,
    Magnify,
    FourierTransform,
    Aperture,
    GaussianAperture,
)

# ── Deprecated-alias shims ──────────────────────────────────────────────
#
# 4.12.0: wire ``_deprecation.deprecated_alias`` (added in 4.7 but never
# imported anywhere -- a Round-4 audit finding) into the top-level
# namespace so users with pre-4.7 code can keep calling the historical
# names with a one-cycle ``DeprecationWarning`` instead of a cold
# ``AttributeError``.  Each shim forwards to the canonical new name.
from ._deprecation import deprecated_alias as _deprecated_alias

# C.2 (v4.7): Zemax-loader renames.
load_zmx_prescription = _deprecated_alias(
    load_zemax_zmx,
    old_name='load_zmx_prescription',
    version_added='4.7',
    version_removed='5.0',
)
load_zemax_prescription_txt = _deprecated_alias(
    load_zemax_prescription_data_txt,
    old_name='load_zemax_prescription_txt',
    version_added='4.7',
    version_removed='5.0',
)

__version__ = "4.16.0"

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
#   Tier 7 -- Specialized physics (thin-grating, BSDF, coatings, vector
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
    # v4.15 (ROADMAP v4.16 #9, #11): Schell-model + annular-incoherent.
    # v4.15.1 (P0-NEW-2): factories now return ensembles + new MCF class.
    'create_gaussian_schell_source',
    'create_schell_model_source',
    'create_annular_incoherent_source',
    'PartialCoherenceMCF',
    'hermite_physicist',
    'laguerre_generalized',

    # Prescriptions (factories + format I/O)
    'make_singlet',
    'make_doublet',
    'make_cylindrical',
    'make_biconic',
    'make_off_axis_parabola',
    'thorlabs_lens',
    'THORLABS_CATALOG',
    'load_zemax_zmx',
    'load_zemax_prescription_data_txt',
    # Deprecated aliases (v4.7 rename, v5.0 removal target)
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
    'normalize_prescription',
    'split_prescription_at_mirrors',
    'has_mirrors',

    # Lens / element models (apply on a field)
    'apply_thin_lens',
    'apply_spherical_lens',
    'apply_aspheric_lens',
    'apply_real_lens',
    'apply_real_lens_traced',
    'close_worker_pool',
    'apply_real_lens_maslov',
    'apply_real_lens_traced_jax',
    'apply_real_lens_maslov_jax',
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
    'surface_sag_q_bfs',
    'surface_sag_q_con',
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
    'SELLMEIER_COEFFICIENTS',
    # v4.16.0 (ROADMAP #14): per-glass Sellmeier validity ranges.
    'GLASS_VALIDITY',
    'list_glasses',
    'search_glasses',

    # DOE / phase mask helpers
    'create_periodic_phase_mask',
    'create_microlens_array',
    'create_diffractive_lens',
    'create_kinoform',
    'create_fresnel_zone_plate',
    'makedammann2d',

    # Other field generators
    'zernike',
    'generate_turbulence_screen',

    # Coronagraph templates
    'apply_lyot_focal_plane_mask',
    'apply_vortex_phase_mask',
    'apply_lyot_stop',
    'apply_apodized_pupil',
    'coronagraph_contrast_curve',

    # Grid / sampling helpers
    'check_grid_vs_apertures',
    'recommend_grid_for_prescription',

    # Operator algebra (4.15.1, Cluster B Item 2; tier moved to Tier-1
    # in v4.15.2 per AUDIT_V4_15_1 P3): Nazarathy/Shamir-style
    # symbolic optical-system construction layered over the existing
    # propagators.  Each primitive carries a closed-form 2x2 ABCD and
    # delegates field application to the canonical LumenAiry function.
    # Operator algebra is a BUILD-TIME construction surface
    # (composing primitives into a system), not a propagation surface
    # -- the dispatcher and the propagator families still live in
    # Tier-2 and consume the field after the build-time composition.
    # See ``lumenairy.algebra`` for details.
    'Operator',
    'CompositeOperator',
    'FreeSpace',
    'ThinLens',
    'CylindricalLens',
    'Magnify',
    'FourierTransform',
    'Aperture',
    'GaussianAperture',

    # ============================================================
    # Tier 2 -- Propagate (dispatcher + propagator families)
    # ============================================================

    # Top-level smart-method propagator
    'propagate',
    'VALID_METHODS',
    'asm_propagate',
    'which_propagator',
    'ASM_FAMILY',
    'PropagationResult',

    # Free-space propagators (low-level)
    'angular_spectrum_propagate',
    'angular_spectrum_propagate_tilted',
    'scalable_angular_spectrum_propagate',
    'fresnel_propagate',
    'fraunhofer_propagate',
    'rayleigh_sommerfeld_propagate',
    'resample_field',
    'apply_fresnel_curvature',
    'fresnel_propagate_mft',
    'fraunhofer_propagate_mft',
    'angular_spectrum_propagate_mft',

    # Huygens-Fresnel family
    'propagate_huygens_fresnel',
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
    # v4.15 (ROADMAP v4.15 #3): ergonomic prescription -> result entry.
    'evaluate',

    # ============================================================
    # Tier 3 -- Trace (geometric + JAX-traceable)
    # ============================================================

    # Geometric ray trace
    'RayBundle',
    'Surface',
    'TraceResult',
    'trace',
    'trace_world',
    'trace_prescription',
    'surfaces_from_prescription',
    'validate_prescription',
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
    'find_paraxial_focus',
    'trace_summary',
    'prescription_summary',
    # Paraxial helpers (4.0+)
    'field_of_view',
    'optical_invariant',
    'f_number',
    'defocus_waves_to_zernike',
    'astigmatism_waves_to_zernike',
    # v4.15.1 (Cluster B Item 6): wave -> ray bridge.
    'rays_from_field',

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
    'beam_diameter',
    'coupling_efficiency',
    'coupling_efficiency_vector',
    'M2',
    'caustic_diagnostic',
    'plot_caustic_diagnostic',
    'CausticDiagnostic',
    'beam_power',
    'strehl_ratio',
    'strehl_marechal',
    'strehl_phase_integral',
    'strehl_vector',
    'check_sampling_conditions',
    'compute_psf',
    'compute_otf',
    'compute_mtf',
    'mtf_radial',
    'mtf_cutoff',
    'encircled_energy_curve',
    'encircled_energy_radius',
    'ee_polychromatic',
    'rayleigh_resolution',
    'sparrow_resolution',
    'fwhm_resolution',
    'depth_of_focus',
    'remove_wavefront_modes',
    'opd_pv_rms',
    'wave_opd_1d',
    'wave_opd_2d',
    'check_opd_sampling',
    'chromatic_focal_shift',
    'polychromatic_strehl',
    'polychromatic_psf',
    'radial_power_bands',
    'zernike_polynomial',
    'zernike_basis_matrix',
    'zernike_decompose',
    'zernike_reconstruct',
    'zernike_index_to_nm',
    'zernike_nm_to_index',
    'astigmatism_mag_angle',

    # Unified aberration analysis (Seidel + LG tensor)
    'AberrationSummary',
    'aberration_summary',
    'format_aberration_summary',
    'ImagePlaneWFE',
    'eval_image_plane_wfe',
    'field_grid_wfe',
    'zemax_pupil_grid',
    'chebyshev_pupil_grid',
    'remove_low_order_aberrations',

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
    'monte_carlo_tolerancing_jax',
    'monte_carlo_tolerancing_linearized',
    'tolerancing_report',

    # Coherence / extended-source imaging
    'koehler_image',
    'extended_source_image',
    'mutual_coherence',

    # Detector / wavefront sensing
    'apply_detector',
    'shack_hartmann',

    # Adaptive-optics primitives
    'DeformableMirror', 'apply_dm',
    'zernike_modal_basis', 'slope_to_modal',
    'LeakyIntegrator',

    # Field-resolved analyses (4.4.0)
    'DistortionVsField', 'distortion_vs_field',
    'DistortionGrid', 'distortion_grid',
    'SurfaceFootprint', 'FieldFootprint', 'footprint_per_surface',
    'SpotDiagramField', 'spot_diagram_vs_field',
    'RelativeIllumination', 'relative_illumination',
    'FieldAberrationSweep', 'field_aberration_sweep',
    'petzval_radius',
    'SensitivityResult', 'sensitivity_ranking',

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
    'non_sequential_stray_light',

    # ============================================================
    # Tier 5 -- Optimize
    # ============================================================

    'DesignParameterization',
    'MultiPrescriptionParameterization',
    'EvaluationContext',
    'DesignResult',
    'Constraint',
    'design_optimize',
    # v4.16 (ROADMAP #11): multi-objective Pareto wrapper (pymoo-optional)
    'ParetoResult',
    'design_optimize_multi_objective',
    'PYMOO_AVAILABLE',
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
    'make_lg_aberration_merit_jax',
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

    # ============================================================
    # Tier 7 -- Specialized physics
    # ============================================================

    # Vector diffraction (high-NA focus)
    'richards_wolf_focus',
    'debye_wolf_psf',

    # Thin-grating scalar diffraction efficiency
    'thin_grating_efficiency_1d',
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
    'plot_wavefront',
    'plot_opd_fan',
    'plot_opd_summary',
    'plot_jones_pupil',
    'compute_jones_pupil',
    'plot_lens_layout',
    'abbe_diagram',
    'plot_glass_map',

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
    'get_fft_threads',
    'set_fft_fallback',
    'reset_fft_backend',

    # v4.16.0 (Agent A __all__-symmetry walker): the backend
    # dispatch helpers below were imported at top level since
    # v3.4.0 but never listed in __all__.  Each is the canonical
    # high-traffic entry point for user code that wants to inspect
    # / dispatch on the array backend explicitly (e.g.
    # ``xp = la.array_namespace(E)``).  Promoted into __all__ so
    # ``from lumenairy import *`` includes them and the v4.14.0
    # P1-NEW-4 sibling-gap meta-pin doesn't re-flag them.
    'array_namespace',
    'is_numpy_array',
    'is_cupy_array',
    'is_jax_array',
    'backend_name',
    'to_numpy',
    'to_backend',
    'RandomState',

    # Precision configuration
    'set_default_complex_dtype',
    'get_default_complex_dtype',
    'DEFAULT_COMPLEX_DTYPE',
    'set_asm_cache_size',
    'get_asm_cache_size',
    'set_fft_plan_cache_size',
    'set_pyfftw_planner',
    'get_pyfftw_planner',
    'set_fft_auto_promote',
    'get_fft_auto_promote',
    'warmup_fft_plans',
    'clear_asm_caches',
    'clear_zernike_basis_cache',
    'clear_lg_polynomial_cache',
    'clear_lg_mode_stack_cache',
    'clear_through_focus_scan_jax_cache',
    'clear_trace_jax_cache',
    'clear_propagate_system_jax_cache',
    'clear_phase_retrieval_caches',
    # v4.16.0 (ROADMAP #15): central cache-clearer registry.
    'register_cache_clearer',
    'list_registered_cache_clearers',

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
    'get_max_ram',

    # Scoped runtime context (4.8.1)
    'lumenairy_context',
    'snapshot_globals',
    'apply_globals',

    # Progress callback infrastructure
    'ProgressCallback',
    'ProgressScaler',
    'call_progress',
    'CancellableProgress',
    'is_cancelled',
]

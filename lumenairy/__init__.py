"""
lumenairy â€” Coherent Optical Field Propagation Library
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

# â”€â”€ Propagation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .analysis import (
    M2,
    CausticDiagnostic,
    astigmatism_mag_angle,
    beam_centroid,
    beam_d4sigma,
    beam_diameter,
    beam_power,
    caustic_diagnostic,
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
    plot_caustic_diagnostic,
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

# â”€â”€ Beam analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .analysis.aberration import (
    AberrationSummary,
    aberration_summary,
    format_aberration_summary,
)

# Coronagraph contrast curve moved to analysis/coronagraph.py in
# 4.3.0; lumenairy.elements re-export still works via a deferred-
# import shim for back-compat.
from .analysis.coronagraph import coronagraph_contrast_curve
from .analysis.image_plane_wfe import (
    ImagePlaneWFE,
    chebyshev_pupil_grid,
    eval_image_plane_wfe,
    field_grid_wfe,
    remove_low_order_aberrations,
    zemax_pupil_grid,
)
from .analysis.phase_retrieval import (
    clear_phase_retrieval_caches,
    error_reduction_jax,
    gerchberg_saxton_jax,
    hybrid_input_output_jax,
)

# v4.12.2: expose the through_focus_scan_jax kernel-cache clear helper
# alongside the other clear_*_cache exports.
from .analysis.through_focus import clear_through_focus_scan_jax_cache

# â”€â”€ Optical elements â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .elements import (
    apply_aperture,
    apply_apodized_pupil,
    apply_gaussian_aperture,
    apply_lyot_focal_plane_mask,
    apply_lyot_stop,
    apply_mask,
    apply_mirror,
    apply_vortex_phase_mask,
    apply_zernike_aberration,
    create_eight_octant_phase_mask,
    create_four_quadrant_phase_mask,
    generate_turbulence_screen,
    zernike,
)

# â”€â”€ Optional optimisation-backend availability flags â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Users / runners can inspect these before toggling tunables that
# depend on the optional backends (e.g. the numexpr-fused phase-screen
# path inside apply_real_lens).  Truthy if the package is importable
# in the current environment.
# â”€â”€ Lenses â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .elements.lenses import (
    NUMEXPR_AVAILABLE,
    PreparedAnalyticLens,
    PreparedTracedLens,
    TiltedCarrier,
    apply_aspheric_lens,
    apply_axicon,
    apply_cylindrical_lens,
    apply_grin_lens,
    apply_real_lens,
    apply_real_lens_gbd,
    apply_real_lens_maslov,
    apply_real_lens_maslov_jax,
    apply_real_lens_traced,
    apply_real_lens_traced_jax,
    apply_real_lens_traced_multi,
    apply_real_lens_traced_multibranch,
    apply_real_lens_traced_segmented,
    apply_real_lens_traced_uniform,
    apply_spherical_lens,
    apply_thin_lens,
    check_grid_vs_apertures,
    clear_pointwise_cos_grid_cache,
    close_worker_pool,
    get_lens_parallel_amp,
    get_lens_sag_dtype,
    get_pointwise_cos_grid_cache_budget,
    lens_sag_float32_opd_error,
    prepare_real_lens,
    prepare_real_lens_traced,
    recommend_grid_for_prescription,
    set_lens_parallel_amp,
    set_lens_sag_dtype,
    set_pointwise_cos_grid_cache_budget,
    surface_sag_biconic,
    surface_sag_general,
)

# v5.21 (__all__-symmetry): Maslov vector entry point + the caustic-uniform
# special functions live in the lenses_maslov submodule __all__ but are not
# re-exported by the elements.lenses aggregate.
from .elements.lenses_maslov import (
    apply_real_lens_maslov_vector,
    pearcey,
    uniform_fold_airy,
)

# â”€â”€ Glass catalog â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .glass import (
    GLASS_REGISTRY,
    # v4.16.0 (ROADMAP #14): per-glass Sellmeier validity ranges.
    GLASS_VALIDITY,
    SELLMEIER_COEFFICIENTS,
    get_glass_index,
    get_glass_index_complex,
    list_glasses,
    search_glasses,
)

# â”€â”€ Backend / runtime helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .memory import available_cpus

# -- Field-aggregation primitives (carrier_field) ------------------------
# The carrier-field value types and their two verbs are public on
# ``lumenairy.propagators`` (and on ``lumenairy.propagators.carrier_field``)
# but landed without a top-level re-export.  That is the v4.14.0 audit
# P1-NEW-4 sibling gap, which the v4.16.0 ``__all__``-symmetry walker
# generalises to every name; the walker is what caught it.
# ``re_reference`` and ``aggregate`` are the verbs -- everything else is a
# value type or the on-disk schema constant.
from .propagators.carrier_field import (
    CARRIER_FIELD_SCHEMA,
    AggregateLedger,
    AggregateResult,
    CarrierField,
    CarrierSpec,
    FieldGrid,
    FieldLedgerRow,
    NyquistReport,
    ReReferenceReport,
    aggregate,
    carrier_difference_nyquist,
    load_carrier_field_zarr,
    re_reference,
    save_carrier_field_zarr,
)
from .propagators.propagation import (
    CUPY_AVAILABLE,
    # Precision configuration (complex64 vs complex128)
    DEFAULT_COMPLEX_DTYPE,
    DEFAULT_DY,
    # v4.16.3 (audit P3-NEW-F2-LOW-1): sibling re-exports for the
    # v4.16.2 default-config knob globals -- ``DEFAULT_COMPLEX_DTYPE``
    # has been at top level since v4.14, but the three v4.16.2
    # globals were only reachable via ``lumenairy.propagators.
    # propagation.DEFAULT_*`` despite the matching setter/getter
    # accessors being top-level since v4.16.2.
    DEFAULT_REAL_DTYPE,
    DEFAULT_WAVE_PROPAGATOR,
    # v5.31 (audit W9-8): the frozen factory value propagate() compares the
    # knob against.  IMMUTABLE, so a static re-export (no live forwarding).
    DEFAULT_WAVE_PROPAGATOR_SHIPPED,
    # FFT backend configuration
    PYFFTW_AVAILABLE,
    CarrierReferencedField,
    TracedCarrierChainMultiResult,
    TracedCarrierChainResult,
    angular_spectrum_propagate,
    angular_spectrum_propagate_mft,
    angular_spectrum_propagate_tilted,
    apply_fresnel_curvature,
    carrier_referenced_aperture,
    carrier_referenced_envelope,
    carrier_referenced_exact_focus_readout,
    carrier_referenced_fit_radius,
    carrier_referenced_focus_readout,
    carrier_referenced_reconstruct,
    clear_asm_caches,
    fraunhofer_propagate,
    fraunhofer_propagate_mft,
    fresnel_propagate,
    fresnel_propagate_mft,
    fresnel_tf_propagate,
    get_asm_cache_size,
    get_default_complex_dtype,
    get_default_dy,
    get_default_real_dtype,
    get_default_wave_propagator,
    get_fft_auto_promote,
    get_fft_double_buffer,
    get_fft_plan_cache_size,
    get_fft_threads,
    get_pyfftw_planner,
    propagate_carrier_referenced,
    propagate_traced_carrier_chain,
    propagate_traced_carrier_chain_multi,
    rayleigh_sommerfeld_propagate,
    resample_field,
    reset_fft_backend,
    restore_fft_state,
    scalable_angular_spectrum_propagate,
    set_asm_cache_size,
    set_default_complex_dtype,
    set_default_dy,
    set_default_real_dtype,
    set_default_wave_propagator,
    set_fft_auto_promote,
    set_fft_double_buffer,
    set_fft_fallback,
    set_fft_plan_cache_size,
    set_fft_threads,
    set_pyfftw_planner,
    snapshot_fft_state,
    warmup_fft_plans,
)

# â”€â”€ Sources â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# â”€â”€ Off-axis + extended source helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .sources import (
    PartialCoherenceMCF,
    Source,
    create_annular_beam,
    create_annular_incoherent_source,
    create_bessel_beam,
    create_fiber_mode,
    create_gaussian_beam,
    # v4.15 (ROADMAP v4.16 #9, #11): Schell-model + annular-incoherent
    # partial-coherence source factories.
    # v4.15.1 (P0-NEW-2): the factories now return ensembles (or a
    # PartialCoherenceMCF) and actually deliver partial coherence.
    create_gaussian_schell_source,
    create_hermite_gauss,
    create_laguerre_gauss,
    create_led_source,
    create_multi_field_sources,
    create_point_source,
    create_schell_model_source,
    create_tilted_plane_wave,
    create_top_hat_beam,
    hermite_physicist,
    laguerre_generalized,
)

# v5.2 (ROADMAP v5.1 partial-coherence/MCF public-API polish):
# short top-level alias for symmetry with ``lumenairy.coherence_at``
# / ``lumenairy.propagate_ensemble``.  The canonical class name
# stays ``PartialCoherenceMCF``; ``MCF`` is a thin alias so the
# import story is uniform across the partial-coherence surface.
MCF = PartialCoherenceMCF

# â”€â”€ High-NA vector diffraction (Richards-Wolf) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# â”€â”€ Central cache-clearer registry (4.16.0) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# v4.16.0 (ROADMAP #15): retires the lazy-import fan-out in
# ``clear_asm_caches`` in favour of a central registry.  Each cache-
# owning module registers its clear function at import time;
# ``clear_asm_caches`` walks the registry rather than enumerating
# calls by hand.  Counter-measure to the "fix N, miss N+1" sibling-
# gap meta-pattern that recurred 5 ways inside v4.14.2 and again at
# v4.14.3.
from ._cache_registry import (
    list_registered_cache_clearers,
    register_cache_clearer,
)

# â”€â”€ Scoped runtime-environment context (4.8.1) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from ._context import (
    _install_atexit_restore,
    apply_globals,
    lumenairy_context,
    snapshot_globals,
)

# v5.4 Phase 5: 2-D Chebyshev fit primitive promoted out of the UI
# dock so notebook scripts and external callers can reuse the same
# tested LS solve.  Lives under the private ``_math`` package; we
# only re-export the fit helper (the Vandermonde-table primitives
# stay internal).
from ._math.chebyshev import chebyshev_fit_2d

# â”€â”€ Adaptive optics primitives â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# AO moved to analysis/ao.py in 4.3.0; lumenairy.ao still works via
# a back-compat shim.
from .analysis.ao import (
    DeformableMirror,
    LeakyIntegrator,
    # v5.2.3 (ROADMAP v5.2.x ao_closed_loop helper): canonical
    # high-level closed-loop AO driver, supersedes the v5.2.0
    # build-it-yourself pattern in examples/11_ao_closed_loop.py.
    ao_closed_loop,
    apply_dm,
    # v5.4 (AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24 P1-A): canonical
    # Shack-Hartmann WFS-callable factory for ao_closed_loop(wfs=...).
    make_shack_hartmann_wfs,
    slope_to_modal,
    zernike_modal_basis,
)

# â”€â”€ Partial coherence / extended-source imaging â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .analysis.coherence import (
    extended_source_image,
    koehler_image,
    mutual_coherence,
)

# â”€â”€ Detector model / wavefront sensing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .analysis.detector import (
    apply_detector,
    shack_hartmann,
)

# â”€â”€ Field-resolved analyses (4.4.0) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Lifted from ui/<dock>.py so distortion / footprint / spot-by-field /
# sensitivity ranking are reachable from scripts.  GUI docks now call
# these public functions and render the result.
from .analysis.field import (
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

# â”€â”€ Ghost analysis â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .analysis.ghost import (
    enumerate_ghost_paths,
    ghost_analysis,
    non_sequential_stray_light,
    retrace_ghost_path,
)

# â”€â”€ Interferometry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .analysis.interferometry import (
    fringe_spacing,
    phase_shift_extract,
    phase_step_roundtrip,
    simulate_interferogram,
)

# â”€â”€ Phase retrieval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .analysis.phase_retrieval import (
    error_reduction,
    gerchberg_saxton,
    hybrid_input_output,
)

# â”€â”€ Through-focus / tolerancing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .analysis.through_focus import (
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

# -- Shared byte-budgeted LRU cache infrastructure (roadmap P0) ------------
# The memory-safety foundation for every N^2-scale perf cache: a single
# collective byte ceiling (LUMENAIRY_CACHE_BUDGET_MB / set_cache_budget),
# LRU eviction within AND across caches, registry-drained by
# ``clear_asm_caches``, and introspectable via ``cache_report``.
from .cache import (
    ByteBudgetedLRU,
    cache_report,
    get_cache_budget,
    set_cache_budget,
)

# â”€â”€ Thin-film coatings â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .elements.berreman import (
    BerremanStack,
    berreman_jones_1d,
)

# -- BOR-PMM (body-of-revolution / axisymmetric cylindrical stack) --------
# v5.25 (audit S5-10 / B2): BORStack is the headline axisymmetric stack
# solver -- the cylindrical-coordinate peer of RCWAStack / PMMStack /
# BerremanStack, which are all top-level exported.  It graduates to the
# top level here for signature symmetry with those engines.  The
# lower-level BOR building blocks and analytic oracles (radial_spectrum,
# fiber_modes, layer_modes, fourier_bessel, ...) stay namespaced under
# ``la.elements.bor.*`` by design -- several carry generic or
# cylindrical-jargon names (``layer_modes`` also names an EME export) that
# would crowd or collide on the top-level surface.
from .elements.bor import BORStack

# â”€â”€ BSDF surface scatter (stray-light analysis) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .elements.bsdf import (
    BSDFModel,
    GaussianBSDF,
    HarveyShackBSDF,
    LambertianBSDF,
    make_bsdf,
    sample_scatter_rays,
)
from .elements.coatings import (
    # v5.4 Phase 5: thin-film coating material database and accessor.
    COATING_MATERIAL_REGISTRY,
    broadband_ar_v_coat,
    coating_reflectance,
    coating_reflectance_jax,
    get_coating_material_index,
    quarter_wave_ar,
)

# â”€â”€ DOE / Gratings / Phase I/O â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .elements.doe import (
    create_diffractive_lens,
    create_fresnel_zone_plate,
    create_kinoform,
    create_microlens_array,
    create_periodic_phase_mask,
    load_fits_field,
    load_phase_file,
    makedammann2d,
    save_fits_field,
    save_phase_file,
)
from .elements.emt import (
    bruggeman,
    maxwell_garnett,
    rytov_segments_tensor,
    rytov_tensor,
)

# â”€â”€ Freeform surfaces â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .elements.freeform import (
    surface_sag_chebyshev,
    surface_sag_freeform,
    surface_sag_q_bfs,
    surface_sag_q_con,
    surface_sag_xy_polynomial,
    surface_sag_zernike_freeform,
)
from .elements.materials import Material
from .elements.pmm import (
    PMMStack,
    classify_from_grating,
    grating_convergence_class,
    pmm_1d,
    pmm_efficiency_1d,
    pmm_efficiency_1d_jax,
    pmm_efficiency_1d_segments,
    pmm_efficiency_1d_slanted,
    pmm_efficiency_1d_vs_wavelength,
    pmm_graded_segments,
    pmm_jones_1d,
    pmm_jones_1d_conical,
    pmm_jones_1d_conical_tensor,
    pmm_jones_1d_segments,
    pmm_jones_1d_segments_vs_wavelength,
    pmm_jones_1d_slanted,
    pmm_jones_1d_slanted_segments,
    pmm_jones_1d_vs_wavelength,
)
from .elements.pmm.stack2d import (
    PMM2DStack,
    PMM2DStack_hybrid,
    PMM2DStackHybrid,
)
from .elements.pmm.stack2d_pure import (
    PMM2DStackPure,
)
from .elements.pmm.twod import (
    PreparedPMM2D,
    pmm_efficiency_2d,
    pmm_efficiency_2d_cell,
    pmm_efficiency_2d_cell_vs_wavelength,
    pmm_efficiency_2d_vs_wavelength,
    prepare_pmm_2d,
    prepare_pmm_2d_cell,
)
from .elements.pmm.twod_jones import (
    pmm_jones_2d,
)
from .elements.pmm.twod_staggered import (
    pmm_efficiency_2d_staggered,
)

# â”€â”€ Polarization / Jones calculus â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .elements.polarization import (
    JonesField,
    apply_half_wave_plate,
    apply_jones_matrix,
    apply_polarizer,
    apply_polarizing_beam_splitter,
    apply_quarter_wave_plate,
    apply_rotator,
    apply_waveplate,
    create_circular_polarized,
    create_elliptical_polarized,
    create_linear_polarized,
    degree_of_polarization,
    jones_field_from_orders,
    polarization_ellipse,
    stokes_parameters,
)

# Rigorous Coupled-Wave Analysis (full vector Fourier Modal Method): 1-D and
# 2-D crossed gratings, metals (Li inverse rule), and anisotropic / LC layers
# (Jones reflection).  The rigorous counterpart to the scalar thin_grating.
from .elements.rcwa import (
    Efficiency2D,
    PreparedRCWA2D,
    RCWAResult,
    RCWAStack,
    RCWAYAverageWarning,
    binary_grating_segments,
    grating_segments,
    interdigitated_grating_segments,
    jones_retardance_diattenuation,
    prepare_rcwa_2d,
    rcwa_blas_threads,
    rcwa_convergence,
    rcwa_efficiency_1d,
    rcwa_efficiency_1d_jax,
    rcwa_efficiency_2d,
    rcwa_efficiency_2d_shapes,
    rcwa_efficiency_2d_vs_wavelength,
    rcwa_efficiency_vs_wavelength,
    rcwa_extrapolate,
    rcwa_jones_1d,
    rcwa_jones_1d_segments,
    rcwa_jones_2d,
    rcwa_jones_vs_wavelength,
    rcwa_jones_vs_wavelength_segments,
    reflective_outcoupling,
    set_blas_threads,
    uniaxial_tensor,
)
from .elements.segment_geometry import (
    BACKGROUND,
    SegmentStackGeometry,
)

# â”€â”€ Grating diffraction efficiency (thin-grating scalar approx) â”€â”€â”€â”€â”€â”€â”€
# The function is named for what it actually computes: an analytical
# scalar thin-phase-grating diffraction-efficiency formula.
from .elements.thin_grating import (
    grating_efficiency_vs_wavelength,
    thin_grating_efficiency_1d,
)

# â”€â”€ Code generation: prescription -> standalone simulation script â”€â”€â”€â”€â”€â”€â”€
from .io.codegen import (
    generate_script_from_txt,
    generate_script_from_zmx,
    generate_simulation_script,
)

# â”€â”€ Lens prescriptions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .io.prescriptions import (
    QUADOA_SCHEMA_VERSION,
    THORLABS_CATALOG,
    combine_prescriptions,
    export_codev_seq,
    export_quadoa_qos,
    export_zemax_lens_data,
    export_zemax_zmx,
    has_mirrors,
    load_codev_seq,
    load_quadoa_qos,
    load_zemax_prescription_data_txt,
    load_zemax_zmx,
    make_biconic,
    make_cylindrical,
    make_doublet,
    make_off_axis_parabola,
    make_singlet,
    normalize_prescription,
    scale_prescription,
    split_prescription_at_mirrors,
    thorlabs_lens,
)

# â”€â”€ Storage (unified HDF5 / Zarr) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .io.storage import (
    TempFieldStore,
    append_plane,
    append_plane_h5,
    default_extension,
    get_storage_backend,
    list_h5_contents,
    list_planes,
    # Backwards-compatible aliases
    list_planes_store,
    load_field_h5,
    load_jones_field_h5,
    load_plane_by_label,
    load_plane_by_label_store,
    load_plane_slice,
    load_plane_slice_store,
    load_planes,
    load_planes_h5,
    read_metadata,
    read_sim_metadata,
    replay_run,
    # HDF5-specific functions
    save_field_h5,
    save_jones_field_h5,
    save_planes_h5,
    # Unified dispatch API
    set_storage_backend,
    write_metadata,
    write_sim_metadata,
)

# â”€â”€ Memory-aware batching helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .memory import (
    array_bytes,
    available_memory_bytes,
    bytes_per_element,
    check_sim_memory,
    estimate_asm_memory,
    estimate_lens_memory,
    estimate_op_memory,
    estimate_sim_memory,
    format_bytes,
    get_max_ram,
    get_ram_budget,
    memory_info,
    pick_batch_size,
    print_memory_report,
    set_low_memory,
    set_max_ram,
    should_split,
    total_memory_bytes,
)

# â”€â”€ Hybrid wave/ray design optimization â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .optimize import (
    # v4.16.0 (Agent A __all__-symmetry walker): the pymoo
    # availability flag is the canonical "is the optional Pareto
    # backend installed?" probe.  Sibling to JAX_AVAILABLE /
    # CUPY_AVAILABLE / NUMEXPR_AVAILABLE / PYFFTW_AVAILABLE.  Was
    # in the multi_objective submodule __all__ but never re-
    # exported at top level.
    PYMOO_AVAILABLE,
    WAVE_PROPAGATOR_REGISTRY,
    BackFocalLengthMerit,
    CallableMerit,
    ChromaticFocalShiftMerit,
    CompositeMerit,
    Constraint,
    DesignParameterization,
    DesignResult,
    EvaluationContext,
    FocalLengthMerit,
    JaxMeritTerm,
    LGAberrationMerit,
    MatchIdealSystemMerit,
    MatchIdealThinLensMerit,
    MatchTargetOPDMerit,
    MaxFNumberMerit,
    MaxThicknessMerit,
    MeritTerm,
    MinBackFocalLengthMerit,
    MinThicknessMerit,
    MultiFieldMerit,
    MultiPrescriptionParameterization,
    MultiWavelengthMerit,
    NormalizedMerit,
    # v4.16 (ROADMAP #11): multi-objective Pareto wrapper (pymoo-optional)
    ParetoResult,
    RawParameterization,
    RMSWavefrontMerit,
    SphericalSeidelMerit,
    SpotSizeMerit,
    StrehlMerit,
    ToleranceAwareMerit,
    ZernikeCoefficientMerit,
    design_optimize,
    design_optimize_multi_objective,
    make_lg_aberration_merit_jax,
    optimize_traced_geometry,
    register_wave_propagator,
    unregister_wave_propagator,
)

# â”€â”€ Multi-configuration + afocal mode â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .optimize.multiconfig import (
    Configuration,
    afocal_angular_magnification,
    beam_expander_prescription,
    create_zoom_configs,
    keplerian_telescope,
    multi_config_merit,
)

# â”€â”€ Progress reporting (opt-in hook for long-running functions) â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .progress import (
    CancellableProgress,
    ProgressCallback,
    ProgressScaler,
    call_progress,
    is_cancelled,
)

# â”€â”€ Phase-space asymptotic propagator + LG aberration tensor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .propagators.asymptotic import (
    AberrationTensorResult,
    CanonicalPolyFit,
    HFPolyFit,
    JaxAberrationTensorResult,
    aberration_tensor,
    aberration_tensor_lg00_jax,
    clear_lg_mode_stack_cache,
    clear_lg_polynomial_cache,
    decompose_hg,
    decompose_lg,
    evaluate_hg_mode,
    evaluate_lg_mode,
    fit_canonical_polynomials,
    fit_canonical_polynomials_jax,
    fit_hf_polynomials,
    gaussian_moment_2d,
    gaussian_moment_table_2d,
    hg_polynomial,
    lg_polynomial,
    lg_seidel_label,
    propagate_hf_chebyshev_quadrature,
    propagate_modal_asymptotic,
    propagate_modal_asymptotic_lg00_jax,
    solve_envelope_stationary,
    solve_envelope_stationary_jax_ift,
)

# â”€â”€ System propagation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .propagators.system import (
    clear_propagate_system_jax_cache,
    # v4.15 (ROADMAP v4.15 #3): ergonomic prescription + Source ->
    # PropagationResult one-call entry, exposed at top level so users
    # can also call ``lumenairy.evaluate(rx, source)`` directly.
    evaluate,
    propagate_through_system,
    propagate_through_system_jax,
)
from .propagators.vector_diffraction import (
    debye_wolf_psf,
    richards_wolf_focus,
)

# â”€â”€ Geometric ray tracing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .raytrace import (
    RAY_APERTURE,
    RAY_EVANESCENT,
    RAY_MISSED_SURFACE,
    RAY_NAN,
    RAY_OK,
    RAY_TIR,
    # v5.21 (__all__-symmetry): differential ray transfer (GBD analytic
    # Jacobian primitive) -- public per raytrace.differential.__all__.
    DifferentialTransfer,
    FirstOrderData,
    LensInfo,
    PupilInfo,
    RayBundle,
    Surface,
    TraceResult,
    apply_doe_phase_traced,
    astigmatism_waves_to_zernike,
    compute_pupils,
    defocus_waves_to_zernike,
    f_number,
    # Paraxial-design one-liner helpers (4.0+)
    field_of_view,
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
    optical_invariant,
    paraxial_focus_world,
    prescription_summary,
    ray_fan_data,
    ray_fan_plot,
    ray_fan_plot_prescription,
    ray_transfer_jacobian,
    ray_transfer_jacobian_analytic,
    ray_transfer_jacobian_jax,
    # v4.15.1 (Cluster B Item 6): wave -> ray bridge.
    rays_from_field,
    raytrace_system,
    refocus,
    seidel_coefficients,
    seidel_field_sweep,
    seidel_prescription,
    seidel_wfe,
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
    world_surfaces_from_prescription,
)

# Snapshot the import-time defaults and register an atexit handler that
# restores them on process shutdown.  Catches the foot-gun where users
# call set_default_complex_dtype / set_pyfftw_planner / etc. at module
# scope inside a long-running process (Jupyter, test harnesses) and
# expect a "clean" library state on the next unrelated run -- the
# handler restores whatever the defaults were at the very first
# ``import lumenairy``.
# v5.3 (AUDIT_V5_2_5 P3-6 closure): the v5.2.5 rename to the
# underscore-prefixed form was documentation-only -- both the
# library bootstrap and the test_context_manager test still used
# the legacy public name.  v5.3 migrates the library bootstrap to
# the underscore form.  The back-compat alias at ``_context.py``
# remains for any external caller still importing the legacy
# name.
_install_atexit_restore()

# â”€â”€ Plotting utilities â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# â”€â”€ Deprecated-alias shims â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#
# 4.12.0: wire ``_deprecation.deprecated_alias`` (added in 4.7 but never
# imported anywhere -- a Round-4 audit finding) into the top-level
# namespace so users with pre-4.7 code can keep calling the historical
# names with a one-cycle ``DeprecationWarning`` instead of a cold
# ``AttributeError``.  Each shim forwards to the canonical new name.
from ._deprecation import deprecated_alias as _deprecated_alias

# â”€â”€ Operator algebra (4.15.1, Cluster B Item 2) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Nazarathy/Shamir-style optical operator algebra over the existing
# array-first propagator infrastructure.  Composable optical primitives
# (FreeSpace, ThinLens, CylindricalLens, Magnify, FourierTransform,
# Aperture, GaussianAperture) with closed-form ABCDs and chain-and-
# delegate field application.  See ``lumenairy.algebra`` and
# ``docs/audits/CLUSTER_B_SPEC.md`` Â§3.
from .algebra import (
    Aperture,
    CompositeOperator,
    CylindricalLens,
    FourierTransform,
    FreeSpace,
    GaussianAperture,
    Magnify,
    Operator,
    ThinLens,
)
from .analysis.plotting import (
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

# â”€â”€ Multi-backend infrastructure (3.4.0) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Foundation for first-class NumPy / CuPy / JAX backend support.
from .backend import (
    JAX_AVAILABLE,
    RandomState,
    array_namespace,
    backend_name,
    is_cupy_array,
    is_jax_array,
    is_numpy_array,
    to_backend,
    to_numpy,
)
from .propagators.dispatch import (
    ASM_FAMILY,
    VALID_METHODS,
    asm_propagate,
    propagate,
    which_propagator,
)

# v4.16.1 (audit AUDIT_V4_16_0_DEEP P5 / P0-1): partial-coherence
# ensemble propagator helper.  Closes the Schell-model workflow that
# v4.15.x left half-shipped (factories returned ensembles, but no
# downstream helper propagated them; coherent propagators rejected
# 3-D inputs via :func:`_check_2d_scalar_field`).  The helper
# iterates each realisation through a coherent propagator and
# returns the partial-coherence intensity ``< |E_k|^2 >_k``.
from .propagators.ensemble import propagate_ensemble
from .propagators.fga import (
    apply_real_lens_auto,
    apply_real_lens_fga,
    apply_real_lens_fga_vector,
    apply_real_lens_universal,
    fga_memory_estimate,
)
from .propagators.gbd import (
    BeamletBundle,
    apply_abcd_to_beamlets,
    apply_aperture_to_beamlets,
    apply_prescription_persurface_to_beamlets,
    apply_thin_lens_to_beamlets,
    asm_field_to_gbd,
    converge_gbd_sampling,
    csp_beamlet_field,
    decompose_field_adaptive,
    decompose_field_to_beamlets,
    frame_completeness,
    gbd_asm_gouy_phase,
    gbd_field_to_asm,
    gbd_ghost_analysis,
    match_global_phase,
    propagate_beamlets_freespace,
    propagate_gbd,  # canonical-order alias
    propagate_gbd_freespace,
    propagate_gbd_freespace_csp,
    propagate_gbd_freespace_spectral,
    propagate_gbd_freespace_vector,
    propagate_gbd_thin_lens,
    propagate_gbd_through_prescription,
    propagate_gbd_vector_through_prescription,
    recommend_gbd_sampling,
    reconstruct_field_from_beamlets,
    reconstruct_vector_field_with_ez,
)
from .propagators.hf import (
    propagate_huygens_fresnel,  # canonical-order alias
    propagate_huygens_fresnel_freespace,
    propagate_huygens_fresnel_through_prescription,
    propagate_huygens_fresnel_with_opl_callable,
)

# â”€â”€ New propagators (3.4.0) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
from .propagators.hfpi import (
    PathBundle,
    accumulate_to_grid,
    apply_aperture_diffraction,
    init_paths_from_field,
    init_paths_stratified,
    propagate_hfpi,  # canonical-order alias
    propagate_hfpi_freespace_aperture,
    propagate_hfpi_through_prescription,
    propagate_to_plane,
)
from .propagators.mhs import (
    HuygensSurface,
    MhsPipeline,
    MhsSubdomain,
    aperture_subdomain,
    asm_subdomain,
    gbd_freespace_subdomain,
    prescription_subdomain,
)
from .propagators.result import PropagationResult
from .propagators.subaperture import (
    PatchGrid,
    combine_patch_fields,
    patch_window,
    patches_for_box,
    propagate_subaperture_asymptotic,
)
from .propagators.vectorial_hfpi import (
    VectorPathBundle,
    accumulate_vector_to_grid,
    apply_vector_aperture_diffraction,
    init_vector_paths_from_field,
    propagate_vector_hfpi_freespace_aperture,
    propagate_vector_to_plane,
)
from .raytrace.jax_trace import (
    JaxRayState,
    clear_trace_jax_cache,
    jax_state_to_raybundle,
    make_jax_ray_state,
    raybundle_to_jax_state,
    trace_jax,
)

# C.2 (v4.7): Zemax-loader renames.
# S4-17 (AUDIT_V5_24_2): these two aliases were originally announced for
# removal in v5.0, but v5.0's back-compat-shim purge (see Migration-Guide
# "5.0.0 -- Back-compat shim removal") did NOT include them and they have
# shipped as convenience re-exports ever since.  The removal target is
# realigned to v6.0 so the emitted warning ("will be removed in v6.0")
# stays self-consistent with the current version instead of naming a
# release that already passed.
load_zmx_prescription = _deprecated_alias(
    load_zemax_zmx,
    old_name='load_zmx_prescription',
    version_added='4.7',
    version_removed='6.0',
)
load_zemax_prescription_txt = _deprecated_alias(
    load_zemax_prescription_data_txt,
    old_name='load_zemax_prescription_txt',
    version_added='4.7',
    version_removed='6.0',
)

__version__ = "5.35.0"

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
    # v5.2: short alias for symmetry with the other top-level
    # partial-coherence symbols.
    'MCF',
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
    'combine_prescriptions',
    'has_mirrors',

    # Lens / element models (apply on a field)
    'apply_thin_lens',
    'apply_spherical_lens',
    'apply_aspheric_lens',
    'apply_real_lens',
    'set_lens_parallel_amp',
    'get_lens_parallel_amp',
    'set_lens_sag_dtype',
    'get_lens_sag_dtype',
    'set_pointwise_cos_grid_cache_budget',
    'get_pointwise_cos_grid_cache_budget',
    'clear_pointwise_cos_grid_cache',
    'lens_sag_float32_opd_error',
    'apply_real_lens_traced',
    'apply_real_lens_traced_multi',
    'apply_real_lens_traced_multibranch',
    'apply_real_lens_traced_uniform',
    'apply_real_lens_traced_segmented',
    'prepare_real_lens_traced',
    'PreparedTracedLens',
    'TiltedCarrier',
    'prepare_real_lens',
    'PreparedAnalyticLens',
    'close_worker_pool',
    'apply_real_lens_gbd',
    'apply_real_lens_fga',
    'apply_real_lens_fga_vector',
    'apply_real_lens_auto',
    'apply_real_lens_universal',
    'fga_memory_estimate',
    'apply_real_lens_maslov',
    'apply_real_lens_traced_jax',
    'apply_real_lens_maslov_jax',
    'apply_real_lens_maslov_vector',
    'pearcey',
    'uniform_fold_airy',
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
    # v5.4 Phase 5: 2-D Chebyshev fit helper (LS solve for the
    # ``{(i, j): c_ij}`` dict that ``surface_sag_chebyshev`` consumes).
    'chebyshev_fit_2d',

    # Polarization / Jones calculus
    'JonesField',
    'apply_jones_matrix',
    'apply_polarizer',
    'apply_polarizing_beam_splitter',
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
    'jones_field_from_orders',

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
    # v5.4 Phase 5: phase-mask coronagraph builders (FQPM + 8OPM)
    # promoted from inline ``coronagraph_dock`` helper to canonical
    # library functions.
    'create_four_quadrant_phase_mask',
    'create_eight_octant_phase_mask',
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
    # v4.16.1: partial-coherence ensemble propagator helper.
    'propagate_ensemble',

    # Free-space propagators (low-level)
    'angular_spectrum_propagate',
    'angular_spectrum_propagate_tilted',
    'scalable_angular_spectrum_propagate',
    'fresnel_propagate',
    'fresnel_tf_propagate',
    'fraunhofer_propagate',
    'rayleigh_sommerfeld_propagate',
    'resample_field',
    'apply_fresnel_curvature',
    # Carrier-referenced ("pilot-beam") Sziklas-Siegman free-space step
    'CarrierReferencedField',
    'TracedCarrierChainResult',
    'TracedCarrierChainMultiResult',
    'propagate_carrier_referenced',
    'carrier_referenced_reconstruct',
    'carrier_referenced_envelope',
    'carrier_referenced_aperture',
    'carrier_referenced_fit_radius',
    'carrier_referenced_focus_readout',
    'carrier_referenced_exact_focus_readout',
    'propagate_traced_carrier_chain',
    'propagate_traced_carrier_chain_multi',

    # Field-aggregation primitives (carrier_field).  Same order as
    # ``lumenairy.propagators.__all__`` so the two lists read alike.
    'CarrierSpec',
    'FieldGrid',
    'CarrierField',
    'NyquistReport',
    'ReReferenceReport',
    'FieldLedgerRow',
    'AggregateLedger',
    'AggregateResult',
    'carrier_difference_nyquist',
    're_reference',
    'aggregate',
    'save_carrier_field_zarr',
    'load_carrier_field_zarr',
    'CARRIER_FIELD_SCHEMA',
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
    'decompose_field_adaptive',
    'frame_completeness',
    'converge_gbd_sampling',
    'csp_beamlet_field',
    'propagate_gbd_freespace_csp',
    'reconstruct_vector_field_with_ez',
    'apply_aperture_to_beamlets',
    'apply_prescription_persurface_to_beamlets',
    'gbd_asm_gouy_phase',
    'gbd_field_to_asm',
    'gbd_ghost_analysis',
    'match_global_phase',
    'asm_field_to_gbd',
    'recommend_gbd_sampling',
    'propagate_beamlets_freespace',
    'apply_thin_lens_to_beamlets',
    'apply_abcd_to_beamlets',
    'reconstruct_field_from_beamlets',
    'propagate_gbd_freespace',
    'propagate_gbd',
    'propagate_gbd_thin_lens',
    'propagate_gbd_through_prescription',
    'propagate_gbd_freespace_spectral',
    'propagate_gbd_freespace_vector',
    'propagate_gbd_vector_through_prescription',

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
    # v5.21: differential ray transfer (GBD analytic-Jacobian primitive)
    'DifferentialTransfer',
    'ray_transfer_jacobian',
    'ray_transfer_jacobian_analytic',
    'ray_transfer_jacobian_jax',
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
    # v5.2.3 (ROADMAP v5.2.x ao_closed_loop helper):
    'ao_closed_loop',
    # v5.4 (AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24 P1-A):
    'make_shack_hartmann_wfs',

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
    'phase_step_roundtrip',
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
    'retrace_ghost_path',

    # ============================================================
    # Tier 5 -- Optimize
    # ============================================================

    'DesignParameterization',
    'MultiPrescriptionParameterization',
    'RawParameterization',
    'EvaluationContext',
    'DesignResult',
    'Constraint',
    'design_optimize',
    # v5.21: traced-lens geometry optimizer (jax geometry-gradient loop)
    'optimize_traced_geometry',
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
    'NormalizedMerit',
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

    # RCWA / Fourier Modal Method (rigorous vector grating solver)
    'rcwa_efficiency_1d',
    'rcwa_efficiency_1d_jax',
    'rcwa_efficiency_2d',
    'rcwa_efficiency_2d_shapes',
    'rcwa_efficiency_2d_vs_wavelength',
    'rcwa_efficiency_vs_wavelength',
    'prepare_rcwa_2d',
    'PreparedRCWA2D',
    'rcwa_extrapolate',
    'rcwa_convergence',
    'rcwa_jones_1d',
    'rcwa_jones_1d_segments',
    'rcwa_jones_2d',
    'rcwa_jones_vs_wavelength',
    'rcwa_jones_vs_wavelength_segments',
    'grating_segments',
    'binary_grating_segments',
    'interdigitated_grating_segments',
    'reflective_outcoupling',
    'jones_retardance_diattenuation',
    'uniaxial_tensor',
    'RCWAStack',
    'RCWAResult',
    'RCWAYAverageWarning',
    'Efficiency2D',
    'set_blas_threads',
    'rcwa_blas_threads',

    # PMM (Polynomial Modal Method -- non-Fourier 1-D modal solver)
    'pmm_efficiency_1d',
    'pmm_efficiency_1d_jax',
    'pmm_efficiency_1d_segments',
    'pmm_efficiency_1d_slanted',
    'pmm_1d',
    'pmm_jones_1d',
    'pmm_jones_1d_conical',
    'pmm_jones_1d_conical_tensor',
    'pmm_jones_1d_segments',
    'pmm_jones_1d_slanted',
    'pmm_jones_1d_slanted_segments',
    'pmm_efficiency_1d_vs_wavelength',
    'Material',
    'SegmentStackGeometry',
    'BACKGROUND',
    'pmm_jones_1d_vs_wavelength',
    'pmm_jones_1d_segments_vs_wavelength',
    'pmm_graded_segments',
    'pmm_efficiency_2d',
    'pmm_efficiency_2d_cell',
    'pmm_efficiency_2d_vs_wavelength',
    'pmm_efficiency_2d_cell_vs_wavelength',
    'pmm_jones_2d',
    'PMM2DStack',
    'PMM2DStackHybrid',
    'PMM2DStack_hybrid',
    'PMM2DStackPure',
    'prepare_pmm_2d',
    'prepare_pmm_2d_cell',
    'PreparedPMM2D',
    'pmm_efficiency_2d_staggered',
    'grating_convergence_class',
    'classify_from_grating',
    'PMMStack',

    # BSDF / surface scatter
    'BSDFModel',
    'LambertianBSDF',
    'GaussianBSDF',
    'HarveyShackBSDF',
    'make_bsdf',
    'sample_scatter_rays',

    # Thin-film coatings
    'coating_reflectance',
    'coating_reflectance_jax',
    'quarter_wave_ar',
    'broadband_ar_v_coat',

    # Berreman 4x4 (anisotropic planar multilayer)
    'berreman_jones_1d',
    'BerremanStack',

    # BOR-PMM axisymmetric cylindrical stack (peer of RCWA/PMM/Berreman
    # stacks).  Lower-level BOR helpers stay under ``la.elements.bor.*``.
    'BORStack',

    # Effective-medium (EMT) homogenization bridge
    'rytov_tensor',
    'rytov_segments_tensor',
    'maxwell_garnett',
    'bruggeman',
    # v5.4 Phase 5: thin-film coating material database.
    'COATING_MATERIAL_REGISTRY',
    'get_coating_material_index',

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

    # Precision + default-config knobs
    'set_default_complex_dtype',
    'get_default_complex_dtype',
    'DEFAULT_COMPLEX_DTYPE',
    'set_default_real_dtype',
    'get_default_real_dtype',
    # v4.16.3 (audit P3-NEW-F2-LOW-1): sibling parity with
    # ``DEFAULT_COMPLEX_DTYPE`` -- the v4.16.2 default-config knob
    # module-level globals are now first-class at top level.
    'DEFAULT_REAL_DTYPE',
    'DEFAULT_WAVE_PROPAGATOR',
    'DEFAULT_WAVE_PROPAGATOR_SHIPPED',
    'DEFAULT_DY',
    'set_default_wave_propagator',
    'get_default_wave_propagator',
    'set_default_dy',
    'get_default_dy',
    'set_asm_cache_size',
    'get_asm_cache_size',
    'set_fft_plan_cache_size',
    'get_fft_plan_cache_size',
    'set_fft_double_buffer',
    'get_fft_double_buffer',
    'set_pyfftw_planner',
    'get_pyfftw_planner',
    'set_fft_auto_promote',
    'get_fft_auto_promote',
    'snapshot_fft_state',
    'restore_fft_state',
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
    # roadmap P0: shared byte-budgeted LRU cache infrastructure.
    'ByteBudgetedLRU',
    'cache_report',
    'get_cache_budget',
    'set_cache_budget',

    # Memory-aware batching
    'available_memory_bytes',
    'total_memory_bytes',
    'memory_info',
    'bytes_per_element',
    'array_bytes',
    'estimate_op_memory',
    'estimate_lens_memory',
    'estimate_asm_memory',
    'estimate_sim_memory',
    'check_sim_memory',
    'set_low_memory',
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


# ---------------------------------------------------------------------------
# A-3: PEP-562 live forwarding for the mutable DEFAULT_* config knobs.
# ---------------------------------------------------------------------------
#
# audit AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 finding A-3: the four
# ``DEFAULT_*`` names re-exported near the top of this file were
# import-time SNAPSHOTS.  The canonical mutable globals live in
# :mod:`lumenairy.propagators.fft_infra` and the four ``set_default_*``
# setters mutate them there, so after
# ``la.set_default_complex_dtype('complex64')`` the getter
# ``la.get_default_complex_dtype()`` returned complex64 while the
# constant ``la.DEFAULT_COMPLEX_DTYPE`` still read complex128 -- the
# same defect, at the same names, that
# ``lumenairy/propagators/propagation.py:296`` already fixes for the
# submodule path.  Replicating that precedent exactly: whitelist the
# live names, DELETE the stale module-level bindings so attribute
# lookup falls through, and forward to the canonical module in
# ``__getattr__``.
#
# Consequences of the fall-through, all intended:
#   * ``la.DEFAULT_COMPLEX_DTYPE`` and ``from lumenairy import
#     DEFAULT_COMPLEX_DTYPE`` both return the CURRENT value.
#   * every ``__all__`` entry still resolves via ``getattr`` (the
#     export-integrity pin in tests/unit/test_public_api.py, which
#     parametrises over all 688 top-level entries, stays green), and a
#     phantom name still raises ``AttributeError``.
#   * ``__dir__`` is overridden so the four names remain visible to
#     ``dir(lumenairy)`` / tab-completion despite not being in
#     ``globals()``.
_LIVE_FORWARD_NAMES = frozenset({
    'DEFAULT_COMPLEX_DTYPE',
    'DEFAULT_REAL_DTYPE',
    'DEFAULT_WAVE_PROPAGATOR',
    'DEFAULT_DY',
})

# Drop the import-time snapshots (see propagation.py:285-293 for the
# identical step).  Without this, normal attribute lookup succeeds on the
# stale binding and ``__getattr__`` is never consulted.
for _name in _LIVE_FORWARD_NAMES:
    if _name in globals():
        del globals()[_name]
del _name


def __getattr__(name):
    """Forward the mutable ``DEFAULT_*`` knobs to their live values.

    Mirrors :func:`lumenairy.propagators.propagation.__getattr__`; the
    canonical globals live in ``propagators.fft_infra`` (the module the
    ``set_default_*`` setters mutate), so one hop gets the current value.
    """
    if name in _LIVE_FORWARD_NAMES:
        from .propagators import fft_infra as _fft_infra
        return getattr(_fft_infra, name)
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LIVE_FORWARD_NAMES)

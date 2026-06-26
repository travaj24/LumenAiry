"""
lumenairy.elements -- optical-element family.

Submodules:

* :mod:`lumenairy.elements.lenses` -- thin/spherical/aspheric/real
  lens phase application, ABCD helpers, Maslov-form lens, the
  largest module.
* :mod:`lumenairy.elements.doe` -- diffractive optical elements
  (binary phase, Dammann gratings, Fresnel zone plates).
* :mod:`lumenairy.elements.coatings` -- thin-film coating models
  (quarter-wave AR, broadband AR-V, generic stack reflectance).
* :mod:`lumenairy.elements.freeform` -- XY polynomial / Q-type
  orthogonal / Chebyshev freeform surface sag.
* :mod:`lumenairy.elements.elements` -- catalog of canonical
  optical elements (mirror, aperture, mask, Zernike phase plate,
  turbulence screen).
* :mod:`lumenairy.elements.thin_grating` -- 1-D thin-grating
  diffraction efficiencies (analytical scalar thin-phase model).
* :mod:`lumenairy.elements.polarization` -- Jones-pupil
  polarization, Jones-field operations.
* :mod:`lumenairy.elements.bsdf` -- BSDF surface scatter for
  stray-light analysis.
"""

from .berreman import (
    BerremanStack,
    berreman_jones_1d,
)
from .bsdf import (
    BSDFModel,
    GaussianBSDF,
    HarveyShackBSDF,
    LambertianBSDF,
    make_bsdf,
    sample_scatter_rays,
)
from .coatings import (
    broadband_ar_v_coat,
    coating_reflectance,
    quarter_wave_ar,
)
from .doe import (
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
    coronagraph_contrast_curve,
    create_eight_octant_phase_mask,
    create_four_quadrant_phase_mask,
    generate_turbulence_screen,
    zernike,
)
from .emt import (
    bruggeman,
    maxwell_garnett,
    rytov_segments_tensor,
    rytov_tensor,
)
from .freeform import (
    surface_sag_chebyshev,
    surface_sag_freeform,
    surface_sag_q_bfs,
    surface_sag_q_con,
    surface_sag_xy_polynomial,
    surface_sag_zernike_freeform,
)
from .lenses import (
    NUMEXPR_AVAILABLE,
    apply_aspheric_lens,
    apply_axicon,
    apply_cylindrical_lens,
    apply_grin_lens,
    apply_real_lens,
    apply_real_lens_maslov,
    apply_real_lens_traced,
    apply_spherical_lens,
    apply_thin_lens,
    check_grid_vs_apertures,
    get_lens_parallel_amp,
    recommend_grid_for_prescription,
    set_lens_parallel_amp,
    surface_sag_biconic,
    surface_sag_general,
)
from .pmm import (
    PMMStack,
    classify_from_grating,
    grating_convergence_class,
    pmm_1d,
    pmm_efficiency_1d,
    pmm_efficiency_1d_jax,
    pmm_efficiency_1d_segments,
    pmm_efficiency_1d_slanted,
    pmm_jones_1d,
    pmm_jones_1d_segments,
    pmm_jones_1d_slanted,
    pmm_jones_1d_slanted_segments,
)
from .pmm.twod import (
    pmm_efficiency_2d,
)
from .pmm.twod_staggered import (
    pmm_efficiency_2d_staggered,
)
from .polarization import (
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
    polarization_ellipse,
    stokes_parameters,
)
from .rcwa import (
    Efficiency2D,
    RCWAResult,
    RCWAStack,
    binary_grating_segments,
    grating_segments,
    interdigitated_grating_segments,
    jones_retardance_diattenuation,
    rcwa_blas_threads,
    rcwa_convergence,
    rcwa_efficiency_1d,
    rcwa_efficiency_1d_jax,
    rcwa_efficiency_2d,
    rcwa_efficiency_2d_shapes,
    rcwa_efficiency_vs_wavelength,
    rcwa_jones_1d,
    rcwa_jones_1d_segments,
    rcwa_jones_2d,
    rcwa_jones_vs_wavelength,
    rcwa_jones_vs_wavelength_segments,
    reflective_outcoupling,
    set_blas_threads,
    uniaxial_tensor,
)
from .thin_grating import (
    grating_efficiency_vs_wavelength,
    thin_grating_efficiency_1d,
)

__all__ = [
    # lenses
    'apply_thin_lens', 'apply_spherical_lens', 'apply_aspheric_lens',
    'apply_real_lens', 'apply_real_lens_traced', 'apply_real_lens_maslov',
    'set_lens_parallel_amp', 'get_lens_parallel_amp',
    'apply_cylindrical_lens', 'apply_grin_lens', 'apply_axicon',
    'surface_sag_general', 'surface_sag_biconic',
    'check_grid_vs_apertures', 'recommend_grid_for_prescription',
    'NUMEXPR_AVAILABLE',
    # doe
    'create_periodic_phase_mask', 'create_microlens_array',
    'create_diffractive_lens', 'create_kinoform',
    'create_fresnel_zone_plate',
    'makedammann2d',
    'load_phase_file', 'save_phase_file', 'load_fits_field', 'save_fits_field',
    # coatings
    'coating_reflectance', 'quarter_wave_ar', 'broadband_ar_v_coat',
    # berreman (anisotropic planar multilayer)
    'berreman_jones_1d', 'BerremanStack',
    # emt (effective-medium homogenization bridge)
    'rytov_tensor', 'rytov_segments_tensor', 'maxwell_garnett', 'bruggeman',
    # freeform
    'surface_sag_xy_polynomial', 'surface_sag_zernike_freeform',
    'surface_sag_chebyshev',
    'surface_sag_q_bfs', 'surface_sag_q_con',
    'surface_sag_freeform',
    # elements
    'apply_mirror', 'apply_aperture', 'apply_gaussian_aperture',
    'apply_mask', 'zernike', 'apply_zernike_aberration',
    'apply_lyot_focal_plane_mask', 'apply_vortex_phase_mask',
    'apply_lyot_stop', 'apply_apodized_pupil',
    # v5.4 Phase 5: canonical phase-mask builders for the four-quadrant
    # (Rouan 2000) and eight-octant (Murakami 2008) focal-plane masks.
    'create_four_quadrant_phase_mask', 'create_eight_octant_phase_mask',
    'coronagraph_contrast_curve',
    'generate_turbulence_screen',
    # thin-grating diffraction efficiency (analytical scalar model)
    'thin_grating_efficiency_1d',
    'grating_efficiency_vs_wavelength',
    # RCWA / Fourier Modal Method (rigorous vector grating solver)
    'rcwa_efficiency_1d',
    'rcwa_efficiency_1d_jax',
    'pmm_1d',
    'pmm_efficiency_1d',
    'pmm_efficiency_1d_jax',
    'pmm_efficiency_1d_segments',
    'pmm_efficiency_1d_slanted',
    'pmm_jones_1d',
    'pmm_jones_1d_segments',
    'pmm_jones_1d_slanted',
    'pmm_jones_1d_slanted_segments',
    'PMMStack',
    'pmm_efficiency_2d',
    'pmm_efficiency_2d_staggered',
    'grating_convergence_class',
    'classify_from_grating',
    'Efficiency2D',
    'rcwa_efficiency_2d',
    'rcwa_efficiency_2d_shapes',
    'rcwa_efficiency_vs_wavelength',
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
    'set_blas_threads',
    'rcwa_blas_threads',
    # polarization
    'JonesField',
    'apply_jones_matrix', 'apply_polarizer', 'apply_waveplate',
    'apply_half_wave_plate', 'apply_quarter_wave_plate', 'apply_rotator',
    'apply_polarizing_beam_splitter',
    'create_linear_polarized', 'create_circular_polarized',
    'create_elliptical_polarized', 'stokes_parameters',
    'degree_of_polarization', 'polarization_ellipse',
    # bsdf
    'BSDFModel', 'LambertianBSDF', 'GaussianBSDF', 'HarveyShackBSDF',
    'make_bsdf', 'sample_scatter_rays',
]

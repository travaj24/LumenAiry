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
* :mod:`lumenairy.elements.rcwa` -- 1-D thin-grating RCWA.
* :mod:`lumenairy.elements.polarization` -- Jones-pupil
  polarization, Jones-field operations.
* :mod:`lumenairy.elements.bsdf` -- BSDF surface scatter for
  stray-light analysis.
"""

from .lenses import (
    apply_thin_lens,
    apply_spherical_lens,
    apply_aspheric_lens,
    apply_real_lens,
    apply_real_lens_traced,
    apply_real_lens_maslov,
    apply_cylindrical_lens,
    apply_grin_lens,
    apply_axicon,
    surface_sag_general,
    surface_sag_biconic,
    check_grid_vs_apertures,
    recommend_grid_for_prescription,
    NUMEXPR_AVAILABLE,
)
from .doe import (
    create_periodic_phase_mask,
    create_microlens_array,
    makedammann2d,
    load_phase_file,
    save_phase_file,
    load_fits_field,
    save_fits_field,
)
from .coatings import (
    coating_reflectance,
    quarter_wave_ar,
    broadband_ar_v_coat,
)
from .freeform import (
    surface_sag_xy_polynomial,
    surface_sag_zernike_freeform,
    surface_sag_chebyshev,
    surface_sag_freeform,
)
from .elements import (
    apply_mirror,
    apply_aperture,
    apply_gaussian_aperture,
    apply_mask,
    zernike,
    apply_zernike_aberration,
    generate_turbulence_screen,
)
from .rcwa import (
    rcwa_1d,
    grating_efficiency_vs_wavelength,
)
from .polarization import (
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
from .bsdf import (
    BSDFModel,
    LambertianBSDF,
    GaussianBSDF,
    HarveyShackBSDF,
    make_bsdf,
    sample_scatter_rays,
)


__all__ = [
    # lenses
    'apply_thin_lens', 'apply_spherical_lens', 'apply_aspheric_lens',
    'apply_real_lens', 'apply_real_lens_traced', 'apply_real_lens_maslov',
    'apply_cylindrical_lens', 'apply_grin_lens', 'apply_axicon',
    'surface_sag_general', 'surface_sag_biconic',
    'check_grid_vs_apertures', 'recommend_grid_for_prescription',
    'NUMEXPR_AVAILABLE',
    # doe
    'create_periodic_phase_mask', 'create_microlens_array', 'makedammann2d',
    'load_phase_file', 'save_phase_file', 'load_fits_field', 'save_fits_field',
    # coatings
    'coating_reflectance', 'quarter_wave_ar', 'broadband_ar_v_coat',
    # freeform
    'surface_sag_xy_polynomial', 'surface_sag_zernike_freeform',
    'surface_sag_chebyshev', 'surface_sag_freeform',
    # elements
    'apply_mirror', 'apply_aperture', 'apply_gaussian_aperture',
    'apply_mask', 'zernike', 'apply_zernike_aberration',
    'generate_turbulence_screen',
    # rcwa
    'rcwa_1d', 'grating_efficiency_vs_wavelength',
    # polarization
    'JonesField',
    'apply_jones_matrix', 'apply_polarizer', 'apply_waveplate',
    'apply_half_wave_plate', 'apply_quarter_wave_plate', 'apply_rotator',
    'create_linear_polarized', 'create_circular_polarized',
    'create_elliptical_polarized', 'stokes_parameters',
    'degree_of_polarization', 'polarization_ellipse',
    # bsdf
    'BSDFModel', 'LambertianBSDF', 'GaussianBSDF', 'HarveyShackBSDF',
    'make_bsdf', 'sample_scatter_rays',
]

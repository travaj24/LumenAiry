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

Known cross-engine gaps (tracked, not yet built)
------------------------------------------------
* **S5-9 (AUDIT_V5_24_2) -- no shared ``LayerSpec`` across stacks.**
  The RCWA stack samples permittivity onto Fourier-order cells, the
  PMM stack carries spectral-element segments, and the Berreman / EME
  / BOR stacks each use their own layer representation.  There is no
  neutral geometry object that every engine can consume, so a single
  physical stack cannot be replayed across engines for cross-
  validation without hand-rebuilding it per engine.  Deferred by
  design (a half-built shared spec that silently drifts from any one
  engine's convention is worse than the current explicit per-engine
  construction).  Tracked in ``ROADMAP.md`` -> "Open / tracked gaps".
"""

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
from .lens_config import (
    LensConfig,
    LensGeometry,
    LensNumerics,
    LensPhysics,
    LensResources,
)
from .lenses import (
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
    apply_real_lens_traced,
    apply_real_lens_traced_multi,
    apply_real_lens_traced_multibranch,
    apply_real_lens_traced_segmented,
    apply_real_lens_traced_uniform,
    apply_spherical_lens,
    apply_thin_lens,
    check_grid_vs_apertures,
    clear_pointwise_cos_grid_cache,
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
    jones_field_from_orders,
    polarization_ellipse,
    stokes_parameters,
)
from .thin_grating import (
    grating_efficiency_vs_wavelength,
    thin_grating_efficiency_1d,
)

# ---------------------------------------------------------------------------
# PEP 562 lazy loading of the rigorous-solver subpackages
# ---------------------------------------------------------------------------
# ``berreman`` / ``bor`` / ``eme`` / ``pmm`` / ``rcwa`` are the heavy end of
# the library: they reach ``scipy.linalg`` + ``scipy.special`` through
# ``lumenairy.backend.scipy``, and a user who only wants ``propagate_asm``
# would otherwise pay for the whole rigorous-solver stack at
# ``import lumenairy`` (audit 2026-09-11 TESTS-ARCH P2-7).  Their names
# are resolved on FIRST ACCESS through the module ``__getattr__`` below.
#
# What still works, and is tested in
# ``tests/unit/test_audit2609_a15b_lazy_elements.py``:
#   * ``from lumenairy.elements import rcwa`` / ``import
#     lumenairy.elements.rcwa as r`` -- normal import machinery, untouched;
#   * ``lumenairy.elements.rcwa_efficiency_1d`` -- resolved here, then CACHED
#     into this module's globals, so the second access is a plain dict hit;
#   * ``lumenairy.elements.rcwa.X`` -- ``__getattr__('rcwa')`` imports the
#     subpackage (and CPython binds it as this package's attribute anyway);
#   * ``dir(lumenairy.elements)``, ``from lumenairy.elements import *`` and
#     the repo's ``__all__`` walkers -- via ``__dir__`` + the unchanged
#     ``__all__`` below;
#   * pickling objects defined in those modules -- pickle resolves by
#     ``__module__``/``__qualname__`` and imports the defining module itself.
#
# An unknown name raises ``AttributeError`` (never ``ImportError``), because
# ``hasattr`` and ``getattr(..., default)`` must keep working.
import importlib as _importlib
# Annotations for the PEP 562 pair below.  Bound under UNDERSCORE names:
# this module's namespace is public API, so ``elements.Any`` would be a
# name the surface walkers would have to learn to ignore.
from typing import Any as _Any
from typing import List as _List

#: Subpackages exposed as lazy attributes of this package.
_LAZY_SUBMODULES = ('berreman', 'bor', 'eme', 'pmm', 'rcwa')

#: Public name -> the submodule (relative to this package) that defines it.
#: One entry per lazily-resolved public name; the ``__all__``
#: list below is unchanged, so the two are cross-checked by the lazy-loading
#: test and by the existing ``__all__``-symmetry walker.
_LAZY_NAMES = {
    # berreman (anisotropic planar multilayer)
    'BerremanStack': 'berreman',
    'berreman_jones_1d': 'berreman',
    # pmm (polynomial modal method)
    'PMMStack': 'pmm',
    'classify_from_grating': 'pmm',
    'grating_convergence_class': 'pmm',
    'pmm_1d': 'pmm',
    'pmm_efficiency_1d': 'pmm',
    'pmm_efficiency_1d_jax': 'pmm',
    'pmm_efficiency_1d_segments': 'pmm',
    'pmm_efficiency_1d_slanted': 'pmm',
    'pmm_jones_1d': 'pmm',
    'pmm_jones_1d_segments': 'pmm',
    'pmm_jones_1d_slanted': 'pmm',
    'pmm_jones_1d_slanted_segments': 'pmm',
    'pmm_efficiency_2d': 'pmm.twod',
    'pmm_efficiency_2d_staggered': 'pmm.twod_staggered',
    # rcwa (Fourier modal method)
    'Efficiency2D': 'rcwa',
    'RCWAResult': 'rcwa',
    'RCWAStack': 'rcwa',
    'binary_grating_segments': 'rcwa',
    'grating_segments': 'rcwa',
    'interdigitated_grating_segments': 'rcwa',
    'jones_retardance_diattenuation': 'rcwa',
    'rcwa_blas_threads': 'rcwa',
    'rcwa_convergence': 'rcwa',
    'rcwa_efficiency_1d': 'rcwa',
    'rcwa_efficiency_1d_jax': 'rcwa',
    'rcwa_efficiency_2d': 'rcwa',
    'rcwa_efficiency_2d_shapes': 'rcwa',
    'rcwa_efficiency_vs_wavelength': 'rcwa',
    'rcwa_jones_1d': 'rcwa',
    'rcwa_jones_1d_segments': 'rcwa',
    'rcwa_jones_2d': 'rcwa',
    'rcwa_jones_vs_wavelength': 'rcwa',
    'rcwa_jones_vs_wavelength_segments': 'rcwa',
    'reflective_outcoupling': 'rcwa',
    'set_blas_threads': 'rcwa',
    'uniaxial_tensor': 'rcwa',
}


def __getattr__(name: str) -> _Any:
    """Resolve a rigorous-solver name on first access (PEP 562).

    ``-> Any`` and not narrower: the table resolves both SUBMODULES and the
    38 solver classes / functions they define, which share no useful type.
    """
    if name in _LAZY_SUBMODULES:
        mod = _importlib.import_module(f'{__name__}.{name}')
        globals()[name] = mod
        return mod
    where = _LAZY_NAMES.get(name)
    if where is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}")
    obj = getattr(_importlib.import_module(f'{__name__}.{where}'), name)
    globals()[name] = obj      # cache: later accesses skip this function
    return obj


def __dir__() -> _List[str]:
    """``dir()`` lists the lazy names too, so tab-completion and the repo's
    surface walkers see the same package they saw when it was eager."""
    return sorted(set(globals()) | set(_LAZY_NAMES) | set(_LAZY_SUBMODULES))

__all__ = [
    # lenses
    'apply_thin_lens', 'apply_spherical_lens', 'apply_aspheric_lens',
    'apply_real_lens', 'apply_real_lens_gbd', 'apply_real_lens_traced',
    'apply_real_lens_maslov',
    'apply_real_lens_traced_multi', 'apply_real_lens_traced_segmented',
    'apply_real_lens_traced_multibranch',
    'apply_real_lens_traced_uniform',
    'prepare_real_lens_traced', 'PreparedTracedLens', 'TiltedCarrier',
    'prepare_real_lens', 'PreparedAnalyticLens',
    # lens_config -- configuration objects for the apply_real_lens family
    'LensGeometry', 'LensNumerics', 'LensResources', 'LensPhysics',
    'LensConfig',
    'set_lens_parallel_amp', 'get_lens_parallel_amp',
    'set_lens_sag_dtype', 'get_lens_sag_dtype',
    'set_pointwise_cos_grid_cache_budget',
    'get_pointwise_cos_grid_cache_budget',
    'clear_pointwise_cos_grid_cache',
    'lens_sag_float32_opd_error',
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
    'jones_field_from_orders',
    # bsdf
    'BSDFModel', 'LambertianBSDF', 'GaussianBSDF', 'HarveyShackBSDF',
    'make_bsdf', 'sample_scatter_rays',
]

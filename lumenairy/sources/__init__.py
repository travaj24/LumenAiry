"""
lumenairy.sources -- source-field generators (Gaussian, plane wave,
fiber mode, point source, top hat, annular, LED, Bessel, Hermite-
Gauss, Laguerre-Gauss).
"""

from .core import (
    Source,
    create_gaussian_beam,
    create_hermite_gauss,
    create_laguerre_gauss,
    hermite_physicist,
    laguerre_generalized,
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
    # v4.15.1 (P0-NEW-2): the 3 factories now return ensembles (or
    # a PartialCoherenceMCF) and actually deliver partial coherence.
    create_gaussian_schell_source,
    create_schell_model_source,
    create_annular_incoherent_source,
    PartialCoherenceMCF,
)


__all__ = [
    'Source',
    'create_gaussian_beam',
    'create_hermite_gauss',
    'create_laguerre_gauss',
    'hermite_physicist',
    'laguerre_generalized',
    'create_tilted_plane_wave',
    'create_point_source',
    'create_multi_field_sources',
    'create_top_hat_beam',
    'create_annular_beam',
    'create_fiber_mode',
    'create_led_source',
    'create_bessel_beam',
    'create_gaussian_schell_source',
    'create_schell_model_source',
    'create_annular_incoherent_source',
    'PartialCoherenceMCF',
]

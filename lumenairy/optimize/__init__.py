"""
lumenairy.optimize -- prescription optimization, multi-config
parameterization, design merits.

Submodules:

* :mod:`lumenairy.optimize.core` -- ``DesignParameterization``,
  the merit-term hierarchy (focal length, BFL, Seidel, Strehl,
  RMS wavefront, spot size, chromatic shift, target-OPD, etc.),
  and ``design_optimize``.
* :mod:`lumenairy.optimize.multiconfig` -- multi-configuration
  parameterization, zoom configs, afocal angular magnification,
  beam-expander / Keplerian-telescope helpers.
"""

from .core import (
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
    design_optimize,
    WAVE_PROPAGATOR_REGISTRY,
    register_wave_propagator,
    unregister_wave_propagator,
)
from .multiconfig import (
    Configuration,
    multi_config_merit,
    create_zoom_configs,
    afocal_angular_magnification,
    beam_expander_prescription,
    keplerian_telescope,
)


__all__ = [
    # core
    'DesignParameterization',
    'MultiPrescriptionParameterization',
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
    'EvaluationContext',
    'DesignResult',
    'design_optimize',
    'WAVE_PROPAGATOR_REGISTRY',
    'register_wave_propagator',
    'unregister_wave_propagator',
    # multiconfig
    'Configuration',
    'multi_config_merit',
    'create_zoom_configs',
    'afocal_angular_magnification',
    'beam_expander_prescription',
    'keplerian_telescope',
]

"""
lumenairy.optimize -- prescription optimization, multi-config
parameterization, design merits.

Submodules:

* :mod:`lumenairy.optimize.core` -- ``DesignParameterization``,
  the merit-term hierarchy (focal length, BFL, Seidel, Strehl,
  RMS wavefront, spot size, chromatic shift, target-OPD, etc.),
  ``design_optimize``, and the v4.16 :class:`Constraint`
  hard-constraint wrapper.
* :mod:`lumenairy.optimize.multiconfig` -- multi-configuration
  parameterization, zoom configs, afocal angular magnification,
  beam-expander / Keplerian-telescope helpers.
* :mod:`lumenairy.optimize.multi_objective` -- v4.16 multi-
  objective Pareto optimisation via pymoo NSGA-II (optional
  dependency; install with ``pip install lumenairy[multi_objective]``
  or ``pip install pymoo>=0.6``).
"""

from .core import (
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
    RawParameterization,
    RMSWavefrontMerit,
    SphericalSeidelMerit,
    SpotSizeMerit,
    StrehlMerit,
    ToleranceAwareMerit,
    ZernikeCoefficientMerit,
    design_optimize,
    make_lg_aberration_merit_jax,
    optimize_traced_geometry,
    register_wave_propagator,
    unregister_wave_propagator,
)

# v4.16 (ROADMAP #11): multi-objective Pareto via pymoo NSGA-II.
# The module imports unconditionally (so users can ``from
# lumenairy.optimize.multi_objective import design_optimize_multi_objective``
# regardless of whether pymoo is installed); ``design_optimize_multi_objective``
# raises ``ImportError`` only when actually called without pymoo.
from .multi_objective import (
    # v4.16.0 (Agent A __all__-symmetry walker): pymoo availability
    # flag re-exported so callers can probe via ``la.PYMOO_AVAILABLE``
    # without an explicit ``lumenairy.optimize.multi_objective`` import.
    # Sibling to CUPY_AVAILABLE / JAX_AVAILABLE.
    PYMOO_AVAILABLE,
    ParetoResult,
    design_optimize_multi_objective,
)
from .multiconfig import (
    Configuration,
    afocal_angular_magnification,
    beam_expander_prescription,
    create_zoom_configs,
    keplerian_telescope,
    multi_config_merit,
)

__all__ = [
    # core
    'DesignParameterization',
    'MultiPrescriptionParameterization',
    'RawParameterization',
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
    'optimize_traced_geometry',
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
    'EvaluationContext',
    'DesignResult',
    'Constraint',
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
    # multi-objective Pareto (v4.16, pymoo-optional)
    'ParetoResult',
    'design_optimize_multi_objective',
    'PYMOO_AVAILABLE',
]

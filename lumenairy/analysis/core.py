"""
Beam measurement and diagnostic functions for optical propagation analysis.

v5.1.0 split: this file is now a thin back-compat re-export shell.  The
actual implementation moved to the topical submodules:

* :mod:`lumenairy.analysis.beam_stats`    -- beam centroid / D4sigma /
  power / diameter / M2 / radial power bands
* :mod:`lumenairy.analysis.strehl`        -- Strehl ratio /
  Marechal / phase-integral / vector / coupling efficiency
* :mod:`lumenairy.analysis.psf_mtf_otf`   -- PSF / OTF / MTF /
  encircled energy / Rayleigh / Sparrow / FWHM resolution
* :mod:`lumenairy.analysis.polychromatic` -- chromatic focal shift /
  polychromatic Strehl + PSF / ee_polychromatic
* :mod:`lumenairy.analysis.zernike`       -- Zernike index <-> nm /
  polynomial / basis matrix / decompose / reconstruct /
  astigmatism_mag_angle
* :mod:`lumenairy.analysis.opd`           -- OPD PV/RMS / wave_opd_1d /
  wave_opd_2d / remove_wavefront_modes / depth_of_focus /
  check_*_sampling

Every previously-public name imported from ``lumenairy.analysis.core``
(or from ``lumenairy.analysis``) continues to resolve here.

Backend awareness preserved across the split: the lightweight beam-
statistic and PSF/MTF helpers still dispatch through
:func:`lumenairy.backend.array_namespace`, so CuPy / JAX input fields
flow through without an implicit host transfer.

Author: Andrew Traverso
"""
from __future__ import annotations

# v5.1.0 (audit P2-NEW-V2 + Wave-4 integration fix-up): re-export from
# every topical submodule.  The submodules are the canonical homes for
# their respective implementations; ``analysis/core.py`` is the
# back-compat shim alone.
from .beam_stats import *  # noqa: F401,F403
from .strehl import *  # noqa: F401,F403
from .psf_mtf_otf import *  # noqa: F401,F403
from .polychromatic import *  # noqa: F401,F403
from .zernike import *  # noqa: F401,F403
from .opd import *  # noqa: F401,F403

from .beam_stats import __all__ as _beam_stats_all
from .strehl import __all__ as _strehl_all
from .psf_mtf_otf import __all__ as _psf_mtf_otf_all
from .polychromatic import __all__ as _polychromatic_all
from .zernike import __all__ as _zernike_all
from .opd import __all__ as _opd_all

# v5.1.0 (Wave-4 integration): re-export private cache + lock objects
# so legacy ``from lumenairy.analysis.core import _ZERNIKE_BASIS_CACHE``
# imports continue to resolve.  The ``import *`` form above does not
# pick up underscored names, but explicit imports do.
from .zernike import (  # noqa: F401
    _ZERNIKE_BASIS_CACHE,
    _ZERNIKE_BASIS_CACHE_LOCK,
    _ZERNIKE_BASIS_CACHE_MAXSIZE,
    _zernike_basis_matrix_build,
)

__all__ = (
    list(_beam_stats_all)
    + list(_strehl_all)
    + list(_psf_mtf_otf_all)
    + list(_polychromatic_all)
    + list(_zernike_all)
    + list(_opd_all)
)

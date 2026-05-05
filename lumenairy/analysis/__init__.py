"""
lumenairy.analysis -- analysis & post-processing tools.

Submodules:
* :mod:`lumenairy.analysis.analysis` -- core analysis
  (Strehl, MTF, PSF, Zernike, OPD).
* :mod:`lumenairy.analysis.detector` -- detector models.
* :mod:`lumenairy.analysis.ghost` -- ghost reflection analysis.
* :mod:`lumenairy.analysis.interferometry` -- simulated
  interferograms.
* :mod:`lumenairy.analysis.phase_retrieval` -- Gerchberg-Saxton /
  error-reduction phase retrieval.
* :mod:`lumenairy.analysis.coherence` -- partially-coherent
  propagation.
* :mod:`lumenairy.analysis.through_focus` -- through-focus scans.
* :mod:`lumenairy.analysis.plotting` -- field / PSF / MTF /
  Stokes plots.

This package's ``__init__`` mirrors all submodule namespaces so
existing user imports of the form ``from lumenairy.analysis import
X`` (or via the top-level shim ``from lumenairy.detector import
Y``) continue to work unchanged.
"""
from . import analysis as _analysis
from . import detector as _detector
from . import ghost as _ghost
from . import interferometry as _interferometry
from . import phase_retrieval as _phase_retrieval
from . import coherence as _coherence
from . import through_focus as _through_focus
from . import plotting as _plotting
for _m in (_analysis, _detector, _ghost, _interferometry,
            _phase_retrieval, _coherence, _through_focus, _plotting):
    globals().update({k: v for k, v in _m.__dict__.items()
                       if not k.startswith('__')})
del _m

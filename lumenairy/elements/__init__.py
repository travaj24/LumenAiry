"""
lumenairy.elements -- optical-element family.

Submodules:
* :mod:`lumenairy.elements.lenses` -- thin/spherical/aspheric/real
  lens phase application + ABCD helpers (the largest module).
* :mod:`lumenairy.elements.doe` -- diffractive optical elements
  (binary phase, Dammann gratings, Fresnel zone plates).
* :mod:`lumenairy.elements.coatings` -- thin-film coating models.
* :mod:`lumenairy.elements.freeform` -- XY polynomial / Q-type
  freeform surface sag.
* :mod:`lumenairy.elements.elements` -- catalog of canonical
  optical elements (axicon, GRIN lens, beam splitter, ...).
* :mod:`lumenairy.elements.rcwa` -- 1-D thin-grating RCWA.
* :mod:`lumenairy.elements.polarization` -- Jones-pupil
  polarization handling, Jones-field propagation.

This package's ``__init__`` mirrors all submodule namespaces so
existing user imports of the form ``from lumenairy.lenses import
X`` (via the top-level shim) or ``from lumenairy.elements import
X`` continue to work unchanged.
"""
from . import lenses as _lenses
from . import doe as _doe
from . import coatings as _coatings
from . import freeform as _freeform
from . import elements as _elements
from . import rcwa as _rcwa
from . import polarization as _polarization
for _m in (_lenses, _doe, _coatings, _freeform, _elements,
            _rcwa, _polarization):
    globals().update({k: v for k, v in _m.__dict__.items()
                       if not k.startswith('__')})
del _m

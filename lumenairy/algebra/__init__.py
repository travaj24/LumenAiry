"""lumenairy.algebra -- Nazarathy/Shamir-style optical operator algebra.

A Python surface for symbolic optical-system construction over
LumenAiry's array-first propagator infrastructure.  Build a system
algebraically::

    import lumenairy as la

    f = 100e-3
    sys = (
        la.FreeSpace(f) * la.ThinLens(f) * la.FreeSpace(2 * f)
        * la.ThinLens(f) * la.FreeSpace(f)
    )
    print(sys.abcd)       # [[-1, 0], [0, -1]]  -- 4f inverter
    print(sys.efl)        # +inf (afocal)

    src = la.Source.gaussian(N=512, dx=1e-6, wavelength=633e-9,
                               w0=100e-6)
    out = sys(src)        # apply the whole chain to a Source

Each operator carries a closed-form 2x2 ABCD (real, float64) and
delegates field-application to existing LumenAiry functions.
Composition multiplies ABCDs (matrix-on-the-left, matching
:func:`lumenairy.raytrace.system_abcd`) and chains the underlying
function calls right-to-left.  The implementation is a faithful
chain-and-delegate -- no symbolic reduction yet; symbolic
collapse of canonical patterns (Q-F-Q sandwiches, lens-lens
combination, etc.) is deferred to a Phase-2 follow-up.

Public symbols
--------------

- :class:`Operator` -- base class for composable optical operators.
- :class:`CompositeOperator` -- multi-stage chain produced by ``*``.
- :class:`FreeSpace` -- free-space propagation.
- :class:`ThinLens` -- paraxial thin lens.
- :class:`CylindricalLens` -- anamorphic thin lens.
- :class:`Magnify` -- geometric magnification (Nazarathy/Shamir
  ``V[a]``).
- :class:`FourierTransform` -- back-focal-plane optical FT.
- :class:`Aperture` -- hard circular / rectangular / annular
  amplitude aperture.
- :class:`GaussianAperture` -- soft Gaussian apodizer.

References
----------
- Nazarathy, M. & Shamir, J., "Fourier optics described by operator
  algebra," JOSA 70 (2), 150-159 (1980).

Author: Andrew Traverso
"""

from __future__ import annotations

from .apertures import Aperture, GaussianAperture
from .base import CompositeOperator, Operator
from .from_prescription import from_prescription as _from_prescription
from .primitives import (
    CylindricalLens,
    FourierTransform,
    FreeSpace,
    Magnify,
    ThinLens,
)


# Attach `Operator.from_prescription` as a classmethod-style factory.
# We use a setattr on the base class so the implementation can live
# in its own module without creating a circular import at the
# ``base.py`` definition site (the factory needs ``ThinLens`` and
# ``FreeSpace`` from ``primitives.py``, which already imports from
# ``base.py``).
def _operator_from_prescription_classmethod(
    cls, prescription, wavelength, *, method='auto',
):
    """Build a :class:`CompositeOperator` from a LumenAiry prescription.

    See :func:`lumenairy.algebra.from_prescription.from_prescription`
    for the full documentation.
    """
    return _from_prescription(prescription, wavelength, method=method)


_operator_from_prescription_classmethod.__doc__ = (
    _from_prescription.__doc__
)
Operator.from_prescription = classmethod(
    _operator_from_prescription_classmethod
)


__all__ = [
    'Operator',
    'CompositeOperator',
    'FreeSpace',
    'ThinLens',
    'CylindricalLens',
    'Magnify',
    'FourierTransform',
    'Aperture',
    'GaussianAperture',
]

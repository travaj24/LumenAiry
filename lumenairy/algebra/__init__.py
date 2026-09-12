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


# v5.46 (audit Z4): make ``lumenairy.algebra.from_prescription`` resolve to
# the FUNCTION, not to the submodule of the same name.
#
# Importing ``.from_prescription`` above made the import system bind the
# SUBMODULE as an attribute of this package, so the call every reader writes
# from the submodule's own ``__all__`` and docstring --
#
#     from lumenairy.algebra import from_prescription
#     from_prescription(rx, 633e-9)
#
# -- raised ``TypeError: 'module' object is not callable``, and the only
# working spellings were ``Operator.from_prescription(rx, wl)`` and the
# fully-qualified ``lumenairy.algebra.from_prescription.from_prescription``.
# Rebinding the attribute after the import fixes the documented spelling.
# The submodule stays in ``sys.modules`` under its own dotted name, so
# ``from lumenairy.algebra.from_prescription import from_prescription`` and
# ``importlib.import_module('lumenairy.algebra.from_prescription')`` keep
# working; what changes is that ATTRIBUTE access
# (``lumenairy.algebra.from_prescription``) now yields the callable, so the
# chained ``...from_prescription.from_prescription`` workaround no longer
# resolves -- drop the duplicated tail.
#
# Deliberately NOT added to this package's ``__all__``: the top-level
# ``lumenairy.__all__`` does not re-export it (``Operator.from_prescription``
# is the canonical entry point, per the documented exemption in
# ``tests/unit/test_v4_16_0_walker_all_symmetry.py``), and the symmetry
# walker requires every submodule ``__all__`` entry to be re-exported or
# exempted.
from_prescription = _from_prescription


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

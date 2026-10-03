# ruff: noqa: I001 -- extracted verbatim from the docs; import order kept
"""V10 -- the Phase D CHANGELOG example EXTRACTED VERBATIM, executed as written."""

import numpy as np
from lumenairy.elements.pmm import Circle, Rect, pmm_jones_2d_staggered
from lumenairy.elements.rcwa._core import uniaxial_tensor

lc = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=np.pi / 6)   # director in the plane, 30 deg
orders, R, T, J = pmm_jones_2d_staggered(
    1.2e-6, 1.2e-6, None, 1.45, 1.0, 0.5e-6, 1.0e-6, n_modes=8,
    shapes=[Rect(0.6e-6, 0.6e-6, 1.2e-6, 1.2e-6, eps=4.0),     # a slab ...
            Circle(0.6e-6, 0.6e-6, 0.36e-6, eps=lc)],          # ... with an LC-filled hole
    background_eps=1.0)

import numpy as _np
print("changelog OK", _np.asarray(R).shape, abs(_np.asarray(J)[0, 1]), float(abs(_np.asarray(R).sum(1) + _np.asarray(T).sum(1) - 1).max()))

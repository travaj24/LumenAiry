# ruff: noqa: I001 -- extracted verbatim from the docs; import order kept
"""V10 -- the Phase D cookbook example EXTRACTED VERBATIM from docs/cookbook.md
(section 'A liquid-crystal-filled circular hole'), executed as written,
plus a print of what it computes."""

import numpy as np
from lumenairy.elements.pmm import Circle, Rect, PMM2DStackPure
from lumenairy.elements.rcwa._core import uniaxial_tensor

p = 1.2e-6
lc = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=np.deg2rad(30.0))  # director at 30 deg

st = PMM2DStackPure(p, n_substrate=1.45, n_modes=7)
# a silicon-nitride slab (n = 2) with a 360 nm-radius hole filled with LC
st.add_layer(0.5e-6, shapes=[Rect(0.6e-6, 0.6e-6, p, p, eps=4.0),
                             Circle(0.6e-6, 0.6e-6, 0.36e-6, eps=lc)],
             background_eps=1.0)
# a magnetic cap layer: the same circle with mu_t = 1.5 (shapes share one map)
st.add_layer(0.1e-6, shapes=[Circle(0.6e-6, 0.6e-6, 0.36e-6, eps=2.25,
                                    mu=np.diag([1.5, 1.5, 1.0]))],
             background_eps=1.0)
st.set_source(1.0e-6)
orders, R, T, J = st.solve()
# J[0, 1] != 0: the rotated in-plane LC director converts x into y polarization

print("cookbook OK", R.shape, T.shape, abs(J[0, 1]), float(abs(R.sum(1) + T.sum(1) - 1).max()))

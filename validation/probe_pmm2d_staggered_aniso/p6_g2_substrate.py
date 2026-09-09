"""Probe 6 -- last discrete G2 reading: is the paper's ``1 - i5`` an INDEX
(so eps_sub = (1-5i)^2) rather than a permittivity?"""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)

LAM = 1.0e-6
EPS_P = np.array([[2.25, -0.5j, 0.0], [0.5j, 2.25, 0.0],
                  [0.0, 0.0, 2.0]], dtype=complex)
EPS_B, EPS_A = np.conj(EPS_P), EPS_P
KEYS = [(1, 1), (-1, 1), (0, -1), (0, 0)]
TAB2 = {(1, 1): 0.0268, (-1, 1): 0.0139, (0, -1): 0.0620, (0, 0): 0.2979}

cands = {"eps_sub = 1+5i (permittivity)": np.sqrt(1.0 + 5.0j),
         "n_sub  = 1+5i (index)": 1.0 + 5.0j,
         "eps_sub = 1 (vacuum)": 1.0,
         "eps_sub = 2.25": 1.5}
for dxdy in ((2.4, 1.4), (1.4, 2.4)):
    for name, nsub in cands.items():
        c = np.empty((2, 2, 3, 3), dtype=complex)
        c[:] = EPS_A
        c[0, 0] = EPS_B
        o, R, T, _ = pmm_jones_2d_staggered(dxdy[0] * LAM, dxdy[1] * LAM, c,
                                            nsub, 1.0, LAM, LAM, degree=7,
                                            n_orders=3)
        idx = {tuple(int(v) for v in r): i for i, r in enumerate(o)}
        v = {k: float(T[0][idx[k]]) for k in KEYS}
        dev = max(abs(v[k] - TAB2[k]) for k in KEYS)
        print(f"d={dxdy}  {name:30s} dev={dev:8.2e}  "
              + "  ".join(f"{str(k)}={v[k]:.5f}" for k in KEYS))

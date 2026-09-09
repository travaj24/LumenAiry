"""Probe 17 -- G7 y-mirror residual at the test's degree (6)."""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

import tests.unit.test_pmm2d_staggered_anisotropic as t  # noqa: E402

C = t._cell(t._LC, t._ISO)
Mm = np.diag([1.0, -1.0, 1.0])
D = np.empty_like(C)
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        D[i, j] = Mm @ C[i, C.shape[1] - 1 - j] @ Mm
oC, RC, _TC, JC = t.pmm_jones_2d_staggered(t._P, t._P, C, **t._G7)
oD, RD, _TD, JD = t.pmm_jones_2d_staggered(t._P, t._P, D, **t._G7)
ic, idd = t._idx(oC), t._idx(oD)
dev = max(abs(RD[r][idd[(m, n)]] - RC[r][ic[(m, -n)]])
          for (m, n) in idd if (m, -n) in ic for r in (0, 1))
print(f"y-mirror: per-order dev = {dev:.3e}   Jones dev = "
      f"{np.max(np.abs(JD - t._M2 @ JC @ t._M2)):.3e}")

"""V12 -- the UNMAPPED y-uniform stripe ladder (the reference of the
y-momentum test, v7 'momentum'): is the reference itself converged?
  python v12_stripe_ref.py <M>
Output: v12_stripe_M<M>.json (the (m, 0) orders, both inputs)."""
import sys

import _vcommon as C
import numpy as np

M = int(sys.argv[1])
th, ph = np.deg2rad(25.0), np.deg2rad(40.0)
vw = np.array([0.0, 0.2, 0.45, 0.7, 1.0]) * C.P
eps = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [4, 4, 4, 4], [1, 1, 1, 1]],
               complex)
o, R, T = C.solve_walls(np.array([0.22, 0.5, 0.9]), vw[1:-1], eps, M, th, ph)
sel = [i for i in range(len(o)) if o[i, 1] == 0 and abs(o[i, 0]) <= 3]
C.dump(f"v12_stripe_M{M}.json", {"M": M, "orders": o[sel], "R": R[:, sel],
                                 "T": T[:, sel]})
print(M, "done")

"""V12 -- the mapped far projector's per-cell PHASE rule on a long cell:
identity-map projector vs the shipped separable one, walls [0, 0.04, 1],
orders to +-10, oblique Bloch phases (0.9, 0.3) k0; M = 4, 5.  Run on the
tip (1.5e-15) and on mutant m09 (fixed 2M + 16 nodes: 2.96e-5).  Output:
printed dict (recorded in the verify doc and the decision test)."""
import _vcommon as C
import numpy as np

from lumenairy.elements.pmm import twod_staggered as TS
from lumenairy.elements.pmm._curvemap import IdentityMap

w = np.array([0.0, 0.04, C.P])
res = {}
for M in (4, 5):
    bx = TS.Basis1D(C.P, w, M, np.exp(-1j * 0.9 * C.K0 * C.P))
    by = TS.Basis1D(C.P, w, M, np.exp(-1j * 0.3 * C.K0 * C.P))
    ox = np.arange(-10, 11)
    Pa = TS._far_projector_2d(bx, by, ox, ox, 0.9 * C.K0, 0.3 * C.K0)
    Pb = TS._far_projector_2d(bx, by, ox, ox, 0.9 * C.K0, 0.3 * C.K0, cmap=IdentityMap(w, w))
    res[M] = max(float(np.abs(Pb[i] - Pa[i]).max() / np.abs(Pa[i]).max()) for i in (0, 1))
print(res)

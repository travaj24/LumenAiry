"""V3b -- the SHIPPED (unmapped) solver's least-squares incident overlap at
OBLIQUE incidence: is it under-determined there too (the F-B4 floor and the
n_orders dependence the Phase C fix removed under a map)?  Integer-grid
pillar, pmm_jones_2d_staggered, PRE and POST trees (must agree: unchanged).
"""
import warnings

import numpy as np
from _vc import BUILD, TREE, dump

from lumenairy.elements.pmm import pmm_jones_2d_staggered

warnings.simplefilter("ignore")
P, WL = 1.2, 1.0
eps3 = np.ones((3, 3), complex)
eps3[1, 1] = 4.0
OUT = {}


def run(M, no, th, ph, pert=0.0):
    o, R, T, J = pmm_jones_2d_staggered(P, P, eps3 * (1 + pert), 1.45, 1.0,
                                        0.5, WL, n_modes=M, n_orders=no,
                                        theta=th, phi=ph)
    o = np.asarray(o)
    sel = [int(np.nonzero((o[:, 0] == m) & (o[:, 1] == n))[0][0])
           for m in (-1, 0, 1) for n in (-1, 0, 1)]
    return np.concatenate([np.asarray(R)[:, sel].ravel(),
                           np.asarray(T)[:, sel].ravel()])


for th, ph in ((0.0, 0.0), (0.3, 0.0), (0.3, 0.6)):
    for M in (4, 5, 6):
        a = run(M, 3, th, ph)
        b = run(M, 3, th, ph, 1e-15)
        c = [run(M, no, th, ph) for no in (2, 5)]
        OUT[f"th{th}_ph{ph}_M{M}"] = dict(
            perturb_1e15=float(np.max(np.abs(a - b))),
            n_orders_2_5_vs_3=float(max(np.max(np.abs(x - a)) for x in c)))
        print(th, ph, M, OUT[f"th{th}_ph{ph}_M{M}"], flush=True)
dump(f"v3b_unmapped_floor_{TREE}_{BUILD}.json", OUT)

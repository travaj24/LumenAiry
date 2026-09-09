"""M5 -- MIXED KINDS through the GENERALIZED mortar twin: an OUT-OF-PLANE
tensor layer (first-order 4 q^2 generator, DISTINCT forward/backward modes) on
one grid over a scalar / in-plane layer on another.

(a) INDEPENDENT ORACLE: a uniform out-of-plane LC slab SPLIT into two halves on
    DIFFERENT grids (2,3) -- the interface is physically absent -- against
    ``berreman_jones_1d``.  Nothing about this answer may depend on the split.
(b) NESTED patterned pair (OOP on N=2 over scalar on N=4) vs the union-grid
    generalized cascade on N=4.
(c) NON-CONFORMING patterned pair (OOP on N=2 over scalar on N=3) vs the
    union-grid generalized cascade on the common refinement N=6."""
import json
import sys
import time
import warnings
import numpy as np
from mortar2d import guard, MortarStack2D, refine_cell
print("lumenairy:", guard(), flush=True)
from lumenairy import berreman_jones_1d
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.rcwa._core import uniaxial_tensor

warnings.simplefilter("ignore")
WL = 1.0e-6
PU = 0.9e-6
P = 1.2e-6
NSUP, NSUB = 1.0, 1.5
OOP = np.asarray(uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0),
                                 phi=np.deg2rad(25.0)), dtype=complex)
ISO = np.eye(3, dtype=complex)
rows = []

# ---------------------------------------------------------------- (a) oracle
print("(a) uniform OOP slab split across grids vs berreman_jones_1d")
DEPU = 0.35e-6
for (th, ph) in ((0.0, 0.0), (np.deg2rad(25.0), np.deg2rad(40.0))):
    Rb, Tb, Jrb, _ = berreman_jones_1d([(OOP, DEPU)], NSUB, NSUP, WL,
                                       theta=th, phi=ph)
    for M in (5, 7):
        for (Na, Nb) in ((2, 2), (2, 3), (3, 4)):
            s = MortarStack2D(PU, PU, n_superstrate=NSUP, n_substrate=NSUB,
                              n_modes=M, n_orders=2)
            s.add_layer(DEPU / 2, eps=OOP, grid=Na)
            s.add_layer(DEPU / 2, eps=OOP, grid=Nb)
            s.set_source(WL, theta=th, phi=ph)
            o, R, T, J = s.solve()
            dJ = float(np.max(np.abs(J - Jrb))) / float(np.max(np.abs(Jrb)))
            clo = max(abs(R[p].sum() + T[p].sum() - 1.0) for p in (0, 1))
            rows.append(dict(part="a", theta=float(th), M=M, Na=Na, Nb=Nb,
                             dJ=dJ, closure=float(clo)))
            print(f"  th={np.rad2deg(th):4.1f}deg M={M} grids({Na},{Nb}) "
                  f"dJones {dJ:9.2e}  |R+T-1| {clo:8.1e}", flush=True)


def cell(t33, n, pillar=None):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:, :] = t33
    if pillar is not None:
        c[0, 0] = pillar
    return c


def scal(n, hi, lo, k):
    c = np.full((n, n), lo + 0j)
    c[:k, :k] = hi
    return c


DEP = 0.4e-6
TH, PH = np.deg2rad(15.0), np.deg2rad(30.0)


def score(J0, J1, R1, T1):
    return (float(np.max(np.abs(J0 - J1))) / float(np.max(np.abs(J0))),
            max(abs(R1[p].sum() + T1[p].sum() - 1.0) for p in (0, 1)))


for tag, Na, Nb, fac_a, fac_b in (("b nested (2,4) -> union 4", 2, 4, 2, 1),
                                  ("c non-conf (2,3) -> union 6", 2, 3, 3, 2)):
    print(f"\n({tag})")
    A = cell(ISO * 2.25, Na, pillar=OOP)             # OOP pillar
    B = scal(Nb, 9.0, 2.0, Nb // 2 if Nb > 2 else 1)  # scalar patterned
    for M in [int(x) for x in (sys.argv[1:] or [4, 5])]:
        t0 = time.perf_counter()
        ref = PMM2DStackPure(P, P, n_superstrate=NSUP, n_substrate=NSUB,
                             n_modes=M, n_orders=2)
        ref.add_layer(DEP, eps_cell=refine_cell(A, fac_a))
        ref.add_layer(0.3e-6, eps_cell=refine_cell(B, fac_b))
        ref.set_source(WL, theta=TH, phi=PH)
        o0, R0, T0, J0 = ref.solve()
        t_ref = time.perf_counter() - t0
        t0 = time.perf_counter()
        mor = MortarStack2D(P, P, n_superstrate=NSUP, n_substrate=NSUB,
                            n_modes=M, n_orders=2)
        mor.add_layer(DEP, eps_cell=A)
        mor.add_layer(0.3e-6, eps_cell=B)
        mor.set_source(WL, theta=TH, phi=PH)
        o1, R1, T1, J1 = mor.solve()
        t_mor = time.perf_counter() - t0
        assert np.array_equal(o0, o1)
        dJ, clo = score(J0, J1, R1, T1)
        clo0 = max(abs(R0[p].sum() + T0[p].sum() - 1.0) for p in (0, 1))
        rows.append(dict(part=tag, M=M, dJ=dJ, closure_mortar=float(clo),
                         closure_ref=float(clo0), t_ref=t_ref, t_mortar=t_mor))
        print(f"  M={M}  dJones vs union {dJ:9.2e} | closure ref {clo0:8.1e} "
              f"mortar {clo:8.1e} | t_ref {t_ref:6.1f}s t_mortar {t_mor:6.1f}s "
              f"({t_ref/t_mor:.2f}x)", flush=True)

json.dump(rows, open("validation/probe_pmm2d_staggered_mortar/m5_oop_mixed.json",
                     "w"), indent=1)

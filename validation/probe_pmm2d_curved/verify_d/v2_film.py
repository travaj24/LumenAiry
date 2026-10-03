"""V2 -- a UNIFORM block-form film (eps tensor, optional mu tensor) under the
verifier's own maps against the verifier's own eps+mu Berreman (v0: equal to
the shipped berreman_jones_1d to 6e-15 incl. the complex reflection Jones).

usage: python v2_film.py MAP EPS MU ANGLE M [MUTATION]
  MAP    h2 | sh4 | c3 | c5 | none
  EPS    lc0 lc30 lc45 lc90 biax gyro rasym lossy | a float
  MU     none | gyro | sym | lossy | a float
  ANGLE  n (0, 0) | o (0.4, 0) | c (0.35, 0.7) | c2 (0.5, -1.2)
  MUTATION  one of _vdcommon.VMUT (optional)

Output v2_<MAP>_<EPS>_<MU>_<ANGLE>_M<M>[_<MUT>].json: the largest |R, T|
error (orders summed per input), the largest complex reflection-Jones error,
the absorption closure (layer_absorption vs the oracle's 1 - R - T), the
Jones itself, wall time."""
import sys
import time
import warnings

import numpy as np
from _vdcommon import (
    G3,
    GYRO_V,
    LOSSY,
    MU_GYRO,
    MU_LOSSY,
    MU_SYM,
    RASYM,
    berreman_eps_mu,
    biaxial,
    dump,
    film_stack,
    lc,
    make_map,
    vmutate,
)

MAPN, EPSN, MUN, ANG, M = sys.argv[1:6]
M = int(M)
MUT = sys.argv[6] if len(sys.argv) > 6 else None
EPS = {"lc0": lc(0), "lc30": lc(30), "lc45": lc(45), "lc90": lc(90),
       "biax": biaxial(), "gyro": GYRO_V, "rasym": RASYM, "lossy": LOSSY}
MUS = {"none": None, "gyro": MU_GYRO, "sym": MU_SYM, "lossy": MU_LOSSY}
ANGS = {"n": (0.0, 0.0), "o": (0.4, 0.0), "c": (0.35, 0.7),
        "c2": (0.5, -1.2)}
eps = EPS[EPSN] if EPSN in EPS else complex(EPSN) * np.eye(3)
mu = MUS[MUN] if MUN in MUS else complex(MUN) * np.eye(3)
th, ph = ANGS[ANG]
cmap = make_map(MAPN, G3["P"])

warnings.simplefilter("ignore")
t0 = time.perf_counter()
if MUT:
    with vmutate(MUT):
        st, o, R, T, J = film_stack(eps, cmap, M, th, ph, mu=mu)
        A = np.asarray(st.layer_absorption())
else:
    st, o, R, T, J = film_stack(eps, cmap, M, th, ph, mu=mu)
    A = np.asarray(st.layer_absorption())
wall = time.perf_counter() - t0
Rb, Tb, rb = berreman_eps_mu([(eps, 1.0 if mu is None else mu, G3["DEP"])],
                             G3["NSUB"], G3["NSUP"], G3["WL"], th, ph)
Rs, Ts = R.sum(1), T.sum(1)
dRT = max(float(np.max(np.abs(Rs - Rb))), float(np.max(np.abs(Ts - Tb))))
dJ = float(np.max(np.abs(J - rb)))
Ab = 1 - Rb - Tb
Asum = A.reshape(-1, 2).sum(0) if A.size else np.zeros(2)
dA = float(np.max(np.abs(np.asarray(Asum).ravel()[:2] - Ab)))
tag = f"v2_{MAPN}_{EPSN}_{MUN}_{ANG}_M{M}" + (f"_{MUT}" if MUT else "")
dump(tag + ".json", {"map": MAPN, "eps": EPSN, "mu": MUN, "angle": [th, ph],
                     "M": M, "mutation": MUT, "dRT": dRT, "dJ": dJ,
                     "dA_layer_absorption": dA, "A_oracle": Ab,
                     "A_layer": np.asarray(A), "J": J, "J_oracle": rb,
                     "R": Rs, "T": Ts, "R_oracle": Rb, "T_oracle": Tb,
                     "wall_s": wall})
print(tag, f"dRT {dRT:.2e} dJ {dJ:.2e} dA {dA:.2e} wall {wall:.1f}s")

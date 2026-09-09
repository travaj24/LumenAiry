"""V1 -- SCALAR PATH UNTOUCHED, across CODE VERSIONS.

Independent adversarial verification of gate G1's "the scalar dispatch is
byte-for-byte the shipped isotropic assembly".  The build doc's T1 compares
the scalar arm against the tensor ``e*I`` arm INSIDE the new build; that shows
the dispatch is internally consistent but it cannot show the shipped scalar
answer did not move.  This probe compares the SAME fixtures across two CODE
VERSIONS -- the pre-build main clone and this worktree -- on one machine, so a
disagreement is a code change, not a build difference.

Usage (each arm separately; the second argument is the expected library root):

    PYTHONPATH=<root> python v1_scalar_identity.py <root> <out.json>
"""
import hashlib
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

ROOT = os.path.abspath(sys.argv[1])
assert os.path.abspath(lumenairy.__file__).startswith(ROOT), (
    f"lumenairy.__file__ = {lumenairy.__file__} is not under {ROOT}")

from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _region_modes,
    pmm_efficiency_2d_staggered,
)


def h(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


# --------------------------------------------------------------- fixtures
P = 0.70e-6
WL = 0.55e-6
DEP = 0.28e-6

F_PILLAR = np.array([[6.25, 2.25], [2.25, 2.25]], dtype=complex)      # (2,2)
F_TWO = np.array([[6.25, 2.25, 2.25],                                 # (3,3)
                  [2.25, 2.25, 4.00],
                  [2.25, 4.00, 2.25]], dtype=complex)
F_LOSSY = np.array([[6.25 + 0.8j, 2.25], [2.25, 2.25]], dtype=complex)

FIXTURES = [
    # name, cell, kwargs to pmm_efficiency_2d_staggered
    ("f1_pillar_22_te_normal", F_PILLAR,
     dict(degree=6, n_orders=3, polarization="te")),
    ("f1_pillar_22_tm_normal", F_PILLAR,
     dict(degree=6, n_orders=3, polarization="tm")),
    ("f2_two_pillar_33_te", F_TWO,
     dict(degree=5, n_orders=3, polarization="te")),
    ("f3_pillar_22_conical", F_PILLAR,
     dict(degree=6, n_orders=3, polarization="te", theta=0.31, phi=0.62)),
    ("f4_lossy_22_tm_oblique", F_LOSSY,
     dict(degree=6, n_orders=3, polarization="tm", theta=0.22, phi=0.0)),
]

out = {"root": ROOT, "lumenairy": lumenairy.__file__,
       "version": lumenairy.__version__,
       "python": sys.version.split()[0], "numpy": np.__version__,
       "fixtures": {}}

for name, cell, kw in FIXTURES:
    o, R, T = pmm_efficiency_2d_staggered(P, P, cell, 1.5, 1.0, DEP, WL, **kw)
    out["fixtures"][name] = {
        "orders": h(np.asarray(o)), "R": h(R), "T": h(T),
        "sumR": float(np.sum(R)), "sumT": float(np.sum(T)),
        "R0": float(R[int(np.where((np.asarray(o)[:, 0] == 0)
                                   & (np.asarray(o)[:, 1] == 0))[0][0])]),
    }

# ------------------------------------------------- assembled operators
OPS = [
    ("op_pillar_22_M6", F_PILLAR, dict(alpha0x=0.31, alpha0y=-0.17,
                                       k0=2 * np.pi / 0.62)),
    ("op_two_33_M5", F_TWO, dict(alpha0x=0.0, alpha0y=0.0,
                                 k0=2 * np.pi / 0.55)),
    ("op_lossy_22_M6", F_LOSSY, dict(alpha0x=0.11, alpha0y=0.29,
                                     k0=2 * np.pi / 0.62)),
]
for name, cell, kw in OPS:
    M = 5 if cell.shape[0] == 3 else 6
    s = Granet2DTransverseE(1.1, 1.1, cell.shape[0], cell.shape[1], M, cell,
                            **kw)
    rec = {k: h(getattr(s, k)) for k in ("Lmat", "Rmat", "Stt", "Schur")}
    rec["Et_blocks0"] = h(s.Et_blocks[0])
    rec["Et_blocks1"] = h(s.Et_blocks[1])
    W, V, lam, g2 = _region_modes(s)
    # eigen-decompositions are NOT bit-reproducible across code layouts in
    # general; record the SORTED |lam| spectrum hash AND its float values so a
    # mismatch can be judged rather than only flagged.
    rec["lam_sorted_hash"] = h(np.sort_complex(np.asarray(lam)))
    rec["lam_absmax"] = float(np.max(np.abs(lam)))
    out["fixtures"][name] = rec

# ------------------------------------------------- pure stack (scalar)
st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.5, n_modes=6,
                    n_orders=3)
st.add_layer(DEP, eps_cell=F_PILLAR)
st.add_layer(0.11e-6, eps=2.25)
st.set_source(WL, theta=0.15, phi=0.4)
o, R, T, J = st.solve(jones=True)
out["fixtures"]["f5_stack_scalar_jones"] = {
    "R": h(R), "T": h(T), "jones": h(J),
    "sumRT": [float(R[r].sum() + T[r].sum()) for r in (0, 1)],
}

json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out, indent=1))

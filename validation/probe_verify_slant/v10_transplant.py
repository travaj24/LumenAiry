"""V10 -- is the SHIPPED assembly BIT-IDENTICAL to the PROTOTYPE's?

The build's gate B1b claims 0 differing bytes over 18 cell x slant x mount
combinations between `Granet2DTransverseE(..., slant=s)` and the prototype
`validation/probe_pmm2d_staggered_slant/slant_lib.SlantSolver(..., slant=-s)`.
That is the check that the transplant did not silently substitute the
element-wise ``b.mixed(b.Btilde, b.B)`` for the distributionally exact
``-(dbt)^H`` in the six blocks, so it is worth re-measuring rather than reading.

Own fixtures: a scalar cell, an IN-PLANE tensor cell, an OUT-OF-PLANE tensor
cell and a NON-RECIPROCAL cell, at four slants (x, y, diagonal, a large one)
and three mounts.
"""
import pathlib
import sys

import numpy as np

from _lib import arm, dump, mx, sha  # noqa: I001

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]
                       / "probe_pmm2d_staggered_slant"))
import slant_lib  # noqa: E402

from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

K0 = 2.0 * np.pi
PX = PY = 0.95
NG, M = 2, 6
TIL = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
INP = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=np.deg2rad(25.0))
NREC = TIL.copy()
NREC[0, 2] += 0.30
NREC[2, 0] -= 0.30
AIR = np.eye(3, dtype=complex)


def tile(t):
    c = np.zeros((NG, NG, 3, 3), dtype=complex)
    c[:, :] = AIR
    c[0, 0] = t
    return c


CELLS = {"scalar": np.array([[4.0, 1.0], [1.0, 1.0]], dtype=complex),
         "inplane": tile(INP), "oop": tile(TIL), "nonrecip": tile(NREC)}
SLANTS = {"x0.30": (0.30, 0.0), "y0.45": (0.0, 0.45),
          "diag": (0.5, -0.25), "big": (1.30, 0.60)}
MOUNTS = {"normal": (0.0, 0.0), "oblique": (0.30, 0.0),
          "conical": (0.25, 0.18)}


def main():
    out = {"rows": {}}
    bad = 0
    for cname, cell in CELLS.items():
        for sname, sl in SLANTS.items():
            for mname, (ax, ay) in MOUNTS.items():
                ship = Granet2DTransverseE(PX, PY, NG, NG, M, cell,
                                           alpha0x=ax * K0, alpha0y=ay * K0,
                                           k0=K0, slant=sl)
                proto = slant_lib.SlantSolver(PX, PY, NG, NG, M, cell,
                                              slant=(-sl[0], -sl[1]),
                                              alpha0x=ax * K0,
                                              alpha0y=ay * K0, k0=K0)
                same_A = sha(ship.Agen) == sha(proto.Agen)
                same_B = sha(ship.Bgen) == sha(proto.Bgen)
                row = dict(A_identical=same_A, B_identical=same_B,
                           dA=mx(ship.Agen, proto.Agen),
                           dB=mx(ship.Bgen, proto.Bgen))
                out["rows"][f"{cname}/{sname}/{mname}"] = row
                if not (same_A and same_B):
                    bad += 1
                    print(f"  DIFFER {cname}/{sname}/{mname}: dA {row['dA']:.3e}"
                          f" dB {row['dB']:.3e}")
    out["n_rows"] = len(out["rows"])
    out["n_differing"] = bad
    print(f"{len(out['rows'])} rows, {bad} differing "
          f"(max dA over all rows "
          f"{max(v['dA'] for v in out['rows'].values()):.3e})")
    # and the WRONG sign relation, as the two-sided arm
    ship = Granet2DTransverseE(PX, PY, NG, NG, M, CELLS["oop"],
                               alpha0x=0.25 * K0, alpha0y=0.18 * K0, k0=K0,
                               slant=(0.5, -0.25))
    proto_wrong = slant_lib.SlantSolver(PX, PY, NG, NG, M, CELLS["oop"],
                                        slant=(0.5, -0.25), alpha0x=0.25 * K0,
                                        alpha0y=0.18 * K0, k0=K0)
    out["wrong_sign_relation_dA"] = mx(ship.Agen, proto_wrong.Agen)
    print(f"wrong sign relation (slant=+s into the prototype): dA "
          f"{out['wrong_sign_relation_dA']:.3e}")
    dump("v10_transplant", out)
    print("arm", arm())


if __name__ == "__main__":
    main()

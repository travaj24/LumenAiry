"""Probe 9 -- G5: genuinely 2-D anisotropic cell across three engines, plus
the no-floor (n_orders-independence) property two-sided."""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.pmm import pmm_jones_2d  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa import rcwa_jones_2d  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

WL = 0.55e-6
P = 0.70e-6
DEP = 0.28e-6
LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
ISO = 4.0 * np.eye(3, dtype=complex)


def cell(up=1, reverse=False):
    n = 2 * up
    c = np.empty((n, n, 3, 3), dtype=complex)
    host, pill = (LC, ISO) if not reverse else (ISO, LC)
    c[:] = host
    c[:up, :up] = pill
    return c


def order0(o, A):
    i = int(np.where((np.asarray(o)[:, 0] == 0)
                     & (np.asarray(o)[:, 1] == 0))[0][0])
    return float(A[0][i]), float(A[1][i])


def main():
    for reverse in (False, True):
        tag = "iso pillar in LC host" if not reverse else "LC pillar in iso host"
        print(f"\n=== {tag} ===")
        res = {}
        o, R, T, J = pmm_jones_2d_staggered(P, P, cell(1, reverse), 1.5, 1.0,
                                            DEP, WL, degree=7, n_orders=5)
        res["staggered M=7"] = (o, R, T, J)
        for deg, nor in ((9, 9), (11, 13)):
            oh, Rh, Th, Jh = pmm_jones_2d(P, P, cell(1, reverse), 1.5, 1.0,
                                          DEP, WL, degree=deg, n_orders=nor)
            res[f"hybrid d={deg} no={nor}"] = (oh, Rh, Th, Jh)
        for nor in (9, 13):
            orc, Rr, Tr, Jr = rcwa_jones_2d(P, P, cell(32, reverse), 1.5, 1.0,
                                            DEP, WL, n_orders_x=nor,
                                            n_orders_y=nor)
            res[f"rcwa no={nor}"] = (orc, Rr, Tr, Jr)
        base = res["staggered M=7"]
        for k, (oo, RR, TT, JJ) in res.items():
            r0 = order0(oo, RR)
            t0 = order0(oo, TT)
            dR = max(abs(a - b) for a, b in zip(r0, order0(base[0], base[1])))
            dT = max(abs(a - b) for a, b in zip(t0, order0(base[0], base[2])))
            dJ = float(np.max(np.abs(JJ - base[3])))
            print(f"{k:22s} R00={r0[0]:.6f},{r0[1]:.6f} "
                  f"T00={t0[0]:.6f},{t0[1]:.6f}  vs staggered: "
                  f"dR={dR:.2e} dT={dT:.2e} dJ={dJ:.2e}  "
                  f"closure={np.max(np.abs(RR.sum(axis=1)+TT.sum(axis=1)-1)):.2e}")

    print("\n=== no-floor: n_orders 4 -> 8 (same M / degree) ===")
    c = cell(1, False)
    vals = {}
    for nor in (4, 8):
        o, R, T, J = pmm_jones_2d_staggered(P, P, c, 1.5, 1.0, DEP, WL,
                                            degree=7, n_orders=nor)
        vals[("stag", nor)] = order0(o, R) + order0(o, T)
        oh, Rh, Th, Jh = pmm_jones_2d(P, P, c, 1.5, 1.0, DEP, WL, degree=9,
                                      n_orders=nor)
        vals[("hyb", nor)] = order0(oh, Rh) + order0(oh, Th)
    for eng in ("stag", "hyb"):
        d = max(abs(a - b) for a, b in zip(vals[(eng, 4)], vals[(eng, 8)]))
        print(f"{eng}: max |change(order-0 R,T)| over n_orders 4->8 = {d:.3e}")


if __name__ == "__main__":
    main()

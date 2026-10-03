"""A8 -- assembly cost of the mapped (quadrature) path against the shipped
Kronecker assembly on the same 3 x 3 grid (operators only, no eig), and the
adaptive node count it ran with.  Wall times on a shared box are upper
bounds; only the ratios measured back to back mean anything.

  python validation/probe_pmm2d_curved/build_a/a8_cost.py

Output: a8_cost.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import numpy as np  # noqa: E402

TS = C.TS
K0 = 2 * np.pi / C.WL


def best(fn, n=3):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    res = {"env": C.env_record(), "rows": []}
    eps = C.cell("stripe")
    for M in (6, 8, 10):
        arms = {"none": None,
                "identity": C.IdentityMap(C.XW, C.YW["stripe"]),
                "stretch_0.05": C.stretch_map(0.05),
                "stretch_0.15": C.stretch_map(0.15)}
        row = {"M": M}
        for name, cm in arms.items():
            if cm is None:
                t = best(lambda: TS.Granet2DTransverseE(
                    C.P, C.P, C.XW, C.YW["stripe"], M, eps, k0=K0))
                row[name] = {"t_assemble": t}
            else:
                bx = TS.Basis1D(C.P, cm.u_walls, M)
                by = TS.Basis1D(C.P, cm.v_walls, M)
                nq = TS._stag_map_nodes(bx, by, cm, M)
                t = best(lambda cm=cm: TS.Granet2DTransverseE(
                    C.P, C.P, cm.u_walls, cm.v_walls, M, eps, k0=K0,
                    cmap=cm))
                row[name] = {"t_assemble": t, "nq": nq}
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(C.HERE, "a8_cost.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

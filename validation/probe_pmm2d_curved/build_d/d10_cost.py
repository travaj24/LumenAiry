"""D10 -- cost of a TENSOR / MAGNETIC mapped cell against the SCALAR mapped
cell (and the unmapped tensor cell), operator construction only (node count,
weights, the 18 quadrature blocks, the Schur term): best of three per arm,
3 x 3 circle map, M = 6 / 8 / 10.  The region eig is the same pencil size
in every arm (2 q^2), so it is not re-timed; one QZ region eig per arm at
M = 6 and 8 is recorded for scale.  The box was loaded throughout: every
time is an UPPER BOUND, only same-run ratios mean anything.

usage: python d10_cost.py <M>     -> d10_cost_M<M>.json
"""
import sys
import time

import _dcommon as D
import numpy as np

P, K0 = 1.2, 2 * np.pi


def arms():
    cm = D.make_map("c3", P)
    sc = np.ones((3, 3), complex)
    sc[1, 1] = 4.0
    te = np.broadcast_to(np.eye(3, dtype=complex), (3, 3, 3, 3)).copy()
    te[1, 1] = D.LC30
    mu = np.broadcast_to(np.eye(3, dtype=complex), (3, 3, 3, 3)).copy()
    mu[1, 1] = np.diag([2.0, 2.0, 1.0])
    w = np.array([0.0, 0.24, 0.96, P])
    return {
        "scalar_mapped": dict(wx=cm.u_walls, wy=cm.v_walls, eps=sc,
                              cmap=cm),
        "tensor_mapped": dict(wx=cm.u_walls, wy=cm.v_walls, eps=te,
                              cmap=cm),
        "tensor_mu_mapped": dict(wx=cm.u_walls, wy=cm.v_walls, eps=te,
                                 mu=mu, cmap=cm),
        "tensor_unmapped": dict(wx=w, wy=w, eps=te),
    }


def build(a, M):
    return D.TS.Granet2DTransverseE(P, P, a["wx"], a["wy"], M, a["eps"],
                                    k0=K0, mu_cell=a.get("mu"),
                                    cmap=a.get("cmap"))


def run(M):
    out = {"M": M}
    for name, a in arms().items():
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            s = build(a, M)
            ts.append(time.perf_counter() - t0)
        row = {"assembly_best_s": min(ts), "assembly_all_s": ts,
               "pencil": int(s.dimtot)}
        if M <= 8:
            t0 = time.perf_counter()
            D.TS._region_modes(s)
            row["region_eig_s"] = time.perf_counter() - t0
        out[name] = row
        print(M, name, row, flush=True)
    base = out["scalar_mapped"]["assembly_best_s"]
    out["ratio_to_scalar_mapped"] = {
        k: out[k]["assembly_best_s"] / base for k in arms()}
    D.dump(f"d10_cost_M{M}.json", out)


if __name__ == "__main__":
    run(int(sys.argv[1]))

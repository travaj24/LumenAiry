"""A7 -- absorption under a map (NOT measured by the planning probes).

A LOSSY layer under the stretch; the claim is the cross-machinery closure
sum_i layer_absorption_i == 1 - sum R - sum T (internal Gram flux against
the Rayleigh far field).  Correct arm: the flux form is the PLAIN (u, v)
block Gram (metric-free: E'_u H'_v - E'_v H'_u carries det J, which cancels
the area element).  FAIL-BEFORE arm: -R = C[chi_t]C as the flux Gram (what
the unmapped stack retains), engineered by replacing the retained Gram after
the solve and re-running the real layer_absorption.

  python validation/probe_pmm2d_curved/build_a/a7_absorption.py

Output: a7_absorption.json -- per (fixture, a, M): the closure defect of
each arm, |sum A - (1 - sum R - sum T)| (max over both polarizations), and
the absorbed fraction itself.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import numpy as np  # noqa: E402

TS = C.TS
EPS_L = 4.0 + 0.5j


def defect(st, R, T):
    A = st.layer_absorption().sum(axis=0)
    return (float(np.max(np.abs(A - (1 - R.sum(axis=1) - T.sum(axis=1))))),
            A.tolist())


def minus_R_gram(st, M):
    """-R of the superstrate region under the stack's map (the flux form
    the unmapped stack retains)."""
    cm = st.cmap
    k0 = 2 * np.pi / C.WL
    s = TS.Granet2DTransverseE(C.P, C.P, cm.u_walls, cm.v_walls, M,
                               np.full(cm.shape, C.N_SUP ** 2 + 0j), k0=k0,
                               cmap=cm)
    return (-s.Rmat).copy()


def main():
    res = {"env": C.env_record(), "eps_layer": [EPS_L.real, EPS_L.imag],
           "rows": []}
    for fx in ("film", "stripe"):
        for a in (0.0, 0.05, 0.15):
            for M in (4, 5, 6, 7, 8):
                t0 = time.perf_counter()
                o, R, T, _J, st = C.solve(fx, a, M, retain=True, eps=EPS_L)
                d_ok, A = defect(st, R, T)
                row = {"fixture": fx, "a": a, "M": M, "defect_correct": d_ok,
                       "absorbed": A}
                if a != 0:
                    st._internal["G"] = minus_R_gram(st, M)
                    d_bad, A_bad = defect(st, R, T)
                    row["defect_minusR_gram"] = d_bad
                    row["absorbed_minusR_gram"] = A_bad
                row["t"] = time.perf_counter() - t0
                res["rows"].append(row)
                print(json.dumps(row), flush=True)
                with open(os.path.join(C.HERE, "a7_absorption.json"), "w") as f:
                    json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

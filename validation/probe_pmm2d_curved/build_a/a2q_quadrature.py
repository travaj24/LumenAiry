"""A2q -- IS THE MAPPED ASSEMBLY'S QUADRATURE ADEQUATE?  (a build finding)

The planning probe sized the per-cell Gauss rule at nq = 2M + 8 nodes per
axis and measured it adequate on the circle map (2.8e-8 against 4x).  Under
the STRETCH x = u + a sin(2 pi u / p) the weights carry 1 / f'(u), and at
a = 0.15 p, f' falls to 0.058: 1 / f' has complex poles close to the real
axis, so a fixed node count converges slowly.  This probe measures the R / T
change as nq grows (factors 1, 2, 4, 8 of the base rule) on the stripe,
at a = 0.05 p and 0.15 p, and reports the rung-to-rung change in M at the
same nq for comparison (the discretisation error the quadrature error must
stay below).

  python validation/probe_pmm2d_curved/build_a/a2q_quadrature.py <label> [fixed|adaptive]

'fixed' forces nq = factor * (2M + 8) for the operator assembly (the far
projector keeps its own rule); 'adaptive' runs the library's rule as
shipped.  Output: a2q_quadrature_<label>.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import numpy as np  # noqa: E402

TS = C.TS


def run(a, M, fac):
    orig = TS._stag_map_quad_rule
    if fac is not None:
        def rule(MM, nq=None, _f=fac):
            return orig(MM, (2 * int(MM) + 8) * _f)
        TS._stag_map_quad_rule = rule
    try:
        o, R, T, _J, st = C.solve("stripe", a, M)
    finally:
        TS._stag_map_quad_rule = orig
    return C.vec(o, R, T)


def main():
    label = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "fixed"
    res = {"env": C.env_record(), "mode": mode, "rows": []}
    for a in (0.05, 0.15):
        for M in (5, 7, 9):
            t0 = time.perf_counter()
            if mode == "fixed":
                vs = {f: run(a, M, f) for f in (1, 2, 4, 8)}
                row = {"a": a, "M": M,
                       "d_vs_x8": {str(f): float(np.max(np.abs(vs[f] - vs[8])))
                                   for f in (1, 2, 4)}}
            else:
                v = run(a, M, None)
                v8 = run(a, M, 8)
                row = {"a": a, "M": M,
                       "adaptive_vs_x8": float(np.max(np.abs(v - v8)))}
            row["t"] = time.perf_counter() - t0
            res["rows"].append(row)
            print(json.dumps(row), flush=True)
    with open(os.path.join(C.HERE, f"a2q_quadrature_{label}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

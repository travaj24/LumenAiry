"""A6 -- the H-partner Gram, two-sided.  Correct arm: every mapped region
recovers the Eq.-25 H partner through the PLAIN (u, v) block Gram (the
library as built).  MIXED arm (engineered through the real stack path,
_common.mixed_hgram): the half-spaces use -R (what the shipped unmapped
_homog_geom_cache did) while the patterned layer keeps the plain Gram.

  python validation/probe_pmm2d_curved/build_a/a6_hgram.py <M> [a_frac]

Output: a6_hgram_M<M>.json -- per fixture (stripe, pillar) the lossless
closure of each arm and the max |dR, dT| between the arms.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
import numpy as np  # noqa: E402


def closure(R, T):
    return float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1)))


def main():
    M = int(sys.argv[1])
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
    res = {"env": C.env_record(), "M": M, "a_over_p": a, "rows": []}
    import warnings
    warnings.simplefilter("ignore")
    for fx in ("stripe", "pillar"):
        t0 = time.perf_counter()
        o, R, T, _J, _st = C.solve(fx, a, M)
        with C.mixed_hgram():
            o2, R2, T2, _J2, _st2 = C.solve(fx, a, M)
        row = {"fixture": fx, "closure_correct": closure(R, T),
               "closure_mixed": closure(R2, T2),
               "mixed_vs_correct": float(np.max(np.abs(C.vec(o2, R2, T2)
                                                       - C.vec(o, R, T)))),
               "t": time.perf_counter() - t0}
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
    with open(os.path.join(C.HERE, f"a6_hgram_M{M}.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

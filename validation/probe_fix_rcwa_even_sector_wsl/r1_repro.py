"""R1 -- reproduce ``test_jones_2d_even_sector_matches_full`` OUTSIDE pytest,
five times in one process, on whichever tree ``PYTHONPATH`` selects.

Five identical repeats in one process decide DETERMINISTIC vs NONDETERMINISTIC
on the build: identical digits across repeats means the build always computes
this number, so "flaky" would be a per-BUILD fact, not a per-RUN one.
"""
from __future__ import annotations

import _lib as L
import numpy as np

from lumenairy.elements.rcwa import rcwa_jones_2d


def one(tc, symmetry):
    return rcwa_jones_2d(L.P_DEFAULT, L.P_DEFAULT, tc, 1.5, 1.0, 0.2e-6,
                         L.WL_DEFAULT, n_orders_x=5, n_orders_y=5,
                         symmetry=symmetry)


def main():
    a = L.arm()
    tc = L.even_sector_cell()
    reps = []
    for k in range(5):
        full = one(tc, False)
        even = one(tc, True)
        dR = float(np.max(np.abs(full[1] - even[1])))
        dJ = float(np.max(np.abs(full[3] - even[3])))
        reps.append(dict(rep=k, dR=dR, dJ=dJ,
                         R_full_max=float(np.max(np.abs(full[1]))),
                         R_even_max=float(np.max(np.abs(even[1]))),
                         R_full_sum=float(np.sum(full[1])),
                         R_even_sum=float(np.sum(even[1])),
                         even_nonfinite=int(np.sum(~np.isfinite(even[1]))),
                         full_nonfinite=int(np.sum(~np.isfinite(full[1]))),
                         passes=bool(dR < 1e-8 and dJ < 1e-8)))
        print("  rep %d  dR %.6e  dJones %.6e  -> %s"
              % (k, dR, dJ, "PASS" if reps[-1]["passes"] else "FAIL"))
    digits = {(r["dR"], r["dJ"]) for r in reps}
    det = len(digits) == 1
    print("%-4s %-16s deterministic=%s  dR=%.6e" %
          (a["build"], a["tree"], det, reps[0]["dR"]))
    L.dump("r1_repro", dict(reps=reps, deterministic=det,
                            bar=1e-8, verdict="PASS" if reps[0]["passes"]
                            else "FAIL"))


if __name__ == "__main__":
    main()

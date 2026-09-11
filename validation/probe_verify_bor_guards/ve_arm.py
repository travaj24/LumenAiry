"""TASK E -- CROSS-ARM answer spread.

The raw LAPACK spectrum's last bits move with the kernel on BOTH builds, so a
moved HASH is not by itself a defect.  What must not move is the ANSWER beyond
roundoff.  This reads the saved forward-``ky`` vectors and reports, per fixture
and per build, the worst |d ky| between any two of the 16 arms and the same
relative to the mode's own size.

A spread of ~1e-13 is the eigensolver.  A spread of ~|ky| is a mode that came
back on the OTHER ROOT on one arm and not another -- an answer decided by the
BLAS kernel.
"""
from __future__ import annotations

import collections
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "runs"


def main():
    data = collections.defaultdict(dict)          # build -> tag -> npz
    for p in sorted(RUNS.glob("ve_ky_*.npz")):
        tag = p.stem[len("ve_ky_"):]
        build = "pre" if tag.startswith("pre_") else "post"
        data[build][tag] = np.load(p)
    names = sorted(set(next(iter(data["post"].values())).files))
    print("%d fixtures x %d arms per build" % (len(names), len(data["post"])))
    print()
    print("%-46s | %-24s | %-24s" % ("fixture", "PRE  worst cross-arm |dky|",
                                     "POST worst cross-arm |dky|"))
    print("-" * 100)
    flag_pre = flag_post = 0
    worst = {}
    for nm in names:
        row = []
        for build in ("pre", "post"):
            arrs = [d[nm] for d in data[build].values()]
            n = min(a.size for a in arrs)
            A = np.stack([a[:n] for a in arrs])
            d = np.max(np.abs(A - A[0]), axis=0)
            scale = np.maximum(np.abs(A[0]), 1.0)
            w = float(np.max(d))
            wr = float(np.max(d / scale))
            nflip = int(np.sum(d > 1e-6 * scale))
            row.append((w, wr, nflip))
        worst[nm] = row
        mark = ""
        if row[0][2]:
            flag_pre += 1
        if row[1][2]:
            flag_post += 1
            mark = "   <== POST ARM-DEPENDENT"
        print("%-46s | %9.3e (%3d flip) | %9.3e (%3d flip)%s"
              % (nm[:46], row[0][0], row[0][2], row[1][0], row[1][2], mark))
    print()
    print("SUMMARY: %d/%d fixtures have an ARM-DECIDED root on PRE, %d/%d on "
          "POST" % (flag_pre, len(names), flag_post, len(names)))
    for build, i in (("PRE", 0), ("POST", 1)):
        w = max(v[i][0] for v in worst.values())
        print("  %-4s worst cross-arm |d ky| over every fixture: %.6e"
              % (build, w))


if __name__ == "__main__":
    main()

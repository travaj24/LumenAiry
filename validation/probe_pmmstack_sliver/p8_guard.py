"""P8 -- the guard, two-sided: refused inside, BIT-IDENTICAL outside.

Runs the same delta ladder with the guard armed and with
``PMM_SLIVER_GUARD = False`` (its fail-before switch) and reports, per row,
whether the guard refused and whether the returned numbers are bit-for-bit the
pre-fix ones wherever it did not.
"""
import json
import os
import warnings

import numpy as np

import lumenairy
from lumenairy.elements.pmm import stack as ps
from p1_repro import PX, frames, orc

print("lumenairy:", lumenairy.__file__, flush=True)
warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
MF_OFF = PX * 1e-10


def attempt(d, deg, mf):
    try:
        return ("ok", orc(frames(d), deg, min_feature=mf))
    except ValueError as exc:
        return ("REFUSED", str(exc))


if __name__ == "__main__":
    rows = []
    for tag, mf in (("default", None), ("snap-off", MF_OFF)):
        for deg in (12, 14):
            for d in (1e-2, 1e-3, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5, 3e-6, 0.0):
                ps.PMM_SLIVER_GUARD = False
                pre = attempt(d, deg, mf)
                ps.PMM_SLIVER_GUARD = True
                post = attempt(d, deg, mf)
                same = None
                if post[0] == "ok" and pre[0] == "ok":
                    same = all(np.array_equal(a, b)
                               for a, b in zip(pre[1], post[1]))
                tot = (float(pre[1][1].sum() + pre[1][2].sum())
                       if pre[0] == "ok" else float("nan"))
                rows.append(dict(tag=tag, degree=deg, delta=d,
                                 pre=pre[0], post=post[0], bit_identical=same,
                                 pre_total=tot))
                print(f"{tag:9s} deg{deg:3d} d={d:8.2e} pre={pre[0]:8s} "
                      f"post={post[0]:8s} bit-identical={same} "
                      f"pre R+T={tot:.6g}", flush=True)
            json.dump(rows, open(os.path.join(HERE, "p8_guard.json"), "w"),
                      indent=1)
    ps.PMM_SLIVER_GUARD = True
    print("\n--- one full refusal message ---", flush=True)
    print(attempt(1e-4, 14, None)[1], flush=True)

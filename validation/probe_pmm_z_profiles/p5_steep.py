"""P5 -- how steep a LINEAR taper can route B carry before the frozen slab's
pencil degrades (the bound P4 found on a rounded rim, where the tilt grows
without limit)?

A ridge whose duty goes 0.2 -> 0.8 over a height chosen so that the
physical tilt of each wall ``|dx_wall/dz|`` is 0.5, 1, 2, 3 and 4 (26.6,
45, 63.4, 71.6 and 76.0 degrees from the vertical).  Route B at K = 16, 32,
64, degree 12 and 16, TE and TM, normal incidence: the lossless closure, the
mirror identity T(-1) = T(+1), and the K ladder's ratio per doubling.

Run:  PYTHONPATH=<worktree> OMP_NUM_THREADS=2 python p5_steep.py
Writes p5_steep.json.
"""
from __future__ import annotations

import os

import _zcommon as zc
import numpy as np
from p2_taper_ladder import EPS_G, EPS_R, EPS_SUB, EPS_SUP, WL, C, P

HERE = os.path.dirname(os.path.abspath(__file__))
DT, DB = 0.2, 0.8


def run(tilt, pol, K, degree):
    h = 0.5 * (DB - DT) * P / tilt

    def xw(w):
        wd = P * (DT + (DB - DT) * w / h)
        return np.array([0.0, C - 0.5 * wd, C + 0.5 * wd, P])
    dm = 0.5 * (DT + DB) * P
    uw = [0.0, C - 0.5 * dm, C + 0.5 * dm, P]
    return zc.solve_taper(period=P, uw=uw, xw_of=xw, h=h,
                          eps_regions=[EPS_G, EPS_R, EPS_G], eps_sup=EPS_SUP,
                          eps_sub=EPS_SUB, wl=WL, pol=pol, degree=degree, K=K)


def main():
    info = zc.assert_tree()
    out = dict(info=info)
    for pol in ("te", "tm"):
        for tilt in (0.5, 1.0, 2.0, 3.0, 4.0):
            for deg in (12, 16):
                r = {K: run(tilt, pol, K, deg) for K in (16, 32, 64)}
                o = r[64]["orders"]
                d1 = zc.dist(r[16], r[32])["amp"]
                d2 = zc.dist(r[32], r[64])["amp"]
                row = dict(closure=max(r[K]["closure"] for K in r),
                           mirror=max(abs(r[K]["T"][o.index(-1)]
                                          - r[K]["T"][o.index(1)])
                                      for K in r),
                           step_16_32=d1, step_32_64=d2,
                           ratio=d1 / d2 if d2 > 0 else float("inf"))
                out[f"{pol}_tilt{tilt}_deg{deg}"] = row
                print(f"{pol} tilt {tilt:3.1f} deg {deg}: closure "
                      f"{row['closure']:.1e} mirror {row['mirror']:.1e} "
                      f"steps {d1:.2e} {d2:.2e} ratio {row['ratio']:.2f}",
                      flush=True)
    zc.dump(os.path.join(HERE, "p5_steep.json"), out)


if __name__ == "__main__":
    main()

"""P2 -- a linearly tapered 1-D ridge: route B (mapped slabs) against the
physical staircase (the shipped ``PMMStack.add_tapered_grating``).

Arms, all at normal incidence, TE and TM, two taper strengths:

  B      route B: K equal slabs, coefficients frozen at the slab midpoint,
         square-matched (``_zcommon.solve_taper``), K = 1 .. 256
  B-RE   Richardson of B, (4 f(2K) - f(K)) / 3  (fourth order if the error
         expansion of the exponential midpoint rule is even in 1/K)
  NOTILT route B with the tilt field dropped (S := 0 in the slabs): the
         engineered defect -- each slab is then the physical slice written on
         the shared u grid and glued by a square match, which is inconsistent
  M5     route B with the M5 prototype's non-conservative i beta <v|S'|phi>
         term added inside every slab (the form that gave a complex
         fundamental on a lossless cell, PMM_M5_2D_FEASIBILITY S3.6)
  STAIR  the shipped physical z-staircase, ``layer_grids='per-layer'``

Reference: B-RE at the top pair.  Distances are the largest change over every
R and T of orders -2..2 ("eff") and over the complex r00, t00 ("amp", B arms
only -- the shipped stack's amplitude phase reference is not compared).

Run:  PYTHONPATH=<worktree> OMP_NUM_THREADS=2 python p2_taper_ladder.py
Writes p2_taper_ladder.json.
"""
from __future__ import annotations

import os
import time
import warnings

import _zcommon as zc
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

P, WL, H = 0.8, 1.0, 0.6
EPS_R, EPS_G = 4.0, 1.0
EPS_SUP, EPS_SUB = 1.0, 1.45 ** 2
C = 0.5 * P
DEG = 16
TAPERS = {"moderate": (0.40, 0.60), "strong": (0.30, 0.70)}   # duty top, bottom
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
NS_STAIR = [2, 4, 8, 16, 32]


def xw_factory(dt, db):
    def xw(w):
        z = w / H                       # w = 0 is the TOP face (incidence side)
        wd = P * (dt + (db - dt) * z)
        return np.array([0.0, C - 0.5 * wd, C + 0.5 * wd, P])
    return xw


def route_b(dt, db, pol, K, **kw):
    dm = 0.5 * (dt + db) * P
    uw = [0.0, C - 0.5 * dm, C + 0.5 * dm, P]     # u walls = mid-height walls
    return zc.solve_taper(period=P, uw=uw, xw_of=xw_factory(dt, db), h=H,
                          eps_regions=[EPS_G, EPS_R, EPS_G], eps_sup=EPS_SUP,
                          eps_sub=EPS_SUB, wl=WL, pol=pol,
                          degree=kw.pop("degree", DEG), K=K, **kw)


def stair(dt, db, pol, ns, degree=DEG):
    from lumenairy.elements.pmm import PMMStack
    st = PMMStack(P, n_substrate=np.sqrt(EPS_SUB), n_superstrate=1.0,
                  degree=degree, far_field_orders=5, layer_grids="per-layer")
    st.add_tapered_grating(H, eps_ridge=EPS_R, eps_groove=EPS_G,
                           duty_top=dt, duty_bottom=db, n_slices=ns)
    st.set_source(WL, theta=0.0)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orders, R, T, _J = st.solve()
    dtm = time.perf_counter() - t0
    row = 1 if pol == "te" else 0
    o = [int(m) for m in np.asarray(orders)]
    sel = [o.index(m) for m in (-2, -1, 0, 1, 2)]
    return dict(R=[float(R[row][i]) for i in sel],
                T=[float(T[row][i]) for i in sel], time=dtm)


def eff_dist(a, b):
    return float(max(np.max(np.abs(np.subtract(a["R"], b["R"]))),
                     np.max(np.abs(np.subtract(a["T"], b["T"])))))


def rich(a, b):
    """(4 b - a) / 3 on every observable of two route-B results."""
    out = dict(orders=a["orders"])
    out["R"] = list((4 * np.asarray(b["R"]) - np.asarray(a["R"])) / 3)
    out["T"] = list((4 * np.asarray(b["T"]) - np.asarray(a["T"])) / 3)
    out["r00"] = (4 * b["r00"] - a["r00"]) / 3
    out["t00"] = (4 * b["t00"] - a["t00"]) / 3
    return out


def main():
    info = zc.assert_tree()
    res = dict(info=info, fixture=dict(P=P, WL=WL, H=H, EPS_R=EPS_R,
                                       EPS_G=EPS_G, EPS_SUP=EPS_SUP,
                                       EPS_SUB=EPS_SUB, DEG=DEG,
                                       TAPERS=TAPERS))
    for name, (dt, db) in TAPERS.items():
        ang = np.degrees(np.arctan(0.5 * (db - dt) * P / H))
        for pol in ("te", "tm"):
            key = f"{name}_{pol}"
            print(f"\n== {key}: duty {dt} -> {db}, sidewall {ang:.2f} deg")
            B, NT, M5 = {}, {}, {}
            for K in KS:
                t0 = time.perf_counter()
                B[K] = route_b(dt, db, pol, K)
                B[K]["time"] = time.perf_counter() - t0
                if K <= 64:
                    NT[K] = route_b(dt, db, pol, K, tilt=False)
                    try:
                        M5[K] = route_b(dt, db, pol, K, m5=True)
                    except Exception as exc:          # noqa: BLE001
                        M5[K] = dict(error=repr(exc))
            REs = {K: rich(B[K], B[2 * K]) for K in KS[:-1]}
            ref = REs[KS[-2]]
            # in-plane check of the reference: same K pair at degree + 4
            hi = rich(route_b(dt, db, pol, 64, degree=DEG + 4),
                      route_b(dt, db, pol, 128, degree=DEG + 4))
            inplane = zc.dist(REs[64], hi)
            rows = []
            for K in KS:
                r = dict(K=K, B=zc.dist(B[K], ref), closure=B[K]["closure"],
                         time=B[K]["time"])
                if K in REs:
                    r["B_RE_K_2K"] = zc.dist(REs[K], ref)
                if K in NT:
                    r["NOTILT"] = zc.dist(NT[K], ref)
                if K in M5:
                    r["M5"] = (zc.dist(M5[K], ref) if "error" not in M5[K]
                               else M5[K]["error"])
                rows.append(r)
                print(f"K {K:4d}: B {r['B']['eff']:.2e}/{r['B']['amp']:.2e}"
                      + (f"  RE {r['B_RE_K_2K']['eff']:.2e}/"
                         f"{r['B_RE_K_2K']['amp']:.2e}" if K in REs else "")
                      + (f"  NOTILT {r['NOTILT']['eff']:.2e}" if K in NT
                         else "")
                      + (f"  M5 {r['M5']['eff']:.2e}" if K in M5
                         and isinstance(r['M5'], dict) else "")
                      + f"  closure {r['closure']:.1e}  {r['time']:.2f}s")
            srows = []
            for ns in NS_STAIR:
                s = stair(dt, db, pol, ns)
                d = eff_dist(s, ref)
                srows.append(dict(ns=ns, eff=d, time=s["time"]))
                print(f"STAIR ns {ns:3d}: eff {d:.2e}  ({s['time']:.1f}s)")
            res[key] = dict(sidewall_deg=float(ang), ladder=rows,
                            stair=srows, ref_inplane_deg_plus4=inplane,
                            ref=dict(R=ref["R"], T=ref["T"],
                                     r00=ref["r00"], t00=ref["t00"]))
            print(f"reference in-plane check (deg {DEG} vs {DEG + 4}): "
                  f"{inplane}")
    zc.dump(os.path.join(HERE, "p2_taper_ladder.json"), res)


if __name__ == "__main__":
    main()

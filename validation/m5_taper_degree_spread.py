"""M5 side-finding -- PMM tapered-stack (degree, n_slice, layer_grids) spread,
adjudicated against the RCWA twin.

ANALYSIS ONLY.  Found while building the M5 SPIKE 2 staircase reference: on a
SIMPLE lossless dielectric tapered grating at in-plane oblique incidence, the
PMM stack's zeroth-order reflectance moves by O(1) across ``degree``,
``n_slice`` and ``layer_grids`` while ``|R+T-1| <= 1e-6`` throughout, i.e. the
shipped energy guard is blind to it.

This is the same configuration class as
``AUDIT_PMM_OBLIQUE_INPLANE_UNION_GRID_2026_07_28.md`` (in-plane oblique + taper
+ deep-ish observable), on a MUCH simpler device -- no coats, no LC, no absorbing
substrate, one tapered region.

The RCWA twin (``RCWAStack.add_tapered_grating``, ``raster='area'``) solves the
IDENTICAL geometry by an independent method and is used here as the arbiter.

Run:  python validation/m5_taper_degree_spread.py [out.json]
"""
from __future__ import annotations

import json
import platform
import sys
import warnings

import numpy as np

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.rcwa import RCWAStack

P, WL, H = 0.700, 1.310, 0.310
EPS_R, EPS_G = 4.0 + 0j, 1.0 + 0j
NSUP = NSUB = 1.5
DUTY = 0.5
THETA = np.deg2rad(8.0)
DD = 2.0 * H * np.tan(np.deg2rad(2.0)) / P     # duty change of a 2 deg taper


def pmm(ns, degree, grids, stabilize=None):
    st = PMMStack(P, n_substrate=NSUB, n_superstrate=NSUP, degree=degree,
                  layer_grids=grids)
    st.add_tapered_grating(H, eps_ridge=EPS_R, eps_groove=EPS_G,
                           duty_top=DUTY - 0.5 * DD,
                           duty_bottom=DUTY + 0.5 * DD,
                           n_slices=ns, rule="midpoint")
    st.set_source(WL, theta=THETA)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        o, R, T, _J = st.solve(stabilize=stabilize)
        guard = any("energy" in str(x.message) for x in caught)
    i0 = int(np.where(np.asarray(o) == 0)[0][0])
    return (float(R[0, i0]), float(R[1, i0]),
            float(np.max(np.sum(R, -1) + np.sum(T, -1))), guard)


def rcwa(ns, n_orders=21, n_x=4096):
    st = RCWAStack(P, n_superstrate=NSUP, n_substrate=NSUB, n_orders=n_orders)
    st.add_tapered_grating(H, eps_ridge=EPS_R, eps_groove=EPS_G,
                           duty_top=DUTY - 0.5 * DD,
                           duty_bottom=DUTY + 0.5 * DD,
                           n_slices=ns, n_x=n_x, raster="area")
    st.set_source(WL, theta=THETA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = st.solve()
    o, R, T = r.efficiencies()
    i0 = int(np.where(np.asarray(o) == 0)[0][0])
    return (float(R[0, i0]), float(R[1, i0]),
            float(np.max(np.sum(R, -1) + np.sum(T, -1))))


def main():
    res = {"platform": platform.platform(), "python": sys.version.split()[0],
           "numpy": np.__version__}
    try:
        res["blas"] = np.__config__.CONFIG["Build Dependencies"]["blas"]["name"]
    except Exception:
        res["blas"] = "unknown"
    print(f"# build: {res['python']} np{res['numpy']} blas={res['blas']} "
          f"{res['platform']}")
    print(f"# device: P={P*1e3:.0f} nm, wl={WL*1e3:.0f} nm, H={H*1e3:.0f} nm, "
          f"duty {DUTY}, eps 4/1, n_sup=n_sub={NSUP}, theta=8 deg (IN-PLANE),")
    print(f"#         2 deg symmetric taper (duty change {DD:.5f} periods)")

    nss = [1, 2, 3, 4, 6, 8, 12, 16]
    print("\n== ARBITER: RCWA twin, identical geometry, n_orders=21, "
          "raster='area' ==")
    ref = {}
    print(f"{'ns':>5}{'R0(te)':>14}{'R0(tm)':>14}{'R+T':>18}")
    for ns in nss + [384]:
        ref[ns] = rcwa(ns)
        print(f"{ns:>5}{ref[ns][0]:>14.9f}{ref[ns][1]:>14.9f}{ref[ns][2]:>18.12f}")
    conv = ref[384]
    print(f"   converged (ns=384): R0(te)={conv[0]:.9f}  R0(tm)={conv[1]:.9f}")
    print(f"   RCWA staircase span over ns=1..16: "
          f"te {max(ref[n][0] for n in nss)-min(ref[n][0] for n in nss):.3e}  "
          f"tm {max(ref[n][1] for n in nss)-min(ref[n][1] for n in nss):.3e}")
    res["rcwa"] = {str(k): v for k, v in ref.items()}

    print("\n== PMM: R0(te) vs (degree, n_slice, layer_grids) ==")
    print("   ('!' = library energy guard fired; 'X' = |R0 - RCWA(ns)| > 5e-3)")
    rows = []
    for grids in ("per-layer", "shared"):
        for deg in (8, 10, 12, 14, 16):
            cells = []
            for ns in nss:
                try:
                    te, tm, tot, g = pmm(ns, deg, grids)
                    bad = abs(te - ref[ns][0]) > 5e-3
                    cells.append(f"{te:.6f}{'!' if g else ''}{'X' if bad else ' '}")
                    rows.append(dict(grids=grids, degree=deg, ns=ns, te=te,
                                     tm=tm, tot=tot, guard=g,
                                     dev=abs(te - ref[ns][0])))
                except Exception as exc:
                    cells.append(f"ERR({type(exc).__name__})")
                    rows.append(dict(grids=grids, degree=deg, ns=ns,
                                     error=str(exc)[:80]))
            print(f"   {grids:<10s} deg={deg:>3}: " + " ".join(cells))
    res["pmm"] = rows
    ok = [r for r in rows if "te" in r]
    print(f"\n   worst |R0(te) - RCWA| over the grid: "
          f"{max(r['dev'] for r in ok):.3e}")
    print(f"   worst |R+T-1| over the grid        : "
          f"{max(abs(r['tot']-1) for r in ok):.3e}  "
          f"<-- the guard's observable")
    for ns in nss:
        sub = [r["te"] for r in ok if r["ns"] == ns]
        if sub:
            print(f"   ns={ns:>3}: PMM R0(te) spread over degree x grids = "
                  f"{max(sub)-min(sub):.3e}   (RCWA value {ref[ns][0]:.6f})")

    # The shipped R-1 union-grid consensus check: does it FIRE on the wrong
    # cells?  (Valid values are None or 'slices'; 'slices' is N/A on the
    # per-layer path by design.)  Slow -- 90-215 s per solve.
    print("\n== stabilize on the affected cells (valid values: None | 'slices') ==")
    for grids in ("shared", "per-layer"):
        for ns in (8, 12):
            for stab in (None, "slices"):
                try:
                    te, tm, tot, g = pmm(ns, 12, grids, stabilize=stab)
                    print(f"   {grids:<10s} deg=12 ns={ns:>2} "
                          f"stabilize={str(stab):8s}: R0(te)={te:.9f}  "
                          f"R+T={tot:.12f}  guard_fired={g}")
                    res.setdefault("stabilize", []).append(
                        dict(grids=grids, ns=ns, stabilize=str(stab), te=te,
                             tot=tot, guard=bool(g)))
                except Exception as exc:
                    print(f"   {grids:<10s} deg=12 ns={ns:>2} "
                          f"stabilize={str(stab):8s}: {type(exc).__name__}: "
                          f"{str(exc)[:110]}")
                    res.setdefault("stabilize", []).append(
                        dict(grids=grids, ns=ns, stabilize=str(stab),
                             error=str(exc)[:200]))

    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        print(f"[json] {sys.argv[1]}")


if __name__ == "__main__":
    main()

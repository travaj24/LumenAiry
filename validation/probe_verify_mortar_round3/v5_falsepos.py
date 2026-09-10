"""V5 -- two FALSE-POSITIVE candidates for the round-3 band warning, and one
SILENT-WRONG-ANSWER hunt for the residual screen.

**FP-1, the UNIFORM LATTICE.**  ``_stag_band_narrowest`` scores a UNIFORM
basis as ``1 / N``, so a per-layer stack carrying a fine uniform lattice
(``N >= 34``, i.e. every cell 2.9e-2 of the period) lands in the band and
warns.  A uniform lattice is not a sliver: every segment is the same width, so
there is no ``1/J_n`` CONTRAST for the mortar to see.  This probe asks whether
the warning fires there and whether the answer is actually degraded, by
comparing the uniform-lattice arm against the SAME device on a coarse grid and
against a conforming (mortar-free) twin.

**FP-2, a sliver on the CONFORMING axis.**  Two layers may differ on x and
agree exactly on y.  The y mortar is then the identity, so a narrow y segment
SHARED by every layer has no cross-grid projection to corrupt -- but
``_stag_band_narrowest`` scans BOTH axes of every grid and warns anyway.  This
probe measures whether such a stack's answer moves with the y width.

**The HUNT.**  An ORDINARY geometry (narrowest segment above the band, so
neither the contract nor the warning fires) whose answer is WRONG while the
residual stays under the 1e-6 bar.  Tried: extreme index contrast, many
orders, ``M`` = 8..10, near-Wood incidence, a dense superstrate, tiny-but-legal
segments just above the contract, and deep stacks, each scored against a
mortar-free twin built on the COMMON REFINEMENT of the same device.

Run: ``python v5_falsepos.py <tag> [fp1|fp2|hunt|all]``
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _path                                     # noqa: E402,F401,I001

import json                                                 # noqa: E402
import time                                                 # noqa: E402
import warnings                                             # noqa: E402

import numpy as np                                          # noqa: E402

import _vfix as F                                           # noqa: E402
from _capture import capture                                # noqa: E402
from lumenairy.elements.pmm import PMM2DStackPure           # noqa: E402
from lumenairy.elements.pmm import _core as _pc             # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts    # noqa: E402
from lumenairy.elements.pmm.twod_staggered import _C        # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
BAR = _pc._MORTAR_RESID_REFUSE


def _solve(st):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        o, R, T = st.solve(jones=False)
    band = [str(w.message) for w in ws if "degradation band" in str(w.message)]
    return {"R00": F.R00(o, R),
            "closure": abs(float(R.sum(axis=1)[1] + T.sum(axis=1)[1]) - 1.0),
            "n_band_warnings": len(band),
            "band_axis": ("y" if band and " y axis" in band[0]
                          else "x" if band else None)}


def _residuals(st):
    """Every mortar residual this stack's solve builds -- BOTH the generalized
    site and the two in-plane ones -- measured the way the shipped screen
    measures it.  An in-plane-only stack reaches no generalized site at all,
    which is itself a reading."""
    with capture(record_all=True) as recs:
        st.solve(jones=False)
    out = []
    for r in recs:
        X = np.linalg.solve(r["A"], r["B"])
        out.append({"n": r["n"], "screen": r["screen"],
                    "exact": float(_pc._mortar_residual(r["A"], X, r["B"],
                                                        probe=False)),
                    "probe": float(_pc._mortar_residual(r["A"], X, r["B"],
                                                        probe=True))})
    return out


# ------------------------------------------------------------------- FP-1
def fp1(rows):
    """A UNIFORM lattice in a per-layer stack: does it warn, and is it worse?

    The device is ONE in-plane pillar layer on an explicit 3-segment grid next
    to a UNIFORM-lattice layer of N cells filled with the SAME uniform
    permittivity, so the DEVICE does not change with N at all and every
    movement is numerical.
    """
    P, WL_, TH_ = 1.0, 0.8, 0.2
    WA = (0.2371, 0.6183)

    def st_of(N, M=5, conforming=False):
        st = PMM2DStackPure(P, n_modes=M, n_orders=1, layer_grids="per-layer")
        st.add_layer(0.12, eps_cell=F.scalar_cell(WA, *WA, eps_in=6.0),
                     x_walls=[w * P for w in WA], y_walls=[w * P for w in WA])
        if conforming:
            st.add_layer(0.09, eps=2.1, x_walls=[w * P for w in WA],
                         y_walls=[w * P for w in WA])
        else:
            st.add_layer(0.09, eps=2.1, grid=N)
        st.set_source(WL_, theta=TH_, phi=0.4)
        return st

    ref = _solve(st_of(0, conforming=True))
    rows.append({"what": "fp1_conforming_reference", **ref})
    print(f"  FP1 conforming twin R00={ref['R00']:.12f}", flush=True)
    # N is capped at 16 BY COST, and that cap is itself the finding: the
    # staggered basis carries q = N (M - 1) per axis and a 2 q^2 region eig,
    # so a uniform lattice fine enough to land in the 3e-2 band (N >= 34) is a
    # 2 q^2 >= 20808 dense eigenproblem at M = 4 -- 7 GiB and hours.  A
    # uniform lattice therefore CANNOT reach the band through the public API,
    # which is what turns FP-1 from a defect into a bounded observation.
    for N in (1, 2, 4, 6, 8):
        t0 = time.time()
        try:
            s = _solve(st_of(N))
            res = _residuals(st_of(N))
        except Exception as e:                              # noqa: BLE001
            rows.append({"what": "fp1", "N": N,
                         "error": f"{type(e).__name__}: {e}"[:200]})
            print(f"  FP1 N={N} ERROR {type(e).__name__}", flush=True)
            continue
        rel = abs(s["R00"] - ref["R00"]) / abs(ref["R00"])
        rows.append({"what": "fp1", "N": N, "cell_fraction": 1.0 / N,
                     "rel_vs_conforming": rel, "residuals": res, **s,
                     "seconds": round(time.time() - t0, 2)})
        print(f"  FP1 N={N:4d} cell={1.0 / N:.4e} R00={s['R00']:.12f} "
              f"rel={rel:.3e} warn={s['n_band_warnings']} "
              f"resid={max((r['exact'] for r in res), default=float('nan')):.2e}",
              flush=True)


# ------------------------------------------------------------------- FP-2
def fp2(rows):
    """A narrow segment on the axis every layer AGREES on.

    Two layers differ on x (so the stack is mortared) and carry the SAME y
    wall array, whose middle segment is ``fy`` wide.  The y walls sit inside a
    region of constant permittivity, so the DEVICE is independent of ``fy``.
    """
    P, WL_, TH_ = 1.07, 0.79, 0.31
    yc = 0.585

    def st_of(fy, M=5, xconf=False):
        yws = [(yc - fy / 2) * P, (yc + fy / 2) * P]
        st = PMM2DStackPure(P, n_modes=M, n_orders=1, layer_grids="per-layer")
        st.add_layer(0.17, eps_cell=np.full((3, 3), _C(2.5)),
                     x_walls=[0.25 * P, 0.60 * P], y_walls=yws)
        st.add_layer(0.17, eps_cell=np.full((3, 3), _C(3.5)),
                     x_walls=([0.25 * P, 0.60 * P] if xconf
                              else [0.31 * P, 0.66 * P]),
                     y_walls=yws)
        st.set_source(WL_, theta=TH_, phi=0.0)
        return st

    base = None
    for fy in (3e-1, 1e-1, 5e-2, 3e-2, 1e-2, 3e-3, 1.2e-3):
        for xconf in (False, True):
            try:
                s = _solve(st_of(fy, xconf=xconf))
            except Exception as e:                          # noqa: BLE001
                rows.append({"what": "fp2", "fy": fy, "x_conforming": xconf,
                             "error": f"{type(e).__name__}: {e}"[:200]})
                continue
            if base is None and not xconf:
                base = s["R00"]
            rel = (abs(s["R00"] - base) / abs(base)
                   if base is not None and not xconf else float("nan"))
            rows.append({"what": "fp2", "fy": fy, "x_conforming": xconf,
                         "rel_vs_widest": rel, **s})
            print(f"  FP2 fy={fy:.1e} xconf={int(xconf)} "
                  f"R00={s['R00']:.12f} move={rel:.3e} "
                  f"warn={s['n_band_warnings']} axis={s['band_axis']}",
                  flush=True)


# -------------------------------------------------------------------- HUNT
def hunt(rows):
    """An ORDINARY geometry -- narrowest segment ABOVE the band, so nothing
    warns and nothing refuses -- whose answer is WRONG while the residual is
    under the bar.  Each case is scored against a mortar-free twin built on
    the COMMON REFINEMENT of the same device."""
    P = 0.87e-6

    def pair(name, WA, WB, *, M, wl, th, ph, eps_in, n_sup=1.0, n_sub=1.45,
             n_orders=2, eps_oop=None, layers=2):
        """(per-layer arm, common-refinement arm) of the SAME device."""
        U = tuple(sorted(set(WA) | set(WB)))

        def build(union):
            st = PMM2DStackPure(P, n_modes=M, n_orders=n_orders,
                                n_superstrate=n_sup, n_substrate=n_sub,
                                layer_grids="per-layer")
            wa, wb = (U, U) if union else (WA, WB)
            st.add_layer(0.118e-6,
                         eps_cell=F.tensor_cell(wa, WA[0], WA[-1],
                                                eps_in=eps_oop),
                         x_walls=F.sc(wa, P), y_walls=F.sc(wa, P))
            st.add_layer(0.094e-6,
                         eps_cell=F.scalar_cell(wb, WB[0], WB[-1],
                                                eps_in=eps_in),
                         x_walls=F.sc(wb, P), y_walls=F.sc(wb, P))
            if layers > 2:
                for i in range(layers - 2):
                    w = wa if i % 2 else wb
                    st.add_layer(0.055e-6,
                                 eps_cell=F.scalar_cell(w, w[0], w[-1],
                                                        eps_in=2.5 + i),
                                 x_walls=F.sc(w, P), y_walls=F.sc(w, P))
            st.set_source(wl, theta=th, phi=ph)
            return st
        return name, build(False), build(True)

    cases = [
        pair("baseline", (0.1873, 0.5412), (0.3106, 0.8039), M=5, wl=0.73e-6,
             th=0.17, ph=1.1, eps_in=8.41),
        pair("extreme_contrast", (0.1873, 0.5412), (0.3106, 0.8039), M=5,
             wl=0.73e-6, th=0.17, ph=1.1, eps_in=144.0),
        pair("lossy_metal", (0.1873, 0.5412), (0.3106, 0.8039), M=5,
             wl=0.73e-6, th=0.17, ph=1.1, eps_in=complex(-20.0, 1.5)),
        pair("many_orders", (0.1873, 0.5412), (0.3106, 0.8039), M=6,
             wl=0.73e-6, th=0.17, ph=1.1, eps_in=8.41, n_orders=5),
        pair("M8", (0.1873, 0.5412), (0.3106, 0.8039), M=8, wl=0.73e-6,
             th=0.17, ph=1.1, eps_in=8.41),
        pair("dense_superstrate", (0.1873, 0.5412), (0.3106, 0.8039), M=5,
             wl=0.73e-6, th=0.17, ph=1.1, eps_in=8.41, n_sup=2.4, n_sub=3.5),
        # near-Wood: theta chosen so sin(th) + wl/P lands on a cutoff
        pair("near_Wood", (0.1873, 0.5412), (0.3106, 0.8039), M=5,
             wl=0.73e-6, th=float(np.arcsin(1.0 - 0.73 / 0.87 + 1e-6)),
             ph=0.0, eps_in=8.41),
        # tiny but LEGAL segments -- just above the contract, and just above
        # the band's own upper edge
        pair("just_above_contract", (0.30, 0.3011), (0.3106, 0.8039), M=5,
             wl=0.73e-6, th=0.17, ph=1.1, eps_in=8.41),
        pair("just_above_band", (0.30, 0.3320), (0.3106, 0.8039), M=5,
             wl=0.73e-6, th=0.17, ph=1.1, eps_in=8.41),
        pair("deep6", (0.1873, 0.5412), (0.3106, 0.8039), M=4, wl=0.73e-6,
             th=0.17, ph=1.1, eps_in=8.41, layers=6),
        pair("strong_oop", (0.1873, 0.5412), (0.3106, 0.8039), M=5,
             wl=0.73e-6, th=0.17, ph=1.1, eps_in=8.41,
             eps_oop=np.array([[3.10, 0.0, 1.55], [0.0, 2.70, 0.0],
                               [1.48, 0.0, 2.45]], dtype=_C)),
    ]
    for name, per_layer, union in cases:
        t0 = time.time()
        try:
            a = _solve(per_layer)
            res = _residuals(per_layer)
            b = _solve(union)
        except Exception as e:                              # noqa: BLE001
            rows.append({"what": "hunt", "name": name,
                         "error": f"{type(e).__name__}: {e}"[:300]})
            print(f"  HUNT {name}: ERROR {type(e).__name__} "
                  f"{str(e)[:110]}", flush=True)
            continue
        rel = abs(a["R00"] - b["R00"]) / max(abs(b["R00"]), 1e-30)
        worst = max((r["exact"] for r in res), default=float("nan"))
        rows.append({"what": "hunt", "name": name, "per_layer": a,
                     "union": b, "rel_vs_union": rel,
                     "worst_residual": worst,
                     "residual_under_bar": bool(worst <= BAR),
                     "residuals": res,
                     "seconds": round(time.time() - t0, 2)})
        print(f"  HUNT {name:22s} R00={a['R00']:.9f} union={b['R00']:.9f} "
              f"rel={rel:.3e} resid={worst:.2e} warn={a['n_band_warnings']} "
              f"clo={a['closure']:.1e} {time.time() - t0:.0f}s", flush=True)


def main(tag, what="all"):
    rows = []
    if what in ("all", "fp1"):
        print("FP1 uniform lattice", flush=True)
        fp1(rows)
    if what in ("all", "fp2"):
        print("FP2 conforming-axis sliver", flush=True)
        fp2(rows)
    if what in ("all", "hunt"):
        print("HUNT silent wrong answer", flush=True)
        hunt(rows)
    out = {"env": F.env(), "tag": tag, "what": what, "bar": BAR,
           "band_edge": _ts._STAG_SLIVER_BAND_FRAC, "rows": rows}
    (HERE / f"v5_falsepos_{tag}_{what}.json").write_text(
        json.dumps(out, indent=1), encoding="cp1252")
    print(f"wrote v5_falsepos_{tag}_{what}.json ({len(rows)} rows)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "win",
         sys.argv[2] if len(sys.argv) > 2 else "all")

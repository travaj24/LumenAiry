"""GATE 0 -- adjudicate the ONE open item of the Stage-B prototype verdict.

``EXPERIMENT_PMM2D_STAGGERED_OOP_2026_09_09.md`` S9.2 / S9.3: on a (3,3) cell
whose feature is an L (a RE-ENTRANT 270-degree corner), the staggered
out-of-plane arm departs from both Fourier oracles by 2.20e-03 on ``sum R``
while the IN-PLANE control on the SAME walls agrees at 3.47e-05.  Neither
Fourier arm was converged there and both route their tensor layer through
``rcwa._core._layer_eigenmodes_tensor``, so their mutual agreement is not
independent evidence about the out-of-plane blocks.

THE GATE (integration brief, 2026-09-09).  Three ladders on the SAME cell:

  (i)   the staggered out-of-plane arm's M-ladder -- is it SELF-CONVERGENT
        (successive differences shrinking) on ``sum R`` and per order?
  (ii)  the hybrid ``pmm_jones_2d`` n_orders ladder, BOTH 'li' and 'laurent'
        -- how far does ITS OWN answer move, and do the two E_z-elimination
        rules agree with each other?  Plus ``rcwa_jones_2d``'s own n_orders
        ladder (the second Fourier arm).
  (iii) the IN-PLANE control on the SAME walls, both arms.

VERDICT RULE.  If the staggered arm is self-convergent AND the Fourier arms'
own drift / mutual floor is of the order of the discrepancy, the item is
BOUNDED (a Fourier out-of-plane floor: the cross-blocks use unfactorized
convolutions) and the integration proceeds with a hybrid-agreement bar derived
from the measured Fourier drift.  If the staggered arm does not converge, or
converges to a value the Fourier arms do not approach as n_orders grows, STOP.

Run:
  cd /c/tmp/lum_aniso_oopint && PYTHONPATH=/c/tmp/lum_aniso_oopint \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python validation/probe_pmm2d_staggered_oop/g0_corner_gate.py
"""
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import probe_common as pc  # noqa: E402

from lumenairy.elements.pmm import pmm_jones_2d  # noqa: E402
from lumenairy.elements.rcwa import rcwa_jones_2d  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

WL = 1.0
PX = PY = 1.2
DEPTH = 0.4
NSUB, NSUP = 1.5, 1.0

#: M-ladder budget for the staggered arm (candidate (a), dim 4*(3*(M-1))^2).
M_LADDER = (4, 5, 6, 7, 8, 9)
#: hybrid n_orders ladder (its 4Nf generator; Nf = (2n+1)^2)
HYB_ORDERS = (7, 9, 11, 13)
#: rcwa n_orders ladder
RCWA_ORDERS = (5, 7, 9)


def strip_oop(t):
    o = np.array(t, dtype=complex)
    o[0, 2] = o[1, 2] = o[2, 0] = o[2, 1] = 0.0
    return o


def lcell(er, eg):
    """The M9 L: three ``er`` pixels with a RE-ENTRANT 270-degree corner."""
    e = np.zeros((3, 3, 3, 3), dtype=complex)
    e[:, :] = eg
    for i, j in ((0, 0), (1, 0), (0, 1)):
        e[i, j] = er
    return e


def upsample(ec, n):
    return np.repeat(np.repeat(np.asarray(ec), n, axis=0), n, axis=1)


def _orders_key(o):
    return [tuple(int(v) for v in row) for row in np.asarray(o)]


def _per_order(o, R, T, keep):
    """Per-order (R, T) for the orders in ``keep`` (a list of (m, n))."""
    idx = {tuple(int(v) for v in row): j for j, row in enumerate(np.asarray(o))}
    out = {}
    for k in keep:
        j = idx.get(k)
        if j is not None:
            out[k] = (np.asarray(R)[:, j].copy(), np.asarray(T)[:, j].copy())
    return out


#: the propagating orders of this cell (px = py = 1.2 lam, n_sub = 1.5): the
#: reflected set is |m|, |n| <= 0 at normal ... measured below and printed.
KEEP = [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1), (1, 1)]


def summarize(tag, rows, key):
    """Print a ladder with successive differences on ``sum R`` per row."""
    print(f"\n   -- {tag}: ladder on {key} --")
    prev = None
    for lab, val, tsec in rows:
        d = "" if prev is None else f"   step |d| = {np.max(np.abs(val - prev)):.3e}"
        print(f"      {lab:>18s}  {np.array2string(val, precision=9)}"
              f"  [{tsec:6.1f} s]{d}")
        prev = val


def main():
    pc.banner("GATE 0 -- re-entrant-corner adjudication (M9 cell)")
    print(f"# cell (3,3) L with a RE-ENTRANT corner; px=py={PX} lam, "
          f"depth={DEPTH} lam, n_sub={NSUB}; normal incidence")
    er_oop = pc.uniaxial(1.5, 1.7, 35.0, azim_deg=25.0)
    eg = np.eye(3, dtype=complex)
    res = {}

    for tag, er in (("OOP", er_oop), ("INPLANE", strip_oop(er_oop))):
        cand = "a" if tag == "OOP" else "eform"
        print(f"\n================ {tag} tensor "
              f"(staggered candidate '{cand}') ================")
        ec = lcell(er, eg)
        block = {}

        # ---------------- (ii) Fourier arms ------------------------------
        hyb = {}
        for form in ("laurent", "li"):
            rows = []
            for no in HYB_ORDERS:
                t0 = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    o, Rm, Tm, J = pmm_jones_2d(
                        PX, PY, ec, NSUB, NSUP, DEPTH, WL, degree=9,
                        n_orders=no, formulation=form, stabilize=True)
                dt = time.time() - t0
                Rt, Tt = Rm.sum(axis=1), Tm.sum(axis=1)
                hyb[(form, no)] = dict(
                    o=o, R=Rm, T=Tm, J=J, Rt=Rt, Tt=Tt,
                    close=float(np.max(np.abs(Rt + Tt - 1))), t=dt)
                rows.append((f"hyb-{form}({no})", Rt, dt))
                block[f"hyb_{form}_{no}"] = dict(
                    sumR=Rt.tolist(), sumT=Tt.tolist(),
                    closure=hyb[(form, no)]["close"], t=dt)
                print(f"      hyb-{form}({no:2d})  sumR = {Rt}  "
                      f"|R+T-1| = {hyb[(form, no)]['close']:.3e}  [{dt:.1f} s]")
            summarize(f"hybrid '{form}'", rows, "sum R")

        rc = {}
        rows = []
        for no in RCWA_ORDERS:
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fine = upsample(ec, int(np.ceil((4 * no + 1) / 3)))
                o, Rm, Tm, J = rcwa_jones_2d(PX, PY, fine, NSUB, NSUP, DEPTH,
                                             WL, n_orders_x=no, n_orders_y=no)
            dt = time.time() - t0
            Rt, Tt = Rm.sum(axis=1), Tm.sum(axis=1)
            rc[no] = dict(o=o, R=Rm, T=Tm, J=J, Rt=Rt, Tt=Tt,
                          close=float(np.max(np.abs(Rt + Tt - 1))), t=dt)
            rows.append((f"rcwa({no})", Rt, dt))
            block[f"rcwa_{no}"] = dict(sumR=Rt.tolist(), sumT=Tt.tolist(),
                                       closure=rc[no]["close"], t=dt)
            print(f"      rcwa({no:2d})       sumR = {Rt}  "
                  f"|R+T-1| = {rc[no]['close']:.3e}  [{dt:.1f} s]")
        summarize("rcwa", rows, "sum R")

        # ---------------- (i)/(iii) staggered arm ------------------------
        stg = {}
        rows = []
        for M in M_LADDER:
            t0 = time.time()
            o, Rm, Tm, J = pc.solve_slab(PX, PY, ec, NSUB, NSUP, DEPTH, WL,
                                         M=M, candidate=cand)
            dt = time.time() - t0
            Rt, Tt = Rm.sum(axis=1), Tm.sum(axis=1)
            stg[M] = dict(o=o, R=Rm, T=Tm, J=J, Rt=Rt, Tt=Tt,
                          close=float(np.max(np.abs(Rt + Tt - 1))), t=dt)
            rows.append((f"stag M={M}", Rt, dt))
            block[f"stag_M{M}"] = dict(sumR=Rt.tolist(), sumT=Tt.tolist(),
                                       closure=stg[M]["close"], t=dt)
            print(f"      stag M={M} dim={4*(3*(M-1))**2:5d}  sumR = {Rt}  "
                  f"|R+T-1| = {stg[M]['close']:.3e}  [{dt:.1f} s]")
        summarize("staggered", rows, "sum R")

        # ---------------- cross-arm distances ----------------------------
        hyb_top = hyb[("laurent", HYB_ORDERS[-1])]
        hyb_top_li = hyb[("li", HYB_ORDERS[-1])]
        rc_top = rc[RCWA_ORDERS[-1]]
        print("\n   -- cross-arm, sum R (top of each ladder) --")
        pairs = {
            "hyb_laurent_vs_hyb_li": np.max(np.abs(hyb_top["Rt"]
                                                   - hyb_top_li["Rt"])),
            "hyb_laurent_vs_rcwa": np.max(np.abs(hyb_top["Rt"] - rc_top["Rt"])),
            "hyb_li_vs_rcwa": np.max(np.abs(hyb_top_li["Rt"] - rc_top["Rt"])),
            "stagMax_vs_hyb_laurent": np.max(np.abs(stg[M_LADDER[-1]]["Rt"]
                                                    - hyb_top["Rt"])),
            "stagMax_vs_hyb_li": np.max(np.abs(stg[M_LADDER[-1]]["Rt"]
                                               - hyb_top_li["Rt"])),
            "stagMax_vs_rcwa": np.max(np.abs(stg[M_LADDER[-1]]["Rt"]
                                             - rc_top["Rt"])),
        }
        for k, v in pairs.items():
            print(f"      {k:28s} = {v:.3e}")
            block[k] = float(v)

        # own drift of each ladder (top step)
        drift = {
            "hyb_laurent_drift": np.max(np.abs(
                hyb[("laurent", HYB_ORDERS[-1])]["Rt"]
                - hyb[("laurent", HYB_ORDERS[-2])]["Rt"])),
            "hyb_li_drift": np.max(np.abs(
                hyb[("li", HYB_ORDERS[-1])]["Rt"]
                - hyb[("li", HYB_ORDERS[-2])]["Rt"])),
            "rcwa_drift": np.max(np.abs(rc[RCWA_ORDERS[-1]]["Rt"]
                                        - rc[RCWA_ORDERS[-2]]["Rt"])),
            "stag_drift": np.max(np.abs(stg[M_LADDER[-1]]["Rt"]
                                        - stg[M_LADDER[-2]]["Rt"])),
        }
        print("\n   -- own drift of the TOP step of each ladder (sum R) --")
        for k, v in drift.items():
            print(f"      {k:28s} = {v:.3e}")
            block[k] = float(v)

        # ---------------- per-order T, top of every ladder ---------------
        print("\n   -- per-order T (row0 = incident Ex), top of each ladder --")
        po = {
            "stag": _per_order(stg[M_LADDER[-1]]["o"], stg[M_LADDER[-1]]["R"],
                               stg[M_LADDER[-1]]["T"], KEEP),
            "stagPrev": _per_order(stg[M_LADDER[-2]]["o"],
                                   stg[M_LADDER[-2]]["R"],
                                   stg[M_LADDER[-2]]["T"], KEEP),
            "hyb_laurent": _per_order(hyb_top["o"], hyb_top["R"],
                                      hyb_top["T"], KEEP),
            "hyb_li": _per_order(hyb_top_li["o"], hyb_top_li["R"],
                                 hyb_top_li["T"], KEEP),
            "rcwa": _per_order(rc_top["o"], rc_top["R"], rc_top["T"], KEEP),
        }
        for k in KEEP:
            if k not in po["stag"]:
                continue
            line = f"      T{str(k):9s}"
            for arm in ("stag", "hyb_laurent", "hyb_li", "rcwa"):
                line += f"  {arm}={po[arm][k][1][0]:.7f}"
            print(line)
        block["per_order_T"] = {
            f"{arm}|{k}": po[arm][k][1].tolist()
            for arm in po for k in KEEP if k in po[arm]}
        # per-order self-convergence of the staggered arm
        po_move = max(float(np.max(np.abs(po["stag"][k][0]
                                          - po["stagPrev"][k][0])))
                      for k in KEEP if k in po["stag"])
        po_moveT = max(float(np.max(np.abs(po["stag"][k][1]
                                           - po["stagPrev"][k][1])))
                       for k in KEEP if k in po["stag"])
        print(f"      staggered per-order self-move M={M_LADDER[-2]} -> "
              f"{M_LADDER[-1]}: R {po_move:.3e}  T {po_moveT:.3e}")
        block["stag_per_order_move_R"] = po_move
        block["stag_per_order_move_T"] = po_moveT
        # per-order distance staggered(top) vs each Fourier arm(top)
        for arm in ("hyb_laurent", "hyb_li", "rcwa"):
            dR = max(float(np.max(np.abs(po["stag"][k][0] - po[arm][k][0])))
                     for k in KEEP if k in po["stag"] and k in po[arm])
            dT = max(float(np.max(np.abs(po["stag"][k][1] - po[arm][k][1])))
                     for k in KEEP if k in po["stag"] and k in po[arm])
            print(f"      per-order stag vs {arm:12s}: R {dR:.3e}  T {dT:.3e}")
            block[f"per_order_stag_vs_{arm}_R"] = dR
            block[f"per_order_stag_vs_{arm}_T"] = dT

        # Jones
        for arm, d in (("hyb_laurent", hyb_top), ("hyb_li", hyb_top_li),
                       ("rcwa", rc_top)):
            dj = float(np.max(np.abs(stg[M_LADDER[-1]]["J"] - d["J"])))
            print(f"      Jones stag vs {arm:12s} = {dj:.3e}")
            block[f"jones_stag_vs_{arm}"] = dj
        block["jones_stag_self_move"] = float(np.max(np.abs(
            stg[M_LADDER[-1]]["J"] - stg[M_LADDER[-2]]["J"])))
        print(f"      Jones staggered self-move = "
              f"{block['jones_stag_self_move']:.3e}")

        res[tag] = block

    with open(os.path.join(OUT, "g0_corner_gate.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("\nwrote results/g0_corner_gate.json")


if __name__ == "__main__":
    main()

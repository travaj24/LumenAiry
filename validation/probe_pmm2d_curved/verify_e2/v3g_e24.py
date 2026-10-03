"""E2-4 / E2-6 on the verifier's CROSSING pairs (v3g_geoms.py).
Modes:
  closure <pair> M            lossless closure, R/T, adaptive n, wall
  absorb  <pair> <l1|l2|both> M   layer_absorption vs 1 - R - T
  vacuum  <circ|fil> M        vacuum-painted shape layer on top of the mapped
                              layer vs the same spacer as a uniform eps=1
                              layer on the shared map, and vs the device alone
  recip   M [noswap]          pair (i) at (20, 0) and (20, 35) deg: closure
                              and reflection reciprocity of order (-1, 0)
  awk     <name> M [force]    awkward geometry: solve, closure, warnings
"""
import sys
import time
import warnings

import numpy as np
from _ve import dump, solve
from v3g_fix import NREC, WL, jones_block, order_index, reverse_angles, stack
from v3g_geoms import awkward, pairs

from lumenairy.elements.pmm import _core

mode = sys.argv[1]
out = {"mode": mode, "argv": sys.argv[1:]}
t00 = time.perf_counter()


def clos(R, T):
    return np.abs(R.sum(1) + T.sum(1) - 1.0)


def run(st, th=0.0, ph=0.0, retain=False):
    k0 = len(NREC)
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        o, R, T, J = solve(st, WL, th, ph, retain)
    return dict(o=o, R=R, T=T, closure=clos(R, T),
                n=[r["n"] for r in NREC[k0:]],
                change=[r["change"] for r in NREC[k0:]],
                xwall=[r["wall"] for r in NREC[k0:]],
                Mab=[(r["Ma"], r["Mb"], r["Na"], r["Nb"]) for r in NREC[k0:]],
                warnings=[str(x.message)[:300] for x in w],
                wall=time.perf_counter() - t0)


def strip(r):
    return {k: v for k, v in r.items() if k not in ("o",)}


if mode == "closure":
    name, M = sys.argv[2], int(sys.argv[3])
    s1, s2, t1, t2 = pairs()[name]
    st = stack([(t1, s1, 1.0), (t2, s2, 1.0)], M)
    assert not st._perlayer_fast_ok(), "merge did not refuse"
    out["refusal"] = st._merge_refusal[:120]
    out["M_layers"] = [int(L.get("M") or M) for L in st._layers]
    r = run(st)
    out.update(strip(r))
    print(name, M, r["closure"], r["n"], r["wall"], r["warnings"])
    dump(f"v3g_e24_closure_{name}_M{M}", out)

elif mode == "absorb":
    name, which, M = sys.argv[2], sys.argv[3], int(sys.argv[4])
    s1, s2, t1, t2 = pairs(which)[name]
    st = stack([(t1, s1, 1.0), (t2, s2, 1.0)], M)
    assert not st._perlayer_fast_ok()
    r = run(st, retain=True)
    A = np.asarray(st.layer_absorption())
    budget = 1.0 - r["R"].sum(1) - r["T"].sum(1)
    out.update(strip(r), absorption=A, budget=budget,
               mismatch=np.abs(A.sum(0) - budget))
    if which == "l1":
        out["lossless_layer_abs"] = np.abs(A[1])
    elif which == "l2":
        out["lossless_layer_abs"] = np.abs(A[0])
    print(name, which, M, "A", A.tolist(), "budget", budget.tolist(),
          "mismatch", out["mismatch"].tolist(),
          "lossless", out.get("lossless_layer_abs"))
    dump(f"v3g_e24_absorb_{name}_{which}_M{M}", out)

elif mode == "vacuum":
    which, M = sys.argv[2], int(sys.argv[3])
    s1, s2, t1, t2 = pairs()["i_circ_sin" if which == "circ"
                             else "iii_fil_sin"]
    # the vacuum-painted shape is the OTHER layer's shape painted eps = 1
    from lumenairy.elements.pmm.shapes2d import SinusoidalWall
    vac = [SinusoidalWall(s2[0].axis, s2[0].x0, s2[0].amplitude,
                          phase=s2[0].phase, eps=1.0)]
    a = run(stack([(t2, vac, 1.0), (t1, s1, 1.0)], M))
    b = run(stack([(t2, None, 1.0), (t1, s1, 1.0)], M, per_layer=False))
    c = run(stack([(t1, s1, 1.0)], M, per_layer=False))
    stn = stack([(t2, vac, 1.0), (t1, s1, 1.0)], M)
    stn._e2_no_ride = True
    d = run(stn)

    def dd(x, y):
        return float(max(np.abs(x["R"] - y["R"]).max(),
                         np.abs(x["T"] - y["T"]).max()))
    out.update(perlayer_vs_shared_spacer=dd(a, b),
               bytes_equal=bool(np.array_equal(a["R"], b["R"])
                                and np.array_equal(a["T"], b["T"])),
               spacer_vs_alone=dd(b, c), perlayer_vs_alone=dd(a, c),
               noride_vs_alone=dd(d, c), noride_n=d["n"],
               perlayer_n=a["n"], closure=a["closure"],
               noride_closure=d["closure"],
               walls=[a["wall"], b["wall"], c["wall"], d["wall"]])
    print(which, M, {k: v for k, v in out.items() if k != "argv"})
    dump(f"v3g_e24_vacuum_{which}_M{M}", out)

elif mode == "recip":
    M = int(sys.argv[2])
    noswap = len(sys.argv) > 3 and sys.argv[3] == "noswap"
    s1, s2, t1, t2 = pairs()["i_circ_sin"]
    lay = [(t1, s1, 1.0), (t2, s2, 1.0)]
    _core.PMM2D_MORTAR_H_SWAP = not noswap
    for th_d, ph_d in ((20.0, 0.0), (20.0, 35.0)):
        th, ph = np.deg2rad(th_d), np.deg2rad(ph_d)
        st = stack(lay, M)
        rf = run(st, th, ph)
        k = order_index(rf["o"], (-1, 0))
        Nf = jones_block(st, k)
        tr, pr = reverse_angles(th, ph, -1, 0)
        st2 = stack(lay, M)
        rr = run(st2, tr, pr)
        Nr = jones_block(st2, order_index(rr["o"], (-1, 0)))
        Nw = jones_block(st2, order_index(rr["o"], (0, 0)))
        sf, sr, sw = (np.linalg.svd(x, compute_uv=False)
                      for x in (Nf, Nr, Nw))
        # full-matrix forms (Cartesian E basis): plain transpose and the
        # sign-flipped variants; reported, the SV gate is the bar
        Dm = np.diag([1.0, -1.0])
        full = {"T": float(np.abs(Nr - Nf.T).max()),
                "DTD": float(np.abs(Nr - Dm @ Nf.T @ Dm).max()),
                "-T": float(np.abs(Nr + Nf.T).max()),
                "-DTD": float(np.abs(Nr + Dm @ Nf.T @ Dm).max())}
        key = f"th{th_d:g}_ph{ph_d:g}"
        out[key] = dict(closure=rf["closure"], closure_rev=rr["closure"],
                        recip_sv=float(np.max(np.abs(sf - sr))),
                        sv_f=sf, sv_r=sr,
                        wrong_pair=float(np.max(np.abs(sf - sw))),
                        full=full, Nf=Nf, Nr=Nr,
                        rev_deg=[float(np.rad2deg(tr)), float(np.rad2deg(pr))],
                        n=rf["n"] + rr["n"], warnings=rf["warnings"]
                        + rr["warnings"], wall=rf["wall"] + rr["wall"])
        print(key, M, "noswap" if noswap else "", {
            k2: v for k2, v in out[key].items()
            if k2 in ("closure", "closure_rev", "recip_sv", "wrong_pair",
                      "full", "n", "wall")}, flush=True)
    _core.PMM2D_MORTAR_H_SWAP = True
    dump(f"v3g_e24_recip_M{M}{'_noswap' if noswap else ''}", out)

elif mode == "awk":
    name, M = sys.argv[2], int(sys.argv[3])
    force = len(sys.argv) > 4 and sys.argv[4] == "force"
    s1, s2 = awkward()[name]
    try:
        st = stack([(0.25, s1, 1.0), (0.2, s2, 1.0)], M)
        out["refusal"] = (st._merge_refusal or "")[:100]
        out["fast"] = st._perlayer_fast_ok()
        if force:
            st._e2_per_layer_maps = True
        r = run(st)
        out.update(strip(r))
        if out["fast"] or force:
            # reference: the merged map (when it exists) or a finer rung
            pass
        print(name, M, force, r["closure"], r["n"], r["change"], r["wall"],
              r["warnings"])
    except Exception as ex:  # noqa: BLE001 -- recorded
        import traceback
        out["exception"] = f"{type(ex).__name__}: {ex}"
        out["tb"] = traceback.format_exc()[-1500:]
        print(name, M, "EXCEPTION", out["exception"])
    dump(f"v3g_e24_awk_{name}_M{M}{'_force' if force else ''}", out)
out["total_wall"] = time.perf_counter() - t00

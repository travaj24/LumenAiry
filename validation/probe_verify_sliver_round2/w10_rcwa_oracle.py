"""W10 -- the RESONANT false refusals, adjudicated by an INDEPENDENT oracle.

`w6_resonant.py` found rows on a guided-mode-resonance grating that the
shipped guard REFUSES although the answer tracks the device's own physical
wall shift.  "Tracks the physical shift" is a within-family statistic, so this
probe adjudicates the same rows against a CROSS-PACKAGE oracle instead:
``RCWAStack`` with ANALYTIC shape form factors, which has no union grid, no
spectral-element Jacobian, and no wall-collision pathology of any kind -- the
sliver simply does not exist for it.  Its own convergence is demonstrated by
an ``n_orders`` ladder on the same geometry.

For each candidate row it reports, on the orders the two packages share:

    |PMM(delta) - RCWA(delta)|      the error of the answer the guard REFUSES
    |PMM_snapped   - RCWA(delta)|   the error of the min_feature the refusal
                                    PRESCRIBES as remedy (1)
    the RCWA order ladder           the oracle's own convergence

A refusal is a FALSE REFUSAL when the first is small (the answer is right) and
is WORSE when the second is larger than the first (the prescribed remedy moves
the answer AWAY from the truth).

    python w10_rcwa_oracle.py [out.json]
"""
import json
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from w6_resonant import gmr  # noqa: E402
from w_fixtures import prescribed, snapped, unguarded  # noqa: E402

import lumenairy  # noqa: E402
from lumenairy.elements.rcwa import RCWAStack  # noqa: E402

P = 1.0e-6
DUTY = 0.5
E_HI, E_LO = 4.0, 3.6
T_GR, T_SLAB = 0.30e-6, 0.10e-6


def rcwa(d, n_orders, *, nl=2, wl=1.7490e-6, th=0.10):
    """The SAME device, solved by RCWA with analytic rectangle form factors --
    no lattice, no union grid, no sliver."""
    st = RCWAStack(P, n_superstrate=1.0, n_substrate=1.45, n_orders=n_orders)
    a = 0.5 - DUTY / 2.0
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        w = DUTY + 2 * dd
        cx = (a - dd + w / 2.0) * P
        st.add_layer(T_GR / nl, eps_background=E_LO,
                     shapes=[{"shape": "rectangle", "eps": E_HI,
                              "size": (w * P, P), "center": (cx, 0.5 * P)}])
    st.add_layer(T_SLAB, eps=E_HI)
    st.set_source(wl, theta=th)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = st.solve()
    o, R, T = res.efficiencies()
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    R, T = np.real(np.asarray(R)), np.real(np.asarray(T))
    if R.ndim == 1:
        R, T = R[None, :], T[None, :]
    return dict(o=o[i], R=R[:, i], T=T[:, i],
                worst=float(np.max(R.sum(-1) + T.sum(-1))))


def cross(a, b, *, pol=None):
    """Max |dR|, |dT| over the orders two packages share -- matched by ORDER
    NUMBER, not by position, so the two order sets need not be the same."""
    c = np.intersect1d(a["o"], b["o"])
    ia, ib = np.searchsorted(a["o"], c), np.searchsorted(b["o"], c)
    pa = slice(None) if pol is None else slice(pol, pol + 1)
    ka = min(a["R"].shape[0], b["R"].shape[0])
    return float(max(
        np.abs(a["R"][:ka, ia][pa] - b["R"][:ka, ib][pa]).max(),
        np.abs(a["T"][:ka, ia][pa] - b["T"][:ka, ib][pa]).max()))


def guarded_verdict(d, deg, kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            gmr(d, deg, **kw).solve()
        except ValueError as exc:
            return ("REFUSED"
                    if "NEAR-COINCIDENT-WALL SLIVER" in str(exc) else "raised")
    return "returned"


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "w10_rcwa_oracle.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib)
    src = json.load(open(os.path.join(HERE, "w6_resonant.json")))
    cand = [r for r in src["rows"]
            if r["verdict"] == "REFUSED" and r["err_over_slope_d"] <= 3.0
            and r["slope_stationary"]]
    cand.sort(key=lambda r: r["err_over_slope_d"])
    # keep the tags this probe can rebuild: a plain wavelength dial, or nl
    def kw_of(tag):
        if tag.startswith("wl"):
            return dict(wl=float(tag[2:]))
        if tag.startswith("nl"):
            return dict(wl=1.7490e-6, nl=int(tag[2:]))
        return None

    rows = []
    for r in cand[:14]:
        kw = kw_of(r["tag"])
        if kw is None:
            continue
        deg, d = r["deg"], r["delta"]
        nl = kw.get("nl", 2)
        wl = kw["wl"]
        # the ORACLE's own convergence, on the same geometry
        ladder = {}
        prev = None
        for no in (31, 41, 51, 61):
            g = rcwa(d, no, nl=nl, wl=wl)
            ladder[no] = dict(worst=g["worst"],
                              step=(cross(g, prev) if prev is not None
                                    else None))
            prev = g
        O = prev
        A = unguarded(gmr(d, deg, **kw))
        pre = prescribed(gmr(d, deg, **kw))
        B = snapped(gmr(d, deg, **kw), pre["mf"])
        # and the SNAPPED geometry's own RCWA truth (the snap really does move
        # the walls, so remedy (1) is a different device -- scored against
        # BOTH: the device the caller asked for, and its own snapped twin)
        Osnap = rcwa(d / 2.0, 61, nl=nl, wl=wl)
        rows.append(dict(
            tag=r["tag"], deg=deg, delta=d, nl=nl, wl=wl,
            slope=r["slope"], err_over_slope_d=r["err_over_slope_d"],
            move_ratio=r["move_ratio"], su_snap=r["su_snap"],
            worst_pmm=A["worst"], worst_snapped=B["worst"],
            worst_rcwa=O["worst"],
            oracle_ladder={str(k): v for k, v in ladder.items()},
            err_refused_answer=cross(A, O),
            err_prescribed_remedy=cross(B, O),
            err_remedy_vs_its_own_device=cross(B, Osnap),
            verdict=guarded_verdict(d, deg, kw)))
        rr = rows[-1]
        print(f"  {r['tag']:22s} deg {deg:2d} d={d:.4e} slope={r['slope']:.1f}"
              f" move={r['move_ratio']:.1f} -> {rr['verdict']}")
        print(f"      oracle converged to "
              f"{ladder[61]['step']:.2e} at n_orders 51->61; "
              f"R+T-1(rcwa) = {O['worst'] - 1:+.2e}")
        print(f"      |PMM      - oracle| = {rr['err_refused_answer']:.4e}"
              f"   (the REFUSED answer)")
        print(f"      |snapped  - oracle| = "
              f"{rr['err_prescribed_remedy']:.4e}   (remedy (1))")
        print(f"      |snapped  - oracle(snapped device)| = "
              f"{rr['err_remedy_vs_its_own_device']:.4e}")
    n_false = sum(1 for r in rows
                  if r["err_refused_answer"] < r["err_prescribed_remedy"])
    summary = dict(lumenairy=lib, python=sys.version.split()[0],
                   numpy=np.__version__, n_candidates=len(rows),
                   n_answer_better_than_remedy=n_false,
                   best_refused_error=min((r["err_refused_answer"]
                                           for r in rows), default=None))
    with open(out_path, "w") as fh:
        json.dump(dict(summary=summary, rows=rows), fh, indent=1)
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

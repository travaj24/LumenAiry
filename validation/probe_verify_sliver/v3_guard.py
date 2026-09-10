"""VERIFY task 3 -- THE GUARD, re-measured and attacked from both sides.

A. DENSE POPULATIONS.  120 log-spaced ``delta`` of my own x degrees 10/14/20 on
   the O-11 fixture, snap disabled.  Classification is the structure's own
   CONTINUITY (err <= 10 delta = correct, err > 100 delta = wrong), the same
   fitted-constant-free rule the fix used, so the two populations are mine.
   Reports max |R+T-1| among CORRECT and min (R+T-1) among WRONG, plus the
   own-scale attribution ratio on every row.

B. FALSE NEGATIVES.  A finer walk (60 log points, 1e-6..3e-5) at degrees
   8/10/12/14/16 -- the decade the fix's grey row sits in -- looking for rows
   the continuity calls WRONG that the guard does NOT refuse, and mapping how
   wide that quiet band is and how wrong the answer is inside it.

C. FALSE POSITIVES.  A battery of solves that ought NOT to be refused:
   an ordinary non-conforming stack, a legitimate 1 nm liner, a lossy
   substrate, a gain layer, an absorbing superstrate, an off-diagonal tensor,
   a CONICAL mount, a SLANTED layer, and a many-slice taper whose super-unity
   comes from the documented quasi-resonance rather than from a sliver.

D. THE M2 AUDIT-CLASS TAPER.  ``_perlayer_window_grids``' own docstring records
   a 0.4127 nm cross-layer cell whose degree ladder collapses to a WRONG answer
   while ``|R+T-1|`` stays at 1e-8 -- i.e. a sliver-family wrong answer the
   guard's conjunct (b) structurally cannot see.  Re-measured here, post
   ``_forward_growth_flip``.

    python validation/probe_verify_sliver/v3_guard.py [out.json] [sections]
"""
import json
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import lumenairy
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as ps

HERE = os.path.dirname(os.path.abspath(__file__))

OP, OWL, OTH = 1.2e-6, 0.85e-6, 0.15
OEH, OEP = 2.25, 9.0
OA, OB = 0.27865, 0.62505
ODZ = 0.32e-6 / 4
NO_SNAP = OP * 1e-12


def _segs(a, b, eh=OEH, ep=OEP):
    return [(a, eh), (b - a, ep), (1.0 - b, eh)]


def _build(d, deg, mf=NO_SNAP):
    st = PMMStack(OP, n_superstrate=1.0, n_substrate=1.0, degree=deg,
                  min_feature=mf)
    for (a, b) in [(OA, OB), (OA - d, OB + d)]:
        st.add_layer(ODZ, segments=_segs(a, b))
    st.set_source(OWL, theta=OTH)
    return st


def _solve_raw(d, deg, mf=NO_SNAP):
    """Unguarded solve: (orders, R1, T1, worst) with the guard OFF."""
    st = _build(d, deg, mf)
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
    return o[i], np.asarray(R)[1][i], np.asarray(T)[1][i], float(np.max(tot)), st


def _predict_refusal(st, worst):
    """EXACTLY what ``_warn_stack_energy`` does, without re-solving."""
    if not (worst > 1.0 + ps._STACK_SUPERUNITY_BAR):
        return None
    return ps._sliver_refusal(st, worst)


def _err(a, b):
    return float(max(np.abs(a[1] - b[1]).max(), np.abs(a[2] - b[2]).max()))


def _ratio(d, mf=NO_SNAP):
    segs = [_segs(OA, OB), _segs(OA - d, OB + d)]
    hit = ps._cross_layer_sliver(segs, mf / OP)
    if hit is None:
        return None
    w, _xl, _xr, _ww, own, n = hit
    return dict(w=w, own=own, ratio=own / w, n=n)


# ==========================================================================
def section_a(degrees=(10, 14, 20), n=120):
    deltas = np.geomspace(3e-3, 1e-6, n)
    rows = []
    t0 = time.time()
    for deg in degrees:
        ref = _solve_raw(0.0, deg)
        for d in deltas:
            d = float(d)
            res = _solve_raw(d, deg)
            e = _err(res, ref)
            kind = ("wrong" if e > 100.0 * d else
                    "right" if e <= 10.0 * d else "grey")
            msg = _predict_refusal(res[4], res[3])
            rows.append(dict(degree=deg, delta=d, err=e,
                             err_over_delta=e / d if d > 0 else 0.0,
                             RplusT=res[3], kind=kind,
                             refused=msg is not None,
                             ratio=(_ratio(d) or {}).get("ratio")))
        print(f"  degree {deg} done  ({time.time()-t0:.0f} s)", flush=True)
    right = [abs(r["RplusT"] - 1.0) for r in rows if r["kind"] == "right"]
    wrong = [r["RplusT"] - 1.0 for r in rows if r["kind"] == "wrong"]
    grey = [r for r in rows if r["kind"] == "grey"]
    summ = dict(
        n_rows=len(rows), n_right=len(right), n_wrong=len(wrong),
        n_grey=len(grey),
        max_absRT1_right=max(right) if right else None,
        min_RT1_wrong=min(wrong) if wrong else None,
        max_err_over_delta_right=max(r["err_over_delta"] for r in rows
                                     if r["kind"] == "right"),
        refused_wrong=sum(1 for r in rows if r["kind"] == "wrong" and r["refused"]),
        refused_right=sum(1 for r in rows if r["kind"] == "right" and r["refused"]),
        refused_grey=sum(1 for r in grey if r["refused"]),
        widest_wrong_ratio=max((r["ratio"] for r in rows
                                if r["kind"] == "wrong" and r["ratio"]),
                               default=None),
        grey_rows=[dict(degree=g["degree"], delta=g["delta"], err=g["err"],
                        eod=g["err_over_delta"], RT=g["RplusT"],
                        refused=g["refused"]) for g in grey],
    )
    print(f"  rows {summ['n_rows']}  right {summ['n_right']} "
          f"wrong {summ['n_wrong']} grey {summ['n_grey']}")
    print(f"  max |R+T-1| among CORRECT = {summ['max_absRT1_right']:.4e}")
    print(f"  min  (R+T-1) among WRONG  = {summ['min_RT1_wrong']:.4e}")
    print(f"  refused: wrong {summ['refused_wrong']}/{summ['n_wrong']}, "
          f"right {summ['refused_right']}/{summ['n_right']}, "
          f"grey {summ['refused_grey']}/{summ['n_grey']}")
    print(f"  widest WRONG own-scale ratio = {summ['widest_wrong_ratio']}")
    for g in summ["grey_rows"]:
        print(f"    GREY deg {g['degree']} delta {g['delta']:.3e} "
              f"err {g['err']:.3e} ({g['eod']:.1f}x) R+T {g['RT']:.6g} "
              f"refused={g['refused']}")
    return dict(rows=rows, summary=summ)


# ==========================================================================
def section_b(degrees=(8, 10, 12, 14, 16), n=60):
    """FALSE-NEGATIVE hunt in the quiet decade."""
    deltas = np.geomspace(3e-5, 1e-6, n)
    rows, misses = [], []
    t0 = time.time()
    for deg in degrees:
        ref = _solve_raw(0.0, deg)
        for d in deltas:
            d = float(d)
            res = _solve_raw(d, deg)
            e = _err(res, ref)
            kind = ("wrong" if e > 100.0 * d else
                    "right" if e <= 10.0 * d else "grey")
            msg = _predict_refusal(res[4], res[3])
            row = dict(degree=deg, delta=d, err=e, err_over_delta=e / d,
                       RplusT=res[3], kind=kind, refused=msg is not None)
            rows.append(row)
            if kind == "wrong" and not row["refused"]:
                misses.append(row)
            elif kind == "right" and row["refused"]:
                misses.append(dict(row, note="FALSE POSITIVE"))
        print(f"  degree {deg} done ({time.time()-t0:.0f} s)", flush=True)
    # how wide is the quiet band, and how wrong inside it
    quiet = [r for r in rows if r["kind"] in ("wrong", "grey")
             and not r["refused"]]
    print(f"  rows {len(rows)}; wrong-and-NOT-refused {len(misses)}; "
          f"quiet (wrong|grey, unrefused) {len(quiet)}")
    for r in sorted(quiet, key=lambda r: -r["err"])[:20]:
        print(f"    QUIET deg {r['degree']:2d} delta {r['delta']:.4e} "
              f"err {r['err']:.4e} ({r['err_over_delta']:.1f}x) "
              f"R+T-1 {r['RplusT']-1.0:+.3e} kind={r['kind']}")
    return dict(rows=rows, misses=misses, n_quiet=len(quiet))


# ==========================================================================
def _try_solve(st, tag):
    """Solve with the guard ARMED; report refusal / warning / totals."""
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = True
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            o, R, T, J = st.solve()
        tot = (np.real(np.asarray(R)).sum(axis=-1)
               + np.real(np.asarray(T)).sum(axis=-1))
        return dict(tag=tag, refused=False, RplusT=[float(x) for x in tot],
                    warned=[str(w.message)[:70] for w in rec][:2])
    except ValueError as exc:
        return dict(tag=tag, refused=True,
                    sliver="NEAR-COINCIDENT-WALL SLIVER" in str(exc),
                    msg=str(exc)[:160])
    finally:
        ps.PMM_SLIVER_GUARD = was


def section_c():
    """FALSE-POSITIVE battery."""
    out = []

    def _liner(walls, eps):
        seg, prev = [], 0.0
        for w, e in zip(list(walls) + [1.0], list(eps)):
            seg.append((w - prev, e))
            prev = w
        return [s for s in seg if s[0] > 0.0]

    # c1: ordinary non-conforming stack, cross-layer cells but no sliver
    st = PMMStack(OP, n_superstrate=1.0, n_substrate=1.0, degree=12,
                  min_feature=NO_SNAP)
    st.add_layer(ODZ, segments=_segs(0.30, 0.50))
    st.add_layer(ODZ, segments=_segs(0.35, 0.55))
    st.set_source(OWL, theta=OTH)
    r = _try_solve(st, "c1_non_conforming")
    r["screen"] = ps._cross_layer_sliver(
        [_segs(0.30, 0.50), _segs(0.35, 0.55)], 1e-12)
    out.append(r)

    # c2: a legitimate 1e-4 liner INSIDE one layer (owned) -- ratio would be
    # 1e4 if the ownership rule were dropped
    segs2 = [_liner([0.30, 0.3001, 0.70], [OEH, OEP, OEH, OEH]),
             _segs(0.30, 0.70)]
    st = PMMStack(OP, n_superstrate=1.0, n_substrate=1.0, degree=12,
                  min_feature=NO_SNAP)
    for s in segs2:
        st.add_layer(ODZ, segments=s)
    st.set_source(OWL, theta=OTH)
    r = _try_solve(st, "c2_owned_liner")
    r["screen"] = ps._cross_layer_sliver(segs2, 1e-12)
    out.append(r)

    # c2b: a legitimate liner in layer 0 SPLIT by a wall of layer 1 that sits
    # 1e-7 away -- the two sub-cells ARE manufactured.  Does the ratio clear
    # 100, and is the answer right?
    segs2b = [_liner([0.30, 0.3010, 0.70], [OEH, OEP, OEH, OEH]),
              _liner([0.30 + 1e-7, 0.70], [OEH, OEP, OEH])]
    st = PMMStack(OP, n_superstrate=1.0, n_substrate=1.0, degree=12,
                  min_feature=NO_SNAP)
    for s in segs2b:
        st.add_layer(ODZ, segments=s)
    st.set_source(OWL, theta=OTH)
    r = _try_solve(st, "c2b_liner_split_by_neighbour")
    r["screen"] = ps._cross_layer_sliver(segs2b, 1e-12)
    out.append(r)

    # c3: the sliver fixture with a LOSSY SUBSTRATE (passive per the helper --
    # is R+T <= 1 still a theorem there?)
    for nsub in (1.5 + 0.05j, 1.5 + 0.5j, 3.0 + 2.0j):
        st = PMMStack(OP, n_superstrate=1.0, n_substrate=nsub, degree=12,
                      min_feature=NO_SNAP)
        st.add_layer(ODZ, segments=_segs(0.35, 0.65))
        st.set_source(OWL, theta=OTH)
        r = _try_solve(st, f"c3_lossy_substrate_{nsub}")
        r["provably_passive"] = bool(ps._stack_provably_passive(st))
        out.append(r)

    # c3b: lossy substrate AND a sliver
    st = _build(1e-4, 14)
    st.n_sub = 1.5 + 0.5j
    st2 = PMMStack(OP, n_superstrate=1.0, n_substrate=1.5 + 0.5j, degree=14,
                   min_feature=NO_SNAP)
    for (a, b) in [(OA, OB), (OA - 1e-4, OB + 1e-4)]:
        st2.add_layer(ODZ, segments=_segs(a, b))
    st2.set_source(OWL, theta=OTH)
    r = _try_solve(st2, "c3b_lossy_substrate_plus_sliver")
    r["provably_passive"] = bool(ps._stack_provably_passive(st2))
    out.append(r)

    # c4: an OFF-DIAGONAL in-plane tensor with the SAME sliver -- not provably
    # passive, so the guard must stay silent (a blind spot by construction)
    e = np.array([[OEP, 0.2, 0.0], [0.2, OEP, 0.0], [0.0, 0.0, OEP]],
                 dtype=complex)
    st = PMMStack(OP, n_superstrate=1.0, n_substrate=1.0, degree=14,
                  min_feature=NO_SNAP)
    for (a, b) in [(OA, OB), (OA - 1e-4, OB + 1e-4)]:
        st.add_layer(ODZ, segments=[(a, OEH), (b - a, e), (1.0 - b, OEH)])
    st.set_source(OWL, theta=OTH)
    r = _try_solve(st, "c4_offdiag_tensor_with_sliver")
    r["provably_passive"] = bool(ps._stack_provably_passive(st))
    out.append(r)

    # c5: CONICAL mount, healthy geometry (no sliver)
    st = PMMStack(OP, n_superstrate=1.0, n_substrate=1.0, degree=12,
                  min_feature=NO_SNAP)
    st.add_layer(ODZ, segments=_segs(0.30, 0.60))
    st.add_layer(ODZ, segments=_segs(0.35, 0.65))
    st.set_source(OWL, theta=OTH, phi=0.5)
    out.append(_try_solve(st, "c5_conical_healthy"))

    # c5b: CONICAL mount WITH the sliver
    st = PMMStack(OP, n_superstrate=1.0, n_substrate=1.0, degree=14,
                  min_feature=NO_SNAP)
    for (a, b) in [(OA, OB), (OA - 1e-4, OB + 1e-4)]:
        st.add_layer(ODZ, segments=_segs(a, b))
    st.set_source(OWL, theta=OTH, phi=0.5)
    out.append(_try_solve(st, "c5b_conical_with_sliver"))

    # c6: SLANTED layer, healthy
    st = PMMStack(OP, n_superstrate=1.0, n_substrate=1.0, degree=10,
                  min_feature=NO_SNAP)
    st.add_layer(ODZ, segments=_segs(0.30, 0.60), slant_angle=0.15)
    st.add_layer(ODZ, segments=_segs(0.35, 0.65))
    st.set_source(OWL, theta=OTH)
    out.append(_try_solve(st, "c6_slant_healthy"))

    # c6b: SLANTED layer WITH the sliver
    st = PMMStack(OP, n_superstrate=1.0, n_substrate=1.0, degree=12,
                  min_feature=NO_SNAP)
    st.add_layer(ODZ, segments=_segs(OA, OB), slant_angle=0.10)
    st.add_layer(ODZ, segments=_segs(OA - 1e-4, OB + 1e-4))
    st.set_source(OWL, theta=OTH)
    out.append(_try_solve(st, "c6b_slant_with_sliver"))

    # c7: a MANY-SLICE taper at LOW degree -- super-unity from the documented
    # quasi-resonance.  Walls collide, so conjunct (a) may also be true.
    for ns, deg in ((16, 6), (24, 6), (32, 6), (24, 8), (40, 6)):
        st = PMMStack(OP, n_superstrate=1.0, n_substrate=1.0, degree=deg,
                      min_feature=NO_SNAP)
        for k in range(ns):
            zeta = (k + 0.5) / ns
            a = 0.20 + 0.10 * zeta
            b = 0.80 - 0.10 * zeta
            st.add_layer(0.4e-6 / ns, segments=_segs(a, b))
        st.set_source(OWL, theta=OTH)
        r = _try_solve(st, f"c7_taper_ns{ns}_deg{deg}")
        segs = []
        for k in range(ns):
            zeta = (k + 0.5) / ns
            segs.append(_segs(0.20 + 0.10 * zeta, 0.80 - 0.10 * zeta))
        hit = ps._cross_layer_sliver(segs, 1e-12)
        r["screen"] = None if hit is None else dict(
            w=hit[0], own=hit[4], ratio=hit[4] / hit[0], n=hit[5])
        out.append(r)

    for r in out:
        print(f"  {r['tag']:38s} refused={r['refused']} "
              f"{('sliver=' + str(r.get('sliver'))) if r['refused'] else ('R+T=' + str([round(x, 6) for x in r['RplusT']]))}")
        if r.get("screen") is not None:
            print(f"      screen: {r['screen']}")
    return out


# ==========================================================================
# D -- the M2 audit-class coated taper: is the energy-BLIND wrong answer live?
# ==========================================================================
NM = 1e-9
M2_PERIOD, M2_WL = 700 * NM, 1310 * NM
M2_SIDE, M2_H1, M2_COAT, M2_WTOP = np.deg2rad(2.0), 310 * NM, 5.0 * NM, 340 * NM
M2_CORE, M2_CO, M2_GR = (3.48 + 0j) ** 2, (1.76 + 0j) ** 2, 1.0 + 0j


def _m2_segments(w_core, coat=M2_COAT):
    w_out = 0.5 * (w_core + 2.0 * coat) / M2_PERIOD
    w_in = 0.5 * w_core / M2_PERIOD
    g = 0.5 - w_out
    c = w_out - w_in
    return [(g, M2_GR), (c, M2_CO), (2 * w_in, M2_CORE), (c, M2_CO), (g, M2_GR)]


def _m2_layers(ns, coat=M2_COAT):
    dz = M2_H1 / ns
    out = []
    for k in range(ns):
        zeta = (k + 0.5) / ns
        a = 0.5 * M2_WTOP - zeta * M2_H1 * np.tan(M2_SIDE)
        out.append((dz, _m2_segments(2.0 * a, coat)))
    return out


def section_d(ns_list=(2, 3, 6), degrees=(6, 8, 10, 12, 14, 16, 18)):
    rows = []
    for ns in ns_list:
        layers = _m2_layers(ns)
        segs = [s for _t, s in layers]
        hit = ps._cross_layer_sliver(segs, 1e-5)      # LIBRARY DEFAULT
        screen_def = None if hit is None else dict(
            w_m=hit[0] * M2_PERIOD, own=hit[4], ratio=hit[4] / hit[0], n=hit[5])
        for mf_tag, mf in (("default", None), ("0.5nm", 0.5 * NM)):
            ladder = []
            for deg in degrees:
                kw = {} if mf is None else dict(min_feature=mf)
                st = PMMStack(M2_PERIOD, n_substrate=1.50, n_superstrate=1.50,
                              degree=deg, far_field_orders=11, **kw)
                for t, s in layers:
                    st.add_layer(t, segments=s)
                st.set_source(M2_WL, theta=np.deg2rad(8.0))
                was = ps.PMM_SLIVER_GUARD
                ps.PMM_SLIVER_GUARD = True
                try:
                    with warnings.catch_warnings(record=True) as rec:
                        warnings.simplefilter("always")
                        o, R, T, _J = st.solve()
                    o = np.asarray(o)
                    m0 = int(np.where(o == 0)[0][0])
                    tot = (np.real(np.asarray(R)).sum(axis=-1)
                           + np.real(np.asarray(T)).sum(axis=-1))
                    ladder.append(dict(degree=deg,
                                       R0=float(np.real(np.asarray(R))[0, m0]),
                                       closure=float(np.max(np.abs(tot - 1.0))),
                                       refused=False,
                                       warned=any("energy not conserved" in
                                                  str(w.message) for w in rec)))
                except ValueError as exc:
                    ladder.append(dict(degree=deg, refused=True,
                                       sliver="SLIVER" in str(exc)))
                finally:
                    ps.PMM_SLIVER_GUARD = was
            good = [r for r in ladder if not r.get("refused")]
            spread = (max(r["R0"] for r in good) - min(r["R0"] for r in good)
                      if good else None)
            rows.append(dict(ns=ns, min_feature=mf_tag, screen_default=screen_def,
                             ladder=ladder, R0_spread=spread,
                             max_closure=max((r["closure"] for r in good),
                                             default=None),
                             any_refused=any(r.get("refused") for r in ladder)))
            print(f"  ns={ns} mf={mf_tag:8s} R0 " +
                  " ".join(f"{r['R0']:.6f}" if not r.get("refused") else "REFUSED"
                           for r in ladder))
            print(f"      spread={spread}  max|R+T-1|={rows[-1]['max_closure']}"
                  f"  screen(default)={screen_def}")
    return rows



# ==========================================================================
# A2 -- the fix's OWN grid (46 log points 3e-3..3e-6, degrees 12/14/20),
# re-run with my script, to separate "the number is wrong" from "the number
# is a property of a 46-point grid".
# ==========================================================================
def section_a2():
    deltas = np.geomspace(3e-3, 3e-6, 46)
    right, wrong, grey, rows = [], [], [], []
    for deg in (12, 14, 20):
        ref = _solve_raw(0.0, deg)
        for d in deltas:
            d = float(d)
            res = _solve_raw(d, deg)
            e = _err(res, ref)
            kind = ("wrong" if e > 100.0 * d else
                    "right" if e <= 10.0 * d else "grey")
            rows.append(dict(degree=deg, delta=d, err=e, RplusT=res[3],
                             err_over_delta=e / d, kind=kind))
            (right if kind == "right" else wrong if kind == "wrong"
             else grey).append(rows[-1])
    summ = dict(n_rows=len(rows), n_right=len(right), n_wrong=len(wrong),
                n_grey=len(grey),
                max_absRT1_right=max(abs(r["RplusT"] - 1.0) for r in right),
                min_RT1_wrong=min(r["RplusT"] - 1.0 for r in wrong),
                max_eod_right=max(r["err_over_delta"] for r in right),
                grey=[dict(degree=g["degree"], delta=g["delta"], err=g["err"],
                           eod=g["err_over_delta"], RT=g["RplusT"])
                      for g in grey])
    print(f"  rows {summ['n_rows']}  right {summ['n_right']} "
          f"wrong {summ['n_wrong']} grey {summ['n_grey']}")
    print(f"  max |R+T-1| among CORRECT = {summ['max_absRT1_right']:.4e}")
    print(f"  min  (R+T-1) among WRONG  = {summ['min_RT1_wrong']:.4e}")
    print(f"  max err/delta among CORRECT = {summ['max_eod_right']:.4f}")
    for g in summ["grey"]:
        print(f"    GREY deg {g['degree']} delta {g['delta']:.3e} "
              f"err {g['err']:.3e} ({g['eod']:.1f}x) R+T {g['RT']:.6g}")
    return dict(rows=rows, summary=summ)


# ==========================================================================
# E -- CONFIRM each false negative END TO END, with independent corroboration
# ==========================================================================
def _solve_guarded(d, deg, mf=NO_SNAP):
    st = _build(d, deg, mf)
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = True
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            o, R, T, _J = st.solve()
        o = np.asarray(o).ravel()
        i = np.argsort(o)
        tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
        return dict(refused=False, R=np.asarray(R)[1][i],
                    T=np.asarray(T)[1][i], worst=float(np.max(tot)),
                    warned=any("energy not conserved" in str(w.message)
                               for w in rec))
    except ValueError as exc:
        return dict(refused=True, msg=str(exc)[:120])
    finally:
        ps.PMM_SLIVER_GUARD = was


def section_e(cands):
    """``cands`` = [(degree, delta), ...] rows section B called wrong-and-
    unrefused.  Each is re-solved END TO END with the guard armed, scored
    against the exact ``delta -> 0`` limit, against the SNAPPED answer the
    refusal would have prescribed, and across a degree ladder."""
    out = []
    for deg, d in cands:
        ref = _solve_raw(0.0, deg)
        g = _solve_guarded(d, deg)
        raw = _solve_raw(d, deg)
        err = _err(raw, ref)
        # the remedy the refusal WOULD have prescribed, had it fired
        hit = ps._cross_layer_sliver([_segs(OA, OB), _segs(OA - d, OB + d)],
                                     NO_SNAP / OP)
        mf_fix = 2.0 * hit[3] * OP if hit else None
        snapped = _solve_raw(d, deg, mf=mf_fix) if mf_fix else None
        err_snap = _err(snapped, ref) if snapped else None
        ladder = []
        for dd in (deg - 2, deg - 1, deg, deg + 1, deg + 2):
            if dd < 4:
                continue
            r = _solve_raw(d, dd)
            ladder.append(dict(degree=dd, R0=float(r[1][np.argmin(np.abs(r[0]))]),
                               RplusT=r[3]))
        out.append(dict(degree=deg, delta=d, err_vs_limit=err,
                        err_over_delta=err / d, RplusT=raw[3],
                        returned_not_refused=(not g["refused"]),
                        energy_warned=g.get("warned"),
                        prescribed_min_feature=mf_fix,
                        err_snapped=err_snap,
                        err_snapped_over_delta=(err_snap / d if err_snap
                                                else None),
                        degree_ladder=ladder))
        print(f"  deg {deg} delta {d:.4e}: err {err:.4e} ({err/d:.0f}x) "
              f"R+T-1 {raw[3]-1.0:+.4e} returned={not g['refused']} "
              f"warned={g.get('warned')} | snapped err {err_snap:.4e} "
              f"({err_snap/d:.2f}x)")
        print("      degree ladder R(order0): " +
              " ".join(f"{r['degree']}:{r['R0']:.6f}" for r in ladder))
    return out



# ==========================================================================
# F -- HOW WIDE is the quiet band?  A fine LINEAR walk of delta around one
# unrefused wrong row, so the band is measured in delta rather than counted
# in log samples.
# ==========================================================================
def section_f(centres=((14, 1.5859431948991427e-06),
                       (20, 1.7130e-06), (12, 2.2413e-06)), n=81, span=0.25):
    out = []
    for deg, d0 in centres:
        ref = _solve_raw(0.0, deg)
        rows = []
        for d in np.linspace(d0 * (1 - span), d0 * (1 + span), n):
            d = float(d)
            res = _solve_raw(d, deg)
            e = _err(res, ref)
            kind = ("wrong" if e > 100.0 * d else
                    "right" if e <= 10.0 * d else "grey")
            rows.append(dict(delta=d, err=e, err_over_delta=e / d,
                             RplusT=res[3], kind=kind,
                             refused=_predict_refusal(res[4], res[3]) is not None))
        quiet = [r for r in rows if r["kind"] != "right" and not r["refused"]]
        # contiguous runs of quiet rows
        runs, cur = [], []
        for r in rows:
            if r in quiet:
                cur.append(r)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        widths = [(rr[-1]["delta"] - rr[0]["delta"]) for rr in runs]
        out.append(dict(degree=deg, centre=d0, n=n, span=span,
                        n_quiet=len(quiet), n_wrong=sum(r["kind"] == "wrong"
                                                        for r in rows),
                        n_refused=sum(r["refused"] for r in rows),
                        n_right=sum(r["kind"] == "right" for r in rows),
                        runs=len(runs), run_widths=widths,
                        max_err_quiet=max((r["err"] for r in quiet),
                                          default=0.0),
                        max_RT1_quiet=max((r["RplusT"] - 1.0 for r in quiet),
                                          default=0.0),
                        rows=rows))
        step = (rows[1]["delta"] - rows[0]["delta"])
        print(f"  deg {deg} centre {d0:.6e}: right {out[-1]['n_right']} "
              f"refused {out[-1]['n_refused']} quiet {out[-1]['n_quiet']} "
              f"in {n} samples (step {step:.3e})")
        print(f"      contiguous quiet runs {len(runs)}, widths "
              f"{['%.3e' % w for w in widths]}, max err {out[-1]['max_err_quiet']:.4e}, "
              f"max R+T-1 {out[-1]['max_RT1_quiet']:+.4e}")
    return out


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "v3_guard.json"))
    want = (sys.argv[2].split(",") if len(sys.argv) > 2
            else ["a", "b", "c", "d"])
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    assert "lum_vsliver" in lib.replace("\\", "/"), lib
    res = dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                         numpy=np.__version__,
                         bar_b=ps._STACK_SUPERUNITY_BAR,
                         bar_a=ps._SLIVER_OWN_SCALE_RATIO))
    if "a" in want:
        print("\n== A: dense populations (120 deltas x 3 degrees) ==")
        res["a"] = section_a()
    if "b" in want:
        print("\n== B: false-negative hunt (60 deltas x 5 degrees) ==")
        res["b"] = section_b()
    if "c" in want:
        print("\n== C: false-positive battery ==")
        res["c"] = section_c()
    if "d" in want:
        print("\n== D: the M2 audit-class coated taper ==")
        res["d"] = section_d()
    if "a2" in want:
        print("\n== A2: the fix's own 46-point grid, degrees 12/14/20 ==")
        res["a2"] = section_a2()
    if "e" in want:
        print("\n== E: the false negatives, end to end ==")
        # EXACT floats from the A/B run -- the pathology moves with the last
        # bits of delta, so a printed 5-digit copy is a DIFFERENT geometry
        # (measured: the 5-digit copy of a quiet row reads R+T-1 = +1.17 where
        # the exact row reads +6.9e-03).
        src = os.environ.get("LUMV_AB_JSON",
                             os.path.join(HERE, "v3_guard_ab.json"))
        prev = json.load(open(src))
        cands = [(r["degree"], r["delta"]) for r in prev["b"]["misses"]]
        cands += [(r["degree"], r["delta"]) for r in prev["b"]["rows"]
                  if r["kind"] in ("wrong", "grey") and not r["refused"]]
        cands += [(g["degree"], g["delta"]) for g in
                  prev["a"]["summary"]["grey_rows"] if not g["refused"]]
        cands += [(r["degree"], r["delta"]) for r in prev["a"]["rows"]
                  if r["kind"] == "wrong" and not r["refused"]]
        cands = sorted(set(cands))
        print(f"  {len(cands)} candidate rows from v3_guard_ab.json")
        res["e"] = section_e(cands)
    if "f" in want:
        print("\n== F: the quiet band's WIDTH in delta ==")
        res["f"] = section_f()
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()

"""TASK D -- the SEM manufactured-element contract, re-measured and attacked.

TERMS.  ``BORStack._solve_sem`` meshes layer ``i`` on the UNION of the ring
walls of layers ``i-1``, ``i`` and ``i+1`` (the "enrichment window").  Two
neighbouring layers whose walls differ by ``delta`` therefore manufacture a
radial element of width ``delta`` in BOTH meshes, which no single layer asked
for.  5.45.1's contract REFUSES the solve when the post-window, post-DPW,
post-``equalize_meshes`` breakpoint set contains such a cross-layer cell
narrower than ``1e-6 * Rbig`` AND that layer's spectrum reads
``|q|max / (n_max k0) > 1e4``; it WARNS (never refuses) in the geometric band
up to ``1e-4 * Rbig``.

WHAT THIS PROBE IS FOR.  The build's own census gives the Q bar only 0.95
decades of room above ordinary geometry, so the two things to try to break are

  * a FALSE REFUSAL -- an ordinary geometry whose cross-layer cell falls below
    ``1e-6 * Rbig`` and whose spectrum is hot, but whose ANSWER is right; and
  * a MISS -- a geometry that is merely warned (or passed) whose answer is
    already wrong.

Both are decided against an EXACT ``delta -> 0`` reference on the same
physical stack, never against an energy closure (the build measured the
closure spreading 1,129x with the BLAS kernel on one rung).

Usage:  python v4_sem.py <pre|post> <part> [outdir]
        part in {census, ladder, hunt, all}
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import _vh  # noqa: E402


def _guard(state):
    """Arm / disarm the shipped contract; a no-op on the PRE tree."""
    try:
        from lumenairy.elements.bor import _sem_contract as C
    except ImportError:
        return None
    prev = C.BOR_SEM_MESH_GUARD
    C.BOR_SEM_MESH_GUARD = state
    return prev


def _report(stack):
    """The shipped per-layer mesh measurement, when the tree has one."""
    rep = getattr(stack, "_sem_mesh_report", None)
    if rep is None:
        return None
    try:
        from lumenairy.elements.bor._sem_contract import verdict
    except ImportError:
        verdict = None
    out = []
    for r in rep:
        d = {k: (float(v) if isinstance(v, (int, float, np.floating))
                 else v) for k, v in r.items() if k != "attribution"}
        d["attribution"] = r.get("attribution")
        if verdict is not None:
            d["verdict"] = verdict(r)
        out.append(d)
    return out


def _solve(stack, arm_guard=True):
    """Solve, recording warnings and the mesh report, with the contract armed
    or disarmed.  A refusal is an answer and is returned as one."""
    prev = _guard(arm_guard)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                res = stack.solve()
                R, T = np.asarray(res["R"]), np.asarray(res["T"])
                out = dict(raised=None, n_channels=int(R.size),
                           hash=_vh.hash_arrays(R, T),
                           R=R.tolist(), T=T.tolist(),
                           closure=float(np.max(np.abs(R + T - 1.0)))
                           if R.size else float("nan"))
            except BaseException as e:              # noqa: BLE001
                out = dict(raised=type(e).__name__, msg=str(e)[:900])
            out["warnings"] = [str(x.message) for x in w]
            out["n_warnings"] = len(w)
            out["report"] = _report(stack)
        return out
    finally:
        if prev is not None:
            _guard(prev)


def _stack(Rbig, m, degree, k0, N=120, eps_seg=1, grade=True):
    from lumenairy.elements.bor.bor_stack import BORStack
    return BORStack(Rbig=Rbig, m=m, N=N, n_superstrate=1.0, n_substrate=1.0,
                    basis="sem", degree=degree, elements_per_segment=eps_seg,
                    grade=grade)


# --------------------------------------------------------------------------- #
#  1. the ORDINARY census, on MY OWN families                                  #
# --------------------------------------------------------------------------- #
def ordinary_families():
    """Every geometry family the BOR suite builds, plus the ones the build's
    census reached for, plus three it did not: ring counts 1..8, a GENTLE
    taper (small radius change, many slices) and hp-refinement at high
    ``elements_per_segment`` with the graded knob on."""
    F = []

    def add(name, fn):
        F.append((name, fn))

    for k0 in (0.8, 2.0, 3.5, 8.0):
        for m in (0, 1, 2, 5):
            def mk(k0=k0, m=m, deg=8):
                s = _stack(24.0, m, deg, k0)
                s.add_layer(0.6, rings=(4.0, 0.5, 1.9, 1.2))
                s.add_layer(0.4, eps=1.44)
                s.set_source(k0=k0)
                return s
            add("ringgrating_k%g_m%d" % (k0, m), mk)

    for nr in (1, 2, 3, 4, 5, 6, 7, 8):
        def mk(nr=nr):
            s = _stack(24.0, 1, 8, 2.0)
            segs = []
            for j in range(nr):
                segs.append((3.0 + 20.0 * j / nr, 2.25 if j % 2 else 1.44))
            segs.append((24.0, 1.0))
            s.add_layer(0.5, segments=segs)
            s.add_layer(0.5, eps=1.21)
            s.set_source(k0=2.0)
            return s
        add("ringcount_%d" % nr, mk)

    # uniform, coincident walls, wall-free spacer, lossy metal ring, aniso
    def mk_uniform():
        s = _stack(24.0, 1, 8, 2.0)
        s.add_layer(0.5, eps=2.25)
        s.add_layer(0.5, eps=1.0)
        s.set_source(k0=2.0)
        return s
    add("uniform_pair", mk_uniform)

    def mk_coincident():
        s = _stack(24.0, 1, 8, 2.0)
        s.add_layer(0.5, segments=[(9.0, 2.25), (24.0, 1.0)])
        s.add_layer(0.5, segments=[(9.0, 1.44), (24.0, 1.0)])
        s.set_source(k0=2.0)
        return s
    add("coincident_walls", mk_coincident)

    def mk_spacer():
        s = _stack(24.0, 1, 8, 2.0)
        s.add_layer(0.5, segments=[(9.0, 2.25), (24.0, 1.0)])
        s.add_layer(0.8, eps=1.0)
        s.add_layer(0.5, segments=[(11.0, 2.25), (24.0, 1.0)])
        s.set_source(k0=2.0)
        return s
    add("wall_free_spacer", mk_spacer)

    def mk_metal():
        s = _stack(24.0, 1, 8, 2.0)
        s.add_layer(0.4, segments=[(8.0, complex(-20.0, 1.5)), (24.0, 1.0)])
        s.add_layer(0.4, eps=2.25)
        s.set_source(k0=2.0)
        return s
    add("lossy_metal_ring", mk_metal)

    def mk_aniso():
        s = _stack(24.0, 1, 8, 2.0)
        s.add_layer(0.5, segments=[(9.0, (2.5, 2.1, 2.3)),
                                   (24.0, (1.0, 1.0, 1.0))])
        s.add_layer(0.5, eps=1.21)
        s.set_source(k0=2.0)
        return s
    add("anisotropic", mk_aniso)

    def mk_nm():
        Rb = 24.0e-6
        s = _stack(Rb, 1, 8, 2.0e6)
        s.add_layer(0.5e-6, segments=[(9.0e-6, 2.25), (Rb, 1.0)])
        s.add_layer(0.5e-6, eps=1.21)
        s.set_source(k0=2.0e6)
        return s
    add("nm_units", mk_nm)

    # hp refinement -- the GRADED knob makes narrow cells at every interval
    # END, and those ends are walls of DIFFERENT layers in a multi-layer stack
    for eps_seg in (2, 4, 8, 16, 32):
        for grade in (True, False):
            def mk(eps_seg=eps_seg, grade=grade):
                s = _stack(24.0, 1, 8, 2.0, eps_seg=eps_seg, grade=grade)
                s.add_layer(0.5, segments=[(9.0, 2.25), (24.0, 1.0)])
                s.add_layer(0.5, segments=[(11.0, 2.25), (24.0, 1.0)])
                s.set_source(k0=2.0)
                return s
            add("hp_eps%d_grade%d" % (eps_seg, int(grade)), mk)

    return F


def taper(ns, degree, k0, r_top=8.0, r_bot=2.0, Rbig=24.0, height=1.2):
    def mk():
        s = _stack(Rbig, 1, degree, k0)
        for j in range(ns):
            r = r_top + (r_bot - r_top) * (j + 0.5) / ns
            s.add_layer(height / ns, segments=[(r, 4.0), (Rbig, 1.0)])
        s.set_source(k0=k0)
        return s
    return mk


def census(fast=False):
    rows = []
    degrees = (6, 8, 12, 16)
    for name, mk in ordinary_families():
        for deg in degrees:
            try:
                s = mk()
                s.degree = deg
                r = _solve(s, arm_guard=False)
            except BaseException as e:              # noqa: BLE001
                r = dict(raised=type(e).__name__, msg=str(e)[:300])
            rows.append(dict(family=name, degree=deg, kind="family", **r))
            print("   %-28s deg%-3d %s" % (
                name, deg,
                _v(r)), flush=True)
    # the taper staircase, out past the build's 256
    slices = (4, 8, 16, 32, 64, 128, 256) if fast else \
             (4, 8, 16, 32, 64, 128, 256, 512)
    for ns in slices:
        for deg, k0 in ((8, 0.8), (8, 2.0), (12, 2.0)):
            try:
                r = _solve(taper(ns, deg, k0)(), arm_guard=False)
            except BaseException as e:              # noqa: BLE001
                r = dict(raised=type(e).__name__, msg=str(e)[:300])
            rows.append(dict(family="taper", n_slices=ns, degree=deg,
                             k0=k0, kind="taper", **r))
            print("   taper n=%-4d deg%d k0=%g %s" % (ns, deg, k0, _v(r)),
                  flush=True)
    # a GENTLE taper: the SAME slice counts but a small radius change, so the
    # manufactured cell goes below 1e-6 Rbig at an ORDINARY slice count
    for dr, ns in ((0.5, 64), (0.1, 64), (0.02, 64), (0.005, 64),
                   (0.001, 64), (0.0002, 64), (0.00005, 64)):
        try:
            r = _solve(taper(ns, 8, 2.0, r_top=8.0, r_bot=8.0 - dr)(),
                       arm_guard=False)
        except BaseException as e:                  # noqa: BLE001
            r = dict(raised=type(e).__name__, msg=str(e)[:300])
        rows.append(dict(family="gentle_taper", dr=dr, n_slices=ns, degree=8,
                         k0=2.0, kind="gentle_taper", **r))
        print("   gentle taper dr=%-9g %s" % (dr, _v(r)), flush=True)
    return rows


def _v(r):
    rep = r.get("report")
    if r.get("raised"):
        return "RAISED %s" % r["raised"]
    if not rep:
        return "closure %.3g (no report)" % r.get("closure", float("nan"))
    wu = min((x["w_min_union_frac"] for x in rep), default=float("inf"))
    qe = max((x["q_excess"] for x in rep), default=float("nan"))
    vs = sorted({x.get("verdict", "?") for x in rep})
    return "w_union/Rbig %.3e  qexc %.4g  %s  nwarn %d" % (wu, qe, vs,
                                                           r["n_warnings"])


# --------------------------------------------------------------------------- #
#  2. the DELTA ladder, scored against an exact delta -> 0 reference           #
# --------------------------------------------------------------------------- #
def two_layer(delta, degree, Rbig=24.0, k0=2.0, m=1, spacer=0):
    def mk():
        s = _stack(Rbig, m, degree, k0)
        s.add_layer(0.5, segments=[(9.0, 4.0), (Rbig, 1.0)])
        for _ in range(spacer):
            s.add_layer(0.3, eps=1.0)
        s.add_layer(0.5, segments=[(9.0 + delta, 2.25), (Rbig, 1.0)])
        s.set_source(k0=k0)
        return s
    return mk


def ladder(spacer=0, degrees=(6, 8, 12), Rbig=24.0):
    ref = {}
    rows = []
    for deg in degrees:
        r0 = _solve(two_layer(0.0, deg, Rbig, spacer=spacer)(),
                    arm_guard=False)
        ref[deg] = r0
        rows.append(dict(delta_frac=0.0, degree=deg, spacer=spacer, **r0))
    for dfrac in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        for deg in degrees:
            d = dfrac * Rbig
            armed = _solve(two_layer(d, deg, Rbig, spacer=spacer)(),
                           arm_guard=True)
            raw = _solve(two_layer(d, deg, Rbig, spacer=spacer)(),
                         arm_guard=False)
            row = dict(delta_frac=dfrac, degree=deg, spacer=spacer,
                       armed_raised=armed.get("raised"),
                       armed_nwarn=armed.get("n_warnings"),
                       armed_msg=(armed.get("msg") or "")[:300],
                       armed_warnings=armed.get("warnings"),
                       report=raw.get("report"),
                       n_channels=raw.get("n_channels"),
                       closure=raw.get("closure"),
                       hash=raw.get("hash"), R=raw.get("R"))
            r0 = ref[deg]
            if raw.get("R") and r0.get("R") and \
                    len(raw["R"]) == len(r0["R"]):
                a = np.asarray(raw["R"])
                b = np.asarray(r0["R"])
                row["dR_max"] = float(np.max(np.abs(a - b)))
                row["dR_rel"] = float(np.max(np.abs(a - b))
                                      / max(np.max(np.abs(b)), 1e-300))
            else:
                row["dR_max"] = None
                row["channel_count_moved"] = True
            rows.append(row)
            print("   spacer%d delta%.0e deg%-3d armed=%-24s dRmax=%s %s"
                  % (spacer, dfrac, deg,
                     row["armed_raised"] or ("warn%d" % row["armed_nwarn"]),
                     row["dR_max"], _v(raw)), flush=True)
    # the MOVE FACTOR: |dR(delta)| / delta, normalized by the same ratio at a
    # rung everyone agrees is ordinary (1e-2).  A smooth answer gives ~1; the
    # 1-D guard's attribution bar is 100x.
    for deg in degrees:
        base = [r for r in rows if r.get("degree") == deg
                and r.get("delta_frac") == 1e-2 and r.get("dR_max")]
        if not base:
            continue
        s0 = base[0]["dR_max"] / 1e-2
        for r in rows:
            if r.get("degree") == deg and r.get("dR_max") and \
                    r.get("delta_frac"):
                r["move_factor"] = (r["dR_max"] / r["delta_frac"]) / s0
    return rows


def liner_ladder():
    """The WITHIN-LAYER liner: the narrow cell is the CALLER's own, so the
    contract must warn and never refuse."""
    rows = []
    Rbig = 24.0
    for wf in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        for deg in (6, 8, 12, 16):
            w = wf * Rbig

            def mk(w=w, deg=deg):
                s = _stack(Rbig, 1, deg, 2.0)
                s.add_layer(0.5, segments=[(9.0, 1.0), (9.0 + w, 4.0),
                                           (Rbig, 1.0)])
                s.add_layer(0.5, eps=1.21)
                s.set_source(k0=2.0)
                return s
            r = _solve(mk(), arm_guard=True)
            rows.append(dict(w_frac=wf, degree=deg,
                             raised=r.get("raised"),
                             n_warnings=r.get("n_warnings"),
                             warnings=r.get("warnings"),
                             report=r.get("report"),
                             closure=r.get("closure")))
            print("   liner w/Rbig %.0e deg%-3d %s" % (wf, deg, _v(r)),
                  flush=True)
    return rows


def fd_immunity():
    """``basis='fd'`` must be uncontracted and, on this uniform radial grid,
    bit-identical at every delta."""
    from lumenairy.elements.bor.bor_stack import BORStack
    rows = []
    for dfrac in (0.0, 1e-4, 1e-6, 1e-7):
        d = dfrac * 24.0
        s = BORStack(Rbig=24.0, m=1, N=200, n_superstrate=1.0,
                     n_substrate=1.0, basis="fd")
        s.add_layer(0.5, segments=[(9.0, 4.0), (24.0, 1.0)])
        s.add_layer(0.5, segments=[(9.0 + d, 2.25), (24.0, 1.0)])
        s.set_source(k0=2.0)
        r = _solve(s, arm_guard=True)
        rows.append(dict(delta_frac=dfrac, hash=r.get("hash"),
                         raised=r.get("raised"),
                         n_warnings=r.get("n_warnings"),
                         has_report=r.get("report") is not None,
                         closure=r.get("closure")))
    return rows


# --------------------------------------------------------------------------- #
#  3. the HUNT -- geometries designed to break the bar in both directions      #
# --------------------------------------------------------------------------- #
def hunt():
    """Three attacks.

    A. THE PHANTOM WALL.  A segment boundary across which ``eps`` is the SAME
       on both sides is physically a no-op, so the stack it describes is
       EXACTLY the stack without it.  Put one a distance ``delta`` from a real
       wall of the neighbouring layer: the window manufactures a sliver, but
       the correct answer is known exactly (it is the answer of the
       phantom-free stack).  A refusal here is a refusal of a solve whose
       right answer is available -- the sharpest false-positive test there is.

    B. THE GENTLE TAPER.  Same geometry class the census calls ordinary, with
       the radius change made small enough that the per-slice wall step falls
       below the refusal's geometric conjunct at an ORDINARY slice count.

    C. THE MISS.  The warn band, scored against the exact reference: is a
       merely-warned rung's answer already wrong?
    """
    out = {}

    # --- A. phantom wall -------------------------------------------------
    ph = []
    for dfrac in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        d = dfrac * 24.0

        def mk_phantom(d=d):
            s = _stack(24.0, 1, 8, 2.0)
            s.add_layer(0.5, segments=[(9.0, 4.0), (24.0, 1.0)])
            # layer 1 is UNIFORM eps 1.0 -- the wall at 9+d is a no-op
            s.add_layer(0.5, segments=[(9.0 + d, 1.0), (24.0, 1.0)])
            s.set_source(k0=2.0)
            return s

        def mk_truth():
            s = _stack(24.0, 1, 8, 2.0)
            s.add_layer(0.5, segments=[(9.0, 4.0), (24.0, 1.0)])
            s.add_layer(0.5, segments=[(24.0, 1.0)])
            s.set_source(k0=2.0)
            return s

        armed = _solve(mk_phantom(), arm_guard=True)
        raw = _solve(mk_phantom(), arm_guard=False)
        truth = _solve(mk_truth(), arm_guard=False)
        rec = dict(delta_frac=dfrac, armed_raised=armed.get("raised"),
                   armed_nwarn=armed.get("n_warnings"),
                   report=raw.get("report"), closure=raw.get("closure"),
                   truth_closure=truth.get("closure"))
        if raw.get("R") and truth.get("R") and \
                len(raw["R"]) == len(truth["R"]):
            # SORTED: the solver's channel ORDER follows the eigensolver and is
            # not stable between two solves, so an elementwise difference of
            # the raw arrays measures a PERMUTATION, not a moved answer.
            a = np.sort(np.asarray(raw["R"]))[::-1]
            b = np.sort(np.asarray(truth["R"]))[::-1]
            rec["R_sorted"] = a.tolist()
            rec["R_truth_sorted"] = b.tolist()
            rec["dR_max_vs_truth"] = float(np.max(np.abs(a - b)))
            rec["dR_rel_vs_truth"] = float(np.max(np.abs(a - b))
                                           / max(np.max(np.abs(b)), 1e-300))
        else:
            rec["channel_count_moved"] = True
            rec["n_ch"] = raw.get("n_channels")
            rec["n_ch_truth"] = truth.get("n_channels")
        ph.append(rec)
        print("   phantom %.0e armed=%-22s dR_vs_truth=%s" % (
            dfrac, rec["armed_raised"] or ("warn%d" % rec["armed_nwarn"]),
            rec.get("dR_max_vs_truth")), flush=True)
    out["phantom_wall"] = ph

    # --- B. gentle taper, scored against its own delta -> 0 ---------------
    gt = []
    for dr in (0.5, 0.05, 0.005, 5e-4, 5e-5, 5e-6):
        armed = _solve(taper(64, 8, 2.0, r_top=8.0, r_bot=8.0 - dr)(),
                       arm_guard=True)
        raw = _solve(taper(64, 8, 2.0, r_top=8.0, r_bot=8.0 - dr)(),
                     arm_guard=False)
        truth = _solve(taper(64, 8, 2.0, r_top=8.0, r_bot=8.0)(),
                       arm_guard=False)
        rec = dict(dr=dr, armed_raised=armed.get("raised"),
                   armed_nwarn=armed.get("n_warnings"),
                   report=raw.get("report"), closure=raw.get("closure"))
        if raw.get("R") and truth.get("R") and len(raw["R"]) == len(truth["R"]):
            a = np.sort(np.asarray(raw["R"]))[::-1]
            b = np.sort(np.asarray(truth["R"]))[::-1]
            rec["R_sorted"] = a.tolist()
            rec["dR_max_vs_flat"] = float(np.max(np.abs(a - b)))
        gt.append(rec)
        print("   gentle dr=%-8g armed=%-22s dR_vs_flat=%s" % (
            dr, rec["armed_raised"] or ("warn%d" % rec["armed_nwarn"]),
            rec.get("dR_max_vs_flat")), flush=True)
    out["gentle_taper"] = gt
    return out


def main():
    build = sys.argv[1]
    part = sys.argv[2] if len(sys.argv) > 2 else "all"
    _vh.require_tree(build)
    a = _vh.arm()
    print("ARM", a, flush=True)
    o = sys.argv[3] if len(sys.argv) > 3 else "."
    res = dict(arm=a, build=build, part=part)
    if part in ("census", "all"):
        with _vh.timed("census"):
            res["census"] = census()
    if part in ("ladder", "all"):
        with _vh.timed("ladder0"):
            res["ladder_adjacent"] = ladder(spacer=0)
        with _vh.timed("ladder1"):
            res["ladder_one_spacer"] = ladder(spacer=1)
        with _vh.timed("ladder2"):
            res["ladder_two_spacers"] = ladder(spacer=2, degrees=(8,))
        with _vh.timed("liner"):
            res["liner"] = liner_ladder()
        with _vh.timed("fd"):
            res["fd"] = fd_immunity()
    if part in ("hunt", "all"):
        with _vh.timed("hunt"):
            res["hunt"] = hunt()
    _vh.dump("%s/v4_sem_%s_%s_%s_%s_t%s.json"
             % (o, part, build, a["platform"], a["loaded_kernel"],
                a["blas_threads"]), res)


if __name__ == "__main__":
    main()
